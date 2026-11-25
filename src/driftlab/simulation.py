"""Core simulation: score detectors on streams, calibrate thresholds, simulate retraining.

Timeline convention (batch index t = 0..T-1):
  * the model predicts batch t when it arrives (prequential evaluation: metrics on batch t are
    computed with the model in force *before* seeing its labels);
  * ground-truth labels of batch t become available at the end of batch t + label_delay;
  * unsupervised detectors score batch t immediately; supervised detectors score batch
    t - label_delay at time t.

Leakage control (Telco): streams are bootstrapped from a finite pool, so a retrained model could
see rows that re-appear in later batches. Performance is therefore measured on an *independent
evaluation sample* drawn at every t from a disjoint pool under the same drift condition; the
retraining data never touches that pool.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split

from driftlab.datasets import Dataset, make_model
from driftlab.detectors import Batch, Spec, make_detector


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


# ----------------------------------------------------------------------------- setup
@dataclass
class Setup:
    """Everything fixed for one outer repetition: data pools, the deployed model, reference."""

    spec: Spec
    model: object
    ref: Batch
    cal: Dataset  # pool for calibration streams (clean)
    stream: Dataset | None  # pool the operational streams are drawn from
    seed: int
    eval: Dataset | None = None  # disjoint pool for leakage-free performance measurement


def telco_setup(ds: Dataset, seed: int, batch_size: int) -> Setup:
    """Stratified, disjoint split:
    30 % train | 10 % reference | 10 % calibration | 25 % operational stream | 25 % evaluation."""
    idx = np.arange(len(ds))

    def take(pool, frac_of_pool):
        return train_test_split(
            pool, train_size=frac_of_pool, stratify=ds.y[pool], random_state=seed
        )

    tr, rest = take(idx, 0.30)
    ref, rest = take(rest, 0.10 / 0.70)
    cal, rest = take(rest, 0.10 / 0.60)
    st, ev = take(rest, 0.5)
    train, refd = ds.subset(tr), ds.subset(ref)
    model = make_model(ds, seed).fit(train.X, train.y)
    ref_batch = Batch(refd.X, refd.y, model.predict_proba(refd.X)[:, 1])
    spec = Spec(ds.numeric, ds.categorical, batch_size)
    return Setup(spec, model, ref_batch, ds.subset(cal), ds.subset(st), seed, ds.subset(ev))


def fit_with_reference(ds_like, X, y, seed: int):
    """Fit a model and build an *unbiased* reference from out-of-fold predictions."""
    model = make_model(ds_like, seed).fit(X, y)
    oof = cross_val_predict(make_model(ds_like, seed), X, y, cv=3, method="predict_proba")[:, 1]
    return model, Batch(X.reset_index(drop=True), np.asarray(y), oof)


# ----------------------------------------------------------------------------- detection
def score_stream(setup: Setup, stream, detector_names, label_delay: int, seed: int):
    """Score every detector on every batch with a FIXED model (no retraining).
    Returns (scores [T x D], model_auc [T])."""
    dets = [make_detector(n, setup.spec, seed).fit(setup.ref) for n in detector_names]
    T = len(stream)
    scores = np.full((T, len(dets)), np.nan)
    model_auc = np.full(T, np.nan)
    probas = []
    for t, (X, y, _alpha) in enumerate(stream):
        p = setup.model.predict_proba(X)[:, 1]
        probas.append(p)
        model_auc[t] = auc(y, p)
        for j, d in enumerate(dets):
            if d.supervised:
                s = t - label_delay
                if s >= 0:
                    scores[t, j] = d.score(Batch(stream[s][0], stream[s][1], probas[s]))
            else:
                scores[t, j] = d.score(Batch(X, None, p))
    return scores, model_auc


def calibrate(setup: Setup, detector_names, n_streams, n_batches, target_fpr, rng_seed):
    """Per-batch false-positive-rate calibration: run clean (no-drift) streams drawn from the
    held-out calibration pool and take the (1 - target_fpr) quantile of each detector's score."""
    from driftlab.scenarios import Scenario, generate_stream

    rng = np.random.default_rng(rng_seed)
    none = Scenario("none")
    all_scores = []
    for k in range(n_streams):
        stream = generate_stream(
            setup.cal.X, setup.cal.y, none, n_batches, setup.spec.batch_size, 0, 0, rng
        )
        s, _ = score_stream(setup, stream, detector_names, 0, rng_seed + k)
        all_scores.append(s)
    S = np.vstack(all_scores)
    return {n: float(np.nanquantile(S[:, j], 1 - target_fpr)) for j, n in enumerate(detector_names)}


# ----------------------------------------------------------------------------- retraining
@dataclass
class RetrainConfig:
    label_delay: int
    window: int  # retrain on the last `window` labelled batches
    wait: int  # after an alarm, wait until `wait` labelled batches of the new regime exist
    cooldown: int  # ignore alarms for `cooldown` batches after a retrain
    periodic_every: int


def simulate_policy(
    policy: str,
    setup: Setup,
    ds_like,
    stream,
    cfg: RetrainConfig,
    thresholds: dict[str, float],
    onset: int | None,
    seed: int,
    eval_stream=None,
) -> pd.DataFrame:
    """policy: 'static' | 'periodic' | 'oracle' | 'always' | 'detector:<NAME>'.
    If `eval_stream` is given, performance at t is measured on eval_stream[t] (leakage-free);
    otherwise on the operational batch itself (valid for real, non-resampled streams)."""
    d = cfg.label_delay
    model = setup.model
    det_name = policy.split(":", 1)[1] if policy.startswith("detector:") else None
    det = make_detector(det_name, setup.spec, seed).fit(setup.ref) if det_name else None
    oracle_times = set()
    if policy == "oracle" and onset is not None:
        oracle_times = {onset + d + cfg.wait - 1, onset + d + cfg.window - 1}

    probas, rows = {}, []
    last_retrain, pending = -(10**9), None

    def retrain(t):
        nonlocal model, det, last_retrain, pending
        avail = [s for s in range(max(0, t - d - cfg.window + 1), t - d + 1)]
        if not avail:
            return False
        Xw = pd.concat([stream[s][0] for s in avail], ignore_index=True)
        yw = np.concatenate([stream[s][1] for s in avail])
        if len(np.unique(yw)) < 2:
            return False
        if det is not None:
            model, ref = fit_with_reference(ds_like, Xw, yw, seed)
            det = make_detector(det_name, setup.spec, seed).fit(ref)
        else:
            model = make_model(ds_like, seed).fit(Xw, yw)
        last_retrain, pending = t, None
        return True

    for t, (X, y, alpha) in enumerate(stream):
        p = model.predict_proba(X)[:, 1]
        probas[t] = p
        if eval_stream is not None:
            Xe, ye, _ = eval_stream[t]
            pe = model.predict_proba(Xe)[:, 1]
        else:
            ye, pe = y, p
        row = {
            "t": t,
            "alpha": alpha,
            "auc": auc(ye, pe),
            "accuracy": accuracy_score(ye, (pe >= 0.5).astype(int)),
            "score": np.nan,
            "alarm": False,
            "retrained": False,
        }

        if det is not None:
            if det.supervised:
                s = t - d
                if s > last_retrain and s >= 0:  # only batches predicted by the current model
                    row["score"] = det.score(Batch(stream[s][0], stream[s][1], probas[s]))
            else:
                row["score"] = det.score(Batch(X, None, p))
            if row["score"] > thresholds[det_name] and t - last_retrain >= cfg.cooldown:
                row["alarm"] = True
                if pending is None:
                    pending = t + cfg.wait - 1 + (0 if det.supervised else d)

        due = (
            (policy == "periodic" and (t + 1) % cfg.periodic_every == 0)
            or policy == "always"
            or t in oracle_times
            or (pending is not None and t >= pending)
        )
        if due:
            row["retrained"] = retrain(t)
        rows.append(row)
    return pd.DataFrame(rows)
