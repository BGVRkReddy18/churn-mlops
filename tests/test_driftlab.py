"""Unit tests for the drift-research framework (fast, synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from churn.data.prepare import clean
from driftlab.analysis import detection_runs, friedman_nemenyi, wilcoxon_holm
from driftlab.datasets import Dataset, make_model
from driftlab.detectors import DETECTORS, SUPERVISED, Batch, Spec, make_detector
from driftlab.scenarios import Scenario, generate_stream, intensity, sample_batch
from driftlab.simulation import RetrainConfig, Setup, fit_with_reference, simulate_policy
from tests.conftest import make_raw_df

SPEC_ARGS = dict(
    numeric=["tenure", "MonthlyCharges", "TotalCharges"],
    categorical=["Contract", "InternetService", "gender"],
)


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw_df(3000, seed=1))
    # give the label real signal so supervised detectors / retraining have something to learn
    df["Churn"] = ((df["Contract"] == "Month-to-month") & (df["tenure"] < 24)).astype(int)
    X = df[SPEC_ARGS["numeric"] + SPEC_ARGS["categorical"]]
    return Dataset("synthetic", X, df["Churn"].to_numpy(), **SPEC_ARGS)


@pytest.fixture(scope="module")
def fitted(data):
    spec = Spec(data.numeric, data.categorical, batch_size=200)
    tr, rest = data.subset(np.arange(1500)), data.subset(np.arange(1500, 3000))
    model = make_model(spec).fit(tr.X, tr.y)
    ref = Batch(rest.X.iloc[:700], rest.y[:700], model.predict_proba(rest.X.iloc[:700])[:, 1])
    pool = rest.subset(np.arange(700, 1500))
    return spec, model, ref, pool


def _batch(model, X, y):
    return Batch(X, y, model.predict_proba(X)[:, 1])


# ---------------------------------------------------------------- scenarios
def test_intensity_sudden_and_gradual():
    assert [intensity(t, 5, 4, "sudden") for t in (4, 5, 9)] == [0, 1, 1]
    assert [intensity(t, 5, 4, "gradual") for t in (4, 5, 6, 8, 20)] == [0, 0.25, 0.5, 1, 1]


def test_concept_drift_changes_labels_not_features(data):
    rng = np.random.default_rng(0)
    Xc, yc = sample_batch(data.X, data.y, 2000, 1.0, Scenario("concept", severity=1.0), rng)
    assert (Xc["tenure"] >= 0).all()
    # long-tenure customers were never churners in this synthetic set; now they are
    assert yc[(Xc["tenure"] > 24).to_numpy()].mean() == 1.0


def test_inflation_scales_charges(data):
    rng = np.random.default_rng(0)
    clean_b, _ = sample_batch(data.X, data.y, 2000, 0.0, Scenario("inflation"), rng)
    drift_b, _ = sample_batch(data.X, data.y, 2000, 1.0, Scenario("inflation", severity=0.5), rng)
    ratio = drift_b["MonthlyCharges"].mean() / clean_b["MonthlyCharges"].mean()
    assert 1.35 < ratio < 1.65


def test_stream_is_reproducible(data):
    a = generate_stream(data.X, data.y, Scenario("subpopulation"), 5, 50, 2, 0,
                        np.random.default_rng(3))  # fmt: skip
    b = generate_stream(data.X, data.y, Scenario("subpopulation"), 5, 50, 2, 0,
                        np.random.default_rng(3))  # fmt: skip
    for (xa, ya, _), (xb, yb, _) in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(xa, xb)
        np.testing.assert_array_equal(ya, yb)


# ---------------------------------------------------------------- detectors
@pytest.mark.parametrize("name", list(DETECTORS))
def test_detector_scores_drift_higher_than_no_drift(fitted, name):
    spec, model, ref, pool = fitted
    rng = np.random.default_rng(0)
    # covariate + label damage at once so every detector family should react
    bug = Scenario("sensor_bug", severity=1.0)
    det = make_detector(name, spec, 0).fit(ref)
    clean_scores = [det.score(_batch(model, *sample_batch(pool.X, pool.y, 200, 0.0, bug, rng)))
                    for _ in range(5)]  # fmt: skip
    det = make_detector(name, spec, 0).fit(ref)
    drift_scores = []
    for _ in range(5):
        Xd, yd = sample_batch(pool.X, pool.y, 200, 1.0, bug, rng)
        yd = 1 - yd if name in SUPERVISED else yd  # supervised detectors need label damage
        drift_scores.append(det.score(_batch(model, Xd, yd)))
    assert np.mean(drift_scores) > np.mean(clean_scores)


def test_psi_is_near_zero_on_reference(fitted):
    spec, model, ref, _ = fitted
    det = make_detector("PSI", spec).fit(ref)
    assert det.score(ref) < 1e-6


# ---------------------------------------------------------------- simulation
def test_static_policy_never_retrains_and_periodic_does(fitted):
    spec, model, ref, pool = fitted
    setup = Setup(spec, model, ref, pool, pool, 0)
    stream = generate_stream(pool.X, pool.y, Scenario("none"), 10, 200, 5, 0,
                             np.random.default_rng(0))  # fmt: skip
    cfg = RetrainConfig(label_delay=1, window=3, wait=1, cooldown=1, periodic_every=4)
    static = simulate_policy("static", setup, spec, stream, cfg, {}, None, 0)
    periodic = simulate_policy("periodic", setup, spec, stream, cfg, {}, None, 0)
    assert static["retrained"].sum() == 0
    assert periodic["retrained"].sum() == 2  # at t = 3 and t = 7


def test_detector_policy_retrains_after_concept_drift(fitted):
    spec, model, ref, pool = fitted
    setup = Setup(spec, model, ref, pool, pool, 0)
    stream = generate_stream(pool.X, pool.y, Scenario("concept", severity=1.0), 14, 200, 4, 0,
                             np.random.default_rng(0))  # fmt: skip
    cfg = RetrainConfig(label_delay=1, window=3, wait=2, cooldown=2, periodic_every=100)
    out = simulate_policy("detector:PerfDrop", setup, spec, stream, cfg, {"PerfDrop": 0.05}, 4, 0)
    first = out.index[out["retrained"]].min()
    assert first >= 4  # never before the drift
    assert out.loc[first + 1 :, "auc"].mean() > out.loc[4:first, "auc"].mean()


def test_fit_with_reference_gives_out_of_fold_scores(data):
    spec = Spec(data.numeric, data.categorical)
    model, ref = fit_with_reference(spec, data.X.iloc[:600], data.y[:600], 0)
    in_sample = model.predict_proba(ref.X)[:, 1]
    assert not np.allclose(in_sample, ref.proba)  # OOF != in-sample (no optimistic bias)


# ---------------------------------------------------------------- statistics
def test_friedman_ranks_clearly_better_method_first():
    rng = np.random.default_rng(0)
    methods = [("good", 0.9), ("ok", 0.8), ("bad", 0.5)]
    rows = [
        {"block": b, "m": m, "v": base + rng.normal(0, 0.01)}
        for b in range(30)
        for m, base in methods
    ]
    res, wide = friedman_nemenyi(pd.DataFrame(rows), ["block"], "m", "v")
    assert list(res["avg_rank"]) == ["good", "ok", "bad"]
    assert res["friedman_p"] < 1e-6
    w = wilcoxon_holm(wide, "good")
    assert w["significant"].all()


def test_detection_runs_keeps_severities_separate():
    """Regression: a severity sweep reuses one scenario name, so runs must be keyed by severity."""
    onset, T = 5, 12
    rows = []
    for sev, jump in [(0.1, 0.0), (0.9, 10.0)]:  # weak drift: no signal; strong drift: big jump
        for t in range(T):
            score = jump if t >= onset else 0.0
            rows.append({"t": t, "split": 0, "stream": 0, "scenario": "inflation-sudden",
                         "kind": "inflation", "mode": "sudden", "severity": sev,
                         **{n: score for n in DETECTORS}})  # fmt: skip
    thr = pd.DataFrame([{"split": 0, "detector": n, "default": 1.0, "calibrated": 1.0}
                        for n in DETECTORS])  # fmt: skip
    c = {"onset": onset, "ramp": 1, "label_delay": 0}
    runs = detection_runs(pd.DataFrame(rows), thr, c)
    cal = runs[runs.threshold == "calibrated"]
    rate = cal.groupby("severity")["detected"].mean()
    assert list(rate.index) == [0.1, 0.9]
    assert rate[0.1] == 0.0 and rate[0.9] == 1.0
