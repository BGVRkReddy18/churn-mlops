"""Experiment runner.

python -m driftlab.run calibrate     # Exp 0: threshold calibration (needed by 1-3)
python -m driftlab.run detection     # Exp 1: detection benchmark
python -m driftlab.run sensitivity   # Exp 3: drift-magnitude sweep
python -m driftlab.run retraining    # Exp 2: retraining policies (+ label-delay sweep)
python -m driftlab.run electricity   # Exp 4: real-world dataset
python -m driftlab.run all
"""

import os

# One thread per worker process: parallelism comes from joblib. Must be set before numpy loads
# (macOS Accelerate BLAS ignores threadpoolctl, which otherwise oversubscribes the CPU).
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
for _var in THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse
import logging
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from churn.config import ROOT
from driftlab.datasets import load_electricity, load_telco
from driftlab.detectors import DETECTORS, Spec, make_detector
from driftlab.scenarios import Scenario, generate_stream
from driftlab.simulation import (
    RetrainConfig,
    Setup,
    calibrate,
    fit_with_reference,
    score_stream,
    simulate_policy,
    telco_setup,
)

CONFIG_PATH = ROOT / "research" / "config.yaml"
RESULTS = ROOT / "research" / "results"
NAMES = list(DETECTORS)
POLICIES = ["static", "periodic", "oracle"] + [f"detector:{n}" for n in NAMES]
OUTPUTS = {
    "calibrate": "thresholds.csv",
    "detection": "detection.csv.gz",
    "sensitivity": "sensitivity.csv.gz",
    "retraining": "retraining.csv.gz",
    "electricity": "electricity.csv.gz",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("driftlab")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def split_seed(split: int) -> int:
    return 1000 + split


def stream_rng(split: int, stream: int, tag: int = 0) -> np.random.Generator:
    # identical seed for every scenario/policy of a (split, stream) -> paired comparisons
    return np.random.default_rng([split, stream, tag])


def telco_scenarios(c) -> list[Scenario]:
    return [Scenario("none")] + [Scenario(k, m) for k in c["scenarios"] for m in c["modes"]]


def _parallel(fn, tasks, n_jobs):
    def run(args):
        with threadpool_limits(1):
            return fn(*args)

    return Parallel(n_jobs=n_jobs, verbose=0)(delayed(run)(t) for t in tasks)


# ============================================================================ Exp 0
def task_calibrate(split, c):
    ds = load_telco()
    st = telco_setup(ds, split_seed(split), c["batch_size"])
    cal = calibrate(
        st,
        NAMES,
        c["calibration"]["n_streams"],
        c["n_batches"],
        c["calibration"]["target_fpr"],
        rng_seed=10_000 + split,
    )
    rows = []
    for n in NAMES:
        default = make_detector(n, st.spec, split).fit(st.ref).default_threshold
        rows.append({"split": split, "detector": n, "calibrated": cal[n], "default": default})
    return pd.DataFrame(rows)


def thresholds_for(split: int, mode: str = "calibrated") -> dict[str, float]:
    df = pd.read_csv(RESULTS / "thresholds.csv")
    df = df[df["split"] == split]
    return dict(zip(df["detector"], df[mode], strict=True))


# ============================================================================ Exp 1 / 3
def task_detection(split, stream_idx, scenarios, c, tag):
    ds = load_telco()
    st = telco_setup(ds, split_seed(split), c["batch_size"])
    out = []
    for sc in scenarios:
        stream = generate_stream(
            st.stream.X, st.stream.y, sc, c["n_batches"], c["batch_size"], c["onset"],
            c["ramp"], stream_rng(split, stream_idx, tag),
        )  # fmt: skip
        S, mauc = score_stream(st, stream, NAMES, c["label_delay"], split * 100 + stream_idx)
        df = pd.DataFrame(S, columns=NAMES)
        df.insert(0, "t", range(len(stream)))
        df["alpha"] = [a for _, _, a in stream]
        df["model_auc"] = mauc
        df["split"], df["stream"] = split, stream_idx
        df["scenario"], df["kind"], df["mode"], df["severity"] = sc.name, sc.kind, sc.mode, sc.s
        out.append(df)
    return pd.concat(out, ignore_index=True)


# ============================================================================ Exp 2
def task_retraining(split, stream_idx, scenarios, c, label_delay):
    ds = load_telco()
    st = telco_setup(ds, split_seed(split), c["batch_size"])
    thr = thresholds_for(split)
    r = c["retraining"]
    cfg = RetrainConfig(label_delay, r["window"], r["wait"], r["cooldown"], r["periodic_every"])
    out = []
    for sc in scenarios:
        stream = generate_stream(
            st.stream.X, st.stream.y, sc, c["n_batches"], c["batch_size"], c["onset"],
            c["ramp"], stream_rng(split, stream_idx),
        )  # fmt: skip
        # independent evaluation sample at every t (disjoint pool -> no train/test leakage)
        eval_stream = generate_stream(
            st.eval.X, st.eval.y, sc, c["n_batches"], c["batch_size"], c["onset"],
            c["ramp"], stream_rng(split, stream_idx, 99),
        )  # fmt: skip
        onset = None if sc.kind == "none" else c["onset"]
        seed = split * 100 + stream_idx
        for pol in POLICIES:
            df = simulate_policy(pol, st, st.spec, stream, cfg, thr, onset, seed, eval_stream)
            df["policy"] = pol.replace("detector:", "")
            df["split"], df["stream"], df["label_delay"] = split, stream_idx, label_delay
            df["scenario"], df["kind"], df["mode"] = sc.name, sc.kind, sc.mode
            out.append(df)
    return pd.concat(out, ignore_index=True)


# ============================================================================ Exp 4
def task_electricity(seed, c):
    ds = load_electricity()
    B = c["batch_size"]
    n_b = len(ds) // B
    chunk = lambda b: np.arange(b * B, (b + 1) * B)  # noqa: E731
    tr = np.concatenate([chunk(b) for b in range(c["train_batches"])])
    ca = np.concatenate(
        [chunk(b) for b in range(c["train_batches"], c["train_batches"] + c["calibration_batches"])]
    )
    first = c["train_batches"] + c["calibration_batches"]
    spec = Spec(ds.numeric, ds.categorical, B)
    train = ds.subset(tr)
    model, ref = fit_with_reference(spec, train.X, train.y, seed)
    setup = Setup(spec, model, ref, ds.subset(ca), None, seed)
    k = c["calibration"]
    thr = calibrate(setup, NAMES, k["n_streams"], k["n_batches"], k["target_fpr"], 20_000 + seed)

    stream = []
    for b in range(first, n_b):
        sub = ds.subset(chunk(b))
        stream.append((sub.X, sub.y, np.nan))
    r = c["retraining"]
    cfg = RetrainConfig(
        c["label_delay"], r["window"], r["wait"], r["cooldown"], r["periodic_every"]
    )
    out = []
    for pol in ["static", "periodic", "always"] + [f"detector:{n}" for n in NAMES]:
        df = simulate_policy(pol, setup, spec, stream, cfg, thr, None, seed)
        df["policy"], df["seed"] = pol.replace("detector:", ""), seed
        df["week"] = df["t"] + first
        out.append(df)
    thr_df = pd.DataFrame([{"seed": seed, "detector": n, "calibrated": v} for n, v in thr.items()])
    return pd.concat(out, ignore_index=True), thr_df


# ============================================================================ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments",
        nargs="+",
        choices=["calibrate", "detection", "sensitivity", "retraining", "electricity", "all"],
        help="one or more stages, run in the order given",
    )
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip stages whose result file already exists (resume an interrupted study)",
    )
    args = parser.parse_args()
    cfg = load_config()
    c = cfg["telco"]
    if args.quick:
        c.update(n_splits=2, streams_per_split=1)
        c["calibration"]["n_streams"] = 3
        c["sensitivity"]["streams_per_split"] = 1
        cfg["electricity"]["seeds"] = 1
    n_jobs = cfg["n_jobs"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    exps = (
        ["calibrate", "detection", "sensitivity", "retraining", "electricity"]
        if "all" in args.experiments
        else args.experiments
    )
    splits = range(c["n_splits"])

    for exp in exps:
        if args.skip_existing and (RESULTS / OUTPUTS[exp]).exists():
            log.info("=== %s === already done (%s) - skipping", exp, OUTPUTS[exp])
            continue
        start = time.time()
        log.info("=== %s ===", exp)
        if exp == "calibrate":
            dfs = _parallel(task_calibrate, [(s, c) for s in splits], n_jobs)
            pd.concat(dfs).to_csv(RESULTS / "thresholds.csv", index=False)

        elif exp == "detection":
            tasks = [
                (s, k, telco_scenarios(c), c, 0)
                for s in splits
                for k in range(c["streams_per_split"])
            ]
            pd.concat(_parallel(task_detection, tasks, n_jobs)).to_csv(
                RESULTS / "detection.csv.gz", index=False
            )

        elif exp == "sensitivity":
            sens = c["sensitivity"]
            scen = [
                Scenario(kind, "sudden", sev)
                for kind, sevs in sens["severities"].items()
                for sev in sevs
            ]
            tasks = [(s, k, scen, c, 1) for s in splits for k in range(sens["streams_per_split"])]
            pd.concat(_parallel(task_detection, tasks, n_jobs)).to_csv(
                RESULTS / "sensitivity.csv.gz", index=False
            )

        elif exp == "retraining":
            scen = telco_scenarios(c)
            tasks = [
                (s, k, scen, c, c["label_delay"])
                for s in splits
                for k in range(c["streams_per_split"])
            ]
            sweep = c["retraining"]["delay_sweep"]
            sweep_scen = [sc for sc in scen if sc.name in sweep["scenarios"]]
            tasks += [
                (s, k, sweep_scen, c, d)
                for d in sweep["label_delays"]
                for s in splits
                for k in range(c["streams_per_split"])
            ]
            pd.concat(_parallel(task_retraining, tasks, n_jobs)).to_csv(
                RESULTS / "retraining.csv.gz", index=False
            )

        elif exp == "electricity":
            e = cfg["electricity"]
            res = _parallel(task_electricity, [(s, e) for s in range(e["seeds"])], n_jobs)
            pd.concat([r[0] for r in res]).to_csv(RESULTS / "electricity.csv.gz", index=False)
            pd.concat([r[1] for r in res]).to_csv(
                RESULTS / "electricity_thresholds.csv", index=False
            )
        log.info("%s finished in %.1f min", exp, (time.time() - start) / 60)


if __name__ == "__main__":
    main()
