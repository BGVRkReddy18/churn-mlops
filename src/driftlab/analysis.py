"""Turn raw experiment logs into metrics, tables and statistical tests.

python -m driftlab.analysis      # writes research/results/tables/*.csv|.md + summary.json
"""

import json

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from churn.config import ROOT
from driftlab.detectors import DETECTORS, SUPERVISED

RESULTS = ROOT / "research" / "results"
TABLES = RESULTS / "tables"
NAMES = list(DETECTORS)
# Nemenyi critical values q_0.05 (Demsar 2006, Table 5), indexed by number of methods k
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
             10: 3.164, 11: 3.219, 12: 3.268}  # fmt: skip


def cfg() -> dict:
    return yaml.safe_load((ROOT / "research" / "config.yaml").read_text())


# ============================================================================ statistics
def friedman_nemenyi(df: pd.DataFrame, block_cols, method_col, value_col, higher_is_better=True):
    """Friedman test + average ranks + Nemenyi critical difference (Demsar 2006)."""
    wide = df.pivot_table(index=block_cols, columns=method_col, values=value_col).dropna()
    ranks = (-wide if higher_is_better else wide).rank(axis=1, method="average")
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    k, n = wide.shape[1], wide.shape[0]
    cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * n))
    return {
        "n_blocks": int(n),
        "k": int(k),
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "nemenyi_cd": float(cd),
        "avg_rank": ranks.mean().sort_values().round(3).to_dict(),
    }, wide


def wilcoxon_holm(wide: pd.DataFrame, control: str, higher_is_better=True) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test of `control` vs every other method, Holm-corrected."""
    rows = []
    for m in wide.columns:
        if m == control:
            continue
        diff = wide[control] - wide[m]
        p = stats.wilcoxon(diff, zero_method="zsplit").pvalue if (diff != 0).any() else 1.0
        rows.append({"vs": m, "median_diff": float(diff.median()), "p": float(p)})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(out)
    adj, running = [], 0.0
    for i, p in enumerate(out["p"]):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    out["p_holm"] = adj
    out["significant"] = out["p_holm"] < 0.05
    out.insert(0, "control", control)
    return out


def ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


# ============================================================================ Exp 1 / 3
def detection_runs(raw: pd.DataFrame, thresholds: pd.DataFrame, c: dict) -> pd.DataFrame:
    """One row per (run, scenario, detector, threshold-mode) with detection metrics."""
    onset, ramp, d = c["onset"], c["ramp"], c["label_delay"]
    thr = thresholds.set_index(["split", "detector"])
    rows = []
    # severity is part of the key: the sensitivity sweep reuses one scenario name per severity
    keys = ["split", "stream", "scenario", "severity"]
    for (split, stream, scen, _sev), g in raw.groupby(keys, sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        kind, mode, sev = g["kind"].iat[0], g["mode"].iat[0], g["severity"].iat[0]
        for det in NAMES:
            s = g[det].to_numpy()
            lag = d if det in SUPERVISED else 0
            full = onset + (ramp - 1 if mode == "gradual" else 0) + lag
            for tmode in ("default", "calibrated"):
                alarm = np.nan_to_num(s, nan=-np.inf) > thr.loc[(split, det), tmode]
                r = dict(split=split, stream=stream, scenario=scen, kind=kind, mode=mode,
                         severity=sev, detector=det, threshold=tmode)  # fmt: skip
                if kind == "none":
                    r["false_alarm_rate"] = alarm.mean()
                    r["any_false_alarm"] = alarm.any()
                else:
                    pre = t < onset
                    r["false_alarm_rate"] = alarm[pre].mean()
                    r["any_false_alarm"] = alarm[pre].any()
                    post_alarms = t[(t >= onset) & alarm]
                    r["detected"] = len(post_alarms) > 0
                    r["delay"] = post_alarms.min() - onset if len(post_alarms) else np.nan
                    if tmode == "calibrated":
                        neg, pos = s[pre & ~np.isnan(s)], s[(t >= full) & ~np.isnan(s)]
                        r["detection_auc"] = (
                            roc_auc_score(
                                np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]
                            )  # fmt: skip
                            if len(neg) and len(pos)
                            else np.nan
                        )
                rows.append(r)
    return pd.DataFrame(rows)


def drift_harm(raw: pd.DataFrame, c: dict) -> pd.DataFrame:
    """How much does each drift actually hurt the (non-retrained) model?"""
    onset, ramp = c["onset"], c["ramp"]
    full = np.where(raw["mode"] == "gradual", onset + ramp - 1, onset)
    period = np.select([raw["t"] < onset, raw["t"] >= full], ["pre", "post"], "ramp")
    per_run = (
        raw.assign(period=period)
        .pivot_table(index=["scenario", "split", "stream"], columns="period", values="model_auc")
        .assign(drop=lambda x: x["pre"] - x["post"])
    )
    out = per_run.groupby("scenario").agg(
        auc_pre=("pre", "mean"), auc_post=("post", "mean"), auc_drop=("drop", "mean")
    )
    out["auc_drop_ci95"] = per_run.groupby("scenario")["drop"].apply(ci95)
    return out.reset_index().sort_values("auc_drop", ascending=False)


def summarise_detection(runs: pd.DataFrame) -> dict:
    cal = runs[runs.threshold == "calibrated"]
    drifted = cal[cal.kind != "none"]
    control = runs[runs.kind == "none"]

    overall = drifted.groupby("detector").agg(
        detection_rate=("detected", "mean"),
        median_delay=("delay", "median"),
        mean_delay=("delay", "mean"),
        detection_auc=("detection_auc", "mean"),
    )
    fa = control.pivot_table(index="detector", columns="threshold", values="false_alarm_rate")
    overall["fa_rate_default"] = fa["default"]
    overall["fa_rate_calibrated"] = fa["calibrated"]
    overall = overall.sort_values("detection_auc", ascending=False)

    by_scen = {
        m: drifted.pivot_table(index="detector", columns="scenario", values=m, aggfunc=agg)
        for m, agg in [("detected", "mean"), ("delay", "median"), ("detection_auc", "mean")]
    }
    default_vs_cal = (
        runs[runs.kind != "none"]
        .pivot_table(index="detector", columns="threshold", values="detected")
        .rename(columns=lambda x: f"detection_rate_{x}")
    )
    stats_auc, wide = friedman_nemenyi(
        drifted, ["split", "stream", "scenario"], "detector", "detection_auc"
    )
    best = next(iter(stats_auc["avg_rank"]))
    return {
        "overall": overall,
        "by_scenario": by_scen,
        "default_vs_calibrated": default_vs_cal.join(fa.add_prefix("fa_rate_")),
        "friedman": stats_auc,
        "wilcoxon": wilcoxon_holm(wide, best),
    }


# ============================================================================ Exp 2
RUN_KEYS = ["label_delay", "split", "stream", "scenario", "kind", "mode", "policy"]
PAIR_KEYS = ["label_delay", "split", "stream", "scenario"]


def retraining_runs(raw: pd.DataFrame, c: dict) -> pd.DataFrame:
    onset = c["onset"]
    raw = raw.assign(
        false_retrain=raw["retrained"] & ((raw["kind"] == "none") | (raw["t"] < onset))
    )
    g = raw.groupby(RUN_KEYS)
    post = raw[raw["t"] >= onset].groupby(RUN_KEYS)
    out = pd.DataFrame(
        {
            "auc_post": post["auc"].mean(),
            "acc_post": post["accuracy"].mean(),
            "auc_all": g["auc"].mean(),
            "retrains": g["retrained"].sum(),
            "false_retrains": g["false_retrain"].sum(),
        }
    ).reset_index()
    static = out[out["policy"] == "static"].set_index(PAIR_KEYS)["auc_post"]
    out["gain_vs_static"] = (
        out["auc_post"].to_numpy()
        - static.reindex(pd.MultiIndex.from_frame(out[PAIR_KEYS])).to_numpy()
    )
    return out


def summarise_retraining(runs: pd.DataFrame, delay: int) -> dict:
    main = runs[runs.label_delay == delay]
    by_scen = main.pivot_table(index="policy", columns="scenario", values="auc_post")
    gain = main.pivot_table(index="policy", columns="scenario", values="gain_vs_static")
    overall = main.groupby("policy").agg(
        auc_post=("auc_post", "mean"),
        gain_vs_static=("gain_vs_static", "mean"),
        retrains=("retrains", "mean"),
        false_retrains=("false_retrains", "mean"),
    )
    harmful = main[main.kind.isin(["concept", "sensor_bug"])]
    overall["gain_harmful_drift"] = harmful.groupby("policy")["gain_vs_static"].mean()
    overall["retrains_no_drift"] = main[main.kind == "none"].groupby("policy")["retrains"].mean()
    overall = overall.sort_values("auc_post", ascending=False)
    fr, wide = friedman_nemenyi(main, ["split", "stream", "scenario"], "policy", "auc_post")
    non_oracle = [p for p in fr["avg_rank"] if p != "oracle"]
    return {
        "overall": overall,
        "auc_by_scenario": by_scen,
        "gain_by_scenario": gain,
        "friedman": fr,
        "wilcoxon": wilcoxon_holm(wide, non_oracle[0]),
    }


def summarise_delay(runs: pd.DataFrame) -> pd.DataFrame:
    swept = runs.loc[runs["label_delay"] != runs["label_delay"].mode()[0], "scenario"].unique()
    sweep = runs[runs["scenario"].isin(swept)]
    return sweep.pivot_table(
        index="policy", columns=["scenario", "label_delay"], values="gain_vs_static"
    )


# ============================================================================ Exp 4
def summarise_electricity(raw: pd.DataFrame) -> dict:
    per_seed = raw.groupby(["policy", "seed"]).agg(
        accuracy=("accuracy", "mean"), auc=("auc", "mean"), retrains=("retrained", "sum")
    )
    overall = per_seed.groupby("policy").mean().sort_values("accuracy", ascending=False)
    overall["accuracy_ci95"] = per_seed.groupby("policy")["accuracy"].apply(ci95)
    weekly = raw.groupby(["week", "policy"])["accuracy"].mean().reset_index()
    weekly["block"] = weekly["week"]
    fr, wide = friedman_nemenyi(weekly, ["block"], "policy", "accuracy")
    non_always = [p for p in fr["avg_rank"] if p != "always"]
    return {
        "overall": overall,
        "friedman": fr,
        "wilcoxon": wilcoxon_holm(wide, non_always[0]),
    }


# ============================================================================ output
def _write(name: str, df: pd.DataFrame, floatfmt=".3f") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv")
    (TABLES / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt) + "\n")


def main() -> None:
    c = cfg()["telco"]
    summary = {}
    thresholds = pd.read_csv(RESULTS / "thresholds.csv")
    _write("thresholds", thresholds.groupby("detector")[["default", "calibrated"]].mean(), ".4f")

    det_raw = pd.read_csv(RESULTS / "detection.csv.gz")
    runs = detection_runs(det_raw, thresholds, c)
    runs.to_csv(RESULTS / "detection_runs.csv.gz", index=False)
    s = summarise_detection(runs)
    _write("exp1_detection_overall", s["overall"])
    for m, df in s["by_scenario"].items():
        _write(f"exp1_{m}_by_scenario", df, ".2f")
    _write("exp1_default_vs_calibrated", s["default_vs_calibrated"])
    _write("exp1_wilcoxon", s["wilcoxon"].set_index("vs"), ".4g")
    summary["exp1_friedman_detection_auc"] = s["friedman"]
    harm = drift_harm(det_raw, c).set_index("scenario")
    _write("exp1_drift_harm", harm)

    sens_raw = pd.read_csv(RESULTS / "sensitivity.csv.gz")
    sens = detection_runs(sens_raw, thresholds, c)
    sens.to_csv(RESULTS / "sensitivity_runs.csv.gz", index=False)
    sens_tab = sens[sens.threshold == "calibrated"].pivot_table(
        index=["kind", "severity"], columns="detector", values="detected"
    )
    _write("exp3_sensitivity_detection_rate", sens_tab, ".2f")
    sens_harm = drift_harm(sens_raw.assign(scenario=sens_raw.kind + "@" +
                                           sens_raw.severity.astype(str)), c)  # fmt: skip
    _write("exp3_sensitivity_harm", sens_harm.set_index("scenario"))

    ret_raw = pd.read_csv(RESULTS / "retraining.csv.gz")
    rr = retraining_runs(ret_raw, c)
    rr.to_csv(RESULTS / "retraining_runs.csv.gz", index=False)
    r = summarise_retraining(rr, c["label_delay"])
    _write("exp2_retraining_overall", r["overall"])
    _write("exp2_auc_by_scenario", r["auc_by_scenario"])
    _write("exp2_gain_by_scenario", r["gain_by_scenario"])
    _write("exp2_wilcoxon", r["wilcoxon"].set_index("vs"), ".4g")
    summary["exp2_friedman_auc_post"] = r["friedman"]
    _write("exp2b_label_delay_gain", summarise_delay(rr))

    el = pd.read_csv(RESULTS / "electricity.csv.gz")
    e = summarise_electricity(el)
    _write("exp4_electricity_overall", e["overall"])
    _write("exp4_wilcoxon", e["wilcoxon"].set_index("vs"), ".4g")
    summary["exp4_friedman_accuracy"] = e["friedman"]

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
