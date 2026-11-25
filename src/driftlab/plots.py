"""Thesis figures (PNG for reading, PDF for LaTeX).

python -m driftlab.plots
"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from churn.config import ROOT
from driftlab.analysis import RESULTS, TABLES, cfg
from driftlab.detectors import DETECTORS, SUPERVISED

FIGURES = ROOT / "research" / "figures"
NAMES = list(DETECTORS)

# --- palette: fixed categorical order, one slot per detector (validated reference palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DET_COLOR = dict(zip(NAMES, SERIES, strict=False))
BASELINE = {"static": "#8a8984", "periodic": "#52514e", "oracle": "#0b0b0b", "always": "#0b0b0b"}
BASE_STYLE = {"static": "-", "periodic": "--", "oracle": ":", "always": "-."}
FAMILY_COLOR = {"baseline": "#52514e", "unsupervised": SERIES[0], "supervised": SERIES[1]}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
SCEN_ORDER = [
    "subpopulation-sudden", "subpopulation-gradual", "inflation-sudden", "inflation-gradual",
    "sensor_bug-sudden", "sensor_bug-gradual", "concept-sudden", "concept-gradual",
]  # fmt: skip

rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#b5b4ae",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


def family(policy: str) -> str:
    if policy in BASELINE:
        return "baseline"
    return "supervised" if policy in SUPERVISED else "unsupervised"


def style(policy: str) -> dict:
    if policy in BASELINE:
        return {"color": BASELINE[policy], "ls": BASE_STYLE[policy]}
    return {"color": DET_COLOR[policy], "ls": "-"}


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")


def pretty(s: str) -> str:
    return s.replace("_", " ").replace("-", " · ")


def heatmap(ax, df: pd.DataFrame, fmt: str, vmin, vmax, cmap=SEQ):
    im = ax.imshow(df.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), [pretty(c) for c in df.columns], rotation=35, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(df.to_numpy(dtype=float)):
        if np.isnan(v):
            ax.text(j, i, "–", ha="center", va="center", color=MUTED)
            continue
        dark = (v - vmin) / (vmax - vmin) > 0.55
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                color="white" if dark else TEXT)  # fmt: skip
    return im


# ============================================================================ Exp 1
def fig_example_scores(det: pd.DataFrame, thr: pd.DataFrame, c: dict) -> None:
    run = det[(det.split == 0) & (det.stream == 0)]
    t0 = thr[thr.split == 0].set_index("detector")["calibrated"]
    scen = ["none", "subpopulation-sudden", "inflation-sudden", "sensor_bug-sudden",
            "concept-sudden", "concept-gradual"]  # fmt: skip
    fig = Figure(figsize=(11, 6.2))
    axes = fig.subplots(2, 3, sharex=True, sharey=True).ravel()
    for ax, sc in zip(axes, scen, strict=True):
        g = run[run.scenario == sc].sort_values("t")
        for n in NAMES:
            base = t0[n] if n not in ("C2ST",) else t0[n] - 0.5
            val = g[n] if n != "C2ST" else g[n] - 0.5
            ratio = np.clip(val / base, 1e-2, 1e3) if base > 0 else val
            ax.plot(g.t, ratio, color=DET_COLOR[n], label=n, lw=1.3)
        ax.axhline(1, color=TEXT, lw=0.9, ls="--")
        if sc != "none":
            ax.axvspan(c["onset"], g.t.max(), color="#f0efec", zorder=0)
        ax.set_yscale("log")
        ax.set_title(pretty(sc))
    for ax in axes[3:]:
        ax.set_xlabel("batch")
    for ax in axes[::3]:
        ax.set_ylabel("score ÷ calibrated threshold")
    axes[0].legend(ncol=2, fontsize=7.5, loc="upper left")
    fig.suptitle("Detector scores on one stream (alarm when above the dashed line; "
                 "shaded = drift active)", x=0.01, ha="left", fontsize=11)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig01_example_scores")


def fig_detection_heatmaps() -> None:
    rate = pd.read_csv(TABLES / "exp1_detected_by_scenario.csv", index_col=0)
    delay = pd.read_csv(TABLES / "exp1_delay_by_scenario.csv", index_col=0)
    dauc = pd.read_csv(TABLES / "exp1_detection_auc_by_scenario.csv", index_col=0)
    cols = [c for c in SCEN_ORDER if c in rate.columns]
    order = NAMES
    fig = Figure(figsize=(12, 3.8))
    axes = fig.subplots(1, 3)
    heatmap(axes[0], rate.loc[order, cols], ".2f", 0, 1)
    axes[0].set_title("Detection rate (calibrated, ↑)")
    heatmap(axes[1], delay.loc[order, cols], ".0f", 0, 15)
    axes[1].set_title("Median delay in batches (↓)")
    heatmap(axes[2], dauc.loc[order, cols], ".2f", 0.5, 1)
    axes[2].set_title("Detection AUC (threshold-free, ↑)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    save(fig, "fig02_detection_heatmaps")


def fig_false_alarms() -> None:
    df = pd.read_csv(TABLES / "exp1_default_vs_calibrated.csv", index_col=0).loc[NAMES]
    target = cfg()["telco"]["calibration"]["target_fpr"]
    fig = Figure(figsize=(9, 3.4))
    a1, a2 = fig.subplots(1, 2)
    x = np.arange(len(NAMES))
    w = 0.38
    for ax, cols, title in [
        (a1, ("fa_rate_default", "fa_rate_calibrated"), "False-alarm rate per batch (no drift)"),
        (a2, ("detection_rate_default", "detection_rate_calibrated"), "Detection rate (drift)"),
    ]:
        for k, (col, lab, colr) in enumerate(
            zip(cols, ("default threshold", "calibrated threshold"), SERIES[:2], strict=True)
        ):
            ax.bar(x + (k - 0.5) * w, df[col], w * 0.92, color=colr, label=lab)
        ax.set_xticks(x, NAMES)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    a1.axhline(target, color=TEXT, lw=0.9, ls="--")
    a1.annotate(f"target {target:.0%}", (len(NAMES) - 0.5, target), xytext=(0, 4),
                textcoords="offset points", ha="right", color=MUTED, fontsize=8)  # fmt: skip
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a1.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig03_default_vs_calibrated")


def fig_cd(avg_rank: dict, cd: float, title: str, name: str) -> None:
    """Critical-difference diagram (Demsar 2006): methods closer than CD are not
    significantly different (Nemenyi, alpha = 0.05)."""
    items = sorted(avg_rank.items(), key=lambda kv: kv[1])
    k = len(items)
    fig = Figure(figsize=(8, 0.45 * k + 1.2))
    ax = fig.subplots()
    ax.grid(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0.5, k + 0.5)
    ax.invert_xaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(range(1, k + 1))
    ax.spines["top"].set_visible(True)
    for i, (m, r) in enumerate(items):
        y = -(i + 1)
        colr = style(m)["color"] if m in BASELINE or m in DET_COLOR else TEXT
        ax.plot([r, r], [0, y], color=GRID, lw=1)
        ax.plot(r, y, "o", color=colr, ms=7, mec="white", mew=1.5)
        # x axis is inverted: a larger rank value sits further left, i.e. clear of the dot
        ax.text(r + 0.12, y, f"{m}  ({r:.2f})", va="center", ha="right", color=TEXT)
    # cliques: maximal groups whose rank range < CD
    ranks = [r for _, r in items]
    bars, last_end = [], -1
    for i in range(k):
        j = max(j for j in range(i, k) if ranks[j] - ranks[i] < cd)
        if j > i and j > last_end:
            bars.append((ranks[i], ranks[j]))
            last_end = j
    for b, (lo, hi) in enumerate(bars):
        y = 0.35 + 0.18 * b
        ax.plot([lo, hi], [y, y], color=TEXT, lw=3, solid_capstyle="round")
    ax.plot([1, 1 + cd], [0.35 + 0.18 * len(bars) + 0.25] * 2, color=SERIES[7], lw=2)
    ax.text(1 + cd / 2, 0.35 + 0.18 * len(bars) + 0.4, f"CD = {cd:.2f}", ha="center",
            color=MUTED, fontsize=8)  # fmt: skip
    ax.set_ylim(-(k + 0.5), 0.35 + 0.18 * len(bars) + 0.8)
    ax.set_title(title, loc="left", pad=22)
    fig.tight_layout()
    save(fig, name)


def fig_drift_harm() -> None:
    df = pd.read_csv(TABLES / "exp1_drift_harm.csv", index_col=0)
    df = df.loc[[s for s in SCEN_ORDER if s in df.index]][::-1]
    fig = Figure(figsize=(6.5, 3.4))
    ax = fig.subplots()
    colors = [SERIES[7] if v > 0.02 else "#b5b4ae" for v in df["auc_drop"]]
    ax.barh(range(len(df)), df["auc_drop"], xerr=df["auc_drop_ci95"], color=colors,
            height=0.7, error_kw={"ecolor": MUTED, "lw": 0.8})  # fmt: skip
    ax.set_yticks(range(len(df)), [pretty(s) for s in df.index])
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_xlabel("drop in model ROC-AUC after drift (higher = more harmful)")
    ax.set_title("Not all drift is harmful", loc="left")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(df["auc_drop"]):
        ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=TEXT)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig04_drift_harm")


def fig_sensitivity(sens_runs: pd.DataFrame) -> None:
    cal = sens_runs[sens_runs.threshold == "calibrated"]
    harm = pd.read_csv(TABLES / "exp3_sensitivity_harm.csv", index_col=0)
    kinds = ["subpopulation", "inflation", "sensor_bug", "concept"]
    fig = Figure(figsize=(12, 5.6))
    axes = fig.subplots(2, 4, sharey="row", gridspec_kw={"height_ratios": [2.2, 1]})
    for j, kind in enumerate(kinds):
        # both rows share one categorical x axis: severities from strongest to weakest
        g = cal[cal.kind == kind].groupby(["severity", "detector"])["detected"].mean().unstack()
        g = g.sort_index(ascending=False)
        x = np.arange(len(g))
        ax = axes[0, j]
        for n in NAMES:
            ax.plot(x, g[n], marker="o", ms=4, color=DET_COLOR[n], label=n)
        ax.set_title(pretty(kind))
        ax.set_xticks(x, [f"{v:g}" for v in g.index])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        h = harm[harm.index.str.startswith(kind + "@")].copy()
        h["sev"] = h.index.str.split("@").str[1].astype(float)
        h = h.set_index("sev").reindex(g.index)
        axes[1, j].bar(x, h["auc_drop"], color="#b5b4ae", width=0.6)
        axes[1, j].set_xticks(x, [f"{v:g}" for v in g.index])
        axes[1, j].set_xlabel("drift severity")
        axes[1, j].axhline(0, color=TEXT, lw=0.7)
        axes[1, j].grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("detection rate")
    axes[1, 0].set_ylabel("model AUC drop")
    axes[0, 0].legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig05_sensitivity")


# ============================================================================ Exp 2
def best_by_family(overall: pd.DataFrame) -> tuple[str, str]:
    det = overall.drop(index=[p for p in BASELINE if p in overall.index])
    unsup = det[~det.index.isin(SUPERVISED)]["auc_post"].idxmax()
    sup = det[det.index.isin(SUPERVISED)]["auc_post"].idxmax()
    return unsup, sup


def fig_retraining_timeseries(ret: pd.DataFrame, overall: pd.DataFrame, c: dict) -> None:
    unsup, sup = best_by_family(overall)
    pols = ["static", "periodic", "oracle", unsup, sup]
    main = ret[ret.label_delay == c["label_delay"]]
    mean = main.groupby(["scenario", "policy", "t"])["auc"].mean().reset_index()
    scen = ["none"] + SCEN_ORDER
    fig = Figure(figsize=(12, 9))
    axes = fig.subplots(3, 3, sharex=True).ravel()
    for ax, sc in zip(axes, scen, strict=True):
        g = mean[mean.scenario == sc]
        for p in pols:
            gp = g[g.policy == p]
            ax.plot(gp.t, gp.auc, label=p, lw=1.8 if p in (unsup, sup) else 1.4, **style(p))
        if sc != "none":
            ax.axvline(c["onset"], color=MUTED, lw=0.8)
        ax.set_title(pretty(sc))
    for ax in axes[6:]:
        ax.set_xlabel("batch")
    for ax in axes[::3]:
        ax.set_ylabel("ROC-AUC (mean of 50 runs)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Model performance over time under each retraining policy "
                 "(vertical line = drift onset)", x=0.01, ha="left", fontsize=11)  # fmt: skip
    fig.tight_layout()
    save(fig, "fig06_retraining_timeseries")


def fig_tradeoff(overall: pd.DataFrame, xcol: str, ycol: str, xlabel, ylabel, title, name):
    fig = Figure(figsize=(7, 4.6))
    ax = fig.subplots()
    for fam, colr in FAMILY_COLOR.items():
        sub = overall[[family(p) == fam for p in overall.index]]
        ax.scatter(sub[xcol], sub[ycol], s=70, color=colr, edgecolor="white", lw=1.5,
                   label=fam, zorder=3)  # fmt: skip
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="center right")
    fig.tight_layout()
    place_labels(fig, ax, overall, xcol, ycol)
    save(fig, name)


def place_labels(fig, ax, df: pd.DataFrame, xcol: str, ycol: str) -> None:
    """Greedy label placement: try a few offsets per point, keep the first that overlaps
    neither an earlier label nor any marker."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()
    pts = ax.transData.transform(df[[xcol, ycol]].to_numpy())
    offsets = [(dx, dy, "left" if dx > 0 else "right")
               for dy in (4, -11, 14, -21) for dx in (6, -6)]  # fmt: skip
    placed = []
    order = np.argsort(-pts[:, 1])  # top to bottom
    for i in order:
        name = df.index[i]
        for dx, dy, ha in offsets:
            t = ax.annotate(name, df[[xcol, ycol]].iloc[i], xytext=(dx, dy),
                            textcoords="offset points", fontsize=8, color=TEXT, ha=ha)  # fmt: skip
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            hits_label = any(bb.overlaps(o) for o in placed)
            hits_dot = any(bb.contains(x, y) for j, (x, y) in enumerate(pts) if j != i)
            if not (hits_label or hits_dot):
                placed.append(bb)
                break
            t.remove()
        else:  # nothing free: fall back to the default position
            placed.append(
                ax.annotate(
                    name,
                    df[[xcol, ycol]].iloc[i],
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=TEXT,
                ).get_window_extent(renderer)  # fmt: skip
            )


def fig_label_delay() -> None:
    df = pd.read_csv(TABLES / "exp2b_label_delay_gain.csv", header=[0, 1], index_col=0)
    scen = list(dict.fromkeys(df.columns.get_level_values(0)))
    fig = Figure(figsize=(12, 3.6))
    axes = fig.subplots(1, len(scen), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scen, strict=True):
        sub = df[sc]
        delays = sub.columns.astype(int)
        for p in ["periodic", "oracle"] + NAMES:
            if p in sub.index:
                ax.plot(delays, sub.loc[p], marker="o", ms=4, label=p, **style(p))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(pretty(sc))
        ax.set_xlabel("label delay (batches)")
        ax.set_xticks(delays)
    np.atleast_1d(axes)[0].set_ylabel("AUC gain vs. never retraining")
    np.atleast_1d(axes)[-1].legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig08_label_delay")


# ============================================================================ Exp 4
def fig_electricity(el: pd.DataFrame, overall: pd.DataFrame) -> None:
    det = overall.drop(index=[p for p in ("static", "periodic", "always") if p in overall.index])
    unsup = det[~det.index.isin(SUPERVISED)]["accuracy"].idxmax()
    sup = det[det.index.isin(SUPERVISED)]["accuracy"].idxmax()
    pols = ["static", "periodic", "always", unsup, sup]
    m = el.groupby(["policy", "week"])["accuracy"].mean().reset_index()
    fig = Figure(figsize=(12, 4.6))
    ax, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for p in pols:
        g = m[m.policy == p].sort_values("week")
        ax.plot(g.week, g.accuracy.rolling(4, min_periods=1).mean(), label=p, **style(p))
    ax.set_ylabel("accuracy (4-week rolling)")
    ax.legend(ncol=5, loc="lower left", fontsize=8)
    ax.set_title("Electricity (real drift): prequential accuracy", loc="left")
    r = el[(el.seed == 0) & (el.policy.isin([unsup, sup])) & el.retrained]
    for k, p in enumerate([unsup, sup]):
        w = r[r.policy == p].week
        ax2.scatter(w, np.full(len(w), k), marker="|", s=120, color=DET_COLOR[p])
    ax2.set_yticks([0, 1], [unsup, sup])
    ax2.set_ylim(-0.7, 1.7)
    ax2.set_xlabel("week")
    ax2.set_title("retraining events (seed 0)", loc="left", fontsize=9, fontweight="normal")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig09_electricity_timeseries")


def main() -> None:
    import json

    c = cfg()["telco"]
    summary = json.loads((RESULTS / "summary.json").read_text())
    det = pd.read_csv(RESULTS / "detection.csv.gz")
    thr = pd.read_csv(RESULTS / "thresholds.csv")
    fig_example_scores(det, thr, c)
    fig_detection_heatmaps()
    fig_false_alarms()
    fig_drift_harm()
    fig_sensitivity(pd.read_csv(RESULTS / "sensitivity_runs.csv.gz"))
    fr = summary["exp1_friedman_detection_auc"]
    fig_cd(fr["avg_rank"], fr["nemenyi_cd"], "Detection AUC: average rank (Friedman + Nemenyi)",
           "fig07a_cd_detection")  # fmt: skip

    ret = pd.read_csv(RESULTS / "retraining.csv.gz")
    overall = pd.read_csv(TABLES / "exp2_retraining_overall.csv", index_col=0)
    fig_retraining_timeseries(ret, overall, c)
    fig_tradeoff(
        overall.drop(index="oracle"), "retrains", "gain_harmful_drift",
        "retrains per stream (mean, all scenarios)",
        "AUC gain vs. static on harmful drift",
        "Retraining benefit vs. cost (Telco)", "fig07_tradeoff_telco",
    )  # fmt: skip
    fr = summary["exp2_friedman_auc_post"]
    fig_cd(fr["avg_rank"], fr["nemenyi_cd"], "Post-drift AUC: average rank of retraining policies",
           "fig07b_cd_retraining")  # fmt: skip
    fig_label_delay()

    el = pd.read_csv(RESULTS / "electricity.csv.gz")
    eo = pd.read_csv(TABLES / "exp4_electricity_overall.csv", index_col=0)
    fig_electricity(el, eo)
    static_acc = eo.loc["static", "accuracy"]
    fig_tradeoff(
        eo.assign(gain=eo["accuracy"] - static_acc), "retrains", "gain",
        "retrains over 2.5 years", "accuracy gain vs. static",
        "Retraining benefit vs. cost (Electricity)", "fig10_tradeoff_electricity",
    )  # fmt: skip
    fr = summary["exp4_friedman_accuracy"]
    fig_cd(fr["avg_rank"], fr["nemenyi_cd"], "Electricity: weekly accuracy rank",
           "fig10b_cd_electricity")  # fmt: skip
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
