"""Drift detectors under one interface.

Every detector is fitted on a *reference* window (held-out data the model was validated on)
and then returns a scalar drift **score** for each new batch - higher means more drift.
Each also has a `default_threshold` (the rule a practitioner would use out-of-the-box);
in the experiments thresholds are additionally *calibrated* to a common false-alarm rate,
so detectors are compared on equal terms.

Unsupervised (input / output only - no labels needed):
  PSI         max Population Stability Index over features        (industry practice)
  KS          per-feature KS (numeric) / chi-squared (categorical), Bonferroni-corrected
  MMD         kernel Maximum Mean Discrepancy on all features jointly (Gretton et al. 2012)
  C2ST        classifier two-sample test / domain classifier (Lopez-Paz & Oquab 2017)
  PredDrift   KS test on the model's predicted probabilities
Supervised (need ground-truth labels, which arrive with a delay):
  PerfDrop    drop in ROC-AUC on a labelled batch vs. the reference
  DDM         Drift Detection Method on the error stream (Gama et al. 2004)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from driftlab.datasets import make_encoder


@dataclass
class Batch:
    X: pd.DataFrame
    y: np.ndarray | None = None
    proba: np.ndarray | None = None


@dataclass
class Spec:
    numeric: list[str]
    categorical: list[str]
    batch_size: int = 200


def _neglog10(p: float) -> float:
    return float(-np.log10(max(p, 1e-300)))


class Detector(ABC):
    name: str = "base"
    supervised: bool = False
    default_threshold: float = 0.0

    def __init__(self, spec: Spec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def fit(self, ref: Batch) -> "Detector": ...

    @abstractmethod
    def score(self, batch: Batch) -> float: ...


# ----------------------------------------------------------------------------- PSI
class PSIDetector(Detector):
    name, default_threshold = "PSI", 0.2  # 0.2 = conventional "significant shift" cut-off
    eps = 1e-4

    def fit(self, ref):
        self.edges, self.ref_pct, self.cats = {}, {}, {}
        for c in self.spec.numeric:
            e = np.unique(np.quantile(ref.X[c], np.linspace(0, 1, 11)))
            e[0], e[-1] = -np.inf, np.inf
            self.edges[c] = e
            self.ref_pct[c] = np.histogram(ref.X[c], e)[0] / len(ref.X)
        for c in self.spec.categorical:
            vc = ref.X[c].value_counts(normalize=True)
            self.cats[c] = list(vc.index)
            self.ref_pct[c] = np.append(vc.to_numpy(), 0.0)  # last bucket = unseen categories
        return self

    def _psi(self, r, q):
        r, q = np.clip(r, self.eps, None), np.clip(q, self.eps, None)
        return float(np.sum((q - r) * np.log(q / r)))

    def feature_scores(self, batch) -> dict[str, float]:
        out = {}
        for c in self.spec.numeric:
            q = np.histogram(batch.X[c], self.edges[c])[0] / len(batch.X)
            out[c] = self._psi(self.ref_pct[c], q)
        for c in self.spec.categorical:
            vc = batch.X[c].value_counts(normalize=True)
            q = np.append(vc.reindex(self.cats[c], fill_value=0).to_numpy(),
                          vc.drop(self.cats[c], errors="ignore").sum())  # fmt: skip
            out[c] = self._psi(self.ref_pct[c], q)
        return out

    def score(self, batch):
        return max(self.feature_scores(batch).values())


# ----------------------------------------------------------------------------- KS / chi2
class KSDetector(Detector):
    name, default_threshold = "KS", _neglog10(0.05)

    def fit(self, ref):
        self.ref = ref.X.reset_index(drop=True)
        return self

    def feature_pvalues(self, batch) -> dict[str, float]:
        out = {}
        for c in self.spec.numeric:
            out[c] = stats.ks_2samp(self.ref[c], batch.X[c]).pvalue
        for c in self.spec.categorical:
            table = pd.concat(
                [self.ref[c].value_counts(), batch.X[c].value_counts()], axis=1
            ).fillna(0)
            table = table[(table.sum(axis=1) > 0)].to_numpy().T
            out[c] = 1.0 if table.shape[1] < 2 else stats.chi2_contingency(table)[1]
        return out

    def score(self, batch):
        p = self.feature_pvalues(batch)
        return _neglog10(min(1.0, min(p.values()) * len(p)))  # Bonferroni


# ----------------------------------------------------------------------------- MMD
class MMDDetector(Detector):
    """Biased MMD^2 with an RBF kernel (median-heuristic bandwidth fixed on the reference).
    Default threshold = 95th percentile of MMD^2 between random reference sub-samples."""

    name = "MMD"
    n_ref = 300

    def fit(self, ref):
        self.enc = make_encoder(self.spec).fit(ref.X)
        Z = self.enc.transform(ref.X)
        self.Zr = Z[self.rng.choice(len(Z), min(self.n_ref, len(Z)), replace=False)]
        d = _sqdist(self.Zr, self.Zr)
        self.gamma = 1.0 / np.median(d[d > 0])
        self.Krr = np.exp(-self.gamma * d).mean()
        # null distribution: reference vs bootstrap batches of the reference itself
        null = []
        for _ in range(100):
            Zb = Z[self.rng.choice(len(Z), self.spec.batch_size)]
            null.append(self._mmd(Zb))
        self.default_threshold = float(np.quantile(null, 0.95))
        return self

    def _mmd(self, Zb):
        kbb = np.exp(-self.gamma * _sqdist(Zb, Zb)).mean()
        krb = np.exp(-self.gamma * _sqdist(self.Zr, Zb)).mean()
        return float(self.Krr + kbb - 2 * krb)

    def score(self, batch):
        return self._mmd(self.enc.transform(batch.X))


def _sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


# ----------------------------------------------------------------------------- C2ST
class C2STDetector(Detector):
    """Domain classifier: can a random forest tell reference rows from batch rows?
    Score = out-of-bag ROC-AUC (0.5 = indistinguishable). Default threshold is the one-sided
    95 % bound of the AUC under H0 (Mann-Whitney normal approximation)."""

    name = "C2ST"
    n_ref = 300

    def fit(self, ref):
        self.enc = make_encoder(self.spec).fit(ref.X)
        Z = self.enc.transform(ref.X)
        self.Zr = Z[self.rng.choice(len(Z), min(self.n_ref, len(Z)), replace=False)]
        n1, n2 = len(self.Zr), self.spec.batch_size
        self.default_threshold = 0.5 + 1.645 * np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        return self

    def score(self, batch):
        Zb = self.enc.transform(batch.X)
        Z = np.vstack([self.Zr, Zb])
        lab = np.r_[np.zeros(len(self.Zr)), np.ones(len(Zb))]
        rf = RandomForestClassifier(
            n_estimators=60,
            max_depth=6,
            min_samples_leaf=5,
            oob_score=True,
            n_jobs=1,
            random_state=int(self.rng.integers(1 << 31)),
        ).fit(Z, lab)
        oob = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=0.5)
        return float(roc_auc_score(lab, oob))


# ----------------------------------------------------------------------------- prediction drift
class PredictionDriftDetector(Detector):
    name, default_threshold = "PredDrift", _neglog10(0.05)

    def fit(self, ref):
        self.ref_proba = np.asarray(ref.proba)
        return self

    def score(self, batch):
        return _neglog10(stats.ks_2samp(self.ref_proba, batch.proba).pvalue)


# ----------------------------------------------------------------------------- performance drop
class PerformanceDropDetector(Detector):
    name, supervised, default_threshold = "PerfDrop", True, 0.05  # 5-point AUC drop

    def fit(self, ref):
        self.ref_auc = roc_auc_score(ref.y, ref.proba)
        return self

    def score(self, batch):
        if len(np.unique(batch.y)) < 2:
            return 0.0
        return float(self.ref_auc - roc_auc_score(batch.y, batch.proba))


# ----------------------------------------------------------------------------- DDM
class DDMDetector(Detector):
    """Gama et al. (2004). Tracks the online error rate p and its std s; signals drift when
    p + s >= p_min + 3 s_min. Score = (p + s - p_min) / s_min, i.e. 'sigmas above best'."""

    name, supervised, default_threshold = "DDM", True, 3.0
    min_samples = 30

    def fit(self, ref):
        self.n, self.err = 0, 0
        self.p_min, self.s_min = np.inf, np.inf
        return self

    def score(self, batch):
        errors = ((batch.proba >= 0.5).astype(int) != batch.y).astype(int)
        best = 0.0
        for e in errors:
            self.n += 1
            self.err += e
            p = self.err / self.n
            s = np.sqrt(p * (1 - p) / self.n)
            if self.n < self.min_samples:
                continue
            if p + s <= self.p_min + self.s_min:
                self.p_min, self.s_min = p, max(s, 1e-6)
            best = max(best, (p + s - self.p_min) / self.s_min)
        return float(best)


DETECTORS: dict[str, type[Detector]] = {
    d.name: d
    for d in (
        PSIDetector,
        KSDetector,
        MMDDetector,
        C2STDetector,
        PredictionDriftDetector,
        PerformanceDropDetector,
        DDMDetector,
    )
}
UNSUPERVISED = [n for n, d in DETECTORS.items() if not d.supervised]
SUPERVISED = [n for n, d in DETECTORS.items() if d.supervised]


def make_detector(name: str, spec: Spec, seed: int = 0) -> Detector:
    return DETECTORS[name](spec, seed)
