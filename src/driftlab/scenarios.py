"""Drift injection for the Telco stream. Each scenario defines how a *drifted* record is
generated; a batch at drift intensity alpha is a mixture of (1-alpha) clean and alpha drifted
records, which gives sudden (alpha jumps 0->1) and gradual (alpha ramps) drift.

Taxonomy (Gama et al. 2014; Moreno-Torres et al. 2012):
  none           control stream, no drift                               -> false-alarm baseline
  subpopulation  covariate shift via sampling: P(X) changes, P(y|X) kept -> "virtual" drift
  inflation      feature shift: charges inflate, behaviour does not       -> covariate shift that
                 moves inputs out of the learned mapping
  concept        real concept drift: P(y|X) changes, P(X) kept            -> invisible to
                 unsupervised detectors. The tenure effect reverses: a competitor poaches
                 long-tenure customers while an onboarding programme retains new ones
  sensor_bug     data-quality incident: tenure logged in days, not months -> upstream pipeline bug
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCENARIO_KINDS = ("none", "subpopulation", "inflation", "concept", "sensor_bug")

DEFAULT_SEVERITY = {
    "none": 0.0,
    "subpopulation": 1.5,  # log-odds boost for month-to-month / new customers
    "inflation": 0.25,  # +25 % on monthly and total charges
    "concept": 0.5,  # probability that an affected customer's outcome flips
    "sensor_bug": 1.0,  # fraction of drifted records with the bug
}


@dataclass(frozen=True)
class Scenario:
    kind: str
    mode: str = "sudden"  # sudden | gradual
    severity: float | None = None

    def __post_init__(self):
        if self.kind not in SCENARIO_KINDS:
            raise ValueError(f"unknown scenario {self.kind!r}")

    @property
    def s(self) -> float:
        return DEFAULT_SEVERITY[self.kind] if self.severity is None else self.severity

    @property
    def name(self) -> str:
        return self.kind if self.kind == "none" else f"{self.kind}-{self.mode}"

    # -- drifted record generators -------------------------------------------------------
    def drifted(self, X: pd.DataFrame, y: np.ndarray, n: int, rng: np.random.Generator):
        if self.kind == "subpopulation":
            score = (X["Contract"] == "Month-to-month").to_numpy() + (X["tenure"] < 12).to_numpy()
            w = np.exp(self.s * score)
            idx = rng.choice(len(X), size=n, p=w / w.sum())
            return X.iloc[idx].reset_index(drop=True), y[idx].copy()

        idx = rng.integers(0, len(X), size=n)
        Xd, yd = X.iloc[idx].reset_index(drop=True).copy(), y[idx].copy()
        if self.kind == "inflation":
            Xd["MonthlyCharges"] *= 1 + self.s
            Xd["TotalCharges"] *= 1 + self.s
        elif self.kind == "concept":
            loyal = (Xd["tenure"] > 24).to_numpy()
            new = (Xd["tenure"] < 12).to_numpy()
            flip = rng.random(n) < self.s
            yd[loyal & (yd == 0) & flip] = 1  # competitor poaches long-tenure customers
            yd[new & (yd == 1) & flip] = 0  # onboarding programme retains new customers
        elif self.kind == "sensor_bug":
            bug = rng.random(n) < self.s
            Xd.loc[bug, "tenure"] = Xd.loc[bug, "tenure"] * 30
        return Xd, yd


def intensity(t: int, onset: int, ramp: int, mode: str) -> float:
    if t < onset:
        return 0.0
    if mode == "sudden" or ramp <= 1:
        return 1.0
    return min(1.0, (t - onset + 1) / ramp)


def sample_batch(X, y, n, alpha, scenario: Scenario, rng):
    k = 0 if scenario.kind == "none" else int(rng.binomial(n, alpha))
    idx = rng.integers(0, len(X), size=n - k)
    Xc, yc = X.iloc[idx].reset_index(drop=True), y[idx]
    if k == 0:
        return Xc, yc
    Xd, yd = scenario.drifted(X, y, k, rng)
    Xb = pd.concat([Xc, Xd], ignore_index=True)
    yb = np.concatenate([yc, yd])
    perm = rng.permutation(n)
    return Xb.iloc[perm].reset_index(drop=True), yb[perm]


def generate_stream(X, y, scenario: Scenario, n_batches, batch_size, onset, ramp, rng):
    """Returns a list of (X_batch, y_batch, alpha)."""
    out = []
    for t in range(n_batches):
        a = intensity(t, onset, ramp, scenario.mode)
        Xb, yb = sample_batch(X, y, batch_size, a, scenario, rng)
        out.append((Xb, yb, a))
    return out
