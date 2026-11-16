"""ExperimentData dataclass: the common format every loader produces.

Also holds the small result dataclasses that the statistics functions return,
so every module shares one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEVELS = ("user", "summary", "aggregated")


@dataclass(frozen=True)
class ExperimentData:
    """One experiment in the common format.

    df has one row per randomisation unit (for level="user"). Downstream code
    reads only the columns named here and never asks which dataset it is.
    """

    name: str
    level: str
    df: pd.DataFrame
    arm_col: str
    control_arm: str
    intended_allocation: dict[str, float]
    metrics: list[str]
    covariates: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")
        needed = [self.arm_col, *self.metrics, *self.covariates, *self.segments]
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise ValueError(f"{self.name}: columns missing from df: {missing}")
        if abs(sum(self.intended_allocation.values()) - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: intended allocation must sum to 1")
        if self.control_arm not in self.intended_allocation:
            raise ValueError(f"{self.name}: control arm {self.control_arm!r} has no allocation")
        unknown = set(self.df[self.arm_col].unique()) - set(self.intended_allocation)
        if unknown:
            raise ValueError(f"{self.name}: arms in data with no intended allocation: {unknown}")

    @property
    def treatment_arms(self) -> list[str]:
        return [a for a in self.intended_allocation if a != self.control_arm]

    def arm_counts(self) -> dict[str, int]:
        """Observed number of units per arm, in allocation order."""
        arms = self.df[self.arm_col]
        return {a: int((arms == a).sum()) for a in self.intended_allocation}

    def values(self, col: str, arm: str) -> np.ndarray:
        """Values of one column for the units in one arm."""
        return self.df.loc[self.df[self.arm_col] == arm, col].to_numpy(dtype=float)


@dataclass(frozen=True)
class NotApplicable:
    """Explicit result for a feature a dataset cannot support."""

    feature: str
    reason: str


@dataclass(frozen=True)
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    dof: int
    p_value: float
    threshold: float
    passed: bool  # True when p_value >= threshold (no mismatch detected)


@dataclass(frozen=True)
class ArmSummary:
    """A point estimate for one arm and the variance OF THAT ESTIMATE.

    For a mean: estimate = ybar, variance = s^2 / n. For a ratio metric the
    variance comes from the delta method. Everything downstream only needs
    these three numbers.
    """

    estimate: float
    variance: float
    n: int


@dataclass(frozen=True)
class EffectResult:
    control_estimate: float
    treatment_estimate: float
    n_control: int
    n_treatment: int
    abs_diff: float
    abs_se: float
    abs_ci: tuple[float, float]
    rel_lift: float  # nan when the control estimate is 0
    rel_se: float
    rel_ci: tuple[float, float]
    z: float
    p_value: float  # raw two-sided p-value of the absolute difference
    p_adjusted: float  # Holm-adjusted when several arms are compared, else = p_value
    alpha: float  # family-wise alpha
    ci_level: float  # 1 - alpha / n_comparisons
    significant: bool  # p_adjusted < alpha


@dataclass(frozen=True)
class CupedResult:
    raw: EffectResult
    adjusted: EffectResult
    covariates: tuple[str, ...]
    theta: tuple[float, ...]
    variance_reduction: float  # 1 - Var(adjusted diff) / Var(raw diff)


@dataclass(frozen=True)
class GuardrailResult:
    metric: str
    margin: float
    margin_type: str  # "relative" or "absolute"
    higher_is_better: bool
    estimate: float  # effect in the "good" direction (sign flipped if lower is better)
    lower_bound: float  # one-sided lower confidence bound of that effect
    upper_bound: float  # one-sided upper confidence bound of that effect
    p_value: float  # non-inferiority p-value
    alpha: float
    status: str  # "pass", "fail_harm_detected", "fail_harm_not_ruled_out"

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class Decision:
    label: str  # "SHIP", "DON'T SHIP", "INCONCLUSIVE", "INVALID"
    reasons: list[str]
