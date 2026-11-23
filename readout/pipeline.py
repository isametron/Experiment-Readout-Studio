"""One entry point that turns an ExperimentData into a full readout.

The app, the BI export and the scripts call run_readout and only display
what it returns. This module computes nothing itself: it calls srm,
metrics, variance_reduction, guardrails and decision in a fixed order.

Order:
  1. SRM over all arms (the gate)
  2. Primary metric: every treatment arm vs control, Holm-adjusted
  3. CUPED on the primary metric, if the dataset has covariates
  4. Guardrails for every treatment arm
  5. Decision per treatment arm
"""

from __future__ import annotations

from dataclasses import dataclass, field

from readout import config
from readout.decision import decide
from readout.guardrails import guardrail_check
from readout.metrics import compare_all_to_control, mean_summary
from readout.schema import (
    CupedResult,
    Decision,
    EffectResult,
    ExperimentData,
    GuardrailResult,
    NotApplicable,
    SRMResult,
)
from readout.srm import srm_check
from readout.variance_reduction import cuped_compare


@dataclass(frozen=True)
class ReadoutSpec:
    """What the analysis commits to before looking at results.

    Writing this down first is a light form of pre-registration: the
    primary metric, guardrails and the practical threshold cannot be
    changed after seeing which way the data went.
    """

    primary_metric: str
    min_practical_lift: float
    guardrails: dict[str, dict] = field(default_factory=dict)
    alpha: float = config.ALPHA


@dataclass(frozen=True)
class Readout:
    experiment: str
    spec: ReadoutSpec
    srm: SRMResult
    primary: dict[str, EffectResult]  # treatment arm -> Holm-adjusted effect
    cuped: dict[str, CupedResult] | NotApplicable
    guardrails: dict[str, list[GuardrailResult]]  # treatment arm -> results
    decisions: dict[str, Decision]  # treatment arm -> decision


def spec_from_config(name: str) -> ReadoutSpec:
    """Build the spec for a dataset from readout/config.py."""
    if name not in config.PRIMARY_METRIC:
        raise KeyError(f"No readout settings for {name!r} in readout/config.py")
    return ReadoutSpec(
        primary_metric=config.PRIMARY_METRIC[name],
        min_practical_lift=config.MIN_PRACTICAL_LIFT[name],
        guardrails=config.GUARDRAILS.get(name, {}),
    )


def run_readout(exp: ExperimentData, spec: ReadoutSpec | None = None) -> Readout:
    """Run every check and test on one experiment and decide per arm.

    The decision uses the raw (unadjusted) primary effect so that it does
    not depend on which covariates a dataset happens to have. CUPED is
    reported next to it as a sharper estimate of the same effect.
    """
    spec = spec or spec_from_config(exp.name)
    _check_metrics_exist(exp, spec)

    srm = srm_check(exp.arm_counts(), exp.intended_allocation)
    primary = _compare_metric(exp, spec.primary_metric, spec.alpha)
    cuped = _cuped(exp, spec)
    guardrails = {arm: [] for arm in exp.treatment_arms}
    for metric, settings in spec.guardrails.items():
        effects = _compare_metric(exp, metric, spec.alpha)
        for arm in exp.treatment_arms:
            guardrails[arm].append(guardrail_check(metric, effects[arm], **settings))

    decisions = {
        arm: decide(srm, guardrails[arm], primary[arm], spec.min_practical_lift)
        for arm in exp.treatment_arms
    }
    return Readout(exp.name, spec, srm, primary, cuped, guardrails, decisions)


def _compare_metric(exp: ExperimentData, metric: str, alpha: float) -> dict[str, EffectResult]:
    """Every treatment arm vs control on one metric, Holm-adjusted."""
    treatments = {arm: mean_summary(exp.values(metric, arm)) for arm in exp.treatment_arms}
    control = mean_summary(exp.values(metric, exp.control_arm))
    return compare_all_to_control(treatments, control, alpha)


def _cuped(exp: ExperimentData, spec: ReadoutSpec) -> dict[str, CupedResult] | NotApplicable:
    if not exp.covariates:
        return NotApplicable("cuped", f"{exp.name} has no pre-experiment covariates")
    df = exp.df
    n_comparisons = len(exp.treatment_arms)
    return {
        arm: cuped_compare(
            df[spec.primary_metric].to_numpy(),
            df[exp.covariates].to_numpy(),
            df[exp.arm_col].to_numpy(),
            treatment=arm,
            control=exp.control_arm,
            covariates=tuple(exp.covariates),
            alpha=spec.alpha,
            n_comparisons=n_comparisons,
        )
        for arm in exp.treatment_arms
    }


def _check_metrics_exist(exp: ExperimentData, spec: ReadoutSpec) -> None:
    """A misnamed metric is a config error: fail loudly, never skip."""
    wanted = [spec.primary_metric, *spec.guardrails]
    missing = [m for m in wanted if m not in exp.df.columns]
    if missing:
        raise ValueError(f"{exp.name}: spec names metrics not in the data: {missing}")
