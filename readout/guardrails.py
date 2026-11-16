"""Guardrail metrics: one-sided non-inferiority tests.

A guardrail is a metric the change must not damage (revenue, latency,
unsubscribes). We do not need it to improve; we need evidence that it did
not get worse by more than a margin we chose in advance.
"""

from __future__ import annotations

from scipy import stats

from readout.config import GUARDRAIL_ALPHA
from readout.schema import EffectResult, GuardrailResult


def guardrail_check(
    metric: str,
    effect: EffectResult,
    margin: float,
    margin_type: str = "relative",
    higher_is_better: bool = True,
    alpha: float = GUARDRAIL_ALPHA,
) -> GuardrailResult:
    """Non-inferiority test of one guardrail metric.

    Let delta be the effect in the "good" direction: the relative lift
    (margin_type="relative") or the absolute difference ("absolute"), with
    the sign flipped when lower values are better. With margin m > 0:

        H0: delta <= -m   (harm of at least the margin)
        H1: delta >  -m   (not worse than the margin)

        z = (delta_hat + m) / SE,    p = 1 - Phi(z)
        lower = delta_hat - z_(1-alpha) * SE
        upper = delta_hat + z_(1-alpha) * SE

    The guardrail passes when p < alpha, which is the same as lower > -m.
    The burden of proof is on the change: with too little data the test
    fails even if nothing is wrong. The status separates the two failures:

        fail_harm_detected       upper < 0: the metric got significantly worse
        fail_harm_not_ruled_out  not enough evidence either way
    """
    if margin_type == "relative":
        estimate, se = effect.rel_lift, effect.rel_se
    elif margin_type == "absolute":
        estimate, se = effect.abs_diff, effect.abs_se
    else:
        raise ValueError(f"margin_type must be 'relative' or 'absolute', got {margin_type!r}")
    if not higher_is_better:
        estimate = -estimate

    z_crit = stats.norm.ppf(1 - alpha)
    lower = estimate - z_crit * se
    upper = estimate + z_crit * se
    p_value = stats.norm.sf((estimate + margin) / se)

    if lower > -margin:
        status = "pass"
    elif upper < 0:
        status = "fail_harm_detected"
    else:
        status = "fail_harm_not_ruled_out"

    return GuardrailResult(
        metric=metric,
        margin=margin,
        margin_type=margin_type,
        higher_is_better=higher_is_better,
        estimate=float(estimate),
        lower_bound=float(lower),
        upper_bound=float(upper),
        p_value=float(p_value),
        alpha=alpha,
        status=status,
    )
