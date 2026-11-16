"""Ship / don't-ship decision rules.

No statistics are computed here. This module only reads results produced by
srm.py, guardrails.py and metrics.py and applies the rules in order.
"""

from __future__ import annotations

from readout.schema import Decision, EffectResult, GuardrailResult, SRMResult

SHIP = "SHIP"
DONT_SHIP = "DON'T SHIP"
INCONCLUSIVE = "INCONCLUSIVE"
INVALID = "INVALID"


def decide(
    srm: SRMResult,
    guardrails: list[GuardrailResult],
    primary: EffectResult,
    min_practical_lift: float,
) -> Decision:
    """Apply the decision rules; the first rule that matches wins.

    1. SRM fails                                  -> INVALID
    2. Any guardrail fails                        -> DON'T SHIP
    3. Primary significant and positive:
         lift >= min_practical_lift               -> SHIP
         lift <  min_practical_lift               -> INCONCLUSIVE (real but too small)
    4. Primary significant and negative           -> DON'T SHIP
    5. Otherwise                                  -> INCONCLUSIVE

    The primary metric is assumed to be "higher is better". `significant`
    already reflects the Holm correction when several arms were compared.
    """
    if not srm.passed:
        return Decision(
            INVALID,
            [
                f"Sample ratio mismatch: chi2={srm.chi2:.2f}, p={srm.p_value:.2g} "
                f"(threshold {srm.threshold}). Investigate assignment and logging "
                "before reading any result."
            ],
        )

    failed = [g for g in guardrails if not g.passed]
    if failed:
        return Decision(DONT_SHIP, [_guardrail_reason(g) for g in failed])

    lift = _lift_text(primary)
    if primary.significant and primary.abs_diff > 0:
        if primary.rel_lift >= min_practical_lift:
            reasons = [
                f"Primary metric improved: {lift}.",
                f"Lift is at or above the minimum practical effect of {min_practical_lift:+.1%}.",
                *_passed_reasons(srm, guardrails),
            ]
            return Decision(SHIP, reasons)
        return Decision(
            INCONCLUSIVE,
            [
                f"Primary metric improved: {lift}.",
                "Statistically significant but below the practical threshold of "
                f"{min_practical_lift:+.1%}: the effect is real but may not be worth shipping.",
            ],
        )

    if primary.significant and primary.abs_diff < 0:
        return Decision(DONT_SHIP, [f"Primary metric got worse: {lift}."])

    low, high = primary.rel_ci
    return Decision(
        INCONCLUSIVE,
        [
            f"Primary metric not significant: {lift}.",
            f"The data rule out lifts above {high:+.1%} and below {low:+.1%}; "
            "anything in between is still possible.",
        ],
    )


def _lift_text(effect: EffectResult) -> str:
    low, high = effect.rel_ci
    return (
        f"lift {effect.rel_lift:+.1%} ({effect.ci_level:.1%} CI {low:+.1%} to {high:+.1%}, "
        f"p={effect.p_adjusted:.3g})"
    )


def _guardrail_reason(g: GuardrailResult) -> str:
    unit = "%" if g.margin_type == "relative" else ""
    scale = 100 if g.margin_type == "relative" else 1
    bound = f"{g.lower_bound * scale:+.2f}{unit}"
    margin = f"{-g.margin * scale:+.2f}{unit}"
    if g.status == "fail_harm_detected":
        what = "harm detected"
    else:
        what = "harm not ruled out (likely too little data)"
    return (
        f"Guardrail '{g.metric}' failed, {what}: lower bound {bound} is not above "
        f"the margin {margin}."
    )


def _passed_reasons(srm: SRMResult, guardrails: list[GuardrailResult]) -> list[str]:
    reasons = [f"No sample ratio mismatch (p={srm.p_value:.3g})."]
    if guardrails:
        names = ", ".join(g.metric for g in guardrails)
        reasons.append(f"All guardrails passed: {names}.")
    return reasons
