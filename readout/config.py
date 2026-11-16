"""Statistical constants and per-dataset settings. No statistics here."""

ALPHA = 0.05  # two-sided, primary metrics
SRM_THRESHOLD = 0.001  # SRM is flagged when p < this
GUARDRAIL_ALPHA = 0.05  # one-sided, non-inferiority

# Guardrail margins per dataset and metric.
# margin_type "relative": margin is a fraction of the control mean (0.10 = 10%).
# margin_type "absolute": margin is in the metric's own units.
GUARDRAILS = {
    "hillstrom": {
        "spend": {"margin": 0.10, "margin_type": "relative", "higher_is_better": True},
    },
}

# Smallest relative lift on the primary metric worth shipping for.
MIN_PRACTICAL_LIFT = {
    "hillstrom": 0.05,
}
