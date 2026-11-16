"""Guardrail non-inferiority tests on simulated data with known effects.

Simulated control mean is 10, so a relative margin of 2% is 0.2 in absolute
terms.
"""

import pytest

from readout.guardrails import guardrail_check
from readout.metrics import compare_arms, mean_summary
from readout.schema import ArmSummary
from sim.simulate import simulate_experiment

N_RUNS = 1000
MARGIN = 0.02


def _status(n, true_effect, seed, **kwargs):
    exp = simulate_experiment(n=n, true_effect=true_effect, seed=seed)
    effect = compare_arms(
        mean_summary(exp.values("y", "treatment")), mean_summary(exp.values("y", "control"))
    )
    return guardrail_check("y", effect, margin=MARGIN, **kwargs).status


def test_hand_calculation():
    # lift = -1%, SE(lift) = sqrt(0.01/100 + 9.9^2*0.01/10^4) ~ 0.01407
    effect = compare_arms(ArmSummary(9.9, 0.01, 100), ArmSummary(10.0, 0.01, 100))
    g = guardrail_check("y", effect, margin=0.05)
    assert g.estimate == pytest.approx(-0.01)
    assert g.lower_bound == pytest.approx(-0.01 - 1.6449 * effect.rel_se, abs=1e-5)
    assert g.passed  # lower bound ~ -3.3% is above -5%
    assert not guardrail_check("y", effect, margin=0.02).passed


def test_type_one_error_at_the_margin_is_five_percent():
    # True harm exactly equal to the margin: passing is a mistake, and it
    # should happen about alpha = 5% of the time.
    passes = sum(_status(4000, -0.2, s) == "pass" for s in range(N_RUNS))
    assert 0.03 <= passes / N_RUNS <= 0.07


def test_no_harm_with_enough_data_passes():
    passes = sum(_status(4000, 0.0, s) == "pass" for s in range(200))
    assert passes / 200 > 0.99


def test_harm_beyond_margin_is_detected():
    statuses = [_status(4000, -0.5, s) for s in range(200)]
    assert statuses.count("fail_harm_detected") / 200 > 0.99


def test_too_little_data_fails_as_not_ruled_out():
    # 10 users per arm: SE(lift) ~ 4.5%, so passing needs an observed lift above
    # about +5.4% (-2% + 1.645 * 4.5%), which happens ~12% of the time when
    # nothing is wrong. Most runs cannot rule out a 2% drop.
    statuses = [_status(20, 0.0, s) for s in range(200)]
    assert statuses.count("fail_harm_not_ruled_out") / 200 > 0.7


def test_lower_is_better_flips_the_direction():
    # The metric rises by 5%. Good if higher is better, harm if lower is better.
    assert _status(4000, 0.5, 0, higher_is_better=True) == "pass"
    assert _status(4000, 0.5, 0, higher_is_better=False) == "fail_harm_detected"


def test_absolute_margin_uses_absolute_difference():
    effect = compare_arms(ArmSummary(9.9, 0.0001, 100), ArmSummary(10.0, 0.0001, 100))
    g = guardrail_check("y", effect, margin=0.2, margin_type="absolute")
    assert g.estimate == pytest.approx(-0.1)
    assert g.passed


def test_unknown_margin_type_raises():
    effect = compare_arms(ArmSummary(9.9, 0.01, 100), ArmSummary(10.0, 0.01, 100))
    with pytest.raises(ValueError):
        guardrail_check("y", effect, margin=0.1, margin_type="percent")
