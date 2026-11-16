"""Decision rules: one test per rule, plus precedence and an end-to-end run."""

from readout.decision import DONT_SHIP, INCONCLUSIVE, INVALID, SHIP, decide
from readout.guardrails import guardrail_check
from readout.metrics import compare_arms, mean_summary
from readout.schema import ArmSummary
from readout.srm import srm_check
from sim.simulate import simulate_experiment

SRM_OK = srm_check({"c": 5000, "t": 5000}, {"c": 0.5, "t": 0.5})
SRM_BAD = srm_check({"c": 4500, "t": 5500}, {"c": 0.5, "t": 0.5})
CONTROL = ArmSummary(10.0, 0.0004, 5000)


def _effect(treatment_mean):
    return compare_arms(ArmSummary(treatment_mean, 0.0004, 5000), CONTROL)


BIG_WIN = _effect(11.0)  # +10%, clearly significant
SMALL_WIN = _effect(10.1)  # +1%, significant (z ~ 3.5)
LOSS = _effect(9.0)  # -10%
FLAT = _effect(10.01)  # +0.1%, not significant

GUARDRAIL_OK = guardrail_check("revenue", _effect(10.0), margin=0.02)
GUARDRAIL_HARM = guardrail_check("revenue", _effect(9.0), margin=0.02)
NOISY = compare_arms(ArmSummary(10.0, 0.5, 20), ArmSummary(10.0, 0.5, 20))
GUARDRAIL_UNDERPOWERED = guardrail_check("revenue", NOISY, margin=0.02)


def test_rule_1_srm_failure_is_invalid():
    d = decide(SRM_BAD, [GUARDRAIL_OK], BIG_WIN, min_practical_lift=0.02)
    assert d.label == INVALID
    assert "Sample ratio mismatch" in d.reasons[0]


def test_rule_2_guardrail_failure_is_dont_ship_even_with_a_big_win():
    d = decide(SRM_OK, [GUARDRAIL_OK, GUARDRAIL_HARM], BIG_WIN, min_practical_lift=0.02)
    assert d.label == DONT_SHIP
    assert len(d.reasons) == 1 and "harm detected" in d.reasons[0]


def test_rule_2_underpowered_guardrail_says_harm_not_ruled_out():
    d = decide(SRM_OK, [GUARDRAIL_UNDERPOWERED], BIG_WIN, min_practical_lift=0.02)
    assert d.label == DONT_SHIP
    assert "harm not ruled out" in d.reasons[0]


def test_rule_3_significant_win_above_threshold_ships():
    d = decide(SRM_OK, [GUARDRAIL_OK], BIG_WIN, min_practical_lift=0.02)
    assert d.label == SHIP
    assert any("guardrails passed" in r for r in d.reasons)


def test_rule_3_significant_win_below_threshold_is_inconclusive():
    assert SMALL_WIN.significant
    d = decide(SRM_OK, [GUARDRAIL_OK], SMALL_WIN, min_practical_lift=0.02)
    assert d.label == INCONCLUSIVE
    assert any("below the practical threshold" in r for r in d.reasons)


def test_rule_4_significant_loss_is_dont_ship():
    d = decide(SRM_OK, [], LOSS, min_practical_lift=0.02)
    assert d.label == DONT_SHIP
    assert "got worse" in d.reasons[0]


def test_rule_5_not_significant_is_inconclusive_and_reports_what_is_ruled_out():
    assert not FLAT.significant
    d = decide(SRM_OK, [GUARDRAIL_OK], FLAT, min_practical_lift=0.02)
    assert d.label == INCONCLUSIVE
    assert any("rule out" in r for r in d.reasons)


def test_srm_takes_precedence_over_guardrails():
    d = decide(SRM_BAD, [GUARDRAIL_HARM], LOSS, min_practical_lift=0.02)
    assert d.label == INVALID


def test_every_decision_has_reasons():
    for primary in (BIG_WIN, SMALL_WIN, LOSS, FLAT):
        assert decide(SRM_OK, [], primary, 0.02).reasons


def test_end_to_end_on_simulated_experiment_with_known_win():
    # True lift = 0.5 / 10 = 5%, far above the 2% threshold; nothing else wrong.
    exp = simulate_experiment(n=20_000, true_effect=0.5, seed=0)
    primary = compare_arms(
        mean_summary(exp.values("y", "treatment")), mean_summary(exp.values("y", "control"))
    )
    srm = srm_check(exp.arm_counts(), exp.intended_allocation)
    guardrail = guardrail_check("y", primary, margin=0.02)
    assert decide(srm, [guardrail], primary, min_practical_lift=0.02).label == SHIP
