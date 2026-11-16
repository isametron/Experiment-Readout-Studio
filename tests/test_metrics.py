"""Unit tests for means, lift, ratio metrics and Holm."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from readout.metrics import (
    compare_all_to_control,
    compare_arms,
    holm_adjust,
    mean_summary,
    ratio_summary,
)
from readout.schema import ArmSummary
from sim.simulate import simulate_experiment, simulate_ratio_experiment

N_RUNS = 1000


def test_mean_summary_hand_calculation():
    s = mean_summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s.estimate == pytest.approx(2.5)
    assert s.variance == pytest.approx((5 / 3) / 4)  # s^2 = 5/3, n = 4
    assert s.n == 4


def test_proportion_is_a_mean_of_zeros_and_ones():
    y = np.array([1] * 30 + [0] * 70)
    s = mean_summary(y)
    assert s.estimate == pytest.approx(0.30)
    assert s.variance == pytest.approx(0.3 * 0.7 / 99)  # p(1-p)/(n-1)


def test_compare_arms_hand_calculation():
    t = ArmSummary(estimate=11.0, variance=0.04, n=100)
    c = ArmSummary(estimate=10.0, variance=0.05, n=100)
    r = compare_arms(t, c)
    assert r.abs_diff == pytest.approx(1.0)
    assert r.abs_se == pytest.approx(0.3)  # sqrt(0.04 + 0.05)
    assert r.z == pytest.approx(1.0 / 0.3)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(1.0 / 0.3))
    assert r.rel_lift == pytest.approx(0.10)
    # Var(lift) = 0.04/100 + 121*0.05/10000 = 0.001005
    assert r.rel_se == pytest.approx(np.sqrt(0.001005))
    assert r.significant


def test_zero_control_mean_gives_nan_lift_not_a_made_up_number():
    r = compare_arms(ArmSummary(1.0, 0.01, 50), ArmSummary(0.0, 0.01, 50))
    assert np.isnan(r.rel_lift)
    assert r.abs_diff == pytest.approx(1.0)


def test_relative_lift_ci_covers_true_lift():
    # True lift = 0.5 / 10 = 5%. The 95% CI should contain it ~95% of the time.
    covered = 0
    for seed in range(N_RUNS):
        exp = simulate_experiment(n=2000, true_effect=0.5, seed=seed)
        r = compare_arms(
            mean_summary(exp.values("y", "treatment")),
            mean_summary(exp.values("y", "control")),
        )
        covered += r.rel_ci[0] <= 0.05 <= r.rel_ci[1]
    assert 0.925 <= covered / N_RUNS <= 0.975


def test_binary_metric_aa_false_positive_rate():
    rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(N_RUNS):
        t = rng.random(3000) < 0.12
        c = rng.random(3000) < 0.12
        rejections += compare_arms(mean_summary(t), mean_summary(c)).significant
    assert 0.03 <= rejections / N_RUNS <= 0.07


# ---- ratio metrics (delta method) ----


def test_ratio_summary_equals_mean_when_denominator_is_constant():
    rng = np.random.default_rng(1)
    num = rng.normal(5, 2, size=500)
    r = ratio_summary(num, np.full(500, 2.0))
    m = mean_summary(num / 2.0)
    assert r.estimate == pytest.approx(m.estimate)
    assert r.variance == pytest.approx(m.variance)


def _ratio_arms(exp):
    return (
        ratio_summary(exp.values("value_sum", "treatment"), exp.values("sessions", "treatment")),
        ratio_summary(exp.values("value_sum", "control"), exp.values("sessions", "control")),
    )


def test_ratio_metric_aa_delta_method_is_calibrated_and_naive_is_not():
    delta_rejections = naive_rejections = 0
    for seed in range(N_RUNS):
        exp, sessions = simulate_ratio_experiment(n_users=1000, seed=seed)
        delta_rejections += compare_arms(*_ratio_arms(exp)).significant
        # Naive: pretend every session is an independent unit.
        naive = compare_arms(
            mean_summary(sessions.loc[sessions["arm"] == "treatment", "value"]),
            mean_summary(sessions.loc[sessions["arm"] == "control", "value"]),
        )
        naive_rejections += naive.significant
    assert 0.03 <= delta_rejections / N_RUNS <= 0.07
    assert naive_rejections / N_RUNS > 0.10  # far too many false positives


def test_ratio_metric_effect_is_unbiased():
    true_effect = 0.3
    estimates = []
    for seed in range(N_RUNS):
        exp, _ = simulate_ratio_experiment(n_users=1000, true_effect=true_effect, seed=seed)
        estimates.append(compare_arms(*_ratio_arms(exp)).abs_diff)
    estimates = np.array(estimates)
    standard_error = estimates.std(ddof=1) / np.sqrt(N_RUNS)
    assert abs(estimates.mean() - true_effect) < 3 * standard_error


# ---- Holm ----


def test_holm_hand_calculation():
    # sorted: 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, 1*0.04 -> 0.03, 0.06, 0.06 (running max)
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(2)
    for _ in range(50):
        p = rng.random(rng.integers(1, 8))
        expected = multipletests(p, method="holm")[1]
        assert holm_adjust(p) == pytest.approx(expected)


def test_holm_controls_family_wise_error_in_three_arm_aa():
    allocation = {"control": 1 / 3, "a": 1 / 3, "b": 1 / 3}
    any_holm = any_raw = 0
    for seed in range(N_RUNS):
        exp = simulate_experiment(n=3000, allocation=allocation, seed=seed)
        results = compare_all_to_control(
            {arm: mean_summary(exp.values("y", arm)) for arm in exp.treatment_arms},
            mean_summary(exp.values("y", "control")),
        )
        any_holm += any(r.significant for r in results.values())
        any_raw += any(r.p_value < 0.05 for r in results.values())
    assert any_holm / N_RUNS <= 0.07
    assert any_raw > any_holm  # without correction the family rejects more often


def test_multi_arm_cis_use_bonferroni_width():
    c = ArmSummary(10.0, 0.01, 100)
    results = compare_all_to_control(
        {"a": ArmSummary(10.2, 0.01, 100), "b": ArmSummary(10.1, 0.01, 100)}, c
    )
    assert results["a"].ci_level == pytest.approx(0.975)
    single = compare_arms(ArmSummary(10.2, 0.01, 100), c)
    assert results["a"].abs_ci[0] < single.abs_ci[0]  # wider than the unadjusted CI
