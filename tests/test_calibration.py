"""Calibration tests: A/A false-positive rate and CUPED unbiasedness.

These run the methods on many simulated experiments where the truth is known
and count how often they are right. Seeds are fixed, so results are
deterministic.

Tolerance for a 5% rate over 1,000 runs: the binomial standard error is
sqrt(0.05 * 0.95 / 1000) = 0.0069, so 0.05 +/- 3 SE is roughly [0.03, 0.07].
"""

import numpy as np
import pytest

from readout.metrics import compare_arms, mean_summary
from readout.srm import srm_check
from readout.variance_reduction import cuped_compare
from sim.simulate import simulate_experiment

N_RUNS = 1000


def _raw(exp):
    return compare_arms(
        mean_summary(exp.values("y", "treatment")),
        mean_summary(exp.values("y", "control")),
    )


def _cuped(exp):
    df = exp.df
    return cuped_compare(
        df["y"].to_numpy(), df["x"].to_numpy(), df["arm"].to_numpy(), "treatment", "control"
    )


def test_aa_false_positive_rate_is_five_percent():
    rejections = sum(_raw(simulate_experiment(n=2000, seed=s)).significant for s in range(N_RUNS))
    assert 0.03 <= rejections / N_RUNS <= 0.07


def test_aa_false_positive_rate_with_unequal_allocation():
    allocation = {"control": 0.15, "treatment": 0.85}
    rejections = sum(
        _raw(simulate_experiment(n=4000, allocation=allocation, seed=s)).significant
        for s in range(N_RUNS)
    )
    assert 0.03 <= rejections / N_RUNS <= 0.07


def test_cuped_aa_false_positive_rate_is_five_percent():
    rejections = sum(
        _cuped(simulate_experiment(n=2000, rho=0.7, seed=s)).adjusted.significant
        for s in range(N_RUNS)
    )
    assert 0.03 <= rejections / N_RUNS <= 0.07


def test_cuped_is_unbiased_and_covers_the_true_effect():
    true_effect = 0.1
    estimates, covered = [], 0
    for seed in range(N_RUNS):
        adjusted = _cuped(
            simulate_experiment(n=2000, true_effect=true_effect, rho=0.7, seed=seed)
        ).adjusted
        estimates.append(adjusted.abs_diff)
        covered += adjusted.abs_ci[0] <= true_effect <= adjusted.abs_ci[1]
    estimates = np.array(estimates)
    standard_error = estimates.std(ddof=1) / np.sqrt(N_RUNS)
    assert abs(estimates.mean() - true_effect) < 3 * standard_error
    assert 0.925 <= covered / N_RUNS <= 0.975


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.8])
def test_cuped_reported_variance_reduction_matches_rho_squared(rho):
    reductions = [
        _cuped(simulate_experiment(n=5000, rho=rho, seed=s)).variance_reduction for s in range(200)
    ]
    assert np.mean(reductions) == pytest.approx(rho**2, abs=0.03)


def test_cuped_actual_variance_of_estimates_falls_by_rho_squared():
    # Not the reported SE: the real spread of the estimates across experiments.
    rho = 0.7
    raw, adjusted = [], []
    for seed in range(N_RUNS):
        result = _cuped(simulate_experiment(n=2000, true_effect=0.1, rho=rho, seed=seed))
        raw.append(result.raw.abs_diff)
        adjusted.append(result.adjusted.abs_diff)
    ratio = np.var(adjusted, ddof=1) / np.var(raw, ddof=1)
    assert ratio == pytest.approx(1 - rho**2, abs=0.07)  # expect 0.51


def test_power_detects_a_real_effect():
    # Effect 0.1 with sigma 1 and 10,000 per arm: z is about 7, power ~ 100%.
    result = _raw(simulate_experiment(n=20_000, true_effect=0.1, seed=0))
    assert result.significant
    assert result.abs_ci[0] <= 0.1 <= result.abs_ci[1]


# ---- SRM ----


def _srm(exp):
    return srm_check(exp.arm_counts(), exp.intended_allocation)


def test_injected_52_48_srm_is_detected():
    # At n = 40,000 a 52/48 split gives chi2 of about 64, far past the
    # p < 0.001 cut-off (chi2 > 10.8). Below roughly n = 10,000 a 52/48 split
    # is NOT reliably detectable at this threshold.
    skewed = {"control": 0.48, "treatment": 0.52}
    for seed in range(50):
        exp = simulate_experiment(n=40_000, srm_allocation=skewed, seed=seed)
        assert not _srm(exp).passed


def test_srm_false_alarm_rate_is_about_one_in_a_thousand():
    false_alarms = sum(not _srm(simulate_experiment(n=5000, seed=s)).passed for s in range(N_RUNS))
    assert false_alarms <= 5  # expected: 1


def test_srm_respects_unequal_intended_allocation():
    allocation = {"control": 0.15, "treatment": 0.85}
    false_alarms = sum(
        not _srm(simulate_experiment(n=5000, allocation=allocation, seed=s)).passed
        for s in range(N_RUNS)
    )
    assert false_alarms <= 5
