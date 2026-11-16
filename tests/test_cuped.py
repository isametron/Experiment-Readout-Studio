"""Unit tests for CUPED / regression adjustment."""

import numpy as np
import pytest

from readout.variance_reduction import cuped_adjust, cuped_compare, cuped_theta
from sim.simulate import simulate_experiment


def _arrays(exp):
    df = exp.df
    return df["y"].to_numpy(), df["x"].to_numpy(), df["arm"].to_numpy()


def test_theta_recovers_true_slope():
    # Y = 10 + sigma * rho * X + noise, so the slope on X is sigma * rho = 2 * 0.6.
    exp = simulate_experiment(n=200_000, rho=0.6, sigma=2.0, true_effect=1.0, seed=0)
    theta = cuped_theta(*_arrays(exp))
    assert theta[0] == pytest.approx(1.2, abs=0.02)


def test_theta_is_not_distorted_by_a_large_treatment_effect():
    # A huge effect shifts Y between arms. Within-arm centring must ignore it.
    exp = simulate_experiment(n=200_000, rho=0.5, true_effect=50.0, seed=1)
    theta = cuped_theta(*_arrays(exp))
    assert theta[0] == pytest.approx(0.5, abs=0.02)


def test_adjustment_keeps_overall_mean():
    exp = simulate_experiment(n=5000, seed=2)
    y, x, arm = _arrays(exp)
    y_adj = cuped_adjust(y, x, cuped_theta(y, x, arm))
    assert y_adj.mean() == pytest.approx(y.mean())


def test_adjusted_difference_corrects_for_covariate_imbalance():
    # adjusted diff = raw diff - theta * (xbar_t - xbar_c).
    # This fails if X is (wrongly) centred per arm.
    exp = simulate_experiment(n=5000, true_effect=0.2, seed=3)
    y, x, arm = _arrays(exp)
    result = cuped_compare(y, x, arm, "treatment", "control")
    imbalance = x[arm == "treatment"].mean() - x[arm == "control"].mean()
    expected = result.raw.abs_diff - result.theta[0] * imbalance
    assert result.adjusted.abs_diff == pytest.approx(expected)
    assert result.adjusted.abs_diff != pytest.approx(result.raw.abs_diff)


def test_two_covariates_reduce_variance_by_r_squared():
    # Y = 0.6*X1 + 0.5*X2 + noise with Var(noise) = 0.39, Var(Y) = 1 -> R^2 = 0.61
    rng = np.random.default_rng(4)
    n = 100_000
    X = rng.standard_normal((n, 2))
    arm = rng.choice(["control", "treatment"], size=n)
    y = X @ np.array([0.6, 0.5]) + np.sqrt(0.39) * rng.standard_normal(n)
    y = y + 0.1 * (arm == "treatment")
    result = cuped_compare(y, X, arm, "treatment", "control", covariates=("x1", "x2"))
    assert result.theta == pytest.approx((0.6, 0.5), abs=0.02)
    assert result.variance_reduction == pytest.approx(0.61, abs=0.02)
    assert result.adjusted.abs_diff == pytest.approx(0.1, abs=0.02)


def test_useless_covariate_gives_no_reduction():
    exp = simulate_experiment(n=50_000, rho=0.0, seed=5)
    result = cuped_compare(*_arrays(exp), "treatment", "control")
    assert result.variance_reduction == pytest.approx(0.0, abs=0.01)
