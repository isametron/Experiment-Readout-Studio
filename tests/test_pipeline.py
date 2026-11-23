"""End-to-end readout tests on simulated experiments with known truth."""

import pytest

from readout.decision import INCONCLUSIVE, INVALID, SHIP
from readout.loaders.hillstrom import DEFAULT_PATH, load_hillstrom
from readout.pipeline import ReadoutSpec, run_readout, spec_from_config
from readout.schema import ExperimentData, NotApplicable
from sim.simulate import simulate_experiment

# Simulated control mean is 10, so an effect of 0.5 is a +5% lift.
SPEC = ReadoutSpec(
    primary_metric="y",
    min_practical_lift=0.02,
    guardrails={"y": {"margin": 0.02, "margin_type": "relative"}},
)


def test_known_win_ships_and_cuped_reduces_variance_by_rho_squared():
    exp = simulate_experiment(n=20_000, true_effect=0.5, rho=0.6, seed=0)
    readout = run_readout(exp, SPEC)
    assert readout.srm.passed
    assert readout.decisions["treatment"].label == SHIP
    assert readout.primary["treatment"].rel_ci[0] <= 0.05 <= readout.primary["treatment"].rel_ci[1]
    assert readout.cuped["treatment"].variance_reduction == pytest.approx(0.36, abs=0.03)


def test_injected_srm_makes_the_readout_invalid_despite_a_win():
    skewed = {"control": 0.48, "treatment": 0.52}
    exp = simulate_experiment(n=40_000, true_effect=0.5, srm_allocation=skewed, seed=1)
    assert run_readout(exp, SPEC).decisions["treatment"].label == INVALID


def test_no_effect_is_inconclusive():
    exp = simulate_experiment(n=20_000, true_effect=0.0, seed=2)
    assert run_readout(exp, SPEC).decisions["treatment"].label == INCONCLUSIVE


def test_multi_arm_uses_holm_and_decides_each_arm():
    allocation = {"control": 1 / 3, "a": 1 / 3, "b": 1 / 3}
    exp = simulate_experiment(
        n=30_000, allocation=allocation, true_effect={"a": 0.5, "b": 0.0}, seed=3
    )
    readout = run_readout(exp, SPEC)
    assert set(readout.decisions) == {"a", "b"}
    assert readout.decisions["a"].label == SHIP
    assert readout.decisions["b"].label == INCONCLUSIVE
    for effect in readout.primary.values():
        assert effect.p_adjusted >= effect.p_value
        assert effect.ci_level == pytest.approx(0.975)


def test_dataset_without_covariates_reports_cuped_not_applicable():
    sim = simulate_experiment(n=2000, seed=4)
    exp = ExperimentData(
        name="no_covariates",
        level="user",
        df=sim.df[["arm", "y"]],
        arm_col="arm",
        control_arm="control",
        intended_allocation=sim.intended_allocation,
        metrics=["y"],
    )
    cuped = run_readout(exp, SPEC).cuped
    assert isinstance(cuped, NotApplicable)
    assert "no pre-experiment covariates" in cuped.reason


def test_misnamed_metric_raises_instead_of_skipping():
    exp = simulate_experiment(n=2000, seed=5)
    bad = ReadoutSpec(primary_metric="y", min_practical_lift=0.02, guardrails={"revenue": {}})
    with pytest.raises(ValueError, match="revenue"):
        run_readout(exp, bad)


def test_unknown_dataset_has_no_config():
    with pytest.raises(KeyError):
        spec_from_config("not_a_dataset")


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="data/raw/hillstrom.csv missing")
def test_hillstrom_readout_runs_from_config():
    readout = run_readout(load_hillstrom())
    assert readout.srm.passed
    assert set(readout.decisions) == {"Mens E-Mail", "Womens E-Mail"}
    assert all(d.reasons for d in readout.decisions.values())
