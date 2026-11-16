"""Unit tests for the SRM check."""

import pytest
from scipy import stats

from readout.srm import srm_check


def test_chi2_matches_hand_calculation():
    # N = 1000, 50/50 design, observed 530/470.
    # chi2 = 30^2/500 + 30^2/500 = 3.6, df = 1
    result = srm_check({"c": 470, "t": 530}, {"c": 0.5, "t": 0.5})
    assert result.chi2 == pytest.approx(3.6)
    assert result.dof == 1
    assert result.p_value == pytest.approx(stats.chi2.sf(3.6, df=1))
    assert result.passed


def test_uses_intended_allocation_not_fifty_fifty():
    # 85/15 design honoured exactly: no mismatch, although far from 50/50.
    result = srm_check({"c": 1500, "t": 8500}, {"c": 0.15, "t": 0.85})
    assert result.chi2 == pytest.approx(0.0)
    assert result.passed
    # The same counts against a 50/50 design are a blatant mismatch.
    assert not srm_check({"c": 1500, "t": 8500}, {"c": 0.5, "t": 0.5}).passed


def test_three_arms_have_two_degrees_of_freedom():
    intended = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    result = srm_check({"a": 1000, "b": 1000, "c": 1000}, intended)
    assert result.dof == 2
    assert result.passed


def test_subset_rescales_intended_shares():
    intended = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    result = srm_check({"a": 1000, "b": 1000, "c": 5000}, intended, arms=["a", "b"])
    assert result.expected == {"a": 1000.0, "b": 1000.0}
    assert result.passed  # the pair looks fine...
    assert not srm_check(
        {"a": 1000, "b": 1000, "c": 5000}, intended
    ).passed  # ...the whole does not
