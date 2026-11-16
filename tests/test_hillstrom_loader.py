"""Hillstrom loader checks. Skipped when the raw file has not been downloaded."""

import pytest

from readout.loaders.hillstrom import DEFAULT_PATH, load_hillstrom

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="data/raw/hillstrom.csv missing")


def test_loads_into_common_format():
    exp = load_hillstrom()
    assert exp.level == "user"
    assert len(exp.df) == 64_000
    assert exp.control_arm == "No E-Mail"
    assert exp.treatment_arms == ["Mens E-Mail", "Womens E-Mail"]
    assert sum(exp.arm_counts().values()) == 64_000
    assert set(exp.df["visit"].unique()) == {0, 1}
    assert exp.df[exp.covariates].notna().all().all()
