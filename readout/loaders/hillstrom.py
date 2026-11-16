"""Loader for the Hillstrom email marketing experiment.

Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008): 64,000
customers who bought in the last 12 months, randomly split in thirds between
a men's merchandise e-mail, a women's merchandise e-mail and no e-mail.
Outcomes are measured over the following two weeks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readout.schema import ExperimentData

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hillstrom.csv"

ARM_COL = "segment"
CONTROL = "No E-Mail"
ARMS = (CONTROL, "Mens E-Mail", "Womens E-Mail")
METRICS = ["visit", "conversion", "spend"]  # outcomes in the 2 weeks after the e-mail
COVARIATES = ["history", "recency"]  # pre-campaign: past-year spend, months since purchase
SEGMENTS = ["channel", "zip_code", "newbie", "history_segment"]
EXPECTED_COLUMNS = [ARM_COL, *METRICS, *COVARIATES, *SEGMENTS, "mens", "womens"]


def load_hillstrom(path: Path | str = DEFAULT_PATH) -> ExperimentData:
    """Read the raw CSV and return it in the common ExperimentData format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/download_data.py")
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Hillstrom file is missing expected columns: {missing}")

    return ExperimentData(
        name="hillstrom",
        level="user",
        df=df,
        arm_col=ARM_COL,
        control_arm=CONTROL,
        intended_allocation=dict.fromkeys(ARMS, 1 / 3),
        metrics=METRICS,
        covariates=COVARIATES,
        segments=SEGMENTS,
    )
