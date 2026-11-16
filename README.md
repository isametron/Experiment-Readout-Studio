# experiment-readout-studio

A readout pipeline for A/B experiments. It runs sample-ratio-mismatch (SRM) checks, variance reduction (CUPED and regression adjustment), segment cuts with multiple-comparison correction, guardrail non-inferiority tests, and produces a ship / don't-ship / inconclusive decision with a memo.

## Layout

- `readout/` — the stats core. All math lives here and nowhere else.
- `sim/` — synthetic experiments with known true effects, used for validation.
- `tests/` — unit tests plus calibration tests (A/A false-positive rate, CUPED unbiasedness).
- `scripts/` — data download and BI export.
- `app/` — Streamlit UI only; it calls `readout.pipeline`.
- `templates/`, `memos/` — memo template and generated example memos.
- `exports/`, `bi/` — CSVs and notes for the Tableau / Power BI dashboard.
- `data/` — raw and processed data, git-ignored.

## Setup

```bash
uv sync
uv run pytest
uv run python scripts/download_data.py
uv run python scripts/run_hillstrom.py
```

Lint with `uv run ruff check . && uv run ruff format .`.

## Status

The stats core is implemented and tested against simulated experiments with known effects: SRM, means and proportions, relative lift, ratio metrics (delta method), Holm correction, CUPED / regression adjustment, guardrails and the decision rules. The Hillstrom loader works end to end. Segments, the memo, the pipeline, the other loaders, the app and the BI export are still stubs.
