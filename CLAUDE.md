# Experiment Readout Studio

Turns public A/B test datasets into a standard experiment readout and a
one-page ship/don't-ship memo. Portfolio project for analyst internship
applications. Deadline: Dec 14, 2026.

## Architecture rules (non-negotiable)
- All statistics live in `readout/`. The Streamlit app and BI exports only
  call `readout.pipeline` and display results. Never compute a p-value,
  CI, or lift in `app/` or in a BI calculated field.
- Every loader returns the common `ExperimentData` format from `schema.py`.
  Downstream code must not know which dataset it came from.
- Features a dataset can't support return an explicit "not applicable"
  result with a reason string. Never silently skip, never fabricate.

## Statistical conventions
- Alpha = 0.05, two-sided, for primary metrics. Significance always comes
  from the z-test on the absolute difference (unpooled variances).
- Guardrails: one-sided non-inferiority tests at alpha = 0.05 with a
  margin defined per metric in `readout/config.py` (each margin is marked
  relative or absolute). A guardrail fails if we cannot rule out harm
  beyond the margin. Failures are labelled `fail_harm_detected` or
  `fail_harm_not_ruled_out` (too little data); both block shipping.
- SRM: chi-square goodness-of-fit against the *intended* allocation
  (not assumed 50/50). Threshold p < 0.001. An SRM failure blocks a
  "ship" decision regardless of other results. In multi-arm tests the
  gate is one test over ALL arms; pairwise checks are diagnostics only.
- Lift is reported as relative lift with a CI. The lift CI uses the delta
  method for every metric (lift is a ratio of two means). Ratio metrics
  (unit of analysis differs from unit of randomisation) also use the
  delta method for their own variance (`metrics.ratio_summary`).
- CUPED: theta = cov(Y, X) / var(X), estimated from within-arm centred
  data pooled across arms (one theta for all arms). X is centred on the
  overall mean, never per arm. Multiple covariates use the same formula
  in matrix form (regression adjustment). Report the variance reduction
  achieved (1 - SE_adj^2 / SE_raw^2) alongside the adjusted estimate.
- Segments are exploratory. Use Benjamini-Hochberg (FDR 0.05) and
  label results "exploratory" in the memo. Segment wins never
  drive a ship decision on their own.
- Multi-arm tests: compare each treatment to control and apply Holm
  correction across the comparisons. Holm has no matching CI, so CIs are
  reported at Bonferroni width (alpha / m) and the memo must say so.

### Open questions (not decided yet)
- ASOS checkpoints: reading a fixed-horizon test at every checkpoint
  inflates false positives. Either use a sequential method or label the
  over-time view "descriptive only".
- Segments: define the BH family (suggested: all levels of all
  dimensions for one metric). A significant segment next to a
  non-significant one is not evidence that the segments differ.

## Decision logic (decision.py)
1. SRM fails → INVALID: investigate before reading results
2. Any guardrail fails → DON'T SHIP
3. Primary significant and positive, point-estimate lift at or above
   the minimum practical effect → SHIP. Significant and positive but
   below it → INCONCLUSIVE with the reason "statistically significant
   but below the practical threshold"
4. Primary significant and negative → DON'T SHIP
5. Otherwise → INCONCLUSIVE (report the CI and what effect size it
   rules out)
Every decision carries a list of human-readable reasons.

## Dataset capability matrix
| Dataset   | Level      | SRM | CUPED                 | Segments            | Notes |
|-----------|------------|-----|-----------------------|---------------------|-------|
| Hillstrom | user       | yes | yes (history, recency)| yes                 | 3 arms: Mens, Womens, No E-Mail |
| Criteo    | user       | yes | regression adj (f0–f11)| yes (binned features)| ~85/15 split; subsample to 500k in dev |
| ASOS      | summary    | yes | no (no user data)     | no                  | checkpoint data; show estimates over time |
| Upworthy  | aggregated | yes | no (no pre-period)    | no                  | many arms per test; multiple testing |
Verify column names against the actual files on first load; update this
table if they differ. Hillstrom verified 2026-11-15: 64,000 rows, arm
column `segment` with values "Mens E-Mail", "Womens E-Mail", "No E-Mail",
no missing values. CUPED on `visit` removes under 1% of variance there
(weak covariates), which is expected and not a bug.

## Status (2026-11-15)
Built and tested (54 tests, simulation-calibrated): `schema.py`,
`config.py`, `srm.py`, `metrics.py` (means, lift, ratio metrics, Holm),
`variance_reduction.py`, `guardrails.py`, `decision.py`,
`loaders/hillstrom.py`, `sim/simulate.py`, `scripts/download_data.py`,
`scripts/run_hillstrom.py`.
Still stubs: `segments.py`, `memo.py`, `pipeline.py`, loaders for
Criteo / ASOS / Upworthy, `app/`, `scripts/export_for_bi.py`,
`templates/memo.md.j2`. `scripts/run_hillstrom.py` shows the call order
that `pipeline.py` should wrap.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format .`
- App: `uv run streamlit run app/streamlit_app.py`
- BI export: `uv run python scripts/export_for_bi.py`

## Working rules for Claude
- Run `uv run pytest` before saying any task is done.
- New statistical functions need a test on simulated data with a known
  true effect, not just a "runs without error" test.
- Prefer plain numpy/scipy/statsmodels; explain any formula in a
  docstring with the equation.
- Don't add dependencies without asking.
- Keep functions small; the owner must be able to explain every line
  in an interview.