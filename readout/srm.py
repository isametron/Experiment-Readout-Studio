"""Chi-square sample ratio mismatch check against intended allocation.

An SRM means the observed split between arms differs from the designed split
by more than chance allows. It signals a bug in assignment or logging, so the
effect estimates cannot be trusted.
"""

from __future__ import annotations

from scipy import stats

from readout.config import SRM_THRESHOLD
from readout.schema import SRMResult


def srm_check(
    observed: dict[str, int],
    intended: dict[str, float],
    arms: list[str] | None = None,
    threshold: float = SRM_THRESHOLD,
) -> SRMResult:
    """Chi-square goodness-of-fit of observed arm counts against the design.

    With N total units, observed count O_i and intended share p_i for arm i:

        E_i  = N * p_i
        chi2 = sum_i (O_i - E_i)^2 / E_i,   df = k - 1

    The p-value is P(chi2_df >= chi2). SRM is flagged when p < threshold.

    By default all arms are tested together (the gate for a multi-arm test).
    Pass `arms` to test a subset as a diagnostic; the intended shares of the
    subset are rescaled to sum to 1.
    """
    arms = list(intended) if arms is None else arms
    share_total = sum(intended[a] for a in arms)
    n_total = sum(observed[a] for a in arms)
    expected = {a: n_total * intended[a] / share_total for a in arms}

    chi2, p_value = stats.chisquare(
        f_obs=[observed[a] for a in arms],
        f_exp=[expected[a] for a in arms],
    )
    return SRMResult(
        observed={a: observed[a] for a in arms},
        expected=expected,
        chi2=float(chi2),
        dof=len(arms) - 1,
        p_value=float(p_value),
        threshold=threshold,
        passed=bool(p_value >= threshold),
    )
