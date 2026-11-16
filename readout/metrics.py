"""Means, proportions, lift, confidence intervals and delta method.

Every metric is first reduced to an ArmSummary (estimate, variance of the
estimate, n) per arm. compare_arms then works the same way for a mean, a
proportion (the mean of a 0/1 column), a ratio metric or a CUPED-adjusted
outcome.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from readout.config import ALPHA
from readout.schema import ArmSummary, EffectResult


def mean_summary(y: np.ndarray) -> ArmSummary:
    """Sample mean and the variance of the sample mean.

        ybar = (1/n) * sum(y_i)
        Var(ybar) = s^2 / n,   s^2 = sum((y_i - ybar)^2) / (n - 1)

    For a 0/1 outcome ybar is the proportion p and s^2 is p(1-p) (times
    n/(n-1)), so proportions need no separate code.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    return ArmSummary(estimate=float(y.mean()), variance=float(y.var(ddof=1) / n), n=n)


def ratio_summary(num: np.ndarray, den: np.ndarray) -> ArmSummary:
    """Ratio metric R = sum(num_i) / sum(den_i) with a delta-method variance.

    Use when the randomisation unit (user) differs from the unit the metric
    is averaged over (session, page view). num_i and den_i are per-user
    totals, for example revenue and number of sessions. Users are independent
    but sessions of one user are not, so treating sessions as independent
    understates the variance.

    With per-user means mu_N, mu_D, variances s_N^2, s_D^2 and covariance s_ND:

        R      = mean(num) / mean(den)
        Var(R) ~ (1/n) * [ s_N^2 / mu_D^2
                           - 2 * mu_N * s_ND / mu_D^3
                           + mu_N^2 * s_D^2 / mu_D^4 ]

    This is the first-order Taylor expansion of N/D around (mu_N, mu_D).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = num.size
    mu_n, mu_d = num.mean(), den.mean()
    cov = np.cov(num, den, ddof=1)
    var_n, var_d, cov_nd = cov[0, 0], cov[1, 1], cov[0, 1]
    variance = (var_n / mu_d**2 - 2 * mu_n * cov_nd / mu_d**3 + mu_n**2 * var_d / mu_d**4) / n
    return ArmSummary(estimate=float(mu_n / mu_d), variance=float(variance), n=n)


def compare_arms(
    treatment: ArmSummary,
    control: ArmSummary,
    alpha: float = ALPHA,
    n_comparisons: int = 1,
) -> EffectResult:
    """Absolute difference (z-test) and relative lift (delta method).

    Absolute difference, arms independent, variances not pooled (Welch):

        d     = est_t - est_c
        SE(d) = sqrt(Var_t + Var_c)
        z     = d / SE(d),   p = 2 * (1 - Phi(|z|))
        CI    = d +/- z_crit * SE(d)

    Relative lift is a ratio of two random quantities, so its SE needs the
    delta method (first-order Taylor expansion of t/c):

        lift      = est_t / est_c - 1
        Var(lift) ~ Var_t / est_c^2 + est_t^2 * Var_c / est_c^4
        CI        = lift +/- z_crit * sqrt(Var(lift))

    Significance always comes from the absolute test; the lift CI is for
    reporting. z_crit uses alpha / n_comparisons (Bonferroni width) so the
    intervals stay valid when several arms are compared to control.
    """
    ci_alpha = alpha / n_comparisons
    z_crit = stats.norm.ppf(1 - ci_alpha / 2)
    t, c = treatment.estimate, control.estimate

    diff = t - c
    se = np.sqrt(treatment.variance + control.variance)
    z = diff / se
    p_value = 2 * stats.norm.sf(abs(z))

    if c == 0:
        lift, lift_se = float("nan"), float("nan")
    else:
        lift = t / c - 1
        lift_se = np.sqrt(treatment.variance / c**2 + t**2 * control.variance / c**4)

    return EffectResult(
        control_estimate=c,
        treatment_estimate=t,
        n_control=control.n,
        n_treatment=treatment.n,
        abs_diff=float(diff),
        abs_se=float(se),
        abs_ci=(float(diff - z_crit * se), float(diff + z_crit * se)),
        rel_lift=float(lift),
        rel_se=float(lift_se),
        rel_ci=(float(lift - z_crit * lift_se), float(lift + z_crit * lift_se)),
        z=float(z),
        p_value=float(p_value),
        p_adjusted=float(p_value),
        alpha=alpha,
        ci_level=1 - ci_alpha,
        significant=bool(p_value < alpha),
    )


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls family-wise error rate).

    Sort the m p-values ascending: p_(1) <= ... <= p_(m). Then

        adj_(i) = max over j <= i of  min(1, (m - j + 1) * p_(j))

    The smallest p-value is multiplied by m, the next by m - 1, and so on;
    the running maximum keeps the adjusted values in the same order.
    Results are returned in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    scaled = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def compare_all_to_control(
    treatments: dict[str, ArmSummary],
    control: ArmSummary,
    alpha: float = ALPHA,
) -> dict[str, EffectResult]:
    """Compare every treatment arm to control with Holm correction.

    p_adjusted is the Holm-adjusted p-value and `significant` means
    p_adjusted < alpha. Holm has no matching confidence interval, so the CIs
    use the Bonferroni width alpha / m. That is slightly more conservative
    than Holm: a result can be Holm-significant while its CI just touches 0.
    """
    m = len(treatments)
    results = {
        arm: compare_arms(summary, control, alpha=alpha, n_comparisons=m)
        for arm, summary in treatments.items()
    }
    adjusted = holm_adjust([r.p_value for r in results.values()])
    return {
        arm: replace(r, p_adjusted=float(p_adj), significant=bool(p_adj < alpha))
        for (arm, r), p_adj in zip(results.items(), adjusted, strict=True)
    }
