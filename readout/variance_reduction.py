"""CUPED and regression adjustment.

Idea: a pre-experiment covariate X cannot be affected by the treatment, so
subtracting the part of Y that X predicts removes noise without changing the
expected treatment effect.

    Y_adj = Y - (X - Xbar) @ theta

One covariate gives classic CUPED with theta = cov(Y, X) / var(X). Several
covariates give regression adjustment with theta = Sxx^-1 Sxy. It is the same
formula, so one code path handles both.
"""

from __future__ import annotations

import numpy as np

from readout.config import ALPHA
from readout.metrics import compare_arms, mean_summary
from readout.schema import CupedResult


def _center_within_arm(values: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Subtract each arm's own mean from its rows."""
    centered = values.astype(float).copy()
    for a in np.unique(arm):
        mask = arm == a
        centered[mask] -= centered[mask].mean(axis=0)
    return centered


def cuped_theta(y: np.ndarray, X: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Pooled within-arm regression coefficient of Y on X.

    Centre Y and X on their own arm means, then solve least squares:

        theta = (Xc' Xc)^-1 Xc' yc        (one covariate: cov(Y, X) / var(X))

    Centring within arm keeps the treatment effect out of the estimate: the
    arm means differ by the effect, and that difference is not something X
    should be asked to explain. All arms share one theta.
    """
    y_c = _center_within_arm(np.asarray(y, dtype=float), arm)
    x_c = _center_within_arm(_as_2d(X), arm)
    theta, *_ = np.linalg.lstsq(x_c, y_c, rcond=None)
    return theta


def cuped_adjust(y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Y_adj = Y - (X - Xbar_overall) @ theta.

    X is centred on the mean over ALL units, not per arm. Then the difference
    in adjusted means is

        (Ybar_t - Ybar_c) - theta * (Xbar_t - Xbar_c)

    which corrects for the chance imbalance in X between arms. Centring X per
    arm would make that correction zero while still shrinking the variance,
    giving a standard error that is too small.
    """
    X = _as_2d(X)
    return np.asarray(y, dtype=float) - (X - X.mean(axis=0)) @ theta


def cuped_compare(
    y: np.ndarray,
    X: np.ndarray,
    arm: np.ndarray,
    treatment: str,
    control: str,
    covariates: tuple[str, ...] = (),
    alpha: float = ALPHA,
    n_comparisons: int = 1,
) -> CupedResult:
    """Raw and CUPED-adjusted treatment-vs-control comparison.

    variance_reduction = 1 - SE(adjusted diff)^2 / SE(raw diff)^2

    With one covariate this is about rho^2, the squared correlation between
    Y and X; with several it is about the R^2 of Y on X (within arm).
    Theta is estimated from every arm passed in, not just the two compared.
    """
    y = np.asarray(y, dtype=float)
    arm = np.asarray(arm)
    theta = cuped_theta(y, X, arm)
    y_adj = cuped_adjust(y, X, theta)

    is_t, is_c = arm == treatment, arm == control
    raw = compare_arms(mean_summary(y[is_t]), mean_summary(y[is_c]), alpha, n_comparisons)
    adjusted = compare_arms(
        mean_summary(y_adj[is_t]), mean_summary(y_adj[is_c]), alpha, n_comparisons
    )
    return CupedResult(
        raw=raw,
        adjusted=adjusted,
        covariates=tuple(covariates),
        theta=tuple(float(t) for t in theta),
        variance_reduction=float(1 - adjusted.abs_se**2 / raw.abs_se**2),
    )


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X
