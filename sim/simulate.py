"""Synthetic experiments with known true effects.

Used to check that the statistics in readout/ are calibrated: we know the
truth here, so we can count how often the methods get it right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readout.schema import ExperimentData

DEFAULT_ALLOCATION = {"control": 0.5, "treatment": 0.5}


def _assign(rng, n: int, allocation: dict[str, float]) -> np.ndarray:
    arms = list(allocation)
    return rng.choice(arms, size=n, p=[allocation[a] for a in arms])


def _effect_per_unit(arm: np.ndarray, allocation: dict, true_effect) -> np.ndarray:
    """Additive effect for each unit. The first arm is control (effect 0).

    true_effect is a float applied to every treatment arm, or a dict
    {arm: effect}.
    """
    control = next(iter(allocation))
    effects = dict.fromkeys(allocation, 0.0)
    for a in allocation:
        if a == control:
            continue
        effects[a] = true_effect[a] if isinstance(true_effect, dict) else true_effect
    return np.array([effects[a] for a in arm])


def simulate_experiment(
    n: int = 10_000,
    true_effect: float | dict[str, float] = 0.0,
    allocation: dict[str, float] | None = None,
    rho: float = 0.5,
    srm_allocation: dict[str, float] | None = None,
    base_mean: float = 10.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> ExperimentData:
    """User-level experiment with one pre-period covariate.

        X ~ N(0, 1)
        Y = base_mean + sigma * (rho * X + sqrt(1 - rho^2) * eps) + effect * T

    so corr(Y, X) = rho within each arm and Var(Y) = sigma^2. CUPED should
    then remove a share rho^2 of the variance.

    `allocation` is the designed split (first key is the control arm).
    `srm_allocation`, when given, is the split units are ACTUALLY assigned
    with while the design still says `allocation`. That is a sample ratio
    mismatch.
    """
    rng = np.random.default_rng(seed)
    allocation = allocation or DEFAULT_ALLOCATION
    arm = _assign(rng, n, srm_allocation or allocation)

    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    effect = _effect_per_unit(arm, allocation, true_effect)
    y = base_mean + sigma * (rho * x + np.sqrt(1 - rho**2) * eps) + effect

    return ExperimentData(
        name="simulated",
        level="user",
        df=pd.DataFrame({"arm": arm, "y": y, "x": x}),
        arm_col="arm",
        control_arm=next(iter(allocation)),
        intended_allocation=allocation,
        metrics=["y"],
        covariates=["x"],
    )


def simulate_ratio_experiment(
    n_users: int = 2_000,
    true_effect: float = 0.0,
    allocation: dict[str, float] | None = None,
    mean_extra_sessions: float = 4.0,
    base_mean: float = 10.0,
    user_sd: float = 1.0,
    session_sd: float = 1.0,
    seed: int | None = None,
) -> tuple[ExperimentData, pd.DataFrame]:
    """Users randomised, metric measured per session (a ratio metric).

        sessions_i ~ 1 + Poisson(mean_extra_sessions)
        value_ij   = base_mean + u_i + effect * T_i + e_ij
        u_i ~ N(0, user_sd^2),  e_ij ~ N(0, session_sd^2)

    u_i makes sessions of the same user correlated. The metric is
    sum(value) / sum(sessions), whose true treatment effect is `true_effect`.

    Returns the user-level ExperimentData (columns `value_sum`, `sessions`)
    and the session-level DataFrame, so tests can show what goes wrong when
    sessions are wrongly treated as independent.
    """
    rng = np.random.default_rng(seed)
    allocation = allocation or DEFAULT_ALLOCATION
    arm = _assign(rng, n_users, allocation)
    effect = _effect_per_unit(arm, allocation, true_effect)

    sessions = 1 + rng.poisson(mean_extra_sessions, size=n_users)
    user_level = base_mean + rng.normal(0, user_sd, size=n_users) + effect

    user_id = np.repeat(np.arange(n_users), sessions)
    value = user_level[user_id] + rng.normal(0, session_sd, size=user_id.size)
    session_df = pd.DataFrame({"user_id": user_id, "arm": arm[user_id], "value": value})

    user_df = pd.DataFrame(
        {
            "arm": arm,
            "value_sum": np.bincount(user_id, weights=value, minlength=n_users),
            "sessions": sessions,
        }
    )
    data = ExperimentData(
        name="simulated_ratio",
        level="user",
        df=user_df,
        arm_col="arm",
        control_arm=next(iter(allocation)),
        intended_allocation=allocation,
        metrics=["value_sum", "sessions"],
    )
    return data, session_df
