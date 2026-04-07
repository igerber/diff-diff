"""
Data-generating processes ported from R's ``did`` package.

Faithfully translates R's ``reset.sim()`` + ``build_sim_dataset()`` from
``R/simulate_data.R`` (github.com/bcallaway11/did) so that Python tests can
exercise the same DGP used to validate the R implementation.

Also includes ``build_two_group_sim_data()`` from ``test_sim_data_2_groups.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Main DGP: reset.sim() + build_sim_dataset()
# ---------------------------------------------------------------------------


def _reset_sim(
    time_periods: int = 4,
    n: int = 5000,
    ipw: bool = True,
    reg: bool = True,
) -> dict:
    """Return default simulation parameters matching R's ``reset.sim()``."""
    T = time_periods
    bett = np.arange(1, T + 1, dtype=float)
    thet = np.arange(1, T + 1, dtype=float)
    theu = thet.copy()
    betu = bett.copy()

    te_bet_ind = np.ones(T)
    te_bet_X = bett.copy()
    te_t = thet.copy()
    te_e = np.zeros(T)
    te = 1.0

    # Generalized propensity score coefficients: c(0, 1:T) / (2*T)
    gamG = np.concatenate([[0.0], np.arange(1, T + 1)]) / (2.0 * T)

    return dict(
        time_periods=T,
        n=n,
        bett=bett,
        thet=thet,
        theu=theu,
        betu=betu,
        te_bet_ind=te_bet_ind,
        te_bet_X=te_bet_X,
        te_t=te_t,
        te_e=te_e,
        te=te,
        gamG=gamG,
        ipw=ipw,
        reg=reg,
    )


def _pnorm(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (matches R's ``pnorm``)."""
    from scipy.stats import norm

    return norm.cdf(x)


def build_csdid_sim_data(
    n: int = 5000,
    time_periods: int = 4,
    *,
    ipw: bool = True,
    reg: bool = True,
    te: float = 1.0,
    te_e: np.ndarray | None = None,
    te_bet_ind: np.ndarray | None = None,
    bett: np.ndarray | None = None,
    betu: np.ndarray | None = None,
    thet: np.ndarray | None = None,
    theu: np.ndarray | None = None,
    te_t: np.ndarray | None = None,
    te_bet_X: np.ndarray | None = None,
    gamG: np.ndarray | None = None,
    seed: int = 91_42_024,
) -> pd.DataFrame:
    """Build simulated panel data matching R's ``build_sim_dataset(reset.sim())``.

    Parameters
    ----------
    n : int
        Total number of cross-sectional units *before* dropping G=1.
    time_periods : int
        Number of time periods (T).
    ipw, reg : bool
        Control DGP specification (see R docs).
    te : float
        Overall treatment effect level.
    te_e : array, optional
        Dynamic (exposure-time) treatment effects, length T.
    te_bet_ind : array, optional
        Group-specific treatment heterogeneity, length T.
    bett, betu, thet, theu, te_t, te_bet_X, gamG : array, optional
        Override default parameters (see ``_reset_sim``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``[unit, period, first_treat, outcome, X, cluster]``
    """
    sp = _reset_sim(time_periods=time_periods, n=n, ipw=ipw, reg=reg)
    sp["te"] = te
    T = sp["time_periods"]

    # Apply overrides
    if te_e is not None:
        sp["te_e"] = np.asarray(te_e, dtype=float)
    if te_bet_ind is not None:
        sp["te_bet_ind"] = np.asarray(te_bet_ind, dtype=float)
    if bett is not None:
        sp["bett"] = np.asarray(bett, dtype=float)
    if betu is not None:
        sp["betu"] = np.asarray(betu, dtype=float)
    if thet is not None:
        sp["thet"] = np.asarray(thet, dtype=float)
    if theu is not None:
        sp["theu"] = np.asarray(theu, dtype=float)
    if te_t is not None:
        sp["te_t"] = np.asarray(te_t, dtype=float)
    if te_bet_X is not None:
        sp["te_bet_X"] = np.asarray(te_bet_X, dtype=float)
    if gamG is not None:
        sp["gamG"] = np.asarray(gamG, dtype=float)

    # NOTE: ipw/reg flags do NOT zero out parameters. They change the
    # covariate transform used in the propensity/outcome models:
    #   ipw=True  → propensity uses X (linear logit — correctly specified)
    #   ipw=False → propensity uses (pnorm(X)+0.5)^2 (nonlinear — misspecified)
    #   reg=True  → outcome uses X (linear — correctly specified)
    #   reg=False → outcome uses X^2 (nonlinear — misspecified for linear reg)

    rng = np.random.default_rng(seed)

    # --- Covariate ---
    X_all = rng.standard_normal(n)

    # --- Group assignment via multinomial logit ---
    gamG_arr = sp["gamG"]  # length T+1, groups 0..T
    if ipw:
        logits = np.outer(X_all, gamG_arr)  # (n, T+1)
    else:
        logits = np.outer((_pnorm(X_all) + 0.5) ** 2, gamG_arr)

    # Softmax for multinomial probabilities
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    pr = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Sample group for each unit: groups are 0, 1, ..., T
    groups_all = np.array([rng.choice(T + 1, p=pr[i]) for i in range(n)])

    # --- Split treated (G>0) and untreated (G==0) ---
    treated_mask = groups_all > 0
    Gt = groups_all[treated_mask]
    Xt_raw = X_all[treated_mask]
    nt = len(Gt)

    untreated_mask = groups_all == 0
    Xu_raw = X_all[untreated_mask]
    nu = int(untreated_mask.sum())

    # Covariate model transform: reg=True → linear X, reg=False → X^2
    if reg:
        Xt = Xt_raw.copy()
        Xu = Xu_raw.copy()
    else:
        Xt = Xt_raw**2
        Xu = Xu_raw**2

    # --- Treated units ---
    Ct = rng.standard_normal(nt) + Gt  # unit FE: N(mean=G, sd=1)

    # Untreated potential outcomes for treated (nt × T)
    # R: thet[t] + Ct + Xt*bett[t] + rnorm(nt), for t in 1:T
    Y0t = np.empty((nt, T))
    for t_idx in range(T):
        Y0t[:, t_idx] = sp["thet"][t_idx] + Ct + Xt * sp["bett"][t_idx] + rng.standard_normal(nt)

    # Treated potential outcomes for treated (nt × T)
    # R code per period t (1-indexed):
    #   te.t[t] + te.bet.ind[Gt]*Ct + Xt*te.bet.X[t]
    #   + (Gt <= t) * te.e[max(t - Gt + 1, 1)] + te + rnorm(nt)
    Y1t = np.empty((nt, T))
    for t_idx in range(T):
        period = t_idx + 1  # R's 1-indexed period

        # te.bet.ind[Gt]: R 1-indexed → Python 0-indexed
        te_ind = sp["te_bet_ind"][Gt - 1]

        # te.e[max(t - Gt + 1, 1)]: R 1-indexed → Python 0-indexed
        # max(period - Gt + 1, 1) - 1 = max(period - Gt, 0)
        exposure_idx = np.clip(period - Gt, 0, T - 1)
        post_treat = (Gt <= period).astype(float)

        Y1t[:, t_idx] = (
            sp["te_t"][t_idx]
            + te_ind * Ct
            + Xt * sp["te_bet_X"][t_idx]
            + post_treat * sp["te_e"][exposure_idx]
            + sp["te"]
            + rng.standard_normal(nt)
        )

    # Observed outcomes for treated: Y = (G<=t)*Y1 + (G>t)*Y0
    Yt = np.empty((nt, T))
    for t_idx in range(T):
        period = t_idx + 1
        post = (Gt <= period).astype(float)
        Yt[:, t_idx] = post * Y1t[:, t_idx] + (1 - post) * Y0t[:, t_idx]

    # --- Untreated units ---
    Cu = rng.standard_normal(nu)  # N(0, 1)

    Y0u = np.empty((nu, T))
    for t_idx in range(T):
        Y0u[:, t_idx] = sp["theu"][t_idx] + Cu + Xu * sp["betu"][t_idx] + rng.standard_normal(nu)

    # --- Combine into long format ---
    n_total = nt + nu
    unit_ids = np.arange(1, n_total + 1)
    group_vals = np.concatenate([Gt, np.zeros(nu, dtype=int)])
    X_vals = np.concatenate([Xt_raw, Xu_raw])

    unit_rep = np.tile(unit_ids, T)
    group_rep = np.tile(group_vals, T)
    X_rep = np.tile(X_vals, T)
    period_rep = np.repeat(np.arange(1, T + 1), n_total)

    outcome_rep = np.empty(T * n_total)
    for t_idx in range(T):
        start = t_idx * n_total
        outcome_rep[start : start + nt] = Yt[:, t_idx]
        outcome_rep[start + nt : start + n_total] = Y0u[:, t_idx]

    # Cluster: random integer 1..50 (no within-cluster correlation)
    cluster_per_unit = rng.integers(1, 51, size=n_total)
    cluster_rep = np.tile(cluster_per_unit, T)

    df = pd.DataFrame(
        {
            "unit": unit_rep,
            "period": period_rep,
            "first_treat": group_rep,
            "outcome": outcome_rep,
            "X": X_rep,
            "cluster": cluster_rep,
        }
    )

    df = df.sort_values(["unit", "period"]).reset_index(drop=True)

    # Drop units with G=1 (treated in first period) — matches R's subset(ddf, G != 1)
    df = df[df["first_treat"] != 1].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Two-group DGP from test_sim_data_2_groups.R
# ---------------------------------------------------------------------------


def build_two_group_sim_data(
    n: int = 5000,
    seed: int = 2024_10_17,
) -> pd.DataFrame:
    """Build two-group simulated data matching R's ``test_sim_data_2_groups.R``.

    DGP: n units, 4 periods, G in {0, 3}.
    Group assignment: P(G=3) = exp(0.5) / (1 + exp(0.5)).
    True ATT = 3 at period 3, with dose-response 1.5*ATT at period 4.

    Parameters
    ----------
    n : int
        Number of cross-sectional units.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``[unit, period, first_treat, outcome]``
    """
    rng = np.random.default_rng(seed)

    p3 = np.exp(0.5) / (1.0 + np.exp(0.5))

    # Group assignment: 1 → G=3, 0 → never-treated
    g_indicator = (rng.uniform(size=n) <= p3).astype(int)
    G = np.where(g_indicator == 1, 3, 0)

    # Unit fixed effect: N(mean=G, sd=1)
    nu = rng.standard_normal(n) + G

    index_trend = 1.0
    index_att = 3.0
    att_rand = rng.standard_normal(n) + index_att  # N(3, 1)

    # Untreated potential outcomes
    Y_untreated = np.empty((n, 4))
    for t in range(4):
        Y_untreated[:, t] = (t + 1) * index_trend + nu + rng.standard_normal(n)

    # Treated potential outcomes (same base + ATT for post-treatment)
    Y_treated = np.empty((n, 4))
    for t in range(4):
        Y_treated[:, t] = (t + 1) * index_trend + nu + rng.standard_normal(n)
    Y_treated[:, 2] += att_rand  # period 3: ATT = att_rand ~ N(3,1)
    Y_treated[:, 3] += 1.5 * att_rand  # period 4: dose-response

    # Observed outcomes
    Y_obs = np.empty((n, 4))
    for t in range(4):
        period = t + 1
        is_treated_now = (G == 3) & (period >= 3)
        Y_obs[:, t] = np.where(is_treated_now, Y_treated[:, t], Y_untreated[:, t])

    # Convert to long format
    unit_ids = np.arange(1, n + 1)
    unit_rep = np.tile(unit_ids, 4)
    group_rep = np.tile(G, 4)
    period_rep = np.repeat(np.arange(1, 5), n)
    outcome_rep = Y_obs.T.ravel()

    df = pd.DataFrame(
        {
            "unit": unit_rep,
            "period": period_rep,
            "first_treat": group_rep,
            "outcome": outcome_rep,
        }
    )

    df = df.sort_values(["unit", "period"]).reset_index(drop=True)
    return df
