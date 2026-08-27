"""Shared doubly-robust DiD scores (private DML infrastructure).

Three distinct score families live here — the repo's methodology review
(``docs/methodology/papers/chang-2020-review.md``) pins that they are NOT
interchangeable:

- :func:`drdid_panel_inf_func` — the Sant'Anna & Zhao (2020) locally efficient
  doubly-robust panel influence function (``DRDID::drdid_panel$att.inf.func``
  parity), including the first-stage nuisance-correction terms.
  Algebra-preserving relocation from ``ContinuousDiD._dr_cell_inf_func``
  (fail-closed input validation added; the mathematical body is unchanged); the relocation is pinned
  by committed oracles in ``tests/test_dr_scores.py``.
- :func:`chang_panel_score` / :func:`chang_panel_score_augmented` — the
  Chang (2020) Neyman-orthogonal Case 1 (repeated outcomes) score for
  cross-fitted (DML) nuisances, normalized by the unconditional treated share
  ``p_hat``, with the augmented variant carrying the finite-dimensional
  treated-share variance correction (``G_1p = -theta/p_hat``).
- :func:`chang_rcs_score` / :func:`chang_rcs_lambda_slope` /
  :func:`chang_rcs_score_augmented` — the Chang (2020) Case 2 (repeated
  cross sections) score on LEVEL outcomes with the post-period sampling share
  ``lam_hat``, whose Theorem 2 variance carries BOTH finite-dimensional
  corrections (the treated-share fold-in plus an EXPLICIT
  ``G_2lambda * (T - lam_hat)`` term). The Case 2 outcome nuisance is a
  SINGLE control-only regression of ``(T - lam_hat) * Y`` on X — deliberately
  different from Sant'Anna-Zhao/DoubleML's four treatment-by-period outcome
  regressions, so ``doubleml.DoubleMLDIDCSBinary`` is a characterization
  anchor, not a parity oracle.

References
----------
Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
estimators. Journal of Econometrics, 219(1), 101-122.

Chang, N.-C. (2020). Double/debiased machine learning for
difference-in-differences models. The Econometrics Journal, 23(2), 177-191.
"""

from typing import Tuple

import numpy as np

from diff_diff.linalg import _rank_guarded_inv

__all__ = [
    "drdid_panel_inf_func",
    "chang_panel_score",
    "chang_panel_score_augmented",
    "chang_rcs_score",
    "chang_rcs_lambda_slope",
    "chang_rcs_score_augmented",
]


def drdid_panel_inf_func(
    dY: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    gamma: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    """DRDID ``drdid_panel`` doubly-robust per-unit influence function.

    Direct port of ``DRDID::drdid_panel$att.inf.func`` (unit weights = 1,
    propensity already clipped so no drop-trimming). Units are ordered
    treated-then-control. Validated to ~1e-13 against DRDID in the original
    spike; the algebra-preserving relocation from
    ``ContinuousDiD._dr_cell_inf_func`` (validation added, body unchanged) is pinned
    by the committed oracles in ``tests/test_dr_scores.py``.

    Parameters
    ----------
    dY : np.ndarray
        Outcome changes, shape (n,).
    D : np.ndarray
        Treatment indicator (1 treated, 0 control), shape (n,).
    X : np.ndarray
        Covariate matrix INCLUDING the intercept column, shape (n, p).
    gamma : np.ndarray
        Outcome-regression coefficients (fit on controls), shape (p,).
    ps : np.ndarray
        Propensity scores, already clipped away from 1, shape (n,).

    Returns
    -------
    np.ndarray
        Per-unit influence function, shape (n,).
    """
    dY = np.asarray(dY, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    context = "drdid_panel_inf_func"
    if dY.ndim != 1 or D.ndim != 1 or ps.ndim != 1 or X.ndim != 2 or gamma.ndim != 1:
        raise ValueError(
            f"{context}: expected dY/D/ps 1-dimensional and X 2-dimensional with "
            "1-dimensional gamma"
        )
    n = dY.shape[0]
    if n == 0:
        raise ValueError(f"{context}: inputs are empty (n=0)")
    if D.shape[0] != n or ps.shape[0] != n or X.shape[0] != n:
        raise ValueError(f"{context}: dY/D/ps/X row counts must all equal {n}")
    if X.shape[1] != gamma.shape[0]:
        raise ValueError(
            f"{context}: X has {X.shape[1]} columns but gamma has {gamma.shape[0]} entries"
        )
    for name, arr in (("dY", dY), ("D", D), ("X", X), ("gamma", gamma), ("ps", ps)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise ValueError(
            f"{context}: both treated and control units are required "
            "(the treated- and control-weight denominators would be zero)"
        )
    if np.any(ps < 0.0) or np.any(ps >= 1.0):
        raise ValueError(
            f"{context}: ps must lie in [0, 1) strictly; clip propensity scores " "before calling"
        )
    out_delta = X @ gamma
    w_treat = D
    w_cont = ps * (1 - D) / (1 - ps)
    dr_treat = w_treat * (dY - out_delta)
    dr_cont = w_cont * (dY - out_delta)
    w_cont_mean = w_cont.mean()
    if not np.isfinite(w_cont_mean) or w_cont_mean <= 0.0:
        raise ValueError(
            f"{context}: the control-odds mass is zero (every control unit has "
            "ps == 0); the comparison population is empty and the influence "
            "function is undefined"
        )
    eta_treat = dr_treat.mean() / w_treat.mean()
    eta_cont = dr_cont.mean() / w_cont_mean
    weights_ols = 1.0 - D
    wols_eX = (weights_ols * (dY - out_delta))[:, np.newaxis] * X
    XpX = ((weights_ols[:, np.newaxis] * X).T @ X) / n
    XpX_inv, _, _ = _rank_guarded_inv(XpX)
    asy_wols = wols_eX @ XpX_inv
    W = ps * (1 - ps)
    score_ps = (D - ps)[:, np.newaxis] * X
    Hess, _, _ = _rank_guarded_inv((X.T @ (W[:, np.newaxis] * X)) / n)
    asy_ps = score_ps @ Hess
    inf_treat_1 = dr_treat - w_treat * eta_treat
    M1 = (w_treat[:, np.newaxis] * X).mean(axis=0)
    inf_treat = (inf_treat_1 - asy_wols @ M1) / w_treat.mean()
    inf_cont_1 = dr_cont - w_cont * eta_cont
    M2 = (w_cont[:, np.newaxis] * (dY - out_delta - eta_cont)[:, np.newaxis] * X).mean(axis=0)
    M3 = (w_cont[:, np.newaxis] * X).mean(axis=0)
    inf_control = (inf_cont_1 + asy_ps @ M2 - asy_wols @ M3) / w_cont_mean
    return inf_treat - inf_control


def _validate_chang_inputs(
    dY: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    context: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for the Chang Case 1 score functions."""
    dY = np.asarray(dY, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    for name, arr in (("dY", dY), ("D", D), ("m_hat", m_hat), ("ps", ps)):
        if arr.ndim != 1:
            raise ValueError(f"{context}: {name} must be 1-dimensional, got ndim={arr.ndim}")
    n = dY.shape[0]
    if n == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    for name, arr in (("dY", dY), ("D", D), ("m_hat", m_hat), ("ps", ps)):
        if arr.shape[0] != n:
            raise ValueError(
                f"{context}: {name} has length {arr.shape[0]}, expected {n} (length of dY)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not np.isfinite(p_hat) or not (0.0 < p_hat < 1.0):
        raise ValueError(
            f"{context}: p_hat must satisfy 0 < p_hat < 1 strictly, got {p_hat!r} "
            "(p_hat = 1 means no comparison population; the estimand is unidentified)"
        )
    if np.any(ps < 0.0) or np.any(ps >= 1.0):
        raise ValueError(
            f"{context}: ps must lie in [0, 1) strictly; clip/trim propensity "
            "scores before calling (values >= 1 would divide by zero)"
        )
    return dY, D, m_hat, ps


def chang_panel_score(
    dY: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
) -> np.ndarray:
    """Chang (2020) Case 1 per-unit UNCENTERED score summand.

    Returns ``summand_i = (D_i - ps_i (1 - D_i) / (1 - ps_i)) * (dY_i -
    m_hat_i) / p_hat``, whose sample mean is the ATT (Equation 3.1's
    ``psi_1`` equals ``summand - theta``). Nuisances ``m_hat`` (control
    outcome-change regression, cross-fitted out-of-fold) and ``ps``
    (propensity, cross-fitted, trimmed by the CALLER's policy) are plug-ins;
    no first-stage correction terms appear — under DML2 cross-fitting the
    score is Neyman-orthogonal in the infinite-dimensional nuisances.

    ``p_hat`` is caller-supplied; the library convention (see the REGISTRY
    "Cross-fitting, DR-score, and ridge infrastructure (DML)" Note) is the
    FULL-SAMPLE treated share.
    """
    dY, D, m_hat, ps = _validate_chang_inputs(dY, D, m_hat, ps, p_hat, "chang_panel_score")
    weight = (D - ps * (1 - D) / (1 - ps)) / p_hat
    return weight * (dY - m_hat)


def chang_panel_score_augmented(
    summand: np.ndarray,
    D: np.ndarray,
    theta: float,
    p_hat: float,
) -> np.ndarray:
    """Chang (2020) augmented score ``psi_bar_i = summand_i - D_i * theta / p_hat``.

    The finite-dimensional treated-share correction Chang's Theorem 2 variance
    requires (``G_1p = -theta/p_hat`` folded into the score). The variance
    estimator is ``SE = sqrt(mean(psi_bar**2) / N)``; this exact object was
    matched to DoubleML's SE at ~6e-17 in the committed parity spike
    (``benchmarks/doubleml/chang_case1_parity.py``).
    """
    summand = np.asarray(summand, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    context = "chang_panel_score_augmented"
    if summand.ndim != 1 or D.ndim != 1:
        raise ValueError(f"{context}: summand and D must be 1-dimensional")
    if summand.shape[0] == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    if summand.shape[0] != D.shape[0]:
        raise ValueError(
            f"{context}: summand has length {summand.shape[0]}, D has length {D.shape[0]}"
        )
    if not np.all(np.isfinite(summand)):
        raise ValueError(f"{context}: summand contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not np.isfinite(theta):
        raise ValueError(f"{context}: theta must be finite, got {theta!r}")
    if not np.isfinite(p_hat) or not (0.0 < p_hat < 1.0):
        raise ValueError(
            f"{context}: p_hat must satisfy 0 < p_hat < 1 strictly, got {p_hat!r} "
            "(p_hat = 1 means no comparison population; the estimand is unidentified)"
        )
    return summand - D * theta / p_hat


def _validate_chang_rcs_inputs(
    y: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    lam_hat: float,
    context: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for the Chang Case 2 (repeated cross sections) scores."""
    y = np.asarray(y, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    m2_hat = np.asarray(m2_hat, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    for name, arr in (("y", y), ("D", D), ("T", T), ("m2_hat", m2_hat), ("ps", ps)):
        if arr.ndim != 1:
            raise ValueError(f"{context}: {name} must be 1-dimensional, got ndim={arr.ndim}")
    n = y.shape[0]
    if n == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    for name, arr in (("y", y), ("D", D), ("T", T), ("m2_hat", m2_hat), ("ps", ps)):
        if arr.shape[0] != n:
            raise ValueError(
                f"{context}: {name} has length {arr.shape[0]}, expected {n} (length of y)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{context}: {name} contains non-finite values")
    if not np.all((D == 0.0) | (D == 1.0)):
        raise ValueError(f"{context}: D must be strictly binary 0/1")
    if not np.all((T == 0.0) | (T == 1.0)):
        raise ValueError(f"{context}: T must be strictly binary 0/1")
    if not np.isfinite(p_hat) or not (0.0 < p_hat < 1.0):
        raise ValueError(
            f"{context}: p_hat must satisfy 0 < p_hat < 1 strictly, got {p_hat!r} "
            "(p_hat = 1 means no comparison population; the estimand is unidentified)"
        )
    if not np.isfinite(lam_hat) or not (0.0 < lam_hat < 1.0):
        raise ValueError(
            f"{context}: lam_hat must satisfy 0 < lam_hat < 1 strictly, got "
            f"{lam_hat!r} (lam_hat = 0 or 1 means all observations lie in one "
            "period; 1/(lam*(1-lam)) is undefined and the estimand is "
            "unidentified)"
        )
    if np.any(ps < 0.0) or np.any(ps >= 1.0):
        raise ValueError(
            f"{context}: ps must lie in [0, 1) strictly; clip/trim propensity "
            "scores before calling (values >= 1 would divide by zero)"
        )
    return y, D, T, m2_hat, ps


def chang_rcs_score(
    y: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    lam_hat: float,
) -> np.ndarray:
    """Chang (2020) Case 2 per-observation UNCENTERED score summand.

    Returns ``summand_i = (D_i - ps_i) / (p_hat * lam_hat * (1 - lam_hat) *
    (1 - ps_i)) * ((T_i - lam_hat) * y_i - m2_hat_i)``, whose sample mean is
    the ATT (Equation 3.2's ``psi_2`` equals ``summand - theta``). ``m2_hat``
    is the SINGLE Case 2 outcome nuisance ``l_20(X) = E[(T - lam) * Y | X,
    D=0]`` — one cross-fitted regression of ``(T - lam_hat) * Y`` on X trained
    on control observations only (Chang's ``I_kz^c``), NOT the four
    treatment-by-period regressions of the Sant'Anna-Zhao/DoubleML RCS score.
    ``ps`` is cross-fitted and trimmed by the CALLER's policy.

    ``p_hat`` and ``lam_hat`` are caller-supplied; the library convention
    (REGISTRY "DMLDiD" Notes) is the FULL-SAMPLE-within-cell treated share and
    post-period sampling share (``mean(D)`` / ``mean(T)``), mirroring the
    Case 1 global-``p_hat`` convention and DoubleML's ``t.mean()``.
    """
    y, D, T, m2_hat, ps = _validate_chang_rcs_inputs(
        y, D, T, m2_hat, ps, p_hat, lam_hat, "chang_rcs_score"
    )
    weight = (D - ps) / (p_hat * lam_hat * (1.0 - lam_hat) * (1.0 - ps))
    return weight * ((T - lam_hat) * y - m2_hat)


def chang_rcs_lambda_slope(
    y: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    lam_hat: float,
) -> float:
    """Chang (2020) Case 2 lambda-slope estimator ``G_2lambda``.

    Sample mean of the closed-form derivative ``d/d(lambda) psi_2`` evaluated
    at the plug-in nuisances (recovered from the proof of Theorem 2, p. 55
    display; the paper prints NO explicit ``G_2lambda`` estimator — this
    natural sample analogue is a documented implementation decision, REGISTRY
    "DMLDiD" Note)::

        d_lam psi_2_i = -((1 - 2*lam) / (lam**2 * (1-lam)**2))
                          * ((D_i - ps_i) / (p_hat * (1 - ps_i)))
                          * ((T_i - lam) * y_i - m2_hat_i)
                        - (y_i / (p_hat * lam * (1-lam)))
                          * ((D_i - ps_i) / (1 - ps_i))

    Only consistency is required of this estimator (Theorem 2 imposes no
    rate). The first term equals ``-((1 - 2*lam) / (lam*(1-lam))) *
    summand_i`` — an algebraic identity the test suite cross-checks.
    """
    y, D, T, m2_hat, ps = _validate_chang_rcs_inputs(
        y, D, T, m2_hat, ps, p_hat, lam_hat, "chang_rcs_lambda_slope"
    )
    return _chang_rcs_lambda_slope_validated(y, D, T, m2_hat, ps, p_hat, lam_hat)


def _chang_rcs_lambda_slope_validated(
    y: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    p_hat: float,
    lam_hat: float,
) -> float:
    # Assumes inputs already coerced/validated by _validate_chang_rcs_inputs.
    odds = (D - ps) / (1.0 - ps)
    term1 = (
        -((1.0 - 2.0 * lam_hat) / (lam_hat**2 * (1.0 - lam_hat) ** 2))
        * (odds / p_hat)
        * ((T - lam_hat) * y - m2_hat)
    )
    term2 = -(y / (p_hat * lam_hat * (1.0 - lam_hat))) * odds
    return float(np.mean(term1 + term2))


def chang_rcs_score_augmented(
    summand: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    theta: float,
    p_hat: float,
    lam_hat: float,
) -> np.ndarray:
    """Chang (2020) Case 2 augmented score with BOTH finite-dim corrections.

    Returns ``psi_bar_i = summand_i - D_i * theta / p_hat + G_2lambda *
    (T_i - lam_hat)`` — Theorem 2's combined score: the treated-share
    correction ``G_2p = -theta/p_hat`` folds into the score exactly as in
    Case 1, while the lambda-correction stays an EXPLICIT extra term
    (``G_2lambda`` computed internally via
    :func:`_chang_rcs_lambda_slope_validated`, the shared slope kernel
    behind :func:`chang_rcs_lambda_slope`).
    The variance estimator is ``SE = sqrt(mean(psi_bar**2) / N)``.

    Per the methodology review: "Omitting the λ-correction term is a
    plausible implementation bug the proof structure warns against" — the
    bare ``psi_2`` squared is NOT the Theorem 2 estimator, and no DoubleML
    parity anchor exists for this object (``DoubleMLDIDCSBinary``'s variance
    omits the lambda term; see the committed characterization spike).
    """
    return _chang_rcs_score_augmented_with_slope(
        summand, D, T, y, m2_hat, ps, theta, p_hat, lam_hat
    )[0]


def _chang_rcs_score_augmented_with_slope(
    summand: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    m2_hat: np.ndarray,
    ps: np.ndarray,
    theta: float,
    p_hat: float,
    lam_hat: float,
) -> Tuple[np.ndarray, float]:
    """Internal variant returning ``(psi_bar, g2_lambda)``.

    Validates once and computes the O(n) lambda-slope pass once, so a
    caller needing both the augmented score and the ``G_2lambda``
    diagnostic (the DMLDiD RCS cell loop) avoids the duplicate
    validation + slope pass of calling the two public functions.
    """
    context = "chang_rcs_score_augmented"
    summand = np.asarray(summand, dtype=np.float64)
    if summand.ndim != 1:
        raise ValueError(f"{context}: summand must be 1-dimensional")
    if summand.shape[0] == 0:
        raise ValueError(f"{context}: inputs are empty (n=0); the score is undefined")
    if not np.all(np.isfinite(summand)):
        raise ValueError(f"{context}: summand contains non-finite values")
    if not np.isfinite(theta):
        raise ValueError(f"{context}: theta must be finite, got {theta!r}")
    y, D, T, m2_hat, ps = _validate_chang_rcs_inputs(y, D, T, m2_hat, ps, p_hat, lam_hat, context)
    if summand.shape[0] != y.shape[0]:
        raise ValueError(f"{context}: summand has length {summand.shape[0]}, expected {y.shape[0]}")
    g2_lambda = _chang_rcs_lambda_slope_validated(y, D, T, m2_hat, ps, p_hat, lam_hat)
    return summand - D * theta / p_hat + g2_lambda * (T - lam_hat), g2_lambda
