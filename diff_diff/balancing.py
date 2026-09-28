"""Covariate balancing weights (entropy balancing).

Implements **entropy balancing** (Hainmueller, J. (2012). "Entropy Balancing for
Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in
Observational Studies." *Political Analysis*, 20(1), 25-46.
https://doi.org/10.1093/pan/mpr025).

Entropy balancing finds nonnegative control weights ``w_i`` that exactly match a set
of target covariate moments (here: the treated-group covariate means) while staying as
close as possible — in the Kullback-Leibler sense — to a set of base weights (uniform by
default). The solution is obtained from the convex dual

    minimize over λ:  L(λ) = log( Σ_i q_i exp(Z_iᵀ λ) ),   Z_i = X_i − target,

whose stationary point ``∇L = Σ_i w_i Z_i = 0`` is exactly first-moment balance, with
``w_i = q_i exp(Z_iᵀ λ) / Σ_j q_j exp(Z_jᵀ λ)``. ``L`` is convex (log-sum-exp), so a
damped Newton iteration (gradient = weighted mean of the centered moments, Hessian =
weighted covariance) converges to the balancing weights whenever the target lies in the
interior of the control covariate convex hull; otherwise no finite λ balances the
moments and the problem is **infeasible**.

This module is dependency-light (numpy, with an optional scipy L-BFGS fallback) and is
used by ``StackedDiD`` to construct the within-sub-experiment design weights ``b_{sa}``
for Covariate-Balanced Weighted Stacked DID (Ustyuzhanin 2026).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["entropy_balance", "BalanceError"]


class BalanceError(ValueError):
    """Raised when entropy balancing fails to achieve first-moment balance.

    Carries the achieved ``max_residual`` and the per-covariate residual vector so
    callers (e.g. ``StackedDiD``) can attach cohort context and report the worst-
    balanced covariate.
    """

    def __init__(self, message: str, *, max_residual: float, residuals: np.ndarray):
        super().__init__(message)
        self.max_residual = max_residual
        self.residuals = residuals


def entropy_balance(
    X: np.ndarray,
    target_means: np.ndarray,
    base_weights: Optional[np.ndarray] = None,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve entropy balancing for control weights matching ``target_means``.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Covariate matrix for the ``n`` control units (``k`` covariates).
    target_means : np.ndarray, shape (k,)
        Target first moments to match — for CBWSDID these are the treated-group
        covariate means.
    base_weights : np.ndarray, shape (n,), optional
        Nonnegative base weights ``q_i`` (the KL reference). Defaults to uniform.
        Internally renormalized to sum to one.
    max_iter : int
        Maximum damped-Newton iterations (then a scipy L-BFGS fallback is attempted).
    tol : float
        Convergence tolerance on the maximum absolute (raw-scale) moment residual
        ``max_r |Σ_i w_i X_{i,r} − target_r|``.

    Returns
    -------
    weights : np.ndarray, shape (n,)
        Nonnegative weights summing to one with ``Σ_i w_i X_i ≈ target_means``.
    info : dict
        ``converged`` (bool), ``max_residual`` (float), ``n_iter`` (int),
        ``ess`` (effective sample size ``1 / Σ_i w_i²``), ``solver`` (str).

    Raises
    ------
    BalanceError
        If neither the damped-Newton nor the L-BFGS fallback drives the maximum moment
        residual below ``tol`` (the target is outside the control covariate hull, i.e.
        infeasible).
    ValueError
        On malformed inputs (shape mismatch, non-finite, negative base weights).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, k); got shape {X.shape}")
    n, k = X.shape
    target = np.asarray(target_means, dtype=np.float64).reshape(-1)
    if target.shape[0] != k:
        raise ValueError(f"target_means length {target.shape[0]} != n_covariates {k}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(target)):
        raise ValueError("X and target_means must be finite")
    if n == 0:
        raise ValueError("X has no control rows")

    if base_weights is None:
        q = np.full(n, 1.0 / n)
    else:
        q = np.asarray(base_weights, dtype=np.float64).reshape(-1)
        if q.shape[0] != n:
            raise ValueError(f"base_weights length {q.shape[0]} != n_control {n}")
        if np.any(q < 0) or not np.all(np.isfinite(q)):
            raise ValueError("base_weights must be nonnegative and finite")
        s = q.sum()
        if s <= 0:
            raise ValueError("base_weights sum to zero")
        q = q / s

    # Centered moments; standardize columns for conditioning (balance set is invariant
    # to the linear rescaling — it is absorbed into the dual variable λ).
    Z = X - target
    scale = Z.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Zs = Z / scale

    def weights_at(lam: np.ndarray) -> np.ndarray:
        logits = Zs @ lam
        logits -= logits.max()
        ew = q * np.exp(logits)
        return ew / ew.sum()

    def dual_loss(lam: np.ndarray) -> float:
        logits = Zs @ lam
        m = logits.max()
        return float(m + np.log(np.sum(q * np.exp(logits - m))))

    def raw_residual(w: np.ndarray) -> np.ndarray:
        return w @ X - target

    lam = np.zeros(k)
    solver = "newton"
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        w = weights_at(lam)
        if np.max(np.abs(raw_residual(w))) < tol:
            break
        g = w @ Zs  # gradient of the dual loss (standardized scale)
        Zc = Zs - g
        H = (w[:, None] * Zc).T @ Zc  # weighted covariance (PSD)
        ridge = 1e-10 * (np.trace(H) / k + 1e-12)
        try:
            direction = -np.linalg.solve(H + ridge * np.eye(k), g)
        except np.linalg.LinAlgError:
            direction = -np.linalg.lstsq(H, g, rcond=None)[0]
        # Backtracking (Armijo) line search on the convex dual loss.
        base = dual_loss(lam)
        slope = float(g @ direction)  # < 0 (descent)
        step = 1.0
        for _ in range(40):
            if dual_loss(lam + step * direction) <= base + 1e-4 * step * slope:
                break
            step *= 0.5
        lam = lam + step * direction
    else:
        w = weights_at(lam)

    w = weights_at(lam)
    resid = raw_residual(w)
    max_resid = float(np.max(np.abs(resid)))

    if max_resid >= tol:
        # Fallback: scipy L-BFGS-B on the convex dual (robust to poor Newton scaling).
        try:
            from scipy.optimize import minimize

            res = minimize(
                dual_loss,
                lam,
                jac=lambda L: weights_at(L) @ Zs,
                method="L-BFGS-B",
                options={"maxiter": 500, "gtol": 1e-12},
            )
            w_lbfgs = weights_at(res.x)
            resid_lbfgs = raw_residual(w_lbfgs)
            if np.max(np.abs(resid_lbfgs)) < max_resid:
                lam, w, resid = res.x, w_lbfgs, resid_lbfgs
                max_resid = float(np.max(np.abs(resid)))
                solver = "lbfgs"
        except Exception:  # pragma: no cover - scipy always present, defensive
            pass

    converged = max_resid < tol
    info: Dict[str, Any] = {
        "converged": converged,
        "max_residual": max_resid,
        "n_iter": n_iter,
        "ess": float(1.0 / np.sum(w**2)),
        "solver": solver,
    }
    if not converged:
        worst = int(np.argmax(np.abs(resid)))
        raise BalanceError(
            "entropy balancing did not converge to first-moment balance "
            f"(max moment residual {max_resid:.3e} >= tol {tol:.1e}; worst covariate "
            f"index {worst}). The target mean is likely outside the convex hull of the "
            "control covariates (infeasible).",
            max_residual=max_resid,
            residuals=resid,
        )
    return w, info
