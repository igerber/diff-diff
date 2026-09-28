"""Conformal inference for counterfactual / synthetic controls (Chernozhukov,
Wüthrich & Zhu 2021, *JASA* 116(536):1849–1864).

Pure, numpy-only building blocks for the conformal-inference layer surfaced as
opt-in methods on :class:`SyntheticControlResults`
(:meth:`~diff_diff.synthetic_control_results.SyntheticControlResults.conformal_test`,
``conformal_confidence_intervals``, ``conformal_average_effect``). See
``docs/methodology/papers/chernozhukov-wuthrich-zhu-2021-review.md`` and the
``## SyntheticControl`` section of ``docs/methodology/REGISTRY.md``.

**Method (CWZ §2.2).** Under a sharp null ``H0: θ = θ0`` over the post period,
impute the counterfactual treated outcomes (``Y^N_{1t} = Y_{1t} − θ0_t`` for
``t > T0``; pre-period unchanged), fit a *time-permutation-invariant* proxy on
**all** periods under that null, take residuals ``û_t = Y^N_{1t} − P̂^N_t``, form
the statistic ``S_q(û) = ((1/√T*)·Σ_{t>T0}|û_t|^q)^{1/q}`` (high → reject), and
compute a permutation p-value (eq 2) ``p̂ = (1/|Π|)·#{π∈Π : S(û_π) ≥ S(û)}`` by
reshuffling residuals over time. ``Π`` contains the identity, so ``p̂ ≥ 1/|Π|``
automatically (there is NO extra ``+1`` — distinct from the cross-unit placebo
``(1+n)/(n+1)``). Inverting the test over a grid of ``θ0`` gives confidence sets.

**Proxy.** The proxy is the canonical CWZ constrained-LS synthetic control
(eqs 3–4): simplex weights minimising ``Σ_{t}(Y^N_{1t} − Σ_j w_j Y^N_{jt})²``
over **all** periods under the null, ``w ≥ 0, Σ w = 1``, **no V-matrix, no
intercept, outcomes by default** (footnote 9: "we estimate w under the null
based on all the data"); optional RAW covariate-matching rows stack into the
same objective when ``covariates=`` is supplied (the paper's note after eq 6),
with outcome-only residuals. This is DISTINCT from the headline ADH V-matrix weights — CWZ's
exactness theory (Lemma 1, Appendix D exchangeability) requires a time-symmetric
proxy, which the ADH pre-period V-fit is not. Reuses the Frank-Wolfe simplex
solver :func:`diff_diff.utils._sc_weight_fw`.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.utils import _sc_weight_fw

__all__: List[str] = []

_INF = float("inf")


# =============================================================================
# Proxy fit (CWZ §2.3, eqs 3-4) — canonical constrained-LS synthetic control
# =============================================================================


def _cwz_proxy_fit(
    y1: np.ndarray,
    Y0: np.ndarray,
    *,
    max_iter: int,
    min_decrease: float,
    init_weights: Optional[np.ndarray] = None,
    x1_rows: Optional[np.ndarray] = None,
    X0_rows: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fit the canonical CWZ constrained-LS SC proxy over ALL given periods (eq 4).

    Minimises ``Σ_t (y1_t − Σ_j w_j Y0_{tj})²`` s.t. ``w ≥ 0, Σ w = 1`` (no
    intercept, no V-matrix). ``y1`` is the (already null-imputed) treated outcome
    vector ``(T,)``; ``Y0`` the donor outcomes ``(T, J)``. Reuses
    :func:`_sc_weight_fw` packed ``(T, J+1)`` (donor columns first, target last),
    ``zeta=0, intercept=False`` — the solver projects onto the unit simplex, so the
    ``w ≥ 0, Σ w = 1`` constraint is delivered with no extra normalization.

    ``min_decrease`` is the ALREADY-SCALED absolute convergence tolerance; the
    caller must pass a θ0-invariant scale (e.g. the pre-window outcome norm) so CI
    membership does not drift with the grid value via the tolerance.

    **Covariates (CWZ §2.3.2, after eq 6):** "it is straightforward to
    incorporate (transformations of) covariates X_jt into the estimation
    problems (4) and (6)." ``x1_rows`` ``(R,)`` / ``X0_rows`` ``(R, J)`` are
    OPTIONAL covariate-matching rows stacked under the outcome rows in the
    same simplex LS objective — each row enters exactly like an outcome
    period, so the stacked objective remains invariant under time
    permutations of the outcome rows (exchangeability preserved; the
    covariate rows are fixed features of every permuted dataset). The
    RESIDUALS returned are OUTCOME rows only — the statistic and the
    permutation p-value are unchanged in form. Covariates are stacked RAW
    (no internal standardization): the paper's "(transformations of)"
    delegates scaling to the caller, and a covariate on a large scale
    dominates the fit exactly as it would in eq 4 — pre-scale accordingly.

    Returns ``(w (J,), resid = y1 − Y0 @ w (T,), converged)``. ``J == 1`` is the
    degenerate single-donor case ``w = [1]`` (no optimisation).
    """
    y1 = np.asarray(y1, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    n_out = y1.shape[0]
    if x1_rows is not None:
        x1_rows = np.asarray(x1_rows, dtype=float)
        X0_rows = np.asarray(X0_rows, dtype=float)
        y1_fit = np.concatenate([y1, x1_rows])
        Y0_fit = np.vstack([Y0, X0_rows])
    else:
        y1_fit, Y0_fit = y1, Y0
    _, J = Y0.shape
    if J == 1:
        w = np.array([1.0], dtype=float)
        return w, y1 - Y0 @ w, True
    packed = np.column_stack([Y0_fit, y1_fit])  # (T+R, J+1); last column is the target
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*did not converge.*", category=UserWarning)
        w, converged = _sc_weight_fw(
            packed,
            zeta=0.0,
            intercept=False,
            init_weights=init_weights,
            min_decrease=float(min_decrease),
            max_iter=int(max_iter),
            return_convergence=True,
        )
    w = np.asarray(w, dtype=float)
    return w, y1[:n_out] - Y0[:n_out] @ w, bool(converged)


# =============================================================================
# Test statistic (CWZ §2.2) and permutation sets
# =============================================================================


def _cwz_statistic(u: np.ndarray, post_mask: np.ndarray, q: Any) -> float:
    """``S_q(û) = ((1/√T*)·Σ_{t>T0}|û_t|^q)^{1/q}`` (CWZ §2.2).

    ``q=∞`` → ``max_{t>T0}|û_t|`` (the ``√T*`` factor vanishes in the sup limit).
    ``post_mask`` is a boolean over the (calendar-ordered) periods selecting the
    post window. Returns NaN if the post window is empty.
    """
    post = np.abs(u[post_mask])
    n_star = post.size
    if n_star == 0:
        return float("nan")
    if q == _INF:
        return float(np.max(post))
    s = float(np.sum(post**q))
    return float((s / np.sqrt(n_star)) ** (1.0 / q))


def _moving_block_perms(m: int) -> np.ndarray:
    """Moving-block permutations ``Π_→`` — ``m`` cyclic shifts (CWZ §2.2).

    Row ``j`` (``j = 0,…,m−1``) is the index array of ``π_j(i) = (i + j) mod m``
    (the paper's 1-based ``i+j`` wrapped at ``m``); ``û_π = û[row]``. Row 0 is the
    identity. Shape ``(m, m)``. Valid under stationary, weakly-dependent errors
    (Assumption 2.2).
    """
    base = np.arange(m)
    return (base[None, :] + base[:, None]) % m


def _iid_perms(m: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. permutations ``Π_all`` (CWZ §2.2).

    Exact ``m!`` enumeration ONLY when genuinely small (``m ≤ 12`` and
    ``m! ≤ n_draws``); otherwise ``n_draws`` random draws with the identity
    prepended (the identity must be in ``Π``). Valid under i.i.d. errors
    (Assumption 2.1). p-values below ``1/n_draws`` are unattainable.
    """
    if m <= 12 and math.factorial(m) <= n_draws:
        from itertools import permutations

        return np.array(list(permutations(range(m))), dtype=int)
    ident = np.arange(m)[None, :]
    if n_draws <= 1:
        return ident
    draws = np.stack([rng.permutation(m) for _ in range(n_draws - 1)])
    return np.vstack([ident, draws]).astype(int)


def _make_perms(m: int, scheme: str, n_iid: int, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the requested permutation set (both include the identity)."""
    if scheme == "moving_block":
        return _moving_block_perms(m)
    if scheme == "iid":
        return _iid_perms(m, n_iid, rng)
    raise ValueError(f"scheme must be 'moving_block' or 'iid', got {scheme!r}")


# =============================================================================
# Permutation p-value (CWZ eq 2)
# =============================================================================


def _cwz_pvalue(
    u: np.ndarray,
    post_mask: np.ndarray,
    perms: np.ndarray,
    q: Any,
    alternative: str = "two-sided",
) -> Tuple[float, float, int]:
    """CWZ eq (2) permutation p-value for residuals ``u``.

    ``p̂ = (1/|Π|)·#{π ∈ Π : S(û_π) ≥ S(û)}``. The residual vector is permuted and
    the statistic always reads the FIXED post-window slots (``post_mask``); the
    identity ``π`` is in ``perms`` so ``p̂ ≥ 1/|Π|`` automatically (no extra ``+1``).
    Ties (``≥``) are counted conservatively. ``perms`` is the ``(|Π|, len(u))``
    integer index array from :func:`_make_perms`. Returns ``(p, S_observed, |Π|)``.

    **One-sided alternatives (CWZ Remark 1).** The permutation argument is
    statistic-agnostic ("other test statistics can be used as well"), so
    ``alternative="greater"`` uses the SIGNED average-effect statistic
    ``S(û) = T_*^{-1/2}·Σ_{t>T0} û_t`` — large positive residuals (treated
    outcome above the counterfactual proxy, i.e. ``θ > θ0``) reject;
    ``"less"`` negates the residuals and applies the same rule. NOTE the
    paper has NO §7 and no named one-sided section — the variants ground in
    Remark 1's statistic freedom + eq 2. ``q`` applies only to the two-sided
    ``S_q`` family and must be 1 for one-sided alternatives (enforced by the
    callers).
    """
    if alternative == "less":
        u = -u
    if alternative in ("greater", "less"):
        post = u[post_mask]
        n_star = post.size
        s_obs = float(np.sum(post) / np.sqrt(n_star))
        s_perm = u[perms][:, post_mask].sum(axis=1) / np.sqrt(n_star)
    else:
        s_obs = _cwz_statistic(u, post_mask, q)
        post_perm = np.abs(u[perms][:, post_mask])  # (|Π|, T*)
        n_star = post_perm.shape[1]
        if q == _INF:
            s_perm = post_perm.max(axis=1)
        else:
            s_perm = (post_perm**q).sum(axis=1) / np.sqrt(n_star)
            s_perm = s_perm ** (1.0 / q)
    n = perms.shape[0]
    tol = 1e-12 * max(abs(s_obs), 1.0)
    n_ge = int(np.sum(s_perm >= s_obs - tol))
    return n_ge / n, float(s_obs), n


# =============================================================================
# Single-null evaluation (joint sharp-null test) and grid inversion (CIs)
# =============================================================================


def _single_null_pvalue(
    y1_obs: np.ndarray,
    Y0: np.ndarray,
    post_mask: np.ndarray,
    effect_on_post: np.ndarray,
    perms: np.ndarray,
    q: Any,
    *,
    max_iter: int,
    min_decrease: float,
    init_weights: Optional[np.ndarray] = None,
    alternative: str = "two-sided",
    x1_rows: Optional[np.ndarray] = None,
    X0_rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build ``Z(θ0)``, fit the proxy under the null, and return the p-value.

    ``effect_on_post`` (length = ``post_mask.sum()``, calendar order) is subtracted
    from the observed treated outcomes in the post window. The proxy is fit ONCE on
    the null-imputed data and residuals are permuted (CWZ footnote 7 — the proxy is
    time-permutation-invariant, so permuting residuals ≡ permuting data).
    """
    y1n = np.asarray(y1_obs, dtype=float).copy()
    y1n[post_mask] = y1n[post_mask] - np.asarray(effect_on_post, dtype=float)
    w, resid, conv = _cwz_proxy_fit(
        y1n,
        Y0,
        max_iter=max_iter,
        min_decrease=min_decrease,
        init_weights=init_weights,
        x1_rows=x1_rows,
        X0_rows=X0_rows,
    )
    p, s_obs, n = _cwz_pvalue(resid, post_mask, perms, q, alternative=alternative)
    return {"p_value": p, "s_observed": s_obs, "n_perms": n, "converged": conv, "weights": w}


def _auto_grid(center: float, spread: float, n_grid: int) -> np.ndarray:
    """Symmetric inversion grid centred on the point estimate.

    Half-width is a generous multiple of the pre-period residual scale (so a valid
    CI lies inside the grid for well-behaved fits); membership outside the grid is
    NOT certified (grid-limited — flagged via ``status`` when the accepted set
    touches an edge).
    """
    half = max(8.0 * spread, 0.5 * abs(center) + 1e-6)
    return np.linspace(center - half, center + half, n_grid)


def _assemble_single_ci(
    thetas: np.ndarray,
    pvals: np.ndarray,
    converged: np.ndarray,
    alpha: float,
    point_estimate: float,
) -> Dict[str, Any]:
    """Assemble a confidence interval from a single-post-slot inversion grid.

    A grid point is **rejected** ONLY when it converged AND ``p ≤ alpha``; a
    non-converged point is **indeterminate** (its p is unreliable) and is therefore
    NOT rejected — keeping it in the set is the conservative choice for an interval
    (excluding tail non-convergence would understate the width, the opposite of
    fail-closed). The confidence set is the complement of the rejections; ``in_set``
    (returned for the grid table) is ``~rejected``. The hull is ``[min, max]`` of the
    in-set ``θ`` with a ``contiguous`` flag (a *rejected* point strictly inside the
    hull ⇒ non-contiguous). ``n_nonconverged`` is surfaced so an interval widened by
    indeterminacy is detectable. ``status`` is one of ``{"ran", "grid_limited",
    "empty"}`` (the granularity-``unbounded`` case is handled by the caller).
    """
    rejected = converged & (pvals <= alpha)
    in_set = ~rejected
    n_nonconv = int(np.sum(~converged))
    acc = thetas[in_set]
    if acc.size == 0:
        return {
            "lower": float("nan"),
            "upper": float("nan"),
            "status": "empty",
            "contiguous": True,
            "point_estimate": float(point_estimate),
            "n_in_set": 0,
            "n_nonconverged": n_nonconv,
            "_in_set": in_set,
        }
    lower, upper = float(acc.min()), float(acc.max())
    inside = (thetas > lower) & (thetas < upper)
    contiguous = bool(not np.any(rejected[inside]))
    touches_edge = bool(in_set[0] or in_set[-1])
    status = "grid_limited" if touches_edge else "ran"
    return {
        "lower": lower,
        "upper": upper,
        "status": status,
        "contiguous": contiguous,
        "point_estimate": float(point_estimate),
        "n_in_set": int(acc.size),
        "n_nonconverged": n_nonconv,
        "_in_set": in_set,
    }


def _invert_single_post(
    y1_obs: np.ndarray,
    Y0: np.ndarray,
    post_idx: int,
    alpha: float,
    perms: np.ndarray,
    *,
    max_iter: int,
    min_decrease: float,
    grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    alternative: str = "two-sided",
    x1_rows: Optional[np.ndarray] = None,
    X0_rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Invert a single-post-slot conformal test over a grid (CWZ Algorithm 1).

    The series ``y1_obs`` ``(m,)`` / ``Y0`` ``(m, J)`` has exactly one post slot at
    ``post_idx`` (a per-period CI sub-series ``(pre ∪ {t})`` or a block-collapsed
    average series). For each candidate effect ``θ`` the post outcome is imputed
    (``y1[post_idx] − θ``), the proxy refit (warm-started across the grid — FW is
    convex so warm-starting only affects speed, not the optimum), and the
    permutation p-value recomputed. With one post slot ``S_q`` reduces to
    ``|û_post|`` for every ``q``, so ``q`` is inert here (fixed to 1); under
    ``alternative="greater"``/``"less"`` the single-slot statistic is the SIGNED
    ``û_post`` (resp. ``−û_post``), and the inversion yields a ONE-SIDED
    confidence set — a half-line whose finite endpoint is the reported bound
    (the infinite side is genuinely accepted, not grid-limited; the caller
    overrides the edge-touch status accordingly). Returns the CI summary plus
    the ``grid`` table of ``(θ, p, in_set, converged)`` rows.

    When ``alpha < 1/|Π|`` every candidate has ``p ≥ 1/|Π| > alpha`` (the identity is
    in ``Π``), so NO value is ever rejected and the confidence set is the whole line:
    this short-circuits to ``status="unbounded"`` with ``(-inf, +inf)`` endpoints and
    an empty grid (the per-point table is uninformative when nothing can be rejected).
    """
    y1_obs = np.asarray(y1_obs, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    m = y1_obs.shape[0]
    post_mask = np.zeros(m, dtype=bool)
    post_mask[post_idx] = True
    n_perms = int(perms.shape[0])

    # Un-absorbed point estimate + grid center from a PRE-ONLY proxy fit: predict the
    # post slot from a proxy fit on the other (pre) slots only. The θ=0 full-series fit
    # would let the proxy soak the effect into the weights, biasing the naive residual
    # toward 0 and mis-centring the auto-grid (CWZ inverts the test precisely to avoid
    # relying on that naive residual). ``spread`` is the pre-fit residual noise scale.
    pre_sel = ~post_mask
    w_pre, resid_pre, _ = _cwz_proxy_fit(
        y1_obs[pre_sel],
        Y0[pre_sel],
        max_iter=max_iter,
        min_decrease=min_decrease,
        x1_rows=x1_rows,
        X0_rows=X0_rows,
    )
    point_est = float(y1_obs[post_idx] - Y0[post_idx] @ w_pre)
    spread = float(np.std(resid_pre)) if resid_pre.size else 1.0

    if alpha < 1.0 / n_perms:
        # Granularity-unbounded: every null is accepted (p >= 1/|Π| > alpha).
        return {
            "lower": -float("inf"),
            "upper": float("inf"),
            "status": "unbounded",
            "contiguous": True,
            "point_estimate": point_est,
            "n_in_set": 0,
            "n_nonconverged": 0,
            "grid": [],
            "n_perms": n_perms,
        }

    if grid is None:
        grid = _auto_grid(point_est, spread, n_grid)
    else:
        grid = np.asarray(grid, dtype=float)

    pvals = np.empty(grid.shape[0], dtype=float)
    converged = np.empty(grid.shape[0], dtype=bool)
    w_prev = w_pre
    for i, theta in enumerate(grid):
        y1n = y1_obs.copy()
        y1n[post_idx] = y1n[post_idx] - float(theta)
        w, resid, conv = _cwz_proxy_fit(
            y1n,
            Y0,
            max_iter=max_iter,
            min_decrease=min_decrease,
            init_weights=w_prev,
            x1_rows=x1_rows,
            X0_rows=X0_rows,
        )
        w_prev = w
        p, _, _ = _cwz_pvalue(resid, post_mask, perms, 1, alternative=alternative)
        pvals[i] = p
        converged[i] = conv

    out = _assemble_single_ci(grid, pvals, converged, alpha, point_est)
    in_set = out.pop("_in_set")
    out["grid"] = list(zip(grid.tolist(), pvals.tolist(), in_set.tolist(), converged.tolist()))
    out["n_perms"] = n_perms
    return out


def _apply_one_sided_endpoints(res: Dict[str, Any], alternative: str) -> Dict[str, Any]:
    """Convert a one-sided inversion's accepted half-line into ±inf endpoints.

    Under ``alternative="greater"`` (H1: θ > θ0) small candidates are rejected
    and the accepted region extends to the grid's upper edge, so the upper
    endpoint becomes ``+inf`` and only a LOWER-edge touch keeps
    ``status="grid_limited"``. ``"less"`` is the mirror image. No-op for
    ``"two-sided"`` or for unbounded results (everything accepted).

    **Hull convention (no monotonicity guarantee).** Algorithm 1 accepts any
    candidate with ``p > α`` after REFITTING the proxy at that candidate; a
    monotone p-curve is typical but NOT guaranteed, so an
    accepted/rejected/accepted pattern is possible. Exactly as in the
    two-sided convention, the reported endpoints are then the HULL of the
    accepted set (with the infinite side attached) and ``contiguous`` is
    RECOMPUTED against the final reported ray — any rejected scanned point
    strictly inside it (including between the finite hull edge and the
    attached infinity) flips ``contiguous=False``, so a rejected pocket is
    disclosed, not hidden (callers warn on it).
    An all-rejected grid (the scan missed the ray — narrow user ``bounds``)
    reports the infinite side + an UNCERTIFIED (NaN) finite endpoint with
    ``grid_limited`` status: the ray always exists beyond the grid on the
    accepted side (``p → 1`` as ``θ0`` walks into the accepted direction).
    """
    if alternative == "two-sided" or res["status"] == "unbounded":
        return res
    if res["status"] == "empty":
        # Every scanned grid point rejected. Under a one-sided alternative
        # the accepted ray ALWAYS exists beyond the grid in the accepted
        # direction (as θ0 → ±∞ on that side, the signed statistic walks
        # into the permutation distribution's far tail and p → 1), so an
        # all-rejected grid means the scan missed it — typically narrow
        # user-supplied bounds. Report the infinite side with the finite
        # endpoint UNCERTIFIED (NaN) and grid_limited status rather than a
        # (false) empty confidence set.
        if alternative == "greater":
            res["upper"] = float("inf")
        else:
            res["lower"] = -float("inf")
        res["status"] = "grid_limited"
        return res
    grid = res.get("grid") or []
    if not grid:
        return res
    in_first = bool(grid[0][2])
    in_last = bool(grid[-1][2])
    if alternative == "greater":
        res["upper"] = float("inf")
        res["status"] = "grid_limited" if in_first else "ran"
        # Recompute contiguity against the FINAL reported ray: the
        # underlying flag only covered the finite accepted hull, so a
        # rejected scanned point ABOVE the hull (inside [lower, +inf) after
        # the rewrite) must flip it (REGISTRY hull-disclosure contract).
        lower = res["lower"]
        res["contiguous"] = bool(res["contiguous"]) and all(
            bool(row[2]) for row in grid if row[0] > lower
        )
    else:  # "less"
        res["lower"] = -float("inf")
        res["status"] = "grid_limited" if in_last else "ran"
        upper = res["upper"]
        res["contiguous"] = bool(res["contiguous"]) and all(
            bool(row[2]) for row in grid if row[0] < upper
        )
    return res


# =============================================================================
# Block collapse for the average-effect test (CWZ Appendix A.1)
# =============================================================================


def _block_collapse(
    y1: np.ndarray, Y0: np.ndarray, n_pre: int, n_post: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Collapse the calendar-ordered panel into non-overlapping ``T*``-blocks (A.1).

    The post block is the last ``n_post`` periods; pre blocks tile the pre-period
    backwards from ``T0``, so the earliest ``n_pre % n_post`` pre-periods are
    DROPPED to make the pre-block count integral (the paper assumes ``T/T*``
    integer). Each block is the per-unit average over its ``n_post`` periods.
    Returns ``(y1_blocks, Y0_blocks, n_dropped)`` with the post block last. Requires
    at least one full pre-block after the drop (``n_pre ≥ n_post``).
    """
    y1 = np.asarray(y1, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    t_star = n_post
    drop = n_pre % t_star
    n_pre_blocks = (n_pre - drop) // t_star
    pre_y1 = y1[drop:n_pre].reshape(n_pre_blocks, t_star).mean(axis=1)
    pre_Y0 = Y0[drop:n_pre].reshape(n_pre_blocks, t_star, Y0.shape[1]).mean(axis=1)
    post_y1 = y1[n_pre:].mean()
    post_Y0 = Y0[n_pre:].mean(axis=0)
    y1_blocks = np.concatenate([pre_y1, [post_y1]])
    Y0_blocks = np.vstack([pre_Y0, post_Y0[None, :]])
    return y1_blocks, Y0_blocks, drop
