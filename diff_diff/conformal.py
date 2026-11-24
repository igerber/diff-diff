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
proxy, which the ADH pre-period V-fit is not. The proxy is solved exactly, by one
non-negative least squares (:func:`diff_diff.utils._sc_weight_nnls`).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.utils import _sc_weight_nnls

__all__: List[str] = []

_INF = float("inf")


# =============================================================================
# Proxy fit (CWZ §2.3, eqs 3-4) — canonical constrained-LS synthetic control
# =============================================================================


def _cwz_proxy_fit(
    y1: np.ndarray,
    Y0: np.ndarray,
    *,
    x1_rows: Optional[np.ndarray] = None,
    X0_rows: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fit the canonical CWZ constrained-LS SC proxy over ALL given periods (eq 4).

    Minimises ``Σ_t (y1_t − Σ_j w_j Y0_{tj})²`` s.t. ``w ≥ 0, Σ w = 1`` (no
    intercept, no V-matrix). ``y1`` is the (already null-imputed) treated outcome
    vector ``(T,)``; ``Y0`` the donor outcomes ``(T, J)``. Solved exactly by
    :func:`_sc_weight_nnls` on the packed ``(T, J+1)`` (donor columns first, target
    last), ``zeta=0`` — one non-negative least squares whose solution lies on the unit
    simplex, so the ``w ≥ 0, Σ w = 1`` constraint is delivered with no extra
    normalization.

    The solve has no iteration budget to tune; ``converged`` is False only if the
    active set exhausted its step guard or the data left the float64 range (neither
    reached in practice), in which case the
    caller treats the grid point as indeterminate.

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
    w, converged = _sc_weight_nnls(packed, return_convergence=True)
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
        x1_rows=x1_rows,
        X0_rows=X0_rows,
    )
    p, s_obs, n = _cwz_pvalue(resid, post_mask, perms, q, alternative=alternative)
    return {"p_value": p, "s_observed": s_obs, "n_perms": n, "converged": conv, "weights": w}


# Auto-range padding (fraction of the bracket width), zoom cap, and the relative
# endpoint tolerance. Module-level so tests can monkeypatch them.
_AUTO_BRACKET_PAD = 0.01
_AUTO_GRID_MAX_ZOOMS = 8
_ENDPOINT_RTOL = 1e-6


def _single_slot_p_floor(perms: np.ndarray, post_idx: int) -> float:
    """Smallest p-value a single-post-slot test can attain.

    With one post slot the statistic of a permuted residual vector depends only on
    WHICH residual lands in that slot, so every permutation that leaves the slot in
    place reproduces the observed statistic and is counted under ``>=``. Hence
    ``p >= mean(perms[:, post_idx] == post_idx)`` for the two-sided and the signed
    one-sided statistic alike. That share is exactly ``1/m`` for the moving-block set
    and for an exact i.i.d. enumeration (``m`` = series length), and the empirical
    share for sampled i.i.d. draws; it is ``>= 1/|Π|`` because the identity is in
    ``Π``. ``alpha`` below it means no candidate can ever be rejected.
    """
    return float(np.mean(perms[:, post_idx] == post_idx))


def _auto_bracket(y1_obs: np.ndarray, Y0: np.ndarray, post_idx: int) -> Tuple[float, float]:
    """Candidate range that contains the finite boundary of the confidence set.

    The proxy weights lie on the unit simplex, so for ANY candidate ``θ`` and any
    fitted weights every pre-slot residual satisfies
    ``|û_t| <= R := max_{t pre, j} |y1_t − Y0_tj|`` and the post-slot residual is
    ``y1_post − θ − Y0_post·w`` with ``Y0_post·w`` inside
    ``[min_j Y0_post,j, max_j Y0_post,j]``. With
    ``θ_min = y1_post − max_j Y0_post,j`` and ``θ_max = y1_post − min_j Y0_post,j``,
    any ``θ`` outside ``[θ_min − R, θ_max + R]`` makes the post residual strictly the
    largest in absolute value: the two-sided p-value then equals the single-slot
    floor (rejected whenever ``alpha >= floor``). For the signed one-sided statistic
    one side of the bracket is rejected the same way and the other has ``p = 1``
    (the accepted ray). So the scan need not guess a scale from the point estimate
    or the pre-period fit.

    The range is padded by a fraction of its width, a small absolute term (so a
    zero-width range still has distinct edges) and a multiple of the floating-point
    resolution of the outcomes (so the strict inequality survives rounding at large
    outcome magnitudes). Raises ``ValueError`` if the range is not representable.
    """
    y1_obs = np.asarray(y1_obs, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    pre = np.ones(y1_obs.shape[0], dtype=bool)
    pre[post_idx] = False
    with np.errstate(over="ignore", invalid="ignore"):
        resid_bound = float(np.max(np.abs(y1_obs[pre][:, None] - Y0[pre]))) if pre.any() else 0.0
        y_post = float(y1_obs[post_idx])
        lo = y_post - float(np.max(Y0[post_idx])) - resid_bound
        hi = y_post - float(np.min(Y0[post_idx])) + resid_bound
        data_scale = max(float(np.max(np.abs(y1_obs))), float(np.max(np.abs(Y0))))
        pad = (
            _AUTO_BRACKET_PAD * (hi - lo)
            + 1e-6 * max(1.0, abs(lo), abs(hi))
            + 64.0 * float(np.finfo(float).eps) * data_scale
        )
        lo, hi = lo - pad, hi + pad
        width = hi - lo
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(width) and width > 0):
        raise ValueError(
            "conformal inversion: the outcome magnitudes are too large to form a finite "
            "candidate range. Rescale the outcome, or pass explicit bounds=(lo, hi)."
        )
    return float(lo), float(hi)


def _assemble_single_ci(
    thetas: np.ndarray,
    pvals: np.ndarray,
    converged: np.ndarray,
    alpha: float,
    point_estimate: float,
) -> Dict[str, Any]:
    """Assemble a confidence interval from single-post-slot inversion candidates.

    ``thetas`` is the sorted array of EVERY candidate evaluated (scan rows, the
    point estimate, endpoint-refinement candidates — see
    :func:`_invert_single_post`). A candidate is **rejected** ONLY when it converged
    AND ``p ≤ alpha``; a
    non-converged point is **indeterminate** (its p is unreliable) and is therefore
    NOT rejected — keeping it in the set is the conservative choice for an interval
    (excluding tail non-convergence would understate the width, the opposite of
    fail-closed). The confidence set is the complement of the rejections; ``in_set``
    (returned for the grid table) is ``~rejected``. The hull is ``[min, max]`` of the
    in-set ``θ`` with a ``contiguous`` flag (a *rejected* point strictly inside the
    hull ⇒ non-contiguous). ``n_nonconverged`` is surfaced so an interval widened by
    indeterminacy is detectable. ``status`` is one of ``{"ran", "grid_limited",
    "empty"}`` (the ``unbounded`` case is handled by the caller): ``"grid_limited"``
    when the outermost candidate on either side is in the set.
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
    grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    alternative: str = "two-sided",
    x1_rows: Optional[np.ndarray] = None,
    X0_rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Invert a single-post-slot conformal test over candidate effects (CWZ Algorithm 1).

    The series ``y1_obs`` ``(m,)`` / ``Y0`` ``(m, J)`` has exactly one post slot at
    ``post_idx`` (a per-period CI sub-series ``(pre ∪ {t})`` or a block-collapsed
    average series). For each candidate effect ``θ`` the post outcome is imputed
    (``y1[post_idx] − θ``), the proxy refit exactly (one NNLS per candidate; no
    warm start is needed or used), and the permutation p-value recomputed. With one
    post slot ``S_q`` reduces to ``|û_post|`` for every ``q``, so ``q`` is inert here
    (fixed to 1); under ``alternative="greater"``/``"less"`` the single-slot statistic
    is the SIGNED ``û_post`` (resp. ``−û_post``), and the inversion yields a ONE-SIDED
    confidence set — a half-line whose finite endpoint is the reported bound (the
    infinite side is genuinely accepted, not grid-limited; the caller overrides the
    edge-touch status accordingly).

    **Candidates.** Algorithm 1 inverts the test over candidate values without
    prescribing them. Every candidate evaluated is kept in ONE store, and the
    interval, ``contiguous``, the counts, and the returned ``grid`` table are all
    computed from that store — nothing evaluated is discarded:

    1. *Scan.* With ``grid`` supplied (the public ``bounds=`` path) its rows are the
       scan. Otherwise ``n_grid`` points are laid on :func:`_auto_bracket`, a range
       that contains the finite boundary of the set whatever the effect size or the
       quality of the pre-period fit.
    2. *Centre (two-sided).* The point estimate is evaluated as an extra candidate
       when it lies strictly inside the scanned range. It is a member of the
       two-sided set: at ``θ = point estimate`` the pre-only minimiser is also the
       full-series minimiser with a zero post residual, so ``p = 1``. (It is
       evaluated, not assumed.) So a scan whose rows all miss a narrow set does not
       report it empty.
    3. *Zoom (automatic range, two-sided).* While fewer than a quarter of the rows of
       the scan just made are in the set, ``n_grid`` more points are laid on the hull
       of ALL accepted candidates padded by one step of that scan, provided this
       halves the range; at most ``_AUTO_GRID_MAX_ZOOMS`` times. This only adds
       resolution inside the hull.
    4. *Endpoints.* Each finite hull endpoint with a (necessarily rejected) outward
       neighbour in the store is bisected against it, keeping an accepted inner and a
       rejected outer candidate, until the bracket is no wider than
       ``max(_ENDPOINT_RTOL · scale, 4 ulp)`` — ``scale`` the reported width
       (two-sided) or the endpoint's distance from the point estimate (one-sided),
       and the ulp taken at the magnitude of the imputed post outcome, below which a
       change in ``θ`` cannot change ``y1[post_idx] − θ``. There is no iteration cap:
       each step strictly shrinks the bracket among floats (the midpoint is formed
       without overflow, so the search also works near the float64 limit).

    So a reported finite endpoint is a candidate that was evaluated and not rejected,
    with a rejected candidate within that tolerance outside it. What is NOT
    guaranteed: Algorithm 1 has no monotonicity guarantee after per-candidate refits,
    so a rejected pocket inside the hull, or an accepted component away from the
    hull, that is narrower than the scan spacing can go unseen; ``contiguous`` and
    the hull reflect the candidates evaluated.

    Returns the CI summary plus the ``grid`` table of ``(θ, p, in_set, converged)``
    rows, one per distinct candidate evaluated, sorted by ``θ``; ``n_in_set`` and
    ``n_nonconverged`` count those rows. ``status`` is ``"grid_limited"`` when the
    outermost candidate on a side is in the set (a ``bounds=`` range that does not
    contain the set, or an automatic-range edge kept through non-convergence),
    ``"empty"`` when no candidate is, and otherwise ``"ran"``.

    When ``alpha`` is below :func:`_single_slot_p_floor` every candidate has
    ``p > alpha``, so NO value is ever rejected and the confidence set is the whole
    line: this short-circuits to ``status="unbounded"`` with ``(-inf, +inf)``
    endpoints and an empty table. (The floor is ``1/m`` for the moving-block set and
    for an exact i.i.d. enumeration — NOT ``1/|Π|``: a larger i.i.d. permutation set
    does not lower it for a single-slot test.)
    """
    y1_obs = np.asarray(y1_obs, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    m = y1_obs.shape[0]
    post_mask = np.zeros(m, dtype=bool)
    post_mask[post_idx] = True
    n_perms = int(perms.shape[0])
    two_sided = alternative == "two-sided"

    # Un-absorbed point estimate from a PRE-ONLY proxy fit: predict the post slot
    # from a proxy fit on the other (pre) slots only. The θ=0 full-series fit would
    # let the proxy soak the effect into the weights, biasing the naive residual
    # toward 0 (CWZ inverts the test precisely to avoid relying on it).
    pre_sel = ~post_mask
    w_pre, _, _ = _cwz_proxy_fit(
        y1_obs[pre_sel],
        Y0[pre_sel],
        x1_rows=x1_rows,
        X0_rows=X0_rows,
    )
    point_est = float(y1_obs[post_idx] - Y0[post_idx] @ w_pre)

    if alpha < _single_slot_p_floor(perms, post_idx):
        # Unbounded: every null is accepted (p >= floor > alpha).
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
            "n_zooms": 0,
        }

    # The candidate store: θ -> (p, converged). Every evaluation lands here once.
    store: Dict[float, Tuple[float, bool]] = {}

    def _evaluate(theta: float) -> Tuple[float, bool]:
        theta = float(theta)
        hit = store.get(theta)
        if hit is None:
            y1n = y1_obs.copy()
            y1n[post_idx] = y1n[post_idx] - theta
            _, resid, conv = _cwz_proxy_fit(
                y1n,
                Y0,
                x1_rows=x1_rows,
                X0_rows=X0_rows,
            )
            p, _, _ = _cwz_pvalue(resid, post_mask, perms, 1, alternative=alternative)
            hit = (float(p), bool(conv))
            store[theta] = hit
        return hit

    def _accepted(theta: float) -> bool:
        # Same rule as _assemble_single_ci: only a CONVERGED p <= alpha rejects.
        p, conv = _evaluate(theta)
        return not (conv and p <= alpha)

    n_zooms = 0
    if grid is not None:
        scan = np.asarray(grid, dtype=float)
        for theta in scan:
            _evaluate(theta)
        if two_sided and float(np.min(scan)) < point_est < float(np.max(scan)):
            _evaluate(point_est)
    else:
        lo, hi = _auto_bracket(y1_obs, Y0, post_idx)
        if two_sided and lo < point_est < hi:
            _evaluate(point_est)
        while True:
            scan = np.linspace(lo, hi, n_grid)
            n_in = sum(_accepted(theta) for theta in scan)
            if not two_sided or 4 * n_in >= n_grid or n_zooms >= _AUTO_GRID_MAX_ZOOMS:
                break
            members = [theta for theta in store if _accepted(theta)]
            if not members:
                break  # nothing accepted anywhere: reported as "empty" below
            step = (hi - lo) / (n_grid - 1)
            new_lo, new_hi = min(members) - step, max(members) + step
            if not (new_hi - new_lo) <= 0.5 * (hi - lo):
                break
            lo, hi = new_lo, new_hi
            n_zooms += 1

    def _assemble() -> Tuple[np.ndarray, Dict[str, Any]]:
        thetas = np.array(sorted(store), dtype=float)
        pvals = np.array([store[t][0] for t in thetas], dtype=float)
        conv = np.array([store[t][1] for t in thetas], dtype=bool)
        return thetas, _assemble_single_ci(thetas, pvals, conv, alpha, point_est)

    thetas, out = _assemble()
    if out["status"] != "empty":
        members_idx = np.flatnonzero(out["_in_set"])
        i_lo, i_hi = int(members_idx[0]), int(members_idx[-1])
        ends = [float(thetas[i_lo]), float(thetas[i_hi])]
        y_post_abs = abs(float(y1_obs[post_idx]))

        def _bisect(side: int, outer: float) -> None:
            inner = ends[side]
            while True:
                scale = (ends[1] - ends[0]) if two_sided else abs(inner - point_est)
                resolution = 4.0 * float(np.spacing(max(y_post_abs, abs(inner), abs(outer))))
                if abs(outer - inner) <= max(_ENDPOINT_RTOL * scale, resolution):
                    break
                # Halve each term before adding: ``0.5 * (inner + outer)`` overflows
                # for same-signed endpoints near the float64 limit, and the resulting
                # inf would be mistaken for "nothing left between them".
                mid = 0.5 * inner + 0.5 * outer
                if not (min(inner, outer) < mid < max(inner, outer)):
                    break  # adjacent floats: nothing left between them
                if _accepted(mid):
                    inner = mid
                    ends[side] = inner
                else:
                    outer = mid

        # Lower first, so the upper side's tolerance sees the refined width. A
        # one-sided set has one finite side: lower for "greater", upper for "less".
        if alternative != "less" and i_lo > 0:
            _bisect(0, float(thetas[i_lo - 1]))
        if alternative != "greater" and i_hi < thetas.shape[0] - 1:
            _bisect(1, float(thetas[i_hi + 1]))
        thetas, out = _assemble()

    in_set = out.pop("_in_set")
    out["grid"] = [
        (float(t), store[float(t)][0], bool(member), store[float(t)][1])
        for t, member in zip(thetas, in_set)
    ]
    out["n_perms"] = n_perms
    out["n_zooms"] = n_zooms
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
