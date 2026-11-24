"""Candidate search of the single-post-slot conformal inversion (CWZ 2021, Algorithm 1).

Pins the contract of ``diff_diff.conformal._invert_single_post``:

* the automatic range is a bracket that contains the finite boundary of the set
  (simplex weights bound every residual), so the result does not depend on the
  size of the effect or the quality of the pre-period fit;
* every candidate evaluated is kept, and the interval, ``contiguous``, the counts
  and the public table all derive from that one set of rows;
* each finite endpoint is an accepted candidate with a rejected candidate just
  outside it (bisection), on the automatic range and on explicit ``bounds=``;
* ``unbounded`` is decided by the single-slot p-value floor, not ``1/|Pi|``.

Helpers added by this change (``_auto_bracket``, ``_single_slot_p_floor``) are
imported inside the tests that use them, so the remaining tests stay runnable
against a tree that predates them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff.conformal as cf
from diff_diff import SyntheticControl
from diff_diff.conformal import (
    _apply_one_sided_endpoints,
    _block_collapse,
    _invert_single_post,
    _make_perms,
    _moving_block_perms,
    _single_null_pvalue,
)

_FAST = dict(n_starts=1, optimizer_options={"maxiter": 50})


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _panel(seed=0, T0=40, J=12, effect=0.0, sigma=1.0):
    """One-factor panel, treated = convex mix of donors + noise; the LAST slot is post."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=T0 + 1))
    lam = rng.uniform(0.5, 1.5, size=J)
    Y0 = 100.0 + np.outer(f, lam) + rng.normal(scale=sigma, size=(T0 + 1, J))
    w = rng.dirichlet(np.ones(J))
    y1 = Y0 @ w + rng.normal(scale=sigma, size=T0 + 1)
    y1[-1] += effect
    return y1, Y0


def _hetero_panel(seed=5, T0=40):
    """Donors at levels 10 / 1e4 / 1e6 / 50: the bracket is ~1e6 times the interval."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=T0 + 1))
    Y0 = np.column_stack(
        [
            10.0 + f + rng.normal(size=T0 + 1),
            1e4 + 3.0 * f + rng.normal(size=T0 + 1),
            1e6 + f + rng.normal(size=T0 + 1),
            50.0 + 2.0 * f + rng.normal(size=T0 + 1),
        ]
    )
    y1 = Y0 @ np.array([0.5, 0.0, 0.0, 0.5]) + rng.normal(size=T0 + 1)
    return y1, Y0


def _unit_set_panel():
    """Pre residuals evenly spaced on [-1, 1], zero post residual, one zero donor.

    The two-sided 80% set is exactly ``[-1, 1]`` (|post residual| must not exceed the
    largest pre residual), which makes coarse-grid behaviour easy to state.
    """
    y1 = np.concatenate([np.linspace(-1.0, 1.0, 10), [0.0]])
    Y0 = np.zeros((11, 1))
    return y1, Y0, _moving_block_perms(11)


def _oracle_p(y1, Y0, theta, perms, alternative="two-sided"):
    """Independent single-null p-value at ``theta`` (the joint-test code path)."""
    post_mask = np.zeros(len(y1), dtype=bool)
    post_mask[-1] = True
    out = _single_null_pvalue(
        y1, Y0, post_mask, np.array([theta]), perms, 1, alternative=alternative
    )
    return float(out["p_value"])


def _invert(y1, Y0, alpha=0.1, perms=None, **kw):
    perms = _moving_block_perms(len(y1)) if perms is None else perms
    return _invert_single_post(y1, Y0, len(y1) - 1, alpha, perms, **kw)


def _brute_hull(y1, Y0, lo, hi, n, alpha, perms=None, alternative="two-sided"):
    perms = _moving_block_perms(len(y1)) if perms is None else perms
    thetas = np.linspace(lo, hi, n)
    acc = np.array([_oracle_p(y1, Y0, t, perms, alternative) > alpha for t in thetas])
    return thetas[acc].min(), thetas[acc].max(), thetas[1] - thetas[0]


def _frame(effect, seed=11, n_donors=8, T0=30, n_post=3, sigma=1.0):
    """Long panel for the public API: residual sd ``sigma``, constant ``effect``."""
    rng = np.random.default_rng(seed)
    T = T0 + n_post
    f = np.cumsum(rng.normal(size=T))
    lam = rng.uniform(0.5, 1.5, size=n_donors)
    Y0 = 100.0 + np.outer(f, lam) + rng.normal(scale=sigma, size=(T, n_donors))
    w = rng.dirichlet(np.ones(n_donors))
    y1 = Y0 @ w + rng.normal(scale=sigma, size=T)
    y1[T0:] += effect
    periods = list(range(1, T + 1))
    rows = [
        {"unit": f"d{j}", "period": periods[t], "y": Y0[t, j], "treated": 0}
        for j in range(n_donors)
        for t in range(T)
    ]
    rows += [
        {"unit": "treated", "period": periods[t], "y": y1[t], "treated": int(t >= T0)}
        for t in range(T)
    ]
    return pd.DataFrame(rows), periods[T0:]


def _fit(effect, **kw):
    df, post = _frame(effect, **kw)
    return SyntheticControl(**_FAST).fit(
        df, "y", "treated", "unit", "period", post_periods=post, treated_unit="treated"
    )


# ---------------------------------------------------------------------------
# 1. the bracket contains the finite boundary of the set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_nothing_outside_the_bracket_is_accepted(seed):
    from diff_diff.conformal import _auto_bracket, _single_slot_p_floor

    y1, Y0 = _panel(seed, T0=4 + 3 * seed, J=2 + seed, effect=40.0 * (seed % 4))
    m = len(y1)
    perms = _moving_block_perms(m)
    lo, hi = _auto_bracket(y1, Y0, m - 1)
    floor = _single_slot_p_floor(perms, m - 1)
    assert floor == pytest.approx(1.0 / m)
    width = hi - lo
    for rel in (1e-9, 1e-3, 1.0, 1e3):
        below, above = lo - rel * width, hi + rel * width
        # two-sided: the post residual is strictly the largest -> p sits on the floor
        assert _oracle_p(y1, Y0, below, perms) == pytest.approx(floor)
        assert _oracle_p(y1, Y0, above, perms) == pytest.approx(floor)
        # one-sided: one side on the floor (rejected), the other p = 1 (the ray)
        assert _oracle_p(y1, Y0, below, perms, "greater") == pytest.approx(floor)
        assert _oracle_p(y1, Y0, above, perms, "greater") == pytest.approx(1.0)
        assert _oracle_p(y1, Y0, below, perms, "less") == pytest.approx(1.0)
        assert _oracle_p(y1, Y0, above, perms, "less") == pytest.approx(floor)


def test_bracket_has_positive_width_on_a_degenerate_fit():
    from diff_diff.conformal import _auto_bracket

    # one donor identical to the treated unit: zero residual bound, zero donor spread
    y = np.arange(8, dtype=float)
    lo, hi = _auto_bracket(y, y[:, None].copy(), 7)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
    # all-zero data: still a usable range
    lo, hi = _auto_bracket(np.zeros(6), np.zeros((6, 2)), 5)
    assert hi > lo


def test_bracket_overflow_raises():
    y1 = np.zeros(11)
    Y0 = np.zeros((11, 2))
    Y0[10] = [1e308, -1e308]
    with pytest.raises(ValueError, match="too large to form a finite candidate range"):
        _invert(y1, Y0, alpha=0.2)


def test_bracket_padding_survives_large_outcome_magnitudes():
    # Every outcome is 1e16: subtracting a candidate below the float spacing (2.0)
    # changes nothing, so the set is about |theta| <= 1. The padding must reach past
    # that resolution for the edges to be rejected.
    y1 = np.full(11, 1e16)
    Y0 = np.full((11, 2), 1e16)
    res = _invert(y1, Y0, alpha=0.2)
    assert res["status"] == "ran"
    assert -4.0 <= res["lower"] <= 0.0 <= res["upper"] <= 4.0


# ---------------------------------------------------------------------------
# 2-4. the interval does not depend on the effect size or on n_grid
# ---------------------------------------------------------------------------


def test_interval_is_shift_equivariant_in_the_effect():
    y1, Y0 = _panel(0)
    base = _invert(y1, Y0)
    assert base["status"] == "ran"
    assert base["upper"] - base["lower"] > 3.0  # residual sd 1, 90% interval
    for effect in (50.0, 500.0, 5000.0):
        shifted = y1.copy()
        shifted[-1] += effect
        res = _invert(shifted, Y0)
        assert res["status"] == "ran"
        np.testing.assert_allclose(res["lower"] - effect, base["lower"], atol=1e-4)
        np.testing.assert_allclose(res["upper"] - effect, base["upper"], atol=1e-4)


@pytest.mark.parametrize("effect", [0.0, 50.0])
def test_endpoints_are_the_boundary_of_the_set(effect):
    from diff_diff.conformal import _auto_bracket

    alpha = 0.1
    y1, Y0 = _panel(1, effect=effect)
    perms = _moving_block_perms(len(y1))
    res = _invert(y1, Y0, alpha)
    lower, upper = res["lower"], res["upper"]
    # accepted at the endpoints, rejected just outside them
    assert _oracle_p(y1, Y0, lower, perms) > alpha
    assert _oracle_p(y1, Y0, upper, perms) > alpha
    tol = 1e-4 * (upper - lower)
    assert _oracle_p(y1, Y0, lower - tol, perms) <= alpha
    assert _oracle_p(y1, Y0, upper + tol, perms) <= alpha
    # ... and they are the hull of an independent brute-force scan of the bracket
    lo, hi = _auto_bracket(y1, Y0, len(y1) - 1)
    b_lo, b_hi, step = _brute_hull(y1, Y0, lo, hi, 4001, alpha)
    assert b_lo - step <= lower <= b_lo
    assert b_hi <= upper <= b_hi + step
    # the endpoints are rows of the table
    params = [row[0] for row in res["grid"]]
    assert lower in params and upper in params


@pytest.mark.parametrize("effect", [0.0, 5000.0])
@pytest.mark.parametrize("n_grid", [2, 3, 4, 5])
def test_tiny_n_grid_gives_the_same_interval(effect, n_grid):
    y1, Y0 = _panel(0, effect=effect)
    ref = _invert(y1, Y0, n_grid=100)
    res = _invert(y1, Y0, n_grid=n_grid)
    assert res["status"] == "ran"
    np.testing.assert_allclose(res["lower"], ref["lower"], atol=1e-4)
    np.testing.assert_allclose(res["upper"], ref["upper"], atol=1e-4)


# ---------------------------------------------------------------------------
# 5. zoom
# ---------------------------------------------------------------------------


def test_zoom_refines_a_set_far_narrower_than_the_bracket(monkeypatch):
    from diff_diff.conformal import _auto_bracket

    alpha = 0.1
    y1, Y0 = _hetero_panel()
    lo, hi = _auto_bracket(y1, Y0, len(y1) - 1)
    res = _invert(y1, Y0, alpha)
    assert res["status"] == "ran"
    assert res["n_zooms"] >= 2
    assert (hi - lo) > 1e5 * (res["upper"] - res["lower"])
    centre = res["point_estimate"]
    assert res["lower"] <= centre <= res["upper"]
    b_lo, b_hi, step = _brute_hull(y1, Y0, centre - 10.0, centre + 10.0, 4001, alpha)
    assert b_lo - step <= res["lower"] <= b_lo
    assert b_hi <= res["upper"] <= b_hi + step
    # With zooming switched off the scan rows all miss the set; the point estimate
    # and bisection still locate the same endpoints, from fewer candidates.
    monkeypatch.setattr(cf, "_AUTO_GRID_MAX_ZOOMS", 0)
    flat = _invert(y1, Y0, alpha)
    assert flat["n_zooms"] == 0 and flat["status"] == "ran"
    np.testing.assert_allclose(flat["lower"], res["lower"], atol=1e-4)
    np.testing.assert_allclose(flat["upper"], res["upper"], atol=1e-4)
    assert len(flat["grid"]) < len(res["grid"])


# ---------------------------------------------------------------------------
# 6. one-sided
# ---------------------------------------------------------------------------


def test_one_sided_finite_endpoint_is_bisected_and_ray_is_not_grid_limited():
    alpha = 0.1
    y1, Y0 = _panel(0)
    perms = _moving_block_perms(len(y1))
    two = _invert(y1, Y0, alpha)
    hi = _apply_one_sided_endpoints(_invert(y1, Y0, alpha, alternative="greater"), "greater")
    lo = _apply_one_sided_endpoints(_invert(y1, Y0, alpha, alternative="less"), "less")
    assert hi["upper"] == np.inf and hi["status"] == "ran" and hi["contiguous"]
    assert lo["lower"] == -np.inf and lo["status"] == "ran" and lo["contiguous"]
    # the finite endpoints are boundaries of the one-sided sets ...
    assert _oracle_p(y1, Y0, hi["lower"], perms, "greater") > alpha
    assert _oracle_p(y1, Y0, hi["lower"] - 1e-4, perms, "greater") <= alpha
    assert _oracle_p(y1, Y0, lo["upper"], perms, "less") > alpha
    assert _oracle_p(y1, Y0, lo["upper"] + 1e-4, perms, "less") <= alpha
    # ... and sit inside the two-sided interval (a one-sided test is less strict there)
    assert two["lower"] <= hi["lower"] and lo["upper"] <= two["upper"]


def test_one_sided_ray_found_when_the_point_estimate_is_rejected():
    # Treated pre-period series strictly below every donor: all proxy residuals are
    # negative, so at theta = point estimate (post residual 0) the signed "greater"
    # statistic is the largest and the point estimate itself is rejected.
    alpha = 0.2
    rng = np.random.default_rng(3)
    T0, J = 12, 4
    Y0 = 10.0 + rng.normal(size=(T0 + 1, J))
    y1 = Y0.min(axis=1) - 1.0 - rng.uniform(0.0, 0.5, size=T0 + 1)
    perms = _moving_block_perms(T0 + 1)
    raw = _invert(y1, Y0, alpha, alternative="greater")
    assert _oracle_p(y1, Y0, raw["point_estimate"], perms, "greater") <= alpha
    res = _apply_one_sided_endpoints(raw, "greater")
    assert res["status"] == "ran" and res["upper"] == np.inf
    assert np.isfinite(res["lower"]) and res["lower"] > res["point_estimate"]
    assert _oracle_p(y1, Y0, res["lower"], perms, "greater") > alpha
    assert _oracle_p(y1, Y0, res["lower"] - 1e-4, perms, "greater") <= alpha


# ---------------------------------------------------------------------------
# 7-8. nothing evaluated is discarded
# ---------------------------------------------------------------------------


def test_non_converged_candidate_survives_a_zoom(monkeypatch):
    # Force exactly ONE first-scan candidate to report a non-converged proxy fit. It
    # is indeterminate, so it stays in the set; the zooms that follow lay new grids
    # that do not contain it, and it must still decide the hull and `contiguous`.
    from diff_diff.conformal import _auto_bracket

    alpha = 0.1
    y1, Y0 = _hetero_panel()
    m = len(y1)
    clean = _invert(y1, Y0, alpha)
    lo, hi = _auto_bracket(y1, Y0, m - 1)
    scan = np.linspace(lo, hi, 100)
    # the first-scan row just above the clean interval: far outside the set, but
    # close enough (one scan step) that the hull still shrinks and the zoom fires
    target = float(scan[scan > clean["upper"]][0])
    assert target - clean["upper"] > 100.0 * (clean["upper"] - clean["lower"])
    imputed = y1[-1] - target
    real = cf._cwz_proxy_fit

    def one_fails(y, Y, **kw):
        w, resid, conv = real(y, Y, **kw)
        if len(y) == m and y[-1] == imputed:
            conv = False
        return w, resid, conv

    monkeypatch.setattr(cf, "_cwz_proxy_fit", one_fails)
    res = _invert(y1, Y0, alpha)
    assert res["n_zooms"] >= 1
    rows = {row[0]: row for row in res["grid"]}
    assert target in rows
    _, _, in_set, converged = rows[target]
    assert in_set and not converged
    assert res["n_nonconverged"] == 1
    assert res["upper"] == target  # the hull reaches the indeterminate candidate
    assert res["contiguous"] is False  # rejected candidates lie between it and the set
    # the other endpoint is still the set's boundary, to the stated tolerance: 1e-6
    # of the REPORTED width, which the far candidate has stretched
    width = res["upper"] - res["lower"]
    assert clean["lower"] <= res["lower"] <= clean["lower"] + 2e-6 * width


def test_table_is_the_single_source_of_the_summary():
    y1, Y0 = _panel(2, effect=50.0)
    res = _invert(y1, Y0)
    params = np.array([row[0] for row in res["grid"]])
    in_set = np.array([row[2] for row in res["grid"]])
    conv = np.array([row[3] for row in res["grid"]])
    assert np.all(np.diff(params) > 0)  # sorted, one row per distinct candidate
    assert res["n_in_set"] == int(in_set.sum())
    assert res["n_nonconverged"] == int((~conv).sum())
    assert res["lower"] == params[in_set].min() and res["upper"] == params[in_set].max()
    assert res["point_estimate"] in params  # the centre is a candidate (two-sided)
    assert not in_set[0] and not in_set[-1]  # bracket edges rejected -> "ran"
    assert res["status"] == "ran"


# ---------------------------------------------------------------------------
# 9-10. explicit bounds, and floating-point limits
# ---------------------------------------------------------------------------


def test_bounds_path_scans_only_the_bounds_and_bisects_inside_them():
    alpha = 0.1
    y1, Y0 = _panel(0, effect=5.0)
    perms = _moving_block_perms(len(y1))
    grid = np.linspace(-2.0, 12.0, 15)  # step 1.0: coarse next to a ~4.3-wide set
    res = _invert(y1, Y0, alpha, grid=grid)
    params = np.array([row[0] for row in res["grid"]])
    assert set(grid) <= set(params)
    assert params.min() == -2.0 and params.max() == 12.0
    assert res["status"] == "ran" and res["n_zooms"] == 0
    assert _oracle_p(y1, Y0, res["lower"], perms) > alpha
    assert _oracle_p(y1, Y0, res["lower"] - 1e-4, perms) <= alpha
    assert _oracle_p(y1, Y0, res["upper"] + 1e-4, perms) <= alpha
    auto = _invert(y1, Y0, alpha)
    np.testing.assert_allclose(
        [res["lower"], res["upper"]], [auto["lower"], auto["upper"]], atol=1e-4
    )


@pytest.mark.parametrize("half", [1e9, 1e100, 1e300])
@pytest.mark.parametrize("n_grid", [2, 3, 4])
def test_coarse_bounds_still_locate_the_set(half, n_grid):
    y1, Y0, perms = _unit_set_panel()
    res = _invert(y1, Y0, 0.2, perms, grid=np.linspace(-half, half, n_grid))
    assert res["status"] == "ran"
    np.testing.assert_allclose([res["lower"], res["upper"]], [-1.0, 1.0], atol=1e-5)


def test_bounds_that_miss_the_set_and_the_point_estimate_are_empty():
    y1, Y0, perms = _unit_set_panel()
    res = _invert(y1, Y0, 0.2, perms, grid=np.linspace(5.0, 9.0, 5))
    assert res["status"] == "empty"
    assert np.isnan(res["lower"]) and np.isnan(res["upper"])
    assert len(res["grid"]) == 5  # the point estimate (0) lies outside the bounds


def test_bounds_edge_inside_the_set_is_grid_limited_and_not_refined():
    y1, Y0, perms = _unit_set_panel()
    res = _invert(y1, Y0, 0.2, perms, grid=np.linspace(-0.5, 5.0, 12))
    assert res["status"] == "grid_limited"
    assert res["lower"] == -0.5  # the accepted edge: nothing outside it was scanned
    np.testing.assert_allclose(res["upper"], 1.0, atol=1e-5)


def test_large_pre_period_donor_does_not_stop_the_bisection():
    # A candidate enters only through the post slot, so one donor that is huge in the
    # PRE period must not coarsen the endpoint search. Set: [-1, 1].
    y1 = np.concatenate([np.tile([1.0, -1.0], 10), [0.0]])
    Y0 = np.zeros((21, 2))
    Y0[:20, 1] = 1e16
    perms = _make_perms(21, "iid", 500, np.random.default_rng(0))
    res = _invert(y1, Y0, 0.1, perms, grid=np.linspace(-10.0, 10.0, 2))
    assert res["status"] == "ran"
    np.testing.assert_allclose([res["lower"], res["upper"]], [-1.0, 1.0], atol=1e-5)


def test_bisection_terminates_at_floating_point_resolution():
    # Point estimate 1e16 with unit-scale residuals: the float spacing there is 2.0,
    # wider than the set, so the interval collapses to the one representable member.
    y1 = np.concatenate([np.tile([1.0, -1.0], 5), [1e16]])
    Y0 = np.zeros((11, 1))
    perms = _moving_block_perms(11)
    res = _invert(y1, Y0, 0.2, perms, grid=np.array([1e16 - 10.0, 1e16 + 10.0]))
    assert res["status"] == "ran"
    assert res["lower"] == res["upper"] == 1e16
    # all-zero data: the set is the single point 0, and the search still terminates
    res = _invert(np.zeros(11), Y0, 0.2, perms)
    assert res["status"] == "ran"
    assert abs(res["lower"]) < 1e-12 and abs(res["upper"]) < 1e-12


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("n_grid", [2, 100])
@pytest.mark.parametrize("explicit", [False, True])
def test_bisection_near_the_float64_limit(sign, n_grid, explicit):
    # Outcomes at 1e307-1e308: the midpoint of two same-signed endpoints must not
    # overflow (an inf midpoint would end the search with the unrefined grid hull
    # reported as "ran"). The same panel divided by 1e307 has set [9, 11].
    y1 = sign * np.array([1e307, -1e307] * 5 + [1e308])
    Y0 = np.zeros((11, 1))
    perms = _moving_block_perms(11)
    grid = np.linspace(sign * 1e308 - 3e307, sign * 1e308 + 3e307, n_grid) if explicit else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = _invert(y1, Y0, 0.2, perms, n_grid=n_grid, grid=grid)
        scaled = _invert(y1 / 1e307, Y0, 0.2, perms, n_grid=100)
        inside = _oracle_p(y1, Y0, sign * 9.5e307, perms)
    assert res["status"] == "ran" and scaled["status"] == "ran"
    np.testing.assert_allclose(
        sorted([abs(scaled["lower"]), abs(scaled["upper"])]), [9.0, 11.0], atol=1e-5
    )
    np.testing.assert_allclose(res["lower"] / 1e307, scaled["lower"], rtol=1e-5)
    np.testing.assert_allclose(res["upper"] / 1e307, scaled["upper"], rtol=1e-5)
    assert inside == 1.0 and res["lower"] < sign * 9.5e307 < res["upper"]


# ---------------------------------------------------------------------------
# 11. the unbounded floor
# ---------------------------------------------------------------------------


def test_single_slot_floor_helper():
    from diff_diff.conformal import _single_slot_p_floor

    rng = np.random.default_rng(0)
    assert _single_slot_p_floor(_moving_block_perms(9), 8) == pytest.approx(1 / 9)
    exact = _make_perms(5, "iid", 10_000, rng)  # 5! = 120 <= n_iid -> enumeration
    assert exact.shape[0] == 120
    assert _single_slot_p_floor(exact, 4) == pytest.approx(1 / 5)
    sampled = _make_perms(20, "iid", 400, rng)
    assert sampled.shape[0] == 400
    floor = _single_slot_p_floor(sampled, 19)
    assert floor == float(np.mean(sampled[:, 19] == 19))
    assert floor >= 1 / 400  # the identity is always in the set


def test_no_candidate_can_be_rejected_below_the_floor():
    # 6 slots, exact i.i.d. enumeration (|Pi| = 720): the old 1/|Pi| check let this
    # through and reported a finite grid_limited interval for a whole-line set.
    y1, Y0 = _panel(4, T0=5, J=3, effect=5.0)
    perms = _make_perms(6, "iid", 2000, np.random.default_rng(0))
    assert perms.shape[0] == 720
    for theta in np.linspace(-1e4, 1e4, 41):
        assert _oracle_p(y1, Y0, theta, perms) >= 1 / 6 - 1e-12
    res = _invert(y1, Y0, 0.1, perms)
    assert res["status"] == "unbounded"
    assert res["lower"] == -np.inf and res["upper"] == np.inf
    assert res["grid"] == []
    bounded = _invert(y1, Y0, 0.2, perms)  # alpha above the floor
    assert bounded["status"] == "ran"


def test_public_average_effect_is_unbounded_below_the_floor():
    # T0 = 30, T* = 5 -> 7 blocks -> floor 1/7 = 0.143 under EITHER scheme.
    res = _fit(5.0, n_post=5)
    for scheme in ("moving_block", "iid"):
        with pytest.warns(UserWarning, match="granularity"):
            s = res.conformal_average_effect(alpha=0.1, scheme=scheme, n_iid=2000, seed=0)
        assert int(s["n_blocks"]) == 7
        assert s["status"] == "unbounded"
        assert s["lower"] == -np.inf and s["upper"] == np.inf
        assert res.get_conformal_grid_df().empty
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a bounded, contiguous interval warns nothing
        s = res.conformal_average_effect(alpha=0.2, scheme="iid", n_iid=2000, seed=0)
    assert s["status"] == "ran" and np.isfinite(s["lower"]) and np.isfinite(s["upper"])


# ---------------------------------------------------------------------------
# 13-14. coverage and the public API at a large effect
# ---------------------------------------------------------------------------


def test_public_api_at_a_large_effect():
    # Residual sd 1, effect 500: the old grid (step ~ 5) stepped over every interval.
    effect = 500.0
    res = _fit(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = res.conformal_confidence_intervals(alpha=0.1)
        ci_grid = res.get_conformal_grid_df()
        avg = res.conformal_average_effect(alpha=0.1)
    assert (ci["status"] == "ran").all()
    assert (ci["upper"] - ci["lower"]).between(0.5, 20.0).all()
    covered = ((ci["lower"] <= effect) & (effect <= ci["upper"])).sum()
    assert covered >= len(ci) - 1  # 90% intervals: allow one miss
    # per-period counts are counts of that period's table rows
    in_set = ci_grid.groupby("period")["in_set"].sum()
    assert in_set.loc[list(ci["period"])].tolist() == ci["n_grid_in_set"].tolist()
    assert avg["status"] == "ran"
    assert np.isfinite(avg["lower"]) and np.isfinite(avg["upper"])
    assert 0.1 < avg["upper"] - avg["lower"] < 20.0
    assert abs(avg["point_estimate"] - effect) < 5.0


def test_public_near_exact_fit_reports_the_whole_hull():
    # Block-collapsed series with as many donors as pre-blocks: the pre-period fit is
    # near exact and the point estimate near zero, so the old estimate-scaled grid
    # (half-width under 0.01) saw only a sliver of a set that is several units wide.
    from diff_diff.conformal import _auto_bracket

    alpha = 0.2
    rng = np.random.default_rng(1028)
    T, T0, J = 20, 16, 5
    Y0 = np.column_stack(
        [
            rng.normal(10, 2) + rng.normal(0, 0.3) * np.arange(T) + rng.normal(0, 0.15, T)
            for _ in range(J)
        ]
    )
    y1 = 0.6 * Y0[:, 0] + 0.4 * Y0[:, 1] + rng.normal(0, 0.08, T)
    y1b, Y0b, _ = _block_collapse(y1, Y0, T0, T - T0)
    nb = len(y1b)
    perms = _make_perms(nb, "iid", 400, np.random.default_rng(28))
    res = _invert_single_post(y1b, Y0b, nb - 1, alpha, perms, n_grid=200)
    assert res["status"] == "ran"
    lo, hi = _auto_bracket(y1b, Y0b, nb - 1)
    b_lo, b_hi, step = _brute_hull(y1b, Y0b, lo, hi, 8001, alpha, perms)
    assert b_lo - step <= res["lower"] <= b_lo + 1e-9
    assert b_hi - 1e-9 <= res["upper"] <= b_hi + step
    assert res["lower"] <= 0.0 <= res["upper"]  # the true effect is 0
    assert res["upper"] - res["lower"] > 100.0 * (0.5 * abs(res["point_estimate"]) + 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("effect", [0.0, 500.0])
def test_coverage_does_not_depend_on_the_effect_size(effect):
    alpha, reps, covered = 0.2, 200, 0
    for rep in range(reps):
        y1, Y0 = _panel(1000 + rep, T0=24, J=6, effect=effect)
        res = _invert(y1, Y0, alpha)
        assert res["status"] == "ran"
        covered += int(res["lower"] <= effect <= res["upper"])
    assert covered / reps >= 0.70, f"coverage {covered / reps:.3f} at effect {effect}"
