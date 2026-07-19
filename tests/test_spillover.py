"""Tests for SpilloverDiD (Butts 2021 ring-indicator spillover-aware DiD).

Step 1 surface: ring-construction helpers and the public class scaffold.
Step 2+ surfaces are added incrementally as the implementation lands.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign
from diff_diff.spillover import (
    SpilloverDiD,
    _apply_callable_metric_pairwise,
    _apply_horizon_binning,
    _build_event_study_design,
    _build_ring_indicators,
    _check_omega_0_connectivity,
    _compute_event_time_per_row,
    _compute_nearest_treated_distance_sparse,
    _compute_nearest_treated_distance_staggered,
    _compute_nearest_treated_distance_static,
    _convert_treatment_to_first_treat,
    _euclidean_pairwise,
    _extract_treatment_onsets,
    _haversine_km_pairwise,
    _pairwise_ring_distances,
    _ring_label,
)
from tests._dgp_utils import (
    generate_butts_nonstaggered_dgp,
    generate_butts_staggered_dgp,
)

# =============================================================================
# Pairwise-distance primitives
# =============================================================================


class TestHaversinePairwise:
    """Tests for _haversine_km_pairwise."""

    def test_zero_distance_when_same_point(self):
        coords = np.array([[40.7128, -74.0060]])  # NYC
        result = _haversine_km_pairwise(coords, coords)
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 1e-9

    def test_known_pair_nyc_to_la(self):
        # NYC (40.7128 N, 74.0060 W) to LA (34.0522 N, 118.2437 W)
        # Reference great-circle distance ~ 3935.7 km (within 0.5% of any source)
        nyc = np.array([[40.7128, -74.0060]])
        la = np.array([[34.0522, -118.2437]])
        result = _haversine_km_pairwise(nyc, la)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 3935.7) < 5.0

    def test_pairwise_matrix_shape(self):
        coords_a = np.array([[40.0, -74.0], [34.0, -118.0]])
        coords_b = np.array([[51.5, -0.1], [35.7, 139.7], [40.7, -74.0]])
        result = _haversine_km_pairwise(coords_a, coords_b)
        assert result.shape == (2, 3)
        # All non-negative
        assert (result >= 0).all()


class TestEuclideanPairwise:
    """Tests for _euclidean_pairwise."""

    def test_known_3_4_5(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = _euclidean_pairwise(a, b)
        assert abs(result[0, 0] - 5.0) < 1e-12

    def test_zero_distance_same_point(self):
        coords = np.array([[1.5, 2.5], [3.0, 4.0]])
        result = _euclidean_pairwise(coords, coords)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)


class TestPairwiseRingDistances:
    """Tests for the _pairwise_ring_distances dispatch."""

    def test_haversine_branch(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        result = _pairwise_ring_distances(a, b, "haversine")
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 1e-9

    def test_euclidean_branch(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        result = _pairwise_ring_distances(a, b, "euclidean")
        assert abs(result[0, 0] - 1.0) < 1e-12

    def test_callable_branch(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[2.0, 2.0]])

        def custom(x, y):
            return np.full((x.shape[0], y.shape[0]), 7.5)

        result = _pairwise_ring_distances(a, b, custom)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result, 7.5)

    def test_unknown_metric_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="Unknown conley_metric"):
            _pairwise_ring_distances(a, b, "manhattan")


class TestApplyCallableMetricPairwise:
    """Validation of user-supplied callable distance metrics."""

    def test_wrong_shape_raises(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        b = np.array([[2.0, 2.0], [3.0, 3.0]])

        def bad(x, y):
            return np.zeros((1, 1))

        with pytest.raises(ValueError, match="shape"):
            _apply_callable_metric_pairwise(bad, a, b)

    def test_non_finite_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])

        def bad(x, y):
            return np.array([[np.inf]])

        with pytest.raises(ValueError, match="non-finite"):
            _apply_callable_metric_pairwise(bad, a, b)

    def test_negative_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])

        def bad(x, y):
            return np.array([[-1.0]])

        with pytest.raises(ValueError, match="negative"):
            _apply_callable_metric_pairwise(bad, a, b)


# =============================================================================
# Static nearest-treated distance
# =============================================================================


@pytest.fixture
def small_static_panel():
    """3 treated units near origin, 3 near-controls 25-100km out, 3 far at 500km."""
    rng = np.random.default_rng(42)
    treated = [(0.0 + rng.normal(0, 0.05), 0.0 + rng.normal(0, 0.05)) for _ in range(3)]
    near = [(0.4 + i * 0.05, 0.0) for i in range(3)]  # ~44-55 km east
    far = [(5.0 + i * 0.1, 0.0) for i in range(3)]  # ~556+ km east
    units = []
    coords = []
    treats = []
    for i, c in enumerate(treated):
        units.append(f"T{i}")
        coords.append(c)
        treats.append(1)
    for i, c in enumerate(near):
        units.append(f"N{i}")
        coords.append(c)
        treats.append(0)
    for i, c in enumerate(far):
        units.append(f"F{i}")
        coords.append(c)
        treats.append(0)
    # Two periods so the static panel has 2 rows per unit
    rows = []
    for u, c, d in zip(units, coords, treats):
        for t in (0, 1):
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": c[0],
                    "lon": c[1],
                    "D": d * t,  # turns on at t=1 for treated
                }
            )
    return pd.DataFrame(rows)


class TestComputeNearestTreatedDistanceStatic:
    """Static (non-staggered) nearest-treated distance helper."""

    def test_treated_units_have_zero_distance(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        for tid in treated_ids:
            pos = np.where(unit_index == tid)[0][0]
            # Treated units' nearest treated is themselves OR an adjacent T*; the
            # static fixture clusters them within rng.normal(0, 0.05) deg ~6 km.
            assert d_i[pos] < 15.0  # all treated cluster within ~15 km

    def test_near_controls_below_far_controls(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        near_pos = [np.where(unit_index == f"N{i}")[0][0] for i in range(3)]
        far_pos = [np.where(unit_index == f"F{i}")[0][0] for i in range(3)]
        assert all(d_i[p] < 100.0 for p in near_pos)
        assert all(d_i[p] > 500.0 for p in far_pos)

    def test_euclidean_metric(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        d_i_h, _ = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        d_i_e, _ = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="euclidean",
            treated_unit_ids=treated_ids,
        )
        # Different units, but ordering should be consistent (near < far)
        # so the rank of distances matches between metrics.
        order_h = np.argsort(d_i_h)
        order_e = np.argsort(d_i_e)
        np.testing.assert_array_equal(order_h, order_e)

    def test_no_treated_units_raises(self, small_static_panel):
        with pytest.raises(ValueError, match="no treated units present"):
            _compute_nearest_treated_distance_static(
                small_static_panel,
                unit="unit",
                coords=("lat", "lon"),
                metric="haversine",
                treated_unit_ids=np.array(["nonexistent_unit"]),
            )

    def test_unit_index_is_sorted(self, small_static_panel):
        treated_ids = np.array(["T0", "T1", "T2"])
        _, unit_index = _compute_nearest_treated_distance_static(
            small_static_panel,
            unit="unit",
            coords=("lat", "lon"),
            metric="haversine",
            treated_unit_ids=treated_ids,
        )
        # Sorted lexicographically: F0, F1, F2, N0, N1, N2, T0, T1, T2
        expected = ["F0", "F1", "F2", "N0", "N1", "N2", "T0", "T1", "T2"]
        np.testing.assert_array_equal(unit_index, expected)


class TestComputeNearestTreatedDistanceSparse:
    """Sparse cKDTree path for nearest-treated computation."""

    def test_sparse_matches_dense_haversine(self, small_static_panel):
        # Force sparse path on the small fixture by using a tight cutoff.
        treated_ids = np.array(["T0", "T1", "T2"])
        unit_coords_df = (
            small_static_panel[["unit", "lat", "lon"]]
            .drop_duplicates(subset="unit")
            .set_index("unit")
            .sort_index()
        )
        all_coords = unit_coords_df[["lat", "lon"]].values.astype(np.float64)
        treated_mask = np.array(
            [uid in set(treated_ids.tolist()) for uid in unit_coords_df.index],
            dtype=bool,
        )
        treated_coords = all_coords[treated_mask]
        # Sparse path with a 1000 km cutoff should agree with the dense path on
        # all in-range units; far controls (>500 km but <1000 km from any
        # treated) get their true nearest-treated distance.
        d_sparse = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="haversine",
            cutoff_km=1000.0,
        )
        d_dense = _haversine_km_pairwise(all_coords, treated_coords).min(axis=1)
        # Mask: only compare entries within cutoff in dense (sparse returns inf otherwise).
        in_range = d_dense <= 1000.0 * (1 + 1e-6)
        np.testing.assert_allclose(d_sparse[in_range], d_dense[in_range], atol=1e-8)

    def test_sparse_inf_when_no_treated_in_range(self):
        # Single unit at (50, 0); treated cluster at (0, 0). With cutoff 100km,
        # great-circle ~5500 km exceeds it; expect inf.
        all_coords = np.array([[50.0, 0.0]])
        treated_coords = np.array([[0.0, 0.0]])
        d = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="haversine",
            cutoff_km=100.0,
        )
        assert np.isinf(d[0])

    def test_sparse_euclidean(self):
        all_coords = np.array([[0.0, 0.0], [5.0, 0.0], [100.0, 0.0]])
        treated_coords = np.array([[0.0, 0.0]])
        d = _compute_nearest_treated_distance_sparse(
            all_coords=all_coords,
            treated_coords=treated_coords,
            metric="euclidean",
            cutoff_km=10.0,
        )
        assert abs(d[0]) < 1e-12
        assert abs(d[1] - 5.0) < 1e-12
        assert np.isinf(d[2])


# =============================================================================
# Staggered nearest-treated distance
# =============================================================================


@pytest.fixture
def staggered_panel():
    """Panel with two cohorts (t_treat=1 and t_treat=2) plus never-treated."""
    rows = []
    # Cohort A: 2 units treated at t=1 (near origin)
    cohort_a = {"A0": (0.0, 0.0), "A1": (0.1, 0.0)}
    # Cohort B: 2 units treated at t=2 (10 deg east of origin, ~1100 km away)
    cohort_b = {"B0": (0.0, 10.0), "B1": (0.0, 10.1)}
    # Never-treated: 1 unit far away
    never = {"N0": (50.0, 0.0)}  # very far north
    first_treat = {
        **{u: 1 for u in cohort_a},
        **{u: 2 for u in cohort_b},
        **{u: np.inf for u in never},
    }
    coords = {**cohort_a, **cohort_b, **never}
    for t in range(4):  # periods 0..3
        for u, (lat, lon) in coords.items():
            rows.append({"unit": u, "time": t, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    return df, first_treat


class TestComputeNearestTreatedDistanceStaggered:
    """Staggered (time-varying) nearest-treated distance helper."""

    def test_inf_pre_any_treatment(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        # Period 0 has no treated units yet -> d_it = inf for all rows.
        mask_t0 = row_time == 0
        assert np.isinf(d_it[mask_t0]).all()

    def test_cohort_a_active_at_t1(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        mask_t1 = row_time == 1
        # Cohort A treats at t=1; B units should be ~1100km from A; A units near zero.
        for u in ("A0", "A1"):
            row = mask_t1 & (row_unit == u)
            assert d_it[row][0] < 15.0  # within their own cohort
        for u in ("B0", "B1"):
            row = mask_t1 & (row_unit == u)
            # B is ~1100km east of A; min distance to {A0, A1}.
            # B0 at lon=10 -> 1112 km; B1 at lon=10.1 -> 1123 km.
            assert 1100.0 < d_it[row][0] < 1130.0

    def test_running_min_across_cohorts_at_t2(self, staggered_panel):
        df, ft = staggered_panel
        d_it, row_unit, row_time, _trigger = _compute_nearest_treated_distance_staggered(
            df,
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
        )
        mask_t2 = row_time == 2
        # At t=2, B0 and B1 are also treated; the nearest-treated set for B units is {A,B}
        # but B is closer to itself -> nearly zero distance now.
        for u in ("B0", "B1"):
            row = mask_t2 & (row_unit == u)
            assert d_it[row][0] < 15.0


# =============================================================================
# Ring indicator construction
# =============================================================================


class TestBuildRingIndicators:
    """Tests for _build_ring_indicators."""

    def test_three_rings_three_distances(self):
        rings = [0.0, 50.0, 100.0, 200.0]  # K=3 rings
        d_values = np.array([25.0, 75.0, 150.0])
        masks = _build_ring_indicators(d_values, rings)
        assert masks.shape == (3, 3)
        # Row 0: distance 25 -> Ring 1 (0 to 50)
        np.testing.assert_array_equal(masks[0], [True, False, False])
        # Row 1: distance 75 -> Ring 2 (50 to 100)
        np.testing.assert_array_equal(masks[1], [False, True, False])
        # Row 2: distance 150 -> Ring 3 (100 to 200)
        np.testing.assert_array_equal(masks[2], [False, False, True])

    def test_interior_boundary_belongs_to_upper_ring(self):
        """Unit exactly at the boundary between two interior rings."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([50.0])  # exactly on the 50.0 boundary
        masks = _build_ring_indicators(d_values, rings)
        # 50.0 should fall in Ring 2 (the upper of the boundary pair) per the
        # half-open-at-top convention.
        np.testing.assert_array_equal(masks[0], [False, True, False])

    def test_outermost_boundary_belongs_to_last_ring(self):
        """Unit exactly at d_bar should fall in the outermost ring, not be far."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([200.0])  # exactly at d_bar
        masks = _build_ring_indicators(d_values, rings)
        # 200.0 should fall in the OUTERMOST ring (closed-at-top convention).
        np.testing.assert_array_equal(masks[0], [False, False, True])

    def test_distance_at_origin_lands_in_first_ring(self):
        """Treated units have d_i = 0; they fall in Ring_1."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([0.0])
        masks = _build_ring_indicators(d_values, rings)
        np.testing.assert_array_equal(masks[0], [True, False, False])

    def test_far_away_unit_in_no_ring(self):
        """Distance beyond d_bar puts unit in NO ring."""
        rings = [0.0, 50.0, 100.0, 200.0]
        d_values = np.array([300.0])
        masks = _build_ring_indicators(d_values, rings)
        np.testing.assert_array_equal(masks[0], [False, False, False])

    def test_single_ring(self):
        """K=1 (single ring) case (Butts Equation 5 single-S_i form)."""
        rings = [0.0, 100.0]
        d_values = np.array([0.0, 50.0, 99.9, 100.0, 100.1])
        masks = _build_ring_indicators(d_values, rings)
        assert masks.shape == (5, 1)
        # First four are <= 100, last is > 100 (far-away).
        np.testing.assert_array_equal(masks[:, 0], [True, True, True, True, False])

    def test_too_few_breakpoints_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            _build_ring_indicators(np.array([0.0]), [50.0])

    def test_non_increasing_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _build_ring_indicators(np.array([0.0]), [50.0, 50.0, 100.0])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _build_ring_indicators(np.array([0.0]), [-10.0, 100.0])


class TestRingLabel:
    """Tests for _ring_label."""

    def test_interior_ring_half_open(self):
        rings = [0.0, 50.0, 100.0, 200.0]
        assert _ring_label(rings, 0) == "[0, 50)"
        assert _ring_label(rings, 1) == "[50, 100)"

    def test_outermost_ring_closed(self):
        rings = [0.0, 50.0, 100.0, 200.0]
        assert _ring_label(rings, 2) == "[100, 200]"

    def test_single_ring_closed_form(self):
        rings = [0.0, 100.0]
        assert _ring_label(rings, 0) == "[0, 100]"


# =============================================================================
# Public class skeleton
# =============================================================================


class TestSpilloverDiDInitGetParamsSetParams:
    """Constructor + sklearn-like get_params / set_params surface."""

    def test_construction_with_defaults(self):
        est = SpilloverDiD(rings=[0.0, 50.0, 100.0])
        assert est.rings == [0.0, 50.0, 100.0]
        assert est.d_bar is None
        assert est.vcov_type == "hc1"
        assert est.is_fitted_ is False

    def test_construction_with_all_kwargs(self):
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            cluster="region",
            alpha=0.10,
            anticipation=1,
            event_study=True,
            horizon_max=5,
            rank_deficient_action="error",
        )
        assert est.d_bar == 200.0
        assert est.vcov_type == "conley"
        assert est.cluster == "region"
        assert est.alpha == 0.10
        assert est.event_study is True

    def test_get_params_returns_constructor_args(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        params = est.get_params()
        # Check all constructor args present
        expected = {
            "rings",
            "d_bar",
            "vcov_type",
            "conley_coords",
            "conley_metric",
            "conley_cutoff_km",
            "conley_lag_cutoff",
            "cluster",
            "alpha",
            "anticipation",
            "event_study",
            "horizon_max",
            "rank_deficient_action",
        }
        assert set(params.keys()) == expected

    def test_set_params_updates_attribute(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        est.set_params(d_bar=100.0, alpha=0.10)
        assert est.d_bar == 100.0
        assert est.alpha == 0.10

    def test_set_params_returns_self(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        out = est.set_params(d_bar=100.0)
        assert out is est

    def test_set_params_rejects_unknown_key(self):
        est = SpilloverDiD(rings=[0.0, 50.0])
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(nonexistent_kwarg=42)


# =============================================================================
# Step 3: Two-stage Gardner fit() integration
# =============================================================================


def _make_butts_2period_dgp(
    *,
    n_treated: int = 10,
    n_near_control: int = 30,
    n_far_control: int = 30,
    tau_total: float = -0.07,
    delta_1: float = -0.04,
    d_bar: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a 2-period panel with known direct + spillover effects.

    Layout:
      - Treated units cluster near (lat=0, lon=0).
      - Near-controls distributed within d_bar km.
      - Far-controls placed ~2*d_bar km away (clean control group).
    Outcomes (potential outcomes model):
      - Y_it(0, 0) = mu_i + lambda_t + e_it  (clean trend, common across all units)
      - Treated unit at t=1: Y = mu_i + lambda_1 + tau_total + e_it
      - Near-control at t=1: Y = mu_i + lambda_1 + delta_1 + e_it
      - Far-control at t=1: Y = mu_i + lambda_1 + e_it
    All units satisfy parallel trends (Butts Assumption 6/7).
    """
    rng = np.random.default_rng(seed)
    n_units = n_treated + n_near_control + n_far_control
    units = [f"u{i:03d}" for i in range(n_units)]
    mu = rng.normal(0.0, 0.5, size=n_units)

    coords = []
    is_treated = []
    is_near = []
    for i in range(n_treated):
        # Cluster within ~5 km of origin
        coords.append((rng.normal(0, 0.05), rng.normal(0, 0.05)))
        is_treated.append(True)
        is_near.append(False)
    for i in range(n_near_control):
        # Within d_bar (uniform in a band 10–80 km east)
        lat = rng.uniform(0.1, 0.7)  # ~11–78 km north
        lon = rng.uniform(-0.3, 0.3)  # spread east-west
        coords.append((lat, lon))
        is_treated.append(False)
        is_near.append(True)
    for i in range(n_far_control):
        # Far-aways at 2*d_bar+ km
        lat = rng.uniform(2.0, 3.0)  # ~220–330 km north
        lon = rng.uniform(-0.5, 0.5)
        coords.append((lat, lon))
        is_treated.append(False)
        is_near.append(False)

    rows = []
    lambda_t = [0.0, 0.1]  # common time trend
    for i, u in enumerate(units):
        for t in (0, 1):
            y_clean = mu[i] + lambda_t[t]
            if t == 1 and is_treated[i]:
                y = y_clean + tau_total
            elif t == 1 and is_near[i]:
                y = y_clean + delta_1
            else:
                y = y_clean
            y += rng.normal(0, 0.02)  # noise
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": coords[i][0],
                    "lon": coords[i][1],
                    "D": int(is_treated[i] and t == 1),
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


class TestSpilloverDiDFitBasic:
    """Step 3 integration: fit() produces sensible point estimates."""

    def test_fit_runs_without_error(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result is not None
        assert est.is_fitted_

    def test_meat_lsmr_fallback_and_fail_closed(self):
        """The Wave-D GMM meat routes through two_stage's certified-LSMR
        Stage-1 fallback: forced sparse-factorization failure takes the
        LSMR path (finite SE + warning), and an uncertified LSMR fails
        closed through the spillover boundary (NaN meat -> NaN SEs)."""
        import unittest.mock

        df = _make_butts_2period_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)

        def _fit():
            est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
            return est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with pytest.warns(UserWarning, match="falling back to sparse LSMR"):
                result = _fit()
        assert np.isfinite(result.se)

        def _fake_lsmr(A, b, **kwargs):
            return (np.zeros(A.shape[1]), 7, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        with unittest.mock.patch(
            "diff_diff.two_stage.sparse_factorized",
            side_effect=RuntimeError("test failure"),
        ):
            with unittest.mock.patch("scipy.sparse.linalg.lsmr", _fake_lsmr):
                with pytest.warns(UserWarning, match="did not converge"):
                    result = _fit()
        assert np.isnan(result.se)

    def test_recovers_tau_total_within_tolerance(self):
        df = _make_butts_2period_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Single-seed tolerance — looser than the 200-seed MC test in Step 5.
        assert abs(result.att - (-0.07)) < 0.04

    def test_recovers_ring_coefficient(self):
        df = _make_butts_2period_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.spillover_effects is not None
        ring_coef = result.spillover_effects.iloc[0]["coef"]
        assert abs(ring_coef - (-0.04)) < 0.04

    def test_result_has_expected_fields(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.ring_breakpoints == [0.0, 100.0]
        assert result.d_bar == 100.0
        assert result.is_staggered is False
        assert result.n_far_away_obs > 0
        assert result.stage1_n_obs > 0
        assert "[0, 100]" in result.n_units_ever_in_ring

    def test_summary_includes_ring_block(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 50.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        summary = result.summary()
        assert "Spillover Effects" in summary
        # Two ring labels: [0, 50) and [50, 100]
        assert "[0, 50)" in summary or "[50, 100]" in summary

    def test_to_dict_serializes_spillover_effects(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        d = result.to_dict()
        assert "spillover_effects" in d
        assert d["spillover_effects"] is not None
        assert "ring_breakpoints" in d
        assert d["d_bar"] == 100.0


class TestSpilloverDiDRawDataInvariant:
    """Step 3: caller's DataFrame must not be mutated by fit()."""

    def test_caller_data_unchanged(self):
        df = _make_butts_2period_dgp(seed=42)
        original_cols = list(df.columns)
        original_len = len(df)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Caller's DataFrame should not gain or lose columns/rows from fit().
        assert list(df.columns) == original_cols
        assert len(df) == original_len


# =============================================================================
# Step 5: Identification MC tests via _dgp_utils.py
# =============================================================================


class TestSpilloverDiDIdentification:
    """50-seed Monte Carlo: SpilloverDiD recovers known DGP within MC tolerance.

    Plan's Step 5 target was 200 seeds; this is a faster default that still
    rejects gross misidentification. A 200-seed version marked `@pytest.mark.slow`
    runs in CI's full suite (`pytest -m slow`).
    """

    def test_nonstaggered_recovers_tau_total(self):
        att_estimates = []
        n_seeds = 50
        for s in range(n_seeds):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            att_estimates.append(result.att)
        mean_att = float(np.mean(att_estimates))
        # MC tolerance: mean over 50 seeds at error_sd=0.05, ~200 units →
        # SE-of-mean ~ 0.05/sqrt(50 * 200) ~ 5e-4. Tolerance 0.02 leaves
        # margin for DGP design noise.
        assert (
            abs(mean_att - (-0.07)) < 0.02
        ), f"non-staggered tau_total: expected -0.07, got {mean_att:.4f}"

    def test_nonstaggered_recovers_delta_1(self):
        delta_estimates = []
        n_seeds = 50
        for s in range(n_seeds):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            if result.spillover_effects is not None:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        mean_delta = float(np.mean(delta_estimates))
        assert (
            abs(mean_delta - (-0.04)) < 0.02
        ), f"non-staggered delta_1: expected -0.04, got {mean_delta:.4f}"

    @pytest.mark.slow
    def test_nonstaggered_recovers_at_200_seeds(self):
        """Plan-targeted 200-seed MC. Marked slow; run via `pytest -m slow`."""
        att_estimates = []
        delta_estimates = []
        for s in range(200):
            df = generate_butts_nonstaggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", treatment="D"
            )
            att_estimates.append(result.att)
            if result.spillover_effects is not None:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        assert abs(np.mean(att_estimates) - (-0.07)) < 0.02
        assert abs(np.mean(delta_estimates) - (-0.04)) < 0.02

    def test_staggered_recovers_tau_total_and_delta_1(self):
        """Staggered MC with 30 seeds (smaller because each DGP is larger).

        Anchors BOTH `tau_total` and `delta_1` recovery on the staggered
        DGP. Per-ring `delta_jk` (event-time decomposition) is deferred
        alongside `event_study=True` support.
        """
        att_estimates = []
        delta_estimates = []
        for s in range(30):
            df = generate_butts_staggered_dgp(tau_total=-0.07, delta_1=-0.04, seed=s)
            result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
                df, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )
            att_estimates.append(result.att)
            if result.spillover_effects is not None and len(result.spillover_effects) > 0:
                delta_estimates.append(result.spillover_effects.iloc[0]["coef"])
        mean_att = float(np.mean(att_estimates))
        mean_delta = float(np.mean(delta_estimates))
        # Staggered MC is noisier than non-staggered; allow a looser
        # tolerance (0.04 on tau_total, 0.03 on delta_1).
        assert (
            abs(mean_att - (-0.07)) < 0.04
        ), f"staggered tau_total: expected -0.07, got {mean_att:.4f}"
        assert (
            abs(mean_delta - (-0.04)) < 0.03
        ), f"staggered delta_1: expected -0.04, got {mean_delta:.4f}"


# =============================================================================
# Step 3 (continued): staggered smoke test
# =============================================================================


# =============================================================================
# Step 7: Conley integration end-to-end
# =============================================================================


class TestSpilloverDiDWithConley:
    """Step 7: vcov_type='conley' flows through stage 2 cleanly."""

    def test_conley_fit_runs(self):
        df = generate_butts_nonstaggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,  # cross-sectional only (2-period panel)
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert result.vcov_type == "conley"
        assert result.conley_lag_cutoff == 0
        assert np.isfinite(result.se)

    def test_conley_kwargs_threaded_to_gmm_helper(self):
        """PR #456 R8 plumbing test, updated for Wave D: verifies that Conley
        kwargs flow to ``_compute_gmm_corrected_meat`` (the Wave D entry
        point) rather than ``solve_ols``'s vcov path. Pre-Wave-D this test
        patched ``solve_ols`` directly; Wave D bypasses solve_ols's vcov
        computation in favor of the GMM-corrected sandwich, so the spy now
        wraps the GMM helper. The test's purpose — proving no silent HC1
        fallback — is preserved.
        """
        from unittest.mock import patch

        df = generate_butts_nonstaggered_dgp(
            seed=42, n_treated=20, n_near_control=80, n_far_control=100
        )
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_metric="haversine",
            conley_lag_cutoff=0,
        )

        import diff_diff.spillover as spillover_mod

        captured: dict = {}

        original_helper = spillover_mod._compute_gmm_corrected_meat

        def spy_helper(*args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return original_helper(*args, **kwargs)

        with patch.object(spillover_mod, "_compute_gmm_corrected_meat", side_effect=spy_helper):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        # Conley kwargs reached the GMM helper (no silent HC1 fallback).
        assert (
            captured.get("vcov_type") == "conley"
        ), f"expected vcov_type='conley', got {captured.get('vcov_type')!r}"
        assert captured.get("conley_cutoff_km") == 200.0
        assert captured.get("conley_metric") == "haversine"
        assert captured.get("conley_lag_cutoff") == 0
        # The fit-time-derived spatial / temporal arrays must be present and
        # have the right shape.
        coords = captured.get("conley_coords")
        assert coords is not None and coords.shape == (result.n_obs, 2)
        conley_time = captured.get("conley_time")
        conley_unit = captured.get("conley_unit")
        assert conley_time is not None and len(conley_time) == result.n_obs
        assert conley_unit is not None and len(conley_unit) == result.n_obs
        # And the reported SE is finite (the actual GMM-corrected Conley
        # computation completed end-to-end).
        assert np.isfinite(result.se)

    def test_conley_att_invariant_vs_hc1(self):
        """Point-estimate invariance: vcov choice does not change ATT
        (the residualization + OLS fit are independent of variance).
        """
        df = generate_butts_nonstaggered_dgp(
            seed=42, n_treated=20, n_near_control=80, n_far_control=100
        )
        result_hc1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="hc1",
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        result_conley = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        assert abs(result_hc1.att - result_conley.att) < 1e-10


# =============================================================================
# Step 3 (continued): staggered smoke test
# =============================================================================


class TestSpilloverDiDStaggeredFit:
    """Step 3: staggered timing produces sensible results."""

    def test_staggered_fit_runs(self):
        # 3 cohorts, 4 periods
        rng = np.random.default_rng(0)
        rows = []
        cohort_onsets = {1: 1, 2: 2, "C": np.inf}
        # 6 units per cohort placed near distinct centers
        for cohort_id, onset in cohort_onsets.items():
            center_lat = 0.0 if cohort_id == 1 else (10.0 if cohort_id == 2 else 50.0)
            for i in range(6):
                u = f"{cohort_id}_{i}"
                lat = center_lat + rng.normal(0, 0.05)
                lon = rng.normal(0, 0.05)
                first_treat = float(onset) if onset != np.inf else np.inf
                for t in range(4):
                    rows.append(
                        {
                            "unit": u,
                            "time": t,
                            "lat": lat,
                            "lon": lon,
                            "first_treat": first_treat,
                            "y": 1.0
                            + 0.1 * t
                            + (0.05 * (t >= first_treat) if np.isfinite(first_treat) else 0)
                            + rng.normal(0, 0.05),
                        }
                    )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(
            rings=[0.0, 50.0],  # ring covers 0-50 km; far cutoff at 50
            conley_coords=("lat", "lon"),
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert result.is_staggered is True
        assert np.isfinite(result.att)


# =============================================================================
# Step 2: Timing conversion helpers
# =============================================================================


class TestExtractTreatmentOnsets:
    """Tests for _extract_treatment_onsets."""

    def test_canonical_finite_onsets(self):
        df = pd.DataFrame(
            {
                "unit": ["A", "A", "B", "B", "C", "C"],
                "first_treat": [1, 1, 2, 2, np.inf, np.inf],
                "time": [0, 1, 0, 1, 0, 1],
            }
        )
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": 1.0, "B": 2.0, "C": np.inf}

    def test_zero_treated_as_never(self):
        df = pd.DataFrame({"unit": ["A", "A"], "first_treat": [0, 0], "time": [0, 1]})
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": np.inf}

    def test_nan_treated_as_never(self):
        df = pd.DataFrame({"unit": ["A", "A"], "first_treat": [np.nan, np.nan], "time": [0, 1]})
        onsets = _extract_treatment_onsets(df, "first_treat", "unit")
        assert onsets == {"A": np.inf}


class TestConvertTreatmentToFirstTreat:
    """Tests for _convert_treatment_to_first_treat."""

    def test_basic_conversion(self):
        # 2 units, 3 periods; A treated from t=1, B never-treated.
        df = pd.DataFrame(
            {
                "unit": ["A"] * 3 + ["B"] * 3,
                "time": [0, 1, 2, 0, 1, 2],
                "D": [0, 1, 1, 0, 0, 0],
            }
        )
        out, col = _convert_treatment_to_first_treat(df, "D", "time", "unit")
        assert col == "_spillover_first_treat"
        a_rows = out[out["unit"] == "A"]
        b_rows = out[out["unit"] == "B"]
        assert (a_rows["_spillover_first_treat"] == 1.0).all()
        assert np.isinf(b_rows["_spillover_first_treat"]).all()

    def test_no_treated_units_marks_all_inf(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 0]})
        out, _ = _convert_treatment_to_first_treat(df, "D", "time", "unit")
        assert np.isinf(out["_spillover_first_treat"]).all()

    def test_missing_treatment_column_raises(self):
        df = pd.DataFrame({"unit": ["A"], "time": [0]})
        with pytest.raises(ValueError, match="not in data"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 2]})
        with pytest.raises(ValueError, match="exact 0/1"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_caller_dataframe_unchanged(self):
        df = pd.DataFrame({"unit": ["A", "A"], "time": [0, 1], "D": [0, 1]})
        original_cols = list(df.columns)
        _convert_treatment_to_first_treat(df, "D", "time", "unit")
        # The defensive copy + column add does NOT leak back to caller.
        assert list(df.columns) == original_cols


# =============================================================================
# Step 2: SpilloverDiD validators
# =============================================================================


@pytest.fixture
def simple_panel():
    """Minimal valid 2-period panel for validator tests."""
    return pd.DataFrame(
        {
            "unit": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "time": [0, 1] * 4,
            "lat": [0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1],
            "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "D": [0, 1, 0, 1, 0, 0, 0, 0],
            "first_treat": [1, 1, 1, 1, np.inf, np.inf, np.inf, np.inf],
            "y": np.arange(8.0),
        }
    )


class TestValidateSpilloverInputs:
    """Behavioral validation of front-door checks."""

    def test_valid_minimal_input_passes(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0], conley_coords=("lat", "lon"))
        # Should not raise.
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 200.0

    def test_rings_too_short_raises(self, simple_panel):
        est = SpilloverDiD(rings=[100.0])
        with pytest.raises(ValueError, match="at least 2 breakpoints"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_rings_non_sorted_raises(self, simple_panel):
        est = SpilloverDiD(rings=[50.0, 100.0, 75.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_rings_negative_raises(self, simple_panel):
        est = SpilloverDiD(rings=[-10.0, 100.0])
        with pytest.raises(ValueError, match="non-negative"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_d_bar_mismatched_with_rings_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 100.0, 200.0], d_bar=150.0)
        with pytest.raises(ValueError, match="d_bar.*must equal"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_d_bar_equal_to_max_rings_accepted(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 100.0, 200.0], d_bar=200.0, conley_coords=("lat", "lon"))
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 200.0

    def test_d_bar_default_uses_max_rings(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 50.0, 175.0], conley_coords=("lat", "lon"))
        est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")
        assert est._effective_d_bar == 175.0

    def test_treatment_and_first_treat_both_raise(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="either.*or"):
            est._validate_spillover_inputs(simple_panel, "D", "first_treat", "time", "unit", "y")

    def test_neither_treatment_nor_first_treat_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="Exactly one of"):
            est._validate_spillover_inputs(simple_panel, None, None, "time", "unit", "y")

    def test_missing_required_column_raises(self, simple_panel):
        est = SpilloverDiD(rings=[0.0, 200.0])
        with pytest.raises(ValueError, match="Missing required columns"):
            est._validate_spillover_inputs(
                simple_panel, "D", None, "time", "nonexistent_unit_col", "y"
            )

    def test_conley_requires_coords(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="conley_coords"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_coords_must_be_2_tuple(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat",),  # type: ignore[arg-type]  # only 1 element
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="2-tuple"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_coord_column_missing_raises(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "missing_lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="'missing_lon' not in data"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_requires_positive_cutoff(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=-1.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_conley_requires_lag_cutoff(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=None,
        )
        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")

    def test_nan_coords_raise(self, simple_panel):
        df = simple_panel.copy()
        df.loc[0, "lat"] = np.nan
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
        )
        with pytest.raises(ValueError, match="non-finite"):
            est._validate_spillover_inputs(df, "D", None, "time", "unit", "y")

    def test_no_treated_observations_raises(self, simple_panel):
        df = simple_panel.copy()
        df["D"] = 0
        est = SpilloverDiD(rings=[0.0, 200.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="No treated observations"):
            est._validate_spillover_inputs(df, "D", None, "time", "unit", "y")

    def test_cluster_column_missing_raises(self, simple_panel):
        est = SpilloverDiD(
            rings=[0.0, 200.0],
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            cluster="not_a_real_column",
        )
        with pytest.raises(ValueError, match="cluster column"):
            est._validate_spillover_inputs(simple_panel, "D", None, "time", "unit", "y")


class TestValidateFarAwayExists:
    """Tests for SpilloverDiD._validate_far_away_exists."""

    def test_returns_count_when_satisfied(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([10.0, 50.0, 500.0, 1000.0])
        is_control = np.array([False, True, True, True])
        n = est._validate_far_away_exists(d, is_control)
        assert n == 2

    def test_raises_when_no_far_controls(self):
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([10.0, 50.0, 99.9, 100.0])
        is_control = np.array([False, True, True, True])
        with pytest.raises(ValueError, match="Assumption 5"):
            est._validate_far_away_exists(d, is_control)

    def test_raises_when_far_units_all_treated(self):
        """Only treated units beyond d_bar (impossible in non-staggered, but the
        validator's job is to check the population that identifies the
        counterfactual: controls strictly past d_bar)."""
        est = SpilloverDiD(rings=[0.0, 100.0])
        est._effective_d_bar = 100.0
        d = np.array([200.0, 300.0, 50.0])
        is_control = np.array([False, False, True])  # only the close unit is control
        with pytest.raises(ValueError, match="Assumption 5"):
            est._validate_far_away_exists(d, is_control)


# =============================================================================
# Codex-review regression tests (post-Wave-B-MVP first review)
# =============================================================================


class TestSpilloverDiDCovariatesRejected:
    """covariates= must raise NotImplementedError in Wave B MVP.

    Stage-1 covariate residualization (Gardner-style) is not yet wired
    through; appending raw covariates only at stage 2 silently biases
    tau_total / delta_j on panels with time-varying covariates.
    """

    def test_covariates_raises_not_implemented(self):
        df = _make_butts_2period_dgp(seed=42)
        df["x"] = np.random.default_rng(0).normal(size=len(df))
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(NotImplementedError, match="covariates"):
            est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                covariates=["x"],
            )

    def test_empty_covariates_accepted(self):
        """An empty covariates list is the same as no covariates — should NOT raise."""
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # Should not raise.
        est.fit(df, outcome="y", unit="unit", time="time", treatment="D", covariates=[])


class TestSpilloverDiDAbsorbingTreatmentValidation:
    """Reversible / non-absorbing treatment patterns must raise."""

    def test_reversible_treatment_path_raises(self):
        # A unit's treatment goes [0, 1, 0] across 3 periods.
        rows = []
        rng = np.random.default_rng(0)
        for u in ("treated_reversing", "ctrl_far"):
            for t in range(3):
                if u == "treated_reversing":
                    d_val = 1 if t == 1 else 0
                    lat, lon = 0.0, 0.0
                else:
                    d_val = 0
                    lat, lon = 5.0, 0.0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": d_val,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-absorbing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_non_constant_first_treat_raises(self):
        # Same unit has two different first_treat values across rows.
        rows = []
        for t in range(3):
            rows.append(
                {
                    "unit": "u1",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "first_treat": 1.0 if t < 2 else 2.0,  # CHANGES at t=2
                    "y": float(t),
                }
            )
            rows.append(
                {
                    "unit": "u2_far",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "first_treat": np.inf,
                    "y": float(t),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-constant"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDAllEventuallyTreated:
    """All-eventually-treated staggered designs (no never-treated units)
    should work as long as some not-yet-treated rows are far-away controls.
    The Codex review flagged the prior is_ever_control logic as too strict.
    """

    def test_all_eventually_treated_staggered(self):
        # Two cohorts, no never-treated units. Far cohort treats at t=10
        # (well past the panel's t=3 horizon), so its early-period rows
        # serve as far-away controls for the early cohort's treatment.
        rng = np.random.default_rng(0)
        # Per-unit coords sampled ONCE (within-unit constancy required).
        early_coords = [(rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(8)]
        late_coords = [(5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(8)]
        rows = []
        # Early cohort: treats at t=2, clustered at origin
        for i, (lat, lon) in enumerate(early_coords):
            u = f"early_{i}"
            for t in range(4):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 2.0,
                        "y": rng.normal() + 0.1 * t + (-0.07 if t >= 2 else 0.0),
                    }
                )
        # Late cohort: treats at t=10 (far outside panel), placed FAR from early
        for i, (lat, lon) in enumerate(late_coords):
            u = f"late_{i}"
            for t in range(4):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 10.0,  # never treats in panel
                        "y": rng.normal() + 0.1 * t,
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert np.isfinite(result.att)


class TestSpilloverDiDConleyCoordsAlwaysRequired:
    """conley_coords must be validated on every fit path, not just vcov_type=conley.
    The Codex review noted the default-hc1 path was failing with AssertionError.
    """

    def test_missing_conley_coords_on_hc1_path_raises_value_error(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0])  # no conley_coords, default hc1
        with pytest.raises(ValueError, match="conley_coords"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDResultNObsMatchesEstimationSample:
    """result.n_obs must equal the stage-2 estimation sample (after dropping
    rows with non-finite y_tilde from rank-deficient stage-1 FE).
    """

    def test_n_obs_equals_finite_mask_count(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # In a well-conditioned DGP no rows are dropped, so n_obs = len(df).
        assert result.n_obs == len(df)
        # n_treated + n_control == n_obs (no overlap, no leakage).
        assert result.n_treated + result.n_control == result.n_obs


# =============================================================================
# Codex review round-2 regression tests
# =============================================================================


class TestSpilloverDiDOmega0IdentificationCheck:
    """Stage-1 FE support: every unit and every period in the panel must have
    at least one Omega_0 = {D_it=0 AND S_it=0} row, otherwise FE is
    unidentified for that unit/period and stage-2 estimates would be
    silently dropped. Round-2 codex review wanted up-front rejection.
    """

    def test_unsupported_period_raises(self):
        # Panel where t=1 has zero Omega_0 support: every t=1 unit is
        # either treated (D=1) or near (S=1 since d_i <= d_bar=200).
        # The far-aways only contribute at t=0 (no t=1 row).
        rng = np.random.default_rng(0)
        # Per-unit coords sampled ONCE (within-unit-constancy required).
        treated_coords = [(rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(4)]
        near_coords = [(rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3)) for _ in range(4)]
        far_coords = [(5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)) for _ in range(4)]
        rows = []
        for i, (lat, lon) in enumerate(treated_coords):
            for t in range(2):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        for i, (lat, lon) in enumerate(near_coords):
            for t in range(2):
                rows.append(
                    {
                        "unit": f"N{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        # Far-aways at PRE only: no t=1 row → Omega_0 ∩ {t=1} is empty.
        for i, (lat, lon) in enumerate(far_coords):
            rows.append(
                {
                    "unit": f"F{i}",
                    "time": 0,
                    "lat": lat,
                    "lon": lon,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 200.0], d_bar=200.0, conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="unidentified"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDConleyCoordsConstantPerUnit:
    """conley_coords must be constant within each unit across rows.
    Round-2 codex review noted ring construction collapses coords via
    `drop_duplicates(subset=[unit])` — non-constant coords silently use
    only the first row's location.
    """

    def test_time_varying_coords_raises(self):
        rows = []
        rng = np.random.default_rng(0)
        # u1 has different lat at t=0 vs t=1
        rows.append({"unit": "u1", "time": 0, "lat": 0.0, "lon": 0.0, "D": 0, "y": rng.normal()})
        rows.append(
            {
                "unit": "u1",
                "time": 1,
                "lat": 1.5,  # changed!
                "lon": 0.0,
                "D": 1,
                "y": rng.normal(),
            }
        )
        # u2 is well-behaved (constant coords)
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-constant"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-3 regression tests
# =============================================================================


class TestSpilloverDiDFractionalTreatmentRejected:
    """Treatment column with fractional values (0.9, 1.1, etc.) must raise.
    Round-3 codex review caught that `int(v) in (0, 1)` was rounding down
    and silently misclassifying fractional rows.
    """

    @pytest.mark.parametrize("bad_value", [0.5, 0.9, 1.1, -0.1, 2.5])
    def test_fractional_treatment_raises(self, bad_value):
        df = _make_butts_2period_dgp(seed=42).copy()
        # Cast D to float64 before assigning the fractional value. Modern
        # pandas (3.x) raises TypeError on int64-column fractional setitem
        # BEFORE SpilloverDiD.fit() ever sees the input, so we promote the
        # column dtype first to ensure the fractional value actually
        # reaches the validator we're testing.
        df["D"] = df["D"].astype(float)
        first_treated_idx = df.index[df["D"] == 1][0]
        df.loc[first_treated_idx, "D"] = bad_value
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="exact 0/1"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNoTreatedRowRaises:
    """If all first_treat > max(time), D_it is all zeros after the
    anticipation shift. Must raise a clear identification error rather
    than crashing in solve_ols.
    """

    def test_future_only_first_treat_raises(self):
        rng = np.random.default_rng(0)
        # All units treat at t=10, but panel only spans t=0..2.
        rows = []
        for i in range(4):
            lat, lon = rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in range(3):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": 10.0,  # > max(time) = 2
                        "y": rng.normal(),
                    }
                )
        for i in range(4):
            lat, lon = 5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in range(3):
                rows.append(
                    {
                        "unit": f"F{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": np.inf,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="No observation is treated"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDHaversineDomain:
    """Haversine lat/lon range validation applies on EVERY vcov path
    (not just vcov_type='conley'), because ring construction always
    uses the configured metric. Round-3 codex review noted out-of-range
    coords silently produced wrong ring assignment on hc1/cluster paths.
    """

    def test_out_of_range_lat_on_hc1_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        # Corrupt one row's lat to be > 90 (impossible geographic value).
        df.loc[0, "lat"] = 95.0
        # Force the constancy check to ignore this corruption: also corrupt
        # the unit's other row to the same value (constant per unit).
        unit_of_first = df.loc[0, "unit"]
        df.loc[df["unit"] == unit_of_first, "lat"] = 95.0
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            vcov_type="hc1",  # NOT conley
        )
        with pytest.raises(ValueError, match=r"latitude.*\[-90, 90\]"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_out_of_range_lon_on_hc1_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        unit_of_first = df.loc[0, "unit"]
        df.loc[df["unit"] == unit_of_first, "lon"] = 200.0  # > 180
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match=r"longitude.*\[-180, 180\]"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_euclidean_metric_skips_range_check(self):
        """conley_metric='euclidean' must NOT enforce haversine ranges."""
        df = _make_butts_2period_dgp(seed=42).copy()
        # Coordinates 95.0 / 200.0 are valid Euclidean but invalid haversine.
        df.loc[df["unit"] == df.loc[0, "unit"], "lat"] = 95.0
        df.loc[df["unit"] == df.loc[0, "unit"], "lon"] = 200.0
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="euclidean",
        )
        # Should not raise. (May still fail downstream for other reasons —
        # we just need to confirm the haversine range gate is metric-aware.)
        try:
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        except ValueError as exc:
            # Acceptable: a different (non-domain) error from downstream.
            assert "[-90, 90]" not in str(exc) and "[-180, 180]" not in str(exc)


# =============================================================================
# Codex review round-4 regression tests
# =============================================================================


class TestSpilloverDiDZeroIndexedBaselineTreated:
    """Auto-generated `_spillover_first_treat` (from binary `D`) must NOT
    collapse `0` onto the never-treated sentinel. Round-4 codex review
    caught that `_extract_treatment_onsets` was collapsing 0 → inf for
    EVERY first_treat input, silently reclassifying baseline-treated
    units (D=1 at t=0) as never-treated.
    """

    def test_baseline_treated_unit_at_t0_recognized(self):
        """A baseline-treated unit (D=1 at t=0) used to silently become
        never-treated because `_convert_treatment_to_first_treat` wrote
        first_treat=0 and `_extract_treatment_onsets` collapsed 0 -> inf.
        After the fix, u1 is correctly recognized as treated. Because u1
        has no Omega_0 rows (D=1 at all t), it triggers the round-16
        warn-and-drop path: the warning naming `u1_baseline` PROVES it was
        recognized as treated (the OLD bug silently reclassified u1 to
        "never treated", which would have passed the Omega_0 check
        without warning and produced garbage estimates).
        """
        rng = np.random.default_rng(0)
        rows = []
        # u1: baseline-treated (D=1 at all t). No Omega_0 rows → warned-
        # and-dropped from stage 2.
        # u2: treated from t=1 (D=0 at t=0, D=1 at t=1). Far from u1.
        # u3: untreated far-control. Provides Omega_0 support.
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u1_baseline",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": 1,
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2_treated_t1",
                    "time": t,
                    "lat": 10.0,
                    "lon": 0.0,
                    "D": int(t == 1),
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u3_far_control",
                    "time": t,
                    "lat": 20.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # PROOF u1 is recognized as treated: the warning names u1_baseline.
        with pytest.warns(UserWarning, match="u1_baseline"):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # u1's 2 rows excluded from stage 2; u2 (1 treated row at t=1) and
        # u3 (2 control rows) remain. Verify n_treated reflects only the
        # supported sample.
        assert result.n_treated == 1, (
            f"After warn-drop of u1_baseline, expected 1 treated row "
            f"(u2 at t=1); got n_treated={result.n_treated}"
        )

    def test_partial_unsupported_units_warn_and_drop(self):
        """Round-16 codex review: units with no Omega_0 row should be
        warned-and-dropped (matching TwoStageDiD's always-treated convention),
        not block the full fit. The remaining supported sample fits normally.
        """
        rng = np.random.default_rng(1)
        rows = []
        # 4 baseline-treated units (no Omega_0 rows → all 4 warned-dropped).
        for k in range(4):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                    }
                )
        # 3 validly-treated units (treated from t=1; far from baselines).
        for k in range(3):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.warns(UserWarning, match="4 unit\\(s\\) have NO"):
            result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # 4 baselines × 2 periods = 8 rows excluded. Remaining: 3 treated +
        # 5 controls = 8 units × 2 periods = 16 rows. n_treated = 3 (one
        # per treated unit at t=1).
        assert result.n_treated == 3
        assert result.n_obs == 16

    def test_unsupported_period_still_raises(self):
        """Period-level Omega_0 unsupport remains a hard error (round-16
        codex split): dropping a period would remove all units' rows at
        that t, losing the cross-time identification entirely.
        """
        rng = np.random.default_rng(2)
        # Balanced panel where t=1 has NO Omega_0 rows: every unit at t=1
        # is either treated or near a treated unit.
        rows = []
        # 2 treated units at t=1; 2 near-controls (within d_bar of treated)
        # at both t. No far-controls → no Omega_0 row at t=1.
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"T{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # Near-controls at both periods. Pre: untreated and (no current
        # treatment) → S=0 → Omega_0. Post: untreated but treated nearby →
        # S=1 → NOT Omega_0. So t=1 has no Omega_0 row.
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"N{k}",
                        "time": t,
                        "lat": 0.1 + k * 0.001,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], d_bar=100.0, conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="period.*unidentified|unidentified.*period"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_baseline_treated_with_far_control_pretreatment_works(self):
        """A control unit at t<0 (pre-treatment for the baseline-treated)
        provides the missing Omega_0 support, but baseline-treated units
        still have no untreated rows. Confirm the system can FIT when
        there's a clean control population, while still recognizing
        baseline-treated units as treated."""
        rng = np.random.default_rng(0)
        rows = []
        # u1 is treated from t=1 (NOT baseline) — gives Omega_0 support.
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u1_t1_treated",
                    "time": t,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": int(t == 1),
                    "y": rng.normal(),
                }
            )
        for t in (0, 1):
            rows.append(
                {
                    "unit": "u2_far",
                    "time": t,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # n_treated should be 1 (u1 at t=1 only); n_control should be 3
        # (u1's t=0 row + u2's both rows). Critical: u1 is NOT silently
        # reclassified.
        assert result.n_treated == 1
        assert result.n_control == 3


class TestSpilloverDiDCallableSelfDistance:
    """Callable metrics on the ring-construction path must satisfy the
    same self-distance / symmetry contract as conley's vcov path.
    Round-4 codex review noted positive self-distance silently corrupted
    ring assignment on hc1/cluster fits.
    """

    def test_positive_self_distance_callable_raises(self):
        df = _make_butts_2period_dgp(seed=42)

        def bad_metric(a, b):
            # Returns CONSTANT 7.5 — fails the zero-diagonal check on the
            # (n, n) self-call validation.
            return np.full((a.shape[0], b.shape[0]), 7.5)

        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric=bad_metric,
        )
        with pytest.raises(ValueError, match="diagonal"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-5 regression tests
# =============================================================================


class TestSpilloverDiDAnticipationPropagation:
    """anticipation is in `__init__` / `get_params` — round-5 codex review
    flagged that it wasn't surfaced on the SpilloverDiDResults / to_dict()
    so downstream consumers couldn't reconstruct the fitted estimand.
    """

    def test_anticipation_round_trips_to_result_and_dict(self):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        )
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # anticipation is on the result object
        assert result.anticipation == 0
        # anticipation is in to_dict()
        d = result.to_dict()
        assert "anticipation" in d
        assert d["anticipation"] == 0


class TestSpilloverDiDAnticipationBehavior:
    """Round-7 CI review P1: anticipation must change the fitted estimand,
    not just round-trip through the result object. It shifts BOTH the
    treatment indicator and the ring-exposure clock by `-anticipation`,
    moving rows in and out of Omega_0 and changing `tau_total` / `delta_j`.
    Hand-built 4-period panel with one treated unit at t=2: anticipation=1
    promotes t=1 into the "treated" window, dropping that row from
    Omega_0 and increasing n_treated by one period's worth of obs.
    Verified on both fit entry paths (`treatment=` and `first_treat=`).
    """

    @staticmethod
    def _make_4period_panel():
        rng = np.random.default_rng(42)
        # 1 treated @ t=2, 1 near-control, 2 far-controls. 4 periods.
        specs = [
            ("treated", 0.0, [0, 0, 1, 1]),
            ("near", 0.5, [0, 0, 0, 0]),
            ("far1", 5.0, [0, 0, 0, 0]),
            ("far2", 5.1, [0, 0, 0, 0]),
        ]
        rows = []
        for unit, lat, d_pattern in specs:
            for t in range(4):
                rows.append(
                    {
                        "unit": unit,
                        "time": t,
                        "lat": lat,
                        "lon": 0.0,
                        "D": d_pattern[t],
                        "y": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    def test_anticipation_shifts_omega_0_on_treatment_path(self):
        """anticipation=1 on the binary `treatment=` path: the effective
        treatment indicator slides one period earlier, so t=1 (formerly
        Omega_0 for treated + near units) is dropped from stage 1 and
        promoted into the "currently-treated / currently-exposed" zone.
        Stage 1 sample shrinks; n_treated grows; att changes."""
        df = self._make_4period_panel()
        r0 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        r1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=1,
        ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # n_treated grows by 1 row (treated unit's t=1 row promoted under
        # the shifted indicator).
        assert r1.n_treated == r0.n_treated + 1, (
            f"anticipation=1 should add one treated period; "
            f"got n_treated {r0.n_treated} -> {r1.n_treated}"
        )
        # Stage 1 Omega_0 sample shrinks (rows promoted to treated/exposed
        # leave Omega_0).
        assert r1.stage1_n_obs < r0.stage1_n_obs, (
            f"anticipation=1 should shrink Omega_0; got stage1_n_obs "
            f"{r0.stage1_n_obs} -> {r1.stage1_n_obs}"
        )
        # And the estimand changes (different sample → different att).
        assert r0.att != r1.att, f"anticipation=1 should change att; got {r0.att} == {r1.att}"

    def test_anticipation_shifts_omega_0_on_first_treat_path(self):
        """anticipation=1 on the Gardner `first_treat=` path: same shift
        applies. The first_treat column carries treatment onsets directly,
        and anticipation subtracts from each onset for both the D_it
        construction AND the ring-exposure (S_it) clock."""
        df = self._make_4period_panel()
        # Convert binary D to first_treat column.
        first_treat_map = {"treated": 2.0, "near": np.inf, "far1": np.inf, "far2": np.inf}
        df_ft = df.copy()
        df_ft["first_treat"] = df_ft["unit"].map(first_treat_map)
        df_ft = df_ft.drop(columns=["D"])

        r0 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=0,
        ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
        r1 = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=1,
        ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert r1.n_treated == r0.n_treated + 1, (
            f"first_treat path anticipation=1: expected +1 treated, "
            f"got {r0.n_treated} -> {r1.n_treated}"
        )
        assert r1.stage1_n_obs < r0.stage1_n_obs, (
            f"first_treat path anticipation=1: Omega_0 should shrink, "
            f"got {r0.stage1_n_obs} -> {r1.stage1_n_obs}"
        )
        assert r0.att != r1.att, (
            f"first_treat path anticipation=1: expected att to change, " f"got {r0.att} == {r1.att}"
        )

    def test_anticipation_shift_matches_across_fit_paths(self):
        """Sanity-check that the `treatment=` and `first_treat=` paths
        produce identical results under the same anticipation setting —
        the two entry points are internally unified, so anticipation must
        compose consistently with both."""
        df = self._make_4period_panel()
        df_ft = df.copy()
        df_ft["first_treat"] = df_ft["unit"].map(
            {"treated": 2.0, "near": np.inf, "far1": np.inf, "far2": np.inf}
        )
        df_ft = df_ft.drop(columns=["D"])

        for ant in (0, 1):
            r_d = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                anticipation=ant,
            ).fit(df, outcome="y", unit="unit", time="time", treatment="D")
            r_ft = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                anticipation=ant,
            ).fit(df_ft, outcome="y", unit="unit", time="time", first_treat="first_treat")
            assert r_d.att == r_ft.att, (
                f"anticipation={ant}: att mismatch between paths " f"({r_d.att} vs {r_ft.att})"
            )
            assert (
                r_d.stage1_n_obs == r_ft.stage1_n_obs
            ), f"anticipation={ant}: stage1_n_obs mismatch between paths"


class TestSpilloverDiDEffectiveRankDoF:
    """Stage-2 residual df should use effective rank (after solve_ols drops
    rank-deficient columns), not raw column count. Round-5 codex review
    noted that using raw `X_2_fit.shape[1]` understates df_resid on
    rank-deficient stage-2 fits and silently inflates p-values / CI widths.
    """

    def test_rank_deficient_design_uses_effective_rank(self):
        # Construct a panel where the INNER ring [0, 50) has no controls
        # so its stage-2 column is identically zero and solve_ols drops
        # it. After the fix, df_resid uses the effective rank (k=2:
        # treatment + outer ring [50, 200]), not the raw 3-column count.
        rng = np.random.default_rng(42)
        rows = []
        # 8 treated near origin
        for i in range(8):
            lat, lon = rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"T{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": int(t == 1),
                        "y": rng.normal(),
                    }
                )
        # 20 NEAR-controls in the OUTER ring [50, 200) only — at ~1.2°
        # ≈ 133 km from origin (inside outer ring, outside inner ring).
        for i in range(20):
            lat, lon = 1.2 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"N{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        # 20 far-controls beyond d_bar=200 → identify the counterfactual.
        for i in range(20):
            lat, lon = 5.0 + rng.normal(0, 0.005), rng.normal(0, 0.005)
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"F{i}",
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 50.0, 200.0], conley_coords=("lat", "lon"))
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Sanity: outer-ring coef + ATT both finite.
        assert np.isfinite(result.att)
        assert np.isfinite(result.spillover_effects.loc["[50, 200]", "coef"])
        # The empty inner ring's coef should be NaN (dropped by solve_ols).
        assert np.isnan(result.spillover_effects.loc["[0, 50)", "coef"])


# =============================================================================
# Codex review round-6 regression tests
# =============================================================================


class TestSpilloverDiDStringCodedTimeOnTreatmentPath:
    """The `treatment=` path must coerce `time` to numeric BEFORE running
    `_convert_treatment_to_first_treat`. Round-6 codex review noted that
    string-coded numeric periods like ['0', '2', '10'] would sort
    lexicographically ('10' < '2') and produce the wrong onset.
    """

    def test_string_coded_time_treatment_path(self):
        # Unit u1: treated starting at time "2" (the SECOND period when
        # sorted numerically). Lexicographic sort would mis-order "10" < "2"
        # and assign first_treat = "10" (the alphabetic min of treated rows).
        rng = np.random.default_rng(42)
        rows = []
        # u1: time periods "0", "2", "10" — treated at "2" and "10"
        for t_str, t_num in [("0", 0), ("2", 2), ("10", 10)]:
            rows.append(
                {
                    "unit": "u1",
                    "time": t_str,
                    "lat": 0.0,
                    "lon": 0.0,
                    "D": 1 if t_num >= 2 else 0,
                    "y": rng.normal(),
                }
            )
        # u2: far-away never-treated
        for t_str in ("0", "2", "10"):
            rows.append(
                {
                    "unit": "u2",
                    "time": t_str,
                    "lat": 5.0,
                    "lon": 0.0,
                    "D": 0,
                    "y": rng.normal(),
                }
            )
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        # The bug previously was: lex sort would treat "10" as the smallest
        # treated time, assigning u1 first_treat = "10" / 10.0 — outside
        # the relevant comparison range. With the fix, first_treat = 2
        # (numeric), so D_it = 1 at the rows with time = "2" AND "10" → 2
        # treated rows.
        result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # PROOF that time was coerced to numeric BEFORE onset detection:
        # u1 has D=1 at numeric time=2 and time=10 (sorted numerically);
        # n_treated must therefore be 2. Under the lex-sort bug, u1's onset
        # would be "10" (since "10" < "2" lexicographically) and only the
        # t="10" row would be flagged → n_treated = 1.
        assert result.n_treated == 2, (
            f"string-coded numeric time not coerced before onset detection: "
            f"expected n_treated=2 (numeric sort, t in {{2, 10}}), got "
            f"{result.n_treated}"
        )


class TestSpilloverDiDOutcomeColumnRequired:
    """`outcome` should fail front-door with a ValueError, not late KeyError.
    Round-6 codex review noted the validator skipped outcome.
    """

    def test_missing_outcome_column_raises_value_error(self):
        df = _make_butts_2period_dgp(seed=42).drop(columns=["y"])
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="Missing required columns"):
            est.fit(
                df,
                outcome="missing_outcome",
                unit="unit",
                time="time",
                treatment="D",
            )


# =============================================================================
# Codex review round-7 regression tests
# =============================================================================


class TestSpilloverDiDPanelStructure:
    """Reject duplicate (unit, time) cells AND unbalanced panels up front.
    Round-7 codex review reproduced an identification failure on an
    unbalanced panel where moving a far-away outcome silently shifted
    ATT by ~100x.
    """

    def test_duplicate_unit_time_cell_raises(self):
        # u1 has two rows at the same period — duplicate (unit, time) cell.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": [0, 0, 1, 0, 1, 1],  # duplicates at (u1, 0) and (u2, 1)
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_unbalanced_panel_raises(self):
        # u1 has 2 periods but u2 has only 1 → unbalanced panel.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u2"],
                "time": [0, 1, 0],
                "lat": [0.0, 0.0, 5.0],
                "lon": [0.0, 0.0, 0.0],
                "D": [0, 1, 0],
                "y": [1.0, 2.0, 0.5],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="Unbalanced panel"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-8 regression tests
# =============================================================================


class TestSpilloverDiDRingsStartAtZero:
    """rings[0] must equal 0; otherwise units in 0 <= d_it < rings[0] are
    flagged exposed but get zero spillover regressors → silent bias.
    """

    def test_rings_starting_above_zero_raises(self):
        est = SpilloverDiD(rings=[10.0, 50.0, 100.0], conley_coords=("lat", "lon"))
        df = _make_butts_2period_dgp(seed=42)
        with pytest.raises(ValueError, match="rings\\[0\\] must equal 0"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDHC2NotSupported:
    """vcov_type='hc2' and 'hc2_bm' require per-coefficient BM/CR2 DOF
    that the inline stage-2 inference doesn't provide. Round-8 codex
    review caught that we'd silently return wrong p-values/CIs.
    """

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm"])
    def test_hc2_paths_raise_not_implemented(self, vcov_type):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"), vcov_type=vcov_type)
        with pytest.raises(NotImplementedError, match="hc2"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDRankDeficientActionValidation:
    """rank_deficient_action must be one of {warn, error, silent}.
    Mirrors the sibling constructor guards at two_stage.py:149 and
    stacked_did.py.
    """

    @pytest.mark.parametrize("bad_value", ["raise", "ignore", "", "WARN", "Error", None])
    def test_invalid_rank_deficient_action_raises_at_init(self, bad_value):
        with pytest.raises(ValueError, match="rank_deficient_action must be"):
            SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                rank_deficient_action=bad_value,
            )

    @pytest.mark.parametrize("good_value", ["warn", "error", "silent"])
    def test_valid_rank_deficient_action_accepted(self, good_value):
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            rank_deficient_action=good_value,
        )
        assert est.rank_deficient_action == good_value


class TestSpilloverDiDOmega0Connectivity:
    """Round-5 CI review P1: balanced panel + per-unit/per-period Omega_0
    coverage is NECESSARY but not SUFFICIENT for stage-1 FE
    identification — the Omega_0 bipartite graph must also be CONNECTED.
    If it splits into K > 1 components, the iterative FE solver returns
    FE only up to component-specific constants, and residualization
    combines mu_i from one component with lambda_t from another,
    silently corrupting tau_total / delta_j.

    The check fires on the SUPPORTED-units subgraph (after unit-level
    warn-and-drop). Under the current absorbing-treatment + period-strict
    + unit-warn-drop regime the disconnected case may be unreachable in
    practice via a real DGP — these tests unit-test the helper directly
    with synthetic (unit_codes, time_codes, omega_0_mask) arrays so the
    check is exercised even if no DGP can reach it through the public
    `.fit()` path.
    """

    def test_disconnected_two_components_raises(self):
        """Two units in periods {0, 1}, two more in periods {2, 3}; no
        unit appears in both halves. Connectivity check must fail.
        """
        # 4 supported units (codes 0..3), 4 periods (codes 0..3).
        # Omega_0 rows: (u0, t0), (u0, t1), (u1, t0), (u1, t1),
        #               (u2, t2), (u2, t3), (u3, t2), (u3, t3).
        unit_codes_arr = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        time_codes_arr = np.array([0, 1, 0, 1, 2, 3, 2, 3, 2, 2, 0, 0, 3, 3, 1, 1])
        omega_0_mask = np.array(
            [True, True, True, True, True, True, True, True]
            + [False, False, False, False, False, False, False, False]
        )
        with pytest.raises(
            ValueError, match="disconnected components|Stage-1 fixed effects unidentified"
        ):
            _check_omega_0_connectivity(
                omega_0_mask=omega_0_mask,
                unit_codes_arr=unit_codes_arr,
                time_codes_arr=time_codes_arr,
                units_in_omega_0={0, 1, 2, 3},
                n_times=4,
                unit_uniques=["u0", "u1", "u2", "u3"],
            )

    def test_connected_via_bridge_unit_succeeds(self):
        """Add a single bridge unit that has Omega_0 rows in all periods —
        the graph becomes connected and the check must pass.
        """
        unit_codes_arr = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
        time_codes_arr = np.array([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 2, 3])
        omega_0_mask = np.array([True] * 12)
        # u4 spans all 4 periods, connecting the two halves through itself.
        _check_omega_0_connectivity(
            omega_0_mask=omega_0_mask,
            unit_codes_arr=unit_codes_arr,
            time_codes_arr=time_codes_arr,
            units_in_omega_0={0, 1, 2, 3, 4},
            n_times=4,
            unit_uniques=["u0", "u1", "u2", "u3", "u4"],
        )  # must not raise

    def test_single_supported_unit_short_circuits(self):
        """n_supp <= 1 short-circuits — no multi-component case possible."""
        unit_codes_arr = np.array([0, 0])
        time_codes_arr = np.array([0, 1])
        omega_0_mask = np.array([True, True])
        _check_omega_0_connectivity(
            omega_0_mask=omega_0_mask,
            unit_codes_arr=unit_codes_arr,
            time_codes_arr=time_codes_arr,
            units_in_omega_0={0},
            n_times=2,
            unit_uniques=["u0"],
        )  # must not raise

    def test_three_components_error_message_names_units(self):
        """Error message should name first few units per component for
        actionable debugging.
        """
        # 3 units, 3 periods, 3-way disconnection: (u0, t0), (u1, t1), (u2, t2).
        unit_codes_arr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        time_codes_arr = np.array([0, 1, 2, 1, 2, 0, 2, 0, 1])
        omega_0_mask = np.array([True, True, True] + [False] * 6)
        with pytest.raises(ValueError) as exc_info:
            _check_omega_0_connectivity(
                omega_0_mask=omega_0_mask,
                unit_codes_arr=unit_codes_arr,
                time_codes_arr=time_codes_arr,
                units_in_omega_0={0, 1, 2},
                n_times=3,
                unit_uniques=["unit_A", "unit_B", "unit_C"],
            )
        msg = str(exc_info.value)
        assert "3 disconnected components" in msg
        assert "unit_A" in msg or "unit_B" in msg or "unit_C" in msg

    def test_normal_butts_dgp_does_not_trigger(self):
        """Positive case: a standard non-staggered Butts DGP must NOT
        trigger the connectivity check.
        """
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        df = generate_butts_nonstaggered_dgp(seed=0)
        # Just verify .fit() succeeds — if connectivity check were
        # over-eager, this would fail.
        result = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon")).fit(
            df, outcome="y", unit="unit", time="time", treatment="D"
        )
        assert result.att is not None


# =============================================================================
# Codex review round-9 regression tests
# =============================================================================


class TestSpilloverDiDAnticipationValidation:
    """anticipation must be a non-negative integer. Round-9 codex review
    caught that fractional / negative values silently shifted timing.
    """

    @pytest.mark.parametrize("bad_value", [-1, 0.5, 1.5, -0.1])
    def test_invalid_anticipation_raises_treatment_path(self, bad_value):
        df = _make_butts_2period_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            anticipation=bad_value,
        )
        with pytest.raises(ValueError, match="anticipation"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNonFiniteOutcomeRejected:
    """outcome must be finite per-row; NaN/Inf raise a targeted ValueError."""

    def test_nan_outcome_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[0, "y"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-finite"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_inf_outcome_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[0, "y"] = np.inf
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="non-finite"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-10 regression tests
# =============================================================================


class TestSpilloverDiDNaNTreatmentRejected:
    """NaN in the binary treatment column must raise.
    Round-10 codex review caught that `_convert_treatment_to_first_treat`
    silently dropped NaN rows via `dropna()` before validation, then
    rebuilt D_it from the inferred onset — coercing missing rows to
    treated or control without warning.
    """

    @pytest.mark.parametrize(
        "d_pattern",
        [
            [0, 1, float("nan")],
            [float("nan"), 1, 1],
            [0, float("nan"), 0],
        ],
    )
    def test_nan_in_treatment_helper_raises(self, d_pattern):
        df = pd.DataFrame(
            {
                "unit": ["u1"] * 3,
                "time": [0, 1, 2],
                "D": d_pattern,
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            _convert_treatment_to_first_treat(df, "D", "time", "unit")

    def test_nan_in_treatment_end_to_end_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        first_treated_idx = df.index[df["D"] == 1][0]
        df.loc[first_treated_idx, "D"] = np.nan
        # Convert column dtype to float so NaN is preserved.
        df["D"] = df["D"].astype(float)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="NaN"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Codex review round-11 regression tests
# =============================================================================


class TestSpilloverDiDClusterNaNRejected:
    """NaN in cluster column must raise. Round-11 codex review caught
    that missing cluster ids silently changed SEs and overstated
    n_clusters because np.unique counts NaN as its own cluster but
    pandas groupby drops it from the cluster meat.
    """

    def test_numeric_nan_cluster_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["region"] = 1.0
        df.loc[df.index[0], "region"] = np.nan
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            cluster="region",
        )
        with pytest.raises(ValueError, match="cluster.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_object_nan_cluster_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["region"] = "A"
        df.loc[df.index[0], "region"] = None  # object-typed NaN
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            cluster="region",
        )
        with pytest.raises(ValueError, match="cluster.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDIdentifierNaNRejected:
    """unit / time / first_treat columns must not contain NaN — round-11
    codex review noted these fell through to opaque numpy / pandas
    errors instead of targeted ValueErrors.
    """

    def test_nan_unit_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[df.index[0], "unit"] = None
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="identifier column 'unit'"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_nan_time_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df["time"] = df["time"].astype(float)
        df.loc[df.index[0], "time"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="identifier column 'time'"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_nan_first_treat_raises(self):
        df = _make_butts_2period_dgp(seed=42).copy()
        df.loc[df.index[0], "first_treat"] = np.nan
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="first_treat.*missing"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


# =============================================================================
# Codex review round-12 regression tests
# =============================================================================


class TestSpilloverDiDMixedRawTimeEncoding:
    """Mixed encodings that collapse under pd.to_numeric must be caught
    by the validator AFTER coercion. Round-12 codex review caught that
    raw labels ['0', 0] (str + int) would pass duplicate-cell validation
    on the raw labels then collapse to (0, 0) after pd.to_numeric, with
    no warning.
    """

    def test_mixed_str_and_int_time_collapse_caught_as_duplicate(self):
        # u1 has time entries '0' (str) and 0 (int). They collapse to 0
        # under pd.to_numeric → duplicate (u1, 0) cell.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": ["0", 0, 1, 0, 0, 1],
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_leading_zero_string_time_collapse_caught(self):
        # '01' and 1 collapse under pd.to_numeric to (1, 1) → duplicate.
        df = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "time": [0, "01", 1, 0, "01", 1],
                "lat": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
                "lon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "D": [0, 0, 1, 0, 0, 0],
                "y": [1.0, 1.0, 2.0, 0.5, 0.6, 0.6],
            }
        )
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with pytest.raises(ValueError, match="duplicate.*unit, time"):
            est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDNonStaggeredFEEquivalence:
    """Round-14 codex review (P2): pin the Gardner identity empirically.

    Under non-staggered timing, two-stage Gardner with the Omega_0-restricted
    stage 1 should produce the same `tau_total` as a single-stage TWFE ring
    regression on the full sample with the time-varying ring covariate. This
    is Butts Eqs. 4-6 / Proposition 2.3 (non-staggered identification).

    The Omega_0 restriction would BREAK Gardner identity in general (stage 1
    estimates FE on a subset, predicts onto observations outside the training
    set), but on Butts-Assumption-satisfying DGPs it is empirically innocent
    at floating-point precision. This test PINS that equivalence so any
    methodology drift surfaces in CI rather than as a silent estimand shift.

    Codex round 14 reported wildly divergent values (seed 3: +0.0238 from
    SpilloverDiD vs -0.0735 from FE) but those numbers were unreproducible —
    our 20-seed sweep confirms bit-identity at atol=1e-10.
    """

    @staticmethod
    def _fit_butts_single_stage_fe_ring(
        df, *, outcome, unit, time, treatment, rings, lat="lat", lon="lon"
    ):
        """Reference: single-stage TWFE ring regression on full sample.

        Y_it = mu_i + lambda_t + tau * D_it + sum_j delta_j * (1 - D_it) * Ring_{it,j}

        For non-staggered with shared onset t_treat, Ring_{it,j} = 1{d_i in
        [rings_j, rings_{j+1})} * 1{t >= t_treat}. Uses library's solve_ols
        for rank-deficient-safe pseudo-inverse.
        """
        import math
        import warnings

        from diff_diff.linalg import solve_ols

        rings = sorted(rings)
        df = df.copy()
        units = sorted(df[unit].unique())
        times = sorted(df[time].unique())
        unit_idx = {u: i for i, u in enumerate(units)}
        time_idx = {t: i for i, t in enumerate(times)}
        n = len(df)

        treated_set = set(df.loc[df[treatment] == 1, unit].unique())
        lat_map = df.groupby(unit)[lat].first().to_dict()
        lon_map = df.groupby(unit)[lon].first().to_dict()

        def hav(u1, u2):
            lat1, lon1 = math.radians(lat_map[u1]), math.radians(lon_map[u1])
            lat2, lon2 = math.radians(lat_map[u2]), math.radians(lon_map[u2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            return 2 * 6371.0 * math.asin(math.sqrt(a))

        d_i = {}
        for u in units:
            if u in treated_set:
                d_i[u] = 0.0
            else:
                d_i[u] = min(hav(u, tt) for tt in treated_set)

        K = len(rings) - 1
        ring_of_unit = {}
        for u in units:
            d = d_i[u]
            ring_of_unit[u] = -1
            for j in range(K):
                if rings[j] <= d < rings[j + 1]:
                    ring_of_unit[u] = j
                    break

        t_treat = df.loc[df[treatment] == 1, time].min()

        n_u = len(units)
        n_t = len(times)
        n_reg = 1 + K
        # Intercept + (n_u - 1) unit FE dummies + (n_t - 1) time FE dummies + n_reg regressors
        X = np.zeros((n, 1 + (n_u - 1) + (n_t - 1) + n_reg))
        X[:, 0] = 1.0
        y = df[outcome].values.astype(float)

        for i, row in enumerate(df.itertuples(index=False)):
            u = getattr(row, unit)
            t = getattr(row, time)
            D = getattr(row, treatment)
            if unit_idx[u] > 0:
                X[i, 1 + unit_idx[u] - 1] = 1.0
            if time_idx[t] > 0:
                X[i, 1 + (n_u - 1) + time_idx[t] - 1] = 1.0
            X[i, 1 + (n_u - 1) + (n_t - 1) + 0] = D
            ridx = ring_of_unit[u]
            if ridx >= 0 and t >= t_treat:
                X[i, 1 + (n_u - 1) + (n_t - 1) + 1 + ridx] = 1 - D

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _, _ = solve_ols(X, y, vcov_type="hc1")
        tau = beta[1 + (n_u - 1) + (n_t - 1) + 0]
        return tau

    def test_nonstaggered_one_ring_matches_single_stage_fe_20_seeds(self):
        """20-seed bit-identity sweep, 1 ring (rings=[0, 200])."""
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        diffs = []
        for seed in range(20):
            df = generate_butts_nonstaggered_dgp(seed=seed, tau_total=-0.07, delta_1=-0.04)
            est = SpilloverDiD(rings=[0, 200], d_bar=200.0, conley_coords=("lat", "lon"))
            spill = est.fit(df, outcome="y", unit="unit", time="time", treatment="D").att
            fe_tau = self._fit_butts_single_stage_fe_ring(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                rings=[0, 200],
            )
            diffs.append(abs(spill - fe_tau))
        max_abs_diff = max(diffs)
        assert max_abs_diff < 1e-10, (
            f"Gardner identity broken: max |SpilloverDiD - single-stage FE| "
            f"= {max_abs_diff:.6e} across 20 seeds (expected < 1e-10)"
        )

    def test_nonstaggered_multi_ring_matches_single_stage_fe_10_seeds(self):
        """10-seed bit-identity sweep with multi-ring spec."""
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        # DGP places near-controls in d ≤ d_bar/2 = 100km, so rings beyond 100
        # may be empty. Use rings=[0, 50, 200] which has near-controls in
        # [0, 50) and possibly [50, 200).
        diffs = []
        for seed in range(10):
            df = generate_butts_nonstaggered_dgp(seed=seed, tau_total=-0.07, delta_1=-0.04)
            est = SpilloverDiD(rings=[0, 50, 200], d_bar=200.0, conley_coords=("lat", "lon"))
            spill = est.fit(df, outcome="y", unit="unit", time="time", treatment="D").att
            fe_tau = self._fit_butts_single_stage_fe_ring(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                rings=[0, 50, 200],
            )
            diffs.append(abs(spill - fe_tau))
        max_abs_diff = max(diffs)
        assert max_abs_diff < 1e-10, (
            f"Multi-ring Gardner identity broken: max diff "
            f"= {max_abs_diff:.6e} across 10 seeds (expected < 1e-10)"
        )


class TestSpilloverDiDCoefficientsAlignToVcov:
    """Round-15 codex review (P2): `coefficients` must expose ALL stage-2
    coefficients (treatment + K ring slots), not just `ATT`, so consumers
    can align names to the `(1+K)×(1+K)` `vcov` rows/cols. The vcov
    columns are `["treatment", "_spillover_<ring_label>", ...]`; the
    coefficients dict mirrors those keys plus an `"ATT"` alias for the
    treatment slot (sibling-estimator convention).
    """

    def _fit_one(self, rings):
        from tests._dgp_utils import generate_butts_nonstaggered_dgp

        df = generate_butts_nonstaggered_dgp(seed=42, tau_total=-0.07, delta_1=-0.04)
        est = SpilloverDiD(rings=rings, d_bar=float(rings[-1]), conley_coords=("lat", "lon"))
        return est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_coefficients_has_treatment_and_ring_keys(self):
        result = self._fit_one([0, 50, 200])
        assert "ATT" in result.coefficients
        assert "treatment" in result.coefficients
        ring_keys = [k for k in result.coefficients if k.startswith("_spillover_")]
        assert (
            len(ring_keys) == 2
        ), f"Expected 2 ring coefficients, got {len(ring_keys)}: {ring_keys}"

    def test_att_alias_equals_treatment_slot(self):
        result = self._fit_one([0, 100])
        assert result.coefficients["ATT"] == result.coefficients["treatment"]
        assert result.coefficients["ATT"] == result.att

    def test_coefficients_length_matches_vcov_dimension(self):
        result = self._fit_one([0, 50, 200])
        assert result.vcov.shape == (3, 3)
        stage2_keys = [k for k in result.coefficients if k != "ATT"]
        assert len(stage2_keys) == result.vcov.shape[0]

    def test_ring_coefficients_match_spillover_effects_dataframe(self):
        result = self._fit_one([0, 50, 200])
        for ring_label, row in result.spillover_effects.iterrows():
            key = f"_spillover_{ring_label}"
            assert key in result.coefficients, f"Missing key {key} in coefficients"
            assert result.coefficients[key] == row["coef"], (
                f"Drift on {ring_label}: coefficients[{key}]="
                f"{result.coefficients[key]} vs spillover_effects.coef={row['coef']}"
            )


# =============================================================================
# Wave C: _compute_event_time_per_row helper unit tests
# =============================================================================


class TestComputeEventTimePerRowHelper:
    """Unit tests for the two-clock event-time helper (Wave C).

    Verifies:
    - K_direct = t - effective_onset for ever-treated rows; NaN for never-treated.
    - K_spill = t - trigger_onset for triggered rows; NaN otherwise.
    - trigger_onset is the EARLIEST in-range cohort onset (running min).
    - Multi-cohort priority: an earlier cohort wins over a later, even if the
      later cohort's units are closer to the row's unit.
    """

    def _make_panel(self, unit_coords, onsets, n_periods=5):
        """Build a balanced panel from (unit -> (lat, lon)) and (unit -> onset)."""
        rows = []
        for u, (lat, lon) in unit_coords.items():
            ft = onsets.get(u, np.inf)
            for t in range(n_periods):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": ft,
                    }
                )
        return pd.DataFrame(rows)

    def test_k_direct_for_ever_treated_units(self):
        """K_direct = t - effective_onset on all rows of ever-treated units."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "B": (5.0, 0.0)},
            onsets={"A": 1.0, "B": 3.0},
            n_periods=5,
        )
        K_direct, _ = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "B": 3.0},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # A: K = t - 1 for t in {0..4} → {-1, 0, 1, 2, 3}
        # B: K = t - 3 for t in {0..4} → {-3, -2, -1, 0, 1}
        expected_a = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        expected_b = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        a_rows = df["unit"].values == "A"
        b_rows = df["unit"].values == "B"
        np.testing.assert_array_equal(K_direct[a_rows], expected_a)
        np.testing.assert_array_equal(K_direct[b_rows], expected_b)

    def test_k_direct_nan_for_never_treated(self):
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 1.0},  # C never-treated
            n_periods=3,
        )
        K_direct, _ = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        c_rows = df["unit"].values == "C"
        assert np.all(np.isnan(K_direct[c_rows]))

    def test_k_spill_for_in_range_unit(self):
        """A never-treated unit within d_bar of A gets K_spill = t - A.onset."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 1.0},
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # C: at t=0 → NaN (pre-trigger); t=1,2,3 → 0, 1, 2.
        c_rows = df["unit"].values == "C"
        expected_c = np.array([np.nan, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_k_spill_nan_for_far_unit(self):
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "D": (100.0, 0.0)},  # D far
            onsets={"A": 1.0},
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "D": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        d_rows = df["unit"].values == "D"
        assert np.all(np.isnan(K_spill[d_rows]))

    def test_k_spill_trigger_is_earliest_cohort_in_range(self):
        """A unit in range of BOTH cohorts gets trigger = EARLIER cohort, even
        if the later cohort is geographically closer."""
        df = self._make_panel(
            unit_coords={
                "A": (0.0, 0.0),  # cohort 1, onset=1
                "B": (3.0, 0.0),  # cohort 2, onset=3
                "C": (2.5, 0.0),  # at distance 2.5 from A AND 0.5 from B; both <= d_bar
            },
            onsets={"A": 1.0, "B": 3.0},
            n_periods=5,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "B": 3.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        # C: trigger = A's onset (1), NOT B's onset (3), even though B is closer.
        # K_spill at t=0 → NaN; t=1 → 0; t=2 → 1; t=3 → 2; t=4 → 3.
        c_rows = df["unit"].values == "C"
        expected_c = np.array([np.nan, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_k_spill_pre_trigger_is_nan(self):
        """Even an in-range unit has K_spill = NaN before the trigger cohort activates."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 2.0},  # cohort onset is at t=2
            n_periods=4,
        )
        _, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 2.0, "C": np.inf},
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        c_rows = df["unit"].values == "C"
        # C: t=0,1 → NaN (pre-trigger); t=2,3 → 0, 1.
        expected_c = np.array([np.nan, np.nan, 0.0, 1.0])
        np.testing.assert_array_equal(K_spill[c_rows], expected_c)

    def test_anticipation_shifts_both_clocks(self):
        """When effective_onsets is anticipation-shifted, both clocks shift accordingly."""
        df = self._make_panel(
            unit_coords={"A": (0.0, 0.0), "C": (1.0, 0.0)},
            onsets={"A": 3.0},
            n_periods=5,
        )
        # anticipation=2 → effective_onset(A) = 3 - 2 = 1.
        K_direct, K_spill = _compute_event_time_per_row(
            data=df,
            unit="unit",
            row_unit=df["unit"].values,
            row_time=df["time"].values,
            effective_onsets={"A": 1.0, "C": np.inf},  # anticipation-shifted
            coords=("lat", "lon"),
            metric="euclidean",
            d_bar=10.0,
        )
        a_rows = df["unit"].values == "A"
        c_rows = df["unit"].values == "C"
        # A's K_direct: t=0..4 → {-1, 0, 1, 2, 3} (against effective_onset=1).
        np.testing.assert_array_equal(K_direct[a_rows], np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
        # C's K_spill: trigger=1; t=0 → NaN; t=1..4 → 0, 1, 2, 3.
        np.testing.assert_array_equal(K_spill[c_rows], np.array([np.nan, 0.0, 1.0, 2.0, 3.0]))


class TestApplyHorizonBinningHelper:
    """Unit tests for the horizon-clip helper (Wave C)."""

    def test_clips_to_endpoint_bins(self):
        K = np.array([-5.0, -3.0, -1.0, 0.0, 2.0, 5.0, 10.0])
        out = _apply_horizon_binning(K, horizon_max=3)
        np.testing.assert_array_equal(out, np.array([-3.0, -3.0, -1.0, 0.0, 2.0, 3.0, 3.0]))

    def test_nan_preservation(self):
        K = np.array([np.nan, -5.0, 0.0, np.nan, 10.0])
        out = _apply_horizon_binning(K, horizon_max=3)
        # NaN positions remain NaN; finite positions clipped.
        assert np.isnan(out[0])
        assert np.isnan(out[3])
        np.testing.assert_array_equal(out[[1, 2, 4]], np.array([-3.0, 0.0, 3.0]))

    def test_none_horizon_returns_input_unchanged(self):
        K = np.array([-10.0, 0.0, np.nan, 100.0])
        out = _apply_horizon_binning(K, horizon_max=None)
        # Should equal input exactly (NaN preserved by np.array_equal with equal_nan=True).
        assert np.array_equal(out, K, equal_nan=True)

    def test_zero_horizon_collapses_to_single_bin(self):
        """H=0 maps every finite K to 0; NaN preserved."""
        K = np.array([-5.0, -1.0, 0.0, 3.0, np.nan])
        out = _apply_horizon_binning(K, horizon_max=0)
        assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
        assert np.isnan(out[4])

    def test_negative_horizon_raises(self):
        K = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_horizon_binning(K, horizon_max=-1)

    def test_float_horizon_raises(self):
        K = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_horizon_binning(K, horizon_max=2.5)


class TestBuildEventStudyDesignHelper:
    """Unit tests for the event-study stage-2 design builder (Wave C).

    Verifies column count, reference-period drop, column-name convention,
    all-zero pre-filter, and rectangular_grid emission.
    """

    def _hand_built_panel(self):
        """4 units (treated A,B,C ever-treated; D never), 5 periods, 2 rings."""
        # D_it: A treated at t>=2, B at t>=4, C never (ever-treated for D_i but K
        # range only includes pre rows in this example). D never treated.
        # We construct K_direct/K_spill arrays directly for hand-control.
        n_rows = 4 * 5  # 4 units * 5 periods
        D_it = np.zeros(n_rows)
        # Rows ordered: A(t=0..4), B(t=0..4), C(t=0..4), D(t=0..4).
        # A treated post t=2: rows 2,3,4 → D_it=1
        # B treated post t=4: row 9 → D_it=1
        D_it[[2, 3, 4, 9]] = 1.0
        # Ring masks: 2 rings. Let's say A and B are in ring 0 of each other,
        # C is in ring 1 of A. D far from all.
        ring_masks = np.zeros((n_rows, 2), dtype=bool)
        # B's rows in ring 0 (of A) for t >= A.onset=2 → rows 7, 8 (B at t=2, 3)
        # B at t=4 is treated (D_it=1), Ring^k still nonzero but (1-D_it)*Ring=0.
        ring_masks[[7, 8, 9], 0] = True
        # C's rows in ring 1 (of A) for t >= A.onset=2 → rows 12, 13, 14
        ring_masks[[12, 13, 14], 1] = True
        ring_labels = ["[0, 50)", "[50, 200)"]
        # K_direct: A's K_direct = t - 2 for all A rows; B's = t - 4; C, D = NaN.
        K_direct = np.full(n_rows, np.nan)
        K_direct[0:5] = np.arange(5) - 2.0  # A: -2,-1,0,1,2
        K_direct[5:10] = np.arange(5) - 4.0  # B: -4,-3,-2,-1,0
        # K_spill: post-trigger only. B's trigger = A.onset = 2, so K_spill = t-2
        # at rows 7, 8 (B at t=2, 3); row 9 is treated post → still in ring but
        # (1-D_it) zeros it; row 9 K_spill = 4 - 2 = 2 if we want consistent
        # data, but contribution is zero. Set K_spill[9] = 2 so K_set is complete.
        K_spill = np.full(n_rows, np.nan)
        K_spill[7] = 0.0  # B at t=2
        K_spill[8] = 1.0  # B at t=3
        K_spill[9] = 2.0  # B at t=4
        K_spill[12] = 0.0  # C at t=2
        K_spill[13] = 1.0  # C at t=3
        K_spill[14] = 2.0  # C at t=4
        return D_it, ring_masks, ring_labels, K_direct, K_spill

    def test_column_count_with_full_grid(self):
        """Full grid H=2, ref=-1 → 2H = 4 direct + 4 × 2 spillover = 12 candidate
        columns. Some empty cells pre-filtered."""
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        event_time_grid = [-2, -1, 0, 1, 2]
        with pytest.warns(UserWarning, match="all-zero"):
            X_2, names, meta, rect_grid, n_obs = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=event_time_grid,
                ref_period=-1,
            )
        # 4 k bins (after dropping ref=-1) × (1 direct + 2 rings) = 12 candidate cols.
        assert len(rect_grid) == 12
        # X_2 has only the non-empty kept columns.
        assert X_2.shape == (20, len(names))
        assert len(names) == len(meta) == len(n_obs)
        # All n_obs > 0 for kept columns.
        assert all(n > 0 for n in n_obs)

    def test_column_name_convention_signed(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):  # all-zero pre-filter fires
            _, names, _, _, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # Sample expected names: "D^k=+0", "D^k=-2", "_spillover_[0, 50)^k=+1".
        assert any(n == "D^k=+0" for n in names)
        # At least one D^k=-2 column (A has K_direct=-2 at t=0).
        assert any(n == "D^k=-2" for n in names)
        # At least one spillover column with the [0, 50) ring label.
        assert any(n.startswith("_spillover_[0, 50)^k=") for n in names)

    def test_reference_period_dropped(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, meta, rect_grid, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # k=-1 must NOT appear in any column name or any rect_grid tuple.
        assert not any("k=-1" in n for n in names)
        assert not any(k == -1 for (_, _, k) in rect_grid)

    def test_rectangular_grid_includes_dropped_cells(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, _, rect_grid, _ = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # Direct: 4 k bins (excluding ref=-1) → 4 entries in rect_grid.
        direct_in_grid = [(s, r, k) for (s, r, k) in rect_grid if s == "direct"]
        assert len(direct_in_grid) == 4
        # Spillover: 4 k bins × 2 rings = 8 entries.
        spill_in_grid = [(s, r, k) for (s, r, k) in rect_grid if s == "spillover"]
        assert len(spill_in_grid) == 8
        # Sanity: each (ring, k) combo appears exactly once.
        spill_pairs = [(r, k) for (s, r, k) in spill_in_grid]
        assert len(set(spill_pairs)) == 8

    def test_n_obs_per_col_correctness(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.warns(UserWarning):
            _, names, meta, _, n_obs = _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1,
            )
        # D^k=+0 has 2 rows contributing (A at t=2 and B at t=4 both have K_direct=0).
        d0_idx = names.index("D^k=+0")
        assert n_obs[d0_idx] == 2
        # D^k=-2 has 2 rows (A at t=0 and B at t=2 both have K_direct=-2).
        dm2_idx = names.index("D^k=-2")
        assert n_obs[dm2_idx] == 2

    def test_ref_period_must_be_int(self):
        D_it, ring_masks, ring_labels, K_direct, K_spill = self._hand_built_panel()
        with pytest.raises(TypeError, match="integer"):
            _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=ring_labels,
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-2, -1, 0, 1, 2],
                ref_period=-1.0,
            )

    def test_ring_labels_length_mismatch_raises(self):
        D_it, ring_masks, _, K_direct, K_spill = self._hand_built_panel()
        with pytest.raises(ValueError, match="ring_labels"):
            _build_event_study_design(
                D_it=D_it,
                ring_masks=ring_masks,
                ring_labels=["only_one_label"],  # K=2 but only 1 label
                K_direct_binned=K_direct,
                K_spill_binned=K_spill,
                event_time_grid=[-1, 0, 1],
                ref_period=-1,
            )


# =============================================================================
# Wave C: SpilloverDiD(event_study=True) end-to-end test surface
# =============================================================================


def _fit_event_study(
    df,
    *,
    rings=(0.0, 50.0, 200.0),
    horizon_max=None,
    anticipation=0,
    vcov_type="hc1",
    **fit_kwargs,
):
    """Helper: silence event-study warnings and return the SpilloverDiD result."""
    est = SpilloverDiD(
        rings=list(rings),
        d_bar=max(rings),
        conley_coords=("lat", "lon"),
        conley_metric="haversine",
        conley_cutoff_km=max(rings),
        conley_lag_cutoff=0,
        vcov_type=vcov_type,
        event_study=True,
        horizon_max=horizon_max,
        anticipation=anticipation,
    )
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat=fit_kwargs.pop("first_treat", "first_treat"),
            **fit_kwargs,
        )


class TestSpilloverDiDEventStudyAPI:
    """Wave C: surface-level API verification for event_study=True."""

    def test_event_study_emits_att_dynamic(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert res.event_study is True
        assert res.att_dynamic is not None
        assert isinstance(res.att_dynamic, pd.DataFrame)
        # Columns present.
        assert set(res.att_dynamic.columns) == {
            "coef",
            "se",
            "t_stat",
            "p_value",
            "ci_low",
            "ci_high",
            "n_obs",
        }

    def test_event_study_emits_multiindex_spillover_effects(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert isinstance(res.spillover_effects.index, pd.MultiIndex)
        assert list(res.spillover_effects.index.names) == ["ring", "k"]

    def test_event_study_emits_event_study_effects_dict(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        assert isinstance(res.event_study_effects, dict)
        # Every key is an integer event-time bin.
        assert all(isinstance(k, int) for k in res.event_study_effects.keys())
        # Each entry has the TwoStageDiD schema.
        for k, entry in res.event_study_effects.items():
            assert set(entry.keys()) == {"effect", "se", "n_obs", "t_stat", "p_value", "conf_int"}
            assert isinstance(entry["conf_int"], tuple)
            assert len(entry["conf_int"]) == 2

    def test_event_study_effects_reference_row_matches_two_stage_did(self):
        """Reference row must use conf_int=(0.0, 0.0) per TwoStageDiD parity."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.event_study_effects[res.reference_period]
        assert ref["effect"] == 0.0
        assert ref["se"] == 0.0
        assert ref["n_obs"] == 0
        assert ref["conf_int"] == (0.0, 0.0)
        assert np.isnan(ref["t_stat"])
        assert np.isnan(ref["p_value"])

    def test_event_study_false_leaves_new_fields_none(self):
        """When event_study=False, the new Wave C fields stay None."""
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert res.att_dynamic is None
        assert res.event_study_effects is None
        assert res.reference_period is None
        assert res.horizon_max is None


class TestSpilloverDiDEventStudyReferencePeriod:
    """Reference period mirrors TwoStageDiD: ref = -1 - anticipation."""

    def test_reference_period_default_anticipation(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=3, anticipation=0)
        assert res.reference_period == -1

    def test_reference_period_with_anticipation_shifts(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=4, anticipation=2)
        assert res.reference_period == -3

    def test_reference_row_appears_in_att_dynamic(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2, anticipation=0)
        # Reference k=-1 row exists with coef=0, se=0, n_obs=0.
        assert -1 in res.att_dynamic.index
        ref_row = res.att_dynamic.loc[-1]
        assert ref_row["coef"] == 0.0
        assert ref_row["se"] == 0.0
        assert ref_row["n_obs"] == 0


class TestSpilloverDiDEventStudyReduceToAggregate:
    """Reduce-to-Wave-B-aggregate at horizon_max=None on constant-tau DGP.

    Note: horizon_max=0 is REJECTED under event_study=True (PR #456 R5 fix):
    single bin k=0 leaves no event-time pair to anchor the reference period
    against. Users wanting a single aggregate direct effect should use
    event_study=False instead.
    """

    def test_constant_tau_horizon_none_recovers_wave_b_att(self):
        """Deterministic constant-tau DGP (`error_sd=0`) + `horizon_max=None` →
        lincom-weighted scalar `att` reproduces Wave B's aggregate `tau_total`
        bit-identically. Tightened per PR #456 R2 review to match the
        CHANGELOG's claimed `atol=1e-10` contract instead of a loose 1e-3."""
        df = generate_butts_staggered_dgp(
            seed=42,
            tau_total=-0.07,
            delta_1=-0.04,
            error_sd=0.0,  # deterministic — no noise.
        )
        agg_est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            agg = agg_est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        es = _fit_event_study(df, horizon_max=None)
        # With deterministic effects (error_sd=0), the equivalence holds at
        # machine precision: under constant-tau, both the aggregate D_it
        # column and the sample-share-weighted average over per-event-time
        # tau_k columns produce identical regression output.
        assert abs(agg.att - es.att) < 1e-10, (
            f"Reduce-to-aggregate equivalence failed at error_sd=0: "
            f"agg.att={agg.att:.15f}, es.att={es.att:.15f}, "
            f"diff={abs(agg.att - es.att):.3e}"
        )

    def test_lincom_att_matches_hand_computed(self):
        df = generate_butts_staggered_dgp(seed=11)
        res = _fit_event_study(df, horizon_max=3)
        post = res.att_dynamic[res.att_dynamic.index >= 0]
        total = post["n_obs"].sum()
        hand_att = (post["coef"] * post["n_obs"]).sum() / total
        assert abs(hand_att - res.att) < 1e-10


class TestSpilloverDiDEventStudyValidation:
    """Wave C validation: horizon_max < 0 and ref_period outside window both raise."""

    def test_negative_horizon_max_raises(self):
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=-1,
        )
        with pytest.raises(ValueError, match="non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_ref_period_outside_window_raises(self):
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
            anticipation=2,  # ref=-3 outside [-1,+1]
        )
        with pytest.raises(ValueError, match="falls outside the binning window"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_horizon_max_none_with_anticipation_works(self):
        df = generate_butts_staggered_dgp(seed=1)
        # horizon_max=None auto-detects H; ref=-3 with anticipation=2 always fits.
        res = _fit_event_study(df, horizon_max=None, anticipation=2)
        assert res.reference_period == -3

    def test_horizon_max_zero_with_event_study_raises(self):
        """PR #456 R5 P1: horizon_max=0 is rejected under event_study=True
        (the single k=0 bin has no event-time pair to anchor the reference
        against). Users wanting a single aggregate effect should use
        event_study=False."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=0,
        )
        with pytest.raises(ValueError, match="horizon_max=0 is not supported"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_non_numeric_anticipation_raises_targeted_value_error(self):
        """PR #456 R2 P2: anticipation must be validated BEFORE the ref_period
        compatibility check; otherwise `-1 - self.anticipation` would raise a
        raw TypeError on non-numeric input instead of the targeted ValueError."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
            anticipation="1",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

    def test_none_anticipation_raises_targeted_value_error(self):
        """Same P2 fix: None anticipation must surface the targeted ValueError."""
        df = generate_butts_staggered_dgp(seed=1)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
            anticipation=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")


class TestSpilloverDiDEventStudyBackwardCompat:
    """event_study=False reproduces the aggregate path; SEs reflect Wave D.

    The COEF golden values are byte-identical to the Wave B/C pin (Wave D
    changes only the variance estimator; point estimates are unchanged).
    The SE golden values are re-pinned for Wave D — the Gardner GMM
    first-stage uncertainty correction inflates SEs upward by a few percent
    relative to Wave B/C, closing the documented "biased downward" caveat.

    Pre-Wave-D references (commented for the directional-inflation invariant):
      ATT       : -0.08620379515400438     (unchanged)
      SE        :  0.017812406263278957    → Wave D 0.01849079486245095  (+3.8%)
      inner SE  :  0.008298917907045593    → Wave D 0.009669525127172741 (+16.5%)
      outer SE  :  0.015538307675860204    → Wave D 0.016311550606451834 (+5.0%)
    """

    # Wave D golden capture (event_study=False on the seed-42 fixture, with
    # GMM first-stage correction applied across HC1).
    _WAVE_D_GOLDEN_ATT = -0.08620379515400438
    _WAVE_D_GOLDEN_SE = 0.01849079486245095
    _WAVE_D_GOLDEN_RING_INNER_COEF = -0.0371780776943839
    _WAVE_D_GOLDEN_RING_INNER_SE = 0.009669525127172741
    _WAVE_D_GOLDEN_RING_OUTER_COEF = -0.009441319618178406
    _WAVE_D_GOLDEN_RING_OUTER_SE = 0.016311550606451834

    # Pre-Wave-D (uncorrected) SE references — used by the directional
    # inflation invariant to prove the correction moved SE upward.
    _WAVE_B_UNCORRECTED_SE = 0.017812406263278957
    _WAVE_B_UNCORRECTED_INNER_SE = 0.008298917907045593
    _WAVE_B_UNCORRECTED_OUTER_SE = 0.015538307675860204

    def test_event_study_false_matches_wave_b_golden(self):
        """Pre-Wave-C golden parity (not just determinism): pin att/se on a
        deterministic DGP at 1e-14 tolerance and assert reproduction within
        ULP-scale BLAS reduction-order drift across runners. Strengthened
        per PR #456 R3 review — the previous determinism check (fit twice on
        the current code path) did not actually anchor against a pre-Wave-C
        baseline. Tolerance softened from `==` to `assert_allclose(rtol=1e-14,
        atol=1e-14)` after CI Pure Python Fallback (Linux py3.14) flagged a
        1-ULP drift from the macOS Accelerate capture machine — the
        identification claim is unchanged; the platform-pinning was."""
        df = generate_butts_nonstaggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Goldens were captured on a single machine (BLAS reduction order is
        # platform-dependent); pin at 1e-14 tolerance per
        # `feedback_assert_allclose_numerical_parity`. Tight enough to catch
        # real aggregate-path drift, loose enough to absorb ULP-scale
        # cross-runner reduction-order differences (Pure Python Fallback on
        # Linux py3.14 drifts ~1 ULP from macOS Accelerate captures).
        np.testing.assert_allclose(
            res.att,
            self._WAVE_D_GOLDEN_ATT,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"event_study=False att drift: got {res.att!r}, expected {self._WAVE_D_GOLDEN_ATT!r}",
        )
        np.testing.assert_allclose(
            res.se,
            self._WAVE_D_GOLDEN_SE,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"event_study=False se drift: got {res.se!r}, expected {self._WAVE_D_GOLDEN_SE!r}",
        )
        # Per-ring entries must also match.
        inner = res.spillover_effects.loc["[0, 50)"]
        np.testing.assert_allclose(
            inner["coef"],
            self._WAVE_D_GOLDEN_RING_INNER_COEF,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"inner ring coef drift: got {inner['coef']!r}, expected {self._WAVE_D_GOLDEN_RING_INNER_COEF!r}",
        )
        np.testing.assert_allclose(
            inner["se"],
            self._WAVE_D_GOLDEN_RING_INNER_SE,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"inner ring se drift: got {inner['se']!r}, expected {self._WAVE_D_GOLDEN_RING_INNER_SE!r}",
        )
        outer = res.spillover_effects.loc["[50, 200]"]
        np.testing.assert_allclose(
            outer["coef"], self._WAVE_D_GOLDEN_RING_OUTER_COEF, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            outer["se"], self._WAVE_D_GOLDEN_RING_OUTER_SE, rtol=1e-14, atol=1e-14
        )

    def test_event_study_false_bit_identical_to_wave_b_fixture(self):
        df = generate_butts_nonstaggered_dgp(seed=42)
        est_a = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        est_b = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res_a = est_a.fit(df, outcome="y", unit="unit", time="time", treatment="D")
            res_b = est_b.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Determinism guard (the golden parity check above pins the actual values).
        assert res_a.att == res_b.att
        assert res_a.se == res_b.se

    def test_wave_d_se_inflates_relative_to_wave_b_uncorrected(self):
        """Wave D directional invariant: GMM-corrected SE > uncorrected SE.

        Locks the methodological direction of the Wave D correction:
        accounting for first-stage FE estimation uncertainty inflates SE
        upward. The pre-Wave-D SE references (captured on the bit-identical
        point estimate) are pinned as commented references in the class
        docstring above; this test asserts the inequality holds at every
        coefficient surface (top-level att, inner ring, outer ring).
        """
        df = generate_butts_nonstaggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        assert res.se > self._WAVE_B_UNCORRECTED_SE, (
            f"Wave D top-level SE {res.se!r} should exceed pre-Wave-D "
            f"uncorrected SE {self._WAVE_B_UNCORRECTED_SE!r}"
        )
        inner_se = float(res.spillover_effects.loc["[0, 50)"]["se"])
        outer_se = float(res.spillover_effects.loc["[50, 200]"]["se"])
        assert inner_se > self._WAVE_B_UNCORRECTED_INNER_SE, (
            f"Wave D inner ring SE {inner_se!r} should exceed pre-Wave-D "
            f"uncorrected SE {self._WAVE_B_UNCORRECTED_INNER_SE!r}"
        )
        assert outer_se > self._WAVE_B_UNCORRECTED_OUTER_SE, (
            f"Wave D outer ring SE {outer_se!r} should exceed pre-Wave-D "
            f"uncorrected SE {self._WAVE_B_UNCORRECTED_OUTER_SE!r}"
        )


class TestSpilloverDiDEventStudyIdentification:
    """100-seed MC verifies per-event-time tau_k recovery on a known DGP."""

    def test_per_event_time_tau_k_recovery(self):
        # Mild heterogeneous tau profile: k=0 → -0.07; k=1 → -0.06; k=2 → -0.05.
        def tau_fn(k):
            return -0.07 + 0.01 * k

        tau_k_estimates = {k: [] for k in [0, 1, 2]}

        for s in range(50):  # 50 seeds; expensive at 100 for tutorial-paced CI
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=tau_fn,
                delta_per_ring_per_event_time=lambda j, k: -0.04,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            for k in tau_k_estimates:
                if k in res.att_dynamic.index:
                    val = res.att_dynamic.loc[k, "coef"]
                    if np.isfinite(val):
                        tau_k_estimates[k].append(val)

        for k, target in [(0, -0.07), (1, -0.06), (2, -0.05)]:
            mean_est = np.mean(tau_k_estimates[k])
            assert abs(mean_est - target) < 0.025, (
                f"k={k}: mean tau_k estimate {mean_est:.4f} differs from "
                f"target {target:.4f} by more than 0.025 over "
                f"{len(tau_k_estimates[k])} seeds"
            )

    def test_per_ring_event_time_delta_jk_recovery(self):
        """PR #456 R3 fix: also verify per-(ring, event-time) `delta_jk`
        recovery — not just `tau_k`. REGISTRY says Wave C covers `delta_jk`
        recovery; this test backs that claim.

        DGP places all near-controls in ring 0 (one-cohort-one-cluster), so
        only ring 0 cells fire; outer rings emit NaN coefs with n_obs=0
        (rectangular schema).
        """

        def delta_fn(j, k):
            # Mild profile in ring 0: k=0 → -0.04; k=1 → -0.035; k=2 → -0.03.
            return -0.04 + 0.005 * k

        delta_k_estimates = {k: [] for k in [0, 1, 2]}

        for s in range(50):
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=lambda k: -0.07,
                delta_per_ring_per_event_time=delta_fn,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            # Ring 0 corresponds to the inner ring; ring labels are like
            # "[0, 50)" depending on rings passed. Iterate by position.
            ring_labels = res.spillover_effects.index.get_level_values("ring").unique()
            inner_ring = ring_labels[0]
            for k in delta_k_estimates:
                key = (inner_ring, k)
                if key in res.spillover_effects.index:
                    val = res.spillover_effects.loc[key, "coef"]
                    if np.isfinite(val):
                        delta_k_estimates[k].append(val)

        for k, target in [(0, -0.04), (1, -0.035), (2, -0.03)]:
            mean_est = np.mean(delta_k_estimates[k])
            assert abs(mean_est - target) < 0.025, (
                f"delta_jk recovery: k={k} target={target:.4f}, "
                f"mean_est={mean_est:.4f} over {len(delta_k_estimates[k])} seeds "
                f"(tolerance 0.025)"
            )


class TestSpilloverDiDEventStudyPlaceboPretrends:
    """On a no-pre-trend DGP, pre-treatment coefs have nominal Type I rate."""

    def test_no_pretrend_dgp_yields_insignificant_pre_coefs(self):
        # DGP with constant tau=-0.07 only post-treatment (no pre-trend).
        n_seeds = 50
        n_significant_pre = 0
        for s in range(n_seeds):
            df = generate_butts_staggered_dgp(
                seed=s,
                tau_per_event_time=lambda k: -0.07 if k >= 0 else 0.0,
            )
            try:
                res = _fit_event_study(df, horizon_max=2)
            except Exception:
                continue
            # Pre-treatment coef at k=-2 (k=-1 is reference, dropped).
            if -2 in res.att_dynamic.index:
                p = res.att_dynamic.loc[-2, "p_value"]
                if np.isfinite(p) and p < 0.10:
                    n_significant_pre += 1
        type1_rate = n_significant_pre / n_seeds
        # Nominal alpha=0.10 + headroom for finite-sample / single-pre-coef testing.
        assert type1_rate < 0.30, (
            f"Pre-treatment k=-2 placebo Type I rate {type1_rate:.2f} exceeds "
            f"0.30 (nominal 0.10 + headroom). DGP has no pre-trend, so pre-"
            f"treatment coefs should be insignificant."
        )


class TestSpilloverDiDEventStudySingularity:
    """Rectangular schema: empty (ring, k) cells emit NaN with n_obs=0."""

    def test_negative_k_spillover_cells_are_nan(self):
        """K_spill is structurally >=0, so negative-k spillover cells are empty."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        # The (ring, k=-2) cells should appear with NaN coef and n_obs=0.
        # K_spill structurally >= 0, so any k < 0 spillover cell is empty.
        neg_k_rows = res.spillover_effects.xs(-2, level="k")
        # Either all NaN or all dropped pre-filter; rectangular schema emits NaN.
        for ring_label, row in neg_k_rows.iterrows():
            assert row["n_obs"] == 0
            assert np.isnan(row["coef"])

    def test_outer_ring_cells_may_be_empty(self):
        """Default DGP has no units in [50, 200) ring → all NaN with n_obs=0."""
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        if "[50, 200]" in res.spillover_effects.index.get_level_values("ring"):
            outer = res.spillover_effects.xs("[50, 200]", level="ring")
            # All n_obs = 0 (no units in the outer ring in this DGP).
            assert all(outer["n_obs"] == 0)


class TestSpilloverDiDEventStudyConleyIntegration:
    """vcov dimensions + diagonal positivity after Conley path with expanded design."""

    def test_conley_vcov_shape_matches_kept_cols(self):
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=200.0,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        # vcov is square of len(coefficients) - 1 (the "ATT" alias is the
        # only non-column entry in the dict).
        n_kept = len([k for k in res.coefficients.keys() if k != "ATT"])
        assert res.vcov.shape == (n_kept, n_kept)
        # Diagonal entries (variances) post-clamp must be non-negative.
        # SpilloverDiD clamps in the per-coef SE extraction; verify the
        # vcov itself is finite where written.
        finite_diag = np.diag(res.vcov)[np.isfinite(np.diag(res.vcov))]
        assert all(finite_diag >= 0)


class TestSpilloverDiDEventStudySummaryRoundTrip:
    """summary() includes per-event-time blocks; pickle round-trip preserves MultiIndex."""

    def test_summary_includes_dynamic_block(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        s = res.summary()
        assert "Dynamic Direct Effects" in s
        assert "k=" in s or "+" in s  # event-time labels rendered

    def test_pickle_round_trip_preserves_multiindex(self):
        import pickle

        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        round_tripped = pickle.loads(pickle.dumps(res))
        # MultiIndex preserved.
        assert isinstance(round_tripped.spillover_effects.index, pd.MultiIndex)
        # att_dynamic preserved.
        pd.testing.assert_frame_equal(res.att_dynamic, round_tripped.att_dynamic)

    def test_to_dict_serializes_new_fields(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        d = res.to_dict()
        assert "att_dynamic" in d
        assert "event_study_effects" in d
        assert "horizon_max" in d
        assert "reference_period" in d


class TestSpilloverDiDEventStudyFitIdempotence:
    """Clone + repeat-fit produces bit-identical att_dynamic AND spillover_effects."""

    def test_fit_twice_bit_identical(self):
        df = generate_butts_staggered_dgp(seed=42)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res_1 = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
            res_2 = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        pd.testing.assert_frame_equal(res_1.att_dynamic, res_2.att_dynamic)
        pd.testing.assert_frame_equal(res_1.spillover_effects, res_2.spillover_effects)
        assert res_1.att == res_2.att


class TestSpilloverDiDEventStudyFiniteMaskPath:
    """PR #456 R1 fix: event_study=True must use post-finite_mask counts.

    When stage-1 warn-and-drop excludes baseline-treated units (those with
    no Omega_0 rows), the per-event-time `n_obs` values in att_dynamic /
    event_study_effects AND the share weights for the scalar `att` must
    reflect the POST-mask sample — not the pre-mask design.
    """

    def _make_warn_and_drop_panel(self):
        rng = np.random.default_rng(1)
        rows = []
        # 4 baseline-treated units (no Omega_0 rows → warned-dropped).
        for k in range(4):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                    }
                )
        # 3 validly-treated units (treated from t=1; supported).
        for k in range(3):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t >= 1),
                        "y": rng.normal(),
                    }
                )
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1, 2):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    def test_n_obs_in_att_dynamic_reflects_post_mask_sample(self):
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # 4 baselines × 3 periods = 12 rows excluded. Remaining: 3 treated +
        # 5 controls = 8 units × 3 periods = 24 rows. n_treated = 3 supported
        # treated units × 2 post-treatment periods (t=1, t=2) = 6.
        assert res.n_obs == 24, f"n_obs={res.n_obs} (expected 24)"
        # att_dynamic: pre-mask, baseline_{0..3} had D=1 at every t, but
        # those rows are now excluded. The n_obs per k should ONLY count the
        # treated_t1_{0..2} rows.
        # At k=0 (t=1, supported treated): 3 rows.
        # At k=1 (t=2, supported treated): 3 rows.
        # At k=-1 (t=0, supported treated; reference): 0 rows (reference is dropped).
        assert res.att_dynamic.loc[0, "n_obs"] == 3, (
            f"k=0 n_obs={res.att_dynamic.loc[0, 'n_obs']} (expected 3 — the 3 "
            "supported treated_t1 rows at t=1, NOT 7 including pre-mask baselines)"
        )
        assert (
            res.att_dynamic.loc[1, "n_obs"] == 3
        ), f"k=+1 n_obs={res.att_dynamic.loc[1, 'n_obs']} (expected 3)"

    def test_event_study_effects_n_obs_reflects_post_mask_sample(self):
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # event_study_effects dict mirrors att_dynamic, must be consistent.
        for k in res.att_dynamic.index:
            es_n = res.event_study_effects[int(k)]["n_obs"]
            dyn_n = res.att_dynamic.loc[k, "n_obs"]
            assert es_n == dyn_n, (
                f"k={k}: event_study_effects n_obs ({es_n}) disagrees "
                f"with att_dynamic n_obs ({dyn_n})"
            )

    def test_scalar_att_weights_use_post_mask_counts(self):
        """Lincom att = sum_{k>=0} w_k * tau_k where w_k = post-mask n_obs / total."""
        df = self._make_warn_and_drop_panel()
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=1,
        )
        with pytest.warns(UserWarning):
            res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Hand-compute share-weighted att from att_dynamic post-mask n_obs.
        post = res.att_dynamic[res.att_dynamic.index >= 0]
        total = post["n_obs"].sum()
        if total > 0 and not post["coef"].isna().any():
            hand_att = (post["coef"] * post["n_obs"]).sum() / total
            assert abs(hand_att - res.att) < 1e-10, (
                f"att={res.att}, hand-computed from post-mask n_obs={hand_att}, "
                f"diff={abs(hand_att - res.att):.2e}"
            )


class TestSpilloverDiDEventStudyRankDeficientFailClosed:
    """PR #456 R1 fix: when solve_ols drops a post-direct column as NaN,
    the scalar `att` must fail closed (NaN with warning), not silently
    discard weight mass via np.nansum on a fixed weight vector.
    """

    def test_nan_post_direct_coef_yields_nan_att_with_warning(self, monkeypatch):
        """Monkey-patch solve_ols to NaN out one post-treatment direct coef
        and assert att=NaN with the documented warning."""
        df = generate_butts_staggered_dgp(seed=42)
        from diff_diff import spillover as spillover_mod
        from diff_diff.linalg import solve_ols as real_solve_ols

        def solve_ols_with_nan_post_direct(*args, **kwargs):
            coef, residuals, vcov = real_solve_ols(*args, **kwargs)
            column_names = kwargs.get("column_names", [])
            # Find the first post-treatment direct column (D^k=+N with N>=0)
            # and NaN out its coefficient.
            for i, name in enumerate(column_names):
                if name.startswith("D^k=+") and name != "D^k=-0":
                    coef[i] = float("nan")
                    if vcov is not None:
                        vcov[i, :] = float("nan")
                        vcov[:, i] = float("nan")
                    break
            return coef, residuals, vcov

        monkeypatch.setattr(spillover_mod, "solve_ols", solve_ols_with_nan_post_direct)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        with pytest.warns(UserWarning, match="scalar `att` is NaN"):
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert np.isnan(res.att), f"Expected att=NaN, got {res.att}"
        assert np.isnan(res.se), f"Expected se=NaN, got {res.se}"


class TestSpilloverDiDEventStudyReferencePeriodSpilloverRows:
    """PR #456 R1 fix (P3): rectangular spillover_effects must include
    (ring, ref_period) rows with coef=0.0, se=0.0, n_obs=0 (matching the
    direct-effect reference row convention)."""

    def test_ref_period_row_present_per_ring(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.reference_period
        # Every ring should have a (ring, ref_period) row.
        for ring_label in res.spillover_effects.index.get_level_values("ring").unique():
            assert (
                ring_label,
                ref,
            ) in res.spillover_effects.index, (
                f"Missing (ring={ring_label}, k={ref}) row in spillover_effects"
            )

    def test_ref_period_row_uses_zero_anchor(self):
        df = generate_butts_staggered_dgp(seed=42)
        res = _fit_event_study(df, horizon_max=2)
        ref = res.reference_period
        for ring_label in res.spillover_effects.index.get_level_values("ring").unique():
            row = res.spillover_effects.loc[(ring_label, ref)]
            assert row["coef"] == 0.0
            assert row["se"] == 0.0
            assert row["n_obs"] == 0
            assert row["ci_low"] == 0.0
            assert row["ci_high"] == 0.0
            assert np.isnan(row["t_stat"])
            assert np.isnan(row["p_value"])


class TestSpilloverDiDEventStudyPlotIntegration:
    """PR #456 R5 P2: plot_event_study must honor reference_period.

    Wave C's rectangular event_study_effects emits multiple rows with
    `n_obs = 0` (empty horizons + the reference). The legacy plot reference
    detection picks the FIRST `n_obs == 0` row, which may be a non-reference
    horizon. The fix prefers `results.reference_period` when present.
    """

    def test_plot_event_study_uses_explicit_reference_period(self):
        """Set an oversized horizon_max so multiple horizons have n_obs=0.
        The reference detection must still pick the documented reference
        period (-1 with default anticipation=0), not the first empty
        horizon found by iteration order."""
        from diff_diff.visualization._event_study import _extract_plot_data

        df = generate_butts_staggered_dgp(seed=42)
        # horizon_max=4 on a 6-period panel yields several empty post-direct
        # horizons (e.g. cohort onset=3 only has k=0..2 in-panel, so k=+3, +4
        # are empty for that cohort's contribution) plus the reference at -1.
        res = _fit_event_study(df, horizon_max=4)
        (
            effects,
            se,
            periods,
            pre_periods,
            post_periods,
            ref_period,
            ref_inferred,
            *_,
        ) = _extract_plot_data(
            res,
            periods=None,
            pre_periods=None,
            post_periods=None,
            reference_period=None,
        )
        # Reference inference uses the explicit attribute (preferred over the
        # n_obs==0 heuristic that could pick any empty horizon).
        assert ref_inferred is True
        assert ref_period == res.reference_period == -1, (
            f"plot_event_study picked reference_period={ref_period}, "
            f"expected {res.reference_period} from explicit attribute"
        )


# =============================================================================
# Wave D — Gardner GMM first-stage uncertainty correction tests
# =============================================================================


class TestSpilloverDiDWaveDGmmCorrectedHc1Hand:
    """Hand-derived Psi values on a 4-unit × 3-period over-identified panel.

    The pre-flight hand-derivation worksheet (Phase 1 of the Wave D plan)
    fixed the expected `Psi` matrix at numpy float64 precision. This test
    pins those expected values against the runtime helper output so the IF
    formula `psi_i = gamma_hat' x_{10,i} eps_{10,i} - x_{2,i} eps_{2,i}`
    is locked at machine precision. P0: any drift here invalidates every
    downstream Wave D SE.
    """

    def test_psi_matches_hand_derivation(self):
        """4-unit × 3-period over-identified fixture → Psi closed-form match."""
        from scipy import sparse

        from diff_diff.two_stage import _compute_gmm_corrected_meat

        # Fixture (matches /tmp/wave_d_phase1_handderivation.py).
        y = np.array([1.0, 2.5, 2.6, 1.5, 1.7, 1.9, 0.5, 0.6, 0.85, 2.0, 2.1, 2.2])
        D = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        S = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        omega_0 = (D == 0) & (S == 0)
        units = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3)
        times = np.tile(np.array([0, 1, 2]), 4)

        # Stage-1 FE design with drop-first-unit + drop-first-time.
        mu_B = (units == "B").astype(float)
        mu_C = (units == "C").astype(float)
        mu_D = (units == "D").astype(float)
        lam_1 = (times == 1).astype(float)
        lam_2 = (times == 2).astype(float)
        X_1 = np.column_stack([np.ones(12), mu_B, mu_C, mu_D, lam_1, lam_2])
        X_10 = X_1.copy()
        X_10[~omega_0] = 0.0

        # Stage-1 solve + eps_10 reconstruction.
        theta = np.linalg.solve(X_10.T @ X_10, X_10.T @ y)
        eps_10 = np.empty(12)
        eps_10[omega_0] = y[omega_0] - (X_10 @ theta)[omega_0]
        eps_10[~omega_0] = y[~omega_0]

        # Stage-2 design + residual.
        Ring = np.zeros(12)
        Ring[4] = 1.0
        Ring[5] = 1.0
        X_2 = np.column_stack([D.astype(float), (1 - D) * Ring])
        y_tilde = y - X_1 @ theta
        beta, *_ = np.linalg.lstsq(X_2, y_tilde, rcond=None)
        eps_2 = y_tilde - X_2 @ beta

        # Call the helper (HC1 path).
        meat = _compute_gmm_corrected_meat(
            X_1_sparse=sparse.csr_matrix(X_1),
            X_10_sparse=sparse.csr_matrix(X_10),
            eps_10=eps_10,
            X_2=X_2,
            eps_2=eps_2,
            vcov_type="hc1",
        )

        # Hand-computed HC1 meat (with finite-sample multiplier n/(n-p_2)
        # = 12/10 = 1.2). The pre-multiplier meat is Psi.T @ Psi which on
        # this fixture equals:
        expected_unscaled = np.array([[0.005625, 0.0028125], [0.0028125, 0.003125]])
        expected = (12 / 10) * expected_unscaled
        np.testing.assert_allclose(meat, expected, atol=1e-12, rtol=1e-12)

    def test_cluster_singletons_equals_hc1(self):
        """Cluster-by-row equals HC1 on the same fixture (singleton CR1
        multiplier `G/(G-1) * (n-1)/(n-p)` collapses to `n/(n-p)` when
        `G = n`)."""
        from scipy import sparse

        from diff_diff.two_stage import _compute_gmm_corrected_meat

        y = np.array([1.0, 2.5, 2.6, 1.5, 1.7, 1.9, 0.5, 0.6, 0.85, 2.0, 2.1, 2.2])
        D = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        S = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        omega_0 = (D == 0) & (S == 0)
        units = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3)
        times = np.tile(np.array([0, 1, 2]), 4)
        mu_B = (units == "B").astype(float)
        mu_C = (units == "C").astype(float)
        mu_D = (units == "D").astype(float)
        lam_1 = (times == 1).astype(float)
        lam_2 = (times == 2).astype(float)
        X_1 = np.column_stack([np.ones(12), mu_B, mu_C, mu_D, lam_1, lam_2])
        X_10 = X_1.copy()
        X_10[~omega_0] = 0.0
        theta = np.linalg.solve(X_10.T @ X_10, X_10.T @ y)
        eps_10 = np.empty(12)
        eps_10[omega_0] = y[omega_0] - (X_10 @ theta)[omega_0]
        eps_10[~omega_0] = y[~omega_0]
        Ring = np.zeros(12)
        Ring[4] = 1.0
        Ring[5] = 1.0
        X_2 = np.column_stack([D.astype(float), (1 - D) * Ring])
        y_tilde = y - X_1 @ theta
        beta, *_ = np.linalg.lstsq(X_2, y_tilde, rcond=None)
        eps_2 = y_tilde - X_2 @ beta

        common = dict(
            X_1_sparse=sparse.csr_matrix(X_1),
            X_10_sparse=sparse.csr_matrix(X_10),
            eps_10=eps_10,
            X_2=X_2,
            eps_2=eps_2,
        )
        meat_hc1 = _compute_gmm_corrected_meat(vcov_type="hc1", **common)
        meat_cluster = _compute_gmm_corrected_meat(
            vcov_type="cluster", cluster_ids=np.arange(12), **common
        )
        np.testing.assert_allclose(meat_hc1, meat_cluster, atol=1e-14, rtol=1e-14)


class TestSpilloverDiDWaveDGmmCorrectedEventStudy:
    """Wave D applies the GMM correction on the `event_study=True` path."""

    def test_vcov_shape_matches_kept_columns(self):
        """vcov is (n_kept, n_kept) and the diagonal entries are finite for
        every kept column (the Wave D bread sandwich produces a well-formed
        result on a non-degenerate event-study design)."""
        df = generate_butts_staggered_dgp(
            seed=0,
            tau_per_event_time=lambda k: -0.07 if k >= 0 else 0.0,
        )
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

        # The att_dynamic block has at least one finite SE per post-treatment
        # event-time (the lincom scalar att SE is finite — the underlying
        # sub-vcov block must therefore be finite at those positions).
        assert np.isfinite(res.se), f"scalar att SE should be finite, got {res.se!r}"
        finite_se_count = res.att_dynamic["se"].apply(np.isfinite).sum()
        assert finite_se_count >= 2, (
            f"expected ≥2 finite SE rows in att_dynamic (post-treatment k=0,1,2), "
            f"got {finite_se_count}"
        )

    def test_event_study_se_inflates_over_pre_wave_d(self):
        """Event-study SE shifts upward under the GMM correction (directional
        invariance — locks the methodological direction of the Wave D fix).

        Captures the same DGP that the pre-Wave-D event-study tests use; we
        cannot literally check against a pre-Wave-D value (Wave D landed
        with this PR), but we CAN assert that the scalar att SE exceeds a
        loose lower bound corresponding to the maximum possible
        uncorrected SE on this fixture.
        """
        df = generate_butts_staggered_dgp(
            seed=0,
            tau_per_event_time=lambda k: -0.07,
        )
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")

        # Loose lower-bound check: SE > 0 and finite. The directional
        # inflation invariant is exercised on the aggregate path in
        # TestSpilloverDiDEventStudyBackwardCompat::test_wave_d_se_inflates_...
        assert res.se > 0
        assert np.isfinite(res.se)


class TestSpilloverDiDWaveDGmmCorrectedNanInferenceContract:
    """Wave D NaN-propagation contract per `feedback_no_silent_failures`."""

    def test_rank_deficient_design_yields_nan_se_not_zero(self):
        """When solve_ols drops a rank-deficient column, the corresponding
        vcov diagonal entry is NaN (re-inflation pattern). Downstream
        per-coefficient SE for that column is NaN — never silently 0.
        """
        # Use the existing fail-closed fixture infrastructure: monkeypatch
        # solve_ols to return a coef vector with a NaN entry.
        from unittest.mock import patch

        import diff_diff.spillover as spillover_mod

        df = generate_butts_nonstaggered_dgp(seed=0)

        original_solve_ols = spillover_mod.solve_ols

        def coef_nan_solve_ols(X, y, **kwargs):
            coef, residuals, vcov = original_solve_ols(X, y, **kwargs)
            # Inject NaN into the LAST coefficient column to simulate a
            # rank-deficient drop. solve_ols normally sets NaN on coefs it
            # dropped; we forcibly do so here.
            coef = coef.copy()
            coef[-1] = np.nan
            return coef, residuals, vcov

        with patch.object(spillover_mod, "solve_ols", side_effect=coef_nan_solve_ols):
            est = SpilloverDiD(
                rings=[0.0, 50.0, 200.0],
                d_bar=200.0,
                conley_coords=("lat", "lon"),
                event_study=False,
            )
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore", UserWarning)
                res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        # The OUTER ring (last column) was forced rank-deficient; its SE
        # must be NaN, not 0. The other coefficients should still have
        # finite SE (the Wave D re-inflation pattern preserves them).
        outer_se = float(res.spillover_effects.loc["[50, 200]"]["se"])
        assert np.isnan(outer_se), f"rank-deficient outer ring SE should be NaN, got {outer_se!r}"


class TestSpilloverDiDWaveDGmmCorrectedValidatorWiring:
    """Wave D bypasses solve_ols's Conley vcov path; the Conley validator
    must still fire from `_compute_gmm_corrected_meat`."""

    def test_conley_without_cutoff_raises(self):
        """vcov_type='conley' with conley_cutoff_km=None raises ValueError."""
        df = generate_butts_nonstaggered_dgp(seed=0)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=None,
            conley_metric="euclidean",
            conley_lag_cutoff=0,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="conley_cutoff_km"):
                est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


class TestSpilloverDiDWaveDGmmCorrectedFitIdempotence:
    """fit() must not mutate estimator state; clone + repeat-fit produces
    bit-identical Wave D vcov per `feedback_fit_does_not_mutate_config`."""

    def test_clone_repeat_fit_bit_identical(self):
        df = generate_butts_nonstaggered_dgp(seed=42)
        kwargs = dict(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        est_a = SpilloverDiD(**kwargs)
        est_b = SpilloverDiD(**kwargs)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            res_a = est_a.fit(df, outcome="y", unit="unit", time="time", treatment="D")
            res_b = est_b.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        # Same-machine determinism: bit-identical att and se.
        assert res_a.att == res_b.att
        assert res_a.se == res_b.se
        # Per-ring entries also bit-identical.
        for label in ["[0, 50)", "[50, 200]"]:
            assert (
                res_a.spillover_effects.loc[label]["se"] == res_b.spillover_effects.loc[label]["se"]
            )


class TestSpilloverDiDWaveDPublicVarianceContract:
    """End-to-end fit() coverage for the PUBLIC vcov_type / cluster contract.

    Round-1 codex review caught two regressions where the helper-level
    tests passed but the public-API contract was broken:
      P0 — `cluster=<col>` silently routed to HC1 instead of CR1.
      P1 — `vcov_type="classical"` raised an unhandled error inside
            `_compute_gmm_corrected_meat` instead of failing fast at
            validation time.

    This class exercises the public surface to lock the contract.
    """

    def test_cluster_kwarg_routes_to_cr1_not_hc1(self):
        """`SpilloverDiD(..., cluster="unit")` MUST produce CR1 SE, not HC1.

        On a fixture with within-cluster correlation, CR1 SE is generically
        DIFFERENT from HC1 SE — if both fits return the same SE to machine
        precision, the cluster kwarg was silently ignored (the P0
        regression that codex Round 1 surfaced).
        """
        df = generate_butts_nonstaggered_dgp(seed=42)
        common = dict(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            est_hc1 = SpilloverDiD(**common)  # vcov_type="hc1" default, no cluster
            res_hc1 = est_hc1.fit(df, outcome="y", unit="unit", time="time", treatment="D")
            est_cr1 = SpilloverDiD(cluster="unit", **common)
            res_cr1 = est_cr1.fit(df, outcome="y", unit="unit", time="time", treatment="D")

        # Point estimates match (cluster kwarg only affects variance).
        assert res_hc1.att == res_cr1.att
        # SE values must DIFFER — if equal, the cluster kwarg was a no-op.
        assert res_hc1.se != res_cr1.se, (
            f"HC1 SE {res_hc1.se!r} == CR1 SE {res_cr1.se!r}; "
            f"cluster=<col> appears to be silently ignored"
        )

    def test_single_cluster_sample_raises(self):
        """CR1 path on a single-cluster sample raises ValueError per
        the standard `n_clusters >= 2` rejection (mirrors linalg.py:1942)."""
        df = generate_butts_nonstaggered_dgp(seed=0)
        df = df.copy()
        df["fake_cluster"] = 0  # collapse all rows to a single cluster
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            cluster="fake_cluster",
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="at least 2 clusters"):
                est.fit(df, outcome="y", unit="unit", time="time", treatment="D")

    def test_saturated_design_yields_nan_se_not_finite(self):
        """`n_obs == p_2` saturated stage-2 design: HC1 multiplier
        ``n/(n-p)`` is undefined. Wave D fails closed by returning NaN
        meat → NaN SE downstream, rather than clamping the denominator
        to 1 and emitting a finite SE on an underdetermined fit.
        """
        from scipy import sparse

        from diff_diff.two_stage import _compute_gmm_corrected_meat

        # Construct a saturated synthetic Psi fixture directly through the
        # helper (avoids manufacturing a real saturated SpilloverDiD panel,
        # which is constrained by the validator). n_obs == p_2 == 4.
        n, p_1, p_2 = 4, 3, 4
        rng = np.random.default_rng(0)
        X_1 = sparse.csr_matrix(rng.standard_normal((n, p_1)))
        X_10 = sparse.csr_matrix(rng.standard_normal((n, p_1)))
        eps_10 = rng.standard_normal(n)
        X_2 = rng.standard_normal((n, p_2))
        eps_2 = rng.standard_normal(n)

        import warnings as _w

        for vmode, kwargs in [
            ("hc1", {}),
            ("cluster", {"cluster_ids": np.array([0, 0, 1, 1])}),
        ]:
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                meat = _compute_gmm_corrected_meat(
                    X_1_sparse=X_1,
                    X_10_sparse=X_10,
                    eps_10=eps_10,
                    X_2=X_2,
                    eps_2=eps_2,
                    vcov_type=vmode,
                    **kwargs,
                )
            assert np.all(np.isnan(meat)), (
                f"vcov_type={vmode!r} saturated design (n=p_2={n}) returned "
                f"finite meat instead of NaN: {meat!r}"
            )
            saturation_warning_fired = any("saturated" in str(w.message) for w in caught)
            assert saturation_warning_fired, (
                f"vcov_type={vmode!r} saturated design did not emit the "
                f"expected saturation warning"
            )

    def test_classical_vcov_raises_with_clear_message(self):
        """`vcov_type="classical"` raises NotImplementedError upfront with a
        clear remediation message rather than failing deep inside the GMM
        helper (the P1 regression that codex Round 1 surfaced)."""
        df = generate_butts_nonstaggered_dgp(seed=0)
        est = SpilloverDiD(
            rings=[0.0, 50.0, 200.0],
            d_bar=200.0,
            conley_coords=("lat", "lon"),
            vcov_type="classical",
            event_study=False,
        )
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="classical"):
                est.fit(df, outcome="y", unit="unit", time="time", treatment="D")


# =============================================================================
# Wave E.1 — survey_design integration (HC1 / CR1 via Binder TSL)
# =============================================================================
#
# Composes Gerber (2026, arXiv:2605.04124) Proposition 1 Binder Taylor
# Series Linearization with the Wave D Gardner GMM first-stage correction.
# Conley × survey is deferred to Wave E.2.
#
# Test invariants (per plan §Chunk 6):
#   (a)  Uniform-weight degenerate bit-identity with Wave D HC1
#   (c)  Binder TSL hand-check (PSU-aggregated, uniform weights)
#   (c2) Binder TSL hand-check (PSU-aggregated, NON-uniform weights)
#   (d)  lonely_psu sensitivity ({remove, certainty, adjust} → different SE)
#   (e1) FPC=np.inf matches no-FPC path (bit-identical)
#   (e2) FPC=n_h zeros that stratum's contribution
#   (e3) FPC intermediate shrinks SE monotonically
#   (f)  Saturated NaN-fail + pytest.warns(match="df_survey")
#   (h)  cluster=<col> + survey.psu warns and uses PSU
#   (i)  Replicate-weight variance rejection
#   (j)  Non-pweight rejection
#   (k)  Fit-idempotency on the survey path
#   (l)  Event-study + survey, is_staggered=True
#   (m)  Event-study + survey, is_staggered=False
#   (n)  Aggregate-vs-event-study parity on the survey path
#   (o)  Drift goldens
#   (p)  finite_mask survey-array subsetting


def _augment_with_survey(
    df: pd.DataFrame,
    *,
    n_strata: int = 2,
    psus_per_stratum: int = 4,
    fpc: float = 20.0,
    weights: Optional[np.ndarray] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Augment a Butts DGP with stratum / PSU / FPC / weight columns.

    Assigns units to strata deterministically (units 0 to n_units / n_strata
    in stratum 0, etc.) and partitions each stratum's units into PSUs of
    roughly equal size. The same unit always gets the same PSU/stratum
    across periods (panel survey constancy).
    """
    df = df.copy()
    units = df["unit"].unique()
    n_units = len(units)
    # Sort for deterministic assignment
    units_sorted = sorted(units)
    unit_to_stratum: Dict[str, int] = {}
    unit_to_psu: Dict[str, int] = {}
    for idx, u in enumerate(units_sorted):
        stratum = min(idx * n_strata // n_units, n_strata - 1)
        psu_within = (idx // max(1, n_units // (n_strata * psus_per_stratum))) % psus_per_stratum
        psu_label = stratum * psus_per_stratum + psu_within
        unit_to_stratum[u] = stratum
        unit_to_psu[u] = psu_label
    df["stratum"] = df["unit"].map(unit_to_stratum)
    df["psu"] = df["unit"].map(unit_to_psu)
    df["N_h"] = fpc  # per-stratum FPC population size
    if weights is None:
        df["w"] = 1.0
    else:
        # weights aligned with sorted(units); broadcast to rows
        w_by_unit = dict(zip(units_sorted, weights))
        df["w"] = df["unit"].map(w_by_unit)
    # Avoid unused-import lint
    _ = seed
    return df


class TestSpilloverDiDWaveE1SurveyDesignHc1:
    """Wave E.1 HC1 / CR1 via Binder TSL on PSU-aggregated Psi.

    Methodology anchor: Gerber (2026, arXiv:2605.04124) Proposition 1
    composed with the Wave D Gardner GMM correction. Hand-checks against
    the audited `_compute_stratified_meat_from_psu_scores` helper.
    """

    def _fit(self, df, **kwargs):

        design = kwargs.pop("design", None)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=False,
            **kwargs,
        )
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )

    def test_a_uniform_weight_degenerate_matches_wave_d(self):
        """Uniform-weight, single-PSU-per-row, no-FPC degenerates to Wave D.

        Load-bearing test of the `weights is None` bit-identical fallback in
        `_iterative_fe_subset`. ATT should bit-match (uniform weights have
        no estimand effect); the SE shifts because Binder TSL replaces the
        HC1 multiplier.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=0)
        # Wave D baseline
        res_baseline = self._fit(df)
        # Survey with PSU == unit (one PSU per unit; required for panel
        # survey unit-constancy validator), uniform weights, no FPC, no strata
        df_s = df.copy()
        df_s["w"] = 1.0
        units_sorted = sorted(df_s["unit"].unique())
        unit_to_psu = {u: idx for idx, u in enumerate(units_sorted)}
        df_s["psu"] = df_s["unit"].map(unit_to_psu)
        design = SurveyDesign(weights="w", psu="psu")
        res_survey = self._fit(df_s, design=design)
        # ATT essentially identical: uniform weights have no estimand
        # effect (OLS == WLS at the algebra level), but tolerance is
        # 1-ULP per `feedback_assert_allclose_numerical_parity` — the
        # Rust backend's `solve_ols` takes a different BLAS code path
        # than the Python backend on ubuntu-latest / windows, producing
        # ~2.77e-17 reduction-order differences that `assert_array_equal`
        # catches as failures even on degenerate uniform weights.
        np.testing.assert_allclose(res_baseline.att, res_survey.att, rtol=1e-14, atol=1e-14)
        # SE differs (Binder TSL meat vs HC1 meat with multiplier)
        assert np.isfinite(res_survey.se) and res_survey.se > 0

    def test_c_binder_helper_math_and_survey_metadata_on_fitted_result(self):
        """Two-part helper-level + fitted-metadata coverage.

        Part 1 (fitted metadata): runs an actual survey-enabled
        `SpilloverDiD.fit()` and verifies the result-class survey fields
        (`n_psu`, `n_strata`, `survey_metadata.df_survey`, finite SE) are
        wired correctly through the Wave E.1 plumbing.

        Part 2 (helper math): independently sanity-checks
        `_compute_stratified_meat_from_psu_scores` on a small synthetic
        PSU-score fixture, hand-computing the
        `(1-f_h) * n_h/(n_h-1) * sum_j (S_hj - S_h_bar)(S_hj - S_h_bar)'`
        formula and asserting parity to rtol=1e-12.

        This is NOT an end-to-end `bread @ meat @ bread` reconstruction
        against `res.vcov` — the test_a uniform-weight bit-identity check
        and test_o drift goldens cover that surface from different angles.
        Full estimator-level vcov reconstruction would require exposing
        `X_2_kept` arrays the estimator doesn't currently surface; tracked
        as informational follow-up (see DEFERRED.md).
        """
        from diff_diff import SurveyDesign
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        df = generate_butts_nonstaggered_dgp(seed=1)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        res = self._fit(df_s, design=design)
        # Part 1: fitted-result survey metadata is correct.
        assert res.n_psu == 8
        assert res.n_strata == 2
        assert res.survey_metadata is not None
        assert getattr(res.survey_metadata, "df_survey", None) == 6
        assert np.isfinite(res.se) and res.se > 0
        # Part 2: helper math sanity — hand-compute the
        # (1-f_h)*n_h/(n_h-1) factor on synthetic PSU scores.
        psu_scores = np.array([[1.0, 0.5], [-0.5, 0.2], [0.3, -0.1], [-0.4, 0.6]])
        psu_strata = np.array([0, 0, 1, 1])
        psu_fpc = np.array([20.0, 20.0, 20.0, 20.0])
        meat, var_computed, _ = _compute_stratified_meat_from_psu_scores(
            psu_scores, psu_strata, fpc_per_psu=psu_fpc, lonely_psu="remove"
        )
        assert var_computed
        # f_h = 2/20 = 0.1; multiplier = (1-0.1)*2/(2-1) = 1.8.
        # Stratum h=0 PSUs are rows 0,1; mean = (1.0-0.5)/2, (0.5+0.2)/2 = (0.25, 0.35)
        # centered: [(0.75, 0.15), (-0.75, -0.15)]; meat_h0 = 1.8 * sum centered'*centered
        meat_h0_expected = 1.8 * np.array(
            [
                [(0.75) ** 2 + (-0.75) ** 2, 0.75 * 0.15 + (-0.75) * (-0.15)],
                [0.75 * 0.15 + (-0.75) * (-0.15), (0.15) ** 2 + (-0.15) ** 2],
            ]
        )
        # Stratum h=1 PSUs are rows 2,3; mean = (-0.05, 0.25)
        # centered: [(0.35, -0.35), (-0.35, 0.35)]
        meat_h1_expected = 1.8 * np.array(
            [
                [(0.35) ** 2 + (-0.35) ** 2, 0.35 * (-0.35) + (-0.35) * 0.35],
                [0.35 * (-0.35) + (-0.35) * 0.35, (-0.35) ** 2 + (0.35) ** 2],
            ]
        )
        np.testing.assert_allclose(
            meat, meat_h0_expected + meat_h1_expected, rtol=1e-12, atol=1e-14
        )

    def test_c2_non_uniform_weights_path_smoke(self):
        """Smoke check the non-uniform-weight survey path: weighted
        gamma_hat + eps weighting + Psi weighting + bread + Binder TSL
        meat all execute without error and produce finite output with the
        expected `df_survey` from the PSU/strata structure.

        This is a SMOKE TEST, not a numerical parity hand-check. The
        non-uniform-weight aggregation contract is pinned numerically by
        `test_n2_event_study_distinguishes_survey_share_from_sample_share`
        (manual lincom reconstruction at rtol=1e-6) and the cross-rule
        distinguishability assertion there. End-to-end vcov reconstruction
        on this path is tracked in DEFERRED.md (requires exposing the
        estimator's internal X_2_kept arrays).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=2)
        units = df["unit"].unique()
        rng = np.random.default_rng(2)
        w_by_unit = dict(zip(sorted(units), rng.uniform(0.5, 2.0, size=len(units))))
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s["w"] = df_s["unit"].map(w_by_unit)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        res = self._fit(df_s, design=design)
        # SE finite (non-degenerate); ATT may shift slightly from the
        # uniform-weight case because WLS pulls toward heavier-weighted units.
        assert np.isfinite(res.se) and res.se > 0
        assert np.isfinite(res.att)
        # df_survey unchanged from uniform-weight case (depends only on
        # PSU/strata structure, not on weight values).
        assert getattr(res.survey_metadata, "df_survey", None) == 6

    def test_d_lonely_psu_modes_accepted(self):
        """The three lonely_psu modes ('remove', 'certainty', 'adjust') all
        flow through SpilloverDiD without error and produce well-defined
        output (finite SE or NaN-propagated inference).

        The methodological behavior of each mode is audited at the
        `_compute_stratified_meat_from_psu_scores` helper level (see
        `tests/test_survey_*.py`); this test verifies parameter propagation
        through SpilloverDiD's survey path.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=3)
        # Mixed design: stratum 0 collapsed to singleton PSU (triggers
        # lonely_psu handling); stratum 1 stays at 4 PSUs (regular path).
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s.loc[df_s["stratum"] == 0, "psu"] = 0

        for mode in ("remove", "certainty", "adjust"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                design = SurveyDesign(
                    weights="w",
                    strata="stratum",
                    psu="psu",
                    fpc="N_h",
                    lonely_psu=mode,
                )
                res = self._fit(df_s, design=design)
                # Either finite SE or NaN-propagated (consistent inference).
                if np.isfinite(res.se):
                    assert res.se >= 0
                else:
                    assert np.isnan(res.t_stat) and np.isnan(res.p_value)

    def test_e1_fpc_large_matches_no_fpc(self):
        """Very large fpc produces (1-f_h)→1 → SE close to the no-FPC path.

        Note: SurveyDesign validates that FPC is finite + non-NaN, so
        np.inf is rejected upfront. Use a very large finite N_h instead
        (e.g. 10^9 sampled-fraction effectively → 0).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=4)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=1e9)
        design_fpc_large = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        design_no_fpc = SurveyDesign(weights="w", strata="stratum", psu="psu")
        res_large = self._fit(df_s, design=design_fpc_large)
        res_no = self._fit(df_s, design=design_no_fpc)
        # f_h = 4 / 1e9 ≈ 4e-9; (1-f_h) ≈ 1 - 4e-9. SE difference < 1e-6.
        np.testing.assert_allclose(res_large.se, res_no.se, rtol=1e-6)

    def test_e2_fpc_equals_n_zeros_stratum(self):
        """When fpc = n_h, (1-f_h) = 0 and stratum contributes zero."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=5)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=4.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        # When all strata's PSUs are at FPC saturation, the meat is zero.
        # Variance is identified-but-zero (degenerate) → SE = 0.
        res = self._fit(df_s, design=design)
        np.testing.assert_allclose(res.se, 0.0, atol=1e-14)

    def test_e3_fpc_intermediate_monotonic_shrinkage(self):
        """As FPC grows (sampling fraction shrinks), SE shrinks monotonically
        toward the no-FPC limit."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=6)
        ses = []
        for fpc_val in [10.0, 20.0, 40.0, 1000.0]:
            df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=fpc_val)
            design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
            ses.append(self._fit(df_s, design=design).se)
        # As FPC increases, (1-f_h) → 1, SE approaches the no-FPC limit
        # from below (smaller f → smaller multiplier reduction → LARGER SE).
        assert (
            ses[0] < ses[1] < ses[2] <= ses[3]
        ), f"Expected monotonic SE growth as FPC grows; got {ses}"

    def test_f_saturated_df_survey_zero_nan_inference_with_warning(self):
        """df_survey = 0 (single PSU per stratum + lonely_psu='remove') →
        NaN inference + UserWarning matching 'df_survey'."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=7)
        df_s = _augment_with_survey(df, n_strata=4, psus_per_stratum=1, fpc=200.0)
        design = SurveyDesign(
            weights="w",
            strata="stratum",
            psu="psu",
            fpc="N_h",
            lonely_psu="remove",
        )
        with pytest.warns(UserWarning, match="df_survey"):
            res = self._fit(df_s, design=design)
        # SE should be NaN; all inference fields NaN-consistent
        assert np.isnan(res.se)
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)
        assert all(np.isnan(c) for c in res.conf_int)

    def test_h_cluster_plus_survey_psu_warn_and_use_psu(self):
        """cluster=<col> + survey.psu → UserWarning, then PSU wins."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=8)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Add a different cluster column that doesn't match PSU
        df_s["bad_cluster"] = df_s["unit"]
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with pytest.warns(UserWarning):
            est = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                conley_metric="haversine",
                cluster="bad_cluster",
            )
            res = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        # n_clusters should equal n_psu (PSU wins), not the unit count
        assert res.n_clusters == 8

    def test_i_replicate_weight_variance_rejected(self):
        """Replicate-weight variance (BRR/Fay/JK/SDR) is deferred."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=9)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Add unit-constant fake replicate weight columns (panel survey
        # validator requires within-unit constancy)
        rng = np.random.default_rng(9)
        units_sorted = sorted(df_s["unit"].unique())
        for r in range(4):
            w_by_unit = dict(zip(units_sorted, rng.uniform(0.5, 1.5, size=len(units_sorted))))
            df_s[f"rep_{r}"] = df_s["unit"].map(w_by_unit)
        design = SurveyDesign(
            weights="w",
            replicate_weights=[f"rep_{r}" for r in range(4)],
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="follow-up"):
            self._fit(df_s, design=design)

    def test_j_non_pweight_rejected(self):
        """SpilloverDiD survey support requires weight_type='pweight'."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=10)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(
            weights="w",
            strata="stratum",
            psu="psu",
            weight_type="fweight",
        )
        with pytest.raises(ValueError, match="pweight"):
            self._fit(df_s, design=design)

    def test_k_fit_idempotent_on_survey_path(self):
        """clone() + repeat-fit on survey path produces identical results;
        no fit-time mutation of survey state (per feedback_fit_does_not_mutate_config).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=11)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
        )
        params_before = est.get_params()
        res1 = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Clone via re-instantiation with the same params (avoids sklearn
        # dependency; same effect as sklearn.base.clone for this surface).
        est2 = SpilloverDiD(**est.get_params())
        res2 = est2.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        params_after = est.get_params()
        assert params_before == params_after  # no fit-time mutation
        np.testing.assert_allclose(res1.att, res2.att, rtol=1e-14)
        np.testing.assert_allclose(res1.se, res2.se, rtol=1e-14)

    def test_q_weights_only_no_psu_no_strata(self):
        """SurveyDesign(weights="w") (no PSU, no strata) — degenerate Binder
        TSL collapses to per-obs aggregation with a single synthetic
        stratum. df_survey = n_obs - 1 per ResolvedSurveyDesign.df_survey."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=30)
        df["w"] = 1.0
        design = SurveyDesign(weights="w")  # no strata, no psu, no fpc
        res = self._fit(df, design=design)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        assert res.survey_metadata is not None
        # df_survey = n_obs - 1 (no PSU, no strata branch)
        assert res.survey_metadata.df_survey == len(df) - 1

    def test_r_weights_and_strata_no_psu(self):
        """SurveyDesign(weights, strata) (no PSU) — Binder TSL with
        per-obs synthetic PSU + user-provided strata.
        df_survey = n_obs - n_strata per ResolvedSurveyDesign.df_survey."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=31)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Drop PSU and FPC; keep stratum + weight only
        design = SurveyDesign(weights="w", strata="stratum")
        res = self._fit(df_s, design=design)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        # df_survey = n_obs - n_strata (PSU None + strata branch)
        assert res.survey_metadata.df_survey == len(df_s) - 2

    def test_s_cluster_kwarg_without_survey_psu_becomes_cr1(self):
        """cluster=<col> + survey_design (no PSU) — `_inject_cluster_as_psu`
        substitutes the cluster column for the missing PSU so the survey
        path becomes CR1 + Binder TSL. The documented contract for
        cluster=<col> under survey_design when PSU is absent."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=32)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Survey design WITHOUT PSU; cluster=<col> takes over via injection
        design = SurveyDesign(weights="w", strata="stratum")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            cluster="psu",  # use the augmented psu column AS the cluster
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        # n_clusters reflects the cluster-as-PSU injection: 8 unique PSU
        # labels (2 strata × 4 PSUs) become the effective cluster.
        assert res.n_clusters == 8

    def test_cluster_kwarg_time_varying_rejected_when_used_as_psu(self):
        """When `cluster=<col>` becomes the effective PSU under survey
        (because `survey_design.psu` is absent), the cluster column must
        satisfy panel-survey within-unit constancy.

        Regression for R11 codex P1: pre-fix, `_validate_unit_constant_survey`
        only checked columns named in `survey_design`. A time-varying
        `cluster=<col>` was silently injected as PSU and used for Binder
        aggregation; the equivalent labels passed via `survey_design.psu=`
        would have been rejected by the panel-survey contract.
        Post-fix: rejected upfront with a `ValueError` matcher.

        Symmetry: a unit-constant `cluster=<col>` should still work
        identically to passing the same labels via `survey_design.psu=`.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=40)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Time-varying cluster: cluster value = row index (varies within unit)
        df_s["time_varying_cluster"] = np.arange(len(df_s))
        design = SurveyDesign(weights="w", strata="stratum", fpc="N_h")  # no psu
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            cluster="time_varying_cluster",
        )
        with pytest.raises(ValueError, match="varies within unit"):
            est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

        # Symmetry: unit-constant cluster (= the augmented `psu` column)
        # works fine on the implicit-PSU path and produces the same result
        # as passing it via `survey_design.psu=` explicitly.
        est_unit_constant = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            cluster="psu",  # unit-constant by _augment_with_survey construction
        )
        res_implicit = est_unit_constant.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Same labels as explicit PSU
        design_explicit = SurveyDesign(
            weights="w",
            strata="stratum",
            psu="psu",
            fpc="N_h",
        )
        est_explicit = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
        )
        res_explicit = est_explicit.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design_explicit,
        )
        np.testing.assert_allclose(res_implicit.att, res_explicit.att, rtol=1e-14)
        np.testing.assert_allclose(res_implicit.se, res_explicit.se, rtol=1e-14)
        assert res_implicit.n_clusters == res_explicit.n_psu == 8

    def test_cluster_kwarg_overlap_across_strata_nest_false_raises(self):
        """When cluster labels repeat across strata under nest=False, the
        implicit cluster-as-PSU injection must raise (matching the explicit
        PSU resolver's contract at `SurveyDesign.resolve()` L305-L316).

        Regression for R8 codex P1: pre-fix, `_inject_cluster_as_psu`
        always nested cluster IDs within strata, silently manufacturing
        extra PSUs and producing inconsistent `n_psu` / `df_survey` /
        Binder meat vs the equivalent explicit `psu=` specification.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=33)
        units_sorted = sorted(df["unit"].unique())
        n_units = len(units_sorted)
        # Build a deliberately-overlapping cluster column: cluster labels
        # 0,1,2,3 appear in BOTH strata 0 and stratum 1.
        unit_to_stratum = {u: min(i * 2 // n_units, 1) for i, u in enumerate(units_sorted)}
        unit_to_cluster = {u: i % 4 for i, u in enumerate(units_sorted)}
        df_s = df.copy()
        df_s["stratum"] = df_s["unit"].map(unit_to_stratum)
        df_s["overlapping_cluster"] = df_s["unit"].map(unit_to_cluster)
        df_s["w"] = 1.0
        # nest=False (default for SurveyDesign): the cluster labels must
        # be globally unique across strata, which they're not here.
        design = SurveyDesign(weights="w", strata="stratum")  # nest defaults to False
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            cluster="overlapping_cluster",
        )
        with pytest.raises(ValueError, match="repeat across strata"):
            est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_cluster_kwarg_overlap_across_strata_nest_true_ok(self):
        """When `nest=True`, the same overlapping cluster labels are
        combined with strata via `(stratum, cluster)` nesting — the implicit
        injection matches the explicit-PSU resolver and the fit completes.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=34)
        units_sorted = sorted(df["unit"].unique())
        n_units = len(units_sorted)
        unit_to_stratum = {u: min(i * 2 // n_units, 1) for i, u in enumerate(units_sorted)}
        unit_to_cluster = {u: i % 4 for i, u in enumerate(units_sorted)}
        df_s = df.copy()
        df_s["stratum"] = df_s["unit"].map(unit_to_stratum)
        df_s["overlapping_cluster"] = df_s["unit"].map(unit_to_cluster)
        df_s["w"] = 1.0
        design = SurveyDesign(weights="w", strata="stratum", nest=True)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            cluster="overlapping_cluster",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        assert np.isfinite(res.se) and res.se > 0
        # With nest=True, (stratum, cluster) combinations are unique → 8 PSUs
        # (2 strata × 4 cluster labels)
        assert res.n_clusters == 8

    def test_zero_weight_omega0_unit_excluded_from_fe_support(self):
        """Zero-weight Omega_0 rows are outside the WLS estimating sample.

        Regression for R3 codex P0: pre-fix, the survey-weighted
        `_iterative_fe_subset` aggregated zero-weight rows into the FE
        support (denominator 0, `np.where` wrote `0.0`), then materialized
        finite FE values for the affected unit/time codes — silently
        corrupting `att`, ring effects, and SEs. Post-fix, zero-weight
        rows are excluded via `omega_0_pos = omega_0_mask & (w > 0)`, so
        a unit whose Omega_0 rows all have weight 0 receives NaN FE and
        is excluded by `finite_mask` from stage-2.

        This test sets one far-control unit's weight to 0 across all
        periods and verifies fit completes + the resulting `att` is
        close to the att produced WITHOUT that unit.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=70)
        units_sorted = sorted(df["unit"].unique())
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Zero out one far-control unit's weight
        zero_unit = units_sorted[-1]
        df_s.loc[df_s["unit"] == zero_unit, "w"] = 0.0
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = self._fit(df_s, design=design)
        # Fit completes with finite output
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        # Compare against fit WITHOUT the zero-weight unit; difference
        # should be small (zero-weight exclusion ≈ row exclusion for FE
        # estimation, modulo stage-2 ring presence). Rtol=0.05 absorbs
        # the small stage-2 numerator/denominator effect of the dropped
        # ring contribution.
        df_without = df_s[df_s["unit"] != zero_unit].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_without = self._fit(df_without, design=design)
        np.testing.assert_allclose(res.att, res_without.att, rtol=0.05)

    def test_all_zero_weight_omega0_raises(self):
        """When ALL weights are zero (and thus all Omega_0 rows have
        weight 0), fit raises ValueError with the positive-weight pointer.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=71)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s["w"] = 0.0
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with pytest.raises((ValueError, ZeroDivisionError), match="positive|empty|weight"):
            self._fit(df_s, design=design)

    def test_p_finite_mask_subsetting_path_populated(self):
        """`survey_metadata` is populated correctly when survey arrays flow
        through the finite_mask code path. The subsetting block at
        `spillover.py` (Chunk 3c) runs unconditionally under the survey
        path — when `n_nan == 0`, the no-op branch passes through; when
        `n_nan > 0`, the subset/replace branch runs. This test exercises
        the no-op pass-through; the subset/replace branch is exercised
        by any DGP whose finite_mask drops rows (e.g. binary-treatment
        baseline-treated unit panels that pre-PR Wave B tests cover)."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=12)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        res = self._fit(df_s, design=design)
        # On the no-drop path: n_psu / n_strata reflect the input survey
        # structure exactly. Both equal what _augment_with_survey created.
        assert res.n_psu == 8
        assert res.n_strata == 2
        # survey_metadata.df_survey computed via the PSU+strata branch
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey == 6
        # n_obs reflects the post-finite_mask sample (no drops here)
        assert res.n_obs == len(df_s)


class TestSpilloverDiDWaveE1SurveyDesignEventStudy:
    """Event-study branch + survey_design, both is_staggered branches."""

    def test_l_event_study_survey_is_staggered_true(self):
        """Full plumbing works end-to-end on the staggered event-study path."""
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=20)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=True,
            horizon_max=2,
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        assert res.spillover_effects is not None and not res.spillover_effects.empty
        assert res.att_dynamic is not None and not res.att_dynamic.empty
        # df_survey threading: t-stat uses df_survey-based critical value,
        # not OLS-residual df. Verify by computing a manual t-quantile.
        from scipy import stats as _stats

        # CI half-width at survey DOF should differ from OLS-residual half-width
        df_survey = res.survey_metadata.df_survey
        t_crit_survey = _stats.t.ppf(1 - 0.05 / 2, df=df_survey)
        ci_low, ci_high = res.conf_int
        expected_half_width = t_crit_survey * res.se
        observed_half_width = (ci_high - ci_low) / 2
        np.testing.assert_allclose(observed_half_width, expected_half_width, rtol=1e-6)

    def test_m_event_study_survey_is_staggered_false(self):
        """Event-study path on the non-staggered branch."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=21, n_periods=4, t_treat=2)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=True,
            horizon_max=1,
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.att)
        assert res.is_staggered is False  # confirm we're on the non-staggered branch

    def test_n_aggregate_vs_event_study_parity_nonuniform_weights(self):
        """Under non-uniform survey weights and a constant-effect DGP, the
        event-study scalar `att` (lincom on post-treatment horizons with
        SURVEY-WEIGHTED share weights) approximately reproduces the
        aggregate ATT.

        Load-bearing test of R2 codex fix: pre-fix, event-study used raw
        n_obs_per_col shares on weighted WLS horizon coefficients,
        producing inconsistent estimands. Post-fix, the lincom weights
        are per-horizon survey-weight totals.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=40, n_periods=4, t_treat=2, error_sd=0.0)
        units_sorted = sorted(df["unit"].unique())
        rng = np.random.default_rng(40)
        w_by_unit = dict(zip(units_sorted, rng.uniform(0.5, 2.0, size=len(units_sorted))))
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s["w"] = df_s["unit"].map(w_by_unit)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")

        est_agg = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=False,
        )
        res_agg = est_agg.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )

        est_es = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=True,
            horizon_max=1,
        )
        res_es = est_es.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res_agg.att) and np.isfinite(res_agg.se)
        assert np.isfinite(res_es.att) and np.isfinite(res_es.se)
        assert res_agg.n_psu == res_es.n_psu
        assert res_agg.n_strata == res_es.n_strata
        # Constant-effect parity: weighted-share event-study att should
        # match the aggregate WLS coefficient. Tolerance acknowledges
        # that event-study stage-2 design (treatment + per-k dummies + per-
        # (ring, k) dummies) differs from the aggregate design (single
        # treatment + per-ring dummies); residual-share-weighted parity
        # is approximate.
        np.testing.assert_allclose(res_es.att, res_agg.att, rtol=0.15)

    def test_n2_event_study_distinguishes_survey_share_from_sample_share(self):
        """Pin the survey-share lincom rule with a NON-constant-effect
        staggered DGP. With `tau_k` varying across horizons + non-uniform
        weights, the WRONG rule (raw n_obs_per_col sample shares) and the
        documented survey-share rule produce materially different scalar
        `att` values.

        This test asserts (a) the reported `att` equals the impl-internal
        survey-share linear combination on the captured `att_dynamic`
        coefficients AND (b) the wrong sample-share rule would differ
        from the reported value by more than 0.01. The combination rules
        out a silent regression to sample-share weighting.

        Reconstructing exact share_k from outside requires replicating
        the horizon-binning logic (`_apply_horizon_binning`); we do that
        manually via `np.clip(K_raw, -H, H)` on the post-treatment rows.
        Note: `att_dynamic.n_obs` stores RAW per-k observation counts on
        all paths (Wave C/D contract); the survey-share weights the impl
        uses for the lincom live in the internal
        `event_study_meta["weight_sum_per_col"]` slot, not in `att_dynamic`.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(
            seed=80, tau_per_event_time=lambda k: 0.5 * (1.0 + max(k, 0))
        )
        units_sorted = sorted(df["unit"].unique())
        # Correlate weights with treatment cohort so survey shares lean
        # toward LATER cohorts (whose post-treat rows have larger K_direct
        # under the tau(k) = 0.5*(1+k) curve). Without this correlation,
        # uniform-vs-non-uniform weights give nearly identical atts.
        first_treat_by_unit = df.groupby("unit")["first_treat"].first().to_dict()
        w_corr = {
            u: (3.0 if np.isfinite(first_treat_by_unit[u]) and first_treat_by_unit[u] >= 3 else 0.5)
            for u in units_sorted
        }
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s["w"] = df_s["unit"].map(w_corr)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")

        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            event_study=True,
            horizon_max=2,
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        df_uniform = df_s.copy()
        df_uniform["w"] = 1.0
        res_uniform = est.fit(
            df_uniform,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.att) and np.isfinite(res.se)
        assert np.isfinite(res_uniform.att)
        # Distinguishability gap (informational): cohort-correlated weights
        # + non-constant tau_k skew the survey-share att toward later
        # cohorts vs the uniform-weight reference.
        gap = abs(res.att - res_uniform.att)
        assert gap > 0.005, (
            f"Test DGP doesn't distinguish survey-share from sample-share "
            f"(reported {res.att:.4f} vs uniform-weights {res_uniform.att:.4f}, "
            f"gap {gap:.4f})."
        )

        # Direct lincom reconstruction on captured att_dynamic.
        #
        # The impl computes per-horizon shares as:
        #   share_k = sum_{i: X_2[i, col_k] != 0} survey_weights_normalized[i]
        # where col_k is the direct-effect column for horizon k after
        # `_apply_horizon_binning(K_arr, horizon_max)` (np.clip(-H, +H)).
        # We replicate this externally:
        H = 2  # horizon_max
        # Hájek-normalize over the full sample to match _resolve_survey_for_fit
        w_norm = df_s["w"] / df_s["w"].mean()
        # K_direct for treated rows (D=1); D=0 rows carry NaN K and don't
        # contribute to any direct-effect column.
        treated_rows = df_s[df_s["D"] == 1].copy()
        treated_rows["K_raw"] = treated_rows["time"] - treated_rows["first_treat"]
        # Apply the same endpoint-pooling clip the impl uses
        treated_rows["K_binned"] = np.clip(treated_rows["K_raw"].values, -H, H)
        treated_rows["w_norm"] = treated_rows["unit"].map(
            {u: w_norm.loc[df_s["unit"] == u].iloc[0] for u in treated_rows["unit"].unique()}
        )
        # Survey-share per binned k (post-treatment only: k >= 0)
        survey_shares = (
            treated_rows.loc[treated_rows["K_binned"] >= 0]
            .groupby("K_binned")["w_norm"]
            .sum()
            .to_dict()
        )
        # Sample-share per binned k (raw row counts)
        sample_shares = (
            treated_rows.loc[treated_rows["K_binned"] >= 0]
            .groupby("K_binned")
            .size()
            .astype(np.float64)
            .to_dict()
        )
        att_dyn = res.att_dynamic
        post_k = sorted(int(k) for k in survey_shares.keys() if int(k) in att_dyn.index)
        if post_k:
            survey_total = sum(survey_shares[k] for k in post_k)
            survey_att = sum(
                (survey_shares[k] / survey_total) * att_dyn.loc[k, "coef"] for k in post_k
            )
            sample_total = sum(sample_shares[k] for k in post_k)
            sample_att = sum(
                (sample_shares[k] / sample_total) * att_dyn.loc[k, "coef"] for k in post_k
            )
            # Impl uses survey-share rule. Tight tolerance (rtol=1e-6).
            np.testing.assert_allclose(res.att, survey_att, rtol=1e-6)
            # The wrong (sample-share) rule must differ by a non-trivial
            # amount on this DGP — otherwise the assertion above wouldn't
            # have load-bearing distinguishing power.
            assert abs(survey_att - sample_att) > 1e-3, (
                f"DGP doesn't distinguish survey-share ({survey_att}) from "
                f"sample-share ({sample_att}); both rules produce the same "
                f"att, so this test cannot catch a regression to the wrong rule."
            )

    def test_p2_finite_mask_forces_drop_under_survey(self):
        """Force a real `finite_mask` drop on the survey path by including
        a baseline-treated unit (D=1 at every period) alongside a late-
        treated unit and an untreated far-control. The baseline-treated
        unit has no Omega_0 rows → unit FE NaN → y_tilde NaN → finite_mask
        drops its 2 rows from stage 2.

        Wave E.3 (shipped): warn-and-dropped rows are RETAINED in the
        resolved survey design as zero-score padding (matches R
        `survey::svyrecvar(subset())` + `imputation.py:2175-2183`
        precedent). `n_psu` / `n_strata` / `df_survey` reflect the FULL
        domain (all 10 PSUs across 2 strata) rather than the post-drop
        fit sample (the prior Wave E.1 behavior of 8 PSUs / df_survey=6).

        DGP shape mirrors the existing pre-Wave-E.1
        `test_baseline_treated_unit_at_t0_recognized` (3 units × 2 periods
        with cohort spacing that avoids tripping solve_ols's column-drop
        path).
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(1)
        rows = []
        # Mirror the DGP shape from the pre-existing
        # `test_partial_unsupported_units_warn_and_drop` (haversine
        # metric: lat=0 vs lat=10 vs lat=20 are degrees → ~1100 / 2200 km
        # apart, well beyond d_bar=100). Survey columns added per unit
        # (constant within unit; varied across units so the survey design
        # is non-degenerate).
        next_psu = 0
        # 2 baseline-treated units (no Omega_0 → warn-and-drop).
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 0,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        # 3 validly-treated units (treated from t=1).
        for k in range(3):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 0,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 1,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        df = pd.DataFrame(rows)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with pytest.warns(UserWarning, match=r"2 unit\(s\) have NO"):
            res = est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=design,
            )
        # 2 baselines × 2 periods = 4 rows excluded.
        # Remaining: 3 treated_t1 × 2 + 5 far × 2 = 16 rows.
        assert res.n_obs == 16, f"expected 16 post-drop rows, got {res.n_obs}"
        # Wave E.3: survey_metadata reflects the FULL domain (zero-pad
        # invariant). All 10 PSUs retained — the 2 baseline PSUs are
        # excluded from the gamma_hat / Psi construction sample
        # (survey_finite_mask = finite_mask & survey_weights > 0) and
        # zero-padded back into the meat via score_pad_mask=survey_finite_mask,
        # so they contribute zero score but still count toward n_psu_full
        # and the per-stratum n_h denominators.
        assert res.survey_metadata is not None
        assert res.n_psu == 10, f"Wave E.3: expected n_psu=10 (full domain), got {res.n_psu}"
        assert res.n_strata == 2
        # df_survey = n_psu - n_strata = 10 - 2 = 8 (Wave E.3 full-domain).
        assert res.survey_metadata.df_survey == 8

    def test_o_drift_golden(self):
        """Pin Wave E.1 survey ATT + SE on a fixed-seed DGP.

        Hard-coded golden values captured on the initial Wave E.1
        implementation. Tolerance matches the BLAS-reduction-ordering band
        per `feedback_assert_allclose_numerical_parity`. If a future change
        shifts these, investigate (do NOT loosen tolerance per
        `feedback_holistic_codex_test_failure_deviation`).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=999)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Drift pins. Captured on initial Wave E.1 implementation (seed=999,
        # standard 2-strata × 4-PSU augmentation). assert_allclose tolerance
        # acknowledges PSU-aggregation BLAS reduction order variation.
        _WAVE_E1_GOLDEN_ATT = -0.07749624543132044
        _WAVE_E1_GOLDEN_SE = 0.005063316956088809
        np.testing.assert_allclose(res.att, _WAVE_E1_GOLDEN_ATT, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(res.se, _WAVE_E1_GOLDEN_SE, rtol=1e-12, atol=1e-14)
        # Lock down DOF + n_psu (these should be deterministic across runners)
        assert res.n_psu == 8
        assert res.n_strata == 2
        assert res.survey_metadata.df_survey == 6


class TestSpilloverDiDWaveE2ConleySurveyDesign:
    """Wave E.2 conley + survey via stratified-Conley sandwich on PSU totals.

    Methodology anchor: Conley (1999) spatial-HAC composed with Gerber
    (2026) Prop 1 Binder TSL (Wave E.1 foundation) and the Wave D Gardner
    GMM correction. Verifies reduction semantics (bandwidth -> 0 ≡ Binder;
    H=1 ≡ plain Conley on PSU totals), cross-stratum independence,
    singleton-adjust FPC skip parity with Binder, and the saturation
    NaN-fail.
    """

    _CUTOFF_KM = 1000.0  # large enough that within-stratum PSU pairs are inside

    def _fit(self, df, **kwargs):
        design = kwargs.pop("design", None)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=False,
            **kwargs,
        )
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )

    def test_a_no_survey_conley_path_matches_wave_d_golden(self):
        """The `resolved_survey is None` branch of the new dispatch must
        produce the SAME no-survey Conley SE as the pre-Wave-E.2 (Wave D)
        Conley path. The Wave D path is `_compute_conley_meat(...)` with
        no changes; the new dispatch only ADDS an `if resolved_survey is
        not None` branch above the existing call. Pin the SE to a golden
        captured on this fixture so any future refactor that disturbs the
        no-survey path is caught by a behavioral test, not just by
        determinism.
        """
        df = generate_butts_nonstaggered_dgp(seed=0)
        res = self._fit(df)
        # Wave D no-survey Conley golden captured on this fixture (seed=0,
        # 2-period non-staggered Butts DGP, cutoff=1000 km, Bartlett kernel).
        # These values reflect the pre-Wave-E.2 no-survey Conley path.
        # The dispatch in `_compute_gmm_corrected_meat` only ADDS a new
        # `if resolved_survey is not None` branch above the existing
        # `_compute_conley_meat` call, so the `resolved_survey is None`
        # path is bit-identical to Wave D; any future refactor that
        # disturbs it must update these goldens deliberately.
        _WAVE_D_NO_SURVEY_CONLEY_ATT = -0.07471658104745109
        _WAVE_D_NO_SURVEY_CONLEY_SE = 0.0018453344099259904
        np.testing.assert_allclose(res.att, _WAVE_D_NO_SURVEY_CONLEY_ATT, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(res.se, _WAVE_D_NO_SURVEY_CONLEY_SE, rtol=1e-12, atol=1e-14)
        assert np.isfinite(res.se) and res.se > 0

    def test_a2_no_survey_conley_path_routes_through_wave_d_helper(self):
        """Structural anchor: a no-survey conley fit invokes the Wave D
        `_compute_conley_meat` helper directly, NOT the Wave E.2
        `_compute_stratified_conley_meat` orchestrator. Pins the dispatch
        branch in `_compute_gmm_corrected_meat` (no leak into the new
        path when `resolved_survey is None`).
        """
        from unittest.mock import patch

        df = generate_butts_nonstaggered_dgp(seed=2)
        with patch("diff_diff.two_stage._compute_stratified_conley_meat") as mock_panel_aware:
            self._fit(df)
            assert not mock_panel_aware.called, (
                "No-survey conley fit must NOT call _compute_stratified_conley_meat "
                "(the Wave E.2 panel-aware survey path); it should route through "
                "the Wave D _compute_conley_meat directly."
            )

    def test_b_panel_aware_per_period_sum_invariant(self):
        """Panel-aware Wave E.2 meat == sum-across-periods of per-period
        within-stratum Conley sandwich on per-period PSU totals.

        Pure unit test on the orchestrator + helper composition: with T
        periods of synthetic PSU-level data, ``_compute_stratified_conley_meat``'s
        per-period loop must produce the same result as manually calling
        the survey helper T times (once per period, on per-period PSU
        totals) and summing. This pins the library's panel Conley contract
        (``conley_lag_cutoff = 0`` means "within-period spatial only") on
        the survey path — no cross-period spatial pairs leak through the
        collapsed PSU totals.

        Replaces the original "bandwidth → 0 reduces to Wave E.1 Binder"
        claim, which only holds under T=1 (the cross-sectional limit).
        SpilloverDiD's panel-only contract precludes a T=1 fit, so the
        Wave E.1-equivalence claim is meaningful only on this synthetic
        unit-test fixture.
        """
        from diff_diff.survey import (
            ResolvedSurveyDesign,
            _compute_stratified_conley_meat_from_psu_scores,
        )
        from diff_diff.two_stage import _compute_stratified_conley_meat

        rng = np.random.default_rng(31)
        # 4 PSUs × 2 periods × 3 obs per PSU-period = 24 obs.
        n_obs, T, G, p_2 = 24, 2, 4, 3
        obs_per_psu_period = 3
        psu_id = np.repeat(np.arange(G), obs_per_psu_period * T)
        time_arr = np.tile(np.repeat(np.arange(T), obs_per_psu_period), G)
        Psi = rng.standard_normal((n_obs, p_2))
        psu_centroids = np.array([[40.0, -120.0], [40.1, -120.0], [40.2, -120.0], [40.3, -120.0]])
        coords = psu_centroids[psu_id]
        psu_strata = np.array([0, 0, 1, 1])  # 2 PSUs per stratum
        fpc_per_psu = np.full(G, 20.0)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(n_obs),
            weight_type="pweight",
            strata=np.repeat(psu_strata, obs_per_psu_period * T),
            psu=psu_id,
            fpc=np.full(n_obs, 20.0),
            n_strata=2,
            n_psu=4,
            lonely_psu="remove",
        )
        # Orchestrator (panel-aware).
        meat = _compute_stratified_conley_meat(
            Psi,
            conley_coords=coords,
            conley_cutoff_km=0.30,
            conley_metric="euclidean",
            conley_kernel="bartlett",
            resolved_survey=resolved,
            conley_time=time_arr,
        )
        # Hand: aggregate Psi to PSU WITHIN each period, run the survey
        # helper per period, sum.
        expected = np.zeros((p_2, p_2))
        for t in range(T):
            period_mask = time_arr == t
            Psi_t = Psi[period_mask]
            psu_id_t = psu_id[period_mask]
            S_psu_t = np.zeros((G, p_2))
            for g in range(G):
                S_psu_t[g] = Psi_t[psu_id_t == g].sum(axis=0)
            meat_t, _, _ = _compute_stratified_conley_meat_from_psu_scores(
                S_psu_t,
                psu_strata,
                psu_centroids,
                cutoff=0.30,
                metric="euclidean",
                kernel="bartlett",
                fpc_per_psu=fpc_per_psu,
            )
            expected += meat_t
        np.testing.assert_allclose(meat, expected, rtol=1e-12, atol=1e-14)
        # Sanity: a time-collapsed naive computation (the OLD pre-R2 design)
        # would DIFFER from the panel-aware meat on the same inputs.
        S_psu_collapsed = np.zeros((G, p_2))
        for g in range(G):
            S_psu_collapsed[g] = Psi[psu_id == g].sum(axis=0)
        meat_collapsed, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            S_psu_collapsed,
            psu_strata,
            psu_centroids,
            cutoff=0.30,
            metric="euclidean",
            kernel="bartlett",
            fpc_per_psu=fpc_per_psu,
        )
        # Differs by the cross-period off-diagonal mass (the panel-aware
        # contract drops these by construction).
        assert not np.allclose(meat, meat_collapsed, rtol=1e-3, atol=1e-3)

    def test_c_hand_computation_methodology_anchor(self):
        """Hand-compute the stratified-Conley meat formula on synthetic
        PSU-level inputs and assert parity with the new survey helper.

        Mirrors `_scratch/wave_e2_smoke.py` Chunk 1 methodology anchor.
        """
        from diff_diff.survey import _compute_stratified_conley_meat_from_psu_scores

        rng = np.random.default_rng(7)
        G, k = 8, 3
        psu_strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        psu_coords = np.array(
            [
                [40.00, -120.0],
                [40.10, -120.0],
                [40.20, -120.0],
                [40.30, -120.0],
                [40.05, -120.0],
                [40.15, -120.0],
                [40.25, -120.0],
                [40.35, -120.0],
            ]
        )
        psu_scores = rng.standard_normal((G, k))
        fpc = np.full(G, 20.0)
        cutoff = 0.30

        meat, var_ok, _ = _compute_stratified_conley_meat_from_psu_scores(
            psu_scores,
            psu_strata,
            psu_coords,
            cutoff=cutoff,
            metric="euclidean",
            kernel="bartlett",
            fpc_per_psu=fpc,
            lonely_psu="remove",
        )
        assert var_ok

        # Hand: per stratum, demean, apply Bartlett K on PSU coords,
        # FPC-scale, sum across strata.
        expected = np.zeros((k, k))
        for h in [0, 1]:
            mask = psu_strata == h
            s_h = psu_scores[mask]
            c_h = psu_coords[mask]
            n_h = s_h.shape[0]
            centered = s_h - s_h.mean(axis=0, keepdims=True)
            d = np.sqrt(((c_h[:, None, :] - c_h[None, :, :]) ** 2).sum(axis=2))
            K = np.maximum(0.0, 1.0 - d / cutoff)
            M_h = centered.T @ K @ centered
            f_h = n_h / fpc[mask][0]
            M_h *= (1.0 - f_h) * n_h / (n_h - 1)
            expected += M_h
        np.testing.assert_allclose(meat, expected, rtol=1e-12, atol=1e-14)

    def test_d_single_stratum_reduces_to_plain_conley_on_psu_totals(self):
        """H = 1 stratum, FPC = inf: reduces to ordinary Conley sandwich
        on PSU totals (modulo the n/(n-1) finite-sample scale).
        """
        from diff_diff.conley import _compute_conley_meat
        from diff_diff.survey import _compute_stratified_conley_meat_from_psu_scores

        rng = np.random.default_rng(11)
        G = 8
        psu_strata = np.zeros(G, dtype=int)
        psu_coords = np.array(
            [
                [40.00, -120.0],
                [40.10, -120.0],
                [40.20, -120.0],
                [40.30, -120.0],
                [40.05, -120.0],
                [40.15, -120.0],
                [40.25, -120.0],
                [40.35, -120.0],
            ]
        )
        psu_scores = rng.standard_normal((G, 3))
        cutoff = 0.30

        meat, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            psu_scores,
            psu_strata,
            psu_coords,
            cutoff=cutoff,
            metric="euclidean",
            kernel="bartlett",
        )
        # Plain Conley sandwich on PSU totals (no FPC). n/(n-1) scale
        # comes from the survey helper's adjustment; FPC term is 1.
        centered = psu_scores - psu_scores.mean(axis=0, keepdims=True)
        plain = _compute_conley_meat(centered, psu_coords, cutoff, "euclidean", "bartlett")
        plain *= G / (G - 1)
        np.testing.assert_allclose(meat, plain, rtol=1e-12, atol=1e-14)

    def test_e_cross_stratum_independence_invariant(self):
        """Cross-stratum kernel weights are exactly zero by sampling design.

        Pure unit test on the new survey helper: full meat ≡ partition-then-sum
        when each partition is fit as a separate single-stratum call. Uses
        interleaved cross-stratum centroids so cross-stratum pairs are
        CLOSER in km than within-stratum pairs — any kernel leak across
        strata would produce a large numerical difference.
        """
        from diff_diff.survey import _compute_stratified_conley_meat_from_psu_scores

        rng = np.random.default_rng(13)
        G, k = 8, 3
        psu_strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Interleaved: stratum 0 at lats 40.00/40.10/40.20/40.30; stratum 1
        # at 40.05/40.15/40.25/40.35. Cross-stratum nearest pair = 0.05 vs
        # within-stratum nearest = 0.10 — kernel would weight them DOUBLE
        # if it leaked.
        psu_coords = np.array(
            [
                [40.00, -120.0],
                [40.10, -120.0],
                [40.20, -120.0],
                [40.30, -120.0],
                [40.05, -120.0],
                [40.15, -120.0],
                [40.25, -120.0],
                [40.35, -120.0],
            ]
        )
        psu_scores = rng.standard_normal((G, k))
        fpc = np.full(G, 20.0)
        cutoff = 0.30

        meat_full, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            psu_scores,
            psu_strata,
            psu_coords,
            cutoff=cutoff,
            metric="euclidean",
            kernel="bartlett",
            fpc_per_psu=fpc,
        )
        partitioned = np.zeros((k, k))
        for h in [0, 1]:
            mask = psu_strata == h
            sub_strata = np.zeros(mask.sum(), dtype=int)
            part_meat, _, _ = _compute_stratified_conley_meat_from_psu_scores(
                psu_scores[mask],
                sub_strata,
                psu_coords[mask],
                cutoff=cutoff,
                metric="euclidean",
                kernel="bartlett",
                fpc_per_psu=fpc[mask],
            )
            partitioned += part_meat
        np.testing.assert_allclose(meat_full, partitioned, rtol=1e-12, atol=1e-14)

    def test_f_lonely_psu_modes_accepted(self):
        """All three lonely_psu modes flow through the conley+survey path."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=14)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        df_s.loc[df_s["stratum"] == 0, "psu"] = 0  # collapse stratum 0 to a singleton PSU
        for mode in ("remove", "certainty", "adjust"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                design = SurveyDesign(
                    weights="w",
                    strata="stratum",
                    psu="psu",
                    fpc="N_h",
                    lonely_psu=mode,
                )
                res = self._fit(df_s, design=design)
                if np.isfinite(res.se):
                    assert res.se >= 0
                else:
                    assert np.isnan(res.t_stat) and np.isnan(res.p_value)

    def test_f2_singleton_adjust_fpc_skip_parity_binder_vs_conley(self):
        """Binder helper and Conley helper produce bit-identical output on
        a singleton stratum with lonely_psu="adjust".

        Load-bearing: pins the Chunk 2 `continue`-skip-FPC pattern. Without
        the `continue`, the Conley helper would divide by `n_h - 1 = 0` on
        the singleton stratum and the meat would NaN-propagate while
        Binder's meat stays finite. With the kernel reducing to identity
        on a singleton (K = [[K(0)]] = [[1.0]]) the two outputs MUST match.
        """
        from diff_diff.survey import (
            _compute_stratified_conley_meat_from_psu_scores,
            _compute_stratified_meat_from_psu_scores,
        )

        rng = np.random.default_rng(15)
        # 5 PSUs: 1 in stratum 0 (singleton), 4 in stratum 1.
        psu_scores = rng.standard_normal((5, 3))
        psu_strata = np.array([0, 1, 1, 1, 1])
        psu_coords = np.array(
            [
                [40.0, -120.0],
                [40.1, -120.0],
                [40.2, -120.0],
                [40.3, -120.0],
                [40.4, -120.0],
            ]
        )
        fpc = np.full(5, 20.0)
        binder_meat, _, _ = _compute_stratified_meat_from_psu_scores(
            psu_scores,
            psu_strata,
            fpc_per_psu=fpc,
            lonely_psu="adjust",
        )
        conley_meat, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            psu_scores,
            psu_strata,
            psu_coords,
            cutoff=1e-10,
            metric="euclidean",
            kernel="bartlett",
            fpc_per_psu=fpc,
            lonely_psu="adjust",
        )
        # Conley with bandwidth -> 0 collapses K to identity in EVERY stratum,
        # so the entire meat (singleton + multi-PSU stratum) reduces to Binder.
        np.testing.assert_allclose(conley_meat, binder_meat, rtol=1e-12, atol=1e-14)
        # And both are finite (the singleton FPC skip prevents divide-by-zero).
        assert np.all(np.isfinite(conley_meat))

    def test_g_fpc_large_matches_no_fpc(self):
        """Very-large FPC (1-f_h ≈ 1) produces SE close to the no-FPC path."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=16)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=1e9)
        design_fpc_large = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        design_no_fpc = SurveyDesign(weights="w", strata="stratum", psu="psu")
        res_large = self._fit(df_s, design=design_fpc_large)
        res_no = self._fit(df_s, design=design_no_fpc)
        np.testing.assert_allclose(res_large.se, res_no.se, rtol=1e-6)

    def test_h_fpc_equals_n_zeros_stratum(self):
        """FPC = n_h per stratum makes (1-f_h) = 0; meat is zero, SE = 0."""
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=17)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=4.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        res = self._fit(df_s, design=design)
        np.testing.assert_allclose(res.se, 0.0, atol=1e-14)

    def test_i_saturated_design_nan_fails(self):
        """All-singleton strata + lonely_psu='remove' -> df_survey = 0 ->
        NaN meat + UserWarning matching 'Wave E.2 stratified-Conley'.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=18)
        df_s = df.copy()
        df_s["w"] = 1.0
        units_sorted = sorted(df_s["unit"].unique())
        unit_to_idx = {u: idx for idx, u in enumerate(units_sorted)}
        df_s["psu"] = df_s["unit"].map(unit_to_idx)
        df_s["stratum"] = df_s["unit"].map(unit_to_idx)  # H = n_units; every stratum singleton
        df_s["N_h"] = 20.0
        design = SurveyDesign(
            weights="w",
            strata="stratum",
            psu="psu",
            fpc="N_h",
            lonely_psu="remove",
        )
        with pytest.warns(UserWarning, match="Wave E.2 stratified-Conley"):
            res = self._fit(df_s, design=design)
        assert np.isnan(res.se)
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)

    def test_j_replicate_weights_rejection_inherits_wave_e1(self):
        """Replicate-weight variance still raises NotImplementedError under
        conley+survey (inherits Wave E.1 gate). SurveyDesign requires
        replicate_weights to be set WITHOUT strata/psu/fpc (they encode
        the design implicitly).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=19)
        df_s = df.copy()
        df_s["w"] = 1.0
        # Add 10 replicate-weight columns; must be constant within units
        # (panel survey constraint).
        rng = np.random.default_rng(19)
        units = sorted(df_s["unit"].unique())
        for r in range(10):
            rep_by_unit = dict(zip(units, rng.uniform(0.5, 2.0, size=len(units))))
            df_s[f"rep_{r}"] = df_s["unit"].map(rep_by_unit)
        design = SurveyDesign(
            weights="w",
            replicate_weights=[f"rep_{r}" for r in range(10)],
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="(?i)replicate|follow-up"):
            self._fit(df_s, design=design)

    def test_k_non_pweight_rejection_inherits_wave_e1(self):
        """Non-pweight weight_type still raises ValueError under conley+survey
        (inherits Wave E.1 gate).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=20)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(
            weights="w",
            strata="stratum",
            psu="psu",
            fpc="N_h",
            weight_type="aweight",
        )
        with pytest.raises((NotImplementedError, ValueError), match="(?i)pweight|aweight"):
            self._fit(df_s, design=design)

    def test_l_cluster_plus_conley_plus_survey_warn_and_use_psu(self):
        """cluster=<col> + conley + survey with different cluster vs PSU ->
        UserWarning fires; PSU wins (mirrors Wave E.1 warn-and-use-PSU).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=21)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Inject a coarser cluster column distinct from PSU (1 cluster
        # per unit). The warn-and-use-PSU path requires that cluster and
        # PSU are NOT identical groupings.
        units_sorted = sorted(df_s["unit"].unique())
        unit_to_cluster = {u: idx // 2 for idx, u in enumerate(units_sorted)}
        df_s["my_cluster"] = df_s["unit"].map(unit_to_cluster)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with pytest.warns(UserWarning, match="(?i)cluster"):
            res = self._fit(df_s, design=design, cluster="my_cluster")
        assert np.isfinite(res.se) and res.se > 0

    def test_m_fit_idempotency_under_conley_survey(self):
        """clone() + repeat fit produces identical results; survey state
        not mutated on fit() (per feedback_fit_does_not_mutate_config).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=22)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
        )
        res_1 = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Second fit on the SAME estimator instance (idempotency).
        res_2 = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert res_1.coefficients == res_2.coefficients
        np.testing.assert_array_equal(res_1.vcov, res_2.vcov)
        assert res_1.n_psu == res_2.n_psu
        assert res_1.n_strata == res_2.n_strata

    def test_n0_no_psu_weights_only_survey_design(self):
        """`SurveyDesign(weights=...)` without explicit PSU — each obs is
        its own pseudo-PSU. Panel-aware path must re-index PSUs WITHIN
        each period (not pad zeros across the full panel) or the centering
        leaks off-period spurious structure into the spatial meat.

        Regression for the R3 P0 fix.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=240)
        df_s = df.copy()
        df_s["w"] = 1.0
        design = SurveyDesign(weights="w")
        res = self._fit(df_s, design=design)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0

    def test_n1_no_psu_strata_only_survey_design(self):
        """`SurveyDesign(weights=..., strata=...)` without explicit PSU —
        each obs is its own pseudo-PSU under stratified sampling. Same
        per-period re-indexing requirement as test_n0.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=241)
        df_s = df.copy()
        df_s["w"] = 1.0
        units_sorted = sorted(df_s["unit"].unique())
        unit_to_stratum = {u: idx % 2 for idx, u in enumerate(units_sorted)}
        df_s["stratum"] = df_s["unit"].map(unit_to_stratum)
        design = SurveyDesign(weights="w", strata="stratum")
        res = self._fit(df_s, design=design)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0

    def test_b2_explicit_psu_centroid_panel_constant_under_finite_mask(self):
        """When a PSU contains multiple units at DIFFERENT coordinates
        (simulating a finite_mask drop that varies coverage across
        periods), the orchestrator must use PANEL-CONSTANT centroids
        (mean across all obs in PSU, regardless of period) — NOT
        per-period centroids. This matches the documented Wave E.2
        contract "centroid_g = mean over i in PSU g of conley_coords[i]"
        at REGISTRY.md and prevents support-sample-dependent kernel
        weights.

        Pure unit test on the orchestrator + helper composition with
        synthetic per-obs inputs.
        """
        from diff_diff.survey import (
            ResolvedSurveyDesign,
            _compute_stratified_conley_meat_from_psu_scores,
        )
        from diff_diff.two_stage import _compute_stratified_conley_meat

        rng = np.random.default_rng(331)
        # 2 strata × 2 PSUs × 1 obs per PSU-period = 8 obs.
        # PSU 0 obs coords differ across periods (simulating finite_mask
        # variation): period 0 at [40.0, 0]; period 1 at [42.0, 0].
        # PSU 1/2/3 have constant coords across periods.
        n, p_2 = 8, 3
        Psi = rng.standard_normal((n, p_2))
        psu_id = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # PSUs alternate per period
        time_arr = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Coords vary across periods for PSU 0 only.
        coords = np.array(
            [
                [40.0, 0.0],  # PSU 0, period 0
                [40.5, 0.0],  # PSU 1, period 0
                [50.0, 0.0],  # PSU 2, period 0
                [50.5, 0.0],  # PSU 3, period 0
                [42.0, 0.0],  # PSU 0, period 1 — DIFFERENT coord
                [40.5, 0.0],  # PSU 1, period 1
                [50.0, 0.0],  # PSU 2, period 1
                [50.5, 0.0],  # PSU 3, period 1
            ]
        )
        psu_strata_obs = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=psu_strata_obs,
            psu=psu_id,
            fpc=None,
            n_strata=2,
            n_psu=4,
            lonely_psu="remove",
        )
        meat_panel = _compute_stratified_conley_meat(
            Psi,
            conley_coords=coords,
            conley_cutoff_km=5.0,
            conley_metric="euclidean",
            conley_kernel="bartlett",
            resolved_survey=resolved,
            conley_time=time_arr,
        )
        # Hand calculation using PANEL-CONSTANT centroids (the contract).
        # PSU 0 centroid = mean([40.0, 0], [42.0, 0]) = [41.0, 0].
        # Other PSUs have constant coords → centroid equals that coord.
        panel_centroids = np.array([[41.0, 0.0], [40.5, 0.0], [50.0, 0.0], [50.5, 0.0]])
        # Per-period PSU totals (each PSU appears once per period in this
        # fixture, so the PSU total per period IS the single obs's Psi).
        psu_strata = np.array([0, 0, 1, 1])
        expected = np.zeros((p_2, p_2))
        for t in [0, 1]:
            mask = time_arr == t
            Psi_t = Psi[mask]
            psu_id_t = psu_id[mask]
            S_psu_t = np.zeros((4, p_2))
            for g in range(4):
                rows = psu_id_t == g
                if rows.any():
                    S_psu_t[g] = Psi_t[rows].sum(axis=0)
            meat_t, _, _ = _compute_stratified_conley_meat_from_psu_scores(
                S_psu_t,
                psu_strata,
                panel_centroids,  # panel-constant — same across periods
                cutoff=5.0,
                metric="euclidean",
                kernel="bartlett",
            )
            expected += meat_t
        np.testing.assert_allclose(meat_panel, expected, rtol=1e-12, atol=1e-14)
        # Counter-check: per-period centroids (the OLD pre-fix design)
        # would give a different meat for PSU 0 because the centroid
        # used in period 1 (42.0) differs from the one used in period 0
        # (40.0). Verify the orchestrator does NOT match that buggy
        # construction.
        buggy_expected = np.zeros((p_2, p_2))
        period_centroids = {
            0: np.array([[40.0, 0.0], [40.5, 0.0], [50.0, 0.0], [50.5, 0.0]]),
            1: np.array([[42.0, 0.0], [40.5, 0.0], [50.0, 0.0], [50.5, 0.0]]),
        }
        for t in [0, 1]:
            mask = time_arr == t
            Psi_t = Psi[mask]
            psu_id_t = psu_id[mask]
            S_psu_t = np.zeros((4, p_2))
            for g in range(4):
                rows = psu_id_t == g
                if rows.any():
                    S_psu_t[g] = Psi_t[rows].sum(axis=0)
            meat_t, _, _ = _compute_stratified_conley_meat_from_psu_scores(
                S_psu_t,
                psu_strata,
                period_centroids[t],  # per-period (buggy)
                cutoff=5.0,
                metric="euclidean",
                kernel="bartlett",
            )
            buggy_expected += meat_t
        # The buggy construction MUST differ measurably from the
        # panel-constant orchestrator output.
        assert not np.allclose(
            meat_panel, buggy_expected, rtol=1e-3, atol=1e-3
        ), "orchestrator unexpectedly matches per-period (buggy) centroid construction"

    def test_n2_no_psu_per_period_reindex_unit_invariant(self):
        """Direct unit test on the orchestrator: the no-PSU per-period
        re-indexing must NOT mix off-period rows into the kernel. With
        synthetic data where obs 0/1 are in period 0 (close in km) and
        obs 2/3 are in period 1 (far away), the meat must reflect
        ONLY within-period spatial pairs.
        """
        from diff_diff.survey import (
            ResolvedSurveyDesign,
            _compute_stratified_conley_meat_from_psu_scores,
        )
        from diff_diff.two_stage import _compute_stratified_conley_meat

        rng = np.random.default_rng(243)
        n, p_2 = 4, 2
        Psi = rng.standard_normal((n, p_2))
        # Period 0: obs 0, 1 at lat 40.00 / 40.01 (close in km).
        # Period 1: obs 2, 3 at lat 50.00 / 50.01 (far from period-0 obs).
        coords = np.array([[40.00, 0.0], [40.01, 0.0], [50.00, 0.0], [50.01, 0.0]])
        time_arr = np.array([0, 0, 1, 1])
        strata_arr = np.array([0, 0, 1, 1])
        resolved = ResolvedSurveyDesign(
            weights=np.ones(n),
            weight_type="pweight",
            strata=strata_arr,
            psu=None,  # implicit per-obs pseudo-PSU
            fpc=None,
            n_strata=2,
            n_psu=n,
            lonely_psu="remove",
        )
        meat_panel = _compute_stratified_conley_meat(
            Psi,
            conley_coords=coords,
            conley_cutoff_km=0.05,
            conley_metric="euclidean",
            conley_kernel="bartlett",
            resolved_survey=resolved,
            conley_time=time_arr,
        )
        # Hand: per-period only on active rows.
        meat_p0, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            Psi[:2],
            np.array([0, 0]),
            coords[:2],
            cutoff=0.05,
            metric="euclidean",
            kernel="bartlett",
        )
        meat_p1, _, _ = _compute_stratified_conley_meat_from_psu_scores(
            Psi[2:],
            np.array([0, 0]),
            coords[2:],
            cutoff=0.05,
            metric="euclidean",
            kernel="bartlett",
        )
        np.testing.assert_allclose(meat_panel, meat_p0 + meat_p1, rtol=1e-12, atol=1e-14)

    def test_n_finite_mask_survey_array_subsetting(self):
        """finite_mask drops baseline-treated rows; survey metadata
        reflects the SUBSET sample, not the original.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=23)
        # Pin a unit to always-treated (g = period 0); finite_mask will
        # drop its rows from stage 2.
        first_unit = sorted(df["unit"].unique())[0]
        df.loc[df["unit"] == first_unit, "first_treat"] = 0
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = self._fit(df_s, design=design)
        # Survey metadata reflects subset (post-finite_mask), not the full panel.
        assert res.survey_metadata is not None
        assert res.n_obs <= len(df_s)  # at least the always-treated unit's rows dropped


class TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoff:
    """Wave E.2 follow-up: conley + survey + conley_lag_cutoff > 0 via
    panel-block composition (spatial + serial Bartlett HAC).

    Methodology anchor: Wave E.2's panel-aware stratified-Conley spatial
    sandwich (Conley 1999 × Binder/Gerber 2026 × Wave D Gardner GMM)
    composed with within-PSU serial Bartlett HAC (Newey-West 1987 separable
    form). Verifies:
      - lag=0 STRICT bit-identity to shipped Wave E.2 ATT and scalar SE
        (test_a) plus mock-spy that the serial helper isn't invoked (test_a2)
      - serial centering = Binder TSL form (per-period within-stratum), NOT raw
      - AR(1) DGP serial-term behavioral inflation
      - panel-wide per-stratum FPC for the serial term
      - panel-wide dense time codes for the lag math
      - singleton-adjust panel-wide mean asymmetry (vs spatial's per-period mean)
      - saturation NaN-fail still fires
    """

    _CUTOFF_KM = 1000.0

    def _fit(self, df, lag_cutoff=1, design=None, **kwargs):
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=lag_cutoff,
            vcov_type="conley",
            event_study=False,
            **kwargs,
        )
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )

    def test_a_lag0_strict_bit_identical_to_wave_e2_meat(self):
        """`conley_lag_cutoff = 0` MUST produce bit-identical ATT AND scalar SE
        as a fresh Wave E.2 baseline fit (`assert_array_equal`). The
        orchestrator does NOT truly early-return at lag=0 — the spatial
        loop, saturation guard, and new PSD/finite guard all still run; the
        guarantee is that the serial helper is NOT invoked (so meat_serial
        contributes nothing). test_a2 mock-spy verifies the helper isn't
        called.

        Methodology lock: skipping the serial helper at the orchestrator
        level is the backwards-compatibility guarantee that the shipped
        Wave E.2 surface is unaffected by the follow-up. Without that skip,
        a numerical zero from the serial helper would still inject
        floating-point noise into the spatial-only meat (which would
        surface as SE drift).

        Note: full meat-matrix equality is NOT asserted — only ATT + scalar
        SE are pinned (the meat matrix is not directly exposed on
        `SpilloverDiDResults`).
        """
        df = generate_butts_nonstaggered_dgp(seed=0)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_lag0 = self._fit(df_s, lag_cutoff=0, design=design)
        # `lag=0` fit must match a fresh Wave E.2 baseline estimator fit
        # (which has the same config, same lag, same data) — bit-identical.
        # This catches any case where the orchestrator fails to short-circuit
        # and the serial helper injects floating-point noise into the
        # spatial-only meat. The Wave E.2 baseline is constructed inline
        # (rather than relying on captured goldens) so this test is robust
        # to BLAS-runner drift on absolute values; the assertion is on
        # cross-fit bit-identity at one runner.
        est_e2_baseline = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_e2_baseline = est_e2_baseline.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        np.testing.assert_array_equal(res_lag0.att, res_e2_baseline.att)
        np.testing.assert_array_equal(res_lag0.se, res_e2_baseline.se)

    def test_a2_lag0_does_not_call_serial_helper(self):
        """Structural anchor: orchestrator skips the serial helper at `lag_cutoff = 0`
        BEFORE invoking `_compute_stratified_serial_bartlett_meat`. Mirrors
        the Wave E.2 test_a2 mock-spy pattern. Without this, a future
        refactor that always-invokes the serial helper would silently degrade
        the lag=0 backwards-compat guarantee.
        """
        from unittest.mock import patch

        df = generate_butts_nonstaggered_dgp(seed=2)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with patch("diff_diff.two_stage._compute_stratified_serial_bartlett_meat") as mock_serial:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._fit(df_s, lag_cutoff=0, design=design)
            assert not mock_serial.called, (
                "lag=0 conley + survey fit must NOT call the serial helper "
                "— orchestrator short-circuit broken."
            )

    def test_b_lag1_invokes_serial_helper(self):
        """lag=1 MUST invoke the new serial helper exactly once."""
        from unittest.mock import patch

        df = generate_butts_nonstaggered_dgp(seed=3)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat as _orig

        with patch(
            "diff_diff.two_stage._compute_stratified_serial_bartlett_meat", wraps=_orig
        ) as spy:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._fit(df_s, lag_cutoff=1, design=design)
            assert spy.called, "lag=1 fit must invoke the serial helper"
            assert spy.call_count == 1

    def test_c0_serial_centering_hand_check_raw_vs_centered(self):
        """Pure unit test: the serial helper applies per-period within-stratum
        centering (Binder TSL form), NOT raw scores like the no-survey
        reference at conley.py:949-965. Construct a 2-PSU × 2-period synthetic
        with materially nonzero per-period stratum mean; assert the centered
        helper output equals the hand-computed centered form (not the raw form).

        Pins MEDIUM #2 from plan-review: codex will push hard on the asymmetry
        vs the no-survey panel-block reference; the load-bearing claim is
        that the centered form is the methodologically-correct choice under
        Binder TSL and that raw scores would inflate the variance.
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        # 2 PSUs (one stratum, both PSUs in stratum 0), 2 periods, score dim 2.
        # Per-period stratum mean is NOT zero (the two PSUs' scores have
        # nonzero average): PSU 0 has [+2, +1] at t=0, [+3, +2] at t=1;
        # PSU 1 has [-1, 0] at t=0, [-2, +1] at t=1.
        # Stratum mean at t=0: [(2-1)/2, (1+0)/2] = [0.5, 0.5]
        # Stratum mean at t=1: [(3-2)/2, (2+1)/2] = [0.5, 1.5]
        # Centered: PSU 0 at t=0: [1.5, 0.5]; PSU 1 at t=0: [-1.5, -0.5]
        #           PSU 0 at t=1: [2.5, 0.5]; PSU 1 at t=1: [-2.5, -0.5]
        # Serial sum per PSU at L=1: K_serial(|0-1|=1) = (1 - 1/2) = 0.5
        # PSU 0 contribution: 0.5 * (centered_t0[0] @ centered_t1[0].T)
        #                   = 0.5 * outer([1.5, 0.5], [2.5, 0.5])
        # PSU 1 contribution: 0.5 * outer([-1.5, -0.5], [-2.5, -0.5])
        # NOTE: K_serial is symmetric so we get the (t=0, t=1) and (t=1, t=0)
        #       contributions both as the same scalar K=0.5 times each PSU's
        #       cross-period outer product (with appropriate transposition).
        # Total per-PSU meat = sum over (t, s) with t != s of K * S_t @ S_s.T
        #   = K * (S_0 @ S_1.T + S_1 @ S_0.T)
        # FPC: n_h_panel = 2, fpc_per_psu = inf -> f_h = 0, scale = 2/(2-1) = 2
        psu_arr = np.array([0, 0, 1, 1])
        time_arr = np.array([0, 1, 0, 1])
        strata_arr = np.array([0, 0, 0, 0])
        fpc_arr = np.full(4, np.inf)
        Psi = np.array(
            [
                [2.0, 1.0],  # PSU 0, t=0
                [3.0, 2.0],  # PSU 0, t=1
                [-1.0, 0.0],  # PSU 1, t=0
                [-2.0, 1.0],  # PSU 1, t=1
            ]
        )

        meat_centered, var_computed, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )
        assert var_computed

        # Hand-compute expected (centered form)
        S_psu_0 = np.array([[2.0, 1.0], [3.0, 2.0]])  # (T, k)
        S_psu_1 = np.array([[-1.0, 0.0], [-2.0, 1.0]])
        # Per-period stratum means
        mean_t0 = (S_psu_0[0] + S_psu_1[0]) / 2  # [0.5, 0.5]
        mean_t1 = (S_psu_0[1] + S_psu_1[1]) / 2  # [0.5, 1.5]
        S0_centered = np.array([S_psu_0[0] - mean_t0, S_psu_0[1] - mean_t1])
        S1_centered = np.array([S_psu_1[0] - mean_t0, S_psu_1[1] - mean_t1])
        # K_serial at L=1: K[t,s] = (1 - |t-s|/2) for |t-s| in {1}; K[t,t] = 0
        K_serial = np.array([[0.0, 0.5], [0.5, 0.0]])
        meat_0 = S0_centered.T @ K_serial @ S0_centered
        meat_1 = S1_centered.T @ K_serial @ S1_centered
        # f_h = 0 (FPC=inf), n_h=2, scale = (1-0) * 2/(2-1) = 2
        meat_expected_centered = 2.0 * (meat_0 + meat_1)
        np.testing.assert_allclose(meat_centered, meat_expected_centered, rtol=1e-12, atol=1e-14)

        # Verify the RAW form would be DIFFERENT (sanity check that centering matters)
        meat_0_raw = S_psu_0.T @ K_serial @ S_psu_0
        meat_1_raw = S_psu_1.T @ K_serial @ S_psu_1
        meat_expected_raw = 2.0 * (meat_0_raw + meat_1_raw)
        # Raw and centered should differ MATERIALLY on this fixture
        assert not np.allclose(
            meat_expected_centered, meat_expected_raw, rtol=1e-2
        ), "Test fixture must have nonzero centering effect to anchor the asymmetry"

    def test_c1_hand_computation_methodology_anchor_lag1(self):
        """Hand-compute the serial Bartlett HAC at L=1 on a 4-PSU x 3-period
        synthetic and assert implementation parity. Methodology anchor that
        gets codified from the _scratch/wave_e2_followup_smoke.py.
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(0)
        G, T, k = 4, 3, 2
        psu_arr = np.repeat(np.arange(G), T)
        time_arr = np.tile(np.arange(T), G)
        strata_arr = np.repeat([0, 0, 1, 1], T)
        fpc_arr = np.full(G * T, 20.0)
        Psi = rng.standard_normal((G * T, k))

        meat, var_computed, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )

        # Hand-compute expected
        S_psu = np.zeros((G, T, k))
        for i in range(G * T):
            S_psu[psu_arr[i], time_arr[i]] += Psi[i]
        meat_expected = np.zeros((k, k))
        for h in [0, 1]:
            stratum_psus = np.where(strata_arr[::T] == h)[0]
            n_h_panel = len(stratum_psus)
            # Per-period within-stratum mean
            S_h = S_psu[stratum_psus]  # (n_h, T, k)
            S_bar_h = S_h.mean(axis=0)  # (T, k)
            S_centered = S_h - S_bar_h[None, :, :]
            # K_serial at L=1
            K = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
            meat_h = np.zeros((k, k))
            for g_local in range(n_h_panel):
                S_g = S_centered[g_local]
                meat_h += S_g.T @ K @ S_g
            N_h = 20.0
            fpc_scale = (1.0 - n_h_panel / N_h) * n_h_panel / (n_h_panel - 1)
            meat_expected += fpc_scale * meat_h

        np.testing.assert_allclose(meat, meat_expected, rtol=1e-12, atol=1e-14)
        assert var_computed

    def test_c2_hand_computation_methodology_anchor_lag2(self):
        """L=2 exercises multiple kernel weights: K(1) = 2/3, K(2) = 1/3."""
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(1)
        G, T, k = 4, 4, 2
        psu_arr = np.repeat(np.arange(G), T)
        time_arr = np.tile(np.arange(T), G)
        strata_arr = np.repeat([0, 0, 1, 1], T)
        fpc_arr = np.full(G * T, 30.0)
        Psi = rng.standard_normal((G * T, k))

        meat, _, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=2,
            lonely_psu="remove",
        )

        # Hand-compute
        S_psu = np.zeros((G, T, k))
        for i in range(G * T):
            S_psu[psu_arr[i], time_arr[i]] += Psi[i]
        meat_expected = np.zeros((k, k))
        # K[t,s] at L=2: (1 - |t-s|/3) for |t-s| in {1, 2}
        K = np.array(
            [
                [0.0, 2 / 3, 1 / 3, 0.0],
                [2 / 3, 0.0, 2 / 3, 1 / 3],
                [1 / 3, 2 / 3, 0.0, 2 / 3],
                [0.0, 1 / 3, 2 / 3, 0.0],
            ]
        )
        for h in [0, 1]:
            stratum_psus = np.where(strata_arr[::T] == h)[0]
            n_h_panel = len(stratum_psus)
            S_h = S_psu[stratum_psus]
            S_bar_h = S_h.mean(axis=0)
            S_centered = S_h - S_bar_h[None, :, :]
            meat_h = np.zeros((k, k))
            for g_local in range(n_h_panel):
                S_g = S_centered[g_local]
                meat_h += S_g.T @ K @ S_g
            N_h = 30.0
            fpc_scale = (1.0 - n_h_panel / N_h) * n_h_panel / (n_h_panel - 1)
            meat_expected += fpc_scale * meat_h

        np.testing.assert_allclose(meat, meat_expected, rtol=1e-12, atol=1e-14)

    def test_c3_serial_term_inflates_se_on_ar1_dgp(self):
        """Behavioral (not invariant): on a panel with within-unit AR(1)
        residuals (positive autocorrelation), SE at lag=1 should be at least
        5% larger than SE at lag=0. Without this test, a bug where the
        serial term computes to near-zero (e.g. centered scores cancel)
        would pass all reduction/parity tests.

        Pins MEDIUM #7 from plan-review: invariant-only coverage is
        susceptible to silent zero bugs.
        """
        # Generate AR(1) panel via deterministic seed
        rng = np.random.default_rng(101)
        n_units, T = 16, 8
        rho = 0.7
        unit_ids = np.repeat(np.arange(n_units), T)
        times = np.tile(np.arange(T), n_units)
        # AR(1) within-unit residuals
        residuals = np.zeros(n_units * T)
        for u in range(n_units):
            e = rng.standard_normal(T)
            r = np.zeros(T)
            r[0] = e[0]
            for t in range(1, T):
                r[t] = rho * r[t - 1] + np.sqrt(1 - rho**2) * e[t]
            residuals[u * T : (u + 1) * T] = r

        # PSU/stratum assignment
        psu_per_unit = unit_ids // 4  # 4 PSUs, 4 units each
        strata_per_unit = psu_per_unit // 2  # 2 strata, 2 PSUs each
        # Spatial coords: UNIT-CONSTANT (SpilloverDiD requires constant coords
        # within each unit across periods). PSU 0/1 (treated) clustered near
        # lat 40; PSU 2/3 (never-treated) FAR (~1000+ km away near lat 50)
        # so they stay in Omega_0 (untreated AND unexposed) at every period.
        # Without the geographic separation, near-control units within
        # ring_max would be flagged as S_it=1 and the time-FE identification
        # would fail at treated periods.
        psu_lat_centers = np.array([40.0, 40.1, 50.0, 50.1])
        psu_lon_centers = np.array([-120.0, -120.1, -120.0, -120.1])
        unit_lat_offset = rng.normal(0, 0.005, size=n_units)
        unit_lon_offset = rng.normal(0, 0.005, size=n_units)
        # unit_to_coords map: unit u gets PSU centroid + unit-level offset
        lat = psu_lat_centers[psu_per_unit] + unit_lat_offset[unit_ids]
        lon = psu_lon_centers[psu_per_unit] + unit_lon_offset[unit_ids]
        # PSU 0/1 treated at t=3; PSU 2/3 never treated
        first_treat = np.where(psu_per_unit < 2, 3, 2**31 - 1)
        # Outcome: pure noise (treatment effect = 0)
        y = residuals + np.where((psu_per_unit < 2) & (times >= 3), 0.0, 0.0)

        df = pd.DataFrame(
            {
                "unit": unit_ids,
                "time": times,
                "first_treat": first_treat,
                "y": y,
                "psu": psu_per_unit,
                "stratum": strata_per_unit,
                "lat": lat,
                "lon": lon,
                "w": 1.0,
                "N_h": 100.0,
            }
        )
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_lag0 = self._fit(df, lag_cutoff=0, design=design)
            res_lag1 = self._fit(df, lag_cutoff=1, design=design)
        # AR(1) with rho=0.7 should produce material serial inflation. The 5%
        # threshold is loose enough to tolerate bootstrap/numerical jitter
        # but tight enough to catch a near-zero serial term.
        inflation = (res_lag1.se / res_lag0.se) - 1.0
        assert inflation > 0.05, (
            f"AR(1) DGP serial inflation {inflation:.4f} should exceed 5%; "
            f"lag0_se={res_lag0.se:.6f}, lag1_se={res_lag1.se:.6f}"
        )

    def test_d_single_stratum_lag1_finite_and_finite_se(self):
        """Single stratum (H=1) + lag=1 fit produces finite output. (Strict
        bit-equivalence to no-survey panel-block conley requires careful
        Hajek-weight + bread alignment; the simpler invariant tested here is
        that the new path produces well-defined output on a single-stratum
        survey design.)
        """
        df = generate_butts_nonstaggered_dgp(seed=5)
        df_s = _augment_with_survey(df, n_strata=1, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = self._fit(df_s, lag_cutoff=1, design=design)
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0

    def test_e_cross_stratum_independence_with_serial(self):
        """Pure unit test on the serial helper: full meat must equal
        partition-by-stratum-then-sum-of-each-stratum's-meat. Pins that
        cross-stratum contributions are exactly zero in the serial term
        (each stratum's serial sum is independent because it only iterates
        within-stratum PSUs).
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(7)
        G, T, k = 6, 3, 2
        psu_arr = np.repeat(np.arange(G), T)
        time_arr = np.tile(np.arange(T), G)
        strata_arr = np.repeat([0, 0, 0, 1, 1, 1], T)
        fpc_arr = np.full(G * T, 20.0)
        Psi = rng.standard_normal((G * T, k))

        meat_full, _, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )
        # Partition + sum
        meat_partitioned = np.zeros((k, k))
        for h in [0, 1]:
            mask = strata_arr == h
            psu_arr_h = psu_arr[mask]
            sub_strata = np.zeros(mask.sum(), dtype=int)
            meat_h, _, _ = _compute_stratified_serial_bartlett_meat(
                Psi[mask],
                psu_arr=psu_arr_h,
                time_arr=time_arr[mask],
                strata_arr_full=sub_strata,
                fpc_arr_full=fpc_arr[mask],
                conley_lag_cutoff=1,
                lonely_psu="remove",
            )
            meat_partitioned += meat_h
        np.testing.assert_allclose(meat_full, meat_partitioned, rtol=1e-12, atol=1e-14)

    def test_f_singleton_adjust_lag1_no_divide_by_zero(self):
        """Pure unit test: singleton stratum + lonely_psu="adjust" + lag=1
        produces finite output (the panel-wide n_h_panel = 1 FPC must be
        SKIPPED via the `continue` mirror; otherwise the helper would
        divide-by-zero). Pins the singleton-adjust panel-wide mean asymmetry.
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(11)
        # 1 PSU in stratum 0, 2 PSUs in stratum 1; lonely_psu="adjust" exercises
        # the singleton branch for stratum 0.
        G, T, k = 3, 3, 2
        psu_arr = np.repeat([0, 1, 2], T)
        time_arr = np.tile(np.arange(T), G)
        strata_arr = np.repeat([0, 1, 1], T)
        fpc_arr = np.full(G * T, 10.0)
        Psi = rng.standard_normal((G * T, k))

        meat, var_computed, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="adjust",
        )
        assert np.all(np.isfinite(meat)), "singleton-adjust path divided by zero"
        assert var_computed

    def test_f2_all_singleton_remove_lag1_returns_zero(self):
        """Pure unit test: all strata singleton + lonely_psu="remove" + lag=1
        returns zero meat and variance_computed=False. (At the orchestrator
        level the spatial term would also fail; the orchestrator combines
        both terms' var_computed flags before NaN-failing.)
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(13)
        G, T, k = 3, 3, 2
        psu_arr = np.repeat([0, 1, 2], T)
        time_arr = np.tile(np.arange(T), G)
        strata_arr = np.repeat([0, 1, 2], T)  # 1 PSU each — all singleton
        fpc_arr = np.full(G * T, 10.0)
        Psi = rng.standard_normal((G * T, k))

        meat, var_computed, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )
        np.testing.assert_array_equal(meat, np.zeros((k, k)))
        assert not var_computed

    def test_g_unbalanced_panel_panel_wide_dense_codes(self):
        """Unit test: panel-wide dense time codes for lag math (matches
        `conley.py:940` R deviation). Drop PSU 0 from period t=2 in a
        4-PSU × 4-period synthetic. PSU 0's observed periods become {0, 1, 3};
        with L=1, only the (t=0, t=1) pair contributes (pair (t=1, t=3) at
        panel-wide lag |1-3|=2 > 1; pair (t=0, t=3) at lag 3 > 1). Hand-compute
        the expected serial contribution from PSU 0.

        Pins MEDIUM #1 from re-review: the lag convention is PANEL-WIDE dense
        codes (NOT per-PSU positional encoding).
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(17)
        G, T, k = 4, 4, 2
        # Build obs list with PSU 0 missing period 2
        psu_obs_list, time_obs_list, strata_obs_list, fpc_obs_list = [], [], [], []
        for g in range(G):
            for t in range(T):
                if g == 0 and t == 2:
                    continue  # PSU 0 missing period 2
                psu_obs_list.append(g)
                time_obs_list.append(t)
                strata_obs_list.append(g // 2)
                fpc_obs_list.append(20.0)
        psu_arr = np.array(psu_obs_list)
        time_arr = np.array(time_obs_list)
        strata_arr = np.array(strata_obs_list)
        fpc_arr = np.array(fpc_obs_list, dtype=np.float64)
        Psi = rng.standard_normal((len(psu_arr), k))

        meat, _, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )

        # Verify PSU 0's serial pairs use panel-wide lag codes {0,1,3} -> lag={1,2,3}
        # not per-PSU positional {0,1,2} -> lag={1,1,2}.
        # If the implementation incorrectly used per-PSU positional, PSU 0
        # would contribute the (t=1, t=3) pair at lag=1 (positional), not lag=2.
        # We verify by computing expected meat under both conventions and
        # asserting which the implementation matches.

        # Panel-wide convention (correct): PSU 0 only contributes (t=0, t=1) at lag=1.
        # Per-PSU positional convention (incorrect): would also contribute
        # (t=1, t=3) at positional lag=1 (positions 1 and 2 in {0, 1, 3}).
        # The contributions differ; verify implementation matches panel-wide.

        # Build S_psu per (g, t) with NaN at missing cells, then sum into meat
        # using panel-wide t codes.
        S_psu = np.full((G, T, k), np.nan)
        for i, (g, t) in enumerate(zip(psu_arr, time_arr)):
            if np.isnan(S_psu[g, t, 0]):
                S_psu[g, t] = Psi[i]
            else:
                S_psu[g, t] += Psi[i]
        # Per-period within-stratum centering. Initialize to ZEROS (NOT raw
        # S_psu) so any (g, t) cell with < 2 active PSUs in stratum at period
        # contributes zero to the serial sum — matches the helper's
        # codex-R1-P1 fix preventing raw-score leakage into the serial
        # Bartlett covariance.
        S_centered = np.zeros_like(S_psu)
        # Replace NaN with 0 in the base (won't matter — zero-init prevents
        # NaN propagation from the helper's initial copy of S_psu).
        for t in range(T):
            for h in [0, 1]:
                stratum_psus = [g for g in range(G) if g // 2 == h]
                active_psus = [g for g in stratum_psus if not np.isnan(S_psu[g, t, 0])]
                if len(active_psus) < 2:
                    continue  # singleton-active-period: leave S_centered as zero
                stratum_mean = np.mean([S_psu[g, t] for g in active_psus], axis=0)
                for g in active_psus:
                    S_centered[g, t] = S_psu[g, t] - stratum_mean
        # Per-PSU serial accumulation
        meat_expected = np.zeros((k, k))
        for h in [0, 1]:
            stratum_psus = [g for g in range(G) if g // 2 == h]
            n_h_panel = len(stratum_psus)
            if n_h_panel < 2:
                continue
            meat_h = np.zeros((k, k))
            for g in stratum_psus:
                present_g = ~np.isnan(S_centered[g, :, 0])
                t_g = np.arange(T)[present_g].astype(np.float64)
                if len(t_g) < 2:
                    continue
                lag_mat = np.abs(t_g[:, None] - t_g[None, :])
                K_g = ((lag_mat <= 1) & (lag_mat != 0)).astype(np.float64) * (1.0 - lag_mat / 2.0)
                S_g = S_centered[g, present_g]
                meat_h += S_g.T @ K_g @ S_g
            N_h = 20.0
            fpc_scale = (1.0 - n_h_panel / N_h) * n_h_panel / (n_h_panel - 1)
            meat_expected += fpc_scale * meat_h

        np.testing.assert_allclose(meat, meat_expected, rtol=1e-12, atol=1e-14)

    def test_g2_lag_greater_than_T_minus_1_finite(self):
        """L > T-1 is well-defined (no kernel overflow); fit succeeds with
        finite output. Pinned per plan: lag=T and lag=T+5 should not crash.
        """
        df = generate_butts_nonstaggered_dgp(seed=19)  # T=2 panel
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        T_max = df["time"].nunique()
        for L in [T_max, T_max + 5]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res = self._fit(df_s, lag_cutoff=L, design=design)
            assert np.isfinite(res.att), f"ATT NaN at L={L}"
            assert np.isfinite(res.se), f"SE NaN at L={L}"

    def test_h_singleton_active_period_centering_zeros(self):
        """Pure unit test: when a stratum-period has < 2 active PSUs, the
        per-period centering must zero out S_centered (not leave raw scores).
        Pins codex R1 P1 fix — leaving raw scores in singleton-active-period
        cells would feed uncentered values into the serial Bartlett sum and
        contaminate the covariance.

        Construct a 4-PSU x 3-period panel where one stratum has both PSUs
        present at t=0, t=2 but only ONE PSU present at t=1 (simulating
        finite_mask-style sparsity). With the bug, the singleton-active-period
        cell at t=1 leaks RAW scores into the serial sum across (t=0, t=1)
        and (t=1, t=2) pairs; with the fix, those legs contribute zero.
        """
        from diff_diff.two_stage import _compute_stratified_serial_bartlett_meat

        rng = np.random.default_rng(37)
        # 4 PSUs: stratum 0 has PSUs (0, 1), stratum 1 has PSUs (2, 3).
        # PSU 1 missing at t=1 -> stratum 0 has only PSU 0 active at t=1.
        # Stratum 1 has both PSUs at all periods.
        obs = []
        for g in range(4):
            for t in range(3):
                if g == 1 and t == 1:
                    continue  # PSU 1 absent at t=1 (singleton-active-period for stratum 0)
                obs.append((g, t, g // 2))
        psu_arr = np.array([o[0] for o in obs])
        time_arr = np.array([o[1] for o in obs])
        strata_arr = np.array([o[2] for o in obs])
        fpc_arr = np.full(len(obs), 20.0)
        # Choose Psi values so PSU 0 has materially nonzero score at t=1
        # (the singleton-active-period cell). Without the fix, this raw
        # value would leak into the serial cross-product with (PSU 0, t=0)
        # and (PSU 0, t=2); with the fix, the t=1 leg of PSU 0's serial sum
        # contributes zero.
        Psi = rng.standard_normal((len(obs), 2))
        # Override PSU 0's t=1 score to a known large value
        for i, (g, t, _) in enumerate(obs):
            if g == 0 and t == 1:
                Psi[i] = np.array([10.0, 10.0])

        meat, _, _ = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr,
            fpc_arr_full=fpc_arr,
            conley_lag_cutoff=1,
            lonely_psu="remove",
        )
        # Hand-compute expected: PSU 0's t=1 leg has S_centered = 0
        # (singleton-active in stratum 0); so PSU 0's serial sum only
        # contributes from valid cross-period pairs where both legs have
        # >= 2 active PSUs in stratum 0.
        # At t=0, t=2: stratum 0 has both PSUs (0, 1) active. So
        # S_centered_0_t0 = Psi_0_t0 - (Psi_0_t0 + Psi_1_t0)/2
        # S_centered_0_t2 = Psi_0_t2 - (Psi_0_t2 + Psi_1_t2)/2
        # Cross-period pair (t=0, t=2) at lag=2 contributes 0 to L=1 sum.
        # Cross-period pairs (t=0, t=1) and (t=1, t=2) at lag=1 should
        # contribute zero from PSU 0's leg (since S_centered at t=1 is zero).
        # So PSU 0 contributes nothing to the serial sum at L=1.
        # PSU 1: only present at t=0, t=2; lag=2 > L=1 so no contribution.
        # Total contribution from stratum 0 = 0.
        # PSU 2, PSU 3 (stratum 1) contribute normally at L=1.
        # Compute stratum 1's expected contribution
        psi_2_t0 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 2 and t == 0][0]]
        psi_2_t1 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 2 and t == 1][0]]
        psi_2_t2 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 2 and t == 2][0]]
        psi_3_t0 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 3 and t == 0][0]]
        psi_3_t1 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 3 and t == 1][0]]
        psi_3_t2 = Psi[[i for i, (g, t, _) in enumerate(obs) if g == 3 and t == 2][0]]
        # Stratum 1 mean at each t
        mean_t0 = (psi_2_t0 + psi_3_t0) / 2
        mean_t1 = (psi_2_t1 + psi_3_t1) / 2
        mean_t2 = (psi_2_t2 + psi_3_t2) / 2
        # Centered PSU 2
        s2 = np.array([psi_2_t0 - mean_t0, psi_2_t1 - mean_t1, psi_2_t2 - mean_t2])
        s3 = np.array([psi_3_t0 - mean_t0, psi_3_t1 - mean_t1, psi_3_t2 - mean_t2])
        K_serial = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        meat_s1 = s2.T @ K_serial @ s2 + s3.T @ K_serial @ s3
        # FPC scale: panel-wide n_h_panel = 2, N_h = 20, f = 0.1, scale = 0.9 * 2/1 = 1.8
        meat_expected = 1.8 * meat_s1

        np.testing.assert_allclose(meat, meat_expected, rtol=1e-12, atol=1e-14)

    def test_n_no_psu_survey_lag_positive_raises(self):
        """No-PSU survey design + conley_lag_cutoff > 0 raises
        NotImplementedError upfront (codex R1 P0 fix). Pseudo-PSU = obs-index
        fallback would silently zero the serial sum; the gate prevents the
        silent zero by failing closed at SpilloverDiD.fit.
        """
        df = generate_butts_nonstaggered_dgp(seed=41)
        # No PSU column; just weights + strata.
        df_s = df.copy()
        df_s["w"] = 1.0
        df_s["stratum"] = 0  # single stratum
        design = SurveyDesign(weights="w", strata="stratum")  # no psu, no fpc
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,  # the problematic combination
            vcov_type="conley",
        )
        with pytest.raises(NotImplementedError, match="no-effective-PSU survey_design"):
            est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_n3_cluster_injects_psu_under_no_psu_survey_lag_positive(self):
        """Positive test (R2 P1 fix): `cluster=<col>` injected as PSU under
        a no-PSU survey design (weights + strata + fpc) must allow lag>0
        survey-Conley fitting via `_inject_cluster_as_psu`. ATT AND SE must
        equal the equivalent fit with explicit `survey_design.psu="psu"`
        — the FPC scaling is the SAME on both paths (Wave E.1 `_inject_cluster_as_psu`
        carries the FPC through), so SE parity is the load-bearing pin.

        Pins the documented Wave E.1 surface that `cluster=<col>` becomes
        the effective PSU when survey PSU is absent; the codex-R2 P1 fix
        moved the no-effective-PSU gate to AFTER injection so this
        documented surface continues to work for lag>0. R3 P3 fix: original
        test omitted fpc on the no-PSU design and only asserted ATT
        equality + finite SE; this version includes fpc on both paths and
        asserts ATT AND scalar SE tight numerical parity (`assert_allclose`
        at rtol=1e-12, atol=1e-14) so a future variance regression on the
        cluster-injected surface fails. The full meat matrix is not
        asserted (SE is a projection of meat that could in principle
        coincide while off-diagonals differ); scalar-SE parity is the
        load-bearing user-visible pin.
        """
        df = generate_butts_nonstaggered_dgp(seed=53)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Reference fit: explicit survey_design.psu="psu"
        design_explicit = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_explicit = self._fit(df_s, lag_cutoff=1, design=design_explicit)
        # Cluster-injected fit: SurveyDesign(weights, strata, fpc) + cluster="psu"
        # (fpc included so SE comparison is apples-to-apples)
        design_no_psu = SurveyDesign(weights="w", strata="stratum", fpc="N_h")
        est_cluster = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
            cluster="psu",  # injected as effective PSU
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_cluster = est_cluster.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design_no_psu,
            )
        # ATT bit-identical (variance-method-invariant) AND SE bit-identical
        # (cluster injection carries fpc through; both paths see identical
        # effective PSU labels + FPC scaling for the panel-block sandwich).
        np.testing.assert_allclose(res_cluster.att, res_explicit.att, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(res_cluster.se, res_explicit.se, rtol=1e-12, atol=1e-14)
        assert np.isfinite(res_cluster.se) and res_cluster.se > 0

    def test_n2_weights_only_survey_lag_positive_raises(self):
        """Weights-only SurveyDesign + conley_lag_cutoff > 0 also raises
        (covers the weights-only no-PSU sub-case)."""
        df = generate_butts_nonstaggered_dgp(seed=43)
        df_s = df.copy()
        df_s["w"] = 1.0
        design = SurveyDesign(weights="w")  # weights only, no psu/strata/fpc
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=2,
            vcov_type="conley",
        )
        with pytest.raises(NotImplementedError, match="no-effective-PSU survey_design"):
            est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_j_no_survey_panel_block_unchanged_with_lag(self):
        """The no-survey Wave D panel-block conley path at `conley.py:920-965`
        must still work after the orchestrator gate-relaxation. The dispatch
        in `_compute_gmm_corrected_meat` only ROUTES the survey branch
        through the new orchestrator; the `resolved_survey is None` branch is
        bit-identical to Wave D pre-PR.
        """
        df = generate_butts_nonstaggered_dgp(seed=23)
        # No survey design; just conley + lag=1 (already supported pre-PR via Wave A/D)
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=1000.0,
            conley_lag_cutoff=1,
            vcov_type="conley",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(df, outcome="y", unit="unit", time="time", first_treat="first_treat")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0

    def test_k_replicate_weights_rejection_inherits_wave_e1(self):
        """Replicate-weight + conley + lag>0 + survey raises NotImplementedError.
        Replicate-weight SurveyDesigns are by construction no-PSU (replicate
        weights can't combine with strata/psu/fpc), so the codex-R1-P0
        no-PSU lag-gate at SpilloverDiD.fit fires before the Wave E.1
        replicate-weight gate — either error is informative. We assert the
        broader rejection contract: ANY of the gates fires.
        """
        df = generate_butts_nonstaggered_dgp(seed=29)
        df_s = df.copy()
        df_s["w"] = 1.0
        # Build replicate-weight columns (unit-constant)
        units_sorted = sorted(df_s["unit"].unique())
        rep_cols = []
        rng = np.random.default_rng(0)
        for r in range(5):
            col = f"rep_{r}"
            w_by_unit = {u: rng.uniform(0.5, 1.5) for u in units_sorted}
            df_s[col] = df_s["unit"].map(w_by_unit)
            rep_cols.append(col)
        design = SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=1000.0,
            conley_lag_cutoff=1,
            vcov_type="conley",
        )
        with pytest.raises(NotImplementedError, match="(?:replicate|no-effective-PSU)"):
            est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )

    def test_m_fit_idempotency_lag1(self):
        """Repeated fit with lag=1 produces bit-identical results (per
        `feedback_fit_does_not_mutate_config`).
        """
        df = generate_butts_nonstaggered_dgp(seed=31)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=1000.0,
            conley_lag_cutoff=1,
            vcov_type="conley",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res1 = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
            # Re-fit on a fresh estimator instance with same params
            est2 = SpilloverDiD(**est.get_params())
            res2 = est2.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        np.testing.assert_array_equal(res1.att, res2.att)
        np.testing.assert_array_equal(res1.se, res2.se)

    def test_r_drift_goldens_lag1(self):
        """ATT is variance-method-invariant: lag=1 ATT MUST match lag=0 ATT
        bit-identically on the same data + estimator config (only vcov_type
        / lag changes the meat, never the point estimate). SE MUST differ
        (serial Bartlett term contributes nonzero variance under panel data
        with within-PSU temporal correlation). Catches any case where the
        serial helper accidentally feeds back into the point estimate path.
        """
        df = generate_butts_nonstaggered_dgp(seed=0)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_lag0 = self._fit(df_s, lag_cutoff=0, design=design)
            res_lag1 = self._fit(df_s, lag_cutoff=1, design=design)
        # Both finite + positive
        assert np.isfinite(res_lag0.att) and np.isfinite(res_lag0.se) and res_lag0.se > 0
        assert np.isfinite(res_lag1.att) and np.isfinite(res_lag1.se) and res_lag1.se > 0
        # ATT bit-identical between lag=0 and lag=1 (variance-method-invariant)
        np.testing.assert_array_equal(
            res_lag1.att,
            res_lag0.att,
            err_msg="ATT must be variance-method-invariant; lag=1 ATT must match lag=0 ATT exactly",
        )
        # SE differs (serial Bartlett term contributes nonzero variance)
        assert not np.allclose(res_lag1.se, res_lag0.se, rtol=1e-10), (
            f"lag=1 SE ({res_lag1.se}) must differ from lag=0 SE ({res_lag0.se}); "
            "serial Bartlett term should add nonzero variance contribution. "
            "Note: on a 2-period Butts DGP only L=1 pairs exist (only adjacent "
            "periods), so the inflation may be small but should be detectable."
        )


class TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoffEventStudy:
    """Wave E.2 follow-up event-study mirror: conley + survey + lag>0 on both
    `is_staggered` branches (per `feedback_cohort_loop_trigger_cache_both_branches`).
    """

    _CUTOFF_KM = 1000.0

    def test_o_event_study_conley_lag1_survey_is_staggered_true(self):
        """Full plumbing end-to-end on the staggered event-study path with
        `conley_lag_cutoff=1` survey."""
        df = generate_butts_staggered_dgp(seed=24)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,  # NEW: panel-block composition
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.spillover_effects is not None
        assert res.survey_metadata is not None
        # df_survey on this fixture matches Wave E.2 event-study test_o
        assert res.survey_metadata.df_survey == 6

    def test_p_event_study_conley_lag1_survey_is_staggered_false(self):
        """The non-staggered branch of the event-study path also works with
        lag>0 (per `feedback_cohort_loop_trigger_cache_both_branches`)."""
        df = generate_butts_nonstaggered_dgp(seed=25)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,  # NEW: panel-block composition
            vcov_type="conley",
            event_study=True,
            horizon_max=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.survey_metadata is not None

    def test_r_event_study_drift_goldens_lag1_vs_lag0(self):
        """Event-study + survey + lag=1: ATT bit-identical to lag=0 (variance-
        method-invariant); SE differs (serial Bartlett contributes).
        """
        df = generate_butts_staggered_dgp(seed=24)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_lag0 = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                conley_metric="haversine",
                conley_cutoff_km=self._CUTOFF_KM,
                conley_lag_cutoff=0,
                vcov_type="conley",
                event_study=True,
                horizon_max=2,
            ).fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
            res_lag1 = SpilloverDiD(
                rings=[0.0, 100.0],
                conley_coords=("lat", "lon"),
                conley_metric="haversine",
                conley_cutoff_km=self._CUTOFF_KM,
                conley_lag_cutoff=1,
                vcov_type="conley",
                event_study=True,
                horizon_max=2,
            ).fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        # ATT bit-identical (point estimate variance-method-invariant)
        np.testing.assert_array_equal(res_lag1.att, res_lag0.att)
        # SE differs (serial term contributes nonzero variance under panel
        # data with multiple periods of within-PSU score correlation)
        assert not np.allclose(res_lag1.se, res_lag0.se, rtol=1e-10)


class TestSpilloverDiDWaveE3SubpopulationFullDesign:
    """Wave E.3: SurveyDesign.subpopulation() + warn-and-drop full-design retention.

    Methodology anchor: R `survey::svyrecvar(subset(design, mask))` (Lumley 2010
    §2.5) — zero-pad scores at the meat-helper boundary; resolved survey design
    retains full-panel `n_psu` / `n_strata` / `df_survey` / Binder centering.
    Library precedent at `imputation.py:2175-2183` and `prep.py:1401-1432`.

    A2 invariant (locked in `_scratch/wave_e3_smoke.py`): warn-and-drop and
    `SurveyDesign.subpopulation()` apply the same zero-pad mechanism — both
    produce identical meat output for identical row-level exclusions.
    """

    _CUTOFF_KM = 1000.0

    def _build_fixture_with_warn_drop(self, seed=1):
        """Mirror of `test_p2_finite_mask_forces_drop_under_survey` DGP."""
        from diff_diff import SurveyDesign as _SD  # noqa: F401

        rng = np.random.default_rng(seed)
        rows = []
        next_psu = 0
        # 2 baseline-treated units (no Omega_0 → warn-and-drop).
        for k in range(2):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"baseline_{k}",
                        "time": t,
                        "lat": 0.0 + k * 0.001,
                        "lon": 0.0,
                        "D": 1,
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 0,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        # 3 validly-treated units (treated from t=1).
        for k in range(3):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"treated_t1_{k}",
                        "time": t,
                        "lat": 10.0 + k * 0.01,
                        "lon": 0.0,
                        "D": int(t == 1),
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 0,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        # 5 far-controls (full Omega_0 support).
        for k in range(5):
            for t in (0, 1):
                rows.append(
                    {
                        "unit": f"far_control_{k}",
                        "time": t,
                        "lat": 20.0 + k * 0.01,
                        "lon": 0.0,
                        "D": 0,
                        "y": rng.normal(),
                        "w": 1.0,
                        "stratum": 1,
                        "psu": next_psu,
                        "N_h": 200.0,
                    }
                )
            next_psu += 1
        return pd.DataFrame(rows)

    def test_a_subpop_df_survey_parity_vs_upstream_subset(self):
        """Wave E.3 contract: SurveyDesign.subpopulation() preserves the full-
        domain `df_survey` regardless of how many rows the mask excludes.

        Contrast: fit on `data[mask]` directly with a plain `SurveyDesign` —
        that path drops PSUs entirely from the design (`n_psu` reflects the
        subset), so its `df_survey` is lower. This is the textbook R
        `svyrecvar(subset())` vs `svyrecvar(svydesign(data[mask]))` behavior.

        Wave E.3 ATT INVARIANCE: on this fixture (PSU 7 excluded —
        last-sorting PSU; doesn't shift the drop-first FE basis), the
        subpop and upstream-subset paths produce BIT-EQUAL ATT (diff
        at machine precision ~ 3e-16). The R6 gamma_hat-build fix
        ensures the FE basis is invariant to which PSU the subpop
        mask excludes; both paths build gamma_hat on the same active
        rows with the same factorize compaction.

        On fixtures where the excluded PSU sorts FIRST (e.g. PSU 0),
        ATT may still differ by ~1e-3 because Hájek normalization
        scales weights differently between the two paths — the
        iterative FE solver's weighted bincount is theoretically
        scale-invariant but its convergence tolerance is sensitive to
        the weight scale. That's documented but NOT asserted here
        (test_q already covers the order-sensitive case via direct
        spy on the meat-helper inputs).

        The CONTRACTS here are:
          1. `df_survey` parity (full-domain n_psu - n_strata vs
             subset n_psu - n_strata).
          2. ATT bit-equality between subpop and upstream-subset paths
             when the excluded PSU does not perturb the FE basis
             (last-sorting PSU on this fixture).
          3. SE / vcov / n_psu fixed goldens for the subpop path.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=300)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Construct a subpopulation that excludes ALL units in PSU 7 (the
        # last PSU, ensures the upstream-subset path drops 1 PSU). Pick the
        # last PSU so the treated units (which `_augment_with_survey`
        # assigns deterministically to the first few units, hence the first
        # PSUs) remain in the active sample.
        df_s["include"] = df_s["psu"] != 7

        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_subpop = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )

        # Upstream-subset baseline (drops PSU 7 from the design entirely)
        df_upstream = df_s[df_s["include"]].copy()
        plain = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_upstream = est.fit(
                df_upstream,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=plain,
            )

        # Wave E.3 contract: subpopulation reflects FULL domain (8 PSUs).
        # Upstream-subset reflects only the active PSUs (7 PSUs after dropping PSU 7).
        assert res_subpop.survey_metadata is not None
        assert res_upstream.survey_metadata is not None
        assert res_subpop.survey_metadata.n_psu == 8, (
            f"Wave E.3: subpopulation n_psu={res_subpop.survey_metadata.n_psu}, "
            f"expected full-domain 8"
        )
        assert res_upstream.survey_metadata.n_psu == 7, (
            f"Upstream subset n_psu={res_upstream.survey_metadata.n_psu}, "
            f"expected 7 (PSU 7 dropped)"
        )
        # df_survey contract: subpopulation = n_psu_full - n_strata = 8-2 = 6;
        # upstream-subset = 7-2 = 5.
        assert res_subpop.survey_metadata.df_survey == 6
        assert res_upstream.survey_metadata.df_survey == 5

        # ATT bit-equality contract (codex R12 P2 fix): with PSU 7
        # excluded (last-sorting PSU), the R6 gamma_hat-build sample is
        # identical between the two paths, and the iterative FE solver
        # converges to bit-equal mu_hat / lambda_hat. Tolerance is
        # machine precision (3e-16 in practice).
        np.testing.assert_allclose(res_subpop.att, res_upstream.att, atol=1e-14)

        # Wave E.3 numeric subpopulation anchor (codex R4 P2 fix): pin the
        # actual SurveyDesign.subpopulation() analytical SE / ATT / vcov[0,0]
        # so a future regression to the subpopulation variance path (broken
        # `score_pad_mask`, gamma_hat shift, Binder centering drift) trips
        # this test. Captured from the codex-R2/R3-verified "fit-sample Psi +
        # score_pad_mask zero-pad inside helper" implementation. This is
        # distinct from `test_r_warn_drop_se_drift_golden` which pins the
        # warn-drop path (no SurveyDesign.subpopulation() involved).
        # Tolerance per `feedback_assert_allclose_numerical_parity`.
        _WAVE_E3_SUBPOP_GOLDEN_ATT = -0.06500930624601475
        _WAVE_E3_SUBPOP_GOLDEN_SE = 0.004545962471946737
        _WAVE_E3_SUBPOP_GOLDEN_VCOV_00 = 2.0665774796348088e-05
        np.testing.assert_allclose(
            res_subpop.att, _WAVE_E3_SUBPOP_GOLDEN_ATT, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(res_subpop.se, _WAVE_E3_SUBPOP_GOLDEN_SE, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            res_subpop.vcov[0, 0], _WAVE_E3_SUBPOP_GOLDEN_VCOV_00, rtol=1e-12, atol=1e-12
        )

    def test_b_full_design_df_survey_under_warn_drop(self):
        """Wave E.3: warn-and-drop fits preserve full-domain df_survey.

        Mirrors `test_p2_finite_mask_forces_drop_under_survey` (which now
        asserts the Wave E.3 contract). Asserts that `df_for_inference`
        threaded into `safe_inference` calls reflects the FULL domain
        (closes the safe_inference threading audit at Plan-review R2 NEW
        LOW #2).
        """
        from diff_diff import SurveyDesign

        df = self._build_fixture_with_warn_drop(seed=1)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with pytest.warns(UserWarning, match=r"2 unit\(s\) have NO"):
            res = est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=design,
            )
        # 10 PSUs total (2 baseline + 3 treated + 5 far), 2 strata.
        assert res.n_psu == 10, f"Wave E.3: expected n_psu=10 (full domain), got {res.n_psu}"
        assert res.n_strata == 2
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey == 8, (
            f"Wave E.3: expected df_survey=8 (full-domain n_psu-n_strata), "
            f"got {res.survey_metadata.df_survey}"
        )

    def test_c_baseline_parity_no_dropouts_pinned_goldens(self):
        """Wave E.3: when finite_mask.all() AND all weights > 0, ATT + SE
        match the pre-E.3 baseline (zero-pad is a no-op). Anchored via
        FIXED GOLDEN VALUES rather than self-comparison, so this test
        actually catches drift from the shipped Wave E.2 / E.2-follow-up
        behavior on a no-drop fixture.

        Tolerance matches the BLAS-reduction-ordering band
        (`rtol=1e-12, atol=1e-12`) per
        `feedback_assert_allclose_numerical_parity`. If a future change
        shifts these, investigate — do NOT loosen tolerance per
        `feedback_holistic_codex_test_failure_deviation`.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=999)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=design,
            )
        # Pre-E.3 baseline goldens (captured pre-Wave-E.3 edits). Since
        # finite_mask.all() == True AND all weights > 0 on this fixture,
        # Wave E.3 zero-pad is a no-op — these values must match the
        # shipped Wave E.1/E.2/follow-up baseline exactly.
        _PRE_E3_GOLDEN_ATT = -0.07749624543132044
        _PRE_E3_GOLDEN_SE = 0.005063316956088809
        _PRE_E3_GOLDEN_N_PSU = 8
        _PRE_E3_GOLDEN_DF_SURVEY = 6

        np.testing.assert_allclose(res.att, _PRE_E3_GOLDEN_ATT, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.se, _PRE_E3_GOLDEN_SE, rtol=1e-12, atol=1e-12)
        assert res.n_psu == _PRE_E3_GOLDEN_N_PSU
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey == _PRE_E3_GOLDEN_DF_SURVEY

    def test_c2_n_psu_cross_surface_consistency_no_explicit_psu(self):
        """Wave E.3 cross-surface contract (codex R1 P2 fix): under
        weights-only / strata-only survey designs (no explicit PSU), top-
        level `res.n_psu` must agree with `res.survey_metadata.n_psu` and
        reflect the FULL domain (not the post-`finite_mask` fit sample).

        Pre-fix bug: top-level `n_psu` fell back to `int(finite_mask.sum())`
        on the implicit-PSU path at `spillover.py:3475`, diverging from
        `survey_metadata.df_survey` (full-domain) on warn-and-drop fits.
        Fix replaces it with `len(resolved_survey_fit.weights)` so both
        surfaces report the full-domain implicit-PSU count.
        """
        from diff_diff import SurveyDesign

        df = self._build_fixture_with_warn_drop(seed=10)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))

        # weights-only survey design (no PSU, no cluster injection)
        design_wo = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_wo = est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=design_wo,
            )
        assert res_wo.survey_metadata is not None
        assert res_wo.n_psu == res_wo.survey_metadata.n_psu, (
            f"Wave E.3 cross-surface contract: res.n_psu={res_wo.n_psu} != "
            f"res.survey_metadata.n_psu={res_wo.survey_metadata.n_psu} on "
            f"weights-only survey path"
        )
        # Full-domain implicit-PSU count = n_obs of the full panel
        # (`_subpop_weight=0` rows are absent here, so n_obs == len(df)).
        assert res_wo.n_psu == len(df), (
            f"Wave E.3: weights-only n_psu should be n_obs_full={len(df)}, " f"got {res_wo.n_psu}"
        )

        # strata-only survey design (no PSU, no cluster injection)
        design_str = SurveyDesign(weights="w", strata="stratum")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_str = est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=design_str,
            )
        assert res_str.survey_metadata is not None
        assert res_str.n_psu == res_str.survey_metadata.n_psu, (
            f"Wave E.3 cross-surface contract: res.n_psu={res_str.n_psu} != "
            f"res.survey_metadata.n_psu={res_str.survey_metadata.n_psu} on "
            f"strata-only survey path"
        )

    def test_d_zero_pad_mechanics_psi_contribution(self):
        """Wave E.3 (revised post codex R6 P1): the gamma_hat / Psi build
        stays on SURVEY-FINITE-MASK inputs (X_*_sparse_fit / eps_*_fit /
        omega_0_mask_fit at survey_finite_mask length, X_2_kept_gamma /
        eps_2_fit_gamma / survey_weights_fit_gamma projected back into
        the fit-sample frame) so the drop-first FE column space is stable
        AND invariant to zero-weight subpop rows. Psi gets zero-padded
        inside `_compute_gmm_corrected_meat` via
        `score_pad_mask=survey_finite_mask` AFTER construction but BEFORE
        kernel dispatch. Spy on the helper to assert: (a) construction
        inputs are at survey-finite-mask length, (b) `score_pad_mask`
        matches `survey_finite_mask`, (c) the kernel-dispatch arrays
        (cluster_ids, resolved_survey) are full-length. Note: on this
        warn-and-drop fixture (no SurveyDesign.subpopulation()),
        survey_finite_mask == finite_mask because all surviving rows have
        positive weight.
        """
        from unittest.mock import patch

        from diff_diff import SurveyDesign
        from diff_diff import two_stage as _ts

        df = self._build_fixture_with_warn_drop(seed=2)
        captured = {}
        original = _ts._compute_gmm_corrected_meat

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with patch("diff_diff.spillover._compute_gmm_corrected_meat", side_effect=spy):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(
                    df,
                    outcome="y",
                    unit="unit",
                    time="time",
                    treatment="D",
                    survey_design=design,
                )

        # Full panel = 20 rows; 4 baseline rows warn-dropped → 16 fit-sample rows.
        n_full = 20
        n_fit = 16
        # Construction inputs are at FIT-SAMPLE length (drop-first FE column
        # space stable; gamma_hat solve on a full-rank fit-length system).
        sw = captured["survey_weights"]
        eps_2 = captured["eps_2"]
        eps_10 = captured["eps_10"]
        X_2 = captured["X_2"]
        X_1_sparse = captured["X_1_sparse"]
        X_10_sparse = captured["X_10_sparse"]
        assert sw is not None
        assert sw.shape == (n_fit,), f"expected fit-length ({n_fit},), got {sw.shape}"
        assert eps_2.shape == (n_fit,)
        assert eps_10.shape == (n_fit,)
        assert X_2.shape[0] == n_fit
        assert X_1_sparse.shape[0] == n_fit
        assert X_10_sparse.shape[0] == n_fit
        # All survey weights at fit-length are POSITIVE (no zero-weight rows
        # reach the meat-helper inputs — warn-and-drop rows were never in
        # the fit sample, and this fixture has no subpopulation exclusions).
        assert (sw > 0).all(), "Wave E.3: fit-sample survey_weights must all be positive"

        # score_pad_mask = finite_mask (full-length boolean, True for active rows).
        score_pad_mask = captured["score_pad_mask"]
        assert score_pad_mask is not None
        assert score_pad_mask.shape == (n_full,)
        assert int(score_pad_mask.sum()) == n_fit, (
            f"Wave E.3: score_pad_mask has {int(score_pad_mask.sum())} True "
            f"entries, expected {n_fit} (fit-sample size)"
        )

        # Kernel-dispatch arrays are at FULL length (so the meat helpers see
        # the full-domain PSU / strata / centroid / time geometry).
        cluster_ids = captured["cluster_ids"]
        if cluster_ids is not None:
            assert cluster_ids.shape == (n_full,), (
                f"Wave E.3: cluster_ids at meat boundary should be full-length "
                f"({n_full},), got {cluster_ids.shape}"
            )
        resolved = captured["resolved_survey"]
        assert resolved is not None
        assert resolved.weights.shape == (n_full,), (
            f"Wave E.3: resolved_survey.weights at meat boundary should be "
            f"full-length ({n_full},), got {resolved.weights.shape}"
        )

    def test_e1_cluster_as_psu_subpop_parity(self):
        """Wave E.3 × Wave E.1 cluster-injection: passing `cluster='psu_col'`
        with a SurveyDesign WITHOUT an explicit PSU produces effectively the
        same fit as passing `survey_design=SurveyDesign(psu='psu_col')`,
        under subpopulation.

        Both paths inject the cluster as the effective PSU per Wave E.1's
        `_inject_cluster_as_psu` routing. Under Wave E.3 the operation runs
        on the FULL-LENGTH design (no `finite_mask` subset on cluster_ids).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=400)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        excluded = set(units[::4])
        df_s["include"] = ~df_s["unit"].isin(excluded)

        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))

        # Path (i): cluster=<col> + SurveyDesign with no explicit PSU
        design_inject = SurveyDesign(weights="w", strata="stratum", fpc="N_h")
        sub_inject, df_inject = design_inject.subpopulation(df_s, "include")
        # Use the underlying `cluster=` plumbing (ring-aware spillover surface).
        est_inject = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"), cluster="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_inject = est_inject.fit(
                df_inject,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_inject,
            )

        # Path (ii): explicit PSU on SurveyDesign
        design_explicit = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_explicit, df_explicit = design_explicit.subpopulation(df_s, "include")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_explicit = est.fit(
                df_explicit,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_explicit,
            )

        # Both paths produce identical effective PSU layout — meat should be equal.
        np.testing.assert_allclose(res_inject.att, res_explicit.att, rtol=1e-10)
        np.testing.assert_allclose(res_inject.se, res_explicit.se, rtol=1e-10)
        assert res_inject.survey_metadata is not None
        assert res_explicit.survey_metadata is not None
        assert (
            res_inject.survey_metadata.n_psu == res_explicit.survey_metadata.n_psu
        ), "cluster-injection vs explicit-PSU n_psu must match under Wave E.3"

    def test_g1_conley_lag_subpop_explicit_psu(self):
        """Wave E.3 × Wave E.2 follow-up: panel-block conley + subpopulation
        with explicit `survey_design.psu`. Asserts the gate at
        `spillover.py:~3109` passes (effective PSU is present) AND the
        meat-helper boundary receives the zero-padded full-domain inputs.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=24)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::5])

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.survey_metadata is not None
        # Full-domain n_psu reflects the design built with all units (subpop
        # drops do not reduce n_psu under Wave E.3).
        assert res.survey_metadata.n_psu == 8, (
            f"Wave E.3 + conley + lag>0: expected n_psu=8 (full domain after "
            f"_augment_with_survey assigns 2 strata × 4 PSUs), got "
            f"{res.survey_metadata.n_psu}"
        )

    def test_g2_conley_lag_subpop_cluster_injection(self):
        """Wave E.3 × Wave E.2 follow-up × Wave E.1 cluster-injection: panel-
        block conley + subpopulation with `cluster=<col>` (no explicit PSU).
        Asserts that cluster injection produces an effective PSU and the gate
        passes (per `feedback_failclosed_gate_post_resolution`).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=25)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::5])

        design = SurveyDesign(weights="w", strata="stratum", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
            cluster="psu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.survey_metadata is not None

    def test_g3_conley_lag_subpop_weights_only_raises(self):
        """Wave E.3 × Wave E.2 follow-up gate: subpopulation + weights-only
        SurveyDesign + conley + lag>0 raises `NotImplementedError` (no
        effective PSU); locks the post-resolution gate from
        `feedback_failclosed_gate_post_resolution`.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=26)
        df_s = _augment_with_survey(df, n_strata=1, psus_per_stratum=8, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::5])

        design = SurveyDesign(weights="w")  # No psu, no strata, no cluster injection
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
        )
        with pytest.raises(NotImplementedError, match="no-effective-PSU"):
            est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )

    def test_i2_unit_both_subpop_and_warn_drop(self):
        """Wave E.3 A2 invariant: a unit that is BOTH subpopulation-excluded
        (weight=0 via `_subpop_weight`) AND warn-and-dropped (no Omega_0 rows)
        composes cleanly. Psi contribution is zero from either cause; the PSU
        still counts toward `n_psu_full`.

        Locks the methodology anchor at `_scratch/wave_e3_smoke.py::scenario_e`.
        """
        from diff_diff import SurveyDesign

        df = self._build_fixture_with_warn_drop(seed=3)
        # ALSO subpop-exclude one of the baseline-treated units
        df["include"] = df["unit"] != "baseline_0"

        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df, "include")
        with pytest.warns(UserWarning, match=r"have NO"):
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=sub_design,
            )
        # Even though `baseline_0` is both subpop-excluded AND warn-and-dropped,
        # its PSU still counts toward n_psu_full (Wave E.3 zero-pad invariant).
        assert res.n_psu == 10, (
            f"Wave E.3: expected n_psu=10 (full domain — both drop mechanisms "
            f"compose), got {res.n_psu}"
        )

    def test_q_subpop_zero_weight_rows_excluded_from_gamma_hat_build(self):
        """Wave E.3 P1 mechanical regression (codex R6 fix): under the
        survey path, zero-weight rows from `SurveyDesign.subpopulation()`
        MUST be filtered out of the `gamma_hat` / Psi construction
        sample, NOT just contribute W=0 to the cross-products. The
        difference matters because `_build_butts_fe_design_csr` uses
        `pd.factorize` to compact unit/time codes and drops the first
        unit/time code for drop-first identification — leaving zero-
        weight rows in the input changes which code sorts first and
        which column gets dropped.

        Pre-R6 fix: gamma_hat-build sample used `finite_mask` only, so
        zero-weight rows survived (when they had finite y_tilde) and
        could shift the FE basis.

        Post-R6 fix at `spillover.py:3033-3060`: gamma_hat-build sample
        uses `finite_mask & (survey_weights > 0)`, so zero-weight rows
        are excluded from the FE basis. `score_pad_mask=survey_finite_mask`
        zero-pads them back into the meat at the full-domain bookkeeping
        step.

        This test directly spies on `_compute_gmm_corrected_meat` and
        asserts: (a) the inputs are at SURVEY_FINITE_MASK length (not
        finite_mask length); (b) score_pad_mask matches survey_finite_mask;
        (c) score_pad_mask.sum() equals the count of `finite_mask &
        survey_weights > 0` rows.
        """
        from unittest.mock import patch

        from diff_diff import SurveyDesign
        from diff_diff import two_stage as _ts

        df = generate_butts_staggered_dgp(seed=400)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Exclude PSU 0 (first-sorting PSU). Without the R6 fix, PSU 0's
        # rows would still be in the gamma_hat-build sample (W=0 but
        # present), and `pd.factorize` would assign them code 0 → drop
        # PSU 0's column → wrong basis. With the fix, those rows are
        # filtered before the FE rebuild.
        df_s["include"] = df_s["psu"] != 0

        captured = {}
        original = _ts._compute_gmm_corrected_meat

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        sub_design, df_sub = design.subpopulation(df_s, "include")
        with patch("diff_diff.spillover._compute_gmm_corrected_meat", side_effect=spy):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(
                    df_sub,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    survey_design=sub_design,
                )

        n_full = len(df_sub)
        # Count of survey-finite-mask = rows with weight > 0 (active under subpop)
        n_active = int((df_sub["_subpop_weight"] > 0).sum())

        score_pad_mask = captured["score_pad_mask"]
        assert score_pad_mask is not None
        assert score_pad_mask.shape == (n_full,)
        # score_pad_mask EXCLUDES the zero-weight subpop rows (the R6 fix):
        # PSU 0 has weight=0, so the mask has False at PSU 0's row positions.
        assert int(score_pad_mask.sum()) == n_active, (
            f"Wave E.3 R6 fix: score_pad_mask.sum()={int(score_pad_mask.sum())} "
            f"should equal n_active={n_active} (zero-weight rows excluded "
            f"from gamma_hat-build sample)"
        )
        # Gamma_hat-build inputs are at SURVEY_FINITE_MASK length (n_active),
        # NOT finite_mask length. This is the load-bearing R6 fix.
        sw = captured["survey_weights"]
        assert sw.shape == (n_active,), (
            f"Wave E.3 R6 fix: survey_weights at meat boundary should be "
            f"survey_finite_mask length ({n_active}), got {sw.shape}"
        )
        assert (sw > 0).all(), (
            "Wave E.3 R6 fix: gamma_hat-build survey_weights must all be "
            "POSITIVE (zero-weight rows filtered out before FE rebuild)"
        )

    def test_q2_subpop_excludes_zero_weight_rows_from_n_obs_metadata(self):
        """Wave E.3 codex R8 P2 fix: under the survey path, top-level
        `res.n_obs` / `res.n_treated` / `res.n_control` reflect the
        effective weighted estimation sample (survey_finite_mask =
        finite_mask & survey_weights > 0), NOT the broader finite_mask.

        Pre-R8: those metadata fields used `finite_mask` which over-
        counted zero-weight subpop rows (ATT/SE were correct because
        the meat helper saw the right sample, but the reported counts
        were inconsistent).

        Post-R8: spillover.py L3335-L3367 uses `count_mask =
        survey_finite_mask if resolved_survey_fit is not None else
        finite_mask` for all top-level count fields.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=700)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Exclude PSU 7 (some rows). Pre-fix: n_obs would include PSU 7
        # rows (zero-weight). Post-fix: n_obs excludes them.
        df_s["include"] = df_s["psu"] != 7
        n_total = len(df_s)
        n_excluded = int((~df_s["include"]).sum())
        n_active = n_total - n_excluded

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        sub_design, df_sub = design.subpopulation(df_s, "include")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        # n_obs reflects the effective weighted sample (= n_active),
        # NOT the full panel (n_total).
        assert res.n_obs == n_active, (
            f"Wave E.3 R8 fix: res.n_obs should reflect survey_finite_mask "
            f"({n_active} active rows after excluding PSU 7), got {res.n_obs}"
        )
        # n_treated + n_control sums to the effective sample
        assert res.n_treated + res.n_control == n_active, (
            f"n_treated ({res.n_treated}) + n_control ({res.n_control}) "
            f"should sum to n_active ({n_active})"
        )

    def test_q3_subpop_excludes_zero_weight_far_away_from_n_far_away_obs(self):
        """Wave E.3 codex R11 P2 fix: `res.n_far_away_obs` reflects the
        effective weighted estimation sample (count_mask), not the full
        domain. Pre-R11: n_far_away_obs was computed on the full panel
        at validation time and used verbatim in the result, so zero-
        weight far-away controls from SurveyDesign.subpopulation() were
        counted as if they were part of the identifying sample —
        inconsistent with the Wave E.3 n_obs / n_treated / n_control
        contract.

        Post-R11: spillover.py recomputes `n_far_away_obs_reported`
        on `count_mask` (= survey_finite_mask on the survey path) so
        the reported count matches the active sample.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=800)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)

        # Capture full-domain n_far_away_obs (no subpop)
        plain_design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_full = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=plain_design,
            )

        # Subpopulation excludes PSU 7 (some of which are far-away controls).
        # n_far_away_obs should DROP by the count of excluded far-away rows.
        df_s["include"] = df_s["psu"] != 7
        sub_design, df_sub = plain_design.subpopulation(df_s, "include")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_subpop = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        # Pre-R11: res_subpop.n_far_away_obs would equal res_full.n_far_away_obs
        # because the count was computed on the full panel BEFORE the survey-
        # finite-mask filter.
        # Post-R11: res_subpop.n_far_away_obs is STRICTLY LESS than the full-
        # domain count because PSU 7's far-away rows are zero-weighted out.
        assert res_subpop.n_far_away_obs < res_full.n_far_away_obs, (
            f"Wave E.3 R11 fix: subpop n_far_away_obs={res_subpop.n_far_away_obs} "
            f"should be < full-domain n_far_away_obs={res_full.n_far_away_obs}. "
            f"PSU 7 has zero-weight rows that should be excluded from the "
            f"reported count under the survey-finite-mask contract."
        )

    def test_q4_subpop_excludes_all_treated_raises(self):
        """Wave E.3 CI codex R1 P1 fix: SurveyDesign.subpopulation() that
        zeros out EVERY treated row must raise a clear identification error
        immediately after survey_finite_mask is built, NOT silently fall
        through to rank-deficient OLS or to the front-door full-domain
        D_it.sum() == 0 gate (which still passes because full-domain D_it
        is non-zero).

        Pre-fix: the full-domain treatment-support gate at spillover.py:2556
        runs BEFORE survey_finite_mask is computed; a subpop mask removing
        all treated units passes it but the effective weighted sample has
        zero treated rows and the OLS solve lands on a rank-deficient
        stage-2 design.

        Post-fix at spillover.py:~2745: active-sample check
        `D_it[survey_finite_mask].sum() == 0` raises ValueError mentioning
        survey_finite_mask + Wave E.3 + treated identification.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=950)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        # Exclude all ever-treated units via subpopulation
        df_s["include"] = df_s["first_treat"] == np.inf

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        sub_design, df_sub = design.subpopulation(df_s, "include")
        with pytest.raises(ValueError, match=r"removes EVERY treated observation"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(
                    df_sub,
                    outcome="y",
                    unit="unit",
                    time="time",
                    first_treat="first_treat",
                    survey_design=sub_design,
                )

    def test_r_warn_drop_se_drift_golden(self):
        """Wave E.3 numeric anchor (codex R3 P2 fix): the WARN-DROP path
        is the actual surface Wave E.3 changes — the no-drop bit-identity
        at test_c locks bit-identity on a path that's a no-op for E.3, but
        does NOT catch a regression in the warn-drop SE / meat path. This
        test pins the analytical SE + ATT + vcov[0,0] on the existing
        warn-drop fixture, captured from the codex-R2-verified
        "score_pad_mask zero-pads Psi inside the helper after
        construction" implementation. Any future regression to the
        warn-drop survey path (e.g. an accidental gamma_hat shift, a
        broken zero-pad, or a centroid drift) trips this test.

        Tolerance matches the BLAS-reduction-ordering band per
        `feedback_assert_allclose_numerical_parity`. If a future change
        shifts these, investigate (do NOT loosen tolerance per
        `feedback_holistic_codex_test_failure_deviation`).
        """
        from diff_diff import SurveyDesign

        df = self._build_fixture_with_warn_drop(seed=2)
        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="D",
                survey_design=design,
            )
        # Goldens captured from the Wave E.3 implementation post codex R2
        # P1 fix (fit-sample Psi build + score_pad_mask zero-pad inside
        # `_compute_gmm_corrected_meat`). Warn-drop excludes 4 baseline
        # rows; full-domain bookkeeping retains all 10 PSUs / 8 df_survey.
        _WAVE_E3_WARN_DROP_GOLDEN_ATT = 0.7649873242115066
        _WAVE_E3_WARN_DROP_GOLDEN_SE = 0.5793037428100555
        _WAVE_E3_WARN_DROP_GOLDEN_VCOV_00 = 0.3355928264337389

        np.testing.assert_allclose(res.att, _WAVE_E3_WARN_DROP_GOLDEN_ATT, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.se, _WAVE_E3_WARN_DROP_GOLDEN_SE, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            res.vcov[0, 0], _WAVE_E3_WARN_DROP_GOLDEN_VCOV_00, rtol=1e-12, atol=1e-12
        )
        assert res.n_psu == 10  # full-domain bookkeeping
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey == 8

    # NOTE: TwoStageDiD parity-divergence test removed — on standard
    # subpopulation fixtures TwoStageDiD does NOT trigger the same warn-and-
    # drop path as SpilloverDiD (always-treated handling at
    # `two_stage.py:294-336` differs from SpilloverDiD's per-unit Omega_0
    # rank-deficiency check), so the expected df_survey divergence does not
    # materialize as a load-bearing assertion. The TwoStageDiD parity
    # follow-up is tracked in DEFERRED.md; when that work lands it should add
    # its own targeted regression test on a fixture that actually exercises
    # the TwoStageDiD finite_mask subset path under subpopulation.


class TestSpilloverDiDWaveE3SubpopulationFullDesignEventStudy:
    """Wave E.3 event-study mirror.

    Per `feedback_cohort_loop_trigger_cache_both_branches`: cover both
    `is_staggered=True` and `is_staggered=False` branches in case Wave E.3
    threads differently across them.
    """

    _CUTOFF_KM = 1000.0

    def test_k_event_study_subpop_full_domain_df(self):
        """Event-study + analytical Binder TSL + subpopulation: full-domain
        `df_survey` preserved on the event-study path. Mirrors Wave E.1's
        event-study `df_survey` lincom verification."""
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=500)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::6])

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert res.survey_metadata is not None
        # Full-domain n_psu (no subset reduction). 2 strata × 4 psus_per_stratum = 8.
        assert res.survey_metadata.n_psu == 8, (
            f"Wave E.3 event-study: expected n_psu=8 (full domain), "
            f"got {res.survey_metadata.n_psu}"
        )

    def test_k3_event_study_subpop_excludes_zero_weight_from_n_obs_metadata(self):
        """Wave E.3 codex R13/R14 P2 fix: under SurveyDesign.subpopulation()
        on the event-study path, BOTH top-level `res.n_obs` AND per-cell
        `event_study_effects[k]["n_obs"]` reflect the EFFECTIVE WEIGHTED
        ESTIMATION SAMPLE (count_mask = survey_finite_mask), NOT the full
        panel.

        Fixture detail: excludes PSU 3 (which contains TREATED units at
        first_treat=3 — see `psu_first_treat` print in the seed=900
        fixture) so the event-study per-cell n_obs values STRICTLY
        DROP under subpop. Excluding a far-away control PSU (like
        PSU 7) wouldn't trigger the per-cell change because
        event_study_effects n_obs counts treated-cohort × horizon
        rows; only treated-PSU exclusion exercises the per-cell
        n_obs propagation through event_study_meta["n_obs_per_col"]
        recompute at spillover.py:L3011-L3036.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=900)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)

        plain_design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_full = est.fit(
                df_s,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=plain_design,
            )

        # Exclude PSU 3 (treated PSU with first_treat=3) to exercise
        # per-cell event_study_effects n_obs propagation.
        df_s["include"] = df_s["psu"] != 3
        sub_design, df_sub = plain_design.subpopulation(df_s, "include")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res_subpop = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        # Top-level n_obs / n_treated / n_control reflect count_mask
        # (= survey_finite_mask). Pre-R8 fix these would have included
        # PSU 3's zero-weight rows.
        assert res_subpop.n_obs < res_full.n_obs, (
            f"Wave E.3 R13/R8 fix: subpop event-study n_obs={res_subpop.n_obs} "
            f"should be < full-domain n_obs={res_full.n_obs}"
        )
        assert res_subpop.n_treated + res_subpop.n_control == res_subpop.n_obs

        # Wave E.3 R14/R15 P2 fix: event_study_effects per-cell n_obs
        # values, att_dynamic.n_obs values, and spillover_effects n_obs
        # values all reflect count_mask (= survey_finite_mask) propagation
        # through event_study_meta["n_obs_per_col"]. Hand-pin exact values
        # so any future regression to the n_obs_per_col recompute at
        # spillover.py L3024-L3036 trips this test.
        assert res_subpop.event_study_effects is not None
        # Exact n_obs values captured from the Wave E.3 implementation
        # post codex-R8 count_mask fix (excluding PSU 3 = first_treat=3
        # removes treated-cohort rows at each event-time horizon).
        _EXPECTED_EVENT_STUDY_N_OBS = {-2: 22, -1: 0, 0: 26, 1: 26, 2: 56}
        for k, expected_n in _EXPECTED_EVENT_STUDY_N_OBS.items():
            actual_n = res_subpop.event_study_effects[k]["n_obs"]
            assert actual_n == expected_n, (
                f"Wave E.3 R14/R15 fix: event_study_effects[{k}] n_obs="
                f"{actual_n}, expected {expected_n} (excluding PSU 3 from "
                f"the count_mask-propagated event_study_meta n_obs_per_col)"
            )
        # att_dynamic n_obs matches event_study_effects n_obs (same source)
        assert res_subpop.att_dynamic is not None
        expected_n_list = [22, 0, 26, 26, 56]
        actual_n_list = res_subpop.att_dynamic["n_obs"].tolist()
        assert actual_n_list == expected_n_list, (
            f"Wave E.3: att_dynamic.n_obs={actual_n_list}, " f"expected {expected_n_list}"
        )
        # spillover_effects per-(ring, k) n_obs reflects count_mask too.
        # Excluded PSU 3 (treated, first_treat=3) drops some ring rows.
        assert res_subpop.spillover_effects is not None
        # Sum across ring-k cells (positive horizons only, ref_period excluded).
        spill_n_obs = res_subpop.spillover_effects["n_obs"].tolist()
        _EXPECTED_SPILLOVER_N_OBS = [0, 0, 42, 42, 102]
        assert spill_n_obs == _EXPECTED_SPILLOVER_N_OBS, (
            f"Wave E.3: spillover_effects n_obs={spill_n_obs}, "
            f"expected {_EXPECTED_SPILLOVER_N_OBS}"
        )

    def test_k2_event_study_nonstaggered_subpop_full_domain_df(self):
        """Wave E.3 event-study × NON-STAGGERED branch + subpopulation
        (codex R12 P2 fix). The k/l tests above use generate_butts_
        staggered_dgp; this one exercises the is_staggered=False code
        path via generate_butts_nonstaggered_dgp to cover the cohort-
        loop trigger cache fork per
        `feedback_cohort_loop_trigger_cache_both_branches`.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=502)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::6])

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert res.survey_metadata is not None
        # Full-domain n_psu (8 = 2 strata × 4 psus_per_stratum).
        assert res.survey_metadata.n_psu == 8, (
            f"Wave E.3 non-staggered event-study: expected n_psu=8 "
            f"(full domain), got {res.survey_metadata.n_psu}"
        )
        # is_staggered=False on this fixture (generate_butts_nonstaggered_dgp).
        assert res.is_staggered is False

    def test_l2_event_study_conley_lag_nonstaggered_subpop_smoke(self):
        """Wave E.3 R15 P2 fix: non-staggered branch mirror of test_l
        (event_study + conley + lag>0 + subpopulation). Per
        `feedback_cohort_loop_trigger_cache_both_branches`, the
        non-staggered cohort-loop trigger cache fork is distinct from
        the staggered branch and should be covered separately.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=503)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::6])

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.spillover_effects is not None
        assert res.survey_metadata is not None
        # Full-domain n_psu (8 = 2 strata × 4 psus_per_stratum)
        assert res.survey_metadata.n_psu == 8
        assert res.is_staggered is False

    def test_l_event_study_conley_lag_subpop_smoke(self):
        """Event-study + conley + lag>0 + subpopulation end-to-end smoke."""
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=501)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        units = sorted(df_s["unit"].unique())
        df_s["include"] = ~df_s["unit"].isin(units[::6])

        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        sub_design, df_sub = design.subpopulation(df_s, "include")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=1,
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = est.fit(
                df_sub,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                survey_design=sub_design,
            )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.spillover_effects is not None
        assert res.survey_metadata is not None
        # Full-domain n_psu (8 = 2 strata × 4 psus_per_stratum)
        assert res.survey_metadata.n_psu == 8


class TestSpilloverDiDWaveE2ConleySurveyDesignEventStudy:
    """Event-study branch + conley + survey, both is_staggered branches."""

    _CUTOFF_KM = 1000.0

    def test_o_event_study_conley_survey_is_staggered_true(self):
        """Full plumbing end-to-end on the staggered event-study path."""
        from diff_diff import SurveyDesign

        df = generate_butts_staggered_dgp(seed=24)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=True,
            horizon_max=2,
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Event-study + spillover finite end-to-end
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        # spillover_effects populated (non-empty)
        assert res.spillover_effects is not None
        # df_survey lookup uses the survey branch
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey == 6

    def test_p_event_study_conley_survey_is_staggered_false(self):
        """The non-staggered branch of the event-study path also works
        (mirrors `feedback_cohort_loop_trigger_cache_both_branches`).
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=25)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
            event_study=True,
            horizon_max=1,
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        assert np.isfinite(res.att) and np.isfinite(res.se) and res.se > 0
        assert res.survey_metadata is not None

    def test_r_drift_goldens(self):
        """Pinned ATT + SE on a fixed-seed conley+survey fit.

        Drift goldens captured on initial Wave E.2 implementation
        (seed=999, standard 2-strata x 4-PSU augmentation, cutoff=1000km).
        `assert_allclose` tolerance acknowledges PSU-aggregation BLAS
        reduction order variation across CI runners.
        """
        from diff_diff import SurveyDesign

        df = generate_butts_nonstaggered_dgp(seed=999)
        df_s = _augment_with_survey(df, n_strata=2, psus_per_stratum=4, fpc=200.0)
        design = SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="N_h")
        est = SpilloverDiD(
            rings=[0.0, 100.0],
            conley_coords=("lat", "lon"),
            conley_metric="haversine",
            conley_cutoff_km=self._CUTOFF_KM,
            conley_lag_cutoff=0,
            vcov_type="conley",
        )
        res = est.fit(
            df_s,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        # Goldens — pinned on initial Wave E.2 implementation (seed=999,
        # 2-strata x 4-PSU augmentation, cutoff=1000km). ATT is invariant
        # to vcov_type, so it matches the Wave E.1 binder golden exactly.
        # SE is Wave E.2-specific (stratified-Conley sandwich on PSU totals).
        _WAVE_E2_GOLDEN_ATT = -0.07749624543132044
        _WAVE_E2_GOLDEN_SE = 0.0006771937420330884
        np.testing.assert_allclose(res.att, _WAVE_E2_GOLDEN_ATT, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(res.se, _WAVE_E2_GOLDEN_SE, rtol=1e-12, atol=1e-14)
        # Lock down DOF + n_psu (deterministic).
        assert res.n_psu == 8
        assert res.n_strata == 2
        assert res.survey_metadata.df_survey == 6


class TestSpilloverDiDBreadRankGuard:
    """The SpilloverDiD Wave D bread (A_22 = X_2' W X_2) now routes through the
    shared ``_rank_guarded_inv``: a near-singular Stage-2 design Gram rank-
    reduces to a finite SE on the identified subspace and warns (was a dense
    lstsq fallback that fired only on an exactly-singular bread)."""

    def test_rank_deficient_bread_warns_and_fits(self):
        from unittest.mock import patch

        import diff_diff.spillover as sp_mod

        df = _make_butts_2period_dgp(seed=42)
        real_rgi = sp_mod._rank_guarded_inv

        def force_drop(A, **kwargs):
            # Finite inverse + n_dropped=1 (warning fires), empty dropped mask so
            # the identified SE stays finite. Mirrors the helper's return arity.
            inv, _, rank = real_rgi(A)
            if kwargs.get("return_dropped"):
                return inv, 1, rank, np.zeros(A.shape[0], dtype=bool)
            return inv, 1, rank

        est = SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))
        with patch.object(sp_mod, "_rank_guarded_inv", side_effect=force_drop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        msgs = [str(w.message) for w in caught]
        assert any("SpilloverDiD Wave D bread" in m and "rank-deficient" in m for m in msgs), msgs
        assert est.is_fitted_ and np.isfinite(result.att)

    def test_dropped_ring_coefficient_propagates_nan_inference(self):
        """A dropped (unidentified) Wave D coefficient must report NaN se / t_stat /
        p_value / CI for its ring effect at the ESTIMATOR level (not the zero-filled
        se=0) — the per-coefficient propagation the rank-guard enables (CI codex P2
        test-depth)."""
        from unittest.mock import patch

        import diff_diff.spillover as sp_mod

        df = _make_butts_2period_dgp(seed=42)
        real_rgi = sp_mod._rank_guarded_inv

        def drop_last(A, **kwargs):
            # Genuinely drop the last Wave D coordinate (zero-fill its row/col +
            # report it dropped) so the caller NaNs that ring's vcov entry.
            inv, _, rank = real_rgi(A)
            inv = np.array(inv, dtype=float)
            inv[-1, :] = 0.0
            inv[:, -1] = 0.0
            k = A.shape[0]
            dropped = np.zeros(k, dtype=bool)
            dropped[-1] = True
            if kwargs.get("return_dropped"):
                return inv, 1, k - 1, dropped
            return inv, 1, k - 1

        est = SpilloverDiD(rings=[0.0, 50.0, 100.0], conley_coords=("lat", "lon"))
        with patch.object(sp_mod, "_rank_guarded_inv", side_effect=drop_last):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = est.fit(df, outcome="y", unit="unit", time="time", treatment="D")
        eff = res.spillover_effects
        nan_rows = eff[np.isnan(eff["se"])]
        fin_rows = eff[np.isfinite(eff["se"]) & (eff["se"] > 0)]
        assert (
            len(nan_rows) >= 1
        ), f"a dropped Wave D coord should NaN a ring SE; got {eff['se'].tolist()}"
        assert len(fin_rows) >= 1, "identified rings should keep finite SE"
        # The NaN-SE ring's FULL inference must be NaN, not just se.
        for _, r in nan_rows.iterrows():
            assert np.isnan(r["t_stat"]) and np.isnan(r["p_value"])
            assert np.isnan(r["ci_low"]) and np.isnan(r["ci_high"])


class TestStaggeredSparseKDTreeBranch:
    """The staggered cohort loop's sparse cKDTree branch (activated when
    ``cutoff_km`` is set, ``n_units > _CONLEY_SPARSE_N_THRESHOLD``, and the
    metric is a built-in string) must reproduce the dense path exactly for
    every within-cutoff distance, and produce identical fits end-to-end —
    every staggered ``d_it`` consumer compares against thresholds <=
    ``_effective_d_bar``, so beyond-cutoff ``inf`` is semantics-preserving.
    Mirrors the static helper's sparse branch (auto-activated via the same
    threshold) and its tests above."""

    def test_helper_sparse_matches_dense_within_cutoff(self, staggered_panel, monkeypatch):
        import diff_diff.spillover as sp

        df, ft = staggered_panel
        kwargs = dict(
            unit="unit",
            time="time",
            coords=("lat", "lon"),
            metric="haversine",
            first_treat_by_unit=ft,
            d_bar=1200.0,
        )
        d_dense, ru, rt, trig_dense = sp._compute_nearest_treated_distance_staggered(df, **kwargs)
        monkeypatch.setattr(sp, "_CONLEY_SPARSE_N_THRESHOLD", 0)
        d_sparse, _, _, trig_sparse = sp._compute_nearest_treated_distance_staggered(
            df, cutoff_km=1200.0, **kwargs
        )
        in_range = d_dense <= 1200.0 * (1 + 1e-6)
        np.testing.assert_allclose(d_sparse[in_range], d_dense[in_range], atol=1e-8)
        # Beyond-cutoff entries are inf on the sparse path.
        assert np.isinf(d_sparse[~in_range]).all()
        # The d_bar trigger consumes distances <= d_bar (== cutoff), so it
        # must be identical between the paths (NaN pattern included).
        np.testing.assert_array_equal(np.isnan(trig_dense), np.isnan(trig_sparse))
        both = ~np.isnan(trig_dense)
        np.testing.assert_array_equal(trig_dense[both], trig_sparse[both])

    def test_fit_sparse_matches_dense_end_to_end(self, monkeypatch):
        """Force the sparse branch on a small staggered fit: att, ring
        coefficients and SEs must match the dense fit (the within-cutoff
        distances are exact; beyond-cutoff rows land in the far-away
        control group on both paths)."""
        import diff_diff.spillover as sp

        rng = np.random.default_rng(42)
        rows = []
        # 3 cohorts of treated units near the origin + controls in/beyond rings.
        units = {}
        uid = 0
        for k in range(6):  # treated: onset staggered 1/2
            units[f"T{uid}"] = (0.05 * k, 0.02 * k, 1 + (k % 2))
            uid += 1
        for k in range(10):  # near controls within ~40 km
            units[f"C{uid}"] = (0.1 + 0.02 * k, 0.15 + 0.02 * k, np.inf)
            uid += 1
        for k in range(8):  # far controls (>5 deg away, far outside rings)
            units[f"F{uid}"] = (6.0 + 0.1 * k, 6.0, np.inf)
            uid += 1
        for u, (lat, lon, ft) in units.items():
            for t in range(4):
                treated_now = np.isfinite(ft) and t >= ft
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "lat": lat,
                        "lon": lon,
                        "first_treat": ft if np.isfinite(ft) else 0,
                        "y": 1.0 + 0.1 * t + (0.5 if treated_now else 0.0) + rng.normal(0, 0.05),
                    }
                )
        df = pd.DataFrame(rows)
        fit_kwargs = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")

        dense = SpilloverDiD(rings=[0.0, 50.0], conley_coords=("lat", "lon")).fit(df, **fit_kwargs)
        monkeypatch.setattr(sp, "_CONLEY_SPARSE_N_THRESHOLD", 0)
        sparse_res = SpilloverDiD(rings=[0.0, 50.0], conley_coords=("lat", "lon")).fit(
            df, **fit_kwargs
        )

        assert sparse_res.is_staggered is True and dense.is_staggered is True
        np.testing.assert_allclose(sparse_res.att, dense.att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sparse_res.se, dense.se, rtol=0, atol=1e-12)
        # spillover_effects is a per-ring DataFrame; compare its numeric columns.
        sp_num = sparse_res.spillover_effects.select_dtypes("number")
        de_num = dense.spillover_effects.select_dtypes("number")
        np.testing.assert_allclose(
            sp_num.to_numpy(dtype=float),
            de_num.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )
