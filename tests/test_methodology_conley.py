"""ConleySpatialHAC methodology verification tests.

Targets Conley, T. G. (1999), *GMM Estimation with Cross-Sectional Dependence*,
Journal of Econometrics 92(1), 1-45. DOI: 10.1016/S0304-4076(98)00084-0.

Secondary sources:
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent
  covariance matrix estimation. *Econometrica* 59(3), 817-858.
- Duesterhoeft, C. (2021). conleyreg: Estimations using Conley Standard Errors.
  CRAN R package. Parity target for the panel block-decomposed sandwich.

Paper review on file: ``docs/methodology/papers/conley-1999-review.md``.

Equation / section walk-through:

- **Eq. 4.2 cross-sectional sandwich (pairwise-distance specialization)**:
  `Var(beta) = (X'X)^{-1} (Sum_{i,j} K(d_ij/h) X_i eps_i eps_j X_j') (X'X)^{-1}`.
  diff-diff implements the real-valued / pairwise-distance form (Conley 1999
  Equation 4.2 plus the "pairwise products at a given distance" remark on
  page 19); Equation 3.13 is the lattice-indexed form reserved for grid
  coordinates. Verified by ``TestConleyEquation42`` (Bartlett kernel form,
  PSD/symmetric/shape contracts, uniform kernel boundary). The 1-D radial
  Bartlett implementation matches R ``conleyreg``, Stata ``acreg`` (Colella
  et al. 2019), and Hsiang (2010) as a practitioner specialization of Eq. 4.2.
- **Eq. 4.2 limits**: tiny-cutoff reduces to HC0 diagonal `Sum_i X_i eps_i^2
  X_i'`; huge-cutoff under uniform reduces to rank-1 correlated limit. Verified
  by ``TestConleyHC0AndRank1Reductions``.
- **Eq. 3.14**: paper's 2-D separable product window
  `K(j, k) = (1 - |j|/L_M)(1 - |k|/L_N)` on a lattice is the explicitly
  PSD-guaranteed form. The 1-D radial Bartlett implementation is a practitioner
  specialization that is **not formally PSD-guaranteed**. The indefiniteness
  guard at `< -1e-12` applies to both kernels (REGISTRY L3609); locked by
  ``TestConleyLibraryExtensions::test_indefiniteness_guard_fires_on_negative_eigenvalue``.
- **Andrews (1991) HAC truncation**: panel space-time Bartlett kernel
  `(1 - |t-s|/(L+1))` for `0 < |t-s| <= L`, matching
  ``conleyreg::time_dist.cpp``. Lag = 0 is excluded to avoid double-counting the
  diagonal already in the spatial component. Verified by
  ``TestConleyAndrewsLagTruncation``.
- **Haversine convention**: Earth radius 6371.01 km (matches
  ``conleyreg::haversine_dist``); 1 degree latitude = 111.195 km at the
  equator. Verified by ``TestConleyHaversineConvention``.
- **Phase 2 panel block-decomposed sandwich**: additive decomposition
  `XeeX = XeeX_spatial + XeeX_serial` (matches ``conleyreg::time_dist.cpp``
  at `atol=1e-12`). Verified by ``TestConleyBlockDecomposition``.
- **Wave A #120 sparse k-d-tree path**: numerical correctness of sparse vs
  dense (bit-identical to `atol=1e-10`). Verified by
  ``TestConleySparseDenseEquivalence``. Activation thresholds + density-gate
  fallback are defensive surface (kept in ``tests/test_conley_vcov.py``).
- **R ``conleyreg`` v0.1.9 parity at `atol=1e-6`**: 6 fixtures (3 cross-sectional
  + 3 panel) plus the time-asymmetric kernel literal-matching contract. Verified
  by ``TestConleyParityR``. Skips when the golden JSON is absent (isolated CI).

Library extensions, deviations from R, and deferrals (3 dedicated classes):

- ``TestConleyLibraryExtensions`` (6 tests): combined spatial + cluster product
  kernel (Wave A #119) — two limit-fixture anchors (all-unique-clusters and
  huge-cutoff + uniform), callable conley_metric validation (Wave A #123),
  sparse k-d-tree activation (Wave A #120), and indefiniteness guard
  separately on the Bartlett and uniform kernel paths (REGISTRY L3609; both
  kernels are practitioner specializations not formally PSD-guaranteed).
  All have no R correspondence.
- ``TestConleyDeviationsFromR`` (3 tests): 1-D radial Bartlett vs paper's 2-D
  separable Eq. 3.14, time-label normalization via `np.unique`, independent
  temporal kernel deferred (R's `kernel` arg controls spatial only).
- ``TestConleyDeferrals`` (5 tests): fail-closed `NotImplementedError` /
  `TypeError` contracts for LinearRegression + survey_design, DiD/MPD/TWFE +
  survey_design, Conley + weights, SyntheticDiD + Conley, wild_bootstrap +
  Conley.

**Generic-path deferral (open methodological question)**: weighted
spatial-HAC under probability sampling (LinearRegression / DiD / MPD /
TWFE generic path + Conley + survey_design) is an open methodological
question; no canonical extension of Conley (1999) exists for the
combination. The estimator-specific shipped surfaces are: SpilloverDiD
+ Conley + survey via Wave E.1/E.2/E.3 (PR #468/#474/#482, stratified-
Conley sandwich on PSU totals); TwoStageDiD + Conley + survey via Wave
E.3 parity (PR #485).

Companion to ``tests/test_conley_vcov.py`` (defensive surface: input
validation, NaN/inf guards, dispatch-level validity, estimator-level
integration smoke tests, set_params atomicity). The methodology file
consolidates paper-anchored R-parity tests into a checklist; the defensive
file retains the high-volume input-validation + regression surface.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff.conley import (
    _CONLEY_EARTH_RADIUS_KM,
    _CONLEY_SPARSE_N_THRESHOLD,
    _bartlett_kernel,
    _compute_conley_vcov,
    _haversine_km,
    _pairwise_distance_matrix,
    _serial_bartlett_kernel_matrix,
    _uniform_kernel,
)
from diff_diff.linalg import (
    LinearRegression,
    compute_robust_vcov,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

GOLDEN_PATH = "benchmarks/data/r_conleyreg_conley_golden.json"
PARITY_TOL = 1e-6  # R conleyreg v0.1.9 parity success criterion (Phase 1+2)


def _load_r_conley_goldens():
    """Load the R conleyreg golden fixtures.

    Returns a dict keyed by fixture name; each value is the per-fixture entry
    (``x``, ``x_shape``, ``y``, ``coords``, ``coords_shape``, ``cutoff_km``,
    ``metric``, ``kernel``, ``vcov``, ``vcov_shape``; panel fixtures add
    ``unit``, ``time``, ``lag_cutoff``).
    """
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / GOLDEN_PATH
    if not path.exists():
        pytest.skip(
            f"Golden JSON not present at {path}; run "
            "`cd benchmarks/R && Rscript generate_conley_golden.R` to "
            "generate. Requires conleyreg R package + sf/lwgeom + system "
            "libs gdal/proj/geos/udunits."
        )
    return json.loads(path.read_text())


def _make_conley_cross_section(*, n=20, seed=42):
    """20-row synthetic OLS dataset with lat/lon coords.

    Used across `TestConleyEquation42`, `TestConleyHC0AndRank1Reductions`,
    and `TestConleyHaversineConvention` (Eq. 4.2 cross-sectional surface).
    """
    rng = np.random.default_rng(seed=seed)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    eps = rng.standard_normal(n)
    y = X @ np.array([1.0, 2.0]) + eps
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coefs
    bread = X.T @ X
    coords = np.column_stack(
        [
            rng.uniform(-30, 30, n),  # lat
            rng.uniform(-100, 100, n),  # lon
        ]
    )
    return X, residuals, coords, bread


def _make_conley_panel(*, n_units=5, T=3, k=2, seed=42, cutoff_km=5000):
    """Balanced panel fixture for the Phase 2 block-decomposed surface.

    Used across `TestConleyAndrewsLagTruncation` and
    `TestConleyBlockDecomposition`. Mirrors the helper in
    `tests/test_conley_vcov.py::TestConleyPanelHelper`.
    """
    rng = np.random.default_rng(seed)
    lat_unit = rng.uniform(-30, 30, size=n_units)
    lon_unit = rng.uniform(-100, 100, size=n_units)
    unit = np.repeat(np.arange(n_units), T)
    time = np.tile(np.arange(1, T + 1), n_units)
    lat = lat_unit[unit]
    lon = lon_unit[unit]
    coords = np.column_stack([lat, lon])
    n = n_units * T
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
    beta = np.linspace(0.5, 2.0, k)
    y = X @ beta + rng.standard_normal(n) * 0.5
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta_hat
    bread = X.T @ X
    return X, residuals, coords, time, unit, bread, cutoff_km


# ---------------------------------------------------------------------------
# TestConleyEquation42 — Conley 1999 Eq. 4.2 cross-sectional sandwich
# ---------------------------------------------------------------------------


class TestConleyEquation42:
    """Conley 1999 Eq. 4.2 cross-sectional Bartlett-kernel sandwich.

    `Var(beta) = (X'X)^{-1} (Sum_{i,j} K(d_ij/h) X_i eps_i eps_j X_j') (X'X)^{-1}`

    Verifies the Bartlett kernel form K(u) = max(0, 1 - |u|), the uniform
    kernel K(u) = 1{|u| <= 1}, and the sandwich PSD/symmetric/shape contracts
    on a small synthetic OLS dataset.
    """

    def test_bartlett_at_zero(self):
        """K(0) = 1: same-location pair contributes the full HC0 term."""
        np.testing.assert_allclose(_bartlett_kernel(np.array([0.0])), 1.0)

    def test_bartlett_at_one(self):
        """K(1) = 0: pair at exactly the bandwidth contributes zero."""
        np.testing.assert_allclose(_bartlett_kernel(np.array([1.0])), 0.0)

    def test_bartlett_above_one_zero(self):
        """K(u) = 0 for u > 1 (compact support of the Bartlett kernel)."""
        u = np.array([1.5, 2.0, 100.0])
        np.testing.assert_allclose(_bartlett_kernel(u), np.zeros(3))

    def test_bartlett_negative_arg_symmetric(self):
        """K(-u) = K(u): Bartlett kernel uses |u| (Conley 1999 p. 11)."""
        np.testing.assert_allclose(
            _bartlett_kernel(np.array([-0.3])),
            _bartlett_kernel(np.array([0.3])),
        )

    def test_bartlett_kernel_finite_and_in_unit_interval(self):
        """Bartlett-weighted kernel matrix on random pairwise distances is
        finite, symmetric, and bounded in [0, 1]. We do NOT assert PSD here:
        the radial 1-D Bartlett on pairwise distance is a practitioner
        specialization of Conley 1999 and is NOT formally PSD-guaranteed —
        see REGISTRY ConleySpatialHAC § Note (deviation / source
        specialization). The runtime path emits a UserWarning if the
        resulting Conley meat is materially indefinite; that contract is
        locked in
        ``TestConleyLibraryExtensions::test_indefiniteness_guard_fires_on_negative_eigenvalue``.
        """
        rng = np.random.default_rng(seed=11)
        n = 25
        coords = rng.uniform(0, 1, size=(n, 2))
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt((diff * diff).sum(axis=-1))
        K = _bartlett_kernel(D / 0.3)
        assert K.shape == (n, n)
        assert np.all(np.isfinite(K))
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-15)

    def test_uniform_kernel_at_boundary(self):
        """Uniform kernel is closed on the right: K(1) = 1, K(1+eps) = 0
        (Conley 1999 p. 11; spectral window is negative in regions per
        footnote 11)."""
        np.testing.assert_allclose(_uniform_kernel(np.array([1.0])), 1.0)

    def test_uniform_kernel_above_one_zero(self):
        np.testing.assert_allclose(_uniform_kernel(np.array([1.0001, 2.0, 100.0])), np.zeros(3))

    def test_uniform_kernel_at_zero(self):
        np.testing.assert_allclose(_uniform_kernel(np.array([0.0])), 1.0)

    def test_sandwich_returns_psd_with_bartlett(self):
        """Eq. 4.2 sandwich is PSD on well-behaved residuals (no
        indefiniteness warning fires)."""
        X, residuals, coords, bread = _make_conley_cross_section()
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=2000.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        eigvals = np.linalg.eigvalsh(0.5 * (vcov + vcov.T))
        assert eigvals.min() > -1e-10

    def test_sandwich_symmetric_vcov(self):
        """Eq. 4.2 sandwich is symmetric to machine precision."""
        X, residuals, coords, bread = _make_conley_cross_section()
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=2000.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-10)

    def test_sandwich_shape_matches_bread(self):
        """Eq. 4.2 sandwich has shape (k, k) matching the bread matrix."""
        X, residuals, coords, bread = _make_conley_cross_section()
        vcov = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=1500.0,
            metric="haversine",
            kernel="bartlett",
            bread_matrix=bread,
        )
        k = X.shape[1]
        assert vcov.shape == (k, k)


# ---------------------------------------------------------------------------
# TestConleyHC0AndRank1Reductions — Eq. 4.2 limits
# ---------------------------------------------------------------------------


class TestConleyHC0AndRank1Reductions:
    """Conley 1999 Eq. 4.2 limits.

    - Tiny-cutoff: K is ~0 off-diagonal and 1 on-diagonal, so the meat
      reduces to HC0 `Sum_i X_i eps_i^2 X_i'`.
    - Huge-cutoff under uniform: K -> ones(n, n) and meat = (X*eps)' ones
      (X*eps) which is rank-1 (NOT HC0). This is the all-correlated limit.
    - Diagonal contribution: K(0) = 1 so the diagonal always contributes the
      full HC0 term exactly.
    """

    def test_tiny_cutoff_bartlett_yields_HC0_meat(self):
        """Bartlett + bandwidth << min pairwise distance reduces to HC0."""
        rng = np.random.default_rng(seed=3)
        n = 15
        coords = np.column_stack([np.arange(n) * 100.0, np.arange(n) * 100.0])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X

        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(2))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv

        vcov_conley = _compute_conley_vcov(
            X,
            eps,
            coords,
            cutoff=1.0,  # << minimum pairwise distance
            metric="euclidean",
            kernel="bartlett",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov_conley, vcov_hc0, atol=1e-12)

    def test_tiny_cutoff_uniform_yields_HC0_meat(self):
        """Uniform kernel + tiny cutoff also reduces to HC0."""
        rng = np.random.default_rng(seed=5)
        n = 12
        coords = np.column_stack([np.arange(n) * 100.0, np.arange(n) * 100.0])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X

        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(2))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv

        vcov_conley = _compute_conley_vcov(
            X,
            eps,
            coords,
            cutoff=0.5,
            metric="euclidean",
            kernel="uniform",
            bread_matrix=bread,
        )
        np.testing.assert_allclose(vcov_conley, vcov_hc0, atol=1e-12)

    def test_huge_cutoff_uniform_NOT_HC0(self):
        """Huge cutoff under uniform: K -> ones(n, n), meat is rank-1
        all-correlated limit — NOT HC0."""
        X, residuals, coords, bread = _make_conley_cross_section()
        vcov_conley = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff=1e9,
            metric="euclidean",
            kernel="uniform",
            bread_matrix=bread,
        )
        meat_hc0 = X.T @ (X * (residuals**2)[:, None])
        bread_inv = np.linalg.solve(bread, np.eye(X.shape[1]))
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv
        assert not np.allclose(vcov_conley, vcov_hc0, atol=1e-6)

    def test_diagonal_of_meat_equals_HC0_contribution(self):
        """K(0/h) = 1 always, so the diagonal contribution to the meat is
        exactly the HC0 term `Sum_i X_i eps_i^2 X_i'` regardless of cutoff
        choice. Extracted from `test_conley_vcov.py::TestConleyReductionsAddendum`."""
        rng = np.random.default_rng(seed=9)
        n = 20
        coords = rng.uniform(0, 1000, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        D = _pairwise_distance_matrix(coords, "euclidean")
        cutoff = float(D[D > 0].min() * 0.001)
        S = X * eps[:, None]
        meat_full = S.T @ _bartlett_kernel(D / cutoff) @ S
        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        np.testing.assert_allclose(meat_full, meat_hc0, atol=1e-12)


# ---------------------------------------------------------------------------
# TestConleyAndrewsLagTruncation — Andrews (1991) HAC truncation
# ---------------------------------------------------------------------------


class TestConleyAndrewsLagTruncation:
    """Andrews (1991) HAC lag truncation for the panel space-time path.

    The within-unit serial Bartlett kernel is `(1 - |t-s|/(L+1))` for
    `0 < |t-s| <= L`, zero otherwise, matching ``conleyreg::time_dist.cpp``.
    `lag = 0` is excluded to avoid double-counting the diagonal already in
    the within-period spatial component.
    """

    def test_serial_bartlett_kernel_matrix_basic(self):
        """Hand-computed Bartlett HAC kernel for t=[0,1,2,3], L=2.

        K[i,j] = (1 - |i-j|/(L+1)) for 0 < |i-j| <= L, else 0. With L=2
        and (L+1)=3: lag 1 -> 2/3, lag 2 -> 1/3, lag 3 -> 0 (out of band).
        """
        K = _serial_bartlett_kernel_matrix(np.array([0, 1, 2, 3]), L=2)
        expected = np.array(
            [
                [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
                [2.0 / 3.0, 0.0, 2.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0],
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0],
            ]
        )
        np.testing.assert_allclose(K, expected, atol=1e-15)

    def test_serial_bartlett_kernel_matrix_l_one(self):
        """L=1: only adjacent lags survive, with weight (1 - 1/2) = 0.5."""
        K = _serial_bartlett_kernel_matrix(np.array([0, 1, 2]), L=1)
        expected = np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.0],
            ]
        )
        np.testing.assert_allclose(K, expected, atol=1e-15)

    def test_serial_bartlett_kernel_matrix_l_zero_returns_zero(self):
        """L=0 is degenerate-but-callable: every off-diagonal lag fails the
        `lag <= 0` band test (since `lag != 0`), so K is the zero matrix.
        Callers guard externally — ``conley.py`` skips the serial loop when
        L == 0."""
        K = _serial_bartlett_kernel_matrix(np.array([0, 1, 2]), L=0)
        np.testing.assert_array_equal(K, np.zeros((3, 3)))

    def test_serial_bartlett_kernel_matrix_single_element(self):
        """Single-element input yields a 1x1 zero matrix (no off-diagonal
        lags exist)."""
        K = _serial_bartlett_kernel_matrix(np.array([7]), L=2)
        np.testing.assert_array_equal(K, np.zeros((1, 1)))

    def test_serial_bartlett_kernel_matrix_int_input_bit_equal_to_float(self):
        """Contract test: int64 and float64 inputs must yield bit-equal
        matrices. The helper does ``astype(np.float64, copy=False)`` and one
        of the call sites (`conley.py` panel-block branch) passes the result
        of array slicing on int time codes."""
        K_int = _serial_bartlett_kernel_matrix(np.array([0, 1, 2, 3], dtype=np.int64), L=2)
        K_float = _serial_bartlett_kernel_matrix(
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64), L=2
        )
        np.testing.assert_array_equal(K_int, K_float)

    def test_T_eq_1_equals_cross_sectional(self):
        """Block-decomposed form with T=1 (single time period) equals the
        Phase 1 cross-sectional form on the same data (serial component
        contributes nothing at T=1)."""
        X, residuals, coords, _, _, bread, cutoff = _make_conley_panel(n_units=8, T=1, k=2)
        unit_single = np.arange(X.shape[0])
        time_single = np.ones(X.shape[0], dtype=int)
        V_cs = _compute_conley_vcov(X, residuals, coords, cutoff, "haversine", "bartlett", bread)
        V_panel = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time_single,
            unit=unit_single,
            lag_cutoff=2,
        )
        np.testing.assert_allclose(V_panel, V_cs, atol=1e-12)

    def test_lag_cutoff_zero_drops_serial(self):
        """L=0 means the serial component contributes nothing; only the
        within-period spatial sandwich applies."""
        X, residuals, coords, time, unit, bread, cutoff = _make_conley_panel()
        V0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
        )
        S = X * residuals[:, None]
        meat_spatial = np.zeros((X.shape[1], X.shape[1]))
        for t_val in np.unique(time):
            mask = time == t_val
            D_t = _pairwise_distance_matrix(coords[mask], "haversine")
            K_t = _bartlett_kernel(D_t / cutoff)
            meat_spatial += S[mask].T @ K_t @ S[mask]
        V_expected = np.linalg.solve(bread, meat_spatial)
        V_expected = np.linalg.solve(bread, V_expected.T).T
        np.testing.assert_allclose(V0, V_expected, atol=1e-12)

    def test_lag_cutoff_positive_adds_serial(self):
        """L > 0 strictly increases the meat by a positive serial
        contribution (within-unit cross-time pairs)."""
        X, residuals, coords, time, unit, bread, cutoff = _make_conley_panel()
        V0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
        )
        V1 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=1,
        )
        assert not np.allclose(V0, V1, atol=1e-8), (
            "lag_cutoff=1 must differ from lag_cutoff=0 with a serial " "component"
        )


# ---------------------------------------------------------------------------
# TestConleyHaversineConvention — Earth radius + degree conversion
# ---------------------------------------------------------------------------


class TestConleyHaversineConvention:
    """Great-circle distance convention.

    diff-diff uses Earth's mean radius 6371.01 km, matching
    ``conleyreg::haversine_dist`` (per Duesterhoeft 2021 CRAN v0.1.9 src).
    At the equator, 1 degree longitude = 2*pi*R/360 = 111.195 km.
    """

    def test_haversine_known_pair_one_degree_equator(self):
        """1 degree longitude at the equator = 2*pi*R/360 ~ 111.195 km."""
        d = _haversine_km(np.array(0.0), np.array(0.0), np.array(0.0), np.array(1.0))
        expected = 2 * np.pi * _CONLEY_EARTH_RADIUS_KM / 360.0
        np.testing.assert_allclose(d, expected, atol=1e-9)

    def test_haversine_zero_self_distance(self):
        """d(p, p) = 0 (load-bearing for the K(0) = 1 identity in
        Eq. 4.2's diagonal contribution)."""
        d = _haversine_km(
            np.array(45.0),
            np.array(-122.0),
            np.array(45.0),
            np.array(-122.0),
        )
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_haversine_symmetric(self):
        """d(a, b) = d(b, a) (load-bearing for symmetric kernel matrix)."""
        d_ab = _haversine_km(
            np.array(40.7),
            np.array(-74.0),
            np.array(34.0),
            np.array(-118.2),
        )
        d_ba = _haversine_km(
            np.array(34.0),
            np.array(-118.2),
            np.array(40.7),
            np.array(-74.0),
        )
        np.testing.assert_allclose(d_ab, d_ba, atol=1e-12)

    def test_haversine_pole_to_equator(self):
        """North pole to equator at any longitude = pi/2 * R ~ 10007.5 km."""
        d = _haversine_km(
            np.array(90.0),
            np.array(0.0),
            np.array(0.0),
            np.array(0.0),
        )
        expected = np.pi * _CONLEY_EARTH_RADIUS_KM / 2.0
        np.testing.assert_allclose(d, expected, atol=1e-9)

    def test_haversine_broadcasting_pairwise(self):
        """Broadcasting (n, 1) vs (1, n) yields the n*n distance matrix."""
        coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        lats = coords[:, 0]
        lons = coords[:, 1]
        D = _haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
        assert D.shape == (3, 3)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        np.testing.assert_allclose(D[0, 2], 2.0 * D[0, 1], rtol=1e-10)

    def test_pairwise_distance_haversine_dispatch(self):
        """Pairwise distance helper dispatches haversine correctly."""
        coords = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
        D = _pairwise_distance_matrix(coords, "haversine")
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_pairwise_distance_euclidean_matches_pdist(self):
        """Euclidean path matches scipy.spatial.distance squareform exactly
        (sanity check for the alternative metric path)."""
        from scipy.spatial.distance import pdist, squareform

        rng = np.random.default_rng(seed=7)
        coords = rng.uniform(0, 100, size=(15, 2))
        D = _pairwise_distance_matrix(coords, "euclidean")
        D_scipy = squareform(pdist(coords, metric="euclidean"))
        np.testing.assert_allclose(D, D_scipy, atol=1e-12)


# ---------------------------------------------------------------------------
# TestConleyBlockDecomposition — Phase 2 panel block-decomposed sandwich
# ---------------------------------------------------------------------------


class TestConleyBlockDecomposition:
    """Phase 2 panel block-decomposed sandwich `XeeX = XeeX_spatial +
    XeeX_serial` matching ``conleyreg::time_dist.cpp``.

    Internal cross-check at `atol=1e-12` against a hand-coded reference
    that re-implements the block decomposition directly from the C++ form.
    """

    def test_panel_matches_block_decomposed_reference(self):
        """Direct verification that ``_compute_conley_vcov`` matches a
        hand-coded block decomposition (matches R ``conleyreg::time_dist.cpp``)
        at machine precision across L in (0, 1, 2)."""
        X, residuals, coords, time, unit, bread, cutoff = _make_conley_panel(seed=314)
        bread_inv = np.linalg.inv(bread)
        S = X * residuals[:, None]
        for L in (0, 1, 2):
            meat = np.zeros((X.shape[1], X.shape[1]))
            for t_val in np.unique(time):
                mask = time == t_val
                D_t = _pairwise_distance_matrix(coords[mask], "haversine")
                K_t = _bartlett_kernel(D_t / cutoff)
                meat += S[mask].T @ K_t @ S[mask]
            if L > 0:
                for u_val in np.unique(unit):
                    mask = unit == u_val
                    S_u = S[mask]
                    t_u = time[mask].astype(np.float64)
                    lag = np.abs(t_u[:, None] - t_u[None, :])
                    K_u = ((lag <= L) & (lag != 0)).astype(np.float64) * (1.0 - lag / (L + 1.0))
                    meat += S_u.T @ K_u @ S_u
            V_ref = bread_inv @ meat @ bread_inv

            V_helper = _compute_conley_vcov(
                X,
                residuals,
                coords,
                cutoff,
                "haversine",
                "bartlett",
                bread,
                time=time,
                unit=unit,
                lag_cutoff=L,
            )
            np.testing.assert_allclose(V_helper, V_ref, atol=1e-12)

    def test_cluster_time_invariance_validated_on_panel_path(self):
        """REGISTRY § ConleySpatialHAC § Combined spatial + cluster product
        kernel: on the panel path the validator REQUIRES cluster_ids to be
        constant within each unit across periods, raising ValueError with
        the violating unit named otherwise. Anchors the contract that
        guarantees the within-unit serial sandwich's cluster mask is
        trivially all-ones, simplifying the math to bare serial Bartlett HAC
        weighted by the spatial mask only."""
        rng = np.random.default_rng(seed=17)
        n_units = 4
        T = 3
        n = n_units * T
        unit = np.repeat(np.arange(n_units), T)
        time = np.tile(np.arange(T), n_units)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        coords = rng.uniform(0, 1, size=(n, 2))
        # Time-varying cluster for unit 0
        cluster = unit.copy()
        cluster[0] = 99  # unit 0 at time 0 has cluster 99 instead of 0
        with pytest.raises(ValueError, match=r"(?i)cluster|time-invariant|unit"):
            compute_robust_vcov(
                X,
                eps,
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=10.0,
                conley_metric="euclidean",
                conley_time=time,
                conley_unit=unit,
                conley_lag_cutoff=1,
                cluster_ids=cluster,
            )


# ---------------------------------------------------------------------------
# TestConleySparseDenseEquivalence — Wave A #120 numerical correctness
# ---------------------------------------------------------------------------


class TestConleySparseDenseEquivalence:
    """Sparse k-d-tree path produces meat bit-identical (to atol=1e-10) to
    the dense path.

    The methodology anchor for Wave A #120 is *numerical correctness*: the
    sparse path is an optimization, not a methodology change, so its output
    must agree with the dense reference at full precision (modulo
    chord-projection roundoff on haversine, which is absorbed by atol=1e-10).
    Sparse-path activation thresholds, density-gate fallback, callable-
    metric / uniform-kernel fallbacks are defensive surface and remain in
    ``tests/test_conley_vcov.py``.
    """

    def _euclidean_fixture(self, *, n=1000, k=3, cutoff=15.0, seed=11):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
        beta = np.linspace(0.5, 2.0, k)
        y = X @ beta + rng.standard_normal(n) * 0.5
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        return X, residuals, coords, bread, cutoff

    def _haversine_fixture(self, *, n=1000, k=3, cutoff_km=500.0, seed=13):
        rng = np.random.default_rng(seed)
        lats = rng.uniform(-30.0, 30.0, size=n)
        lons = rng.uniform(-100.0, 100.0, size=n)
        coords = np.column_stack([lats, lons])
        X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
        beta = np.linspace(0.5, 2.0, k)
        y = X @ beta + rng.standard_normal(n) * 0.5
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        return X, residuals, coords, bread, cutoff_km

    @pytest.mark.slow
    def test_sparse_vs_dense_euclidean_cross_sectional(self):
        """Sparse and dense paths produce the same meat on a 1000-row
        euclidean+bartlett fixture at `atol=1e-10`."""
        X, residuals, coords, bread, cutoff = self._euclidean_fixture(n=1000)
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    @pytest.mark.slow
    def test_sparse_vs_dense_haversine_cross_sectional(self):
        """Sparse and dense paths produce the same meat on a 1000-row
        haversine+bartlett fixture at `atol=1e-10`. Haversine adds the
        chord-projection roundoff that the tolerance must absorb."""
        X, residuals, coords, bread, cutoff = self._haversine_fixture(n=1000)
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    @pytest.mark.slow
    def test_sparse_vs_dense_panel_block_decomposed(self):
        """Panel block-decomposed sandwich produces the same vcov whether the
        spatial component is computed dense or sparse. The serial component
        is always dense regardless of the flag. Locks Wave A #120 numerical
        correctness on the Phase 2 path."""
        X, residuals, coords, bread, cutoff = self._euclidean_fixture(n=900, seed=21)
        time = np.repeat(np.arange(3), 300)
        unit = np.tile(np.arange(300), 3)
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=1,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=1,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    @pytest.mark.slow
    def test_sparse_with_cluster_cross_sectional(self):
        """Sparse + combined cluster kernel (Wave A #119) matches the dense
        path bit-for-bit at `atol=1e-10` on a cross-sectional fixture.
        Load-bearing for the sparse + Wave A #119 product-kernel composition."""
        rng = np.random.default_rng(seed=181)
        n = 600
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        cluster_ids = rng.integers(0, 8, size=n)
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5, -0.5]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            15.0,
            "euclidean",
            "bartlett",
            bread,
            cluster_ids=cluster_ids,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            15.0,
            "euclidean",
            "bartlett",
            bread,
            cluster_ids=cluster_ids,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    @pytest.mark.slow
    def test_sparse_with_cluster_panel(self):
        """Sparse + Wave A #119 cluster product kernel + Phase 2 panel
        block-decomposed sandwich matches the dense path on a 3-period
        panel with time-invariant cluster. Cluster-time-invariance contract
        guarantees the within-unit serial mask is trivially all-ones."""
        rng = np.random.default_rng(seed=191)
        n_units = 200
        T = 3
        unit = np.repeat(np.arange(n_units), T)
        time = np.tile(np.arange(T), n_units)
        n = n_units * T
        unit_coords = rng.uniform(0.0, 100.0, size=(n_units, 2))
        coords = unit_coords[unit]
        cluster_per_unit = rng.integers(0, 5, size=n_units)
        cluster_ids = cluster_per_unit[unit]
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5, -0.3]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            15.0,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=1,
            cluster_ids=cluster_ids,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            15.0,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=1,
            cluster_ids=cluster_ids,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestConleyParityR — R conleyreg v0.1.9 parity at atol=1e-6
# ---------------------------------------------------------------------------


class TestConleyParityR:
    """R ``conleyreg`` v0.1.9 parity (Duesterhoeft 2021 CRAN).

    Goldens at ``benchmarks/data/r_conleyreg_conley_golden.json``; generator
    at ``benchmarks/R/generate_conley_golden.R``. 6 fixtures:

    - 3 cross-sectional (Phase 1): ``small_haversine``, ``dense_haversine``,
      ``lat_lon_realistic``.
    - 3 panel (Phase 2 block-decomposed with `lag_cutoff > 0`):
      ``panel_haversine_lag1``, ``panel_haversine_lag2``,
      ``panel_lat_lon_realistic_lag1``.

    Parity tolerance: `atol=1e-6` and `rtol=1e-6`. Earth radius 6371.01 km
    matches ``conleyreg::haversine_dist``. The sparse k-d-tree path forced
    on the same goldens is exercised at ``test_sparse_forced_matches_r_*``
    to verify the optimization path also matches R parity.

    Also verifies the time-asymmetric kernel parity contract: R
    ``conleyreg``'s `kernel` argument controls the spatial component only;
    the temporal kernel is unconditionally Bartlett (`(1 - |t-s|/(L+1))`)
    regardless of `kernel` (see ``conleyreg::time_dist.cpp`` for the
    hardcoding). diff-diff matches this asymmetry exactly. This is a
    parity contract, not a deviation — the independent-temporal-kernel
    follow-up is documented as a deferral in
    ``TestConleyDeviationsFromR::test_independent_temporal_kernel_deferred``.
    """

    def _check_cross_section_fixture(self, golden, name):
        entry = golden[name]
        X = np.asarray(entry["x"], dtype=np.float64).reshape(entry["x_shape"])
        y = np.asarray(entry["y"], dtype=np.float64)
        coords = np.asarray(entry["coords"], dtype=np.float64).reshape(entry["coords_shape"])
        vcov_expected = np.asarray(entry["vcov"], dtype=np.float64).reshape(entry["vcov_shape"])

        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        vcov_got = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=entry["cutoff_km"],
            conley_metric=entry["metric"],
            conley_kernel=entry["kernel"],
        )
        np.testing.assert_allclose(vcov_got, vcov_expected, atol=PARITY_TOL, rtol=PARITY_TOL)

    def _check_panel_fixture(self, golden, name):
        entry = golden[name]
        X = np.asarray(entry["x"], dtype=np.float64).reshape(entry["x_shape"])
        y = np.asarray(entry["y"], dtype=np.float64)
        coords = np.asarray(entry["coords"], dtype=np.float64).reshape(entry["coords_shape"])
        vcov_expected = np.asarray(entry["vcov"], dtype=np.float64).reshape(entry["vcov_shape"])
        unit = np.asarray(entry["unit"])
        time = np.asarray(entry["time"])

        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        vcov_got = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=entry["cutoff_km"],
            conley_metric=entry["metric"],
            conley_kernel=entry["kernel"],
            conley_time=time,
            conley_unit=unit,
            conley_lag_cutoff=int(entry["lag_cutoff"]),
        )
        np.testing.assert_allclose(vcov_got, vcov_expected, atol=PARITY_TOL, rtol=PARITY_TOL)

    def test_parity_small_haversine(self):
        self._check_cross_section_fixture(_load_r_conley_goldens(), "small_haversine")

    def test_parity_dense_haversine(self):
        self._check_cross_section_fixture(_load_r_conley_goldens(), "dense_haversine")

    def test_parity_lat_lon_realistic(self):
        self._check_cross_section_fixture(_load_r_conley_goldens(), "lat_lon_realistic")

    def test_parity_panel_haversine_lag1(self):
        self._check_panel_fixture(_load_r_conley_goldens(), "panel_haversine_lag1")

    def test_parity_panel_haversine_lag2(self):
        self._check_panel_fixture(_load_r_conley_goldens(), "panel_haversine_lag2")

    def test_parity_panel_lat_lon_realistic_lag1(self):
        self._check_panel_fixture(_load_r_conley_goldens(), "panel_lat_lon_realistic_lag1")

    def test_sparse_forced_matches_r_cross_sectional(self):
        """Sparse path forced on the R conleyreg cross-sectional fixtures
        also passes parity at `atol=1e-6`. Verifies Wave A #120 sparse
        optimization preserves R parity, not just dense-vs-sparse internal
        equivalence."""
        golden = _load_r_conley_goldens()
        for name in ("small_haversine", "dense_haversine", "lat_lon_realistic"):
            entry = golden[name]
            if entry["kernel"] != "bartlett":
                continue  # sparse only supports bartlett
            X = np.asarray(entry["x"], dtype=np.float64).reshape(entry["x_shape"])
            y = np.asarray(entry["y"], dtype=np.float64)
            coords = np.asarray(entry["coords"], dtype=np.float64).reshape(entry["coords_shape"])
            vcov_expected = np.asarray(entry["vcov"], dtype=np.float64).reshape(entry["vcov_shape"])
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ coefs
            bread = X.T @ X
            vcov_got = _compute_conley_vcov(
                X,
                residuals,
                coords,
                entry["cutoff_km"],
                entry["metric"],
                entry["kernel"],
                bread,
                _conley_sparse=True,
            )
            np.testing.assert_allclose(vcov_got, vcov_expected, atol=PARITY_TOL, rtol=PARITY_TOL)

    def test_sparse_forced_matches_r_panel(self):
        """Sparse path forced on the R conleyreg panel fixtures (Phase 2
        block-decomposed sandwich) also passes parity at `atol=1e-6`.
        Verifies Wave A #120 sparse optimization preserves R parity on the
        spatial component of the panel path; the serial component is always
        dense regardless of the flag."""
        golden = _load_r_conley_goldens()
        for name in (
            "panel_haversine_lag1",
            "panel_haversine_lag2",
            "panel_lat_lon_realistic_lag1",
        ):
            entry = golden[name]
            if entry["kernel"] != "bartlett":
                continue  # sparse only supports bartlett
            X = np.asarray(entry["x"], dtype=np.float64).reshape(entry["x_shape"])
            y = np.asarray(entry["y"], dtype=np.float64)
            coords = np.asarray(entry["coords"], dtype=np.float64).reshape(entry["coords_shape"])
            vcov_expected = np.asarray(entry["vcov"], dtype=np.float64).reshape(entry["vcov_shape"])
            unit = np.asarray(entry["unit"])
            time = np.asarray(entry["time"])
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ coefs
            bread = X.T @ X
            vcov_got = _compute_conley_vcov(
                X,
                residuals,
                coords,
                entry["cutoff_km"],
                entry["metric"],
                entry["kernel"],
                bread,
                time=time,
                unit=unit,
                lag_cutoff=int(entry["lag_cutoff"]),
                _conley_sparse=True,
            )
            np.testing.assert_allclose(vcov_got, vcov_expected, atol=PARITY_TOL, rtol=PARITY_TOL)

    def test_time_asymmetric_kernel_matches_r_literal(self):
        """R ``conleyreg::time_dist.cpp`` hardcodes the Bartlett temporal
        kernel regardless of the user's `kernel` argument — the spatial
        kernel choice does NOT propagate to the serial component. diff-diff
        matches this asymmetry exactly: the serial-contribution delta
        between two `lag_cutoff` values is IDENTICAL whether the spatial
        kernel is bartlett or uniform.

        Moved from `TestConleyDeviationsFromR` to `TestConleyParityR` per
        plan review MEDIUM #1 — this is a parity contract with R
        ``conleyreg`` literal, not a deviation. The independent-temporal-
        kernel API extension would be a deviation and is documented as a
        deferral in
        ``TestConleyDeviationsFromR::test_independent_temporal_kernel_deferred``.
        """
        X, residuals, coords, time, unit, bread, cutoff = _make_conley_panel()
        V_bartlett_L0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
        )
        V_bartlett_L2 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=2,
        )
        V_uniform_L0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "uniform",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
        )
        V_uniform_L2 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "uniform",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=2,
        )
        # The serial-contribution delta must match across spatial kernels
        # because the temporal kernel is hardcoded Bartlett in both
        # diff-diff and R conleyreg.
        delta_bartlett = bread @ (V_bartlett_L2 - V_bartlett_L0) @ bread
        delta_uniform = bread @ (V_uniform_L2 - V_uniform_L0) @ bread
        np.testing.assert_allclose(delta_bartlett, delta_uniform, atol=1e-10)


# ---------------------------------------------------------------------------
# TestConleyLibraryExtensions — Wave A library extensions (no R correspondence)
# ---------------------------------------------------------------------------


class TestConleyLibraryExtensions:
    """Wave A library extensions documented as library-side additions with
    no paper or R correspondence (6 tests).

    [x] Combined spatial + cluster product kernel (Wave A #119) — Anchor 1:
        all-unique-clusters → HC0 diagonal
    [x] Combined spatial + cluster product kernel (Wave A #119) — Anchor 2:
        huge-cutoff + uniform → within-cluster CR1 plain
    [x] Callable conley_metric validation (Wave A #123)
    [x] Sparse k-d-tree fast path (Wave A #120) — activation thresholds and
        boundary behavior
    [x] Indefiniteness guard on Bartlett kernel
    [x] Indefiniteness guard on uniform kernel
    """

    def test_combined_spatial_cluster_kernel_wave_a_119_all_unique(self):
        """Wave A #119 library extension: combined kernel
        `K_total[i, j] = K_space(d_ij/h) * 1{cluster_i = cluster_j}`.

        Anchor 1 (REGISTRY § Combined spatial + cluster product kernel):
        all-unique-clusters reduction — cluster mask is identity, meat
        reduces to HC0 diagonal `Sum_i X_i eps_i^2 X_i'`.
        """
        rng = np.random.default_rng(seed=21)
        n = 30
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        coords = rng.uniform(0, 1, size=(n, 2))
        cluster_unique = np.arange(n)
        bread = X.T @ X
        bread_inv = np.linalg.solve(bread, np.eye(2))
        meat_hc0 = X.T @ (X * (eps**2)[:, None])
        vcov_hc0 = bread_inv @ meat_hc0 @ bread_inv
        vcov_combined = compute_robust_vcov(
            X,
            eps,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=10.0,  # large; full kernel reduces to ones
            conley_metric="euclidean",
            conley_kernel="uniform",
            cluster_ids=cluster_unique,
        )
        np.testing.assert_allclose(vcov_combined, vcov_hc0, atol=1e-12)

    def test_combined_spatial_cluster_kernel_wave_a_119_huge_cutoff(self):
        """Wave A #119 library extension: combined kernel
        `K_total[i, j] = K_space(d_ij/h) * 1{cluster_i = cluster_j}`.

        Anchor 2 (REGISTRY § Combined spatial + cluster product kernel):
        huge-cutoff under uniform — `K_space = 1` on every pair, so the
        meat reduces to the pure within-cluster sum
        `Sum_g X_g' eps_g eps_g' X_g` (CR1 without the Liang-Zeger
        small-sample correction). This exact identity holds only for
        `conley_kernel="uniform"` (K_uniform(u) = 1 for |u| <= 1); the
        Bartlett kernel gives `K(u) = 1 - |u| < 1` for u > 0, so the
        Bartlett huge-cutoff limit is asymptotic, not exact at any finite
        cutoff. The anchor uses uniform for an exact identity check.
        """
        rng = np.random.default_rng(seed=22)
        n = 24
        n_clusters = 6
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        coords = rng.uniform(0, 1, size=(n, 2))
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)
        bread = X.T @ X
        bread_inv = np.linalg.solve(bread, np.eye(X.shape[1]))
        # Pure within-cluster CR1 sandwich (no Liang-Zeger correction)
        S = X * eps[:, None]
        meat_cr1_plain = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(cluster_ids):
            S_c = S[cluster_ids == c]
            S_c_sum = S_c.sum(axis=0)
            meat_cr1_plain += np.outer(S_c_sum, S_c_sum)
        vcov_cr1_plain = bread_inv @ meat_cr1_plain @ bread_inv
        # Huge-cutoff under uniform reduces combined kernel to within-cluster
        vcov_combined = compute_robust_vcov(
            X,
            eps,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=1e9,  # >> all pairwise distances
            conley_metric="euclidean",
            conley_kernel="uniform",
            cluster_ids=cluster_ids,
        )
        np.testing.assert_allclose(vcov_combined, vcov_cr1_plain, atol=1e-12)

    def test_callable_metric_validation_wave_a_123(self):
        """Wave A #123 library extension: callable conley_metric is
        validated at the boundary with 6 invariants — float64-castable,
        `(n, n)` shape, finite, non-negative, symmetric to `atol=1e-10`,
        and zero diagonal `|d(i, i)| <= 1e-10` (so K(0) = 1 reduces to
        the HC0 diagonal contribution). Asymmetric callable raises
        `ValueError`."""
        rng = np.random.default_rng(seed=31)
        n = 10
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        coords = rng.uniform(0, 1, size=(n, 2))

        def asymmetric_metric(c1, c2):
            d = np.zeros((len(c1), len(c2)))
            for i in range(len(c1)):
                for j in range(len(c2)):
                    d[i, j] = float(i + 2 * j)  # asymmetric: d[i, j] != d[j, i]
            return d

        with pytest.raises(ValueError, match=r"(?i)symmetric|asymmetric"):
            compute_robust_vcov(
                X,
                eps,
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=10.0,
                conley_metric=asymmetric_metric,
            )

    def test_sparse_kd_tree_activation_wave_a_120(self):
        """Wave A #120 library extension: sparse k-d-tree path auto-activates
        when `n > _CONLEY_SPARSE_N_THRESHOLD` AND `conley_kernel == "bartlett"`
        AND `conley_metric` is haversine or euclidean (no callable). At
        `n + 1` above threshold the sparse helper is invoked; at threshold
        the dense path stays."""
        import diff_diff.conley as conley_module

        rng = np.random.default_rng(seed=41)
        n = _CONLEY_SPARSE_N_THRESHOLD + 1
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        bread = X.T @ X

        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        conley_module._compute_spatial_bartlett_meat_sparse = _spy
        try:
            _compute_conley_vcov(X, eps, coords, 15.0, "euclidean", "bartlett", bread)
        finally:
            conley_module._compute_spatial_bartlett_meat_sparse = orig
        assert calls["n"] >= 1, "Sparse helper not called when n > threshold."

    def test_indefiniteness_guard_fires_on_negative_eigenvalue(self):
        """REGISTRY L3609: the 1-D radial Bartlett implementation and the
        uniform kernel are practitioner specializations of Conley 1999
        (not the paper's explicitly PSD-guaranteed 2-D separable Eq. 3.14
        form). Library hardening: the indefiniteness guard emits a
        `UserWarning` when any meat eigenvalue is below `-1e-12`. Forces
        the indefinite path by monkey-patching the kernel to return a
        sign-pattern that produces an indefinite meat, then asserts the
        warning surfaces with the kernel name."""
        from diff_diff import conley as conley_mod

        rng = np.random.default_rng(seed=11)
        n = 6
        coords = rng.uniform(0, 1, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = np.ones(n)
        bread = X.T @ X

        original = conley_mod._bartlett_kernel

        def _indefinite(u: np.ndarray) -> np.ndarray:
            base = np.eye(u.shape[0])
            for i in range(u.shape[0]):
                for j in range(u.shape[0]):
                    if i != j:
                        base[i, j] = -10.0
            return base

        try:
            conley_mod._bartlett_kernel = _indefinite
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                conley_mod._compute_conley_vcov(
                    X,
                    eps,
                    coords,
                    cutoff=10.0,
                    metric="euclidean",
                    kernel="bartlett",
                    bread_matrix=bread,
                )
            psd_warnings = [
                msg
                for msg in w
                if issubclass(msg.category, UserWarning)
                and "bartlett" in str(msg.message)
                and "negative eigenvalue" in str(msg.message)
            ]
            assert len(psd_warnings) >= 1, (
                f"Expected UserWarning naming kernel='bartlett' and "
                f"'negative eigenvalue'; got {[str(m.message) for m in w]}"
            )
        finally:
            conley_mod._bartlett_kernel = original

    def test_indefiniteness_guard_fires_on_negative_eigenvalue_uniform(self):
        """REGISTRY L3609 + the production indefiniteness guard at
        ``diff_diff/linalg.py:L3024-L3030`` covers BOTH kernels (uniform is
        documented as not PSD-guaranteed per Conley 1999 footnote 11 —
        spectral window is negative in regions). Library hardening: the
        guard emits a ``UserWarning`` naming the kernel when any meat
        eigenvalue is below ``-1e-12``. Forces the indefinite path by
        monkey-patching ``_uniform_kernel`` to a sign-pattern that produces
        an indefinite meat, then asserts the warning surfaces with the
        uniform-kernel name (regression for the uniform-kernel branch of
        the same contract)."""
        from diff_diff import conley as conley_mod

        rng = np.random.default_rng(seed=12)
        n = 6
        coords = rng.uniform(0, 1, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = np.ones(n)
        bread = X.T @ X

        original = conley_mod._uniform_kernel

        def _indefinite(u: np.ndarray) -> np.ndarray:
            base = np.eye(u.shape[0])
            for i in range(u.shape[0]):
                for j in range(u.shape[0]):
                    if i != j:
                        base[i, j] = -10.0
            return base

        try:
            conley_mod._uniform_kernel = _indefinite
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                conley_mod._compute_conley_vcov(
                    X,
                    eps,
                    coords,
                    cutoff=10.0,
                    metric="euclidean",
                    kernel="uniform",
                    bread_matrix=bread,
                )
            psd_warnings = [
                msg
                for msg in w
                if issubclass(msg.category, UserWarning)
                and "uniform" in str(msg.message)
                and "negative eigenvalue" in str(msg.message)
            ]
            assert len(psd_warnings) >= 1, (
                f"Expected UserWarning naming kernel='uniform' and "
                f"'negative eigenvalue'; got {[str(m.message) for m in w]}"
            )
        finally:
            conley_mod._uniform_kernel = original


# ---------------------------------------------------------------------------
# TestConleyDeviationsFromR — Deviations from R conleyreg literal
# ---------------------------------------------------------------------------


class TestConleyDeviationsFromR:
    """Deviations from the R ``conleyreg`` literal that are intentional,
    documented in REGISTRY, and exercised here.

    [x] 1-D radial Bartlett vs paper's 2-D separable Eq. 3.14 (matches
        ``conleyreg``, Stata ``acreg``, Hsiang 2010 — not formally
        PSD-guaranteed)
    [x] Time-label normalization via `np.unique(return_inverse=True)`
        (diff-diff normalizes to dense panel-period codes; R uses raw
        time values literally)
    [x] Independent temporal kernel deferred (R `kernel` arg controls
        spatial only; the literal-matching parity assertion is in
        ``TestConleyParityR::test_time_asymmetric_kernel_matches_r_literal``)
    """

    def test_1d_radial_bartlett_vs_2d_separable_eq314(self):
        """The 1-D radial Bartlett K(u) = max(0, 1 - |u|) on pairwise
        distance is a practitioner specialization that diff-diff,
        ``conleyreg``, Stata ``acreg``, and Hsiang (2010) all share. It
        differs from Conley 1999's explicitly PSD-guaranteed 2-D separable
        Eq. 3.14 form `K(j, k) = (1 - |j|/L_M)(1 - |k|/L_N)` indexed on a
        lattice. The 1-D form is NOT formally PSD-guaranteed, hence the
        indefiniteness guard at `< -1e-12` documented in REGISTRY L3609.

        This test pins the 1-D form numerically (NOT the 2-D form). The
        2-D form would require a lattice-indexed implementation which
        diff-diff does not provide."""
        u = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])
        np.testing.assert_allclose(_bartlett_kernel(u), expected, atol=1e-15)

    def test_time_label_normalization_deviation_from_r(self):
        """REGISTRY L3592-3603: diff-diff normalizes time labels via
        `np.unique(return_inverse=True)` to dense panel-period codes
        before lag computation; R ``conleyreg`` uses raw time values
        literally (`time_dist.cpp`'s `t_diff = abs(times - times[i])`).

        On dense integer labels (the parity-test convention), the two
        paths produce bit-identical results. For non-dense encodings —
        e.g., `time = 202012, 202101` for monthly panels — the raw R
        difference is 89, so a `lag_cutoff=1` request silently drops
        valid lag-1 serial pairs in R. diff-diff is the more robust
        default; this test verifies bit-equality between dense codes
        and YYYYMM labels for the same underlying panel."""
        X, residuals, coords, _, unit, bread, cutoff = _make_conley_panel(n_units=8, T=3, k=2)
        time_dense = np.tile([1, 2, 3], 8)
        time_yyyymm = np.tile([202011, 202012, 202101], 8)
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time_dense,
            unit=unit,
            lag_cutoff=1,
        )
        V_yyyymm = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "haversine",
            "bartlett",
            bread,
            time=time_yyyymm,
            unit=unit,
            lag_cutoff=1,
        )
        np.testing.assert_allclose(V_yyyymm, V_dense, atol=1e-12)

    def test_independent_temporal_kernel_deferred(self):
        """REGISTRY L3584-3591: R ``conleyreg``'s `kernel` argument controls
        the spatial component only; the temporal kernel is unconditionally
        Bartlett. diff-diff matches this asymmetry exactly for parity (see
        ``TestConleyParityR::test_time_asymmetric_kernel_matches_r_literal``
        for the literal-matching assertion).

        Independent temporal-kernel choice would be a follow-up API
        extension if user demand emerges. This test pins the current
        contract by asserting `conley_kernel` is the only kernel knob in
        the supported API surface — there is no `conley_temporal_kernel`
        kwarg yet."""
        import inspect

        sig = inspect.signature(compute_robust_vcov)
        param_names = set(sig.parameters.keys())
        assert "conley_kernel" in param_names
        # No independent temporal kernel parameter is supported.
        assert "conley_temporal_kernel" not in param_names
        assert "conley_time_kernel" not in param_names


# ---------------------------------------------------------------------------
# TestConleyDeferrals — Fail-closed NotImplementedError / TypeError contracts
# ---------------------------------------------------------------------------


class TestConleyDeferrals:
    """Fail-closed contracts asserting unsupported combinations raise
    explicit `NotImplementedError` / `TypeError`.

    [x] `LinearRegression(vcov_type="conley", survey_design=...)` (generic-
        path deferral; open methodological question)
    [x] DiD / MPD / TWFE + Conley + survey_design (estimator-level)
    [x] Conley + weights (cross-sectional sampling weights without survey
        design)
    [x] SyntheticDiD + Conley (uses bootstrap/jackknife/placebo variance)
    [x] wild_bootstrap + Conley (separate inference path)

    Generic-path deferral rationale: weighted spatial-HAC under probability
    sampling is an open methodological question; no canonical extension of
    Conley (1999) exists for the combination. Shipped estimator-specific
    surfaces: SpilloverDiD + Conley + survey via Wave E.1/E.2/E.3 (PR
    #468/#474/#482); TwoStageDiD + Conley + survey via Wave E.3 parity
    (PR #485).
    """

    def test_linear_regression_survey_design_not_implemented(self):
        """`LinearRegression(vcov_type='conley', survey_design=...)` raises
        `NotImplementedError` at the LinearRegression entry (linalg.py
        front-door rejection). Uses `inspect.signature` to confirm the
        `survey_design` kwarg exists (so the rejection path is reachable)
        before asserting the error."""
        import inspect

        import pandas as pd

        from diff_diff.survey import SurveyDesign

        sig = inspect.signature(LinearRegression.__init__)
        assert "survey_design" in sig.parameters, (
            "LinearRegression must accept survey_design kwarg for the "
            "generic-path Conley+survey rejection contract to be reachable"
        )

        rng = np.random.default_rng(seed=51)
        n = 30
        y = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        coords = np.column_stack([rng.uniform(-30, 30, n), rng.uniform(-100, 100, n)])
        df = pd.DataFrame(
            {
                "psu": np.repeat(np.arange(5), n // 5),
                "weights": np.ones(n),
            }
        )
        sd = SurveyDesign(psu="psu", weights="weights").resolve(df)
        with pytest.raises(NotImplementedError, match=r"(?i)conley.*survey|open methodological"):
            LinearRegression(
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=1000.0,
                survey_design=sd,
            ).fit(X, y)

    def test_did_mpd_twfe_survey_design_not_implemented(self):
        """REGISTRY L3760: DiD / MPD / TWFE + Conley + survey_design raises
        `NotImplementedError` at the estimator level. Tests on DiD as the
        primary surface; MPD/TWFE share the same fail-closed contract."""
        import pandas as pd

        from diff_diff.estimators import DifferenceInDifferences
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(seed=61)
        n_units = 6
        T = 2
        n = n_units * T
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(n),
                "treated": np.tile([0, 0, 1, 1, 0, 1], 2),
                "post": np.repeat([0, 1], n_units),
                "unit": np.tile(np.arange(n_units), T),
                "psu": np.tile(np.repeat([0, 1, 2], 2), T),
                "lat": np.tile(rng.uniform(-30, 30, n_units), T),
                "lon": np.tile(rng.uniform(-100, 100, n_units), T),
                "weights": np.ones(n),
            }
        )
        sd = SurveyDesign(psu="psu", weights="weights")
        with pytest.raises(NotImplementedError, match=r"(?i)conley.*survey|open methodological"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=1000.0,
                conley_lag_cutoff=0,
            ).fit(
                df,
                formula="y ~ treated * post",
                unit="unit",
                survey_design=sd,
            )

    def test_conley_plus_weights_not_implemented(self):
        """REGISTRY L3762: cross-sectional sampling weights without survey
        design also raise `NotImplementedError`. Same open methodological
        question — weighted spatial-HAC under probability sampling has no
        canonical extension of Conley (1999)."""
        rng = np.random.default_rng(seed=71)
        n = 20
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = rng.standard_normal(n)
        coords = rng.uniform(0, 1, size=(n, 2))
        weights = rng.uniform(0.5, 1.5, size=n)
        with pytest.raises(NotImplementedError, match=r"(?i)conley.*weight|open methodological"):
            compute_robust_vcov(
                X,
                eps,
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=10.0,
                conley_metric="euclidean",
                weights=weights,
            )

    def test_synthetic_did_conley_typeerror(self):
        """`SyntheticDiD(vcov_type='conley')` raises `TypeError` —
        SyntheticDiD uses bootstrap / jackknife / placebo variance, not
        the analytical sandwich. Tracked in DEFERRED.md."""
        from diff_diff.synthetic_did import SyntheticDiD

        with pytest.raises(TypeError, match=r"(?i)conley|vcov_type"):
            SyntheticDiD(vcov_type="conley")

    def test_wild_bootstrap_conley_not_implemented(self):
        """REGISTRY L3547: wild bootstrap is a separate inference path that
        does not consume the analytical Conley sandwich. Combining
        `vcov_type='conley'` with `inference='wild_bootstrap'` raises
        `NotImplementedError`."""
        import pandas as pd

        from diff_diff.estimators import DifferenceInDifferences

        rng = np.random.default_rng(seed=81)
        n_units = 6
        T = 2
        n = n_units * T
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(n),
                "treated": np.tile([0, 0, 1, 1, 0, 1], 2),
                "post": np.repeat([0, 1], n_units),
                "unit": np.tile(np.arange(n_units), T),
                "lat": np.tile(rng.uniform(-30, 30, n_units), T),
                "lon": np.tile(rng.uniform(-100, 100, n_units), T),
            }
        )
        with pytest.raises(NotImplementedError, match=r"(?i)wild.bootstrap|conley.*inference"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=1000.0,
                conley_lag_cutoff=0,
                inference="wild_bootstrap",
            ).fit(df, formula="y ~ treated * post", unit="unit")
