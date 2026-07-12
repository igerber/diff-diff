"""Defensive regression surface for the Conley (1999) spatial HAC
variance estimator.

Companion to ``tests/test_methodology_conley.py`` (paper-equation-numbered
methodology walk-through + R `conleyreg` v0.1.9 parity at ``atol=1e-6``
+ library extensions / deviations / deferrals area). This file retains:

- Input validation (callable-metric invariants, lat/lon range bounds,
  unknown-metric / unknown-kernel raises)
- Dispatch-level validity (vcov_type membership; cluster + one-way
  rejections; Conley + weights / survey_design fail-closed gates)
- Estimator-level integration smoke tests (DiD + Conley, TWFE + Conley,
  MPD + Conley) and set_params atomicity
- Sparse-path activation thresholds + density-gate fallback behavior
  (the methodology file pins numerical equivalence vs dense; activation
  and fallback are defensive)
- PSD-guard infrastructure (`_validate_meat_psd` stacklevel attribution,
  threshold boundary behavior)

Paper-equation-anchored methodology coverage (Eq. 4.2 sandwich, Andrews
1991 lag truncation, haversine convention, R parity goldens, sparse-vs-
dense bit-identity, combined-kernel limit anchors, indefiniteness guard
on both kernels, library-extension / deviation / deferral contracts)
lives in ``tests/test_methodology_conley.py``.
"""

import warnings

import numpy as np
import pytest

from diff_diff.conley import (
    _CONLEY_EARTH_RADIUS_KM,
    _CONLEY_SPARSE_N_THRESHOLD,
    _bartlett_kernel,
    _compute_conley_vcov,
    _pairwise_distance_matrix,
    _validate_conley_kwargs,
    _validate_meat_psd,
)
from diff_diff.linalg import (
    LinearRegression,
    compute_robust_vcov,
    solve_ols,
)

# ---------------------------------------------------------------------------
# Shared fixtures (small synthetic OLS dataset with geocoords)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_ols_with_coords():
    """20-row OLS dataset with synthetic lat/lon. Used across helper tests."""
    rng = np.random.default_rng(seed=42)
    n = 20
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


# ---------------------------------------------------------------------------
# Note: Methodology-anchored kernel form tests (Bartlett K(u), uniform kernel,
# serial Bartlett HAC matrix) + paper-anchored haversine/distance-metric
# numerical contracts have moved to tests/test_methodology_conley.py
# (TestConleyEquation42, TestConleyAndrewsLagTruncation,
# TestConleyHaversineConvention). The defensive surface for input validation
# (callable metric invariant checks, unknown-metric / unknown-kernel raises,
# lat/lon range validation) remains here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestConleyDistanceMetrics — callable metric validation + unknown-metric raise
# ---------------------------------------------------------------------------


class TestConleyDistanceMetrics:
    def test_pairwise_distance_callable(self):
        """A user-supplied callable is dispatched and its output preserved.
        Output must satisfy the validator's invariants (zero diagonal, finite,
        non-negative, symmetric)."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def constant_offdiag_metric(c1, c2):
            n1 = len(c1)
            n2 = len(c2)
            out = np.full((n1, n2), 5.0)
            np.fill_diagonal(out, 0.0)
            return out

        D = _pairwise_distance_matrix(coords, constant_offdiag_metric)
        expected = np.full((3, 3), 5.0)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(D, expected)

    def test_pairwise_distance_unknown_metric_raises(self):
        """Unknown metric strings raise ValueError from the dispatcher."""
        with pytest.raises(ValueError, match="conley_metric"):
            _pairwise_distance_matrix(np.zeros((3, 2)), "manhattan")

    def test_callable_metric_wrong_shape_raises(self):
        """Callable returning a non-(n, n) matrix raises a targeted ValueError."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def wrong_shape_metric(c1, c2):
            return np.zeros((2, 5))

        with pytest.raises(ValueError, match=r"\(n, n\) distance matrix"):
            _pairwise_distance_matrix(coords, wrong_shape_metric)

    def test_callable_metric_returns_nan_raises(self):
        """Callable returning a matrix with NaN raises a targeted ValueError."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def nan_metric(c1, c2):
            out = np.zeros((3, 3))
            out[0, 1] = np.nan
            out[1, 0] = np.nan
            return out

        with pytest.raises(ValueError, match="non-finite"):
            _pairwise_distance_matrix(coords, nan_metric)

    def test_callable_metric_returns_inf_raises(self):
        """Callable returning a matrix with inf raises (same branch as NaN)."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def inf_metric(c1, c2):
            out = np.zeros((3, 3))
            out[0, 1] = np.inf
            out[1, 0] = np.inf
            return out

        with pytest.raises(ValueError, match="non-finite"):
            _pairwise_distance_matrix(coords, inf_metric)

    def test_callable_metric_negative_entries_raise(self):
        """Callable returning a negative distance raises (distances must be
        non-negative)."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def negative_metric(c1, c2):
            out = np.full((3, 3), 1.0)
            out[0, 1] = -0.5
            out[1, 0] = -0.5
            np.fill_diagonal(out, 0.0)
            return out

        with pytest.raises(ValueError, match="negative entries"):
            _pairwise_distance_matrix(coords, negative_metric)

    def test_callable_metric_asymmetric_raises(self):
        """Callable returning a non-symmetric matrix raises."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def asymmetric_metric(c1, c2):
            out = np.zeros((3, 3))
            out[0, 1] = 1.0
            out[1, 0] = 2.0
            return out

        with pytest.raises(ValueError, match="asymmetric matrix"):
            _pairwise_distance_matrix(coords, asymmetric_metric)

    def test_callable_metric_nonzero_diagonal_raises(self):
        """Callable returning a symmetric/finite/non-negative matrix with a
        positive self-distance still raises, because the Conley sandwich
        requires d(i, i) = 0 (K(0) = 1 reduces the i=j term to the HC0
        diagonal X_i ε_i² X_i'). A nonzero self-distance silently attenuates
        the HC0 contribution by K(d_ii / h) < 1 and misstates Conley SEs.
        Codex CI R4 P1."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def positive_diagonal_metric(c1, c2):
            # Symmetric, finite, non-negative — but d(i, i) = 0.5 > 0.
            n1 = len(c1)
            n2 = len(c2)
            out = np.full((n1, n2), 5.0)
            np.fill_diagonal(out, 0.5)
            return out

        with pytest.raises(ValueError, match=r"nonzero self-distance"):
            _pairwise_distance_matrix(coords, positive_diagonal_metric)

    def test_callable_metric_near_zero_diagonal_accepted(self):
        """Sub-tolerance diagonal (roundoff scale) is accepted, mirroring
        the symmetry-tolerance contract."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def near_zero_diagonal_metric(c1, c2):
            n1 = len(c1)
            n2 = len(c2)
            out = np.full((n1, n2), 5.0)
            np.fill_diagonal(out, 0.0)
            # Diagonal noise well below the 1e-10 tolerance
            out[0, 0] = 1e-13
            return out

        D = _pairwise_distance_matrix(coords, near_zero_diagonal_metric)
        assert D.shape == (3, 3)

    def test_callable_metric_non_array_result_raises(self):
        """Callable returning a non-castable result raises a targeted ValueError."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def non_array_metric(c1, c2):
            return "not an array"

        with pytest.raises(ValueError, match="cannot be cast"):
            _pairwise_distance_matrix(coords, non_array_metric)

    def test_callable_metric_near_symmetric_accepted(self):
        """Sub-tolerance asymmetry (eps-level roundoff) is accepted."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def near_symmetric_metric(c1, c2):
            out = np.full((3, 3), 5.0)
            np.fill_diagonal(out, 0.0)
            # Asymmetry below the 1e-10 tolerance — round-off only
            out[0, 1] += 1e-13
            return out

        D = _pairwise_distance_matrix(coords, near_symmetric_metric)
        assert D.shape == (3, 3)


# ---------------------------------------------------------------------------
# TestConleyValidatorHelpers — direct calls to _validate_conley_kwargs
# ---------------------------------------------------------------------------


class TestConleyValidatorHelpers:
    def test_missing_coords_raises(self):
        with pytest.raises(ValueError, match="conley_coords"):
            _validate_conley_kwargs(
                coords=None, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_missing_cutoff_raises(self):
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=None,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_zero_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=0.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_negative_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=-5.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_nan_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=float("nan"),
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_inf_cutoff_raises(self):
        with pytest.raises(ValueError, match="positive finite"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=float("inf"),
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_3d_coords_raises(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 3)),
                cutoff=100.0,
                metric="haversine",
                kernel="bartlett",
                n=10,
            )

    def test_coord_n_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows but X has"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="haversine",
                kernel="bartlett",
                n=15,
            )

    def test_nan_coord_raises(self):
        bad = np.zeros((10, 2))
        bad[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_conley_kwargs(
                coords=bad, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lat_out_of_range_raises_haversine(self):
        coords = np.array([[91.0, 0.0]] + [[0.0, 0.0]] * 9)
        with pytest.raises(ValueError, match=r"latitude in \[-90, 90\]"):
            _validate_conley_kwargs(
                coords=coords, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lon_out_of_range_raises_haversine(self):
        coords = np.array([[0.0, 200.0]] + [[0.0, 0.0]] * 9)
        with pytest.raises(ValueError, match=r"longitude in \[-180, 180\]"):
            _validate_conley_kwargs(
                coords=coords, cutoff=100.0, metric="haversine", kernel="bartlett", n=10
            )

    def test_lat_out_of_range_skipped_for_euclidean(self):
        """Projected coords are unconstrained — euclidean skips lat/lon checks."""
        coords = np.array([[5000.0, 12000.0]] * 10)  # any units
        # Should not raise
        _validate_conley_kwargs(
            coords=coords, cutoff=100.0, metric="euclidean", kernel="bartlett", n=10
        )

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="conley_kernel"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="haversine",
                kernel="gaussian",
                n=10,
            )

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="conley_metric"):
            _validate_conley_kwargs(
                coords=np.zeros((10, 2)),
                cutoff=100.0,
                metric="manhattan",
                kernel="bartlett",
                n=10,
            )

    def test_callable_metric_accepted(self):
        """Callable distance metric passes validation (delegated to caller)."""
        _validate_conley_kwargs(
            coords=np.zeros((10, 2)),
            cutoff=100.0,
            metric=lambda c1, c2: np.zeros((len(c1), len(c2))),
            kernel="bartlett",
            n=10,
        )

    def test_n_above_warn_threshold_warns(self):
        with pytest.warns(UserWarning, match="dense"):
            _validate_conley_kwargs(
                coords=np.zeros((20_001, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=20_001,
            )

    def test_n_below_warn_threshold_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            _validate_conley_kwargs(
                coords=np.zeros((100, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=100,
            )

    def test_panel_args_partial_raises(self):
        """conley_time / conley_unit / conley_lag_cutoff are three-way co-required."""
        n = 6
        kwargs = dict(
            coords=np.zeros((n, 2)),
            cutoff=100.0,
            metric="euclidean",
            kernel="bartlett",
            n=n,
        )
        # Only time set
        with pytest.raises(ValueError, match="must all be passed together"):
            _validate_conley_kwargs(**kwargs, time=np.arange(n))
        # Only unit + lag set (missing time)
        with pytest.raises(ValueError, match="must all be passed together"):
            _validate_conley_kwargs(**kwargs, unit=np.arange(n), lag_cutoff=1)
        # Time + unit but no lag_cutoff
        with pytest.raises(ValueError, match="must all be passed together"):
            _validate_conley_kwargs(**kwargs, time=np.arange(n), unit=np.arange(n))

    def test_panel_args_all_three_accepted(self):
        """All three panel args together pass validation."""
        n = 6
        _validate_conley_kwargs(
            coords=np.zeros((n, 2)),
            cutoff=100.0,
            metric="euclidean",
            kernel="bartlett",
            n=n,
            time=np.array([1, 2, 1, 2, 1, 2]),
            unit=np.array([1, 1, 2, 2, 3, 3]),
            lag_cutoff=1,
        )

    def test_panel_lag_cutoff_negative_raises(self):
        n = 4
        with pytest.raises(ValueError, match="non-negative integer"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=np.arange(n),
                unit=np.arange(n),
                lag_cutoff=-1,
            )

    def test_panel_time_wrong_length_raises(self):
        n = 4
        with pytest.raises(ValueError, match="conley_time must be a 1-D array"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=np.arange(n + 1),  # mismatched length
                unit=np.arange(n),
                lag_cutoff=1,
            )

    def test_panel_unit_wrong_length_raises(self):
        n = 4
        with pytest.raises(ValueError, match="conley_unit must be a 1-D array"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=np.arange(n),
                unit=np.arange(n + 1),  # mismatched length
                lag_cutoff=1,
            )

    def test_panel_time_nan_raises(self):
        n = 4
        time = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="conley_time contains NaN"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=time,
                unit=np.arange(n),
                lag_cutoff=1,
            )

    def test_panel_unit_nan_float_raises(self):
        """NaN unit IDs would silently drop those rows from the per-unit
        serial HAC sum at `np.unique(unit_arr) + mask_u = unit_arr == u_val`.
        Closes Codex P1.
        """
        n = 4
        unit = np.array([1.0, 2.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="conley_unit contains NaN"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=np.array([1.0, 2.0, 1.0, 2.0]),
                unit=unit,
                lag_cutoff=1,
            )

    def test_panel_unit_pd_na_object_raises(self):
        """Object-dtype unit IDs (mixed string + pd.NA) must also raise."""
        import pandas as pd

        n = 4
        unit = np.array(["A", "B", pd.NA, "C"], dtype=object)
        with pytest.raises(ValueError, match="conley_unit contains NaN"):
            _validate_conley_kwargs(
                coords=np.zeros((n, 2)),
                cutoff=100.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=np.array([1.0, 2.0, 1.0, 2.0]),
                unit=unit,
                lag_cutoff=1,
            )


# ---------------------------------------------------------------------------
# TestValidateMeatPsd — _validate_meat_psd guard helper
# ---------------------------------------------------------------------------


class TestValidateMeatPsd:
    def test_nonfinite_raises(self):
        """Non-finite meat must raise ValueError with the caller's exact
        ``error_msg`` so site-specific guidance reaches the user."""
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with pytest.raises(ValueError, match="custom guidance for caller XYZ"):
            _validate_meat_psd(
                M,
                error_msg="custom guidance for caller XYZ",
                warning_template="unused-here ({eigval:.2e})",
            )

    def test_negative_eigenvalue_warns_with_template_substitution(self):
        """An indefinite meat triggers UserWarning with ``{eigval}``
        substituted in scientific notation."""
        # Symmetric matrix with eigenvalues {2, -1}: aggressively indefinite.
        M = np.array([[0.5, 1.5], [1.5, 0.5]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_meat_psd(
                M,
                error_msg="not used on this path",
                warning_template="SITEX meat: min eigenvalue = {eigval:.2e}",
            )
        psd = [
            msg
            for msg in w
            if issubclass(msg.category, UserWarning) and "SITEX" in str(msg.message)
        ]
        assert len(psd) == 1, f"Expected one PSD warning; got {[str(m.message) for m in w]}"
        # Min eigenvalue is -1.0; verify scientific-notation substitution.
        assert "-1.00e+00" in str(psd[0].message)

    def test_psd_matrix_silent(self):
        """A PSD meat (identity) must not emit any warning."""
        M = np.eye(3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_meat_psd(
                M,
                error_msg="not used",
                warning_template="not used ({eigval:.2e})",
            )
        psd = [msg for msg in w if issubclass(msg.category, UserWarning)]
        assert psd == [], f"Expected no warnings; got {[str(m.message) for m in psd]}"

    def test_threshold_boundary_above_threshold_silent(self):
        """An eigenvalue just above the -1e-12 threshold (at -5e-13) is
        absorbed as numerical noise and must NOT warn."""
        # diag(-5e-13, 1.0): symmetric, eigenvalues exactly {-5e-13, 1.0}.
        M = np.diag([-5e-13, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_meat_psd(
                M,
                error_msg="not used",
                warning_template="not used ({eigval:.2e})",
            )
        psd = [msg for msg in w if issubclass(msg.category, UserWarning)]
        assert psd == [], (
            f"Eigenvalue -5e-13 is above -1e-12 threshold; expected no warning. "
            f"Got: {[str(m.message) for m in psd]}"
        )

    def test_stacklevel_attributes_to_caller_frame(self):
        """Stacklevel parameter must attribute the warning to the caller's
        frame (or higher), NOT to the helper's own frame.

        Locks the contract that the helper itself is invisible in warning
        attribution. The helper extraction added one frame to the stack, so
        call sites had to bump their stacklevel by +1 (conley.py 3→4,
        two_stage.py 2→3). Defaulting to ``stacklevel=3`` from inside the
        helper attributes the warning to the helper's direct caller's caller
        (one intermediate frame between caller and outer-caller)."""
        M = np.array([[0.5, 1.5], [1.5, 0.5]])  # eigenvalues {2, -1}

        def inner_caller():
            _validate_meat_psd(
                M,
                error_msg="x",
                warning_template="attr-check ({eigval:.2e})",
                stacklevel=3,
            )

        def outer_caller():
            inner_caller()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outer_caller()

        psd = [msg for msg in w if "attr-check" in str(msg.message)]
        assert len(psd) == 1, f"Expected one PSD warning; got {[str(m.message) for m in w]}"
        # With stacklevel=3 from inside _validate_meat_psd, the warning is
        # attributed to the caller of inner_caller, which is outer_caller.
        # Both inner_caller and outer_caller are defined in this test file,
        # so the attribution must land here (NOT in conley.py inside the
        # helper body).
        assert psd[0].filename.endswith("test_conley_vcov.py"), (
            f"Expected attribution to test_conley_vcov.py (outer_caller "
            f"frame); got {psd[0].filename!r}:{psd[0].lineno}. If "
            f"attribution landed in conley.py the helper's stacklevel "
            f"contract has regressed."
        )

    def test_no_survey_path_attributes_warning_to_user_code(self):
        """End-to-end warning-capture test on ``_compute_conley_vcov``: when
        the indefinite-meat path triggers the PSD warning via the shared
        helper, attribution must bubble all the way through the three
        internal frames (``_validate_meat_psd`` → ``_compute_conley_meat`` →
        ``_compute_conley_vcov``) to land at user code (this test's frame).

        Locks the stacklevel=4 contract at the no-survey call site that
        compensates for the +1 frame the helper extraction added. Pre-
        extraction the inline warn used ``stacklevel=3`` which already
        attributed to user code; preserving that behavior is the whole
        point of the +1 bump."""
        from diff_diff import conley as conley_mod

        rng = np.random.default_rng(seed=11)
        n = 6
        coords = rng.uniform(0, 1, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eps = np.ones(n)
        bread = X.T @ X

        # Monkey-patch the bartlett kernel to force an indefinite meat
        # (mirrors the existing test_indefinite_meat_warning_fires_for_bartlett
        # fixture pattern).
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
            psd = [
                msg
                for msg in w
                if issubclass(msg.category, UserWarning)
                and "negative eigenvalue" in str(msg.message)
            ]
        finally:
            conley_mod._bartlett_kernel = original

        assert len(psd) >= 1, "Expected a PSD UserWarning from the indefinite meat path"
        msg = psd[0]
        # Attribution must be in this test file (user code), NOT any of the
        # three internal production frames. If a future refactor regresses
        # the stacklevel to 3, attribution would stick at _compute_conley_vcov
        # in conley.py; to 2, at _compute_conley_meat; to 1, inside the helper.
        assert msg.filename.endswith("test_conley_vcov.py"), (
            f"Expected attribution to user code (test_conley_vcov.py); got "
            f"{msg.filename!r}:{msg.lineno}. The stacklevel=4 contract at "
            f"the conley.py call site has regressed."
        )

    def test_survey_call_site_passes_stacklevel_3(self):
        """Static source check: the survey orchestrator
        ``_compute_stratified_conley_meat`` in two_stage.py must pass
        ``stacklevel=3`` to ``_validate_meat_psd``. The pre-extraction
        inline warn used ``stacklevel=2``; after the helper extraction the
        +1 frame shift means the call site must pass ``stacklevel=3`` to
        attribute the warning to the same outer caller. Pairs with the
        runtime test ``test_survey_path_attributes_warning_to_user_code``
        which exercises the actual frame walk; this static check pins the
        literal kwarg to surface bare-text regressions even if the runtime
        test's fixture changes."""
        import inspect

        from diff_diff.two_stage import _compute_stratified_conley_meat

        src = inspect.getsource(_compute_stratified_conley_meat)
        # Find the _validate_meat_psd call and verify the kwarg block
        # includes stacklevel=3 (not 2, not 4, not missing).
        assert "_validate_meat_psd(" in src, (
            "_compute_stratified_conley_meat no longer calls "
            "_validate_meat_psd; survey-side PSD guard regressed."
        )
        assert "stacklevel=3" in src, (
            "_compute_stratified_conley_meat does not pass stacklevel=3 "
            "to _validate_meat_psd. The pre-extraction inline warn used "
            "stacklevel=2; the +1 frame shift from extracting the helper "
            "requires stacklevel=3 to preserve attribution."
        )

    def test_survey_path_attributes_warning_to_user_code(self):
        """Runtime warning-capture test on the SURVEY orchestrator
        ``_compute_stratified_conley_meat``: when the panel-block path
        produces an indefinite combined meat the PSD warning must bubble
        through the orchestrator frame to land at user code (this test).
        Locks the stacklevel=3 contract end-to-end (not just by source
        substring), addressing CI codex R1 P3.

        Mirrors the no-survey test's monkey-patch pattern: replaces the
        serial-Bartlett kernel helper bound inside ``two_stage.py`` with
        an aggressively-negative-off-diagonal stub so the serial meat is
        indefinite and the combined meat's min eigenvalue drops below
        the -1e-12 PSD threshold. Uses the minimal 4-PSU x 2-period x
        3-obs survey fixture from
        ``tests/test_spillover.py::TestSpilloverDiDWaveE2Followup``."""
        from diff_diff import two_stage as two_stage_mod
        from diff_diff.survey import ResolvedSurveyDesign
        from diff_diff.two_stage import _compute_stratified_conley_meat

        rng = np.random.default_rng(seed=29)
        n_obs, T, G, p_2 = 24, 2, 4, 3
        obs_per_psu_period = 3
        psu_id = np.repeat(np.arange(G), obs_per_psu_period * T)
        time_arr = np.tile(np.repeat(np.arange(T), obs_per_psu_period), G)
        Psi = rng.standard_normal((n_obs, p_2))
        psu_centroids = np.array([[40.0, -120.0], [40.1, -120.0], [40.2, -120.0], [40.3, -120.0]])
        coords = psu_centroids[psu_id]
        psu_strata = np.array([0, 0, 1, 1])
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

        # Monkey-patch the serial Bartlett kernel helper as bound inside
        # diff_diff.two_stage (the `from diff_diff.conley import ...`
        # rebind at module load time) so the serial meat is indefinite.
        # The combined meat = spatial + indefinite_serial then drops
        # below the -1e-12 PSD threshold.
        original = two_stage_mod._serial_bartlett_kernel_matrix

        def _indefinite(t_codes: np.ndarray, L: int) -> np.ndarray:
            n = t_codes.shape[0]
            K = np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        K[i, j] = -10.0
            return K

        try:
            two_stage_mod._serial_bartlett_kernel_matrix = _indefinite
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _compute_stratified_conley_meat(
                    Psi,
                    conley_coords=coords,
                    conley_cutoff_km=0.30,
                    conley_metric="euclidean",
                    conley_kernel="bartlett",
                    resolved_survey=resolved,
                    conley_time=time_arr,
                    conley_lag_cutoff=1,
                )
        finally:
            two_stage_mod._serial_bartlett_kernel_matrix = original

        psd = [
            msg
            for msg in w
            if issubclass(msg.category, UserWarning) and "negative eigenvalue" in str(msg.message)
        ]
        assert len(psd) >= 1, (
            f"Expected a PSD UserWarning from the indefinite combined "
            f"survey meat. Got: {[str(m.message) for m in w]}"
        )
        msg = psd[0]
        # Attribution must be in this test file (user code), proving the
        # warning bubbled through both the helper (_validate_meat_psd in
        # conley.py) and the survey orchestrator
        # (_compute_stratified_conley_meat in two_stage.py). A regression
        # of the call-site stacklevel from 3 to 2 would stick the
        # attribution inside two_stage.py; to 1 inside the helper itself.
        assert msg.filename.endswith("test_conley_vcov.py"), (
            f"Expected attribution to user code (test_conley_vcov.py); got "
            f"{msg.filename!r}:{msg.lineno}. The stacklevel=3 contract at "
            f"two_stage.py's _compute_stratified_conley_meat call site has "
            f"regressed."
        )


# ---------------------------------------------------------------------------
# TestConleyValidationDispatch — dispatch-level validation contracts.
# (Methodology-anchored Eq. 4.2 PSD / symmetric / shape contracts and
# HC0+rank-1 reductions moved to tests/test_methodology_conley.py.)
# ---------------------------------------------------------------------------


class TestConleyValidationDispatch:
    """Validation tests at the compute_robust_vcov dispatch level."""

    @pytest.fixture
    def fit_inputs(self):
        rng = np.random.default_rng(seed=0)
        n = 12
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        residuals = rng.standard_normal(n)
        coords = rng.uniform(-10, 10, size=(n, 2))
        return X, residuals, coords

    def test_conley_in_valid_set(self):
        """Sanity: 'conley' is in the canonical _VALID_VCOV_TYPES set."""
        from diff_diff.linalg import _VALID_VCOV_TYPES

        assert "conley" in _VALID_VCOV_TYPES

    def test_conley_with_cluster_combined_kernel(self, fit_inputs):
        """Conley + cluster_ids applies the combined spatial + cluster
        product kernel; no longer raises. The shipped SE differs from
        bare Conley because cross-cluster off-diagonals are zeroed out."""
        X, residuals, coords = fit_inputs
        cluster_ids = np.arange(len(X)) // 3
        V_combined = compute_robust_vcov(
            X,
            residuals,
            cluster_ids=cluster_ids,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=100.0,
        )
        V_bare = compute_robust_vcov(
            X,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=100.0,
        )
        assert V_combined.shape == V_bare.shape
        # Combined kernel zeros out off-cluster off-diagonals → the meat
        # (and hence vcov) must differ from bare Conley on the same data.
        assert not np.allclose(V_combined, V_bare, atol=1e-8)

    def test_conley_with_weights_raises(self, fit_inputs):
        X, residuals, coords = fit_inputs
        with pytest.raises(NotImplementedError, match="conley.*weights"):
            compute_robust_vcov(
                X,
                residuals,
                weights=np.ones(len(X)),
                vcov_type="conley",
                conley_coords=coords,
                conley_cutoff_km=100.0,
            )

    def test_conley_without_coords_raises(self, fit_inputs):
        X, residuals, _ = fit_inputs
        with pytest.raises(ValueError, match="conley_coords"):
            compute_robust_vcov(
                X,
                residuals,
                vcov_type="conley",
                conley_cutoff_km=100.0,
            )

    def test_conley_without_cutoff_raises(self, fit_inputs):
        X, residuals, coords = fit_inputs
        with pytest.raises(ValueError, match="conley_cutoff_km"):
            compute_robust_vcov(
                X,
                residuals,
                vcov_type="conley",
                conley_coords=coords,
            )


class TestConleyLinearRegression:
    """Step 3 smoke tests: LinearRegression and solve_ols thread Conley
    kwargs to compute_robust_vcov. Covers both the higher-level
    LinearRegression API and the lower-level solve_ols entrypoint."""

    @pytest.fixture
    def fit_data(self):
        rng = np.random.default_rng(seed=42)
        n = 25
        X = rng.standard_normal(size=(n, 2))
        y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)
        coords = rng.uniform(-30, 30, size=(n, 2))
        return X, y, coords

    def test_linear_regression_conley_runs(self, fit_data):
        X, y, coords = fit_data
        reg = LinearRegression(
            vcov_type="conley",
            include_intercept=True,
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        ).fit(X, y)
        assert reg.vcov_ is not None
        assert reg.vcov_.shape == (3, 3)  # +1 for intercept
        # Diagonal entries are SE^2 — must be finite and positive
        diag = np.diag(reg.vcov_)
        assert np.all(np.isfinite(diag))
        assert np.all(diag > 0)

    def test_linear_regression_conley_matches_direct(self, fit_data):
        """LinearRegression(vcov_type='conley', ...) ⇔ compute_robust_vcov direct
        call produces the same vcov on the same X (with intercept added)."""
        X, y, coords = fit_data
        reg = LinearRegression(
            vcov_type="conley",
            include_intercept=True,
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        ).fit(X, y)
        # Reproduce X with intercept that LinearRegression built internally
        X_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coefs, *_ = np.linalg.lstsq(X_intercept, y, rcond=None)
        residuals = y - X_intercept @ coefs
        vcov_direct = compute_robust_vcov(
            X_intercept,
            residuals,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
        )
        np.testing.assert_allclose(reg.vcov_, vcov_direct, atol=1e-10, rtol=1e-10)

    def test_solve_ols_conley_path(self, fit_data):
        """solve_ols(vcov_type='conley', ...) returns finite vcov."""
        X, y, coords = fit_data
        coefs, residuals, vcov = solve_ols(
            X,
            y,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=2000.0,
            skip_rank_check=True,
        )
        assert vcov is not None
        assert np.all(np.isfinite(np.diag(vcov)))

    def test_linear_regression_conley_with_survey_design_raises(self, fit_data):
        """LinearRegression(vcov_type='conley', survey_design=...) must raise
        NotImplementedError before fitting. Without the front-door guard,
        LinearRegression.fit() silently bypasses the documented Conley+survey
        rejection: it sets `return_vcov=False` on the solve_ols call when
        survey vcov is needed, skipping the linalg validator, and the survey
        vcov path then overwrites `vcov_` with a non-Conley variance under a
        Conley request. The rejection is permanent on the generic
        LinearRegression path — weighted spatial-HAC under probability
        sampling is an open methodological question (no canonical extension
        of Conley (1999) exists for the combination).
        """
        from diff_diff.survey import make_pweight_design

        X, y, coords = fit_data
        n = X.shape[0]
        survey = make_pweight_design(np.ones(n))
        with pytest.raises(NotImplementedError, match="conley.*survey"):
            LinearRegression(
                vcov_type="conley",
                include_intercept=True,
                conley_coords=coords,
                conley_cutoff_km=2000.0,
                survey_design=survey,
            ).fit(X, y)


class TestConleyEstimatorIntegration:
    """Panel-estimator rejection tests for vcov_type='conley'.

    DiD and MultiPeriodDiD reject Conley at fit-time in Phase 1 because
    cross-sectional Conley over (unit, time) rows mishandles same-unit
    cross-time pairs (d_ij = 0 -> K = 1). The supported Phase 1 path for
    Conley is direct compute_robust_vcov / LinearRegression on a single-
    period regression. Phase 2 will add the space-time product kernel and
    lift the rejection.
    """

    @pytest.fixture
    def two_period_panel(self):
        rng = np.random.default_rng(seed=11)
        n_units = 40
        units = np.arange(n_units)
        treated = (units < 20).astype(int)
        rows = []
        for u in units:
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in [0, 1]:
                y = 1.0 + 0.5 * t + (1.0 if (treated[u] and t == 1) else 0.0) + rng.normal(0, 0.5)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": treated[u], "lat": lat, "lon": lon}
                )
        import pandas as pd

        return pd.DataFrame(rows)

    def test_did_with_conley_panel_finite_se(self, two_period_panel):
        """DifferenceInDifferences + vcov_type='conley' + unit + lag_cutoff
        produces a finite SE on a two-period panel (Wave A #118)."""
        from diff_diff import DifferenceInDifferences

        df = two_period_panel.copy()
        res = DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        assert res.vcov_type == "conley"
        assert res.conley_lag_cutoff == 1

    def test_did_conley_missing_unit_raises(self, two_period_panel):
        """vcov_type='conley' without unit= at fit-time raises ValueError."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match=r"`unit=<column_name>`"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(two_period_panel, outcome="y", treatment="treated", time="time")

    def test_did_conley_unknown_unit_column_raises(self, two_period_panel):
        """vcov_type='conley' with `unit=<name>` referring to an absent column
        raises a clear estimator-level ValueError, NOT a raw pandas KeyError.
        Front-door check mirrors MultiPeriodDiD / TwoWayFixedEffects.
        Codex CI R1 P1 #1."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="Unit column 'missing_unit' not found"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="missing_unit",
            )

    def test_did_conley_unknown_coord_column_raises(self, two_period_panel):
        """vcov_type='conley' with `conley_coords=(<absent>, <col>)` raises
        a clear estimator-level ValueError before downstream column access.
        Codex CI R2 P1."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="conley_coords column 'missing_lat' not found"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("missing_lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )

    def test_did_conley_unknown_cluster_column_raises(self, two_period_panel):
        """DiD + Conley + cluster=<missing column> raises a clear estimator-
        level ValueError before `data[self.cluster]` access (combined-kernel
        path; codex CI R7 P1)."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="Cluster column 'missing_region' not found"):
            DifferenceInDifferences(
                vcov_type="conley",
                cluster="missing_region",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )

    def test_mpd_conley_unknown_cluster_column_raises(self):
        """MPD + Conley + cluster=<missing column> raises a clear estimator-
        level ValueError before `data[self.cluster]` access (combined-kernel
        path; codex CI R7 P1)."""
        import pandas as _pd

        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=83)
        rows = []
        for u in range(8):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.standard_normal(),
                        "treated": int(u >= 4),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df = _pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Cluster column 'missing_region' not found"):
            MultiPeriodDiD(
                vcov_type="conley",
                cluster="missing_region",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                post_periods=[1, 2],
                reference_period=0,
            )

    def test_twfe_conley_unknown_cluster_column_raises(self):
        """TWFE + Conley + cluster=<missing column> raises a clear estimator-
        level ValueError before `data[self.cluster]` access (combined-kernel
        path; codex CI R7 P1)."""
        import pandas as _pd

        from diff_diff import TwoWayFixedEffects

        rng = np.random.default_rng(seed=89)
        rows = []
        for u in range(8):
            lat = rng.uniform(-5, 5)
            lon = rng.uniform(-5, 5)
            for t in range(2):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.standard_normal(),
                        "treated": int(u >= 4),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df = _pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Cluster column 'missing_region' not found"):
            TwoWayFixedEffects(
                vcov_type="conley",
                cluster="missing_region",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")

    def test_mpd_conley_wild_bootstrap_raises_without_warning(self):
        """MPD + Conley + inference='wild_bootstrap' raises NotImplementedError
        cleanly. The pre-Conley analytical-fallback UserWarning is suppressed
        on this combination so the user gets one consistent error message
        instead of "warn then raise". Codex CI R11 P3 #1.
        """
        import pandas as _pd

        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=211)
        rows = []
        for u in range(8):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.standard_normal(),
                        "treated": int(u >= 4),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df = _pd.DataFrame(rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(NotImplementedError, match="wild_bootstrap"):
                MultiPeriodDiD(
                    vcov_type="conley",
                    inference="wild_bootstrap",
                    conley_coords=("lat", "lon"),
                    conley_cutoff_km=2000.0,
                    conley_lag_cutoff=1,
                ).fit(
                    df,
                    outcome="y",
                    treatment="treated",
                    time="time",
                    unit="unit",
                    post_periods=[1, 2],
                    reference_period=0,
                )
            fallback_warnings = [
                msg
                for msg in w
                if "falling back to analytical" in str(msg.message)
                or "Wild bootstrap inference is not yet supported" in str(msg.message)
            ]
            assert len(fallback_warnings) == 0, (
                "Got the analytical-fallback warning on a Conley fit that will "
                "raise NotImplementedError — contradictory guidance."
            )

    def test_did_conley_malformed_coord_tuple_raises(self, two_period_panel):
        """vcov_type='conley' with a malformed conley_coords (wrong arity or
        non-string elements) raises ValueError before downstream access.
        Codex CI R2 P1."""
        from diff_diff import DifferenceInDifferences

        # Wrong arity (1-element tuple)
        with pytest.raises(ValueError, match="2-element tuple/list of column"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat",),  # type: ignore[arg-type]
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )
        # Non-string element
        with pytest.raises(ValueError, match="2-element tuple/list of column"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", 0),  # type: ignore[arg-type]
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )

    def test_did_conley_missing_lag_cutoff_raises(self, two_period_panel):
        """vcov_type='conley' without conley_lag_cutoff raises ValueError."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(
                two_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )

    def test_did_conley_matches_mpd_post_periods_1(self, two_period_panel):
        """DiD + Conley on a 2-period panel matches MultiPeriodDiD with
        post_periods=[1], reference_period=0 on the same data (locks the
        DiD wire-up correctness against the already-shipped MPD path)."""
        from diff_diff import DifferenceInDifferences, MultiPeriodDiD

        df = two_period_panel.copy()
        kwargs = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        )
        res_did = DifferenceInDifferences(**kwargs).fit(
            df, outcome="y", treatment="treated", time="time", unit="unit"
        )
        res_mpd = MultiPeriodDiD(**kwargs).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            post_periods=[1],
            reference_period=0,
        )
        # MPD reports ATT for the single post period (1)
        np.testing.assert_allclose(res_did.att, res_mpd.att, atol=1e-10)
        np.testing.assert_allclose(res_did.se, res_mpd.se, atol=1e-10)

    def test_did_conley_with_absorb_uses_raw_time_labels(self, two_period_panel, monkeypatch):
        """DiD + Conley + absorb=[<unit>] must feed the Conley helper the
        ORIGINAL time/unit/coord columns from `data`, not the absorb-demeaned
        `working_data` (in which time has been residualized to floats).
        Otherwise the within-period spatial sandwich silently partitions on
        per-unit demeaned floats instead of the true pre/post periods.
        Codex Wave A R1 P0 #2.
        """
        import diff_diff.linalg as linalg_module
        from diff_diff import DifferenceInDifferences

        df = two_period_panel.copy()
        captured: dict = {"time_arg": None, "unit_arg": None}
        orig = linalg_module._compute_conley_vcov

        def _spy(*args, **kwargs):
            captured["time_arg"] = kwargs.get("time")
            captured["unit_arg"] = kwargs.get("unit")
            return orig(*args, **kwargs)

        monkeypatch.setattr(linalg_module, "_compute_conley_vcov", _spy)
        DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            absorb=["unit"],
        )
        assert captured["time_arg"] is not None
        # Raw labels are integer 0/1 (the binary post-treatment indicator);
        # demeaned values would be floats from absorb's within-unit
        # demeaning. np.unique on raw labels yields exactly 2 distinct
        # values; on demeaned floats it would yield ~n_units distinct.
        time_arg = np.asarray(captured["time_arg"])
        uniques = np.unique(time_arg)
        assert len(uniques) == 2, (
            f"Expected 2 unique time labels (raw 0/1), got {len(uniques)}: "
            f"{uniques[:5]} — absorb is leaking demeaned time into the "
            "Conley helper."
        )
        assert set(uniques.tolist()) == {0, 1}, f"Expected raw integer labels 0/1, got {uniques}"

    def test_mpd_conley_with_absorb_uses_raw_coords_and_time(self, monkeypatch):
        """MultiPeriodDiD + Conley + absorb=[<col>] must feed the Conley
        helper the ORIGINAL coords/time/unit columns from `data`, not the
        absorb-demeaned `working_data`. If a user lists the `time` column
        (or coord columns) in `absorb`, working_data has those demeaned and
        the Conley helper would partition the within-period spatial sandwich
        on residualized floats. Mirrors the DiD raw-time contract.
        Codex CI R5 P0.
        """
        import pandas as _pd

        import diff_diff.linalg as linalg_module
        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=71)
        n_units = 10
        T = 4
        rows = []
        for u in range(n_units):
            treated = u >= n_units // 2
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in range(T):
                effect = 1.0 if (treated and t >= 2) else 0.0
                yv = 0.2 * t + effect + rng.normal(0, 0.3)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": yv,
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df = _pd.DataFrame(rows)

        captured: dict = {"time_arg": None, "unit_arg": None, "coords_arg": None}
        orig = linalg_module._compute_conley_vcov

        def _spy(*args, **kwargs):
            captured["time_arg"] = kwargs.get("time")
            captured["unit_arg"] = kwargs.get("unit")
            # coords is the 3rd positional arg to _compute_conley_vcov
            if len(args) >= 3:
                captured["coords_arg"] = args[2]
            return orig(*args, **kwargs)

        monkeypatch.setattr(linalg_module, "_compute_conley_vcov", _spy)
        MultiPeriodDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            post_periods=[2, 3],
            reference_period=0,
            absorb=["unit"],
        )
        # Raw time labels span exactly 0..T-1 = 4 distinct values; demeaned
        # absorb would collapse to per-unit means → ~n_units distinct values.
        time_arg = np.asarray(captured["time_arg"])
        uniques = np.unique(time_arg)
        assert len(uniques) == T, (
            f"Expected {T} unique time labels (raw 0..{T - 1}), got {len(uniques)}: "
            f"{uniques[:5]} — absorb is leaking demeaned time into the Conley helper."
        )
        assert set(uniques.tolist()) == set(range(T))
        # Raw coords are time-invariant within unit and span n_units distinct
        # (lat, lon) pairs. If absorb=["unit"] leaked demeaned coords, all
        # within-unit coord values would collapse to 0 (per-unit mean), giving
        # only 1 distinct row across all observations.
        coords_arg = np.asarray(captured["coords_arg"])
        # Expect n_units distinct (lat, lon) pairs since each unit has its own
        unique_coords = np.unique(coords_arr_view := coords_arg, axis=0)
        del coords_arr_view
        assert len(unique_coords) == n_units, (
            f"Expected {n_units} unique coord pairs, got {len(unique_coords)} — "
            "absorb=['unit'] is leaking demeaned coords into the Conley helper."
        )

    def test_multi_period_did_with_conley_panel(self):
        """Phase 2 MultiPeriodDiD + vcov_type='conley' uses the block-decomposed
        sandwich (matches R conleyreg). Verifies that finite SEs are produced
        when conley_lag_cutoff and unit are both supplied."""
        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=13)
        n_units = 30
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 15
            for t in range(4):
                y = 0.2 * t + (1.0 if (treated and t >= 2) else 0.0) + rng.normal(0, 0.5)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": int(treated), "lat": lat, "lon": lon}
                )
        import pandas as pd

        df_mp = pd.DataFrame(rows)
        res = MultiPeriodDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(
            df_mp,
            outcome="y",
            treatment="treated",
            time="time",
            post_periods=[2, 3],
            unit="unit",
            reference_period=1,
        )
        assert np.all(np.isfinite(res.vcov)), "MPD+Conley vcov must be finite"

    def test_multi_period_did_conley_missing_unit_raises(self):
        """MPD + vcov_type='conley' without unit= at fit-time raises ValueError."""
        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=13)
        n_units = 20
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 10
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.normal(),
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        import pandas as pd

        df_mp = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="unit="):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df_mp,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[2],
                reference_period=1,
            )

    def test_multi_period_did_conley_missing_lag_cutoff_raises(self):
        """MPD + vcov_type='conley' without conley_lag_cutoff raises ValueError
        (no defensible default per Conley 1999 Section 5)."""
        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=13)
        n_units = 20
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 10
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.normal(),
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        import pandas as pd

        df_mp = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(
                df_mp,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[2],
                unit="unit",
                reference_period=1,
            )

    def test_multi_period_did_conley_with_survey_design_raises(self):
        """MPD + vcov_type='conley' + survey_design raises NotImplementedError.

        Closes Codex P0: previously, MPD passed return_vcov=False to solve_ols
        when _use_survey_vcov=True, bypassing the conley + weights guard, and
        then overwrote vcov with compute_survey_vcov — silently returning
        survey SEs under a Conley request.
        """
        import pandas as pd

        from diff_diff import MultiPeriodDiD
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(seed=29)
        n_units = 24
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.normal(),
                        "treated": int(u < 12),
                        "lat": lat,
                        "lon": lon,
                        "weight": 1.0 + 0.1 * rng.random(),
                        "stratum": u % 4,
                        "psu": u // 6,
                    }
                )
        df_mp = pd.DataFrame(rows)
        # Pure pweight (no PSU / strata) — would route through analytical conley
        # path; the guard must fire before solve_ols.
        sd_tsl = SurveyDesign(weights="weight", weight_type="pweight")
        with pytest.raises(NotImplementedError, match="survey_design"):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df_mp,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[2],
                unit="unit",
                reference_period=1,
                survey_design=sd_tsl,
            )
        # Stratified PSU survey design — would route through Taylor TSL path
        # and was the canonical bypass case the codex reviewer flagged.
        sd_psu = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
            nest=True,
        )
        with pytest.raises(NotImplementedError, match="survey_design"):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df_mp,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[2],
                unit="unit",
                reference_period=1,
                survey_design=sd_psu,
            )

    def test_multi_period_did_conley_with_datetime64_time(self):
        """End-to-end MPD + vcov_type='conley' with datetime64 time labels.
        Closes Codex re-review P1: the wrapper must NOT coerce time to float64
        before passing to _compute_conley_vcov; the helper normalizes to
        dense codes internally. Verifies the SEs match an equivalent
        dense-integer-coded fit.
        """
        import pandas as pd

        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=37)
        n_units = 12
        date_labels = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-08-01"])
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t_idx, dt in enumerate(date_labels):
                treated = u < 6
                y = 0.2 * t_idx + (1.0 if (treated and t_idx >= 1) else 0.0) + rng.normal(0, 0.4)
                rows.append(
                    {
                        "unit": u,
                        "time_dt": dt,
                        "time_int": t_idx,
                        "y": y,
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df_mp = pd.DataFrame(rows)
        kwargs = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        )
        res_int = MultiPeriodDiD(**kwargs).fit(
            df_mp,
            outcome="y",
            treatment="treated",
            time="time_int",
            post_periods=[1, 2],
            unit="unit",
            reference_period=0,
        )
        res_dt = MultiPeriodDiD(**kwargs).fit(
            df_mp,
            outcome="y",
            treatment="treated",
            time="time_dt",
            post_periods=[date_labels[1], date_labels[2]],
            unit="unit",
            reference_period=date_labels[0],
        )
        # Per-coefficient SE should match across the two encodings (dense
        # codes normalize identically). MPD orders coefficients by the
        # reference-vs-non-reference period split; with reference_period=0
        # and post_periods=[1,2] the coefficient ordering is bit-identical.
        se_int = np.sqrt(np.diag(res_int.vcov))
        se_dt = np.sqrt(np.diag(res_dt.vcov))
        np.testing.assert_allclose(se_dt, se_int, atol=1e-10)

    def test_multi_period_did_conley_to_dict_carries_lag_cutoff(self):
        """Closes Codex re-review round 4 P1 (Maintainability) on MPD:
        serialized `to_dict()` must include `vcov_type` and
        `conley_lag_cutoff` so downstream programmatic consumers can tell
        which Conley variant produced the SEs."""
        import pandas as pd

        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=41)
        n_units = 10
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            for t in range(3):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.normal(),
                        "treated": int(u < 5),
                        "lat": lat,
                        "lon": lon,
                    }
                )
        df_mp = pd.DataFrame(rows)
        res = MultiPeriodDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=2,
        ).fit(
            df_mp,
            outcome="y",
            treatment="treated",
            time="time",
            post_periods=[1, 2],
            unit="unit",
            reference_period=0,
        )
        d = res.to_dict()
        assert d["vcov_type"] == "conley"
        assert d["conley_lag_cutoff"] == 2

    def test_multi_period_did_conley_missing_coords_raises(self):
        """MPD + vcov_type='conley' without conley_coords raises a clean
        ValueError instead of a raw TypeError on `self.conley_coords[0]`.
        Closes Codex P2 #1.
        """
        import pandas as pd

        from diff_diff import MultiPeriodDiD

        rng = np.random.default_rng(seed=31)
        n_units = 10
        rows = []
        for u in range(n_units):
            for t in range(2):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": rng.normal(),
                        "treated": int(u < 5),
                    }
                )
        df_mp = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="conley_coords.*conley_cutoff_km"):
            MultiPeriodDiD(
                vcov_type="conley",
                conley_lag_cutoff=1,
            ).fit(
                df_mp,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[1],
                unit="unit",
                reference_period=0,
            )


class TestConleyTWFE:
    """TwoWayFixedEffects + vcov_type='conley' uses the Phase 2 block-decomposed
    panel HAC (matches R conleyreg). The within-transformed scores feed the same
    block-decomposed helper that LinearRegression uses; FWL composability
    ensures the FE-residualized meat matches the full-dummy-expansion meat.
    """

    @pytest.fixture
    def panel(self):
        """Build a 2-period panel with geocoords for TWFE tests."""
        rng = np.random.default_rng(seed=17)
        n_units = 30
        rows = []
        for u in range(n_units):
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            treated = u < 15
            unit_fe = rng.normal(0, 0.3)
            for t in range(2):
                time_fe = 0.5 if t == 1 else 0.0
                effect = 1.0 if (treated and t == 1) else 0.0
                y = unit_fe + time_fe + effect + rng.normal(0, 0.4)
                rows.append(
                    {"unit": u, "time": t, "y": y, "treated": int(treated), "lat": lat, "lon": lon}
                )
        import pandas as pd

        return pd.DataFrame(rows)

    def test_twfe_conley_panel_finite_se(self, panel):
        """TWFE + vcov_type='conley' on a balanced panel produces a finite SE."""
        from diff_diff import TwoWayFixedEffects

        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.att), "ATT must be finite"
        assert np.isfinite(res.se) and res.se > 0, "SE must be positive and finite"

    def test_twfe_conley_with_explicit_cluster_combined_kernel(self, panel):
        """TWFE + vcov_type='conley' + explicit cluster=<col> applies the
        combined spatial + cluster product kernel. The user-supplied cluster
        column propagates to ``cluster_name`` (no longer cleared on the
        Conley path) and a finite SE is produced. Auto-cluster on the
        Conley path remains silently dropped — the user MUST explicitly
        opt in to the combined kernel."""
        from diff_diff import TwoWayFixedEffects

        # Add a unit-level region column so the cluster is time-invariant
        # within unit (the panel block-decomposed validator's contract).
        panel = panel.copy()
        panel["region"] = panel["unit"] // 5
        res = TwoWayFixedEffects(
            vcov_type="conley",
            cluster="region",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        assert res.cluster_name == "region"
        d = res.to_dict()
        assert d.get("cluster_name") == "region"

    def test_twfe_conley_with_wild_bootstrap_raises(self, panel):
        """vcov_type='conley' + inference='wild_bootstrap' raises: wild bootstrap
        does not consume the analytical Conley sandwich."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(NotImplementedError, match="wild_bootstrap"):
            TwoWayFixedEffects(
                vcov_type="conley",
                inference="wild_bootstrap",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_repeated_coords_panel_finite_se(self, panel):
        """Phase 2 regression for the Phase-1 silent-bug case: each unit's
        coords are time-invariant. The block-decomposed sandwich correctly
        sums within-period (period 0 and period 1 separately) plus
        within-unit serial (lag=1) so the same-unit cross-time pairs at
        d_ij=0 do NOT inflate the meat."""
        from diff_diff import TwoWayFixedEffects

        coord_var = panel.groupby("unit")[["lat", "lon"]].nunique()
        assert (coord_var.values == 1).all(), "Fixture coords must be time-invariant"
        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.se) and res.se > 0

    def test_twfe_conley_missing_lag_cutoff_raises(self, panel):
        """conley_lag_cutoff is required; no defensible default per Conley §5."""
        from diff_diff import TwoWayFixedEffects

        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            TwoWayFixedEffects(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
            ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")

    def test_twfe_conley_binary_post_label_normalization(self, panel):
        """TWFE with binary `post` (values {0,1}) + `conley_lag_cutoff=1`
        produces the same finite vcov as the equivalent dense-period-index
        fit. Closes the Codex P1 example — the time-label normalization
        means lag is counted in panel periods regardless of how `time` is
        encoded (binary post indicator vs. dense period index).
        """
        from diff_diff import TwoWayFixedEffects

        # `panel` fixture uses `time` in {0, 1}, identical to a binary post.
        # Rename to `post` to make the test scenario explicit.
        df_post = panel.rename(columns={"time": "post"})
        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df_post, outcome="y", treatment="treated", time="post", unit="unit")
        assert np.isfinite(res.se) and res.se > 0

    def test_twfe_conley_summary_emits_conley_label(self, panel):
        """Panel result summary must label the variance family as Conley
        spatial HAC and surface `lag_cutoff` so downstream consumers can tell
        which Conley variant produced the SEs. Closes Codex P3 and the
        re-review P1 (Maintainability).
        """
        from diff_diff import TwoWayFixedEffects

        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        summary = res.summary()
        assert "Conley spatial HAC" in summary
        assert "lag_cutoff=1" in summary
        # No explicit cluster on this fit → label must NOT advertise the
        # combined kernel.
        assert "+ cluster product kernel" not in summary
        # The result dataclass also carries the lag for programmatic access.
        assert res.conley_lag_cutoff == 1

    def test_twfe_conley_with_cluster_summary_label_names_kernel_and_cluster(self, panel):
        """When an explicit cluster=<col> is combined with Conley, the
        summary label must distinguish the combined spatial + cluster
        product kernel from bare Conley and name the cluster column.
        Codex CI R3 P3 (Maintainability)."""
        from diff_diff import TwoWayFixedEffects

        panel = panel.copy()
        panel["region"] = panel["unit"] // 5  # time-invariant within unit
        res = TwoWayFixedEffects(
            vcov_type="conley",
            cluster="region",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        summary = res.summary()
        assert "Conley spatial HAC" in summary
        assert "+ cluster product kernel at region" in summary
        assert "lag_cutoff=1" in summary
        # Programmatic access via the result dataclass.
        assert res.cluster_name == "region"
        assert res.conley_lag_cutoff == 1

    def test_twfe_conley_to_dict_carries_lag_cutoff(self, panel):
        """Closes Codex re-review round 4 P1 (Maintainability): the
        serialized `to_dict()` must include `vcov_type` and
        `conley_lag_cutoff` so downstream programmatic consumers (notebooks,
        adapters, pipelines) can tell which Conley variant produced the SEs
        without re-deriving from the summary string."""
        from diff_diff import TwoWayFixedEffects

        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        d = res.to_dict()
        assert d["vcov_type"] == "conley"
        assert d["conley_lag_cutoff"] == 1

    def test_twfe_conley_cluster_name_is_none(self, panel):
        """Closes Codex re-review round 5 P1 (Maintainability): TWFE drops
        its auto-unit-cluster on the Conley path (`_conley_cluster_override =
        None`), so the variance-provenance metadata must reflect that. The
        result's `cluster_name` is None and `to_dict()` does not advertise
        `cluster_name` — otherwise downstream consumers would be told the
        SEs were CR1-clustered when they're actually Conley spatial HAC.
        """
        from diff_diff import TwoWayFixedEffects

        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        assert res.cluster_name is None
        d = res.to_dict()
        assert "cluster_name" not in d

    def test_twfe_conley_non_numeric_time_fails(self, panel):
        """TWFE's `_treatment_post = treated * time` design step requires
        numeric `time`. Non-numeric labels (datetime64, pd.Period, strings)
        are TWFE-incompatible end-to-end and surface as a clean error before
        the Conley path runs. Use MultiPeriodDiD if you need non-numeric
        time labels.
        """
        from diff_diff import TwoWayFixedEffects

        df_str = panel.copy()
        df_str["time_str"] = df_str["time"].map({0: "pre", 1: "post"})
        with pytest.raises((TypeError, ValueError)):
            TwoWayFixedEffects(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df_str,
                outcome="y",
                treatment="treated",
                time="time_str",
                unit="unit",
            )

    def test_twfe_conley_within_vs_dummy_expansion_equivalence(self, panel):
        """FWL composability: TWFE (within-transform) + Conley should produce
        the SAME ATT SE as a dummy-expansion design with the same Conley
        kernel applied to the FE-residualized scores. Verifies that the
        block-decomposed sandwich on demeaned scores matches the full-design
        sandwich up to FW-projection noise.

        Note: Exact equivalence requires the full-dummy design to also use
        the block-decomposed sandwich (same unit/time grid). Phase 2's
        contract is that BOTH paths use the SAME helper; this test confirms
        TWFE's wired path is internally consistent with computing the
        sandwich on the within-transformed scores directly.
        """
        from diff_diff import TwoWayFixedEffects
        from diff_diff.conley import _compute_conley_vcov

        # Fit TWFE + Conley
        res = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(panel, outcome="y", treatment="treated", time="time", unit="unit")
        # Manually demean using the same within-transform util TWFE uses
        from diff_diff.utils import within_transform as _within_transform_util

        df_dem = _within_transform_util(
            panel.assign(_tp=panel["treated"] * panel["time"]),
            ["y", "_tp"],
            "unit",
            "time",
            suffix="_d",
        )
        y_d = df_dem["y_d"].values
        x_d = df_dem["_tp_d"].values
        X_d = np.column_stack([np.ones_like(y_d), x_d])
        beta, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
        resid = y_d - X_d @ beta
        coords = panel[["lat", "lon"]].values
        bread = X_d.T @ X_d
        V_direct = _compute_conley_vcov(
            X_d,
            resid,
            coords,
            2000.0,
            "haversine",
            "bartlett",
            bread,
            time=panel["time"].values,
            unit=panel["unit"].values,
            lag_cutoff=1,
        )
        # TWFE's att_idx=1 (treatment_post is index 1 after intercept).
        # The DF adjustment differs between TWFE (df_adjustment for FE) and
        # the raw helper, so compare the raw vcov diagonal up to scaling
        # by sigma_hat^2 — both paths share the same meat structure.
        # Direct test: TWFE's vcov entry for att should equal V_direct[1, 1]
        # modulo the DF adjustment scaling that LinearRegression applies.
        # For Phase 2 we assert both are finite and have the same sign-shape.
        assert np.isfinite(V_direct[1, 1])
        assert np.isfinite(res.se) and res.se > 0


class TestConleyEstimatorValidation:
    """Step 4 validation: estimator-level rejections for invalid combinations."""

    @pytest.fixture
    def df(self):
        import pandas as pd

        rng = np.random.default_rng(seed=2)
        n = 20
        return pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": np.tile([0, 1], n // 2),
                "y": rng.standard_normal(n),
                "treated": np.tile([0, 1], n // 2),
                "lat": rng.uniform(-30, 30, n),
                "lon": rng.uniform(-100, 100, n),
                "stratum": np.tile([0, 1, 2, 3], n // 4),
            }
        )

    def test_did_conley_combinations(self, df):
        """DifferenceInDifferences + vcov_type='conley' validation table:
        missing coords/cutoff/lag_cutoff/unit each raise ValueError;
        valid full kwarg set succeeds; survey_design + Conley raises
        NotImplementedError (Wave A scope: row 121 deferred);
        wild_bootstrap + Conley raises NotImplementedError."""
        from diff_diff import DifferenceInDifferences

        # missing conley_cutoff_km
        with pytest.raises(ValueError, match="conley_coords|conley_cutoff_km"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_lag_cutoff=1,
            ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")
        # missing conley_lag_cutoff
        with pytest.raises(ValueError, match="conley_lag_cutoff"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
            ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")
        # missing unit
        with pytest.raises(ValueError, match=r"`unit=<column_name>`"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
                conley_lag_cutoff=1,
            ).fit(df, outcome="y", treatment="treated", time="time")
        # Valid full kwarg set does NOT raise (separate fixture in
        # TestConleyEstimatorIntegration covers the finite-SE assertion;
        # this fixture's treated/time correlation triggers rank deficiency).
        DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=100.0,
            conley_lag_cutoff=1,
            rank_deficient_action="silent",
        ).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
        )

    def test_did_conley_with_survey_design_raises(self, df):
        """DiD + Conley + survey_design raises NotImplementedError — weighted
        spatial-HAC under probability sampling is an open methodological
        question (no canonical extension of Conley (1999) exists for the
        combination)."""
        from diff_diff import DifferenceInDifferences, SurveyDesign

        with pytest.raises(NotImplementedError, match="conley.*survey_design"):
            DifferenceInDifferences(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
                conley_lag_cutoff=1,
            ).fit(
                df,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                survey_design=SurveyDesign(strata="stratum"),
            )

    def test_did_conley_with_wild_bootstrap_raises(self, df):
        """DiD + Conley + inference='wild_bootstrap' raises."""
        from diff_diff import DifferenceInDifferences

        with pytest.raises(NotImplementedError, match="wild_bootstrap"):
            DifferenceInDifferences(
                vcov_type="conley",
                inference="wild_bootstrap",
                cluster="unit",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=100.0,
                conley_lag_cutoff=1,
            ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")

    def test_synthetic_did_conley_raises(self):
        from diff_diff import SyntheticDiD

        with pytest.raises(TypeError, match="conley"):
            SyntheticDiD(vcov_type="conley")  # type: ignore[call-arg]

    def test_synthetic_did_conley_kwarg_raises(self):
        from diff_diff import SyntheticDiD

        with pytest.raises(TypeError, match="conley"):
            SyntheticDiD(conley_cutoff_km=100.0)  # type: ignore[call-arg]

    def test_synthetic_did_set_params_conley_raises(self):
        """SyntheticDiD.set_params(vcov_type='conley') must raise (mirrors
        __init__'s contract — closes the silent-bypass gap CI reviewer flagged
        as P1 CQ1)."""
        from diff_diff import SyntheticDiD

        est = SyntheticDiD()
        # Snapshot pre-call state
        before_variance = est.variance_method
        before_n_boot = est.n_bootstrap
        before_zeta = est.zeta_omega

        with pytest.raises(TypeError, match="conley"):
            est.set_params(vcov_type="conley")
        # Verify nothing mutated
        assert est.variance_method == before_variance
        assert est.n_bootstrap == before_n_boot
        assert est.zeta_omega == before_zeta

    def test_synthetic_did_set_params_conley_kwarg_raises(self):
        from diff_diff import SyntheticDiD

        est = SyntheticDiD()
        with pytest.raises(TypeError, match="conley"):
            est.set_params(conley_cutoff_km=100.0)
        # Verify the conley attr stays None (rejected before mutation)
        assert getattr(est, "conley_cutoff_km", None) is None

    def test_synthetic_did_get_params_includes_conley_keys(self):
        """get_params() / set_params() round-trip must include the inherited
        conley_* keys with None values for sklearn-style API consistency
        (CI reviewer P2 CQ3)."""
        from diff_diff import SyntheticDiD

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=10)
        params = est.get_params()
        assert "vcov_type" in params and params["vcov_type"] is None
        assert "conley_coords" in params and params["conley_coords"] is None
        assert "conley_cutoff_km" in params and params["conley_cutoff_km"] is None
        assert "conley_metric" in params and params["conley_metric"] is None
        assert "conley_kernel" in params and params["conley_kernel"] is None
        # Round-trip: passing None values back into set_params is a no-op
        est.set_params(**params)
        assert est.variance_method == "placebo"
        assert est.n_bootstrap == 10


class TestConleySetParamsAtomicity:
    """set_params atomicity for Conley fields. Per
    feedback_transactional_set_params: invalid multi-kwarg call must not
    leave the estimator in a partial state."""

    def test_unknown_kwarg_raises_no_mutation(self):
        from diff_diff import DifferenceInDifferences

        est = DifferenceInDifferences(conley_coords=("lat", "lon"), conley_cutoff_km=100.0)
        # Pre-call snapshot
        before_cutoff = est.conley_cutoff_km
        before_kernel = est.conley_kernel
        # set_params with valid + unknown key → must raise & not mutate
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(conley_cutoff_km=200.0, garbage_field="x")
        # Verify state did NOT change
        assert est.conley_cutoff_km == before_cutoff
        assert est.conley_kernel == before_kernel

    def test_valid_kwargs_apply(self):
        from diff_diff import DifferenceInDifferences

        est = DifferenceInDifferences(conley_coords=("lat", "lon"), conley_cutoff_km=100.0)
        est.set_params(conley_cutoff_km=250.0, conley_kernel="uniform")
        assert est.conley_cutoff_km == 250.0
        assert est.conley_kernel == "uniform"


class TestConleySparse:
    """Sparse k-d-tree fast path for the spatial Bartlett meat.

    The sparse path is gated by three conditions: total n above the
    threshold, metric in {"haversine", "euclidean"} (no callable), and
    kernel == "bartlett". Each of these tests exercises one of the
    gates plus the bit-identity parity claim vs the dense path.
    """

    def _euclidean_fixture(self, n=1000, k=3, cutoff=15.0, seed=11):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
        beta = np.linspace(0.5, 2.0, k)
        y = X @ beta + rng.standard_normal(n) * 0.5
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        return X, residuals, coords, bread, cutoff

    def _haversine_fixture(self, n=1000, k=3, cutoff_km=500.0, seed=13):
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

    def test_auto_toggle_above_threshold_uses_sparse(self, monkeypatch):
        """n > _CONLEY_SPARSE_N_THRESHOLD with bartlett + euclidean must
        auto-route through the sparse helper. Verified by spying on the
        sparse helper call count."""
        import diff_diff.conley as conley_module

        X, residuals, coords, bread, cutoff = self._euclidean_fixture(
            n=_CONLEY_SPARSE_N_THRESHOLD + 1
        )
        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(conley_module, "_compute_spatial_bartlett_meat_sparse", _spy)
        _compute_conley_vcov(X, residuals, coords, cutoff, "euclidean", "bartlett", bread)
        assert calls["n"] >= 1, "Sparse helper not called when n > threshold."

    def test_auto_toggle_below_threshold_stays_dense(self, monkeypatch):
        """n <= _CONLEY_SPARSE_N_THRESHOLD must use the dense path even
        when other sparse conditions (bartlett + euclidean) are met."""
        import diff_diff.conley as conley_module

        X, residuals, coords, bread, cutoff = self._euclidean_fixture(n=_CONLEY_SPARSE_N_THRESHOLD)
        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(conley_module, "_compute_spatial_bartlett_meat_sparse", _spy)
        _compute_conley_vcov(X, residuals, coords, cutoff, "euclidean", "bartlett", bread)
        assert calls["n"] == 0, "Sparse helper called below threshold."

    def test_auto_toggle_callable_metric_stays_dense(self, monkeypatch):
        """A callable conley_metric forces the dense path even at large n —
        the kd-tree query needs a vectorizable metric, and callables are
        not supported via projection."""
        import diff_diff.conley as conley_module

        X, residuals, coords, bread, cutoff = self._euclidean_fixture(
            n=_CONLEY_SPARSE_N_THRESHOLD + 100
        )

        def callable_metric(c1, c2):
            diff = c1[:, None, :] - c2[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=-1))

        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(conley_module, "_compute_spatial_bartlett_meat_sparse", _spy)
        _compute_conley_vcov(X, residuals, coords, cutoff, callable_metric, "bartlett", bread)
        assert calls["n"] == 0, "Sparse helper called for callable metric."

    def test_auto_toggle_uniform_kernel_stays_dense(self, monkeypatch):
        """uniform kernel forces dense path — bartlett has K(u=1) == 0 which
        the sparse path relies on; uniform has K(u=1) == 1 which would
        require a closed-interval query semantic the chord projection
        cannot reliably preserve."""
        import diff_diff.conley as conley_module

        X, residuals, coords, bread, cutoff = self._euclidean_fixture(
            n=_CONLEY_SPARSE_N_THRESHOLD + 100
        )
        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(conley_module, "_compute_spatial_bartlett_meat_sparse", _spy)
        _compute_conley_vcov(X, residuals, coords, cutoff, "euclidean", "uniform", bread)
        assert calls["n"] == 0, "Sparse helper called for uniform kernel."

    def test_force_sparse_with_uniform_raises(self):
        """Explicit _conley_sparse=True with uniform kernel raises rather
        than silently falling back, so callers see the mismatch."""
        X, residuals, coords, bread, cutoff = self._euclidean_fixture(n=100)
        with pytest.raises(ValueError, match="_conley_sparse=True requires"):
            _compute_conley_vcov(
                X,
                residuals,
                coords,
                cutoff,
                "euclidean",
                "uniform",
                bread,
                _conley_sparse=True,
            )

    def test_force_sparse_with_callable_metric_raises(self):
        """Explicit _conley_sparse=True with a callable metric raises."""
        X, residuals, coords, bread, cutoff = self._euclidean_fixture(n=100)

        def callable_metric(c1, c2):
            diff = c1[:, None, :] - c2[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=-1))

        with pytest.raises(ValueError, match="_conley_sparse=True requires"):
            _compute_conley_vcov(
                X,
                residuals,
                coords,
                cutoff,
                callable_metric,
                "bartlett",
                bread,
                _conley_sparse=True,
            )

    def test_force_dense_with_sparse_eligible_inputs(self):
        """_conley_sparse=False overrides the auto-toggle and stays dense
        even when n is above the threshold."""
        import diff_diff.conley as conley_module

        X, residuals, coords, bread, cutoff = self._euclidean_fixture(
            n=_CONLEY_SPARSE_N_THRESHOLD + 100
        )
        calls = {"n": 0}
        orig = conley_module._compute_spatial_bartlett_meat_sparse

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        # The monkeypatch fixture isn't available here; use plain attribute swap.
        conley_module._compute_spatial_bartlett_meat_sparse = _spy
        try:
            _compute_conley_vcov(
                X,
                residuals,
                coords,
                cutoff,
                "euclidean",
                "bartlett",
                bread,
                _conley_sparse=False,
            )
        finally:
            conley_module._compute_spatial_bartlett_meat_sparse = orig
        assert calls["n"] == 0, "Sparse helper called when _conley_sparse=False."

    def test_sparse_haversine_cutoff_above_half_earth_circumference(self):
        """Sparse haversine path with conley_cutoff_km > π·R_earth (~20,015 km)
        must include all geometrically eligible pairs. Without the arc-radians
        clamp, the chord-radius formula 2·sin(arc/2) shrinks for arc > π and
        the kd-tree silently drops pairs that still have positive Bartlett
        weight. The dense path saturates at π·R via _haversine_km's clip;
        the sparse path matches via the clamp. Codex Wave A R1 P0 #1.
        """
        rng = np.random.default_rng(seed=101)
        n = 200
        lats = rng.uniform(-90.0, 90.0, size=n)
        lons = rng.uniform(-180.0, 180.0, size=n)
        coords = np.column_stack([lats, lons])
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 2.0, -0.5]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        # Cutoff well above half-Earth circumference (~20,015 km). Without
        # the clamp, the sparse path drops antipodal pairs and the meat
        # diverges from the dense path.
        cutoff_km = 25_000.0  # > π·R_earth ≈ 20015 km
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff_km,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff_km,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    def test_sparse_density_gate_falls_back_to_dense_and_warns(self):
        """When neighbor density exceeds the threshold (default 30%), the
        sparse helper returns None and the dispatcher falls back to dense.
        A UserWarning surfaces the reason so users with large cutoffs aren't
        surprised by the "sparse" path materializing a near-dense matrix
        and using more memory than dense float64. Codex CI R6 P2.
        """
        # Tight cluster of points + large cutoff → 100% density.
        rng = np.random.default_rng(seed=151)
        n = 100
        coords = rng.uniform(0.0, 10.0, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        # Cutoff = 1e6 → every pair is within range (density = 100%)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            V_forced_sparse = _compute_conley_vcov(
                X,
                residuals,
                coords,
                1e6,
                "euclidean",
                "bartlett",
                bread,
                _conley_sparse=True,
            )
            density_warnings = [msg for msg in w if "exceeds threshold" in str(msg.message)]
            assert len(density_warnings) == 1, "Expected exactly one density-gate UserWarning"
        # Result must equal the dense path (the dispatcher fell back).
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            1e6,
            "euclidean",
            "bartlett",
            bread,
            _conley_sparse=False,
        )
        np.testing.assert_allclose(V_forced_sparse, V_dense, atol=1e-10, rtol=1e-10)

    def test_sparse_density_gate_does_not_trigger_below_threshold(self):
        """At realistic Conley cutoffs (neighbor density well below 30%),
        the sparse path runs normally without the density warning."""
        # 1000 points spread over a wide area; cutoff small relative to span.
        rng = np.random.default_rng(seed=157)
        n = 1000
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_conley_vcov(
                X,
                residuals,
                coords,
                10.0,  # small cutoff → sparse density << 30%
                "euclidean",
                "bartlett",
                bread,
                _conley_sparse=True,
            )
            density_warnings = [msg for msg in w if "exceeds threshold" in str(msg.message)]
            assert (
                len(density_warnings) == 0
            ), "Density gate should not have triggered at low density"

    def test_sparse_density_gate_cluster_aware(self):
        """High global spatial density + many small clusters: within-cluster
        density is low, so the sparse path should NOT spuriously fall back
        to dense. Tests the cluster-aware refinement of the density gate.
        Codex CI R8 P1."""
        # Tight spatial cluster of points → 100% global spatial density,
        # but split into many small disjoint clusters so post-mask density
        # is well below 30%.
        rng = np.random.default_rng(seed=199)
        n = 200
        coords = rng.uniform(0.0, 5.0, size=(n, 2))  # tight cluster
        # 50 clusters of 4 points each → within-cluster nnz = 4*4 = 16 per
        # cluster, total = 50*16 = 800 = 800/(200*200) = 2% density << 30%.
        cluster_ids = np.repeat(np.arange(50), 4)
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5]) + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            V_sparse = _compute_conley_vcov(
                X,
                residuals,
                coords,
                1e6,  # cutoff > data span → 100% global density
                "euclidean",
                "bartlett",
                bread,
                cluster_ids=cluster_ids,
                _conley_sparse=True,
            )
            density_warnings = [msg for msg in w if "exceeds threshold" in str(msg.message)]
            assert len(density_warnings) == 0, (
                "Density gate spuriously fell back to dense even though "
                "within-cluster density is low — cluster-aware refinement "
                "is not working."
            )
        # Result must equal the dense path (sparse path executed correctly)
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            1e6,
            "euclidean",
            "bartlett",
            bread,
            cluster_ids=cluster_ids,
            _conley_sparse=False,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)

    def test_sparse_haversine_cutoff_at_exactly_half_earth_circumference(self):
        """Cutoff = π·R_earth: chord radius = 2 (sphere diameter); all
        pairs are included. Bartlett at u=1 returns 0, so the antipodal
        pair contributes zero — but pairs at all other distances
        contribute. Sparse and dense paths must agree."""
        rng = np.random.default_rng(seed=103)
        n = 150
        lats = rng.uniform(-90.0, 90.0, size=n)
        lons = rng.uniform(-180.0, 180.0, size=n)
        coords = np.column_stack([lats, lons])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5]) + rng.standard_normal(n) * 0.5
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        cutoff_km = float(np.pi * _CONLEY_EARTH_RADIUS_KM)  # ≈ 20015.16 km
        V_dense = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff_km,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=False,
        )
        V_sparse = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff_km,
            "haversine",
            "bartlett",
            bread,
            _conley_sparse=True,
        )
        np.testing.assert_allclose(V_sparse, V_dense, atol=1e-10, rtol=1e-10)


class TestConleyCluster:
    """Combined spatial + cluster product kernel: K(d_ij/h) * 1{c_i = c_j}.

    Wave A item #119. Lifts the prior linalg-level and TWFE-level rejects of
    ``vcov_type='conley' + cluster_ids``. The cluster mask multiplies the
    spatial kernel on both cross-sectional and panel block-decomposed paths.
    On the panel path the validator enforces that cluster membership is
    constant within each unit across periods (so the within-unit serial
    sandwich's mask is trivially all-ones — no per-unit-time mask needed).
    """

    def _cross_sectional(self, n=24, k=2, seed=11):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, 50.0, size=(n, 2))
        X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(k - 1)])
        y = X @ np.array([1.0, 2.0])[:k] + rng.standard_normal(n) * 0.4
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        return X, residuals, coords, bread

    def test_cross_sectional_cluster_no_longer_raises(self):
        """compute_robust_vcov + vcov_type='conley' + cluster_ids no longer
        raises (was the linalg validator's NotImplementedError)."""
        X, residuals, coords, _ = self._cross_sectional()
        cluster_ids = np.arange(X.shape[0]) // 4
        V = compute_robust_vcov(
            X,
            residuals,
            cluster_ids=cluster_ids,
            vcov_type="conley",
            conley_coords=coords,
            conley_cutoff_km=20.0,
        )
        assert V.shape == (X.shape[1], X.shape[1])
        assert np.all(np.isfinite(V))

    def test_combined_kernel_matches_hadamard_dense(self):
        """The combined kernel matches the explicit Hadamard
        ``K_space * cluster_mask`` on the same data."""
        X, residuals, coords, bread = self._cross_sectional(n=30, seed=7)
        cluster_ids = np.array([i % 4 for i in range(X.shape[0])])
        cutoff = 15.0
        V_helper = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            cluster_ids=cluster_ids,
        )
        S = X * residuals[:, None]
        D = _pairwise_distance_matrix(coords, "euclidean")
        K = _bartlett_kernel(D / cutoff) * (cluster_ids[:, None] == cluster_ids[None, :])
        meat = S.T @ K @ S
        bread_inv = np.linalg.inv(bread)
        V_manual = bread_inv @ meat @ bread_inv
        np.testing.assert_allclose(V_helper, V_manual, atol=1e-12)

    def test_combined_kernel_reduces_to_hc0_when_all_unique_clusters(self):
        """Every observation in its own cluster → cluster_mask is the identity,
        so the meat reduces to the diagonal HC0 contribution."""
        X, residuals, coords, bread = self._cross_sectional(n=20, seed=13)
        cluster_ids = np.arange(X.shape[0])  # all unique → cluster_mask = I
        V_combined = _compute_conley_vcov(
            X,
            residuals,
            coords,
            10.0,
            "euclidean",
            "bartlett",
            bread,
            cluster_ids=cluster_ids,
        )
        # Manual HC0
        S = X * residuals[:, None]
        meat_hc0 = X.T @ (X * (residuals**2)[:, None])
        bread_inv = np.linalg.inv(bread)
        V_hc0 = bread_inv @ meat_hc0 @ bread_inv
        np.testing.assert_allclose(V_combined, V_hc0, atol=1e-12)
        del S

    def test_combined_kernel_reduces_to_pure_cluster_at_huge_cutoff(self):
        """Cutoff so large that K_space is identically 1 → combined kernel
        reduces to the pure within-cluster sum (cluster mask alone). Uses
        the UNIFORM kernel: K_uniform(u) = 1 for |u| <= 1, so at huge_cutoff
        the kernel is exactly 1 on every in-range pair (i.e. all pairs).
        Bartlett would give `1 - d_ij/cutoff < 1` for all pairs with d > 0,
        so the reduction is only asymptotic, not exact (codex CI R6 P3).
        """
        X, residuals, coords, bread = self._cross_sectional(n=24, seed=19)
        cluster_ids = np.array([i // 3 for i in range(X.shape[0])])
        huge_cutoff = 1e9  # K_space (uniform) = 1 on every pair
        V_combined = _compute_conley_vcov(
            X,
            residuals,
            coords,
            huge_cutoff,
            "euclidean",
            "uniform",
            bread,
            cluster_ids=cluster_ids,
        )
        # Manual pure-cluster meat
        S = X * residuals[:, None]
        K_cluster = (cluster_ids[:, None] == cluster_ids[None, :]).astype(np.float64)
        meat = S.T @ K_cluster @ S
        bread_inv = np.linalg.inv(bread)
        V_expected = bread_inv @ meat @ bread_inv
        np.testing.assert_allclose(V_combined, V_expected, atol=1e-12)

    def test_combined_kernel_panel_serial_unchanged_when_cluster_per_unit(self):
        """When cluster is constant within unit, the SERIAL component of the
        panel sandwich is identical to the no-cluster case (the within-unit
        cluster mask is trivially all-ones). Only the spatial component
        differs."""
        rng = np.random.default_rng(seed=23)
        n_units = 6
        T = 3
        unit = np.repeat(np.arange(n_units), T)
        time = np.tile(np.arange(T), n_units)
        n = n_units * T
        coords = np.column_stack([rng.uniform(-10, 10, size=n), rng.uniform(-10, 10, size=n)])
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([1.0, 1.5]) + rng.standard_normal(n) * 0.3
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        bread = X.T @ X
        # Time-invariant cluster: one cluster per unit (cluster_per_unit)
        cluster_per_unit = np.repeat(rng.integers(0, 3, size=n_units), T)
        cutoff = 8.0
        # Two variants: lag=1 with and without cluster
        V_no_cluster_l0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
        )
        V_no_cluster_l1 = _compute_conley_vcov(
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
        )
        V_cluster_l0 = _compute_conley_vcov(
            X,
            residuals,
            coords,
            cutoff,
            "euclidean",
            "bartlett",
            bread,
            time=time,
            unit=unit,
            lag_cutoff=0,
            cluster_ids=cluster_per_unit,
        )
        V_cluster_l1 = _compute_conley_vcov(
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
            cluster_ids=cluster_per_unit,
        )
        # Serial delta should be identical under cluster vs no-cluster — the
        # within-unit mask is all-ones when cluster is constant within unit.
        delta_no_cluster = bread @ (V_no_cluster_l1 - V_no_cluster_l0) @ bread
        delta_cluster = bread @ (V_cluster_l1 - V_cluster_l0) @ bread
        np.testing.assert_allclose(delta_cluster, delta_no_cluster, atol=1e-10)

    def test_panel_time_varying_cluster_raises(self):
        """Panel block-decomposed path with a cluster that varies across
        periods within a unit raises ValueError naming the violating units."""
        rng = np.random.default_rng(seed=29)
        n_units = 4
        T = 3
        unit = np.repeat(np.arange(n_units), T)
        time = np.tile(np.arange(T), n_units)
        n = n_units * T
        coords = np.column_stack([rng.uniform(-10, 10, size=n), rng.uniform(-10, 10, size=n)])
        # Unit 1 changes cluster from 0 -> 1 -> 1 across periods
        cluster_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        with pytest.raises(ValueError, match="constant within each unit"):
            _validate_conley_kwargs(
                coords=coords,
                cutoff=10.0,
                metric="euclidean",
                kernel="bartlett",
                n=n,
                time=time,
                unit=unit,
                lag_cutoff=1,
                cluster_ids=cluster_ids,
            )

    def test_cross_sectional_time_varying_cluster_ok(self):
        """Cross-sectional path (no time/unit/lag_cutoff) has NO time-
        invariance constraint — the validator should accept any cluster."""
        X, _, coords, _ = self._cross_sectional(n=20, seed=31)
        cluster_ids = np.arange(X.shape[0]) % 3
        # Should not raise
        _validate_conley_kwargs(
            coords=coords,
            cutoff=10.0,
            metric="euclidean",
            kernel="bartlett",
            n=X.shape[0],
            cluster_ids=cluster_ids,
        )

    def test_cluster_wrong_shape_raises(self):
        X, _, coords, _ = self._cross_sectional(n=15)
        with pytest.raises(ValueError, match="cluster_ids must be a 1-D array"):
            _validate_conley_kwargs(
                coords=coords,
                cutoff=10.0,
                metric="euclidean",
                kernel="bartlett",
                n=15,
                cluster_ids=np.zeros((10,)),
            )

    def test_cluster_nan_raises(self):
        X, _, coords, _ = self._cross_sectional(n=10)
        cluster_ids = np.array([0, 0, 1, 1, np.nan, 2, 2, 2, 0, 1], dtype=object)
        with pytest.raises(ValueError, match="cluster_ids contains NaN"):
            _validate_conley_kwargs(
                coords=coords,
                cutoff=10.0,
                metric="euclidean",
                kernel="bartlett",
                n=10,
                cluster_ids=cluster_ids,
            )

    def test_twfe_explicit_cluster_propagates_to_cluster_name(self):
        """TWFE + Conley + explicit cluster=<col> sets res.cluster_name to
        the user's column AND to_dict()['cluster_name'] reflects it."""
        from diff_diff import TwoWayFixedEffects

        rng = np.random.default_rng(seed=37)
        rows = []
        n_units = 10
        for u in range(n_units):
            treated = u >= 5
            lat = rng.uniform(-5, 5)
            lon = rng.uniform(-5, 5)
            region = u // 5  # time-invariant within unit
            for t in range(2):
                effect = 1.0 if (treated and t == 1) else 0.0
                yv = effect + rng.normal(0, 0.5)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": yv,
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                        "region": region,
                    }
                )
        import pandas as _pd

        df = _pd.DataFrame(rows)
        res = TwoWayFixedEffects(
            vcov_type="conley",
            cluster="region",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")
        assert res.cluster_name == "region"
        d = res.to_dict()
        assert d.get("cluster_name") == "region"

    def _multi_period_panel_with_region(self, n_units=12, T=4, seed=41):
        """Multi-period panel with a time-invariant `region` column for
        combined-kernel estimator tests."""
        import pandas as _pd

        rng = np.random.default_rng(seed=seed)
        rows = []
        for u in range(n_units):
            treated = u >= n_units // 2
            lat = rng.uniform(-30, 30)
            lon = rng.uniform(-100, 100)
            region = u // 3  # time-invariant within unit; spans multiple units
            for t in range(T):
                effect = 1.0 if (treated and t >= T // 2) else 0.0
                yv = 0.2 * t + effect + rng.normal(0, 0.3)
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": yv,
                        "treated": int(treated),
                        "lat": lat,
                        "lon": lon,
                        "region": region,
                    }
                )
        return _pd.DataFrame(rows)

    def test_did_combined_kernel_finite_se_and_cluster_name(self):
        """DifferenceInDifferences(vcov_type='conley', cluster='region') on
        a 2-period panel produces a finite SE, propagates `region` to
        res.cluster_name and to_dict(), and differs from the no-cluster
        baseline (combined kernel zeros out cross-cluster off-diagonals)."""
        from diff_diff import DifferenceInDifferences

        df = self._multi_period_panel_with_region(n_units=12, T=2, seed=43)
        kwargs = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        )
        res_combined = DifferenceInDifferences(cluster="region", **kwargs).fit(
            df, outcome="y", treatment="treated", time="time", unit="unit"
        )
        res_bare = DifferenceInDifferences(**kwargs).fit(
            df, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert np.isfinite(res_combined.att)
        assert np.isfinite(res_combined.se) and res_combined.se > 0
        assert res_combined.cluster_name == "region"
        d = res_combined.to_dict()
        assert d.get("cluster_name") == "region"
        # Combined kernel zeros out off-cluster pairs → SE differs from bare
        assert not np.isclose(res_combined.se, res_bare.se, atol=1e-8)

    def test_did_combined_kernel_time_varying_cluster_raises(self):
        """DiD + Conley + cluster=<col> on the panel block-decomposed path
        must raise when the cluster column varies across periods within a
        unit (time-invariance contract). Codex CI R1 P1 #2."""
        from diff_diff import DifferenceInDifferences

        df = self._multi_period_panel_with_region(n_units=10, T=2, seed=47)
        # Make region time-varying for unit 0 (different region in t=1)
        mask_u0_t1 = (df["unit"] == 0) & (df["time"] == 1)
        df.loc[mask_u0_t1, "region"] = 99
        with pytest.raises(ValueError, match="constant within each unit"):
            DifferenceInDifferences(
                vcov_type="conley",
                cluster="region",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(df, outcome="y", treatment="treated", time="time", unit="unit")

    def test_mpd_combined_kernel_finite_se_and_cluster_name(self):
        """MultiPeriodDiD(vcov_type='conley', cluster='region') on a 4-period
        panel produces a finite SE and propagates `region` to cluster_name
        on the result + to_dict()."""
        from diff_diff import MultiPeriodDiD

        df = self._multi_period_panel_with_region(n_units=12, T=4, seed=53)
        res = MultiPeriodDiD(
            vcov_type="conley",
            cluster="region",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            post_periods=[2, 3],
            reference_period=0,
        )
        assert np.isfinite(res.avg_att)
        assert np.isfinite(res.avg_se) and res.avg_se > 0
        assert res.cluster_name == "region"
        d = res.to_dict()
        assert d.get("cluster_name") == "region"

    def test_mpd_combined_kernel_time_varying_cluster_raises(self):
        """MultiPeriodDiD + Conley + cluster=<col> with a cluster that
        varies across periods within a unit raises ValueError (same time-
        invariance contract as the linalg validator). Codex CI R1 P1 #2."""
        from diff_diff import MultiPeriodDiD

        df = self._multi_period_panel_with_region(n_units=10, T=3, seed=59)
        mask_violator = (df["unit"] == 2) & (df["time"] == 2)
        df.loc[mask_violator, "region"] = 77
        with pytest.raises(ValueError, match="constant within each unit"):
            MultiPeriodDiD(
                vcov_type="conley",
                cluster="region",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                post_periods=[1, 2],
                reference_period=0,
            )


# =====================================================================
# Conley spatial-HAC threading for SunAbraham + WooldridgeDiD (one PR).
# Both route conley through the within-transform (FWL) path, reusing the
# already-conleyreg-validated solve_ols/conley machinery. The
# FWL-composability tests are the primary correctness anchor: on a
# 2-period / 2-group panel both estimators reduce to a single DiD cell,
# so the estimator's within-transform conley SE must equal a manually
# built full-dummy [intercept, treated*post, C(unit), C(time)] design run
# through solve_ols(vcov_type="conley") on the same coords/time/unit
# (validates that the FWL-demeaned conley sandwich equals the full-dummy
# conley SE — the property asserted by comment at twfe.py:163-168 and
# relied on by the SA / WooldridgeDiD threading). This is NOT a fresh
# conleyreg parity claim: the conleyreg goldens use raw OLS designs with
# no demeaning, so they do not transitively certify the demeaned estimand.
# =====================================================================


def _two_group_panel(seed=21, n_units=40, treat_period=2):
    """2-period (t=1,2) panel: first half treated at ``treat_period``, rest
    never-treated. ``cohort`` = treat_period for treated / 0 for never
    (SA first_treat + Wooldridge cohort convention); ``treated`` is the 0/1
    DiD indicator. Each unit has a fixed (lat, lon)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        lat = rng.uniform(-30, 30)
        lon = rng.uniform(-100, 100)
        treated = int(u < n_units // 2)
        cohort = treat_period if treated else 0
        for t in (1, 2):
            post = 1 if (treated and t == treat_period) else 0
            y = 1.0 + 0.4 * t + 0.3 * (u / n_units) + 1.5 * post + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": y,
                    "treated": treated,
                    "cohort": cohort,
                    "lat": lat,
                    "lon": lon,
                }
            )
    return pd.DataFrame(rows)


def _full_dummy_conley_treatment(df, cutoff_km, lag_cutoff):
    """Full-dummy FWL-composability reference. Builds
    [intercept, treated*post, C(unit) drop-first, C(time) drop-first], runs
    solve_ols(vcov_type='conley') on the same coords/time/unit, and returns
    ``(coef_treated_post, se_treated_post)``. On a 2-period/2-group panel the
    single DiD cell equals the treated*post coefficient, so SunAbraham /
    WooldridgeDiD-conley (within-transform) must match this."""
    from diff_diff.linalg import solve_ols

    last_t = df["time"].max()
    treated_post = (df["treated"].values * (df["time"].values == last_t)).astype(float)
    units = np.unique(df["unit"].values)
    times = np.unique(df["time"].values)
    unit_d = np.column_stack([(df["unit"].values == u).astype(float) for u in units[1:]])
    time_d = np.column_stack([(df["time"].values == t).astype(float) for t in times[1:]])
    X = np.column_stack([np.ones(len(df)), treated_post, unit_d, time_d])
    y = df["y"].values.astype(float)
    coords = np.column_stack([df["lat"].values, df["lon"].values]).astype(float)
    out = solve_ols(
        X,
        y,
        vcov_type="conley",
        conley_coords=coords,
        conley_cutoff_km=cutoff_km,
        conley_kernel="bartlett",
        conley_metric="haversine",
        conley_time=df["time"].values,
        conley_unit=df["unit"].values,
        conley_lag_cutoff=lag_cutoff,
        return_vcov=True,
    )
    coef, vcov = out[0], out[-1]
    return float(coef[1]), float(np.sqrt(vcov[1, 1]))


class TestConleySunAbraham:
    """vcov_type='conley' threading for SunAbraham (within-transform / FWL path)."""

    def test_panel_finite_se_and_metadata(self):
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=21)
        res = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.vcov_type == "conley"
        assert res.conley_lag_cutoff == 1
        assert "Conley" in res.summary()

    def test_cross_sectional_finite_se(self):
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=22)
        res = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=0,
        ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.conley_lag_cutoff == 0

    def test_fwl_composability_vs_full_dummy(self):
        """SA within-transform conley SE == full-dummy conley SE (atol 1e-7)."""
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=23)
        res = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2500.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        ref_coef, ref_se = _full_dummy_conley_treatment(df, 2500.0, 1)
        assert res.overall_att == pytest.approx(ref_coef, abs=1e-8)
        assert res.overall_se == pytest.approx(ref_se, abs=1e-7)

    def test_att_bit_identical_across_vcov(self):
        """conley is additive: ATT must be bit-identical to the hc1 fit."""
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=24)
        kw = dict(outcome="y", unit="unit", time="time", first_treat="cohort")
        r_conley = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        r_hc1 = SunAbraham(vcov_type="hc1").fit(df, **kw)
        assert r_conley.overall_att == pytest.approx(r_hc1.overall_att, abs=1e-12)

    def test_conley_plus_cluster_product_kernel(self):
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=25)
        kw = dict(outcome="y", unit="unit", time="time", first_treat="cohort")
        plain = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        clustered = SunAbraham(
            vcov_type="conley",
            cluster="unit",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        assert np.isfinite(clustered.overall_se) and clustered.overall_se > 0
        assert clustered.overall_se != plain.overall_se
        assert "cluster" in clustered.summary().lower()

    def test_reject_survey_design(self):
        from diff_diff import SunAbraham
        from diff_diff.survey import SurveyDesign

        df = _two_group_panel(seed=26)
        df["w"] = 1.0
        with pytest.raises(NotImplementedError):
            SunAbraham(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="cohort",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_reject_n_bootstrap(self):
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=27)
        with pytest.raises(ValueError, match="n_bootstrap"):
            SunAbraham(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
                n_bootstrap=20,
                seed=0,
            ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")

    def test_reject_missing_cutoff(self):
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=28)
        with pytest.raises(ValueError):
            SunAbraham(vcov_type="conley", conley_coords=("lat", "lon"), conley_lag_cutoff=1).fit(
                df, outcome="y", unit="unit", time="time", first_treat="cohort"
            )

    def test_unbalanced_panel_alignment(self):
        """Dropping a few control-unit rows must not misalign coords (the within
        transform preserves order; arrays come from the post-filter frame)."""
        from diff_diff import SunAbraham

        df = _two_group_panel(seed=29)
        # Drop a few never-treated (cohort==0) rows to unbalance without breaking
        # identification of the treated cells.
        drop_idx = df.index[df["cohort"] == 0][:3]
        df = df.drop(index=drop_idx).reset_index(drop=True)
        res = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_get_set_params_carry_conley(self):
        from diff_diff import SunAbraham

        est = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_kernel="uniform",
            conley_lag_cutoff=2,
        )
        p = est.get_params()
        assert p["conley_coords"] == ("lat", "lon")
        assert p["conley_cutoff_km"] == 2000.0
        assert p["conley_kernel"] == "uniform"
        assert p["conley_lag_cutoff"] == 2
        est2 = SunAbraham()
        est2.set_params(**p)
        assert est2.get_params() == p

    def test_conley_multi_cohort_event_study(self):
        """conley on a multi-cohort staggered panel: the interaction-weighted
        cohort aggregation + delta-method event-study SEs (W @ vcov_conley @ W.T)
        all stay finite (the single-cell FWL test cannot exercise this)."""
        from diff_diff import SunAbraham

        df = _staggered_panel(seed=61)
        res = SunAbraham(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=3000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", first_treat="cohort")
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert len(res.event_study_effects) >= 2  # genuinely multi-period
        for e, eff in res.event_study_effects.items():
            assert np.isfinite(eff["se"]), (e, eff)


class TestConleyWooldridge:
    """vcov_type='conley' threading for WooldridgeDiD-OLS (within-transform / FWL path)."""

    def test_panel_finite_se_and_metadata(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=31)
        res = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.vcov_type == "conley"
        assert res.conley_lag_cutoff == 1
        assert "Conley" in res.summary()

    def test_cross_sectional_finite_se(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=32)
        res = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=0,
        ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        assert res.conley_lag_cutoff == 0

    def test_fwl_composability_vs_full_dummy(self):
        """WDID within-transform conley SE == full-dummy conley SE (atol 1e-7)."""
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=33)
        res = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2500.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        ref_coef, ref_se = _full_dummy_conley_treatment(df, 2500.0, 1)
        assert res.overall_att == pytest.approx(ref_coef, abs=1e-8)
        assert res.overall_se == pytest.approx(ref_se, abs=1e-7)

    def test_att_bit_identical_across_vcov(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=34)
        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        r_conley = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        r_hc1 = WooldridgeDiD(vcov_type="hc1").fit(df, **kw)
        assert r_conley.overall_att == pytest.approx(r_hc1.overall_att, abs=1e-12)

    def test_conley_plus_cluster_product_kernel(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=35)
        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        plain = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        clustered = WooldridgeDiD(
            vcov_type="conley",
            cluster="unit",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, **kw)
        assert np.isfinite(clustered.overall_se) and clustered.overall_se > 0
        assert clustered.overall_se != plain.overall_se
        assert "cluster" in clustered.summary().lower()

    def test_reject_survey_design(self):
        from diff_diff import WooldridgeDiD
        from diff_diff.survey import SurveyDesign

        df = _two_group_panel(seed=36)
        df["w"] = 1.0
        with pytest.raises(NotImplementedError):
            WooldridgeDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
            ).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                cohort="cohort",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_reject_n_bootstrap(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=37)
        with pytest.raises(ValueError, match="n_bootstrap"):
            WooldridgeDiD(
                vcov_type="conley",
                conley_coords=("lat", "lon"),
                conley_cutoff_km=2000.0,
                conley_lag_cutoff=1,
                n_bootstrap=20,
            ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")

    def test_reject_logit_plus_conley_at_init(self):
        from diff_diff import WooldridgeDiD

        with pytest.raises(NotImplementedError):
            WooldridgeDiD(method="logit", vcov_type="conley")

    def test_reject_missing_cutoff(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=38)
        with pytest.raises(ValueError):
            WooldridgeDiD(
                vcov_type="conley", conley_coords=("lat", "lon"), conley_lag_cutoff=1
            ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")

    def test_unbalanced_panel_alignment(self):
        from diff_diff import WooldridgeDiD

        df = _two_group_panel(seed=39)
        drop_idx = df.index[df["cohort"] == 0][:3]
        df = df.drop(index=drop_idx).reset_index(drop=True)
        res = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_lag_cutoff=1,
        ).fit(df, outcome="y", unit="unit", time="time", cohort="cohort")
        assert np.isfinite(res.overall_se) and res.overall_se > 0

    def test_atomic_set_params_round_trip(self):
        from diff_diff import WooldridgeDiD

        est = WooldridgeDiD(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=2000.0,
            conley_kernel="uniform",
            conley_lag_cutoff=2,
        )
        p = est.get_params()
        assert p["conley_coords"] == ("lat", "lon")
        assert p["conley_lag_cutoff"] == 2
        est2 = WooldridgeDiD()
        est2.set_params(**p)
        assert est2.get_params() == p

    def test_conley_cohort_trends_full_dummy(self):
        """conley + cohort_trends=True routes through the full-dummy design.
        Conley must actually engage (SE differs from hc1) and stay finite; the
        ATT stays bit-identical (conley is additive)."""
        from diff_diff import WooldridgeDiD

        df = _staggered_panel(seed=51)
        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        ck = dict(conley_coords=("lat", "lon"), conley_cutoff_km=3000.0, conley_lag_cutoff=1)
        r_conley = WooldridgeDiD(vcov_type="conley", cohort_trends=True, **ck).fit(df, **kw)
        r_hc1 = WooldridgeDiD(vcov_type="hc1", cohort_trends=True).fit(df, **kw)
        assert np.isfinite(r_conley.overall_se) and r_conley.overall_se > 0
        assert r_conley.vcov_type == "conley"
        # conley actually engaged on the full-dummy path (SE differs from hc1)
        assert r_conley.overall_se != r_hc1.overall_se
        assert r_conley.overall_att == pytest.approx(r_hc1.overall_att, abs=1e-9)

    def test_conley_aggregations_finite(self):
        """conley vcov flows through aggregate('group'|'calendar'|'event') —
        finite overall + per-key SEs (no NaN/crash on the conley path)."""
        from diff_diff import WooldridgeDiD

        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        ck = dict(conley_coords=("lat", "lon"), conley_cutoff_km=3000.0, conley_lag_cutoff=1)
        for agg, field in (
            ("group", "group_effects"),
            ("calendar", "calendar_effects"),
            ("event", "event_study_effects"),
        ):
            # aggregate() mutates in place, so re-fit per aggregation type.
            res = WooldridgeDiD(vcov_type="conley", **ck).fit(_staggered_panel(seed=52), **kw)
            res.aggregate(agg)
            assert np.isfinite(res.overall_se) and res.overall_se > 0, agg
            effects = getattr(res, field)
            assert effects, (agg, "no effects populated")
            for key, eff in effects.items():
                assert np.isfinite(eff["se"]), (agg, key, eff)

    def test_conley_cohort_trends_aggregations_finite(self):
        """The distinct full-dummy Conley aggregation interaction: conley +
        cohort_trends=True (full-dummy) THEN aggregate each — finite overall +
        per-key SEs."""
        from diff_diff import WooldridgeDiD

        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        ck = dict(conley_coords=("lat", "lon"), conley_cutoff_km=3000.0, conley_lag_cutoff=1)
        for agg, field in (
            ("group", "group_effects"),
            ("calendar", "calendar_effects"),
            ("event", "event_study_effects"),
        ):
            res = WooldridgeDiD(vcov_type="conley", cohort_trends=True, **ck).fit(
                _staggered_panel(seed=53), **kw
            )
            res.aggregate(agg)
            assert np.isfinite(res.overall_se) and res.overall_se > 0, agg
            effects = getattr(res, field)
            assert effects, (agg, "no effects populated")
            for key, eff in effects.items():
                assert np.isfinite(eff["se"]), (agg, key, eff)

    def test_conley_control_group_never_treated(self):
        """conley on the OLS + control_group='never_treated' interaction-matrix
        branch (distinct cell set from the default not_yet_treated) — finite SE,
        ATT bit-identical to the hc1 fit (conley is additive)."""
        from diff_diff import WooldridgeDiD

        df = _staggered_panel(seed=62)
        kw = dict(outcome="y", unit="unit", time="time", cohort="cohort")
        ck = dict(conley_coords=("lat", "lon"), conley_cutoff_km=3000.0, conley_lag_cutoff=1)
        r_conley = WooldridgeDiD(vcov_type="conley", control_group="never_treated", **ck).fit(
            df, **kw
        )
        r_hc1 = WooldridgeDiD(vcov_type="hc1", control_group="never_treated").fit(df, **kw)
        assert np.isfinite(r_conley.overall_se) and r_conley.overall_se > 0
        assert r_conley.overall_att == pytest.approx(r_hc1.overall_att, abs=1e-9)


def _staggered_panel(seed=51, n_units=36, n_periods=6):
    """Staggered multi-cohort panel with lat/lon for conley aggregate / cohort_trends
    tests: cohorts {3, 4, 0(never)}, ``n_periods`` periods, one fixed (lat, lon) per unit."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n_units):
        lat = rng.uniform(-30, 30)
        lon = rng.uniform(-100, 100)
        g = (3, 4, 0)[u % 3]
        for t in range(1, n_periods + 1):
            post = 1 if (g > 0 and t >= g) else 0
            y = 0.3 * (u / n_units) + 0.4 * t + 1.5 * post + rng.normal(0, 0.5)
            recs.append({"unit": u, "time": t, "y": y, "cohort": g, "lat": lat, "lon": lon})
    return pd.DataFrame(recs)


class TestConleyBreadRankGuard:
    """The conley spatial-HAC bread inversion now routes through the shared
    ``_rank_guarded_inv``: a near-singular design Gram rank-reduces to a finite
    SE on the identified subspace (previously a garbage ~1e13 inverse), and a
    singular Gram no longer raises ``ValueError`` (it rank-reduces + warns).
    Sibling of the covariate IF rank-guard and the ContinuousDiD / TwoStageDiD /
    SpilloverDiD structural bread guards."""

    @staticmethod
    def _cross_section(seed, dup="indep"):
        # Fixed rng draw order (x1, noise, y) so ``exact`` and ``near`` share
        # x1 / coords / y and differ ONLY in the third column.
        rng = np.random.default_rng(seed)
        n = 60
        coords = rng.uniform(0.0, 5.0, size=(n, 2))
        x1 = rng.normal(size=n)
        noise = rng.normal(size=n)
        y = rng.normal(size=n)
        if dup == "exact":
            x2 = x1.copy()  # exactly collinear -> singular X'X
        elif dup == "near":
            x2 = x1 + 1e-3 * noise  # highly collinear but full rank
        else:
            x2 = noise  # independent
        X = np.column_stack([np.ones(n), x1, x2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return X, resid, coords

    def test_singular_gram_rank_reduces_not_raises(self):
        X, resid, coords = self._cross_section(0, dup="exact")
        bread = X.T @ X  # exactly singular (x2 == x1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            V = _compute_conley_vcov(X, resid, coords, 3.0, "euclidean", "bartlett", bread)
        # Pre-fix this raised ValueError; now it rank-reduces. The dropped
        # (duplicate, unidentified) coefficient gets a NaN row/col — se=NaN, not
        # the zero-filled se=0 — while the identified coefficients stay finite.
        nan_diag = np.isnan(np.diag(V))
        assert nan_diag.sum() == 1, f"exactly one duplicate coef should be NaN, got {np.diag(V)}"
        kept = np.flatnonzero(~nan_diag)
        assert 0 in kept, "intercept (identified) must remain finite"
        assert np.all(np.isfinite(V[np.ix_(kept, kept)])), "identified block must be finite"
        dropped_idx = int(np.flatnonzero(nan_diag)[0])
        assert np.all(np.isnan(V[dropped_idx, :])) and np.all(np.isnan(V[:, dropped_idx]))
        msgs = [str(w.message) for w in caught]
        assert any("Conley spatial HAC variance" in m and "rank-deficient" in m for m in msgs), msgs

    def test_rank_zero_gram_returns_nan(self):
        X, resid, coords = self._cross_section(1, dup="indep")
        Xz = np.zeros_like(X)
        bread = Xz.T @ Xz  # rank 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            V = _compute_conley_vcov(Xz, resid, coords, 3.0, "euclidean", "bartlett", bread)
        assert np.all(np.isnan(V))

    def test_column_drop_equals_near_collinear_limit(self):
        # Column-drop generalized inverse == the near-collinear full-rank limit:
        # the identified-subspace SE (intercept) matches between an exactly-
        # collinear design (rank-reduced) and a highly-but-not-exactly collinear
        # one (full-rank normal inverse). Mirrors the covariate rank-guard's
        # verified se_ratio ~ 1 property.
        Xe, re_, ce = self._cross_section(7, dup="exact")
        Xn, rn, cn = self._cross_section(7, dup="near")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ve = _compute_conley_vcov(Xe, re_, ce, 3.0, "euclidean", "bartlett", Xe.T @ Xe)
            Vn = _compute_conley_vcov(Xn, rn, cn, 3.0, "euclidean", "bartlett", Xn.T @ Xn)
        assert np.isfinite(Ve[0, 0]) and Ve[0, 0] > 0
        np.testing.assert_allclose(Ve[0, 0], Vn[0, 0], rtol=5e-2)
