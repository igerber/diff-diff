"""
B-spline utilities for continuous Difference-in-Differences estimation.

Provides basis construction, evaluation, and derivative computation for
the dose-response curve estimation in ContinuousDiD.
"""

import warnings

import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "build_bspline_basis",
    "bspline_design_matrix",
    "bspline_derivative_design_matrix",
    "default_dose_grid",
    "saturated_dose_levels",
    "saturated_design_matrix",
    "saturated_derivative_design_matrix",
    "SATURATED_TOL",
]

# Tolerance used consistently for BOTH discrete dose-level construction and
# level matching in the saturated (discrete-treatment) basis. Using the same
# value in both places guarantees a dose can never fall between two
# near-duplicate levels or double-match one.
SATURATED_TOL = 1e-9


def build_bspline_basis(dose, degree=3, num_knots=0):
    """
    Construct B-spline knot vector from positive dose values.

    Interior knots are placed at quantiles of the dose distribution,
    matching R's ``choose_knots_quantile`` convention.

    Parameters
    ----------
    dose : array-like
        Positive dose values from treated units.
    degree : int, default=3
        Degree of the B-spline (3 = cubic).
    num_knots : int, default=0
        Number of interior knots.

    Returns
    -------
    knots : np.ndarray
        Full knot vector with boundary clamping.
    degree : int
        The B-spline degree (echoed back for convenience).
    """
    dose = np.asarray(dose, dtype=float)
    d_L = float(np.min(dose))
    d_U = float(np.max(dose))

    if num_knots > 0:
        # Interior knots at evenly-spaced quantiles of dose distribution
        probs = np.linspace(0, 1, num_knots + 2)[1:-1]
        interior_knots = np.quantile(dose, probs)
    else:
        interior_knots = np.array([])

    # Full knot vector: clamped at boundaries
    knots = np.concatenate(
        [
            np.repeat(d_L, degree + 1),
            interior_knots,
            np.repeat(d_U, degree + 1),
        ]
    )

    return knots, degree


def bspline_design_matrix(x, knots, degree, include_intercept=True):
    """
    Evaluate B-spline basis functions at points ``x``.

    To match R's ``splines2::bSpline(intercept=FALSE)`` plus an explicit
    intercept column: drop the first B-spline column and prepend a
    column of ones.

    Parameters
    ----------
    x : array-like
        Evaluation points, shape ``(n,)``.
    knots : np.ndarray
        Full knot vector (from :func:`build_bspline_basis`).
    degree : int
        B-spline degree.
    include_intercept : bool, default=True
        If True, drop first B-spline column and prepend intercept column.

    Returns
    -------
    np.ndarray
        Design matrix, shape ``(n, n_cols)``.
    """
    x = np.asarray(x, dtype=float)

    # scipy requires evaluation within [knots[degree], knots[-(degree+1)]]
    # Clamp to boundary knots to avoid extrapolation issues
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    x_clamped = np.clip(x, t_min, t_max)

    # Sparse design matrix from scipy, convert to dense
    B = BSpline.design_matrix(x_clamped, knots, degree).toarray()

    if include_intercept:
        # Drop first B-spline column, prepend intercept
        B = np.column_stack([np.ones(len(x)), B[:, 1:]])

    return B


def bspline_derivative_design_matrix(x, knots, degree, include_intercept=True):
    """
    Evaluate first derivatives of B-spline basis functions at points ``x``.

    Parameters
    ----------
    x : array-like
        Evaluation points, shape ``(n,)``.
    knots : np.ndarray
        Full knot vector.
    degree : int
        B-spline degree.
    include_intercept : bool, default=True
        If True, drop derivative of first B-spline (replaced by intercept
        whose derivative is 0) and prepend a zeros column.

    Returns
    -------
    np.ndarray
        Derivative design matrix, shape ``(n, n_cols)``.
    """
    x = np.asarray(x, dtype=float)

    # Number of basis functions
    n_basis = len(knots) - degree - 1

    # Clamp evaluation points to boundary
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    x_clamped = np.clip(x, t_min, t_max)

    # Build derivative for each basis function
    dB = np.zeros((len(x), n_basis))

    # Check if knot vector is degenerate (all identical, e.g. single dose)
    if knots[0] == knots[-1]:
        # All knots identical: derivatives are all zero — this is a
        # mathematically well-defined degenerate case (single dose value
        # means no dose variation to differentiate), handled silently.
        pass
    else:
        failed_basis_indices = []
        for j in range(n_basis):
            c = np.zeros(n_basis)
            c[j] = 1.0
            try:
                spline_j = BSpline(knots, c, degree)
                deriv_j = spline_j.derivative()
                dB[:, j] = deriv_j(x_clamped)
            except ValueError:
                # Finding #12 (axis C, silent-failures audit): silent pass
                # on ValueError meant a malformed knot vector (too few
                # knots for the degree, non-monotonic, etc.) quietly set
                # whole columns of the derivative design matrix to zero.
                # Downstream ContinuousDiD inference then used a silently
                # biased dPsi matrix. Track affected basis indices so we
                # can surface ONE aggregate warning.
                failed_basis_indices.append(j)

        if failed_basis_indices:
            warnings.warn(
                f"B-spline derivative construction failed for "
                f"{len(failed_basis_indices)} of {n_basis} basis function(s) "
                f"(indices {failed_basis_indices}); their derivative columns "
                f"are zero. This typically indicates a malformed knot vector "
                f"(too few knots for the chosen degree, non-monotonic, or "
                f"repeated interior knots). Both ACRT point estimates and "
                f"analytical/bootstrap inference depend on this derivative "
                f"matrix, so both may be biased. Consider increasing the "
                f"number of distinct doses or reducing the B-spline degree.",
                UserWarning,
                stacklevel=2,
            )

    if include_intercept:
        # Drop first column (intercept derivative = 0), prepend zeros
        dB = np.column_stack([np.zeros(len(x)), dB[:, 1:]])

    return dB


def default_dose_grid(dose, lower_quantile=0.10, upper_quantile=0.99):
    """
    Compute a quantile-based evaluation grid from positive dose values.

    Matches R's default: ``quantile(dose[dose > 0], probs=seq(0.10, 0.99, 0.01))``,
    producing 90 evaluation points.

    Parameters
    ----------
    dose : array-like
        Dose values (only positive values are used).
    lower_quantile : float, default=0.10
        Lower quantile bound.
    upper_quantile : float, default=0.99
        Upper quantile bound.

    Returns
    -------
    np.ndarray
        Dose evaluation grid.
    """
    dose = np.asarray(dose, dtype=float)
    positive_dose = dose[dose > 0]
    if len(positive_dose) == 0:
        return np.array([])
    probs = np.arange(lower_quantile, upper_quantile + 0.005, 0.01)
    return np.quantile(positive_dose, probs)


# ----------------------------------------------------------------------
# Saturated (discrete-treatment) basis
#
# For a multi-valued / discrete dose taking distinct levels d_1 < ... < d_J,
# the dose-response is estimated by a *saturated* regression (CGBS 2024
# Eq. 4.1): one indicator per level, so beta_j = mean_{D=d_j}(delta_tilde_Y)
# = ATT(d_j) (a per-level 2x2 DiD). These three functions mirror the B-spline
# trio (build_bspline_basis / bspline_design_matrix /
# bspline_derivative_design_matrix) so ContinuousDiD can swap the basis and
# reuse the entire linear influence-function / bootstrap / covariate / survey
# machinery unchanged: att_d = Psi_eval @ beta, acrt_d = dPsi_eval @ beta.
# ----------------------------------------------------------------------


def saturated_dose_levels(dose, tol=SATURATED_TOL):
    """
    Distinct positive dose levels for the saturated (discrete) basis.

    Sorted unique positive doses, clustered at ``tol`` (values within ``tol``
    of an accepted level collapse to it) so level construction uses the same
    tolerance as matching in :func:`saturated_design_matrix`. Analogous to
    :func:`build_bspline_basis` returning the knot vector.

    Parameters
    ----------
    dose : array-like
        Dose values from treated units (only positive values are used).
    tol : float, default=:data:`SATURATED_TOL`
        Clustering tolerance.

    Returns
    -------
    np.ndarray
        Sorted distinct dose levels, shape ``(J,)``.
    """
    dose = np.asarray(dose, dtype=float)
    positive = np.sort(dose[dose > 0])
    levels: list = []
    for v in positive:
        if not levels or (v - levels[-1]) > tol:
            levels.append(float(v))
    return np.array(levels)


def _match_levels(x, levels, tol):
    """Map each ``x_i`` to the index of its dose level; raise if unmatched."""
    x = np.asarray(x, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if len(levels) == 0:
        raise ValueError("saturated basis requires at least one dose level.")
    diff = np.abs(x[:, np.newaxis] - levels[np.newaxis, :])
    idx = np.argmin(diff, axis=1)
    nearest = diff[np.arange(len(x)), idx]
    if np.any(nearest > tol):
        bad = x[nearest > tol]
        raise ValueError(
            f"{int(np.sum(nearest > tol))} dose value(s) match no observed dose "
            f"level within tol={tol} (e.g. {float(bad[0])}). The saturated "
            "(discrete) basis can only be evaluated at observed dose levels."
        )
    return idx


def saturated_design_matrix(x, levels, tol=SATURATED_TOL):
    """
    Indicator design matrix for the saturated (discrete) basis.

    Column ``j`` is ``1{x_i == levels[j]}`` (match within ``tol``). Serves both
    the treated design (``x`` = treated doses) and the evaluation matrix
    (``x`` = dose grid; when the grid equals ``levels`` this is the identity).
    Fail-closed: an ``x_i`` matching no level raises ``ValueError`` (no silent
    all-zero row). Analogous to :func:`bspline_design_matrix`.

    Parameters
    ----------
    x : array-like
        Evaluation points, shape ``(n,)``.
    levels : array-like
        Distinct dose levels (from :func:`saturated_dose_levels`), shape ``(J,)``.
    tol : float, default=:data:`SATURATED_TOL`
        Matching tolerance.

    Returns
    -------
    np.ndarray
        Indicator design matrix, shape ``(n, J)``.
    """
    x = np.asarray(x, dtype=float)
    levels = np.asarray(levels, dtype=float)
    idx = _match_levels(x, levels, tol)
    B = np.zeros((len(x), len(levels)))
    B[np.arange(len(x)), idx] = 1.0
    return B


def saturated_derivative_design_matrix(x, levels, tol=SATURATED_TOL):
    """
    Finite-difference derivative rows for the saturated (discrete) basis.

    ACRT for a discrete dose is the paper's backward difference of the level
    effects (CGBS 2024 §3.2 / §4.1) on the grid ``{d_0 = 0, d_1, ..., d_J}``,
    where ``d_0 = 0`` is the omitted (untreated) category with ``ATT(0) = 0``:
    ``ACRT(d_j) = [ATT(d_j) - ATT(d_{j-1})] / (d_j - d_{j-1})``. At the lowest
    positive level this references the zero-dose baseline,
    ``ACRT(d_1) = [ATT(d_1) - 0] / (d_1 - 0) = ATT(d_1) / d_1`` — so a single
    positive dose (``J = 1``, e.g. binary ``D in {0, 1}``) gives
    ``ACRT(d_1) = ATT(d_1) / d_1`` and, for ``d_1 = 1``, the documented binary
    identity ``ACRT = ATT``. This is a linear operator ``L`` on ``beta``, and
    ``acrt = L @ beta``. Only the lowest level's row references ``d_0 = 0`` (so
    that row does NOT sum to 0); the ``j >= 2`` rows are ordinary adjacent
    backward differences (rows sum to 0). Returns the ``L`` row for each ``x_i``
    at its dose level. Analogous to :func:`bspline_derivative_design_matrix`.

    Parameters
    ----------
    x : array-like
        Evaluation points, shape ``(n,)``.
    levels : array-like
        Distinct dose levels, shape ``(J,)``.
    tol : float, default=:data:`SATURATED_TOL`
        Matching tolerance.

    Returns
    -------
    np.ndarray
        Derivative design matrix, shape ``(n, J)``.
    """
    levels = np.asarray(levels, dtype=float)
    idx = _match_levels(x, levels, tol)
    J = len(levels)
    L = np.zeros((J, J))
    # Row 0 (lowest positive dose d_1): backward difference to the zero-dose
    # baseline d_0 = 0, ATT(0) = 0 -> ACRT(d_1) = ATT(d_1) / d_1. Rows j >= 1:
    # ordinary adjacent backward differences between positive doses.
    for j in range(J):
        if j == 0:
            L[0, 0] = 1.0 / levels[0]
        else:
            h = levels[j] - levels[j - 1]
            L[j, j - 1] = -1.0 / h
            L[j, j] = 1.0 / h
    return L[idx]
