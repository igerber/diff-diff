"""
B-spline utilities for continuous Difference-in-Differences estimation.

Provides basis construction, evaluation, and derivative computation for
the dose-response curve estimation in ContinuousDiD.
"""

import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "build_bspline_basis",
    "bspline_design_matrix",
    "bspline_derivative_design_matrix",
    "default_dose_grid",
]


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
    knots = np.concatenate([
        np.repeat(d_L, degree + 1),
        interior_knots,
        np.repeat(d_U, degree + 1),
    ])

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
        # All knots identical: derivatives are all zero
        pass
    else:
        for j in range(n_basis):
            c = np.zeros(n_basis)
            c[j] = 1.0
            try:
                spline_j = BSpline(knots, c, degree)
                deriv_j = spline_j.derivative()
                dB[:, j] = deriv_j(x_clamped)
            except ValueError:
                # Degenerate knot vector: derivative is zero
                pass

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
