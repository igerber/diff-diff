"""
Kernels and univariate local-linear regression at a boundary.

Ships the foundational RDD infrastructure that downstream estimators compose:

- Bounded one-sided kernels (Epanechnikov, triangular, uniform) on ``[0, 1]``
  suitable for boundary-point nonparametric regression.
- Closed-form kernel-moment constants ``kappa_k := int_0^1 t^k * k(t) dt`` and the
  derived boundary-kernel constant ``C = (kappa_2^2 - kappa_1 kappa_3) /
  (kappa_0 kappa_2 - kappa_1^2)`` that appears in the asymptotic bias of
  local-linear at a boundary.
- A univariate local-linear regression fitter ``local_linear_fit`` that estimates
  the conditional mean ``m(d0) := E[Y | D = d0]`` at the boundary of ``D``'s
  support via kernel-weighted OLS.

This module is used by the :class:`HeterogeneousAdoptionDiD` phases:

- Phase 1a ships the kernels and fitter (this module).
- Phase 1b ships the MSE-optimal bandwidth selector
  ``mse_optimal_bandwidth`` / ``BandwidthResult`` (this module), a thin
  wrapper over the Calonico-Cattaneo-Farrell (2018) plug-in selector
  ported in ``diff_diff/_nprobust_port.py``.
- Phase 1c ships the bias-corrected local-linear fit
  ``bias_corrected_local_linear`` / ``BiasCorrectedFit`` (this module), a
  thin wrapper over the Calonico-Cattaneo-Titiunik (2014) robust-bias
  correction ported in ``diff_diff/_nprobust_port.py``. This produces
  the mu-scale point estimate and CI for Equation 8 of de Chaisemartin,
  Ciccia, D'Haultfoeuille & Knau (2026, arXiv:2405.04465v6); Phase 2
  applies the ``(1/G) * sum(D_{g,2})`` beta-scale rescaling.

References
----------
- de Chaisemartin, C., Ciccia, D., D'Haultfoeuille, X., & Knau, F. (2026).
  Difference-in-Differences Estimators When No Unit Remains Untreated.
  arXiv:2405.04465v6. Section 3.1.3 defines the kernel-moment constants used
  here.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias
  estimation on coverage accuracy in nonparametric inference. Journal of the
  American Statistical Association, 113(522), 767-779.
- Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its
  Applications. Chapman & Hall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy import integrate

from diff_diff.linalg import solve_ols

__all__ = [
    "BandwidthResult",
    "BiasCorrectedFit",
    "bias_corrected_local_linear",
    "epanechnikov_kernel",
    "triangular_kernel",
    "uniform_kernel",
    "KERNELS",
    "kernel_moments",
    "LocalLinearFit",
    "local_linear_fit",
    "mse_optimal_bandwidth",
]


# =============================================================================
# Kernel functions
# =============================================================================
#
# Each kernel is defined on [0, 1] (one-sided, for boundary estimation where
# the running variable D is supported on [0, infinity) and the evaluation point
# is at d0 = 0). Kernels return 0 outside [0, 1].


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel on ``[0, 1]``.

    ``k(u) = (3/4)(1 - u^2)`` for ``u in [0, 1]``, zero elsewhere.

    Parameters
    ----------
    u : np.ndarray
        Points on the scaled domain ``u = (d - d0) / h``.

    Returns
    -------
    np.ndarray
        Kernel values, same shape as ``u``.
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 0.75 * (1.0 - u * u), 0.0)


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel on ``[0, 1]``.

    ``k(u) = 1 - u`` for ``u in [0, 1]``, zero elsewhere.

    Using the convention ``int_0^1 k(u) du = 1/2`` to match Epanechnikov's
    one-sided normalization.
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 1.0 - u, 0.0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform (rectangular) kernel on ``[0, 1]``.

    ``k(u) = 1`` for ``u in [0, 1]``, zero elsewhere.

    Already normalized to ``int_0^1 k(u) du = 1`` (no factor of 1/2 like the
    other two).
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 1.0, 0.0)


KERNELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "epanechnikov": epanechnikov_kernel,
    "triangular": triangular_kernel,
    "uniform": uniform_kernel,
}


# =============================================================================
# Closed-form kernel moments
# =============================================================================
#
# For each kernel k(t) on [0, 1], the moments
#
#     kappa_k := int_0^1 t^k * k(t) dt
#
# admit closed forms. These are the values from elementary integration; the
# test suite verifies each against a numerical scipy.integrate.quad call.
#
# The derived constant C from the paper's Section 3.1.3 is
#
#     C = (kappa_2^2 - kappa_1 * kappa_3) / (kappa_0 * kappa_2 - kappa_1^2)
#
# and can be negative depending on kernel shape (e.g. Epanechnikov C < 0).


_CLOSED_FORM_MOMENTS: Dict[str, Dict[str, float]] = {
    "epanechnikov": {
        "kappa_0": 1.0 / 2.0,
        "kappa_1": 3.0 / 16.0,
        "kappa_2": 1.0 / 10.0,
        "kappa_3": 1.0 / 16.0,
        "kappa_4": 3.0 / 70.0,
    },
    "triangular": {
        "kappa_0": 1.0 / 2.0,
        "kappa_1": 1.0 / 6.0,
        "kappa_2": 1.0 / 12.0,
        "kappa_3": 1.0 / 20.0,
        "kappa_4": 1.0 / 30.0,
    },
    "uniform": {
        "kappa_0": 1.0,
        "kappa_1": 1.0 / 2.0,
        "kappa_2": 1.0 / 3.0,
        "kappa_3": 1.0 / 4.0,
        "kappa_4": 1.0 / 5.0,
    },
}


def kernel_moments(kernel: str = "epanechnikov") -> Dict[str, float]:
    """Return kernel-moment constants used in boundary local-linear asymptotics.

    The returned dict contains five raw moments ``kappa_k`` for
    ``k in {0, 1, 2, 3, 4}``, plus two derived constants:

    - ``"C"``: the paper's boundary-kernel constant used in the asymptotic
      bias term ``h^2 * C * m''(0)``. Per de Chaisemartin, Ciccia,
      D'Haultfoeuille & Knau (2026, Section 3.1.3),
      ``C = (kappa_2^2 - kappa_1 * kappa_3) / (kappa_0 * kappa_2 - kappa_1^2)``.
    - ``"kstar_L2_norm"``: the asymptotic-variance constant
      ``int_0^1 k*(t)^2 dt`` where
      ``k*(t) = (kappa_2 - kappa_1 * t) / (kappa_0 * kappa_2 - kappa_1^2) * k(t)``
      is the equivalent kernel for local-linear at a boundary. Computed by
      numerical integration.

    Parameters
    ----------
    kernel : str
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.

    Returns
    -------
    dict of {str: float}
        Keys ``kappa_0``, ``kappa_1``, ``kappa_2``, ``kappa_3``, ``kappa_4``,
        ``C``, ``kstar_L2_norm``.

    Raises
    ------
    ValueError
        If ``kernel`` is not a recognized name.
    """
    if kernel not in _CLOSED_FORM_MOMENTS:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of " f"{sorted(_CLOSED_FORM_MOMENTS.keys())}."
        )

    kappas = dict(_CLOSED_FORM_MOMENTS[kernel])

    k0, k1, k2, k3 = kappas["kappa_0"], kappas["kappa_1"], kappas["kappa_2"], kappas["kappa_3"]
    denom = k0 * k2 - k1 * k1
    C = (k2 * k2 - k1 * k3) / denom
    kappas["C"] = C

    kfun = KERNELS[kernel]

    def _kstar_sq(t: float) -> float:
        kt = kfun(np.array([t]))[0]
        return ((k2 - k1 * t) / denom) ** 2 * kt * kt

    val, _ = integrate.quad(_kstar_sq, 0.0, 1.0, limit=200)
    kappas["kstar_L2_norm"] = float(val)

    return kappas


# =============================================================================
# Local-linear regression at a boundary
# =============================================================================


@dataclass
class LocalLinearFit:
    """Result of a local-linear regression at a boundary.

    Attributes
    ----------
    intercept : float
        Estimated conditional mean at the boundary, ``mu_hat_h(d0)``.
    slope : float
        Estimated slope of the local linear fit (coefficient on ``d - d0``).
    n_effective : int
        Count of observations with strictly positive kernel weight (within
        ``[d0, d0 + h]`` for the one-sided kernels shipped here).
    bandwidth : float
        Bandwidth ``h`` used.
    kernel : str
        Kernel name.
    boundary : float
        Evaluation point ``d0``.
    residuals : np.ndarray, shape (n_effective,)
        Residuals from the weighted OLS fit, in the order of the retained
        observations.
    kernel_weights : np.ndarray, shape (n_effective,)
        Kernel weights ``k((d_i - d0) / h)``. These are the pre-scaled weights;
        the ``1/h`` scaling cancels out of the weighted-OLS estimator (a
        constant factor on all weights does not change the point estimate).
    design_matrix : np.ndarray, shape (n_effective, 2)
        Design matrix ``X = [1, d_i - d0]`` used in the fit. Preserved for
        Phase 1c bias-correction machinery.
    """

    intercept: float
    slope: float
    n_effective: int
    bandwidth: float
    kernel: str
    boundary: float
    residuals: np.ndarray
    kernel_weights: np.ndarray
    design_matrix: np.ndarray


def local_linear_fit(
    d: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    boundary: float = 0.0,
    kernel: str = "epanechnikov",
    weights: Optional[np.ndarray] = None,
) -> LocalLinearFit:
    """Local-linear regression of ``y`` on ``d`` at a boundary.

    Fits ``y ~ a + b * (d - boundary)`` using kernel weights
    ``k((d - boundary) / h)`` on observations with ``d in [boundary,
    boundary + h]``. Returns the intercept (the boundary estimate ``mu_hat``)
    and slope.

    Parameters
    ----------
    d : np.ndarray, shape (n,)
        Regressor values. For the HAD application, ``d`` is the period-2 dose
        ``D_{g,2}`` and the boundary is 0.
    y : np.ndarray, shape (n,)
        Outcome values. For the HAD application, ``y`` is the first-difference
        ``Delta Y_g``.
    bandwidth : float
        Bandwidth ``h > 0``.
    boundary : float, default=0.0
        Evaluation point ``d0``. Observations with ``d < d0`` are excluded
        (one-sided boundary estimation).
    kernel : str, default="epanechnikov"
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.
    weights : np.ndarray or None, optional
        Optional per-observation weights ``w_i >= 0`` multiplied into the
        kernel weights. Useful for survey weighting; when ``None``, treated as
        unit weights.

    Returns
    -------
    LocalLinearFit
        Named container with ``intercept``, ``slope``, ``n_effective``, and
        diagnostics needed by downstream bias-correction phases.

    Raises
    ------
    ValueError
        If ``bandwidth <= 0``, ``kernel`` is unknown, ``d`` and ``y`` differ
        in length, or the bandwidth window retains fewer than 2 observations.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth}")
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {sorted(KERNELS.keys())}.")

    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(f"d and y must have the same shape; got {d.shape} and {y.shape}")

    # Explicit NaN / Inf validation at the API boundary so the caller gets a
    # targeted error rather than a downstream failure inside the kernel or OLS.
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")

    if weights is None:
        user_w = np.ones_like(d)
    else:
        user_w = np.asarray(weights, dtype=np.float64).ravel()
        if user_w.shape != d.shape:
            raise ValueError(
                f"weights must have the same shape as d; got " f"{user_w.shape} vs {d.shape}"
            )
        if not np.all(np.isfinite(user_w)):
            raise ValueError("weights contains non-finite values (NaN or Inf)")
        if np.any(user_w < 0):
            raise ValueError("weights must be nonnegative")

    # Kernel weights on the scaled domain u = (d - d0) / h.
    kfun = KERNELS[kernel]
    u = (d - boundary) / bandwidth
    k_weights_full = kfun(u)

    # Compose with user weights and restrict to the bandwidth window.
    combined = k_weights_full * user_w
    retain = combined > 0.0
    n_effective = int(retain.sum())
    if n_effective < 2:
        raise ValueError(
            f"bandwidth window retained {n_effective} observation(s); "
            f"need at least 2. Widen the bandwidth or move the boundary."
        )

    d_in = d[retain]
    y_in = y[retain]
    w_in = combined[retain]
    k_in = k_weights_full[retain]

    design = np.column_stack([np.ones_like(d_in), d_in - boundary])

    # Weighted OLS via solve_ols. "aweight" treats the weights as analytic
    # frequency weights so the unweighted-OLS formulas apply with w-scaled X.
    # We only need the coefficients and residuals, not a vcov for the fit
    # itself (Phase 1c will build its own bias-aware variance).
    coef, residuals, _ = solve_ols(
        design,
        y_in,
        cluster_ids=None,
        return_vcov=False,
        weights=w_in,
        weight_type="aweight",
    )

    return LocalLinearFit(
        intercept=float(coef[0]),
        slope=float(coef[1]),
        n_effective=n_effective,
        bandwidth=float(bandwidth),
        kernel=kernel,
        boundary=float(boundary),
        residuals=np.asarray(residuals, dtype=np.float64),
        kernel_weights=np.asarray(k_in, dtype=np.float64),
        design_matrix=np.asarray(design, dtype=np.float64),
    )


# =============================================================================
# MSE-optimal bandwidth selector (Phase 1b)
# =============================================================================
#
# Public wrapper around diff_diff._nprobust_port.lpbwselect_mse_dpi. The port
# is a faithful Python translation of nprobust::lpbwselect(bwselect="mse-dpi")
# (R package nprobust 0.5.0, SHA 36e4e53); see diff_diff/_nprobust_port.py for
# source mapping.
#
# The public API here is a thin wrapper that:
# 1. Accepts the diff-diff library's kernel naming convention (full words) and
#    translates to nprobust's short codes ("epa", "tri", "uni").
# 2. Converts the internal MseDpiStages dataclass to the user-facing
#    BandwidthResult dataclass.
# 3. Enforces input validation consistent with local_linear_fit.
# 4. Rejects unsupported combinations (e.g. weights=) with NotImplementedError
#    (nprobust has no weight support).


@dataclass
class BandwidthResult:
    """MSE-optimal bandwidth selector output plus per-stage diagnostics.

    Returned by ``mse_optimal_bandwidth(..., return_diagnostics=True)``.
    Mirrors the five-bandwidth + four-stage structure of
    ``nprobust::lpbwselect.mse.dpi``; see
    ``diff_diff/_nprobust_port.py`` for the source mapping.

    Attributes
    ----------
    h_mse : float
        Final MSE-optimal bandwidth ``h*`` for local-linear estimation at
        ``boundary``. The argument to pass to
        ``local_linear_fit(..., bandwidth=h_mse)``.
    b_mse : float
        Bias-correction bandwidth. Consumed by Phase 1c for the
        bias-corrected confidence interval (CCF 2018 Equation 8).
    c_bw : float
        Stage 1 preliminary bandwidth used as ``h.V`` in every
        ``lprobust.bw`` call downstream:
        ``C_kernel * min(sd(d), IQR(d)/1.349) * G^{-1/5}``.
        Kernel constants: ``epa=2.34``, ``uni=1.843``, ``tri=2.576``.
    bw_mp2 : float
        Stage 2 pilot bandwidth for the ``m^{(q+1)}`` derivative estimator.
    bw_mp3 : float
        Stage 2 pilot bandwidth for the ``m^{(q+2)}`` derivative estimator.
    stage_d1_V, stage_d1_B1, stage_d1_B2, stage_d1_R : float
        Variance and bias coefficients from the first Stage-2
        ``lprobust.bw`` call (order ``q+1``, reading the ``(q+1)``-th
        derivative). Parity-checked to 1% against R.
    stage_d2_V, stage_d2_B1, stage_d2_B2, stage_d2_R : float
        Same for the second Stage-2 call (order ``q+2``).
    stage_b_V, stage_b_B1, stage_b_B2, stage_b_R : float
        Same for the Stage-3 bias-bandwidth call (order ``q``, nu ``p+1``).
    stage_h_V, stage_h_B1, stage_h_B2, stage_h_R : float
        Same for the final Stage-3 main-bandwidth call (order ``p``,
        nu ``deriv``).
    n : int
        Sample size.
    kernel : str
        Kernel name as user supplied it ("epanechnikov" / "triangular" /
        "uniform").
    boundary : float
        Evaluation point ``d_0``.
    """

    h_mse: float
    b_mse: float
    c_bw: float
    bw_mp2: float
    bw_mp3: float
    stage_d1_V: float
    stage_d1_B1: float
    stage_d1_B2: float
    stage_d1_R: float
    stage_d2_V: float
    stage_d2_B1: float
    stage_d2_B2: float
    stage_d2_R: float
    stage_b_V: float
    stage_b_B1: float
    stage_b_B2: float
    stage_b_R: float
    stage_h_V: float
    stage_h_B1: float
    stage_h_B2: float
    stage_h_R: float
    n: int
    kernel: str
    boundary: float


# Mapping between diff-diff's full kernel names and nprobust's short codes.
_KERNEL_NAME_TO_NPROBUST = {
    "epanechnikov": "epa",
    "triangular": "tri",
    "uniform": "uni",
}


def _validate_had_inputs(
    d: np.ndarray,
    y: np.ndarray,
    boundary: float,
) -> tuple[np.ndarray, np.ndarray]:
    """HAD-scope input validation shared across Phase 1b and Phase 1c.

    Coerces ``d`` and ``y`` to 1-D ``float64`` arrays and enforces the
    HAD input contract used by both ``mse_optimal_bandwidth`` (Phase 1b)
    and ``bias_corrected_local_linear`` (Phase 1c). Each caller handles
    its own ``weights=`` rejection upfront because the phase name and
    message differ; every other rule lives here.

    Rules (matching Phase 1b's original inline checks verbatim):

    - ``d`` and ``y`` must be shape-matched, non-empty, and finite.
    - ``boundary`` must be finite.
    - Doses are nonnegative (HAD support ``D_{g,2} >= 0``).
    - ``boundary`` is approximately ``0`` (Design 1') or approximately
      ``d.min()`` (Design 1 continuous-near-d_lower); other boundaries
      raise so off-support calls do not silently target a different
      estimand.
    - Mass-point designs at ``d.min() > 0`` with modal-min fraction
      ``> 2%`` raise ``NotImplementedError`` pointing to Phase 2's 2SLS
      path, per the paper's Section 3.2.4.
    - Design 1' plausibility heuristic: when ``boundary ~ 0``, require
      ``d.min() <= 0.05 * median(|d|)``; samples above that ratio are
      redirected to ``boundary=float(d.min())``.

    Parameters
    ----------
    d, y : np.ndarray
        Regressor (dose) and outcome.
    boundary : float
        Evaluation point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coerced ``(d, y)`` as 1-D float64 arrays.

    Raises
    ------
    ValueError, NotImplementedError
    """
    if not np.isfinite(boundary):
        raise ValueError(f"boundary must be finite; got {boundary}")

    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(f"d and y must have the same shape; got {d.shape} and {y.shape}")
    if d.size == 0:
        raise ValueError(
            "d and y must be non-empty; the selector cannot estimate a "
            "bandwidth from zero observations."
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")

    # HAD support restriction: de Chaisemartin et al. (2026) Assumption
    # (dose definition in Section 2) treats ``D_{g,2}`` as the period-2
    # treatment dose with ``D_{g,2} >= 0``. Negative dose values are
    # outside the HAD design and would silently calibrate the selector
    # against a symmetric-kernel two-sided problem while the downstream
    # fitter remains one-sided. Reject front-door rather than produce a
    # plausible bandwidth on a malformed input.
    d_neg = d < 0.0
    if np.any(d_neg):
        n_neg = int(d_neg.sum())
        min_neg = float(d[d_neg].min())
        raise ValueError(
            f"Negative dose values detected in d (n_neg={n_neg}, "
            f"min={min_neg!r}). The HAD estimator (de Chaisemartin et "
            f"al. 2026) requires the period-2 dose D_{{g,2}} >= 0. "
            f"Nonnegative-dose data is required for both Design 1' "
            f"(d_lower = 0) and Design 1 (d_lower > 0)."
        )

    # Boundary-applicability check (Phase 1b scope).
    # The exported wrapper is scoped to the two documented HAD
    # nonparametric estimands:
    #   - Design 1' evaluates m(0) = lim_{d down 0} E[Delta Y | D_2 <= d]
    #     with ``boundary = 0``.
    #   - Design 1 continuous-near-d_lower evaluates m(d_lower) with
    #     ``boundary = d_lower = min(D_2)``.
    # Any other off-support boundary (interior, upper-boundary, or an
    # arbitrary value between 0 and d.min()) would silently target an
    # undocumented limit and is rejected.
    d_min = float(d.min())
    _boundary_tol = 1e-12 * max(1.0, abs(d_min), abs(boundary))
    _at_zero = abs(boundary) <= _boundary_tol
    _at_d_min = abs(boundary - d_min) <= _boundary_tol
    if not (_at_zero or _at_d_min):
        raise ValueError(
            f"boundary={boundary!r} is not at a supported HAD estimand. "
            f"The Phase 1b public wrapper accepts only boundary ~ 0 "
            f"(Design 1') or boundary ~ d.min()={d_min!r} (Design 1 "
            f"continuous-near-d_lower). Off-support values would "
            f"silently target an undocumented limit. For interior or "
            f"other boundary points, use "
            f"diff_diff._nprobust_port.lpbwselect_mse_dpi directly and "
            f"note that those paths are not separately parity-tested."
        )

    # Mass-point design check (paper Section 3.2.4, REGISTRY 2% rule).
    # Must fire BEFORE the Design 1' support check: mass-point data is
    # never appropriate for the CCF nonparametric selector regardless
    # of the boundary the caller supplied. The correct remediation is
    # the 2SLS sample-average path (Phase 2), not a boundary
    # reclassification.
    #
    # The check explicitly excludes d_min ~ 0 (the Design 1'
    # "untreated units present" subcase that the paper's simulations
    # and the Garrett et al. (2020) application accept).
    _MASS_POINT_THRESHOLD = 0.02  # REGISTRY rule: > 2% modal-min
    if d_min > _boundary_tol:
        eps_eq = 1e-12 * max(1.0, abs(d_min))
        at_d_min_mask = np.abs(d - d_min) <= eps_eq
        modal_fraction = float(np.mean(at_d_min_mask))
        if modal_fraction > _MASS_POINT_THRESHOLD:
            raise NotImplementedError(
                f"Detected mass-point design at d.min()={d_min!r} "
                f"(modal fraction {modal_fraction:.4f} > "
                f"{_MASS_POINT_THRESHOLD:.2f}). Per de Chaisemartin et "
                f"al. (2026) Section 3.2.4, Design 1 mass-point cases "
                f"require the 2SLS sample-average estimator with "
                f"instrument 1{{D_2 > d_lower}}, not the CCF "
                f"nonparametric selector. That path is queued for "
                f"Phase 2 (HeterogeneousAdoptionDiD). For continuous "
                f"near-d_lower designs (modal fraction <= "
                f"{_MASS_POINT_THRESHOLD:.2f}), this wrapper is "
                f"applicable."
            )

    # Design 1' support check: boundary ~ 0 requires the realized
    # sample minimum to be compatible with a population support
    # infimum at 0. Otherwise the selector calibrates ``h_mse`` at
    # an off-support limit.
    #
    # Rule: when boundary ~ 0 (not also at d.min()), require
    # d.min() <= 5% * median(|d|). The 5% threshold is generous
    # enough to accept Design 1' samples with vanishing boundary
    # density (Beta(2,2): d.min/median ~ 3%) while rejecting samples
    # substantially off-support (U(0.5, 1): d.min/median ~ 1.0).
    # Samples just between these (e.g. U(0.05, 1), d.min/median ~ 10%)
    # are directed to boundary=float(d.min()) for the continuous-
    # near-d_lower path.
    _DESIGN_1_PRIME_RATIO = 0.05
    if _at_zero and not _at_d_min:
        d_median_abs = float(np.median(np.abs(d)))
        effective_threshold = _DESIGN_1_PRIME_RATIO * max(d_median_abs, 1e-12)
        if d_min > effective_threshold:
            raise ValueError(
                f"boundary ~ 0 selected but d.min()={d_min!r} is not "
                f"compatible with a Design 1' support infimum at 0 "
                f"(rule: d.min() <= "
                f"{_DESIGN_1_PRIME_RATIO} * median(|d|) = "
                f"{effective_threshold!r}). This sample is not "
                f"Design 1'. Either: (a) pass boundary=float(d.min()) "
                f"for the Design 1 continuous-near-d_lower path, or "
                f"(b) verify the population support actually has "
                f"infimum at 0 (in which case the realized d.min() "
                f"would be closer to zero relative to the data scale)."
            )

    return d, y


def mse_optimal_bandwidth(
    d: np.ndarray,
    y: np.ndarray,
    boundary: float = 0.0,
    kernel: str = "epanechnikov",
    weights: Optional[np.ndarray] = None,
    bwcheck: Optional[int] = 21,
    bwregul: float = 1.0,
    return_diagnostics: bool = False,
):
    """MSE-optimal bandwidth for local-linear regression at a boundary.

    Port of ``nprobust::lpbwselect(bwselect="mse-dpi")`` (R package
    ``nprobust`` 0.5.0, SHA ``36e4e53``). Implements the Calonico,
    Cattaneo, and Farrell (2018, JASA 113(522)) direct-plug-in DPI
    bandwidth selector for the local-linear boundary estimator used in
    Design 1' of de Chaisemartin et al. (2026) (HAD).

    The three-stage algorithm produces five bandwidths (``c_bw``,
    ``bw_mp2``, ``bw_mp3``, ``b_mse``, ``h_mse``); see
    ``BandwidthResult`` for full diagnostics.

    **Public API scope (Phase 1b of HAD).** This wrapper is intentionally
    restricted to the HAD configuration: ``p=1`` (local-linear),
    ``deriv=0`` (mean regression), ``interior=False`` (boundary eval
    point), ``vce="nn"`` (nearest-neighbor variance), and ``nnmatch=3``
    are hard-coded. The underlying ``diff_diff._nprobust_port`` supports
    additional ``vce`` modes (``hc0`` / ``hc1`` / ``hc2`` / ``hc3``),
    interior evaluation, and higher polynomial orders, but those paths
    are NOT parity-tested against ``nprobust`` and are deferred to Phase
    1c / Phase 2. Do not rely on this public wrapper for anything
    outside HAD Phase 1b; use the port directly if you need the broader
    surface and accept that parity has not been separately verified.

    Parameters
    ----------
    d : np.ndarray, shape (G,)
        Regressor values (the dose ``D_{g,2}`` in HAD).
    y : np.ndarray, shape (G,)
        Outcome values (the first-difference ``Delta Y_g`` in HAD).
    boundary : float, default=0.0
        Evaluation point ``d_0``. The Phase 1b wrapper accepts only
        two values (within float tolerance): ``boundary = 0`` for
        Design 1' or ``boundary = float(d.min())`` for Design 1
        continuous-near-``d_lower``. Use the sample minimum
        ``d.min()`` (not a known theoretical lower bound of the
        support), because the downstream selector operates on the
        realized data. Any other value -- including
        ``boundary < d.min()``, interior points, or
        ``boundary > d.min()`` -- raises ``ValueError``.
    kernel : str, default="epanechnikov"
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.
    weights : np.ndarray or None, default=None
        Not supported in Phase 1b (raises ``NotImplementedError``).
        ``nprobust::lpbwselect`` has no weight argument and thus no
        parity anchor. Weighted-data support is queued for Phase 2+.
    bwcheck : int or None, default=21
        If set, clip ``c_bw`` (and all downstream bandwidths) below the
        distance to the ``bwcheck``-th nearest neighbor of ``boundary``.
        Matches ``nprobust::lpbwselect`` default.
    bwregul : float, default=1.0
        Bias-regularization scale used in Stage 3. ``bwregul=0`` disables
        the regularization term; ``bwregul=1`` matches ``nprobust``
        default.
    return_diagnostics : bool, default=False
        When ``False`` (default) return the final bandwidth as a
        ``float``. When ``True`` return a ``BandwidthResult`` with all
        five stage bandwidths plus per-stage ``(V, B1, B2, R)``
        diagnostics.

    Returns
    -------
    float or BandwidthResult
        The MSE-optimal bandwidth ``h*``, or (if
        ``return_diagnostics=True``) a ``BandwidthResult`` dataclass.

    Raises
    ------
    ValueError
        Raised on: shape mismatch between ``d`` and ``y``; non-finite
        values in ``d``, ``y``, or ``boundary``; unknown ``kernel``
        name; ``bwcheck`` outside ``[1, len(d)]``; ``boundary`` that
        is not approximately 0 or approximately ``d.min()`` (the only
        two supported HAD estimands in Phase 1b); or a rank-deficient
        / under-determined pilot fit inside the DPI port (surfaced
        from ``qrXXinv`` or the per-stage count guards in
        ``lprobust_bw``).
    NotImplementedError
        Raised on: ``weights=`` passed (no nprobust parity anchor);
        detected Design 1 mass-point design (``d.min() > 0`` and
        modal fraction at ``d.min()`` exceeds 2%, per the paper's
        Section 3.2.4 redirection to the 2SLS sample-average
        estimator, queued for Phase 2).

    Notes
    -----
    The port parity-tests at 1% relative error against R
    ``nprobust::lpbwselect(bwselect="mse-dpi", vce="nn")`` on three
    deterministic DGPs (see
    ``benchmarks/R/generate_nprobust_golden.R``). In practice the
    agreement is at machine precision (0.0000% on the golden tests);
    the 1% tolerance is the hard gate before a deviation is considered
    a regression.
    """
    if weights is not None:
        raise NotImplementedError(
            "weights= is not supported in Phase 1b of the MSE-optimal "
            "bandwidth selector. nprobust::lpbwselect has no weight "
            "argument, so there is no parity anchor. Weighted-data "
            "support is queued for Phase 2+ (survey-design adaptation)."
        )

    if kernel not in _KERNEL_NAME_TO_NPROBUST:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of "
            f"{sorted(_KERNEL_NAME_TO_NPROBUST.keys())}."
        )
    nprobust_kernel = _KERNEL_NAME_TO_NPROBUST[kernel]

    d, y = _validate_had_inputs(d, y, boundary)

    # Defer heavy import to call time to avoid import-cycle risk.
    from diff_diff._nprobust_port import lpbwselect_mse_dpi

    stages = lpbwselect_mse_dpi(
        y=y,
        x=d,
        cluster=None,
        eval_point=float(boundary),
        p=1,
        q=2,
        deriv=0,
        kernel=nprobust_kernel,
        bwcheck=bwcheck,
        bwregul=bwregul,
        vce="nn",
        nnmatch=3,
        interior=False,
    )

    if not return_diagnostics:
        return stages.h_mse_dpi

    return BandwidthResult(
        h_mse=stages.h_mse_dpi,
        b_mse=stages.b_mse_dpi,
        c_bw=stages.c_bw,
        bw_mp2=stages.bw_mp2,
        bw_mp3=stages.bw_mp3,
        stage_d1_V=stages.stage_d1.V,
        stage_d1_B1=stages.stage_d1.B1,
        stage_d1_B2=stages.stage_d1.B2,
        stage_d1_R=stages.stage_d1.R,
        stage_d2_V=stages.stage_d2.V,
        stage_d2_B1=stages.stage_d2.B1,
        stage_d2_B2=stages.stage_d2.B2,
        stage_d2_R=stages.stage_d2.R,
        stage_b_V=stages.stage_b.V,
        stage_b_B1=stages.stage_b.B1,
        stage_b_B2=stages.stage_b.B2,
        stage_b_R=stages.stage_b.R,
        stage_h_V=stages.stage_h.V,
        stage_h_B1=stages.stage_h.B1,
        stage_h_B2=stages.stage_h.B2,
        stage_h_R=stages.stage_h.R,
        n=int(d.shape[0]),
        kernel=kernel,
        boundary=float(boundary),
    )


# =============================================================================
# Bias-corrected local-linear fit (Phase 1c)
# =============================================================================
#
# Public wrapper around diff_diff._nprobust_port.lprobust. The port is a
# faithful Python translation of the single-eval-point path of
# nprobust::lprobust (R package nprobust 0.5.0, SHA 36e4e53) implementing
# Calonico, Cattaneo, and Titiunik (2014) robust bias correction.
#
# This wrapper produces the mu-scale quantities of Equation 8 in de
# Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026): a classical and
# bias-corrected point estimate plus naive and robust standard errors and
# the bias-corrected confidence interval [tau.bc +/- z_{1-alpha/2} * se.rb].
# The beta-scale rescaling in Equation 8 (divide by ``(1/G) * sum(D_{g,2})``)
# is applied by Phase 2's ``HeterogeneousAdoptionDiD.fit()``, not here.


@dataclass
class BiasCorrectedFit:
    """Bias-corrected local-linear fit at a boundary (Phase 1c).

    Output of :func:`bias_corrected_local_linear`. Produces the mu-scale
    quantities needed by Equation 8 of de Chaisemartin, Ciccia,
    D'Haultfoeuille, and Knau (2026). Phase 2's ``HeterogeneousAdoptionDiD``
    class applies the beta-scale ``(1/G) * sum(D_{g,2})`` rescaling.

    Attributes
    ----------
    estimate_classical : float
        Classical point estimate ``tau.cl`` from ``nprobust::lprobust``
        (local-linear boundary intercept at ``h``; no bias correction).
    estimate_bias_corrected : float
        Bias-corrected point estimate ``tau.bc = mu_hat + M_hat`` from the
        Calonico-Cattaneo-Titiunik (2014) combined design-matrix statistic.
    se_classical : float
        Naive plug-in standard error.
    se_robust : float
        Robust standard error accounting for the additional variability
        introduced by the bias-correction term (CCT 2014).
    ci_low, ci_high : float
        Endpoints of the bias-corrected CI: ``tau.bc +/- z_{1-alpha/2} *
        se.rb``.
    alpha : float
        CI level (``0.05`` gives a 95% CI).
    h, b : float
        Main and bias-correction bandwidths actually used (post-``bwcheck``
        floor).
    bandwidth_source : {"auto", "user"}
        ``"auto"`` when the wrapper called the Phase 1b DPI selector
        (``_nprobust_port.lpbwselect_mse_dpi``) internally with the
        caller's ``cluster`` / ``vce`` / ``nnmatch``; ``"user"`` when the
        caller passed explicit bandwidths. Auto mode then enforces
        nprobust's ``rho=1`` default by setting ``b = h``; the
        selector's distinct ``b_mse`` is surfaced via
        ``bandwidth_diagnostics`` but not applied.
    bandwidth_diagnostics : BandwidthResult or None
        Full Phase 1b selector output when ``bandwidth_source == "auto"``;
        ``None`` when the user supplied bandwidths (to avoid a redundant
        selector call).
    n_used : int
        Observations retained in the active kernel window (``sum(ind.b)``
        when ``h <= b`` and ``sum(ind.h)`` when ``h > b``; with the
        ``rho=1`` default the two coincide).
    n_total : int
        Total observations passed in (before kernel filtering).
    kernel : str
        Kernel name as supplied by the caller.
    boundary : float
        Evaluation point ``c``.

    Notes
    -----
    ``p=1``, ``q=2``, ``deriv=0`` are hard-coded for HAD Phase 1c and are
    not exposed as fields. Phase 2 may surface them on the estimator-level
    result class if a use case materializes.
    """

    estimate_classical: float
    estimate_bias_corrected: float
    se_classical: float
    se_robust: float
    ci_low: float
    ci_high: float
    alpha: float
    h: float
    b: float
    bandwidth_source: str
    bandwidth_diagnostics: Optional[BandwidthResult]
    n_used: int
    n_total: int
    kernel: str
    boundary: float
    influence_function: Optional[np.ndarray] = None
    """Per-observation influence function of the BIAS-CORRECTED point
    estimate ``tau.bc`` (Phase 4.5 survey composition). Aligned with
    the original caller-supplied ``d``/``y`` ordering; observations
    outside the active kernel window have IF=0. Populated only when
    ``return_influence=True``; ``None`` otherwise.

    Derived from ``Q_q`` + ``res_b`` so the variance self-check is
    ``sum(IF^2) == V_Y_bc[0, 0]`` under unclustered HC0 — matching the
    bias-corrected scale of ``estimate_bias_corrected``. Using the
    classical IF here would silently under-estimate survey SE by
    ignoring the CCT-2014 bias-correction variance inflation."""


def bias_corrected_local_linear(
    d: np.ndarray,
    y: np.ndarray,
    boundary: float = 0.0,
    kernel: str = "epanechnikov",
    h: Optional[float] = None,
    b: Optional[float] = None,
    alpha: float = 0.05,
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
    nnmatch: int = 3,
    weights: Optional[np.ndarray] = None,
    return_influence: bool = False,
) -> BiasCorrectedFit:
    """Bias-corrected local-linear fit with robust CI at a boundary.

    Ports the single-eval-point path of ``nprobust::lprobust`` (Calonico,
    Cattaneo, and Titiunik 2014) and produces the mu-scale outputs of
    Equation 8 in de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau
    (2026). Phase 2's ``HeterogeneousAdoptionDiD`` class applies the
    beta-scale rescaling to return the final estimand.

    **Public API scope (Phase 1c of HAD).** Hard-coded for the HAD
    configuration: ``p=1`` (local-linear), ``q=2`` (bias-correction
    order), ``deriv=0`` (level), ``interior=False`` (boundary eval
    point), ``bwcheck=21``, ``bwregul=1``, and ``vce="nn"`` (the only
    variance mode golden-parity-tested against R in this phase). The
    underlying ``diff_diff._nprobust_port.lprobust`` accepts the broader
    surface (hc0/hc1/hc2/hc3, higher ``p``, interior eval), but those
    paths are not separately parity-tested and remain private until
    Phase 2+ ships dedicated goldens.

    Parameters
    ----------
    d : np.ndarray, shape (G,)
        Regressor (dose ``D_{g,2}`` in HAD).
    y : np.ndarray, shape (G,)
        Outcome (first-difference ``Delta Y_g`` in HAD).
    boundary : float, default=0.0
        Evaluation point ``d_0``. Accepts only ``boundary ~ 0`` (Design 1')
        or ``boundary ~ float(d.min())`` (Design 1 continuous-near-d_lower);
        see Phase 1b's input contract.
    kernel : {"epanechnikov", "triangular", "uniform"}, default="epanechnikov"
    h : float or None, default=None
        Main bandwidth. ``None`` auto-selects ``h`` by calling
        ``diff_diff._nprobust_port.lpbwselect_mse_dpi`` directly with the
        supplied ``cluster``, ``vce``, and ``nnmatch`` so the selector
        and the final fit use the same estimator. ``b`` is then set to
        ``h`` per nprobust's ``rho=1`` default; the selector's distinct
        ``b_mse`` is surfaced through
        ``BiasCorrectedFit.bandwidth_diagnostics`` for inspection but
        not applied. If ``h`` is provided and ``b`` is ``None``,
        ``b = h`` likewise.
    b : float or None, default=None
        Bias-correction bandwidth. Pairs with ``h`` (see above). ``b``
        provided without ``h`` raises ``ValueError``.
    alpha : float, default=0.05
        CI level; ``0.05`` gives a 95% CI. Must be in ``(0, 1)``.
    vce : {"nn"}, default="nn"
        Variance-estimation method. Only ``"nn"`` is supported in
        Phase 1c; hc0/hc1/hc2/hc3 are queued for Phase 2+ pending
        dedicated R parity goldens. Passing anything else raises
        ``NotImplementedError``.
    cluster : np.ndarray or None
        Per-observation cluster IDs for cluster-robust variance. Missing
        (NaN) cluster IDs raise ``ValueError`` rather than silently
        dropping rows.
    nnmatch : int, default=3
        Number of nearest neighbors for ``vce="nn"`` residuals.
    weights : np.ndarray or None, default=None
        Per-unit non-negative weights (e.g., survey sampling weights).
        Forwarded to the final ``lprobust`` fit; propagates through
        kernel composition, design matrices, Q.q bias correction, and
        variance matrices. When ``weights=np.ones(G)`` the output is
        bit-identical to the unweighted path. **Known methodology gap**:
        the auto-bandwidth MSE-optimal DPI (Phase 1b) remains unweighted
        in Phase 4.5; pass ``h``/``b`` explicitly for a weight-aware
        bandwidth. See REGISTRY "Weighted extension (Phase 4.5)" for the
        analytic derivation + parity-ceiling note.

    Returns
    -------
    BiasCorrectedFit

    Raises
    ------
    ValueError
        Shape mismatch, non-finite inputs, off-support boundary, negative
        doses, ``alpha`` outside ``(0, 1)``, unknown ``kernel``,
        NaN / None cluster IDs, ``b`` supplied without ``h``, negative or
        non-finite ``weights``, or a rank-deficient window.
    NotImplementedError
        ``vce != "nn"`` (hc0/hc1/hc2/hc3 deferred to Phase 2+ pending
        dedicated R parity goldens); a Design 1 mass-point sample
        (redirects to Phase 2's 2SLS sample-average path per the paper's
        Section 3.2.4).

    Notes
    -----
    Parity against ``nprobust::lprobust(..., bwselect="mse-dpi")`` is
    asserted at ``atol=1e-12`` on ``tau_cl``, ``tau_bc``, ``se_cl``,
    ``se_rb``, ``ci_low``, and ``ci_high`` across the three unclustered
    golden DGPs; DGP 1 and DGP 3 typically land closer to ``1e-13``.
    The Python wrapper computes its own ``z_{1-alpha/2}`` via
    ``scipy.stats.norm.ppf`` inside ``safe_inference()``; R's ``qnorm``
    value is stored in the golden JSON for audit, and the parity harness
    compares Python's CI bounds to R's pre-computed CI bounds, so any
    residual drift is purely the floating-point arithmetic in
    ``tau.bc +/- z * se.rb``, not a critical-value disagreement.
    Clustered DGP 4 achieves bit-parity (``atol=1e-14``) when cluster
    IDs are in first-appearance order; otherwise BLAS reduction
    ordering can drift to ``atol=1e-10``.
    """
    # Zero-weight unit handling (Phase 4.5 survey support). Filter
    # d/y/weights to the positive-weight support BEFORE
    # ``_validate_had_inputs`` runs. Otherwise zero-weight observations
    # at the boundary or at ``d.min()`` taint the Design 1' support
    # heuristic and the mass-point threshold, causing spurious
    # off-support rejections or design misidentification on valid
    # weighted domain fits (e.g. ``SurveyDesign.subpopulation()``).
    # The auto-bandwidth MSE-DPI selector below also sees only
    # positive-weight observations — kernel density + variance estimates
    # at the boundary are incorrect if zero-weight units contaminate
    # the sample. Callers that want the IF aligned with the ORIGINAL
    # ``d`` ordering get zero-padded positions back via
    # ``return_influence=True``: positive-weight positions carry the
    # active IF, zero-weight positions are 0 (consistent with their
    # zero contribution to the fit).
    _positive_mask_full: Optional[np.ndarray] = None
    _n_full_for_if: Optional[int] = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        d_np = np.asarray(d).ravel()
        # Weight validation fires BEFORE the zero-weight filter below,
        # so invalid weights (NaN, Inf, negative, zero-sum) raise
        # consistent ValueErrors regardless of how many zero-weight
        # units the caller also passed. Duplicates the lprobust-level
        # validation locally so errors surface at the public-wrapper
        # boundary with the same messages the port raises downstream.
        if weights.shape[0] != d_np.shape[0]:
            raise ValueError(
                f"weights length ({weights.shape[0]}) does not match " f"d/y ({d_np.shape[0]})."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values (NaN or Inf).")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if np.sum(weights) <= 0:
            raise ValueError("weights sum to zero — no observations have positive weight.")
        if weights.shape[0] == d_np.shape[0]:
            _positive_mask_full = weights > 0.0
            if not bool(_positive_mask_full.all()):
                _n_full_for_if = int(d_np.shape[0])
                d = d_np[_positive_mask_full]
                y = np.asarray(y).ravel()[_positive_mask_full]
                weights = weights[_positive_mask_full]
                if cluster is not None:
                    cluster = np.asarray(cluster).ravel()[_positive_mask_full]
        # NOTE: bandwidth selection (auto mode) on the POSITIVE-weight
        # subset remains unweighted — the plug-in MSE-optimal DPI is
        # not yet weight-aware. Users who want a weight-aware bandwidth
        # should pass ``h``/``b`` that reflect the weighted DGP. See
        # REGISTRY "Weighted extension" subsection.

    if kernel not in _KERNEL_NAME_TO_NPROBUST:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of "
            f"{sorted(_KERNEL_NAME_TO_NPROBUST.keys())}."
        )
    nprobust_kernel = _KERNEL_NAME_TO_NPROBUST[kernel]

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    # Phase 1c public-wrapper vce restriction.
    # Only ``vce="nn"`` is golden-tested against R in Phase 1c. Exposing
    # hc0/hc1/hc2/hc3 on the public surface would ship non-parity-verified
    # inference -- and nprobust's internal hc2/hc3 residual path reuses
    # the p-fit hat-matrix leverage for the q-fit residuals (lprobust.R
    # :229-241), a detail that would need its own R parity anchor before
    # the Python port can advertise it as CCT-2014-compliant. Defer the
    # hc-mode public surface to Phase 2+ with dedicated goldens. The
    # port-level ``diff_diff._nprobust_port.lprobust`` still accepts
    # hc0..hc3 for callers who need the broader surface and accept that
    # the hc-mode behavior has not been separately parity-tested.
    if vce != "nn":
        raise NotImplementedError(
            f"vce={vce!r} is not supported on bias_corrected_local_linear "
            "in Phase 1c. Only vce='nn' is golden-tested against R "
            "nprobust::lprobust in this phase; hc0/hc1/hc2/hc3 are queued "
            "for Phase 2+ pending dedicated R parity goldens. If you need "
            "the hc-mode port path for exploratory use, call "
            "diff_diff._nprobust_port.lprobust directly and accept that "
            "those paths are not separately parity-tested."
        )

    # HAD-scope input validation (shared with Phase 1b via _validate_had_inputs).
    d, y = _validate_had_inputs(d, y, boundary)
    n_total = int(d.shape[0])

    # Reject missing cluster IDs up front (Phase 1b convention). Delegates
    # to the dtype-agnostic `_cluster_has_missing` helper in the port so
    # wrapper, port-level `lprobust`, and `lpbwselect_mse_dpi` all enforce
    # the same missing-sentinel contract across float / object / string
    # dtypes (CI review PR #340 P1 follow-up).
    cluster_arr: Optional[np.ndarray] = None
    if cluster is not None:
        from diff_diff._nprobust_port import _cluster_has_missing

        cluster_arr = np.asarray(cluster).ravel()
        if cluster_arr.shape[0] != n_total:
            raise ValueError(
                f"cluster length ({cluster_arr.shape[0]}) does not match " f"d/y ({n_total})."
            )
        if _cluster_has_missing(cluster_arr):
            raise ValueError(
                "cluster contains missing values (NaN / None). Filter "
                "your data before the call or drop missing observations "
                "explicitly."
            )

    # --- Resolve (h, b) ---
    # nprobust's lprobust() with the default rho=1 sets b = h / rho = h
    # whenever h is unspecified (R lprobust.R:121-124 and 139: even though
    # lpbwselect returns a distinct b_mse, `if (rho>0) b <- h/rho` discards
    # it). Auto mode here matches that behavior to preserve bit-parity. The
    # distinct b_mse from Phase 1b's selector is still surfaced via
    # bandwidth_diagnostics.b_mse for callers that want to inspect or
    # override. The paper (de Chaisemartin et al. 2026) likewise uses a
    # single h*_G throughout Equation 8.
    #
    # In auto mode, cluster / vce / nnmatch are forwarded to
    # ``lpbwselect_mse_dpi`` so bandwidth selection reflects the same
    # estimator the final ``lprobust`` call will use. Calling
    # ``mse_optimal_bandwidth`` (the public wrapper) instead would hard-code
    # ``cluster=None, vce="nn", nnmatch=3`` and silently mismatch the
    # downstream fit — a methodology bug (CI review PR #340 P1).
    bw_source: str
    bw_diag: Optional[BandwidthResult] = None
    if h is None and b is None:
        # Defer heavy import to call time to avoid import-cycle risk.
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        stages = lpbwselect_mse_dpi(
            y=y,
            x=d,
            cluster=cluster_arr,
            eval_point=float(boundary),
            p=1,
            q=2,
            deriv=0,
            kernel=nprobust_kernel,
            bwcheck=21,
            bwregul=1.0,
            vce=vce,
            nnmatch=nnmatch,
            interior=False,
        )
        bw_diag = BandwidthResult(
            h_mse=stages.h_mse_dpi,
            b_mse=stages.b_mse_dpi,
            c_bw=stages.c_bw,
            bw_mp2=stages.bw_mp2,
            bw_mp3=stages.bw_mp3,
            stage_d1_V=stages.stage_d1.V,
            stage_d1_B1=stages.stage_d1.B1,
            stage_d1_B2=stages.stage_d1.B2,
            stage_d1_R=stages.stage_d1.R,
            stage_d2_V=stages.stage_d2.V,
            stage_d2_B1=stages.stage_d2.B1,
            stage_d2_B2=stages.stage_d2.B2,
            stage_d2_R=stages.stage_d2.R,
            stage_b_V=stages.stage_b.V,
            stage_b_B1=stages.stage_b.B1,
            stage_b_B2=stages.stage_b.B2,
            stage_b_R=stages.stage_b.R,
            stage_h_V=stages.stage_h.V,
            stage_h_B1=stages.stage_h.B1,
            stage_h_B2=stages.stage_h.B2,
            stage_h_R=stages.stage_h.R,
            n=int(d.shape[0]),
            kernel=kernel,
            boundary=float(boundary),
        )
        h_val = float(bw_diag.h_mse)
        b_val = h_val  # rho=1 default to match nprobust
        bw_source = "auto"
    elif h is not None:
        h_val = float(h)
        if b is None:
            b_val = h_val  # nprobust rho=1 default
        else:
            b_val = float(b)
        if not (np.isfinite(h_val) and h_val > 0):
            raise ValueError(f"h must be finite and positive; got {h!r}.")
        if not (np.isfinite(b_val) and b_val > 0):
            raise ValueError(f"b must be finite and positive; got {b!r}.")
        bw_source = "user"
    else:
        # h is None but b is not None: ambiguous.
        raise ValueError(
            "b provided without h; pass both bandwidths explicitly or "
            "leave both as None to auto-select via mse_optimal_bandwidth."
        )

    # --- Call the port ---
    # Defer heavy import to call time to avoid import-cycle risk.
    from diff_diff._nprobust_port import lprobust

    result = lprobust(
        y=y,
        x=d,
        eval_point=float(boundary),
        h=h_val,
        b=b_val,
        p=1,
        q=2,
        deriv=0,
        kernel=nprobust_kernel,
        vce=vce,
        cluster=cluster_arr,
        nnmatch=nnmatch,
        bwcheck=21,
        weights=weights,
        return_influence=return_influence,
    )

    # --- Bias-corrected CI via safe_inference (NaN-safe gate) ---
    # When se_robust is zero, negative, or non-finite (e.g., exact-fit
    # cases where the residual vector collapses), ALL inference fields —
    # including the CI — must return NaN. This enforces the repo-wide
    # inference contract (CLAUDE.md Key Design Pattern #6; CI review
    # PR #340 P0) rather than returning a misleading zero-width or
    # infinite CI. safe_inference computes the critical value z_{1-α/2}
    # via scipy.stats.norm.ppf; the parity tests compare Python's
    # scipy-computed ci_low/ci_high to R's qnorm-computed ci_low/ci_high
    # stored in the golden JSON. The golden JSON also exports R's raw
    # `z` value for audit/reference so a reviewer can verify the two
    # critical values agree to machine precision.
    from diff_diff.utils import safe_inference

    _, _, (ci_low, ci_high) = safe_inference(result.tau_bc, result.se_rb, alpha=float(alpha))

    # Zero-pad the IF back to the ORIGINAL (pre-positive-weight-filter)
    # sample ordering when the caller passed a zero-weight vector, so
    # downstream estimator-level Binder-TSL composition lines up with
    # the full survey design rather than with the positive-weight
    # subset. Zero-weight positions get IF=0 (consistent with their
    # zero contribution to the fit); the survey-design variance
    # (``compute_survey_if_variance``) can then preserve PSU / stratum
    # / df_survey structure of the full design — the correct
    # subpopulation / domain-analysis convention.
    if_out = result.influence_function
    if if_out is not None and _positive_mask_full is not None and _n_full_for_if is not None:
        if_full = np.zeros(_n_full_for_if, dtype=np.float64)
        if_full[_positive_mask_full] = if_out
        if_out = if_full

    return BiasCorrectedFit(
        estimate_classical=result.tau_cl,
        estimate_bias_corrected=result.tau_bc,
        se_classical=result.se_cl,
        se_robust=result.se_rb,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=float(alpha),
        h=result.h,
        b=result.b,
        bandwidth_source=bw_source,
        bandwidth_diagnostics=bw_diag,
        n_used=result.n_used,
        n_total=int(_n_full_for_if) if _n_full_for_if is not None else n_total,
        kernel=kernel,
        boundary=float(boundary),
        influence_function=if_out,
    )
