"""
Heterogeneous Adoption Difference-in-Differences (HAD) estimator (Phase 2a).

Implements the de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026)
Weighted-Average-Slope (WAS) estimator with three design-dispatch paths.
All three paths produce a beta-scale point estimate of the form
``(mean(Delta Y) - [boundary limit]) / [expected dose gap]`` (Design 1
family) or the Wald-IV ratio (mass-point), then route inference through
:func:`diff_diff.utils.safe_inference`.

1. Design 1' (``continuous_at_zero``): ``d_lower = 0``, boundary density
   continuous at zero, Assumption 3. Theorem 1 / Equation 3
   (identification); Equation 7 (sample estimator):

       beta = (E[Delta Y] - lim_{d v 0} E[Delta Y | D_2 <= d]) / E[D_2]

   The bias-corrected local-linear fit at ``boundary = 0`` estimates
   the boundary limit; ``E[D_2]`` and ``E[Delta Y]`` are plugin sample
   means.

2. Design 1 continuous-near-d_lower (``continuous_near_d_lower``):
   ``d_lower > 0``, continuous boundary density, Assumption 5 or 6.
   Theorem 3 / Equation 11 (``WAS_{d_lower}`` under Assumption 6;
   Theorem 4 is the QUG null test, not this estimand):

       beta = (E[Delta Y] - lim_{d v d_lower} E[Delta Y | D_2 <= d])
              / E[D_2 - d_lower]

   The local-linear fit is anchored at ``d_lower`` via the regressor
   shift ``D' = D - d_lower``, evaluated at ``boundary = 0`` on the
   shifted scale.

3. Design 1 mass-point (``mass_point``): ``d_lower > 0``, modal fraction
   at ``d.min()`` exceeds 2%. Sample-average 2SLS with instrument
   ``Z_g = 1{D_{g,2} > d_lower}`` (paper Section 3.2.4). Point estimate
   equals the Wald-IV ratio
   ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``. Standard
   error via the structural-residual 2SLS sandwich
   (see ``_fit_mass_point_2sls``).

Phase 2a ships the single-period WAS estimator (``aggregate="overall"``).
Phase 2b adds the multi-period event-study extension (paper Appendix B.2)
via ``aggregate="event_study"``: per-horizon WAS estimates with pointwise
CIs, including pre-period placebos, reusing the three Phase 2a design
paths on per-horizon first differences anchored at ``Y_{g, F-1}``.
Staggered-timing panels are auto-filtered to the last-treatment cohort
per paper Appendix B.2 prescription.

Infrastructure reused from Phase 1:
- ``diff_diff.local_linear.bias_corrected_local_linear`` (Phase 1c)
- ``diff_diff.local_linear.BiasCorrectedFit``, ``BandwidthResult``
- ``diff_diff.utils.safe_inference`` (NaN-safe CI gating)
- ``diff_diff.prep.balance_panel`` (panel validation)

References
----------
- de Chaisemartin, C., Ciccia, D., D'Haultfoeuille, X., & Knau, F. (2026).
  Difference-in-Differences Estimators When No Unit Remains Untreated.
  arXiv:2405.04465v6.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust
  nonparametric confidence intervals for regression-discontinuity designs.
  Econometrica, 82(6), 2295-2326.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd

from diff_diff.bootstrap_chunking import (
    compute_block_size,
    iter_survey_multiplier_weight_blocks,
    iter_weight_blocks,
)
from diff_diff.local_linear import (
    BandwidthResult,
    BiasCorrectedFit,
    bias_corrected_local_linear,
)
from diff_diff.survey import (
    SurveyMetadata,
    compute_survey_metadata,
)
from diff_diff.utils import safe_inference

__all__ = [
    "HeterogeneousAdoptionDiD",
    "HeterogeneousAdoptionDiDResults",
    "HeterogeneousAdoptionDiDEventStudyResults",
]


# =============================================================================
# Module-level constants
# =============================================================================

# REGISTRY line 2320 auto-detect boundary: d.min() < 0.01 * median(|d|) resolves
# to Design 1' (continuous_at_zero). Distinct from Phase 1c's stricter
# _DESIGN_1_PRIME_RATIO = 0.05 in ``local_linear.py`` which is a plausibility
# guard on the nonparametric fit, not the auto-detect rule.
_CONTINUOUS_AT_ZERO_THRESHOLD = 0.01

# REGISTRY line 2320 mass-point rule: modal fraction at d.min() > 0.02 resolves
# to Design 1 mass-point. Mirrored (not imported) from
# ``diff_diff.local_linear._validate_had_inputs`` where the same constant
# enforces the upstream mass-point rejection on the nonparametric path.
_MASS_POINT_THRESHOLD = 0.02

_VALID_DESIGNS = (
    "auto",
    "continuous_at_zero",
    "continuous_near_d_lower",
    "mass_point",
)
_VALID_AGGREGATES = ("overall", "event_study")
# Mass-point 2SLS supports: classical, hc1 (HC0 is an unscaled variant not
# publicly exposed; hc2 and hc2_bm raise NotImplementedError pending a
# 2SLS-specific leverage derivation and R parity anchor).
_MASS_POINT_VCOV_SUPPORTED = ("classical", "hc1")
_MASS_POINT_VCOV_UNSUPPORTED = ("hc2", "hc2_bm")

# Extensive-margin / positive-untreated-mass warning (TODO L74). The paper (de
# Chaisemartin et al. 2026, Section 2 / Assumption 3) defines HAD for the case
# where no genuine untreated group exists, and recommends users with a real
# untreated mass consider a standard DiD instead. The paper prescribes "warn"
# but NO numeric cutoff, and explicitly RETAINS small untreated shares (Garrett
# et al.: 12/2954 ~ 0.4%, with nominal coverage), so this fit-time UserWarning
# fires only above a library-convention fraction of EXACTLY-zero post-period
# doses. Overall path ONLY — the event-study path requires never-treated units
# per Appendix B.2, so an untreated mass is expected there, not a misuse signal.
_HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC = 0.10

# Target-parameter label per design. Design 1' targets the WAS (Assumption 3);
# Design 1 targets WAS_{d_lower} (Assumption 5 or 6), which also applies to
# the mass-point path (paper Section 3.2.4).
_TARGET_PARAMETER = {
    "continuous_at_zero": "WAS",
    "continuous_near_d_lower": "WAS_d_lower",
    "mass_point": "WAS_d_lower",
}


# =============================================================================
# JSON-serialization helpers
# =============================================================================


def _json_safe_scalar(x: Any) -> Any:
    """Coerce a scalar to a JSON-serializable type.

    - NumPy scalars (``np.int64``, ``np.float64``, ``np.bool_``) ->
      native Python via ``.item()``
    - ``pd.Timestamp`` / ``pd.Timedelta`` -> ISO 8601 string via
      ``.isoformat()``
    - Everything else returned as-is.

    The ``to_dict`` methods use this to keep the returned dict
    serializable via ``json.dumps`` regardless of the underlying
    pandas/numpy dtype of the time / first_treat columns.
    """
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except (AttributeError, ValueError, TypeError):
            return x
    return x


def _json_safe_filter_info(
    filter_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize a ``filter_info`` dict to JSON-safe scalars.

    Returns ``None`` unchanged; otherwise coerces ``F_last`` and each
    entry in ``dropped_cohorts`` via :func:`_json_safe_scalar`. Int
    counts are cast to ``int`` for stability.
    """
    if filter_info is None:
        return None
    return {
        "F_last": _json_safe_scalar(filter_info.get("F_last")),
        "n_kept": int(filter_info.get("n_kept", 0)),
        "n_dropped": int(filter_info.get("n_dropped", 0)),
        "dropped_cohorts": [_json_safe_scalar(c) for c in filter_info.get("dropped_cohorts", [])],
    }


# =============================================================================
# Results dataclass
# =============================================================================


@dataclass
class HeterogeneousAdoptionDiDResults:
    """Estimator output for :class:`HeterogeneousAdoptionDiD`.

    NaN-safe inference: the three downstream fields ``t_stat``,
    ``p_value``, and ``conf_int`` are routed through
    :func:`diff_diff.utils.safe_inference`, which returns NaN on all
    three whenever ``se`` is non-finite, zero, or negative. ``att`` and
    ``se`` themselves are RAW estimator outputs from the chosen fit
    path and are NOT gated by ``safe_inference``:

    - On the degenerate fit configurations (constant outcome on the
      continuous paths, all-units-at-d_lower / no-dose-variation on the
      mass-point path), the fit path explicitly returns
      ``(att=nan, se=nan)``, which combined with the safe-inference
      gate yields all five fields NaN together.
    - On the degenerate CR1 cluster configuration (mass-point path
      with a single cluster), ``_fit_mass_point_2sls`` returns
      ``(att=beta_hat, se=nan)`` - ``att`` stays finite because the
      Wald-IV ratio is well defined, but the cluster-robust SE is
      not, so ``se`` is NaN and the downstream triple becomes NaN
      via the safe-inference gate.

    So the guaranteed NaN coupling is on the downstream triple
    (``t_stat``, ``p_value``, ``conf_int``), not on ``att``. The
    ``assert_nan_inference`` fixture in ``tests/conftest.py`` checks
    the downstream triple against the gate contract and does not
    assume ``att`` is NaN.

    Attributes
    ----------
    att : float
        Point estimate of the WAS parameter on the beta-scale.

        - Design 1' (paper Theorem 1 / Equation 3 identification;
          Equation 7 sample estimator):
          ``att = (mean(ΔY) - tau_bc) / D_bar``
          where ``tau_bc`` is the bias-corrected local-linear estimate
          of ``lim_{d v 0} E[ΔY | D_2 <= d]`` and
          ``D_bar = (1/G) * sum(D_{g,2})``.
        - Design 1 continuous-near-d_lower (paper Theorem 3 /
          Equation 11, ``WAS_{d_lower}`` under Assumption 6):
          ``att = (mean(ΔY) - tau_bc) / mean(D_2 - d_lower)``
          where ``tau_bc`` is the bias-corrected local-linear estimate
          of ``lim_{d v d_lower} E[ΔY | D_2 <= d]``.
        - Mass-point (paper Section 3.2.4): the Wald-IV / 2SLS
          coefficient directly -
          ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``.
    se : float
        Standard error on the beta-scale. For continuous designs:

        - Unweighted: CCT-2014 robust SE from Phase 1c divided by ``|den|``
          (``den`` = raw denominator).
        - ``survey_design=SurveyDesign(...)``: Binder (1983) Taylor-series
          linearization of the per-unit IF (bias-corrected scale,
          aligned with ``tau_bc``) routed through
          :func:`compute_survey_if_variance` for PSU-aggregated,
          FPC/strata-adjusted variance, divided by ``|den|``.

        In both cases the higher-order variance from ``mean(ΔY)`` is
        dominated by the nonparametric boundary estimate in large
        samples and is not included in the leading-order formula. For
        mass-point, the 2SLS structural-residual sandwich SE.
    t_stat, p_value, conf_int : inference fields
        Routed through ``safe_inference``; NaN when SE is non-finite.
    alpha : float
        CI level used at fit time (0.05 for a 95% CI).
    design : str
        Resolved design mode: ``"continuous_at_zero"``,
        ``"continuous_near_d_lower"``, or ``"mass_point"``. ``"auto"`` is
        resolved to one of the three concrete modes before storing.
    target_parameter : str
        Estimand label: ``"WAS"`` for Design 1', ``"WAS_d_lower"`` for the
        other two. Pins the estimand semantically even when two designs
        share the same divisor.
    d_lower : float
        Support infimum ``d_lower``. ``0.0`` for Design 1';
        ``float(d.min())`` for the other two.
    dose_mean : float
        ``D_bar = (1/G) * sum(D_{g,2})``.
    n_obs : int
        Number of units contributing to the estimator (post panel
        aggregation to unit-level first differences).
    n_treated : int
        Number of units with ``D_{g,2} > d_lower``.
    n_control : int
        Number of units at or below ``d_lower`` (the "not-treated" subset).
    n_mass_point : int or None
        Mass-point path only: number of units with ``D_{g,2} == d_lower``.
        ``None`` on continuous paths.
    n_above_d_lower : int or None
        Mass-point path only: number of units with ``D_{g,2} > d_lower``.
        ``None`` on continuous paths.
    inference_method : str
        ``"analytical_nonparametric"`` (continuous designs) or
        ``"analytical_2sls"`` (mass-point).
    vcov_type : str or None
        Effective variance-covariance family used. On continuous paths:
        ``None`` when unclustered (they use the CCT-2014 robust SE from
        Phase 1c, not the library's ``vcov_type`` enum) and ``"cr1"`` when
        ``cluster`` is supplied (the CCT SE becomes cluster-robust; paired
        with ``inference_method="analytical_nonparametric"`` this identifies
        the clustered CCT variance, distinct from the mass-point 2SLS CR1
        sandwich). Mass-point: ``"classical"`` or ``"hc1"`` when ``cluster``
        is not supplied, and ``"cr1"`` whenever ``cluster`` is supplied
        (cluster-robust CR1 is computed regardless of the requested
        ``vcov_type`` because classical/hc1 + cluster collapses to the same
        CR1 sandwich). Downstream consumers reading ``result.to_dict()`` can
        inspect this field directly to determine the effective SE family.
    cluster_name : str or None
        Column name of the cluster variable when cluster-robust SE is
        requested — on the mass-point path (2SLS CR1) or the continuous
        paths (clustered CCT SE, Phase 2a). ``None`` otherwise.
    survey_metadata : SurveyMetadata or None
        Repo-standard survey metadata dataclass from
        :class:`diff_diff.survey.SurveyMetadata`. ``None`` when ``fit()``
        was called without ``survey_design=``; populated on the weighted
        paths via
        :func:`diff_diff.survey.compute_survey_metadata`. Exposes
        ``weight_type``, ``effective_n``, ``design_effect``,
        ``sum_weights``, ``n_strata``, ``n_psu``, ``weight_range``, and
        ``df_survey`` for downstream reporting consumers (BusinessReport,
        DiagnosticReport) that read these fields via attribute access.
        HAD-specific inference-method info (Binder-TSL) is carried on
        ``inference_method`` and ``variance_formula``.
    bandwidth_diagnostics : BandwidthResult or None
        Full Phase 1b MSE-DPI selector output on the continuous paths
        (when bandwidths were auto-selected). ``None`` on the mass-point
        path (parametric, no bandwidth).
    bias_corrected_fit : BiasCorrectedFit or None
        Full Phase 1c bias-corrected local-linear fit on the continuous
        paths. ``None`` on the mass-point path.
    """

    # Point estimate + inference (safe_inference-gated)
    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    alpha: float

    # Design metadata
    design: str
    target_parameter: str
    d_lower: float
    dose_mean: float

    # Sample counts
    n_obs: int
    n_treated: int
    n_control: int
    n_mass_point: Optional[int]
    n_above_d_lower: Optional[int]

    # Inference metadata
    inference_method: str
    vcov_type: Optional[str]
    cluster_name: Optional[str]
    survey_metadata: Optional[SurveyMetadata]

    # Nonparametric-only diagnostics
    bandwidth_diagnostics: Optional[BandwidthResult]
    bias_corrected_fit: Optional[BiasCorrectedFit]

    # Phase 4.5 weighted-path extras (optional so unweighted fits stay unchanged)
    variance_formula: Optional[str] = None
    """HAD-specific label for the SE formula on weighted fits, populated
    on BOTH continuous and mass-point designs (Phase 4.5 A / B):
    ``"survey_binder_tsl"`` (continuous, Binder 1983 TSL with PSU/strata/FPC
    under ``survey_design=SurveyDesign(...)``) or ``"survey_binder_tsl_2sls"``
    (mass-point, Binder 1983 TSL under ``survey_design=``; requires
    ``vcov_type="hc1"`` / ``robust=True`` — the mass-point survey path rejects
    ``vcov_type="classical"``, and ``cluster=`` + ``survey_design=`` is
    rejected, so PSU clustering is expressed via
    ``SurveyDesign(weights='<weight_col>', psu='<cluster_col>')``).
    ``None`` on unweighted fits.
    Orthogonal to ``survey_metadata`` which is the repo-standard
    :class:`diff_diff.survey.SurveyMetadata` shared with downstream
    report/diagnostic consumers (no HAD-specific leakage)."""
    effective_dose_mean: Optional[float] = None
    """Weighted denominator used by the beta-scale rescaling, populated
    on weighted fits across all designs: ``sum(w_g · D_g) / sum(w_g)``
    on ``continuous_at_zero``, ``sum(w_g · (D_g - d_lower)) / sum(w_g)``
    on ``continuous_near_d_lower``, and the weighted Wald-IV dose gap
    ``mean(D | Z=1, w) - mean(D | Z=0, w)`` on ``mass_point`` (where
    ``Z = 1{D > d_lower}``). On the continuous designs reduces
    bit-exactly to ``dose_mean`` / ``mean(D - d_lower)`` when weights
    are uniform or absent. ``None`` when ``fit()`` was called without
    ``survey_design=`` (use ``dose_mean`` there). Exists because
    ``dose_mean`` is the raw sample mean of the
    dose column; under weighted fits the estimator's actual denominator
    is the weighted form above, and users reconstructing the β-scale
    value by hand need the weighted one."""

    def __repr__(self) -> str:
        base = (
            f"HeterogeneousAdoptionDiDResults("
            f"att={self.att:.4f}, se={self.se:.4f}, "
            f"design={self.design!r}, n_obs={self.n_obs}"
        )
        # Surface weighted-path identity when the fit was weighted, so the
        # one-line repr makes it unambiguous which inference family was
        # used (pweight-shortcut vs full Binder-TSL survey) and the
        # effective denominator flows into ad-hoc log output.
        if self.variance_formula is not None:
            base += f", variance_formula={self.variance_formula!r}"
        if self.effective_dose_mean is not None:
            base += f", effective_dose_mean={self.effective_dose_mean:.4g}"
        return base + ")"

    def summary(self) -> str:
        """Formatted summary table."""
        width = 72
        conf_level = int((1 - self.alpha) * 100)
        lines = [
            "=" * width,
            "HeterogeneousAdoptionDiD Estimation Results".center(width),
            "=" * width,
            "",
            f"{'Design:':<30} {self.design:>20}",
            f"{'Target parameter:':<30} {self.target_parameter:>20}",
            f"{'d_lower:':<30} {self.d_lower:>20.6g}",
            f"{'D_bar (dose mean):':<30} {self.dose_mean:>20.6g}",
            f"{'Observations (units):':<30} {self.n_obs:>20}",
            f"{'Above d_lower:':<30} {self.n_treated:>20}",
            f"{'At/below d_lower:':<30} {self.n_control:>20}",
        ]
        if self.n_mass_point is not None:
            lines.append(f"{'At d_lower (mass point):':<30} {self.n_mass_point:>20}")
        if self.n_above_d_lower is not None:
            lines.append(f"{'Strictly above d_lower:':<30} {self.n_above_d_lower:>20}")
        lines.append(f"{'Inference method:':<30} {self.inference_method:>20}")
        if self.vcov_type is not None:
            if self.cluster_name is not None:
                # Cluster-robust (CR1): the stored vcov_type is already "cr1",
                # but render with the cluster column for clarity.
                label = f"CR1 at {self.cluster_name}"
            else:
                label = self.vcov_type
            lines.append(f"{'Variance:':<30} {label:>20}")
        if self.bandwidth_diagnostics is not None:
            bw = self.bandwidth_diagnostics
            lines.append(f"{'Bandwidth h (MSE-DPI):':<30} {bw.h_mse:>20.6g}")
            lines.append(f"{'Bandwidth b (bias):':<30} {bw.b_mse:>20.6g}")
        if self.bias_corrected_fit is not None:
            bc = self.bias_corrected_fit
            lines.append(f"{'Bandwidth h used:':<30} {bc.h:>20.6g}")
            lines.append(f"{'Obs in window (n_used):':<30} {bc.n_used:>20}")
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            vf_label = self.variance_formula or "unknown"
            lines.append(f"{'Variance formula:':<30} {vf_label:>20}")
            lines.append(f"{'Effective sample size:':<30} {sm.effective_n:>20.6g}")
            if self.effective_dose_mean is not None:
                lines.append(
                    f"{'Weighted D̄ (denominator):':<30} " f"{self.effective_dose_mean:>20.6g}"
                )
            if sm.df_survey is not None:
                lines.append(f"{'Survey df:':<30} {sm.df_survey:>20}")
        param_label = self.target_parameter
        lines.extend(
            [
                "",
                "-" * width,
                (
                    f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10}"
                ),
                "-" * width,
                (
                    f"{param_label:<15} {self.att:>12.4f} {self.se:>12.4f} "
                    f"{self.t_stat:>10.3f} {self.p_value:>10.4f}"
                ),
                "-" * width,
                "",
                (
                    f"{conf_level}% Confidence Interval: "
                    f"[{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]"
                ),
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a dict of scalars + weighted-path surfaces.

        Always-present keys mirror the dataclass fields: ``att``, ``se``,
        ``t_stat``, ``p_value``, ``conf_int_lower`` / ``conf_int_upper``,
        ``alpha``, ``design``, ``target_parameter``, ``d_lower``,
        ``dose_mean``, ``n_obs`` / ``n_treated`` / ``n_control`` /
        ``n_mass_point`` / ``n_above_d_lower``, ``inference_method``,
        ``vcov_type``, ``cluster_name``.

        Weighted-path keys (``None`` on unweighted fits):

        - ``survey_metadata``: repo-standard
          :class:`diff_diff.survey.SurveyMetadata` dataclass (object, not
          dict) carrying ``weight_type`` / ``effective_n`` /
          ``design_effect`` / ``sum_weights`` / ``weight_range`` +
          ``n_strata`` / ``n_psu`` / ``df_survey`` (populated from the
          resolved ``survey_design=``).
        - ``variance_formula``: HAD-specific SE label, populated on BOTH
          continuous and mass-point designs (Phase 4.5 A / B):
          ``"survey_binder_tsl"`` (continuous, Binder 1983 TSL under
          ``survey_design=``) or ``"survey_binder_tsl_2sls"`` (mass-point,
          Binder 1983 TSL under ``survey_design=``; requires
          ``vcov_type="hc1"`` / ``robust=True`` — the mass-point survey path
          rejects ``vcov_type="classical"``). See the field docstring above
          for the full contract.
        - ``effective_dose_mean``: weighted denominator used by the
          beta-scale rescaling - weighted ``mean(D)`` on
          ``continuous_at_zero``, weighted ``mean(D - d_lower)`` on
          ``continuous_near_d_lower``, or the weighted Wald-IV dose gap
          ``mean(D | Z=1, w) - mean(D | Z=0, w)`` on ``mass_point``."""
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "alpha": self.alpha,
            "design": self.design,
            "target_parameter": self.target_parameter,
            "d_lower": self.d_lower,
            "dose_mean": self.dose_mean,
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_mass_point": self.n_mass_point,
            "n_above_d_lower": self.n_above_d_lower,
            "inference_method": self.inference_method,
            "vcov_type": self.vcov_type,
            "cluster_name": self.cluster_name,
            "survey_metadata": self.survey_metadata,
            "variance_formula": self.variance_formula,
            "effective_dose_mean": self.effective_dose_mean,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class HeterogeneousAdoptionDiDEventStudyResults:
    """Event-study results for :class:`HeterogeneousAdoptionDiD` (Phase 2b).

    Per-horizon arrays align with ``event_times`` by index; all per-horizon
    arrays have shape ``(n_horizons,)``. The anchor horizon ``e = -1``
    (i.e., ``t = F - 1``) is NOT included because
    ``Y_{g, F-1} - Y_{g, F-1} = 0`` trivially and the WAS is not identified
    there.

    Per-horizon inference fields (``t_stat``, ``p_value``, ``conf_int_low``,
    ``conf_int_high``) are NaN-coupled to the per-horizon ``se`` via
    :func:`diff_diff.utils.safe_inference`; ``att`` and ``se`` themselves
    are raw estimator outputs from the chosen design path on each
    horizon's first differences.

    Design resolution is SHARED across horizons: the design, ``d_lower``,
    ``target_parameter``, and ``inference_method`` are single scalars
    determined once from the post-period dose distribution ``D_{g, F}``
    (paper Appendix B.2 convention — the dose regressor is invariant
    across event-time horizons).

    Attributes
    ----------
    event_times : np.ndarray, shape (n_horizons,)
        Integer event-time labels ``e = t - F``, sorted ascending.
        Excludes ``e = -1`` (the anchor). Post-period horizons have
        ``e >= 0``; pre-period placebos have ``e <= -2``.
    att : np.ndarray, shape (n_horizons,)
        Per-horizon WAS point estimate on the beta-scale (see
        :class:`HeterogeneousAdoptionDiDResults.att` for the per-design
        formula, applied to ``ΔY_t = Y_{g,t} - Y_{g,F-1}``).
    se : np.ndarray, shape (n_horizons,)
        Per-horizon standard error on the beta-scale. Three regimes:

        - **Unweighted**: per-horizon INDEPENDENT analytical sandwich
          (continuous: CCT-2014 robust divided by ``|den|``;
          mass-point: structural-residual 2SLS sandwich via
          ``_fit_mass_point_2sls``). No cross-horizon covariance.
        - **``survey_design=``**: each horizon composes Binder (1983)
          Taylor-series linearization via
          :func:`compute_survey_if_variance` on the per-unit β̂-scale
          IF (continuous + mass-point both route through the same
          helper). ``df_survey`` threads into ``safe_inference`` for
          t-inference.

        Pointwise CIs are always populated; a simultaneous confidence
        band (``cband_*`` below) is available on the weighted/survey path
        OR whenever ``cluster=`` is set (the clustered band fires even on
        an unweighted fit). Joint cross-horizon analytical covariance is
        not computed in this release (tracked in TODO.md).
    t_stat, p_value : np.ndarray, shape (n_horizons,)
        Per-horizon inference triple element.
    conf_int_low, conf_int_high : np.ndarray, shape (n_horizons,)
        Per-horizon CI endpoints at level ``alpha``.
    n_obs_per_horizon : np.ndarray, shape (n_horizons,)
        Per-horizon sample size (units contributing at that event time).
        In Phase 2b this equals ``n_units`` for every horizon because the
        validator rejects NaN in outcome / dose / unit columns upstream;
        tracked as a field for future flexibility (e.g., per-period
        missingness).
    alpha : float
        CI level used at fit time (0.05 for a 95% CI).
    design : str
        Resolved design mode, shared across horizons:
        ``"continuous_at_zero"``, ``"continuous_near_d_lower"``, or
        ``"mass_point"``.
    target_parameter : str
        Estimand label: ``"WAS"`` for Design 1' (continuous_at_zero),
        ``"WAS_d_lower"`` for the other two.
    d_lower : float
        Support infimum used for all horizons. ``0.0`` for Design 1';
        ``float(d.min())`` otherwise.
    dose_mean : float
        ``D_bar = (1/G) * sum(D_{g,F})`` computed on the fit sample (after
        the staggered last-cohort filter, if applied).
    F : object
        First-treatment period label (arbitrary dtype — int, str,
        datetime). Identified by the multi-period dose invariant from the
        fitted data.
    n_units : int
        Number of unique units contributing to the fit. After staggered
        auto-filter: last-cohort units PLUS never-treated (``first_treat = 0``)
        units retained as the untreated-group comparison per paper
        Appendix B.2. Only earlier-treated cohorts are dropped.
    inference_method : str
        ``"analytical_nonparametric"`` (continuous designs) or
        ``"analytical_2sls"`` (mass-point). Shared across horizons.
    vcov_type : str or None
        Effective variance-covariance family. Mass-point path:
        ``"classical"``, ``"hc1"``, or ``"cr1"`` when ``cluster=`` is
        supplied. Continuous paths: ``None`` (CCT-2014 robust SE), or
        ``"cr1"`` when ``cluster=`` is supplied (cluster-robust CCT SE).
    cluster_name : str or None
        Column name of the cluster variable when cluster-robust SE is
        requested. ``None`` otherwise.
    survey_metadata : SurveyMetadata or None
        Repo-standard survey metadata dataclass from
        :class:`diff_diff.survey.SurveyMetadata`. ``None`` when
        ``fit()`` was called without ``survey_design=``;
        populated on the weighted event-study path (Phase 4.5 B). See
        :class:`HeterogeneousAdoptionDiDResults.survey_metadata` for
        the attribute contract.
    variance_formula : str or None
        Per-horizon variance family (applied uniformly across horizons).
        ``"survey_binder_tsl"`` / ``"survey_binder_tsl_2sls"`` on the
        ``survey_design=`` path. ``None`` on unweighted fits.
    effective_dose_mean : float or None
        Weighted denominator used by the β̂-scale rescaling (continuous
        paths: weighted sample mean of ``d`` or ``d - d_lower``;
        mass-point: weighted Wald-IV dose gap). ``None`` on unweighted
        fits.
    cband_low, cband_high : np.ndarray or None, shape (n_horizons,)
        Simultaneous confidence-band endpoints constructed by the
        multiplier-bootstrap sup-t procedure. ``None`` when
        ``fit(..., cband=False)`` is passed or on unweighted, unclustered
        fits; a clustered fit (``cluster=``) populates the band even when
        unweighted. Horizons with ``se <= 0`` or non-finite ``se`` are NaN
        (matches the pointwise inference gate from ``safe_inference``).
    cband_crit_value : float or None
        Sup-t multiplier-bootstrap critical value at level
        ``1 - alpha``. Under a trivial resolved design (no strata /
        PSU / FPC) at ``H=1`` reduces to ``Φ⁻¹(1 − alpha/2) ≈ 1.96``
        up to Monte Carlo error; under stratified designs the helper
        applies PSU-aggregation + stratum-demeaning + ``sqrt(n_h /
        (n_h - 1))`` small-sample correction so the bootstrap
        variance matches the analytical Binder-TSL target term-for-
        term.
    cband_method : str or None
        ``"multiplier_bootstrap"`` on the weighted/survey event-study band,
        ``"cluster_multiplier_bootstrap"`` on the clustered band, else
        ``None``.
    cband_n_bootstrap : int or None
        Number of multiplier-bootstrap replicates used to compute the
        sup-t critical value.
    bandwidth_diagnostics : list[BandwidthResult] or None
        Per-horizon bandwidth diagnostics on the continuous paths;
        ``None`` on the mass-point path. When non-None, aligned with
        ``event_times`` by index.
    bias_corrected_fit : list[BiasCorrectedFit] or None
        Per-horizon bias-corrected fit on the continuous paths; ``None``
        on the mass-point path. When non-None, aligned with
        ``event_times`` by index.
    filter_info : dict or None
        Populated when the staggered-timing last-cohort auto-filter fires.
        Keys: ``"F_last"`` (kept cohort label), ``"n_kept"`` (units
        retained), ``"n_dropped"`` (units dropped), ``"dropped_cohorts"``
        (list of dropped cohort labels). ``None`` when no filter was
        applied.
    """

    # Per-horizon arrays
    event_times: np.ndarray
    att: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    conf_int_low: np.ndarray
    conf_int_high: np.ndarray
    n_obs_per_horizon: np.ndarray

    # Shared metadata
    alpha: float
    design: str
    target_parameter: str
    d_lower: float
    dose_mean: float
    F: Any
    n_units: int
    inference_method: str
    vcov_type: Optional[str]
    cluster_name: Optional[str]
    survey_metadata: Optional[SurveyMetadata]

    # Per-horizon diagnostics (lists, None on mass-point).
    # List entries may be None for horizons where the continuous-path fit
    # caught a degenerate bandwidth-selector failure (constant / perfectly-
    # linear outcome); att / se for those horizons are NaN as well.
    bandwidth_diagnostics: Optional[List[Optional[BandwidthResult]]]
    bias_corrected_fit: Optional[List[Optional[BiasCorrectedFit]]]

    # Staggered auto-filter metadata
    filter_info: Optional[Dict[str, Any]]

    # Phase 4.5 B weighted / survey-path extras (optional so unweighted
    # fits stay unchanged). ``variance_formula`` / ``effective_dose_mean``
    # are None on unweighted fits; the ``cband_*`` fields are also populated
    # on unweighted CLUSTERED fits (Phase 2b clustered band).
    variance_formula: Optional[str] = None
    """Per-horizon variance family label (applied uniformly across all
    horizons in the fit). One of ``"survey_binder_tsl"`` /
    ``"survey_binder_tsl_2sls"`` (when a SurveyDesign was supplied via
    ``survey_design=``; continuous / mass-point) or ``None`` on unweighted
    fits. Mirrors the static-path ``variance_formula`` field."""
    effective_dose_mean: Optional[float] = None
    """Weighted denominator used by the β̂-scale rescaling. For continuous
    designs: weighted ``sum(w · d)/sum(w)`` (continuous_at_zero) or
    ``sum(w · (d − d_lower))/sum(w)`` (continuous_near_d_lower). For
    mass-point: weighted Wald-IV dose gap. ``None`` on unweighted fits."""
    cband_low: Optional[np.ndarray] = None
    """Simultaneous confidence-band lower endpoints, shape ``(n_horizons,)``.
    ``None`` when ``cband=False``, or on unweighted, unclustered fits (a
    clustered fit populates the band even when unweighted). Derived from the
    multiplier-bootstrap sup-t critical value:
    ``cband_low[e] = att[e] − cband_crit_value * se[e]``."""
    cband_high: Optional[np.ndarray] = None
    """Simultaneous confidence-band upper endpoints, shape
    ``(n_horizons,)``. See ``cband_low``."""
    cband_crit_value: Optional[float] = None
    """Sup-t multiplier-bootstrap critical value at level ``1 - alpha``.
    Reduces to ``Φ⁻¹(1 − alpha/2) ≈ 1.96`` at ``H=1`` up to Monte Carlo
    error. ``None`` when ``cband=False`` or on unweighted, unclustered fits."""
    cband_method: Optional[str] = None
    """``"multiplier_bootstrap"`` (weighted/survey band) or
    ``"cluster_multiplier_bootstrap"`` (clustered band) when populated, else
    ``None``."""
    cband_n_bootstrap: Optional[int] = None
    """Number of multiplier-bootstrap replicates used to compute the sup-t
    critical value. ``None`` when ``cband=False`` or on unweighted,
    unclustered fits."""

    def __repr__(self) -> str:
        base = (
            f"HeterogeneousAdoptionDiDEventStudyResults("
            f"n_horizons={len(self.event_times)}, "
            f"design={self.design!r}, n_units={self.n_units}"
        )
        if self.variance_formula is not None:
            base += f", variance_formula={self.variance_formula!r}"
        if self.cband_crit_value is not None:
            base += f", cband_crit={self.cband_crit_value:.3f}"
        return base + ")"

    def summary(self) -> str:
        """Formatted per-horizon summary table."""
        width = 80
        conf_level = int((1 - self.alpha) * 100)
        lines = [
            "=" * width,
            "HeterogeneousAdoptionDiD Event-Study Results".center(width),
            "=" * width,
            "",
            f"{'Design:':<30} {self.design:>20}",
            f"{'Target parameter:':<30} {self.target_parameter:>20}",
            f"{'d_lower:':<30} {self.d_lower:>20.6g}",
            f"{'D_bar (dose mean):':<30} {self.dose_mean:>20.6g}",
            f"{'First-treatment period (F):':<30} {str(self.F):>20}",
            f"{'Observations (units):':<30} {self.n_units:>20}",
            f"{'Horizons:':<30} {len(self.event_times):>20}",
            f"{'Inference method:':<30} {self.inference_method:>20}",
        ]
        if self.vcov_type is not None:
            if self.cluster_name is not None:
                label = f"CR1 at {self.cluster_name}"
            else:
                label = self.vcov_type
            lines.append(f"{'Variance:':<30} {label:>20}")
        if self.filter_info is not None:
            lines.append(
                f"{'Last-cohort filter (F_last):':<30} "
                f"{str(self.filter_info.get('F_last')):>20}"
            )
            lines.append(
                f"{'  Units kept / dropped:':<30} "
                f"{self.filter_info.get('n_kept', 0)} / "
                f"{self.filter_info.get('n_dropped', 0):<8}".rjust(51)
            )
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            vf_label = self.variance_formula or "unknown"
            lines.append(f"{'Variance formula:':<30} {vf_label:>20}")
            lines.append(f"{'Effective sample size:':<30} {sm.effective_n:>20.6g}")
            if self.effective_dose_mean is not None:
                lines.append(
                    f"{'Weighted D̄ (denominator):':<30} " f"{self.effective_dose_mean:>20.6g}"
                )
            if sm.df_survey is not None:
                lines.append(f"{'Survey df:':<30} {sm.df_survey:>20}")
        if self.cband_crit_value is not None:
            lines.append(f"{'Sup-t crit (bootstrap):':<30} " f"{self.cband_crit_value:>20.4f}")
            lines.append(f"{'Bootstrap replicates:':<30} " f"{self.cband_n_bootstrap or 0:>20}")
        lines.extend(
            [
                "",
                "-" * width,
                (
                    f"{'Event-time':>10} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} "
                    f"{str(conf_level) + '% CI':>22}"
                ),
                "-" * width,
            ]
        )
        for i, e in enumerate(self.event_times):
            ci_str = f"[{self.conf_int_low[i]:.4f}, {self.conf_int_high[i]:.4f}]"
            # Default float formatting renders non-finite values as "nan";
            # we do not override this here since the column width is fixed
            # and lowercase "nan" is unambiguous.
            se_i = self.se[i]
            t_i = self.t_stat[i]
            p_i = self.p_value[i]
            lines.append(
                f"{int(e):>10} {self.att[i]:>12.4f} "
                f"{se_i:>12.4f} {t_i:>10.3f} {p_i:>10.4f} {ci_str:>22}"
            )
        lines.extend(
            [
                "-" * width,
                "",
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a dict with per-horizon arrays and scalars.

        Per-horizon arrays are converted to Python lists via
        ``ndarray.tolist()`` (which unwraps NumPy scalar elements to
        native ``int`` / ``float``); scalar fields are coerced to
        native Python types via ``_json_safe_scalar`` where relevant
        (NumPy scalars -> ``.item()``, pandas ``Timestamp`` -> ISO
        string, ``Timedelta`` -> ISO string). The returned dict is
        JSON-serializable directly via ``json.dumps``.
        """
        return {
            "event_times": self.event_times.tolist(),
            "att": self.att.tolist(),
            "se": self.se.tolist(),
            "t_stat": self.t_stat.tolist(),
            "p_value": self.p_value.tolist(),
            "conf_int_low": self.conf_int_low.tolist(),
            "conf_int_high": self.conf_int_high.tolist(),
            "n_obs_per_horizon": self.n_obs_per_horizon.tolist(),
            "alpha": float(self.alpha),
            "design": self.design,
            "target_parameter": self.target_parameter,
            "d_lower": float(self.d_lower),
            "dose_mean": float(self.dose_mean),
            "F": _json_safe_scalar(self.F),
            "n_units": int(self.n_units),
            "inference_method": self.inference_method,
            "vcov_type": self.vcov_type,
            "cluster_name": self.cluster_name,
            "filter_info": _json_safe_filter_info(self.filter_info),
            # Phase 4.5 B weighted/survey-path surfaces. survey_metadata /
            # variance_formula / effective_dose_mean are None on unweighted
            # fits; the cband_* fields are also populated on unweighted
            # CLUSTERED fits (Phase 2b clustered band). The full
            # SurveyMetadata dataclass is carried as an object, matching the
            # static-path ``to_dict`` contract — consumers read attributes
            # uniformly.
            "survey_metadata": self.survey_metadata,
            "variance_formula": self.variance_formula,
            "effective_dose_mean": self.effective_dose_mean,
            "cband_low": (self.cband_low.tolist() if self.cband_low is not None else None),
            "cband_high": (self.cband_high.tolist() if self.cband_high is not None else None),
            "cband_crit_value": self.cband_crit_value,
            "cband_method": self.cband_method,
            "cband_n_bootstrap": self.cband_n_bootstrap,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy per-horizon DataFrame.

        Columns: ``event_time, att, se, t_stat, p_value, conf_int_low,
        conf_int_high, n_obs``. One row per event-time horizon. When a
        simultaneous band was computed (``cband=True`` with a
        weighted/survey OR clustered fit), also includes ``cband_low`` and
        ``cband_high`` columns.
        """
        data: Dict[str, Any] = {
            "event_time": self.event_times,
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_low": self.conf_int_low,
            "conf_int_high": self.conf_int_high,
            "n_obs": self.n_obs_per_horizon,
        }
        if self.cband_low is not None:
            data["cband_low"] = self.cband_low
        if self.cband_high is not None:
            data["cband_high"] = self.cband_high
        return pd.DataFrame(data)


# =============================================================================
# Panel validation and aggregation
# =============================================================================


def _validate_had_panel(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> Tuple[int, int]:
    """Validate a HAD panel and return ``(t_pre, t_post)``.

    Enforces the Phase 2a panel contract:

    - All required columns present.
    - Exactly two distinct time periods. Staggered timing (``>2`` periods)
      with ``first_treat_col=None`` raises; with ``first_treat_col`` it
      also raises (multi-period reduction is Phase 2b).
    - Balanced panel (all units observed at both periods).
    - ``D_{g, t_pre} == 0`` for all units (HAD no-unit-untreated pre-period).
    - No NaN in outcome, dose, or unit columns.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    first_treat_col : str or None
        Optional column for cross-validation. Supplied column must contain
        ``0`` for never-treated and the post-period value for treated units.

    Returns
    -------
    tuple[Any, Any]
        ``(t_pre, t_post)`` - the two period identifiers identified by
        the HAD dose invariant (``t_pre`` is the period with dose == 0
        for all units; ``t_post`` is the other period). Supports
        arbitrary-dtype period labels (int, str, datetime, etc.) rather
        than relying on ordinal / lexicographic sort.

    Raises
    ------
    ValueError
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    if first_treat_col is not None:
        required.append(first_treat_col)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")

    periods_list = list(data[time_col].unique())
    if len(periods_list) < 2:
        raise ValueError(
            f"HAD requires a two-period panel; got {len(periods_list)} distinct "
            f"period(s) in column {time_col!r}."
        )
    if len(periods_list) > 2:
        raise ValueError(
            f"HAD with aggregate='overall' requires exactly two time "
            f"periods (got {len(periods_list)} in {time_col!r}). For "
            f"multi-period panels, pass aggregate='event_study' (paper "
            f"Appendix B.2 multi-period event-study extension) which "
            f"produces per-event-time WAS estimates."
        )

    # Balanced-panel check: every unit appears exactly once per period.
    counts = data.groupby([unit_col, time_col]).size()
    if (counts != 1).any():
        n_bad = int((counts != 1).sum())
        raise ValueError(
            f"Unbalanced panel: {n_bad} unit-period cells have != 1 "
            f"observation. HAD requires a balanced two-period panel "
            f"(each unit observed exactly once at each period)."
        )
    unit_counts = data.groupby(unit_col)[time_col].nunique()
    incomplete = unit_counts[unit_counts != 2]
    if len(incomplete) > 0:
        raise ValueError(
            f"Unbalanced panel: {len(incomplete)} unit(s) do not appear "
            f"in both periods. HAD requires a balanced two-period panel."
        )

    # NaN checks on key columns.
    for col in [outcome_col, dose_col, unit_col]:
        if bool(data[col].isna().any()):
            n_nan = int(data[col].isna().sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r}. HAD "
                f"does not silently drop NaN rows; drop or impute before "
                f"calling fit()."
            )

    # Identify t_pre and t_post by the HAD invariant rather than by
    # lexicographic sort on the time labels: D_{g, t_pre} = 0 for all
    # units (paper Section 2 no-unit-untreated pre-period convention).
    # Sorting labels alphabetically reverses valid chronologies like
    # ("pre", "post") where ordering is semantic, not alphabetic.
    per_period_nonzero: Dict[Any, int] = {}
    for p in periods_list:
        p_doses = np.asarray(data.loc[data[time_col] == p, dose_col], dtype=np.float64)
        per_period_nonzero[p] = int((p_doses != 0).sum())
    all_zero_periods = [p for p, nz in per_period_nonzero.items() if nz == 0]
    if len(all_zero_periods) == 0:
        # Neither period has all-zero dose: HAD pre-period contract violated.
        stats_str = ", ".join(f"{p!r}: {nz} nonzero" for p, nz in per_period_nonzero.items())
        raise ValueError(
            f"HAD requires D_{{g,1}} = 0 for all units (pre-period "
            f"untreated). Neither period in column {time_col!r} has "
            f"all-zero dose ({stats_str}). Exactly one period must be "
            f"the pre-treatment period with D_{{g,1}} = 0 for every unit; "
            f"drop rows with nonzero pre-period dose or verify the dose "
            f"column."
        )
    if len(all_zero_periods) == 2:
        raise ValueError(
            f"HAD requires variation in D_{{g,2}} for estimation. Both "
            f"periods in column {time_col!r} have all-zero dose, so "
            f"there is no treatment assignment to estimate."
        )
    t_pre = all_zero_periods[0]
    t_post = [p for p in periods_list if p != t_pre][0]

    # Post-period nonnegative-dose check on the ORIGINAL (unshifted) dose
    # scale. Front-door rejection per paper Assumption (dose definition
    # Section 2) which treats D_{g,2} as nonnegative. Without this
    # check, negative original doses would only surface after the
    # regressor shift in ``_fit_continuous`` via Phase 1c's
    # ``_validate_had_inputs``, which references the shifted values
    # and would confuse users about which column is malformed.
    post_mask = data[time_col] == t_post
    post_doses = np.asarray(data.loc[post_mask, dose_col], dtype=np.float64)
    neg_post = post_doses < 0
    if neg_post.any():
        n_neg = int(neg_post.sum())
        min_neg = float(post_doses[neg_post].min())
        raise ValueError(
            f"HAD requires D_{{g,2}} >= 0 for all units (paper Section "
            f"2 dose definition). {n_neg} unit(s) have negative post-"
            f"period dose at t_post={t_post} (min={min_neg!r}). Drop "
            f"these units or verify the dose column."
        )

    # Optional value-domain validation via first_treat_col: if supplied,
    # every unit's first_treat value must be in {0, t_post} (0 = never
    # treated, t_post = treated in the second period). This is a value-
    # domain check that catches typos and staggered-timing mix-ups; it
    # does NOT cross-validate first_treat against post-period dose
    # (D_{g, t_post} remains the primary signal). Extended cross-checks
    # are queued for a follow-up PR. The check is DTYPE-AGNOSTIC: it uses
    # pd.isna() for missingness and raw-value membership against
    # {0, t_post} so that string-labelled periods (e.g., ("A", "B")) with
    # first_treat in {0, "B"} are supported.
    if first_treat_col is not None:
        # Row-level NaN check: `groupby().first()` skips NaNs silently, so a
        # unit with rows [valid, NaN] would pass a collapsed check. Validate
        # raw per-row values first, then verify per-unit constancy with
        # `nunique(dropna=False)` so within-unit NaN variation is caught.
        ft_raw = data[first_treat_col]
        if bool(ft_raw.isna().any()):
            n_nan = int(ft_raw.isna().sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains "
                f"{n_nan} NaN value(s) at the row level. Use 0 for "
                f"never-treated units and t_post for treated, and drop "
                f"or impute any NaN rows before calling fit()."
            )
        # Row-level domain check: every row (not just the collapsed first())
        # must be in {0, t_post}. Catches mixed-row malformed inputs where
        # a unit has [valid, invalid].
        valid_values = {0, t_post}
        observed_raw = set(ft_raw.unique().tolist())
        bad = sorted(observed_raw - valid_values, key=lambda x: str(x))
        if bad:
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains value(s) "
                f"{bad} outside the allowed set {{0, {t_post!r}}} for a "
                f"two-period HAD panel. Staggered timing with multiple "
                f"cohorts is Phase 2b."
            )
        # Within-unit consistency: every unit must have a single
        # first_treat value across its rows. Uses dropna=False so a unit
        # with [value, NaN] counts as 2 unique values (caught above by
        # the NaN check anyway, but this is belt-and-suspenders).
        ft_per_unit_nunique = data.groupby(unit_col)[first_treat_col].nunique(dropna=False)
        if (ft_per_unit_nunique > 1).any():
            n_bad = int((ft_per_unit_nunique > 1).sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} is not constant "
                f"within unit for {n_bad} unit(s). Each unit must have "
                f"a single first_treat value across both observed periods."
            )

    return t_pre, t_post


def _validate_had_panel_event_study(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> Tuple[Any, List[Any], List[Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """Validate a HAD panel for multi-period event-study mode (Phase 2b).

    Implements paper Appendix B.2 contract: a common treatment date ``F``
    where ``D_{g,t} = 0`` for all units at ``t < F`` and some units have
    ``D_{g,t} > 0`` for ``t >= F``. Requires ``len(periods) > 2`` with at
    least one pre-period (``t < F``, all D=0) and at least one post-period
    (``t >= F``, some D > 0).

    Staggered-timing handling: when ``first_treat_col`` is supplied and
    indicates more than one nonzero cohort, the panel is auto-filtered to
    the LAST cohort (``F_last = max(cohorts)``) per paper Appendix B.2
    prescription "did_had may be used only for the last treatment cohort
    in a staggered design". A ``UserWarning`` is emitted with drop-counts.

    Parameters
    ----------
    data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        As in :func:`_validate_had_panel`.

    Returns
    -------
    F : period label
        First-treatment period (the earliest period where any unit has
        ``D > 0`` in the filtered data).
    t_pre_list : list
        Pre-period labels (``t < F``, all D=0), sorted by natural ordering
        on the column dtype.
    t_post_list : list
        Post-period labels (``t >= F``, some D > 0), sorted.
    data_filtered : pd.DataFrame
        Input with earlier cohorts (``first_treat`` in ``dropped_cohorts``)
        dropped if staggered; never-treated units (``first_treat = 0``)
        are RETAINED per paper Appendix B.2's "there must be an untreated
        group" requirement. Identical to input when no staggered filter
        applies.
    filter_info : dict or None
        Populated on staggered filter with keys ``F_last`` (kept cohort
        label), ``n_kept`` (last-cohort units PLUS never-treated units),
        ``n_dropped`` (earlier-cohort units removed), ``dropped_cohorts``
        (list of earlier cohort labels). ``None`` otherwise.

    Raises
    ------
    ValueError
        On missing columns, NaN in key columns, malformed panel, dose-
        invariant violations, or no-treatment detected.
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    if first_treat_col is not None:
        required.append(first_treat_col)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")

    periods_list = list(data[time_col].unique())
    if len(periods_list) < 3:
        raise ValueError(
            f"HAD with aggregate='event_study' requires more than two "
            f"time periods (got {len(periods_list)} in {time_col!r}). "
            f"For two-period panels, pass aggregate='overall' (Phase 2a "
            f"single-period WAS)."
        )

    # Ordered-time-type check. Paper Appendix B.2 event-time horizons
    # require chronological ordering of periods (anchor at F-1, horizons
    # e = t - F relative to F). Phase 2a two-period panels can use the
    # dose invariant alone to distinguish pre from post without needing
    # chronological order, so string labels ("pre", "post") work there.
    # For multi-period event-study, multiple pre-periods all have D=0
    # and multiple post-periods may both have D>0, so dose alone cannot
    # recover chronology: we must trust the time column's natural order.
    # Raw lexicographic sort on object/string labels silently misorders
    # panels like "pre1"/"pre2"/"post1"/"post2" or month-name labels.
    # Require an explicitly-ordered time representation.
    time_dtype = data[time_col].dtype
    if not (
        pd.api.types.is_numeric_dtype(time_dtype)
        or pd.api.types.is_datetime64_any_dtype(time_dtype)
        or (isinstance(time_dtype, pd.CategoricalDtype) and bool(time_dtype.ordered))
    ):
        raise ValueError(
            f"HAD aggregate='event_study' requires an ordered time "
            f"column. time_col={time_col!r} has dtype={time_dtype!r}, "
            f"which has no defined chronological order; raw sort would "
            f"fall back to lexicographic ordering and silently misindex "
            f"event-time horizons (e.g., 'pre1'/'pre2'/'post1'/'post2' "
            f"sorts lexicographically but not chronologically). "
            f"Convert time_col to numeric (e.g., integer year), "
            f"datetime, or ordered categorical "
            f"(``pd.Categorical(..., ordered=True, categories=[...])``) "
            f"before calling fit() with aggregate='event_study'."
        )

    # Construct the chronological sort key once, shared across every
    # downstream ordering: cohort ranking, pre/post period sorting, and
    # contiguity checks. Ordered categoricals use their declared
    # category index (``list(categorical)`` strips the ordering and
    # falls back to string comparison); numeric / datetime use natural
    # Python order. Reused by ``_aggregate_multi_period_first_differences``
    # via a parallel construction in that helper (both read the same
    # ``time_dtype``).
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        _cat_order = {c: i for i, c in enumerate(time_dtype.categories)}

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, _cat_order.get(x, len(_cat_order)))

    else:

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, x)

    # NaN checks on key columns (before any filter).
    for col in [outcome_col, dose_col, unit_col]:
        if bool(data[col].isna().any()):
            n_nan = int(data[col].isna().sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r}. HAD "
                f"does not silently drop NaN rows; drop or impute before "
                f"calling fit()."
            )

    # Cohort detection and staggered-timing auto-filter.
    filter_info: Optional[Dict[str, Any]] = None
    data_filtered = data
    if first_treat_col is not None:
        ft_raw = data[first_treat_col]
        if bool(ft_raw.isna().any()):
            n_nan = int(ft_raw.isna().sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains "
                f"{n_nan} NaN value(s) at the row level. Use 0 for "
                f"never-treated units and the treatment-start period "
                f"for treated units. Drop or impute any NaN rows "
                f"before calling fit()."
            )
        # Within-unit constancy check.
        ft_per_unit_nunique = data.groupby(unit_col)[first_treat_col].nunique(dropna=False)
        if (ft_per_unit_nunique > 1).any():
            n_bad = int((ft_per_unit_nunique > 1).sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} is not constant "
                f"within unit for {n_bad} unit(s). Each unit must have "
                f"a single first_treat value across all observed periods."
            )
        # Cross-validate first_treat_col against observed first-positive-
        # dose period for every unit. A mislabeled cohort column would
        # otherwise silently select the wrong cohort as F_last and return
        # event-study estimates for the wrong units. Contract:
        #   - declared first_treat == 0: unit must have D == 0 at all t
        #     (never-treated)
        #   - declared first_treat == F_g > 0: unit's first period with
        #     D > 0 must equal F_g
        df_for_check = data.sort_values([unit_col, time_col])
        pos_rows = df_for_check.loc[df_for_check[dose_col] > 0]
        actual_first_pos = pos_rows.groupby(unit_col)[time_col].first()
        declared_ft = df_for_check.groupby(unit_col)[first_treat_col].first()
        n_mismatch = 0
        example_mismatch: Optional[Tuple[Any, Any, Any]] = None
        for u, declared in declared_ft.items():
            actual = actual_first_pos.get(u, None)
            if declared == 0:
                if actual is not None:
                    n_mismatch += 1
                    if example_mismatch is None:
                        example_mismatch = (u, declared, actual)
            else:
                if actual is None or actual != declared:
                    n_mismatch += 1
                    if example_mismatch is None:
                        example_mismatch = (u, declared, actual)
        if n_mismatch > 0:
            u, declared, actual = example_mismatch  # type: ignore[misc]
            raise ValueError(
                f"first_treat_col={first_treat_col!r} disagrees with the "
                f"observed dose path for {n_mismatch} unit(s). Example: "
                f"unit={u!r} declares first_treat={declared!r} but the "
                f"unit's first period with D>0 is {actual!r} "
                f"(None means never-treated). A mislabeled cohort column "
                f"would silently select the wrong cohort as F_last in the "
                f"last-cohort auto-filter. Fix the first_treat_col values "
                f"to equal each unit's first positive-dose period (or 0 "
                f"for never-treated) before calling fit()."
            )
        # Identify cohorts (nonzero first_treat values). Sort using
        # ``_sort_key`` (chronological order from ``time_dtype``), NOT
        # raw Python sort: first_treat values are period labels and
        # must rank chronologically so ``F_last = cohorts[-1]`` is the
        # chronologically latest cohort. Under ordered-categorical time
        # labels (e.g. month names), raw Python sort is lexicographic
        # and would silently pick the wrong ``F_last``.
        ft_unique = list(pd.unique(ft_raw))
        cohorts = sorted(
            [v for v in ft_unique if v != 0 and not (isinstance(v, float) and np.isnan(v))],
            key=_sort_key,
        )
        if len(cohorts) == 0:
            raise ValueError(
                f"first_treat_col={first_treat_col!r} has no nonzero "
                f"cohort values (all units appear never-treated). HAD "
                f"requires at least one treated cohort with "
                f"first_treat > 0 to identify a WAS effect."
            )
        if len(cohorts) > 1:
            F_last = cohorts[-1]
            dropped_cohorts = cohorts[:-1]
            # Filter: keep last-cohort AND never-treated (first_treat == 0).
            # Paper Appendix B.2: "in designs with variation in treatment
            # timing, there must be an untreated group, at least till the
            # period where the last cohort gets treated". Never-treated
            # units (first_treat=0) satisfy the dose invariant for every
            # period (D=0 throughout) and serve as the untreated-group
            # comparison at every pre-period horizon. Keeping them matches
            # the paper's "there must be an untreated group" language and
            # preserves Design 1' identifiability (boundary at 0) when the
            # last-cohort doses are uniformly positive. Only earlier-treated
            # cohorts (first_treat in dropped_cohorts) are dropped.
            keep_mask = (data[first_treat_col] == F_last) | (data[first_treat_col] == 0)
            dropped_unit_ids = set(data.loc[~keep_mask, unit_col].unique())
            kept_unit_ids = set(data.loc[keep_mask, unit_col].unique())
            data_filtered = data.loc[keep_mask].copy()
            n_dropped = len(dropped_unit_ids - kept_unit_ids)
            n_kept = len(kept_unit_ids)
            if n_kept == 0:
                raise ValueError(
                    f"Staggered auto-filter to last cohort "
                    f"(F_last={F_last!r}) left 0 units. Verify "
                    f"first_treat_col={first_treat_col!r} contains the "
                    f"expected cohort labels."
                )
            filter_info = {
                "F_last": F_last,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
                "dropped_cohorts": dropped_cohorts,
            }
            warnings.warn(
                f"Staggered-timing panel detected: {len(cohorts)} distinct "
                f"nonzero cohorts in first_treat_col={first_treat_col!r} "
                f"({cohorts!r}). Auto-filtering to the last cohort "
                f"(F_last={F_last!r}) plus never-treated units "
                f"(first_treat=0): {n_kept} units kept, {n_dropped} "
                f"earlier-cohort units dropped (from cohorts "
                f"{dropped_cohorts!r}). HAD applies only to the last "
                f"treatment cohort in staggered designs (paper Appendix "
                f"B.2); never-treated units are retained as the untreated-"
                f"group comparison per the paper's \"there must be an "
                f'untreated group" requirement. For earlier-cohort '
                f"effects, use ChaisemartinDHaultfoeuille "
                f"(did_multiplegt_dyn).",
                UserWarning,
                stacklevel=3,
            )
            # After filter, re-read periods_list (cohort filter may have
            # dropped some periods if earlier cohorts contributed uniquely).
            periods_list = list(data_filtered[time_col].unique())
            if len(periods_list) < 3:
                raise ValueError(
                    f"After staggered auto-filter to last cohort "
                    f"(F_last={F_last!r}), only {len(periods_list)} "
                    f"distinct time periods remain in {time_col!r}. "
                    f"Event-study requires >2 periods; the filtered "
                    f"panel is too small. Pass aggregate='overall' on "
                    f"a two-period subset, or supply data with more "
                    f"pre- or post-periods for the last cohort."
                )

    # Balanced panel on the (possibly-filtered) data: every unit appears
    # exactly once per period. ``observed=True`` tells categorical
    # groupby to count only OBSERVED unit-period cells. Without it, a
    # time_col with an ordered-categorical dtype carrying extra unused
    # category levels (beyond the periods actually present in the data)
    # would expand to zero-count cells and the balance check would
    # falsely reject valid panels. The rest of the validator is keyed
    # to ``periods_list`` (observed unique values) so this stays
    # consistent.
    counts = data_filtered.groupby([unit_col, time_col], observed=True).size()
    if (counts != 1).any():
        n_bad = int((counts != 1).sum())
        raise ValueError(
            f"Unbalanced panel: {n_bad} unit-period cells have != 1 "
            f"observation. HAD requires a balanced panel (each unit "
            f"observed exactly once at each period)."
        )
    unit_counts = data_filtered.groupby(unit_col)[time_col].nunique()
    incomplete = unit_counts[unit_counts != len(periods_list)]
    if len(incomplete) > 0:
        raise ValueError(
            f"Unbalanced panel: {len(incomplete)} unit(s) do not appear "
            f"in all {len(periods_list)} periods. HAD requires a balanced "
            f"panel (each unit observed at every period)."
        )

    # Dose-invariant period classification on filtered data.
    per_period_nonzero: Dict[Any, int] = {}
    for p in periods_list:
        p_doses = np.asarray(
            data_filtered.loc[data_filtered[time_col] == p, dose_col], dtype=np.float64
        )
        per_period_nonzero[p] = int((p_doses != 0).sum())
    t_pre_list_unsorted = [p for p, nz in per_period_nonzero.items() if nz == 0]
    t_post_list_unsorted = [p for p, nz in per_period_nonzero.items() if nz > 0]

    if len(t_pre_list_unsorted) == 0:
        stats_str = ", ".join(f"{p!r}: {nz} nonzero" for p, nz in per_period_nonzero.items())
        raise ValueError(
            f"HAD requires D_{{g,t}} = 0 for all units in at least one "
            f"pre-period. No period in column {time_col!r} has all-zero "
            f"dose ({stats_str}). The panel has no identifiable baseline."
        )
    if len(t_post_list_unsorted) == 0:
        raise ValueError(
            f"HAD requires at least one period with nonzero dose for "
            f"some unit. All periods in column {time_col!r} have all-"
            f"zero dose; there is no treatment to estimate."
        )

    # Sort using the same ``_sort_key`` already constructed for cohorts
    # (ordered-categorical uses declared category order; numeric /
    # datetime use natural Python order).
    t_pre_list = sorted(t_pre_list_unsorted, key=_sort_key)
    t_post_list = sorted(t_post_list_unsorted, key=_sort_key)

    # Contiguity check: all pre < all post in the natural ordering.
    # The HAD dose invariant requires a single transition from all-zero
    # to any-nonzero; interleaved pre/post periods indicate a malformed
    # panel (e.g., dose going back to zero after treatment, or mixing
    # never-treated units with out-of-order labels). Uses ``_sort_key``
    # so ordered categoricals respect their declared category order.
    if t_pre_list and t_post_list:
        max_pre = t_pre_list[-1]
        min_post = t_post_list[0]
        contiguous = _sort_key(max_pre) < _sort_key(min_post)
        if not contiguous:
            raise ValueError(
                f"HAD dose invariant violated: pre-periods (all D=0) "
                f"and post-periods (some D>0) are not contiguous. "
                f"Pre-periods: {t_pre_list!r}; post-periods: "
                f"{t_post_list!r}. The dose sequence must transition "
                f"from all-zero to nonzero exactly once. For panels "
                f"where dose varies non-monotonically (e.g., reversed "
                f"treatment, switching), use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn)."
            )

    F = t_post_list[0]  # earliest post-period

    # Post-period nonnegative-dose check on the filtered data.
    post_mask = data_filtered[time_col].isin(t_post_list)
    post_doses = np.asarray(data_filtered.loc[post_mask, dose_col], dtype=np.float64)
    neg_post = post_doses < 0
    if neg_post.any():
        n_neg = int(neg_post.sum())
        min_neg = float(post_doses[neg_post].min())
        raise ValueError(
            f"HAD requires D_{{g,t}} >= 0 for all units in post-periods "
            f"(paper Section 2 dose definition). {n_neg} unit-period "
            f"cell(s) have negative dose at t >= F={F!r} "
            f"(min={min_neg!r}). Drop these units or verify the dose "
            f"column."
        )

    # Staggered-without-``first_treat_col`` detection. When cohort metadata
    # is not supplied, the dose-invariant period classification still
    # declares t=F=min-post-period based on "any unit has nonzero dose".
    # That silently accepts staggered panels where units have DIFFERENT
    # first-positive-dose periods: the later-treated cohorts enter
    # ``d_arr`` as zero-dose "controls" at the inferred F, violating
    # paper Appendix B.2's last-cohort-only contract. Compute per-unit
    # first-positive-dose period directly from the dose path and raise
    # if multiple cohorts are present, directing users to pass
    # ``first_treat_col`` (which activates the last-cohort auto-filter)
    # or to use ChaisemartinDHaultfoeuille for full staggered support.
    if first_treat_col is None:
        df_sorted = data_filtered.sort_values([unit_col, time_col])
        # For each unit, the first period at which dose > 0.
        pos_mask_global = df_sorted[dose_col] > 0
        first_pos_per_unit = df_sorted.loc[pos_mask_global].groupby(unit_col)[time_col].first()
        cohort_labels = list(first_pos_per_unit.unique())
        if len(cohort_labels) > 1:
            # Sort chronologically via the validated time-column order.
            distinct_cohorts = sorted(cohort_labels, key=_sort_key)
            raise ValueError(
                f"Staggered-timing panel detected (first_treat_col is "
                f"None): {len(distinct_cohorts)} distinct first-positive-"
                f"dose periods {distinct_cohorts!r} across units. HAD's "
                f"last-cohort auto-filter (paper Appendix B.2) only runs "
                f"when first_treat_col is supplied so the estimator can "
                f"identify cohorts. Pass first_treat_col=<column> to "
                f"enable the auto-filter to the last cohort, or use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn) for "
                f"full staggered support."
            )

    # Constant post-period dose check. Paper Appendix B.2 assumes
    # "once treated, stay treated with the same dose"; the event-study
    # aggregation uses ``D_{g, F}`` as the single regressor for every
    # event-time horizon. Panels where a unit's dose varies across
    # post-periods (e.g., phased adoption, dose changes after F) would
    # silently misattribute later-horizon effects to the period-F dose.
    # Reject front-door with a redirect to ChaisemartinDHaultfoeuille
    # for genuinely time-varying post-treatment doses.
    if len(t_post_list) > 1:
        post_data = data_filtered.loc[post_mask]
        dose_spread_per_unit = post_data.groupby(unit_col)[dose_col].agg(
            lambda x: float(x.max() - x.min())
        )
        abs_max_dose = float(np.max(np.abs(post_doses))) if post_doses.size else 0.0
        tol = 1e-12 * max(1.0, abs_max_dose)
        bad_mask = dose_spread_per_unit > tol
        if bool(bad_mask.any()):
            n_bad = int(bad_mask.sum())
            max_spread = float(dose_spread_per_unit.max())
            raise ValueError(
                f"HAD event-study requires constant dose within unit for "
                f"all post-treatment periods t >= F={F!r}. {n_bad} unit(s) "
                f"have time-varying doses across post-periods "
                f"{t_post_list!r} (max within-unit spread={max_spread!r}, "
                f"tolerance={tol!r}). The aggregation uses D_{{g, F}} as "
                f"the single regressor for every event-time horizon "
                f"(paper Appendix B.2 constant-dose convention), so "
                f"silently accepting time-varying post-treatment doses "
                f"would misattribute later-horizon effects. For genuinely "
                f"time-varying post-treatment doses use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn)."
            )

    return F, t_pre_list, t_post_list, data_filtered, filter_info


def _aggregate_first_difference(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    t_pre: Any,
    t_post: Any,
    cluster_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Reduce a balanced two-period panel to unit-level first differences.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
        Post-period dose ``D_{g,2}`` per unit, ordered by sorted unit id.
    dy_arr : np.ndarray, shape (G,)
        First-difference outcome ``Y_{g,2} - Y_{g,1}`` per unit.
    cluster_arr : np.ndarray or None, shape (G,)
        Cluster IDs per unit (must be unit-constant); ``None`` when
        ``cluster_col`` is ``None``.
    unit_ids : np.ndarray, shape (G,)
        Sorted unit identifiers (useful for debugging / test inspection).
    """
    df = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    pre = df[df[time_col] == t_pre].set_index(unit_col).sort_index()
    post = df[df[time_col] == t_post].set_index(unit_col).sort_index()

    if bool(pre.index.equals(post.index)) is False:
        # Should not reach here after _validate_had_panel balanced-panel
        # check, but belt-and-suspenders:
        raise ValueError(
            "Internal error: pre and post period unit indices do not "
            "match after balanced-panel validation. Please report this."
        )

    unit_ids = np.asarray(pre.index)
    d_arr = post[dose_col].to_numpy(dtype=np.float64)
    dy_arr = post[outcome_col].to_numpy(dtype=np.float64) - pre[outcome_col].to_numpy(
        dtype=np.float64
    )

    cluster_arr: Optional[np.ndarray] = None
    if cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(f"cluster column {cluster_col!r} not found in data.")
        # Row-level NaN check: `groupby().first()` skips NaNs silently, so a
        # unit with rows [valid, NaN] would pass a collapsed check. Validate
        # raw per-row values first so any NaN in the cluster column is
        # rejected up-front, not masked by the per-unit collapse.
        cluster_raw = data[cluster_col]
        if bool(cluster_raw.isna().any()):
            n_nan = int(cluster_raw.isna().sum())
            raise ValueError(
                f"cluster column {cluster_col!r} contains {n_nan} NaN "
                f"value(s) at the row level. Silent row dropping is "
                f"disabled; drop or impute cluster ids before calling fit()."
            )
        # Cluster must be unit-constant. `nunique(dropna=False)` counts NaN
        # as a distinct value so rows like [value, NaN] register as 2
        # uniques (also caught by the row-level NaN check above).
        cluster_per_unit = df.groupby(unit_col)[cluster_col].nunique(dropna=False)
        if (cluster_per_unit > 1).any():
            n_bad = int((cluster_per_unit > 1).sum())
            raise ValueError(
                f"cluster column {cluster_col!r} is not constant within "
                f"unit for {n_bad} unit(s). Cluster must be unit-level "
                f"(e.g., cluster_col=unit_col, or a coarser grouping "
                f"where each unit belongs to a single cluster)."
            )
        cluster_arr = df.groupby(unit_col)[cluster_col].first().sort_index().to_numpy()

    return d_arr, dy_arr, cluster_arr, unit_ids


def _aggregate_unit_weights(
    data: pd.DataFrame,
    weights_arr: np.ndarray,
    unit_col: str,
) -> np.ndarray:
    """Aggregate per-row weights to per-unit, enforcing constant-within-unit.

    HAD's continuous path operates at the per-unit level (G rows, one per
    unit) so survey weights — which typically arrive per-row in a long
    panel — must collapse to a single value per unit. The paper's weighted
    extension assumes sampling weights assigned at the sampling-unit level
    (standard BRFSS/CPS/NHANES convention), so any within-unit weight
    variance is interpreted as user error and rejected front-door rather
    than silently mean-rolled (per ``feedback_no_silent_failures``).

    Parameters
    ----------
    data : pd.DataFrame
        Long panel; the same frame passed to ``_aggregate_first_difference``.
    weights_arr : np.ndarray, shape (n_rows,)
        Row-aligned weights.
    unit_col : str
        Unit identifier column.

    Returns
    -------
    w_unit : np.ndarray, shape (G,)
        Per-unit weights sorted by unit id to align with the d/dy arrays
        returned by ``_aggregate_first_difference``.

    Raises
    ------
    ValueError
        Shape mismatch, non-finite weights, negative weights, zero-sum
        weights, or weights that vary within a unit.
    """
    n_rows = int(data.shape[0])
    w = np.asarray(weights_arr, dtype=np.float64).ravel()
    if w.shape[0] != n_rows:
        raise ValueError(
            f"weights length ({w.shape[0]}) does not match number of " f"rows in data ({n_rows})."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contains non-finite values (NaN or Inf).")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if np.sum(w) <= 0:
        raise ValueError("weights sum to zero — no observations have positive weight.")

    df = data.reset_index(drop=True).copy()
    df["_w_tmp__"] = w
    w_per_unit = df.groupby(unit_col)["_w_tmp__"].agg(lambda s: (float(s.min()), float(s.max())))
    varying = w_per_unit.apply(lambda t: not np.isclose(t[0], t[1], rtol=1e-12, atol=0.0))
    if bool(varying.any()):
        n_bad = int(varying.sum())
        raise ValueError(
            f"weights vary within {n_bad} unit(s). HAD's continuous path "
            f"requires unit-level sampling weights (constant within each "
            f"unit). Either pre-aggregate your weights to the unit level "
            f"or pass a unit-constant weight column. Per-obs weights that "
            f"vary within unit would require an obs-level estimator not "
            f"available on this path."
        )
    w_unit = df.groupby(unit_col)["_w_tmp__"].first().sort_index().to_numpy(dtype=np.float64)
    return w_unit


def _aggregate_unit_resolved_survey(
    data: pd.DataFrame,
    resolved: Any,  # ResolvedSurveyDesign
    unit_col: str,
) -> Any:  # ResolvedSurveyDesign at unit level
    """Collapse a row-level ResolvedSurveyDesign to a unit-level analogue.

    HAD's continuous path operates at G per-unit rows, so the survey
    design (weights / strata / PSU / FPC) must collapse the same way.
    Each design column is required to be constant-within-unit (same
    invariant as ``_aggregate_unit_weights``); a within-unit inconsistency
    raises ``ValueError`` per ``feedback_no_silent_failures``.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel.
    resolved : ResolvedSurveyDesign
        Resolved design with ``(n_rows,)`` arrays.
    unit_col : str

    Returns
    -------
    ResolvedSurveyDesign
        New resolved design with ``(G,)`` arrays aligned to sorted unit ids.

    Raises
    ------
    ValueError
        Strata / PSU / FPC vary within unit, or replicate-weight designs
        are passed (replicate-weight HAD is Phase 4.5 C scope, not this
        commit).
    NotImplementedError
        ``resolved.replicate_weights is not None`` — replicate-weight HAD
        is deferred.
    """
    from diff_diff.survey import ResolvedSurveyDesign

    if resolved.replicate_weights is not None:
        raise NotImplementedError(
            "Replicate-weight SurveyDesign on HAD is deferred to Phase 4.5 C. "
            "Pass a SurveyDesign with weights/strata/psu/fpc (Taylor-series "
            "linearization path) for this PR."
        )

    n_rows = int(data.shape[0])
    if resolved.weights.shape[0] != n_rows:
        raise ValueError(
            f"ResolvedSurveyDesign.weights length ({resolved.weights.shape[0]}) "
            f"does not match data rows ({n_rows}). The SurveyDesign must "
            f"have been resolved against the same DataFrame passed to fit()."
        )

    def _collapse(arr: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"ResolvedSurveyDesign.{name} length does not match " f"data rows ({n_rows})."
            )
        df = data.reset_index(drop=True).copy()
        # Use a stable per-row column so we can take first() per unit and
        # verify constancy. For numeric designs, use np.isclose; for
        # object/string (rare), require equality.
        df["_tmp__"] = arr
        unit_min = df.groupby(unit_col)["_tmp__"].min()
        unit_max = df.groupby(unit_col)["_tmp__"].max()
        # nunique=1 is the robust check across dtypes.
        nunique = df.groupby(unit_col)["_tmp__"].nunique(dropna=False)
        if (nunique > 1).any():
            n_bad = int((nunique > 1).sum())
            raise ValueError(
                f"ResolvedSurveyDesign.{name} varies within {n_bad} unit(s). "
                f"Survey design columns must be constant within each unit "
                f"(sampling-unit-level assignment convention). If your "
                f"panel uses unit-varying design columns, aggregate them "
                f"to unit-level before calling fit()."
            )
        # The unit-level min and max coincide by the nunique check; take min.
        _ = unit_max  # referenced for symmetry of intent; min suffices
        return unit_min.sort_index().to_numpy(dtype=arr.dtype)

    w_unit = _collapse(resolved.weights, "weights")
    assert w_unit is not None  # resolved.weights is non-Optional
    strata_unit = _collapse(resolved.strata, "strata")
    psu_unit = _collapse(resolved.psu, "psu")
    fpc_unit = _collapse(resolved.fpc, "fpc")

    # Recompute n_strata and n_psu at the unit level.
    n_strata_unit = int(np.unique(strata_unit).shape[0]) if strata_unit is not None else 1
    n_psu_unit = int(np.unique(psu_unit).shape[0]) if psu_unit is not None else int(w_unit.shape[0])

    return ResolvedSurveyDesign(
        weights=w_unit,
        weight_type=resolved.weight_type,
        strata=strata_unit,
        psu=psu_unit,
        fpc=fpc_unit,
        n_strata=n_strata_unit,
        n_psu=n_psu_unit,
        lonely_psu=resolved.lonely_psu,
        replicate_weights=None,
        replicate_method=None,
        fay_rho=0.0,
        n_replicates=0,
        replicate_strata=None,
        combined_weights=resolved.combined_weights,
        replicate_scale=None,
        replicate_rscales=None,
        mse=resolved.mse,
    )


def _aggregate_multi_period_first_differences(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    F: Any,
    t_pre_list: List[Any],
    t_post_list: List[Any],
    cluster_col: Optional[str],
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Optional[np.ndarray], np.ndarray, Any]:
    """Reduce a multi-period HAD panel to per-horizon first differences.

    For each period ``t`` other than the anchor ``t_anchor = F - 1`` (the
    last pre-period), computes the unit-level first difference
    ``ΔY_{g,t} = Y_{g,t} - Y_{g, t_anchor}`` and stores it under the event
    time ``e = rank(t) - rank(F)`` where ``rank`` is the natural ordering
    on the period column (so ``e = 0`` at ``t = F``, ``e = 1`` at the next
    post-period, etc., and ``e <= -2`` for pre-period placebos).

    The single dose regressor is ``D_{g, F}`` (the dose at the first
    treatment period), reused for every horizon. Paper Appendix B.2
    convention assumes "once treated, stay treated with same dose"; this
    helper uses the period-F dose for every horizon, so time-varying
    post-period dose is treated as the realized F-period dose.

    Parameters
    ----------
    data : pd.DataFrame
        Validated multi-period panel (already passed
        ``_validate_had_panel_event_study`` and any staggered filter).
    outcome_col, dose_col, time_col, unit_col, cluster_col : str
        Column names.
    F : period label
        First-treatment period (from the validator).
    t_pre_list, t_post_list : list
        Period labels partitioned by the dose invariant, sorted
        ascending.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
        Post-period dose ``D_{g, F}`` per unit.
    dy_dict : dict[int, np.ndarray]
        Maps event time ``e`` to first-difference outcome
        ``Y_{g, t} - Y_{g, F-1}`` per unit, for every ``t`` except the
        anchor. Keys cover all horizons EXCEPT ``e = -1`` (the anchor
        gives ``ΔY = 0`` trivially).
    cluster_arr : np.ndarray or None, shape (G,)
        Cluster IDs per unit when ``cluster_col`` is supplied.
    unit_ids : np.ndarray, shape (G,)
        Sorted unit identifiers.
    t_anchor : period label
        The anchor period used (``F - 1`` in the natural period ordering;
        equal to the LAST pre-period).
    """
    df = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    # Period sort respects ordered categorical dtypes (matches
    # ``_validate_had_panel_event_study``). The validator already
    # enforces a numeric / datetime / ordered-categorical dtype on the
    # event-study path, so ``_sort_key`` lookups are well-defined here.
    time_dtype = data[time_col].dtype
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        _cat_order = {c: i for i, c in enumerate(time_dtype.categories)}

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, _cat_order.get(x, len(_cat_order)))

    else:

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, x)

    all_periods = sorted(t_pre_list + t_post_list, key=_sort_key)
    # Event-time mapping: natural rank of each period relative to F.
    F_idx = all_periods.index(F)
    period_to_event_time: Dict[Any, int] = {p: (i - F_idx) for i, p in enumerate(all_periods)}
    # Anchor = F-1 in natural rank (i.e., the last pre-period). By the
    # validator's contiguity guarantee, this IS t_pre_list[-1].
    if not t_pre_list:
        raise ValueError(
            "Internal error: event-study aggregation called with no "
            "pre-periods. _validate_had_panel_event_study should have "
            "rejected this upstream."
        )
    t_anchor = t_pre_list[-1]

    # Pivot to wide: units x periods.
    wide_y = df.pivot(index=unit_col, columns=time_col, values=outcome_col)
    wide_y = wide_y.sort_index()
    unit_ids = np.asarray(wide_y.index)

    # Dose at F (the single regressor for ALL horizons).
    d_at_F = (
        df[df[time_col] == F].set_index(unit_col).sort_index()[dose_col].to_numpy(dtype=np.float64)
    )

    dy_dict: Dict[int, np.ndarray] = {}
    anchor_y = wide_y[t_anchor].to_numpy(dtype=np.float64)
    for p in all_periods:
        if p == t_anchor:
            continue  # anchor gives ΔY = 0 trivially; skipped by contract
        e = period_to_event_time[p]
        # e = -1 corresponds to t_anchor, which we've already skipped.
        # All other periods get a horizon. The F-1 anchor / e=-1 coincidence
        # is preserved: event time -1 means "one period before F", which is
        # by definition the anchor.
        y_t = wide_y[p].to_numpy(dtype=np.float64)
        dy_dict[int(e)] = y_t - anchor_y

    cluster_arr: Optional[np.ndarray] = None
    if cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(f"cluster column {cluster_col!r} not found in data.")
        cluster_raw = data[cluster_col]
        if bool(cluster_raw.isna().any()):
            n_nan = int(cluster_raw.isna().sum())
            raise ValueError(
                f"cluster column {cluster_col!r} contains {n_nan} NaN "
                f"value(s) at the row level. Silent row dropping is "
                f"disabled; drop or impute cluster ids before calling fit()."
            )
        cluster_per_unit = df.groupby(unit_col)[cluster_col].nunique(dropna=False)
        if (cluster_per_unit > 1).any():
            n_bad = int((cluster_per_unit > 1).sum())
            raise ValueError(
                f"cluster column {cluster_col!r} is not constant within "
                f"unit for {n_bad} unit(s). Cluster must be unit-level."
            )
        cluster_arr = df.groupby(unit_col)[cluster_col].first().sort_index().to_numpy()

    return d_at_F, dy_dict, cluster_arr, unit_ids, t_anchor


# =============================================================================
# Design auto-detection
# =============================================================================


def _detect_design(d_arr: np.ndarray) -> str:
    """Resolve ``design="auto"`` to a concrete mode string.

    Implements the REGISTRY line-2320 rule:

    1. If ``d.min() == 0`` exactly, resolve unconditionally to
       ``continuous_at_zero``. (The modal-min check only runs when
       ``d.min() > 0`` since otherwise the "at d_lower" subset is always
       the "zero-dose" subset and the continuous-at-zero path is the
       paper-endorsed handling.)
    2. Otherwise, if ``d.min() < 0.01 * median(|d|)``, resolve to
       ``continuous_at_zero`` (small-share-of-treated samples that
       effectively satisfy Design 1').
    3. Else, if the modal fraction at ``d.min()`` exceeds 2%, resolve to
       ``mass_point``.
    4. Else, resolve to ``continuous_near_d_lower``.

    The rule is strict first-match: when both the ``<0.01 * median``
    threshold and the modal-fraction rule fire simultaneously (e.g., 3%
    of units at ``D = 0`` + 97% ``Uniform(0.5, 1)``), the resolution is
    ``continuous_at_zero``. This matches the paper-endorsed Design 1'
    handling of small-share-of-treated samples.

    Parameters
    ----------
    d_arr : np.ndarray, shape (G,)
        Unit-level post-period doses.

    Returns
    -------
    str
        One of ``"continuous_at_zero"``, ``"continuous_near_d_lower"``,
        ``"mass_point"``.
    """
    d_min = float(d_arr.min())

    # Tie-break: d.min() == 0 always resolves to Design 1'.
    # Use an absolute tolerance that scales with the range of the data.
    scale = max(1.0, float(np.max(np.abs(d_arr))))
    if abs(d_min) <= 1e-12 * scale:
        return "continuous_at_zero"

    median_abs = float(np.median(np.abs(d_arr)))
    # REGISTRY rule: d.min() < threshold * median(|d|) -> continuous_at_zero.
    # Guard against median_abs == 0 (all doses at zero; handled above).
    if median_abs > 0 and d_min < _CONTINUOUS_AT_ZERO_THRESHOLD * median_abs:
        return "continuous_at_zero"

    # Modal-fraction rule for mass-point detection.
    eps = 1e-12 * max(1.0, abs(d_min))
    at_d_min = np.abs(d_arr - d_min) <= eps
    modal_fraction = float(np.mean(at_d_min))
    if modal_fraction > _MASS_POINT_THRESHOLD:
        return "mass_point"

    return "continuous_near_d_lower"


# =============================================================================
# Sup-t multiplier bootstrap (Phase 4.5 B event-study simultaneous CI)
# =============================================================================


def _sup_t_multiplier_bootstrap(
    influence_matrix: np.ndarray,
    att_per_horizon: np.ndarray,
    se_per_horizon: np.ndarray,
    resolved_survey: Any,  # Optional[ResolvedSurveyDesign]
    *,
    n_bootstrap: int,
    alpha: float,
    seed: Optional[int],
    bootstrap_weights: str = "rademacher",
    cluster_ids: Optional[np.ndarray] = None,
    cluster_if_scale: float = 1.0,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], int]:
    """Compute sup-t simultaneous CI via PSU-level multiplier bootstrap.

    Reuses :func:`diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch`
    (survey path, ``resolved_survey`` not None) / :func:`generate_bootstrap_weights_batch`
    (unit- or cluster-level Rademacher when ``resolved_survey`` is None) to draw
    ``n_bootstrap`` replicates of multiplier weights
    shaped ``(n_bootstrap, n_units)``; the helpers handle stratum
    centering, lonely-PSU, and FPC scaling so this function only
    composes them with the per-unit influence function.

    Construction (mirrors `staggered_bootstrap.py:354-373` perturbation
    idiom and `:497-533` sup-t quantile idiom; NO ``(1/n)`` prefactor —
    ``psi`` is already on the θ̂-scale per the Phase 4.5 B IF scale
    convention):

    1. Draw multiplier weights ``xi`` shape ``(n_bootstrap, n_units)``.
    2. Perturbations: ``delta[b, e] = sum_g xi[b, g] * psi[g, e]``.
    3. t-statistics: ``t[b, e] = delta[b, e] / se[e]``.
    4. Sup-t: ``sup_t[b] = max_e |t[b, e]|``.
    5. Critical value: ``q = quantile(sup_t[isfinite], 1 - alpha)``.

    Under ``H=1`` the sup reduces to the marginal, so
    ``q -> Phi^{-1}(1 - alpha/2) ≈ 1.96`` at ``alpha=0.05`` (locked by
    the ``TestSupTReducesToNormalAtH1`` reduction-invariant test).

    Parameters
    ----------
    influence_matrix : np.ndarray, shape (n_units, n_horizons)
        Per-unit per-horizon influence function on the β̂-scale. NaN
        columns are treated as degenerate-horizon placeholders and drop
        out of the sup via the finite mask.
    att_per_horizon : np.ndarray, shape (n_horizons,)
        Per-horizon point estimates (used to assemble the simultaneous
        band: ``att ± q · se``).
    se_per_horizon : np.ndarray, shape (n_horizons,)
        Per-horizon analytical SE (Binder-TSL on the survey path, HC1
        sandwich on the unweighted path).
    resolved_survey : Optional[ResolvedSurveyDesign]
        ``None`` → unit-level Rademacher draw via
        ``generate_bootstrap_weights_batch``. Otherwise →
        PSU-level draw via ``generate_survey_multiplier_weights_batch``
        (stratum-centered, FPC-scaled, lonely-PSU-aware).
    n_bootstrap : int
        Number of multiplier replicates.
    alpha : float
        CI level (``0.05`` for 95% simultaneous band).
    seed : int or None
        RNG seed for reproducibility.
    bootstrap_weights : str
        Passed through to the helper: ``"rademacher"``, ``"mammen"``, or
        ``"webb"``. Default ``"rademacher"`` (binary ±1 multipliers).
    cluster_ids : np.ndarray or None, shape (n_units,)
        Per-unit cluster labels aligned to ``influence_matrix`` rows. When
        supplied, a dedicated CLUSTER-robust branch fires (bypassing both
        the survey and unit-level branches): the per-unit IF is aggregated
        to cluster level and cluster-level iid multipliers are drawn, so
        ``Var_xi(xi @ Psi_cl) = sum_c (sum_{i in c} psi_i)^2`` — the RAW
        cluster sandwich matching the analytical cluster-robust SE (no
        stratum-centering / FPC / Bessel; the continuous CCT cluster meat
        carries no ``g/(g-1)`` correction). ``resolved_survey`` must be
        ``None`` when this is set (``cluster= + survey_design=`` is rejected
        up-front). ``None`` restores the survey / unit-level dispatch.
    cluster_if_scale : float
        Path-specific finite-sample scalar applied to the cluster-aggregated
        IF before the draw: ``1.0`` for the CCT continuous path (exact by
        construction), ``sqrt(G/(G-1))`` for the mass-point CR1 path (the
        returned mass-point IF carries ``sqrt((n-1)/(n-k))`` but not the CR1
        ``G/(G-1)`` factor, so it is restored here). ``G`` is the full-array
        cluster count, identical to the ``n_clusters`` this branch computes.

    Returns
    -------
    (cband_crit_value, cband_low, cband_high, n_valid) : tuple
        ``cband_crit_value`` is the sup-t quantile (float); ``cband_low``
        / ``cband_high`` are simultaneous-band endpoints shape
        ``(n_horizons,)``; ``n_valid`` is the count of finite sup-t
        draws (<= ``n_bootstrap``). Returns ``(nan, None, None,
        n_valid)`` when fewer than half the draws are finite — warns the
        caller.
    """
    from diff_diff.bootstrap_utils import apply_stratum_centering

    influence_matrix = np.asarray(influence_matrix, dtype=np.float64)
    att_per_horizon = np.asarray(att_per_horizon, dtype=np.float64)
    se_per_horizon = np.asarray(se_per_horizon, dtype=np.float64)
    n_units, n_horizons = influence_matrix.shape
    rng = np.random.default_rng(seed)

    # Review R3 P1: the survey-aware branch must fire for ANY non-None
    # resolved_survey, including the trivial ``SurveyDesign(weights=...)``
    # case (no explicit strata / PSU / FPC). The analytical Binder-TSL
    # target under that design still applies the centered
    # (n/(n-1)) · sum(psi − psi_bar)² formula, so the bootstrap must
    # also use PSU-aggregation + stratum-demeaning + sqrt(n/(n-1))
    # scaling — not raw unit-level Rademacher draws (which skip the
    # centering and small-sample factor). The unit-level branch below is
    # reserved for the unweighted path (no survey object at all).
    use_survey_bootstrap = resolved_survey is not None

    if cluster_ids is not None:
        # Cluster-robust multiplier bootstrap (had.py clustered event-study
        # band). Aggregate the per-unit IF to cluster level and draw
        # cluster-level iid multipliers, so
        #   Var_xi(xi @ Psi_cl) = sum_c (sum_{i in c} psi_i)^2
        # — the RAW cluster sandwich, matching the analytical cluster-robust
        # SE. No stratum-centering / FPC / Bessel: the continuous CCT cluster
        # meat (lprobust_vce) carries no g/(g-1) correction, so raw Rademacher
        # is exact (scale 1.0); the mass-point CR1 path restores its G/(G-1)
        # factor via ``cluster_if_scale`` before the draw. ``G`` == this
        # branch's ``n_clusters`` == the count the analytical CR uses (full
        # array, incl. wholly-zero-weight clusters that contribute 0).
        cluster_ids_arr = np.asarray(cluster_ids).ravel()
        codes, cluster_labels = pd.factorize(cluster_ids_arr, sort=False)
        n_clusters = len(cluster_labels)
        if n_clusters < 2:
            # Cluster-robust simultaneous band undefined with one cluster.
            return float("nan"), None, None, 0
        Psi_cl = np.zeros((n_clusters, n_horizons), dtype=np.float64)
        np.add.at(Psi_cl, codes, influence_matrix)
        Psi_cl *= float(cluster_if_scale)
        perturbations = np.empty((n_bootstrap, n_horizons), dtype=np.float64)
        for _cs, _w_block in iter_weight_blocks(n_bootstrap, n_clusters, bootstrap_weights, rng):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                perturbations[_cs : _cs + _w_block.shape[0]] = _w_block @ Psi_cl
    elif use_survey_bootstrap:
        # Review R2 P1: lonely_psu="adjust" pools singleton strata into a
        # pseudo-stratum with NONZERO multipliers in the bootstrap helper,
        # but the analytical compute_survey_if_variance target for
        # singletons is centered at the global mean of PSU scores. Since
        # this PR's stratum-demean loop only matches the within-stratum
        # Binder-TSL target (and skips singletons assuming zero
        # contribution), pooled singleton multipliers would diverge from
        # the analytical variance without an additional pseudo-stratum
        # centering step. Reject with a clear pointer until the matching
        # transform is derived; "remove" / "certainty" (singleton
        # multipliers forced to zero) are fine.
        _lonely = getattr(resolved_survey, "lonely_psu", "remove")
        if _lonely == "adjust":
            strata_arr = resolved_survey.strata
            psu_arr = resolved_survey.psu
            _has_singleton = False
            if strata_arr is not None:
                for h in np.unique(strata_arr):
                    mask_h = np.asarray(strata_arr) == h
                    if psu_arr is not None:
                        n_psu_h = int(np.unique(np.asarray(psu_arr)[mask_h]).shape[0])
                    else:
                        n_psu_h = int(mask_h.sum())
                    if n_psu_h < 2:
                        _has_singleton = True
                        break
            if _has_singleton:
                raise NotImplementedError(
                    "HeterogeneousAdoptionDiD event-study sup-t bootstrap "
                    "does not yet support SurveyDesign(lonely_psu='adjust') "
                    "with singleton strata: the bootstrap helper pools "
                    "singletons with nonzero multipliers while the "
                    "analytical Binder-TSL target centers singleton PSU "
                    "scores at the global mean, and the matching "
                    "pseudo-stratum centering transform has not been "
                    "implemented. Use lonely_psu='remove' (drops singleton "
                    "contributions; matches the 'remove' analytical target) "
                    "or pass cband=False to skip the simultaneous band."
                )
        # PSU-level multiplier weights, generated one draw-block at a time so the
        # (n_bootstrap, n_psu) matrix is never materialized in full (n_psu ==
        # n_units when each unit is its own PSU). Unstratified designs tile the
        # generation; stratified / degenerate designs fall back to full
        # generation + slicing. Weight stream bit-identical to the un-chunked
        # generate_survey_multiplier_weights_batch.
        _block_size = compute_block_size(n_units, n_bootstrap)
        psu_ids, _psu_blocks = iter_survey_multiplier_weight_blocks(
            n_bootstrap, resolved_survey, bootstrap_weights, rng, block_size=_block_size
        )
        # Aggregate Psi to PSU level, stratum-demean, and apply the
        # small-sample correction so Var_xi(xi @ Psi_psu_scaled) matches
        # the analytical Binder-TSL variance exactly (review R1 P1).
        # Target:
        #   V = sum_h (1 - f_h) (n_h / (n_h - 1)) sum_j (psi_hj - psi_h_bar)²
        # ``generate_survey_multiplier_weights_batch`` already bakes the
        # (1 - f_h) FPC factor into the multipliers, so we only need to
        # pre-process Psi at the PSU level (aggregate → stratum-demean →
        # sqrt(n_h / (n_h - 1)) rescale).
        n_psu = len(psu_ids)
        psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
        Psi_psu = np.zeros((n_psu, n_horizons), dtype=np.float64)
        if resolved_survey.psu is not None:
            unit_psu = np.asarray(resolved_survey.psu)
            for i in range(n_units):
                col = psu_id_to_col[int(unit_psu[i])]
                Psi_psu[col] += influence_matrix[i]
        else:
            # Each unit is its own PSU (psu_ids = np.arange(n_units)).
            Psi_psu = influence_matrix.copy()

        # Stratum centering + Bessel rescale on the PSU-aggregated
        # influence tensor. Shared with the HAD Stute survey-bootstrap
        # family (had_pretests.stute_test / stute_joint_pretest) which
        # applies the same algebra to PSU multipliers instead — see
        # ``apply_stratum_centering`` docstring and REGISTRY
        # § "Note (Stute stratified survey-bootstrap calibration)".
        apply_stratum_centering(Psi_psu, resolved_survey, psu_ids, psu_axis=0)

        # PSU-level perturbations: (B, H) = (B, n_psu) @ (n_psu, H), accumulated
        # one draw-block at a time (output is the small (B, H), so only the weight
        # block is large). No (1/n) prefactor — Psi_psu_scaled is already on the
        # θ̂-scale matched to the analytical variance.
        perturbations = np.empty((n_bootstrap, n_horizons), dtype=np.float64)
        for _cs, _psu_block in _psu_blocks:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                perturbations[_cs : _cs + _psu_block.shape[0]] = _psu_block @ Psi_psu
    else:
        # Unit-level iid multipliers, generated one draw-block at a time at unit
        # width (no stratum centering). Var(xi @ Psi) = sum_g psi_g² matches the
        # trivial analytical variance from compute_survey_if_variance at the
        # IF-scale-invariant tolerance (PR #359 convention).
        perturbations = np.empty((n_bootstrap, n_horizons), dtype=np.float64)
        for _cs, _w_block in iter_weight_blocks(n_bootstrap, n_units, bootstrap_weights, rng):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                perturbations[_cs : _cs + _w_block.shape[0]] = _w_block @ influence_matrix

    # t-statistics via per-horizon analytical SE.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_se = np.where(
            (se_per_horizon > 0) & np.isfinite(se_per_horizon),
            se_per_horizon,
            np.nan,
        )
        t_dist = perturbations / safe_se[np.newaxis, :]  # (B, H)
        # Suppress the all-NaN-slice warning from nanmax on rows with
        # every horizon degenerate — the finite mask drops those.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sup_t_dist = np.nanmax(np.abs(t_dist), axis=1)  # (B,)

    finite_mask = np.isfinite(sup_t_dist)
    n_valid = int(finite_mask.sum())

    if n_valid < max(1, int(0.5 * n_bootstrap)):
        warnings.warn(
            f"Too few valid sup-t bootstrap samples ({n_valid}/{n_bootstrap}) "
            f"— returning NaN simultaneous-band critical value. Possible "
            f"causes: near-singular per-horizon SEs, or a survey design "
            f"with all-zero PSU multipliers after stratum-level FPC. "
            f"Inspect the per-horizon ``se`` array and consider raising "
            f"`n_bootstrap` or passing `cband=False` to skip the sup-t "
            f"band.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan"), None, None, n_valid

    q = float(np.quantile(sup_t_dist[finite_mask], 1.0 - alpha))
    # NaN-gate simultaneous-band endpoints for degenerate horizons the
    # same way ``safe_inference`` gates pointwise output: a horizon with
    # ``se <= 0`` or non-finite ``se`` gets a NaN band instead of the
    # point estimate ± 0, avoiding misleading precision (review R1 P0).
    se_valid_mask = (se_per_horizon > 0) & np.isfinite(se_per_horizon)
    cband_low = np.where(se_valid_mask, att_per_horizon - q * se_per_horizon, np.nan)
    cband_high = np.where(se_valid_mask, att_per_horizon + q * se_per_horizon, np.nan)
    return q, cband_low, cband_high, n_valid


# =============================================================================
# Mass-point 2SLS
# =============================================================================


@overload
def _fit_mass_point_2sls(
    d: np.ndarray,
    dy: np.ndarray,
    d_lower: float,
    cluster: Optional[np.ndarray],
    vcov_type: str,
    *,
    weights: Optional[np.ndarray] = ...,
    return_influence: bool = ...,
    return_intercept_se: Literal[False] = ...,
) -> Tuple[float, float, Optional[np.ndarray]]: ...


@overload
def _fit_mass_point_2sls(
    d: np.ndarray,
    dy: np.ndarray,
    d_lower: float,
    cluster: Optional[np.ndarray],
    vcov_type: str,
    *,
    weights: Optional[np.ndarray] = ...,
    return_influence: bool = ...,
    return_intercept_se: Literal[True],
) -> Tuple[float, float, Optional[np.ndarray], float]: ...


def _fit_mass_point_2sls(
    d: np.ndarray,
    dy: np.ndarray,
    d_lower: float,
    cluster: Optional[np.ndarray],
    vcov_type: str,
    *,
    weights: Optional[np.ndarray] = None,
    return_influence: bool = False,
    return_intercept_se: bool = False,
) -> Union[
    Tuple[float, float, Optional[np.ndarray]],
    Tuple[float, float, Optional[np.ndarray], float],
]:
    """Wald-IV point estimate and structural-residual 2SLS sandwich SE.

    The just-identified binary instrument ``Z_g = 1{D_{g,2} > d_lower}``
    gives a 2SLS estimator that collapses to the Wald-IV sample-average
    ratio
    ``beta_hat = (Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``.

    The STANDARD ERROR is computed via the 2SLS sandwich
    ``V = [Z'WX]^{-1} * Omega * [Z'WX]^{-T}`` where ``Omega`` is built
    from the STRUCTURAL residuals
    ``u = dy - alpha_hat - beta_hat * d`` (NOT the reduced-form
    residuals). This is the canonical 2SLS inference path and matches
    what ``estimatr::iv_robust`` / Stata ``ivregress`` would produce.

    Supported ``vcov_type``:

    - ``"classical"``: constant variance ``sigma_hat^2 = sum(w² u²) /
      (sum(w) - k)`` (weighted) or ``sum(u^2) / (n-k)`` (unweighted);
      sandwich form with ``w²`` in the meat when weighted.
    - ``"hc1"``: heteroskedasticity-robust with small-sample DOF scaling
      ``n / (n - k)``; meat = ``Z' diag(w² u²) Z`` (pweight convention,
      Wooldridge 2010 Eq. 12.37; matches ``estimatr::iv_robust(...,
      weights=..., se_type="HC1")`` bit-exactly). With ``cluster``
      supplied, switches to CR1 (Liang-Zeger) with cluster score
      ``Z'_c (w · u)_c``.

    ``"hc2"`` and ``"hc2_bm"`` raise ``NotImplementedError`` (2SLS-specific
    leverage derivation pending).

    Parameters
    ----------
    d : np.ndarray, shape (n,)
        Post-period doses ``D_{g,2}``.
    dy : np.ndarray, shape (n,)
        First-difference outcome ``Delta Y_g``.
    d_lower : float
        Support infimum / evaluation point.
    cluster : np.ndarray or None, shape (n,)
        Cluster ids per unit (``None`` for no clustering).
    vcov_type : str
        One of ``"classical"``, ``"hc1"``.
    weights : np.ndarray or None, shape (n,)
        Per-unit sampling weights (pweight convention). ``None`` for the
        unweighted path — that branch is numerically bit-exact with
        pre-Phase 4.5 B output. Zero-weight units contribute zero to
        every sum-weighted expression and zero to the returned IF.
    return_influence : bool
        When True, returns the per-unit influence function (IF) on the
        β̂-scale, shape ``(n,)`` with zeros at zero-weight rows. The IF
        is scaled so that under a trivial ``ResolvedSurveyDesign``
        (single stratum, each unit its own PSU, no FPC),
        ``compute_survey_if_variance(IF, trivial)`` ≈ ``V_HC1[1, 1]``
        at ``atol=1e-10`` (PR #359 convention; see the "IF scale
        convention" section of the Phase 4.5 B plan for derivation).
    return_intercept_se : bool
        When True, the returned tuple gains a 4th element: the intercept
        standard error ``sqrt(V[0, 0])`` (``V[1, 1]`` is the slope
        variance that drives ``se_beta``). ``NaN`` whenever ``se_beta``
        is ``NaN`` (early-exit / singular paths). Test-only surfacing hook
        for estimatr-parity coverage; production callers leave it False and
        the return is the unchanged 3-tuple.

    Returns
    -------
    tuple[float, float, np.ndarray or None]
        ``(beta_hat, se_beta, psi)``. ``psi`` is the per-unit IF when
        ``return_influence=True``, else ``None``. NaN for SE when the
        dose-gap vanishes (``Dbar_{Z=1} == Dbar_{Z=0}``) or the
        sandwich is singular; in those cases ``psi`` is returned as a
        length-``n`` zero array when ``return_influence=True``. When
        ``return_intercept_se=True`` a 4th element ``se_intercept`` is
        appended.
    """
    d = np.asarray(d, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    n = d.shape[0]

    # Weight validation / normalization. The unweighted branch preserves
    # numerical bit-parity vs pre-Phase 4.5 B by skipping np.average and
    # using plain sums / means exactly as before. `w_arr`, `w_sum`, and
    # `pos_mask` are initialized as sentinels so static typing flows
    # cleanly; only the weighted branch populates them with real values.
    weighted = weights is not None
    w_arr: np.ndarray = np.ones(n, dtype=np.float64)
    w_sum: float = float(n)
    pos_mask: np.ndarray = np.ones(n, dtype=bool)
    if weighted:
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if w_arr.shape[0] != n:
            raise ValueError(
                f"weights length ({w_arr.shape[0]}) does not match d / dy length ({n})."
            )
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("weights contains non-finite values (NaN or Inf).")
        if np.any(w_arr < 0):
            raise ValueError("weights must be non-negative.")
        w_sum = float(w_arr.sum())
        if w_sum <= 0:
            raise ValueError("weights sum to zero — no observations have positive weight.")
        pos_mask = w_arr > 0

    Z = (d > d_lower).astype(np.float64)

    # Degeneracy checks on the positive-weight subset (zero-weight units
    # do not drive design resolution; same subpopulation convention as
    # PR #359). Under unweighted fits the "positive" subset is the full
    # sample so behavior is unchanged.
    if weighted:
        n_above = int(((Z == 1) & pos_mask).sum())
        n_at_or_below = int(((Z == 0) & pos_mask).sum())
    else:
        n_above = int(Z.sum())
        n_at_or_below = n - n_above

    _null_psi = np.zeros(n, dtype=np.float64) if return_influence else None

    def _pack(
        beta: float,
        se_b: float,
        psi_out: Optional[np.ndarray],
        se_int: float = float("nan"),
    ) -> Union[
        Tuple[float, float, Optional[np.ndarray]],
        Tuple[float, float, Optional[np.ndarray], float],
    ]:
        # Centralize the optional 4th (intercept-SE) element so every one of
        # the function's early-exit and success returns keeps a consistent
        # arity. The @overload stubs give callers the precise per-flag type;
        # the True branch appends se_int for the test-only surfacing hook.
        if return_intercept_se:
            return (beta, se_b, psi_out, se_int)
        return (beta, se_b, psi_out)

    if n_above == 0 or n_at_or_below == 0:
        return _pack(float("nan"), float("nan"), _null_psi)

    # Point estimate: weighted Wald-IV ratio (reduces to unweighted at w=1).
    if weighted:
        Z1_idx = (Z == 1) & pos_mask
        Z0_idx = (Z == 0) & pos_mask
        w_Z1 = float(w_arr[Z1_idx].sum())
        w_Z0 = float(w_arr[Z0_idx].sum())
        if w_Z1 <= 0.0 or w_Z0 <= 0.0:
            return _pack(float("nan"), float("nan"), _null_psi)
        dose_gap = float(
            (w_arr[Z1_idx] * d[Z1_idx]).sum() / w_Z1 - (w_arr[Z0_idx] * d[Z0_idx]).sum() / w_Z0
        )
        d_bar = float((w_arr * d).sum() / w_sum)
        dy_gap = float(
            (w_arr[Z1_idx] * dy[Z1_idx]).sum() / w_Z1 - (w_arr[Z0_idx] * dy[Z0_idx]).sum() / w_Z0
        )
        dy_bar = float((w_arr * dy).sum() / w_sum)
    else:
        dose_gap = d[Z == 1].mean() - d[Z == 0].mean()
        d_bar = float(d.mean())
        dy_gap = dy[Z == 1].mean() - dy[Z == 0].mean()
        dy_bar = float(dy.mean())

    if abs(dose_gap) < 1e-12 * max(1.0, abs(d_bar)):
        # No dose variation around d_lower -> beta undefined.
        return _pack(float("nan"), float("nan"), _null_psi)

    # dy_gap / dy_bar were computed inside the weighted/unweighted blocks above.
    beta_hat = float(dy_gap / dose_gap)
    alpha_hat = float(dy_bar - beta_hat * d_bar)

    # STRUCTURAL residuals: u = y - alpha - beta*x. The Wald-IV /
    # OLS-on-indicator shortcut would use reduced-form residuals
    # u_rf = dy - (alpha_rf + gamma * Z), which differ in finite
    # samples and substantively under clustering.
    u = dy - alpha_hat - beta_hat * d

    # Design matrices: X = [1, d] (endogenous), Z_d = [1, Z] (instrument).
    X = np.column_stack([np.ones(n, dtype=np.float64), d])
    Zd = np.column_stack([np.ones(n, dtype=np.float64), Z])

    if weighted:
        # Weighted bread: Z' diag(w) X. Zero-weight rows contribute 0.
        ZtWX = Zd.T @ (w_arr[:, None] * X)
    else:
        ZtWX = Zd.T @ X

    try:
        ZtWX_inv = np.linalg.inv(ZtWX)
    except np.linalg.LinAlgError:
        return _pack(beta_hat, float("nan"), _null_psi)

    vcov_type = vcov_type.lower()
    if vcov_type in _MASS_POINT_VCOV_UNSUPPORTED:
        raise NotImplementedError(
            f"vcov_type={vcov_type!r} is not supported on the "
            f"HeterogeneousAdoptionDiD mass-point path. HC2 / HC2-BM "
            f"require a 2SLS-specific leverage derivation "
            f"`x_i' (Z'WX)^{{-1}}(...)(X'WZ)^{{-1}} x_i` that differs "
            f"from the OLS leverage `x_i' (X'WX)^{{-1}} x_i`. Derivation "
            f"+ R parity anchor are queued for a follow-up PR. Use "
            f"vcov_type='hc1' or 'classical' for now."
        )

    k = 2  # intercept + dose
    if cluster is not None:
        # CR1 (Liang-Zeger) cluster-robust sandwich.
        # Use pd.unique to match R's first-appearance order (stable for
        # cross-runtime reproducibility).
        clusters_unique = pd.unique(cluster)
        Omega = np.zeros((2, 2), dtype=np.float64)
        wu = (w_arr * u) if weighted else u
        for c in clusters_unique:
            idx = cluster == c
            # score per cluster: s_c = Zd[idx]' @ (w · u)[idx]
            s = Zd[idx].T @ wu[idx]
            Omega += np.outer(s, s)
        n_clusters = len(clusters_unique)
        if n_clusters < 2:
            # Cluster-robust SE undefined with a single cluster.
            return _pack(beta_hat, float("nan"), _null_psi)
        Omega *= (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    elif vcov_type == "classical":
        if weighted:
            dof = w_sum - k
            if dof <= 0:
                return _pack(beta_hat, float("nan"), _null_psi)
            sigma2 = float((w_arr * w_arr * u * u).sum()) / dof
            Omega = sigma2 * (Zd.T @ ((w_arr * w_arr)[:, None] * Zd))
        else:
            dof = n - k
            if dof <= 0:
                return _pack(beta_hat, float("nan"), _null_psi)
            sigma2 = float((u * u).sum()) / dof
            Omega = sigma2 * (Zd.T @ Zd)
    elif vcov_type == "hc1":
        dof = n - k
        if dof <= 0:
            return _pack(beta_hat, float("nan"), _null_psi)
        if weighted:
            # Pweight HC1: meat = Z' diag(w² u²) Z (Wooldridge 2010 Eq 12.37,
            # matches linalg.py:1141 convention and estimatr::iv_robust
            # HC1 bit-exactly).
            Omega = (n / dof) * (Zd.T @ ((w_arr * w_arr * u * u)[:, None] * Zd))
        else:
            Omega = (n / dof) * (Zd.T @ ((u * u)[:, None] * Zd))
    else:
        raise ValueError(
            f"Unsupported vcov_type={vcov_type!r} on the HAD mass-point "
            f"path. Supported: {_MASS_POINT_VCOV_SUPPORTED} (plus "
            f"cluster-robust CR1 via cluster=)."
        )

    V = ZtWX_inv @ Omega @ ZtWX_inv.T
    var_beta = float(V[1, 1])
    if not np.isfinite(var_beta) or var_beta < 0:
        return _pack(beta_hat, float("nan"), _null_psi)
    se_beta = float(np.sqrt(var_beta))
    # Intercept SE = sqrt(V[0, 0]); surfaced only when return_intercept_se
    # (see _pack). NaN if the intercept variance is non-finite/negative.
    var_intercept = float(V[0, 0])
    se_intercept = (
        float(np.sqrt(var_intercept))
        if np.isfinite(var_intercept) and var_intercept >= 0
        else float("nan")
    )

    if not return_influence:
        return _pack(beta_hat, se_beta, None, se_intercept)

    # Per-unit influence function on β̂-scale, scaled so that
    # compute_survey_if_variance(psi, trivial_resolved) ≈ V_HC1[1, 1]
    # at atol=1e-10. Derivation (see Phase 4.5 B plan, "IF scale
    # convention"): V_HC1[1,1] = (n/(n-k)) · sum_j psi₀_j² with
    # psi₀_j = [(Z'WX)^{-1} z_j w_j u_j][1]; trivial
    # compute_survey_if_variance reduces to (n/(n-1)) · sum_j psi_j².
    # Setting psi_j = psi₀_j · sqrt((n-1)/(n-k)) makes the two agree.
    # For the unweighted path w=1 is the degenerate case and psi₀_j
    # equals the standard OLS IF.
    bread_row = ZtWX_inv[1, :]  # shape (2,)
    if weighted:
        psi0 = (Zd @ bread_row) * (w_arr * u)  # shape (n,)
    else:
        psi0 = (Zd @ bread_row) * u
    dof_psi = max(n - k, 1)
    psi = psi0 * np.sqrt((n - 1) / dof_psi)
    return _pack(beta_hat, se_beta, psi, se_intercept)


# =============================================================================
# Main estimator class
# =============================================================================


class HeterogeneousAdoptionDiD:
    """Heterogeneous Adoption Difference-in-Differences estimator.

    Implements de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026)
    Weighted-Average-Slope (WAS) estimator with three design-dispatch
    paths: Design 1' (continuous-at-zero), Design 1 continuous-near-
    d_lower, and Design 1 mass-point (2SLS sample-average per paper
    Section 3.2.4). Two aggregation modes:

    - ``aggregate="overall"`` (Phase 2a, default) returns a single-period
      :class:`HeterogeneousAdoptionDiDResults` on a two-period panel.
    - ``aggregate="event_study"`` (Phase 2b, paper Appendix B.2) returns
      a :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
      event-time WAS estimates on a multi-period panel, using a uniform
      ``F-1`` anchor and pointwise CIs per horizon. Staggered-timing
      panels auto-filter to the last-treatment cohort plus never-treated
      units (paper Appendix B.2 prescription).

    Parameters
    ----------
    design : {"auto", "continuous_at_zero", "continuous_near_d_lower", "mass_point"}
        Design-dispatch strategy. Defaults to ``"auto"`` which resolves
        via the REGISTRY auto-detect rule on the fitted dose data
        (see :func:`_detect_design`).

        Explicit overrides are checked against the paper's
        regime-partition contract (Section 3.2) at fit time:

        - ``"continuous_at_zero"`` (Design 1'): paper requires the
          support infimum ``d_lower = 0``. Phase 1c's
          ``_validate_had_inputs`` rejects mass-point samples passed
          to this path.
        - ``"continuous_near_d_lower"`` (Design 1, continuous density
          near ``d_lower``): requires ``d_lower > 0`` and a
          non-mass-point sample (modal fraction at ``d.min()`` must be
          <= 2%). ``d_lower`` must equal ``float(d.min())`` within
          float tolerance; non-support-infimum thresholds are off-
          support and raise.
        - ``"mass_point"`` (Design 1 mass-point): requires
          ``d_lower > 0`` AND a mass-point sample (modal fraction at
          ``d.min()`` must be > 2%). ``d_lower`` must equal
          ``float(d.min())`` within float tolerance. Forcing this
          design on a ``d_lower = 0`` sample or on a continuous
          (non-mass-point) sample raises; in either case 2SLS
          identifies a different estimand than the paper's Design 1
          mass-point WAS.

        Mismatched overrides raise ``ValueError`` pointing at the
        correct design rather than silently identifying a different
        estimand.
    d_lower : float or None
        Support infimum ``d_lower``. ``None`` means use ``0.0`` on the
        Design 1' path and ``float(d.min())`` on the other two paths.
        On Design 1 paths (``continuous_near_d_lower`` and
        ``mass_point``), an explicit ``d_lower`` must equal
        ``float(d.min())`` within float tolerance AND must be strictly
        positive; zero-valued or mismatched thresholds raise.
    kernel : {"epanechnikov", "triangular", "uniform"}
        Forwarded to :func:`bias_corrected_local_linear` on the continuous
        paths. Ignored on the mass-point path.
    alpha : float
        CI level (0.05 for 95% CI).
    vcov_type : {"classical", "hc1"} or None
        Mass-point-path only. When ``None``, the effective family falls
        back to the ``robust`` flag: ``robust=True`` -> ``"hc1"``,
        ``robust=False`` -> ``"classical"`` (the default construction).
        Explicit ``"hc2"`` and ``"hc2_bm"`` raise ``NotImplementedError``
        pending a 2SLS-specific leverage derivation. Ignored on the
        continuous paths (which use the CCT-2014 robust SE from
        Phase 1c); passing a non-default ``vcov_type`` on a continuous
        path emits a ``UserWarning`` per fit call.
    robust : bool
        Backward-compat alias used only when ``vcov_type is None``:
        ``True`` -> ``"hc1"``, ``False`` -> ``"classical"``. Explicit
        ``vcov_type`` takes precedence (e.g.,
        ``vcov_type="classical", robust=True`` runs classical). Only
        the mass-point path consumes these; continuous paths ignore
        both with a warning.
    cluster : str or None
        Column name for cluster-robust SE. On the mass-point path this is
        the 2SLS CR1 sandwich; on the continuous (``continuous_at_zero`` /
        ``continuous_near_d_lower``) paths (Phase 2a) it threads the cluster
        IDs into ``bias_corrected_local_linear`` so ``se_robust`` is the
        cluster-robust CCT-2014 nonparametric SE (``β̂``-scale
        ``se = se_robust / |den|``). A bare ``cluster=`` gives unweighted
        cluster-robust inference; the cluster + ``survey_design=``
        composition raises ``NotImplementedError`` (the Binder-TSL survey
        variance would override the cluster-robust SE — for weighted
        clustering route through
        ``survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>')`` instead).
        Cluster must be constant within unit. On the event-study path
        (``aggregate="event_study"``, Phase 2b) ``cluster=`` provides
        cluster-robust per-horizon pointwise CIs (both designs) AND a
        cluster-robust simultaneous sup-t band (``cband=True``, fires even
        on unweighted fits); ``cluster=`` + ``survey_design=`` is rejected
        there too.

    Notes
    -----
    **Non-testable assumptions (paper Section 3.1.2).** Point identification
    of ``WAS_{d_lower}`` on the Design 1 family
    (``continuous_near_d_lower`` and ``mass_point``) requires Assumption 6
    in addition to parallel trends; sign identification requires
    Assumption 5. Neither is testable via pre-trends:

    - Assumption 5 (sign identification): the boundary slope-ratio
      ``lim_{d down d_lower} E(TE_2 | D_2 <= d) / WAS < E(D_2) / d_lower``
      relates the conditional expectation near the boundary to the
      overall WAS; it cannot be inferred from pre-period outcome
      trajectories alone.
    - Assumption 6 (point identification): the counterfactual-mean
      alignment ``lim_{d down d_lower} E[Y_2(d_lower) - Y_2(0) | D_2 <= d]
      = E[Y_2(d_lower) - Y_2(0)]`` is a statement about an unobserved
      counterfactual at the support infimum.

    The fit() method emits a ``UserWarning`` whenever ``resolved_design``
    is on the Design 1 family (``continuous_near_d_lower`` or
    ``mass_point``) so users are not silently led to interpret point
    estimates as full point identification. The available pre-tests
    verify ADJACENT identifying conditions:

    - :func:`diff_diff.qug_test`: Theorem 4 / Design 1' support-infimum
      null ``d_lower = 0`` (adjacent evidence on the ``d_lower = 0``
      clause of Assumption 4 only, NOT a test of the full Assumption 4
      statement which also covers boundary-density positivity,
      conditional-mean smoothness, conditional-variance regularity, and
      bandwidth conditions).
    - :func:`diff_diff.stute_test` / :func:`diff_diff.yatchew_hr_test`:
      Assumption 8 linearity of ``E[ΔY | D_2]`` in ``D_2`` (residuals
      from ``dy ~ 1 + d``).
    - :func:`diff_diff.joint_pretrends_test`: Assumption 7
      mean-independence pre-trends across multi-period placebos
      (intercept-only residual form via ``null_form="mean_independence"``;
      the raw ``stute_test`` / ``yatchew_hr_test`` helpers do NOT cover
      Assumption 7 on their own).

    None of these test Assumptions 5 or 6 directly. The Assumption 5/6
    non-testability caveat is surfaced by the Design 1 fit-time
    ``UserWarning`` and by T21 (HAD pretest workflow tutorial) prose,
    NOT by the composite workflow verdict string (which only flags the
    Assumption 7 step-2 gap on the two-period ``aggregate="overall"``
    path).

    **Diagnostics coverage.** ``HeterogeneousAdoptionDiDResults.bandwidth_diagnostics``
    and ``.bias_corrected_fit`` are populated only on the continuous
    paths; both are ``None`` on the mass-point path (which is parametric
    and has no bandwidth). Conversely, ``.n_mass_point`` and
    ``.n_above_d_lower`` are populated only on the mass-point path.

    **Clone idempotence.** ``self.design`` stores the RAW user input
    (e.g., ``"auto"``); the resolved mode is stored on the result object
    at fit time. This mirrors Phase 1a's ``_vcov_type_arg`` pattern and
    keeps ``get_params()`` / ``sklearn.clone()`` round-trips exact.

    Examples
    --------
    Construct a two-period HAD panel by hand. Phase 2a requires exactly
    two periods with ``D_{g,1} = 0`` for every unit.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from diff_diff import HeterogeneousAdoptionDiD
    >>> rng = np.random.default_rng(42)  # doctest: +SKIP
    >>> G = 500  # doctest: +SKIP
    >>> dose_post = rng.uniform(0.0, 1.0, G)  # doctest: +SKIP
    >>> dose_post[0] = 0.0  # at least one zero-dose unit for Design 1'
    >>> delta_y = 0.3 * dose_post + 0.1 * rng.standard_normal(G)  # doctest: +SKIP
    >>> data = pd.DataFrame({  # doctest: +SKIP
    ...     "unit": np.repeat(np.arange(G), 2),
    ...     "period": np.tile([1, 2], G),
    ...     "dose": np.column_stack([np.zeros(G), dose_post]).ravel(),
    ...     "outcome": np.column_stack([np.zeros(G), delta_y]).ravel(),
    ... })
    >>> est = HeterogeneousAdoptionDiD(design="auto")  # doctest: +SKIP
    >>> result = est.fit(  # doctest: +SKIP
    ...     data, outcome_col="outcome", dose_col="dose",
    ...     time_col="period", unit_col="unit",
    ... )
    >>> result.design  # doctest: +SKIP
    'continuous_at_zero'
    """

    def __init__(
        self,
        design: str = "auto",
        d_lower: Optional[float] = None,
        kernel: str = "epanechnikov",
        alpha: float = 0.05,
        vcov_type: Optional[str] = None,
        robust: bool = False,
        cluster: Optional[str] = None,
        n_bootstrap: int = 999,
        seed: Optional[int] = None,
    ) -> None:
        self.design = design
        self.d_lower = d_lower
        self.kernel = kernel
        self.alpha = alpha
        self.vcov_type = vcov_type
        self.robust = robust
        self.cluster = cluster
        # Event-study sup-t simultaneous-CI support. ``n_bootstrap`` =
        # number of multiplier-bootstrap replicates for the sup-t band;
        # ``seed`` = reproducibility seed for the multiplier draws. Both are
        # consulted when ``aggregate="event_study"`` + ``cband=True``
        # (default) AND either ``survey_design=`` (weighted/survey band,
        # Phase 4.5 B) OR ``cluster=`` (cluster-robust band, Phase 2b
        # — fires even unweighted). An unweighted, unclustered event-study
        # skips the bootstrap entirely (pre-Phase 4.5 B output preserved).
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self._validate_constructor_args()

    def _validate_constructor_args(self) -> None:
        if self.design not in _VALID_DESIGNS:
            raise ValueError(
                f"Invalid design={self.design!r}. Must be one of " f"{_VALID_DESIGNS}."
            )
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}.")
        # d_lower must be None or a finite scalar. NaN / +/-inf would
        # bypass every downstream comparison-based guard (`>`, `<=`,
        # tolerance-abs-diff) because those return False for NaN, so we
        # front-door reject non-finite values here. This fires under
        # __init__ and set_params (which dry-runs through the
        # constructor), keeping the contract uniform across all entry
        # points.
        if self.d_lower is not None:
            d_lower_float = float(self.d_lower)
            if not np.isfinite(d_lower_float):
                raise ValueError(
                    f"d_lower must be None or a finite scalar; got "
                    f"d_lower={self.d_lower!r}. NaN and +/-inf are not "
                    f"valid support-infimum values."
                )
        if self.vcov_type is not None:
            if self.vcov_type.lower() in _MASS_POINT_VCOV_UNSUPPORTED:
                # Don't raise here — the raise happens at fit() time on
                # the mass-point path so users who set vcov_type=hc2 for
                # a continuous fit (which ignores the knob) don't get
                # blocked. But we also don't silently accept it at
                # construction if it's outside our documented enum.
                pass
            elif (
                self.vcov_type.lower()
                not in _MASS_POINT_VCOV_SUPPORTED + _MASS_POINT_VCOV_UNSUPPORTED
            ):
                raise ValueError(
                    f"Invalid vcov_type={self.vcov_type!r}. Must be one "
                    f"of {_MASS_POINT_VCOV_SUPPORTED + _MASS_POINT_VCOV_UNSUPPORTED}, "
                    f"or None."
                )
        # Phase 4.5 B: n_bootstrap must be a positive int; seed must be
        # None or a nonneg int (numpy default_rng contract).
        if not isinstance(self.n_bootstrap, (int, np.integer)):
            raise ValueError(f"n_bootstrap must be an int; got {type(self.n_bootstrap).__name__}.")
        if int(self.n_bootstrap) < 1:
            raise ValueError(f"n_bootstrap must be >= 1; got {self.n_bootstrap!r}.")
        if self.seed is not None:
            if not isinstance(self.seed, (int, np.integer)):
                raise ValueError(f"seed must be None or an int; got {type(self.seed).__name__}.")
            if int(self.seed) < 0:
                raise ValueError(f"seed must be nonneg; got {self.seed!r}.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the raw constructor parameters (sklearn-compatible).

        Matches the :meth:`sklearn.base.BaseEstimator.get_params`
        signature. Preserves the user's original inputs - in particular,
        ``design`` returns ``"auto"`` when the user set it to ``"auto"``
        (even after fit), so ``sklearn.base.clone(est)`` round-trips
        exactly.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for sklearn-contract compatibility. This estimator
            has no nested sub-estimator parameters, so ``deep=False``
            and ``deep=True`` return the same dict.
        """
        del deep  # accepted for compat; this estimator has no nested params
        return {
            "design": self.design,
            "d_lower": self.d_lower,
            "kernel": self.kernel,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
            "robust": self.robust,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> "HeterogeneousAdoptionDiD":
        """Set estimator parameters and return self (sklearn-compatible).

        Only keys returned by :meth:`get_params` are accepted. Passing
        any other attribute name (including method names like ``fit``)
        raises ``ValueError`` so the estimator cannot be silently
        corrupted by a mistyped or attacker-supplied key.

        Mutation is ATOMIC: validation runs on a proposed merged
        parameter dict before any attribute is overwritten. A failing
        call (invalid key, or an otherwise valid key whose value
        violates the constructor constraints) leaves ``self`` unchanged
        and safe to reuse.
        """
        valid_keys = set(self.get_params().keys())
        invalid = [k for k in params if k not in valid_keys]
        if invalid:
            raise ValueError(
                f"Invalid parameter: {invalid[0]!r}. Valid parameters: " f"{sorted(valid_keys)}."
            )
        # Dry-run validation by constructing a fresh instance with the
        # merged state. If the constructor raises, self is not mutated.
        merged = self.get_params()
        merged.update(params)
        type(self)(**merged)  # raises ValueError on invalid combination
        # All checks passed; apply atomically.
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Main fit entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        dose_col: str,
        time_col: str,
        unit_col: str,
        first_treat_col: Optional[str] = None,
        aggregate: str = "overall",
        *,
        cband: bool = True,
        survey_design: Any = None,
        trends_lin: bool = False,
        covariates: Any = None,
    ) -> Union[HeterogeneousAdoptionDiDResults, HeterogeneousAdoptionDiDEventStudyResults]:
        """Fit the HAD estimator.

        ``aggregate="overall"`` (default) fits on a two-period panel and
        returns a :class:`HeterogeneousAdoptionDiDResults` with the
        single-period WAS estimate. ``aggregate="event_study"`` fits on
        a multi-period panel (``T > 2``) and returns a
        :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
        event-time WAS estimates using a uniform ``F-1`` anchor (paper
        Appendix B.2).

        Both the overall and event-study paths are **panel-only**: the paper
        (Section 2) defines HAD on panel or repeated-cross-section data,
        but this implementation requires a balanced panel with a unit
        identifier so that unit-level first differences
        ``ΔY_{g,t} = Y_{g,t} - Y_{g,t_anchor}`` can be formed.
        Repeated-cross-section inputs (disjoint unit IDs between
        periods) are rejected by the balanced-panel validator.
        Repeated-cross-section support is queued for a follow-up PR
        (tracked in ``TODO.md``); it requires a separate identification
        path based on pre/post cell means rather than unit-level
        differences.

        Parameters
        ----------
        data : pd.DataFrame
        outcome_col, dose_col, time_col, unit_col : str
            Column names.
        first_treat_col : str or None
            Optional first-treatment column (the period at which each
            unit first receives treatment; ``0`` for never-treated).
            For common-adoption panels the column is optional; when
            omitted, the event-study path infers the first-treatment
            period ``F`` from the dose invariant. **Staggered-timing
            contract (HAD Appendix B.2):**

            - **`first_treat_col` supplied + multiple cohorts detected**:
              auto-filter to the last-treatment cohort + never-treated
              units with a ``UserWarning`` naming kept / dropped counts.
            - **`first_treat_col` omitted + multiple distinct first-
              positive-dose cohorts inferred from the dose path**: the
              estimator FAIL-CLOSES with ``ValueError`` directing the
              user to either pass ``first_treat_col`` (activates the
              auto-filter) or use :class:`ChaisemartinDHaultfoeuille`
              (``did_multiplegt_dyn``) for full staggered support. See
              REGISTRY § "Library extension: Staggered-timing fail-
              closed" for the rationale on raising vs. warning.
        aggregate : {"overall", "event_study"}
            ``"overall"`` (default): returns a single-period
            :class:`HeterogeneousAdoptionDiDResults` (Phase 2a). Requires
            exactly two time periods.
            ``"event_study"`` (Phase 2b): returns a
            :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
            event-time WAS estimates on the multi-period panel (paper
            Appendix B.2). Requires more than two time periods. Pointwise
            CIs per horizon; joint cross-horizon covariance is deferred
            to a follow-up PR. Staggered-timing panels: see the
            ``first_treat_col`` contract above (auto-filter to last
            cohort + never-treated with ``UserWarning`` when supplied;
            fail-closed ``ValueError`` when omitted on a staggered
            panel).
        survey_design : SurveyDesign or None, keyword-only
            Survey design (sampling weights + optional strata / PSU / FPC)
            for design-based inference. Supported on ALL design × aggregate
            combinations after Phase 4.5 B: continuous paths
            (``continuous_at_zero``, ``continuous_near_d_lower``) on both
            ``aggregate="overall"`` and ``aggregate="event_study"``, AND
            the ``mass_point`` design on both aggregates. Continuous paths
            compose the SE via :func:`compute_survey_if_variance` (Binder
            1983 TSL); weights propagate pointwise into the lprobust
            kernel. Mass-point composes the per-unit 2SLS IF on the
            HC1-scale and Binder-TSL-aggregates that — requires
            ``vcov_type='hc1'`` (the classical default raises
            ``NotImplementedError`` on the survey path). Event-study fits
            with ``cband=True`` add a multiplier-bootstrap simultaneous
            confidence band. Only ``weight_type="pweight"`` is supported
            (``aweight`` / ``fweight`` raise ``NotImplementedError``).
            Survey design columns (strata / PSU / FPC) must be constant
            within unit (sampling-unit-level assignment); within-unit
            variance raises ``ValueError``. Replicate-weight designs raise
            ``NotImplementedError``. See
            ``docs/methodology/REGISTRY.md`` § HeterogeneousAdoptionDiD —
            "Note (HAD survey-design API consolidation)" for the full
            dispatch matrix.
        cband : bool, default True
            Controls the multiplier-bootstrap simultaneous confidence band
            on the event-study path. When ``True`` (default) and
            ``aggregate="event_study"`` AND either (a) ``survey_design=``
            is supplied (weighted/survey band,
            Phase 4.5 B) OR (b) ``cluster=`` is supplied (cluster-robust
            band — fires even on an unweighted fit, Phase 2b), the fit
            populates ``cband_low`` / ``cband_high`` / ``cband_crit_value``
            / ``cband_method`` / ``cband_n_bootstrap`` on the result. When
            ``False`` those fields stay ``None``. No effect on
            ``aggregate="overall"`` or on unweighted, unclustered event-
            study. ``n_bootstrap`` and ``seed`` (constructor params)
            control replicate count and RNG; defaults are 999 / ``None``.
        trends_lin : bool, default False, keyword-only
            When ``True``, applies paper Eq 17 linear-trend detrending
            to per-event-time outcome evolutions. Mirrors R
            ``DIDHAD::did_had(..., trends_lin=TRUE)``. Per-group slope
            is estimated as ``Y[g, F-1] - Y[g, F-2]``; each event-time
            ``e`` evolution is replaced by ``dy_dict[e] - (e+1) ×
            slope`` (uniform formula that absorbs both effect-side
            detrending and placebo-side anchor swap). Requires
            ``aggregate="event_study"`` AND ``F >= 3`` (panel must
            include both ``F-1`` and ``F-2``); raises
            ``NotImplementedError`` on ``aggregate="overall"`` and
            ``ValueError`` on ``F < 3``. The "consumed" placebo at
            event time ``e=-2`` is auto-dropped (R reduces max
            placebo lag by 1 with the same effect). Mutually
            exclusive with survey weighting (``survey_design``);
            raises ``NotImplementedError`` if combined. Default
            ``False`` preserves bit-exact
            backcompat with all pre-PR fits.
        covariates : array-like or None, default None, keyword-only
            NOT YET IMPLEMENTED. Reserved for the covariate-adjusted HAD
            identification of de Chaisemartin et al. (2026), Appendix B.1 /
            Theorem 6 (the multivariate-covariate extension). A non-None
            value raises ``NotImplementedError`` with a pointer to that
            extension; pre-residualize the outcome on the covariates before
            calling ``fit()``, or omit ``covariates=`` for the unconditional
            WAS estimand.

        Returns
        -------
        HeterogeneousAdoptionDiDResults
            When ``aggregate="overall"`` (the default; two-period only):
            single-period WAS estimate plus shared metadata.
        HeterogeneousAdoptionDiDEventStudyResults
            When ``aggregate="event_study"`` (multi-period panel; on
            staggered panels auto-filters to the last cohort plus
            never-treated): per-event-time WAS estimates with per-
            horizon arrays.

        Notes
        -----
        On the ``aggregate="overall"`` path, ``fit()`` emits a ``UserWarning``
        when a non-trivial fraction (``>= 10%``, a library convention) of
        units have exactly-zero post-period dose — a genuine untreated mass
        for which a standard DiD may be more appropriate (de Chaisemartin
        et al. 2026, Section 2). The event-study path does not warn: it
        *requires* never-treated units per Appendix B.2.
        """
        # ---- aggregate / survey_design validation ----
        if aggregate not in _VALID_AGGREGATES:
            raise ValueError(
                f"Invalid aggregate={aggregate!r}. Must be one of " f"{_VALID_AGGREGATES}."
            )

        # ---- covariates= future-work trap (TODO L73) ----
        # Covariate-adjusted HAD identification (de Chaisemartin et al. 2026,
        # Appendix B.1 / Theorem 6 — the multivariate-covariate extension) is not
        # implemented. An explicit param + NotImplementedError surfaces the roadmap
        # (vs the bare TypeError a missing kwarg would raise) while leaving
        # unknown-kwarg typos a normal TypeError. Placed after the survey/weights
        # mutex and before the event-study dispatch so the single raise covers BOTH
        # aggregate="overall" and aggregate="event_study".
        if covariates is not None:
            raise NotImplementedError(
                "HeterogeneousAdoptionDiD.fit(covariates=...) is not yet "
                "implemented. Covariate-adjusted HAD identification (de "
                "Chaisemartin et al. 2026, Appendix B.1 / Theorem 6 — the "
                "multivariate-covariate extension) requires a multivariate "
                "nonparametric regression of dY on (D, X) at the dose boundary, "
                "which is not derived here. Pre-residualize the outcome on the "
                "covariates before calling fit(), or omit covariates= for the "
                "unconditional WAS estimand."
            )

        # ---- trends_lin scope gates (PR #389 / Phase 4 R-parity).
        # `trends_lin=True` implements paper Eq 17 linear-trend detrending
        # (per-group slope from Y[F-1]-Y[F-2], applied to per-event-time
        # outcome evolutions). Requires F >= 3 (need both F-1 and F-2 in
        # panel) and is currently event-study-only — the overall path is
        # 2-period and cannot accommodate the F-2 row.
        if trends_lin:
            if aggregate != "event_study":
                raise NotImplementedError(
                    "HAD.fit(trends_lin=True) requires "
                    "aggregate='event_study' (the linear-trend slope "
                    "estimator needs Y at F-2, which a 2-period panel "
                    "does not contain). Pass a panel with at least 3 "
                    "periods and aggregate='event_study'; the per-"
                    "horizon arrays in the resulting "
                    "HeterogeneousAdoptionDiDEventStudyResults provide "
                    "the same single-effect / per-effect estimates as "
                    "the overall path."
                )
            if survey_design is not None:
                raise NotImplementedError(
                    "HAD.fit(trends_lin=True) is not yet supported with "
                    "survey weighting (`survey_design=`). The per-group slope "
                    "estimator's weighted variant (weighted-OLS slope? per-PSU "
                    "slope?) is not derived from the paper. Use trends_lin=True "
                    "WITHOUT survey weights, or use survey weights WITHOUT "
                    "trends_lin. Tracked in TODO.md as a follow-up if user "
                    "demand emerges."
                )

        # `survey_design=` is the sole public weighting entry. The fit body
        # reads the internal `survey` name for the (possibly-None) SurveyDesign.
        survey = survey_design

        # Type guard on the data-in surface (PR #376 R8 P1): HAD.fit()
        # accepts a SurveyDesign that gets resolved against `data` at fit
        # time; a pre-resolved ResolvedSurveyDesign (or its
        # make_pweight_design factory output) goes to the array-in pretest
        # helpers, NOT to fit(). Reject explicitly with migration guidance
        # rather than letting `survey.resolve(data)` AttributeError or
        # `survey.weights` (a numpy array on Resolved) be misinterpreted as
        # a column name. Mirrors the array-in helpers' isinstance-SurveyDesign
        # rejection in stute_test/yatchew_hr_test/stute_joint_pretest.
        if survey is not None and not hasattr(survey, "resolve"):
            raise TypeError(
                "HeterogeneousAdoptionDiD.fit: `survey_design=` accepts a "
                "SurveyDesign instance (column-referencing, gets "
                "`.resolve(data)`'d at fit time) on the data-in estimator "
                "surface. Got "
                f"{type(survey).__name__} (no `.resolve()` method). "
                "If you have a pre-resolved ResolvedSurveyDesign or used "
                "`make_pweight_design(arr)`, that pattern is for the "
                "array-in pretest helpers (`stute_test`, `yatchew_hr_test`, "
                "`stute_joint_pretest`). On HAD.fit, add the weights as a "
                "column on `data` and pass "
                "`survey_design=SurveyDesign(weights='col_name', ...)`."
            )

        # Dispatch the event-study path to a dedicated method so the
        # single-period path stays unchanged (Phase 2a contract preserved).
        # The static return-type annotation on `fit()` is
        # Union[HeterogeneousAdoptionDiDResults,
        # HeterogeneousAdoptionDiDEventStudyResults]; callers should
        # narrow via isinstance (or by the aggregate they passed) to
        # access aggregate-specific fields.
        if aggregate == "event_study":
            return self._fit_event_study(  # type: ignore[return-value]
                data=data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
                survey=survey,
                cband=cband,
                trends_lin=trends_lin,
            )

        # ---- Resolve effective fit-time state (local vars only, per
        # feedback_fit_does_not_mutate_config; do not mutate self.*) ----
        design_arg = self.design
        d_lower_arg = self.d_lower
        vcov_type_arg = self.vcov_type
        robust_arg = self.robust
        cluster_arg = self.cluster

        # ---- Validate panel contract ----
        t_pre, t_post = _validate_had_panel(
            data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        )

        # ---- Aggregate to unit-level first differences (no cluster yet) ----
        # Cluster validation/extraction is re-run below (after design
        # resolution) whenever cluster= is set, so this first aggregation
        # skips the cluster column. Both the mass-point 2SLS sandwich and the
        # continuous (Phase 2a) bias_corrected_local_linear path consume the
        # extracted cluster IDs.
        d_arr, dy_arr, _, _ = _aggregate_first_difference(
            data,
            outcome_col,
            dose_col,
            time_col,
            unit_col,
            t_pre,
            t_post,
            None,
        )

        # ---- Extensive-margin / positive-untreated-mass warning (TODO L74) ----
        # d_arr is the per-unit post-period dose D_{g,2} (D_{g,1}=0, so dD = D_2);
        # exactly-zero entries are genuinely untreated units. The `== 0.0` test
        # mirrors the qug_test `d == 0` convention (had_pretests.py). Fraction-only
        # (no absolute floor); fires at/above the library-convention cutoff. See the
        # _HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC definition for the paper rationale.
        # This runs on the overall path only: the event-study dispatch returns
        # above, and the event-study path requires never-treated units (App. B.2).
        n_zero = int((d_arr == 0.0).sum())
        if n_zero and n_zero / d_arr.shape[0] >= _HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC:
            frac_zero = n_zero / d_arr.shape[0]
            warnings.warn(
                f"{n_zero}/{d_arr.shape[0]} units ({frac_zero:.0%}) have exactly-"
                f"zero post-period dose. HeterogeneousAdoptionDiD targets a "
                f"Weighted Average Slope under the assumption that all units "
                f"receive a positive, heterogeneous dose with no genuine control "
                f"group (de Chaisemartin et al. 2026, Section 2 / Assumption 3). A "
                f"substantial untreated mass suggests a genuine extensive margin, "
                f"where a standard DiD using the untreated units as controls may be "
                f"more appropriate. (The paper retains small untreated shares — "
                f"e.g. 12/2954 in Garrett et al. — with nominal coverage; this "
                f"warning fires only at/above a "
                f"{_HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC:.0%} library-convention "
                f"cutoff.)",
                UserWarning,
                stacklevel=2,
            )

        # Resolve survey/weights into per-unit weights + optional
        # ResolvedSurveyDesign (for PSU/strata/FPC composition).
        # - `survey_design=SurveyDesign(weights="col", ...)` → resolve the design
        #   and produce a unit-level analogue for per-unit IF composition.
        # - neither → unweighted path.
        weights_unit: Optional[np.ndarray] = None
        raw_weights_unit: Optional[np.ndarray] = None
        resolved_survey_unit: Any = None  # ResolvedSurveyDesign (G,) when weighted
        if survey is not None:
            if not hasattr(survey, "weights"):
                raise TypeError(
                    f"survey_design= must be a SurveyDesign-like object with a "
                    f".weights attribute; got {type(survey).__name__}. "
                    f"Construct a SurveyDesign via diff_diff.survey."
                )
            if getattr(survey, "weights", None) is None:
                raise NotImplementedError(
                    "survey_design= without weights is not yet supported. Pass "
                    "survey_design=SurveyDesign(weights='<col>', ...) with a "
                    "per-row weight column."
                )
            # HAD's weighted local-linear treats ``weights`` as sampling
            # (probability) weights: the kernel-composition formula
            # ``W_combined = k((D-d̲)/h) · w`` is the inverse-probability
            # weighting convention. Frequency weights (``fweight``)
            # would imply replicating observations, and analytic weights
            # (``aweight``, inverse-variance) would imply a different
            # inferential target. Reject those up front rather than
            # silently reinterpreting.
            weight_type = getattr(survey, "weight_type", "pweight")
            if weight_type != "pweight":
                raise NotImplementedError(
                    f"survey_design=SurveyDesign(weight_type={weight_type!r}) is "
                    f"not supported on HeterogeneousAdoptionDiD's "
                    f"continuous path. Only ``weight_type='pweight'`` "
                    f"(sampling / inverse-probability weights) is "
                    f"implemented in Phase 4.5 A. Frequency weights "
                    f"(fweight) and analytic weights (aweight) would "
                    f"imply different estimands and are not yet derived."
                )
            # Capture the RAW pre-normalization weight column before
            # ``resolve()`` rescales pweights/aweights to mean=1. Needed
            # below so ``compute_survey_metadata`` receives raw weights
            # (per its contract — ``sum_weights`` / ``weight_range`` are
            # raw-scale quantities; passing normalized weights would make
            # the metadata disagree with ``compute_survey_metadata``'s
            # raw-scale contract in ``survey.py``).
            weights_col_name = survey.weights  # known non-None from guard above
            if weights_col_name not in data.columns:
                raise ValueError(f"survey.weights column {weights_col_name!r} not found in data.")
            raw_weights_row = np.asarray(data[weights_col_name].values, dtype=np.float64)
            raw_weights_unit = _aggregate_unit_weights(data, raw_weights_row, unit_col)
            # Resolve the SurveyDesign against the long-panel data. This
            # validates column names, applies pweight/aweight normalization
            # to mean=1, and extracts numpy arrays for all design columns.
            resolved_survey_row = survey.resolve(data)
            # Collapse design columns to unit-level (constant-within-unit
            # invariant) so the IF-based variance composition operates at
            # the G-unit scale that matches the local-linear fit.
            resolved_survey_unit = _aggregate_unit_resolved_survey(
                data, resolved_survey_row, unit_col
            )
            weights_unit = np.asarray(resolved_survey_unit.weights, dtype=np.float64)

        # Zero-weight units (e.g. SurveyDesign.subpopulation() output, or
        # a user-supplied pweight column with excluded observations) must
        # not drive design resolution — ``_detect_design`` / ``d_lower``
        # / mass-point threshold / cohort counts run on the POSITIVE-
        # weight subset. But the survey VARIANCE and ``SurveyMetadata``
        # preserve the FULL ResolvedSurveyDesign (zero-weight PSUs /
        # strata kept in the design with zero in-domain mass) — that is
        # the standard subpopulation / domain-estimation convention in
        # ``diff_diff.survey``: keep the sampling frame, zero the
        # contributions. The weighted kernel in ``lprobust`` drops
        # zero-weight observations via its ``w > 0`` selector, and
        # ``bias_corrected_local_linear`` zero-pads the returned IF back
        # to the full unit ordering so the survey composition at the
        # HAD level sees IF=0 for zero-weight units on the FULL design.
        # (CI review PR #359 round 5 P0 + round 6 P1 cascade.)
        d_arr_full = d_arr  # unfiltered (G units); passed to _fit_continuous
        dy_arr_full = dy_arr
        weights_unit_full = weights_unit  # may contain zeros; used for FIT
        resolved_survey_unit_full = resolved_survey_unit  # full design for VARIANCE
        raw_weights_unit_full = raw_weights_unit  # full for SurveyMetadata
        if weights_unit is not None:
            positive_mask = weights_unit > 0.0
            if not bool(positive_mask.all()):
                n_dropped = int((~positive_mask).sum())
                warnings.warn(
                    f"HAD continuous path: {n_dropped} unit(s) have "
                    f"weight == 0 and are excluded from design resolution "
                    f"(auto-detect design, d_lower, mass-point threshold, "
                    f"cohort counts). They are RETAINED in the survey "
                    f"design for variance + SurveyMetadata (subpopulation "
                    f"convention: zero-weight contributions but full "
                    f"sampling frame), and their IF is 0 on the full "
                    f"design.",
                    UserWarning,
                    stacklevel=2,
                )
                # Filter arrays used for DESIGN-RESOLUTION ONLY.
                d_arr = d_arr[positive_mask]
                dy_arr = dy_arr[positive_mask]
                weights_unit = weights_unit[positive_mask]

        n_obs = int(d_arr.shape[0])
        if n_obs < 3:
            raise ValueError(
                f"HAD requires at least 3 units for inference; got "
                f"n_obs={n_obs} after aggregation."
            )

        # ---- Resolve design ----
        if design_arg == "auto":
            resolved_design = _detect_design(d_arr)
        else:
            resolved_design = design_arg

        # ---- Extract cluster IDs (mass-point + continuous paths) ----
        # Reject the continuous cluster + ``survey_design=`` composition BEFORE
        # extracting/validating the cluster column, so the unsupported-
        # composition error is predictable even when the cluster column itself
        # is malformed (a nonexistent column would otherwise raise ``ValueError``
        # first). The Binder-TSL survey path composes variance from the per-unit
        # IF and would silently override the cluster-robust local-linear SE.
        if (
            cluster_arg is not None
            and resolved_survey_unit_full is not None
            and resolved_design in ("continuous_at_zero", "continuous_near_d_lower")
        ):
            raise NotImplementedError(
                f"cluster={cluster_arg!r} + survey_design= on the "
                f"'{resolved_design}' path is not yet supported: the survey path "
                f"composes Binder-TSL variance via compute_survey_if_variance and "
                f"would silently override the cluster-robust local-linear SE. Pass "
                f"cluster= alone (unweighted cluster-robust), or "
                f"survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>') to cluster through "
                f"the survey (Binder-TSL) path."
            )
        # Re-run the aggregation with cluster_col so validation (missing column /
        # NaN / within-unit variance) fires only when cluster is actually going to
        # be used. The per-unit cluster array is threaded into the 2SLS sandwich on
        # the mass-point path and into ``bias_corrected_local_linear`` on the
        # continuous paths (Phase 2a).
        cluster_arr: Optional[np.ndarray] = None
        if cluster_arg is not None:
            _, _, cluster_arr, _ = _aggregate_first_difference(
                data,
                outcome_col,
                dose_col,
                time_col,
                unit_col,
                t_pre,
                t_post,
                cluster_arg,
            )

        # ---- Resolve d_lower ----
        if resolved_design == "continuous_at_zero":
            # Design 1' regime (paper Section 3.2) is defined at d_lower = 0.
            # Reject explicit nonzero d_lower overrides front-door rather
            # than silently coerce to zero. Tolerance family matches the
            # Design 1 d_lower guards below.
            if d_lower_arg is not None:
                scale = max(1.0, float(np.max(np.abs(d_arr))))
                if abs(float(d_lower_arg)) > 1e-12 * scale:
                    raise ValueError(
                        f"design='continuous_at_zero' (Design 1') requires "
                        f"d_lower == 0 within float tolerance (paper Section "
                        f"3.2 Design 1' regime). Got d_lower="
                        f"{float(d_lower_arg)!r}. For d_lower > 0 use "
                        f"design='continuous_near_d_lower' (local-linear "
                        f"boundary-limit estimator) or design='mass_point' "
                        f"(2SLS) as appropriate, or design='auto' which "
                        f"auto-detects the correct path from the dose "
                        f"distribution."
                    )
            d_lower_val = 0.0
        elif d_lower_arg is None:
            d_lower_val = float(d_arr.min())
        else:
            d_lower_val = float(d_lower_arg)

        # ---- Regime partition: d_lower > 0 for Design 1 paths ----
        # Paper Section 3.2 partitions HAD into d_lower = 0 (Design 1',
        # continuous_at_zero) and d_lower > 0 (Design 1, continuous_near
        # _d_lower or mass_point). The auto-detect rule already enforces
        # this partition; explicit overrides must respect it too, otherwise
        # `design="mass_point", d_lower=0` returns a finite but
        # paper-incompatible 2SLS result and `design="continuous_near_d_lower"`
        # with d_lower=0 reduces to Design 1' algebra while mislabeling the
        # estimand as `WAS_d_lower` and emitting the wrong Assumption 5/6
        # warning. Use the same float-tolerance family as _detect_design's
        # d.min()==0 tie-break.
        if resolved_design in ("mass_point", "continuous_near_d_lower"):
            scale = max(1.0, float(np.max(np.abs(d_arr))))
            if abs(d_lower_val) <= 1e-12 * scale:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower > 0 (paper "
                    f"Section 3.2 reserves the d_lower=0 regime for Design 1' "
                    f"/ `continuous_at_zero`). Got d_lower={d_lower_val!r}. "
                    f"Use design='continuous_at_zero' (explicit) or "
                    f"design='auto' (auto-detect) for samples with support "
                    f"infimum at zero."
                )

        # ---- Original-scale modal-fraction / regime check ----
        # Paper Section 3.2 splits Design 1 into two REGIME-SPECIFIC
        # estimators by modal-min fraction:
        #   - continuous_near_d_lower: modal fraction at d.min() <= 2%
        #     (local-linear boundary-limit estimator).
        #   - mass_point: modal fraction at d.min() > 2%
        #     (Wald-IV / 2SLS identified by the instrument 1{D_2 > d_lower}).
        # The auto-detect rule already enforces this; explicit overrides
        # must too, otherwise the wrong estimand is returned silently.
        # Both guards are symmetric around the 2% threshold used in
        # _detect_design() and Phase 1c's _validate_had_inputs().
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            d_min_orig = float(d_arr.min())
            if d_min_orig > 0:
                eps_mp = 1e-12 * max(1.0, abs(d_min_orig))
                at_d_min_mask_orig = np.abs(d_arr - d_min_orig) <= eps_mp
                modal_fraction_orig = float(np.mean(at_d_min_mask_orig))
                if (
                    resolved_design == "continuous_near_d_lower"
                    and modal_fraction_orig > _MASS_POINT_THRESHOLD
                ):
                    raise ValueError(
                        f"design='continuous_near_d_lower' cannot be used on a "
                        f"mass-point sample (modal fraction {modal_fraction_orig:.4f} "
                        f"at d.min()={d_min_orig!r} exceeds the "
                        f"{_MASS_POINT_THRESHOLD:.2f} threshold from paper Section "
                        f"3.2.4). Use design='mass_point' (Wald-IV / 2SLS) or "
                        f"design='auto' which will auto-detect. Forcing the "
                        f"continuous path on a mass-point sample would produce "
                        f"the wrong estimand."
                    )
                if resolved_design == "mass_point" and modal_fraction_orig <= _MASS_POINT_THRESHOLD:
                    raise ValueError(
                        f"design='mass_point' requires a modal mass at d.min() "
                        f"exceeding the {_MASS_POINT_THRESHOLD:.2f} threshold "
                        f"(paper Section 3.2.4). Got modal fraction "
                        f"{modal_fraction_orig:.4f} at d.min()={d_min_orig!r}. "
                        f"For continuous-near-d_lower samples use "
                        f"design='continuous_near_d_lower' (local-linear "
                        f"boundary-limit estimator) or design='auto' which "
                        f"will auto-detect. Forcing 2SLS on a continuous "
                        f"sample identifies the exact-d.min() cell rather "
                        f"than the paper's boundary-limit estimand."
                    )

        # ---- d_lower contract for Design 1 paths ----
        # Paper Sections 3.2.2-3.2.4 define the Design 1 estimators at
        # d_lower = support infimum of D_{g,2}. For the mass-point path,
        # the instrument Z = 1{D_{g,2} > d_lower} requires d_lower to be
        # the lower-support mass point. For the continuous-near-d_lower
        # path, evaluating the local-linear fit after the regressor
        # shift (D - d_lower) at boundary=0 only makes sense when the
        # shift anchors to the realized sample minimum; otherwise the
        # boundary evaluation is off-support (no observations near zero
        # on the shifted scale) and Phase 1c's 5% plausibility heuristic
        # may fail to catch the mismatch. We enforce d_lower == d.min()
        # within float tolerance on both Design 1 paths; mismatched
        # overrides raise with a clear pointer to the unsupported
        # estimand.
        if resolved_design in ("mass_point", "continuous_near_d_lower") and d_lower_arg is not None:
            d_min = float(d_arr.min())
            tol = 1e-12 * max(1.0, abs(d_min))
            if abs(d_lower_val - d_min) > tol:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower to equal "
                    f"the support infimum float(d.min())={d_min!r}; got "
                    f"d_lower={d_lower_val!r}. The paper's Design 1 "
                    f"estimators (Sections 3.2.2-3.2.4) identify at the "
                    f"lower-support boundary, not at an arbitrary "
                    f"threshold. Pass d_lower=None to auto-resolve, or "
                    f"d_lower=float(d.min()) explicitly. Non-support-"
                    f"infimum thresholds identify a different (LATE-like "
                    f"for mass_point, off-support for continuous_near_"
                    f"d_lower) estimand that is out of Phase 2a scope."
                )
            # Snap tolerance-accepted overrides back to the exact support
            # infimum. Float-rounding drift matters downstream: on the
            # mass-point path, `Z = d > d_lower` with d_lower = d.min() - ε
            # puts the mass-point units into Z=1 (control group empties);
            # on the continuous-near-d_lower path, d_lower = d.min() + ε
            # makes `d - d_lower` negative and trips Phase 1c's
            # _validate_had_inputs negative-dose guard. Snapping preserves
            # the "within tolerance" contract while keeping downstream
            # algebra exact.
            d_lower_val = d_min

        # ---- Compute cohort counts ----
        if resolved_design == "mass_point":
            eps = 1e-12 * max(1.0, abs(d_lower_val))
            at_d_min_mask = np.abs(d_arr - d_lower_val) <= eps
            above_mask = d_arr > d_lower_val
            n_mass_point: Optional[int] = int(at_d_min_mask.sum())
            n_above_d_lower: Optional[int] = int(above_mask.sum())
            n_treated = n_above_d_lower
            n_control = int(at_d_min_mask.sum())
        else:
            n_mass_point = None
            n_above_d_lower = None
            above_mask = d_arr > d_lower_val
            n_treated = int(above_mask.sum())
            n_control = n_obs - n_treated

        dose_mean = float(d_arr.mean())

        # ---- Assumption 5/6 warning on Design 1 paths ----
        # Paper Sections 3.2.2-3.2.4: when d_lower > 0 (Design 1 family),
        # point identification of WAS_{d_lower} requires Assumption 6 in
        # addition to parallel trends (Assumption 1-3); Assumption 5 gives
        # only sign identification. These extra assumptions are NOT
        # testable via pre-trends. Surface this to the user front-door so
        # results are not silently interpreted as full point identification.
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            warnings.warn(
                f"design={resolved_design!r} (Design 1, d_lower > 0) requires "
                f"Assumption 6 from de Chaisemartin et al. (2026) for point "
                f"identification of WAS_{{d_lower}}, or Assumption 5 for "
                f"sign identification only. Neither is testable via "
                f"pre-trends. Confirm the extra assumption is defensible "
                f"for your setting before interpreting the returned "
                f"point estimate as the WAS.",
                UserWarning,
                stacklevel=2,
            )

        # ---- Dispatch ----
        if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
            # Warn when the user set a mass-point-only knob that's ignored
            # on the continuous path. (Emitted per fit call; this is not
            # suppressed after the first call.)
            if vcov_type_arg is not None:
                warnings.warn(
                    f"vcov_type={vcov_type_arg!r} is ignored on the "
                    f"'{resolved_design}' path (the continuous designs "
                    f"use the CCT-2014 robust SE from Phase 1c). "
                    f"vcov_type applies only to design='mass_point'.",
                    UserWarning,
                    stacklevel=2,
                )
            if robust_arg:
                warnings.warn(
                    f"robust=True is ignored on the '{resolved_design}' "
                    f"path (the continuous designs use the CCT-2014 "
                    f"robust SE from Phase 1c unconditionally; the "
                    f"robust flag is a mass-point-only backward-compat "
                    f"alias for vcov_type).",
                    UserWarning,
                    stacklevel=2,
                )
            # (cluster= + survey_design= was rejected up front, before cluster
            # extraction — see the guard in the design-resolution block above.)
            # Fit on FULL (unfiltered) arrays so the IF aligns with the
            # full survey design. bias_corrected_local_linear drops
            # zero-weight rows internally for its validation + selector +
            # fit, then zero-pads the IF back to full length, and filters the
            # cluster IDs by the same positive-weight mask. Survey
            # composition below runs on the full design, preserving
            # domain-estimation semantics.
            att, se, bc_fit, bw_diag = self._fit_continuous(
                d_arr_full,
                dy_arr_full,
                resolved_design,
                d_lower_val,
                weights_arr=weights_unit_full,
                resolved_survey_unit=resolved_survey_unit_full,
                cluster_arr=cluster_arr,
            )
            inference_method = "analytical_nonparametric"
            # Cluster-robust (Phase 2a): the CCT nonparametric SE from
            # bias_corrected_local_linear is cluster-robust when cluster= is
            # threaded. vcov_type="cr1" + inference_method="analytical_
            # nonparametric" together identify the clustered CCT variance
            # (distinct from the mass-point 2SLS CR1 sandwich).
            vcov_label: Optional[str] = "cr1" if cluster_arg is not None else None
            cluster_label: Optional[str] = cluster_arg if cluster_arg is not None else None
        elif resolved_design == "mass_point":
            # survey_design= + cluster= is a silent-mismatch case (the
            # Binder-TSL override would overwrite the CR1 sandwich while
            # result metadata still advertises vcov_type='cr1'), so it is
            # rejected below. The unweighted cluster= path returns the CR1
            # sandwich from _fit_mass_point_2sls directly and matches
            # estimatr::iv_robust (se_type="stata") bit-exactly — see
            # tests/test_estimatr_iv_robust_parity.py::TestEstimatrIVRobustCR1Parity.
            if cluster_arg is not None and resolved_survey_unit_full is not None:
                raise NotImplementedError(
                    f"cluster={cluster_arg!r} + survey_design= on "
                    f"design='mass_point' is not yet supported: the "
                    f"survey path composes Binder-TSL variance via "
                    f"compute_survey_if_variance and would silently "
                    f"override the CR1 cluster-robust sandwich while "
                    f"result metadata still advertises "
                    f"vcov_type='cr1'. Pass cluster= alone "
                    f"(unweighted CR1), or "
                    f"survey_design=SurveyDesign(weights='<col>', psu='<cluster>') "
                    f"to cluster through the survey (Binder-TSL) path. Combined "
                    f"cluster-robust + survey inference is deferred to a "
                    f"follow-up PR."
                )
            # Resolve the EFFECTIVE vcov family first (vcov_type_arg,
            # with default-mapping from robust= when unset). Reject the
            # classical-on-IF-consumption combination against the
            # resolved value, NOT the raw kwarg, so the default
            # `vcov_type=None, robust=False` case (which maps to
            # classical) hits the guard too (review R5 P1 — previous
            # fix only fired on explicit vcov_type='classical').
            if vcov_type_arg is None:
                # Backward-compat: robust=True -> hc1, robust=False -> classical.
                vcov_requested = "hc1" if robust_arg else "classical"
            else:
                vcov_requested = vcov_type_arg.lower()
            # Review R3 P1 / R5 P1: the weighted mass-point path returns
            # an HC1-scaled influence function (IF scale convention
            # locks compute_survey_if_variance(psi, trivial) ≈ V_HC1[1,1]
            # via the sqrt((n-1)/(n-k)) factor in _fit_mass_point_2sls).
            # On the survey path the analytical SE is ALWAYS overwritten
            # with that HC1-scale Binder-TSL composition, so effective
            # classical + survey_design= would silently report an HC1-target
            # SE under a classical label. Reject until a
            # classical-aligned IF is derived.
            if vcov_requested == "classical" and resolved_survey_unit_full is not None:
                raise NotImplementedError(
                    "vcov_type='classical' (resolved — either explicit or "
                    "from the default robust=False mapping) + survey_design= on "
                    "design='mass_point' is not yet supported: the "
                    "survey path composes Binder-TSL variance via the "
                    "HC1-scale influence function, which targets V_HC1 "
                    "rather than the classical sandwich "
                    "V_cl = σ² · (Z'WX)^{-1}(Z'W²Z)(X'WZ)^{-1}. Use "
                    "vcov_type='hc1' (or leave vcov_type unset with "
                    "robust=True) on the weighted path; a classical-"
                    "aligned IF derivation is queued for a follow-up PR."
                )
            # Phase 4.5 B: accept weights_unit (None on unweighted fits).
            # return_influence=True only on the survey path because
            # Binder-TSL composition consumes the IF; the unweighted path
            # uses the analytical sandwich SE directly. ``psi_mp`` is per-unit IF on β̂-scale or None.
            # Fit on FULL (unfiltered) arrays so the IF aligns with the
            # full survey design (subpopulation convention: zero-weight
            # units contribute 0 to all sums; IF zero-padded back to full
            # length). Under unweighted fits d_arr_full == d_arr and
            # weights_unit_full is None, so behavior is unchanged.
            att, se, psi_mp = _fit_mass_point_2sls(
                d_arr_full,
                dy_arr_full,
                d_lower_val,
                cluster_arr,
                vcov_requested,
                weights=weights_unit_full,
                return_influence=resolved_survey_unit_full is not None,
            )
            # Survey path: compose Binder-TSL variance from per-unit IF
            # (replaces analytical sandwich SE). Mirrors continuous-path
            # branch at lines 3082-3099. Under trivial resolved (single
            # stratum, no PSU/FPC, uniform w), this reduces to analytical
            # HC1 within the IF-scale-convention tolerance (atol=1e-10).
            if resolved_survey_unit_full is not None and psi_mp is not None:
                from diff_diff.survey import compute_survey_if_variance

                v_survey = compute_survey_if_variance(psi_mp, resolved_survey_unit_full)
                if np.isfinite(v_survey) and v_survey > 0.0:
                    se = float(np.sqrt(v_survey))
                else:
                    se = float("nan")
            bc_fit = None
            bw_diag = None
            inference_method = "analytical_2sls"
            # Store the EFFECTIVE variance family so downstream consumers
            # (to_dict, to_dataframe, summary) see the actual SE type that
            # was computed. When cluster is supplied, _fit_mass_point_2sls
            # unconditionally computes CR1 regardless of vcov_requested
            # (e.g. classical+cluster -> CR1), so we surface that here.
            vcov_label = "cr1" if cluster_arg is not None else vcov_requested
            cluster_label = cluster_arg if cluster_arg is not None else None
        else:
            raise ValueError(f"Internal error: unhandled design={resolved_design!r}.")

        # ---- Route all inference fields through safe_inference ----
        # Survey path: use t-distribution with ``df_survey = n_psu -
        # n_strata`` (or replicate-QR rank − 1) so small-PSU designs
        # don't get Normal-theory inference that overstates precision.
        # Non-survey path (unweighted): use the existing
        # Normal-theory default.
        df_infer: Optional[int] = None
        if resolved_survey_unit is not None:
            df_infer = resolved_survey_unit.df_survey
        t_stat, p_value, conf_int = safe_inference(att, se, alpha=float(self.alpha), df=df_infer)

        # Build survey metadata (repo-standard SurveyMetadata from
        # diff_diff.survey.compute_survey_metadata) when weights/survey
        # were supplied, so downstream report/diagnostic consumers can
        # read attributes uniformly. HAD-specific extras (variance-
        # formula label, effective-denominator value) live on dedicated
        # result fields rather than being folded into the survey dict.
        survey_metadata: Optional[SurveyMetadata] = None
        variance_formula_label: Optional[str] = None
        effective_dose_mean_value: Optional[float] = None
        if weights_unit_full is not None:
            if resolved_survey_unit_full is not None:
                # survey path: build metadata from the FULL
                # ResolvedSurveyDesign (pre-zero-weight-filter), so
                # ``n_strata`` / ``n_psu`` / ``df_survey`` / weight sums
                # reflect the sampling frame, not the in-domain subset.
                # Pass the RAW pre-normalization per-unit weights
                # (captured before survey.resolve() rescaled pweights/
                # aweights to mean=1) so ``sum_weights`` / ``weight_range``
                # reflect the user-supplied scale — matching both the
                # ``compute_survey_metadata``'s contract.
                assert raw_weights_unit_full is not None  # set in survey branch
                survey_metadata = compute_survey_metadata(
                    resolved_survey_unit_full, raw_weights_unit_full
                )
                # Design-specific label — continuous uses bias-corrected CCT
                # IF, mass-point uses 2SLS IF; both route through Binder TSL.
                variance_formula_label = (
                    "survey_binder_tsl_2sls"
                    if resolved_design == "mass_point"
                    else "survey_binder_tsl"
                )
            # Expose the effective weighted denominator used by the
            # beta-scale rescaling. Continuous paths use a weighted sample
            # mean of d (or d − d_lower). Mass-point uses the weighted
            # Wald-IV dose-gap (the denominator of β̂ =
            # dy_gap_w / dose_gap_w), computed from the FULL arrays
            # (zero-weight units contribute 0 to both subgroup sums).
            if resolved_design == "continuous_at_zero":
                effective_dose_mean_value = float(np.average(d_arr_full, weights=weights_unit_full))
            elif resolved_design == "continuous_near_d_lower":
                effective_dose_mean_value = float(
                    np.average(d_arr_full - d_lower_val, weights=weights_unit_full)
                )
            elif resolved_design == "mass_point":
                # Weighted Wald-IV dose gap: mean(d | Z=1, w) - mean(d | Z=0, w).
                # Surface this as the "effective denominator" so downstream
                # reporting displays the β̂-scale denominator consistently
                # across designs. Guard against empty subgroups (handled
                # upstream by _fit_mass_point_2sls returning NaN, so we
                # only reach here on a successful fit with positive mass
                # on both sides).
                Z_mp = (d_arr_full > d_lower_val).astype(np.float64)
                pos_mp = weights_unit_full > 0
                Z1_mp = (Z_mp == 1) & pos_mp
                Z0_mp = (Z_mp == 0) & pos_mp
                w_Z1_mp = float(weights_unit_full[Z1_mp].sum())
                w_Z0_mp = float(weights_unit_full[Z0_mp].sum())
                if w_Z1_mp > 0.0 and w_Z0_mp > 0.0:
                    effective_dose_mean_value = float(
                        (weights_unit_full[Z1_mp] * d_arr_full[Z1_mp]).sum() / w_Z1_mp
                        - (weights_unit_full[Z0_mp] * d_arr_full[Z0_mp]).sum() / w_Z0_mp
                    )

        return HeterogeneousAdoptionDiDResults(
            att=float(att),
            se=float(se),
            t_stat=float(t_stat),
            p_value=float(p_value),
            conf_int=(float(conf_int[0]), float(conf_int[1])),
            alpha=float(self.alpha),
            design=resolved_design,
            target_parameter=_TARGET_PARAMETER[resolved_design],
            d_lower=d_lower_val,
            dose_mean=dose_mean,
            n_obs=n_obs,
            n_treated=n_treated,
            n_control=n_control,
            n_mass_point=n_mass_point,
            n_above_d_lower=n_above_d_lower,
            inference_method=inference_method,
            vcov_type=vcov_label,
            cluster_name=cluster_label,
            survey_metadata=survey_metadata,
            bandwidth_diagnostics=bw_diag,
            bias_corrected_fit=bc_fit,
            variance_formula=variance_formula_label,
            effective_dose_mean=effective_dose_mean_value,
        )

    # ------------------------------------------------------------------
    # Continuous-design dispatch (Design 1' + Design 1 continuous-near-d_lower)
    # ------------------------------------------------------------------

    def _fit_continuous(
        self,
        d_arr: np.ndarray,
        dy_arr: np.ndarray,
        resolved_design: str,
        d_lower_val: float,
        weights_arr: Optional[np.ndarray] = None,
        resolved_survey_unit: Any = None,  # ResolvedSurveyDesign (G,) or None
        force_return_influence: bool = False,
        cluster_arr: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Optional[BiasCorrectedFit], Optional[BandwidthResult]]:
        """Fit Phase 1c ``bias_corrected_local_linear`` and form the WAS estimate.

        Implements de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau
        (2026) continuous-design estimators:

        - Design 1' (``continuous_at_zero``), paper Theorem 1 /
          Equation 3 (identification); Equation 7 (sample estimator):

              beta = (E[Delta Y] - lim_{d v 0} E[Delta Y | D_2 <= d]) / E[D_2]

          Regressor passed to the local-linear boundary fit is
          ``d_arr``; the boundary is ``0``.

        - Design 1 (``continuous_near_d_lower``), paper Theorem 3 /
          Equation 11 (``WAS_{d_lower}`` under Assumption 6; note
          Theorem 4 is the QUG null test, not this estimand):

              beta = (E[Delta Y] - lim_{d v d_lower} E[Delta Y | D_2 <= d])
                     / E[D_2 - d_lower]

          Regressor passed to the local-linear fit is
          ``d_arr - d_lower`` so the boundary evaluation point is ``0``
          on the shifted scale; the numerator and denominator are
          computed on the original scale.

        The bias-corrected boundary estimate ``tau_bc`` from
        :func:`bias_corrected_local_linear` corresponds to the limit
        ``lim_{d v d_lower} E[Delta Y | D_2 <= d]``. The beta-scale
        estimator and standard error are then:

            att = (mean(dy) - tau_bc) / den
            se  = se_robust / |den|

        where ``den`` is the expectation in the denominator (``D_bar``
        for Design 1', ``mean(D_2 - d_lower)`` for Design 1). The
        confidence interval is ``att +/- z * se`` computed in
        :func:`diff_diff.utils.safe_inference`; the endpoints reverse
        relative to the boundary-limit CI because the numerator
        transformation is ``ΔȲ - tau_bc``.
        """
        # Weighted population moments (Phase 4.5). Under uniform weights,
        # `np.average(.., weights=np.ones(G)) == a.mean()` bit-exactly, so
        # the unweighted path is bit-identical when `weights_arr is None`.
        if resolved_design == "continuous_at_zero":
            d_reg = d_arr
            boundary = 0.0
            if weights_arr is None:
                den = float(d_arr.mean())
            else:
                den = float(np.average(d_arr, weights=weights_arr))
        elif resolved_design == "continuous_near_d_lower":
            d_reg = d_arr - d_lower_val
            boundary = 0.0
            if weights_arr is None:
                den = float(d_reg.mean())
            else:
                den = float(np.average(d_reg, weights=weights_arr))
        else:
            raise ValueError(
                f"_fit_continuous called with non-continuous " f"design={resolved_design!r}"
            )

        # Phase 1b/1c's bandwidth selector can hit degenerate ratios
        # (divide-by-zero in the bias estimator) when the outcome is
        # exactly constant or exactly linear in d (zero residuals).
        # The upstream fix lives in ``_nprobust_port.lpbwselect_mse_dpi``
        # and is deliberately out of Phase 2a scope (per plan "No changes
        # to local_linear.py / _nprobust_port.py"). Catch the symptomatic
        # exceptions here and surface a NaN result via ``safe_inference``
        # so users get a well-shaped ``HeterogeneousAdoptionDiDResults``
        # rather than a traceback.
        try:
            bc_fit = bias_corrected_local_linear(
                d=d_reg,
                y=dy_arr,
                boundary=boundary,
                kernel=self.kernel,
                alpha=float(self.alpha),
                weights=weights_arr,
                # Request per-unit influence function ONLY when we need it
                # for survey-composed variance (compute_survey_if_variance).
                # Unconditional IF computation would add a small O(G) cost
                # to every fit; gate it on the survey path.
                return_influence=(resolved_survey_unit is not None or force_return_influence),
                # Cluster-robust CCT variance (Phase 2a): when ``cluster_arr``
                # is provided, ``bias_corrected_local_linear`` forwards it to
                # the bandwidth selector and the ``lprobust`` variance so
                # ``se_robust`` is the cluster-robust nonparametric SE. It also
                # filters ``cluster_arr`` by the same positive-weight mask it
                # uses to drop zero-weight rows, so a full-length array aligns.
                # A bare ``cluster=`` gives unweighted cluster-robust
                # inference; the ``cluster=`` + ``survey_design=`` composition is rejected
                # up-front in ``fit()`` (Binder-TSL would override se_robust).
                cluster=cluster_arr,
            )
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            return float("nan"), float("nan"), None, None

        # Guard against degenerate denominators: if all units are at
        # d_lower (continuous_near_d_lower) or if D_bar rounds to zero
        # (continuous_at_zero with a vanishing sample mean), the beta-
        # scale estimator is undefined. Fall through to NaN via
        # safe_inference in fit().
        if abs(den) < 1e-12:
            att = float("nan")
            se = float("nan")
        else:
            if weights_arr is None:
                dy_mean = float(dy_arr.mean())
            else:
                dy_mean = float(np.average(dy_arr, weights=weights_arr))
            tau_bc = float(bc_fit.estimate_bias_corrected)
            att = (dy_mean - tau_bc) / den
            if resolved_survey_unit is not None and bc_fit.influence_function is not None:
                # Survey-composed variance via Binder (1983) Taylor
                # linearization. Paper Equation 8 treats D_bar as fixed
                # (it's a sqrt(G)-rate quantity, faster than the
                # G^{2/5}-rate nonparametric estimator), so the leading-
                # order variance is V(mu_hat) / D_bar^2. Under survey
                # design, V(mu_hat) = compute_survey_if_variance(psi,
                # resolved) with PSU aggregation + strata sum + FPC.
                from diff_diff.survey import compute_survey_if_variance

                v_survey = compute_survey_if_variance(
                    bc_fit.influence_function, resolved_survey_unit
                )
                if np.isfinite(v_survey) and v_survey > 0.0:
                    se = float(np.sqrt(v_survey)) / abs(den)
                else:
                    se = float("nan")
            else:
                se = float(bc_fit.se_robust) / abs(den)

        # Note: cluster-robust inference with fewer than two clusters in the
        # active kernel window is NaN'd inside bias_corrected_local_linear /
        # _nprobust_port.lprobust (``se_robust`` becomes NaN), so ``se`` here is
        # already NaN in that degenerate case and safe_inference NaNs the
        # downstream t-stat / p-value / CI (att stays the raw point estimate,
        # mirroring the mass-point CR1 single-cluster contract).
        return att, se, bc_fit, bc_fit.bandwidth_diagnostics

    # ------------------------------------------------------------------
    # Event-study dispatch (Phase 2b, paper Appendix B.2)
    # ------------------------------------------------------------------

    def _fit_event_study(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        dose_col: str,
        time_col: str,
        unit_col: str,
        first_treat_col: Optional[str],
        survey: Any = None,
        cband: bool = True,
        trends_lin: bool = False,
    ) -> HeterogeneousAdoptionDiDEventStudyResults:
        """Multi-period event-study fit (paper Appendix B.2).

        Delegates to the multi-period panel validator (including staggered
        last-cohort auto-filter), aggregates per-horizon first differences
        against a common anchor ``Y_{g, F-1}``, resolves the design ONCE
        on the period-F dose distribution, and then fits the chosen design
        path independently on each event-time horizon's first differences.

        Per-horizon variance regimes (matches the static-path contract):

        - **``survey_design=``**: Binder (1983) Taylor-series linearization
          via :func:`compute_survey_if_variance` on the per-unit
          β̂-scale influence function (continuous + mass-point both
          route through the same helper). Inference is
          t-distribution with ``df_survey``.
        - **Unweighted**: per-horizon independent analytical sandwich
          (continuous: CCT-2014 robust ``bc_fit.se_robust / |den|``;
          mass-point: 2SLS sandwich via ``_fit_mass_point_2sls``).
          Inference is Normal (``df=None``).

        The simultaneous confidence band (when ``cband=True``) is
        constructed by a multiplier bootstrap over the stacked per-horizon
        β̂-scale IF matrix via :func:`_sup_t_multiplier_bootstrap`. On the
        weighted/survey path it draws PSU-level multipliers; when
        ``cluster=`` is set it takes the clustered branch (cluster-level
        multipliers, raw cluster sandwich; fires even unweighted). An
        unweighted, unclustered event-study skips the bootstrap (pre-Phase
        4.5 B numerical output preserved).
        """
        # ---- Resolve effective fit-time state (local vars only,
        # feedback_fit_does_not_mutate_config). ----
        design_arg = self.design
        d_lower_arg = self.d_lower
        vcov_type_arg = self.vcov_type
        robust_arg = self.robust
        cluster_arg = self.cluster
        n_bootstrap_eff = int(self.n_bootstrap)
        seed_eff = None if self.seed is None else int(self.seed)

        # ---- survey_design contract validation (front-door).
        # Mirrors the static-path gates at fit(): survey_design= without a
        # weights column is unsupported; non-pweight SurveyDesigns are
        # rejected with a NotImplementedError pointing at Phase 4.5 C. ----
        if survey is not None:
            if not hasattr(survey, "weights"):
                raise TypeError(
                    f"survey_design= must be a SurveyDesign-like object with a "
                    f".weights attribute; got {type(survey).__name__}. "
                    f"Construct a SurveyDesign via diff_diff.survey."
                )
            if getattr(survey, "weights", None) is None:
                raise NotImplementedError(
                    "survey_design= without weights is not yet supported. Pass "
                    "survey_design=SurveyDesign(weights='<col>', ...) with a "
                    "per-row weight column."
                )
            weight_type = getattr(survey, "weight_type", "pweight")
            if weight_type != "pweight":
                raise NotImplementedError(
                    f"survey_design=SurveyDesign(weight_type={weight_type!r}) is "
                    f"not yet supported on HeterogeneousAdoptionDiD. "
                    f"HAD's weighted local-linear treats weights as "
                    f"sampling (probability) weights only. Replicate "
                    f"designs (BRR/Fay/JK1/JKn/SDR) and frequency / "
                    f"analytic weights are deferred to Phase 4.5 C."
                )

        # ---- Validate multi-period panel and apply staggered filter ----
        F, t_pre_list, t_post_list, data_filtered, filter_info = _validate_had_panel_event_study(
            data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        )

        # ---- Filter weights / resolve survey on the FILTERED frame.
        # ``data_filtered`` preserves the original row index (via
        # ``.loc[keep_mask].copy()`` in the validator), so the filtered
        # weights array is ``weights[data_filtered.index]``. Survey
        # resolution runs against ``data_filtered`` directly — the
        # SurveyDesign reads its columns by name from the DataFrame. ----
        weights_unit: Optional[np.ndarray] = None
        raw_weights_unit: Optional[np.ndarray] = None
        resolved_survey_unit: Any = None  # ResolvedSurveyDesign (G,) or None
        if survey is not None:
            from diff_diff.survey import ResolvedSurveyDesign  # noqa: F401

            resolved_survey_row = survey.resolve(data_filtered)
            # Capture RAW pre-normalization per-row weights for
            # compute_survey_metadata (matches PR #359 static-path
            # contract: sum_weights / weight_range surface the user-
            # supplied scale, not the resolver's mean=1 normalization).
            weight_col_name = getattr(survey, "weights", None)
            if isinstance(weight_col_name, str):
                raw_row_w = np.asarray(data_filtered[weight_col_name], dtype=np.float64)
                raw_weights_unit = _aggregate_unit_weights(data_filtered, raw_row_w, unit_col)
            resolved_survey_unit = _aggregate_unit_resolved_survey(
                data_filtered, resolved_survey_row, unit_col
            )
            weights_unit = np.asarray(resolved_survey_unit.weights, dtype=np.float64)

        # ---- Aggregate to per-horizon first differences ----
        # Cluster extraction is deferred until after design resolution.
        d_arr, dy_dict, _, _, _ = _aggregate_multi_period_first_differences(
            data_filtered,
            outcome_col,
            dose_col,
            time_col,
            unit_col,
            F,
            t_pre_list,
            t_post_list,
            None,
        )

        n_units = int(d_arr.shape[0])
        if n_units < 3:
            raise ValueError(
                f"HAD event-study requires at least 3 units for inference; "
                f"got n_units={n_units} after aggregation."
            )

        # ---- Apply trends_lin detrending (paper Eq 17, R `did_had(trends_lin=TRUE)`).
        # Compute the per-group linear-trend slope from the F-2 → F-1
        # outcome change and subtract a (e+1) × slope adjustment from
        # every event-time evolution. This is the unified formula that
        # absorbs both R's effect-side detrending (Effect_i -= i × slope,
        # i = e+1 for our event-time convention) AND R's placebo-side
        # anchor shift (placebos under trends_lin re-anchor to F-2 and
        # add i × slope; algebraically equivalent to dy_dict[e] -
        # (e+1) × slope on the F-1 anchor convention).
        #
        # The shallowest placebo (e=-2) is dropped under trends_lin
        # because the F-2 → F-1 evolution is "consumed" by the slope
        # estimator (R reduces max placebo lag by 1; same effect).
        #
        # Optimization: dy_dict[-2] = Y[F-2] - Y[F-1] = -slope, so we
        # can read slope directly from the dy_dict entry without
        # re-pivoting the wide outcome matrix.
        if trends_lin:
            if len(t_pre_list) < 2:
                raise ValueError(
                    "HAD.fit(trends_lin=True) requires F >= 3 (at least "
                    "two pre-treatment periods F-2 and F-1) so the per-"
                    "group linear-trend slope Y[F-1] - Y[F-2] is "
                    f"identified. Got F={F!r} with pre-periods "
                    f"{t_pre_list!r} (need at least 2 entries)."
                )
            if -2 not in dy_dict:
                # Should be unreachable given the len(t_pre_list) >= 2
                # check above (the validator guarantees pre-periods are
                # contiguous, so F-2 → e=-2 must be in dy_dict). Defensive
                # belt-and-suspenders.
                raise ValueError(
                    "HAD.fit(trends_lin=True): expected event time -2 "
                    "(period F-2) to be present in the panel so the "
                    "linear-trend slope can be estimated. Got dy_dict "
                    f"keys {sorted(dy_dict.keys())!r}."
                )
            # slope[g] = Y[g, F-1] - Y[g, F-2] = -dy_dict[-2][g].
            lin_trend_slope = -dy_dict[-2]
            # Detrend every event-time evolution. Apply BEFORE dropping
            # e=-2 so the math is unambiguous (the e=-2 entry itself
            # detrends to zero by construction; we drop it anyway).
            dy_dict = {e: dy_dict[e] - (e + 1) * lin_trend_slope for e in dy_dict.keys()}
            # Drop the consumed placebo (F-2 row used by the slope
            # estimator). R reduces max placebo lag by 1 with the same
            # effect (R's placebo lags shift from {1, 2, ..., F-2} to
            # {1, 2, ..., F-3}; in our event_time convention this maps
            # to dropping e = -2).
            del dy_dict[-2]

        # ---- Zero-weight subpopulation convention (mirrors static path).
        # Design decisions (auto-detect, d_lower, mass-point threshold)
        # run on the positive-weight subset; variance composition runs
        # on the FULL design (zero contributions but preserved frame). ----
        d_arr_full = d_arr  # unfiltered; pass to per-horizon fits
        dy_dict_full = dy_dict
        weights_unit_full = weights_unit
        resolved_survey_unit_full = resolved_survey_unit
        raw_weights_unit_full = raw_weights_unit
        if weights_unit is not None:
            positive_mask = weights_unit > 0.0
            if not bool(positive_mask.all()):
                n_dropped = int((~positive_mask).sum())
                warnings.warn(
                    f"HAD event-study: {n_dropped} unit(s) have weight == 0 "
                    f"and are excluded from design resolution (auto-detect, "
                    f"d_lower, mass-point threshold). Retained in the survey "
                    f"design for variance + SurveyMetadata (subpopulation "
                    f"convention).",
                    UserWarning,
                    stacklevel=2,
                )
                d_arr = d_arr[positive_mask]
                dy_dict = {e: v[positive_mask] for e, v in dy_dict.items()}
                weights_unit = weights_unit[positive_mask]
                n_units = int(d_arr.shape[0])
                if n_units < 3:
                    raise ValueError(
                        f"HAD event-study requires at least 3 positive-"
                        f"weight units for inference; got n_units={n_units} "
                        f"after the zero-weight filter."
                    )

        # ---- Resolve design (once, from D_{g, F} distribution) ----
        if design_arg == "auto":
            resolved_design = _detect_design(d_arr)
        else:
            resolved_design = design_arg

        # ---- Resolve d_lower ----
        if resolved_design == "continuous_at_zero":
            if d_lower_arg is not None:
                scale = max(1.0, float(np.max(np.abs(d_arr))))
                if abs(float(d_lower_arg)) > 1e-12 * scale:
                    raise ValueError(
                        f"design='continuous_at_zero' (Design 1') requires "
                        f"d_lower == 0 within float tolerance (paper Section "
                        f"3.2 Design 1' regime). Got d_lower="
                        f"{float(d_lower_arg)!r}. For d_lower > 0 use "
                        f"design='continuous_near_d_lower' or "
                        f"design='mass_point', or design='auto'."
                    )
            d_lower_val = 0.0
        elif d_lower_arg is None:
            d_lower_val = float(d_arr.min())
        else:
            d_lower_val = float(d_lower_arg)

        # ---- Regime-partition guards (mirror Phase 2a) ----
        if resolved_design in ("mass_point", "continuous_near_d_lower"):
            scale = max(1.0, float(np.max(np.abs(d_arr))))
            if abs(d_lower_val) <= 1e-12 * scale:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower > 0 (paper "
                    f"Section 3.2 reserves the d_lower=0 regime for Design 1' "
                    f"/ `continuous_at_zero`). Got d_lower={d_lower_val!r}."
                )

        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            d_min_orig = float(d_arr.min())
            if d_min_orig > 0:
                eps_mp = 1e-12 * max(1.0, abs(d_min_orig))
                at_d_min_mask_orig = np.abs(d_arr - d_min_orig) <= eps_mp
                modal_fraction_orig = float(np.mean(at_d_min_mask_orig))
                if (
                    resolved_design == "continuous_near_d_lower"
                    and modal_fraction_orig > _MASS_POINT_THRESHOLD
                ):
                    raise ValueError(
                        f"design='continuous_near_d_lower' cannot be used on a "
                        f"mass-point sample (modal fraction {modal_fraction_orig:.4f} "
                        f"at d.min()={d_min_orig!r} exceeds the "
                        f"{_MASS_POINT_THRESHOLD:.2f} threshold)."
                    )
                if resolved_design == "mass_point" and modal_fraction_orig <= _MASS_POINT_THRESHOLD:
                    raise ValueError(
                        f"design='mass_point' requires a modal mass at d.min() "
                        f"exceeding the {_MASS_POINT_THRESHOLD:.2f} threshold "
                        f"(paper Section 3.2.4). Got modal fraction "
                        f"{modal_fraction_orig:.4f} at d.min()={d_min_orig!r}."
                    )

        if resolved_design in ("mass_point", "continuous_near_d_lower") and d_lower_arg is not None:
            d_min = float(d_arr.min())
            tol = 1e-12 * max(1.0, abs(d_min))
            if abs(d_lower_val - d_min) > tol:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower to equal "
                    f"the support infimum float(d.min())={d_min!r}; got "
                    f"d_lower={d_lower_val!r}."
                )
            d_lower_val = d_min  # snap

        dose_mean = float(d_arr.mean())

        # ---- Extract cluster IDs (mass-point + continuous paths) ----
        # cluster= threads into the per-horizon CR1 sandwich (mass-point) or
        # the cluster-robust CCT SE (continuous, Phase 2a parity), AND into a
        # cluster-robust sup-t simultaneous band (both designs) via the
        # clustered branch of ``_sup_t_multiplier_bootstrap``. The one
        # incompatible composition is cluster= + survey_design= (either design): the
        # survey path composes Binder-TSL variance and would silently override
        # the cluster-robust sandwich while metadata still reports the
        # cluster-robust vcov. Reject it BEFORE extracting the column (mirrors
        # the static-path guard had.py:3361) so the error is predictable even
        # for a malformed cluster column. The clustered band reconciles the
        # variance family (raw cluster Rademacher; mass-point sqrt(G/(G-1))
        # CR1 scaling).
        cluster_arr: Optional[np.ndarray] = None
        if cluster_arg is not None and resolved_survey_unit_full is not None:
            raise NotImplementedError(
                f"cluster={cluster_arg!r} + survey_design= on "
                f"design={resolved_design!r} event-study is not supported: "
                f"the survey path composes Binder-TSL variance per horizon "
                f"and would silently override the cluster-robust sandwich. "
                f"Pass cluster= alone (unweighted cluster-robust), or "
                f"survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>') to "
                f"cluster through the survey (Binder-TSL) path."
            )
        if cluster_arg is not None:
            _, _, cluster_arr, _, _ = _aggregate_multi_period_first_differences(
                data_filtered,
                outcome_col,
                dose_col,
                time_col,
                unit_col,
                F,
                t_pre_list,
                t_post_list,
                cluster_arg,
            )

        # ---- One-time warnings (per fit call, not per horizon) ----
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            warnings.warn(
                f"design={resolved_design!r} (Design 1, d_lower > 0) requires "
                f"Assumption 6 from de Chaisemartin et al. (2026) for point "
                f"identification of WAS_{{d_lower}}, or Assumption 5 for "
                f"sign identification only. Neither is testable via "
                f"pre-trends.",
                UserWarning,
                stacklevel=3,
            )

        if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
            if vcov_type_arg is not None:
                warnings.warn(
                    f"vcov_type={vcov_type_arg!r} is ignored on the "
                    f"'{resolved_design}' path (continuous designs use the "
                    f"CCT-2014 robust SE from Phase 1c).",
                    UserWarning,
                    stacklevel=3,
                )
            if robust_arg:
                warnings.warn(
                    f"robust=True is ignored on the '{resolved_design}' " f"path.",
                    UserWarning,
                    stacklevel=3,
                )
            # cluster= is now threaded into the continuous-path CCT SE and the
            # cluster-robust sup-t band (Phase 2b parity with the static path);
            # no "cluster ignored" warning.

        # ---- Resolve vcov label for mass-point ----
        if resolved_design == "mass_point":
            # Resolve the EFFECTIVE vcov family first (review R5 P1 —
            # previous fix only fired on explicit vcov_type='classical'
            # and missed the default vcov_type=None, robust=False →
            # 'classical' mapping).
            if vcov_type_arg is None:
                vcov_requested = "hc1" if robust_arg else "classical"
            else:
                vcov_requested = vcov_type_arg.lower()
            # Review R3/R5 P1 (event-study arm): reject effective
            # classical when the survey_design= path will compute the IF
            # (the survey path composes Binder-TSL on the HC1-scale IF, so
            # a classical analytical SE would give wrong t-stats). Matches
            # the static-path rejection — weighted mass-point paths use the
            # HC1-scale IF
            # convention uniformly.
            # When cluster= is set, the mass-point path computes the CR1
            # cluster-robust sandwich regardless of vcov_type (classical/hc1
            # are ignored), and the clustered sup-t band normalizes by that
            # CR1 SE — so there is no classical-vs-HC1 variance-family
            # mismatch to reject. Guard on ``cluster_arg is None``.
            _uses_if_matrix = cluster_arg is None and resolved_survey_unit_full is not None
            if vcov_requested == "classical" and _uses_if_matrix:
                raise NotImplementedError(
                    "vcov_type='classical' (resolved — either explicit "
                    "or from the default robust=False mapping) + "
                    "survey_design= on design='mass_point' event-study "
                    "is not yet supported: the per-horizon IF matrix is "
                    "HC1-scale (targets V_HC1 via "
                    "compute_survey_if_variance) and mixing it with a "
                    "classical analytical SE via the survey Binder-TSL "
                    "override would produce an inconsistent variance "
                    "family. cband=False does not avoid this — the survey "
                    "path consumes the HC1-scaled IF for the analytical SE "
                    "regardless. Use vcov_type='hc1' (or leave vcov_type "
                    "unset with robust=True) on the survey_design= "
                    "event-study path."
                )
            inference_method = "analytical_2sls"
            vcov_label: Optional[str] = "cr1" if cluster_arg is not None else vcov_requested
            cluster_label: Optional[str] = cluster_arg if cluster_arg is not None else None
        else:
            vcov_requested = ""
            inference_method = "analytical_nonparametric"
            # cluster-robust CCT SE when cluster= is set (Phase 2a static-path
            # parity, had.py:3615); labelled "cr1" for surface consistency
            # with the mass-point CR1 path.
            vcov_label = "cr1" if cluster_arg is not None else None
            cluster_label = cluster_arg if cluster_arg is not None else None

        # ---- Per-horizon loop ----
        # On the weighted path, every horizon uses the FULL arrays
        # (zero-weight units padded to 0 contribution) so the stacked IF
        # matrix aligns with the full survey design. On unweighted fits,
        # `d_arr_full == d_arr` and `dy_dict_full == dy_dict`, so this
        # branch is a no-op.
        event_times_sorted = sorted(dy_dict.keys())
        n_horizons = len(event_times_sorted)
        # Use the full arrays when weighted so the IF matrix aligns with
        # the survey design; unweighted uses the same d_arr either way.
        weighted_es = weights_unit_full is not None
        d_arr_loop = d_arr_full if weighted_es else d_arr
        dy_dict_loop = dy_dict_full if weighted_es else dy_dict
        G_full = int(d_arr_full.shape[0])

        att_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        se_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        t_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        p_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        ci_lo_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        ci_hi_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        # Review R4 P2: report the POSITIVE-WEIGHT contributing sample
        # size, not the full pre-filter design size. Matches the
        # static-path n_obs contract where zero-weight units are
        # excluded from the reported count (survey_metadata still
        # carries the full-design effective_n / n_psu / etc.).
        n_obs_arr = np.full(n_horizons, n_units, dtype=np.int64)

        # Two IF-consumption flags (review R6 P2): the PER-HORIZON IF is
        # needed when the survey_design= path composes Binder-TSL variance
        # (via compute_survey_if_variance inside _fit_continuous or the
        # mass-point override below); the STACKED (G, H) IF matrix is
        # needed only when the sup-t multiplier bootstrap runs
        # (``cband=True`` on a survey_design= OR clustered fit). Splitting
        # them avoids allocating / filling Psi on the common opt-out path
        # (``cband=False`` on an unweighted, unclustered fit), where no IF
        # consumer exists.
        # The clustered sup-t band consumes the stacked IF too, and (unlike
        # the survey band) fires even on the UNWEIGHTED path — so the
        # stacked-IF flag also switches on when cluster= is set.
        needs_per_horizon_if = resolved_survey_unit_full is not None or (weighted_es and cband)
        needs_stacked_if_matrix = cband and (weighted_es or cluster_arg is not None)
        if needs_stacked_if_matrix:
            Psi = np.full((G_full, n_horizons), np.nan, dtype=np.float64)
        else:
            Psi = np.zeros((0, 0), dtype=np.float64)  # sentinel, not used

        # Collect per-horizon diagnostics on continuous paths. Entries may be
        # None for horizons where ``_fit_continuous`` caught a degenerate
        # bandwidth-selector failure (constant/perfectly-linear outcome).
        bc_fits: Optional[List[Optional[BiasCorrectedFit]]] = (
            [] if resolved_design in ("continuous_at_zero", "continuous_near_d_lower") else None
        )
        bw_diags: Optional[List[Optional[BandwidthResult]]] = (
            [] if resolved_design in ("continuous_at_zero", "continuous_near_d_lower") else None
        )

        # df_survey for t-inference on survey path (mirrors static path).
        df_infer: Optional[int] = None
        if resolved_survey_unit_full is not None:
            df_infer = resolved_survey_unit_full.df_survey

        # Whenever the sup-t multiplier bootstrap runs (weighted/survey OR
        # clustered — see ``needs_stacked_if_matrix``), it operates on the
        # per-horizon IF matrix, so we must force the IF computation even
        # when _fit_continuous would normally skip it (an unweighted
        # clustered fit; no survey structure). Pass through
        # the actual ``resolved_survey_unit_full`` (None on those paths) so
        # the per-horizon analytical SE still matches the static-path
        # convention (bc_fit.se_robust on unweighted/clustered; Binder-TSL on
        # the survey_design= path). IF return is gated on
        # `force_return_influence=True`.

        # Track the Binder-TSL den for continuous paths so we can
        # reconstruct the per-unit IF (psi / den) for the sup-t bootstrap
        # where both numerator IF and denominator divide are needed.
        for i, e in enumerate(event_times_sorted):
            dy_e = dy_dict_loop[e]
            if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
                att_e, se_e, bc_fit_e, bw_diag_e = self._fit_continuous(
                    d_arr_loop,
                    dy_e,
                    resolved_design,
                    d_lower_val,
                    weights_arr=weights_unit_full,
                    resolved_survey_unit=resolved_survey_unit_full,
                    # Force IF return only when the sup-t bootstrap
                    # needs the stacked matrix AND the survey_design= gate
                    # won't already produce it. Under the survey_design=
                    # path, _fit_continuous returns the IF automatically
                    # (resolved_survey_unit_full != None); on an unweighted
                    # CLUSTERED fit with cband=True, force it here;
                    # otherwise skip the O(G) IF work (review R6 P2).
                    force_return_influence=(
                        needs_stacked_if_matrix and resolved_survey_unit_full is None
                    ),
                    # Phase 2b: cluster-robust CCT SE per horizon (static-path
                    # parity). None on unclustered fits (byte-unchanged).
                    cluster_arr=cluster_arr,
                )
                if bc_fits is not None:
                    bc_fits.append(bc_fit_e)
                if bw_diags is not None:
                    bw_diags.append(bw_diag_e)
                # Collect per-unit IF on β̂-scale (psi_bc / den) into
                # Psi ONLY when the sup-t bootstrap will consume it.
                if (
                    needs_stacked_if_matrix
                    and bc_fit_e is not None
                    and bc_fit_e.influence_function is not None
                ):
                    if resolved_design == "continuous_at_zero":
                        den_e = float(np.average(d_arr_full, weights=weights_unit_full))
                    else:
                        den_e = float(
                            np.average(
                                d_arr_full - d_lower_val,
                                weights=weights_unit_full,
                            )
                        )
                    if abs(den_e) > 1e-12:
                        Psi[:, i] = bc_fit_e.influence_function / abs(den_e)
            elif resolved_design == "mass_point":
                att_e, se_e, psi_e = _fit_mass_point_2sls(
                    d_arr_loop,
                    dy_e,
                    d_lower_val,
                    cluster_arr,
                    vcov_requested,
                    weights=weights_unit_full,
                    # Return IF when a consumer exists: the survey_design=
                    # path needs it for the per-horizon Binder-TSL override;
                    # the sup-t bootstrap (survey OR clustered) needs it for
                    # the stacked matrix. An unweighted, unclustered fit with
                    # cband=False skips IF computation entirely (review R6 P2).
                    return_influence=(needs_per_horizon_if or needs_stacked_if_matrix),
                )
                # Survey path: override analytical sandwich SE with
                # Binder-TSL via compute_survey_if_variance (matches
                # continuous-path convention from PR #359).
                if resolved_survey_unit_full is not None and psi_e is not None:
                    from diff_diff.survey import compute_survey_if_variance

                    v_survey = compute_survey_if_variance(psi_e, resolved_survey_unit_full)
                    if np.isfinite(v_survey) and v_survey > 0.0:
                        se_e = float(np.sqrt(v_survey))
                    else:
                        se_e = float("nan")
                if needs_stacked_if_matrix and psi_e is not None:
                    Psi[:, i] = psi_e
            else:
                raise ValueError(f"Internal error: unhandled design={resolved_design!r}.")

            t_stat_e, p_value_e, conf_int_e = safe_inference(
                att_e, se_e, alpha=float(self.alpha), df=df_infer
            )
            att_arr[i] = float(att_e)
            se_arr[i] = float(se_e)
            t_arr[i] = float(t_stat_e)
            p_arr[i] = float(p_value_e)
            ci_lo_arr[i] = float(conf_int_e[0])
            ci_hi_arr[i] = float(conf_int_e[1])

        # ---- Sup-t simultaneous confidence band (weighted/survey OR
        # clustered, + cband) ----
        cband_low_arr: Optional[np.ndarray] = None
        cband_high_arr: Optional[np.ndarray] = None
        cband_crit_value: Optional[float] = None
        cband_method_label: Optional[str] = None
        cband_n_bootstrap_eff: Optional[int] = None
        if cband and cluster_arg is not None and n_horizons >= 1:
            # Cluster-robust simultaneous band (Phase 2b). Route cluster= as
            # cluster-level multipliers on the per-unit IF: the raw cluster
            # Rademacher variance sum_c (sum_{i in c} psi_i)^2 equals the
            # analytical cluster-robust SE² for the continuous CCT path (no
            # g/(g-1) correction; scale 1.0) and, for the mass-point CR1 path,
            # matches after restoring the CR1 sqrt(G/(G-1)) factor (the
            # returned mass-point IF already carries sqrt((n-1)/(n-k)) but not
            # G/(G-1)). G is the full-array cluster count — identical to the
            # bootstrap branch's own n_clusters, so scale and aggregation
            # share one count (a wholly-zero-weight cluster contributes 0 to
            # both yet is counted in G by both). Fires on the UNWEIGHTED path
            # too (cluster= + survey_design= is rejected up-front, so no survey
            # composition). See REGISTRY "Note (HAD clustered event-study
            # sup-t band)".
            _cl_G = int(len(pd.unique(np.asarray(cluster_arr))))
            if _cl_G < 2:
                warnings.warn(
                    f"cluster={cluster_arg!r} yields a single cluster; the "
                    f"cluster-robust sup-t band and pointwise SEs are "
                    f"undefined (NaN band returned).",
                    RuntimeWarning,
                    stacklevel=3,
                )
                # Undefined band, NOT skipped: mirror the degenerate
                # weighted-band convention (crit=NaN, method + replicate
                # count populated, endpoints left None) so callers can tell
                # "attempted but undefined" apart from "no band requested".
                cband_crit_value = float("nan")
                cband_method_label = "cluster_multiplier_bootstrap"
                cband_n_bootstrap_eff = n_bootstrap_eff
            else:
                _cl_scale = (
                    float(np.sqrt(_cl_G / (_cl_G - 1))) if resolved_design == "mass_point" else 1.0
                )
                q, cband_low_arr, cband_high_arr, _n_valid = _sup_t_multiplier_bootstrap(
                    influence_matrix=Psi,
                    att_per_horizon=att_arr,
                    se_per_horizon=se_arr,
                    resolved_survey=None,
                    n_bootstrap=n_bootstrap_eff,
                    alpha=float(self.alpha),
                    seed=seed_eff,
                    cluster_ids=cluster_arr,
                    cluster_if_scale=_cl_scale,
                )
                cband_crit_value = q
                cband_method_label = "cluster_multiplier_bootstrap"
                cband_n_bootstrap_eff = n_bootstrap_eff
        elif weighted_es and cband and n_horizons >= 1:
            # The per-unit influence function returned by _fit_continuous
            # / _fit_mass_point_2sls is HC1-scaled per the PR #359
            # convention — compute_survey_if_variance(psi, resolved) ≈
            # V_HC1. The sup-t multiplier bootstrap fires the survey-aware
            # branch on the resolved survey design (centered +
            # sqrt(n/(n-1))-corrected), matching the variance family of the
            # analytical per-horizon Binder-TSL SE.
            resolved_for_bootstrap: Any = resolved_survey_unit_full
            q, cband_low_arr, cband_high_arr, _n_valid = _sup_t_multiplier_bootstrap(
                influence_matrix=Psi,
                att_per_horizon=att_arr,
                se_per_horizon=se_arr,
                resolved_survey=resolved_for_bootstrap,
                n_bootstrap=n_bootstrap_eff,
                alpha=float(self.alpha),
                seed=seed_eff,
            )
            cband_crit_value = q
            cband_method_label = "multiplier_bootstrap"
            cband_n_bootstrap_eff = n_bootstrap_eff

        # ---- Build survey metadata + variance_formula + effective_dose_mean
        # (mirrors static-path branch). ----
        survey_metadata: Optional[SurveyMetadata] = None
        variance_formula_label: Optional[str] = None
        effective_dose_mean_value: Optional[float] = None
        if weights_unit_full is not None:
            if resolved_survey_unit_full is not None:
                assert raw_weights_unit_full is not None
                survey_metadata = compute_survey_metadata(
                    resolved_survey_unit_full, raw_weights_unit_full
                )
                variance_formula_label = (
                    "survey_binder_tsl_2sls"
                    if resolved_design == "mass_point"
                    else "survey_binder_tsl"
                )
            if resolved_design == "continuous_at_zero":
                effective_dose_mean_value = float(np.average(d_arr_full, weights=weights_unit_full))
            elif resolved_design == "continuous_near_d_lower":
                effective_dose_mean_value = float(
                    np.average(d_arr_full - d_lower_val, weights=weights_unit_full)
                )
            elif resolved_design == "mass_point":
                Z_mp = (d_arr_full > d_lower_val).astype(np.float64)
                pos_mp = weights_unit_full > 0
                Z1_mp = (Z_mp == 1) & pos_mp
                Z0_mp = (Z_mp == 0) & pos_mp
                w_Z1_mp = float(weights_unit_full[Z1_mp].sum())
                w_Z0_mp = float(weights_unit_full[Z0_mp].sum())
                if w_Z1_mp > 0.0 and w_Z0_mp > 0.0:
                    effective_dose_mean_value = float(
                        (weights_unit_full[Z1_mp] * d_arr_full[Z1_mp]).sum() / w_Z1_mp
                        - (weights_unit_full[Z0_mp] * d_arr_full[Z0_mp]).sum() / w_Z0_mp
                    )

        return HeterogeneousAdoptionDiDEventStudyResults(
            event_times=np.asarray(event_times_sorted, dtype=np.int64),
            att=att_arr,
            se=se_arr,
            t_stat=t_arr,
            p_value=p_arr,
            conf_int_low=ci_lo_arr,
            conf_int_high=ci_hi_arr,
            n_obs_per_horizon=n_obs_arr,
            alpha=float(self.alpha),
            design=resolved_design,
            target_parameter=_TARGET_PARAMETER[resolved_design],
            d_lower=d_lower_val,
            dose_mean=dose_mean,
            F=F,
            # Review R4 P2: report positive-weight contributing count
            # (matches n_obs_per_horizon; full-design size surfaces
            # through survey_metadata.n_psu / effective_n / etc.).
            n_units=n_units,
            inference_method=inference_method,
            vcov_type=vcov_label,
            cluster_name=cluster_label,
            survey_metadata=survey_metadata,
            bandwidth_diagnostics=bw_diags,
            bias_corrected_fit=bc_fits,
            filter_info=filter_info,
            variance_formula=variance_formula_label,
            effective_dose_mean=effective_dose_mean_value,
            cband_low=cband_low_arr,
            cband_high=cband_high_arr,
            cband_crit_value=cband_crit_value,
            cband_method=cband_method_label,
            cband_n_bootstrap=cband_n_bootstrap_eff,
        )
