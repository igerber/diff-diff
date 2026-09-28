"""Sensitivity analysis for LWDiD estimator.

Assesses robustness of ATT estimates along the two axes with direct
theoretical grounding in Lee & Wooldridge (2025, 2026):
- Pre-period selection sensitivity (T0-robustness)
- No-anticipation assumption sensitivity

Classification thresholds (a diff-diff LIBRARY HEURISTIC — the Lee &
Wooldridge papers recommend the diagnostics but define no categorical
robustness scale; see docs/methodology/REGISTRY.md, LWDiD):
  sensitivity_ratio < 10%  → 'highly_robust'
  10% ≤ ratio < 25%       → 'moderately_robust'
  25% ≤ ratio < 50%       → 'sensitive'
  ratio ≥ 50%             → 'highly_sensitive'
  ratio is NaN            → 'not_estimable' (baseline ATT non-finite or
                            fewer than two specifications produced finite
                            estimates; robustness cannot be assessed)

References
----------
Lee, S. J. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
  to Difference-in-Differences Estimation for Panel Data." SSRN 4516518.
Lee, S. J. & Wooldridge, J. M. (2026). "Simple Approaches to Inference
  with Difference-in-Differences Estimators with Small Cross-Sectional
  Sample Sizes." SSRN 5325686.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

_ROBUSTNESS_THRESHOLDS = {
    "highly_robust": 0.10,
    "moderately_robust": 0.25,
    "sensitive": 0.50,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SpecificationResult:
    """Result from a single specification in sensitivity analysis.

    Attributes
    ----------
    label : str
        Human-readable label describing this specification.
    rolling : str
        Transformation method used ('demean' or 'detrend').
    estimation_method : str
        Estimation method used ('reg', 'ipw', 'dr').
    n_pre_periods : int
        Number of pre-treatment periods used. -1 if all periods used.
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of ATT.
    pvalue : float
        Two-sided p-value for testing H0: ATT = 0.
    """

    label: str
    rolling: str
    estimation_method: str
    n_pre_periods: int
    att: float
    se: float
    pvalue: float
    #: Fitted confidence interval endpoints (round-21 review: plots must
    #: render the fitted interval, not a normal-theory +/-1.96*SE).
    conf_int: Optional[Tuple[float, float]] = None

    @property
    def is_significant(self) -> float:
        """1.0 / 0.0 for a decidable 5%-level test, NaN when the p-value
        is missing or non-finite (a failed specification must never
        publish "not significant" - fix-wave WS10)."""
        if self.pvalue is None or not np.isfinite(self.pvalue):
            return float("nan")
        return float(self.pvalue < 0.05)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "label": self.label,
            "rolling": self.rolling,
            "estimation_method": self.estimation_method,
            "n_pre_periods": self.n_pre_periods,
            "att": self.att,
            "se": self.se,
            "pvalue": self.pvalue,
            "conf_int": self.conf_int,
            "significant_05": self.is_significant,
        }


@dataclass
class SensitivityResult:
    """Result of comprehensive sensitivity analysis.

    Attributes
    ----------
    specifications : List[SpecificationResult]
        Results from each non-baseline specification.
    baseline_att : float
        ATT from the baseline specification.
    baseline_se : float
        Standard error from the baseline specification.
    sensitivity_ratio : float
        (max_att - min_att) / |baseline_att|, measuring estimate instability.
        NaN when robustness cannot be assessed (non-finite baseline ATT or
        fewer than two finite estimates).
    robustness_level : str
        Categorical assessment: 'highly_robust', 'moderately_robust',
        'sensitive', 'highly_sensitive', or 'not_estimable'. The
        'not_estimable' level indicates the sensitivity ratio is NaN
        because too few specifications produced finite estimates.
    n_specifications : int
        Total number of specifications tested (including baseline).
    """

    specifications: List[SpecificationResult]
    baseline_att: float
    baseline_se: float
    sensitivity_ratio: float
    robustness_level: str
    n_specifications: int
    #: Baseline specification's p-value (None only on legacy construction;
    #: NaN when the baseline fit failed).
    baseline_pvalue: Optional[float] = None

    def summary(self) -> str:
        """Return a formatted summary of sensitivity analysis results.

        Returns
        -------
        str
            Multi-line string summarizing the sensitivity analysis.
        """
        lines = [
            "=" * 60,
            "LWDiD Sensitivity Analysis Summary",
            "=" * 60,
            f"Baseline ATT:       {self.baseline_att:.6f}",
            f"Baseline SE:        {self.baseline_se:.6f}",
            f"Sensitivity Ratio:  {self.sensitivity_ratio:.4f} "
            f"({self.sensitivity_ratio * 100:.1f}%)",
            f"Robustness Level:   {self.robustness_level}",
            f"N Specifications:   {self.n_specifications}",
            "-" * 60,
        ]

        if self.specifications:
            lines.append(f"{'Label':<25} {'ATT':>10} {'SE':>10} {'p-value':>10}")
            lines.append("-" * 60)
            for spec in self.specifications:
                lines.append(
                    f"{spec.label:<25} {spec.att:>10.6f} " f"{spec.se:>10.6f} {spec.pvalue:>10.4f}"
                )
        else:
            lines.append("No alternative specifications computed.")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all specification results to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: label, rolling, estimation_method,
            n_pre_periods, att, se, pvalue, significant_05.
        """
        baseline_p = self.baseline_pvalue
        if baseline_p is None or not np.isfinite(baseline_p):
            baseline_sig = float("nan")
            baseline_p = float("nan") if baseline_p is None else baseline_p
        else:
            baseline_sig = float(baseline_p < 0.05)
        rows = [
            {
                "label": "baseline",
                "rolling": "",
                "estimation_method": "",
                "n_pre_periods": -1,
                "att": self.baseline_att,
                "se": self.baseline_se,
                "pvalue": baseline_p,
                "significant_05": baseline_sig,
            }
        ]
        for spec in self.specifications:
            rows.append(spec.to_dict())
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(baseline_att={self.baseline_att:.4f}, "
            f"ratio={self.sensitivity_ratio:.4f}, "
            f"level='{self.robustness_level}', "
            f"n_specs={self.n_specifications})"
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _classify_robustness(ratio: float) -> str:
    """Classify sensitivity ratio into robustness level.

    Parameters
    ----------
    ratio : float
        Sensitivity ratio (range / |baseline|). NaN indicates the ratio
        could not be estimated.

    Returns
    -------
    str
        One of 'highly_robust', 'moderately_robust', 'sensitive',
        'highly_sensitive', or 'not_estimable' (when ratio is NaN).
    """
    if np.isnan(ratio):
        return "not_estimable"
    if ratio < _ROBUSTNESS_THRESHOLDS["highly_robust"]:
        return "highly_robust"
    elif ratio < _ROBUSTNESS_THRESHOLDS["moderately_robust"]:
        return "moderately_robust"
    elif ratio < _ROBUSTNESS_THRESHOLDS["sensitive"]:
        return "sensitive"
    else:
        return "highly_sensitive"


def _compute_sensitivity_ratio(baseline_att: float, all_atts: List[float]) -> float:
    """Compute sensitivity ratio from ATT estimates.

    Parameters
    ----------
    baseline_att : float
        Baseline ATT estimate.
    all_atts : list of float
        All ATT estimates including baseline.

    Returns
    -------
    float
        Sensitivity ratio: (max - min) / |baseline|. NaN when the baseline
        ATT is non-finite, fewer than two estimates are finite, or the
        baseline is (numerically) zero -- the relative ratio is undefined
        there, so robustness cannot be assessed (classified as
        'not_estimable', never 'highly_robust').
    """
    if not np.isfinite(baseline_att):
        return float(np.nan)
    finite_atts = [a for a in all_atts if np.isfinite(a)]
    if len(finite_atts) <= 1:
        return float(np.nan)
    if abs(baseline_att) < 1e-10:
        return float(np.nan)
    return (max(finite_atts) - min(finite_atts)) / abs(baseline_att)


def _fit_single_spec(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    cohort: Optional[str],
    rolling: str,
    estimation_method: str,
    vcov_type: str,
    cluster: Optional[str],
    controls: Optional[List[str]],
    control_group: str = "not_yet_treated",
    raise_errors: bool = False,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Fit a single LWDiD specification and return
    (att, se, pvalue, conf_int).

    Column existence is validated eagerly: missing columns raise
    ValueError instead of being silently converted to NaN. Only
    data-dependent failures of the fit itself (ValueError from a
    degenerate specification, e.g. no remaining pre-periods, or a
    LinAlgError from a singular design) are mapped to (nan, nan, nan);
    any other exception is a programming error and propagates.

    ``raise_errors=True`` (the BASELINE fit in both public helpers, on
    the full frame) propagates every fit error: a full-frame failure is
    a configuration/support problem (e.g. covariate-free PSM), not a
    restricted-specification non-estimability, and must not be reported
    as ``not_estimable`` (round-10 review).
    """
    from diff_diff.lwdid import LWDiD

    required = {
        "outcome": outcome,
        "unit": unit,
        "time": time,
        "treatment": treatment,
    }
    if cohort is not None:
        required["cohort"] = cohort
    if cluster is not None:
        required["cluster"] = cluster
    missing = [f"{role}={name!r}" for role, name in required.items() if name not in data.columns]
    if controls is not None:
        missing.extend(f"control={c!r}" for c in controls if c not in data.columns)
    if missing:
        raise ValueError(
            f"Column(s) not found in data for sensitivity analysis: {', '.join(missing)}"
        )

    est = LWDiD(
        rolling=rolling,
        estimation_method=estimation_method,
        vcov_type=vcov_type,
        cluster=cluster,
        control_group=control_group,
    )
    try:
        res = est.fit(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            treatment=treatment,
            first_treat=cohort,
            covariates=controls,
        )
        return res.att, res.se, res.p_value, res.conf_int
    except (ValueError, np.linalg.LinAlgError):
        if raise_errors:
            raise
        return np.nan, np.nan, np.nan, (np.nan, np.nan)


def _prevalidate_frame(data, outcome, unit, time, treatment, cohort, cluster, controls) -> None:
    """Run LWDiD's shared input validation on the full frame (raises).

    Includes the treatment-design check (absorbing treatment, common-
    timing onset homogeneity, D_it/cohort consistency) with the same
    encode-then-normalize ordering as ``fit()`` - round-7 review: without
    it, a structurally invalid design (e.g. a 1 -> 0 treatment reversal)
    was swallowed by the per-spec ValueError handler and reported as
    ``robustness_level='not_estimable'`` instead of raising.
    """
    from diff_diff.lwdid import (
        LWDiD,
        _check_treatment_design,
        _encode_staggered_time_scale,
        _normalize_cohorts,
    )
    from diff_diff.utils import validate_binary

    probe = LWDiD(cluster=cluster)
    frame = data.copy()
    probe._validate_inputs(
        frame, outcome, unit, time, treatment, cohort, cluster, list(controls or [])
    )
    validate_binary(frame[treatment].values, treatment)
    time_col, cohort_col = time, cohort
    if cohort is not None:
        frame, time_col, cohort_col, _ = _encode_staggered_time_scale(frame, time, cohort)
        with warnings.catch_warnings():
            # fit() re-normalizes and re-warns; suppress the duplicate here.
            warnings.simplefilter("ignore")
            frame[cohort_col], _, _ = _normalize_cohorts(
                frame[cohort_col], max_time=frame[time_col].max()
            )
    _check_treatment_design(frame, unit, time_col, treatment, cohort_col)
    if cohort is not None:
        # Multi-cohort rejection on the NORMALIZED cohorts (round-10
        # review: counting raw values rejected valid single-cohort
        # designs whose beyond-window/inf encodings normalize to
        # never-treated).
        values = frame[cohort_col].to_numpy(dtype=float)
        treated_cohorts = np.unique(values[np.isfinite(values) & (values > 0)])
        if len(treated_cohorts) > 1:
            raise ValueError(
                f"Sensitivity analyses currently support a single treated "
                f"cohort; found {len(treated_cohorts)} distinct cohorts in "
                f"'{cohort}' (after never-treated normalization). "
                f"Pre-period exclusions are defined relative to the "
                f"earliest adoption, which would mislabel the samples used "
                f"for later cohorts' transformations. Run the analysis per "
                f"cohort, or see DEFERRED.md (cohort-relative sensitivity "
                f"exclusions)."
            )


def _get_pre_periods(data: pd.DataFrame, time: str, treatment: str) -> np.ndarray:
    """Identify pre-treatment periods from the data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    time : str
        Time column name.
    treatment : str
        Treatment indicator column name.

    Returns
    -------
    np.ndarray
        Sorted array of pre-treatment period values.
    """
    # Partition at the single onset S = min(observed treated period),
    # matching fit()'s calendar rule (round-9 review: the former
    # any-unit-treated rule classified a controls-only post period as
    # pre-treatment, so exclusions could remove a POST period while
    # labeling it an excluded pre-period).
    all_periods = np.sort(data[time].unique())
    treated_times = data.loc[data[treatment] == 1, time]
    if len(treated_times) == 0:
        return all_periods
    onset_s = treated_times.min()
    return np.array([p for p in all_periods if p < onset_s])


# =============================================================================
# Public API: robustness_pre_periods
# =============================================================================


def robustness_pre_periods(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    rolling: str = "demean",
    estimation_method: str = "reg",
    vcov_type: str = "hc1",
    cluster: Optional[str] = None,
    controls: Optional[List[str]] = None,
    control_group: str = "not_yet_treated",
    k_min: int = 2,
    k_max: Optional[int] = None,
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> SensitivityResult:
    """Assess sensitivity of ATT to number of pre-treatment periods used.

    For each k in range(k_min, k_max+1), restricts the data to use only
    the last k pre-treatment periods for rolling transformation, then fits
    LWDiD and collects the ATT estimate.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    outcome : str
        Outcome column name. (alias: y)
    unit : str
        Unit identifier column name. (alias: ivar)
    time : str
        Time period column name. (alias: tvar)
    treatment : str
        Binary treatment indicator column name. (alias: d)
    cohort : str, optional
        Cohort variable for staggered designs. (alias: gvar) At most ONE
        distinct treated cohort is supported: the exclusion windows are
        defined relative to the earliest adoption, which would mislabel
        later cohorts' transformation samples (multi-cohort inputs raise
        ValueError; see DEFERRED.md, cohort-relative exclusions).
    rolling : str, default 'demean'
        Transformation method.
    estimation_method : str, default 'reg'
        Estimation method.
    vcov_type : str, default 'hc1'
        Variance-covariance family.
    cluster : str, optional
        Cluster variable for standard errors.
    controls : list of str, optional
        Control variable column names.
    k_min : int, default 2
        Minimum number of pre-treatment periods to test.
    k_max : int, optional
        Maximum number of pre-treatment periods. If None, uses all available.

    Returns
    -------
    SensitivityResult
        Sensitivity analysis result with per-specification ATT estimates
        and overall robustness classification.
    """
    if kwargs:
        raise TypeError(
            f"robustness_pre_periods() got unexpected keyword argument(s): " f"{sorted(kwargs)}"
        )
    # Resolve lwdid-py aliases
    outcome = outcome or y
    unit = unit or ivar
    time = time or tvar
    treatment = treatment or d
    cohort = cohort or gvar

    # Validate required params
    if outcome is None:
        raise ValueError("'outcome' (or 'y') parameter is required")
    if unit is None:
        raise ValueError("'unit' (or 'ivar') parameter is required")
    if time is None:
        raise ValueError("'time' (or 'tvar') parameter is required")
    if treatment is None:
        raise ValueError("'treatment' (or 'd') parameter is required")

    # Pre-validate the FULL frame once so genuine specification errors
    # (missing/NaN key columns, non-binary treatment, malformed panels)
    # RAISE here instead of being swallowed as per-spec "failed fits"
    # inside _fit_single_spec (campaign finding: a string covariate's
    # ValueError became a silent NaN spec).
    _prevalidate_frame(data, outcome, unit, time, treatment, cohort, cluster, controls)

    for name, value in (("k_min", k_min), ("k_max", k_max)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")

    pre_periods = _get_pre_periods(data, time, treatment)
    n_pre = len(pre_periods)

    if k_max is None:
        k_max = n_pre

    k_max = min(k_max, n_pre)
    # Transformation-aware minimum (round-16 review: the former
    # unconditional max(k_min, 2) SILENTLY dropped an explicitly
    # requested, methodologically valid k=1 demeaning specification -
    # demeaning needs one pre-period, detrending two).
    min_required = 1 if rolling in ("demean", "demeanq") else 2
    if k_min < min_required:
        raise ValueError(
            f"k_min={k_min} is below the minimum pre-period requirement "
            f"for rolling='{rolling}' ({min_required}; detrending needs "
            f"two pre-periods for its rank condition)."
        )

    if k_min > k_max:
        warnings.warn(
            f"k_min ({k_min}) > k_max ({k_max}). "
            "Insufficient pre-treatment periods for robustness analysis.",
            UserWarning,
            stacklevel=2,
        )
        # Return degenerate result with baseline only
        att, se, pval, spec_ci = _fit_single_spec(
            data,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimation_method,
            vcov_type,
            cluster,
            controls,
            control_group=control_group,
            raise_errors=True,
        )
        degenerate_ratio = _compute_sensitivity_ratio(att, [att])
        return SensitivityResult(
            specifications=[],
            baseline_att=att,
            baseline_se=se,
            sensitivity_ratio=degenerate_ratio,
            robustness_level=_classify_robustness(degenerate_ratio),
            n_specifications=1,
            baseline_pvalue=pval,
        )

    # Baseline: use all pre-periods
    baseline_att, baseline_se, baseline_pval, _baseline_ci = _fit_single_spec(
        data,
        outcome,
        unit,
        time,
        treatment,
        cohort,
        rolling,
        estimation_method,
        vcov_type,
        cluster,
        controls,
        control_group=control_group,
        raise_errors=True,
    )

    # ALL observed periods >= S are post (round-9 review: the any-unit-
    # treated rule dropped controls-only post periods from the subset).
    _onset_s = data.loc[data[treatment] == 1, time].min()
    post_periods = np.sort(data.loc[data[time] >= _onset_s, time].unique())

    specs: List[SpecificationResult] = []

    for k in range(k_min, k_max + 1):
        if k == n_pre:
            # Same as baseline, skip
            continue

        # Keep only the last k pre-periods + all post-periods
        keep_pre = pre_periods[-k:]
        keep_periods = np.concatenate([keep_pre, post_periods])
        subset = data[data[time].isin(keep_periods)].copy()

        att, se, pval, spec_ci = _fit_single_spec(
            subset,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimation_method,
            vcov_type,
            cluster,
            controls,
            control_group=control_group,
        )

        specs.append(
            SpecificationResult(
                label=f"k={k}_pre_periods",
                rolling=rolling,
                estimation_method=estimation_method,
                n_pre_periods=k,
                att=att,
                se=se,
                pvalue=pval,
                conf_int=spec_ci,
            )
        )

    # Compute sensitivity ratio
    all_atts = [baseline_att] + [s.att for s in specs]
    ratio = _compute_sensitivity_ratio(baseline_att, all_atts)
    level = _classify_robustness(ratio)

    if level == "not_estimable":
        warnings.warn(
            "Sensitivity ratio could not be estimated: baseline ATT is "
            "non-finite or fewer than two specifications produced finite "
            "estimates. Robustness to pre-period selection cannot be "
            "assessed.",
            UserWarning,
            stacklevel=2,
        )
    elif level in ("sensitive", "highly_sensitive"):
        warnings.warn(
            f"ATT estimates are {level} to pre-period selection "
            f"(ratio={ratio:.3f}). Consider investigating data structure.",
            UserWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        specifications=specs,
        baseline_att=baseline_att,
        baseline_se=baseline_se,
        sensitivity_ratio=ratio,
        robustness_level=level,
        n_specifications=len(specs) + 1,
        baseline_pvalue=baseline_pval,
    )


# =============================================================================
# Public API: sensitivity_no_anticipation
# =============================================================================


def sensitivity_no_anticipation(
    data: pd.DataFrame,
    outcome: str = None,
    unit: str = None,
    time: str = None,
    treatment: str = None,
    cohort: Optional[str] = None,
    exclude_periods: Optional[List[int]] = None,
    rolling: str = "demean",
    estimation_method: str = "reg",
    vcov_type: str = "hc1",
    cluster: Optional[str] = None,
    controls: Optional[List[str]] = None,
    control_group: str = "not_yet_treated",
    # lwdid-py compatible aliases
    y: Optional[str] = None,
    ivar: Optional[str] = None,
    tvar: Optional[str] = None,
    d: Optional[str] = None,
    gvar: Optional[str] = None,
    **kwargs,
) -> SensitivityResult:
    """Assess sensitivity to potential anticipation effects.

    For each n_exclude in exclude_periods, drops the last n_exclude
    pre-treatment periods and re-estimates LWDiD. If ATT changes
    substantially when excluding periods just before treatment,
    this suggests anticipation effects may be present.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    outcome : str
        Outcome column name. (alias: y)
    unit : str
        Unit identifier column name. (alias: ivar)
    time : str
        Time period column name. (alias: tvar)
    treatment : str
        Binary treatment indicator column name. (alias: d)
    cohort : str, optional
        Cohort variable for staggered designs. (alias: gvar) At most ONE
        distinct treated cohort is supported: the exclusion windows are
        defined relative to the earliest adoption, which would mislabel
        later cohorts' transformation samples (multi-cohort inputs raise
        ValueError; see DEFERRED.md, cohort-relative exclusions).
    exclude_periods : list of int, optional
        Number of pre-treatment periods to exclude in each test.
        Default is [1, 2, 3].
    rolling : str, default 'demean'
        Transformation method.
    estimation_method : str, default 'reg'
        Estimation method.
    vcov_type : str, default 'hc1'
        Variance-covariance family.
    cluster : str, optional
        Cluster variable for standard errors.
    controls : list of str, optional
        Control variable column names.

    Returns
    -------
    SensitivityResult
        Sensitivity result with per-exclusion ATT estimates and
        overall robustness classification.
    """
    if kwargs:
        raise TypeError(
            f"sensitivity_no_anticipation() got unexpected keyword argument(s): "
            f"{sorted(kwargs)}"
        )
    # Resolve lwdid-py aliases
    outcome = outcome or y
    unit = unit or ivar
    time = time or tvar
    treatment = treatment or d
    cohort = cohort or gvar

    # Validate required params
    if outcome is None:
        raise ValueError("'outcome' (or 'y') parameter is required")
    if unit is None:
        raise ValueError("'unit' (or 'ivar') parameter is required")
    if time is None:
        raise ValueError("'time' (or 'tvar') parameter is required")
    if treatment is None:
        raise ValueError("'treatment' (or 'd') parameter is required")

    _prevalidate_frame(data, outcome, unit, time, treatment, cohort, cluster, controls)

    if exclude_periods is None:
        exclude_periods = [1, 2, 3]
    validated_exclusions: List[int] = []
    for value in exclude_periods:
        # Round-3 review: exclude 0 sliced pre_periods[:-0] == EMPTY
        # (dropping every pre-period instead of none); negative values
        # selected the wrong window; bool is an int subclass.
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(
                f"exclude_periods entries must be positive integers "
                f"(number of trailing pre-periods to drop); got {value!r}."
            )
        validated_exclusions.append(int(value))
    if len(set(validated_exclusions)) != len(validated_exclusions):
        raise ValueError(f"exclude_periods contains duplicate entries: {exclude_periods!r}.")
    exclude_periods = validated_exclusions

    pre_periods = _get_pre_periods(data, time, treatment)
    n_pre = len(pre_periods)

    # Baseline: no exclusion
    baseline_att, baseline_se, baseline_pval, _baseline_ci = _fit_single_spec(
        data,
        outcome,
        unit,
        time,
        treatment,
        cohort,
        rolling,
        estimation_method,
        vcov_type,
        cluster,
        controls,
        control_group=control_group,
        raise_errors=True,
    )

    # ALL observed periods >= S are post (round-9 review: the any-unit-
    # treated rule dropped controls-only post periods from the subset).
    _onset_s = data.loc[data[treatment] == 1, time].min()
    post_periods = np.sort(data.loc[data[time] >= _onset_s, time].unique())

    specs: List[SpecificationResult] = []

    for n_exclude in exclude_periods:
        if n_exclude >= n_pre:
            warnings.warn(
                f"Cannot exclude {n_exclude} periods with only {n_pre} "
                "pre-treatment periods. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Exclude the last n_exclude pre-periods
        remaining_pre = pre_periods[:-n_exclude]
        keep_periods = np.concatenate([remaining_pre, post_periods])
        subset = data[data[time].isin(keep_periods)].copy()

        att, se, pval, spec_ci = _fit_single_spec(
            subset,
            outcome,
            unit,
            time,
            treatment,
            cohort,
            rolling,
            estimation_method,
            vcov_type,
            cluster,
            controls,
            control_group=control_group,
        )

        specs.append(
            SpecificationResult(
                label=f"exclude_{n_exclude}_periods",
                rolling=rolling,
                estimation_method=estimation_method,
                n_pre_periods=n_pre - n_exclude,
                att=att,
                se=se,
                pvalue=pval,
                conf_int=spec_ci,
            )
        )

    # Compute sensitivity ratio
    all_atts = [baseline_att] + [s.att for s in specs]
    ratio = _compute_sensitivity_ratio(baseline_att, all_atts)
    level = _classify_robustness(ratio)

    if level == "not_estimable":
        warnings.warn(
            "Sensitivity ratio could not be estimated: baseline ATT is "
            "non-finite or fewer than two specifications produced finite "
            "estimates. Robustness to anticipation exclusions cannot be "
            "assessed.",
            UserWarning,
            stacklevel=2,
        )
    elif level in ("sensitive", "highly_sensitive"):
        warnings.warn(
            f"ATT estimates are {level} to anticipation exclusions "
            f"(ratio={ratio:.3f}). Potential anticipation effects detected.",
            UserWarning,
            stacklevel=2,
        )

    return SensitivityResult(
        specifications=specs,
        baseline_att=baseline_att,
        baseline_se=baseline_se,
        sensitivity_ratio=ratio,
        robustness_level=level,
        n_specifications=len(specs) + 1,
        baseline_pvalue=baseline_pval,
    )
