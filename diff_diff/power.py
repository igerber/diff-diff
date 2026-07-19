"""
Power analysis tools for difference-in-differences study design.

This module provides power calculations and simulation-based power analysis
for DiD study design, helping practitioners answer questions like:
- "How many units do I need to detect an effect of size X?"
- "What is the minimum detectable effect given my sample size?"
- "What power do I have to detect a given effect?"

References
----------
Bloom, H. S. (1995). "Minimum Detectable Effects: A Simple Way to Report the
    Statistical Power of Experimental Designs." Evaluation Review, 19(5), 547-556.

Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    Journal of Development Economics, 144, 102458.

Djimeu, E. W., & Houndolo, D.-G. (2016). "Power Calculation for Causal Inference
    in Social Science: Sample Size and Minimum Detectable Effect Determination."
    Journal of Development Effectiveness, 8(4), 508-527.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.results_base import Diagnostic

# Maximum sample size returned when effect is too small to detect
# (e.g., zero effect or extremely small relative to noise)
MAX_SAMPLE_SIZE = 2**31 - 1


# ---------------------------------------------------------------------------
# Estimator registry — maps estimator class names to DGP/fit/extract profiles
# ---------------------------------------------------------------------------


@dataclass
class _EstimatorProfile:
    """Internal profile describing how to run power simulations for an estimator."""

    default_dgp: Callable
    dgp_kwargs_builder: Callable
    fit_kwargs_builder: Callable
    result_extractor: Callable
    min_n: int = 20


# ---------------------------------------------------------------------------
# SurveyPowerConfig — carries DGP survey params for simulation power
# ---------------------------------------------------------------------------


@dataclass
class SurveyPowerConfig:
    """Configuration for survey-aware power simulations.

    When passed to :func:`simulate_power`, :func:`simulate_mde`, or
    :func:`simulate_sample_size`, the simulation loop generates data with
    :func:`~diff_diff.prep.generate_survey_did_data` and automatically
    injects a ``SurveyDesign`` into the estimator's ``fit()`` call.

    Parameters
    ----------
    n_strata : int, default=5
        Number of geographic strata.
    psu_per_stratum : int, default=8
        Number of primary sampling units (PSUs) per stratum. Must be >= 2
        for Taylor Series Linearization variance estimation.
    fpc_per_stratum : float, default=200.0
        Finite population correction (total PSUs per stratum).
    weight_variation : str, default="moderate"
        Sampling weight dispersion: ``"none"`` (all equal), ``"moderate"``
        (range ~1-2), ``"high"`` (range ~1-4).
    psu_re_sd : float, default=2.0
        Standard deviation of PSU random effects. Controls intra-cluster
        correlation and drives DEFF > 1.
    psu_period_factor : float, default=0.5
        Multiplier for PSU-period interaction shocks.
    icc : float, optional
        Target intra-class correlation (0 < icc < 1). Overrides
        ``psu_re_sd`` via variance decomposition.
    weight_cv : float, optional
        Target coefficient of variation for weights. Overrides
        ``weight_variation``.
    informative_sampling : bool, default=False
        If True, weights correlate with Y(0).
    heterogeneous_te_by_strata : bool, default=False
        If True, treatment effect varies by stratum.
    include_replicate_weights : bool, default=False
        If True, add JK1 delete-one-PSU replicate weight columns.
    survey_design : SurveyDesign, optional
        Override the auto-built SurveyDesign. When None, a default
        ``SurveyDesign(weights="weight", strata="stratum", psu="psu",
        fpc="fpc")`` is used, matching ``generate_survey_did_data`` output.

    Examples
    --------
    >>> from diff_diff import CallawaySantAnna, simulate_power, SurveyPowerConfig
    >>> config = SurveyPowerConfig(n_strata=5, psu_per_stratum=8, icc=0.05)
    >>> results = simulate_power(
    ...     CallawaySantAnna(),
    ...     n_units=200,
    ...     treatment_effect=2.0,
    ...     survey_config=config,
    ...     n_simulations=100,
    ...     seed=42,
    ... )
    """

    n_strata: int = 5
    psu_per_stratum: int = 8
    fpc_per_stratum: float = 200.0
    weight_variation: str = "moderate"
    psu_re_sd: float = 2.0
    psu_period_factor: float = 0.5
    icc: Optional[float] = None
    weight_cv: Optional[float] = None
    informative_sampling: bool = False
    heterogeneous_te_by_strata: bool = False
    include_replicate_weights: bool = False
    survey_design: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {self.n_strata}")
        if self.psu_per_stratum < 2:
            raise ValueError(
                f"psu_per_stratum must be >= 2 for TSL variance estimation, "
                f"got {self.psu_per_stratum}"
            )
        if self.weight_variation not in ("none", "moderate", "high"):
            raise ValueError(
                f"weight_variation must be 'none', 'moderate', or 'high', "
                f"got '{self.weight_variation}'"
            )
        if not np.isfinite(self.psu_re_sd) or self.psu_re_sd < 0:
            raise ValueError(f"psu_re_sd must be finite and >= 0, got {self.psu_re_sd}")
        if not np.isfinite(self.fpc_per_stratum):
            raise ValueError(f"fpc_per_stratum must be finite, got {self.fpc_per_stratum}")
        if self.icc is not None and not (0 < self.icc < 1):
            raise ValueError(f"icc must be between 0 and 1 (exclusive), got {self.icc}")
        if self.icc is not None and self.psu_re_sd != 2.0:
            raise ValueError(
                "Cannot specify both icc and a non-default psu_re_sd. "
                "icc overrides psu_re_sd via the ICC formula."
            )
        if self.weight_cv is not None:
            if not np.isfinite(self.weight_cv) or self.weight_cv <= 0:
                raise ValueError(f"weight_cv must be finite and > 0, got {self.weight_cv}")
            if self.weight_variation != "moderate":
                raise ValueError(
                    "Cannot specify both weight_cv and a non-default "
                    "weight_variation. weight_cv overrides weight_variation."
                )
        if not np.isfinite(self.psu_period_factor) or self.psu_period_factor < 0:
            raise ValueError(
                f"psu_period_factor must be finite and >= 0, got {self.psu_period_factor}"
            )
        if self.fpc_per_stratum < self.psu_per_stratum:
            raise ValueError(
                f"fpc_per_stratum ({self.fpc_per_stratum}) must be >= "
                f"psu_per_stratum ({self.psu_per_stratum})"
            )

    def _build_survey_design(self) -> Any:
        """Return a SurveyDesign for this config.

        Reflects the live ``self.survey_design`` value every call (no
        caching). Finding #28 (axis J, silent-failures audit): the
        previous ``_cached_survey_design`` was populated on first call
        and never invalidated on mutation, so ``config.survey_design =
        other_design`` silently kept returning the original. Since the
        default ``SurveyDesign(...)`` construction is microseconds and
        user-provided designs are just reference copies, there's no cache
        cost worth keeping.
        """
        if self.survey_design is not None:
            return self.survey_design
        from diff_diff.survey import SurveyDesign

        return SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc")

    @property
    def min_viable_n(self) -> int:
        """Minimum n_units for a viable survey design (>= 2 units per PSU)."""
        return self.n_strata * self.psu_per_stratum * 2


# -- DGP kwargs adapters -----------------------------------------------------


def _basic_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
) -> Dict[str, Any]:
    return dict(
        n_units=n_units,
        n_periods=n_periods,
        treatment_effect=treatment_effect,
        treatment_fraction=treatment_fraction,
        treatment_period=treatment_period,
        noise_sd=sigma,
    )


def _staggered_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
) -> Dict[str, Any]:
    return dict(
        n_units=n_units,
        n_periods=n_periods,
        treatment_effect=treatment_effect,
        never_treated_frac=1 - treatment_fraction,
        cohort_periods=[treatment_period],
        dynamic_effects=False,
        noise_sd=sigma,
    )


def _factor_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
) -> Dict[str, Any]:
    n_pre = treatment_period
    n_post = n_periods - treatment_period
    return dict(
        n_units=n_units,
        n_pre=n_pre,
        n_post=n_post,
        n_treated=max(1, int(n_units * treatment_fraction)),
        treatment_effect=treatment_effect,
        noise_sd=sigma,
    )


def _ddd_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
) -> Dict[str, Any]:
    return dict(
        n_per_cell=max(2, n_units // 8),
        treatment_effect=treatment_effect,
        noise_sd=sigma,
    )


def _ddd_panel_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
) -> Dict[str, Any]:
    # Panel DDD DGP (n_periods > 2). `n_units` maps directly (no 8-cell //8
    # rounding). `treatment_fraction` is intentionally NOT mapped — DDD is a
    # balanced factorial, so group_frac/partition_frac are left at the DGP
    # default 0.5. Omitting group_frac/partition_frac here also keeps them out
    # of the _PROTECTED_DGP_KEYS collision check, so a user can still override
    # the group/partition split via data_generator_kwargs.
    return dict(
        n_units=n_units,
        n_periods=n_periods,
        treatment_period=treatment_period,
        treatment_effect=treatment_effect,
        noise_sd=sigma,
    )


# -- Fit kwargs builders ------------------------------------------------------


def _basic_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(outcome="outcome", treatment="treated", time="post")


def _twfe_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(outcome="outcome", treatment="treated", time="post", unit="unit")


def _multiperiod_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(
        outcome="outcome",
        treatment="treated",
        time="period",
        post_periods=list(range(treatment_period, n_periods)),
    )


def _staggered_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _ddd_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(outcome="outcome", group="group", partition="partition", time="time")


def _ddd_panel_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    # Panel DDD: time="post" is generate_ddd_panel_data's derived binary
    # pre/post indicator (vs the cross-sectional "time"). Clustering is NOT a
    # fit kwarg — it resolves from the estimator's cluster="unit" attribute
    # against the DGP's "unit" column.
    return dict(outcome="outcome", group="group", partition="partition", time="post")


def _trop_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    return dict(outcome="outcome", treatment="treated", unit="unit", time="period")


def _sdid_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
) -> Dict[str, Any]:
    periods = sorted(data["period"].unique())
    post_periods = [p for p in periods if p >= treatment_period]
    return dict(
        outcome="outcome",
        treatment="treat",
        unit="unit",
        time="period",
        post_periods=post_periods,
    )


# -- Survey-aware DGP kwargs adapter ------------------------------------------


def _survey_dgp_kwargs(
    n_units: int,
    n_periods: int,
    treatment_effect: float,
    treatment_fraction: float,
    treatment_period: int,
    sigma: float,
    survey_config: SurveyPowerConfig,
) -> Dict[str, Any]:
    """Build kwargs for generate_survey_did_data from simulate_power params."""
    return dict(
        n_units=n_units,
        n_periods=n_periods,
        treatment_effect=treatment_effect,
        never_treated_frac=1 - treatment_fraction,
        # 0-indexed treatment_period → 1-indexed cohort_periods
        cohort_periods=[treatment_period + 1],
        noise_sd=sigma,
        dynamic_effects=False,
        n_strata=survey_config.n_strata,
        psu_per_stratum=survey_config.psu_per_stratum,
        fpc_per_stratum=survey_config.fpc_per_stratum,
        weight_variation=survey_config.weight_variation,
        psu_re_sd=survey_config.psu_re_sd,
        psu_period_factor=survey_config.psu_period_factor,
        icc=survey_config.icc,
        weight_cv=survey_config.weight_cv,
        informative_sampling=survey_config.informative_sampling,
        heterogeneous_te_by_strata=survey_config.heterogeneous_te_by_strata,
        include_replicate_weights=survey_config.include_replicate_weights,
        return_true_population_att=True,
    )


# -- Survey-aware fit kwargs builders -----------------------------------------


def _survey_basic_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
    survey_config: SurveyPowerConfig,
) -> Dict[str, Any]:
    """Fit kwargs for DifferenceInDifferences with survey design.

    Uses ``ever_treated`` (time-invariant group indicator) rather than the
    survey DGP's ``treated`` column (which is post-only: 1{g>0, t>=g}).
    DifferenceInDifferences internally constructs ``treatment * time``,
    so passing the post-only flag would make that interaction rank-deficient.
    """
    return dict(
        outcome="outcome",
        treatment="ever_treated",
        time="post",
        survey_design=survey_config._build_survey_design(),
    )


def _survey_twfe_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
    survey_config: SurveyPowerConfig,
) -> Dict[str, Any]:
    """Fit kwargs for TwoWayFixedEffects with survey design."""
    return dict(
        outcome="outcome",
        treatment="ever_treated",
        time="post",
        unit="unit",
        survey_design=survey_config._build_survey_design(),
    )


def _survey_multiperiod_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
    survey_config: SurveyPowerConfig,
) -> Dict[str, Any]:
    """Fit kwargs for MultiPeriodDiD with survey design (1-indexed periods)."""
    return dict(
        outcome="outcome",
        treatment="ever_treated",
        unit="unit",
        time="period",
        # 1-indexed: post periods run from treatment_period+1 to n_periods
        post_periods=list(range(treatment_period + 1, n_periods + 1)),
        survey_design=survey_config._build_survey_design(),
    )


def _survey_staggered_fit_kwargs(
    data: pd.DataFrame,
    n_units: int,
    n_periods: int,
    treatment_period: int,
    survey_config: SurveyPowerConfig,
) -> Dict[str, Any]:
    """Fit kwargs for staggered estimators (CS, SA, etc.) with survey design."""
    return dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        survey_design=survey_config._build_survey_design(),
    )


# -- Result extractors --------------------------------------------------------


def _extract_simple(result: Any) -> Tuple[float, float, float, Tuple[float, float]]:
    return (result.att, result.se, result.p_value, result.conf_int)


def _extract_multiperiod(
    result: Any,
) -> Tuple[float, float, float, Tuple[float, float]]:
    return (result.avg_att, result.avg_se, result.avg_p_value, result.avg_conf_int)


def _extract_staggered(
    result: Any,
) -> Tuple[float, float, float, Tuple[float, float]]:
    _nan = float("nan")
    _nan_ci = (_nan, _nan)

    def _first(r: Any, *attrs: str, default: Any = _nan) -> Any:
        for a in attrs:
            v = getattr(r, a, None)
            if v is not None:
                return v
        return default

    return (
        result.overall_att,
        _first(result, "overall_se", "overall_att_se"),
        _first(result, "overall_p_value", "overall_att_p_value"),
        _first(result, "overall_conf_int", "overall_att_ci", default=_nan_ci),
    )


# Keys derived from simulate_power() public params — overriding these
# via data_generator_kwargs would desync the DGP from the result object.
_PROTECTED_DGP_KEYS = frozenset(
    {
        "treatment_effect",  # → true_effect in results / MDE search variable
        "noise_sd",  # → sigma param
        "n_units",  # → sample-size search variable
        "n_periods",  # → n_periods param
        "treatment_fraction",  # → treatment_fraction param
        "treatment_period",  # → treatment_period param
        "n_pre",  # → derived from treatment_period in factor-model DGPs
        "n_post",  # → derived from n_periods - treatment_period in factor-model DGPs
    }
)

# Keys managed by SurveyPowerConfig — block in data_generator_kwargs when
# survey_config is active to prevent silent conflicts.
_SURVEY_CONFIG_KEYS = frozenset(
    {
        "n_strata",
        "psu_per_stratum",
        "fpc_per_stratum",
        "weight_variation",
        "psu_re_sd",
        "psu_period_factor",
        "icc",
        "weight_cv",
        "informative_sampling",
        "heterogeneous_te_by_strata",
        "include_replicate_weights",
        "return_true_population_att",
        "dynamic_effects",
        "cohort_periods",
        "never_treated_frac",
    }
)


# -- Staggered DGP compatibility check ----------------------------------------

_STAGGERED_ESTIMATORS = frozenset(
    {
        "CallawaySantAnna",
        "SunAbraham",
        "ImputationDiD",
        "TwoStageDiD",
        "StackedDiD",
        "EfficientDiD",
    }
)

# Estimators that need a derived `post` column when using survey DGP
# (survey DGP produces `period`/`first_treat` but not `post`).
_SURVEY_POST_ESTIMATORS = frozenset({"DifferenceInDifferences", "TwoWayFixedEffects"})

# Survey fit kwargs builder lookup — maps estimator name to builder function.
_SURVEY_FIT_BUILDERS: Dict[str, Callable] = {
    "DifferenceInDifferences": _survey_basic_fit_kwargs,
    "TwoWayFixedEffects": _survey_twfe_fit_kwargs,
    "MultiPeriodDiD": _survey_multiperiod_fit_kwargs,
    **{name: _survey_staggered_fit_kwargs for name in _STAGGERED_ESTIMATORS},
}

# Unsupported: factor-model and triple-diff estimators (survey DGP produces
# staggered cohort data, not factor-model or 2x2x2 data).
_SURVEY_UNSUPPORTED = frozenset({"TROP", "SyntheticDiD", "TripleDifference"})


def _check_staggered_dgp_compat(
    estimator: Any,
    data_generator_kwargs: Optional[Dict[str, Any]],
) -> None:
    """Warn if a staggered estimator's settings don't match the default DGP."""
    name = type(estimator).__name__
    if name not in _STAGGERED_ESTIMATORS:
        return

    dgp_overrides = data_generator_kwargs or {}
    cohort_periods = dgp_overrides.get("cohort_periods")
    has_multi_cohort = cohort_periods is not None and len(set(cohort_periods)) >= 2
    issues: List[str] = []

    # Check control_group="not_yet_treated" (CS, SA)
    cg = getattr(estimator, "control_group", "never_treated")
    if cg == "not_yet_treated" and not has_multi_cohort:
        issues.append(
            f'  - {name} has control_group="not_yet_treated" but the default '
            f"DGP generates a single treatment cohort with never-treated "
            f"controls. Power may not reflect the intended not-yet-treated "
            f"design.\n"
            f"    Fix: pass data_generator_kwargs="
            f'{{"cohort_periods": [2, 4], "never_treated_frac": 0.0}} '
            f"(or a custom data_generator)."
        )

    # Check anticipation > 0 (all staggered)
    antic = getattr(estimator, "anticipation", 0)
    if antic > 0:
        issues.append(
            f"  - {name} has anticipation={antic} but the default DGP does "
            f"not model anticipatory effects. The estimator will look for "
            f"treatment effects {antic} period(s) before the DGP generates "
            f"them, biasing power estimates.\n"
            f"    Fix: supply a custom data_generator that shifts the "
            f"effect onset."
        )

    # Check clean_control on StackedDiD
    if name == "StackedDiD":
        cc = getattr(estimator, "clean_control", "not_yet_treated")
        if cc == "strict" and not has_multi_cohort:
            issues.append(
                '  - StackedDiD has clean_control="strict" but the default '
                "single-cohort DGP makes strict controls equivalent to "
                "never-treated controls.\n"
                "    Fix: pass data_generator_kwargs="
                '{"cohort_periods": [2, 4]} '
                "to test true strict clean-control behavior."
            )

    if issues:
        msg = (
            f"Staggered power DGP mismatch for {name}. The default "
            f"single-cohort DGP may not match the estimator "
            f"configuration:\n" + "\n".join(issues)
        )
        warnings.warn(msg, UserWarning, stacklevel=2)


def _ddd_effective_n(
    n_units: int, data_generator_kwargs: Optional[Dict[str, Any]]
) -> Optional[int]:
    """Return effective DDD sample size, or None if no rounding occurred."""
    overrides = data_generator_kwargs or {}
    if "n_per_cell" in overrides:
        eff = overrides["n_per_cell"] * 8
    else:
        eff = max(2, n_units // 8) * 8
    return eff if eff != n_units else None


def _ddd_panel_cells_populated(n: int, group_frac: float, partition_frac: float) -> bool:
    """Whether ``generate_ddd_panel_data`` would populate all 4 (group,partition)
    cells at ``n`` units. Mirrors the rounded stratified allocation + non-empty
    validation in ``prep_dgp.generate_ddd_panel_data`` (kept in lockstep)."""
    if n < 4:
        return False
    n_g1 = int(round(n * group_frac))
    n_g0 = n - n_g1
    n_p1_g0 = int(round(n_g0 * partition_frac))
    n_p1_g1 = int(round(n_g1 * partition_frac))
    cells = (n_g0 - n_p1_g0, n_p1_g0, n_g1 - n_p1_g1, n_p1_g1)
    return min(cells) >= 1


def _ddd_panel_viable_min_n(
    group_frac: float, partition_frac: float, floor: int = 16, search_max: int = 100000
) -> int:
    """Smallest ``n_units`` for which ``generate_ddd_panel_data`` populates all
    four (group,partition) cells under the given split, floored at ``floor``.

    For the balanced default (0.5/0.5) this is 4, so the result is ``floor``;
    skewed splits (e.g. 0.1/0.1) need more units before every cell is non-empty,
    so the sample-size search must bracket above this value (the registry
    documents group_frac/partition_frac as data_generator_kwargs overrides)."""
    # Validate up front (matching generate_ddd_panel_data) so an out-of-range
    # split raises the same clear message here — before the bracketing logic can
    # surface a misleading "n_range below the minimum" error downstream.
    if not (0.0 < group_frac < 1.0):
        raise ValueError(f"group_frac must be in (0, 1); got {group_frac}.")
    if not (0.0 < partition_frac < 1.0):
        raise ValueError(f"partition_frac must be in (0, 1); got {partition_frac}.")
    for n in range(4, search_max + 1):
        if _ddd_panel_cells_populated(n, group_frac, partition_frac):
            return max(floor, n)
    raise ValueError(
        f"No panel-DDD sample size <= {search_max} populates all four "
        f"(group, partition) cells for group_frac={group_frac}, "
        f"partition_frac={partition_frac}; move the split closer to 0.5."
    )


def _check_ddd_dgp_compat(
    n_units: int,
    n_periods: int,
    treatment_fraction: float,
    treatment_period: int,
    data_generator_kwargs: Optional[Dict[str, Any]],
) -> None:
    """Warn when simulation inputs don't match DDD's fixed 2×2×2 design."""
    issues: List[str] = []

    # DDD is a fixed 2-period factorial; n_periods and treatment_period are ignored
    if n_periods != 2:
        issues.append(
            f"n_periods={n_periods} is ignored (DDD uses a fixed " f"2-period design: pre/post)"
        )
    if treatment_period != 1:
        issues.append(
            f"treatment_period={treatment_period} is ignored (DDD "
            f"always treats in the second period)"
        )

    # DDD's 2×2×2 factorial has inherent 50% treatment fraction
    if treatment_fraction != 0.5:
        issues.append(
            f"treatment_fraction={treatment_fraction} is ignored "
            f"(DDD uses a balanced 2×2×2 factorial where 50% of "
            f"groups are treated)"
        )

    # n_units rounding: n_per_cell = max(2, n_units // 8)
    eff_n = _ddd_effective_n(n_units, data_generator_kwargs)
    if eff_n is not None:
        eff_n_per_cell = eff_n // 8
        issues.append(
            f"effective sample size is {eff_n} "
            f"(n_per_cell={eff_n_per_cell} × 8 cells), "
            f"not the requested n_units={n_units}"
        )

    if issues:
        warnings.warn(
            "TripleDifference uses a fixed 2×2×2 factorial DGP "
            "(group × partition × time). "
            + "; ".join(issues)
            + ". Pass a custom data_generator for non-standard DDD designs.",
            UserWarning,
            stacklevel=2,
        )


def _check_ddd_panel_dgp_compat(
    estimator: Any,
    treatment_fraction: float,
    data_generator_kwargs: Optional[Dict[str, Any]],
) -> None:
    """Compat checks for the panel DDD power path (``n_periods > 2``).

    Unlike the cross-sectional ``_check_ddd_dgp_compat``, ``n_periods`` and
    ``treatment_period`` are honored here (no warning). ``treatment_fraction``
    is still inert (the panel DGP is a balanced 2×2×2). The key addition is the
    clustering caveat: ``generate_ddd_panel_data`` has within-unit serial
    correlation, so unclustered SEs overstate power.
    """
    overrides = data_generator_kwargs or {}
    if "n_per_cell" in overrides:
        raise ValueError(
            "data_generator_kwargs contains 'n_per_cell', a cross-sectional "
            "generate_ddd_data parameter. The panel DDD power path "
            "(n_periods > 2) uses generate_ddd_panel_data, which sizes the panel "
            "by n_units directly. Control the design via n_units and the "
            "group_frac / partition_frac data_generator_kwargs instead."
        )

    if treatment_fraction != 0.5:
        warnings.warn(
            f"treatment_fraction={treatment_fraction} is ignored for "
            f"TripleDifference power: generate_ddd_panel_data uses a balanced "
            f"2×2×2 design (group_frac=partition_frac=0.5). Pass group_frac / "
            f"partition_frac via data_generator_kwargs to vary the split.",
            UserWarning,
            stacklevel=2,
        )

    # The panel DGP hard-names its unit column "unit", and TripleDifference
    # resolves clustering from `self.cluster` against a same-named data column,
    # so on this auto-DGP path the only correct value is the literal "unit" —
    # this is NOT a general clustering check.
    if getattr(estimator, "cluster", None) != "unit":
        warnings.warn(
            "TripleDifference power on the panel DGP (n_periods > 2) has "
            "within-unit serial correlation, so unclustered standard errors are "
            "anti-conservative and overstate power. Construct the estimator as "
            'TripleDifference(cluster="unit") so the reported power reflects '
            "cluster-robust (Liang-Zeger CR1) inference.",
            UserWarning,
            stacklevel=2,
        )


def _check_sdid_placebo_data(
    data: pd.DataFrame,
    estimator: Any,
    est_kwargs: Dict[str, Any],
) -> None:
    """Check SyntheticDiD placebo feasibility on realized data.

    This catches infeasible designs on the custom-DGP path where the
    pre-generation check (which uses ``n_units * treatment_fraction``)
    cannot run because treatment allocation is determined by the DGP.
    """
    vm = getattr(estimator, "variance_method", "placebo")
    if vm != "placebo":
        return

    treat_col = est_kwargs.get("treatment", "treat")
    unit_col = est_kwargs.get("unit", "unit")

    if treat_col not in data.columns or unit_col not in data.columns:
        return  # fit will fail with a more specific error

    unit_treat = data.groupby(unit_col)[treat_col].first()
    n_treated = int(unit_treat.sum())
    n_control = len(unit_treat) - n_treated

    if n_control <= n_treated:
        raise ValueError(
            f"SyntheticDiD placebo variance requires more control than "
            f"treated units, but the generated data has n_control={n_control}, "
            f"n_treated={n_treated}. Either adjust your data_generator so that "
            f"n_control > n_treated, or use "
            f"SyntheticDiD(variance_method='bootstrap') (paper-faithful refit; "
            f"~5-30x slower than placebo) or SyntheticDiD(variance_method='jackknife')."
        )


# -- Registry construction (deferred to avoid import-time cost) ---------------

_ESTIMATOR_REGISTRY: Optional[Dict[str, _EstimatorProfile]] = None


def _get_registry() -> Dict[str, _EstimatorProfile]:
    """Lazily build and return the estimator registry."""
    global _ESTIMATOR_REGISTRY  # noqa: PLW0603
    if _ESTIMATOR_REGISTRY is not None:
        return _ESTIMATOR_REGISTRY

    from diff_diff.prep import (
        generate_ddd_data,
        generate_did_data,
        generate_factor_data,
        generate_staggered_data,
    )

    _ESTIMATOR_REGISTRY = {
        # --- Basic DiD group ---
        "DifferenceInDifferences": _EstimatorProfile(
            default_dgp=generate_did_data,
            dgp_kwargs_builder=_basic_dgp_kwargs,
            fit_kwargs_builder=_basic_fit_kwargs,
            result_extractor=_extract_simple,
            min_n=20,
        ),
        "TwoWayFixedEffects": _EstimatorProfile(
            default_dgp=generate_did_data,
            dgp_kwargs_builder=_basic_dgp_kwargs,
            fit_kwargs_builder=_twfe_fit_kwargs,
            result_extractor=_extract_simple,
            min_n=20,
        ),
        "MultiPeriodDiD": _EstimatorProfile(
            default_dgp=generate_did_data,
            dgp_kwargs_builder=_basic_dgp_kwargs,
            fit_kwargs_builder=_multiperiod_fit_kwargs,
            result_extractor=_extract_multiperiod,
            min_n=20,
        ),
        # --- Staggered group ---
        "CallawaySantAnna": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        "SunAbraham": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        "ImputationDiD": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        "TwoStageDiD": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        "StackedDiD": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        "EfficientDiD": _EstimatorProfile(
            default_dgp=generate_staggered_data,
            dgp_kwargs_builder=_staggered_dgp_kwargs,
            fit_kwargs_builder=_staggered_fit_kwargs,
            result_extractor=_extract_staggered,
            min_n=40,
        ),
        # --- Factor model group ---
        "TROP": _EstimatorProfile(
            default_dgp=generate_factor_data,
            dgp_kwargs_builder=_factor_dgp_kwargs,
            fit_kwargs_builder=_trop_fit_kwargs,
            result_extractor=_extract_simple,
            min_n=30,
        ),
        "SyntheticDiD": _EstimatorProfile(
            default_dgp=generate_factor_data,
            dgp_kwargs_builder=_factor_dgp_kwargs,
            fit_kwargs_builder=_sdid_fit_kwargs,
            result_extractor=_extract_simple,
            min_n=30,
        ),
        # --- Triple difference ---
        "TripleDifference": _EstimatorProfile(
            default_dgp=generate_ddd_data,
            dgp_kwargs_builder=_ddd_dgp_kwargs,
            fit_kwargs_builder=_ddd_fit_kwargs,
            result_extractor=_extract_simple,
            min_n=64,
        ),
    }
    return _ESTIMATOR_REGISTRY


def _ddd_panel_profile() -> "_EstimatorProfile":
    """Profile for panel DDD power simulations (n_periods > 2).

    Routes TripleDifference power analysis to ``generate_ddd_panel_data``
    (which honors ``n_periods``/``treatment_period``) instead of the
    cross-sectional 2x2x2 ``generate_ddd_data``. See ``simulate_power``.
    """
    from diff_diff.prep import generate_ddd_panel_data

    # min_n=16 is intentionally lower than the cross-sectional DDD min_n=64
    # (which counts 8 cells x n_per_cell); the panel DGP maps n_units directly
    # and only requires n_units >= 4, so 16 gives >=1 unit per (group,partition)
    # cell with margin.
    return _EstimatorProfile(
        default_dgp=generate_ddd_panel_data,
        dgp_kwargs_builder=_ddd_panel_dgp_kwargs,
        fit_kwargs_builder=_ddd_panel_fit_kwargs,
        result_extractor=_extract_simple,
        min_n=16,
    )


@dataclass
class PowerResults(Diagnostic):
    """
    Results from analytical power analysis.

    Attributes
    ----------
    power : float
        Statistical power (probability of rejecting H0 when effect exists).
    mde : float
        Minimum detectable effect size.
    required_n : int
        Required total sample size (treated + control).
    effect_size : float
        Effect size used in calculation.
    alpha : float
        Significance level.
    alternative : str
        Alternative hypothesis ('two-sided', 'greater', 'less').
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    n_pre : int
        Number of pre-treatment periods.
    n_post : int
        Number of post-treatment periods.
    sigma : float
        Residual standard deviation.
    rho : float
        Within-unit (serial) equicorrelation (Burlig 2020 Eq. 2 equicorrelated case).
    deff : float
        Survey design effect (variance inflation factor).
    design : str
        Study design type ('basic_did', 'panel', 'staggered').
    """

    power: float
    mde: float
    required_n: int
    effect_size: float
    alpha: float
    alternative: str
    n_treated: int
    n_control: int
    n_pre: int
    n_post: int
    sigma: float
    rho: float = 0.0
    deff: float = 1.0
    design: str = "basic_did"

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"PowerResults(power={self.power:.3f}, mde={self.mde:.4f}, "
            f"required_n={self.required_n})"
        )

    def summary(self) -> str:
        """
        Generate a formatted summary of power analysis results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 60,
            "Power Analysis for Difference-in-Differences".center(60),
            "=" * 60,
            "",
            f"{'Design:':<30} {self.design}",
            f"{'Significance level (alpha):':<30} {self.alpha:.3f}",
            f"{'Alternative hypothesis:':<30} {self.alternative}",
            "",
            "-" * 60,
            "Sample Size".center(60),
            "-" * 60,
            f"{'Treated units:':<30} {self.n_treated:>10}",
            f"{'Control units:':<30} {self.n_control:>10}",
            f"{'Total units:':<30} {self.n_treated + self.n_control:>10}",
            f"{'Pre-treatment periods:':<30} {self.n_pre:>10}",
            f"{'Post-treatment periods:':<30} {self.n_post:>10}",
            "",
            "-" * 60,
            "Variance Parameters".center(60),
            "-" * 60,
            f"{'Residual SD (sigma):':<30} {self.sigma:>10.4f}",
            f"{'Within-unit equicorrelation:':<30} {self.rho:>10.4f}",
            *([f"{'Design effect (DEFF):':<30} {self.deff:>10.4f}"] if self.deff != 1.0 else []),
            "",
            "-" * 60,
            "Power Analysis Results".center(60),
            "-" * 60,
            f"{'Effect size:':<30} {self.effect_size:>10.4f}",
            f"{'Power:':<30} {self.power:>10.1%}",
            f"{'Minimum detectable effect:':<30} {self.mde:>10.4f}",
            f"{'Required sample size:':<30} {self.required_n:>10}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all power analysis results.
        """
        return {
            "power": self.power,
            "mde": self.mde,
            "required_n": self.required_n,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "sigma": self.sigma,
            "rho": self.rho,
            "deff": self.deff,
            "design": self.design,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with power analysis results.
        """
        return pd.DataFrame([self.to_dict()])


@dataclass
class SimulationPowerResults(Diagnostic):
    """
    Results from simulation-based power analysis.

    Attributes
    ----------
    power : float
        Estimated power (proportion of simulations rejecting H0).
    power_se : float
        Standard error of power estimate.
    power_ci : Tuple[float, float]
        Confidence interval for power estimate.
    rejection_rate : float
        Proportion of simulations with p-value < alpha.
    mean_estimate : float
        Mean treatment effect estimate across simulations.
    std_estimate : float
        Standard deviation of estimates across simulations.
    mean_se : float
        Mean standard error across simulations.
    coverage : float
        Proportion of CIs containing true effect.
    n_simulations : int
        Number of simulations performed (successful count; see
        ``n_simulation_failures`` for failed-replicate count).
    n_simulation_failures : int
        Number of simulations at the primary effect size whose `estimator.fit`
        (or result extraction) raised an exception and was skipped. Lets
        callers programmatically detect fragile DGP/estimator pairings; a
        proportional warning is also emitted above a 10% failure rate.
    effect_sizes : List[float]
        Effect sizes tested (if multiple).
    powers : List[float]
        Power at each effect size (if multiple).
    true_effect : float
        True treatment effect used in simulation.
    alpha : float
        Significance level.
    estimator_name : str
        Name of the estimator used.
    effective_n_units : int or None
        Effective sample size when it differs from the requested ``n_units``
        (e.g., due to DDD grid rounding). ``None`` when no rounding occurred.
    """

    power: float
    power_se: float
    power_ci: Tuple[float, float]
    rejection_rate: float
    mean_estimate: float
    std_estimate: float
    mean_se: float
    coverage: float
    n_simulations: int
    effect_sizes: List[float]
    powers: List[float]
    true_effect: float
    alpha: float
    estimator_name: str
    bias: float = field(init=False)
    rmse: float = field(init=False)
    simulation_results: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    effective_n_units: Optional[int] = None
    survey_config: Optional[Any] = field(default=None, repr=False)
    mean_deff: Optional[float] = None
    mean_icc_realized: Optional[float] = None
    n_simulation_failures: int = 0

    def __post_init__(self):
        """Compute derived statistics."""
        self.bias = self.mean_estimate - self.true_effect
        self.rmse = np.sqrt(self.bias**2 + self.std_estimate**2)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"SimulationPowerResults(power={self.power:.3f} "
            f"[{self.power_ci[0]:.3f}, {self.power_ci[1]:.3f}], "
            f"n_simulations={self.n_simulations})"
        )

    def summary(self) -> str:
        """
        Generate a formatted summary of simulation power results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 65,
            "Simulation-Based Power Analysis Results".center(65),
            "=" * 65,
            "",
            f"{'Estimator:':<35} {self.estimator_name}",
            f"{'Number of simulations:':<35} {self.n_simulations}",
            f"{'Simulation failures:':<35} {self.n_simulation_failures}",
            f"{'True treatment effect:':<35} {self.true_effect:.4f}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            "",
            "-" * 65,
            "Power Estimates".center(65),
            "-" * 65,
            f"{'Power (rejection rate):':<35} {self.power:.1%}",
            f"{'Standard error:':<35} {self.power_se:.4f}",
            f"{'95% CI:':<35} [{self.power_ci[0]:.3f}, {self.power_ci[1]:.3f}]",
            "",
            "-" * 65,
            "Estimation Performance".center(65),
            "-" * 65,
            f"{'Mean estimate:':<35} {self.mean_estimate:.4f}",
            f"{'Bias:':<35} {self.bias:.4f}",
            f"{'Std. deviation of estimates:':<35} {self.std_estimate:.4f}",
            f"{'RMSE:':<35} {self.rmse:.4f}",
            f"{'Mean standard error:':<35} {self.mean_se:.4f}",
            f"{'Coverage (CI contains true):':<35} {self.coverage:.1%}",
        ]
        if self.effective_n_units is not None:
            lines.append(
                f"{'Effective sample size:':<35} {self.effective_n_units}" f" (DDD grid-rounded)"
            )
        if self.survey_config is not None:
            lines.extend(
                [
                    "",
                    "-" * 65,
                    "Survey Design".center(65),
                    "-" * 65,
                    f"{'Strata:':<35} {self.survey_config.n_strata}",
                    f"{'PSUs per stratum:':<35} {self.survey_config.psu_per_stratum}",
                ]
            )
            if self.mean_deff is not None:
                lines.append(f"{'Mean Kish DEFF:':<35} {self.mean_deff:.4f}")
            if self.mean_icc_realized is not None:
                lines.append(f"{'Mean realized ICC:':<35} {self.mean_icc_realized:.4f}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation power results.
        """
        d: Dict[str, Any] = {
            "power": self.power,
            "power_se": self.power_se,
            "power_ci_lower": self.power_ci[0],
            "power_ci_upper": self.power_ci[1],
            "rejection_rate": self.rejection_rate,
            "mean_estimate": self.mean_estimate,
            "std_estimate": self.std_estimate,
            "bias": self.bias,
            "rmse": self.rmse,
            "mean_se": self.mean_se,
            "coverage": self.coverage,
            "n_simulations": self.n_simulations,
            "n_simulation_failures": self.n_simulation_failures,
            "true_effect": self.true_effect,
            "alpha": self.alpha,
            "estimator_name": self.estimator_name,
            "effective_n_units": self.effective_n_units,
            "mean_deff": self.mean_deff,
            "mean_icc_realized": self.mean_icc_realized,
        }
        return d

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation power results.
        """
        return pd.DataFrame([self.to_dict()])

    def power_curve_df(self) -> pd.DataFrame:
        """
        Get power curve data as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with effect_size and power columns.
        """
        return pd.DataFrame({"effect_size": self.effect_sizes, "power": self.powers})


class PowerAnalysis:
    """
    Power analysis for difference-in-differences designs.

    Provides analytical power calculations for basic 2x2 DiD and panel DiD
    designs. For complex designs like staggered adoption, use simulate_power()
    instead.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    power : float, default=0.80
        Target statistical power.
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.

    Examples
    --------
    Calculate minimum detectable effect:

    >>> from diff_diff import PowerAnalysis
    >>> pa = PowerAnalysis(alpha=0.05, power=0.80)
    >>> results = pa.mde(n_treated=50, n_control=50, sigma=1.0)
    >>> print(f"MDE: {results.mde:.3f}")

    Calculate required sample size:

    >>> results = pa.sample_size(effect_size=0.5, sigma=1.0)
    >>> print(f"Required N: {results.required_n}")

    Calculate power for given sample and effect:

    >>> results = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
    >>> print(f"Power: {results.power:.1%}")

    Notes
    -----
    The power calculations are based on the variance of the DiD estimator.

    Critical values use the **normal (z)** distribution following Bloom (1995):
    ``MDE = (z_{1-alpha/2} + z_{1-kappa}) * SE``. This is a large-sample
    approximation to Burlig et al.'s t-based multiplier (their Eq. 1) and is
    mildly anti-conservative for very small numbers of units.

    The variance is the **within-unit equicorrelated special case of Burlig,
    Preonas & Woerman (2020), Eq. 2** (psi^B = psi^A = psi^X = rho * sigma^2), for
    m = n_pre pre-periods and r = n_post post-periods::

        Var(ATT) = sigma^2 * (1/N_treated + 1/N_control) * (1/m + 1/r) * (1 - rho)

    where rho is the within-unit (serial) equicorrelation. Cross-period
    correlation **lowers** the DiD variance (differencing cancels the shared
    within-unit component), so the MDE *decreases* as rho increases -- the
    opposite of a Moulton mean-inflation factor.

    The basic 2x2 design (n_pre = n_post = 1) is the m = r = 1 special case
    (Burlig footnote 11), Var(ATT) = 2 * sigma^2 * (1/N_treated + 1/N_control) *
    (1 - rho), reducing to Bloom (1995) Eq. 1's DiD analog at rho = 0.

    The fully general serial-correlation-robust form (independent psi^B, psi^A,
    psi^X) is not implemented; see ``docs/methodology/REGISTRY.md``
    ``## PowerAnalysis`` and the source audits under ``docs/methodology/papers/``.

    References
    ----------
    Bloom, H. S. (1995). "Minimum Detectable Effects."
    Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: str = "two-sided",
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < power < 1:
            raise ValueError("power must be between 0 and 1")
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        self.alpha = alpha
        self.target_power = power
        self.alternative = alternative

    @staticmethod
    def _validate_deff(deff: float) -> None:
        """Validate deff parameter and warn if < 1."""
        if not np.isfinite(deff) or deff <= 0:
            raise ValueError(f"deff must be finite and > 0, got {deff}")
        if deff < 1.0:
            warnings.warn(
                f"deff={deff:.4f} < 1.0 implies net variance reduction "
                f"(e.g., from stratification). This is valid but unusual.",
                stacklevel=3,
            )

    @staticmethod
    def _validate_design_params(n_pre: int, n_post: int, rho: float) -> None:
        """Validate analytical-power inputs for the Burlig (2020) Eq. 2 variance.

        Applies to BOTH the 2x2 (n_pre = n_post = 1) and multi-period panel paths
        -- the 2x2 case is the m = r = 1 special case of the same equicorrelated
        variance. Requires at least one pre- and one post-period, and a within-unit
        equicorrelation ``rho`` in ``[-1/(T-1), 1)`` (T = n_pre + n_post):
        ``rho >= 1`` yields a non-positive residual variance, and
        ``rho < -1/(T-1)`` is not a valid equicorrelation structure.
        """
        if n_pre < 1 or n_post < 1:
            raise ValueError(
                "Power analysis requires n_pre >= 1 and n_post >= 1 (a DiD design "
                f"needs both pre- and post-periods), got n_pre={n_pre}, n_post={n_post}."
            )
        T = n_pre + n_post
        rho_min = -1.0 / (T - 1)
        if not rho_min <= rho < 1.0:
            raise ValueError(
                f"rho must lie in [{rho_min:.4g}, 1) (valid within-unit "
                f"equicorrelation over T={T} periods; rho >= 1 implies zero "
                f"residual variance), got rho={rho}."
            )

    def _get_critical_values(self) -> Tuple[float, float]:
        """Get normal (z) critical values for alpha and power.

        Uses standard-normal quantiles (the Bloom 1995 multiplier) -- a
        large-sample approximation to Burlig et al. (2020) Eq. 1's t-based
        multiplier.
        """
        if self.alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        z_beta = stats.norm.ppf(self.target_power)
        return z_alpha, z_beta

    def _compute_variance(
        self,
        n_treated: int,
        n_control: int,
        n_pre: int,
        n_post: int,
        sigma: float,
        rho: float = 0.0,
        deff: float = 1.0,
        design: str = "basic_did",
    ) -> float:
        """
        Compute variance of the DiD estimator.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        n_pre : int
            Number of pre-treatment periods.
        n_post : int
            Number of post-treatment periods.
        sigma : float
            Residual standard deviation.
        rho : float
            Within-unit (serial) equicorrelation for the panel design (Burlig
            2020 Eq. 2, equicorrelated case); higher rho lowers the variance.
        deff : float
            Survey design effect (variance inflation factor). Not redundant
            with ``rho``: ``rho`` models within-unit (serial) equicorrelation
            (Burlig 2020 Eq. 2 ``(1/m+1/r)(1-rho)`` factor), ``deff`` models
            survey clustering/weighting.
        design : str
            Study design type.

        Returns
        -------
        float
            Variance of the DiD estimator.
        """
        # Validate inputs before routing so invalid two-period shapes (e.g.
        # n_pre=0) and out-of-range rho cannot fall through to basic_did silently.
        self._validate_design_params(n_pre, n_post, rho)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma (residual SD) must be finite and >= 0, got {sigma}")
        if n_treated <= 0 or n_control <= 0:
            raise ValueError(
                "n_treated and n_control must be > 0, got "
                f"n_treated={n_treated}, n_control={n_control}"
            )
        if design == "basic_did":
            # 2x2 DiD (n_pre = n_post = 1): the m = r = 1 special case of the
            # equicorrelated Burlig (2020) Eq. 2 variance (footnote 11 drops the
            # within-pre / within-post covariance terms, leaving the cross-period
            # term). Reduces to Bloom (1995) Eq. 1's DiD analog
            # 2 * sigma^2 * (1/n_T + 1/n_C) at rho = 0; the (1 - rho) factor
            # applies the correlation between the single pre and post observation.
            n_t_pre = n_treated  # treated units in pre-period
            n_t_post = n_treated  # treated units in post-period
            n_c_pre = n_control
            n_c_post = n_control

            cell_factor = 1 / n_t_post + 1 / n_t_pre + 1 / n_c_post + 1 / n_c_pre
            variance = sigma**2 * cell_factor * (1 - rho)
        elif design == "panel":
            # Burlig, Preonas & Woerman (2020), Eq. 2, specialized to within-unit
            # equicorrelation (psi^B = psi^A = psi^X = rho * sigma^2):
            #     Var(ATT) = sigma^2 (1/n_T + 1/n_C) (1/m + 1/r) (1 - rho)
            # with m = n_pre, r = n_post. Cross-period correlation (rho) LOWERS the
            # DiD variance because differencing cancels the shared within-unit
            # component -- the opposite sign of a Moulton mean-inflation factor.
            period_factor = 1 / n_pre + 1 / n_post  # = (m + r) / (m * r)
            base_var = sigma**2 * (1 / n_treated + 1 / n_control)
            variance = base_var * period_factor * (1 - rho)
        else:
            raise ValueError(f"Unknown design: {design}")

        # Survey design effect (multiplicative variance inflation)
        variance *= deff

        return variance

    def power(
        self,
        effect_size: float,
        n_treated: int,
        n_control: int,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        deff: float = 1.0,
    ) -> PowerResults:
        """
        Calculate statistical power for given effect size and sample.

        Parameters
        ----------
        effect_size : float
            Expected treatment effect size.
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Within-unit (serial) equicorrelation for panel designs. Higher rho
            LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
            valid range [-1/(T-1), 1).
        deff : float, default=1.0
            Survey design effect (variance inflation factor). Not redundant
            with ``rho``: ``rho`` models within-unit serial correlation,
            ``deff`` models survey clustering/weighting.

        Returns
        -------
        PowerResults
            Power analysis results.

        Examples
        --------
        >>> pa = PowerAnalysis()
        >>> results = pa.power(effect_size=2.0, n_treated=50, n_control=50, sigma=5.0)
        >>> print(f"Power: {results.power:.1%}")
        """
        self._validate_deff(deff)
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, deff=deff, design=design
        )
        se = np.sqrt(variance)

        # Calculate power
        if self.alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            # Power = P(reject | effect) = P(|Z| > z_alpha | effect)
            power_val = (
                1
                - stats.norm.cdf(z_alpha - effect_size / se)
                + stats.norm.cdf(-z_alpha - effect_size / se)
            )
        elif self.alternative == "greater":
            z_alpha = stats.norm.ppf(1 - self.alpha)
            power_val = 1 - stats.norm.cdf(z_alpha - effect_size / se)
        else:  # less
            z_alpha = stats.norm.ppf(1 - self.alpha)
            power_val = stats.norm.cdf(-z_alpha - effect_size / se)

        # Also compute MDE and required N for reference
        mde = self._compute_mde_from_se(se)
        required_n = self._compute_required_n(
            effect_size,
            sigma,
            n_pre,
            n_post,
            rho,
            design,
            n_treated / (n_treated + n_control),
            deff=deff,
        )

        return PowerResults(
            power=power_val,
            mde=mde,
            required_n=required_n,
            effect_size=effect_size,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            deff=deff,
            design=design,
        )

    def _compute_mde_from_se(self, se: float) -> float:
        """Compute MDE given standard error."""
        z_alpha, z_beta = self._get_critical_values()
        return (z_alpha + z_beta) * se

    def mde(
        self,
        n_treated: int,
        n_control: int,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        deff: float = 1.0,
    ) -> PowerResults:
        """
        Calculate minimum detectable effect given sample size.

        The MDE is the smallest effect size that can be detected with the
        specified power and significance level.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Within-unit (serial) equicorrelation for panel designs. Higher rho
            LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
            valid range [-1/(T-1), 1).
        deff : float, default=1.0
            Survey design effect (variance inflation factor).

        Returns
        -------
        PowerResults
            Power analysis results including MDE.

        Examples
        --------
        >>> pa = PowerAnalysis(power=0.80)
        >>> results = pa.mde(n_treated=100, n_control=100, sigma=10.0)
        >>> print(f"MDE: {results.mde:.2f}")
        """
        self._validate_deff(deff)
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, deff=deff, design=design
        )
        se = np.sqrt(variance)

        mde = self._compute_mde_from_se(se)

        return PowerResults(
            power=self.target_power,
            mde=mde,
            required_n=n_treated + n_control,
            effect_size=mde,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            deff=deff,
            design=design,
        )

    def _compute_required_n(
        self,
        effect_size: float,
        sigma: float,
        n_pre: int,
        n_post: int,
        rho: float,
        design: str,
        treat_frac: float = 0.5,
        deff: float = 1.0,
    ) -> int:
        """Compute required sample size for given effect.

        Note: this method has its own formula independent of _compute_variance,
        so deff must be applied here separately (not double-counting).
        """
        # Validate inputs before routing (mirrors _compute_variance).
        self._validate_design_params(n_pre, n_post, rho)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma (residual SD) must be finite and >= 0, got {sigma}")
        if not 0 < treat_frac < 1:
            raise ValueError(f"treat_frac must be in (0, 1), got {treat_frac}")

        # Handle edge case of zero effect size
        if effect_size == 0:
            return MAX_SAMPLE_SIZE  # Can't detect zero effect

        z_alpha, z_beta = self._get_critical_values()

        if design == "basic_did":
            # 2x2 DiD = the m = r = 1 equicorrelated case (period factor
            # 1/1 + 1/1 = 2); the (1 - rho) factor mirrors _compute_variance and
            # reduces to Bloom's 2 * sigma^2 * (z..)^2 / (delta^2 f(1-f)) at rho = 0.
            n_total = (
                2
                * sigma**2
                * (z_alpha + z_beta) ** 2
                * (1 - rho)
                / (effect_size**2 * treat_frac * (1 - treat_frac))
            )
        else:  # panel
            # Burlig (2020) Eq. 2 (equicorrelated), inverted for required N.
            # period_factor = 1/n_pre + 1/n_post equals 2 at n_pre=n_post=1, so this
            # is continuous with the basic_did branch.
            period_factor = 1 / n_pre + 1 / n_post
            n_total = (
                sigma**2
                * (z_alpha + z_beta) ** 2
                * period_factor
                * (1 - rho)
                / (effect_size**2 * treat_frac * (1 - treat_frac))
            )

        # Survey design effect (multiplicative sample size inflation)
        n_total *= deff

        # Handle infinity case (extremely small effect)
        if np.isinf(n_total):
            return MAX_SAMPLE_SIZE

        return max(4, int(np.ceil(n_total)))  # At least 4 units

    def sample_size(
        self,
        effect_size: float,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        treat_frac: float = 0.5,
        deff: float = 1.0,
    ) -> PowerResults:
        """
        Calculate required sample size to detect given effect.

        Parameters
        ----------
        effect_size : float
            Treatment effect to detect.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Within-unit (serial) equicorrelation for panel designs. Higher rho
            LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
            valid range [-1/(T-1), 1).
        treat_frac : float, default=0.5
            Fraction of units assigned to treatment.
        deff : float, default=1.0
            Survey design effect (variance inflation factor).

        Returns
        -------
        PowerResults
            Power analysis results including required sample size.

        Examples
        --------
        >>> pa = PowerAnalysis(power=0.80)
        >>> results = pa.sample_size(effect_size=5.0, sigma=10.0)
        >>> print(f"Required N: {results.required_n}")
        """
        self._validate_deff(deff)
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        n_total = self._compute_required_n(
            effect_size, sigma, n_pre, n_post, rho, design, treat_frac, deff=deff
        )

        n_treated = max(2, int(np.ceil(n_total * treat_frac)))
        n_control = max(2, n_total - n_treated)
        n_total = n_treated + n_control

        # Compute actual power achieved
        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, deff=deff, design=design
        )
        se = np.sqrt(variance)
        mde = self._compute_mde_from_se(se)

        return PowerResults(
            power=self.target_power,
            mde=mde,
            required_n=n_total,
            effect_size=effect_size,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            deff=deff,
            design=design,
        )

    def power_curve(
        self,
        n_treated: int,
        n_control: int,
        sigma: float,
        effect_sizes: Optional[List[float]] = None,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        deff: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute power for a range of effect sizes.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        effect_sizes : list of float, optional
            Effect sizes to evaluate. If None, uses a range from 0 to 3*MDE.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Within-unit (serial) equicorrelation for panel designs. Higher rho
            LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
            valid range [-1/(T-1), 1).
        deff : float, default=1.0
            Survey design effect (variance inflation factor).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'effect_size' and 'power'.

        Examples
        --------
        >>> pa = PowerAnalysis()
        >>> curve = pa.power_curve(n_treated=50, n_control=50, sigma=5.0)
        >>> print(curve)
        """
        # First get MDE to determine default range
        mde_result = self.mde(n_treated, n_control, sigma, n_pre, n_post, rho, deff=deff)

        if effect_sizes is None:
            # Generate range from 0 to 2*MDE
            effect_sizes = np.linspace(0, 2.5 * mde_result.mde, 50).tolist()

        powers = []
        for es in effect_sizes:
            result = self.power(
                effect_size=es,
                n_treated=n_treated,
                n_control=n_control,
                sigma=sigma,
                n_pre=n_pre,
                n_post=n_post,
                rho=rho,
                deff=deff,
            )
            powers.append(result.power)

        return pd.DataFrame({"effect_size": effect_sizes, "power": powers})

    def sample_size_curve(
        self,
        effect_size: float,
        sigma: float,
        sample_sizes: Optional[List[int]] = None,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        treat_frac: float = 0.5,
        deff: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute power for a range of sample sizes.

        Parameters
        ----------
        effect_size : float
            Treatment effect size.
        sigma : float
            Residual standard deviation.
        sample_sizes : list of int, optional
            Total sample sizes to evaluate. If None, uses sensible range.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Within-unit (serial) equicorrelation for panel designs. Higher rho
            LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
            valid range [-1/(T-1), 1).
        treat_frac : float, default=0.5
            Fraction assigned to treatment.
        deff : float, default=1.0
            Survey design effect (variance inflation factor).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'sample_size' and 'power'.
        """
        # Get required N to determine default range
        required = self.sample_size(effect_size, sigma, n_pre, n_post, rho, treat_frac, deff=deff)

        if sample_sizes is None:
            min_n = max(10, required.required_n // 4)
            max_n = required.required_n * 2
            sample_sizes = list(range(min_n, max_n + 1, max(1, (max_n - min_n) // 50)))

        powers = []
        for n in sample_sizes:
            n_treated = max(2, int(n * treat_frac))
            n_control = max(2, n - n_treated)
            result = self.power(
                effect_size=effect_size,
                n_treated=n_treated,
                n_control=n_control,
                sigma=sigma,
                n_pre=n_pre,
                n_post=n_post,
                rho=rho,
                deff=deff,
            )
            powers.append(result.power)

        return pd.DataFrame({"sample_size": sample_sizes, "power": powers})


def simulate_power(
    estimator: Any,
    n_units: int = 100,
    n_periods: int = 4,
    treatment_effect: float = 5.0,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    sigma: float = 1.0,
    n_simulations: int = 500,
    alpha: float = 0.05,
    effect_sizes: Optional[List[float]] = None,
    seed: Optional[int] = None,
    data_generator: Optional[Callable] = None,
    data_generator_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    result_extractor: Optional[Callable] = None,
    progress: bool = True,
    survey_config: Optional[SurveyPowerConfig] = None,
) -> SimulationPowerResults:
    """
    Estimate power using Monte Carlo simulation.

    This function simulates datasets with known treatment effects and estimates
    power as the fraction of simulations where the null hypothesis is rejected.
    Most built-in estimators are supported via an internal registry that selects
    the appropriate data-generating process and fit signature automatically.

    Parameters
    ----------
    estimator : estimator object
        DiD estimator to use (e.g., DifferenceInDifferences, CallawaySantAnna).
    n_units : int, default=100
        Number of units per simulation.
    n_periods : int, default=4
        Number of time periods.
    treatment_effect : float, default=5.0
        True treatment effect to simulate.
    treatment_fraction : float, default=0.5
        Fraction of units that are treated.
    treatment_period : int, default=2
        First post-treatment period (0-indexed).
    sigma : float, default=1.0
        Residual standard deviation (noise level).
    n_simulations : int, default=500
        Number of Monte Carlo simulations.
    alpha : float, default=0.05
        Significance level for hypothesis tests.
    effect_sizes : list of float, optional
        Multiple effect sizes to evaluate for power curve.
        If None, uses only treatment_effect.
    seed : int, optional
        Random seed for reproducibility.
    data_generator : callable, optional
        Custom data generation function. When provided, bypasses the
        registry DGP and calls this function with the standard kwargs
        (n_units, n_periods, treatment_effect, etc.).
    data_generator_kwargs : dict, optional
        Additional keyword arguments for data generator.
    estimator_kwargs : dict, optional
        Additional keyword arguments for estimator.fit().
    result_extractor : callable, optional
        Custom function to extract results from the estimator output.
        Takes the estimator result object and returns a tuple of
        ``(att, se, p_value, conf_int)``. Useful for unregistered
        estimators with non-standard result schemas.
    progress : bool, default=True
        Whether to print progress updates.
    survey_config : SurveyPowerConfig, optional
        When provided, generates survey-structured data via
        ``generate_survey_did_data`` and injects ``SurveyDesign`` into
        estimator ``fit()``. Mutually exclusive with ``data_generator``.
        Supported estimators: DiD, TWFE, MultiPeriod, CS, SA, Imputation,
        TwoStage, Stacked, Efficient. Unsupported: TROP, SyntheticDiD,
        TripleDifference. ``heterogeneous_te_by_strata`` must be False.

    Returns
    -------
    SimulationPowerResults
        Simulation-based power analysis results.

    Examples
    --------
    Basic power simulation:

    >>> from diff_diff import DifferenceInDifferences, simulate_power
    >>> did = DifferenceInDifferences()
    >>> results = simulate_power(
    ...     estimator=did,
    ...     n_units=100,
    ...     treatment_effect=5.0,
    ...     sigma=5.0,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> print(f"Power: {results.power:.1%}")

    Power curve over multiple effect sizes:

    >>> results = simulate_power(
    ...     estimator=did,
    ...     effect_sizes=[1.0, 2.0, 3.0, 5.0, 7.0],
    ...     n_simulations=200,
    ...     seed=42
    ... )
    >>> print(results.power_curve_df())

    With Callaway-Sant'Anna (auto-detected, no custom DGP needed):

    >>> from diff_diff import CallawaySantAnna
    >>> cs = CallawaySantAnna()
    >>> results = simulate_power(cs, n_simulations=200, seed=42)

    Notes
    -----
    The simulation approach:
    1. Generate data with known treatment effect
    2. Fit the estimator and record the p-value
    3. Repeat n_simulations times
    4. Power = fraction of simulations where p-value < alpha

    The analytical reference formulas this Monte Carlo path complements (the Bloom 1995 normal
    multiplier and the Burlig et al. 2020 Eq. 2 equicorrelated panel variance) are documented in
    ``docs/methodology/REGISTRY.md`` ``## PowerAnalysis``.

    References
    ----------
    Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    """
    rng = np.random.default_rng(seed)

    estimator_name = type(estimator).__name__
    registry = _get_registry()
    profile = registry.get(estimator_name)

    # If no profile and no custom data_generator, raise
    if profile is None and data_generator is None:
        raise ValueError(
            f"Estimator '{estimator_name}' not in registry. "
            f"Provide a custom data_generator and estimator_kwargs "
            f"(the full dict of keyword arguments for estimator.fit(), "
            f"e.g. dict(outcome='y', treatment='treat', time='period'))."
        )

    # When a custom data_generator is provided, bypass registry DGP
    use_custom_dgp = data_generator is not None
    use_survey_dgp = survey_config is not None

    # Route DDD power to the panel DGP when n_periods > 2. The cross-sectional
    # 2x2x2 generate_ddd_data ignores n_periods; generate_ddd_panel_data honors
    # it. Swapping the profile here (before the collision check and the DDD
    # compat warnings) makes every downstream consumer (dgp_kwargs_builder,
    # default_dgp, fit_kwargs_builder, result_extractor, min_n) use the panel
    # variant automatically.
    use_ddd_panel = (
        estimator_name == "TripleDifference"
        and n_periods > 2
        and not use_custom_dgp
        and not use_survey_dgp
    )
    if use_ddd_panel:
        profile = _ddd_panel_profile()

    # --- Survey config validation ---
    if use_survey_dgp:
        assert survey_config is not None  # for type narrowing
        if estimator_name in _SURVEY_UNSUPPORTED:
            raise ValueError(
                f"survey_config is not supported with {estimator_name}. "
                f"generate_survey_did_data produces staggered cohort data "
                f"incompatible with this estimator's DGP. Use the custom "
                f"data_generator path for survey power with {estimator_name}."
            )
        if use_custom_dgp:
            raise ValueError(
                "survey_config and data_generator are mutually exclusive. "
                "survey_config uses generate_survey_did_data internally."
            )
        if treatment_period < 1:
            raise ValueError(
                f"treatment_period must be >= 1 with survey_config "
                f"(need at least one pre-treatment period), got {treatment_period}."
            )
        if estimator_name not in _SURVEY_FIT_BUILDERS:
            raise ValueError(
                f"No survey power profile for {estimator_name}. "
                f"Supported: {sorted(_SURVEY_FIT_BUILDERS.keys())}."
            )
        if survey_config.heterogeneous_te_by_strata:
            raise ValueError(
                "heterogeneous_te_by_strata=True is not supported with "
                "simulation power analysis. The DGP's population ATT diverges "
                "from the input treatment_effect under heterogeneous effects, "
                "which would make bias/coverage/RMSE metrics misleading."
            )

    data_gen_kwargs = data_generator_kwargs or {}
    est_kwargs = estimator_kwargs or {}

    # Block survey_design in estimator_kwargs when survey_config is active.
    # Custom survey design overrides go through SurveyPowerConfig.survey_design.
    if use_survey_dgp and "survey_design" in est_kwargs:
        raise ValueError(
            "estimator_kwargs cannot contain 'survey_design' when survey_config "
            "is set. To override the auto-built SurveyDesign, pass it via "
            "SurveyPowerConfig(survey_design=...)."
        )

    # Block survey-config-managed keys in data_generator_kwargs
    if use_survey_dgp and data_gen_kwargs:
        collisions = _SURVEY_CONFIG_KEYS & set(data_gen_kwargs)
        if collisions:
            raise ValueError(
                f"data_generator_kwargs contains keys managed by survey_config: "
                f"{sorted(collisions)}. Set these on SurveyPowerConfig instead."
            )
        # Block DGP params that make realized ATT diverge from scalar input,
        # which would misstate bias/coverage/RMSE (same rationale as
        # heterogeneous_te_by_strata rejection above).
        te_interaction = data_gen_kwargs.get("te_covariate_interaction", 0.0)
        if te_interaction != 0.0:
            raise ValueError(
                f"te_covariate_interaction={te_interaction} is not supported "
                f"with survey_config. The DGP's population ATT diverges from "
                f"the input treatment_effect under covariate-interaction "
                f"heterogeneity, which would make bias/coverage/RMSE misleading."
            )

    # Enforce panel-mode alignment between DGP and estimator.
    # Runs even with empty data_gen_kwargs to catch CS(panel=False) + default DGP.
    if use_survey_dgp:
        dgp_panel = data_gen_kwargs.get("panel", True)
        est_panel = getattr(estimator, "panel", True)
        if not dgp_panel:
            if estimator_name != "CallawaySantAnna":
                raise ValueError(
                    f"panel=False (repeated cross-sections) is not supported "
                    f"with {estimator_name} under survey_config. Only "
                    f"CallawaySantAnna supports repeated cross-sections."
                )
            if est_panel:
                raise ValueError(
                    "data_generator_kwargs has panel=False but "
                    "CallawaySantAnna.panel=True. Use "
                    "CallawaySantAnna(panel=False) to match."
                )
        elif estimator_name == "CallawaySantAnna" and not est_panel:
            raise ValueError(
                "CallawaySantAnna(panel=False) requires "
                "data_generator_kwargs={'panel': False} to generate "
                "repeated cross-section data."
            )
        # Reject estimator settings that require a multi-cohort DGP.
        # survey_config hard-codes a single-cohort DGP and blocks
        # cohort_periods/never_treated_frac overrides.
        control_group = getattr(estimator, "control_group", "never_treated")
        clean_control = getattr(estimator, "clean_control", None)
        if control_group in ("not_yet_treated", "last_cohort"):
            raise ValueError(
                f"survey_config does not support control_group='{control_group}' "
                "(requires multi-cohort DGP). Use the custom data_generator "
                "path for survey power with this control-group design."
            )
        if clean_control == "strict":
            raise ValueError(
                "survey_config does not support clean_control='strict' "
                "(requires multi-cohort DGP). Use the custom data_generator "
                "path for survey power with strict clean controls."
            )

    # SyntheticDiD placebo variance requires n_control > n_treated.
    # Check after merging data_generator_kwargs so overrides of n_treated
    # are accounted for.
    if estimator_name == "SyntheticDiD" and not use_custom_dgp:
        vm = getattr(estimator, "variance_method", "placebo")
        effective_n_treated = data_gen_kwargs.get(
            "n_treated", max(1, int(n_units * treatment_fraction))
        )
        n_control = n_units - effective_n_treated
        if vm == "placebo" and n_control <= effective_n_treated:
            raise ValueError(
                f"SyntheticDiD placebo variance requires more control than "
                f"treated units (got n_control={n_control}, "
                f"n_treated={effective_n_treated}). Either lower "
                f"treatment_fraction so that n_control > n_treated, or use "
                f"SyntheticDiD(variance_method='bootstrap') (paper-faithful refit; "
                f"~5-30x slower than placebo) or "
                f"SyntheticDiD(variance_method='jackknife')."
            )

    # Warn if staggered estimator settings don't match auto DGP
    if profile is not None and not use_custom_dgp:
        _check_staggered_dgp_compat(estimator, data_generator_kwargs)

    # Block registry-path collisions on search-critical keys
    if profile is not None and not use_custom_dgp and data_gen_kwargs:
        sample_dgp_keys = set(
            profile.dgp_kwargs_builder(
                n_units=n_units,
                n_periods=n_periods,
                treatment_effect=treatment_effect,
                treatment_fraction=treatment_fraction,
                treatment_period=treatment_period,
                sigma=sigma,
            ).keys()
        )
        collisions = _PROTECTED_DGP_KEYS & set(data_gen_kwargs) & sample_dgp_keys
        if collisions:
            raise ValueError(
                f"data_generator_kwargs contains keys that conflict with "
                f"registry-managed simulation inputs: {sorted(collisions)}. "
                f"These are controlled by simulate_power() parameters directly. "
                f"Use the corresponding function parameters instead, or pass a "
                f"custom data_generator to override the DGP entirely."
            )

    # Warn if DDD design inputs are silently ignored
    if use_ddd_panel:
        # Panel path honors n_periods/treatment_period; n_units maps directly
        # (no //8 rounding). Different compat surface than the cross-sectional
        # 2x2x2 design.
        _check_ddd_panel_dgp_compat(estimator, treatment_fraction, data_generator_kwargs)
        effective_n_units = None
    elif estimator_name == "TripleDifference" and not use_custom_dgp:
        _check_ddd_dgp_compat(
            n_units,
            n_periods,
            treatment_fraction,
            treatment_period,
            data_generator_kwargs,
        )
        effective_n_units = _ddd_effective_n(n_units, data_generator_kwargs)
    else:
        effective_n_units = None

    # Determine effect sizes to test
    if effect_sizes is None:
        effect_sizes = [treatment_effect]

    all_powers = []

    # For the primary effect, collect detailed results
    if len(effect_sizes) == 1:
        primary_idx = 0
    else:
        primary_idx = -1
        for i, es in enumerate(effect_sizes):
            if np.isclose(es, treatment_effect):
                primary_idx = i
                break
        if primary_idx == -1:
            primary_idx = len(effect_sizes) - 1

    primary_effect = effect_sizes[primary_idx]

    # Initialize so they are always bound
    primary_estimates: List[float] = []
    primary_ses: List[float] = []
    primary_p_values: List[float] = []
    primary_rejections: List[bool] = []
    primary_ci_contains: List[bool] = []
    primary_n_failures = 0

    # Survey DGP truth accumulation (DEFF/ICC are DGP properties,
    # independent of effect size, so averaging across all sims is correct)
    deff_values: List[float] = []
    icc_values: List[float] = []

    # Lazy import for survey DGP (mirrors registry's lazy import pattern)
    _generate_survey_did_data: Optional[Callable] = None
    if use_survey_dgp:
        from diff_diff.prep import generate_survey_did_data as _generate_survey_did_data

    for effect_idx, effect in enumerate(effect_sizes):
        is_primary = effect_idx == primary_idx

        estimates: List[float] = []
        ses: List[float] = []
        p_values: List[float] = []
        rejections: List[bool] = []
        ci_contains_true: List[bool] = []
        n_failures = 0

        for sim in range(n_simulations):
            if progress and sim % 100 == 0 and sim > 0:
                pct = (sim + effect_idx * n_simulations) / (len(effect_sizes) * n_simulations)
                print(f"  Simulation progress: {pct:.0%}")

            sim_seed = rng.integers(0, 2**31)

            # --- Generate data ---
            if use_survey_dgp:
                assert survey_config is not None
                assert _generate_survey_did_data is not None
                dgp_kwargs = _survey_dgp_kwargs(
                    n_units=n_units,
                    n_periods=n_periods,
                    treatment_effect=effect,
                    treatment_fraction=treatment_fraction,
                    treatment_period=treatment_period,
                    sigma=sigma,
                    survey_config=survey_config,
                )
                dgp_kwargs.update(data_gen_kwargs)
                dgp_kwargs.pop("seed", None)
                data = _generate_survey_did_data(seed=sim_seed, **dgp_kwargs)

                # Derive columns for non-staggered estimators.
                # Survey DGP's `treated` is time-varying (1{g>0, t>=g}); basic/TWFE/
                # MultiPeriod need a time-invariant group indicator (`ever_treated`).
                if estimator_name not in _STAGGERED_ESTIMATORS:
                    data["ever_treated"] = (data["first_treat"] > 0).astype(int)
                # Basic/TWFE also need a `post` period indicator.
                if estimator_name in _SURVEY_POST_ESTIMATORS:
                    data["post"] = (data["period"] >= treatment_period + 1).astype(int)

                # Collect DGP truth for metadata
                dgp_truth = data.attrs.get("dgp_truth", {})
                if dgp_truth:
                    kish = dgp_truth.get("deff_kish")
                    icc_r = dgp_truth.get("icc_realized")
                    if kish is not None:
                        deff_values.append(kish)
                    if icc_r is not None:
                        icc_values.append(icc_r)

            elif use_custom_dgp:
                assert data_generator is not None
                data = data_generator(
                    n_units=n_units,
                    n_periods=n_periods,
                    treatment_effect=effect,
                    treatment_fraction=treatment_fraction,
                    treatment_period=treatment_period,
                    noise_sd=sigma,
                    seed=sim_seed,
                    **data_gen_kwargs,
                )
            else:
                assert profile is not None
                dgp_kwargs = profile.dgp_kwargs_builder(
                    n_units=n_units,
                    n_periods=n_periods,
                    treatment_effect=effect,
                    treatment_fraction=treatment_fraction,
                    treatment_period=treatment_period,
                    sigma=sigma,
                )
                dgp_kwargs.update(data_gen_kwargs)
                dgp_kwargs.pop("seed", None)
                data = profile.default_dgp(seed=sim_seed, **dgp_kwargs)

            # Check SDID placebo feasibility on realized data (custom DGP path)
            if effect_idx == 0 and sim == 0 and estimator_name == "SyntheticDiD":
                _check_sdid_placebo_data(data, estimator, est_kwargs)

            try:
                # --- Fit estimator ---
                if use_survey_dgp:
                    assert survey_config is not None
                    fit_builder = _SURVEY_FIT_BUILDERS[estimator_name]
                    fit_kwargs = fit_builder(
                        data, n_units, n_periods, treatment_period, survey_config
                    )
                    fit_kwargs.update(est_kwargs)
                elif profile is not None and not use_custom_dgp:
                    fit_kwargs = profile.fit_kwargs_builder(
                        data, n_units, n_periods, treatment_period
                    )
                    fit_kwargs.update(est_kwargs)
                else:
                    # Custom DGP fallback: use registry fit kwargs if available,
                    # otherwise use basic DiD signature
                    if profile is not None:
                        fit_kwargs = profile.fit_kwargs_builder(
                            data, n_units, n_periods, treatment_period
                        )
                        fit_kwargs.update(est_kwargs)
                    else:
                        fit_kwargs = dict(est_kwargs)

                result = estimator.fit(data, **fit_kwargs)

                # --- Extract results ---
                if profile is not None:
                    att, se, p_val, ci = profile.result_extractor(result)
                elif result_extractor is not None:
                    att, se, p_val, ci = result_extractor(result)
                else:
                    att = result.att if hasattr(result, "att") else result.avg_att
                    se = result.se if hasattr(result, "se") else result.avg_se
                    p_val = result.p_value if hasattr(result, "p_value") else result.avg_p_value
                    ci = result.conf_int if hasattr(result, "conf_int") else result.avg_conf_int

                # NaN p-value → treat as non-rejection
                rejected = bool(p_val < alpha) if not np.isnan(p_val) else False

                estimates.append(att)
                ses.append(se)
                p_values.append(p_val)
                rejections.append(rejected)
                ci_contains_true.append(ci[0] <= effect <= ci[1])

            except (
                ValueError,
                np.linalg.LinAlgError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
            ) as e:
                n_failures += 1
                if progress:
                    print(f"  Warning: Simulation {sim} failed: {e}")
                continue

        # Warn if too many simulations failed
        failure_rate = n_failures / n_simulations
        if failure_rate > 0.1:
            warnings.warn(
                f"{n_failures}/{n_simulations} simulations ({failure_rate:.1%}) "
                f"failed for effect_size={effect}. "
                f"Check estimator and data generator.",
                UserWarning,
            )

        if len(estimates) == 0:
            raise RuntimeError("All simulations failed. Check estimator and data generator.")

        power_val = np.mean(rejections)
        all_powers.append(power_val)

        if is_primary:
            primary_estimates = estimates
            primary_ses = ses
            primary_p_values = p_values
            primary_rejections = rejections
            primary_ci_contains = ci_contains_true
            primary_n_failures = n_failures

    # Compute confidence interval for power (primary effect)
    power_val = all_powers[primary_idx]
    n_valid = len(primary_rejections)
    power_se = np.sqrt(power_val * (1 - power_val) / n_valid)
    z = stats.norm.ppf(0.975)
    power_ci = (
        max(0.0, power_val - z * power_se),
        min(1.0, power_val + z * power_se),
    )

    mean_estimate = np.mean(primary_estimates)
    std_estimate = np.std(primary_estimates, ddof=1)
    mean_se = np.mean(primary_ses)
    coverage = np.mean(primary_ci_contains)

    return SimulationPowerResults(
        power=power_val,
        power_se=power_se,
        power_ci=power_ci,
        rejection_rate=power_val,
        mean_estimate=mean_estimate,
        std_estimate=std_estimate,
        mean_se=mean_se,
        coverage=coverage,
        n_simulations=n_valid,
        n_simulation_failures=primary_n_failures,
        effect_sizes=effect_sizes,
        powers=all_powers,
        true_effect=primary_effect,
        alpha=alpha,
        estimator_name=estimator_name,
        simulation_results=[
            {"estimate": e, "se": s, "p_value": p, "rejected": r}
            for e, s, p, r in zip(
                primary_estimates,
                primary_ses,
                primary_p_values,
                primary_rejections,
            )
        ],
        effective_n_units=effective_n_units,
        survey_config=survey_config,
        mean_deff=float(np.nanmean(deff_values)) if deff_values else None,
        mean_icc_realized=float(np.nanmean(icc_values)) if icc_values else None,
    )


# ---------------------------------------------------------------------------
# Simulation-based MDE and sample-size search
# ---------------------------------------------------------------------------


@dataclass
class SimulationMDEResults(Diagnostic):
    """
    Results from simulation-based minimum detectable effect search.

    Attributes
    ----------
    mde : float
        Minimum detectable effect (smallest effect achieving target power).
    power_at_mde : float
        Power achieved at the MDE.
    target_power : float
        Target power used in the search.
    alpha : float
        Significance level.
    n_units : int
        Sample size used.
    n_simulations_per_step : int
        Number of simulations per bisection step.
    n_steps : int
        Number of bisection steps performed.
    search_path : list of dict
        Diagnostic trace of ``{effect_size, power}`` at each step.
    estimator_name : str
        Name of the estimator used.
    effective_n_units : int or None
        Effective sample size when it differs from the requested ``n_units``
        (e.g., due to DDD grid rounding). ``None`` when no rounding occurred.
    """

    mde: float
    power_at_mde: float
    target_power: float
    alpha: float
    n_units: int
    n_simulations_per_step: int
    n_steps: int
    search_path: List[Dict[str, float]]
    estimator_name: str
    effective_n_units: Optional[int] = None
    survey_config: Optional[Any] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"SimulationMDEResults(mde={self.mde:.4f}, "
            f"power_at_mde={self.power_at_mde:.3f}, "
            f"n_steps={self.n_steps})"
        )

    def summary(self) -> str:
        """Generate a formatted summary."""
        lines = [
            "=" * 65,
            "Simulation-Based MDE Results".center(65),
            "=" * 65,
            "",
            f"{'Estimator:':<35} {self.estimator_name}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            f"{'Target power:':<35} {self.target_power:.1%}",
            f"{'Sample size (n_units):':<35} {self.n_units}",
        ]
        if self.effective_n_units is not None:
            lines.append(
                f"{'Effective sample size:':<35} {self.effective_n_units}" f" (DDD grid-rounded)"
            )
        lines += [
            f"{'Simulations per step:':<35} {self.n_simulations_per_step}",
            "",
            "-" * 65,
            "Search Results".center(65),
            "-" * 65,
            f"{'Minimum detectable effect:':<35} {self.mde:.4f}",
            f"{'Power at MDE:':<35} {self.power_at_mde:.1%}",
            f"{'Bisection steps:':<35} {self.n_steps}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "mde": self.mde,
            "power_at_mde": self.power_at_mde,
            "target_power": self.target_power,
            "alpha": self.alpha,
            "n_units": self.n_units,
            "effective_n_units": self.effective_n_units,
            "n_simulations_per_step": self.n_simulations_per_step,
            "n_steps": self.n_steps,
            "estimator_name": self.estimator_name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class SimulationSampleSizeResults(Diagnostic):
    """
    Results from simulation-based sample size search.

    Attributes
    ----------
    required_n : int
        Required number of units to achieve target power.
    power_at_n : float
        Power achieved at the required N.
    target_power : float
        Target power used in the search.
    alpha : float
        Significance level.
    effect_size : float
        Effect size used in the search.
    n_simulations_per_step : int
        Number of simulations per bisection step.
    n_steps : int
        Number of bisection steps performed.
    search_path : list of dict
        Diagnostic trace of ``{n_units, power}`` at each step.
    estimator_name : str
        Name of the estimator used.
    effective_n_units : int or None
        Effective sample size when it differs from ``required_n``
        (e.g., due to DDD grid rounding). ``None`` when no rounding occurred
        or when the search already snapped to the estimator's grid.
    """

    required_n: int
    power_at_n: float
    target_power: float
    alpha: float
    effect_size: float
    n_simulations_per_step: int
    n_steps: int
    search_path: List[Dict[str, float]]
    estimator_name: str
    effective_n_units: Optional[int] = None
    survey_config: Optional[Any] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"SimulationSampleSizeResults(required_n={self.required_n}, "
            f"power_at_n={self.power_at_n:.3f}, "
            f"n_steps={self.n_steps})"
        )

    def summary(self) -> str:
        """Generate a formatted summary."""
        lines = [
            "=" * 65,
            "Simulation-Based Sample Size Results".center(65),
            "=" * 65,
            "",
            f"{'Estimator:':<35} {self.estimator_name}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            f"{'Target power:':<35} {self.target_power:.1%}",
            f"{'Effect size:':<35} {self.effect_size:.4f}",
            f"{'Simulations per step:':<35} {self.n_simulations_per_step}",
            "",
            "-" * 65,
            "Search Results".center(65),
            "-" * 65,
            f"{'Required sample size:':<35} {self.required_n}",
            f"{'Power at required N:':<35} {self.power_at_n:.1%}",
            f"{'Bisection steps:':<35} {self.n_steps}",
        ]
        if self.effective_n_units is not None:
            lines.append(
                f"{'Effective sample size:':<35} {self.effective_n_units}" f" (DDD grid-rounded)"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "required_n": self.required_n,
            "power_at_n": self.power_at_n,
            "target_power": self.target_power,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "n_simulations_per_step": self.n_simulations_per_step,
            "n_steps": self.n_steps,
            "estimator_name": self.estimator_name,
            "effective_n_units": self.effective_n_units,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


def simulate_mde(
    estimator: Any,
    n_units: int = 100,
    n_periods: int = 4,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    sigma: float = 1.0,
    n_simulations: int = 200,
    power: float = 0.80,
    alpha: float = 0.05,
    effect_range: Optional[Tuple[float, float]] = None,
    tol: float = 0.02,
    max_steps: int = 15,
    seed: Optional[int] = None,
    data_generator: Optional[Callable] = None,
    data_generator_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    result_extractor: Optional[Callable] = None,
    progress: bool = True,
    survey_config: Optional[SurveyPowerConfig] = None,
) -> SimulationMDEResults:
    """
    Find the minimum detectable effect via simulation-based bisection search.

    Searches over effect sizes to find the smallest effect that achieves the
    target power, using ``simulate_power()`` at each step.

    Parameters
    ----------
    estimator : estimator object
        DiD estimator to use.
    n_units : int, default=100
        Number of units per simulation.
    n_periods : int, default=4
        Number of time periods.
    treatment_fraction : float, default=0.5
        Fraction of units that are treated.
    treatment_period : int, default=2
        First post-treatment period (0-indexed).
    sigma : float, default=1.0
        Residual standard deviation.
    n_simulations : int, default=200
        Simulations per bisection step.
    power : float, default=0.80
        Target power.
    alpha : float, default=0.05
        Significance level.
    effect_range : tuple of (float, float), optional
        ``(lo, hi)`` bracket for the search. If None, auto-brackets.
    tol : float, default=0.02
        Convergence tolerance on power.
    max_steps : int, default=15
        Maximum bisection steps.
    seed : int, optional
        Random seed for reproducibility.
    data_generator : callable, optional
        Custom data generation function.
    data_generator_kwargs : dict, optional
        Additional keyword arguments for data generator.
    estimator_kwargs : dict, optional
        Additional keyword arguments for estimator.fit().
    result_extractor : callable, optional
        Custom function to extract results from the estimator output.
        Forwarded to ``simulate_power()``.
    progress : bool, default=True
        Whether to print progress updates.
    survey_config : SurveyPowerConfig, optional
        Survey-aware simulation config. Forwarded to ``simulate_power()``.
        See :func:`simulate_power` for details and constraints.

    Returns
    -------
    SimulationMDEResults
        Results including the MDE and search diagnostics.

    Examples
    --------
    >>> from diff_diff import simulate_mde, DifferenceInDifferences
    >>> result = simulate_mde(DifferenceInDifferences(), n_simulations=100, seed=42)
    >>> print(f"MDE: {result.mde:.3f}")
    """
    master_rng = np.random.default_rng(seed)
    estimator_name = type(estimator).__name__
    search_path: List[Dict[str, float]] = []

    # Compute effective N for DDD (N is fixed throughout MDE search). Only the
    # cross-sectional 2x2x2 DGP (n_periods <= 2) rounds n_units to a multiple of
    # 8; the panel DGP (n_periods > 2) maps n_units directly, so report None.
    if estimator_name == "TripleDifference" and data_generator is None and n_periods <= 2:
        effective_n_units = _ddd_effective_n(n_units, data_generator_kwargs)
    else:
        effective_n_units = None

    common_kwargs: Dict[str, Any] = dict(
        estimator=estimator,
        n_units=n_units,
        n_periods=n_periods,
        treatment_fraction=treatment_fraction,
        treatment_period=treatment_period,
        sigma=sigma,
        n_simulations=n_simulations,
        alpha=alpha,
        data_generator=data_generator,
        data_generator_kwargs=data_generator_kwargs,
        estimator_kwargs=estimator_kwargs,
        result_extractor=result_extractor,
        progress=False,
        survey_config=survey_config,
    )

    def _power_at(effect: float) -> float:
        step_seed = int(master_rng.integers(0, 2**31))
        res = simulate_power(treatment_effect=effect, seed=step_seed, **common_kwargs)
        pwr = float(res.power)
        search_path.append({"effect_size": effect, "power": pwr})
        if progress:
            print(f"  MDE search: effect={effect:.4f}, power={pwr:.3f}")
        return pwr

    # --- Bracket ---
    if effect_range is not None:
        lo, hi = effect_range
        power_lo = _power_at(lo)
        power_hi = _power_at(hi)
        if power_lo >= power:
            warnings.warn(
                f"Power at effect={lo} is {power_lo:.2f} >= target {power}. "
                f"Lower bound already exceeds target power. Returning lo as MDE.",
                UserWarning,
            )
            return SimulationMDEResults(
                mde=lo,
                power_at_mde=power_lo,
                target_power=power,
                alpha=alpha,
                n_units=n_units,
                n_simulations_per_step=n_simulations,
                n_steps=len(search_path),
                search_path=search_path,
                estimator_name=estimator_name,
                effective_n_units=effective_n_units,
                survey_config=survey_config,
            )
        if power_hi < power:
            warnings.warn(
                f"Target power {power} not bracketed: power at effect={hi} "
                f"is {power_hi:.2f}. Upper bound may be too low.",
                UserWarning,
            )
    else:
        lo = 0.0
        # Check that power at zero is below target (no inflated Type I error)
        power_at_zero = _power_at(0.0)
        if power_at_zero >= power:
            warnings.warn(
                f"Power at effect=0 is {power_at_zero:.2f} >= target {power}. "
                f"This suggests inflated Type I error. Returning MDE=0.",
                UserWarning,
            )
            return SimulationMDEResults(
                mde=0.0,
                power_at_mde=power_at_zero,
                target_power=power,
                alpha=alpha,
                n_units=n_units,
                n_simulations_per_step=n_simulations,
                n_steps=len(search_path),
                search_path=search_path,
                estimator_name=estimator_name,
                survey_config=survey_config,
                effective_n_units=effective_n_units,
            )

        hi = sigma
        for _ in range(10):
            if _power_at(hi) >= power:
                break
            hi *= 2
        else:
            warnings.warn(
                f"Could not bracket MDE (power at effect={hi} still below "
                f"{power}). Returning best upper bound.",
                UserWarning,
            )

    # --- Bisect ---
    best_effect = hi
    best_power = search_path[-1]["power"] if search_path else 0.0

    for _ in range(max_steps):
        mid = (lo + hi) / 2
        pwr = _power_at(mid)

        if pwr >= power:
            hi = mid
            best_effect = mid
            best_power = pwr
        else:
            lo = mid

        # Convergence: effect range is tight or power is close enough
        if hi - lo < max(tol * hi, 1e-6) or abs(pwr - power) < tol:
            break

    return SimulationMDEResults(
        mde=best_effect,
        power_at_mde=best_power,
        target_power=power,
        alpha=alpha,
        n_units=n_units,
        n_simulations_per_step=n_simulations,
        n_steps=len(search_path),
        search_path=search_path,
        estimator_name=estimator_name,
        effective_n_units=effective_n_units,
        survey_config=survey_config,
    )


def simulate_sample_size(
    estimator: Any,
    treatment_effect: float = 5.0,
    n_periods: int = 4,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    sigma: float = 1.0,
    n_simulations: int = 200,
    power: float = 0.80,
    alpha: float = 0.05,
    n_range: Optional[Tuple[int, int]] = None,
    max_steps: int = 15,
    seed: Optional[int] = None,
    data_generator: Optional[Callable] = None,
    data_generator_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    result_extractor: Optional[Callable] = None,
    progress: bool = True,
    survey_config: Optional[SurveyPowerConfig] = None,
) -> SimulationSampleSizeResults:
    """
    Find the required sample size via simulation-based bisection search.

    Searches over ``n_units`` to find the smallest N that achieves the
    target power, using ``simulate_power()`` at each step.

    Parameters
    ----------
    estimator : estimator object
        DiD estimator to use.
    treatment_effect : float, default=5.0
        True treatment effect to simulate.
    n_periods : int, default=4
        Number of time periods.
    treatment_fraction : float, default=0.5
        Fraction of units that are treated.
    treatment_period : int, default=2
        First post-treatment period (0-indexed).
    sigma : float, default=1.0
        Residual standard deviation.
    n_simulations : int, default=200
        Simulations per bisection step.
    power : float, default=0.80
        Target power.
    alpha : float, default=0.05
        Significance level.
    n_range : tuple of (int, int), optional
        ``(lo, hi)`` bracket for sample size. If None, auto-brackets.
    max_steps : int, default=15
        Maximum bisection steps.
    seed : int, optional
        Random seed for reproducibility.
    data_generator : callable, optional
        Custom data generation function.
    data_generator_kwargs : dict, optional
        Additional keyword arguments for data generator.
    estimator_kwargs : dict, optional
        Additional keyword arguments for estimator.fit().
    result_extractor : callable, optional
        Custom function to extract results from the estimator output.
        Forwarded to ``simulate_power()``.
    progress : bool, default=True
        Whether to print progress updates.
    survey_config : SurveyPowerConfig, optional
        Survey-aware simulation config. Forwarded to ``simulate_power()``.
        When set, the bisection floor is raised to
        ``survey_config.min_viable_n`` to ensure viable survey structure.
        See :func:`simulate_power` for details and constraints.

    Returns
    -------
    SimulationSampleSizeResults
        Results including the required N and search diagnostics.

    Examples
    --------
    >>> from diff_diff import simulate_sample_size, DifferenceInDifferences
    >>> result = simulate_sample_size(
    ...     DifferenceInDifferences(), treatment_effect=5.0, n_simulations=100, seed=42
    ... )
    >>> print(f"Required N: {result.required_n}")
    """
    master_rng = np.random.default_rng(seed)
    estimator_name = type(estimator).__name__
    search_path: List[Dict[str, float]] = []

    # Determine min_n from registry. DDD splits cross-sectional (n_periods <= 2,
    # 2x2x2 factorial) from panel (n_periods > 2, generate_ddd_panel_data).
    registry = _get_registry()
    profile = registry.get(estimator_name)
    is_ddd = estimator_name == "TripleDifference" and data_generator is None
    is_ddd_panel = is_ddd and n_periods > 2
    if is_ddd_panel:
        # The panel DGP requires every (group,partition) cell non-empty, which
        # for a skewed group_frac/partition_frac override needs more than the
        # default 16-unit floor. Bracket above the viable minimum so the search
        # never probes an infeasible n (which would raise in the DGP).
        _ddd_overrides = data_generator_kwargs or {}
        min_n = _ddd_panel_viable_min_n(
            _ddd_overrides.get("group_frac", 0.5),
            _ddd_overrides.get("partition_frac", 0.5),
            floor=_ddd_panel_profile().min_n,
        )
    else:
        min_n = profile.min_n if profile is not None else 20

    # Grid snapping: the cross-sectional 2x2x2 DDD DGP rounds n_units to a
    # multiple of 8, so bisection candidates must snap to that grid. The panel
    # DGP maps n_units directly → continuous (step-1) search like every other
    # estimator.
    is_ddd_grid = is_ddd and not is_ddd_panel
    grid_step = 8 if is_ddd_grid else 1
    convergence_threshold = grid_step + 1  # 9 for cross-sectional DDD, 2 otherwise

    if is_ddd_grid and data_generator_kwargs and "n_per_cell" in data_generator_kwargs:
        raise ValueError(
            "data_generator_kwargs contains 'n_per_cell', which conflicts with "
            "the sample-size search in simulate_sample_size(). For "
            "TripleDifference, n_per_cell is derived from n_units (the search "
            "variable). Use simulate_power() with a fixed n_per_cell override "
            "instead, or pass a custom data_generator."
        )

    def _snap_n(n: int, direction: str = "down", floor: Optional[int] = None) -> int:
        actual_floor = floor if floor is not None else min_n
        if grid_step == 1:
            return max(actual_floor, n)
        if direction == "up":
            return max(actual_floor, ((n + grid_step - 1) // grid_step) * grid_step)
        return max(actual_floor, (n // grid_step) * grid_step)

    common_kwargs: Dict[str, Any] = dict(
        estimator=estimator,
        n_periods=n_periods,
        treatment_effect=treatment_effect,
        treatment_fraction=treatment_fraction,
        treatment_period=treatment_period,
        sigma=sigma,
        n_simulations=n_simulations,
        alpha=alpha,
        data_generator=data_generator,
        data_generator_kwargs=data_generator_kwargs,
        estimator_kwargs=estimator_kwargs,
        result_extractor=result_extractor,
        progress=False,
        survey_config=survey_config,
    )

    def _power_at_n(n: int) -> float:
        step_seed = int(master_rng.integers(0, 2**31))
        res = simulate_power(n_units=n, seed=step_seed, **common_kwargs)
        pwr = float(res.power)
        search_path.append({"n_units": float(n), "power": pwr})
        if progress:
            print(f"  Sample size search: n={n}, power={pwr:.3f}")
        return pwr

    # Block strata_sizes in sample-size search (same class as n_per_cell for DDD):
    # strata_sizes requires sum(strata_sizes) == n_units, but n_units varies
    # during bisection so a fixed strata_sizes would fail mid-search.
    if survey_config is not None and data_generator_kwargs:
        if "strata_sizes" in data_generator_kwargs:
            raise ValueError(
                "strata_sizes in data_generator_kwargs is not supported with "
                "simulate_sample_size() because n_units varies during the "
                "bisection search. Use simulate_power() with a fixed n_units "
                "and strata_sizes instead."
            )

    # --- Bracket ---
    # Cross-sectional DDD wants a >=16 floor so the 8 G×P×T cells are populated;
    # the panel path uses its split-aware viable floor (min_n above, which is
    # >=16 and higher for skewed group_frac/partition_frac); everything else
    # floors at 4.
    if is_ddd_panel:
        abs_min = min_n
    elif is_ddd:
        abs_min = 16
    else:
        abs_min = 4
    if survey_config is not None:
        abs_min = max(abs_min, survey_config.min_viable_n)
    if is_ddd_panel and n_range is not None and n_range[1] < abs_min:
        raise ValueError(
            f"n_range upper bound ({n_range[1]}) is below the minimum panel-DDD "
            f"sample size ({abs_min}) needed to populate all (group, partition) "
            f"cells for group_frac/partition_frac. Raise the upper bound or move "
            f"the split closer to 0.5."
        )
    if n_range is not None:
        lo, hi = _snap_n(n_range[0], "up", floor=abs_min), _snap_n(
            n_range[1], "down", floor=abs_min
        )
        if lo > hi:
            lo = hi  # collapsed bracket — evaluate single point
        power_lo = _power_at_n(lo)
        if power_lo >= power:
            warnings.warn(
                f"Power at n={lo} is {power_lo:.2f} >= target {power}. "
                f"Lower bound already achieves target power. Returning lo.",
                UserWarning,
            )
            return SimulationSampleSizeResults(
                required_n=lo,
                power_at_n=power_lo,
                target_power=power,
                alpha=alpha,
                effect_size=treatment_effect,
                n_simulations_per_step=n_simulations,
                n_steps=len(search_path),
                search_path=search_path,
                estimator_name=estimator_name,
                survey_config=survey_config,
            )
        power_hi = _power_at_n(hi)
        if power_hi < power:
            warnings.warn(
                f"Target power {power} not bracketed: power at n={hi} "
                f"is {power_hi:.2f}. Upper bound may be too low.",
                UserWarning,
            )
    else:
        lo = max(min_n, abs_min)
        power_lo = _power_at_n(lo)
        if power_lo >= power:
            # Floor achieves target — search downward for true minimum
            hi = lo
            found_lower = False
            probe = _snap_n(max(abs_min, lo // 2), floor=abs_min)
            for _ in range(8):
                if probe >= hi or probe < abs_min:
                    break
                pwr = _power_at_n(probe)
                if pwr < power:
                    lo = probe
                    found_lower = True
                    break
                hi = probe
                probe = _snap_n(max(abs_min, probe // 2), floor=abs_min)
            if not found_lower:
                # Even smallest viable N achieves target — return best found
                best = min(
                    (s for s in search_path if s["power"] >= power),
                    key=lambda s: s["n_units"],
                )
                # Clamp to abs_min (enforces survey min_viable_n contract)
                best_n = max(int(best["n_units"]), abs_min)
                warnings.warn(
                    f"Power at n={best_n} is "
                    f"{best['power']:.2f} >= target {power}. Could not "
                    f"find a smaller N below target power. Pass "
                    f"n_range=(lo, hi) to refine.",
                    UserWarning,
                )
                return SimulationSampleSizeResults(
                    required_n=best_n,
                    power_at_n=best["power"],
                    target_power=power,
                    alpha=alpha,
                    effect_size=treatment_effect,
                    n_simulations_per_step=n_simulations,
                    n_steps=len(search_path),
                    search_path=search_path,
                    estimator_name=estimator_name,
                    survey_config=survey_config,
                )
            # Fall through to bisection with lo..hi bracket
        else:
            hi = max(2 * lo, abs_min, 100)
            for _ in range(10):
                if _power_at_n(hi) >= power:
                    break
                hi *= 2
            else:
                warnings.warn(
                    f"Could not bracket required N (power at n={hi} still "
                    f"below {power}). Returning best upper bound.",
                    UserWarning,
                )

    # --- Bisect on integer n_units ---
    best_n = hi
    # Look up power at hi (search_path[-1] may not be hi after downward search)
    best_power = next(
        (s["power"] for s in reversed(search_path) if int(s["n_units"]) == hi),
        search_path[-1]["power"] if search_path else 0.0,
    )

    for _ in range(max_steps):
        if hi - lo <= convergence_threshold:
            break
        mid = _snap_n((lo + hi) // 2, floor=abs_min)
        if mid <= lo or mid >= hi:
            break
        pwr = _power_at_n(mid)

        if pwr >= power:
            hi = mid
            best_n = mid
            best_power = pwr
        else:
            lo = mid

    # Final answer is hi (conservative ceiling) — skip if already evaluated
    if best_n != hi:
        final_pwr = _power_at_n(hi)
        if final_pwr >= power:
            best_n = hi
            best_power = final_pwr

    return SimulationSampleSizeResults(
        required_n=best_n,
        power_at_n=best_power,
        target_power=power,
        alpha=alpha,
        effect_size=treatment_effect,
        n_simulations_per_step=n_simulations,
        n_steps=len(search_path),
        search_path=search_path,
        estimator_name=estimator_name,
        survey_config=survey_config,
    )


def compute_mde(
    n_treated: int,
    n_control: int,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
    deff: float = 1.0,
) -> float:
    """
    Convenience function to compute minimum detectable effect.

    Parameters
    ----------
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    sigma : float
        Residual standard deviation.
    power : float, default=0.80
        Target statistical power.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Within-unit (serial) equicorrelation for panel designs. Higher rho
        LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
        valid range [-1/(T-1), 1).
    deff : float, default=1.0
        Survey design effect (variance inflation factor).

    Returns
    -------
    float
        Minimum detectable effect size.

    Examples
    --------
    >>> mde = compute_mde(n_treated=50, n_control=50, sigma=10.0)
    >>> print(f"MDE: {mde:.2f}")
    """
    pa = PowerAnalysis(alpha=alpha, power=power)
    result = pa.mde(n_treated, n_control, sigma, n_pre, n_post, rho, deff=deff)
    return result.mde


def compute_power(
    effect_size: float,
    n_treated: int,
    n_control: int,
    sigma: float,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
    deff: float = 1.0,
) -> float:
    """
    Convenience function to compute power for given effect and sample.

    Parameters
    ----------
    effect_size : float
        Expected treatment effect.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    sigma : float
        Residual standard deviation.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Within-unit (serial) equicorrelation for panel designs. Higher rho
        LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
        valid range [-1/(T-1), 1).
    deff : float, default=1.0
        Survey design effect (variance inflation factor).

    Returns
    -------
    float
        Statistical power.

    Examples
    --------
    >>> power = compute_power(effect_size=5.0, n_treated=50, n_control=50, sigma=10.0)
    >>> print(f"Power: {power:.1%}")
    """
    pa = PowerAnalysis(alpha=alpha)
    result = pa.power(effect_size, n_treated, n_control, sigma, n_pre, n_post, rho, deff=deff)
    return result.power


def compute_sample_size(
    effect_size: float,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
    treat_frac: float = 0.5,
    deff: float = 1.0,
) -> int:
    """
    Convenience function to compute required sample size.

    Parameters
    ----------
    effect_size : float
        Treatment effect to detect.
    sigma : float
        Residual standard deviation.
    power : float, default=0.80
        Target statistical power.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Within-unit (serial) equicorrelation for panel designs. Higher rho
        LOWERS the MDE (Burlig et al. 2020, Eq. 2, equicorrelated case);
        valid range [-1/(T-1), 1).
    treat_frac : float, default=0.5
        Fraction assigned to treatment.
    deff : float, default=1.0
        Survey design effect (variance inflation factor).

    Returns
    -------
    int
        Required total sample size.

    Examples
    --------
    >>> n = compute_sample_size(effect_size=5.0, sigma=10.0)
    >>> print(f"Required N: {n}")
    """
    pa = PowerAnalysis(alpha=alpha, power=power)
    result = pa.sample_size(effect_size, sigma, n_pre, n_post, rho, treat_frac, deff=deff)
    return result.required_n
