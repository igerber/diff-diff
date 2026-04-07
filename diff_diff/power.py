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
            f"SyntheticDiD(variance_method='bootstrap')."
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


@dataclass
class PowerResults:
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
        Intra-cluster correlation (for panel data).
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
            f"{'Intra-cluster correlation:':<30} {self.rho:>10.4f}",
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
class SimulationPowerResults:
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
        Number of simulations performed.
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
            "true_effect": self.true_effect,
            "alpha": self.alpha,
            "estimator_name": self.estimator_name,
            "effective_n_units": self.effective_n_units,
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
    The power calculations are based on the variance of the DiD estimator:

    For basic 2x2 DiD:
        Var(ATT) = sigma^2 * (1/n_treated_post + 1/n_treated_pre
                            + 1/n_control_post + 1/n_control_pre)

    For panel DiD with T periods:
        Var(ATT) = sigma^2 * (1/(N_treated * T) + 1/(N_control * T))
                 * (1 + (T-1)*rho) / (1 + (T-1)*rho)

    Where rho is the intra-cluster correlation coefficient.

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

    def _get_critical_values(self) -> Tuple[float, float]:
        """Get z critical values for alpha and power."""
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
            Intra-cluster correlation (for panel data).
        design : str
            Study design type.

        Returns
        -------
        float
            Variance of the DiD estimator.
        """
        if design == "basic_did":
            # For basic 2x2 DiD, each cell has n_treated/2 or n_control/2 obs
            # assuming balanced design
            n_t_pre = n_treated  # treated units in pre-period
            n_t_post = n_treated  # treated units in post-period
            n_c_pre = n_control
            n_c_post = n_control

            variance = sigma**2 * (1 / n_t_post + 1 / n_t_pre + 1 / n_c_post + 1 / n_c_pre)
        elif design == "panel":
            # Panel DiD with multiple periods
            # Account for serial correlation via ICC
            T = n_pre + n_post

            # Design effect for clustering
            design_effect = 1 + (T - 1) * rho

            # Base variance (as if independent)
            base_var = sigma**2 * (1 / n_treated + 1 / n_control)

            # Adjust for clustering (Moulton factor)
            variance = base_var * design_effect / T
        else:
            raise ValueError(f"Unknown design: {design}")

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
            Intra-cluster correlation for panel data.

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
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(n_treated, n_control, n_pre, n_post, sigma, rho, design)
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
            effect_size, sigma, n_pre, n_post, rho, design, n_treated / (n_treated + n_control)
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
            Intra-cluster correlation for panel data.

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
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(n_treated, n_control, n_pre, n_post, sigma, rho, design)
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
    ) -> int:
        """Compute required sample size for given effect."""
        # Handle edge case of zero effect size
        if effect_size == 0:
            return MAX_SAMPLE_SIZE  # Can't detect zero effect

        z_alpha, z_beta = self._get_critical_values()

        T = n_pre + n_post

        if design == "basic_did":
            # Var = sigma^2 * (1/n_t + 1/n_t + 1/n_c + 1/n_c) = sigma^2 * (2/n_t + 2/n_c)
            # For balanced: Var = sigma^2 * 4/n where n = n_t = n_c
            # SE = sqrt(Var), effect_size = (z_alpha + z_beta) * SE
            # n = 4 * sigma^2 * (z_alpha + z_beta)^2 / effect_size^2

            # For general allocation with treat_frac:
            # Var = sigma^2 * 2 * (1/(N*p) + 1/(N*(1-p)))
            #     = 2 * sigma^2 / N * (1/p + 1/(1-p))
            #     = 2 * sigma^2 / N * (1/(p*(1-p)))

            n_total = (
                2
                * sigma**2
                * (z_alpha + z_beta) ** 2
                / (effect_size**2 * treat_frac * (1 - treat_frac))
            )
        else:  # panel
            design_effect = 1 + (T - 1) * rho

            # Var = sigma^2 * (1/n_t + 1/n_c) * design_effect / T
            # For balanced: Var = 2 * sigma^2 / N * design_effect / T

            n_total = (
                2
                * sigma**2
                * (z_alpha + z_beta) ** 2
                * design_effect
                / (effect_size**2 * treat_frac * (1 - treat_frac) * T)
            )

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
            Intra-cluster correlation for panel data.
        treat_frac : float, default=0.5
            Fraction of units assigned to treatment.

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
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        n_total = self._compute_required_n(
            effect_size, sigma, n_pre, n_post, rho, design, treat_frac
        )

        n_treated = max(2, int(np.ceil(n_total * treat_frac)))
        n_control = max(2, n_total - n_treated)
        n_total = n_treated + n_control

        # Compute actual power achieved
        variance = self._compute_variance(n_treated, n_control, n_pre, n_post, sigma, rho, design)
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
            Intra-cluster correlation.

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
        mde_result = self.mde(n_treated, n_control, sigma, n_pre, n_post, rho)

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
            Intra-cluster correlation.
        treat_frac : float, default=0.5
            Fraction assigned to treatment.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'sample_size' and 'power'.
        """
        # Get required N to determine default range
        required = self.sample_size(effect_size, sigma, n_pre, n_post, rho, treat_frac)

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

    data_gen_kwargs = data_generator_kwargs or {}
    est_kwargs = estimator_kwargs or {}

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
                f"SyntheticDiD(variance_method='bootstrap')."
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
    if estimator_name == "TripleDifference" and not use_custom_dgp:
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
            if use_custom_dgp:
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
                if profile is not None and not use_custom_dgp:
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

            except Exception as e:
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
    )


# ---------------------------------------------------------------------------
# Simulation-based MDE and sample-size search
# ---------------------------------------------------------------------------


@dataclass
class SimulationMDEResults:
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
class SimulationSampleSizeResults:
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

    # Compute effective N for DDD (N is fixed throughout MDE search)
    if estimator_name == "TripleDifference" and data_generator is None:
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

    # Determine min_n from registry
    registry = _get_registry()
    profile = registry.get(estimator_name)
    min_n = profile.min_n if profile is not None else 20

    # DDD grid snapping: bisection candidates must be multiples of 8
    is_ddd_grid = estimator_name == "TripleDifference" and data_generator is None
    grid_step = 8 if is_ddd_grid else 1
    convergence_threshold = grid_step + 1  # 9 for DDD, 2 for others

    if is_ddd_grid and data_generator_kwargs and "n_per_cell" in data_generator_kwargs:
        raise ValueError(
            "data_generator_kwargs contains 'n_per_cell', which conflicts with "
            "the sample-size search in simulate_sample_size(). For "
            "TripleDifference, n_per_cell is derived from n_units (the search "
            "variable). Use simulate_power() with a fixed n_per_cell override "
            "instead, or pass a custom data_generator."
        )

    def _snap_n(n: int, direction: str = "down", floor: Optional[int] = None) -> int:
        if grid_step == 1:
            return n
        actual_floor = floor if floor is not None else min_n
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
    )

    def _power_at_n(n: int) -> float:
        step_seed = int(master_rng.integers(0, 2**31))
        res = simulate_power(n_units=n, seed=step_seed, **common_kwargs)
        pwr = float(res.power)
        search_path.append({"n_units": float(n), "power": pwr})
        if progress:
            print(f"  Sample size search: n={n}, power={pwr:.3f}")
        return pwr

    # --- Bracket ---
    abs_min = 16 if is_ddd_grid else 4
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
            )
        power_hi = _power_at_n(hi)
        if power_hi < power:
            warnings.warn(
                f"Target power {power} not bracketed: power at n={hi} "
                f"is {power_hi:.2f}. Upper bound may be too low.",
                UserWarning,
            )
    else:
        lo = min_n
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
                warnings.warn(
                    f"Power at n={int(best['n_units'])} is "
                    f"{best['power']:.2f} >= target {power}. Could not "
                    f"find a smaller N below target power. Pass "
                    f"n_range=(lo, hi) to refine.",
                    UserWarning,
                )
                return SimulationSampleSizeResults(
                    required_n=int(best["n_units"]),
                    power_at_n=best["power"],
                    target_power=power,
                    alpha=alpha,
                    effect_size=treatment_effect,
                    n_simulations_per_step=n_simulations,
                    n_steps=len(search_path),
                    search_path=search_path,
                    estimator_name=estimator_name,
                )
            # Fall through to bisection with lo..hi bracket
        else:
            hi = max(100, 2 * min_n)
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
        Intra-cluster correlation.

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
    result = pa.mde(n_treated, n_control, sigma, n_pre, n_post, rho)
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
        Intra-cluster correlation.

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
    result = pa.power(effect_size, n_treated, n_control, sigma, n_pre, n_post, rho)
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
        Intra-cluster correlation.
    treat_frac : float, default=0.5
        Fraction assigned to treatment.

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
    result = pa.sample_size(effect_size, sigma, n_pre, n_post, rho, treat_frac)
    return result.required_n
