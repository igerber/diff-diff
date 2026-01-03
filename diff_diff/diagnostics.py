"""
Diagnostic tools for validating Difference-in-Differences assumptions.

This module provides placebo tests and other diagnostic tools for assessing
the validity of the parallel trends assumption in DiD designs.

References
----------
Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust
Differences-in-Differences Estimates? The Quarterly Journal of Economics,
119(1), 249-275.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.utils import compute_confidence_interval, compute_p_value


def _get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    return ""


@dataclass
class PlaceboTestResults:
    """
    Results from a placebo test for DiD assumption validation.

    Attributes
    ----------
    test_type : str
        Type of placebo test performed.
    placebo_effect : float
        Estimated placebo treatment effect.
    se : float
        Standard error of the placebo effect.
    t_stat : float
        T-statistic for the placebo effect.
    p_value : float
        P-value for testing placebo_effect = 0.
    conf_int : tuple
        Confidence interval for the placebo effect.
    n_obs : int
        Number of observations used in the test.
    is_significant : bool
        Whether the placebo effect is significant at alpha=0.05.
    original_effect : float, optional
        Original ATT estimate for comparison.
    original_se : float, optional
        Original SE for comparison.
    permutation_distribution : np.ndarray, optional
        Distribution of permuted effects (for permutation test).
    leave_one_out_effects : dict, optional
        Unit-specific effects (for leave-one-out test).
    fake_period : any, optional
        The fake treatment period used (for timing test).
    fake_group : list, optional
        The fake treatment group used (for group test).
    """

    test_type: str
    placebo_effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    is_significant: bool
    alpha: float = 0.05

    # Optional fields for specific test types
    original_effect: Optional[float] = None
    original_se: Optional[float] = None
    permutation_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    leave_one_out_effects: Optional[Dict[Any, float]] = field(default=None)
    fake_period: Optional[Any] = None
    fake_group: Optional[List[Any]] = field(default=None)
    n_permutations: Optional[int] = None

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)

    def summary(self) -> str:
        """Generate formatted summary of placebo test results."""
        conf_level = int((1 - self.alpha) * 100)

        lines = [
            "=" * 65,
            f"Placebo Test Results: {self.test_type}".center(65),
            "=" * 65,
            "",
            f"{'Placebo effect:':<25} {self.placebo_effect:>12.4f}",
            f"{'Standard error:':<25} {self.se:>12.4f}",
            f"{'T-statistic:':<25} {self.t_stat:>12.4f}",
            f"{'P-value:':<25} {self.p_value:>12.4f}",
            f"{conf_level}% CI: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            "",
            f"{'Observations:':<25} {self.n_obs:>12}",
        ]

        if self.original_effect is not None:
            lines.extend([
                "",
                "-" * 65,
                "Comparison with Original Estimate".center(65),
                "-" * 65,
                f"{'Original ATT:':<25} {self.original_effect:>12.4f}",
            ])
            if self.original_se is not None:
                lines.append(f"{'Original SE:':<25} {self.original_se:>12.4f}")

        if self.n_permutations is not None:
            lines.append(f"{'Number of permutations:':<25} {self.n_permutations:>12}")

        if self.fake_period is not None:
            lines.append(f"{'Fake treatment period:':<25} {str(self.fake_period):>12}")

        if self.leave_one_out_effects is not None:
            n_units = len(self.leave_one_out_effects)
            effects = list(self.leave_one_out_effects.values())
            lines.extend([
                "",
                "-" * 65,
                "Leave-One-Out Summary".center(65),
                "-" * 65,
                f"{'Units analyzed:':<25} {n_units:>12}",
                f"{'Mean effect:':<25} {np.mean(effects):>12.4f}",
                f"{'Std. dev.:':<25} {np.std(effects, ddof=1):>12.4f}",
                f"{'Min effect:':<25} {np.min(effects):>12.4f}",
                f"{'Max effect:':<25} {np.max(effects):>12.4f}",
            ])

        # Interpretation
        lines.extend([
            "",
            "-" * 65,
            "Interpretation".center(65),
            "-" * 65,
        ])

        if self.is_significant:
            lines.append(
                "WARNING: Significant placebo effect detected (p < 0.05)."
            )
            lines.append(
                "This suggests potential violations of the parallel trends assumption."
            )
        else:
            lines.append(
                "No significant placebo effect detected (p >= 0.05)."
            )
            lines.append(
                "This is consistent with the parallel trends assumption."
            )

        lines.append("=" * 65)

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        result = {
            "test_type": self.test_type,
            "placebo_effect": self.placebo_effect,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "is_significant": self.is_significant,
        }

        if self.original_effect is not None:
            result["original_effect"] = self.original_effect
        if self.original_se is not None:
            result["original_se"] = self.original_se
        if self.n_permutations is not None:
            result["n_permutations"] = self.n_permutations

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        return pd.DataFrame([self.to_dict()])


def run_placebo_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: Optional[str] = None,
    test_type: str = "fake_timing",
    fake_treatment_period: Optional[Any] = None,
    fake_treatment_group: Optional[List[Any]] = None,
    post_periods: Optional[List[Any]] = None,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    **estimator_kwargs
) -> PlaceboTestResults:
    """
    Run a placebo test to validate DiD assumptions.

    Placebo tests provide evidence on the validity of the parallel trends
    assumption by testing whether "fake" treatments produce significant effects.
    A significant placebo effect suggests the parallel trends assumption may
    be violated.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data for DiD analysis.
    outcome : str
        Name of outcome variable column.
    treatment : str
        Name of treatment indicator column (0/1).
    time : str
        Name of time period column.
    unit : str, optional
        Name of unit identifier column. Required for some test types.
    test_type : str, default="fake_timing"
        Type of placebo test:
        - "fake_timing": Assign treatment at a fake (earlier) time period
        - "fake_group": Run DiD designating some control units as "fake treated"
        - "permutation": Randomly reassign treatment and compute distribution
        - "leave_one_out": Drop each treated unit and re-estimate
    fake_treatment_period : any, optional
        For "fake_timing": The fake treatment period to test.
        Should be a pre-treatment period.
    fake_treatment_group : list, optional
        For "fake_group": List of control unit IDs to designate as fake treated.
    post_periods : list, optional
        List of post-treatment periods. Required for fake_timing test.
    n_permutations : int, default=1000
        For "permutation": Number of random treatment assignments.
    alpha : float, default=0.05
        Significance level.
    seed : int, optional
        Random seed for reproducibility.
    **estimator_kwargs
        Additional arguments passed to the DiD estimator.

    Returns
    -------
    PlaceboTestResults
        Object containing placebo effect estimates, p-values, and diagnostics.

    Examples
    --------
    Fake timing test:

    >>> results = run_placebo_test(
    ...     data, outcome='sales', treatment='treated', time='period',
    ...     test_type='fake_timing',
    ...     fake_treatment_period=1,  # Pre-treatment period
    ...     post_periods=[2, 3, 4]
    ... )
    >>> if results.is_significant:
    ...     print("Warning: Pre-treatment differential trends detected!")

    Permutation test:

    >>> results = run_placebo_test(
    ...     data, outcome='sales', treatment='treated', time='period',
    ...     unit='unit_id',
    ...     test_type='permutation',
    ...     n_permutations=1000,
    ...     seed=42
    ... )
    >>> print(f"Permutation p-value: {results.p_value:.4f}")

    References
    ----------
    Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should
    We Trust Differences-in-Differences Estimates? The Quarterly Journal
    of Economics, 119(1), 249-275.
    """
    test_type = test_type.lower()
    valid_types = ["fake_timing", "fake_group", "permutation", "leave_one_out"]

    if test_type not in valid_types:
        raise ValueError(
            f"test_type must be one of {valid_types}, got '{test_type}'"
        )

    if test_type == "fake_timing":
        return placebo_timing_test(
            data=data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            fake_treatment_period=fake_treatment_period,
            post_periods=post_periods,
            alpha=alpha,
            **estimator_kwargs
        )

    elif test_type == "fake_group":
        if unit is None:
            raise ValueError("unit is required for fake_group test")
        if fake_treatment_group is None or len(fake_treatment_group) == 0:
            raise ValueError("fake_treatment_group is required for fake_group test")
        return placebo_group_test(
            data=data,
            outcome=outcome,
            time=time,
            unit=unit,
            fake_treated_units=fake_treatment_group,
            post_periods=post_periods,
            alpha=alpha,
            **estimator_kwargs
        )

    elif test_type == "permutation":
        if unit is None:
            raise ValueError("unit is required for permutation test")
        return permutation_test(
            data=data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit=unit,
            n_permutations=n_permutations,
            alpha=alpha,
            seed=seed,
            **estimator_kwargs
        )

    elif test_type == "leave_one_out":
        if unit is None:
            raise ValueError("unit is required for leave_one_out test")
        return leave_one_out_test(
            data=data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit=unit,
            alpha=alpha,
            **estimator_kwargs
        )


def placebo_timing_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    fake_treatment_period: Any,
    post_periods: Optional[List[Any]] = None,
    alpha: float = 0.05,
    **estimator_kwargs
) -> PlaceboTestResults:
    """
    Test for pre-treatment effects by moving treatment timing earlier.

    Creates a fake "post" indicator using pre-treatment data only, then
    estimates a DiD model. A significant effect suggests pre-existing
    differential trends.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    treatment : str
        Treatment indicator column.
    time : str
        Time period column.
    fake_treatment_period : any
        Period to use as fake treatment timing (should be a pre-treatment period).
    post_periods : list, optional
        List of actual post-treatment periods. If None, infers from data.
    alpha : float, default=0.05
        Significance level.
    **estimator_kwargs
        Arguments passed to DifferenceInDifferences.

    Returns
    -------
    PlaceboTestResults
        Results of the fake timing placebo test.
    """
    all_periods = sorted(data[time].unique())

    # Infer post periods if not provided
    if post_periods is None:
        # Use second half of periods as post
        mid = len(all_periods) // 2
        post_periods = all_periods[mid:]

    # Validate fake_treatment_period is pre-treatment
    if fake_treatment_period in post_periods:
        raise ValueError(
            f"fake_treatment_period ({fake_treatment_period}) must be a "
            f"pre-treatment period, not in post_periods ({post_periods})"
        )

    # Use only pre-treatment data
    pre_periods = [p for p in all_periods if p not in post_periods]
    pre_data = data[data[time].isin(pre_periods)].copy()

    # Create fake post indicator
    pre_data["_fake_post"] = (pre_data[time] >= fake_treatment_period).astype(int)

    # Fit DiD on pre-treatment data with fake post
    did = DifferenceInDifferences(**estimator_kwargs)
    results = did.fit(
        pre_data,
        outcome=outcome,
        treatment=treatment,
        time="_fake_post"
    )

    # Also fit on full data for comparison
    data_with_post = data.copy()
    data_with_post["_post"] = data_with_post[time].isin(post_periods).astype(int)
    did_full = DifferenceInDifferences(**estimator_kwargs)
    results_full = did_full.fit(
        data_with_post,
        outcome=outcome,
        treatment=treatment,
        time="_post"
    )

    return PlaceboTestResults(
        test_type="fake_timing",
        placebo_effect=results.att,
        se=results.se,
        t_stat=results.t_stat,
        p_value=results.p_value,
        conf_int=results.conf_int,
        n_obs=results.n_obs,
        is_significant=bool(results.p_value < alpha),
        alpha=alpha,
        original_effect=results_full.att,
        original_se=results_full.se,
        fake_period=fake_treatment_period,
    )


def placebo_group_test(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    unit: str,
    fake_treated_units: List[Any],
    post_periods: Optional[List[Any]] = None,
    alpha: float = 0.05,
    **estimator_kwargs
) -> PlaceboTestResults:
    """
    Test for differential trends among never-treated units.

    Assigns some never-treated units as "fake treated" and estimates a
    DiD model using only never-treated data. A significant effect suggests
    heterogeneous trends in the control group.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    time : str
        Time period column.
    unit : str
        Unit identifier column.
    fake_treated_units : list
        List of control unit IDs to designate as "fake treated".
    post_periods : list, optional
        List of post-treatment period values.
    alpha : float, default=0.05
        Significance level.
    **estimator_kwargs
        Arguments passed to DifferenceInDifferences.

    Returns
    -------
    PlaceboTestResults
        Results of the fake group placebo test.
    """
    if fake_treated_units is None or len(fake_treated_units) == 0:
        raise ValueError("fake_treated_units must be a non-empty list")

    all_periods = sorted(data[time].unique())

    # Infer post periods if not provided
    if post_periods is None:
        mid = len(all_periods) // 2
        post_periods = all_periods[mid:]

    # Create fake treatment indicator
    fake_data = data.copy()
    fake_data["_fake_treated"] = fake_data[unit].isin(fake_treated_units).astype(int)
    fake_data["_post"] = fake_data[time].isin(post_periods).astype(int)

    # Fit DiD
    did = DifferenceInDifferences(**estimator_kwargs)
    results = did.fit(
        fake_data,
        outcome=outcome,
        treatment="_fake_treated",
        time="_post"
    )

    return PlaceboTestResults(
        test_type="fake_group",
        placebo_effect=results.att,
        se=results.se,
        t_stat=results.t_stat,
        p_value=results.p_value,
        conf_int=results.conf_int,
        n_obs=results.n_obs,
        is_significant=bool(results.p_value < alpha),
        alpha=alpha,
        fake_group=list(fake_treated_units),
    )


def permutation_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: str,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    **estimator_kwargs
) -> PlaceboTestResults:
    """
    Compute permutation-based p-value for DiD estimate.

    Randomly reassigns treatment status at the unit level and computes the
    DiD estimate for each permutation. The p-value is the proportion of
    permuted estimates at least as extreme as the original.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    treatment : str
        Treatment indicator column.
    time : str
        Time period column.
    unit : str
        Unit identifier column.
    n_permutations : int, default=1000
        Number of random permutations.
    alpha : float, default=0.05
        Significance level.
    seed : int, optional
        Random seed for reproducibility.
    **estimator_kwargs
        Arguments passed to DifferenceInDifferences.

    Returns
    -------
    PlaceboTestResults
        Results with permutation distribution and p-value.

    Notes
    -----
    The permutation test is exact and does not rely on asymptotic
    approximations, making it valid with any sample size.
    """
    rng = np.random.default_rng(seed)

    # First, fit original model
    did = DifferenceInDifferences(**estimator_kwargs)
    original_results = did.fit(
        data,
        outcome=outcome,
        treatment=treatment,
        time=time
    )
    original_att = original_results.att

    # Get unit-level treatment assignment
    unit_treatment = (
        data.groupby(unit)[treatment]
        .first()
        .reset_index()
    )
    units = unit_treatment[unit].values
    n_treated = int(unit_treatment[treatment].sum())

    # Permutation loop
    permuted_effects = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Randomly assign treatment to units
        perm_treated_units = rng.choice(units, size=n_treated, replace=False)

        # Create permuted data
        perm_data = data.copy()
        perm_data["_perm_treatment"] = perm_data[unit].isin(perm_treated_units).astype(int)

        # Fit DiD
        try:
            perm_did = DifferenceInDifferences(**estimator_kwargs)
            perm_results = perm_did.fit(
                perm_data,
                outcome=outcome,
                treatment="_perm_treatment",
                time=time
            )
            permuted_effects[i] = perm_results.att
        except Exception:
            # Handle edge cases where fitting fails
            permuted_effects[i] = np.nan

    # Remove any NaN values
    valid_effects = permuted_effects[~np.isnan(permuted_effects)]

    if len(valid_effects) == 0:
        raise RuntimeError("All permutations failed - check your data")

    # Compute p-value: proportion of |permuted| >= |original|
    p_value = np.mean(np.abs(valid_effects) >= np.abs(original_att))

    # Ensure p-value is at least 1/(n_permutations + 1)
    p_value = max(p_value, 1 / (len(valid_effects) + 1))

    # Compute SE and CI from permutation distribution
    se = np.std(valid_effects, ddof=1)
    ci_lower = np.percentile(valid_effects, alpha / 2 * 100)
    ci_upper = np.percentile(valid_effects, (1 - alpha / 2) * 100)

    # T-stat from original estimate
    t_stat = original_att / se if se > 0 else 0.0

    return PlaceboTestResults(
        test_type="permutation",
        placebo_effect=np.mean(valid_effects),  # Mean of null distribution
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int=(ci_lower, ci_upper),
        n_obs=len(data),
        is_significant=bool(p_value < alpha),
        alpha=alpha,
        original_effect=original_att,
        original_se=original_results.se,
        permutation_distribution=valid_effects,
        n_permutations=len(valid_effects),
    )


def leave_one_out_test(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: str,
    alpha: float = 0.05,
    **estimator_kwargs
) -> PlaceboTestResults:
    """
    Assess sensitivity by dropping each treated unit in turn.

    For each treated unit, drops that unit and re-estimates the DiD model.
    Large variation in estimates suggests results are driven by a single unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    treatment : str
        Treatment indicator column.
    time : str
        Time period column.
    unit : str
        Unit identifier column.
    alpha : float, default=0.05
        Significance level.
    **estimator_kwargs
        Arguments passed to DifferenceInDifferences.

    Returns
    -------
    PlaceboTestResults
        Results with leave_one_out_effects dict mapping unit -> ATT estimate.
    """
    # Fit original model
    did = DifferenceInDifferences(**estimator_kwargs)
    original_results = did.fit(
        data,
        outcome=outcome,
        treatment=treatment,
        time=time
    )
    original_att = original_results.att

    # Get treated units
    treated_units = data[data[treatment] == 1][unit].unique()

    # Leave-one-out loop
    loo_effects = {}

    for u in treated_units:
        # Drop this unit
        loo_data = data[data[unit] != u].copy()

        # Check we still have treated units
        if loo_data[treatment].sum() == 0:
            continue

        try:
            loo_did = DifferenceInDifferences(**estimator_kwargs)
            loo_results = loo_did.fit(
                loo_data,
                outcome=outcome,
                treatment=treatment,
                time=time
            )
            loo_effects[u] = loo_results.att
        except Exception:
            # Skip units that cause fitting issues
            loo_effects[u] = np.nan

    # Remove NaN values for statistics
    valid_effects = [v for v in loo_effects.values() if not np.isnan(v)]

    if len(valid_effects) == 0:
        raise RuntimeError("All leave-one-out estimates failed")

    # Statistics of LOO distribution
    mean_effect = np.mean(valid_effects)
    se = np.std(valid_effects, ddof=1) if len(valid_effects) > 1 else 0.0
    t_stat = mean_effect / se if se > 0 else 0.0

    # Use t-distribution for p-value
    df = len(valid_effects) - 1 if len(valid_effects) > 1 else 1
    p_value = compute_p_value(t_stat, df=df)

    # CI
    conf_int = compute_confidence_interval(mean_effect, se, alpha, df=df)

    return PlaceboTestResults(
        test_type="leave_one_out",
        placebo_effect=mean_effect,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int=conf_int,
        n_obs=len(data),
        is_significant=bool(p_value < alpha),
        alpha=alpha,
        original_effect=original_att,
        original_se=original_results.se,
        leave_one_out_effects=loo_effects,
    )


def run_all_placebo_tests(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: str,
    pre_periods: List[Any],
    post_periods: List[Any],
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    **estimator_kwargs
) -> Dict[str, Union[PlaceboTestResults, Dict[str, str]]]:
    """
    Run a comprehensive suite of placebo tests.

    Runs fake timing tests for each pre-period, a permutation test, and
    a leave-one-out sensitivity analysis. If a test fails, the result
    will be a dict with an "error" key containing the error message.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    treatment : str
        Treatment indicator column.
    time : str
        Time period column.
    unit : str
        Unit identifier column.
    pre_periods : list
        List of pre-treatment periods.
    post_periods : list
        List of post-treatment periods.
    n_permutations : int, default=500
        Permutations for permutation test.
    alpha : float, default=0.05
        Significance level.
    seed : int, optional
        Random seed.
    **estimator_kwargs
        Arguments passed to estimators.

    Returns
    -------
    dict
        Dictionary mapping test names to PlaceboTestResults.
        Keys: "fake_timing_{period}", "permutation", "leave_one_out"
    """
    results = {}

    # Fake timing tests for each pre-period (except first)
    for period in pre_periods[1:]:  # Skip first period
        try:
            test_result = placebo_timing_test(
                data=data,
                outcome=outcome,
                treatment=treatment,
                time=time,
                fake_treatment_period=period,
                post_periods=post_periods,
                alpha=alpha,
                **estimator_kwargs
            )
            results[f"fake_timing_{period}"] = test_result
        except Exception as e:
            # Store error info
            results[f"fake_timing_{period}"] = {"error": str(e)}

    # Permutation test
    try:
        perm_result = permutation_test(
            data=data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit=unit,
            n_permutations=n_permutations,
            alpha=alpha,
            seed=seed,
            **estimator_kwargs
        )
        results["permutation"] = perm_result
    except Exception as e:
        results["permutation"] = {"error": str(e)}

    # Leave-one-out test
    try:
        loo_result = leave_one_out_test(
            data=data,
            outcome=outcome,
            treatment=treatment,
            time=time,
            unit=unit,
            alpha=alpha,
            **estimator_kwargs
        )
        results["leave_one_out"] = loo_result
    except Exception as e:
        results["leave_one_out"] = {"error": str(e)}

    return results
