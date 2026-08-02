"""
Data preparation utilities for difference-in-differences analysis.

This module provides helper functions to prepare data for DiD estimation,
including creating treatment indicators, reshaping panel data, and
generating synthetic datasets for testing.

Data generation functions (generate_*) are defined in prep_dgp.py and
re-exported here for backward compatibility.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Re-export data generation functions from prep_dgp for backward compatibility
from diff_diff._deprecation import NOT_SUPPLIED, require_arg, resolve_renamed_kwarg
from diff_diff.prep_dgp import (  # noqa: F401
    generate_continuous_did_data,
    generate_ddd_data,
    generate_ddd_panel_data,
    generate_did_data,
    generate_event_study_data,
    generate_factor_data,
    generate_panel_data,
    generate_reversible_did_data,
    generate_staggered_data,
    generate_staggered_ddd_data,
    generate_survey_did_data,
    generate_synthetic_control_data,
)
from diff_diff.survey import (
    ResolvedSurveyDesign,
    SurveyDesign,
    _compute_if_variance_fast,
    _precompute_psu_scaffolding,
    _PsuScaffolding,
    compute_replicate_if_variance,
    compute_survey_if_variance,
)
from diff_diff.utils import _compute_noise_level, _sc_weight_fw

# Constants for rank_control_units
_SIMILARITY_THRESHOLD_SD = 0.5  # Controls within this many SDs are "similar"
_OUTLIER_PENALTY_WEIGHT = 0.3  # Penalty weight for outcome outliers in treatment candidate scoring


def make_treatment_indicator(
    data: pd.DataFrame,
    column: str,
    treated_values: Optional[Union[Any, List[Any]]] = None,
    threshold: Optional[float] = None,
    above_threshold: bool = True,
    new_column: str = "treated",
) -> pd.DataFrame:
    """
    Create a binary treatment indicator column from various input types.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to use for creating the treatment indicator.
    treated_values : Any or list, optional
        Value(s) that indicate treatment. Units with these values get
        treatment=1, others get treatment=0.
    threshold : float, optional
        Numeric threshold for creating treatment. Used when the treatment
        is based on a continuous variable (e.g., treat firms above median size).
    above_threshold : bool, default=True
        If True, values >= threshold are treated. If False, values <= threshold
        are treated. Only used when threshold is specified.
    new_column : str, default="treated"
        Name of the new treatment indicator column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new treatment indicator column added.

    Examples
    --------
    Create treatment from categorical variable:

    >>> df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'y': [1, 2, 3, 4]})
    >>> df = make_treatment_indicator(df, 'group', treated_values='A')
    >>> df['treated'].tolist()
    [1, 1, 0, 0]

    Create treatment from numeric threshold:

    >>> df = pd.DataFrame({'size': [10, 50, 100, 200], 'y': [1, 2, 3, 4]})
    >>> df = make_treatment_indicator(df, 'size', threshold=75)
    >>> df['treated'].tolist()
    [0, 0, 1, 1]

    Treat units below a threshold:

    >>> df = make_treatment_indicator(df, 'size', threshold=75, above_threshold=False)
    >>> df['treated'].tolist()
    [1, 1, 0, 0]
    """
    df = data.copy()

    if treated_values is not None and threshold is not None:
        raise ValueError("Specify either 'treated_values' or 'threshold', not both.")

    if treated_values is None and threshold is None:
        raise ValueError("Must specify either 'treated_values' or 'threshold'.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if treated_values is not None:
        # Convert single value to list
        if not isinstance(treated_values, (list, tuple, set)):
            treated_values = [treated_values]
        df[new_column] = df[column].isin(treated_values).astype(int)
    else:
        # Use threshold
        if above_threshold:
            df[new_column] = (df[column] >= threshold).astype(int)
        else:
            df[new_column] = (df[column] <= threshold).astype(int)

    return df


def make_post_indicator(
    data: pd.DataFrame,
    time_column: str,
    post_periods: Optional[Union[Any, List[Any]]] = None,
    treatment_start: Optional[Any] = None,
    new_column: str = "post",
) -> pd.DataFrame:
    """
    Create a binary post-treatment indicator column.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    time_column : str
        Name of the time/period column.
    post_periods : Any or list, optional
        Specific period value(s) that are post-treatment. Periods matching
        these values get post=1, others get post=0.
    treatment_start : Any, optional
        The first post-treatment period. All periods >= this value get post=1.
        Works with numeric periods, strings (sorted alphabetically), or dates.
    new_column : str, default="post"
        Name of the new post indicator column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new post indicator column added.

    Examples
    --------
    Using specific post periods:

    >>> df = pd.DataFrame({'year': [2018, 2019, 2020, 2021], 'y': [1, 2, 3, 4]})
    >>> df = make_post_indicator(df, 'year', post_periods=[2020, 2021])
    >>> df['post'].tolist()
    [0, 0, 1, 1]

    Using treatment start:

    >>> df = make_post_indicator(df, 'year', treatment_start=2020)
    >>> df['post'].tolist()
    [0, 0, 1, 1]

    Works with date columns:

    >>> df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01'])})
    >>> df = make_post_indicator(df, 'date', treatment_start='2020-06-01')
    """
    df = data.copy()

    if post_periods is not None and treatment_start is not None:
        raise ValueError("Specify either 'post_periods' or 'treatment_start', not both.")

    if post_periods is None and treatment_start is None:
        raise ValueError("Must specify either 'post_periods' or 'treatment_start'.")

    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame.")

    if post_periods is not None:
        # Convert single value to list
        if not isinstance(post_periods, (list, tuple, set)):
            post_periods = [post_periods]
        df[new_column] = df[time_column].isin(post_periods).astype(int)
    else:
        # Use treatment_start - convert to same type as column if needed
        col_dtype = df[time_column].dtype
        if pd.api.types.is_datetime64_any_dtype(col_dtype):
            treatment_start = pd.to_datetime(treatment_start)
        df[new_column] = (df[time_column] >= treatment_start).astype(int)

    return df


def wide_to_long(
    data: pd.DataFrame,
    value_columns: List[str],
    id_column: str,
    time_name: str = "period",
    value_name: str = "value",
    time_values: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Convert wide-format panel data to long format for DiD analysis.

    Wide format has one row per unit with multiple columns for each time period.
    Long format has one row per unit-period combination.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format DataFrame with one row per unit.
    value_columns : list of str
        Column names containing the outcome values for each period.
        These should be in chronological order.
    id_column : str
        Column name for the unit identifier.
    time_name : str, default="period"
        Name for the new time period column.
    value_name : str, default="value"
        Name for the new value/outcome column.
    time_values : list, optional
        Values to use for time periods. If None, uses 0, 1, 2, ...
        Must have same length as value_columns.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per unit-period.

    Examples
    --------
    >>> wide_df = pd.DataFrame({
    ...     'firm_id': [1, 2, 3],
    ...     'sales_2019': [100, 150, 200],
    ...     'sales_2020': [110, 160, 210],
    ...     'sales_2021': [120, 170, 220]
    ... })
    >>> long_df = wide_to_long(
    ...     wide_df,
    ...     value_columns=['sales_2019', 'sales_2020', 'sales_2021'],
    ...     id_column='firm_id',
    ...     time_name='year',
    ...     value_name='sales',
    ...     time_values=[2019, 2020, 2021]
    ... )
    >>> len(long_df)
    9
    >>> long_df.columns.tolist()
    ['firm_id', 'year', 'sales']
    """
    if not value_columns:
        raise ValueError("value_columns cannot be empty.")

    if id_column not in data.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")

    for col in value_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if time_values is None:
        time_values = list(range(len(value_columns)))

    if len(time_values) != len(value_columns):
        raise ValueError(
            f"time_values length ({len(time_values)}) must match "
            f"value_columns length ({len(value_columns)})."
        )

    # Get other columns to preserve (not id or value columns)
    other_cols = [c for c in data.columns if c != id_column and c not in value_columns]

    # Use pd.melt for better performance (vectorized)
    long_df = pd.melt(
        data,
        id_vars=[id_column] + other_cols,
        value_vars=value_columns,
        var_name="_temp_var",
        value_name=value_name,
    )

    # Map column names to time values
    col_to_time = dict(zip(value_columns, time_values))
    long_df[time_name] = long_df["_temp_var"].map(col_to_time)
    long_df = long_df.drop("_temp_var", axis=1)

    # Reorder columns and sort
    cols = [id_column, time_name, value_name] + other_cols
    return long_df[cols].sort_values([id_column, time_name]).reset_index(drop=True)


def balance_panel(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    method: str = "inner",
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Balance a panel dataset to ensure all units have all time periods.

    Parameters
    ----------
    data : pd.DataFrame
        Unbalanced panel data.
    unit_column : str
        Column name for unit identifier.
    time_column : str
        Column name for time period.
    method : str, default="inner"
        Balancing method:
        - "inner": Keep only units that appear in all periods (drops units)
        - "outer": Include all unit-period combinations (creates NaN)
        - "fill": Include all combinations and fill missing values
    fill_value : float, optional
        Value to fill missing observations when method="fill".
        If None with method="fill", uses column-specific forward fill.

    Returns
    -------
    pd.DataFrame
        Balanced panel DataFrame.

    Examples
    --------
    Keep only complete units:

    >>> df = pd.DataFrame({
    ...     'unit': [1, 1, 1, 2, 2, 3, 3, 3],
    ...     'period': [1, 2, 3, 1, 2, 1, 2, 3],
    ...     'y': [10, 11, 12, 20, 21, 30, 31, 32]
    ... })
    >>> balanced = balance_panel(df, 'unit', 'period', method='inner')
    >>> balanced['unit'].unique().tolist()
    [1, 3]

    Include all combinations:

    >>> balanced = balance_panel(df, 'unit', 'period', method='outer')
    >>> len(balanced)
    9
    """
    if unit_column not in data.columns:
        raise ValueError(f"Column '{unit_column}' not found in DataFrame.")
    if time_column not in data.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame.")

    if method not in ["inner", "outer", "fill"]:
        raise ValueError(f"method must be 'inner', 'outer', or 'fill', got '{method}'")

    all_units = data[unit_column].unique()
    all_periods = sorted(data[time_column].unique())
    n_periods = len(all_periods)

    if method == "inner":
        # Keep only units that have all periods
        unit_counts = data.groupby(unit_column)[time_column].nunique()
        complete_units = unit_counts[unit_counts == n_periods].index
        return data[data[unit_column].isin(complete_units)].copy()

    elif method in ["outer", "fill"]:
        # Create full grid of unit-period combinations
        full_index = pd.MultiIndex.from_product(
            [all_units, all_periods], names=[unit_column, time_column]
        )
        full_df = pd.DataFrame(index=full_index).reset_index()

        # Merge with original data
        result = full_df.merge(data, on=[unit_column, time_column], how="left")

        if method == "fill":
            # Identify columns to fill (exclude unit and time columns)
            cols_to_fill = [c for c in result.columns if c not in [unit_column, time_column]]

            if fill_value is not None:
                # Fill specified columns with fill_value
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in cols_to_fill:
                        result[col] = result[col].fillna(fill_value)
            else:
                # Forward fill within each unit for non-key columns
                result = result.sort_values([unit_column, time_column])
                result[cols_to_fill] = result.groupby(unit_column)[cols_to_fill].ffill()
                # Backward fill any remaining NaN at start
                result[cols_to_fill] = result.groupby(unit_column)[cols_to_fill].bfill()

        return result

    return data


def validate_did_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: Optional[str] = None,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Validate that data is properly formatted for DiD analysis.

    Checks for common data issues and provides informative error messages.

    Parameters
    ----------
    data : pd.DataFrame
        Data to validate.
    outcome : str
        Name of outcome variable column.
    treatment : str
        Name of treatment indicator column.
    time : str
        Name of time/post indicator column.
    unit : str, optional
        Name of unit identifier column (for panel data validation).
    raise_on_error : bool, default=True
        If True, raises ValueError on validation failures.
        If False, returns validation results without raising.

    Returns
    -------
    dict
        Validation results with keys:
        - valid: bool indicating if data passed all checks
        - errors: list of error messages
        - warnings: list of warning messages
        - summary: dict with data summary statistics

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'y': [1, 2, 3, 4],
    ...     'treated': [0, 0, 1, 1],
    ...     'post': [0, 1, 0, 1]
    ... })
    >>> result = validate_did_data(df, 'y', 'treated', 'post', raise_on_error=False)
    >>> result['valid']
    True
    """
    errors = []
    warnings = []

    # Check columns exist
    required_cols = [outcome, treatment, time]
    if unit is not None:
        required_cols.append(unit)

    for col in required_cols:
        if col not in data.columns:
            errors.append(f"Required column '{col}' not found in DataFrame.")

    if errors:
        if raise_on_error:
            raise ValueError("\n".join(errors))
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # Check outcome is numeric
    if not pd.api.types.is_numeric_dtype(data[outcome]):
        errors.append(
            f"Outcome column '{outcome}' must be numeric. " f"Got type: {data[outcome].dtype}"
        )

    # Check treatment is binary
    treatment_vals = data[treatment].dropna().unique()
    if not set(treatment_vals).issubset({0, 1}):
        errors.append(
            f"Treatment column '{treatment}' must be binary (0 or 1). "
            f"Found values: {sorted(treatment_vals)}"
        )

    # Check time is binary for simple DiD
    time_vals = data[time].dropna().unique()
    if len(time_vals) == 2 and not set(time_vals).issubset({0, 1}):
        warnings.append(
            f"Time column '{time}' has 2 values but they are not 0 and 1: {sorted(time_vals)}. "
            "For basic DiD, use 0 for pre-treatment and 1 for post-treatment."
        )

    # Check for missing values
    for col in required_cols:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            errors.append(
                f"Column '{col}' has {n_missing} missing values. "
                "Please handle missing data before fitting."
            )

    # Calculate summary statistics
    summary = {}
    if not errors:
        summary["n_obs"] = len(data)
        summary["n_treated"] = int((data[treatment] == 1).sum())
        summary["n_control"] = int((data[treatment] == 0).sum())
        summary["n_periods"] = len(time_vals)

        if unit is not None:
            summary["n_units"] = data[unit].nunique()

        # Check for sufficient variation
        if summary["n_treated"] == 0:
            errors.append("No treated observations found (treatment column is all 0).")
        if summary["n_control"] == 0:
            errors.append("No control observations found (treatment column is all 1).")

        # Check for each treatment-time combination
        if len(time_vals) == 2:
            # For 2-period DiD, check all four cells
            for t_val in [0, 1]:
                for p_val in time_vals:
                    count = len(data[(data[treatment] == t_val) & (data[time] == p_val)])
                    if count == 0:
                        errors.append(
                            f"No observations for treatment={t_val}, time={p_val}. "
                            "DiD requires observations in all treatment-time cells."
                        )
        else:
            # For multi-period, check that both treatment groups exist in multiple periods
            for t_val in [0, 1]:
                n_periods_with_obs = data[data[treatment] == t_val][time].nunique()
                if n_periods_with_obs < 2:
                    group_name = "Treated" if t_val == 1 else "Control"
                    errors.append(
                        f"{group_name} group has observations in only {n_periods_with_obs} period(s). "
                        "DiD requires multiple periods per group."
                    )

    # Panel-specific validation
    if unit is not None and not errors:
        # Check treatment is constant within units
        unit_treatment_var = data.groupby(unit)[treatment].nunique()
        units_with_varying_treatment = unit_treatment_var[unit_treatment_var > 1]
        if len(units_with_varying_treatment) > 0:
            warnings.append(
                f"Treatment varies within {len(units_with_varying_treatment)} unit(s). "
                "For standard DiD, treatment should be constant within units. "
                "This may be intentional for staggered adoption designs."
            )

        # Check panel balance
        periods_per_unit = data.groupby(unit)[time].nunique()
        if periods_per_unit.min() != periods_per_unit.max():
            warnings.append(
                f"Unbalanced panel detected. Units have between "
                f"{periods_per_unit.min()} and {periods_per_unit.max()} periods. "
                "Consider using balance_panel() to balance the data."
            )

    valid = len(errors) == 0

    if raise_on_error and not valid:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))

    return {"valid": valid, "errors": errors, "warnings": warnings, "summary": summary}


def summarize_did_data(
    data: pd.DataFrame, outcome: str, treatment: str, time: str, unit: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate summary statistics by treatment group and time period.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    outcome : str
        Name of outcome variable column.
    treatment : str
        Name of treatment indicator column.
    time : str
        Name of time/period column.
    unit : str, optional
        Name of unit identifier column.

    Returns
    -------
    pd.DataFrame
        Summary statistics with columns for each treatment-time combination.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'y': [10, 11, 12, 13, 20, 21, 22, 23],
    ...     'treated': [0, 0, 1, 1, 0, 0, 1, 1],
    ...     'post': [0, 1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> summary = summarize_did_data(df, 'y', 'treated', 'post')
    >>> print(summary)
    """
    # Group by treatment and time
    summary = (
        data.groupby([treatment, time])[outcome]
        .agg([("n", "count"), ("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")])
        .round(4)
    )

    # Calculate time values for labeling
    time_vals = sorted(data[time].unique())

    # Add group labels based on sorted time values (not literal 0/1)
    if len(time_vals) == 2:
        pre_val, post_val = time_vals[0], time_vals[1]

        def format_label(x: tuple) -> str:
            treatment_label = "Treated" if x[0] == 1 else "Control"
            time_label = "Post" if x[1] == post_val else "Pre"
            return f"{treatment_label} - {time_label}"

        summary.index = summary.index.map(format_label)

        # Calculate means for each cell
        treated_pre = data[(data[treatment] == 1) & (data[time] == pre_val)][outcome].mean()
        treated_post = data[(data[treatment] == 1) & (data[time] == post_val)][outcome].mean()
        control_pre = data[(data[treatment] == 0) & (data[time] == pre_val)][outcome].mean()
        control_post = data[(data[treatment] == 0) & (data[time] == post_val)][outcome].mean()

        # Calculate DiD
        treated_diff = treated_post - treated_pre
        control_diff = control_post - control_pre
        did_estimate = treated_diff - control_diff

        # Add to summary as a new row
        did_row = pd.DataFrame(
            {"n": ["-"], "mean": [did_estimate], "std": ["-"], "min": ["-"], "max": ["-"]},
            index=["DiD Estimate"],
        )
        summary = pd.concat([summary, did_row])
    else:
        summary.index = summary.index.map(
            lambda x: f"{'Treated' if x[0] == 1 else 'Control'} - Period {x[1]}"
        )

    return summary


def create_event_time(
    data: pd.DataFrame, time_column: str, treatment_time_column: str, new_column: str = "event_time"
) -> pd.DataFrame:
    """
    Create an event-time column relative to treatment timing.

    Useful for event study designs where treatment occurs at different
    times for different units.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    time_column : str
        Name of the calendar time column.
    treatment_time_column : str
        Name of the column indicating when each unit was treated.
        Units with NaN or infinity are considered never-treated.
    new_column : str, default="event_time"
        Name of the new event-time column.

    Returns
    -------
    pd.DataFrame
        DataFrame with event-time column added. Values are:
        - Negative for pre-treatment periods
        - 0 for the treatment period
        - Positive for post-treatment periods
        - NaN for never-treated units

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'unit': [1, 1, 1, 2, 2, 2],
    ...     'year': [2018, 2019, 2020, 2018, 2019, 2020],
    ...     'treatment_year': [2019, 2019, 2019, 2020, 2020, 2020]
    ... })
    >>> df = create_event_time(df, 'year', 'treatment_year')
    >>> df['event_time'].tolist()
    [-1, 0, 1, -2, -1, 0]
    """
    df = data.copy()

    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame.")
    if treatment_time_column not in df.columns:
        raise ValueError(f"Column '{treatment_time_column}' not found in DataFrame.")

    # Calculate event time
    df[new_column] = df[time_column] - df[treatment_time_column]

    # Handle never-treated (inf or NaN in treatment time)
    col = df[treatment_time_column]
    if pd.api.types.is_numeric_dtype(col):
        never_treated = col.isna() | np.isinf(col)
    else:
        never_treated = col.isna()
    df.loc[never_treated, new_column] = np.nan

    return df


def aggregate_to_cohorts(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    treatment_column: str,
    outcome: str,
    covariates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate unit-level data to treatment cohort means.

    Useful for visualization and cohort-level analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Unit-level panel data.
    unit_column : str
        Name of unit identifier column.
    time_column : str
        Name of time period column.
    treatment_column : str
        Name of treatment indicator column.
    outcome : str
        Name of outcome variable column.
    covariates : list of str, optional
        Additional columns to aggregate (will compute means).

    Returns
    -------
    pd.DataFrame
        Cohort-level data with mean outcomes by treatment status and period.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'unit': [1, 1, 2, 2, 3, 3, 4, 4],
    ...     'period': [0, 1, 0, 1, 0, 1, 0, 1],
    ...     'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    ...     'y': [10, 15, 12, 17, 8, 10, 9, 11]
    ... })
    >>> cohort_df = aggregate_to_cohorts(df, 'unit', 'period', 'treated', 'y')
    >>> len(cohort_df)
    4
    """
    agg_cols = {outcome: "mean", unit_column: "nunique"}

    if covariates:
        for cov in covariates:
            agg_cols[cov] = "mean"

    cohort_data = data.groupby([treatment_column, time_column]).agg(agg_cols).reset_index()

    # Rename columns
    cohort_data = cohort_data.rename(columns={unit_column: "n_units", outcome: f"mean_{outcome}"})

    return cohort_data


def rank_control_units(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    outcome_column: str,
    treatment_column: Optional[str] = None,
    treated_units: Optional[List[Any]] = None,
    pre_periods: Optional[List[Any]] = None,
    covariates: Optional[List[str]] = None,
    outcome_weight: float = 0.7,
    covariate_weight: float = 0.3,
    exclude_units: Optional[List[Any]] = None,
    require_units: Optional[List[Any]] = None,
    n_top: Optional[int] = None,
    suggest_treatment_candidates: bool = False,
    n_treatment_candidates: int = 5,
    lambda_reg: float = 0.0,
) -> pd.DataFrame:
    """
    Rank potential control units by their suitability for DiD analysis.

    Evaluates control units based on pre-treatment outcome trend similarity
    and optional covariate matching to treated units. Returns a ranked list
    with quality scores.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    unit_column : str
        Column name for unit identifier.
    time_column : str
        Column name for time periods.
    outcome_column : str
        Column name for outcome variable.
    treatment_column : str, optional
        Column with binary treatment indicator (0/1). Used to identify
        treated units from data.
    treated_units : list, optional
        Explicit list of treated unit IDs. Alternative to treatment_column.
    pre_periods : list, optional
        Pre-treatment periods for comparison. If None, uses first half of periods.
    covariates : list of str, optional
        Covariate columns for matching. Similarity is based on pre-treatment means.
    outcome_weight : float, default=0.7
        Weight for pre-treatment outcome trend similarity (0-1).
    covariate_weight : float, default=0.3
        Weight for covariate distance (0-1). Ignored if no covariates.
    exclude_units : list, optional
        Units that cannot be in control group.
    require_units : list, optional
        Units that must be in control group (will always appear in output).
    n_top : int, optional
        Return only top N control units. If None, return all.
    suggest_treatment_candidates : bool, default=False
        If True and no treated units specified, identify potential treatment
        candidates instead of ranking controls.
    n_treatment_candidates : int, default=5
        Number of treatment candidates to suggest.
    lambda_reg : float, default=0.0
        Regularization for synthetic weights. Higher values give more uniform
        weights across controls.

    Returns
    -------
    pd.DataFrame
        Ranked control units with columns:

        - unit: Unit identifier
        - quality_score: Combined quality score (0-1, higher is better)
        - outcome_trend_score: Pre-treatment outcome trend similarity
        - covariate_score: Covariate match score (NaN if no covariates)
        - synthetic_weight: Informational heuristic weight from a single-pass
          uncentered Frank-Wolfe solve; does NOT factor into ``quality_score``
          (ranking) and is NOT the canonical SDID unit weight. For canonical
          SDID weights use ``SyntheticDiD.fit()``.
        - pre_trend_rmse: RMSE of pre-treatment outcome vs treated mean
        - is_required: Whether unit was in require_units

        If suggest_treatment_candidates=True (and no treated units):

        - unit: Unit identifier
        - treatment_candidate_score: Suitability as treatment unit
        - avg_outcome_level: Pre-treatment outcome mean
        - outcome_trend: Pre-treatment trend slope
        - n_similar_controls: Count of similar potential controls

    Examples
    --------
    Rank controls against treated units:

    >>> data = generate_did_data(n_units=30, n_periods=6, seed=42)
    >>> ranking = rank_control_units(
    ...     data,
    ...     unit_column='unit',
    ...     time_column='period',
    ...     outcome_column='outcome',
    ...     treatment_column='treated',
    ...     n_top=10
    ... )
    >>> ranking['quality_score'].is_monotonic_decreasing
    True

    With covariates:

    >>> data['size'] = np.random.randn(len(data))
    >>> ranking = rank_control_units(
    ...     data,
    ...     unit_column='unit',
    ...     time_column='period',
    ...     outcome_column='outcome',
    ...     treatment_column='treated',
    ...     covariates=['size']
    ... )

    Filter data for SyntheticDiD:

    >>> top_controls = ranking['unit'].tolist()
    >>> filtered = data[(data['treated'] == 1) | (data['unit'].isin(top_controls))]
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    for col in [unit_column, time_column, outcome_column]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if treatment_column is not None and treatment_column not in data.columns:
        raise ValueError(f"Treatment column '{treatment_column}' not found in DataFrame.")

    if covariates:
        for cov in covariates:
            if cov not in data.columns:
                raise ValueError(f"Covariate column '{cov}' not found in DataFrame.")

    if not 0 <= outcome_weight <= 1:
        raise ValueError("outcome_weight must be between 0 and 1")
    if not 0 <= covariate_weight <= 1:
        raise ValueError("covariate_weight must be between 0 and 1")

    if treated_units is not None and treatment_column is not None:
        raise ValueError("Specify either 'treated_units' or 'treatment_column', not both.")

    if require_units and exclude_units:
        invalid_required = [u for u in require_units if u in exclude_units]
        if invalid_required:
            raise ValueError(f"Units cannot be both required and excluded: {invalid_required}")

    # -------------------------------------------------------------------------
    # Determine pre-treatment periods
    # -------------------------------------------------------------------------
    all_periods = sorted(data[time_column].unique())
    if pre_periods is None:
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]
    else:
        pre_periods = list(pre_periods)

    if len(pre_periods) == 0:
        raise ValueError("No pre-treatment periods specified or inferred.")

    # -------------------------------------------------------------------------
    # Identify treated and control units
    # -------------------------------------------------------------------------
    all_units = list(data[unit_column].unique())

    if treated_units is not None:
        treated_set = set(treated_units)
    elif treatment_column is not None:
        unit_treatment = data.groupby(unit_column)[treatment_column].first()
        treated_set = set(unit_treatment[unit_treatment == 1].index)
    elif suggest_treatment_candidates:
        # Treatment candidate discovery mode - no treated units
        treated_set = set()
    else:
        raise ValueError(
            "Must specify treated_units, treatment_column, or set "
            "suggest_treatment_candidates=True"
        )

    # -------------------------------------------------------------------------
    # Treatment candidate discovery mode
    # -------------------------------------------------------------------------
    if suggest_treatment_candidates and len(treated_set) == 0:
        return _suggest_treatment_candidates(
            data, unit_column, time_column, outcome_column, pre_periods, n_treatment_candidates
        )

    if len(treated_set) == 0:
        raise ValueError("No treated units found.")

    # Determine control candidates
    control_candidates = [u for u in all_units if u not in treated_set]

    if exclude_units:
        control_candidates = [u for u in control_candidates if u not in exclude_units]

    if len(control_candidates) == 0:
        raise ValueError("No control units available after exclusions.")

    # -------------------------------------------------------------------------
    # Create outcome matrices (pre-treatment)
    # -------------------------------------------------------------------------
    pre_data = data[data[time_column].isin(pre_periods)]
    pivot = pre_data.pivot(index=time_column, columns=unit_column, values=outcome_column)

    # Filter to pre_periods that exist in data
    valid_pre_periods = [p for p in pre_periods if p in pivot.index]
    if len(valid_pre_periods) == 0:
        raise ValueError("No data found for specified pre-treatment periods.")

    # Filter control_candidates to those present in pivot (handles unbalanced panels)
    control_candidates = [c for c in control_candidates if c in pivot.columns]
    if len(control_candidates) == 0:
        raise ValueError("No control units found in pre-treatment data.")

    # Control outcomes: shape (n_pre_periods, n_control_candidates)
    Y_control = pivot.loc[valid_pre_periods, control_candidates].values.astype(float)

    # Treated outcomes mean: shape (n_pre_periods,)
    treated_list = [u for u in treated_set if u in pivot.columns]
    if len(treated_list) == 0:
        raise ValueError("Treated units not found in pre-treatment data.")
    Y_treated_mean = pivot.loc[valid_pre_periods, treated_list].mean(axis=1).values.astype(float)

    # -------------------------------------------------------------------------
    # Compute outcome trend scores
    # -------------------------------------------------------------------------
    # Informational `synthetic_weight` column. This is a RANKING HEURISTIC,
    # not an estimator: it gives a rough "which controls would a synthetic
    # regression weight heavily" signal that's reported alongside RMSE and
    # covariate distance. The actual ranking (`quality_score`) is computed
    # below from `outcome_trend_score` (RMSE-based) + `covariate_score`; the
    # `synthetic_weight` column does NOT factor into the ranking decision.
    #
    # Solver choice. We use a single-pass uncentered Frank-Wolfe via the
    # shared `_sc_weight_fw` dispatcher to solve:
    #
    #     min_w  ||Y_treated_mean - Y_control @ w||^2 + lambda_reg * ||w||^2
    #         s.t. w >= 0, sum(w) = 1
    #
    # Mapped to the FW objective `zeta^2 ||w||^2 + (1/N) ||Aw - b||^2` via
    # `zeta = sqrt(lambda_reg / N)`. intercept=False because this QP does
    # no column-centering, max_iter=1000 to bound ranking-loop cost,
    # min_weight=1e-6 post-processing for interpretability.
    #
    # NOTE — this is INTENTIONALLY NOT the canonical SDID / R
    # `synthdid::sc.weight.fw` two-pass unit-weight procedure (that uses
    # intercept=TRUE, 100-iter -> sparsify -> 10000-iter). SDID estimation
    # still uses that canonical path in `_sc_weight_fw_numpy` at
    # `utils.py:_sc_weight_fw_numpy` via `compute_sdid_unit_weights`; this
    # ranking heuristic uses a simpler single-pass call to the same solver
    # for a cheap diagnostic score.
    #
    # Replaces the former `compute_synthetic_weights` wrapper whose Rust
    # and Python backends had divergent PGD implementations (audit
    # finding #22). Net effect: users on default `lambda_reg=0` with
    # typical data see `synthetic_weight` values that agree with the old
    # code to ~1e-7; extreme Y or `lambda_reg > 0` cases produce values
    # that differ from the old code (which was mathematically wrong).
    _Y_control = np.ascontiguousarray(Y_control, dtype=np.float64)
    _Y_treated_mean = np.ascontiguousarray(Y_treated_mean, dtype=np.float64)
    _n_pre, _n_control = _Y_control.shape
    if _n_control == 0:
        synthetic_weights = np.array([], dtype=np.float64)
    elif _n_control == 1:
        synthetic_weights = np.array([1.0])
    else:
        _zeta = float(np.sqrt(lambda_reg / _n_pre)) if lambda_reg > 0 else 0.0
        # Scale stopping threshold by noise level so convergence stays
        # meaningful at any data magnitude.
        _sigma = _compute_noise_level(_Y_control)
        _min_decrease = 1e-5 * max(_sigma, 1e-12)
        _Y_fw = np.column_stack([_Y_control, _Y_treated_mean])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*did not converge.*",
                category=UserWarning,
            )
            synthetic_weights = _sc_weight_fw(
                _Y_fw,
                zeta=_zeta,
                intercept=False,
                min_decrease=_min_decrease,
                max_iter=1000,
            )
        # Set small weights to zero for interpretability, then renormalize.
        synthetic_weights = np.asarray(synthetic_weights, dtype=np.float64)
        _min_weight = 1e-6
        synthetic_weights[synthetic_weights < _min_weight] = 0.0
        _total = float(np.sum(synthetic_weights))
        if _total > 0:
            synthetic_weights = synthetic_weights / _total
        else:
            synthetic_weights = np.ones(_n_control) / _n_control

    # RMSE for each control vs treated mean (use nanmean to handle missing data)
    rmse_scores = []
    for j in range(len(control_candidates)):
        y_c = Y_control[:, j]
        rmse = np.sqrt(np.nanmean((y_c - Y_treated_mean) ** 2))
        rmse_scores.append(rmse)

    # Convert RMSE to similarity score (lower RMSE = higher score)
    max_rmse = max(rmse_scores) if rmse_scores else 1.0
    min_rmse = min(rmse_scores) if rmse_scores else 0.0
    rmse_range = max_rmse - min_rmse

    if rmse_range < 1e-10:
        # All controls have identical/similar pre-trends (includes single control case)
        outcome_trend_scores = [1.0] * len(rmse_scores)
    else:
        # Normalize so best control gets 1.0, worst gets 0.0
        outcome_trend_scores = [1 - (rmse - min_rmse) / rmse_range for rmse in rmse_scores]

    # -------------------------------------------------------------------------
    # Compute covariate scores (if covariates provided)
    # -------------------------------------------------------------------------
    if covariates and len(covariates) > 0:
        # Get unit-level covariate values (pre-treatment mean)
        cov_data = pre_data.groupby(unit_column)[covariates].mean()

        # Treated covariate profile (mean across treated units)
        treated_cov = cov_data.loc[list(treated_set)].mean()

        # Standardize covariates
        cov_mean = cov_data.mean()
        cov_std = cov_data.std().replace(0, 1)  # Avoid division by zero
        cov_standardized = (cov_data - cov_mean) / cov_std
        treated_cov_std = (treated_cov - cov_mean) / cov_std

        # Euclidean distance in standardized space (vectorized)
        control_cov_matrix = cov_standardized.loc[control_candidates].values
        treated_cov_vector = treated_cov_std.values
        covariate_distances = np.sqrt(
            np.sum((control_cov_matrix - treated_cov_vector) ** 2, axis=1)
        )

        # Convert distance to similarity score (min-max normalization)
        max_dist = covariate_distances.max() if len(covariate_distances) > 0 else 1.0
        min_dist = covariate_distances.min() if len(covariate_distances) > 0 else 0.0
        dist_range = max_dist - min_dist

        if dist_range < 1e-10:
            # All controls have identical/similar covariate profiles
            covariate_scores = [1.0] * len(covariate_distances)
        else:
            # Normalize so best control (closest) gets 1.0, worst gets 0.0
            covariate_scores = (1 - (covariate_distances - min_dist) / dist_range).tolist()
    else:
        covariate_scores = [np.nan] * len(control_candidates)

    # -------------------------------------------------------------------------
    # Compute combined quality score
    # -------------------------------------------------------------------------
    # Normalize weights
    total_weight = outcome_weight + covariate_weight
    if total_weight > 0:
        norm_outcome_weight = outcome_weight / total_weight
        norm_covariate_weight = covariate_weight / total_weight
    else:
        norm_outcome_weight = 1.0
        norm_covariate_weight = 0.0

    quality_scores = []
    for i in range(len(control_candidates)):
        outcome_score = outcome_trend_scores[i]
        cov_score = covariate_scores[i]

        if np.isnan(cov_score):
            # No covariates - use only outcome score
            combined = outcome_score
        else:
            combined = norm_outcome_weight * outcome_score + norm_covariate_weight * cov_score

        quality_scores.append(combined)

    # -------------------------------------------------------------------------
    # Build result DataFrame
    # -------------------------------------------------------------------------
    require_set = set(require_units) if require_units else set()

    result = pd.DataFrame(
        {
            "unit": control_candidates,
            "quality_score": quality_scores,
            "outcome_trend_score": outcome_trend_scores,
            "covariate_score": covariate_scores,
            "synthetic_weight": synthetic_weights,
            "pre_trend_rmse": rmse_scores,
            "is_required": [u in require_set for u in control_candidates],
        }
    )

    # Sort by quality score (descending)
    result = result.sort_values("quality_score", ascending=False)

    # Apply n_top limit if specified
    if n_top is not None and n_top < len(result):
        # Always include required units
        required_df = result[result["is_required"]]
        non_required_df = result[~result["is_required"]]

        # Take top from non-required to fill remaining slots
        remaining_slots = max(0, n_top - len(required_df))
        top_non_required = non_required_df.head(remaining_slots)

        result = pd.concat([required_df, top_non_required])
        result = result.sort_values("quality_score", ascending=False)

    return result.reset_index(drop=True)


def _suggest_treatment_candidates(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    outcome_column: str,
    pre_periods: List[Any],
    n_candidates: int,
) -> pd.DataFrame:
    """
    Identify units that would make good treatment candidates.

    A good treatment candidate:
    1. Has many similar control units available (for matching)
    2. Has stable pre-treatment trends (predictable counterfactual)
    3. Is not an extreme outlier

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    unit_column : str
        Unit identifier column.
    time_column : str
        Time period column.
    outcome_column : str
        Outcome variable column.
    pre_periods : list
        Pre-treatment periods.
    n_candidates : int
        Number of candidates to return.

    Returns
    -------
    pd.DataFrame
        Treatment candidates with scores.
    """
    all_units = list(data[unit_column].unique())
    pre_data = data[data[time_column].isin(pre_periods)]

    candidate_info = []

    for unit in all_units:
        unit_data = pre_data[pre_data[unit_column] == unit]

        if len(unit_data) == 0:
            continue

        # Average outcome level
        avg_outcome = unit_data[outcome_column].mean()

        # Trend (simple linear regression slope)
        times = unit_data[time_column].values
        outcomes = unit_data[outcome_column].values
        if len(times) > 1:
            times_norm = np.arange(len(times))
            try:
                slope = np.polyfit(times_norm, outcomes, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
        else:
            slope = 0.0

        # Count similar potential controls
        other_units = [u for u in all_units if u != unit]
        other_means = (
            pre_data[pre_data[unit_column].isin(other_units)]
            .groupby(unit_column)[outcome_column]
            .mean()
        )

        if len(other_means) > 0:
            sd = other_means.std()
            if sd > 0:
                n_similar = int(
                    np.sum(np.abs(other_means - avg_outcome) < _SIMILARITY_THRESHOLD_SD * sd)
                )
            else:
                n_similar = len(other_means)
        else:
            n_similar = 0

        candidate_info.append(
            {
                "unit": unit,
                "avg_outcome_level": avg_outcome,
                "outcome_trend": slope,
                "n_similar_controls": n_similar,
            }
        )

    if len(candidate_info) == 0:
        return pd.DataFrame(
            columns=[
                "unit",
                "treatment_candidate_score",
                "avg_outcome_level",
                "outcome_trend",
                "n_similar_controls",
            ]
        )

    result = pd.DataFrame(candidate_info)

    # Score: prefer units with many similar controls and moderate outcome levels
    max_similar = result["n_similar_controls"].max()
    if max_similar > 0:
        similarity_score = result["n_similar_controls"] / max_similar
    else:
        similarity_score = pd.Series([0.0] * len(result))

    # Penalty for outliers in outcome level
    outcome_mean = result["avg_outcome_level"].mean()
    outcome_std = result["avg_outcome_level"].std()
    if outcome_std > 0:
        outcome_z = np.abs((result["avg_outcome_level"] - outcome_mean) / outcome_std)
    else:
        outcome_z = pd.Series([0.0] * len(result))

    result["treatment_candidate_score"] = (
        similarity_score - _OUTLIER_PENALTY_WEIGHT * outcome_z
    ).clip(0, 1)

    # Return top candidates
    result = result.nlargest(n_candidates, "treatment_candidate_score")
    return result.reset_index(drop=True)


def trim_weights(
    data: pd.DataFrame,
    weights: Any = NOT_SUPPLIED,
    upper: Optional[float] = None,
    quantile: Optional[float] = None,
    lower: Optional[float] = None,
    weight_col: Any = NOT_SUPPLIED,
) -> pd.DataFrame:
    """Trim (winsorize) survey weights to reduce influence of extreme values.

    Caps weights at specified thresholds. Useful for reducing variance from
    extreme survey weights before DiD estimation. Federal agencies (e.g., NCHS)
    recommend reviewing weights with CV > 30%.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    weights : str
        Name of the weight column.
    upper : float, optional
        Absolute upper cap. Weights above this value are set to it.
        Mutually exclusive with ``quantile``.
    quantile : float, optional
        Quantile-based upper cap (e.g., 0.99). Weights above the quantile
        value are capped at it. Mutually exclusive with ``upper``.
    lower : float, optional
        Absolute lower floor. Weights below this value are set to it.
        Can be combined with either ``upper`` or ``quantile``.
    weight_col : str, optional
        Deprecated alias for ``weights`` (row M-113); warns with
        ``FutureWarning`` and will be removed in 4.0.

    Returns
    -------
    pd.DataFrame
        Copy of data with trimmed weights.

    Raises
    ------
    ValueError
        If both ``upper`` and ``quantile`` are provided, or if ``weights``
        is not in the DataFrame.
    """
    weights = resolve_renamed_kwarg(
        "trim_weights",
        "weight_col",
        weight_col,
        "weights",
        weights,
        default=NOT_SUPPLIED,
    )
    require_arg("trim_weights", "weights", weights)
    # Body-local name; the public parameter is weights (M-113).
    weight_col = weights
    if upper is not None and quantile is not None:
        raise ValueError("Specify either 'upper' or 'quantile', not both.")
    if weight_col not in data.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame.")

    result = data.copy()
    w = result[weight_col].values.copy()

    if quantile is not None:
        if not (0 < quantile < 1):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        upper = float(np.nanquantile(w, quantile))

    # Validate cap values are finite and non-negative
    if upper is not None:
        if not np.isfinite(upper) or upper < 0:
            raise ValueError(f"upper must be finite and >= 0, got {upper}")
    if lower is not None:
        if not np.isfinite(lower) or lower < 0:
            raise ValueError(f"lower must be finite and >= 0, got {lower}")
    if upper is not None and lower is not None and lower > upper:
        raise ValueError(
            f"lower ({lower}) must be <= upper ({upper}). "
            f"When using quantile, the resolved upper cap may be below lower."
        )

    if upper is not None:
        w = np.minimum(w, upper)
    if lower is not None:
        w = np.maximum(w, lower)

    result[weight_col] = w
    return result


# ---------------------------------------------------------------------------
# Survey aggregation helpers
# ---------------------------------------------------------------------------


def _cell_mean_variance(
    y_full: np.ndarray,
    full_resolved: ResolvedSurveyDesign,
    cell_mask: np.ndarray,
    min_n: int,
    scaffolding: Optional[_PsuScaffolding] = None,
) -> Tuple[float, float, int, bool]:
    """Compute design-based mean and variance of the weighted mean for one cell.

    Uses full-design domain estimation: the influence function is zero-padded
    outside the cell, preserving the full strata/PSU structure for variance
    estimation. This is the methodologically correct approach for domain
    estimation under complex survey designs (Lumley 2004, Section 3.4).

    Parameters
    ----------
    y_full : np.ndarray
        Outcome values for the full dataset (may contain NaN).
    full_resolved : ResolvedSurveyDesign
        Full-sample resolved survey design.
    cell_mask : np.ndarray
        Boolean mask identifying cell members in the full dataset.
    min_n : int
        Minimum valid observations for design-based variance. Below this
        threshold, SRS fallback is used.

    Returns
    -------
    mean : float
        Design-weighted cell mean.
    variance : float
        Design-based variance of the cell mean (>= 0). Uses SRS fallback
        when the design-based estimate is unidentifiable or n_valid < min_n.
    n_valid : int
        Number of non-missing observations in the cell.
    used_srs_fallback : bool
        True if SRS variance was used instead of design-based.
    """
    y_cell = y_full[cell_mask]
    w_cell = full_resolved.weights[cell_mask]
    # Valid = non-missing AND positive weight (zero-weight rows are padding)
    valid = ~np.isnan(y_cell) & (w_cell > 0)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return np.nan, np.nan, 0, False

    if n_valid < 2:
        y_bar = float(y_cell[valid][0])
        return y_bar, np.nan, 1, False

    # Weighted mean from cell members (NaN-safe)
    w_valid = w_cell * valid.astype(np.float64)
    y_clean = np.where(valid, y_cell, 0.0)
    sum_w = float(np.sum(w_valid))

    if sum_w <= 0:
        return np.nan, np.nan, n_valid, False

    y_bar = float(np.sum(w_valid * y_clean) / sum_w)

    # SRS fallback if below min_n threshold
    # Normalize positive weights to mean=1 so fallback is scale-invariant
    # (replicate designs preserve raw weight scale per survey.py:L189-240)
    used_srs = False
    if n_valid < min_n:
        w_norm = w_valid.copy()
        w_pos = w_norm[w_norm > 0]
        if len(w_pos) > 0:
            w_norm[w_norm > 0] = w_pos / w_pos.mean()
        sum_wn = float(np.sum(w_norm))
        resid_sq = w_norm * (y_clean - y_bar) ** 2
        variance = float(np.sum(resid_sq) / (sum_wn**2) * n_valid / (n_valid - 1))
        return y_bar, max(variance, 0.0), n_valid, True

    # Full-design domain estimation: construct full-length psi with zeros
    # outside the cell, preserving full strata/PSU structure for variance
    n_total = len(y_full)
    psi = np.zeros(n_total)
    # Positions in full array where cell member has valid data
    cell_indices = np.where(cell_mask)[0]
    valid_positions = cell_indices[valid]
    psi[valid_positions] = w_valid[valid] * (y_clean[valid] - y_bar) / sum_w

    # Route to TSL or replicate variance using the full design.  When a
    # design-level scaffolding is provided (aggregate_survey's fast path),
    # use it to skip the per-call pandas groupby / np.unique setup that
    # otherwise dominates runtime at BRFSS scale.
    if full_resolved.uses_replicate_variance:
        variance, _ = compute_replicate_if_variance(psi, full_resolved)
    elif scaffolding is not None:
        variance = _compute_if_variance_fast(psi, scaffolding)
    else:
        variance = compute_survey_if_variance(psi, full_resolved)

    # SRS fallback when design-based variance is unidentifiable
    if np.isnan(variance):
        w_norm = w_valid.copy()
        w_pos = w_norm[w_norm > 0]
        if len(w_pos) > 0:
            w_norm[w_norm > 0] = w_pos / w_pos.mean()
        sum_wn = float(np.sum(w_norm))
        resid_sq = w_norm * (y_clean - y_bar) ** 2
        variance = float(np.sum(resid_sq) / (sum_wn**2) * n_valid / (n_valid - 1))
        used_srs = True

    return y_bar, max(float(variance), 0.0), n_valid, used_srs


def aggregate_survey(
    data: pd.DataFrame,
    by: Union[str, List[str]],
    outcomes: Union[str, List[str]],
    survey_design: SurveyDesign,
    covariates: Optional[Union[str, List[str]]] = None,
    min_n: int = 2,
    lonely_psu: Optional[str] = None,
    second_stage_weights: str = "pweight",
) -> Tuple[pd.DataFrame, SurveyDesign]:
    """Aggregate survey microdata to geographic-period cells with design-based precision.

    Computes design-weighted cell means and their Taylor-linearized (or
    replicate-based) standard errors for each cell defined by the ``by``
    columns. Returns a panel-ready DataFrame and a pre-configured
    :class:`SurveyDesign` for second-stage DiD estimation.

    Each cell is treated as a subpopulation/domain of the full survey
    design: influence function values are zero-padded outside the cell,
    preserving full strata/PSU structure for variance estimation per
    Lumley (2004) Section 3.4.

    Parameters
    ----------
    data : pd.DataFrame
        Individual-level microdata.
    by : str or list of str
        Columns defining cells (e.g., ``["state", "year"]``). The first
        element is used as the clustering variable in the returned
        SurveyDesign (geographic unit for second-stage inference).
    outcomes : str or list of str
        Outcome variable(s) to aggregate with full precision tracking.
        Each outcome produces ``{name}_mean``, ``{name}_se``,
        ``{name}_n``, and ``{name}_precision`` columns. When multiple
        outcomes are given, panel filtering (non-estimable cell
        removal, zero-weight PSU pruning) is based on the **first**
        outcome only, consistent with the returned SurveyDesign. For
        independent per-outcome support, call once per outcome.
    survey_design : SurveyDesign
        Survey design specification for the microdata.
    covariates : str or list of str, optional
        Additional variables to aggregate as design-weighted means only
        (no SE/precision columns).
    min_n : int, default 2
        Minimum respondents per cell. Cells below this threshold use
        simple random sampling variance as a fallback.
    lonely_psu : str, optional
        Override the survey design's ``lonely_psu`` setting for within-cell
        computation. One of ``"remove"``, ``"certainty"``, ``"adjust"``.
    second_stage_weights : str, default "pweight"
        Weight type for the returned second-stage ``SurveyDesign``:

        - ``"pweight"`` (default): Population weights - the mean of
          per-cell survey weight sums within each geographic unit
          (first ``by`` column), constant across periods. Targets
          population-weighted second-stage estimation. Compatible
          with all survey-capable estimators including those that
          require unit-constant survey columns.
        - ``"aweight"``: Precision weights - inverse variance
          (``1 / V(y_bar)``). Targets precision-weighted second-stage
          estimation via WLS. Compatible with estimators that accept ``aweight``
          (DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD,
          SunAbraham, ContinuousDiD, EfficientDiD); rejected by
          ``pweight``-only estimators.

    Returns
    -------
    panel_df : pd.DataFrame
        Aggregated panel with columns: grouping variables,
        ``{outcome}_mean``, ``{outcome}_se``, ``{outcome}_n``,
        ``{outcome}_precision``, ``{outcome}_weight``,
        ``{covariate}_mean``, ``cell_n``, ``cell_n_eff``,
        ``cell_sum_w``, ``srs_fallback``. The ``_weight`` column
        contains unit-constant population weights (mean of
        ``cell_sum_w`` within each geographic unit) in pweight mode,
        or cleaned precision (NaN/Inf mapped to 0.0) in aweight mode.
        ``cell_sum_w`` is always present as a diagnostic column
        containing the sum of normalized survey weights per cell
        (proportional to estimated population).
    second_stage_design : SurveyDesign
        Pre-configured for second-stage estimation with the chosen
        ``weight_type``, weights from the first outcome, and
        geographic clustering via ``psu``.

    Examples
    --------
    >>> design = SurveyDesign(weights="finalwt", strata="strat", psu="psu")
    >>> panel, stage2 = aggregate_survey(
    ...     microdata, by=["state", "year"],
    ...     outcomes="smoking_rate", survey_design=design,
    ... )
    >>> # stage2 has weight_type="pweight" — compatible with all estimators.
    >>> # Add treatment/time indicators at the panel level, then fit:
    >>> # panel["first_treat"] = panel["state"].map(policy_year).fillna(0)
    >>> # result = CallawaySantAnna().fit(
    >>> #     panel, outcome="smoking_rate_mean",
    >>> #     unit="state", time="year", first_treat="first_treat",
    >>> #     survey_design=stage2,
    >>> # )
    """
    import warnings
    from dataclasses import replace

    # --- Normalize inputs ---
    by_cols = [by] if isinstance(by, str) else list(by)
    outcome_cols = [outcomes] if isinstance(outcomes, str) else list(outcomes)
    cov_cols = (
        [covariates] if isinstance(covariates, str) else list(covariates) if covariates else []
    )

    # --- Validate ---
    if not by_cols:
        raise ValueError("'by' must specify at least one grouping column")
    if not outcome_cols:
        raise ValueError("'outcomes' must specify at least one outcome variable")

    all_cols = by_cols + outcome_cols + cov_cols
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    overlap = set(by_cols) & (set(outcome_cols) | set(cov_cols))
    if overlap:
        raise ValueError(f"Columns appear in both 'by' and outcomes/covariates: {overlap}")

    if not isinstance(survey_design, SurveyDesign):
        raise TypeError(
            f"survey_design must be a SurveyDesign instance, got {type(survey_design).__name__}"
        )

    _valid_second_stage = {"pweight", "aweight"}
    if second_stage_weights not in _valid_second_stage:
        raise ValueError(
            f"second_stage_weights must be one of {sorted(_valid_second_stage)}, "
            f"got '{second_stage_weights}'."
        )

    if min_n < 1:
        raise ValueError(f"min_n must be >= 1, got {min_n}")

    if lonely_psu is not None and lonely_psu not in ("remove", "certainty", "adjust"):
        raise ValueError(
            f"lonely_psu must be 'remove', 'certainty', or 'adjust', got '{lonely_psu}'"
        )

    # --- Empty-input guard ---
    if data.empty:
        raise ValueError("data must be non-empty")

    # --- Validate grouping columns have no missing values ---
    by_missing = data[by_cols].isna().any()
    cols_with_na = list(by_missing[by_missing].index)
    if cols_with_na:
        raise ValueError(
            f"Missing values in grouping column(s): {cols_with_na}. "
            f"Drop or fill NaN values before calling aggregate_survey()."
        )

    # --- Resolve design once on full data ---
    effective_design = (
        replace(survey_design, lonely_psu=lonely_psu) if lonely_psu else survey_design
    )
    full_resolved = effective_design.resolve(data)

    # Precompute stratum/PSU scaffolding once per design.  Amortizes
    # per-cell pandas groupby + np.unique + stratum FPC lookup that
    # otherwise dominate runtime at scale (see _compute_if_variance_fast).
    # Replicate-weight designs use a different variance surface and stay
    # on the legacy path.
    _tsl_scaffolding: Optional[_PsuScaffolding] = (
        _precompute_psu_scaffolding(full_resolved)
        if not full_resolved.uses_replicate_variance
        else None
    )

    # --- Precompute full-length outcome/covariate arrays ---
    n_total = len(data)
    all_vars = outcome_cols + cov_cols
    non_numeric = [v for v in all_vars if not pd.api.types.is_numeric_dtype(data[v])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric column(s) in outcomes/covariates: {non_numeric}. "
            f"All outcome and covariate columns must be numeric."
        )
    y_arrays: Dict[str, np.ndarray] = {var: data[var].values.astype(np.float64) for var in all_vars}

    # --- Per-cell computation ---
    # Use groupby().indices for position-based cell membership (safe with
    # duplicate DataFrame indices, no column injection into user data)
    grouped = data.groupby(by_cols, sort=True)
    cell_indices = grouped.indices  # dict of cell_key → positional indices
    rows: List[Dict[str, Any]] = []
    srs_cells: List[str] = []
    zero_var_cells: List[str] = []

    for cell_key, pos_idx in cell_indices.items():
        # Boolean mask for full-design domain estimation
        cell_mask = np.zeros(n_total, dtype=bool)
        cell_mask[pos_idx] = True

        cell_n = int(np.sum(cell_mask))
        cell_key_str = str(cell_key)

        # Cell-level statistics (Kish ESS is a property of the cell)
        cell_w = full_resolved.weights[cell_mask]
        sum_w = float(np.sum(cell_w))
        sum_w2 = float(np.sum(cell_w**2))
        cell_n_eff = (sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0

        # Build row dict with grouping columns
        row: Dict[str, Any] = {}
        if len(by_cols) == 1:
            row[by_cols[0]] = cell_key
        else:
            for i, col in enumerate(by_cols):
                row[col] = cell_key[i]

        row["cell_n"] = cell_n
        row["cell_n_eff"] = cell_n_eff
        row["cell_sum_w"] = sum_w

        cell_srs_fallback = False

        # Outcomes: mean + SE + n + precision (full-design domain estimation)
        for var in outcome_cols:
            y_bar, variance, n_valid, used_srs = _cell_mean_variance(
                y_arrays[var],
                full_resolved,
                cell_mask,
                min_n,
                scaffolding=_tsl_scaffolding,
            )
            se = float(np.sqrt(variance)) if not np.isnan(variance) else np.nan

            if used_srs:
                cell_srs_fallback = True

            # Zero variance → precision NaN
            if se == 0.0:
                precision = np.nan
                zero_var_cells.append(cell_key_str)
            elif np.isnan(se):
                precision = np.nan
            else:
                precision = 1.0 / variance

            row[f"{var}_mean"] = y_bar
            row[f"{var}_se"] = se
            row[f"{var}_n"] = n_valid
            row[f"{var}_precision"] = precision

        # Covariates: design-weighted mean only
        for var in cov_cols:
            y_cell = y_arrays[var][cell_mask]
            valid = ~np.isnan(y_cell)
            w_valid = cell_w * valid.astype(np.float64)
            sw = float(np.sum(w_valid))
            if sw > 0:
                row[f"{var}_mean"] = float(np.sum(w_valid * np.where(valid, y_cell, 0.0)) / sw)
            else:
                row[f"{var}_mean"] = np.nan

        row["srs_fallback"] = cell_srs_fallback
        if cell_srs_fallback:
            srs_cells.append(cell_key_str)

        rows.append(row)

    # --- Warnings ---
    if srs_cells:
        warnings.warn(
            f"Design-based variance not estimable for {len(srs_cells)} cell(s); "
            f"using SRS fallback: {srs_cells[:5]}"
            + (f" ... and {len(srs_cells) - 5} more" if len(srs_cells) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )
    if zero_var_cells:
        warnings.warn(
            f"Zero variance in {len(zero_var_cells)} cell(s) (precision set to NaN): "
            f"{zero_var_cells[:5]}"
            + (f" ... and {len(zero_var_cells) - 5} more" if len(zero_var_cells) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )

    # --- Assemble output ---
    panel_df = pd.DataFrame(rows)

    # Sort by grouping columns
    panel_df = panel_df.sort_values(by_cols).reset_index(drop=True)

    # --- Drop non-estimable cells ---
    # Cells with non-finite mean (n_valid==0 or all-missing) cannot contribute
    # to second-stage estimation and would cause fit() to reject NaN outcomes.
    # Dropping them also removes all-zero-weight PSUs from the panel.
    first_outcome = outcome_cols[0]
    mean_col = f"{first_outcome}_mean"
    nonestimable = ~np.isfinite(panel_df[mean_col].values)
    if np.any(nonestimable):
        n_dropped = int(np.sum(nonestimable))
        dropped_keys = panel_df.loc[nonestimable, by_cols].values.tolist()
        # Warn about secondary outcomes losing valid data in dropped cells
        secondary_loss = []
        for var in outcome_cols[1:]:
            valid_secondary = np.isfinite(panel_df.loc[nonestimable, f"{var}_mean"].values)
            if np.any(valid_secondary):
                secondary_loss.append(var)
        msg = (
            f"Dropped {n_dropped} non-estimable cell(s) (based on first outcome "
            f"'{first_outcome}'): {dropped_keys[:5]}"
            + (f" ... and {n_dropped - 5} more" if n_dropped > 5 else "")
        )
        if secondary_loss:
            msg += (
                f". Note: {secondary_loss} had valid data in dropped cells. "
                f"For independent per-outcome support, call once per outcome."
            )
        warnings.warn(msg, UserWarning, stacklevel=2)
        panel_df = panel_df[~nonestimable].reset_index(drop=True)

    # --- Construct second-stage SurveyDesign ---
    geo_col = by_cols[0]
    weight_col = f"{first_outcome}_weight"
    if second_stage_weights == "pweight":
        # Unit-level population weight: average cell_sum_w across periods
        # within each geographic unit.  This produces a unit-constant
        # weight that satisfies _validate_unit_constant_survey() for
        # panel estimators, while representing each unit's average
        # population share (averaging out period-to-period sampling
        # variability in per-cell weight sums).
        panel_df[weight_col] = panel_df.groupby(geo_col)["cell_sum_w"].transform("mean")
    else:
        # Precision weight: inverse variance, with NaN/Inf -> 0.0 so
        # downstream resolve() doesn't reject missing weights.
        # Diagnostic *_precision column is kept unchanged.
        panel_df[weight_col] = np.where(
            np.isfinite(panel_df[f"{first_outcome}_precision"]),
            panel_df[f"{first_outcome}_precision"],
            0.0,
        )

    # Drop geographic units (PSUs) with zero total weight — they would
    # inflate survey df and distort second-stage variance estimation.
    # Under pweight mode, unit-averaged cell_sum_w > 0 for all surviving
    # cells, so this block is a defensive no-op.  Under aweight, NaN
    # precision maps to 0.0 and geographic units with all-zero precision
    # are pruned here.
    geo_weight = panel_df.groupby(geo_col)[weight_col].sum()
    zero_geos = geo_weight[geo_weight == 0].index
    if len(zero_geos) > 0:
        n_before = len(panel_df)
        panel_df = panel_df[~panel_df[geo_col].isin(zero_geos)].reset_index(drop=True)
        n_after = len(panel_df)
        warnings.warn(
            f"Dropped {n_before - n_after} cell(s) from {len(zero_geos)} "
            f"geographic unit(s) with zero total weight: "
            f"{list(zero_geos[:5])}"
            + (f" ... and {len(zero_geos) - 5} more" if len(zero_geos) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )

    # Guard: all cells dropped
    if panel_df.empty:
        raise ValueError(
            "No estimable cells remain after aggregation. "
            "All cells had missing outcomes or zero effective weight."
        )

    second_stage_design = SurveyDesign(
        weights=weight_col,
        weight_type=second_stage_weights,
        psu=geo_col,
    )

    return panel_df, second_stage_design
