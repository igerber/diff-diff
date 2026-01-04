"""
Data preparation utilities for difference-in-differences analysis.

This module provides helper functions to prepare data for DiD estimation,
including creating treatment indicators, reshaping panel data, and
generating synthetic datasets for testing.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from diff_diff.utils import compute_synthetic_weights

# Constants for rank_control_units
_SIMILARITY_THRESHOLD_SD = 0.5  # Controls within this many SDs are "similar"
_OUTLIER_PENALTY_WEIGHT = 0.3  # Penalty weight for outcome outliers in treatment candidate scoring


def make_treatment_indicator(
    data: pd.DataFrame,
    column: str,
    treated_values: Optional[Union[Any, List[Any]]] = None,
    threshold: Optional[float] = None,
    above_threshold: bool = True,
    new_column: str = "treated"
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
    new_column: str = "post"
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
    time_values: Optional[List[Any]] = None
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
        var_name='_temp_var',
        value_name=value_name
    )

    # Map column names to time values
    col_to_time = dict(zip(value_columns, time_values))
    long_df[time_name] = long_df['_temp_var'].map(col_to_time)
    long_df = long_df.drop('_temp_var', axis=1)

    # Reorder columns and sort
    cols = [id_column, time_name, value_name] + other_cols
    return long_df[cols].sort_values([id_column, time_name]).reset_index(drop=True)


def balance_panel(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    method: str = "inner",
    fill_value: Optional[float] = None
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
            [all_units, all_periods],
            names=[unit_column, time_column]
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
    raise_on_error: bool = True
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
            f"Outcome column '{outcome}' must be numeric. "
            f"Got type: {data[outcome].dtype}"
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

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary
    }


def summarize_did_data(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: Optional[str] = None
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
    summary = data.groupby([treatment, time])[outcome].agg([
        ("n", "count"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max")
    ]).round(4)

    # Calculate time values for labeling
    time_vals = sorted(data[time].unique())

    # Add group labels based on sorted time values (not literal 0/1)
    if len(time_vals) == 2:
        pre_val, post_val = time_vals[0], time_vals[1]

        def format_label(x):
            treatment_label = 'Treated' if x[0] == 1 else 'Control'
            time_label = 'Post' if x[1] == post_val else 'Pre'
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
            {
                "n": ["-"],
                "mean": [did_estimate],
                "std": ["-"],
                "min": ["-"],
                "max": ["-"]
            },
            index=["DiD Estimate"]
        )
        summary = pd.concat([summary, did_row])
    else:
        summary.index = summary.index.map(
            lambda x: f"{'Treated' if x[0] == 1 else 'Control'} - Period {x[1]}"
        )

    return summary


def generate_did_data(
    n_units: int = 100,
    n_periods: int = 4,
    treatment_effect: float = 5.0,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    unit_fe_sd: float = 2.0,
    time_trend: float = 0.5,
    noise_sd: float = 1.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic data for DiD analysis with known treatment effect.

    Creates a balanced panel dataset with realistic features including
    unit fixed effects, time trends, and a known treatment effect.

    Parameters
    ----------
    n_units : int, default=100
        Number of units in the panel.
    n_periods : int, default=4
        Number of time periods.
    treatment_effect : float, default=5.0
        True average treatment effect on the treated.
    treatment_fraction : float, default=0.5
        Fraction of units that receive treatment.
    treatment_period : int, default=2
        First post-treatment period (0-indexed). Periods >= this are post.
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    time_trend : float, default=0.5
        Linear time trend coefficient.
    noise_sd : float, default=1.0
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic panel data with columns:
        - unit: Unit identifier
        - period: Time period
        - treated: Treatment indicator (0/1)
        - post: Post-treatment indicator (0/1)
        - outcome: Outcome variable
        - true_effect: The true treatment effect (for validation)

    Examples
    --------
    Generate simple data for testing:

    >>> data = generate_did_data(n_units=50, n_periods=4, treatment_effect=3.0, seed=42)
    >>> len(data)
    200
    >>> data.columns.tolist()
    ['unit', 'period', 'treated', 'post', 'outcome', 'true_effect']

    Verify treatment effect recovery:

    >>> from diff_diff import DifferenceInDifferences
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, outcome='outcome', treatment='treated', time='post')
    >>> abs(results.att - 3.0) < 1.0  # Close to true effect
    True
    """
    rng = np.random.default_rng(seed)

    # Determine treated units
    n_treated = int(n_units * treatment_fraction)
    treated_units = set(range(n_treated))

    # Generate unit fixed effects
    unit_fe = rng.normal(0, unit_fe_sd, n_units)

    # Build data
    records = []
    for unit in range(n_units):
        is_treated = unit in treated_units

        for period in range(n_periods):
            is_post = period >= treatment_period

            # Base outcome
            y = 10.0  # Baseline
            y += unit_fe[unit]  # Unit fixed effect
            y += time_trend * period  # Time trend

            # Treatment effect (only for treated units in post-period)
            effect = 0.0
            if is_treated and is_post:
                effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(is_post),
                "outcome": y,
                "true_effect": effect
            })

    return pd.DataFrame(records)


def create_event_time(
    data: pd.DataFrame,
    time_column: str,
    treatment_time_column: str,
    new_column: str = "event_time"
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
    covariates: Optional[List[str]] = None
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
    cohort_data = cohort_data.rename(columns={
        unit_column: "n_units",
        outcome: f"mean_{outcome}"
    })

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
        - synthetic_weight: Weight from synthetic control optimization
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
            data, unit_column, time_column, outcome_column,
            pre_periods, n_treatment_candidates
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
    # Synthetic weights (higher = better match)
    synthetic_weights = compute_synthetic_weights(
        Y_control, Y_treated_mean, lambda_reg=lambda_reg
    )

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

    result = pd.DataFrame({
        'unit': control_candidates,
        'quality_score': quality_scores,
        'outcome_trend_score': outcome_trend_scores,
        'covariate_score': covariate_scores,
        'synthetic_weight': synthetic_weights,
        'pre_trend_rmse': rmse_scores,
        'is_required': [u in require_set for u in control_candidates]
    })

    # Sort by quality score (descending)
    result = result.sort_values('quality_score', ascending=False)

    # Apply n_top limit if specified
    if n_top is not None and n_top < len(result):
        # Always include required units
        required_df = result[result['is_required']]
        non_required_df = result[~result['is_required']]

        # Take top from non-required to fill remaining slots
        remaining_slots = max(0, n_top - len(required_df))
        top_non_required = non_required_df.head(remaining_slots)

        result = pd.concat([required_df, top_non_required])
        result = result.sort_values('quality_score', ascending=False)

    return result.reset_index(drop=True)


def _suggest_treatment_candidates(
    data: pd.DataFrame,
    unit_column: str,
    time_column: str,
    outcome_column: str,
    pre_periods: List[Any],
    n_candidates: int
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
        other_means = pre_data[
            pre_data[unit_column].isin(other_units)
        ].groupby(unit_column)[outcome_column].mean()

        if len(other_means) > 0:
            sd = other_means.std()
            if sd > 0:
                n_similar = int(np.sum(
                    np.abs(other_means - avg_outcome) < _SIMILARITY_THRESHOLD_SD * sd
                ))
            else:
                n_similar = len(other_means)
        else:
            n_similar = 0

        candidate_info.append({
            'unit': unit,
            'avg_outcome_level': avg_outcome,
            'outcome_trend': slope,
            'n_similar_controls': n_similar
        })

    if len(candidate_info) == 0:
        return pd.DataFrame(columns=[
            'unit', 'treatment_candidate_score', 'avg_outcome_level',
            'outcome_trend', 'n_similar_controls'
        ])

    result = pd.DataFrame(candidate_info)

    # Score: prefer units with many similar controls and moderate outcome levels
    max_similar = result['n_similar_controls'].max()
    if max_similar > 0:
        similarity_score = result['n_similar_controls'] / max_similar
    else:
        similarity_score = pd.Series([0.0] * len(result))

    # Penalty for outliers in outcome level
    outcome_mean = result['avg_outcome_level'].mean()
    outcome_std = result['avg_outcome_level'].std()
    if outcome_std > 0:
        outcome_z = np.abs((result['avg_outcome_level'] - outcome_mean) / outcome_std)
    else:
        outcome_z = pd.Series([0.0] * len(result))

    result['treatment_candidate_score'] = (
        similarity_score - _OUTLIER_PENALTY_WEIGHT * outcome_z
    ).clip(0, 1)

    # Return top candidates
    result = result.nlargest(n_candidates, 'treatment_candidate_score')
    return result.reset_index(drop=True)
