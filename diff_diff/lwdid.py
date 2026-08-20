"""LWDiD: Lee & Wooldridge (2025, 2026) rolling-transformation DiD.

Converts panel DiD into cross-sectional estimation via unit-specific
rolling transformations of the outcome variable. Supports common timing
and staggered adoption designs with regression adjustment ('reg'),
inverse probability weighting ('ipw'), doubly robust ('dr'), and
propensity score matching ('psm') estimation.

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg

from diff_diff._base import BaseEstimator
from diff_diff.linalg import _detect_rank_deficiency, solve_logit, solve_ols
from diff_diff.lwdid_results import LWDiDResults
from diff_diff.utils import safe_inference, validate_binary

_VALID_ROLLING = ("demean", "detrend", "demeanq", "detrendq")
_VALID_ESTIMATION_METHODS = ("reg", "ipw", "dr", "psm")
_VALID_VCOV_TYPES = ("classical", "hc1", "hc2", "hc3")
_VALID_CONTROL_GROUPS = ("never_treated", "not_yet_treated")

# Propensity score trimming bounds for numerical stability
#: Column names written into internal estimation/plotting frames. A user
#: role column with one of these names would be silently overwritten
#: (e.g. cluster='_treat' turned the cluster labels into the treatment
#: regressor), so they are rejected up front by _validate_inputs.
_RESERVED_INTERNAL_COLUMNS = frozenset(
    {
        "_treat",
        "_ydot",
        "_ydot_avg",
        "_ever_treated",
        "_boot_unit",
        "_lwdid_time_pos",
        "_lwdid_cohort_pos",
        "_lwdid_season",
    }
)


def _normalize_cohorts(
    cohort_series: pd.Series,
    *,
    max_time: Any,
) -> Tuple[pd.Series, int, int]:
    """Canonicalize NUMERIC-encoded cohort values to the house convention.

    Never-treated is encoded as ``NaN`` or ``0`` downstream. Two additional
    encodings are recoded here with a warning (no silent reinterpretation):

    - ``np.inf`` -> ``0`` (CallawaySantAnna convention: CS recodes exactly
      ``0``/``inf``; the NaN limb accepted downstream is an LWDiD-only
      extension needed for datetime scales, documented in REGISTRY).
    - finite ``g > max_time`` (beyond-window) -> ``0``: a unit that never
      switches on inside the observed window is within-sample
      never-treated. This is a documented DEVIATION from CS (which keeps
      finite cohorts out of never-treated); under
      ``control_group='never_treated'`` recoded units join the control
      pool and contribute no pre-period event cells.

    Negative finite values and ``-inf`` are nonsensical cohort encodings
    and raise ``ValueError`` (never silently classified).

    Parameters
    ----------
    cohort_series : pd.Series
        Row-level cohort column, already numeric (datetime/Period panels
        must be encoded to integer positions first — see
        ``_encode_staggered_time_scale``).
    max_time : scalar
        Largest observed time value on the same numeric scale.

    Returns
    -------
    tuple
        ``(normalized_series, n_inf_rows_recoded, n_beyond_rows_recoded)``.
    """
    values = pd.to_numeric(cohort_series, errors="raise")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    negative = finite & (values.to_numpy(dtype=float, na_value=np.nan) < 0)
    neg_inf = np.isneginf(values.to_numpy(dtype=float, na_value=np.nan))
    if negative.any() or neg_inf.any():
        bad = sorted(pd.unique(values[negative | neg_inf]).tolist())
        raise ValueError(
            f"Cohort column contains negative value(s) {bad[:5]}: cohort "
            f"values must be 0/NaN (never-treated), np.inf (recoded to "
            f"never-treated), or an observed treatment period."
        )
    inf_mask = np.isposinf(values.to_numpy(dtype=float, na_value=np.nan))
    n_inf = int(inf_mask.sum())
    if n_inf:
        warnings.warn(
            f"first_treat=inf found on {n_inf} row(s); recoding to 0 "
            f"(never-treated). Use first_treat=0 to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
    beyond_mask = finite & (values.to_numpy(dtype=float, na_value=np.nan) > 0)
    beyond_mask &= values.to_numpy(dtype=float, na_value=np.nan) > float(max_time)
    n_beyond = int(beyond_mask.sum())
    if n_beyond:
        bad_vals = sorted(pd.unique(values[beyond_mask]).tolist())
        warnings.warn(
            f"Cohort value(s) {bad_vals[:5]} exceed the last observed period "
            f"({max_time}); units in these cohorts never switch on within "
            f"the sample and are recoded to never-treated (0). This deviates "
            f"from CallawaySantAnna, which keeps finite cohorts out of "
            f"never-treated; see docs/methodology/REGISTRY.md (LWDiD).",
            UserWarning,
            stacklevel=3,
        )
    if n_inf or n_beyond:
        values = values.copy()
        values[inf_mask | beyond_mask] = 0
    return values, n_inf, n_beyond


def _check_treatment_design(
    df: pd.DataFrame,
    unit: str,
    time: str,
    treatment: str,
    first_treat: Optional[str] = None,
) -> None:
    """Validate the treatment design in a single vectorized pass.

    One sort + groupby covers three checks:

    1. Absorbing treatment: within each unit the sequence D_it must be
       non-decreasing over time (once treated, always treated).
    2. Common timing (``first_treat is None``): every treated unit must
       first switch to D_it = 1 in the same period; heterogeneous onsets
       require the staggered interface (``first_treat`` cohort column).
    3. Staggered (``first_treat`` given): over each unit's OBSERVED rows,
       the treatment indicator must equal ``1[t >= g_i]`` exactly — no
       D_it = 1 before the cohort value and no D_it = 0 at or after it;
       units with cohort NaN/0 (never treated) must have no D_it = 1
       rows. The row at ``t == g_i`` itself may be unobserved (unbalanced
       panels with a missing onset row are accepted). Finite positive
       cohort values must be members of the observed time support —
       numeric between-period cohorts are rejected (datetime/Period
       cohorts are mapped to the next observed period by the encoding
       step before this check).

    Precondition: on the staggered path the cohort column is expected to
    be already normalized+encoded (``_encode_staggered_time_scale`` +
    ``_normalize_cohorts``) so that never-treated is NaN/0 and
    beyond-window/inf sentinels no longer occur.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    unit, time, treatment : str
        Column names of the unit identifier, time period, and binary
        treatment indicator.
    first_treat : str or None, default None
        Cohort (first-treatment-time) column for staggered designs.

    Raises
    ------
    ValueError
        If any applicable design check fails.
    """
    cols = [unit, time, treatment]
    if first_treat is not None:
        cols.append(first_treat)
    ordered = df[cols].sort_values([unit, time], kind="stable")

    # (1) Absorbing treatment: within-unit first difference must never
    # be negative (a 1 -> 0 switch).
    diffs = ordered.groupby(unit, sort=False)[treatment].diff()
    non_absorbing = (diffs < 0).to_numpy()
    if non_absorbing.any():
        bad_units = pd.unique(ordered.loc[non_absorbing, unit])
        preview = ", ".join(repr(u) for u in bad_units[:5])
        suffix = "" if len(bad_units) <= 5 else f", ... ({len(bad_units)} units total)"
        raise ValueError(
            f"Non-absorbing treatment detected for unit(s) {preview}{suffix}: "
            f"treatment switches from 1 to 0. LWDiD requires absorbing treatment."
        )

    # First observed treatment period per treated unit (rows are already
    # time-sorted within unit, so first() is the onset).
    treated_rows = ordered.loc[ordered[treatment] == 1]
    onset = treated_rows.groupby(unit, sort=False)[time].first()

    if first_treat is None:
        # (2) Common timing: every ever-treated unit's observed rows must
        # satisfy D_it = 1[t >= S] for the single global onset
        # S = min(first observed treated period). Comparing first OBSERVED
        # treated rows directly would falsely reject a unit whose t = S
        # row is simply missing (round-8 review; the staggered branch
        # already permits an unobserved onset row). A genuinely
        # heterogeneous unit (true onset S' > S) has an observed
        # UNTREATED row at t >= S and is rejected.
        if len(onset) == 0:
            return
        onset_s = onset.min()
        ever_treated = set(onset.index)
        ever_row = ordered[unit].isin(ever_treated).to_numpy()
        late_zero = (
            ever_row
            & (ordered[time] >= onset_s).to_numpy()
            & (ordered[treatment].to_numpy(dtype=float) == 0)
        )
        if late_zero.any():
            bad_units = sorted(pd.unique(ordered.loc[late_zero, unit]).tolist())
            preview = ", ".join(repr(u) for u in bad_units[:5])
            suffix = "" if len(bad_units) <= 5 else f", ... ({len(bad_units)} units total)"
            raise ValueError(
                f"Treated unit(s) {preview}{suffix} have untreated observed "
                f"rows at or after the common onset {onset_s!r}: treated "
                f"units have heterogeneous first-treatment periods, but no "
                f"cohort column was given. Common-timing LWDiD requires a single "
                f"treatment onset; pass first_treat= to use the staggered "
                f"(cohort) interface."
            )
        return

    # (3a) Support membership: finite positive cohorts must be observed
    # time values. Post-normalization, beyond-window sentinels no longer
    # occur, so this targets exactly numeric BETWEEN-period cohorts
    # (e.g. g=4.5 with observed times {4, 5}).
    cohort_by_unit = ordered.groupby(unit, sort=False)[first_treat].first()
    observed_times = pd.Index(pd.unique(ordered[time]))
    positive = cohort_by_unit.notna() & (cohort_by_unit > 0)
    off_support = positive & ~cohort_by_unit.isin(observed_times)
    if off_support.to_numpy().any():
        bad_vals = sorted(pd.unique(cohort_by_unit[off_support]).tolist())
        raise ValueError(
            f"Cohort value(s) {bad_vals[:5]} in column '{first_treat}' are "
            f"not observed time periods. Numeric between-period cohorts are "
            f"not supported; use an observed period value (datetime/Period "
            f"cohorts are mapped to the next observed period automatically)."
        )

    # (3b) Per unit, over OBSERVED rows, D_it must equal 1[t >= g_i]:
    # never-treated units (cohort NaN/0) have no D=1 rows; treated-cohort
    # units have no D=1 before g and no D=0 at/after g. The onset row
    # itself may be unobserved (unbalanced panels are accepted).
    g_by_row = ordered[unit].map(cohort_by_unit)
    g_arr = g_by_row.to_numpy(dtype=float, na_value=np.nan)
    never_row = np.isnan(g_arr) | (g_arr == 0)
    d_arr = ordered[treatment].to_numpy(dtype=float)
    t_arr = ordered[time].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        premature = ~never_row & (t_arr < g_arr) & (d_arr == 1)
        untreated_post = ~never_row & (t_arr >= g_arr) & (d_arr == 0)
    never_treated_rows = never_row & (d_arr == 1)
    violation = premature | untreated_post | never_treated_rows
    if violation.any():
        bad_units = pd.unique(ordered.loc[violation, unit])
        preview = ", ".join(repr(u) for u in bad_units[:5])
        suffix = "" if len(bad_units) <= 5 else f", ... ({len(bad_units)} units total)"
        raise ValueError(
            f"Treatment column '{treatment}' is inconsistent with cohort "
            f"column '{first_treat}' for unit(s) {preview}{suffix}: over "
            f"observed rows, treatment must equal 1[t >= cohort] — no "
            f"treatment=1 before the cohort period, no treatment=0 at or "
            f"after it, and never-treated units (cohort NaN or 0) must have "
            f"no treatment=1 rows."
        )


def _encode_staggered_time_scale(
    df: pd.DataFrame,
    time: str,
    first_treat: str,
) -> Tuple[pd.DataFrame, str, str, Optional[Dict[str, Dict[int, Any]]]]:
    """Re-encode datetime/Period time scales as integer positions.

    The staggered machinery relies on integer time semantics: cohort
    eligibility comparisons (``g > 0``, ``g > t``), event times ``t - g``,
    and the never-treated sentinel 0. Datetime and Period panels are
    therefore mapped onto the ordered support of observed time values --
    the k-th observed period becomes position k (1-based, so 0 stays free
    for the never-treated sentinel, coded NaT in datetime panels). Cohort
    values between observed periods map to the next observed position and
    cohorts beyond the window map to T + 1, preserving the onset
    consistency checks. Numeric panels are returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (a private copy owned by the caller; encoded position
        columns are added in place).
    time, first_treat : str
        Time and cohort column names.

    Returns
    -------
    tuple
        ``(df, time_column, cohort_column, label_maps)`` where
        ``label_maps`` is None for numeric panels and otherwise maps
        integer positions back to the original time/cohort labels.

    Raises
    ------
    ValueError
        If the two columns do not share the same time family: exactly one
        of them is date-like, one is datetime64 while the other is Period
        (either direction), or both are Period with different frequencies.
    """
    time_is_datetime = pd.api.types.is_datetime64_any_dtype(df[time])
    cohort_is_datetime = pd.api.types.is_datetime64_any_dtype(df[first_treat])
    time_is_period = isinstance(df[time].dtype, pd.PeriodDtype)
    cohort_is_period = isinstance(df[first_treat].dtype, pd.PeriodDtype)
    if not (time_is_datetime or time_is_period or cohort_is_datetime or cohort_is_period):
        return df, time, first_treat, None
    # datetime64 and Period are distinct time families: position lookups
    # (searchsorted, dict membership) crash inside pandas when mixed, so
    # both directions are rejected up front with the documented ValueError.
    if time_is_datetime != cohort_is_datetime or time_is_period != cohort_is_period:
        raise ValueError(
            f"Columns '{time}' (time) and '{first_treat}' (first_treat) must "
            f"share the same time scale; got dtypes {df[time].dtype} and "
            f"{df[first_treat].dtype}. Encode both as datetime64, both as "
            f"Period with the same frequency, or both as numeric."
        )
    if time_is_period and df[time].dtype.freq != df[first_treat].dtype.freq:
        raise ValueError(
            f"Columns '{time}' (time) and '{first_treat}' (first_treat) are "
            f"Period columns with different frequencies ({df[time].dtype} vs "
            f"{df[first_treat].dtype}). Convert them to a common frequency "
            f"before fitting."
        )

    support = pd.Index(pd.unique(df[time])).sort_values()
    time_pos: Dict[Any, int] = {value: index + 1 for index, value in enumerate(support)}

    def cohort_pos(value: Any) -> int:
        if value in time_pos:
            return time_pos[value]
        # Between observed periods -> next observed position; beyond the
        # window -> T + 1 (vacuously consistent, like numeric cohorts
        # past the last observed period).
        return int(support.searchsorted(value, side="left")) + 1

    cohort_map: Dict[Any, int] = {
        value: cohort_pos(value)
        for value in pd.Index(pd.unique(df[first_treat]))
        if pd.notna(value)
    }
    df["_lwdid_time_pos"] = df[time].map(time_pos).astype(int)
    df["_lwdid_cohort_pos"] = df[first_treat].map(cohort_map).fillna(0).astype(int)
    # Preserve the CALENDAR season before time values become dense
    # positions: the seasonal transforms' numeric fallback derives quarter
    # as (t - 1) % 4 + 1, which relabels every season after a globally
    # missing calendar period (review round 3 P0 - silent seasonal
    # mixing). The q-variant transforms prefer this column when present.
    if time_is_datetime:
        df["_lwdid_season"] = df[time].dt.quarter.to_numpy()
    else:
        df["_lwdid_season"] = np.array([value.quarter for value in df[time]])
    time_reverse = {position: value for value, position in time_pos.items()}
    # Cohort positions relabel to the CANONICAL observed period at that
    # position (round-24 review: reversing cohort_map let two raw
    # between-period labels mapping to the same onset collide, with the
    # surviving label depending on input row order); off-support
    # positions (beyond-window T+1) keep a deterministic raw label and
    # are normalized to never-treated downstream anyway.
    label_maps = {
        "time": time_reverse,
        "cohort": {
            position: time_reverse.get(position, value)
            for value, position in sorted(cohort_map.items(), key=lambda kv: str(kv[0]))
        },
    }
    return df, "_lwdid_time_pos", "_lwdid_cohort_pos", label_maps


def _relabel_staggered_results(
    results: LWDiDResults,
    label_maps: Dict[str, Dict[int, Any]],
) -> LWDiDResults:
    """Map integer time positions in staggered results back to original labels.

    Cohort and calendar-time keys (and the nested ``'cohort'``/``'time'``
    entries) are restored to the user's datetime/Period labels. Relative
    event times remain integers: they are position differences on the
    ordered time support.
    """
    time_labels = label_maps["time"]
    cohort_labels = label_maps["cohort"]
    if results.cohort_effects is not None:
        relabeled_cohorts: Dict[Any, Dict[str, Any]] = {}
        for g, info in results.cohort_effects.items():
            info["cohort"] = cohort_labels.get(info["cohort"], info["cohort"])
            relabeled_cohorts[cohort_labels.get(g, g)] = info
        results.cohort_effects = relabeled_cohorts
    if results.cohort_time_effects is not None:
        relabeled_cells: Dict[Any, Dict[str, Any]] = {}
        for (g, t), info in results.cohort_time_effects.items():
            info["cohort"] = cohort_labels.get(info["cohort"], info["cohort"])
            info["time"] = time_labels.get(info["time"], info["time"])
            relabeled_cells[(cohort_labels.get(g, g), time_labels.get(t, t))] = info
        results.cohort_time_effects = relabeled_cells
    return results


class LWDiD(BaseEstimator):
    """Lee & Wooldridge rolling-transformation DiD estimator.

    Parameters
    ----------
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}, default 'demean'
        Unit-specific transformation method.
        'demean': subtract pre-treatment mean
        'detrend': subtract pre-treatment linear trend
        'demeanq': subtract unit-specific seasonal (quarterly) means
        'detrendq': subtract unit-specific linear trend + seasonal effects
    estimation_method : {'reg', 'ipw', 'dr', 'psm'}, default 'reg'
        Treatment effect estimation method.
        'reg': regression adjustment (OLS)
        'ipw': inverse probability weighting
        'dr': doubly robust (augmented IPW)
        'psm': propensity score matching (1:n_neighbors nearest-neighbor,
        1:1 by default);
        POINT ESTIMATES ONLY - inference is NaN pending an
        Abadie-Imbens matching variance (see DEFERRED.md)
    vcov_type : {'classical', 'hc1', 'hc2', 'hc3'}, default 'hc1'
        Variance-covariance estimator.
        'hc2': leverage-corrected (u_i^2 / (1-h_ii))
        'hc3': jackknife-style leverage correction (u_i^2 / (1-h_ii)^2)
        The full set applies to ``estimation_method='reg'`` only:
        'ipw'/'dr'/'psm' accept 'hc1' alone (the influence-function
        variance on ipw/dr; psm reports NaN inference - see below).
        Cluster-robust (CR1) inference activates via the ``cluster=``
        parameter, not through a ``vcov_type`` value, composes only with
        'hc1', and is rejected for 'psm'.
    cluster : str or None, default None
        Column name for cluster-robust (CR1) standard errors. When set,
        clustered inference is active for the whole fit.
    control_group : {'never_treated', 'not_yet_treated'}, default 'not_yet_treated'
        Control group definition for staggered designs. Both options
        require never-treated units: 'never_treated' needs at least two,
        and 'not_yet_treated' needs at least one so that every cohort-time
        cell keeps a valid control pool. Panels where all units are
        eventually treated are rejected with a ValueError.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default 0
        Number of bootstrap replications (0 = analytical inference).
    seed : int or None, default None
        Random seed for bootstrap inference.
    pscore_trim : float, default 0.01
        Propensity score trimming threshold. Scores below this value
        or above (1 - pscore_trim) are clipped. Used by IPW/DR/PSM.
    n_neighbors : int, default 1
        Number of nearest neighbors for PSM matching.
    caliper : float or None, default None
        Maximum allowable distance for PSM matches. Unmatched treated
        units (no control within caliper) receive NaN.
    with_replacement : bool, default True
        Whether PSM matching is done with replacement.
    n_jobs : int, default 1
        Execution parallelism for the common-timing bootstrap
        (ThreadPoolExecutor when > 1; experimental). Purely an execution
        setting: seeded bootstrap draws are identical for every value
        (per-replicate SeedSequence streams), so it never affects any
        reported number and is not stored in result provenance.

    Notes
    -----
    **Parameter mapping from lwdid-py to diff-diff:**

    The standalone ``lwdid-py`` package (``from lwdid import lwdid``) uses a
    functional interface with separate ``d`` (ever-treated indicator) and
    ``post`` (post-period indicator) columns.  In diff-diff, the ``treatment``
    column is the time-varying binary indicator ``D_i * post_t``—i.e., the
    product of the two lwdid-py columns.

    .. code-block:: python

        # lwdid-py (functional API):
        lwdid(data, y='y', d='d', ivar='unit', tvar='time', post='post',
              rolling='demean', estimator='ra', vce=None)

        # Equivalent in diff-diff (class-based API):
        LWDiD(rolling='demean', estimation_method='reg',
              vcov_type='classical').fit(
            data, outcome='y', unit='unit', time='time', treatment='treat')
        # where data['treat'] == data['d'] * data['post']

    Parameter correspondence:

    =================  ==================  ====================================
    lwdid-py           diff-diff           Notes
    =================  ==================  ====================================
    y                  outcome             Outcome column name
    d + post           treatment           Binary D_it (ever-treated × post)
    ivar               unit                Unit identifier
    tvar               time                Time variable
    gvar               first_treat         Cohort (first treatment period)
    rolling            rolling             Same values
    estimator='ra'     estimation_method   'ra' -> 'reg', 'ipwra' -> 'dr'
    vce=None           vcov_type           Homoskedastic == 'classical'
    vce='hc1'          vcov_type='hc1'     Heteroskedasticity-robust
    vce='cluster'      cluster=<column>    Constructor cluster= parameter
    cluster_var        cluster             Cluster variable name
    controls           covariates          fit() covariates= parameter
    control_group      control_group       Same values
    =================  ==================  ====================================

    **Results mapping:**

    ==================  =========================  ==============================
    lwdid-py            diff-diff                  Notes
    ==================  =========================  ==============================
    result.att          result.att                 ATT point estimate
    result.se_att       result.se                  Standard error
    result.t_stat       result.t_stat              t-statistic
    result.pvalue       result.p_value             p-value (note underscore)
    result.ci_lower     result.conf_int[0]         CI lower bound
    result.ci_upper     result.conf_int[1]         CI upper bound
    result.nobs         result.n_obs               Number of observations
    result.n_treated    result.n_treated           Treated units
    result.n_control    result.n_control           Control units
    result.vce_type     result.vcov_type           Variance family
    result.cluster_var  result.cluster_name        Cluster variable name
    result.n_clusters   result.n_clusters          Number of clusters
    ==================  =========================  ==============================

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from diff_diff.lwdid import LWDiD
    >>> from diff_diff import generate_staggered_data
    >>> data = generate_staggered_data(n_units=100, n_periods=8, seed=0)
    >>> model = LWDiD(rolling='demean', estimation_method='reg')
    >>> result = model.fit(data, outcome='outcome', unit='unit',
    ...                    time='period', treatment='treated',
    ...                    first_treat='first_treat')
    >>> result.att != 0
    True
    """

    def __init__(
        self,
        rolling: str = "demean",
        estimation_method: str = "reg",
        vcov_type: str = "hc1",
        cluster: Optional[str] = None,
        control_group: str = "not_yet_treated",
        alpha: float = 0.05,
        n_bootstrap: int = 0,
        seed: Optional[int] = None,
        # Engineering parameters:
        pscore_trim: float = 0.01,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
        with_replacement: bool = True,
        n_jobs: int = 1,
    ) -> None:
        # Validate rolling
        if rolling not in _VALID_ROLLING:
            raise ValueError(f"rolling must be one of {_VALID_ROLLING}, got '{rolling}'")
        # Validate estimation_method
        if estimation_method not in _VALID_ESTIMATION_METHODS:
            raise ValueError(
                f"estimation_method must be one of {_VALID_ESTIMATION_METHODS}, "
                f"got '{estimation_method}'"
            )
        # Validate vcov_type ('cluster' is retired as a MODE value: clustering
        # activates via the cluster= constructor parameter)
        if vcov_type == "cluster":
            raise ValueError(
                "vcov_type='cluster' is retired; pass the cluster= constructor "
                "parameter (column name) to activate cluster-robust inference."
            )
        if vcov_type not in _VALID_VCOV_TYPES:
            raise ValueError(f"vcov_type must be one of {_VALID_VCOV_TYPES}, got '{vcov_type}'")
        self._validate_vcov_config(vcov_type, estimation_method, cluster)
        # Validate control_group
        if control_group not in _VALID_CONTROL_GROUPS:
            raise ValueError(
                f"control_group must be one of {_VALID_CONTROL_GROUPS}, " f"got '{control_group}'"
            )
        # Validate alpha
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha)
            or not (0 < alpha < 1)
        ):
            # Round-22 review: a one-element array passed the range check
            # and failed later inside inference with a raw TypeError.
            raise ValueError(f"alpha must be a scalar in (0, 1), got {alpha!r}")
        alpha = float(alpha)
        # Validate n_bootstrap (0 = analytical; a bootstrap needs >= 2
        # replicates for a sample standard deviation - review finding:
        # n_bootstrap=1 was accepted and produced NaN downstream)
        if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be a non-negative integer, " f"got {n_bootstrap}")
        if n_bootstrap == 1:
            raise ValueError(
                "n_bootstrap must be 0 (analytical inference) or >= 2 (a "
                "bootstrap standard deviation needs at least 2 replicates)."
            )

        self.rolling = rolling
        self.estimation_method = estimation_method
        self.vcov_type = vcov_type
        self.cluster = cluster
        self.control_group = control_group
        self.alpha = alpha
        self.n_bootstrap = int(n_bootstrap)
        self.seed = seed

        # Engineering parameters (validated, never silently coerced -
        # review finding: fractional n_neighbors truncated, strings became
        # with_replacement=True, negative calipers matched nothing)
        if not isinstance(pscore_trim, (int, float, np.integer, np.floating)) or isinstance(
            pscore_trim, bool
        ):
            raise ValueError(f"pscore_trim must be a number, got {pscore_trim!r}")
        self.pscore_trim = float(pscore_trim)
        if not np.isfinite(self.pscore_trim) or not (0.0 < self.pscore_trim < 0.5):
            raise ValueError("pscore_trim must be between 0 and 0.5")
        if not isinstance(n_neighbors, (int, np.integer)) or isinstance(n_neighbors, bool):
            raise ValueError(f"n_neighbors must be an integer, got {n_neighbors!r}")
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if caliper is not None:
            if not isinstance(caliper, (int, float, np.integer, np.floating)) or isinstance(
                caliper, bool
            ):
                raise ValueError(f"caliper must be a positive number or None, got {caliper!r}")
            caliper = float(caliper)
            if not np.isfinite(caliper) or caliper <= 0:
                raise ValueError(f"caliper must be a positive finite number, got {caliper}")
        self.caliper = caliper
        if not isinstance(with_replacement, (bool, np.bool_)):
            raise ValueError(f"with_replacement must be a boolean, got {with_replacement!r}")
        self.with_replacement = bool(with_replacement)
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer, got {n_jobs!r}")
        self.n_jobs = int(n_jobs)

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        first_treat: Optional[str] = None,
        covariates: Optional[List[str]] = None,
    ) -> LWDiDResults:
        """Fit the LWDiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel dataset in long format.
        outcome : str
            Column name of the outcome variable.
        unit : str
            Column name of the unit identifier.
        time : str
            Column name of the time period variable.
        treatment : str
            Column name of the binary treatment indicator (0/1).
        first_treat : str, optional
            Column name of the first-treatment-time (cohort) variable.
            If None, assumes common timing (all treated units adopt
            treatment simultaneously).
        covariates : list of str, optional
            Column names for control variables (covariates). Every LWDiD
            path requires unit-constant (time-invariant) covariates;
            time-varying columns raise a ValueError. The same
            unit-constancy contract applies to the constructor's
            ``cluster=`` column.

        Returns
        -------
        LWDiDResults
            Object containing ATT estimates, standard errors, and
            inference results.

        Raises
        ------
        ValueError
            If required columns are missing, treatment is not binary,
            or panel structure is invalid.
        """
        # --- Input validation ---
        df = data.copy()
        cluster = self.cluster
        # Re-check the vcov configuration at fit time (set_params probe
        # re-init covers most mutations; this closes direct-attribute edits).
        self._validate_vcov_config(self.vcov_type, self.estimation_method, cluster)
        self._validate_inputs(df, outcome, unit, time, treatment, first_treat, cluster, covariates)
        if cluster is not None:
            from diff_diff.linalg import effective_cluster_count

            n_cl = effective_cluster_count(df[cluster].to_numpy())
            if n_cl < 2:
                raise ValueError(
                    f"cluster='{cluster}' has {n_cl} effective cluster(s); "
                    f"cluster-robust inference requires at least 2."
                )

        # Validate treatment is binary
        validate_binary(df[treatment].values, treatment)

        # Datetime/Period time scales are re-encoded as integer positions
        # before design validation: the staggered checks and estimation
        # compare cohorts against the never-treated sentinel 0 and build
        # event times as t - g, which are undefined for datetime values.
        label_maps = None
        if first_treat is not None:
            df, time, first_treat, label_maps = _encode_staggered_time_scale(df, time, first_treat)
            # Cohort normalization runs AFTER encoding (numeric positions
            # only — datetime beyond-window cohorts arrive as T+1 and are
            # caught by the g > max_time rule) and BEFORE the design check
            # (which requires canonical never-treated encodings).
            df[first_treat], _, _ = _normalize_cohorts(df[first_treat], max_time=df[time].max())

        # Unified treatment-design validation (absorbing + timing
        # consistency) covering both dispatch paths
        _check_treatment_design(df, unit, time, treatment, first_treat)

        # Normalize covariates
        if covariates is None:
            covariates = []

        if self.estimation_method == "psm" and self.n_bootstrap > 0:
            # Review findings (rounds 1-3): the staggered multiplier
            # bootstrap silently did nothing for PSM (no influence-function
            # representation), and the common-timing unit bootstrap
            # replaced the documented fail-closed NaN inference with a
            # naive pairs-bootstrap SE - invalid for nearest-neighbor
            # matching with replacement (Abadie & Imbens 2008, "On the
            # Failure of the Bootstrap for Matching Estimators").
            raise ValueError(
                "estimation_method='psm' does not support n_bootstrap > 0: "
                "matching has no influence-function representation for the "
                "staggered multiplier bootstrap, and the standard bootstrap "
                "is invalid for nearest-neighbor matching estimators "
                "(Abadie & Imbens 2008). Use n_bootstrap=0, or "
                "estimation_method='dr'."
            )
        if self.estimation_method == "psm" and not covariates:
            # Review round 3: without covariates there is no propensity
            # model to match on; the silent delegation to regression
            # adjustment returned a finite OLS SE while the results
            # metadata reported method 'psm' under its documented
            # fail-closed NaN-inference contract.
            raise ValueError(
                "estimation_method='psm' requires covariates: without them "
                "there is no propensity score to match on (PSM would reduce "
                "to a difference in means). Use estimation_method='reg', or "
                "supply covariates."
            )

        # Dispatch to common timing or staggered
        if first_treat is None:
            if isinstance(df[time].dtype, pd.CategoricalDtype) and df[time].dtype.ordered:
                # Ordered categoricals declare the chronology; encode to
                # codes so every comparison/sort respects it (round-20).
                df[time] = df[time].cat.codes.astype(int)
            self._validate_common_time_scale(df, time)
            return self._fit_common_timing(df, outcome, unit, time, treatment, cluster, covariates)
        from diff_diff.lwdid_staggered import fit_staggered

        results = fit_staggered(self, df, outcome, unit, time, first_treat, cluster, covariates)
        if label_maps is not None:
            _relabel_staggered_results(results, label_maps)
        return results

    def _validate_common_time_scale(self, df: pd.DataFrame, time: str) -> None:
        """Common-timing time-scale contract, SHARED by fit() and
        get_transformation_diagnostics() (round-18 review: diagnostics
        bypassed these checks and reached the transforms' raw float
        conversion errors)."""
        if self.rolling in ("detrend", "detrendq") and isinstance(df[time].dtype, pd.PeriodDtype):
            # Period values cannot be cast to float for the unit trend
            # design (review finding: validation accepted PeriodDtype
            # but the transform raised a raw TypeError). datetime64
            # works (nanosecond ordinals).
            raise ValueError(
                f"rolling='{self.rolling}' does not support a Period "
                f"time column on the common-timing path; convert with "
                f".dt.to_timestamp() or encode the time column "
                f"numerically."
            )
        if self.rolling != "demean" and not (
            pd.api.types.is_numeric_dtype(df[time])
            or pd.api.types.is_datetime64_any_dtype(df[time])
            or isinstance(df[time].dtype, pd.PeriodDtype)
        ):
            # Campaign finding: detrend/demeanq/detrendq cast the time
            # column to float for the trend/quarter design and raised a
            # raw numpy conversion error on string times (while demean,
            # which never touches the time values, succeeded).
            raise ValueError(
                f"rolling='{self.rolling}' requires a numeric or "
                f"datetime/Period time column (the unit-specific trend/"
                f"seasonal design uses the time values); column "
                f"'{time}' has dtype {df[time].dtype}. Encode the time "
                f"column numerically, or use rolling='demean'."
            )
        if self.rolling == "demean" and not (
            pd.api.types.is_numeric_dtype(df[time])
            or pd.api.types.is_datetime64_any_dtype(df[time])
            or isinstance(df[time].dtype, pd.PeriodDtype)
        ):
            # Round-20 review: plain object labels sort LEXICOGRAPHICALLY
            # ('Q10' < 'Q2'), silently corrupting the pre/post partition
            # and event-time positions; an ordered categorical declares
            # the chronology explicitly and is encoded to its codes.
            dtype = df[time].dtype
            if not (isinstance(dtype, pd.CategoricalDtype) and dtype.ordered):
                raise ValueError(
                    f"rolling='demean' with a non-numeric, non-datetime "
                    f"time column requires an ORDERED categorical (the "
                    f"chronology cannot be inferred from labels - "
                    f"lexicographic order breaks at e.g. 'Q10' < 'Q2'). "
                    f"Column '{time}' has dtype {dtype}. Use "
                    f"pd.Categorical(values, categories=..., ordered=True) "
                    f"or encode the time column numerically."
                )

    def get_transformation_diagnostics(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        first_treat: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the transformation step and return diagnostics without full estimation.

        This is useful for inspecting pre-treatment fit quality before running
        the full estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Name of the outcome column.
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        treatment : str
            Name of the treatment indicator column.
        first_treat : str or None, default None
            Name of the first-treatment-time (cohort) column, for
            staggered designs.

        Returns
        -------
        dict
            Common timing: transformation diagnostics (see _transform_*
            docstrings). Staggered: per-cohort diagnostics organized as
            ``{'method': ..., 'design': 'staggered', 'by_cohort': {g:
            diagnostics_g}}`` where each cohort g uses its own pre-period
            definition ``time < g`` and the same unit subset as estimation
            (cohort-g treated units plus the control superset implied by
            ``control_group``).
        """
        df = data.copy()

        # Same front-door validation as fit() (review round 3: diagnostics
        # previously accepted designs fit() rejects - non-binary treatment,
        # reserved-name collisions, duplicate panels, incoherent cohorts).
        self._validate_inputs(df, outcome, unit, time, treatment, first_treat, None, None)
        validate_binary(df[treatment].values, treatment)

        if first_treat is not None:
            # Staggered: each cohort g has its own pre-period t < g,
            # mirroring _transform_for_cohort in estimation. Datetime and
            # Period panels use the same integer-position encoding as fit().
            df, time, first_treat, label_maps = _encode_staggered_time_scale(df, time, first_treat)
            # Same cohort normalization as fit(): inf and beyond-window
            # cohorts are recoded to never-treated here too, so the
            # diagnostics iterate the same cohort set estimation uses.
            df[first_treat], _, _ = _normalize_cohorts(df[first_treat], max_time=df[time].max())
            _check_treatment_design(df, unit, time, treatment, first_treat)
            cohort_by_unit = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)[
                first_treat
            ]
            never_mask = cohort_by_unit.isna() | (cohort_by_unit == 0)
            never_units = cohort_by_unit.index[never_mask].to_list()
            treated_cohorts = sorted(
                value for value in pd.unique(df[first_treat]) if pd.notna(value) and value > 0
            )
            if not treated_cohorts:
                # Round-20 review: an all-never-treated panel returned an
                # empty {'by_cohort': {}} that read as successful
                # diagnostics; fit_staggered rejects the same input.
                raise ValueError("No treated cohorts found.")
            by_cohort: Dict[Any, Dict[str, Any]] = {}
            for g in treated_cohorts:
                treated_units = cohort_by_unit.index[cohort_by_unit == g].to_list()
                if self.control_group == "never_treated":
                    control_superset = never_units
                else:
                    later = cohort_by_unit.index[cohort_by_unit > g].to_list()
                    control_superset = never_units + later
                relevant_units = list(dict.fromkeys(treated_units + control_superset))
                cohort_frame = df.loc[df[unit].isin(relevant_units)].copy()
                pre_mask = cohort_frame[time] < g
                by_cohort[g] = self._run_transformation_diagnostics(
                    cohort_frame, outcome, unit, time, pre_mask
                )
            if label_maps is not None:
                cohort_labels = label_maps["cohort"]
                by_cohort = {cohort_labels.get(g, g): value for g, value in by_cohort.items()}
            return {
                "method": self.rolling,
                "design": "staggered",
                "by_cohort": by_cohort,
            }

        # Common timing: partition at the single onset S, same as
        # _fit_common_timing (round-9 review: the per-period max(D) rule
        # here still classified a controls-only post period as pre).
        if isinstance(df[time].dtype, pd.CategoricalDtype) and df[time].dtype.ordered:
            df[time] = df[time].cat.codes.astype(int)
        self._validate_common_time_scale(df, time)
        _check_treatment_design(df, unit, time, treatment, None)
        treated_times = df.loc[df[treatment] == 1, time]
        if len(treated_times) == 0:
            raise ValueError(
                "No post-treatment periods found. At least one period "
                "with some treatment=1 is required."
            )
        pre_mask = df[time] < treated_times.min()
        return self._run_transformation_diagnostics(df, outcome, unit, time, pre_mask)

    def _run_transformation_diagnostics(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        pre_mask: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Dispatch to the configured transformation with diagnostics enabled."""
        if self.rolling == "demean":
            _, diagnostics = self._transform_demean(
                df, outcome, unit, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "detrend":
            _, diagnostics = self._transform_detrend(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "demeanq":
            _, diagnostics = self._transform_demeanq(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        elif self.rolling == "detrendq":
            _, diagnostics = self._transform_detrendq(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )
        else:
            _, diagnostics = self._transform_detrend(
                df, outcome, unit, time, pre_mask, return_diagnostics=True
            )

        return diagnostics

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cohort: Optional[str],
        cluster: Optional[str],
        controls: Optional[List[str]],
    ) -> None:
        """Validate that all required columns exist and data is valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe.
        outcome, unit, time, treatment : str
            Required column names.
        cohort, cluster : str or None
            Optional column names.
        controls : list of str or None
            Optional control variable column names.

        Raises
        ------
        ValueError
            If any specified column is not in the dataframe.
        """
        required_cols = [outcome, unit, time, treatment]
        if cohort is not None:
            required_cols.append(cohort)
        if cluster is not None:
            required_cols.append(cluster)
        if controls:
            required_cols.extend(controls)

        # Internal working columns are written into the estimation frames;
        # a user role column bearing one of these names is silently
        # overwritten (review round 2: cluster='_treat' reported the
        # cluster labels' coefficient as the ATT).
        reserved = _RESERVED_INTERNAL_COLUMNS.intersection(required_cols)
        if reserved:
            raise ValueError(
                f"Column name(s) {sorted(reserved)} are reserved for LWDiD "
                f"internal use and cannot be supplied as outcome, unit, "
                f"time, treatment, first_treat, cluster, or covariate "
                f"columns. Rename the column(s) before fitting."
            )
        core_roles = {
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "treatment": treatment,
        }
        if cohort is not None:
            core_roles["first_treat"] = cohort
        seen: Dict[str, str] = {}
        for role, name in core_roles.items():
            if name in seen:
                raise ValueError(
                    f"Column '{name}' was supplied as both '{seen[name]}' and "
                    f"'{role}'; each role requires a distinct column."
                )
            seen[name] = role
        overlap = set(controls or []).intersection(core_roles.values())
        if overlap:
            raise ValueError(
                f"Covariate column(s) {sorted(overlap)} are already supplied "
                f"as outcome/unit/time/treatment/first_treat columns."
            )
        if controls and len(set(controls)) != len(controls):
            duplicated = sorted({c for c in controls if controls.count(c) > 1})
            raise ValueError(
                f"Covariate list contains duplicate column(s): {duplicated} "
                f"(a repeated covariate makes the design matrix rank-"
                f"deficient by construction)."
            )

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        # Check for NaN in key columns
        for col in [outcome, unit, time, treatment]:
            if df[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing values. "
                    f"Please handle missing data before fitting."
                )
        # Round-9 review: Inf outcomes passed the NaN check and were
        # silently np.isfinite-filtered inside staggered cells (changing
        # the estimation sample with no warning); non-numeric outcomes
        # crashed with raw conversion errors.
        try:
            outcome_values = df[outcome].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Outcome column '{outcome}' is not numeric (dtype "
                f"{df[outcome].dtype}); encode it numerically before fitting."
            ) from exc
        n_nonfinite_y = int((~np.isfinite(outcome_values)).sum())
        if n_nonfinite_y > 0:
            raise ValueError(
                f"Outcome column '{outcome}' contains {n_nonfinite_y} "
                f"non-finite value(s) (Inf). LWDiD does not silently drop "
                f"outcome rows; remove or recode them before fitting."
            )
        # Numeric TIME values must be finite too (round-14 review: +/-Inf
        # passed the NaN check and reached event-time arithmetic, raising
        # a raw OverflowError). Datetime/Period/ordered-label columns are
        # untouched.
        if pd.api.types.is_numeric_dtype(df[time]):
            time_values = df[time].to_numpy(dtype=float)
            n_nonfinite_t = int((~np.isfinite(time_values)).sum())
            if n_nonfinite_t > 0:
                raise ValueError(
                    f"Time column '{time}' contains {n_nonfinite_t} "
                    f"non-finite value(s) (Inf). Time periods must be "
                    f"finite; remove or recode them before fitting."
                )

        # Check panel structure: each unit-time pair should be unique
        duplicates = df.duplicated(subset=[unit, time], keep=False)
        if duplicates.any():
            n_dup = duplicates.sum()
            raise ValueError(
                f"Panel is not balanced: {n_dup} duplicate "
                f"unit-time observations found. Each (unit, time) "
                f"pair must be unique."
            )

        # Panel balance check
        obs_per_unit = df.groupby(unit)[time].nunique()
        if obs_per_unit.nunique() > 1:
            n_short = (obs_per_unit < obs_per_unit.max()).sum()
            warnings.warn(
                f"Unbalanced panel: {n_short} of {obs_per_unit.shape[0]} units have "
                f"fewer than {obs_per_unit.max()} time periods. LWDiD assumes balanced "
                "panels for optimal performance.",
                UserWarning,
                stacklevel=2,
            )

        # Unit-constancy contracts, shared by BOTH dispatch paths: LWDiD
        # collapses the panel to one row per unit, reading unit-level
        # covariate and cluster values. A time-varying column would make
        # the estimate depend on the row order of the input frame (and,
        # in staggered designs, silently pull post-treatment covariate
        # values into the cohort-time cells), so it is rejected here.
        for column in controls or []:
            n_missing = int(df[column].isna().sum())
            if n_missing > 0:
                # Campaign finding: NaN covariates silently dropped units
                # on the cell paths while poisoning the common-timing OLS
                # into a NaN ATT - unify by rejecting up front.
                raise ValueError(
                    f"Covariate '{column}' contains {n_missing} missing "
                    f"value(s). LWDiD does not silently drop or impute "
                    f"covariate rows; remove or impute them before fitting."
                )
            # Round-8 review: Inf passed the NaN check and was silently
            # filtered per staggered cell (changing the estimation sample)
            # or crashed inside the solver on the common-timing path;
            # non-numeric covariates crashed with a raw conversion error.
            try:
                column_values = df[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Covariate '{column}' is not numeric (dtype "
                    f"{df[column].dtype}); encode it numerically before "
                    f"fitting."
                ) from exc
            n_nonfinite = int((~np.isfinite(column_values)).sum())
            if n_nonfinite > 0:
                raise ValueError(
                    f"Covariate '{column}' contains {n_nonfinite} "
                    f"non-finite value(s) (Inf). LWDiD does not silently "
                    f"drop covariate rows; remove or recode them before "
                    f"fitting."
                )
            varying = df.groupby(unit)[column].nunique(dropna=False)
            if (varying > 1).any():
                raise ValueError(
                    f"Covariate '{column}' is not unit-constant; time-varying "
                    "covariates are not supported by LWDiD. Aggregate the "
                    "column to one value per unit (e.g. its pre-treatment "
                    "value) before fitting."
                )
        if cluster is not None and cluster != unit:
            n_missing = int(df[cluster].isna().sum())
            if n_missing > 0:
                raise ValueError(
                    f"Cluster column '{cluster}' contains {n_missing} missing "
                    f"value(s); every observation must belong to a cluster."
                )
            varying = df.groupby(unit)[cluster].nunique(dropna=False)
            if (varying > 1).any():
                raise ValueError(
                    f"Cluster column '{cluster}' is not unit-constant; each "
                    "unit must belong to exactly one cluster. Assign one "
                    "cluster value per unit before fitting."
                )

    def _fit_common_timing(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cluster: Optional[str],
        controls: List[str],
    ) -> LWDiDResults:
        """Estimate ATT under common treatment timing.

        All treated units adopt treatment at the same time period.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable column.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        treatment : str
            Binary treatment indicator column.
        cluster : str or None
            Cluster variable for cluster-robust SEs.
        controls : list of str
            Control variable columns.

        Returns
        -------
        LWDiDResults
            Estimation results.
        """
        # Treatment-design validation (absorbing + common timing) is
        # performed by _check_treatment_design in fit() before dispatch.

        # Step 1: Partition the calendar support at the single adoption
        # period S (validated by _check_treatment_design): pre = t < S,
        # post = t >= S. Round-8 review: the previous per-period
        # `groupby(time)[treatment].max()` partition silently classified a
        # post period with no observed TREATED rows (controls only) as
        # pre-treatment, contaminating the rolling pre window and biasing
        # a zero-effect trend panel to ATT ~ 0.75.
        treated_times = df.loc[df[treatment] == 1, time]
        if len(treated_times) == 0:
            raise ValueError(
                "No post-treatment periods found. At least one period "
                "with some treatment=1 is required."
            )
        onset_s = treated_times.min()
        support = sorted(pd.unique(df[time]))
        pre_periods = [t for t in support if t < onset_s]
        post_periods = [t for t in support if t >= onset_s]

        if len(pre_periods) == 0:
            raise ValueError(
                "No pre-treatment periods found. At least one period "
                "with all treatment=0 is required."
            )
        if len(post_periods) == 0:
            raise ValueError(
                "No post-treatment periods found. At least one period "
                "with some treatment=1 is required."
            )

        # Identify treated and control units
        unit_ever_treated = df.groupby(unit)[treatment].max()
        treated_units = unit_ever_treated[unit_ever_treated == 1].index.tolist()
        control_units = unit_ever_treated[unit_ever_treated == 0].index.tolist()
        treated_set = set(treated_units)

        if len(treated_units) == 0:
            raise ValueError("No treated units found in the data.")
        if len(control_units) == 0:
            raise ValueError(
                "No control units found. At least one never-treated " "unit is required."
            )

        # Step 2: Apply transformation
        pre_mask = df[time].isin(pre_periods)

        if self.rolling == "demean":
            df = self._transform_demean(df, outcome, unit, pre_mask)
        elif self.rolling == "detrend":
            df = self._transform_detrend(df, outcome, unit, time, pre_mask)
        elif self.rolling == "demeanq":
            df = self._transform_demeanq(df, outcome, unit, time, pre_mask)
        elif self.rolling == "detrendq":
            df = self._transform_detrendq(df, outcome, unit, time, pre_mask)
        else:
            df = self._transform_detrend(df, outcome, unit, time, pre_mask)

        # Per-period event-study surface (LW 2026 eq. 2.20): each post
        # period is one small cross-sectional regression on the transformed
        # outcome, so the surface is populated at fit time, exactly like
        # the staggered path.
        (
            event_effects,
            reference_periods,
            event_vcov,
            event_vcov_index,
            event_study_df,
            cband_method,
            cband_crit_value,
            cband_n_bootstrap,
        ) = self._common_timing_event_study(
            df, unit, time, cluster, controls, post_periods, treated_set
        )

        # Step 3: Take post-treatment cross-section of transformed outcomes
        # Average transformed outcome over the FIXED post window per unit.
        # The estimand is the paper's fixed-window post average (LW 2026,
        # denominator T - S + 1): units not observing EVERY post period are
        # dropped as complete cases with a warning (round-6 review: the
        # previous per-unit mean over whichever post periods a unit
        # observed let calendar composition masquerade as treatment effect
        # - a zero-effect panel where treated units observed one extra
        # post period reported that period's trend as ATT).
        post_mask = df[time].isin(post_periods)
        post_df = df.loc[post_mask].copy()

        # Reindex over EVERY panel unit so zero-post-row units are counted
        # as incomplete too (round-10 review: they were absent from the
        # counts, silently vanished in the merge, and the documented
        # fixed-window drop warning never fired for them).
        all_panel_units = pd.Index(pd.unique(df[unit]))
        post_counts = (
            post_df.loc[np.isfinite(post_df["_ydot"])]
            .groupby(unit)["_ydot"]
            .size()
            .reindex(all_panel_units, fill_value=0)
        )
        complete_units = set(post_counts.index[post_counts == len(post_periods)])
        n_incomplete = int((post_counts < len(post_periods)).sum())
        if n_incomplete > 0:
            warnings.warn(
                f"LWDiD: {n_incomplete} unit(s) dropped from the collapsed "
                f"cross-section for lacking a finite transformed outcome in "
                f"every post-treatment period (missing rows or failed "
                f"transformation): the headline ATT is the fixed-window "
                f"post average over complete cases. See "
                f"docs/methodology/REGISTRY.md (LWDiD).",
                UserWarning,
                stacklevel=2,
            )
        post_df = post_df.loc[post_df[unit].isin(complete_units)]

        # Compute unit-level average of transformed outcome in post periods
        unit_post_avg = post_df.groupby(unit)["_ydot"].mean().reset_index()
        unit_post_avg.columns = [unit, "_ydot_avg"]

        # Build cross-sectional dataset
        # Take first observation per unit for controls
        cs_df = df.drop_duplicates(subset=[unit], keep="first")[[unit] + controls].copy()
        # Treatment indicator: 1 if unit is ever-treated
        cs_df["_treat"] = cs_df[unit].isin(treated_set).astype(float)
        if cluster is not None:
            # Get cluster from original data
            if cluster == unit:
                cs_df[cluster] = cs_df[unit]
            else:
                cluster_map = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)[
                    cluster
                ]
                cs_df[cluster] = cs_df[unit].map(cluster_map)

        cs_df = cs_df.merge(unit_post_avg, on=unit, how="inner")

        # After merge, drop units whose transformation produced NaN
        n_before_drop = len(cs_df)
        cs_df = cs_df.dropna(subset=["_ydot_avg"])
        n_dropped = n_before_drop - len(cs_df)
        if n_dropped > 0 and len(cs_df) > 0:
            warnings.warn(
                f"LWDiD: {n_dropped} unit(s) dropped due to NaN transformed outcomes "
                f"(insufficient pre-treatment periods for '{self.rolling}' transformation).",
                UserWarning,
                stacklevel=2,
            )
        if len(cs_df) == 0:
            nan = float("nan")
            warnings.warn(
                f"All units have NaN transformed outcomes for rolling='{self.rolling}'. "
                "Likely insufficient pre-treatment periods. Cannot estimate ATT.",
                UserWarning,
                stacklevel=2,
            )
            return LWDiDResults(
                att=nan,
                se=nan,
                t_stat=nan,
                p_value=nan,
                conf_int=(nan, nan),
                n_obs=0,
                n_treated=0,
                n_control=0,
                rolling=self.rolling,
                estimation_method=self.estimation_method,
                vcov_type=self.vcov_type,
                control_group=self.control_group,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed,
                cluster_name=cluster,
                pscore_trim=(
                    self.pscore_trim if self.estimation_method in ("ipw", "dr", "psm") else None
                ),
                psm_config=(
                    {
                        "pscore_trim": self.pscore_trim,
                        "n_neighbors": self.n_neighbors,
                        "caliper": self.caliper,
                        "with_replacement": self.with_replacement,
                    }
                    if self.estimation_method == "psm"
                    else None
                ),
                alpha=self.alpha,
                event_study_effects=event_effects,
                event_study_vcov=event_vcov,
                event_study_vcov_index=event_vcov_index,
                event_study_df=event_study_df,
                reference_periods=reference_periods,
                cband_method=cband_method,
                cband_crit_value=cband_crit_value,
                cband_n_bootstrap=cband_n_bootstrap,
            )

        # Step 4: Estimate ATT
        y = cs_df["_ydot_avg"].values.astype(np.float64)
        treat = cs_df["_treat"].values.astype(np.float64)
        n_obs = len(y)
        n_treated = int(treat.sum())
        n_control = n_obs - n_treated

        if n_treated == 0 or n_control == 0:
            # Round-6 review: the raw-panel arm counts are checked before
            # the transformation, but complete-case/NaN drops can empty an
            # arm - the pre-fix code dispatched a one-arm design into the
            # estimators (rank warnings, NaN arithmetic, and the IPW family
            # entered propensity fitting with an empty group).
            raise ValueError(
                f"After the transformation and complete-case drops, the "
                f"collapsed cross-section has {n_treated} treated and "
                f"{n_control} control unit(s); estimation requires at "
                f"least one of each. Likely insufficient pre-treatment "
                f"periods or incomplete post-period coverage for one arm "
                f"under rolling='{self.rolling}'."
            )

        # Guard: if transformation produced all-NaN outcomes, return NaN result
        if np.all(np.isnan(y)):
            warnings.warn(
                f"All transformed outcomes are NaN (likely insufficient "
                f"pre-treatment periods for '{self.rolling}' transformation). "
                f"Cannot estimate ATT.",
                UserWarning,
                stacklevel=2,
            )
            nan = float("nan")
            return LWDiDResults(
                att=nan,
                se=nan,
                t_stat=nan,
                p_value=nan,
                conf_int=(nan, nan),
                n_obs=n_obs,
                n_treated=n_treated,
                n_control=n_control,
                rolling=self.rolling,
                estimation_method=self.estimation_method,
                vcov_type=self.vcov_type,
                control_group=self.control_group,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed,
                cluster_name=cluster,
                pscore_trim=(
                    self.pscore_trim if self.estimation_method in ("ipw", "dr", "psm") else None
                ),
                psm_config=(
                    {
                        "pscore_trim": self.pscore_trim,
                        "n_neighbors": self.n_neighbors,
                        "caliper": self.caliper,
                        "with_replacement": self.with_replacement,
                    }
                    if self.estimation_method == "psm"
                    else None
                ),
                alpha=self.alpha,
                event_study_effects=event_effects,
                event_study_vcov=event_vcov,
                event_study_vcov_index=event_vcov_index,
                event_study_df=event_study_df,
                reference_periods=reference_periods,
                cband_method=cband_method,
                cband_crit_value=cband_crit_value,
                cband_n_bootstrap=cband_n_bootstrap,
            )

        # Build controls matrix
        controls_matrix = None
        if controls:
            controls_matrix = cs_df[controls].values.astype(np.float64)

        # Get cluster ids (clustered inference activates via the cluster=
        # constructor parameter)
        cluster_ids = None
        collapsed_single_cluster = False
        if cluster is not None:
            cluster_ids = cs_df[cluster].values
            if len(np.unique(cluster_ids)) < 2:
                collapsed_single_cluster = True
                # The NaN-transformation dropna can reduce the collapsed
                # cross-section below 2 clusters even when the raw panel
                # passed the fit-level guard - fail closed rather than let
                # a single-cluster CR1 SE through on roundoff.
                warnings.warn(
                    "LWDiD: after transformation drops, the collapsed "
                    "cross-section contains fewer than 2 clusters; "
                    "cluster-robust inference is not identified. The point "
                    "estimate is retained with NaN inference.",
                    UserWarning,
                    stacklevel=2,
                )
                cluster_ids = None  # estimate the POINT unclustered

        # Estimate
        att, se, coefs, vcov, n_params, _ = self._dispatch_estimator(
            y, treat, controls_matrix, cluster_ids, n_obs
        )
        if collapsed_single_cluster:
            se = np.nan  # fail-closed (warned above); point retained

        # Step 5: Compute inference
        # n_params is the fitted design's parameter count, so the residual
        # df is design-coherent (LW 2026 Section 2): T_{N-2} without
        # controls, T_{N-K-2} for the plain design and T_{N-2K-2} when the
        # treatment-covariate interaction is active.
        if self.estimation_method == "psm":
            # PSM inference is NaN by contract and uses no residual df
            # (round-20 review: the exact-OLS residual-df guard below
            # rejected valid point-only matching fits whose nominal
            # propensity width exhausted an OLS df count PSM never uses).
            df_dof = None
        elif n_obs < 3 or n_obs - n_params <= 0:
            # Registry small-sample guards (N >= 3; N > K + 2 with controls):
            # coercing the residual df to 1 fabricated exact inference on
            # invalid designs, and N=2 reached sse/(n-k) division by zero
            # (review finding).
            raise ValueError(
                f"Invalid exact-inference design: {n_obs} collapsed "
                f"observation(s) with {n_params} fitted parameter(s). LWDiD "
                f"requires at least 3 cross-sectional units and a positive "
                f"residual df (N > K + 2 with controls)."
            )
        else:
            df_dof = n_obs - n_params

        # Issue 3: Cluster-robust inference uses df = G-1
        if cluster_ids is not None:
            df_dof = max(int(len(np.unique(cluster_ids))) - 1, 1)
        elif collapsed_single_cluster:
            df_dof = 0  # safe_inference fails the tuple closed

        # Scale-equivariant degenerate-SE guard, same rule as the
        # staggered/event surfaces (round-21 review: an exactly fitted
        # panel produced se ~ 1e-16 and t ~ 1e16 on the common headline).
        from diff_diff.lwdid_staggered import _guard_standard_error

        se = _guard_standard_error(att, se, scale=float(np.max(np.abs(y))) if len(y) else 0.0)

        t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_dof)

        # Step 6: Bootstrap if requested
        inference_basis = None
        if self.n_bootstrap > 0 and collapsed_single_cluster:
            # Round-5 review: the unconditional bootstrap call overwrote
            # the single-effective-cluster fail-closed NaN inference with a
            # finite (near-zero) SE built from the raw cluster map. The
            # fail-closed state wins; the earlier warning already fired.
            warnings.warn(
                "LWDiD: bootstrap skipped - fewer than 2 effective clusters "
                "survive the transformation, so clustered inference is not "
                "identified (the NaN inference tuple is retained).",
                UserWarning,
                stacklevel=2,
            )
        elif self.n_bootstrap > 0:
            att, se, t_stat, p_value, conf_int, df_dof = self._bootstrap(
                df,
                outcome,
                unit,
                time,
                treatment,
                cluster,
                controls,
                pre_periods,
                post_periods,
                treated_units,
                control_units,
            )
            # Provenance (review round 3): the headline se/p/CI now come
            # from the resampling bootstrap while params/vcov remain the
            # analytical regression quantities - record which family backs
            # the headline so consumers (and summary()) can tell.
            inference_basis = "cluster_bootstrap" if cluster is not None else "unit_bootstrap"

        result = LWDiDResults(
            inference_basis=inference_basis,
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=n_obs,
            n_treated=n_treated,
            n_control=n_control,
            rolling=self.rolling,
            estimation_method=self.estimation_method,
            vcov_type=self.vcov_type,
            alpha=self.alpha,
            cluster_name=cluster if cluster_ids is not None else None,
            control_group=self.control_group,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
            pscore_trim=(
                self.pscore_trim if self.estimation_method in ("ipw", "dr", "psm") else None
            ),
            psm_config=(
                {
                    "pscore_trim": self.pscore_trim,
                    "n_neighbors": self.n_neighbors,
                    "caliper": self.caliper,
                    "with_replacement": self.with_replacement,
                }
                if self.estimation_method == "psm"
                else None
            ),
            n_clusters=int(len(np.unique(cluster_ids))) if cluster_ids is not None else None,
            cohort_effects=None,
            params=coefs,
            vcov=vcov,
            df_inference=df_dof,
            event_study_effects=event_effects,
            event_study_vcov=event_vcov,
            event_study_vcov_index=event_vcov_index,
            event_study_df=event_study_df,
            reference_periods=reference_periods,
            cband_method=cband_method,
            cband_crit_value=cband_crit_value,
            cband_n_bootstrap=cband_n_bootstrap,
        )

        # Fit-time replay spec for the post-fit advanced-inference methods
        # (round-5 review: they previously accepted arbitrary caller arrays
        # and a non-interacted design, caching p-values for a DIFFERENT
        # estimand than .att on covariate-unbalanced RA fits).
        object.__setattr__(
            result,
            "_replay_spec",
            {
                "y": y.copy(),
                "treatment": treat.copy(),
                "controls": controls_matrix.copy() if controls_matrix is not None else None,
                "cluster_ids": cluster_ids.copy() if cluster_ids is not None else None,
            },
        )

        # Final safety net: warn if result has NaN ATT
        if np.isnan(result.att):
            warnings.warn(
                f"LWDiD estimation returned NaN ATT. This typically indicates "
                f"insufficient data for the '{self.rolling}' transformation or "
                f"numerical issues in estimation. Check your data structure and "
                f"consider using a simpler transformation (e.g., rolling='demean').",
                UserWarning,
                stacklevel=2,
            )

        return result

    def _common_timing_event_study(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        cluster: Optional[str],
        controls: List[str],
        post_periods: List[Any],
        treated_set: set,
    ) -> Tuple[
        Dict[int, Dict[str, Any]],
        Tuple[int, ...],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Dict[int, Any],
        Optional[str],
        Optional[float],
        Optional[int],
    ]:
        """Per-period event-study surface for a common-timing fit.

        LW (2026) eq. (2.20): after the rolling transformation, the effect
        for post period t is the coefficient on D in the cross-sectional
        regression of the transformed outcome at t -- numerically identical
        to a standard DiD on the subset panel {1, ..., S-1, t}. Each post
        period is therefore one small regression run through the same
        ``_dispatch_estimator`` path as the overall ATT, and the surface
        follows the staggered storage contract (integer event-time keys,
        position-difference convention).

        Reference anchors are the transformation's nominal anchors
        (``-1`` for demean/demeanq, ``-2, -1`` for detrend/detrendq)
        restricted to genuinely observed relative times: an unobserved
        anchor is never synthesized.

        Returns
        -------
        tuple
            ``(event_effects, reference_periods, event_vcov,
            event_vcov_index, event_study_df, cband_method,
            cband_crit_value, cband_n_bootstrap)``.
        """
        from diff_diff.lwdid_staggered import (
            _guard_standard_error,
            compute_event_study_bands,
        )

        # Event-time convention, shared with the staggered path (round-9
        # review: the interfaces previously disagreed on gapped numeric
        # calendars - common used ordered-support positions while
        # staggered used arithmetic t - g): NUMERIC calendars use the
        # Registry's arithmetic r = t - S (validated integral so distinct
        # horizons can never merge under the integer storage keys);
        # datetime/Period calendars use position differences (matching
        # _encode_staggered_time_scale, which encodes them to positions
        # before the staggered machinery runs).
        all_times = sorted(pd.unique(df[time]))
        onset_s = min(post_periods)
        if not pd.api.types.is_numeric_dtype(df[time]):
            # datetime/Period (position-encoded on the staggered path) and
            # ordered string labels (demean-only contract): position
            # differences on the ordered support.
            time_pos = {value: index for index, value in enumerate(all_times)}
            g_pos = time_pos[onset_s]
            relative_of = {t: int(time_pos[t] - g_pos) for t in all_times}
        else:
            relative_of = {}
            for t in all_times:
                rel = float(t) - float(onset_s)
                if abs(rel - round(rel)) > 1e-9:
                    raise ValueError(
                        f"Event time t - S = {rel!r} for period {t!r} is not "
                        f"an integer: the event-study surface stores integer "
                        f"event-time keys and cannot represent fractional "
                        f"horizons without silently merging them. Encode the "
                        f"time column as consecutive integer periods or as "
                        f"datetime/Period values."
                    )
                relative_of[t] = int(round(rel))
        nominal_anchors = (-1,) if self.rolling in ("demean", "demeanq") else (-2, -1)
        observed_relative = set(relative_of.values())
        reference_periods = tuple(r for r in nominal_anchors if r in observed_relative)

        unit_rows = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)
        all_units = unit_rows.index.to_list()
        unit_to_index = {value: index for index, value in enumerate(all_units)}
        global_cluster_ids = None
        if cluster is not None:
            if cluster == unit:
                global_cluster_ids = unit_rows.index.to_numpy()
            else:
                global_cluster_ids = unit_rows.loc[all_units, cluster].to_numpy()

        event_effects: Dict[int, Dict[str, Any]] = {}
        event_influence: Dict[int, np.ndarray] = {}
        skipped: List[Tuple[int, str]] = []
        for t in post_periods:
            relative_time = relative_of[t]
            columns = [unit, "_ydot"] + controls
            if cluster is not None and cluster not in columns:
                columns.append(cluster)
            cell = df.loc[df[time] == t, columns].drop_duplicates(subset=[unit], keep="first")
            finite = np.isfinite(cell["_ydot"].to_numpy(dtype=float))
            if controls:
                finite &= np.all(np.isfinite(cell[controls].to_numpy(dtype=float)), axis=1)
            cell = cell.loc[finite].copy()
            treatment_vec = cell[unit].isin(treated_set).to_numpy(dtype=float)
            n_treated = int(treatment_vec.sum())
            n_control = int(len(treatment_vec) - n_treated)
            if n_treated == 0 or n_control == 0:
                skipped.append((relative_time, "zero_treated_control"))
                continue

            y = cell["_ydot"].to_numpy(dtype=float)
            controls_matrix = cell[controls].to_numpy(dtype=float) if controls else None
            cluster_ids = None
            single_cluster_period = False
            if cluster is not None:
                cluster_ids = cell[cluster].to_numpy()
                if len(np.unique(cluster_ids)) < 2:
                    warnings.warn(
                        "LWDiD: a common-timing event-study period cell "
                        "contains fewer than 2 clusters; its cluster-robust "
                        "inference is not identified (point retained, "
                        "inference NaN).",
                        UserWarning,
                        stacklevel=2,
                    )
                    single_cluster_period = True
                    cluster_ids = None  # estimate the POINT unclustered
            try:
                att, se, _, _, n_params, influence = self._dispatch_estimator(
                    y, treatment_vec, controls_matrix, cluster_ids, len(cell)
                )
            except ValueError as exc:
                if "Invalid exact-inference design" in str(exc):
                    # Non-estimable period cell (Registry: NaN, not a
                    # mid-fit raise; only the OVERALL design raises).
                    skipped.append((relative_time, "insufficient_sample"))
                    continue
                raise
            if not np.isfinite(att):
                skipped.append((relative_time, "non_finite_estimate"))
                continue

            se = _guard_standard_error(att, se, scale=float(np.max(np.abs(y))))
            if single_cluster_period:
                se = np.nan  # fail-closed (warned above); point retained
                influence = None
            if cluster_ids is not None:
                df_event = max(len(np.unique(cluster_ids)) - 1, 1)
            else:
                # Raw residual df: safe_inference fails the tuple closed
                # when df <= 0 (no fabricated df=1 - review finding).
                df_event = len(cell) - n_params
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_event)
            event_effects[relative_time] = {
                "effect": float(att),
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_treated": n_treated,
                "n_control": n_control,
                "n_cells": 1,
                "df": df_event,
            }
            if influence is not None and np.isfinite(se):
                global_influence = np.zeros(len(all_units), dtype=float)
                for local_index, unit_value in enumerate(cell[unit].to_list()):
                    global_influence[unit_to_index[unit_value]] = influence[local_index]
                event_influence[relative_time] = global_influence

        if skipped:
            preview = ", ".join(f"r={r}: {reason}" for r, reason in skipped[:6])
            suffix = "" if len(skipped) <= 6 else f"; plus {len(skipped) - 6} more"
            warnings.warn(
                f"LWDiD skipped {len(skipped)} per-period effect(s): {preview}{suffix}. "
                "The event-study surface omits these event times.",
                UserWarning,
                stacklevel=2,
            )

        (
            event_vcov,
            event_vcov_index,
            cband_method,
            cband_crit_value,
            cband_n_bootstrap,
        ) = compute_event_study_bands(self, event_effects, event_influence, global_cluster_ids)
        event_study_df = {
            label: value["df"]
            for label, value in event_effects.items()
            if value.get("df") is not None
        }
        return (
            event_effects,
            reference_periods,
            event_vcov,
            event_vcov_index,
            event_study_df,
            cband_method,
            cband_crit_value,
            cband_n_bootstrap,
        )

    def _composite_regression_aggregation(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
    ) -> Tuple[float, float, int, int, int, float, Dict[Any, int]]:
        """Compute tau_omega via composite outcome regression (LW 2026 Eq 7.18/7.19).

        For staggered designs, constructs a composite outcome vector:
        - Treated units in cohort g: use their cohort's transformed outcome
        - Never-treated units: weighted average of all cohort transformations
        Then runs a single cross-sectional OLS: y_composite ~ [1, D_ever_treated]

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        outcome : str
            Outcome variable column.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        cohort : str
            Cohort (first treatment time) column.

        Returns
        -------
        att : float
            ATT from composite regression coefficient on D.
        se : float
            Classical OLS SE from composite regression.
        dof : int
            Degrees of freedom (n_complete_case_units - 2).
        n_treated_dropped : int
            Treated units dropped by the complete-case resolution.
        n_controls_dropped : int
            Control units dropped by the complete-case resolution.
        """
        # Step 1: Identify cohorts and unit membership
        fy = df.groupby(unit)[cohort].first()
        cohorts = sorted([g for g in fy.unique() if g > 0 and not np.isnan(g)])
        n_treat = int((fy > 0).sum())

        if n_treat == 0:
            return np.nan, np.nan, 0, 0, 0, 0.0, {}

        # Step 2: For each cohort g, compute per-unit post-average transformed outcome
        # using cohort g's pre-period for ALL units
        ydot_by_cohort: Dict[Any, pd.Series] = {}
        for g in cohorts:
            # pre_mask: periods < g (i.e., time <= g-1)
            pre_mask_g = df[time] < g
            post_mask_g = df[time] >= g

            # Apply transformation to full dataset. The composite (tau_omega)
            # estimand is defined for the plain demean/detrend transforms
            # only; the routing gate in lwdid_staggered restricts rolling,
            # and this raise keeps a future gate change from silently
            # substituting a non-seasonal transform for a q-variant again
            # (campaign finding: demeanq/detrendq were mapped to
            # demean/detrend here, moving the point by ~8% silently).
            if self.rolling == "demean":
                df_transformed = self._transform_demean(df, outcome, unit, pre_mask_g)
            elif self.rolling == "detrend":
                df_transformed = self._transform_detrend(df, outcome, unit, time, pre_mask_g)
            else:
                raise ValueError(
                    f"Internal error: composite (tau_omega) aggregation is "
                    f"only defined for rolling in ('demean', 'detrend'); got "
                    f"{self.rolling!r}. The routing gate should not have "
                    f"dispatched here."
                )

            # Per-unit average of transformed outcome in post-periods
            # (>= g). Completeness semantics (ADJUDICATED, pinned by the
            # acceptance suite's frozen reference oracle): a unit
            # contributes cohort g's component iff its OBSERVED post-g
            # rows yield a finite average - partial post windows are
            # averaged over the observed rows, symmetrically for treated
            # and control units. A stricter every-period window rule was
            # considered in review round 8 and NOT adopted: it changes the
            # estimand the acceptance oracle pins; the unbalanced
            # composition caveat is documented in REGISTRY.
            post_data = df_transformed.loc[post_mask_g]  # type: ignore[union-attr]
            unit_avg_g = post_data.groupby(unit)["_ydot"].mean()
            ydot_by_cohort[g] = unit_avg_g

        # Step 3: Complete-case resolution (deterministic, one-directional).
        # Fixed cohort weights omega_g = N_g / N_treat are defined on the
        # ESTIMATION sample: units that cannot contribute their required
        # transformed outcomes are dropped with a warning (never silently
        # zero-filled or asymmetrically reweighted by the OLS finite mask).
        all_units = fy.index

        # 3.1: Treated units must have a finite post-window average for
        # their OWN cohort (missing post rows or a NaN transform both
        # count - the pre-fix code silently NaN'd these out of the OLS,
        # implicitly reweighting the treated side).
        surviving_treated: List[Any] = []
        n_treated_dropped = 0
        for u in all_units:
            g_u = fy[u]
            if not (g_u > 0):
                continue
            value = ydot_by_cohort[g_u].get(u, np.nan)
            if np.isfinite(value):
                surviving_treated.append(u)
            else:
                n_treated_dropped += 1
        if n_treated_dropped:
            warnings.warn(
                f"LWDiD tau_omega composite: dropped {n_treated_dropped} "
                f"treated unit(s) with no finite post-window transformed "
                f"outcome for their cohort (complete-case estimation; cohort "
                f"weights are recomputed on the surviving sample).",
                UserWarning,
                stacklevel=3,
            )

        # 3.2: Recompute cohort masses on the surviving treated sample.
        fy_surviving = fy.loc[surviving_treated]
        cohort_sizes = {g: int((fy_surviving == g).sum()) for g in cohorts}
        weighted_cohorts = [g for g in cohorts if cohort_sizes[g] > 0]
        n_treat_cc = len(surviving_treated)

        # 3.3: Control units must observe every surviving-weight cohort's
        # post window with a finite transformed outcome (the pre-fix code
        # injected a literal 0.0 for missing entries, biasing the
        # composite control mean toward zero).
        control_units = [u for u in all_units if not (fy[u] > 0)]
        surviving_controls: List[Any] = []
        n_controls_dropped = 0
        for u in control_units:
            vals = [ydot_by_cohort[g].get(u, np.nan) for g in weighted_cohorts]
            if vals and np.all(np.isfinite(vals)):
                surviving_controls.append(u)
            else:
                n_controls_dropped += 1
        if n_controls_dropped:
            warnings.warn(
                f"LWDiD tau_omega composite: dropped {n_controls_dropped} "
                f"control unit(s) not observing every treated cohort's post "
                f"window with a finite transformed outcome (complete-case "
                f"estimation with fixed cohort weights).",
                UserWarning,
                stacklevel=3,
            )

        # 3.4: Empty-arm fail-closed guard (the pre-drop n_treat check
        # does not cover drops emptying an arm).
        if n_treat_cc == 0 or not surviving_controls:
            warnings.warn(
                "LWDiD tau_omega composite: complete-case filtering left an "
                "empty treated or control arm; the composite ATT and its "
                "inference are NaN.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.nan, 0, n_treated_dropped, n_controls_dropped, 0.0, dict(cohort_sizes)

        # Step 4: Assemble composite outcome vector on the complete-case
        # sample (finite by construction).
        included = surviving_treated + surviving_controls
        n = len(included)
        y_composite = np.empty(n, dtype=np.float64)
        d_ever_treated = np.empty(n, dtype=np.float64)
        for i, u in enumerate(included):
            g_u = fy[u]
            if g_u > 0:
                y_composite[i] = float(ydot_by_cohort[g_u][u])
                d_ever_treated[i] = 1.0
            else:
                weighted_sum = 0.0
                for g in weighted_cohorts:
                    w_g = cohort_sizes[g] / n_treat_cc
                    weighted_sum += w_g * float(ydot_by_cohort[g][u])
                y_composite[i] = weighted_sum
                d_ever_treated[i] = 0.0

        if n < 3:
            return np.nan, np.nan, 0, n_treated_dropped, n_controls_dropped, 0.0, dict(cohort_sizes)

        # Step 5: Single OLS regression y_composite ~ [1, D] via the house
        # linalg engine (classical SE from the same regression).
        X = np.column_stack([np.ones(n, dtype=np.float64), d_ever_treated])
        coefs, _, vcov = solve_ols(X, y_composite, return_vcov=True, vcov_type="classical")
        att = float(coefs[1])
        dof = n - 2
        if vcov is not None and np.isfinite(vcov[1, 1]):
            se = float(np.sqrt(max(vcov[1, 1], 0.0)))
        else:
            se = np.nan

        # Data scale for the degenerate-SE guard (scale-equivariant
        # roundoff reference - see _guard_standard_error).
        y_scale = float(np.max(np.abs(y_composite))) if len(y_composite) else 0.0
        # Survivor cohort masses (round-12 review: the drops-route
        # aggregation needs these - raw masses left dropped treated
        # units in the cohort weights).
        return att, se, dof, n_treated_dropped, n_controls_dropped, y_scale, dict(cohort_sizes)

    def _transform_demean(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific demeaning transformation.

        For each unit, compute the mean of the outcome in pre-treatment
        periods, then subtract that mean from ALL periods (pre and post).

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing demeaned outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()

        # Compute pre-treatment mean for each unit
        pre_df = df.loc[pre_mask, [unit_col, outcome_col]]
        pre_means = pre_df.groupby(unit_col)[outcome_col].mean()

        # Collect per-unit diagnostics if requested
        per_unit: Dict[Any, Dict[str, Any]] = {}
        if return_diagnostics:
            pre_stds = pre_df.groupby(unit_col)[outcome_col].std()
            pre_counts = pre_df.groupby(unit_col)[outcome_col].count()
            post_mask_inv = ~pre_mask
            post_df = df.loc[post_mask_inv, [unit_col, outcome_col]]
            post_counts = post_df.groupby(unit_col)[outcome_col].count()
            all_units = df[unit_col].unique()
            for uid in all_units:
                has_pre = uid in pre_means.index
                info: Dict[str, Any] = {
                    "pre_mean": float(pre_means[uid]) if has_pre else float("nan"),
                    "pre_n_periods": int(pre_counts.get(uid, 0)),
                    "pre_std": float(pre_stds.get(uid, float("nan"))),
                    "post_n_periods": int(post_counts.get(uid, 0)),
                    "valid": has_pre,
                }
                per_unit[uid] = info

        # Map pre-means back to all observations
        unit_means = df[unit_col].map(pre_means)

        # Check for units with no pre-treatment obs (shouldn't happen
        # after validation, but guard defensively)
        no_pre = unit_means.isna()
        if no_pre.any():
            n_missing = df.loc[no_pre, unit_col].nunique()
            warnings.warn(
                f"{n_missing} unit(s) have no pre-treatment observations. "
                f"Their transformed outcomes will be NaN.",
                UserWarning,
                stacklevel=2,
            )

        # Subtract pre-treatment mean from all periods
        df["_ydot"] = df[outcome_col].values - unit_means.values

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            pre_period_counts = [per_unit[uid]["pre_n_periods"] for uid in valid_units]
            diagnostics: Dict[str, Any] = {
                "method": "demean",
                "description": "\u0232_{i,pre} subtracted from all periods (Procedure 2.1, Eq 2.12)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                    "mean_pre_periods": (
                        float(np.mean(pre_period_counts)) if pre_period_counts else 0.0
                    ),
                    "min_pre_periods": int(np.min(pre_period_counts)) if pre_period_counts else 0,
                    "max_pre_periods": int(np.max(pre_period_counts)) if pre_period_counts else 0,
                },
            }
            return df, diagnostics

        return df

    def _transform_detrend(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific linear detrending transformation.

        For each unit, fit y = alpha + beta*t on pre-treatment periods
        using scipy.linalg.lstsq, then subtract the fitted trend from
        ALL periods.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing detrended outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        time_arr = df[time_col].values.astype(np.float64)
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            t_u = time_arr[idx_u]
            y_u = y_arr[idx_u]
            pre_u = pre_arr[idx_u]

            # Pre-treatment data for this unit
            pre_sel = pre_u.astype(bool)
            n_pre = int(pre_sel.sum())

            if n_pre < 2:
                warnings.warn(
                    f"Unit {uid}: detrend requires at least 2 "
                    f"pre-treatment periods, found {n_pre}. "
                    f"Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "pre_n_periods": n_pre,
                        "residual_std": float("nan"),
                        "r_squared": float("nan"),
                        "valid": False,
                    }
                continue

            # Extract pre-treatment time and outcome
            t_pre = t_u[pre_sel]
            y_pre = y_u[pre_sel]

            # Center time for numerical stability
            t_mean = t_pre.mean()
            t_pre_centered = t_pre - t_mean

            # Build design matrix [intercept, centered_time]
            X_pre = np.column_stack(
                [
                    np.ones(n_pre, dtype=np.float64),
                    t_pre_centered,
                ]
            )

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]  # [alpha, beta]

            # Check for valid coefficients
            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: detrending produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "pre_n_periods": n_pre,
                        "residual_std": float("nan"),
                        "r_squared": float("nan"),
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit (using same centering)
            t_all_centered = t_u - t_mean
            y_hat = coefs[0] + coefs[1] * t_all_centered

            # Residuals = outcome - fitted trend
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                y_hat_pre = X_pre @ coefs
                residuals_pre = y_pre - y_hat_pre
                residual_std = float(np.std(residuals_pre, ddof=2)) if n_pre > 2 else float("nan")
                ss_res = float(np.sum(residuals_pre**2))
                ss_tot = float(np.sum((y_pre - y_pre.mean()) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                per_unit[uid] = {
                    "alpha": float(coefs[0]),
                    "beta": float(coefs[1]),
                    "pre_n_periods": n_pre,
                    "residual_std": residual_std,
                    "r_squared": r_squared,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            betas = [per_unit[uid]["beta"] for uid in valid_units]
            r2s = [
                per_unit[uid]["r_squared"]
                for uid in valid_units
                if np.isfinite(per_unit[uid]["r_squared"])
            ]
            diagnostics: Dict[str, Any] = {
                "method": "detrend",
                "description": "Y_{it} - (\u03b1\u0302_i + \u03b2\u0302_i * t) based on pre-treatment OLS (Procedure 3.1)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                    "mean_beta": float(np.mean(betas)) if betas else float("nan"),
                    "std_beta": float(np.std(betas)) if betas else float("nan"),
                    "mean_r_squared": float(np.mean(r2s)) if r2s else float("nan"),
                },
            }
            return df, diagnostics

        return df

    def _transform_demeanq(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific seasonal (quarterly) demeaning transformation.

        For each unit, fit Y on [1, Q2, Q3, Q4] dummies using pre-treatment
        periods only, then subtract fitted values from ALL periods.
        Quarter is determined by time_col % 4.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing seasonally-demeaned outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Determine quarter from time column (0-indexed modulo 4 → 1-4).
        # Encoded staggered frames carry the CALENDAR season in
        # _lwdid_season (the time column holds dense positions there, and
        # (pos - 1) % 4 + 1 would relabel seasons after a calendar gap).
        t_series = df[time_col]
        if "_lwdid_season" in df.columns:
            quarters = df["_lwdid_season"].to_numpy()
        elif pd.api.types.is_datetime64_any_dtype(t_series):
            quarters = t_series.dt.quarter.to_numpy()
        elif hasattr(t_series.iloc[0], "quarter"):
            quarters = np.array([v.quarter for v in t_series])
        else:
            t_vals = t_series.to_numpy()
            quarters = (t_vals.astype(np.int64) - 1) % 4 + 1

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            y_u = y_arr[idx_u]
            q_u = quarters[idx_u]
            pre_u = pre_arr[idx_u].astype(bool)

            # Pre-treatment data for this unit
            n_pre = int(pre_u.sum())

            # Need at least as many pre-obs as parameters (intercept + up to 3 dummies)
            q_pre = q_u[pre_u]
            observed_seasons = sorted(np.unique(q_pre))
            n_params = len(observed_seasons)  # = 1 intercept + (n_seasons-1) dummies

            if n_pre < n_params:
                warnings.warn(
                    f"Unit {uid}: demeanq requires at least as many pre-treatment "
                    f"observations as seasonal parameters ({n_params}), "
                    f"found {n_pre}. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "intercept": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Build seasonal dummy design matrix for pre-treatment
            y_pre = y_u[pre_u]

            # Create dummies: drop first category (reference)
            X_pre_parts = [np.ones(n_pre, dtype=np.float64)]
            for s in observed_seasons[1:]:
                X_pre_parts.append((q_pre == s).astype(np.float64))
            X_pre = np.column_stack(X_pre_parts)

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]

            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: demeanq produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "intercept": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Season coverage: a quarter never observed in the unit's
            # pre-period has no estimated effect; predicting it at the
            # reference-season level is a silent out-of-support
            # extrapolation (campaign finding) -> warn + NaN the unit.
            unobserved_seasons = sorted(set(q_u.tolist()) - set(observed_seasons))
            if unobserved_seasons:
                warnings.warn(
                    f"Unit {uid}: demeanq cannot predict quarter(s) "
                    f"{unobserved_seasons} that never appear in the unit's "
                    f"pre-treatment periods. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "intercept": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit
            n_all = len(q_u)
            X_all_parts = [np.ones(n_all, dtype=np.float64)]
            for s in observed_seasons[1:]:
                X_all_parts.append((q_u == s).astype(np.float64))
            X_all = np.column_stack(X_all_parts)
            y_hat = X_all @ coefs

            # Residuals
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                seasonal_effects = {
                    int(s): float(coefs[idx + 1]) for idx, s in enumerate(observed_seasons[1:])
                }
                per_unit[uid] = {
                    "intercept": float(coefs[0]),
                    "seasonal_effects": seasonal_effects,
                    "pre_n_periods": n_pre,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            diagnostics: Dict[str, Any] = {
                "method": "demeanq",
                "description": "Remove unit-specific seasonal (quarterly) fixed effects from pre-treatment",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                },
            }
            return df, diagnostics

        return df

    def _transform_detrendq(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        unit_col: str,
        time_col: str,
        pre_mask: Union[pd.Series, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Apply unit-specific linear detrending with seasonal adjustment.

        For each unit, fit Y on [1, t, Q2, Q3, Q4] using pre-treatment
        periods only, then subtract fitted values from ALL periods.
        Quarter is determined by time_col % 4.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Name of the outcome column.
        unit_col : str
            Name of the unit identifier column.
        time_col : str
            Name of the time period column.
        pre_mask : Series or ndarray of bool
            Boolean mask indicating pre-treatment observations.
        return_diagnostics : bool, default False
            If True, return (df, diagnostics) tuple instead of just df.

        Returns
        -------
        pd.DataFrame or (pd.DataFrame, dict)
            Input data with '_ydot' column containing detrended+seasonally-adjusted outcomes.
            If return_diagnostics=True, also returns diagnostics dict.
        """
        df = df.copy()
        df["_ydot"] = np.nan

        # Determine quarter from time column. Encoded staggered frames
        # carry the CALENDAR season in _lwdid_season (see _transform_demeanq).
        t_series = df[time_col]
        if "_lwdid_season" in df.columns:
            quarters = df["_lwdid_season"].to_numpy()
        elif pd.api.types.is_datetime64_any_dtype(t_series):
            quarters = t_series.dt.quarter.to_numpy()
        elif hasattr(t_series.iloc[0], "quarter"):
            quarters = np.array([v.quarter for v in t_series])
        else:
            t_vals = t_series.to_numpy()
            quarters = (t_vals.astype(np.int64) - 1) % 4 + 1

        # Pre-extract numpy arrays to avoid repeated df.loc[] overhead
        unit_arr = df[unit_col].values
        time_arr = df[time_col].values.astype(np.float64)
        y_arr = df[outcome_col].values.astype(np.float64)
        pre_arr = pre_mask.values if hasattr(pre_mask, "values") else np.asarray(pre_mask)

        units = df[unit_col].unique()
        per_unit: Dict[Any, Dict[str, Any]] = {}
        ydot_out = np.full(len(df), np.nan)

        for uid in units:
            mask_u = unit_arr == uid
            idx_u = np.where(mask_u)[0]
            t_u = time_arr[idx_u]
            y_u = y_arr[idx_u]
            q_u = quarters[idx_u]
            pre_u = pre_arr[idx_u].astype(bool)

            # Pre-treatment data for this unit
            n_pre = int(pre_u.sum())

            if n_pre < 2:
                warnings.warn(
                    f"Unit {uid}: detrendq requires at least 2 "
                    f"pre-treatment periods, found {n_pre}. "
                    f"Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Check seasonal parameters
            q_pre = q_u[pre_u]
            t_pre = t_u[pre_u]
            observed_seasons = sorted(np.unique(q_pre))
            # Parameters: intercept + slope + (n_seasons - 1) dummies
            n_params = 1 + len(observed_seasons)

            y_pre = y_u[pre_u]

            # Center time for numerical stability
            t_mean = t_pre.mean()
            t_pre_centered = t_pre - t_mean

            # Insufficient pre-observations for the seasonal model: fail
            # closed like demeanq (warn + NaN the unit). The pre-fix code
            # silently fit intercept+trend only (per-unit detrend) while
            # the fit still reported rolling='detrendq' - with quarterly
            # data and <= 5 pre-periods EVERY unit fell back, making the
            # whole fit numerically identical to detrend with no trace
            # (campaign finding).
            if n_pre < n_params:
                warnings.warn(
                    f"Unit {uid}: detrendq requires at least as many "
                    f"pre-treatment observations as seasonal parameters "
                    f"({n_params}), found {n_pre}. Transformed outcome set "
                    f"to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Season coverage: quarters unobserved in the pre-period have
            # no estimated effect; predicting them at the reference-season
            # level is silent extrapolation (campaign finding).
            unobserved_seasons = sorted(set(q_u.tolist()) - set(observed_seasons))
            if unobserved_seasons:
                warnings.warn(
                    f"Unit {uid}: detrendq cannot predict quarter(s) "
                    f"{unobserved_seasons} that never appear in the unit's "
                    f"pre-treatment periods. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            use_seasonal = True
            # Build design matrix: [1, t_centered, Q2, Q3, Q4]
            X_pre_parts = [
                np.ones(n_pre, dtype=np.float64),
                t_pre_centered,
            ]
            for s in observed_seasons[1:]:
                X_pre_parts.append((q_pre == s).astype(np.float64))
            X_pre = np.column_stack(X_pre_parts)

            # Solve via scipy.linalg.lstsq
            result = scipy_linalg.lstsq(X_pre, y_pre, cond=None)
            coefs = result[0]

            if not np.all(np.isfinite(coefs)):
                warnings.warn(
                    f"Unit {uid}: detrendq produced non-finite "
                    f"coefficients. Transformed outcome set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                if return_diagnostics:
                    per_unit[uid] = {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "seasonal_effects": {},
                        "pre_n_periods": n_pre,
                        "valid": False,
                    }
                continue

            # Predict on ALL periods for this unit
            t_all_centered = t_u - t_mean
            n_all = len(t_u)

            X_all_parts = [
                np.ones(n_all, dtype=np.float64),
                t_all_centered,
            ]
            if use_seasonal:
                for s in observed_seasons[1:]:
                    X_all_parts.append((q_u == s).astype(np.float64))
            X_all = np.column_stack(X_all_parts)
            y_hat = X_all @ coefs

            # Residuals
            ydot_out[idx_u] = y_u - y_hat

            # Collect diagnostics for this unit
            if return_diagnostics:
                if use_seasonal:
                    seasonal_effects = {
                        int(s): float(coefs[idx + 2]) for idx, s in enumerate(observed_seasons[1:])
                    }
                else:
                    seasonal_effects = {}
                per_unit[uid] = {
                    "alpha": float(coefs[0]),
                    "beta": float(coefs[1]),
                    "seasonal_effects": seasonal_effects,
                    "pre_n_periods": n_pre,
                    "valid": True,
                }

        df["_ydot"] = ydot_out

        if return_diagnostics:
            valid_units = [uid for uid, info in per_unit.items() if info["valid"]]
            n_valid = len(valid_units)
            n_total = len(per_unit)
            diagnostics: Dict[str, Any] = {
                "method": "detrendq",
                "description": "Remove unit-specific trend + seasonal effects (α̂_i + β̂_i*t + Σγ̂_q*Q_q)",
                "per_unit": per_unit,
                "summary": {
                    "n_units_total": n_total,
                    "n_units_valid": n_valid,
                    "n_units_dropped": n_total - n_valid,
                },
            }
            return df, diagnostics

        return df

    def _dispatch_estimator(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        Optional[np.ndarray],
    ]:
        """Dispatch estimation to the appropriate method based on self.estimation_method.

        This is the central routing function that maps the user's estimation-method
        choice to the corresponding implementation. After unit-specific rolling transformation
        converts the panel into a cross-sectional dataset, this method applies the
        chosen treatment-effect estimator to obtain the ATT.

        Corresponds to Step 2 of the Lee & Wooldridge (2025, 2026) procedure:
        after computing Ẏ_{ir} (transformed outcome), apply reg/ipw/dr/psm
        to the cross-section {(Ẏ_{ir}, D_i, X_i)}.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Transformed outcome variable (\u1e8e_{ir} in paper notation).
            This is the post-transformation average residual for each unit.
        treatment : np.ndarray of shape (n,)
            Binary treatment indicator (D_i). 1 = treated, 0 = control.
        controls_matrix : np.ndarray of shape (n, K) or None
            Covariate matrix (X_i). None if no controls specified.
            Used for regression adjustment, propensity score, and matching.
        cluster_ids : np.ndarray of shape (n,) or None
            Cluster identifiers for cluster-robust variance estimation.
            None unless the cluster= constructor parameter is set.
        n_obs : int
            Number of cross-sectional observations (units).

        Returns
        -------
        tuple of (att, se, coefs, vcov, n_params, influence)
            att : float
                Estimated average treatment effect on the treated (\u03c4\u0302 in paper).
            se : float
                Standard error of the ATT estimate.
            coefs : np.ndarray or None
                Full coefficient vector from the regression (RA/IPW paths).
                None for PSM.
            vcov : np.ndarray or None
                Variance-covariance matrix of coefficients.
                None for PSM.
            n_params : int
                Number of parameters in the fitted design, used for the
                residual degrees of freedom: df = N - n_params. For the reg
                path this is design-coherent (LW 2026 Section 2): N - 2
                without controls, N - K - 2 for the plain design and
                N - 2K - 2 when the treatment-covariate interaction is
                active.
            influence : np.ndarray or None
                Observation-aligned influence contributions. Their unit-level
                or cluster-level norm reproduces ``se`` and they can therefore
                be combined across staggered cohort-time cells without assuming
                independence. Matching returns ``None``.

        Raises
        ------
        ValueError
            If self.estimation_method is not in {'reg', 'ipw', 'dr', 'psm'}.
            (Should not occur if __init__ validation passed.)

        Notes
        -----
        Routing logic:
        - 'reg'   → _estimate_reg(): OLS of Ẏ on [1, D, X, D*(X-X̄₁)]
                    per Equation 3.3 in Lee & Wooldridge (2025)
        - 'ipw'   → _estimate_ipw(): Inverse probability weighting via
                    logit propensity score, Hajek-style normalization
        - 'dr'    → _estimate_dr(): Doubly-robust augmented IPW
                    combining outcome model and propensity weighting
        - 'psm'   → _estimate_psm(): Nearest-neighbor propensity score
                    matching (1:n with optional caliper)

        When controls_matrix is None, IPW/DR/PSM fall back to regression
        adjustment (simple difference in means) with a warning.

        The variance family (self.vcov_type) determines which variance
        estimator is used:
        - 'classical': homoskedastic OLS variance
        - 'hc1': HC1 (White) heteroskedasticity-robust
        - 'hc2': HC2 (leverage-adjusted)
        - 'hc3': HC3 (jackknife-style leverage adjustment)
        Cluster-robust (Liang-Zeger CR1) inference activates via the
        cluster= constructor parameter.

        References
        ----------
        Lee, S. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
            to Difference-in-Differences Estimation for Panel Data."
            Procedure 3.1, Equation 3.3.
        Lee, S. J. & Wooldridge, J. M. (2026). "Simple Approaches to
            Inference with Difference-in-Differences Estimators with
            Small Cross-Sectional Sample Sizes." Procedure 2.1.
        """
        if self.estimation_method == "reg":
            return self._estimate_reg(y, treatment, controls_matrix, cluster_ids, n_obs)
        elif self.estimation_method == "ipw":
            return self._estimate_ipw(y, treatment, controls_matrix, cluster_ids, n_obs)
        elif self.estimation_method == "psm":
            return self._estimate_psm(y, treatment, controls_matrix, cluster_ids, n_obs)
        else:  # dr
            return self._estimate_dr(y, treatment, controls_matrix, cluster_ids, n_obs)

    @staticmethod
    def _finalize_influence(
        influence: np.ndarray,
        se: float,
    ) -> Optional[np.ndarray]:
        """Drop influence contributions that cannot support joint inference."""
        if not np.isfinite(se) or se <= 0 or not np.all(np.isfinite(influence)):
            return None
        return influence

    def _ols_treatment_influence(
        self,
        X: np.ndarray,
        xtx_inv: np.ndarray,
        residuals: np.ndarray,
        n_obs: int,
        n_params: int,
        cluster_ids: Optional[np.ndarray],
        coef_index: int = 1,
    ) -> np.ndarray:
        r"""Influence contributions for the OLS treatment coefficient.

        ``coef_index`` is the treatment coefficient's position in the
        (possibly rank-reduced) design actually passed in - the caller
        remaps it when solve_ols dropped columns (fix-wave WS8: computing
        the bread/leverage on the full-width design while only the df
        moved left the influence function describing a min-norm fit).

        The asymptotically linear representation of :math:`\hat\tau` is
        :math:`\psi_i = e_2' (X'X)^{-1} x_i \varepsilon_i`. Each variance
        estimator is a reweighting of those contributions, so applying the
        estimator's own weights here makes the sum of squared contributions
        (summed within clusters when clustering) reproduce the reported
        standard error exactly, while preserving the per-unit structure that
        cross-cell covariance needs. Every branch, including classical,
        keeps the residual-based direction: replacing residuals by their
        homoskedastic magnitude (``sigma * basis``) would fabricate
        covariance between staggered cells that merely share control units.
        """
        basis = X @ xtx_inv[:, coef_index]
        dof = max(n_obs - n_params, 1)
        psi = basis * residuals

        if cluster_ids is not None:
            n_clusters = len(np.unique(cluster_ids))
            if n_clusters > 1:
                cr1 = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / dof)
                return psi * float(np.sqrt(cr1))
            # Degenerate single-cluster fallback that solve_ols resolves
            # to hc1.
            return psi * float(np.sqrt(n_obs / dof))

        if self.vcov_type == "classical":
            # Homoskedastic magnitude: sum_i sigma^2 (x_i' a)^2 = sigma^2
            # (X'X)^{-1}_22, the textbook OLS variance. The contributions are
            # the residual-based psi rescaled to that magnitude, so a single
            # cell reproduces the classical SE exactly while cross-cell
            # products retain the unit-level residual dependence.
            sigma = float(np.sqrt(float(residuals @ residuals) / dof))
            target = sigma * float(np.sqrt(float(basis @ basis)))
            norm = float(np.sqrt(float(psi @ psi)))
            if norm > 0.0 and np.isfinite(norm):
                return psi * (target / norm)
            return psi

        if self.vcov_type in ("hc2", "hc3"):
            raw_leverage = np.sum((X @ xtx_inv) * X, axis=1)
            if self.vcov_type in ("hc2", "hc3") and np.any(raw_leverage >= 1.0 - 1e-8):
                # Match the shared linalg fail-closed contract (round-10
                # review: clipping fabricated a finite HC3 influence
                # vector for a design whose HC3 vcov is NaN, so aggregate
                # inference disagreed with the cell's own).
                return np.full_like(psi, np.nan)
            leverage = np.clip(raw_leverage, 0.0, 1.0 - 1e-10)
            if self.vcov_type == "hc2":
                return psi / np.sqrt(1.0 - leverage)
            return psi / (1.0 - leverage)

        # hc1
        return psi * float(np.sqrt(n_obs / dof))

    def _moment_influence(
        self,
        psi_full: np.ndarray,
        n_obs: int,
        cluster_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Rescale a semiparametric influence function to ATT scale.

        Mirrors the variance formulas used by the IPW/IPWRA paths, so the
        sum of squared contributions reproduces their reported variance.
        """
        if cluster_ids is not None:
            n_clusters = len(np.unique(cluster_ids))
            if n_clusters > 1:
                return psi_full * float(np.sqrt(n_clusters / (n_clusters - 1))) / n_obs
        return (psi_full - float(np.mean(psi_full))) / float(np.sqrt(n_obs * (n_obs - 1)))

    def _estimate_reg(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        Optional[np.ndarray],
    ]:
        """Estimate ATT via regression adjustment (OLS).

        Fits y = alpha + tau*D + X*beta + D*(X - X_bar_1)*gamma + epsilon
        and returns tau as the ATT estimate (LW2025 Equation 3.3).

        The interaction term D*(X - X_bar_1) allows covariate effects to
        differ between treated and control groups. It is only included when
        both N_treated > K+1 and N_control > K+1.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Control variables.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers for cluster-robust SEs.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            Treatment effect coefficient.
        se : float
            Standard error of treatment coefficient.
        coefs : ndarray
            Full coefficient vector.
        vcov : ndarray or None
            Variance-covariance matrix.
        n_params : int
            Number of parameters in the regression.
        """
        # Build design matrix: [intercept, treatment, controls, interaction]
        parts = [np.ones((n_obs, 1)), treatment.reshape(-1, 1)]
        if controls_matrix is not None:
            parts.append(controls_matrix)
            # Add D*(X - X_bar_1) interaction term when sample sizes permit
            # (LW2025 Eq 3.3: requires N_0 > K+1 and N_1 > K+1). K is the
            # IDENTIFIED control dimension (round-11 review: the nominal
            # column count let a perfectly collinear control flip the gate
            # off and silently change the ATT while adding no information).
            K = int(
                _detect_rank_deficiency(np.column_stack([np.ones(n_obs), controls_matrix]))[0] - 1
            )
            treated_mask = treatment == 1
            n_treated = int(treated_mask.sum())
            n_control = n_obs - n_treated
            if n_treated > K + 1 and n_control > K + 1:
                X_bar_1 = controls_matrix[treated_mask].mean(axis=0)
                interaction = treatment.reshape(-1, 1) * (controls_matrix - X_bar_1)
                parts.append(interaction)
        X = np.hstack(parts)
        # EFFECTIVE design rank on the column-equilibrated matrix
        # (round-13 review: the nominal width rejected designs whose
        # redundant columns the rank-aware solver drops, e.g. N=4 with
        # [1, D, x, 2x] has rank 3 and one residual df; equilibration
        # keeps the rank decision scale-invariant).
        # Round-17 review: matrix_rank's looser default tolerance
        # disagreed with solve_ols's scale-invariant pivoted-QR 1e-7
        # convention on near-collinear columns, so the gate and the fit
        # could select different designs - use the SHARED detector.
        rank_eff = int(_detect_rank_deficiency(X)[0])
        if X.shape[0] < 3 or X.shape[0] - rank_eff <= 0:
            # Registry small-sample guards (N >= 3; positive residual df,
            # i.e. N > K + 2 with controls / N > 2K + 2 interacted): the
            # shared classical vcov divides by n - k, so an exactly-
            # saturated design reached ZeroDivisionError (review finding).
            raise ValueError(
                f"Invalid exact-inference design: {X.shape[0]} "
                f"observation(s) with {rank_eff} identified parameter(s). "
                f"LWDiD requires at least 3 cross-sectional units and a "
                f"positive residual df (N > K + 2 with controls)."
            )

        # Determine vcov_type for solve_ols (hc3 routes through the shared
        # linalg backend; clustered fits resolve to CR1 via cluster_ids)
        vcov_type = self._resolve_vcov_type()

        # Call solve_ols
        coefs, residuals, vcov = solve_ols(
            X,
            y,
            cluster_ids=cluster_ids,
            return_vcov=True,
            vcov_type=vcov_type,
        )

        # ATT = coefficient on treatment (index 1)
        att = float(coefs[1])
        # SE from vcov diagonal
        if vcov is not None and np.isfinite(vcov[1, 1]):
            se = float(np.sqrt(max(vcov[1, 1], 0.0)))
        else:
            se = np.nan

        # Return the fitted design's EFFECTIVE parameter count so callers
        # compute a design-coherent residual df: N - 2 without controls,
        # N - K - 2 for the plain design (1, D, X), N - 2K - 2 when the
        # interaction D*(X - X_bar_1) is active (LW 2026 Section 2) - and,
        # under rank deficiency, the KEPT-column count (fix-wave WS8: the
        # nominal count understated the df and the full-width pinv bread
        # broke the IF == solve_ols SE identity).
        nan_mask = np.isnan(coefs)
        n_params_effective = int(np.sum(~nan_mask))
        if nan_mask.any():
            if nan_mask[1]:
                # The treatment column itself was pivoted out: the ATT is
                # unidentified (solve_ols already emitted the rank warning).
                return np.nan, np.nan, coefs, vcov, n_params_effective, None
            kept = np.flatnonzero(~nan_mask)
            X_used = X[:, kept]
            coef_index = int(np.flatnonzero(kept == 1)[0])
        else:
            X_used = X
            coef_index = 1
        # Scale-equilibrated bread (round-13 review: the raw-Gram pinv
        # silently dropped low-scale directions at large covariate units -
        # cell coefficients/SEs from solve_ols were invariant while the
        # reconstructed influence, and therefore every aggregate SE and
        # multiplier-bootstrap input, was not). With column scales D,
        # (X'X)^{-1} = D^{-1} (Xs'Xs)^{-1} D^{-1} for Xs = X D^{-1}.
        used_scales = np.linalg.norm(X_used, axis=0)
        used_scales[used_scales == 0] = 1.0
        X_scaled = X_used / used_scales
        xtx_inv = np.linalg.pinv(X_scaled.T @ X_scaled) / np.outer(used_scales, used_scales)
        if self.vcov_type == "hc2" and cluster_ids is None:
            # Round-21 review: the shared hc2 kernel keeps its RELEASED
            # 1 - h floor (tracked separately), but the NEW LWDiD surface
            # must not report a fabricated finite variance for a
            # perfectly-leveraged design - fail closed HERE, mirroring
            # hc3 (point retained, inference NaN).
            leverage_used = np.sum((X_used @ xtx_inv) * X_used, axis=1)
            if np.any(leverage_used >= 1.0 - 1e-8):
                n_lev1 = int(np.sum(leverage_used >= 1.0 - 1e-8))
                warnings.warn(
                    f"HC2 variance is undefined for this design: {n_lev1} "
                    f"observation(s) have hat-matrix leverage ~1 (e.g. a "
                    f"single treated unit). Returning NaN inference (point "
                    f"retained); use vcov_type='classical' exact inference "
                    f"or add treated units.",
                    UserWarning,
                    stacklevel=2,
                )
                se = np.nan
        influence = self._finalize_influence(
            self._ols_treatment_influence(
                X_used,
                xtx_inv,
                residuals,
                n_obs,
                n_params_effective,
                cluster_ids,
                coef_index=coef_index,
            ),
            se,
        )
        return att, se, coefs, vcov, n_params_effective, influence

    def _estimate_ipw(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        Optional[np.ndarray],
    ]:
        """Estimate ATT via inverse probability weighting.

        Uses propensity scores to reweight control observations.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates for propensity score model.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            IPW-estimated ATT.
        se : float
            Standard error.
        coefs : ndarray or None
            Not returned for IPW (None).
        vcov : ndarray or None
            Not returned for IPW (None).
        n_params : int
            Number of parameters in the underlying regression.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Without covariates, IPW reduces to simple difference
            # in means (propensity score is constant)
            warnings.warn(
                "IPW without control variables reduces to a simple "
                "difference in means. Consider using estimation_method='reg'.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_reg(
                y, treatment, None, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        # Step 1: Estimate propensity score via logit
        # solve_logit adds intercept automatically
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Rank/convergence handling (round-11 review): the shared solver
        # marks DROPPED collinear columns with NaN coefficients while the
        # fitted probabilities remain valid - that reduced-rank fit stays
        # an IPW fit (the pre-fix code silently substituted regression
        # adjustment under ipw provenance). Only a genuinely failed solve
        # (non-finite probabilities) falls back.
        kept_ps = np.isfinite(coefs_logit)
        if not kept_ps.all():
            if np.all(np.isfinite(probs)):
                warnings.warn(
                    f"Propensity model is rank-deficient: "
                    f"{int((~kept_ps).sum())} collinear column(s) dropped; "
                    f"continuing IPW with the reduced-rank propensity fit.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Logistic regression did not converge (non-finite "
                    "probabilities). Falling back to 'reg' estimation. "
                    "Consider standardizing controls.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._estimate_reg(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Step 2: Trim propensity scores to [pscore_trim, 1 - pscore_trim]
        trim_lo, trim_hi = self.pscore_trim, 1.0 - self.pscore_trim
        n_trimmed = int((probs < trim_lo).sum() + (probs > trim_hi).sum())
        if n_trimmed > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed} observation(s) had propensity scores trimmed "
                f"to [{self.pscore_trim:.3f}, {1-self.pscore_trim:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs_raw = probs
        probs = np.clip(probs_raw, trim_lo, trim_hi)
        # Observations at the clip boundary have ZERO weight-derivative in
        # gamma (round-11 review: using clipped probabilities in the logit
        # score/Hessian broke the estimating-equation linearization - the
        # score at the MLE is ~0 in the RAW fitted probabilities only).
        unclipped = (probs_raw > trim_lo) & (probs_raw < trim_hi)

        # Step 3: Compute IPW weights
        # For treated: weight = 1
        # For control: weight = p(x) / (1 - p(x))
        # Normalized so control weights sum to n_treated
        ipw_weights = np.where(
            treatment == 1,
            1.0,
            probs / (1.0 - probs),
        )

        # Normalize weights: treated get weight 1/n_treated,
        # control weights normalized to sum to 1
        treat_mask = treatment == 1
        ctrl_mask = treatment == 0

        w_ctrl_sum = ipw_weights[ctrl_mask].sum()
        if w_ctrl_sum <= 0:
            warnings.warn(
                "IPW control weights sum to zero. Falling back to " "unweighted 'reg' estimation.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_reg(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Hajek-style ATT estimator
        att_treated = y[treat_mask].mean()
        att_control = np.sum(ipw_weights[ctrl_mask] * y[ctrl_mask]) / w_ctrl_sum
        att = float(att_treated - att_control)

        # Step 4: Compute SE via semiparametric influence function
        # (Lunceford & Davidian 2004 AIPW form - the documented,
        # adjudicated alternative to the papers'/Stata package's stacked
        # E.3/E.4 form; see the REGISTRY IPWRA-variance Note).
        # The full IF consists of the Hajek main term plus a propensity score
        # estimation uncertainty correction.
        n_treated_f = float(treat_mask.sum())
        p_bar = n_treated_f / n_obs  # P(D=1) estimate

        # --- Hajek influence function (main term) ---
        w_ctrl = ipw_weights[ctrl_mask]  # p/(1-p) for controls

        psi_ht = np.zeros(n_obs)
        psi_ht[treat_mask] = (y[treat_mask] - att_treated) / p_bar
        psi_ht[ctrl_mask] = -w_ctrl * (y[ctrl_mask] - att_control) / p_bar

        # --- Propensity score estimation uncertainty correction ---
        # Design matrix with intercept (solve_logit adds intercept internally,
        # so we reconstruct it here for the IF computation), restricted to
        # the KEPT (identified) propensity columns under rank deficiency.
        X_ps = np.column_stack([np.ones(n_obs), controls_matrix])[:, kept_ps]

        # Logit score: S_i = (D_i - p_i) * X_i, at the RAW fitted
        # probabilities (the score of the actual MLE; clipped probabilities
        # are a weighting choice, not the estimating equation).
        S_gamma = (treatment - probs_raw)[:, np.newaxis] * X_ps

        # Logit Hessian: H = -(1/n) * X' diag(p*(1-p)) X (raw fit)
        W_ps = probs_raw * (1 - probs_raw)
        H_gamma = -(X_ps.T * W_ps) @ X_ps / n_obs
        try:
            H_gamma_inv = np.linalg.inv(H_gamma)
        except np.linalg.LinAlgError:
            H_gamma_inv = np.linalg.pinv(H_gamma)

        # Sensitivity: dATT/dgamma
        # dw/dgamma_i = w_i * X_i (logit chain rule) for UNCLIPPED
        # observations; a clipped weight is locally constant in gamma.
        # dATT/dgamma = -(1/w_sum) * sum_ctrl(w_i * X_i * (Y_i - mu_0))
        # The (Y_i - mu_0) centering comes from the quotient rule for the
        # Hajek estimator (d/dgamma of Sigma(wY)/Sigma(w)) and ensures
        # translation invariance of the resulting SE.
        dw_dgamma_ctrl = (w_ctrl * unclipped[ctrl_mask])[:, np.newaxis] * X_ps[ctrl_mask]
        Y_ctrl_centered = y[ctrl_mask] - att_control
        dATT_dgamma = -(dw_dgamma_ctrl * Y_ctrl_centered[:, np.newaxis]).sum(axis=0) / (
            n_obs * p_bar
        )

        # PS adjustment: psi_adj_i = (S_i @ H^{-1}) @ dATT_dgamma
        ps_adjustment = (S_gamma @ H_gamma_inv.T) @ dATT_dgamma

        # Full IF = main term - PS correction
        psi_full = psi_ht - ps_adjustment

        # --- Variance estimation ---
        if cluster_ids is not None:
            cluster_df = pd.DataFrame({"psi": psi_full, "cluster": cluster_ids})
            cluster_sums = cluster_df.groupby("cluster")["psi"].sum().values
            n_clusters = len(cluster_sums)
            if n_clusters <= 1:
                warnings.warn(
                    "Only 1 cluster found; falling back to non-clustered "
                    "variance for IPW influence function.",
                    UserWarning,
                    stacklevel=2,
                )
                var_att = float(np.var(psi_full, ddof=1) / n_obs)
            else:
                var_att = float(
                    (n_clusters / (n_clusters - 1)) * np.sum(cluster_sums**2) / n_obs**2
                )
        else:
            var_att = float(np.var(psi_full, ddof=1) / n_obs)

        se = float(np.sqrt(max(var_att, 0.0)))

        # n_params: IDENTIFIED propensity-model rank (round-12 review:
        # the nominal count let a redundant control shrink residual df).
        n_params = int(kept_ps.sum())
        influence = self._finalize_influence(
            self._moment_influence(psi_full, n_obs, cluster_ids), se
        )
        return att, se, None, None, n_params, influence

    def _estimate_psm(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        Optional[np.ndarray],
    ]:
        """Estimate ATT via propensity score matching.

        For each treated unit, find the nearest control unit(s) by
        propensity score (1:n_neighbors nearest-neighbor matching, 1:1 by
        default, with replacement by default), then compute ATT as the
        average difference between treated and matched controls.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates for propensity score model.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            PSM-estimated ATT.
        se : float
            Always NaN (fail-closed): no valid matching variance is
            implemented (the naive matched-pairs formula ignored control
            reuse and first-stage uncertainty; an Abadie-Imbens variance
            is tracked in DEFERRED.md).
        coefs : ndarray or None
            Not returned for PSM (None).
        vcov : ndarray or None
            Not returned for PSM (None).
        n_params : int
            Effective number of parameters.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Unreachable from fit() (config guard rejects covariate-less
            # PSM); kept as defense in depth for direct callers.
            raise ValueError(
                "estimation_method='psm' requires covariates: without them "
                "there is no propensity score to match on. Use "
                "estimation_method='reg', or supply covariates."
            )

        treat_mask = treatment == 1
        ctrl_mask = treatment == 0
        n_treated = int(treat_mask.sum())
        n_control = int(ctrl_mask.sum())

        if n_treated == 0 or n_control == 0:
            warnings.warn(
                "PSM estimation failed: no treated or no control units available. "
                "Returning NaN results.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None, None, 2, None

        # Step 1: Estimate propensity score via logit
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Rank/convergence handling (round-19 review; mirrors ipw/dr):
        # NaN coefficients with FINITE probabilities are a reduced-rank
        # propensity fit - matching needs only the probabilities, so PSM
        # continues (the pre-fix path substituted a regression-adjustment
        # point under psm provenance). Only genuinely failed solves
        # (non-finite probabilities) fall back, fail-closed.
        kept_ps_match = np.isfinite(coefs_logit)
        if not kept_ps_match.all():
            if np.all(np.isfinite(probs)):
                warnings.warn(
                    f"Propensity model is rank-deficient: "
                    f"{int((~kept_ps_match).sum())} collinear column(s) "
                    f"dropped; continuing PSM with the reduced-rank "
                    f"propensity fit.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # Review round 3: the pre-fix fallback returned the
                # regression point WITH its finite OLS inference while the
                # results metadata still said 'psm'. Point retained,
                # inference NaN.
                warnings.warn(
                    "Logistic regression did not converge (non-finite "
                    "probabilities); the point estimate falls back to "
                    "regression adjustment, and inference is NaN under the "
                    "PSM fail-closed contract. Consider standardizing "
                    "controls or using estimation_method='reg'.",
                    UserWarning,
                    stacklevel=2,
                )
                att_fb, _, _, _, n_params_fb, _ = self._estimate_reg(
                    y, treatment, controls_matrix, cluster_ids, n_obs
                )
                return att_fb, np.nan, None, None, n_params_fb, None

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Step 2: Trim propensity scores to [pscore_trim, 1 - pscore_trim]
        trim_lo, trim_hi = self.pscore_trim, 1.0 - self.pscore_trim
        n_trimmed = int((probs < trim_lo).sum() + (probs > trim_hi).sum())
        if n_trimmed > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed} observation(s) had propensity scores trimmed "
                f"to [{self.pscore_trim:.3f}, {1-self.pscore_trim:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs = np.clip(probs, trim_lo, trim_hi)

        # Step 3: Nearest-neighbor matching (with replacement)
        p_treated = probs[treat_mask]
        p_control = probs[ctrl_mask]
        y_treated = y[treat_mask]
        y_control = y[ctrl_mask]

        # For each treated unit, find n_neighbors nearest controls
        matched_y_control = np.empty(n_treated)
        available_mask = np.ones(n_control, dtype=bool)
        n_partial_matches = 0

        for i in range(n_treated):
            valid_control_idx = np.where(available_mask)[0]
            if len(valid_control_idx) == 0:
                matched_y_control[i] = np.nan
                continue

            distances = np.abs(p_treated[i] - p_control[valid_control_idx])

            if self.caliper is not None:
                within_caliper = distances <= self.caliper
                if not within_caliper.any():
                    matched_y_control[i] = np.nan
                    continue
                distances = np.where(within_caliper, distances, np.inf)

            nearest_local = np.argsort(distances)[: self.n_neighbors]
            # Caliper contract: only within-caliper controls may be
            # averaged. argsort places np.inf (out-of-caliper) LAST but
            # still returns it, so a partial shortfall (>=1 but
            # < n_neighbors controls inside the caliper) used to average
            # arbitrarily distant controls into the counterfactual
            # (campaign finding: deterministic ATT of -49 vs the correct
            # caliper-respecting 1.0 on the repro fixture).
            nearest_local = nearest_local[np.isfinite(distances[nearest_local])]
            if len(nearest_local) < self.n_neighbors:
                n_partial_matches += 1
            nearest_global = valid_control_idx[nearest_local]
            matched_y_control[i] = y_control[nearest_global].mean()

            if not self.with_replacement:
                available_mask[nearest_global] = False

        # Step 4: Compute ATT = mean(Y_treated - Y_matched_control)
        # Exclude NaN matches (from caliper)
        valid_matches = np.isfinite(matched_y_control)
        n_unmatched = int(np.isnan(matched_y_control).sum())
        if n_unmatched > 0:
            warnings.warn(
                f"LWDiD PSM: {n_unmatched} treated unit(s) could not be matched "
                f"within caliper={self.caliper}. ATT computed from {n_treated - n_unmatched} matches.",
                UserWarning,
                stacklevel=2,
            )
        if n_partial_matches > 0:
            warnings.warn(
                f"LWDiD PSM: {n_partial_matches} treated unit(s) had fewer "
                f"than n_neighbors={self.n_neighbors} control(s) within "
                f"caliper={self.caliper}; their matches average the "
                f"within-caliper control(s) only.",
                UserWarning,
                stacklevel=2,
            )
        if not valid_matches.any():
            warnings.warn(
                "PSM estimation failed: no valid matches found (all exceeded caliper). "
                "Returning NaN results.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None, None, 2, None
        diffs = y_treated[valid_matches] - matched_y_control[valid_matches]
        att = float(np.mean(diffs))

        # Step 5: Inference fails closed (review finding). The former
        # sqrt(var(diffs)/n) treated matched differences as INDEPENDENT -
        # with replacement matching a control can appear in many treated
        # counterfactuals, so their common uncertainty cancels out of that
        # formula - and it omits the propensity/matching first-stage
        # uncertainty entirely. A valid matching variance (Abadie-Imbens)
        # is tracked in DEFERRED.md; until it lands the point is retained
        # and the inference tuple is NaN (same convention as the staggered
        # 'unavailable_matching' basis).
        warnings.warn(
            "LWDiD PSM: no valid matching variance estimator is implemented "
            "(the naive var(diffs)/n formula ignores matched-control reuse "
            "and first-stage matching uncertainty). The ATT point estimate "
            "is reported with NaN inference; use estimation_method='dr' for "
            "a doubly robust alternative with valid inference.",
            UserWarning,
            stacklevel=2,
        )
        se = np.nan

        # Effective n_params: intercept + controls (for propensity model)
        n_params = 1 + controls_matrix.shape[1]
        return att, se, None, None, n_params, None

    def _estimate_dr(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        controls_matrix: Optional[np.ndarray],
        cluster_ids: Optional[np.ndarray],
        n_obs: int,
    ) -> Tuple[
        float,
        float,
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        Optional[np.ndarray],
    ]:
        """Estimate ATT via augmented IPW (doubly robust).

        Combines regression adjustment with inverse probability weighting
        for double robustness.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Transformed outcome.
        treatment : ndarray of shape (n,)
            Binary treatment indicator.
        controls_matrix : ndarray of shape (n, p) or None
            Covariates.
        cluster_ids : ndarray of shape (n,) or None
            Cluster identifiers.
        n_obs : int
            Number of observations.

        Returns
        -------
        att : float
            Doubly-robust ATT estimate.
        se : float
            Standard error.
        coefs : ndarray or None
            Not returned for DR (None).
        vcov : ndarray or None
            Not returned for DR (None).
        n_params : int
            Effective number of parameters for df computation.

        Notes
        -----
        **Variance form (paper mapping).** The reported SE and the influence
        function consumed by the multiplier bootstrap use the AIPW efficient
        influence function (Lunceford & Davidian 2004) — NOT the stacked
        M-estimator form of Lee & Wooldridge (2026) Appendix E.3 that the
        authors' Stata package implements. This is a documented,
        independently anchored alternative, adjudicated in PR #588's final
        round: the AIPW EIF is the standard doubly-robust influence function
        in the causal-inference literature and is anchored by the RA config's
        bootstrap-SE parity gate against the Stata golden plus the suite's
        analytical/bootstrap cross-path pins. Measured on the Walmart
        application (2026-08-16): DR point estimates agree with the authors'
        package to ~1e-3, while DR (IPWRA) multiplier-bootstrap SEs diverge
        systematically by ~15%; the RA config's SEs agree within Monte-Carlo
        bounds. See the LWDiD IPWRA-variance note in
        ``docs/methodology/REGISTRY.md``; implementing the E.3 stacked form
        remains an available follow-up if package-form SE parity is preferred.
        """
        if controls_matrix is None or controls_matrix.shape[1] == 0:
            # Without covariates, DR reduces to regression adjustment.
            # Say so, matching the routing docstring and the ipw branch
            # (no-silent-failures contract).
            warnings.warn(
                "DR (doubly robust) without control variables reduces to "
                "regression adjustment. Consider using "
                "estimation_method='reg'.",
                UserWarning,
                stacklevel=2,
            )
            return self._estimate_reg(
                y, treatment, None, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        treat_mask = treatment == 1
        ctrl_mask = treatment == 0
        n_treated = int(treat_mask.sum())
        n_control = int(ctrl_mask.sum())

        # Step 1: Get propensity scores
        coefs_logit, probs = solve_logit(controls_matrix, treatment)

        # Rank/convergence handling (round-11 review; mirrors _estimate_ipw):
        # NaN coefficients with finite probabilities = a reduced-rank
        # propensity fit that remains a DR fit; only non-finite
        # probabilities fall back to regression adjustment.
        kept_ps = np.isfinite(coefs_logit)
        if not kept_ps.all():
            if np.all(np.isfinite(probs)):
                warnings.warn(
                    f"Propensity model is rank-deficient: "
                    f"{int((~kept_ps).sum())} collinear column(s) dropped; "
                    f"continuing DR with the reduced-rank propensity fit.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Logistic regression did not converge (non-finite "
                    "probabilities). Falling back to 'reg' estimation. "
                    "Consider standardizing controls.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._estimate_reg(y, treatment, controls_matrix, cluster_ids, n_obs)

        # Convergence check: complete/quasi-complete separation
        if np.any(probs < 1e-8) or np.any(probs > 1 - 1e-8):
            warnings.warn(
                "Possible complete separation detected in propensity score model. "
                "Some predicted probabilities are near 0 or 1. "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        trim_lo_dr, trim_hi_dr = self.pscore_trim, 1.0 - self.pscore_trim
        n_trimmed_dr = int((probs < trim_lo_dr).sum() + (probs > trim_hi_dr).sum())
        if n_trimmed_dr > 0:
            warnings.warn(
                f"LWDiD: {n_trimmed_dr} observation(s) had propensity scores trimmed "
                f"to [{self.pscore_trim:.3f}, {1-self.pscore_trim:.3f}].",
                UserWarning,
                stacklevel=2,
            )
        probs_raw = probs
        probs = np.clip(probs_raw, self.pscore_trim, 1.0 - self.pscore_trim)
        # Zero weight-derivative at the clip boundary; score/Hessian use
        # the RAW fitted probabilities (round-11 review; see _estimate_ipw).
        unclipped = (probs_raw > trim_lo_dr) & (probs_raw < trim_hi_dr)

        # Step 2: Fit outcome model on control units only using WLS with IPW weights
        # This matches the Stata/lwdid-py reference: outcome model is fitted on
        # controls with weights w_i = p(X_i)/(1-p(X_i)) to target ATT.
        X_ctrl = np.column_stack([np.ones(n_control), controls_matrix[ctrl_mask]])
        y_ctrl = y[ctrl_mask]

        # IPW weights for control units
        ipw_ctrl = probs[ctrl_mask] / (1.0 - probs[ctrl_mask])
        ipw_ctrl_sum = ipw_ctrl.sum()

        if ipw_ctrl_sum <= 0:
            # Fall back to RA if IPW weights degenerate
            return self._estimate_reg(
                y, treatment, controls_matrix, cluster_ids, n_obs
            )  # returns 5-tuple including n_params

        # WLS via sqrt(w) transformation through the shared RANK-AWARE
        # solver (round-12 review: the raw inv/pinv Gram on the nominal
        # columns was not scale-invariant - an exactly redundant
        # 1e12-rescaled duplicate changed the DR SE by ~2.5x). Dropped
        # collinear columns get NaN coefficients; the identified mask is
        # reused for prediction and every outcome-model IF term.
        sqrt_w = np.sqrt(ipw_ctrl)
        X_ctrl_w = X_ctrl * sqrt_w[:, np.newaxis]
        y_ctrl_w = y_ctrl * sqrt_w
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # rank warning surfaced below
            coefs_full, _, _ = solve_ols(X_ctrl_w, y_ctrl_w)
        kept_om = np.isfinite(coefs_full)
        if not kept_om.all():
            warnings.warn(
                f"DR outcome model is rank-deficient: "
                f"{int((~kept_om).sum())} collinear column(s) dropped; "
                f"continuing with the identified outcome design.",
                UserWarning,
                stacklevel=2,
            )
        coefs_outcome = coefs_full[kept_om]

        # Predict counterfactual for all units (identified columns only)
        X_all = np.column_stack([np.ones(n_obs), controls_matrix])[:, kept_om]
        mu_0 = X_all @ coefs_outcome

        # Step 3: Compute AIPW/IPWRA estimator (Hajek normalization)
        # ATT = mean_{D=1}(Y - mu_0) - sum_{D=0}[w*(Y-mu_0)] / sum_{D=0}(w)
        resid = y - mu_0
        resid_ctrl = resid[ctrl_mask]

        # Treated component
        att_treated_part = resid[treat_mask].mean()

        # Control component (Hajek: divide by sum of weights)
        weights_sum = ipw_ctrl_sum
        att_ctrl_part = np.sum(ipw_ctrl * resid_ctrl) / weights_sum

        att = float(att_treated_part - att_ctrl_part)

        # Step 4: Compute SE via full semiparametric influence function
        # The IPWRA IF consists of 3 components (Cattaneo 2010, Lunceford & Davidian 2004):
        # 1. Hajek main term (plug-in IF)
        # 2. Propensity score estimation uncertainty correction
        # 3. Outcome model estimation uncertainty correction
        n_treated_f = float(n_treated)
        p_bar = n_treated_f / n_obs  # P(D=1) estimate

        # Control term (Hajek weighted mean of control residuals)
        control_term = att_ctrl_part  # = sum(w*resid_C) / sum(w)

        # ================================================================
        # Component 1: Hajek influence function (main term)
        # Hajek linearization for ATT = mean_T(resid) - sum_C(w*resid)/sum_C(w)
        # ================================================================
        psi = np.zeros(n_obs)
        psi[treat_mask] = (resid[treat_mask] - att) / p_bar
        psi[ctrl_mask] = -ipw_ctrl * (resid_ctrl - control_term) / weights_sum * n_obs

        # ================================================================
        # Component 2: Propensity score estimation uncertainty correction
        # S_gamma_i = (D_i - p_i) * X_i (logit score)
        # H_gamma = -(1/n) * X' diag(p*(1-p)) X (logit Hessian)
        # dATT/dgamma = -sum_C[dw/dgamma * (resid - B)] / sum_C(w)
        # ================================================================
        X_ps = np.column_stack([np.ones(n_obs), controls_matrix])[:, kept_ps]

        # Logit score (RAW fitted probabilities - the actual MLE's
        # estimating equation; clipping is a weighting choice)
        S_gamma = (treatment - probs_raw)[:, np.newaxis] * X_ps

        # Logit Hessian (raw fit)
        W_ps = probs_raw * (1 - probs_raw)
        H_gamma = -(X_ps.T * W_ps) @ X_ps / n_obs
        try:
            H_gamma_inv = np.linalg.inv(H_gamma)
        except np.linalg.LinAlgError:
            H_gamma_inv = np.linalg.pinv(H_gamma)

        # Sensitivity of ATT to propensity score parameters
        # dw/dgamma_i = w_i * X_i for UNCLIPPED observations (a clipped
        # weight is locally constant in gamma); chain through the Hajek
        # control term
        r_minus_B = resid_ctrl - control_term
        dw_dgamma_ctrl = (ipw_ctrl * unclipped[ctrl_mask])[:, np.newaxis] * X_ps[ctrl_mask]
        dATT_dgamma = -(dw_dgamma_ctrl * r_minus_B[:, np.newaxis]).sum(axis=0) / weights_sum

        # PS adjustment
        ps_adjustment = (S_gamma @ H_gamma_inv.T) @ dATT_dgamma

        # ================================================================
        # Component 3: Outcome model estimation uncertainty correction
        # The outcome model is WLS fitted on controls with IPW weights:
        #   E[Y|X, D=0] fitted by WLS with w_i = p/(1-p).
        # S_beta_i = w_i * resid_i * X_i * I(D_i=0) (WLS score)
        # H_beta = -(1/n) * X_ctrl' diag(w) X_ctrl (WLS Hessian)
        # dATT/dbeta = -mean_T(X_i) + sum_C(w_i*X_i) / sum_C(w)
        # ================================================================
        X_om = np.column_stack([np.ones(n_obs), controls_matrix])[:, kept_om]
        X_ctrl_om = X_om[ctrl_mask]

        # WLS score (nonzero only for control units)
        S_beta = np.zeros((n_obs, X_om.shape[1]))
        S_beta[ctrl_mask] = ipw_ctrl[:, np.newaxis] * resid_ctrl[:, np.newaxis] * X_ctrl_om

        # WLS Hessian: H_beta = -(1/n) * X_ctrl' diag(w) X_ctrl
        H_beta = -(X_ctrl_om.T * ipw_ctrl) @ X_ctrl_om / n_obs
        try:
            H_beta_inv = np.linalg.inv(H_beta)
        except np.linalg.LinAlgError:
            H_beta_inv = np.linalg.pinv(H_beta)

        # Sensitivity of ATT to outcome model parameters
        # dATT/dbeta = -mean_T(X_i) + weighted_mean_C(X_i)
        X_bar_treated = X_om[treat_mask].mean(axis=0)
        X_bar_ctrl_w = (ipw_ctrl[:, np.newaxis] * X_ctrl_om).sum(axis=0) / weights_sum
        dATT_dbeta = -X_bar_treated + X_bar_ctrl_w

        # Outcome model adjustment
        om_adjustment = (S_beta @ H_beta_inv.T) @ dATT_dbeta

        # ================================================================
        # Combine: full IF = main - PS correction - outcome correction
        # ================================================================
        psi_full = psi - ps_adjustment - om_adjustment

        # --- Variance estimation ---
        if cluster_ids is not None:
            # Cluster-robust: sum phi within clusters, then outer product
            cluster_df = pd.DataFrame({"psi": psi_full, "cluster": cluster_ids})
            cluster_sums = cluster_df.groupby("cluster")["psi"].sum().values
            n_clusters = len(cluster_sums)
            if n_clusters <= 1:
                warnings.warn(
                    "Only 1 cluster found; falling back to non-clustered "
                    "variance for DR influence function.",
                    UserWarning,
                    stacklevel=2,
                )
                var_att = float(np.var(psi_full, ddof=1) / n_obs)
            else:
                var_att = float(
                    (n_clusters / (n_clusters - 1)) * np.sum(cluster_sums**2) / n_obs**2
                )
        else:
            var_att = float(np.var(psi_full, ddof=1) / n_obs)

        se = float(np.sqrt(max(var_att, 0.0)))

        # Effective n_params: treatment dimension + the IDENTIFIED
        # outcome-model rank (round-12 review: the nominal count let an
        # exactly redundant control shrink residual df and move
        # p-values/CIs while ATT and SE were unchanged).
        n_params = 1 + int(kept_om.sum())
        influence = self._finalize_influence(
            self._moment_influence(psi_full, n_obs, cluster_ids), se
        )
        return att, se, None, None, n_params, influence

    @staticmethod
    def _validate_vcov_config(vcov_type, estimation_method, cluster) -> None:
        """Config-only vcov coherence checks (called from __init__ AND fit).

        Accepted sets (campaign finding: vcov_type was silently inert for
        ipw/dr/psm - the influence-function / matching variance is always
        used there, so only the value whose behavior is real is accepted):

        - ``reg``: {classical, hc1, hc2, hc3}
        - ``ipw`` / ``dr``: {hc1} only (the default; implemented as the
          heteroskedasticity-robust influence-function sandwich)
        - ``psm``: {hc1} only, and ``cluster=`` is rejected (the matching
          SE has no clustered form - pre-fix it presented a non-clustered
          SE under cluster-robust metadata)

        ``cluster=`` composes ONLY with hc1 (for any method): pre-fix,
        classical/hc2/hc3 + cluster were silently remapped to CR1 while
        the results object kept the requested label.
        """
        if estimation_method in ("ipw", "dr") and vcov_type != "hc1":
            raise ValueError(
                f"estimation_method='{estimation_method}' supports "
                f"vcov_type='hc1' only (the influence-function sandwich; "
                f"heteroskedasticity-robust by construction). Got "
                f"vcov_type='{vcov_type}', which would be silently inert."
            )
        if estimation_method == "psm":
            if vcov_type != "hc1":
                raise ValueError(
                    f"estimation_method='psm' accepts vcov_type='hc1' only "
                    f"(the accepted configuration; matching inference is "
                    f"currently unavailable and reported as NaN - see "
                    f"DEFERRED.md). Got vcov_type='{vcov_type}', which would "
                    f"be silently inert."
                )
            if cluster is not None:
                raise ValueError(
                    "estimation_method='psm' does not support cluster=: the "
                    "matching SE has no cluster-robust form. Use "
                    "estimation_method='dr' for a doubly robust alternative "
                    "with clustered inference."
                )
        if cluster is not None and vcov_type not in ("hc1",):
            raise ValueError(
                f"cluster= composes only with vcov_type='hc1' (CR1); got "
                f"vcov_type='{vcov_type}'. Cluster-robust leverage-corrected "
                f"families are not implemented for LWDiD."
            )

    def _resolve_vcov_type(self) -> str:
        """Map the requested variance family to a solve_ols vcov_type.

        Returns
        -------
        str
            The vcov_type string compatible with solve_ols. When the
            cluster= constructor parameter is set, cluster-robust (CR1)
            inference is requested via hc1 plus cluster_ids.
        """
        # Post fix-wave WS6, the config validator guarantees cluster only
        # composes with hc1, so the requested family IS the resolved family
        # on every path (no silent remap can occur).
        if self.cluster is not None:
            assert self.vcov_type == "hc1", "validator invariant violated"
        return self.vcov_type

    def _bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treatment: str,
        cluster: Optional[str],
        controls: List[str],
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any],
    ) -> Tuple[float, float, float, float, Tuple[float, float], int]:
        """Compute bootstrap standard errors.

        Uses unit-level block bootstrap for panel data; when ``cluster`` is
        set, whole clusters are resampled instead (a cluster may contain
        both treated and control units, so replicates with an empty arm are
        counted as failed).

        Parameters
        ----------
        df : pd.DataFrame
            Full panel data.
        outcome : str
            Outcome column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.
        treatment : str
            Treatment indicator column name.
        cluster : str or None
            Cluster column name.
        controls : list of str
            Control variable column names.
        pre_periods : list
            Pre-treatment period values.
        post_periods : list
            Post-treatment period values.
        treated_units : list
            Treated unit identifiers.
        control_units : list
            Control unit identifiers.

        Returns
        -------
        att : float
            Point estimate from full sample.
        se : float
            Bootstrap standard error.
        t_stat : float
            t-statistic.
        p_value : float
            Two-sided p-value.
        conf_int : tuple of float
            Confidence interval (lower, upper).
        df_used : int
            Degrees of freedom the p-value/CI actually used (G-1 when
            clustered, N-k otherwise) - stored as ``df_inference``.
        """
        # Full-sample estimate
        treated_set = set(treated_units)
        pre_mask = df[time].isin(pre_periods)
        if self.rolling == "demean":
            df_t = self._transform_demean(df, outcome, unit, pre_mask)
        elif self.rolling == "detrend":
            df_t = self._transform_detrend(df, outcome, unit, time, pre_mask)
        elif self.rolling == "demeanq":
            df_t = self._transform_demeanq(df, outcome, unit, time, pre_mask)
        elif self.rolling == "detrendq":
            df_t = self._transform_detrendq(df, outcome, unit, time, pre_mask)
        else:
            df_t = self._transform_detrend(df, outcome, unit, time, pre_mask)

        post_mask = df_t[time].isin(post_periods)  # type: ignore[union-attr, call-overload]
        post_df = df_t.loc[post_mask]  # type: ignore[union-attr]
        # Same fixed-window complete-case rule as _fit_common_timing (the
        # bootstrap must estimate the same estimand as the point path).
        post_counts = post_df.loc[np.isfinite(post_df["_ydot"])].groupby(unit)["_ydot"].size()
        complete_units = set(post_counts.index[post_counts == len(post_periods)])
        post_df = post_df.loc[post_df[unit].isin(complete_units)]
        unit_post_avg = post_df.groupby(unit)["_ydot"].mean()

        cs_df = df.drop_duplicates(subset=[unit], keep="first")[[unit] + controls].copy()
        cs_df["_treat"] = cs_df[unit].isin(treated_set).astype(float)
        cs_df["_ydot_avg"] = cs_df[unit].map(unit_post_avg)
        cs_df = cs_df.dropna(subset=["_ydot_avg"])

        y_full = cs_df["_ydot_avg"].values.astype(np.float64)
        treat_full = cs_df["_treat"].values.astype(np.float64)
        controls_mat = cs_df[controls].values.astype(np.float64) if controls else None

        att_full, _, _, _, n_params_full, _ = self._dispatch_estimator(
            y_full, treat_full, controls_mat, None, len(y_full)
        )

        # Bootstrap replications. Resampling level: units (default) or
        # whole CLUSTERS when cluster= is set (campaign finding: the
        # cluster parameter was silently ignored here, producing an iid
        # unit bootstrap labeled as clustered). The resampling population
        # is restricted to units SURVIVING the transformation and finite
        # filters (round-5 review: the raw-panel population let a raw
        # cluster map claim G clusters when fewer contribute).
        surviving = set(cs_df[unit])
        treated_arr = np.array([u for u in treated_units if u in surviving])
        control_arr = np.array([u for u in control_units if u in surviving])
        n_treated = len(treated_arr)
        n_control = len(control_arr)
        unit_counts = df.groupby(unit).size().to_dict()

        # Positional row map computed ONCE (campaign finding: the previous
        # code collected index LABELS then fetched rows POSITIONALLY via
        # .iloc - crashing on non-default indexes and silently resampling
        # the wrong rows when labels were permuted relative to positions).
        unit_col_arr = df[unit].to_numpy()
        all_unit_ids = np.concatenate([treated_arr, control_arr])
        unit_positions = {u: np.flatnonzero(unit_col_arr == u) for u in all_unit_ids}

        cluster_draw: Optional[Dict[Any, np.ndarray]] = None
        if cluster is not None:
            # cluster is unit-constant (validated); map cluster -> units.
            cluster_by_unit = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)[
                cluster
            ]
            cluster_lists: Dict[Any, List[Any]] = {}
            for u in all_unit_ids:
                cluster_lists.setdefault(cluster_by_unit[u], []).append(u)
            cluster_draw = {cl: np.asarray(us) for cl, us in cluster_lists.items()}
            if len(cluster_draw) < 2:
                # Fewer than 2 effective clusters survive the
                # transformation: clustered bootstrap inference is not
                # identified (round-5 review - the raw cluster map
                # previously produced a near-zero SE here). Point
                # retained, inference NaN.
                warnings.warn(
                    "LWDiD bootstrap: fewer than 2 effective clusters "
                    "survive the transformation; clustered bootstrap "
                    "inference is not identified (NaN).",
                    UserWarning,
                    stacklevel=2,
                )
                return att_full, np.nan, np.nan, np.nan, (np.nan, np.nan), 0
        treated_set_all = set(treated_units)

        def _draw_units(rng_b: np.random.Generator) -> np.ndarray:
            if cluster_draw is not None:
                # Cluster-level draws (no treated/control stratification: a
                # cluster may contain both arms). A draw collapsing onto a
                # single distinct cluster carries no between-cluster
                # variation - counted as a failed replicate (round-5
                # review), signalled by an empty draw.
                cluster_keys = list(cluster_draw)
                picks = rng_b.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
                if len(set(picks.tolist())) < 2:
                    return np.array([], dtype=object)
                return np.concatenate([cluster_draw[cluster_keys[i]] for i in picks])
            boot_treated = rng_b.choice(treated_arr, size=n_treated, replace=True)
            boot_control = rng_b.choice(control_arr, size=n_control, replace=True)
            return np.concatenate([boot_treated, boot_control])

        def _replicate_att(boot_units: np.ndarray) -> float:
            """Estimate one bootstrap replicate (shared serial/parallel)."""
            if boot_units.size == 0:
                return np.nan  # single-distinct-cluster draw (failed)
            boot_indices = np.concatenate([unit_positions[u] for u in boot_units])
            boot_df = df.iloc[boot_indices].copy()
            # Occurrence-specific synthetic unit ids keep duplicate draws
            # distinct through the transform step.
            repeat_counts = [unit_counts[u] for u in boot_units]
            boot_df["_boot_unit"] = np.repeat(np.arange(len(boot_units)), repeat_counts)

            # Treatment is EVER-TREATED MEMBERSHIP of the source unit, not
            # the collapsed row's time-varying D (with unsorted input the
            # drop_duplicates(keep="first") row is arbitrary - typically
            # pre-treatment, which would zero the treatment vector).
            boot_treat_vec = np.array(
                [1.0 if u in treated_set_all else 0.0 for u in boot_units], dtype=np.float64
            )
            if boot_treat_vec.sum() == 0 or boot_treat_vec.sum() == len(boot_units):
                return np.nan  # invalid replicate: an arm is empty

            # Apply transformation
            pre_mask_b = boot_df[time].isin(pre_periods)
            if self.rolling == "demean":
                boot_df = self._transform_demean(boot_df, outcome, "_boot_unit", pre_mask_b)
            elif self.rolling == "detrend":
                boot_df = self._transform_detrend(boot_df, outcome, "_boot_unit", time, pre_mask_b)
            elif self.rolling == "demeanq":
                boot_df = self._transform_demeanq(boot_df, outcome, "_boot_unit", time, pre_mask_b)
            elif self.rolling == "detrendq":
                boot_df = self._transform_detrendq(boot_df, outcome, "_boot_unit", time, pre_mask_b)
            else:
                boot_df = self._transform_detrend(boot_df, outcome, "_boot_unit", time, pre_mask_b)

            # Cross-sectional estimate
            post_mask_b = boot_df[time].isin(post_periods)  # type: ignore[union-attr, call-overload]
            post_b = boot_df.loc[post_mask_b]  # type: ignore[union-attr]
            unit_avg_b = post_b.groupby("_boot_unit")["_ydot"].mean()

            first_rows = boot_df.drop_duplicates(subset=["_boot_unit"], keep="first")  # type: ignore[union-attr]
            cs_b = first_rows[["_boot_unit"]].copy()
            if controls:
                for c in controls:
                    cs_b[c] = first_rows[c].values

            cs_b["_treat"] = cs_b["_boot_unit"].map(
                dict(zip(range(len(boot_units)), boot_treat_vec))
            )
            cs_b["_ydot_avg"] = cs_b["_boot_unit"].map(unit_avg_b)
            cs_b = cs_b.dropna(subset=["_ydot_avg"])

            if len(cs_b) < 3:
                return np.nan

            y_b = cs_b["_ydot_avg"].values.astype(np.float64)
            treat_b = cs_b["_treat"].values.astype(np.float64)
            ctrl_b = cs_b[controls].values.astype(np.float64) if controls else None

            try:
                att_b, _, _, _, _, _ = self._dispatch_estimator(
                    y_b, treat_b, ctrl_b, None, len(y_b)
                )
                return float(att_b)
            except (np.linalg.LinAlgError, ValueError):
                return np.nan

        # Per-replicate RNG streams via SeedSequence spawning, IDENTICAL for
        # every n_jobs: replicate b always draws from child stream b, so a
        # seeded fit is reproducible regardless of the execution mode
        # (review round 2: the serial path consumed one sequential stream
        # while the parallel path spawned, so the same seed produced
        # different bootstrap SEs across n_jobs). seed=None still draws
        # fresh OS entropy (non-deterministic).
        seed_seq = np.random.SeedSequence(self.seed)
        child_seqs = seed_seq.spawn(self.n_bootstrap)
        boot_unit_samples = [
            _draw_units(np.random.default_rng(child_seqs[b])) for b in range(self.n_bootstrap)
        ]

        if self.n_jobs == 1:
            # --- Serial path ---
            boot_atts = np.array([_replicate_att(sample) for sample in boot_unit_samples])
        else:
            # --- Parallel path (n_jobs > 1) ---
            from concurrent.futures import ThreadPoolExecutor

            warnings.warn(
                "Parallel bootstrap (n_jobs > 1) is experimental. "
                "ThreadPoolExecutor is used; speedup depends on "
                "GIL-releasing operations in numpy/scipy.",
                UserWarning,
                stacklevel=2,
            )

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                boot_atts = np.array(list(executor.map(_replicate_att, boot_unit_samples)))

        # Compute bootstrap SE
        n_failed = int(np.isnan(boot_atts).sum())
        if n_failed > 0:
            warnings.warn(
                f"LWDiD bootstrap: {n_failed}/{self.n_bootstrap} replication(s) failed "
                f"(returned NaN). Results based on {self.n_bootstrap - n_failed} valid replications.",
                UserWarning,
                stacklevel=2,
            )
        valid_boots = boot_atts[np.isfinite(boot_atts)]
        if len(valid_boots) < 2:
            se = np.nan
        else:
            from diff_diff.lwdid_staggered import _guard_standard_error

            se = _guard_standard_error(
                att_full,
                float(np.std(valid_boots, ddof=1)),
                scale=float(np.max(np.abs(y_full))) if len(y_full) else 0.0,
            )

        # df matches the analytical path's rule (campaign finding: the
        # reported df_inference was G-1 under cluster= while the bootstrap
        # p-value used N-k): G-1 when clustered, N-k otherwise. The df
        # actually used is returned so the caller can store it.
        if cluster_draw is not None:
            df_used = max(len(cluster_draw) - 1, 1)
        else:
            df_used = len(y_full) - n_params_full
        t_stat, p_value, conf_int = safe_inference(att_full, se, alpha=self.alpha, df=df_used)

        return att_full, se, t_stat, p_value, conf_int, df_used

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        params = self.get_params()
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"LWDiD({params_str})"


def validate_staggered_data(data, unit, time, cohort) -> Dict[str, Any]:
    """Validate panel data structure for staggered DiD estimation.

    Checks (using the same never-treated definition as ``fit()``:
    cohort ``NaN``/``NaT``, ``0``, ``np.inf`` (recoded), or a finite value
    beyond the last observed period (recoded)):

    - Panel is complete (all unit×time combinations exist)
    - Cohort is time-invariant within units (missing values included,
      matching ``fit_staggered``'s ``nunique(dropna=False)`` check)
    - At least one never-treated unit exists
    - At least one treated cohort remains after normalization
    - Time and cohort columns share the same time family

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    cohort : str
        Cohort column name (``0``/``NaN``/``NaT`` = never-treated;
        ``np.inf`` and beyond-window values are recoded to never-treated
        with a warning, as in ``fit()``).

    Returns
    -------
    dict
        Validation results with keys: 'valid', 'warnings', 'errors',
        'n_units', 'n_periods', 'n_cohorts', 'n_never_treated'.
    """

    df = data.copy()

    results: dict[str, Any] = {"valid": True, "warnings": [], "errors": []}

    # Check required columns exist
    for col in [unit, time, cohort]:
        if col not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' not found in data")
            return results

    # Dtype coherence + normalization (dtype-aware; mirrors fit()'s
    # encode-then-normalize pipeline without building position maps).
    # Round-8 review: datetime64 and Period are DISTINCT families (mixing
    # them crashes pandas position lookups), so both directions and
    # Period-frequency mismatches are rejected exactly like
    # _encode_staggered_time_scale - a single "datelike" flag previously
    # let a datetime time column meet a Period cohort column in an
    # invalid cross-family comparison below.
    cohort_is_datetime = pd.api.types.is_datetime64_any_dtype(df[cohort])
    time_is_datetime = pd.api.types.is_datetime64_any_dtype(df[time])
    cohort_is_period = isinstance(df[cohort].dtype, pd.PeriodDtype)
    time_is_period = isinstance(df[time].dtype, pd.PeriodDtype)
    cohort_datelike = cohort_is_datetime or cohort_is_period
    if time_is_datetime != cohort_is_datetime or time_is_period != cohort_is_period:
        results["valid"] = False
        results["errors"].append(
            f"Columns '{time}' (time) and '{cohort}' (cohort) must share the "
            f"same time scale; got dtypes {df[time].dtype} and {df[cohort].dtype}. "
            f"Encode both as datetime64, both as Period with the same "
            f"frequency, or both as numeric."
        )
        return results
    if time_is_period and df[time].dtype.freq != df[cohort].dtype.freq:
        results["valid"] = False
        results["errors"].append(
            f"Columns '{time}' (time) and '{cohort}' (cohort) are Period "
            f"columns with different frequencies ({df[time].dtype} vs "
            f"{df[cohort].dtype}). Convert them to a common frequency."
        )
        return results
    if cohort_datelike:
        # Datetime/Period: NaT = never-treated; beyond-window recodes to
        # NaT via a same-dtype comparison (no numeric sentinel exists).
        max_time = df[time].max()
        beyond = df[cohort].notna() & (df[cohort] > max_time)
        if beyond.to_numpy().any():
            results["warnings"].append(
                f"{int(beyond.sum())} row(s) have cohort values beyond the "
                f"last observed period ({max_time}); treated as never-treated."
            )
            df.loc[beyond, cohort] = pd.NaT
        never_mask_series = df[cohort].isna()
        treated_vals = df.loc[~never_mask_series, cohort]
    else:
        try:
            df[cohort], _, _ = _normalize_cohorts(df[cohort], max_time=df[time].max())
        except ValueError as exc:
            results["valid"] = False
            results["errors"].append(str(exc))
            return results
        never_mask_series = df[cohort].isna() | (df[cohort] == 0)
        treated_vals = df.loc[~never_mask_series, cohort]

    # Check cohort time-invariance (missing values included, matching
    # fit_staggered's nunique(dropna=False) — a unit mixing NaT/NaN with a
    # finite cohort must fail here, not later inside fit).
    cohort_per_unit = df.groupby(unit)[cohort].nunique(dropna=False)
    varying = cohort_per_unit[cohort_per_unit > 1]
    if len(varying) > 0:
        results["valid"] = False
        results["errors"].append(f"{len(varying)} units have time-varying cohort values")

    # Check for never-treated. All-eventually-treated panels are rejected
    # by fit() for staggered designs, so mirror that hard-error here
    # instead of reporting a valid-with-warning contradiction.
    never_treated = df.loc[never_mask_series, unit].nunique()
    if never_treated == 0:
        results["valid"] = False
        results["errors"].append(
            "No never-treated units found (cohort NaN/NaT or 0); staggered "
            "LWDiD estimation requires a never-treated control group."
        )

    # Treated-cohort coherence: if normalization recoded every cohort,
    # fit_staggered would raise "No treated cohorts found." — report the
    # same failure here instead of valid-with-zero-cohorts.
    n_cohorts = int(treated_vals.nunique())
    if n_cohorts == 0:
        results["valid"] = False
        results["errors"].append("No treated cohorts found.")

    # Duplicate (unit, time) cells are INVALID (fit() rejects them), and
    # a duplicate can mask a missing cell in the row-count balance check
    # below (round-11 review).
    n_dup_cells = int(df.duplicated(subset=[unit, time]).sum())
    if n_dup_cells > 0:
        results["valid"] = False
        results["errors"].append(
            f"{n_dup_cells} duplicate (unit, time) observation(s); each " f"pair must be unique."
        )

    # Check panel balance
    n_units = df[unit].nunique()
    n_times = df[time].nunique()
    expected_rows = n_units * n_times
    if len(df) - n_dup_cells != expected_rows:
        results["warnings"].append(
            f"Unbalanced panel: {len(df) - n_dup_cells} distinct cell(s) vs "
            f"{expected_rows} expected"
        )

    # Missing unit/time values are ERRORS (fit() rejects the same frame;
    # round-23 review: warning-only let 'valid: True' disagree with fit).
    # Cohort NaN/NaT stays a documented never-treated encoding.
    for col in [unit, time]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            results["valid"] = False
            results["errors"].append(f"{n_missing} missing values in '{col}'")

    results["n_units"] = n_units
    results["n_periods"] = n_times
    results["n_cohorts"] = n_cohorts
    results["n_never_treated"] = never_treated

    return results


def is_never_treated(data, unit, cohort, time=None) -> np.ndarray:
    """Identify never-treated units in staggered design.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset.
    unit : str
        Unit identifier column name.
    cohort : str
        Cohort column name. Never-treated encodings: ``0``, ``NaN``/``NaT``,
        and ``np.inf``.
    time : str or None, default None
        Time column name. When provided, beyond-window classification also
        applies (dtype-aware): a finite cohort value greater than the last
        observed period counts as never-treated, matching ``fit()``'s
        normalization. Without it, only the sentinel encodings above are
        classified.

    Returns
    -------
    np.ndarray of bool
        True for never-treated units (one entry per unique unit).
    """
    unit_cohort = data.groupby(unit)[cohort].first()
    never = (unit_cohort == 0) | unit_cohort.isna()
    if pd.api.types.is_numeric_dtype(unit_cohort):
        never |= np.isposinf(unit_cohort.to_numpy(dtype=float, na_value=np.nan))
    if time is not None:
        max_time = data[time].max()
        never |= unit_cohort.notna() & (unit_cohort > max_time)
    return np.asarray(never)
