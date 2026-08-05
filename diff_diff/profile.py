"""Descriptive panel-profiling utility for agent-facing use.

``profile_panel()`` inspects a DiD panel and returns a :class:`PanelProfile`
dataclass of structural facts — panel balance, treatment-type classification,
outcome characteristics, and a list of factual :class:`Alert` observations.

This module is descriptive, not opinionated. Alerts report what is (e.g.
"smallest cohort has 7 units"), never what to do about it. Estimator
selection is the caller's responsibility; consult
``diff_diff.get_llm_guide("autonomous")`` for the estimator-support matrix
and per-design-feature reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import pandas as pd

_OBSERVATION_COVERAGE_THRESHOLD = 0.70
_MIN_COHORT_SIZE_THRESHOLD = 10
_SHORT_PRE_PANEL_THRESHOLD = 3
_SHORT_POST_PANEL_THRESHOLD = 3


@dataclass(frozen=True)
class Alert:
    """A factual observation about a panel.

    ``severity`` is ``"info"`` (descriptive) or ``"warn"`` (descriptive and
    likely relevant to the caller's estimator choice). Alerts never
    recommend a specific estimator.
    """

    code: str
    severity: str
    message: str
    observed: Any


@dataclass(frozen=True)
class OutcomeShape:
    """Distributional shape of a numeric outcome column.

    Populated on :class:`PanelProfile` when the outcome dtype is integer or
    float (``np.dtype(...).kind in {"i", "u", "f"}``); ``None`` otherwise.
    Descriptive only — these fields surface what is observed in the outcome
    distribution. They never recommend a specific estimator family.
    """

    n_distinct_values: int
    pct_zeros: float
    value_min: float
    value_max: float
    skewness: Optional[float]
    excess_kurtosis: Optional[float]
    is_integer_valued: bool
    is_count_like: bool
    is_bounded_unit: bool


@dataclass(frozen=True)
class TreatmentDoseShape:
    """Distributional shape of a continuous treatment dose.

    Populated on :class:`PanelProfile` only when ``treatment_type ==
    "continuous"``; ``None`` otherwise. Most fields are descriptive
    distributional context.

    **profile_panel only sees the dose column**, not the separate
    ``first_treat`` column ``ContinuousDiD.fit()`` consumes. In the
    canonical ``ContinuousDiD`` setup (Callaway, Goodman-Bacon,
    Sant'Anna 2024) the dose ``D_i`` is **time-invariant per unit**
    (``D_i = 0`` for never-treated, ``D_i > 0`` constant across all
    periods for treated unit i) and ``first_treat`` is a **separate
    column** the caller supplies — not derived from the dose column.
    Under that canonical setup, several profile-side facts on the
    dose column predict ``ContinuousDiD.fit()`` outcomes:

    1. ``PanelProfile.has_never_treated == True`` (some unit has
       dose 0 in every period). Predicts the estimator's
       ``P(D=0) > 0`` requirement under the default
       ``control_group="never_treated"`` /
       ``control_group="not_yet_treated"`` (the canonical setup ties
       ``first_treat == 0`` to ``D_i == 0``). When it is ``False``,
       ``control_group="lowest_dose"`` (Remark 3.1) is the route —
       the lowest-dose group becomes the comparison (needs a mass
       point at the minimum dose) — so failure of (1) no longer rules
       ``ContinuousDiD`` out; see routing notes below.
    2. ``PanelProfile.treatment_varies_within_unit == False``
       (per-unit full-path dose constancy on the dose column). This
       IS the actual fit-time gate, matching
       ``ContinuousDiD.fit()``'s
       ``df.groupby(unit)[dose].nunique() > 1`` rejection at line
       222-228; holds regardless of ``first_treat``. ``True`` rules
       ``ContinuousDiD`` out — for graded-adoption panels with
       dose changes use ``HeterogeneousAdoptionDiD``.
    3. ``PanelProfile.is_balanced == True``. Actual fit-time gate
       (``continuous_did.py:653-661``); not ``first_treat``-dependent.
    4. Absence of the ``duplicate_unit_time_rows`` alert. The
       precompute path silently resolves duplicate ``(unit, time)``
       cells via last-row-wins (``continuous_did.py:1256-1261``);
       **not** a fit-time raise. The agent must deduplicate before
       fit because ``ContinuousDiD`` will otherwise overwrite
       silently.
    5. ``treatment_dose.dose_min > 0`` (over non-zero doses).
       Predicts ``ContinuousDiD.fit()``'s strictly-positive-treated-
       dose requirement (raises ``ValueError`` on negative dose for
       ``first_treat > 0`` units, ``continuous_did.py:586-593``).
       Failure means some treated units have negative dose; see
       routing notes below.

    Routing alternatives when (1) or (5) fails:

    - When (1) fails (no never-treated controls but all observed
      doses non-negative): use ``control_group="lowest_dose"``
      (Remark 3.1) if there is a mass point at the minimum dose
      (``>= 2`` units at ``d_L``) — the lowest-dose group becomes the
      comparison and the estimand is ``ATT(d) - ATT(d_L)``.
      Otherwise ``HeterogeneousAdoptionDiD`` IS a candidate for
      graded-adoption designs (HAD's contract requires non-negative
      dose, satisfied here); linear DiD with the treatment as a
      continuous covariate is another.
    - When (5) fails (negative treated doses):
      ``HeterogeneousAdoptionDiD`` is **not** a fallback either —
      HAD raises on negative post-period dose (``had.py:1563-1572``,
      paper Section 2). Linear DiD with the treatment as a signed
      continuous covariate is the applicable routing alternative.
    - Re-encoding the treatment column (shifting, absolute value,
      etc.) is an agent-side preprocessing choice that changes the
      estimand and is not documented in REGISTRY as a supported
      fallback; if the agent re-encodes to non-negative support,
      both ``ContinuousDiD`` and ``HeterogeneousAdoptionDiD``
      become candidates again on the re-encoded scale.
    - Do **not** relabel positive- or negative-dose units as
      ``first_treat == 0``: that triggers ``ContinuousDiD.fit()``'s
      force-zero coercion path, which is implementation behavior
      for inconsistent inputs (e.g., an accidentally-nonzero row on
      a never-treated unit), not a documented routing option.

    The agent must still validate the supplied ``first_treat``
    column independently: it must contain at least one
    ``first_treat == 0`` unit (``P(D=0) > 0``), be non-negative
    integer-valued (or ``+inf`` / 0 for never-treated), and be
    consistent with the dose column on per-unit treated/untreated
    status. ``profile_panel`` does not see ``first_treat`` and
    cannot validate it.

    ``has_zero_dose`` is a row-level fact ("at least one observation has
    dose == 0"); it is NOT a substitute for ``has_never_treated``, which
    is the unit-level field. A panel can have ``has_zero_dose == True``
    (pre-treatment zero rows) while ``has_never_treated == False`` (every
    unit eventually treated), in which case the standard-workflow agent
    would conclude no never-treated controls exist before calling
    ``ContinuousDiD.fit()``.
    """

    n_distinct_doses: int
    has_zero_dose: bool
    dose_min: float
    dose_max: float
    dose_mean: float


@dataclass(frozen=True)
class PanelProfile:
    """Structural facts about a DiD panel.

    Returned by :func:`profile_panel`. Mirrors the ``BusinessContext``
    frozen-dataclass pattern. Consume ``.to_dict()`` for a JSON-serializable
    representation and reason against the bundled
    ``llms-autonomous.txt`` guide.
    """

    n_units: int
    n_periods: int
    n_obs: int
    is_balanced: bool  # every (unit, time) cell appears at least once
    observation_coverage: float  # unique (unit, time) keys / (n_units * n_periods)

    treatment_type: str
    is_staggered: bool
    n_cohorts: int
    cohort_sizes: Mapping[Any, int]
    has_never_treated: bool
    has_always_treated: bool
    treatment_varies_within_unit: bool

    first_treatment_period: Optional[Any]
    last_treatment_period: Optional[Any]
    min_pre_periods: Optional[int]
    min_post_periods: Optional[int]

    outcome_dtype: str
    outcome_is_binary: bool
    outcome_has_zeros: bool
    outcome_has_negatives: bool
    outcome_missing_fraction: float
    outcome_summary: Mapping[str, float]

    alerts: Tuple[Alert, ...]

    # Wave 2 additions are kept defaulted so direct PanelProfile(...)
    # construction by external callers does not break when the new
    # fields are not supplied. profile_panel() always populates both.
    outcome_shape: Optional[OutcomeShape] = None
    treatment_dose: Optional[TreatmentDoseShape] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of the profile."""
        return {
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "n_obs": self.n_obs,
            "is_balanced": self.is_balanced,
            "observation_coverage": self.observation_coverage,
            "treatment_type": self.treatment_type,
            "is_staggered": self.is_staggered,
            "n_cohorts": self.n_cohorts,
            "cohort_sizes": {_jsonable_key(k): int(v) for k, v in self.cohort_sizes.items()},
            "has_never_treated": self.has_never_treated,
            "has_always_treated": self.has_always_treated,
            "treatment_varies_within_unit": self.treatment_varies_within_unit,
            "first_treatment_period": _jsonable(self.first_treatment_period),
            "last_treatment_period": _jsonable(self.last_treatment_period),
            "min_pre_periods": self.min_pre_periods,
            "min_post_periods": self.min_post_periods,
            "outcome_dtype": self.outcome_dtype,
            "outcome_is_binary": self.outcome_is_binary,
            "outcome_has_zeros": self.outcome_has_zeros,
            "outcome_has_negatives": self.outcome_has_negatives,
            "outcome_missing_fraction": self.outcome_missing_fraction,
            "outcome_summary": {k: float(v) for k, v in self.outcome_summary.items()},
            "outcome_shape": (
                None
                if self.outcome_shape is None
                else {
                    "n_distinct_values": int(self.outcome_shape.n_distinct_values),
                    "pct_zeros": float(self.outcome_shape.pct_zeros),
                    "value_min": float(self.outcome_shape.value_min),
                    "value_max": float(self.outcome_shape.value_max),
                    "skewness": (
                        None
                        if self.outcome_shape.skewness is None
                        else float(self.outcome_shape.skewness)
                    ),
                    "excess_kurtosis": (
                        None
                        if self.outcome_shape.excess_kurtosis is None
                        else float(self.outcome_shape.excess_kurtosis)
                    ),
                    "is_integer_valued": bool(self.outcome_shape.is_integer_valued),
                    "is_count_like": bool(self.outcome_shape.is_count_like),
                    "is_bounded_unit": bool(self.outcome_shape.is_bounded_unit),
                }
            ),
            "treatment_dose": (
                None
                if self.treatment_dose is None
                else {
                    "n_distinct_doses": int(self.treatment_dose.n_distinct_doses),
                    "has_zero_dose": bool(self.treatment_dose.has_zero_dose),
                    "dose_min": float(self.treatment_dose.dose_min),
                    "dose_max": float(self.treatment_dose.dose_max),
                    "dose_mean": float(self.treatment_dose.dose_mean),
                }
            ),
            "alerts": [
                {
                    "code": a.code,
                    "severity": a.severity,
                    "message": a.message,
                    "observed": _jsonable(a.observed),
                }
                for a in self.alerts
            ],
        }


def profile_panel(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    treatment: str,
    outcome: str,
) -> PanelProfile:
    """Describe the structure of a DiD panel.

    Reports structural facts — balance, treatment-type classification,
    outcome characteristics, factual alerts. Descriptive, not opinionated:
    the profile says what is, never what to do about it. Estimator
    selection is up to the caller.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel data containing the four named columns.
    unit : str
        Column identifying the cross-sectional unit.
    time : str
        Column identifying the time period.
    treatment : str
        Column holding the treatment indicator or dose. See Notes for the
        classification rules.
    outcome : str
        Column holding the outcome variable.

    Returns
    -------
    PanelProfile
        Frozen dataclass. Call ``.to_dict()`` for a JSON-serializable view.

    Raises
    ------
    ValueError
        If any of the four column names is not present in ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from diff_diff import profile_panel
    >>> df = pd.DataFrame({
    ...     "u":  [1, 1, 2, 2],
    ...     "t":  [0, 1, 0, 1],
    ...     "tr": [0, 0, 1, 1],
    ...     "y":  [0.1, 0.2, 0.1, 0.9],
    ... })
    >>> profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    >>> profile.is_balanced
    True
    >>> profile.treatment_type
    'binary_absorbing'

    Notes
    -----
    Classification rules for ``treatment_type``:

    - ``"binary_absorbing"``: numeric treatment whose observed non-NaN
      values are a subset of :math:`\\{0, 1\\}` (one or two distinct
      values) AND each unit's treatment sequence (ordered by ``time``)
      is weakly monotone non-decreasing. All-zero and all-one panels
      are valid degenerate cases.
    - ``"binary_non_absorbing"``: values a subset of :math:`\\{0, 1\\}`
      with at least two distinct values observed, where at least one
      unit switches from 1 back to 0.
    - ``"continuous"``: numeric treatment with more than two distinct
      values, or a 2-valued numeric whose values are not in
      :math:`\\{0, 1\\}` (matches the ``ContinuousDiD`` convention).
    - ``"categorical"``: non-numeric dtype (object / category) or a
      column that is entirely NaN.

    Bool-dtype columns (``True`` / ``False``) are classified the same
    way as numeric ``{0, 1}``: the library's binary estimators validate
    on value support via :func:`diff_diff.utils.validate_binary`, so
    ``True`` / ``False`` behave like ``1`` / ``0`` for absorbing /
    non-absorbing classification.

    ``has_never_treated`` is computed across both binary and
    continuous numeric treatment types: some unit has ``treatment ==
    0`` in every observed non-NaN row. For binary this flags the
    clean-control group; for continuous this flags zero-dose controls
    (required by ``ContinuousDiD``). Always ``False`` for
    ``"categorical"``.

    ``has_always_treated`` has binary-only semantics: some unit has
    ``treatment == 1`` in every observed non-NaN row (no pre-treatment
    information in the DiD sense). For ``"continuous"`` and
    ``"categorical"`` treatment this field is always ``False``
    regardless of dose positivity — pre-treatment periods on
    continuous DiD are determined by the separate ``first_treat``
    column passed to ``ContinuousDiD.fit``, not by whether the dose
    is strictly positive.

    Rows with ``NaN`` in ``unit`` or ``time`` are dropped up front and
    surfaced via the ``missing_id_rows_dropped`` alert; all subsequent
    structural facts are computed on the non-missing subset, so
    ``observation_coverage`` is always in ``[0, 1]``. Duplicate
    ``(unit, time)`` rows are surfaced separately via the
    ``duplicate_unit_time_rows`` alert.

    The profile does not recommend an estimator. Consult
    ``diff_diff.get_llm_guide("autonomous")`` for the estimator-support
    matrix and per-design-feature reasoning.
    """
    _validate_columns(df, unit=unit, time=time, treatment=treatment, outcome=outcome)

    input_row_count = int(len(df))
    if input_row_count == 0:
        raise ValueError("profile_panel: DataFrame is empty; at least one row is required.")

    missing_id_mask = cast(pd.Series, df[[unit, time]].isna().any(axis=1))
    n_rows_with_missing_id = int(missing_id_mask.sum())
    if n_rows_with_missing_id > 0:
        df = df.loc[~missing_id_mask]
    n_obs = int(len(df))
    if n_obs == 0:
        raise ValueError(
            f"profile_panel: no rows remain after dropping "
            f"{n_rows_with_missing_id} row(s) with missing unit or time "
            "identifier; at least one valid row is required."
        )

    n_units = int(df[unit].nunique())
    n_periods = int(df[time].nunique())
    n_unique_keys = int(df[[unit, time]].drop_duplicates().shape[0])
    denom = n_units * n_periods
    observation_coverage = float(n_unique_keys / denom) if denom > 0 else 0.0
    is_balanced = n_unique_keys == denom
    n_duplicate_rows = n_obs - n_unique_keys

    (
        treatment_type,
        is_staggered,
        cohort_sizes,
        has_never_treated,
        has_always_treated,
        first_tp,
        last_tp,
    ) = _classify_treatment(df, unit=unit, time=time, treatment=treatment)

    if pd.api.types.is_numeric_dtype(df[treatment]) or pd.api.types.is_bool_dtype(df[treatment]):
        per_unit_distinct = df.groupby(unit)[treatment].nunique(dropna=True)
        treatment_varies_within_unit = bool((per_unit_distinct > 1).any())
    else:
        treatment_varies_within_unit = False

    min_pre, min_post = _compute_pre_post(
        df,
        unit=unit,
        time=time,
        treatment=treatment,
        treatment_type=treatment_type,
    )

    outcome_col = cast(pd.Series, df[outcome])
    outcome_dtype = str(outcome_col.dtype)
    valid = cast(pd.Series, outcome_col.dropna())
    outcome_missing_fraction = (
        float(1.0 - len(valid) / len(outcome_col)) if len(outcome_col) > 0 else 0.0
    )
    outcome_is_binary, outcome_has_zeros, outcome_has_negatives = _classify_outcome(valid)
    outcome_summary = _summarize_outcome(valid)

    dtype_kind = getattr(outcome_col.dtype, "kind", "O")
    outcome_shape = _compute_outcome_shape(valid, dtype_kind)
    treatment_dose = _compute_treatment_dose(df, treatment=treatment, treatment_type=treatment_type)
    alerts = _compute_alerts(
        n_periods=n_periods,
        observation_coverage=observation_coverage,
        cohort_sizes=cohort_sizes,
        has_never_treated=has_never_treated,
        has_always_treated=has_always_treated,
        min_pre_periods=min_pre,
        min_post_periods=min_post,
        outcome_is_binary=outcome_is_binary,
        outcome_dtype_kind=dtype_kind,
        n_duplicate_rows=n_duplicate_rows,
        n_rows_with_missing_id=n_rows_with_missing_id,
    )

    return PanelProfile(
        n_units=n_units,
        n_periods=n_periods,
        n_obs=n_obs,
        is_balanced=is_balanced,
        observation_coverage=observation_coverage,
        treatment_type=treatment_type,
        is_staggered=is_staggered,
        n_cohorts=len(cohort_sizes),
        cohort_sizes=cohort_sizes,
        has_never_treated=has_never_treated,
        has_always_treated=has_always_treated,
        treatment_varies_within_unit=treatment_varies_within_unit,
        first_treatment_period=first_tp,
        last_treatment_period=last_tp,
        min_pre_periods=min_pre,
        min_post_periods=min_post,
        outcome_dtype=outcome_dtype,
        outcome_is_binary=outcome_is_binary,
        outcome_has_zeros=outcome_has_zeros,
        outcome_has_negatives=outcome_has_negatives,
        outcome_missing_fraction=outcome_missing_fraction,
        outcome_summary=outcome_summary,
        outcome_shape=outcome_shape,
        treatment_dose=treatment_dose,
        alerts=tuple(alerts),
    )


def _validate_columns(df: pd.DataFrame, **cols: str) -> None:
    missing = [(role, name) for role, name in cols.items() if name not in df.columns]
    if missing:
        pairs = ", ".join(f"{role}={name!r}" for role, name in missing)
        raise ValueError(
            f"profile_panel: column(s) not found in DataFrame: {pairs}. "
            f"Provided columns: {list(df.columns)}"
        )


def _classify_treatment(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    treatment: str,
) -> Tuple[
    str,
    bool,
    Dict[Any, int],
    bool,
    bool,
    Optional[Any],
    Optional[Any],
]:
    """Return (type, is_staggered, cohort_sizes, has_never, has_always, first_tp, last_tp)."""
    col = df[treatment]
    is_numeric = pd.api.types.is_numeric_dtype(col)
    is_bool = pd.api.types.is_bool_dtype(col)

    # Bool-dtype treatment columns are treated as binary 0/1 inputs.
    # The library's binary estimators validate value support via
    # `validate_binary`, which accepts bool because True/False coerce
    # to 1/0 numerically. Classifying bool columns as "categorical"
    # here would route a valid binary design away from the supported
    # estimator set.
    if (not is_numeric) and (not is_bool):
        return ("categorical", False, {}, False, False, None, None)

    distinct = col.dropna().unique()
    n_distinct = len(distinct)
    values_set = set(distinct.tolist())
    if n_distinct == 0:
        return ("categorical", False, {}, False, False, None, None)

    # has_never_treated has a single well-defined meaning across binary
    # and continuous numeric treatment: some unit has treatment == 0 in
    # every observed non-NaN row. For binary this is the clean-control
    # group; for continuous this is the zero-dose control proxy used
    # alongside ContinuousDiD's `P(D=0) > 0` requirement.
    #
    # On binary panels (values in {0, 1}), `unit_max == 0` is equivalent
    # to "all observed values are 0". On continuous panels with negative
    # dose support, that equivalence breaks: a unit with path `[-1, 0]`
    # would falsely satisfy `unit_max == 0` even though no row is a
    # zero-dose control. Check both endpoints (`unit_max == 0` AND
    # `unit_min == 0`) to enforce the documented contract — for any
    # numeric panel this is exactly "every observed dose is 0".
    unit_max = df.groupby(unit)[treatment].max().to_numpy()
    unit_min = df.groupby(unit)[treatment].min().to_numpy()
    has_never_treated = bool(np.any((unit_max == 0) & (unit_min == 0)))

    is_binary_valued = values_set <= {0, 1, 0.0, 1.0}
    # has_always_treated has binary-only semantics: "unit is treated in
    # every observed period" = unit_min == 1 on a binary panel (no
    # pre-treatment information). For continuous panels, positive dose
    # throughout does not mean "always treated in the DiD sense"
    # (pre-treatment periods are determined by `first_treat`, not by
    # whether the dose is positive), so this field is False for
    # continuous / categorical types.
    has_always_treated = is_binary_valued and bool(np.any(unit_min == 1))

    if not is_binary_valued:
        return (
            "continuous",
            False,
            {},
            has_never_treated,
            has_always_treated,
            None,
            None,
        )

    sorted_df = df.sort_values([unit, time])

    # Monotonicity check on the observed non-NaN subsequence per unit.
    # A path like [0, 1, NaN, 0] must be detected as non-absorbing: the
    # non-NaN subsequence [0, 1, 0] violates weak monotonicity.
    is_absorbing = True
    for _, group in sorted_df.groupby(unit, sort=False):
        vals = group[treatment].to_numpy()
        mask = ~pd.isna(vals)
        # Cast to int so np.diff on a bool-dtype column performs
        # arithmetic (1 - 0 = 1, 0 - 1 = -1) rather than XOR (which
        # would mask a True -> False transition).
        observed = vals[mask].astype(np.int64, copy=False)
        if len(observed) >= 2 and bool(np.any(np.diff(observed) < 0)):
            is_absorbing = False
            break

    if not is_absorbing:
        return (
            "binary_non_absorbing",
            False,
            {},
            has_never_treated,
            has_always_treated,
            None,
            None,
        )

    first_treat = sorted_df[sorted_df[treatment] == 1].groupby(unit, sort=False)[time].min()
    cohort_counts = first_treat.value_counts().sort_index()
    cohort_sizes: Dict[Any, int] = {k: int(v) for k, v in cohort_counts.items()}
    first_tp = min(cohort_sizes) if cohort_sizes else None
    last_tp = max(cohort_sizes) if cohort_sizes else None
    is_staggered = len(cohort_sizes) >= 2

    return (
        "binary_absorbing",
        is_staggered,
        cohort_sizes,
        has_never_treated,
        has_always_treated,
        first_tp,
        last_tp,
    )


def _compute_pre_post(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    treatment: str,
    treatment_type: str,
) -> Tuple[Optional[int], Optional[int]]:
    """Return (min_pre, min_post) across treated units using each unit's
    observed (unit, time) support. On unbalanced panels this correctly
    reflects the actual pre/post exposure of the least-supported treated
    unit, rather than the global panel period set which could overstate
    exposure and suppress short-panel alerts.
    """
    if treatment_type != "binary_absorbing":
        return None, None

    support = df[[unit, time]].drop_duplicates()
    sorted_df = df.sort_values([unit, time])
    first_treat_per_unit = (
        sorted_df[sorted_df[treatment] == 1].groupby(unit, sort=False)[time].min()
    )
    if first_treat_per_unit.empty:
        return None, None

    pre_counts: List[int] = []
    post_counts: List[int] = []
    treated_units = first_treat_per_unit.index.tolist()
    for u in treated_units:
        c_u = first_treat_per_unit.loc[u]
        unit_periods = support.loc[support[unit] == u, time]
        pre_counts.append(int((unit_periods < c_u).sum()))
        post_counts.append(int((unit_periods >= c_u).sum()))

    return int(min(pre_counts)), int(min(post_counts))


def _classify_outcome(valid: pd.Series) -> Tuple[bool, bool, bool]:
    n_distinct = valid.nunique(dropna=False)
    if n_distinct == 0:
        return False, False, False

    is_numeric = pd.api.types.is_numeric_dtype(valid)
    if is_numeric:
        distinct_set = set(valid.unique().tolist())
        is_binary = n_distinct == 2 and (distinct_set <= {0, 1} or distinct_set <= {0.0, 1.0})
        has_zeros = bool((valid == 0).any())
        has_negatives = bool((valid < 0).any())
        return is_binary, has_zeros, has_negatives

    return False, False, False


def _summarize_outcome(valid: pd.Series) -> Dict[str, float]:
    if len(valid) == 0 or not pd.api.types.is_numeric_dtype(valid):
        return {}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
    }


def _compute_outcome_shape(valid: pd.Series, outcome_dtype_kind: str) -> Optional[OutcomeShape]:
    """Compute distributional shape for a numeric outcome.

    Returns ``None`` for non-numeric dtypes (kind not in ``{"i", "u", "f"}``)
    or empty series. Skewness and excess kurtosis are gated on
    ``n_distinct_values >= 3`` and non-zero variance; otherwise ``None``.
    """
    if outcome_dtype_kind not in {"i", "u", "f"}:
        return None
    if len(valid) == 0:
        return None

    arr = np.asarray(valid, dtype=float)
    n = int(arr.size)
    n_distinct = int(np.unique(arr).size)
    pct_zeros = float(np.sum(arr == 0)) / n
    value_min = float(arr.min())
    value_max = float(arr.max())

    # Tolerance-aware integer detection: a CSV-roundtripped count column
    # may carry float64 representation noise (e.g., 1.0 stored as
    # 1.0000000000000002), and that should still classify as
    # integer-valued for the purpose of the count-like heuristic.
    is_integer_valued = bool(np.all(np.isclose(arr, np.round(arr), rtol=0.0, atol=1e-12)))
    is_bounded_unit = bool(np.all((arr >= 0.0) & (arr <= 1.0)))

    skewness: Optional[float] = None
    excess_kurtosis: Optional[float] = None
    if n_distinct >= 3:
        mean = float(np.mean(arr))
        m2 = float(np.mean((arr - mean) ** 2))
        if m2 > 0.0:
            std = m2**0.5
            m3 = float(np.mean((arr - mean) ** 3))
            m4 = float(np.mean((arr - mean) ** 4))
            skewness = float(m3 / (std**3))
            excess_kurtosis = float(m4 / (m2**2) - 3.0)

    # Non-negativity is part of the contract: `is_count_like == True`
    # is the routing signal toward `WooldridgeDiD(method="poisson")`,
    # which hard-rejects negative outcomes at fit time
    # (`wooldridge.py:1105` raises `ValueError` on `y < 0`). Without the
    # `value_min >= 0` guard, a right-skewed integer outcome with zeros
    # and some negatives could set `is_count_like=True` and steer an
    # agent toward an estimator that will then refuse to fit.
    is_count_like = bool(
        is_integer_valued
        and pct_zeros > 0.0
        and skewness is not None
        and skewness > 0.5
        and n_distinct > 2
        and value_min >= 0.0
    )

    return OutcomeShape(
        n_distinct_values=n_distinct,
        pct_zeros=pct_zeros,
        value_min=value_min,
        value_max=value_max,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        is_integer_valued=is_integer_valued,
        is_count_like=is_count_like,
        is_bounded_unit=is_bounded_unit,
    )


def _compute_treatment_dose(
    df: pd.DataFrame,
    *,
    treatment: str,
    treatment_type: str,
) -> Optional[TreatmentDoseShape]:
    """Compute distributional shape for a continuous-treatment dose column.

    Returns ``None`` unless ``treatment_type == "continuous"``. Most
    fields are descriptive distributional context; ``dose_min > 0``
    is one of the profile-side screening checks for ``ContinuousDiD``
    (see :class:`TreatmentDoseShape` docstring for the full screening
    set and the ``first_treat`` validation that
    ``ContinuousDiD.fit()`` applies separately).
    """
    if treatment_type != "continuous":
        return None

    col = df[treatment].dropna()
    if col.empty:
        return None

    n_distinct_doses = int(col.nunique())
    has_zero_dose = bool((col == 0).any())

    # `treatment_type == "continuous"` is reached only when the
    # treatment column has more than two distinct values OR a 2-valued
    # numeric outside `{0, 1}` (see `_classify_treatment`). An all-zero
    # numeric column is classified as `binary_absorbing` and never
    # reaches this branch, so `nonzero` is guaranteed non-empty.
    nonzero = col[col != 0]
    dose_min = float(nonzero.min())
    dose_max = float(nonzero.max())
    dose_mean = float(nonzero.mean())

    return TreatmentDoseShape(
        n_distinct_doses=n_distinct_doses,
        has_zero_dose=has_zero_dose,
        dose_min=dose_min,
        dose_max=dose_max,
        dose_mean=dose_mean,
    )


def _compute_alerts(
    *,
    n_periods: int,
    observation_coverage: float,
    cohort_sizes: Mapping[Any, int],
    has_never_treated: bool,
    has_always_treated: bool,
    min_pre_periods: Optional[int],
    min_post_periods: Optional[int],
    outcome_is_binary: bool,
    outcome_dtype_kind: str,
    n_duplicate_rows: int,
    n_rows_with_missing_id: int,
) -> List[Alert]:
    alerts: List[Alert] = []

    if n_rows_with_missing_id > 0:
        alerts.append(
            Alert(
                code="missing_id_rows_dropped",
                severity="warn",
                message=(
                    f"Dropped {n_rows_with_missing_id} row(s) with missing "
                    "unit or time identifier; structural facts are computed "
                    "from the non-missing subset."
                ),
                observed=int(n_rows_with_missing_id),
            )
        )

    if n_duplicate_rows > 0:
        alerts.append(
            Alert(
                code="duplicate_unit_time_rows",
                severity="warn",
                message=(
                    f"Found {n_duplicate_rows} duplicate (unit, time) row(s); "
                    "balance and coverage are computed from the unique support."
                ),
                observed=int(n_duplicate_rows),
            )
        )

    if cohort_sizes:
        smallest = min(cohort_sizes.values())
        if smallest < _MIN_COHORT_SIZE_THRESHOLD:
            alerts.append(
                Alert(
                    code="min_cohort_size_below_10",
                    severity="warn",
                    message=(
                        f"Smallest cohort has {smallest} units; "
                        "cohort-level inference will be noisy."
                    ),
                    observed=int(smallest),
                )
            )
        if len(cohort_sizes) == 1:
            alerts.append(
                Alert(
                    code="only_one_cohort",
                    severity="info",
                    message=("All treated units adopt at the same time " "(non-staggered design)."),
                    observed=1,
                )
            )
            if not has_never_treated:
                alerts.append(
                    Alert(
                        code="all_units_treated_simultaneously",
                        severity="info",
                        message=(
                            "Every unit is treated and every treated unit "
                            "adopts in the same period; no untreated "
                            "comparison group exists in the panel."
                        ),
                        observed=None,
                    )
                )

    if min_pre_periods is not None and min_pre_periods < _SHORT_PRE_PANEL_THRESHOLD:
        alerts.append(
            Alert(
                code="short_pre_panel",
                severity="warn",
                message=(
                    f"Minimum pre-treatment periods across treated units is "
                    f"{min_pre_periods}; parallel-trends and event-study "
                    "diagnostics have limited power."
                ),
                observed=int(min_pre_periods),
            )
        )
    if min_post_periods is not None and min_post_periods < _SHORT_POST_PANEL_THRESHOLD:
        alerts.append(
            Alert(
                code="short_post_panel",
                severity="info",
                message=(
                    f"Minimum post-treatment periods across treated units is "
                    f"{min_post_periods}; dynamic-effect estimation is "
                    "limited."
                ),
                observed=int(min_post_periods),
            )
        )

    if cohort_sizes and not has_never_treated:
        alerts.append(
            Alert(
                code="no_never_treated",
                severity="info",
                message=(
                    "No never-treated comparison units; every unit in the "
                    "panel is eventually treated."
                ),
                observed=False,
            )
        )

    if has_always_treated:
        alerts.append(
            Alert(
                code="has_always_treated_units",
                severity="info",
                message=(
                    "Some units are treated in every observed period; they "
                    "provide no pre-treatment information."
                ),
                observed=True,
            )
        )

    if observation_coverage < _OBSERVATION_COVERAGE_THRESHOLD:
        alerts.append(
            Alert(
                code="panel_highly_unbalanced",
                severity="warn",
                message=(
                    f"Observation coverage is {observation_coverage:.1%}; "
                    "panel is highly unbalanced."
                ),
                observed=float(observation_coverage),
            )
        )

    if n_periods == 2:
        alerts.append(
            Alert(
                code="only_two_periods",
                severity="info",
                message="Only two time periods are observed (2x2 design).",
                observed=2,
            )
        )

    if outcome_is_binary and outcome_dtype_kind == "f":
        alerts.append(
            Alert(
                code="outcome_looks_binary_but_dtype_float",
                severity="info",
                message=("Outcome takes values in {0, 1} but is stored with a " "float dtype."),
                observed=None,
            )
        )

    return alerts


def _jsonable(x: Any) -> Any:
    """Coerce a value to a JSON-serializable primitive."""
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float, str)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(x)
    if isinstance(x, dict):
        return {_jsonable_key(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def _jsonable_key(k: Any) -> Any:
    """Coerce a mapping key to a JSON-compatible primitive."""
    if isinstance(k, bool):
        return bool(k)
    if isinstance(k, (int, float, str)):
        return k
    if isinstance(k, np.bool_):
        return bool(k)
    if isinstance(k, np.integer):
        return int(k)
    if isinstance(k, np.floating):
        return float(k)
    return str(k)
