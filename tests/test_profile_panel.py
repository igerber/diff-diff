"""Tests for ``diff_diff.profile_panel`` and the ``PanelProfile`` dataclass."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import PanelProfile, profile_panel
from diff_diff.profile import Alert


def _make_panel(
    *,
    n_units: int,
    periods: Iterable[int],
    first_treat: Optional[Dict[int, int]] = None,
    outcome_fn: Any = None,
) -> pd.DataFrame:
    """Build a balanced long panel with optional per-unit first-treatment timing.

    ``first_treat`` maps unit -> first treatment period (inclusive). Units not
    in the mapping are never-treated.
    """
    first_treat = first_treat or {}
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, n_units + 1):
        for t in periods:
            tr = 1 if (u in first_treat and t >= first_treat[u]) else 0
            if outcome_fn is not None:
                y = outcome_fn(u, t, tr, rng)
            else:
                y = float(u) + 0.1 * t + 0.5 * tr
            rows.append({"u": u, "t": t, "tr": tr, "y": y})
    return pd.DataFrame(rows)


def _alert_codes(profile: PanelProfile) -> set[str]:
    return {a.code for a in profile.alerts}


def test_balanced_binary_2x2():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=[0, 1], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.is_staggered is False
    assert profile.has_never_treated is True
    assert profile.n_units == 20
    assert profile.n_periods == 2
    assert profile.is_balanced is True


def test_staggered_multi_cohort():
    first_treat: Dict[int, int] = {}
    first_treat.update({u: 3 for u in range(1, 11)})
    first_treat.update({u: 5 for u in range(11, 21)})
    first_treat.update({u: 7 for u in range(21, 31)})
    df = _make_panel(n_units=40, periods=range(1, 9), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.is_staggered is True
    assert profile.n_cohorts == 3
    assert profile.cohort_sizes == {3: 10, 5: 10, 7: 10}
    assert profile.first_treatment_period == 3
    assert profile.last_treatment_period == 7
    assert profile.has_never_treated is True


def test_binary_non_absorbing_switcher():
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, 21):
        treat_seq = [0, 1, 1, 0, 0] if u > 10 else [0, 0, 0, 0, 0]
        for t, tr in enumerate(treat_seq):
            rows.append({"u": u, "t": t, "tr": tr, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_non_absorbing"
    assert profile.cohort_sizes == {}
    assert profile.is_staggered is False
    assert profile.has_never_treated is True


def test_continuous_treatment():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 41):
        dose = float(rng.uniform(0, 5))
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.cohort_sizes == {}
    assert profile.is_staggered is False
    # Each unit has a constant dose across all periods → time-invariant.
    assert profile.treatment_varies_within_unit is False


def test_continuous_treatment_with_time_varying_dose():
    """Time-varying dose must be flagged so agents routed to
    ContinuousDiD do not hit the fit-time "dose must be time-invariant"
    ValueError. treatment_varies_within_unit == True signals the
    incompatibility."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 21):
        for t in range(4):
            dose = float(rng.uniform(0, 5))
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.treatment_varies_within_unit is True


def test_binary_absorbing_varies_within_unit():
    """Binary-absorbing panels have within-unit treatment variation by
    construction (0 pre, 1 post). The field is True."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_varies_within_unit is True


def test_continuous_positive_dose_does_not_fire_has_always_treated():
    """Valid ContinuousDiD panels have units with a constant positive
    dose across all periods AND well-defined pre-treatment periods
    (via a separate `first_treat` column). `has_always_treated` has
    binary-only semantics, so it must be False on continuous panels
    regardless of dose positivity. Previously the field conflated
    "positive dose throughout" with "always treated in the DiD sense",
    which fired the misleading `has_always_treated_units` alert on
    valid continuous-DiD panels."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 21):
        dose = 0.0 if u <= 5 else 2.5
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is False, (
        "has_always_treated must be False on continuous panels regardless "
        "of dose positivity (binary-only semantics)"
    )
    assert "has_always_treated_units" not in _alert_codes(profile)


def test_bool_dtype_treatment_is_binary_absorbing():
    """Bool-dtype treatment columns (True/False) must classify the same
    way as numeric {0, 1}. The library's binary estimators validate on
    value support via `validate_binary`, which accepts bool because
    True/False coerce to 1/0 numerically. Classifying bool as
    "categorical" would silently route valid binary DiD panels away
    from the supported estimator set."""
    first_treat = {u: 2 for u in range(11, 21)}
    rows = []
    for u in range(1, 21):
        for t in range(4):
            treated = u in first_treat and t >= first_treat[u]
            rows.append({"u": u, "t": t, "tr": bool(treated), "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    assert df["tr"].dtype == bool
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is False
    assert profile.treatment_varies_within_unit is True
    assert profile.cohort_sizes == {2: 10}


def test_bool_dtype_non_absorbing():
    """Reversible 0 -> 1 -> 0 treatment expressed as a bool column must
    classify as binary_non_absorbing, same as numeric."""
    rows = []
    for u in range(1, 11):
        seq = [False, True, True, False, False] if u > 5 else [False] * 5
        for t, tr in enumerate(seq):
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    assert df["tr"].dtype == bool
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_non_absorbing"
    assert profile.has_never_treated is True


def test_categorical_treatment_object_dtype():
    rows = []
    for u in range(1, 11):
        arm = "A" if u <= 5 else "B"
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": arm, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "categorical"
    assert profile.has_never_treated is False
    assert profile.has_always_treated is False


def test_no_never_treated_alert():
    first_treat = {u: 2 for u in range(1, 21)}
    df = _make_panel(n_units=20, periods=range(0, 5), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.has_never_treated is False
    codes = _alert_codes(profile)
    assert "no_never_treated" in codes


def test_has_always_treated_alert():
    rows = []
    for u in range(1, 21):
        for t in range(5):
            tr = 1 if u <= 5 else (1 if t >= 3 else 0)
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.has_always_treated is True
    codes = _alert_codes(profile)
    assert "has_always_treated_units" in codes


def test_unbalanced_panel_below_threshold():
    first_treat = {u: 3 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 5), first_treat=first_treat)
    df = df.iloc[::3].reset_index(drop=True)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.is_balanced is False
    assert profile.observation_coverage < 0.70
    codes = _alert_codes(profile)
    assert "panel_highly_unbalanced" in codes


def test_binary_outcome_float_dtype_alert():
    first_treat = {u: 2 for u in range(11, 31)}
    df = _make_panel(
        n_units=30,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, rng: float(tr),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.outcome_is_binary is True
    assert profile.outcome_dtype == "float64"
    codes = _alert_codes(profile)
    assert "outcome_looks_binary_but_dtype_float" in codes


def test_outcome_missing_fraction_computed():
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df.loc[0:9, "y"] = np.nan
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert 0.0 < profile.outcome_missing_fraction < 1.0
    assert profile.outcome_missing_fraction == pytest.approx(10 / len(df))


def test_short_pre_panel_alert():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=[0, 1, 2, 3], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.min_pre_periods == 1
    codes = _alert_codes(profile)
    assert "short_pre_panel" in codes


def test_missing_column_raises_value_error():
    df = pd.DataFrame({"u": [1, 2], "t": [0, 1], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="treatment"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")


def test_panel_profile_is_frozen():
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.n_units = 999  # type: ignore[misc]


def test_to_dict_is_json_serializable():
    first_treat = {u: 3 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 6), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    payload = profile.to_dict()
    as_json = json.dumps(payload)
    roundtripped = json.loads(as_json)
    assert roundtripped["treatment_type"] == "binary_absorbing"
    assert set(roundtripped.keys()) >= {
        "n_units",
        "n_periods",
        "n_obs",
        "is_balanced",
        "observation_coverage",
        "treatment_type",
        "is_staggered",
        "n_cohorts",
        "cohort_sizes",
        "has_never_treated",
        "has_always_treated",
        "treatment_varies_within_unit",
        "first_treatment_period",
        "last_treatment_period",
        "min_pre_periods",
        "min_post_periods",
        "outcome_dtype",
        "outcome_is_binary",
        "outcome_has_zeros",
        "outcome_has_negatives",
        "outcome_missing_fraction",
        "outcome_summary",
        "outcome_shape",
        "treatment_dose",
        "alerts",
    }


def test_alerts_are_factual_no_recommender_language():
    first_treat = {u: 1 for u in range(11, 21)}
    df = _make_panel(n_units=12, periods=[0, 1, 2, 3], first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    forbidden_substrings = (
        "recommend",
        "should use",
        "use estimator",
        "we suggest",
        "you should",
    )
    for alert in profile.alerts:
        lowered = alert.message.lower()
        for phrase in forbidden_substrings:
            assert phrase not in lowered, (
                f"alert {alert.code!r} contains recommender-adjacent phrase "
                f"{phrase!r} in message: {alert.message!r}"
            )


def test_alert_dataclass_is_frozen():
    a = Alert(code="x", severity="info", message="m", observed=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.code = "y"  # type: ignore[misc]


def test_all_zero_treatment_is_binary_absorbing():
    """Degenerate binary: no unit is ever treated. Must classify as binary,
    not continuous, so the documented taxonomy matches the implementation."""
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=None)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is False
    assert profile.cohort_sizes == {}
    assert profile.n_cohorts == 0


def test_all_one_treatment_is_binary_absorbing_always_treated():
    """Degenerate binary: every unit treated in every period. Must classify as
    binary_absorbing with has_always_treated=True."""
    rows = []
    for u in range(1, 21):
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": 1, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.has_never_treated is False
    assert profile.has_always_treated is True
    codes = _alert_codes(profile)
    assert "has_always_treated_units" in codes


def test_binary_with_nans_only_zeros_observed_is_binary():
    """Binary panel with some NaNs and only 0 observed among non-NaN values —
    still classify as binary, not continuous."""
    rows = []
    for u in range(1, 11):
        for t in range(4):
            tr = 0 if (u + t) % 2 == 0 else np.nan
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"


def test_all_nan_treatment_is_categorical():
    """Treatment column entirely NaN — classify as categorical (no info)."""
    rows = []
    for u in range(1, 11):
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": np.nan, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "categorical"


def test_top_level_import_surface():
    """profile_panel, PanelProfile, Alert, OutcomeShape, and
    TreatmentDoseShape must be importable from the top-level namespace
    so `help(diff_diff)` points at real symbols."""
    import diff_diff

    assert callable(diff_diff.profile_panel)
    assert diff_diff.PanelProfile.__name__ == "PanelProfile"
    assert diff_diff.Alert.__name__ == "Alert"
    assert diff_diff.OutcomeShape.__name__ == "OutcomeShape"
    assert diff_diff.TreatmentDoseShape.__name__ == "TreatmentDoseShape"
    for name in (
        "profile_panel",
        "PanelProfile",
        "Alert",
        "OutcomeShape",
        "TreatmentDoseShape",
    ):
        assert name in diff_diff.__all__, f"{name} missing from __all__"


def test_duplicate_unit_time_rows_do_not_inflate_coverage():
    """Duplicate (unit, time) rows must not make a panel look balanced.
    observation_coverage must stay in [0, 1] and derive from the unique
    (unit, time) support, and the duplicate_unit_time_rows alert fires."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_dup = pd.concat([df, df.iloc[:5].copy()], ignore_index=True)
    profile = profile_panel(df_dup, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.is_balanced is True
    assert 0.0 <= profile.observation_coverage <= 1.0
    assert "duplicate_unit_time_rows" in _alert_codes(profile)

    df_missing_cell = df.drop(df.index[0]).reset_index(drop=True)
    df_dup_missing = pd.concat(
        [df_missing_cell, df_missing_cell.iloc[:5].copy()], ignore_index=True
    )
    profile2 = profile_panel(df_dup_missing, unit="u", time="t", treatment="tr", outcome="y")
    assert profile2.is_balanced is False
    assert profile2.observation_coverage < 1.0
    assert "duplicate_unit_time_rows" in _alert_codes(profile2)


def test_reversal_through_nan_is_binary_non_absorbing():
    """A 0 -> 1 -> NaN -> 0 path must be detected as non-absorbing: the
    observed non-NaN subsequence violates weak monotonicity. Previously a
    NaN-inclusive diff could report False monotonicity violation."""
    rows = []
    for u in range(1, 11):
        treat_seq = [0, 1, np.nan, 0]
        for t, tr in enumerate(treat_seq):
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_non_absorbing"


def test_continuous_zero_dose_controls_flag_has_never_treated():
    """Continuous treatment with some zero-dose units must flag
    has_never_treated=True. Previously continuous panels hardcoded
    has_never_treated=False regardless of control availability.
    has_always_treated has binary-only semantics and must remain
    False on continuous panels regardless of dose positivity."""
    rows = []
    rng = np.random.default_rng(0)
    for u in range(1, 21):
        dose = 0.0 if u <= 5 else float(rng.uniform(0.5, 3.0))
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": rng.normal()})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.has_never_treated is True
    assert profile.has_always_treated is False


def test_continuous_negative_then_zero_does_not_count_as_never_treated():
    """Regression: on continuous panels with negative dose support, a
    unit path like `[-1, 0]` has `groupby(unit).max() == 0` but is NOT
    a zero-dose never-treated control (`-1` is non-zero). The implementation
    must check both endpoints (`unit_max == 0` AND `unit_min == 0`) so
    `has_never_treated` matches the documented contract: "some unit has
    treatment == 0 in every observed non-NaN row." The autonomous guide's
    ContinuousDiD preflight uses this field as the `P(D=0) > 0` proxy, so
    a false-positive here would silently mislead an agent about control
    availability."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(1, 21):
        for t in range(4):
            if u <= 5:
                # Mixed negative/zero path - max is 0 but min < 0;
                # must NOT count as never-treated.
                dose = -1.0 if t < 2 else 0.0
            else:
                dose = float(rng.uniform(0.5, 3.0))
            rows.append({"u": u, "t": t, "tr": dose, "y": float(rng.normal())})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "continuous"
    assert profile.has_never_treated is False, (
        "Unit path [-1, -1, 0, 0] has unit_max == 0 but is not a "
        "zero-dose never-treated control (some rows are negative). "
        "Profile must enforce the documented contract: 'treatment == 0 "
        "in every observed non-NaN row' (i.e., unit_min == unit_max == 0)."
    )


def test_guide_api_strings_resolve_against_public_api():
    """Sanity-check that every estimator referenced in the autonomous guide
    exists in the public API, plus the `hausman_pretest` classmethod location
    and the `not_yet_treated` control-group string. Guards against guide
    drift that the CI reviewer has previously flagged."""
    import diff_diff
    from diff_diff import get_llm_guide

    text = get_llm_guide("autonomous")

    for name in (
        "DifferenceInDifferences",
        "MultiPeriodDiD",
        "TwoWayFixedEffects",
        "CallawaySantAnna",
        "SunAbraham",
        "ChaisemartinDHaultfoeuille",
        "ImputationDiD",
        "TwoStageDiD",
        "StackedDiD",
        "WooldridgeDiD",
        "EfficientDiD",
        "SyntheticDiD",
        "TROP",
        "TripleDifference",
        "StaggeredTripleDifference",
        "ContinuousDiD",
        "HeterogeneousAdoptionDiD",
    ):
        assert name in text, f"estimator {name!r} missing from guide"
        assert hasattr(diff_diff, name), f"{name!r} in guide but not exported"

    assert hasattr(
        diff_diff.EfficientDiD, "hausman_pretest"
    ), "EfficientDiD.hausman_pretest classmethod missing from the public API"

    assert "EfficientDiD.hausman_pretest" in text
    assert "Hausman.hausman_pretest" not in text

    assert 'control_group="not_yet_treated"' in text
    assert "notyettreated" not in text

    # HAD targets WAS / WAS_d_lower, not ATT; event-study is per-event-
    # time, not per-cohort. Guard against the guide drifting back to
    # ATT-shaped / per-cohort phrasing.
    assert "Weighted Average Slope (WAS)" in text
    assert "WAS_d_lower" in text
    assert "per-cohort Pierce-Schott" not in text

    # EfficientDiD has three paths when no never-treated exists:
    # PT-Post, PT-All, or control_group="last_cohort". The guide must
    # mention last_cohort in the no-never-treated section so agents do
    # not rule out the supported path.
    assert 'control_group="last_cohort"' in text

    # SunAbraham requires a never-treated cohort; the fit path raises a
    # ValueError when none exists. Guard the matrix / prose contract so
    # the guide cannot drift back to claiming SunAbraham is optional.
    sun_abraham_row = next(
        line for line in text.splitlines() if "`SunAbraham`" in line and "|" in line
    )
    cells = [cell.strip() for cell in sun_abraham_row.strip("|").split("|")]
    # Column order: estimator, binary_absorbing, staggered, continuous,
    # triple-diff, never-treated-required, covariate, few-treated,
    # heterogeneous-adoption, clustered-SE.
    assert cells[5] == "✓", (
        "SunAbraham matrix row must mark never-treated-required=✓ " f"(row: {sun_abraham_row!r})"
    )

    # HAD Assumption 3 is not testable per REGISTRY.md; the guide must
    # not claim otherwise.
    assert "Assumption 3" in text  # mentioned as untestable, not as validated
    assert "validate Assumptions 3 and 7" not in text
    assert "not testable" in text

    # EfficientDiD requires never-treated under BOTH assumption="PT-All"
    # and assumption="PT-Post" — PT-Post is not a "drop the requirement"
    # escape hatch. Only control_group="last_cohort" admits all-treated
    # panels. Guard against guide drift back to the incorrect wording.
    assert "PT-Post is the weaker" in text or "both" in text.lower()
    # The old claim "switch to `assumption=\"PT-Post\"` to drop" must
    # not reappear in any form.
    assert 'switch to `assumption="PT-Post"` to drop' not in text

    # Matrix covariate cells: SyntheticDiD accepts fit(covariates=...)
    # and residualizes the outcome; ContinuousDiD.fit has no covariate
    # surface. Guard the matrix rows against drift.
    sdid_row = next(line for line in text.splitlines() if "`SyntheticDiD`" in line and "|" in line)
    sdid_cells = [c.strip() for c in sdid_row.strip("|").split("|")]
    assert sdid_cells[6] in ("✓", "partial"), (
        "SyntheticDiD covariate-adjustment cell must be ✓ or partial "
        f"(residualization path exists); got {sdid_cells[6]!r}"
    )
    cdid_row = next(line for line in text.splitlines() if "`ContinuousDiD`" in line and "|" in line)
    cdid_cells = [c.strip() for c in cdid_row.strip("|").split("|")]
    assert cdid_cells[6] == "✗", (
        "ContinuousDiD covariate-adjustment cell must be ✗ "
        f"(no covariate surface on fit()); got {cdid_cells[6]!r}"
    )

    # §5 API signatures: compute_pretrends_power takes a fitted results
    # object (not df), plot_sensitivity takes SensitivityResults,
    # plot_honest_event_study takes HonestDiDResults. Guard against
    # drift back to the df-first / results-only signatures.
    assert "`compute_pretrends_power(results" in text
    assert "`plot_sensitivity(sensitivity_results" in text
    assert "`plot_honest_event_study(honest_results" in text

    # §6 BR/DR schema alignment. The emitted top-level keys are
    # singular / underscored ("assumption", "pre_trends", "sample"),
    # not the plural / run-together variants. DiagnosticReport emits
    # sections at the top level (not nested under a "checks" dict)
    # and uses "estimator" (the string class name) / "headline_metric"
    # / "estimator_native_diagnostics". Guard each real key and
    # forbid the obsolete ones.
    for real_key in (
        "`assumption: dict`",
        "`pre_trends: dict`",
        "`sample: dict`",
        "`headline_metric: dict`",
        "`estimator_native_diagnostics: dict`",
        "`overall_interpretation: str`",
    ):
        assert real_key in text, f"BR/DR §6 missing real key: {real_key}"
    for obsolete_key in (
        "`assumptions: dict`",
        "`pretrends: dict`",
        "`main_result: dict`",
        "`sample_summary: dict`",
        "`estimator_type: str`",
        "`checks: dict`",
    ):
        assert obsolete_key not in text, f"BR/DR §6 still lists obsolete key: {obsolete_key}"

    # BR `diagnostics` is a wrapper (status + schema/reason + possibly
    # overall_interpretation), not the DR payload directly. Guard the
    # wrapper wording so the guide does not drift back to telling
    # agents to parse BR["diagnostics"] as the DR schema.
    assert 'diagnostics["schema"]' in text
    # target_parameter includes a `reference` field per
    # describe_target_parameter(); guard its documentation.
    assert "`reference` (REGISTRY.md citation string)" in text

    # Methodology source attribution: EfficientDiD is Chen, Sant'Anna,
    # Xie (2025), not Arkhangelsky-Imbens. ContinuousDiD is Callaway,
    # Goodman-Bacon, Sant'Anna (2024). Guard both attributions in the
    # §4 prose and the §7 citation list.
    assert "Chen, Sant'Anna, Xie 2025" in text
    assert "(Arkhangelsky-Imbens)" not in text
    assert "Callaway, Goodman-Bacon, Sant'Anna 2024" in text
    # ContinuousDiD prose must distinguish the PT vs SPT identified
    # targets rather than collapsing everything into "ACR".
    assert "ATT(d|d)" in text
    assert "ACRT" in text
    assert "Strong Parallel Trends" in text

    # ContinuousDiD requires zero-dose (P(D=0) > 0) because Remark 3.1
    # lowest-dose-as-control is unimplemented; matrix col 5 must be ✓.
    assert cdid_cells[5] == "✓", (
        "ContinuousDiD matrix row must mark never-treated-required=✓ "
        f"(P(D=0) > 0 required per Remark 3.1); got {cdid_cells[5]!r}"
    )
    assert "P(D=0) > 0" in text or "P(D=0) &gt; 0" in text

    # ContinuousDiD DOES support staggered adoption natively (via the
    # `first_treat` column). Matrix column 2 (staggered) must be ✓.
    assert cdid_cells[2] == "✓", (
        "ContinuousDiD matrix row must mark staggered=✓ "
        "(adoption timing via first_treat is supported); "
        f"got {cdid_cells[2]!r}"
    )

    # ContinuousDiD also requires dose to be time-invariant per unit;
    # this is the second eligibility prerequisite the guide must spell
    # out. Guide text must mention the invariant explicitly AND the
    # `treatment_varies_within_unit` field used to detect it.
    assert "time-invariant" in text
    assert "treatment_varies_within_unit" in text

    # DR §6 section statuses: execution-state vocabulary must include
    # the actual emitted values ("ran", "not_applicable", "not_run",
    # "no_scalar_by_design", "skipped"), and `verdict` must be
    # documented separately from `status`. Guard against drift back
    # to the pass/warn/inconclusive-as-status framing.
    for real_status in (
        '"ran"',
        '"not_applicable"',
        '"not_run"',
        '"no_scalar_by_design"',
    ):
        assert real_status in text, f"DR §6 section-status vocabulary must document {real_status}"
    # `status` must not be described as "pass/warn/inconclusive" —
    # those belong under `verdict`.
    assert '`"pass"` / `"warn"` / `"inconclusive"`' not in text
    assert "verdict" in text.lower()

    # Balanced-panel eligibility: ContinuousDiD, EfficientDiD,
    # SyntheticDiD, and HeterogeneousAdoptionDiD all hard-reject
    # unbalanced panels at fit() time. The guide must surface this
    # so agents gate these estimators on PanelProfile.is_balanced
    # before selecting them.
    assert "is_balanced" in text, (
        "Guide must mention PanelProfile.is_balanced as an eligibility "
        "check for balance-sensitive estimators"
    )
    for estimator in (
        "ContinuousDiD",
        "EfficientDiD",
        "SyntheticDiD",
        "HeterogeneousAdoptionDiD",
        "StaggeredTripleDifference",
    ):
        idx = 0
        found = False
        while idx < len(text):
            loc = text.find(estimator, idx)
            if loc < 0:
                break
            window = text[max(0, loc - 400) : loc + 400]
            if "balanced" in window.lower() or "is_balanced" in window:
                found = True
                break
            idx = loc + 1
        assert found, (
            f"Guide must mention a balanced-panel constraint near the "
            f"{estimator!r} bullet / row (hard-rejects unbalanced panels "
            "at fit time)"
        )

    # HeterogeneousAdoptionDiD staggered support is `partial` and
    # specifically last-cohort-only (Appendix B.2): with first_treat
    # supplied, fit() auto-filters to F_last + never-treated; without
    # first_treat, a multi-cohort panel raises. Guide must surface
    # this explicitly so agents don't route a general staggered panel
    # to HAD expecting a multi-cohort estimand.
    assert "last-cohort-only" in text or "last cohort" in text.lower(), (
        "Guide must name the last-cohort-only restriction on HAD "
        "staggered support (Appendix B.2)"
    )
    assert "first_treat" in text, (
        "Guide must mention that first_treat is required to activate "
        "HAD's staggered last-cohort auto-filter"
    )
    assert "ChaisemartinDHaultfoeuille" in text, (
        "Guide must point at ChaisemartinDHaultfoeuille as the fallback "
        "for full staggered support"
    )

    # Balanced-panel gate is incomplete with `is_balanced` alone because
    # duplicate (unit, time) rows don't flip is_balanced. Guide must
    # require BOTH is_balanced == True AND absence of the
    # duplicate_unit_time_rows alert before routing to the duplicate-
    # intolerant estimators (ContinuousDiD silently overwrites
    # duplicates via last-row-wins; EfficientDiD/HAD raise).
    assert "duplicate_unit_time_rows" in text, (
        "Guide must name the duplicate_unit_time_rows alert as part of "
        "the balanced-panel eligibility gate"
    )
    assert "BOTH" in text or "both" in text, (
        "Guide must require BOTH is_balanced and absence of the "
        "duplicate_unit_time_rows alert before routing to duplicate-"
        "intolerant estimators"
    )

    # ChaisemartinDHaultfoeuille handles non-absorbing / reversible
    # treatment; SUTVA is still assumed (no native interference or
    # spillover support per REGISTRY.md). Guard against the guide
    # drifting back to advertising dCDH as "robust to spillover
    # designs" or similar.
    for phrase in (
        "robust to spillover",
        "interference-robust",
        "supports spillover",
        "and to spillover",
    ):
        assert phrase not in text, (
            f"Guide must not advertise unsupported dCDH capability "
            f"{phrase!r}: SUTVA is assumed across the estimator suite."
        )

    # Repeated-cross-section (§4.10) must not claim broad
    # applicability. The documented RCS-capable estimators are
    # CallawaySantAnna(panel=False), TripleDifference, and
    # StaggeredTripleDifference; EfficientDiD and
    # HeterogeneousAdoptionDiD explicitly reject RCS per REGISTRY.md.
    assert "most estimators remain applicable" not in text, (
        "§4.10 must not claim broad RCS applicability; only the "
        "explicitly documented RCS-capable subset is applicable."
    )
    assert "panel=False" in text, (
        "§4.10 must point at CallawaySantAnna(panel=False) as the " "explicit RCS mode"
    )
    # The section must explicitly name at least one panel-only
    # estimator as rejected for RCS, so agents do not silently route
    # RCS data to it.
    rcs_section_start = text.find("§4.10 Repeated cross-sections")
    assert rcs_section_start >= 0
    rcs_section = text[rcs_section_start : rcs_section_start + 2500]
    for panel_only in (
        "EfficientDiD",
        "HeterogeneousAdoptionDiD",
        "StaggeredTripleDifference",
    ):
        assert panel_only in rcs_section, (
            f"§4.10 must explicitly name {panel_only!r} as panel-only "
            "so RCS data is not routed to it"
        )

    # The explicit RCS-capable bullet list must NOT put
    # StaggeredTripleDifference next to the RCS-support language.
    # The estimator has no panel=False mode and fit() rejects
    # unbalanced input; only TripleDifference (non-staggered) is
    # cross-sectional-DDD-capable.
    explicit_support_block = text.find("Explicit RCS support", rcs_section_start)
    rejected_block = text.find("Explicitly rejected for RCS", rcs_section_start)
    assert 0 <= explicit_support_block < rejected_block, (
        "§4.10 must separate an Explicit RCS support list from the " "Explicitly rejected list"
    )
    explicit_segment = text[explicit_support_block:rejected_block]
    assert "StaggeredTripleDifference" not in explicit_segment, (
        "StaggeredTripleDifference must NOT appear in the Explicit RCS "
        "support list — it is panel-only and balance-enforced."
    )


def test_min_pre_post_use_per_unit_observed_support():
    """On an unbalanced panel where one treated unit is missing its
    earliest pre-period, min_pre_periods must reflect that unit's actual
    observed support. Previously _compute_pre_post used the global period
    set, which could hide short-panel cases and suppress the short_pre_panel
    alert."""
    rows = []
    for u in range(1, 21):
        first_treat = 3
        for t in range(0, 6):
            if u == 1 and t <= 1:
                continue
            tr = 1 if t >= first_treat else 0
            rows.append({"u": u, "t": t, "tr": tr, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.min_pre_periods == 1
    assert "short_pre_panel" in _alert_codes(profile)


def test_missing_unit_or_time_ids_are_dropped_consistently():
    """NaN values in unit or time must not push observation_coverage above
    1.0. `nunique()` drops NaN while `drop_duplicates()` keeps NaN as a
    distinct key, which previously produced coverage > 1 silently. The
    fix drops NaN-id rows up front, emits the missing_id_rows_dropped
    alert, and computes all structural facts on the non-missing subset."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_with_missing = df.copy()
    df_with_missing.loc[[0, 1, 2], "u"] = np.nan
    df_with_missing.loc[[5, 6], "t"] = np.nan
    profile = profile_panel(df_with_missing, unit="u", time="t", treatment="tr", outcome="y")
    assert 0.0 <= profile.observation_coverage <= 1.0
    codes = _alert_codes(profile)
    assert "missing_id_rows_dropped" in codes
    drop_alert = next(a for a in profile.alerts if a.code == "missing_id_rows_dropped")
    assert drop_alert.observed == 5


def test_row_with_both_ids_missing_counted_once():
    """A row with BOTH unit and time NaN must count as one dropped row,
    not two. Previously `isna().sum()` summed the two columns and
    double-counted rows missing both identifiers."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    df_both_missing = df.copy()
    df_both_missing.loc[0, "u"] = np.nan
    df_both_missing.loc[0, "t"] = np.nan
    profile = profile_panel(df_both_missing, unit="u", time="t", treatment="tr", outcome="y")
    drop_alert = next(a for a in profile.alerts if a.code == "missing_id_rows_dropped")
    assert drop_alert.observed == 1


def test_empty_dataframe_raises_value_error():
    """Direct empty input must raise, not silently return a 'balanced'
    profile with zero units/periods."""
    df = pd.DataFrame({"u": [], "t": [], "tr": [], "y": []})
    with pytest.raises(ValueError, match="empty"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")


def test_empty_after_id_drop_raises_value_error():
    """If every row has a missing unit or time identifier, the panel is
    empty after the drop; raise rather than returning is_balanced=True
    on zero rows."""
    df = pd.DataFrame(
        {
            "u": [np.nan, np.nan],
            "t": [0, 1],
            "tr": [0, 1],
            "y": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="no rows remain"):
        profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")


# ---------------------------------------------------------------------------
# Outcome shape (numeric outcome distributional facts)
# ---------------------------------------------------------------------------


def test_outcome_shape_count_like_poisson():
    """A Poisson-distributed outcome (integer-valued, has zeros, right-skewed)
    must satisfy is_count_like=True so an agent can gate WooldridgeDiD over
    linear OLS pre-fit. Uses Poisson(lambda=0.5) on a 200-unit panel for a
    reliably-skewed empirical sample (theoretical skew = 1/sqrt(0.5) ~ 1.41)."""
    rng = np.random.default_rng(7)
    first_treat = {u: 2 for u in range(101, 201)}
    df = _make_panel(
        n_units=200,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: int(rng.poisson(0.5 + 0.2 * tr)),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.is_integer_valued is True
    assert shape.is_count_like is True
    assert shape.pct_zeros > 0.1
    assert shape.skewness is not None and shape.skewness > 0.5
    assert shape.n_distinct_values >= 3


def test_outcome_shape_binary_outcome_not_count_like():
    """Binary 0/1 outcome must NOT trigger is_count_like (only 2 distinct
    values; skewness gate fires None) and is_bounded_unit must be True."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(
        n_units=20,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: int(tr),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.is_count_like is False
    assert shape.is_bounded_unit is True
    assert shape.skewness is None
    assert shape.excess_kurtosis is None
    assert shape.n_distinct_values == 2


def test_outcome_shape_count_like_excludes_negative_support():
    """`is_count_like` is the routing signal toward
    `WooldridgeDiD(method="poisson")`, which raises `ValueError` on
    negative outcomes (`wooldridge.py:1105`). The heuristic must
    therefore gate on `value_min >= 0`: a right-skewed integer outcome
    with zeros AND some negative values must NOT set `is_count_like`,
    even though the other four conditions (integer-valued, has zeros,
    skewness > 0.5, > 2 distinct values) are satisfied. Otherwise the
    autonomous-guide §5.3 reasoning chain would steer agents toward an
    estimator that will then refuse to fit the panel."""
    rng = np.random.default_rng(37)
    first_treat = {u: 2 for u in range(101, 201)}

    # Mix Poisson with a small share of negative integers - integer-valued,
    # has zeros (Poisson(0.5) yields ~60% zeros), and right-skewed; but
    # value_min < 0 should kill is_count_like.
    def _outcome_fn(u, t, tr, _rng):
        base = int(rng.poisson(0.5 + 0.2 * tr))
        # 5% of cells become a small negative integer, e.g. -1 or -2.
        if rng.random() < 0.05:
            return -int(rng.integers(1, 3))
        return base

    df = _make_panel(
        n_units=200,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=_outcome_fn,
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.value_min < 0, "fixture must include some negative outcomes"
    assert shape.is_integer_valued is True
    assert shape.pct_zeros > 0.1
    assert shape.skewness is not None
    assert shape.is_count_like is False, (
        "is_count_like must be False when value_min < 0; otherwise an "
        "agent following the §5.3 worked example would route the panel "
        "to WooldridgeDiD(method='poisson'), which raises ValueError on "
        "negative outcomes."
    )


def test_outcome_shape_continuous_normal():
    """Normally-distributed float outcome must have is_integer_valued=False,
    is_count_like=False, is_bounded_unit=False (signed values exit [0,1])."""
    rng = np.random.default_rng(11)
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(
        n_units=20,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: float(rng.normal(0.0, 1.0)),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.is_integer_valued is False
    assert shape.is_count_like is False
    assert shape.is_bounded_unit is False
    assert shape.skewness is not None
    assert shape.excess_kurtosis is not None


def test_outcome_shape_bounded_unit():
    """Outcome confined to [0, 1] (e.g., a proportion / probability) must
    set is_bounded_unit=True so an agent can flag the bounded-support
    consideration when interpreting linear-OLS estimates."""
    rng = np.random.default_rng(13)
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(
        n_units=20,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: float(rng.uniform(0.0, 1.0)),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.is_bounded_unit is True
    assert 0.0 <= shape.value_min <= shape.value_max <= 1.0


def test_outcome_shape_categorical_returns_none():
    """Object-dtype outcome (string labels) must return outcome_shape=None;
    skewness/kurtosis are not defined for non-numeric data."""
    first_treat = {u: 2 for u in range(11, 21)}
    rows = []
    for u in range(1, 21):
        for t in range(4):
            tr = 1 if (u in first_treat and t >= first_treat[u]) else 0
            rows.append({"u": u, "t": t, "tr": tr, "y": "high" if tr else "low"})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.outcome_shape is None


def test_outcome_shape_skewness_kurtosis_gated_on_distinct_values():
    """Outcomes with fewer than 3 distinct values must report
    skewness=None and excess_kurtosis=None — moments are undefined or
    degenerate at that distinctness floor."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(
        n_units=20,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: float(tr),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    shape = profile.outcome_shape
    assert shape is not None
    assert shape.n_distinct_values == 2
    assert shape.skewness is None
    assert shape.excess_kurtosis is None


def test_outcome_shape_to_dict_roundtrips_through_json():
    """outcome_shape must serialize to a JSON-compatible nested dict and
    survive ``json.loads(json.dumps(...))`` without loss."""
    rng = np.random.default_rng(17)
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(
        n_units=20,
        periods=range(0, 4),
        first_treat=first_treat,
        outcome_fn=lambda u, t, tr, _rng: int(rng.poisson(2.0)),
    )
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    payload = profile.to_dict()
    roundtrip = json.loads(json.dumps(payload))
    shape_dict = roundtrip["outcome_shape"]
    assert shape_dict is not None
    assert set(shape_dict.keys()) == {
        "n_distinct_values",
        "pct_zeros",
        "value_min",
        "value_max",
        "skewness",
        "excess_kurtosis",
        "is_integer_valued",
        "is_count_like",
        "is_bounded_unit",
    }


# ---------------------------------------------------------------------------
# Treatment dose (continuous-treatment distributional facts)
# ---------------------------------------------------------------------------


def test_treatment_dose_continuous_zero_baseline():
    """Continuous panel with zero-dose controls and multiple distinct
    treated doses populates treatment_dose with has_zero_dose=True and
    the correct n_distinct_doses + dose support."""
    rng = np.random.default_rng(19)
    rows = []
    for u in range(1, 21):
        # 5 zero-dose units, 15 with one of three nonzero dose levels
        if u <= 5:
            dose = 0.0
        elif u <= 10:
            dose = 1.0
        elif u <= 15:
            dose = 2.5
        else:
            dose = 4.0
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": float(rng.normal())})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    dose = profile.treatment_dose
    assert profile.treatment_type == "continuous"
    assert dose is not None
    assert dose.has_zero_dose is True
    assert dose.n_distinct_doses == 4  # {0, 1, 2.5, 4}
    assert dose.dose_min == pytest.approx(1.0)
    assert dose.dose_max == pytest.approx(4.0)


def test_treatment_dose_descriptive_fields_supplement_existing_gates():
    """Regression: most TreatmentDoseShape fields are descriptive
    distributional context. In the canonical ContinuousDiD setup
    (Callaway, Goodman-Bacon, Sant'Anna 2024) the dose `D_i` is
    time-invariant per unit and `first_treat` is a separate
    caller-supplied column — `profile_panel` sees only the dose
    column. Profile-side screening on the dose column includes
    `treatment_varies_within_unit` (the actual fit-time gate; rules
    out `0,0,d,d` paths) and `has_never_treated` (predicts
    `P(D=0) > 0` under the canonical convention that `first_treat ==
    0` ties to `D_i == 0`). The two cases below exercise those
    signals:

    1. `0,0,d,d` within-unit dose path: a single unit toggles between
       zero (pre-treatment) and a single nonzero dose `d` (post). The
       PanelProfile.treatment_varies_within_unit field correctly fires
       True. This IS the actual fit-time gate (line 222-228 of
       continuous_did.py uses `df.groupby(unit)[dose].nunique() > 1`
       independent of `first_treat`). The canonical ContinuousDiD
       setup expects time-invariant per-unit dose, so a `0,0,d,d`
       path is incompatible regardless of how `first_treat` is
       constructed.
    2. Row-level zeros without never-treated: every unit eventually
       gets treated, but pre-treatment rows have dose=0. has_zero_dose
       fires True (row-level), while has_never_treated correctly fires
       False (unit-level). With a `first_treat` column consistent
       with the dose column on per-unit treated/untreated status,
       `ContinuousDiD.fit()` will reject the panel via
       `n_control == 0`. `ContinuousDiD` as currently implemented
       does not apply (Remark 3.1 lowest-dose-as-control is not
       implemented). Routing alternatives that do not require
       `P(D=0) > 0` include linear DiD with the treatment as a
       continuous covariate and `HeterogeneousAdoptionDiD` for
       graded-adoption designs. Re-encoding the treatment column
       is an agent-side preprocessing choice that changes the
       estimand and is not documented in REGISTRY. Do not relabel
       units to manufacture controls via the force-zero coercion
       path."""
    # Case 1: 0,0,d,d within-unit path
    rows1 = []
    for u in range(1, 21):
        for t in range(4):
            if u <= 5:
                dose = 0.0  # never-treated controls
            else:
                # 0,0,2.5,2.5 path - constant nonzero dose, but full
                # path nunique == 2
                dose = 0.0 if t < 2 else 2.5
            rows1.append({"u": u, "t": t, "tr": dose, "y": 0.0})
    df1 = pd.DataFrame(rows1)
    profile1 = profile_panel(df1, unit="u", time="t", treatment="tr", outcome="y")
    assert profile1.treatment_type == "continuous"
    assert profile1.treatment_varies_within_unit is True, (
        "treatment_varies_within_unit must fire True on `0,0,d,d` paths "
        "(matching ContinuousDiD.fit() unit-level full-path nunique check at "
        "line 222-228 of continuous_did.py); this is an actual fit-time gate "
        "that holds independent of `first_treat`."
    )

    # Case 2: row-level zeros, no never-treated units
    rows2 = []
    for u in range(1, 21):
        # Every unit eventually treated; pre-treatment dose=0, post-treatment
        # constant nonzero dose. No unit is never-treated.
        adopt_period = 1 if u <= 10 else 2
        for t in range(4):
            dose = 1.5 if t >= adopt_period else 0.0
            rows2.append({"u": u, "t": t, "tr": dose, "y": 0.0})
    df2 = pd.DataFrame(rows2)
    profile2 = profile_panel(df2, unit="u", time="t", treatment="tr", outcome="y")
    assert profile2.treatment_type == "continuous"
    dose2 = profile2.treatment_dose
    assert dose2 is not None
    assert dose2.has_zero_dose is True, "row-level zero dose rows are present"
    assert profile2.has_never_treated is False, (
        "every unit eventually treated -> has_never_treated must be False, "
        "even though row-level has_zero_dose==True. Under the canonical "
        "ContinuousDiD setup, has_never_treated==False signals to the "
        "agent that no `first_treat == 0` units exist; ContinuousDiD.fit() "
        "will reject the panel via `n_control == 0`. ContinuousDiD as "
        "currently implemented does not apply on this panel; routing "
        "alternatives include HeterogeneousAdoptionDiD or linear DiD with "
        "a continuous covariate. Re-encoding is an agent-side preprocessing "
        "choice not documented in REGISTRY as a supported fallback."
    )


def test_treatment_dose_min_flags_negative_dose_continuous_panels():
    """Regression: a balanced, never-treated, time-invariant continuous
    panel with negative non-zero treated doses fails the canonical-
    setup `dose_min > 0` preflight check. The canonical ContinuousDiD
    setup uses time-invariant per-unit dose `D_i` and a separate
    caller-supplied `first_treat` column; with a `first_treat`
    consistent with the dose column on per-unit treated/untreated
    status (negative-dose units labeled `first_treat > 0`),
    `ContinuousDiD.fit()` would raise `ValueError` at
    `continuous_did.py:586-593` ("Dose must be strictly positive for
    treated units (D > 0)"). `ContinuousDiD` as currently
    implemented does not apply on this panel. `HeterogeneousAdoptionDiD`
    is also NOT a routing alternative here: HAD requires non-negative
    dose support (`had.py:1563-1572`, paper Section 2). The applicable
    alternative on the negative-dose branch is linear DiD with the
    treatment as a signed continuous covariate. Re-encoding the
    treatment to a non-negative scale is an agent-side preprocessing
    choice that changes the estimand and is not documented in
    REGISTRY as a supported fallback. The force-zero coercion path
    on `first_treat == 0` rows is implementation behavior for
    inconsistent inputs and is not a documented routing option. This
    test asserts that the profile correctly surfaces `dose_min < 0`
    so an agent can choose an applicable alternative before reaching
    `fit()`."""
    rng = np.random.default_rng(41)
    rows = []
    for u in range(1, 21):
        # 5 zero-dose (never-treated) units, 15 with one of three
        # NEGATIVE non-zero dose levels held constant across periods.
        if u <= 5:
            dose = 0.0
        elif u <= 10:
            dose = -1.0
        elif u <= 15:
            dose = -2.5
        else:
            dose = -4.0
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": float(rng.normal())})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    # All other canonical-setup preflight checks pass:
    assert profile.treatment_type == "continuous"
    assert profile.has_never_treated is True
    assert profile.treatment_varies_within_unit is False
    assert profile.is_balanced is True
    assert "duplicate_unit_time_rows" not in {a.code for a in profile.alerts}
    # But dose_min > 0 fails: under the canonical ContinuousDiD setup
    # (per-unit time-invariant dose + separate first_treat with
    # negative-dose units labeled first_treat > 0), fit() would raise
    # at line 287-294 ("Dose must be strictly positive for treated
    # units"). ContinuousDiD as currently implemented does not apply.
    # HeterogeneousAdoptionDiD is also NOT a routing alternative on
    # this branch: HAD requires non-negative dose support (had.py:
    # 1450-1459, paper Section 2). The applicable alternative is
    # linear DiD with the treatment as a signed continuous covariate.
    # Re-encoding is an agent-side preprocessing choice not
    # documented in REGISTRY as a fallback. Relabeling-to-
    # first_treat==0 is not a documented routing option.
    dose = profile.treatment_dose
    assert dose is not None
    assert dose.dose_min < 0, (
        "Fixture must have negative dose_min so the canonical-setup "
        "preflight check (`dose_min > 0`) correctly fires False on "
        "this panel."
    )


def test_treatment_dose_continuous_no_zero_dose():
    """If every unit has a strictly positive dose throughout, has_zero_dose
    must be False — descriptive flag for the absence of dose-zero rows in
    the panel."""
    rng = np.random.default_rng(29)
    rows = []
    for u in range(1, 21):
        dose = float(rng.uniform(0.5, 3.0))
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": float(rng.normal())})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    dose = profile.treatment_dose
    assert dose is not None
    assert dose.has_zero_dose is False


def test_treatment_dose_binary_treatment_returns_none():
    """Binary treatment must produce treatment_dose=None — dose semantics
    only apply to continuous treatments."""
    first_treat = {u: 2 for u in range(11, 21)}
    df = _make_panel(n_units=20, periods=range(0, 4), first_treat=first_treat)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "binary_absorbing"
    assert profile.treatment_dose is None


def test_treatment_dose_categorical_treatment_returns_none():
    """Categorical (object-dtype) treatment must produce treatment_dose=None."""
    rows = []
    for u in range(1, 11):
        arm = "A" if u <= 5 else "B"
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": arm, "y": float(u) + 0.1 * t})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    assert profile.treatment_type == "categorical"
    assert profile.treatment_dose is None


def test_treatment_dose_to_dict_roundtrips_through_json():
    """treatment_dose must serialize to a JSON-compatible nested dict."""
    rng = np.random.default_rng(31)
    rows = []
    for u in range(1, 21):
        dose = 0.0 if u <= 5 else 2.5
        for t in range(4):
            rows.append({"u": u, "t": t, "tr": dose, "y": float(rng.normal())})
    df = pd.DataFrame(rows)
    profile = profile_panel(df, unit="u", time="t", treatment="tr", outcome="y")
    payload = profile.to_dict()
    roundtrip = json.loads(json.dumps(payload))
    dose_dict = roundtrip["treatment_dose"]
    assert dose_dict is not None
    assert set(dose_dict.keys()) == {
        "n_distinct_doses",
        "has_zero_dose",
        "dose_min",
        "dose_max",
        "dose_mean",
    }


# ---------------------------------------------------------------------------
# Frozen invariants
# ---------------------------------------------------------------------------


def test_outcome_shape_dataclass_is_frozen():
    from diff_diff.profile import OutcomeShape

    s = OutcomeShape(
        n_distinct_values=3,
        pct_zeros=0.1,
        value_min=0.0,
        value_max=1.0,
        skewness=0.5,
        excess_kurtosis=0.0,
        is_integer_valued=False,
        is_count_like=False,
        is_bounded_unit=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_distinct_values = 999  # type: ignore[misc]


def test_panel_profile_direct_construction_without_wave2_fields():
    """`PanelProfile` is a public top-level type. Wave 2 added two
    optional fields (`outcome_shape`, `treatment_dose`) with `None`
    defaults so direct callers that instantiated `PanelProfile(...)`
    pre-Wave-2 do not break with `TypeError: missing keyword
    argument`. Verify direct construction without the new fields
    succeeds and yields the documented defaults."""
    profile = PanelProfile(
        n_units=2,
        n_periods=2,
        n_obs=4,
        is_balanced=True,
        observation_coverage=1.0,
        treatment_type="binary_absorbing",
        is_staggered=False,
        n_cohorts=1,
        cohort_sizes={1: 1},
        has_never_treated=True,
        has_always_treated=False,
        treatment_varies_within_unit=True,
        first_treatment_period=1,
        last_treatment_period=1,
        min_pre_periods=1,
        min_post_periods=1,
        outcome_dtype="float64",
        outcome_is_binary=False,
        outcome_has_zeros=False,
        outcome_has_negatives=False,
        outcome_missing_fraction=0.0,
        outcome_summary={"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5},
        alerts=(),
    )
    assert profile.outcome_shape is None
    assert profile.treatment_dose is None
    # to_dict() must serialize the defaulted fields as None values
    payload = profile.to_dict()
    assert payload["outcome_shape"] is None
    assert payload["treatment_dose"] is None


def test_treatment_dose_dataclass_is_frozen():
    from diff_diff.profile import TreatmentDoseShape

    d = TreatmentDoseShape(
        n_distinct_doses=3,
        has_zero_dose=True,
        dose_min=1.0,
        dose_max=3.0,
        dose_mean=2.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.has_zero_dose = False  # type: ignore[misc]
