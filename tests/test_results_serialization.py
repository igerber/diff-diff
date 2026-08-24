"""Serialization contract for the Phase 2 results foundation (spec section 5).

- The seven newly-added ``to_dict()`` methods emit canonical keys, values
  equal to the property surface, and JSON-serializable scalars.
- Every estimator results class subclasses ``BaseResults``; no class is
  both ``BaseResults`` and ``Diagnostic``; ``EventStudyResults`` is
  ``BaseResults`` and not ``Diagnostic``.
"""

import json
import warnings

import numpy as np
import pytest

import diff_diff
from diff_diff import BaseResults, Diagnostic, EventStudyResults

# Estimator results classes that gained to_dict() in this PR.
NEW_TO_DICT = [
    "CallawaySantAnnaResults",
    "StackedDiDResults",
    "ContinuousDiDResults",
    "SunAbrahamResults",
    "WooldridgeDiDResults",
    "ChaisemartinDHaultfoeuilleResults",
    "BaconDecompositionResults",
]

# Every public estimator results class (must be BaseResults, never Diagnostic).
ESTIMATOR_RESULTS = [
    "DiDResults",
    "SpilloverDiDResults",
    "MultiPeriodDiDResults",
    "SyntheticDiDResults",
    "CallawaySantAnnaResults",
    "DMLDiDResults",
    "SunAbrahamResults",
    "ImputationDiDResults",
    "TwoStageDiDResults",
    "StackedDiDResults",
    "EfficientDiDResults",
    "ContinuousDiDResults",
    "WooldridgeDiDResults",
    "ChaisemartinDHaultfoeuilleResults",
    "LPDiDResults",
    "StaggeredTripleDiffResults",
    "TripleDifferenceResults",
    "TROPResults",
    "SyntheticControlResults",
    "ChangesInChangesResults",
    "HeterogeneousAdoptionDiDResults",
    "HeterogeneousAdoptionDiDEventStudyResults",
    "RegressionDiscontinuityResults",
]

CANONICAL_QUINTET_KEYS = {"att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"}


# ---------------------------------------------------------------------------
# to_dict() behavioral tests (real small fits)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_results():
    """One fitted instance per to_dict-gaining estimator (except Bacon fits
    its own staggered panel; all analytical)."""
    out = {}
    from diff_diff import (
        BaconDecomposition,
        CallawaySantAnna,
        StackedDiD,
        WooldridgeDiD,
    )
    from diff_diff.continuous_did import ContinuousDiD
    from diff_diff.datasets import load_mpdta
    from diff_diff.prep import generate_staggered_data as prep_staggered
    from diff_diff.prep_dgp import (
        generate_continuous_did_data,
    )
    from diff_diff.prep_dgp import generate_staggered_data as dgp_staggered
    from diff_diff.sun_abraham import SunAbraham

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cs_data = prep_staggered(
            n_units=80, n_periods=8, cohort_periods=[4], treatment_effect=2.0, seed=42
        )
        out["CallawaySantAnnaResults"] = CallawaySantAnna(n_bootstrap=0).fit(
            cs_data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )

        out["SunAbrahamResults"] = SunAbraham().fit(
            _sa_panel(), outcome="outcome", unit="unit", time="time", first_treat="first_treat"
        )

        st_data = dgp_staggered(
            n_units=120,
            n_periods=12,
            cohort_periods=[4, 6, 8],
            never_treated_frac=0.3,
            treatment_effect=5.0,
            dynamic_effects=True,
            seed=42,
        )
        out["StackedDiDResults"] = StackedDiD(kappa_pre=2, kappa_post=2).fit(
            st_data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )

        cd_data = generate_continuous_did_data(
            n_units=200, n_periods=5, cohort_periods=[2, 4], seed=42, noise_sd=0.5
        )
        out["ContinuousDiDResults"] = ContinuousDiD(
            control_group="not_yet_treated", n_bootstrap=0
        ).fit(cd_data, "outcome", "unit", "period", "first_treat", "dose")

        out["WooldridgeDiDResults"] = WooldridgeDiD().fit(
            load_mpdta(), outcome="lemp", unit="countyreal", time="year", first_treat="first_treat"
        )

        out["ChaisemartinDHaultfoeuilleResults"] = _dcdh_fit()

        out["BaconDecompositionResults"] = BaconDecomposition().fit(
            _bacon_panel(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
    return out


@pytest.mark.parametrize("name", NEW_TO_DICT)
def test_to_dict_returns_dict(name, fitted_results):
    d = fitted_results[name].to_dict()
    assert isinstance(d, dict) and d


@pytest.mark.parametrize("name", [n for n in NEW_TO_DICT if n != "BaconDecompositionResults"])
def test_to_dict_has_canonical_quintet(name, fitted_results):
    d = fitted_results[name].to_dict()
    assert CANONICAL_QUINTET_KEYS.issubset(d.keys()), (
        f"{name}.to_dict() missing canonical keys: " f"{CANONICAL_QUINTET_KEYS - set(d.keys())}"
    )


@pytest.mark.parametrize("name", [n for n in NEW_TO_DICT if n != "BaconDecompositionResults"])
def test_to_dict_values_match_property_surface(name, fitted_results):
    res = fitted_results[name]
    d = res.to_dict()
    assert d["att"] == pytest.approx(res.att, nan_ok=True)
    assert d["se"] == pytest.approx(res.se, nan_ok=True)
    assert d["conf_int_lower"] == pytest.approx(res.conf_int[0], nan_ok=True)
    assert d["conf_int_upper"] == pytest.approx(res.conf_int[1], nan_ok=True)


def test_bacon_to_dict_has_no_quintet(fitted_results):
    d = fitted_results["BaconDecompositionResults"].to_dict()
    assert "att" not in d
    assert "twfe_estimate" in d


@pytest.mark.parametrize("name", NEW_TO_DICT)
def test_to_dict_no_deprecated_names(name, fitted_results):
    # Serialization emits canonical names only (spec section 5): no
    # overall_*/avg_* keys leak into to_dict output.
    d = fitted_results[name].to_dict()
    leaked = [k for k in d if k.startswith("overall_") or k.startswith("avg_")]
    assert not leaked, f"{name}.to_dict() leaked deprecated keys: {leaked}"


@pytest.mark.parametrize("name", NEW_TO_DICT)
def test_to_dict_scalars_json_serializable(name, fitted_results):
    d = fitted_results[name].to_dict()
    # numpy scalar types are the common JSON offender; coerce and dump.
    coerced = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in d.items()
        if not isinstance(v, (list, dict, np.ndarray))
    }
    json.dumps(coerced)


# ---------------------------------------------------------------------------
# BaseResults / Diagnostic coverage (introspection - no fits)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ESTIMATOR_RESULTS)
def test_estimator_result_is_base_results(name):
    cls = getattr(diff_diff, name)
    assert issubclass(cls, BaseResults), f"{name} must subclass BaseResults"


def test_no_class_is_both_base_and_diagnostic():
    # Scan every public results/diagnostic class in the namespace.
    import diff_diff.diagnostic_report as drm

    both = []
    for name in dir(diff_diff):
        obj = getattr(diff_diff, name)
        if isinstance(obj, type) and issubclass(obj, (BaseResults, Diagnostic)):
            if issubclass(obj, BaseResults) and issubclass(obj, Diagnostic):
                both.append(name)
    # DiagnosticReportResults lives in diagnostic_report; check it too.
    drr = drm.DiagnosticReportResults
    if issubclass(drr, BaseResults) and issubclass(drr, Diagnostic):
        both.append("DiagnosticReportResults")
    assert not both, f"classes are BOTH BaseResults and Diagnostic: {both}"


def test_event_study_results_is_base_not_diagnostic():
    assert issubclass(EventStudyResults, BaseResults)
    assert not issubclass(EventStudyResults, Diagnostic)


# ---------------------------------------------------------------------------
# Panel builders (small, analytical)
# ---------------------------------------------------------------------------


def _sa_panel(n_units=80, n_periods=8, n_cohorts=3, seed=42):
    import pandas as pd

    np.random.seed(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    first_treat = np.zeros(n_units)
    if n_treated > 0:
        first_treat[n_never:] = cohort_periods[
            np.random.choice(len(cohort_periods), size=n_treated)
        ]
    fte = np.repeat(first_treat, n_periods)
    unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    post = (times >= fte) & (fte > 0)
    y = unit_fe + time_fe + 2.0 * post + np.random.randn(len(units)) * 0.5
    return pd.DataFrame(
        {"unit": units, "time": times, "outcome": y, "first_treat": fte.astype(int)}
    )


def _bacon_panel(n_units=100, n_periods=10, n_cohorts=3, seed=42):
    import pandas as pd

    np.random.seed(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    n_never = int(n_units * 0.3)
    n_treated = n_units - n_never
    cohort_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    first_treat = np.zeros(n_units)
    if n_treated > 0:
        first_treat[n_never:] = cohort_periods[
            np.random.choice(len(cohort_periods), size=n_treated)
        ]
    fte = np.repeat(first_treat, n_periods)
    unit_fe = np.repeat(np.random.randn(n_units) * 2, n_periods)
    time_fe = np.tile(np.linspace(0, 1, n_periods), n_units)
    post = (times >= fte) & (fte > 0)
    y = unit_fe + time_fe + 2.0 * post + np.random.randn(len(units)) * 0.5
    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": y,
            "first_treat": fte.astype(int),
            "treated": post.astype(int),
        }
    )


def _dcdh_fit():
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
    from diff_diff.prep_dgp import generate_reversible_did_data

    data = generate_reversible_did_data(n_groups=50, n_periods=8, pattern="joiners_only", seed=42)
    return ChaisemartinDHaultfoeuille(twfe_diagnostic=False).fit(
        data, outcome="outcome", unit="group", time="period", treatment="treatment", L_max=1
    )
