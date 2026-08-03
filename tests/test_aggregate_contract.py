"""Behavioral contract for post-fit ``results.aggregate()`` (spec section 6).

The ``test_ref`` for ledger rows M-020 (``fit(aggregate=)`` shim), M-117
(``balance_e`` moves onto ``aggregate()``) and M-122 (``AggregationResult``).

The headline gate is NUMERICAL INERTNESS: for every supported type,
``fit(aggregate=T)`` and ``fit(); .aggregate(T)`` must agree to 1e-14. The
refactor that enabled post-fit aggregation touched the influence-function
path, so drift here means a regression, not a design change.
"""

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, StaggeredTripleDifference
from diff_diff.aggregation import AGGREGATION_SCHEMA, AggregationResult
from diff_diff.results_base import EventStudyResults

FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")


def _panel(seed=11, n_units=80, n_periods=7):
    rng = np.random.default_rng(seed)
    cohorts = [0, 3, 4, 5, 6]
    rows = []
    for u in range(n_units):
        g = cohorts[u % len(cohorts)]
        ui = rng.normal(0, 0.5)
        for t in range(1, n_periods + 1):
            treated = g != 0 and t >= g
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": g,
                    "y": ui
                    + 0.4 * t
                    + (1.5 + 0.3 * (t - g) if treated else 0.0)
                    + rng.normal(0, 0.25),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel):
    """A plain fit - no aggregate= argument, so no deprecation warning."""
    return CallawaySantAnna().fit(panel, **FIT_KW)


@pytest.fixture(scope="module")
def fit_time(panel):
    """The DEPRECATED fit-time aggregation, for the inertness comparison."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return CallawaySantAnna().fit(panel, aggregate="all", **FIT_KW)


# --------------------------------------------------------------------------- #
# Numerical inertness (the headline gate)
# --------------------------------------------------------------------------- #


class TestInertness:
    def test_simple_matches_fit_time(self, fitted, fit_time):
        got = fitted.aggregate("simple")
        assert np.allclose(got.att[0], fit_time.overall_att, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.se[0], fit_time.overall_se, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.p_value[0], fit_time.overall_p_value, rtol=1e-14, atol=1e-14)

    def test_group_matches_fit_time(self, fitted, fit_time):
        frame = fitted.aggregate("group").to_dataframe()
        assert len(frame) == len(fit_time.group_effects)
        for _, row in frame.iterrows():
            native = fit_time.group_effects[row["label"]]
            for col, key in (
                ("att", "effect"),
                ("se", "se"),
                ("t_stat", "t_stat"),
                ("p_value", "p_value"),
            ):
                assert np.allclose(
                    row[col], native[key], rtol=1e-14, atol=1e-14, equal_nan=True
                ), f"group[{row['label']}].{col} drifted"

    def test_event_study_matches_fit_time(self, fitted, fit_time):
        frame = fitted.aggregate("event_study").to_dataframe()
        compared = 0
        for _, row in frame.iterrows():
            if bool(row["is_reference"]):
                continue
            native = fit_time.event_study_effects[row["event_time"]]
            for col, key in (
                ("att", "effect"),
                ("se", "se"),
                ("t_stat", "t_stat"),
                ("p_value", "p_value"),
            ):
                assert np.allclose(
                    row[col], native[key], rtol=1e-14, atol=1e-14, equal_nan=True
                ), f"event_study[{row['event_time']}].{col} drifted"
            compared += 1
        assert compared > 0, "no non-reference event times compared"

    @pytest.mark.parametrize("balance_e", [0, 1, 2])
    def test_balance_e_matches_fit_time(self, panel, fitted, balance_e):
        """M-117: balance_e on aggregate() reproduces fit(balance_e=)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna().fit(
                panel, aggregate="event_study", balance_e=balance_e, **FIT_KW
            )
        frame = fitted.aggregate("event_study", balance_e=balance_e).to_dataframe()
        non_ref = frame[~frame["is_reference"].astype(bool)]
        assert len(non_ref) == len(
            [e for e in native.event_study_effects if e in set(non_ref["event_time"])]
        )
        for _, row in non_ref.iterrows():
            got = native.event_study_effects[row["event_time"]]
            assert np.allclose(row["att"], got["effect"], rtol=1e-14, atol=1e-14, equal_nan=True)
            assert np.allclose(row["se"], got["se"], rtol=1e-14, atol=1e-14, equal_nan=True)


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


class TestImmutability:
    #: The attributes the aggregators used to mutate on their host.
    MUTATED = (
        "event_study_effects",
        "group_effects",
        "event_study_vcov",
        "event_study_vcov_index",
        "event_study_df",
    )

    def test_parent_unchanged_across_mixed_calls(self, fitted):
        before = {f: copy.deepcopy(getattr(fitted, f, None)) for f in self.MUTATED}
        for level in ("event_study", "group", "simple", "group", "event_study"):
            fitted.aggregate(level)
        for f in self.MUTATED:
            after = getattr(fitted, f, None)
            if before[f] is None:
                assert after is None, f"{f} was populated by aggregate()"
            else:
                assert str(before[f]) == str(after), f"{f} changed"

    def test_repeated_calls_agree(self, fitted):
        a = fitted.aggregate("group").to_dataframe()
        b = fitted.aggregate("group").to_dataframe()
        # assert_frame_equal treats corresponding NaNs as equal and handles the
        # mixed object/float dtypes; a bare np.array_equal does neither.
        pd.testing.assert_frame_equal(a, b)

    def test_order_independent(self, fitted):
        es_first = fitted.aggregate("event_study").to_dataframe()
        fitted.aggregate("group")
        es_after = fitted.aggregate("event_study").to_dataframe()
        assert np.allclose(es_first["att"].to_numpy(), es_after["att"].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# Retention kit
# --------------------------------------------------------------------------- #


class TestRetentionKit:
    def test_kit_holds_no_dataframe(self, fitted):
        """The source panel must never be retained on a results object."""
        kit = fitted._aggregation_kit
        frames = [k for k, v in kit.bookkeeping.items() if isinstance(v, pd.DataFrame)]
        assert frames == [], f"kit retained DataFrame(s): {frames}"
        assert not isinstance(kit.influence, pd.DataFrame)

    def test_no_dataframe_anywhere_on_results(self, fitted):
        frames = [
            f
            for f in fitted.__dataclass_fields__
            if isinstance(getattr(fitted, f, None), pd.DataFrame)
        ]
        assert frames == [], f"results retained DataFrame(s): {frames}"

    def test_aggregate_survives_pickle(self, fitted):
        """No live reference to the estimator or its frame."""
        revived = pickle.loads(pickle.dumps(fitted))
        got = revived.aggregate("group").to_dataframe()
        want = fitted.aggregate("group").to_dataframe()
        assert np.allclose(got["att"].to_numpy(), want["att"].to_numpy(), equal_nan=True)

    def test_no_raw_unit_identifiers_are_retained(self):
        """Data minimization: results objects are picklable and get shared, so
        the kit must not become a carrier for unit identifiers - which are
        routinely names, emails or administrative IDs. The kit needs only
        POSITION (influence arrays index by ``treated_idx``/``control_idx``),
        so it stores canonical 0..n-1 codes.

        Searches the whole serialized artifact, not just the kit: a recursive
        check is what makes this a real guarantee rather than a spot check.
        """
        sentinel = "SENTINEL-ID-{}@example.invalid"
        rng = np.random.default_rng(4)
        rows = []
        for i in range(45):
            g = [0, 3, 4][i % 3]
            for t in range(1, 7):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": sentinel.format(i),
                        "time": t,
                        "first_treat": g,
                        "y": 0.4 * t + (1.5 if treated else 0.0) + rng.normal(0, 0.25),
                    }
                )
        res = CallawaySantAnna().fit(pd.DataFrame(rows), **FIT_KW)
        assert b"SENTINEL-ID" not in pickle.dumps(res), "raw unit ids reached the artifact"
        # And the minimization is inert: aggregation still produces numbers.
        for level in ("simple", "group", "event_study"):
            assert np.isfinite(np.asarray(res.aggregate(level).att)).any()

    def test_identifier_minimization_is_numerically_inert(self, panel, fitted):
        """Canonical codes must not perturb any aggregate: compared at
        atol=rtol=0, since substituting positions for labels should not change
        a single floating-point operation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna().fit(panel, aggregate="all", **FIT_KW)
        got = fitted.aggregate("simple")
        assert got.att[0] == native.overall_att
        assert got.se[0] == native.overall_se


# --------------------------------------------------------------------------- #
# Fail-closed contract
# --------------------------------------------------------------------------- #


class TestFailClosed:
    def test_calendar_unsupported_by_cs(self, fitted):
        with pytest.raises(ValueError, match="calendar"):
            fitted.aggregate("calendar")

    def test_all_is_not_a_post_fit_type(self, fitted):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            fitted.aggregate("all")

    def test_error_names_supported_types(self, fitted):
        with pytest.raises(ValueError) as exc:
            fitted.aggregate("nonsense")
        for level in ("simple", "event_study", "group"):
            assert level in str(exc.value)

    def test_weights_rejected(self, fitted):
        """CS exposes no weighting selector, so anything but None fails closed."""
        with pytest.raises(ValueError, match="weights"):
            fitted.aggregate("simple", "cohort_share")

    @pytest.mark.parametrize("level", ["simple", "group"])
    def test_balance_e_rejected_where_inert(self, fitted, level):
        """balance_e applies to event-study aggregation ONLY - silently
        ignoring it elsewhere would accept a user argument that does nothing."""
        with pytest.raises(ValueError, match="balance_e"):
            fitted.aggregate(level, balance_e=2)

    def test_bootstrap_fit_raises(self, panel):
        """Percentile-bootstrap inference cannot be reproduced from the
        analytical state retained here, so aggregate() must not substitute
        analytical numbers silently."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = CallawaySantAnna(n_bootstrap=49, seed=42).fit(panel, **FIT_KW)
        with pytest.raises(NotImplementedError, match="bootstrap"):
            boot.aggregate("group")


# --------------------------------------------------------------------------- #
# Container schema (M-122)
# --------------------------------------------------------------------------- #


class TestAggregationResult:
    def test_pinned_schema(self, fitted):
        assert tuple(fitted.aggregate("group").to_dataframe().columns) == AGGREGATION_SCHEMA

    def test_event_study_returns_the_unified_container(self, fitted):
        """M-092's container finally gets a public producer."""
        assert isinstance(fitted.aggregate("event_study"), EventStudyResults)

    def test_non_event_study_returns_aggregation_result(self, fitted):
        assert isinstance(fitted.aggregate("group"), AggregationResult)
        assert isinstance(fitted.aggregate("simple"), AggregationResult)

    def test_serializers_present(self, fitted):
        got = fitted.aggregate("group")
        assert isinstance(got.to_dict(), dict)
        assert isinstance(got.to_dataframe(), pd.DataFrame)
        assert isinstance(got.summary(), str)

    def test_group_weight_is_none_not_fabricated(self, fitted):
        """_aggregate_by_group weights (g,t) cells equally WITHIN a cohort and
        forms no cross-cohort mass, so there is no per-row weight to report."""
        assert fitted.aggregate("group").weight is None
        assert fitted.aggregate("group").to_dict()["weight"] is None

    def test_n_kind_is_declared(self, fitted):
        """`n` means different things per level; n_kind is what disambiguates."""
        assert fitted.aggregate("group").n_kind == "cells"
        assert fitted.aggregate("simple").n_kind == "units"

    def test_target_is_per_row(self, fitted):
        got = fitted.aggregate("group")
        assert got.target.shape == got.label.shape

    def test_zero_row_is_a_supported_boundary(self):
        empty = AggregationResult(
            level="group",
            label=np.array([], dtype=object),
            target=np.array([], dtype=object),
            att=np.array([]),
            se=np.array([]),
            t_stat=np.array([]),
            p_value=np.array([]),
            conf_int_lower=np.array([]),
            conf_int_upper=np.array([]),
            n=np.array([]),
            df=np.array([]),
        )
        assert len(empty.to_dataframe()) == 0
        assert isinstance(empty.summary(), str)

    def test_non_estimable_row_keeps_point_estimate(self):
        """safe_inference NaNs t/p/CI only - att and se are inputs, not
        outputs, so NaN-ing the whole quintet would erase valid estimates."""
        got = AggregationResult(
            level="group",
            label=np.array([1], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([2.0]),
            se=np.array([np.nan]),
            t_stat=np.array([np.nan]),
            p_value=np.array([np.nan]),
            conf_int_lower=np.array([np.nan]),
            conf_int_upper=np.array([np.nan]),
            n=np.array([3.0]),
            df=61.0,
        )
        assert got.att[0] == 2.0
        assert np.isnan(got.t_stat[0])
        assert np.isnan(got.df[0]), "df must be NaN where p_value is non-finite"


# --------------------------------------------------------------------------- #
# Deprecation shim (M-020 / M-117)
# --------------------------------------------------------------------------- #


class TestFitShim:
    def test_plain_fit_does_not_warn(self, panel):
        """The sentinel default is what makes this true - a bare None default
        could not tell 'not passed' from 'passed None'."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CallawaySantAnna().fit(panel, **FIT_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    @pytest.mark.parametrize("kwargs", [{"aggregate": "group"}, {"balance_e": 1}])
    def test_deprecated_args_warn(self, panel, kwargs):
        with pytest.warns(FutureWarning, match="aggregate"):
            CallawaySantAnna().fit(panel, **kwargs, **FIT_KW)

    def test_deprecated_path_still_populates_legacy_surface(self, fit_time):
        """Downstream consumers (honest_did, pretrends,
        build_event_study_surface) read these off the results object."""
        assert fit_time.event_study_effects is not None
        assert fit_time.group_effects is not None


def _rcs(seed=17, n_per_cell=6, n_periods=6):
    """Repeated cross-sections: a DIFFERENT set of units each period, so unit
    ids must be globally unique rather than recycled across periods."""
    rng = np.random.default_rng(seed)
    cohorts = [0, 3, 4, 5]
    rows = []
    uid = 0
    for t in range(1, n_periods + 1):
        for g in cohorts:
            for _ in range(n_per_cell):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "first_treat": g,
                        "y": 0.4 * t + (1.5 if treated else 0.0) + rng.normal(0, 0.25),
                    }
                )
                uid += 1
    return pd.DataFrame(rows)


class TestInertnessAcrossDesigns:
    """The inertness gate on designs the balanced-panel fixture does not reach.

    The refactor made the aggregators pure by threading bookkeeping that the
    panel path and the repeated-cross-section path populate differently (RCS
    makes several kit keys observation-length rather than unit-length), and
    ``anticipation`` shifts which ``(g, t)`` cells are eligible - so both need
    their own round-trip, not just the panel.
    """

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_repeated_cross_sections_match_fit_time(self, level):
        data = _rcs()
        post = CallawaySantAnna(panel=False).fit(data, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(panel=False).fit(data, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_anticipation_matches_fit_time(self, panel, level):
        post = CallawaySantAnna(anticipation=1).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(anticipation=1).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_universal_base_period_matches_fit_time(self, panel, level):
        """REGISTRY gives universal bases their own reference-cell and
        VCV-index semantics (a zero reference cell per cohort), which the
        default 'varying' fixture never exercises."""
        kw = dict(base_period="universal")
        post = CallawaySantAnna(**kw).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(**kw).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_not_yet_treated_with_anticipation_matches_fit_time(self, panel, level):
        """The not-yet-treated control group interacts with anticipation in
        picking each (g, t) comparison set - a different code path from the
        never-treated default."""
        kw = dict(control_group="not_yet_treated", anticipation=1)
        post = CallawaySantAnna(**kw).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(**kw).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_unbalanced_panel_matches_fit_time(self, panel, level):
        """allow_unbalanced_panel changes which units survive into the
        influence vectors, so the retained bookkeeping must agree with what
        the fit actually used."""
        thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
        kw = dict(allow_unbalanced_panel=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            post = CallawaySantAnna(**kw).fit(thinned, **FIT_KW)
            native = CallawaySantAnna(**kw).fit(thinned, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_survey_numbers_match_fit_time(self, survey_fit, level):
        """Survey parity on the ESTIMATES, not just the df metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native = CallawaySantAnna().fit(
                _survey_panel(),
                survey_design=_survey_design(),
                aggregate="all",
                **FIT_KW,
            )
        _assert_level_matches(survey_fit, native, level)


def _assert_level_matches(post, native, level):
    """Compare a post-fit aggregation against the fit-time surface at 1e-14."""
    if level == "simple":
        got = post.aggregate("simple")
        assert np.allclose(got.att[0], native.overall_att, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.se[0], native.overall_se, rtol=1e-14, atol=1e-14)
        return

    frame = post.aggregate(level).to_dataframe()
    if level == "group":
        expected, key_col = native.group_effects, "label"
    else:
        expected, key_col = native.event_study_effects, "event_time"
    compared = 0
    for _, row in frame.iterrows():
        if level == "event_study" and bool(row["is_reference"]):
            continue
        ref = expected[row[key_col]]
        for col, key in (("att", "effect"), ("se", "se"), ("p_value", "p_value")):
            assert np.allclose(
                row[col], ref[key], rtol=1e-14, atol=1e-14, equal_nan=True
            ), f"{level}[{row[key_col]}].{col} drifted"
        compared += 1
    assert compared > 0, f"no {level} rows compared"


def _survey_panel():
    rng = np.random.default_rng(3)
    rows = []
    for u in range(80):
        g = [0, 4, 6][u % 3]
        for t in range(1, 9):
            treated = g != 0 and t >= g
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": g,
                    "psu": u % 20,
                    "stratum": u % 4,
                    "w": 1.0 + (u % 5) * 0.1,
                    "y": 0.3 * t + (2.0 if treated else 0.0) + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def _survey_design():
    from diff_diff import SurveyDesign

    return SurveyDesign(weights="w", strata="stratum", psu="psu")


@pytest.fixture(scope="module")
def survey_fit():
    """An EXPLICIT survey design - the branch where ``df_inference`` is
    intentionally None and ``survey_metadata.df_survey`` is the real carrier."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return CallawaySantAnna().fit(_survey_panel(), survey_design=_survey_design(), **FIT_KW)


class TestInferenceDfProvenance:
    """``df`` is per-row PROVENANCE - the df that actually produced the stored
    p-value/CI - so it must come from the carrier that governed inference.

    Regression: ``simple`` read ``df_inference`` (documented to stay None on
    explicit ``survey_design=`` fits) and ``group`` read ``event_study_df`` (a
    DIFFERENT aggregation's df, and None after a plain fit). Both reported
    ``df=NaN`` - implying normal/undefined inference - while the interval they
    carried was built on a finite t-reference.
    """

    def test_explicit_survey_df_reaches_every_level(self, survey_fit):
        """The canonical carrier is ``survey_metadata.df_survey``; on an
        explicit survey design ``df_inference`` is intentionally None."""
        expected = float(survey_fit.survey_metadata.df_survey)
        assert survey_fit.df_inference is None, "fixture no longer exercises the bug"
        for level in ("simple", "group", "event_study"):
            df_col = np.asarray(survey_fit.aggregate(level).df, dtype=float)
            finite = df_col[np.isfinite(df_col)]
            assert finite.size > 0, f"{level} reported no df at all"
            assert np.all(finite == expected), f"{level} df {finite} != {expected}"

    @pytest.mark.parametrize("survey", [False, True])
    def test_bootstrap_clears_group_df_provenance(self, survey):
        """When bootstrap replaces a group row's se/p/CI with percentile values,
        the retained analytical df described inference that no longer exists.
        Leaving it finite would claim a t-reference governed percentile-bootstrap
        numbers - the same false provenance the event-study path already clears.
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(7)
        rows = []
        for u in range(90):
            g = [0, 4, 6][u % 3]
            for t in range(1, 9):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": g,
                        "psu": u % 18,
                        "stratum": u % 3,
                        "w": 1.0 + (u % 4) * 0.1,
                        "y": 0.3 * t + (2.0 if treated else 0.0) + rng.normal(0, 0.5),
                    }
                )
        # cluster= is a CONSTRUCTOR argument; survey_design= is a fit() argument.
        ctor = {} if survey else {"cluster": "psu"}
        fit_extra = (
            {"survey_design": SurveyDesign(weights="w", strata="stratum", psu="psu")}
            if survey
            else {}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(n_bootstrap=50, seed=1, **ctor).fit(
                pd.DataFrame(rows), aggregate="all", **fit_extra, **FIT_KW
            )
        assert res.group_effects, "fixture produced no group rows"
        for g, eff in res.group_effects.items():
            assert eff.get("df_used") is None, f"group {g} kept analytical df on a bootstrap fit"
        # The event-study path's equivalent clearing must not have regressed.
        assert res.event_study_df is None

    @pytest.mark.parametrize("survey", [False, True])
    def test_sddd_bootstrap_clears_group_df_provenance(self, survey):
        """StaggeredTripleDifference carries its OWN copy of the bootstrap
        group-replacement loop, so the clearing rule has to hold there too -
        fixing one twin and testing only the other is how this class of bug
        survived in the first place.

        Coverage is ASYMMETRIC and deliberately so: only ``survey=True``
        exercises the clearing (verified by reverting the fix - just that case
        fails). SDDD's plain-``cluster`` path never resolves a finite analytical
        df, so ``df_used`` is already None there and nothing needs clearing. The
        non-survey case is kept as a cheap guard in case that ever changes, not
        because it currently reproduces the bug. The CallawaySantAnna
        equivalent above DOES fail in both configurations.
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(9)
        rows = []
        for u in range(96):
            g = [0, 3, 4][u % 3]
            elig = u % 2
            for t in range(1, 7):
                treated = g != 0 and t >= g and elig == 1
                rows.append(
                    {
                        "unit": u,
                        "period": t,
                        "first_treat": g,
                        "eligibility": elig,
                        "psu": u % 16,
                        "stratum": u % 4,
                        "w": 1.0 + (u % 3) * 0.1,
                        "outcome": 0.3 * t + (1.2 if treated else 0.0) + rng.normal(0, 0.2),
                    }
                )
        ctor = {} if survey else {"cluster": "psu"}
        fit_extra = (
            {"survey_design": SurveyDesign(weights="w", strata="stratum", psu="psu")}
            if survey
            else {}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StaggeredTripleDifference(n_bootstrap=40, seed=3, **ctor).fit(
                pd.DataFrame(rows),
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                aggregate="all",
                **fit_extra,
            )
        assert res.group_effects, "fixture produced no SDDD group rows"
        for g, eff in res.group_effects.items():
            assert eff.get("df_used") is None, f"SDDD group {g} kept analytical df on bootstrap"

    def test_group_df_is_not_the_event_study_df(self, fitted):
        """``group`` must not borrow ``event_study_df``: on a plain fit that
        field is None, and it is a different aggregation's denominator."""
        assert fitted.event_study_df is None
        grp = fitted.aggregate("group")
        assert np.asarray(grp.df).shape == np.asarray(grp.att).shape


class TestContainerNormalization:
    """``AggregationResult`` normalizes its own columns; doing so must not
    reach back into caller-owned memory or mask a shape error."""

    @staticmethod
    def _kw():
        return dict(
            level="group",
            label=np.array(["a", "b"], dtype=object),
            target=np.array(["att", "att"], dtype=object),
            att=np.array([1.0, 2.0]),
            se=np.array([0.1, 0.2]),
            t_stat=np.array([10.0, np.nan]),
            p_value=np.array([0.01, np.nan]),
            conf_int_lower=np.array([0.8, np.nan]),
            conf_int_upper=np.array([1.2, np.nan]),
            n=np.array([5.0, 6.0]),
        )

    def test_df_input_is_not_mutated(self):
        """The NaN-out of non-estimable rows wrote THROUGH to the caller's
        array when df was normalized with ``asarray`` instead of ``array``."""
        caller = np.array([10.0, 20.0])
        AggregationResult(df=caller, **self._kw())
        assert np.array_equal(caller, [10.0, 20.0]), "caller's df array was mutated"

    def test_read_only_df_is_accepted(self):
        frozen = np.array([10.0, 20.0])
        frozen.flags.writeable = False
        got = AggregationResult(df=frozen, **self._kw())
        assert np.isnan(got.df[1]), "non-estimable row should still be NaN'd"

    def test_zero_dim_label_raises_value_error(self):
        """Not IndexError: the shape check must precede the shape read."""
        with pytest.raises(ValueError, match="one-dimensional"):
            AggregationResult(
                level="simple",
                label=np.array("x", dtype=object),
                target=np.array(["att"], dtype=object),
                att=np.array([1.0]),
                se=np.array([0.1]),
                t_stat=np.array([1.0]),
                p_value=np.array([0.1]),
                conf_int_lower=np.array([0.0]),
                conf_int_upper=np.array([2.0]),
                n=np.array([5.0]),
                df=None,
            )

    def test_off_vocabulary_n_kind_raises(self):
        """``n_kind`` is a routing key shared with EventStudyResults, so an
        unknown value is a contract break rather than a free-form label."""
        with pytest.raises(ValueError, match="vocabulary"):
            AggregationResult(df=None, n_kind="widgets", **self._kw())

    @pytest.mark.parametrize("level,expected", [("simple", "units"), ("group", "cells")])
    def test_cs_n_kinds_are_in_the_shared_vocabulary(self, fitted, level, expected):
        from diff_diff.results_base import N_KIND_VOCABULARY

        got = fitted.aggregate(level)
        assert got.n_kind == expected
        assert got.n_kind in N_KIND_VOCABULARY

    def test_rcs_simple_reports_obs_not_units(self):
        """On panel=False, fit() counts ROWS (there is no unit tracking), so
        labelling that total "units" would misdescribe the sample - the exact
        conflation the shared vocabulary forbids."""
        data = _rcs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = CallawaySantAnna(panel=False).fit(data, **FIT_KW)
        got = res.aggregate("simple")
        assert got.n_kind == "obs"
        assert got.n[0] == float(res.n_treated_units + res.n_control_units)
        # The panel fixture must still say units, or this would be a blanket rename.
        panel_res = CallawaySantAnna().fit(_panel(), **FIT_KW)
        assert panel_res.aggregate("simple").n_kind == "units"

    def test_shared_vocabulary_is_enforced_on_both_containers(self, fitted):
        """The vocabulary is declared SHARED, so validating it on only one
        container would let an unknown value reach a consumer through the
        unchecked side. Every value a real producer emits must still pass.
        """
        from diff_diff.results_base import N_KIND_VOCABULARY

        base = dict(
            event_time=np.array([-1, 0, 1]),
            att=np.array([0.0, 1.0, 2.0]),
            se=np.array([0.1, 0.2, 0.3]),
            t_stat=np.array([0.0, 5.0, 6.0]),
            p_value=np.array([1.0, 0.01, 0.01]),
            conf_int_lower=np.array([-0.2, 0.6, 1.4]),
            conf_int_upper=np.array([0.2, 1.4, 2.6]),
            is_reference=np.array([True, False, False]),
            n=np.array([np.nan, 4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="vocabulary"):
            EventStudyResults(n_kind="widgets", **base)
        # Every value shipped producers actually emit stays constructible.
        for kind in ("groups", "switcher_cells", "cells", "units", "obs", None):
            assert EventStudyResults(n_kind=kind, **base).n_kind == kind
        assert set(N_KIND_VOCABULARY) >= {"groups", "switcher_cells", "cells", "units", "obs"}

    def test_summary_refuses_to_relabel_a_stored_interval(self, fitted):
        """summary(alpha=) never recomputes, so printing the passed alpha would
        assert a confidence level the stored interval was not built at. Raises
        instead, matching EventStudyResults.summary."""
        got = fitted.aggregate("group")
        assert f"alpha={got.alpha}" in got.summary()
        assert f"alpha={got.alpha}" in got.summary(alpha=got.alpha)
        with pytest.raises(ValueError, match="never recomputes"):
            got.summary(alpha=0.10)


# --------------------------------------------------------------------------- #
# Cross-estimator regression
# --------------------------------------------------------------------------- #


def test_staggered_triple_diff_overall_att_es_still_works():
    """StaggeredTripleDifference inherits the CS aggregation mixin and is the
    ONLY reader of the Eq. 4.14 overall, which the purity refactor converted
    from a self-attribute to a returned value."""
    rng = np.random.default_rng(5)
    rows = []
    for u in range(80):
        g = [0, 3, 4][u % 3]
        elig = u % 2
        for t in range(1, 6):
            treated = g != 0 and t >= g and elig == 1
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "first_treat": g,
                    "eligibility": elig,
                    "outcome": 0.3 * t + (1.2 if treated else 0.0) + rng.normal(0, 0.2),
                }
            )
    data = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = StaggeredTripleDifference().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
    assert res.overall_att_es is not None
    assert np.isfinite(res.overall_att_es)


# --------------------------------------------------------------------------- #
# dCDH (row M-026): fit(aggregate=) shim + the VIEW-based aggregate()
# --------------------------------------------------------------------------- #

DCDH_KW = dict(outcome="outcome", unit="unit", time="period", treatment="treat")


def _dcdh_panel(seed=5, n_units=40, n_periods=6, switch_t=4):
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_units):
        s_t = switch_t if u < n_units // 2 else 10**6
        for t in range(1, n_periods + 1):
            d = 1 if t >= s_t else 0
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "outcome": u / 10 + 0.2 * t + 1.5 * d + rng.randn() * 0.3,
                    "treat": d,
                }
            )
    return pd.DataFrame(rows)


def _dcdh_survey_panel(seed=7, n_units=60, n_periods=6, switch_t=4):
    df = _dcdh_panel(seed=seed, n_units=n_units, n_periods=n_periods, switch_t=switch_t)
    df["survey_weights"] = 1.0 + 0.1 * (df["unit"] % 5)
    df["strata"] = df["unit"] % 4
    df["psu"] = df["unit"]
    return df


def _fit_dcdh(data, *, est_kw=None, **fit_kw):
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ChaisemartinDHaultfoeuille(**(est_kw or {})).fit(data, **DCDH_KW, **fit_kw)


@pytest.fixture(scope="module")
def dcdh_panel():
    return _dcdh_panel()


@pytest.fixture(scope="module")
def dcdh_fitted(dcdh_panel):
    """Phase-1 fit (L_max=None)."""
    return _fit_dcdh(dcdh_panel)


class TestDcdhShim:
    def test_plain_fit_does_not_warn(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, **DCDH_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, aggregate=None, **DCDH_KW)

    def test_non_none_value_warns_then_raises(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match=r"results\.aggregate"):
                ChaisemartinDHaultfoeuille().fit(dcdh_panel, aggregate="event_study", **DCDH_KW)

    def test_wrapper_forwarded_aggregate_warns(self, dcdh_panel):
        # chaisemartin_dhaultfoeuille() splits **kwargs by signature and
        # forwards non-__init__ names into fit(), so the shim is reachable
        # through the wrapper too.
        from diff_diff.chaisemartin_dhaultfoeuille import chaisemartin_dhaultfoeuille

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            chaisemartin_dhaultfoeuille(
                dcdh_panel,
                outcome="outcome",
                group="unit",
                time="period",
                treatment="treat",
                aggregate=None,
            )


class TestDcdhAggregate:
    def _assert_surface_matches_builder(self, res):
        from diff_diff.results_base import build_event_study_surface

        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        built = build_event_study_surface(res)
        # The dataclass's generated == raises on ndarray fields; compare
        # to_dataframe rows per the file's precedent.
        a, b = es.to_dataframe(), built.to_dataframe()
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape
        for col in a.columns:
            av, bv = a[col].to_numpy(), b[col].to_numpy()
            if av.dtype.kind in "fc":
                assert np.allclose(av.astype(float), bv.astype(float), equal_nan=True)
            else:
                assert list(av) == list(bv)
        return es

    def test_simple_view_bit_exact_phase1(self, dcdh_fitted):
        agg = dcdh_fitted.aggregate("simple")
        assert isinstance(agg, AggregationResult)
        assert agg.level == "simple"
        assert list(agg.label) == ["overall"]
        assert list(agg.target) == ["DID_M"]
        assert float(agg.att[0]) == dcdh_fitted.overall_att
        assert float(agg.se[0]) == dcdh_fitted.overall_se
        assert float(agg.t_stat[0]) == dcdh_fitted.overall_t_stat
        assert float(agg.p_value[0]) == dcdh_fitted.overall_p_value
        assert float(agg.conf_int_lower[0]) == dcdh_fitted.overall_conf_int[0]
        assert float(agg.conf_int_upper[0]) == dcdh_fitted.overall_conf_int[1]
        assert float(agg.n[0]) == float(dcdh_fitted.n_switcher_cells)
        assert agg.n_kind == "switcher_cells"
        # Non-survey analytical inference is z-based: no df.
        assert np.isnan(agg.df[0])
        assert agg.estimator == "ChaisemartinDHaultfoeuille"

    def test_simple_view_lmax1_groups(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=1)
        agg = res.aggregate("simple")
        assert list(agg.target) == ["DID_1"]
        assert agg.n_kind == "groups"
        assert float(agg.n[0]) == float(res.n_switcher_cells)

    def test_simple_view_lmax2_delta(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=2)
        agg = res.aggregate("simple")
        assert list(agg.target) == ["delta"]
        # The delta averages horizon-specific N_l: no truthful scalar count.
        assert np.isnan(agg.n[0])
        assert agg.n_kind is None
        assert float(agg.att[0]) == res.overall_att

    def test_simple_view_trends_linear_all_nan_relay(self, dcdh_panel):
        # trends_linear + L_max>=2 suppresses the delta by design: every
        # overall_* field is NaN and the estimand label points at
        # linear_trends_effects. The view relays the all-NaN row honestly.
        res = _fit_dcdh(dcdh_panel, L_max=2, trends_linear=True)
        agg = res.aggregate("simple")
        assert "fd" in str(agg.target[0])
        assert np.isnan(agg.att[0])
        assert np.isnan(agg.se[0])
        assert np.isnan(agg.p_value[0])
        assert np.isnan(agg.conf_int_lower[0]) and np.isnan(agg.conf_int_upper[0])
        assert np.isnan(agg.df[0])

    def test_simple_view_bootstrap_percentile_relay(self, dcdh_panel):
        # Bootstrap fits are PERMITTED (pure view): the row relays the
        # stored percentile-bootstrap inference; df is NaN (no df used).
        res = _fit_dcdh(dcdh_panel, est_kw=dict(n_bootstrap=49, seed=3))
        agg = res.aggregate("simple")
        assert float(agg.att[0]) == res.overall_att
        assert float(agg.se[0]) == res.overall_se
        assert np.isnan(agg.df[0])

    def test_event_study_container_threads_survey_df(self):
        # CQ1 (local review R3): the dCDH builder threads the scalar
        # df_survey provenance too - a survey fit's container must carry
        # survey_metadata.df_survey, not None.
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd)
        surface = res.aggregate("event_study")
        assert res.survey_metadata is not None
        assert surface.df_survey == float(res.survey_metadata.df_survey)

    def test_simple_view_survey_analytical_df(self):
        # Analytical survey fit: the stored p/CI used the survey df; the
        # view relays it (event_study_df carries it here).
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd)
        agg = res.aggregate("simple")
        assert res.survey_metadata is not None
        expected = res.survey_metadata.df_survey
        assert expected is not None and np.isfinite(agg.df[0])
        assert float(agg.df[0]) == float(expected)

    def test_simple_view_lmax2_survey_bootstrap_finite_df(self):
        # THE df-provenance pin: under n_bootstrap>0 the event_study_df
        # channel is cleared, but the L_max>=2 delta's stored p/CI still
        # came from analytical safe_inference with the survey df (REGISTRY
        # Note, Phase 2 cost-benefit delta SE). The view must report that
        # finite df, not NaN.
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd, est_kw=dict(n_bootstrap=49, seed=3))
        assert res.event_study_df is None  # cleared under bootstrap
        assert np.isfinite(res.overall_p_value)  # delta stayed analytical
        agg = res.aggregate("simple")
        assert res.survey_metadata is not None
        assert float(agg.df[0]) == float(res.survey_metadata.df_survey)

    def test_event_study_view_phase1_two_rows(self, dcdh_fitted):
        es = self._assert_surface_matches_builder(dcdh_fitted)
        # Phase-1 (L_max=None): the 2-row l=1 view, l1 convention - NOT an
        # error (fit populates event_study_effects={1: ...} on this path).
        assert es.event_time.tolist() == [0, 1]
        assert es.event_time_convention == "l1_first_switch"
        assert es.n_kind == "switcher_cells"

    def test_event_study_view_multi_horizon(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=2)
        es = self._assert_surface_matches_builder(res)
        assert 2 in es.event_time.tolist()
        assert es.n_kind == "groups"

    def test_balance_e_rejected_empty_vocabulary(self, dcdh_fitted):
        with pytest.raises(ValueError, match="no aggregation type on this estimator"):
            dcdh_fitted.aggregate("event_study", balance_e=1)

    def test_weights_rejected(self, dcdh_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            dcdh_fitted.aggregate("simple", weights="cell")

    @pytest.mark.parametrize("bad", ["group", "calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, dcdh_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            dcdh_fitted.aggregate(bad)

    def test_mixin_hooks_are_not_dataclass_fields(self):
        # Regression: on a dataclass results class, annotating the mixin
        # routing hooks without ClassVar turns them into __init__ fields,
        # widening the public constructor/repr/equality surface. Enforced
        # dynamically for EVERY dataclass that mixes AggregationMixin in,
        # so later 2(b) waves are enrolled automatically.
        import dataclasses
        import inspect

        import diff_diff
        from diff_diff.aggregation import AggregationMixin

        hooks = ("_AGGREGATE_SUPPORTED", "_AGGREGATE_BALANCE_E_TYPES")
        checked = []
        for name in dir(diff_diff):
            obj = getattr(diff_diff, name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, AggregationMixin)
                and obj is not AggregationMixin
                and dataclasses.is_dataclass(obj)
            ):
                checked.append(name)
                # dataclasses.fields() (not __dataclass_fields__, which
                # also lists ClassVar pseudo-fields) = the real
                # init/repr/eq surface.
                fields = {f.name for f in dataclasses.fields(obj)}
                params = inspect.signature(obj.__init__).parameters
                for hook in hooks:
                    assert hook not in fields, f"{name}.{hook} leaked into fields"
                    assert hook not in params, f"{name}.{hook} leaked into __init__"
        # The roster must at least cover the shipped mixin adopters.
        assert "CallawaySantAnnaResults" in checked
        assert "ChaisemartinDHaultfoeuilleResults" in checked
        assert "StackedDiDResults" in checked


# --------------------------------------------------------------------------- #
# StackedDiD (row M-024): fit(aggregate=) shim + the VIEW-based aggregate()
# --------------------------------------------------------------------------- #

STACKED_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _stacked_panel(seed=42, n_units=120, n_periods=12):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[4, 6, 8], seed=seed
    )


def _fit_stacked(data, *, est_kw=None, **fit_kw):
    from diff_diff import StackedDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StackedDiD(**(est_kw or {"kappa_pre": 2, "kappa_post": 2})).fit(
            data, **STACKED_KW, **fit_kw
        )


@pytest.fixture(scope="module")
def stacked_panel():
    return _stacked_panel()


@pytest.fixture(scope="module")
def stacked_fitted(stacked_panel):
    """Plain hc1 fit - the surface is always materialized (M-024)."""
    return _fit_stacked(stacked_panel)


class TestStackedShim:
    def test_plain_fit_does_not_warn(self, stacked_panel):
        from diff_diff import StackedDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StackedDiD(kappa_pre=2, kappa_post=2).fit(stacked_panel, **STACKED_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, stacked_panel):
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            StackedDiD(kappa_pre=2, kappa_post=2).fit(stacked_panel, aggregate=None, **STACKED_KW)

    def test_deprecated_value_warns_and_still_works(self, stacked_panel, stacked_fitted):
        # CS-style warn-and-still-work (the param genuinely worked here,
        # unlike dCDH's raise): the deprecated path returns an object whose
        # surface equals a plain fit's - the surface is always computed.
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            res = StackedDiD(kappa_pre=2, kappa_post=2).fit(
                stacked_panel, aggregate="event_study", **STACKED_KW
            )
        assert res.overall_att == stacked_fitted.overall_att
        assert res.event_study_effects is not None
        assert sorted(res.event_study_effects) == sorted(stacked_fitted.event_study_effects)
        np.testing.assert_array_equal(res.event_study_vcov, stacked_fitted.event_study_vcov)

    def test_wrapper_forwarded_aggregate_warns(self, stacked_panel):
        # stacked_did() declares aggregate explicitly with its own sentinel
        # default and forwards verbatim into fit().
        from diff_diff.stacked_did import stacked_did

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            stacked_did(
                stacked_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                kappa_pre=2,
                kappa_post=2,
                aggregate="simple",
            )

    def test_plain_wrapper_call_does_not_warn(self, stacked_panel):
        from diff_diff.stacked_did import stacked_did

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = stacked_did(
                stacked_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                kappa_pre=2,
                kappa_post=2,
            )
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []
        assert res.event_study_effects is not None

    def test_group_warns_then_raises_educational_error(self, stacked_panel):
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            with pytest.raises(ValueError, match="not supported by StackedDiD"):
                StackedDiD(kappa_pre=2, kappa_post=2).fit(
                    stacked_panel, aggregate="group", **STACKED_KW
                )


class TestStackedAggregate:
    def _assert_surface_matches_builder(self, res):
        from diff_diff.results_base import build_event_study_surface

        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        built = build_event_study_surface(res)
        a, b = es.to_dataframe(), built.to_dataframe()
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape
        for col in a.columns:
            av, bv = a[col].to_numpy(), b[col].to_numpy()
            if av.dtype.kind in "fc":
                assert np.allclose(av.astype(float), bv.astype(float), equal_nan=True)
            else:
                assert list(av) == list(bv)
        return es

    def test_event_study_view_matches_builder(self, stacked_fitted):
        es = self._assert_surface_matches_builder(stacked_fitted)
        # kappa 2/2 grid: {-2, ref -1, 0, 1, 2}; n_kind from n_obs cells.
        assert sorted(es.event_time.tolist()) == [-2, -1, 0, 1, 2]
        assert es.n_kind == "obs"
        assert es.base_period == "universal"
        assert es.reference_event_times == (-1,)

    def test_event_study_view_does_not_alias_fit_vcov(self, stacked_fitted):
        es = stacked_fitted.aggregate("event_study")
        assert not np.shares_memory(es.vcov, stacked_fitted.event_study_vcov)
        # int labels survive to_dict (no float coercion of the index)
        assert es.to_dict()["vcov_index"] == [-2, 0, 1, 2]

    def test_simple_bit_exact_relay_hc1(self, stacked_fitted):
        agg = stacked_fitted.aggregate("simple")
        assert isinstance(agg, AggregationResult)
        assert agg.level == "simple"
        assert list(agg.label) == ["overall"]
        # target is "att": overall_att is the equally-weighted post-period
        # average, NOT the per-event trimmed aggregate ATT (M-024 Note).
        assert list(agg.target) == ["att"]
        assert float(agg.att[0]) == stacked_fitted.overall_att
        assert float(agg.se[0]) == stacked_fitted.overall_se
        assert float(agg.t_stat[0]) == stacked_fitted.overall_t_stat
        assert float(agg.p_value[0]) == stacked_fitted.overall_p_value
        assert (float(agg.conf_int_lower[0]), float(agg.conf_int_upper[0])) == tuple(
            stacked_fitted.overall_conf_int
        )
        # Treated-unit count: control units OVERLAP treated across
        # sub-experiments, so no disjoint total exists (M-024 Note).
        assert float(agg.n[0]) == float(stacked_fitted.n_treated_units)
        assert agg.n_kind == "units"
        assert float(agg.df[0]) == float(stacked_fitted.inference_df)
        assert agg.estimator == "StackedDiD"

    @pytest.mark.parametrize("weighting", ["aggregate", "population", "sample_share"])
    def test_simple_target_att_on_all_weighting_schemes(self, stacked_panel, weighting):
        fit_kw = {}
        panel = stacked_panel
        if weighting == "population":
            panel = stacked_panel.copy()
            panel["pop"] = 100.0 + (panel["unit"] % 7)
            fit_kw["population"] = "pop"
        res = _fit_stacked(
            panel,
            est_kw={"kappa_pre": 2, "kappa_post": 2, "weighting": weighting},
            **fit_kw,
        )
        agg = res.aggregate("simple")
        assert list(agg.target) == ["att"]
        assert agg.n_kind == "units"

    def test_simple_relay_hc2_bm_df_is_overall_bm_dof(self, stacked_panel):
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        agg = res.aggregate("simple")
        assert np.isfinite(agg.df[0])
        assert float(agg.df[0]) == float(res.inference_df)
        # per-row BM dfs are present WITHOUT fit-time aggregate (M-024)
        assert res.event_study_df is not None
        assert all(np.isfinite(v) for v in res.event_study_df.values())

    def test_simple_relay_bm_failure_nan_inference(self, stacked_panel, monkeypatch):
        # The hc2_bm fail-closed state (finite att/se, jointly-NaN t/p/CI,
        # inference_df None) must RELAY through the simple view - the df
        # comparison is np.isnan, never df == inference_df (nan == None).
        import diff_diff.linalg as dl

        def _nan_dof(X, cluster_ids, bread_matrix, contrasts, weights=None):
            return np.full(contrasts.shape[1], np.nan)

        monkeypatch.setattr(dl, "_compute_cr2_bm_contrast_dof", _nan_dof)
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        assert res.inference_df is None
        agg = res.aggregate("simple")
        assert np.isfinite(agg.att[0]) and np.isfinite(agg.se[0])
        assert np.isnan(agg.t_stat[0])
        assert np.isnan(agg.p_value[0])
        assert np.isnan(agg.conf_int_lower[0]) and np.isnan(agg.conf_int_upper[0])
        assert np.isnan(agg.df[0])

    def test_simple_relay_survey_tsl_df(self, stacked_panel):
        from diff_diff.survey import SurveyDesign

        panel = stacked_panel.copy()
        panel["w"] = 1.0 + 0.1 * (panel["unit"] % 5)
        panel["strata"] = panel["unit"] % 4
        panel["psu"] = panel["unit"]
        res = _fit_stacked(
            panel,
            survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
        )
        agg = res.aggregate("simple")
        assert float(agg.att[0]) == res.overall_att
        # The stored overall inference used the survey df; the relay
        # carries exactly that provenance.
        assert float(agg.df[0]) == float(res.inference_df)

    def test_bm_dof_batch_parity_and_overall_reconstruction(self, stacked_panel, monkeypatch):
        # Pin (a): per-contrast Satterthwaite dof is column-independent in
        # VALUE on a well-conditioned design - the m=1 evaluation of each
        # contrast equals its column in the batched call (the batch-relative
        # noise floor changes only the DEGENERACY GUARD's scale, documented
        # in the REGISTRY M-024 Note).
        # Pin (b): the pre-change PLAIN-fit overall inference (m=1 batch) is
        # reconstructed in-process from the spy-captured fit-time locals and
        # must match the post-change stored overall inference at 1e-14.
        import diff_diff.linalg as dl
        from diff_diff.utils import safe_inference

        real = dl._compute_cr2_bm_contrast_dof
        captured = {}

        def spy(X, cluster_ids, bread_matrix, contrasts, weights=None):
            captured.update(
                X=X,
                cluster_ids=cluster_ids,
                bread=bread_matrix,
                contrasts=contrasts,
                weights=weights,
            )
            return real(X, cluster_ids, bread_matrix, contrasts, weights=weights)

        monkeypatch.setattr(dl, "_compute_cr2_bm_contrast_dof", spy)
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        assert captured, "spy never fired"
        contrasts = captured["contrasts"]
        batched = real(
            captured["X"],
            captured["cluster_ids"],
            captured["bread"],
            contrasts,
            weights=captured["weights"],
        )
        for j in range(contrasts.shape[1]):
            single = real(
                captured["X"],
                captured["cluster_ids"],
                captured["bread"],
                contrasts[:, [j]],
                weights=captured["weights"],
            )
            np.testing.assert_allclose(single[0], batched[j], rtol=1e-14)
        # (b) overall contrast is appended LAST at the fit site; its m=1
        # dof + safe_inference reproduce the stored overall inference.
        overall_dof_m1 = float(
            real(
                captured["X"],
                captured["cluster_ids"],
                captured["bread"],
                contrasts[:, [-1]],
                weights=captured["weights"],
            )[0]
        )
        np.testing.assert_allclose(res.inference_df, overall_dof_m1, rtol=1e-14)
        t, p, ci = safe_inference(
            res.overall_att, res.overall_se, alpha=res.alpha, df=overall_dof_m1
        )
        np.testing.assert_allclose(
            [t, ci[0], ci[1]],
            [
                res.overall_t_stat,
                res.overall_conf_int[0],
                res.overall_conf_int[1],
            ],
            rtol=1e-14,
        )
        # The p-value gets a looser rtol: batched-vs-m=1 BLAS kernels differ
        # at ~1 ULP in the dof (platform-dependent - observed on Linux
        # x86 OpenBLAS), and the deep tail amplifies that relative noise by
        # ~t^2 (d ln p ~ -t dt; t ~ 29 here => ~1e3x), so p carries ~1e-13
        # relative noise while t/CI stay at 1e-14.
        np.testing.assert_allclose(p, res.overall_p_value, rtol=1e-12)

    def test_legacy_pickle_absent_surface_hint(self, stacked_fitted):
        import dataclasses

        legacy = dataclasses.replace(
            stacked_fitted,
            event_study_effects=None,
            event_study_vcov=None,
            event_study_vcov_index=None,
            event_study_df=None,
        )
        with pytest.raises(ValueError, match=r"diff-diff >= 3\.9"):
            legacy.aggregate("event_study")

    def test_balance_e_rejected_empty_vocabulary(self, stacked_fitted):
        with pytest.raises(ValueError, match="no aggregation type on this estimator"):
            stacked_fitted.aggregate("event_study", balance_e=1)

    def test_weights_rejected(self, stacked_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            stacked_fitted.aggregate("simple", weights="cell")

    @pytest.mark.parametrize("bad", ["group", "calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, stacked_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            stacked_fitted.aggregate(bad)
