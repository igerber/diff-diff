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
