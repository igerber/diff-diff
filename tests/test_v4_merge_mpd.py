"""Phase 3(a) merge tests: TWFE event-study mode + MPD deprecation + time->post.

``test_ref`` for ledger rows M-010 (TwoWayFixedEffects absorbs
MultiPeriodDiD) and M-082 (static ``fit(time=)`` -> ``post=``), covering the
v4-design section 4.1 gate triple, the mode/rename validation surfaces, the
day-one auto-cluster carve-outs (user decision 2026-08-07), the deprecation
choreography (M-060's EventStudy warning rides the parent shim), the unified
surface contract, warning attribution, and the consumer ports.

Conventions: message pins via ``re.escape`` of the exact text; parity
assertions are BIT-EXACT (``assert_array_equal``) where the two sides run
the SAME code path in the same process (pooled vs MultiPeriodDiD - the
shared core), and ``assert_allclose`` at tight tolerance where the designs
differ (within vs pooled equivalence). PreTrendsPower parity lanes pin
``pretest_form="wald"``: the default ``"nis"`` box probability uses scipy's
``multivariate_normal.cdf``, which is internally randomized at ~1e-9 even
for identical repeated calls.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    EventStudy,
    MultiPeriodDiD,
    TwoWayFixedEffects,
    compute_honest_did,
)
from diff_diff.pretrends import compute_pretrends_power
from diff_diff.results_base import EventStudyResults, _from_mpd

# ---------------------------------------------------------------------------
# Pinned messages (the deprecation/validation contract)
# ---------------------------------------------------------------------------

MPD_DEPRECATION_MSG = (
    "MultiPeriodDiD is deprecated and will be removed in 4.0; use "
    "TwoWayFixedEffects().fit(..., event_study=True) instead - "
    "spec='pooled' reproduces the MultiPeriodDiD design; the default "
    "spec='within' adds unit fixed effects. The EventStudy alias is "
    "deprecated with it."
)
RENAME_MSG = (
    "TwoWayFixedEffects.fit(time=) is deprecated and will be removed in "
    "4.0; use post= instead. From 4.0, time= means the event-study "
    "calendar column only."
)
ES_WILD_MSG = (
    "inference='wild_bootstrap' is not supported in event-study mode: the "
    "wild cluster bootstrap covers the static ATT only. Use "
    "inference='analytical' for event-study fits."
)
WITHIN_UNIT_MSG = (
    "spec='within' requires unit=; the unit fixed effects are absorbed at "
    "the unit level. spec='pooled' is the only event-study spec that works "
    "without a unit id (repeated cross-sections)."
)
ES_POST_MSG = (
    "event-study mode takes time= (calendar column) as a keyword; post= is "
    "the static-mode 0/1 dummy. Pass "
    "fit(..., event_study=True, time='<calendar column>')."
)
ES_POST_PERIODS_MSG = (
    "event-study mode requires an explicit post_periods= (the "
    "post-treatment calendar periods): the treatment boundary is not "
    "observable from a time-invariant ever-treated treatment indicator, "
    "and the deprecated MultiPeriodDiD midpoint default (last half of the "
    "calendar) is deliberately not carried into the merged mode."
)


# ---------------------------------------------------------------------------
# Seeded DGPs
# ---------------------------------------------------------------------------


def _panel(seed=42, n_units=40, n_periods=6, covariate=False, unbalanced=False, str_periods=False):
    """Balanced simultaneous-adoption panel; knobs make it diverge.

    ``unbalanced=True`` drops the first two periods for a third of the
    control units; ``covariate=True`` adds a treatment-correlated covariate
    ``x`` entering the outcome. Both make the unit-FE projection change the
    point estimates (the documented within-vs-pooled estimate shift).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        ti = 1 if u < n_units // 2 else 0
        a_u = rng.normal() * 2.0
        x_u = rng.normal() + 0.8 * ti
        for p in range(n_periods):
            if unbalanced and ti == 0 and u % 3 == 0 and p < 2:
                continue
            x = x_u + 0.5 * rng.normal()
            y = (
                a_u
                + 0.3 * p
                + rng.normal() * 0.5
                + 0.8 * ti * (p >= 3)
                + (0.7 * x if covariate else 0.0)
            )
            rows.append(dict(unit=u, period=f"P{p}" if str_periods else p, treated=ti, x=x, y=y))
    return pd.DataFrame(rows)


def _post(n_periods=6, cut=3, str_periods=False):
    ps = list(range(cut, n_periods))
    return [f"P{p}" for p in ps] if str_periods else ps


@pytest.fixture()
def panel():
    return _panel()


def _mpd(**kwargs):
    """Construct MultiPeriodDiD with its deprecation warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return MultiPeriodDiD(**kwargs)


def _static_df(panel):
    return panel.assign(post=(panel["period"] >= 3).astype(int))


def _eq_with_nans(a, b):
    """Bit-exact parity that cannot mask NaN-vs-zero regressions: the NaN
    masks must match exactly, then the finite entries compare bit-equal
    (R4 - ``nan_to_num`` would silently equate a NaN on one side with a
    0.0 on the other, the exact partial-NaN shape the inference contract
    forbids)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def _close_with_nans(a, b, **tol):
    """allclose sibling of ``_eq_with_nans`` (same mask-first contract)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], **tol)


# ---------------------------------------------------------------------------
# The section 4.1 gate triple (+ the within numerical gate)
# ---------------------------------------------------------------------------


class TestGateTriple:
    def test_a_equivalence_balanced_no_covariates(self, panel):
        """(a) balanced / no covariates / simultaneous: within == pooled points."""
        w = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        p = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        _close_with_nans(w.att, p.att, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize(
        "dgp_kwargs, fit_kwargs, floor",
        [
            # measured max|within - pooled| = 0.2092 on seed 42 -> floor an
            # order of magnitude below, orders above numerical noise
            (dict(unbalanced=True), {}, 0.02),
            # measured 0.0214 on seed 42
            (dict(covariate=True), dict(covariates=["x"]), 0.002),
        ],
        ids=["unbalanced", "covariate"],
    )
    def test_b_divergence_locks_estimate_shift(self, dgp_kwargs, fit_kwargs, floor):
        """(b) unbalanced-or-covariate: within != pooled (the documented shift)."""
        df = _panel(**dgp_kwargs)
        w = TwoWayFixedEffects().fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
            **fit_kwargs,
        )
        p = TwoWayFixedEffects().fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
            **fit_kwargs,
        )
        assert np.nanmax(np.abs(w.att - p.att)) > floor

    def test_c_pooled_parity_unitless_repeated_cross_sections(self, panel):
        """(c)(i) unit-less pooled == 3.x MPD default lane, BIT-EXACT.

        Same code path in the same process (the shared core), so exact
        equality is the deliberate bar - the design's "reproduced exactly"
        gate, not a cross-implementation comparison.
        """
        s = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(_mpd().fit(panel, "y", "treated", "period", post_periods=_post()))
        for field in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            _eq_with_nans(getattr(s, field), getattr(m, field))
        np.testing.assert_array_equal(s.event_time, m.event_time)
        assert s.reference_period == m.reference_period
        _eq_with_nans(s.vcov, m.vcov)

    def test_c_pooled_parity_explicit_cluster(self, panel):
        """(c)(ii) pooled + matched explicit cluster= == MPD, BIT-EXACT."""
        s = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(
            _mpd(cluster="unit").fit(panel, "y", "treated", "period", post_periods=_post())
        )
        for field in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            _eq_with_nans(getattr(s, field), getattr(m, field))
        _eq_with_nans(s.vcov, m.vcov)

    def test_c_pooled_parity_hc2_bm_per_period_df(self, panel):
        """(c)(iii) hc2_bm: the per-period BM-DOF df channel round-trips."""
        s = TwoWayFixedEffects(vcov_type="hc2_bm").fit(
            panel,
            "y",
            "treated",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(
            _mpd(vcov_type="hc2_bm").fit(panel, "y", "treated", "period", post_periods=_post())
        )
        _eq_with_nans(s.df, m.df)
        # the df column is the finite per-period BM-DOF channel (on this
        # balanced DGP the per-period DOFs coincide numerically; the parity
        # equality above is the load-bearing pin)
        assert np.isfinite(np.asarray(s.df)[~s.is_reference]).all()
        for field in ("att", "se", "p_value"):
            _eq_with_nans(getattr(s, field), getattr(m, field))

    @pytest.mark.parametrize(
        "est_kwargs",
        [dict(cluster="unit"), dict(vcov_type="hc2")],
        ids=["explicit-cluster", "explicit-hc2-oneway"],
    )
    def test_d_within_matches_mpd_absorb_unit(self, panel, est_kwargs):
        """(d) within == MPD.fit(absorb=[unit]) under MATCHED inference.

        The designs coincide (MPD's absorb path snaps the unit-absorbed D
        column and rank-drops it; on the hc2 lane the absorb ->
        fixed_effects auto-route drops a redundant column instead - same
        column space either way), so the interaction block's quintet, ES
        vcov, and per-row df agree. Matched-inference lanes only: within
        auto-clusters at unit while MPD never auto-clusters, so their
        DEFAULTS diverge by construction (no "plain hc1" lane exists).
        """
        w = TwoWayFixedEffects(**est_kwargs).fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(
            _mpd(**est_kwargs).fit(
                panel, "y", "treated", "period", post_periods=_post(), absorb=["unit"]
            )
        )
        for field in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            _close_with_nans(getattr(w, field), getattr(m, field), atol=1e-12, rtol=1e-12)
        _close_with_nans(w.vcov, m.vcov, atol=1e-12, rtol=1e-12)
        _eq_with_nans(w.df, m.df)


# ---------------------------------------------------------------------------
# Mode validation + the rename shim
# ---------------------------------------------------------------------------


class TestModeValidation:
    def test_bad_spec_rejected_any_mode(self, panel):
        with pytest.raises(
            ValueError, match=re.escape("spec must be one of ('within', 'pooled'), got 'bogus'")
        ):
            TwoWayFixedEffects().fit(
                panel, "y", "treated", event_study=True, time="period", spec="bogus"
            )
        df = _static_df(panel)
        with pytest.raises(ValueError, match=re.escape("got 'bogus'")):
            TwoWayFixedEffects().fit(df, "y", "treated", post="post", unit="unit", spec="bogus")

    @pytest.mark.parametrize("vcov_type", ["hc2", "hc2_bm"])
    @pytest.mark.parametrize("missing", ["unit", "period"])
    def test_hc2_preflight_missing_columns_get_estimator_errors(self, panel, vcov_type, missing):
        """R7: the within HC2/HC2-BM memory preflight must not preempt
        column validation - a missing unit/calendar column raises the
        core's normal ValueError, never a raw pandas KeyError."""
        df = panel.drop(columns=[missing])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                TwoWayFixedEffects(vcov_type=vcov_type).fit(
                    df,
                    "y",
                    "treated",
                    unit="unit",
                    event_study=True,
                    time="period",
                    post_periods=_post(),
                )

    def test_post_periods_required_in_event_study(self, panel):
        """R4: the calendar partition is explicit - omitting (or emptying)
        post_periods= fails loud instead of inheriting the MPD midpoint
        guess (the boundary is unobservable from time-invariant D_i)."""
        for bad_kwargs in ({}, dict(post_periods=[])):
            with pytest.raises(ValueError, match=re.escape(ES_POST_PERIODS_MSG)):
                TwoWayFixedEffects().fit(
                    panel,
                    "y",
                    "treated",
                    unit="unit",
                    event_study=True,
                    time="period",
                    **bad_kwargs,
                )
        # MPD's own midpoint default is UNCHANGED (compatibility bar)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _mpd().fit(panel, "y", "treated", "period")
        assert np.isfinite(m.avg_att)
        # duplicates fail at the front door, before the regression runs
        # (R10 - the container's own validation would reject only at
        # conversion time)
        with pytest.raises(ValueError, match="duplicate labels"):
            TwoWayFixedEffects().fit(
                panel,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=[3, 3, 4],
            )
        # a one-shot iterable is materialized once, not exhausted by the
        # emptiness check (R8)
        gen = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=(p for p in _post()),
        )
        assert gen.post_periods == tuple(_post())

    @pytest.mark.parametrize(
        "kwargs, names",
        [
            (dict(spec="pooled"), "spec"),
            (dict(reference_period=2), "reference_period"),
            (dict(post_periods=[3, 4, 5]), "post_periods"),
        ],
    )
    def test_static_rejects_event_study_params(self, panel, kwargs, names):
        df = _static_df(panel)
        with pytest.raises(ValueError, match=rf"{names}.*require\(s\) event_study=True"):
            TwoWayFixedEffects().fit(df, "y", "treated", post="post", unit="unit", **kwargs)

    def test_within_requires_unit(self, panel):
        with pytest.raises(ValueError, match=re.escape(WITHIN_UNIT_MSG)):
            TwoWayFixedEffects().fit(panel, "y", "treated", event_study=True, time="period")

    def test_event_study_rejects_post(self, panel):
        with pytest.raises(ValueError, match=re.escape(ES_POST_MSG)):
            TwoWayFixedEffects().fit(
                panel, "y", "treated", post="period", unit="unit", event_study=True, spec="pooled"
            )

    def test_event_study_positional_slot4_lands_in_post(self, panel):
        # A positional 4th argument is the static post slot; under
        # event_study=True it hits the post= rejection steering to time=.
        with pytest.raises(ValueError, match=re.escape(ES_POST_MSG)):
            TwoWayFixedEffects().fit(
                panel, "y", "treated", "period", "unit", event_study=True, spec="pooled"
            )

    def test_event_study_missing_time_raises(self, panel):
        with pytest.raises(
            TypeError, match=re.escape("TwoWayFixedEffects.fit() missing required argument: 'time'")
        ):
            TwoWayFixedEffects().fit(panel, "y", "treated", unit="unit", event_study=True)

    def test_static_missing_post_raises(self, panel):
        with pytest.raises(
            TypeError, match=re.escape("TwoWayFixedEffects.fit() missing required argument: 'post'")
        ):
            TwoWayFixedEffects().fit(panel, "y", "treated")

    def test_static_missing_unit_raises(self, panel):
        df = _static_df(panel)
        with pytest.raises(
            TypeError, match=re.escape("TwoWayFixedEffects.fit() missing required argument: 'unit'")
        ):
            TwoWayFixedEffects().fit(df, "y", "treated", post="post")


class TestRenameShim:
    def test_canonical_post_silent(self, panel):
        df = _static_df(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r_kw = TwoWayFixedEffects().fit(df, "y", "treated", post="post", unit="unit")
            r_pos = TwoWayFixedEffects().fit(df, "y", "treated", "post", "unit")
        assert not [w for w in rec if issubclass(w.category, FutureWarning)]
        assert r_kw.att == r_pos.att

    def test_time_warns_and_routes_identically(self, panel):
        df = _static_df(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_new = TwoWayFixedEffects().fit(df, "y", "treated", post="post", unit="unit")
        with pytest.warns(FutureWarning, match=re.escape(RENAME_MSG)):
            r_old = TwoWayFixedEffects().fit(df, "y", "treated", time="post", unit="unit")
        assert r_old.att == r_new.att
        assert r_old.se == r_new.se
        assert r_old.t_stat == r_new.t_stat
        assert r_old.p_value == r_new.p_value
        assert r_old.conf_int == r_new.conf_int

    def test_both_supplied_raises(self, panel):
        df = _static_df(panel)
        with pytest.raises(ValueError, match=r"pass only post="):
            TwoWayFixedEffects().fit(df, "y", "treated", post="post", time="post", unit="unit")

    def test_event_study_time_emits_no_rename_warning(self, panel):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            TwoWayFixedEffects().fit(
                panel,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=_post(),
            )
        assert not [w for w in rec if "time=" in str(w.message) and "deprecated" in str(w.message)]


# ---------------------------------------------------------------------------
# Wild raise + auto-cluster carve-outs
# ---------------------------------------------------------------------------


class TestWildRaise:
    @pytest.mark.parametrize("n_bootstrap", [0, 1, 999])
    def test_wild_raises_at_any_n_bootstrap(self, panel, n_bootstrap):
        est = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=n_bootstrap)
        with pytest.raises(ValueError, match=re.escape(ES_WILD_MSG)):
            est.fit(panel, "y", "treated", unit="unit", event_study=True, time="period")

    def test_wild_raise_precedes_survey_front_door(self, panel):
        from diff_diff import SurveyDesign

        df = panel.assign(w=1.0)
        est = TwoWayFixedEffects(inference="wild_bootstrap")
        with pytest.raises(ValueError, match=re.escape(ES_WILD_MSG)):
            est.fit(
                df,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_wild_raise_precedes_conley_front_door(self, panel):
        df = panel.assign(lat=40.0, lon=-100.0)
        est = TwoWayFixedEffects(
            inference="wild_bootstrap",
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=100.0,
            conley_lag_cutoff=1,
        )
        with pytest.raises(ValueError, match=re.escape(ES_WILD_MSG)):
            est.fit(df, "y", "treated", unit="unit", event_study=True, time="period")


class TestAutoCluster:
    def test_auto_cluster_equals_explicit_unit(self, panel):
        r_auto = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        r_expl = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        _eq_with_nans(r_auto.se, r_expl.se)

    def test_pooled_without_unit_stays_one_way(self, panel):
        """No unit id -> no auto-cluster; == the MPD default (hc1) lane."""
        r = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(_mpd().fit(panel, "y", "treated", "period", post_periods=_post()))
        _eq_with_nans(r.se, m.se)

    def test_explicit_one_way_hc2_drops_auto_cluster(self, panel):
        """The one-way exception mirror: hc2 + analytical == MPD hc2."""
        r = TwoWayFixedEffects(vcov_type="hc2").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _from_mpd(
            _mpd(vcov_type="hc2").fit(panel, "y", "treated", "period", post_periods=_post())
        )
        _eq_with_nans(r.se, m.se)

    def test_conley_carve_out_drops_auto_cluster(self):
        """ES pooled + unit + conley + cluster=None == MPD conley cluster=None
        (auto-cluster dropped - no implicit spatial x unit product kernel);
        an explicit cluster= legitimately combines and differs."""
        rng = np.random.default_rng(3)
        rows = []
        for u in range(30):
            ti = 1 if u < 15 else 0
            lat, lon = rng.uniform(30, 45), rng.uniform(-100, -80)
            for p in range(5):
                rows.append(
                    dict(
                        unit=u,
                        period=p,
                        treated=ti,
                        lat=lat,
                        lon=lon,
                        y=rng.normal() + 0.6 * ti * (p >= 3),
                    )
                )
        df = pd.DataFrame(rows)
        kw = dict(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=500.0,
            conley_lag_cutoff=1,
        )
        r = TwoWayFixedEffects(**kw).fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=[3, 4],
        )
        m = _from_mpd(
            _mpd(**kw).fit(df, "y", "treated", "period", post_periods=[3, 4], unit="unit")
        )
        _eq_with_nans(r.se, m.se)
        _eq_with_nans(r.vcov, m.vcov)
        r_ex = TwoWayFixedEffects(cluster="unit", **kw).fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=[3, 4],
        )
        assert not np.allclose(np.nan_to_num(r_ex.se), np.nan_to_num(r.se))
        # within smoke: carve-out lane runs with finite inference
        w = TwoWayFixedEffects(**kw).fit(
            df, "y", "treated", unit="unit", event_study=True, time="period", post_periods=[3, 4]
        )
        assert np.isfinite(np.asarray(w.se)[~w.is_reference]).all()

    def test_survey_carve_out_no_implicit_psu(self, panel):
        """ES pooled + unit + no-PSU survey == MPD survey cluster=None
        (implicit per-observation PSUs - the auto-cluster is never injected);
        explicit cluster= injects the PSU and changes the SEs (matching
        MPD's own explicit-cluster behavior). Both lanes use a survey design
        WITHOUT its own PSU: when survey_design.psu is set, the shared
        resolver gives the PSU precedence over any cluster on every class."""
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(11)
        df = panel.assign(w=rng.uniform(0.5, 2.0, len(panel)))
        r = TwoWayFixedEffects().fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
            survey_design=SurveyDesign(weights="w"),
        )
        m = _from_mpd(
            _mpd().fit(
                df,
                "y",
                "treated",
                "period",
                post_periods=_post(),
                survey_design=SurveyDesign(weights="w"),
            )
        )
        _eq_with_nans(r.se, m.se)
        r_ex = TwoWayFixedEffects(cluster="unit").fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
            survey_design=SurveyDesign(weights="w"),
        )
        assert not np.allclose(np.nan_to_num(r_ex.se), np.nan_to_num(r.se))


# ---------------------------------------------------------------------------
# Deprecation choreography (M-010 warning; M-060 rides it)
# ---------------------------------------------------------------------------


class TestDeprecation:
    def test_mpd_construction_warns_pinned_message(self):
        with pytest.warns(FutureWarning, match=re.escape(MPD_DEPRECATION_MSG)):
            MultiPeriodDiD()

    def test_event_study_alias_warns_and_is_same_class(self):
        assert EventStudy is MultiPeriodDiD
        with pytest.warns(FutureWarning, match=re.escape(MPD_DEPRECATION_MSG)):
            EventStudy()

    def test_mpd_warns_and_still_works(self, panel):
        with pytest.warns(FutureWarning, match=re.escape(MPD_DEPRECATION_MSG)):
            est = MultiPeriodDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(panel, "y", "treated", "period", post_periods=_post())
        assert np.isfinite(r.avg_att)

    def test_successors_do_not_warn(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            TwoWayFixedEffects()
            DifferenceInDifferences()
        assert not [w for w in rec if issubclass(w.category, FutureWarning)]

    def test_mpd_introspection_intact(self):
        """The __signature__ mirror keeps the BaseEstimator contract."""
        m = _mpd()
        d = DifferenceInDifferences()
        assert set(m.get_params()) == set(d.get_params())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            m2 = m.set_params(alpha=0.10)
        assert m2.get_params()["alpha"] == 0.10


# ---------------------------------------------------------------------------
# The unified surface contract
# ---------------------------------------------------------------------------


class TestSurfaceContract:
    def test_non_midpoint_partition_selects_correct_reference(self):
        """R4: 8 periods with a declared boundary at period 2 - the
        explicit partition [2..7] selects reference 1 (last pre). The
        rejected midpoint guess would have partitioned at 4 and
        misclassified periods 2-3 as pre-treatment."""
        df = _panel(n_periods=8)
        s = TwoWayFixedEffects().fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=list(range(2, 8)),
        )
        assert s.post_periods == tuple(range(2, 8))
        assert s.reference_period == 1
        ref_mask = np.asarray(s.is_reference)
        assert s.event_time[ref_mask].tolist() == [1]

    def test_surface_provenance(self, panel):
        s = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        assert isinstance(s, EventStudyResults)
        assert s.time_scale == "calendar"
        assert s.source == "TwoWayFixedEffects"
        assert s.estimation_spec == "within"
        assert s.post_periods == tuple(_post())
        assert s.reference_period == 2  # default: last pre-period (e=-1)
        assert bool(s.is_reference[np.asarray(s.event_time) == 2][0])
        assert s.vcov is not None and s.vcov_index is not None
        p = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        assert p.estimation_spec == "pooled"

    def test_explicit_reference_period_honored(self, panel):
        s = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
            reference_period=1,
        )
        assert s.reference_period == 1

    def test_no_legacy_reference_warning_in_event_study_mode(self, panel):
        """The M-007 transition warning is MPD-only (v4-design section 4.1)."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            TwoWayFixedEffects().fit(
                panel,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=_post(),
            )
        assert not [w for w in rec if "reference_period has changed" in str(w.message)]
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            _mpd().fit(panel, "y", "treated", "period", post_periods=_post())
        assert [w for w in rec2 if "reference_period has changed" in str(w.message)]

    def test_renderers_run(self, panel):
        s = TwoWayFixedEffects().fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        assert "TwoWayFixedEffects" in s.summary() or len(s.summary()) > 0
        df = s.to_dataframe()
        assert len(df) == 6
        d = s.to_dict()
        assert d["estimation_spec"] == "within"
        assert d["post_periods"] == list(_post())

    def test_refit_static_then_event_study_and_back(self, panel):
        """Mode transitions on one estimator leave no stale state."""
        est = TwoWayFixedEffects()
        df = _static_df(panel)
        r1 = est.fit(df, "y", "treated", post="post", unit="unit")
        assert hasattr(r1, "att") and np.isfinite(r1.att)
        s = est.fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        assert isinstance(s, EventStudyResults)
        assert est.results_ is s
        r2 = est.fit(df, "y", "treated", post="post", unit="unit")
        assert r2.att == r1.att
        assert est.results_ is r2

    def test_survey_replicate_within_matches_mpd_absorb(self, panel, ci_params):
        """Numerical replicate-lane pin: the include_treatment_main threading
        through the absorb replicate-refit closure produces the same numbers
        as MPD.fit(absorb=[unit]) under matched explicit cluster."""
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(5)
        df = panel.assign(w=rng.uniform(0.5, 2.0, len(panel)))
        n_rep = max(4, min(8, ci_params.bootstrap(8)))
        rep_cols = {}
        for j in range(n_rep):
            rep_cols[f"rw{j}"] = df["w"] * rng.uniform(0.5, 1.5, len(df))
        df = df.assign(**rep_cols)
        sd_kwargs = dict(
            weights="w",
            replicate_weights=[f"rw{j}" for j in range(n_rep)],
            replicate_method="JK1",
        )
        w = TwoWayFixedEffects(cluster="unit").fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
            survey_design=SurveyDesign(**sd_kwargs),
        )
        m = _from_mpd(
            _mpd(cluster="unit").fit(
                df,
                "y",
                "treated",
                "period",
                post_periods=_post(),
                absorb=["unit"],
                survey_design=SurveyDesign(**sd_kwargs),
            )
        )
        for field in ("att", "se", "p_value"):
            _close_with_nans(getattr(w, field), getattr(m, field), atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Inference integrity: joint-NaN, estimator_name, warning attribution
# ---------------------------------------------------------------------------


class TestInferenceIntegrity:
    def test_within_rank_deficient_period_jointly_nan(self):
        """A period with no treated observations drops its interaction: that
        row's FULL inference tuple is jointly NaN; identified rows finite."""
        rng = np.random.default_rng(7)
        rows = []
        for u in range(30):
            ti = 1 if u < 15 else 0
            for p in range(5):
                if ti == 1 and p == 1:
                    continue  # no treated obs in period 1 -> interaction drops
                rows.append(
                    dict(unit=u, period=p, treated=ti, y=rng.normal() + 0.5 * ti * (p >= 3))
                )
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = TwoWayFixedEffects(rank_deficient_action="silent").fit(
                df,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=[3, 4],
            )
        et = np.asarray(s.event_time)
        bad = (et == 1) & (~s.is_reference)
        good = (~s.is_reference) & (et != 1)
        assert bad.sum() == 1
        for field in ("se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
            arr = np.asarray(getattr(s, field))
            assert np.isnan(arr[bad]).all(), f"{field} not NaN on the dropped row"
            assert np.isfinite(arr[good]).all(), f"{field} not finite on identified rows"
        assert np.isnan(np.asarray(s.df)[bad]).all()

    def test_estimator_name_threading(self, panel):
        """TWFE event-study messages name TwoWayFixedEffects, never the
        deprecated class; MPD's own messages are bit-identical legacy."""
        stag = panel.copy()
        # two adoption cohorts among former controls (0->1 transitions at
        # periods 1 and 3) so the staggered-adoption advisory fires
        stag.loc[(stag.unit >= 30) & (stag.unit < 33) & (stag.period >= 1), "treated"] = 1
        stag.loc[(stag.unit >= 33) & (stag.unit < 36) & (stag.period >= 3), "treated"] = 1
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            TwoWayFixedEffects().fit(
                stag,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                spec="pooled",
                time="period",
                post_periods=_post(),
            )
        adv = [w for w in rec if "simultaneous adoption" in str(w.message)]
        assert adv and "TwoWayFixedEffects" in str(adv[0].message)
        assert "MultiPeriodDiD" not in str(adv[0].message)
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            _mpd().fit(stag, "y", "treated", "period", post_periods=_post(), unit="unit")
        adv2 = [w for w in rec2 if "simultaneous adoption" in str(w.message)]
        assert adv2 and "MultiPeriodDiD" in str(adv2[0].message)
        # validation errors carry the producer too
        with pytest.raises(ValueError, match="TwoWayFixedEffects"):
            TwoWayFixedEffects().fit(
                panel.assign(const=1.0),
                "y",
                "treated",
                unit="unit",
                event_study=True,
                spec="pooled",
                time="period",
                post_periods=_post(),
                covariates=["const"],
            )

    def test_staggered_timing_undetectable_under_time_invariant_di(self, panel):
        """Documented detection limit (REGISTRY 'staggered-adoption
        detection limit' Notes; TODO.md 3(a) R2 cohort-timing row): with
        the contract-valid time-invariant ever-treated D_i, adoption
        timing is not observable in the inputs, so a genuinely staggered
        two-cohort design fits with NO staggered advisory and NO D_it
        warning. This pins the documented limitation so the behavior
        change is visible when the cohort= validation input lands."""
        stag = panel.copy()
        # two cohorts adopting at periods 1 and 3 - but encoded as the
        # documented ever-treated D_i (1 in ALL periods for both cohorts),
        # so the timing difference never appears in the treatment column
        stag.loc[(stag.unit >= 30) & (stag.unit < 36), "treated"] = 1
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = TwoWayFixedEffects().fit(
                stag,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                spec="pooled",
                time="period",
                post_periods=_post(),
            )
        assert not any("simultaneous adoption" in str(w.message) for w in rec)
        assert not any("varies within units" in str(w.message) for w in rec)
        assert np.all(np.isfinite(res.att))

    def test_singleton_units_retained_class_consistently(self, panel):
        """R5: singleton units are RETAINED, not dropped - the REGISTRY
        'Deviation from R' Note (reghdfe iteratively drops singletons,
        fixest retains them; diff-diff matches the fixest default,
        class-wide). The singleton's unit-demeaned row is zero, so
        event-study points are unchanged while N/G/df count it (SEs
        shift); the within spec inherits the class behavior EXACTLY -
        bit-parity with the same-data MPD absorb fit holds ON the
        singleton fixture, so the new mode introduces no divergence.
        Opt-in pruning is the TODO.md 3(a) R5 row."""
        single = pd.DataFrame([dict(unit=999, period=0, treated=0, x=0.0, y=0.0)])
        aug = pd.concat([panel, single], ignore_index=True)
        base = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        with_s = TwoWayFixedEffects(cluster="unit").fit(
            aug,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )
        _close_with_nans(base.att, with_s.att, atol=1e-9, rtol=1e-9)
        assert not np.allclose(np.nan_to_num(base.se), np.nan_to_num(with_s.se))
        m = _from_mpd(
            _mpd(cluster="unit").fit(
                aug, "y", "treated", "period", post_periods=_post(), absorb=["unit"]
            )
        )
        for field in ("att", "se", "t_stat", "p_value"):
            _close_with_nans(getattr(with_s, field), getattr(m, field), atol=1e-12, rtol=1e-12)
        # static path: the same retained-singleton behavior (point pinned
        # to ~ULP - the two-way alternating-projection demeaning remixes
        # means - while the SE visibly shifts with N/G/df)
        st = _static_df(panel)
        st_aug = pd.concat([st, single.assign(post=0)], ignore_index=True)
        s0 = TwoWayFixedEffects().fit(st, "y", "treated", post="post", unit="unit")
        s1 = TwoWayFixedEffects().fit(st_aug, "y", "treated", post="post", unit="unit")
        assert np.isclose(s0.att, s1.att)
        assert s0.se != s1.se

    def test_warning_attribution_baselines(self, panel):
        """The three A1 attribution pins: inline + snap -> USER file;
        the solve_ols rank-deficiency chain -> estimators.py (preserved
        library attribution; never a user-file pin on that class)."""
        stag = panel.copy()
        # two adoption cohorts among former controls (0->1 transitions at
        # periods 1 and 3) so the staggered-adoption advisory fires
        stag.loc[(stag.unit >= 30) & (stag.unit < 33) & (stag.period >= 1), "treated"] = 1
        stag.loc[(stag.unit >= 33) & (stag.unit < 36) & (stag.period >= 3), "treated"] = 1
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _mpd().fit(stag, "y", "treated", "period", post_periods=_post(), unit="unit")
        inline = [w for w in rec if "simultaneous adoption" in str(w.message)]
        assert inline and inline[0].filename.endswith("test_v4_merge_mpd.py")
        # snap (class-(i)-via-helper): unit-constant covariate under absorb
        df2 = panel.assign(ucov=(panel["unit"] % 3).astype(float))
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            _mpd().fit(
                df2,
                "y",
                "treated",
                "period",
                post_periods=_post(),
                absorb=["unit"],
                covariates=["ucov"],
                reference_period=2,
            )
        snap = [w for w in rec2 if "collinear with the absorbed" in str(w.message)]
        assert snap and snap[0].filename.endswith("test_v4_merge_mpd.py")
        rank = [w for w in rec2 if "Rank-deficient design matrix" in str(w.message)]
        assert rank and rank[0].filename.endswith("estimators.py")
        # the TWFE event-study path preserves the same attribution classes
        with warnings.catch_warnings(record=True) as rec3:
            warnings.simplefilter("always")
            TwoWayFixedEffects().fit(
                df2,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=_post(),
                covariates=["ucov"],
                reference_period=2,
            )
        snap3 = [w for w in rec3 if "collinear with the absorbed" in str(w.message)]
        assert snap3 and snap3[0].filename.endswith("test_v4_merge_mpd.py")


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


class TestConsumers:
    def _surfaces(self, panel, **mpd_kwargs):
        s = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        m = _mpd(cluster="unit").fit(panel, "y", "treated", "period", post_periods=_post())
        return s, m

    def test_honest_did_parity_standard_geometry(self, panel):
        s, m = self._surfaces(panel)
        for M in (0.5, 1.0):
            hs = compute_honest_did(s, M=M)
            hn = compute_honest_did(m, M=M)
            assert hs.ci_lb == hn.ci_lb and hs.ci_ub == hn.ci_ub
            assert hs.pre_periods_used == hn.pre_periods_used

    def test_honest_did_survey_df_scalar_threads(self, panel):
        """Survey-backed lane: the df_survey scalar that moves FLCI critical
        values threads identically through the calendar route."""
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(13)
        df = panel.assign(w=rng.uniform(0.5, 2.0, len(panel)))
        s = TwoWayFixedEffects().fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
            survey_design=SurveyDesign(weights="w"),
        )
        m = _mpd().fit(
            df,
            "y",
            "treated",
            "period",
            post_periods=_post(),
            survey_design=SurveyDesign(weights="w"),
        )
        hs = compute_honest_did(s, M=1.0)
        hn = compute_honest_did(m, M=1.0)
        assert hs.ci_lb == hn.ci_lb and hs.ci_ub == hn.ci_ub

    @pytest.mark.parametrize(
        "fit_kwargs",
        [
            dict(post_periods=[2, 5], reference_period=4),
            dict(post_periods=[3, 4, 5], reference_period=1),
        ],
        ids=["non-suffix", "non-last-reference"],
    )
    def test_honest_did_rejects_non_chronological_geometry(self, panel, fit_kwargs):
        s = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            **fit_kwargs,
        )
        with pytest.raises(ValueError, match="Registry-valid chronological geometry"):
            compute_honest_did(s, M=1.0)

    def test_honest_did_rejects_missing_provenance_and_multi_reference(self):
        base = dict(
            att=[0.1, 0.0, 0.5, 0.6],
            se=[0.1, np.nan, 0.1, 0.1],
            t_stat=[1.0, np.nan, 5.0, 6.0],
            p_value=[0.3, np.nan, 0.01, 0.01],
            conf_int_lower=[0.0] * 4,
            conf_int_upper=[1.0] * 4,
            n=[np.nan] * 4,
            time_scale="calendar",
            source="TwoWayFixedEffects",
        )
        no_prov = EventStudyResults(
            event_time=[0, 1, 2, 3], is_reference=[False, True, False, False], **base
        )
        with pytest.raises(TypeError, match="post_periods partition provenance"):
            compute_honest_did(no_prov, M=1.0)
        with pytest.raises(TypeError, match="post_periods partition provenance"):
            compute_pretrends_power(no_prov, M=0.3)
        multi_base = dict(base)
        multi_base.update(
            att=[0.0, 0.0, 0.5, 0.6],
            se=[np.nan, np.nan, 0.1, 0.1],
            t_stat=[np.nan, np.nan, 5.0, 6.0],
            p_value=[np.nan, np.nan, 0.01, 0.01],
        )
        multi_ref = EventStudyResults(
            event_time=[0, 1, 2, 3],
            is_reference=[True, True, False, False],
            post_periods=(2, 3),
            **multi_base,
        )
        with pytest.raises(ValueError, match="exactly one reference row"):
            compute_honest_did(multi_ref, M=1.0)
        with pytest.raises(ValueError, match="exactly one reference row"):
            compute_pretrends_power(multi_ref, M=0.3)

    def test_calendar_route_rejects_foreign_source(self):
        foreign = EventStudyResults(
            event_time=[0, 1, 2],
            att=[0.0, 0.5, 0.6],
            se=[np.nan, 0.1, 0.1],
            t_stat=[np.nan, 5.0, 6.0],
            p_value=[np.nan, 0.01, 0.01],
            conf_int_lower=[0.0] * 3,
            conf_int_upper=[1.0] * 3,
            is_reference=[True, False, False],
            n=[np.nan] * 3,
            time_scale="calendar",
            source="SomethingElse",
            post_periods=(1, 2),
        )
        with pytest.raises(TypeError, match="TwoWayFixedEffects event-study mode"):
            compute_honest_did(foreign, M=1.0)
        with pytest.raises(TypeError, match="TwoWayFixedEffects event-study mode"):
            compute_pretrends_power(foreign, M=0.3)

    def test_no_unknown_provenance_warning_on_first_party_surface(self, panel):
        s, _ = self._surfaces(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            compute_honest_did(s, M=1.0)
        assert not [w for w in rec if "provenance" in str(w.message)]

    @pytest.mark.parametrize(
        "fit_kwargs",
        [
            dict(post_periods=[3, 4, 5]),
            dict(post_periods=[2, 5], reference_period=4),
            dict(post_periods=[3, 4, 5], reference_period=1),
        ],
        ids=["suffix", "non-suffix", "non-last-reference"],
    )
    def test_pretrends_parity_wald(self, panel, fit_kwargs):
        """PreTrendsPower is NOT geometry-scoped: all three lanes are parity
        lanes (wald form - the nis box probability is internally randomized)."""
        s = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            **fit_kwargs,
        )
        m = _mpd(cluster="unit").fit(panel, "y", "treated", "period", **fit_kwargs)
        a = compute_pretrends_power(s, M=0.3, pretest_form="wald")
        b = compute_pretrends_power(m, M=0.3, pretest_form="wald")
        assert a.power == b.power and a.mdv == b.mdv
        assert a.covariance_source == b.covariance_source == "full_pre_period_vcov"

    def test_pretrends_string_labels_degrade_like_native(self):
        """String calendar labels reproduce the native route's gamma-unit
        degradation warning - never silence, never bypass."""
        df = _panel(str_periods=True)
        posts = _post(str_periods=True)
        s = TwoWayFixedEffects(cluster="unit").fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=posts,
        )
        m = _mpd(cluster="unit").fit(df, "y", "treated", "period", post_periods=posts)
        with warnings.catch_warnings(record=True) as r1:
            warnings.simplefilter("always")
            q1 = compute_pretrends_power(s, M=0.3, pretest_form="wald")
        with warnings.catch_warnings(record=True) as r2:
            warnings.simplefilter("always")
            q2 = compute_pretrends_power(m, M=0.3, pretest_form="wald")
        w1 = sorted(str(w.message) for w in r1 if "reference_period" in str(w.message))
        w2 = sorted(str(w.message) for w in r2 if "reference_period" in str(w.message))
        assert w1 == w2 and len(w1) == 1
        assert (q1.mdv == q2.mdv) or (np.isnan(q1.mdv) and np.isnan(q2.mdv))

    def test_pretrends_explicit_pre_periods_validated(self, panel):
        """R8: an explicit pre_periods= selection on the calendar route
        is VALIDATED and chronologically ordered (the relative container
        route's contract) - unknown labels, the reference row,
        duplicates all fail loud, and a reversed-but-valid selection
        gives the identical positional analysis (last_period targets the
        chronological grid, not argument order)."""
        s, _ = self._surfaces(panel)
        with pytest.raises(ValueError, match="not eligible pre-treatment periods"):
            compute_pretrends_power(s, M=0.3, pre_periods=[0, 999], pretest_form="wald")
        with pytest.raises(ValueError, match="not eligible pre-treatment periods"):
            # period 2 is the reference row
            compute_pretrends_power(s, M=0.3, pre_periods=[0, 2], pretest_form="wald")
        with pytest.raises(ValueError, match="duplicate labels"):
            compute_pretrends_power(s, M=0.3, pre_periods=[0, 0], pretest_form="wald")
        fwd = compute_pretrends_power(
            s, M=0.3, pre_periods=[0, 1], violation_type="last_period", pretest_form="wald"
        )
        rev = compute_pretrends_power(
            s, M=0.3, pre_periods=[1, 0], violation_type="last_period", pretest_form="wald"
        )
        assert fwd.power == rev.power and fwd.mdv == rev.mdv
        # R9: a NaN coefficient beside a finite SE (hand-built shape -
        # producer rows are NaN-consistent) is ineligible: dropped from
        # automatic selection, rejected when explicitly requested
        import dataclasses

        row_of = {t: i for i, t in enumerate(s.event_time.tolist())}
        att_bad = np.asarray(s.att, dtype=float).copy()
        att_bad[row_of[0]] = np.nan
        bad = dataclasses.replace(s, att=att_bad)
        auto = compute_pretrends_power(bad, M=0.3, pretest_form="wald")
        assert np.isfinite(auto.power)  # period 0 excluded, period 1 carries
        with pytest.raises(ValueError, match="not eligible pre-treatment periods"):
            compute_pretrends_power(bad, M=0.3, pre_periods=[0, 1], pretest_form="wald")

    def test_honest_string_labels_warn_ambiguous_chronology(self):
        """R6: string calendar labels cannot prove chronology - the
        HonestDiD calendar route warns loudly (sorted() order is only
        ASSUMED; unpadded numeric suffixes would reorder, silently
        shifting the positional l_vec target) while staying
        native-route consistent: the fit itself applied the same
        sorted() rule, so both routes share one ordering."""
        df = _panel(str_periods=True)
        posts = _post(str_periods=True)
        s = TwoWayFixedEffects(cluster="unit").fit(
            df,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=posts,
        )
        m = _mpd(cluster="unit").fit(df, "y", "treated", "period", post_periods=posts)
        with pytest.warns(UserWarning, match="STRING calendar labels"):
            hs = compute_honest_did(s, M=1.0)
        hn = compute_honest_did(m, M=1.0)
        assert hs.ci_lb == hn.ci_lb and hs.ci_ub == hn.ci_ub

    def test_calendar_vcov_integrity_rejections(self, panel):
        """R3 hardening: the calendar routes share the relative container
        path's vcov integrity contract instead of silently degrading to
        diag(se^2) - duplicate/incomplete vcov_index and malformed
        sub-blocks fail loud; HonestDiD additionally rejects singular
        blocks (allow_singular=False) while PreTrendsPower keeps its
        documented singular handling."""
        import dataclasses

        s, _ = self._surfaces(panel)
        labels = list(s.vcov_index.tolist())
        row_of = {t: i for i, t in enumerate(s.event_time.tolist())}

        dup = dataclasses.replace(s, vcov_index=np.array([labels[0]] + labels[1:-1] + [labels[0]]))
        for consumer in (compute_honest_did, compute_pretrends_power):
            with pytest.raises(ValueError, match="carries duplicate"):
                consumer(dup, M=0.5)

        # index/matrix shrunk together so the container validates, but the
        # first pre-period label is gone from the covariance index
        shrunk = dataclasses.replace(
            s,
            vcov=np.asarray(s.vcov)[1:, 1:],
            vcov_index=np.array(labels[1:]),
        )
        with pytest.raises(ValueError, match="retained horizon"):
            compute_honest_did(shrunk, M=0.5)
        with pytest.raises(ValueError, match="missing one of the pre-period labels"):
            compute_pretrends_power(shrunk, M=0.5)

        asym = np.asarray(s.vcov, dtype=float).copy()
        asym[0, 1] = asym[0, 1] + 0.01  # not mirrored
        for consumer in (compute_honest_did, compute_pretrends_power):
            with pytest.raises(ValueError, match="not symmetric"):
                consumer(dataclasses.replace(s, vcov=asym), M=0.5)

        for consumer in (compute_honest_did, compute_pretrends_power):
            with pytest.raises(ValueError, match="inconsistent with the stored standard"):
                consumer(dataclasses.replace(s, vcov=np.asarray(s.vcov) * 4.0), M=0.5)

        # rank-1 PSD matrix with the exact se^2 diagonal: singular but
        # otherwise well-formed
        se_r = np.array([float(s.se[row_of[t]]) for t in labels])
        rank1 = dataclasses.replace(s, vcov=np.outer(se_r, se_r))
        with pytest.raises(ValueError, match="singular or near-singular"):
            compute_honest_did(rank1, M=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_pretrends_power(rank1, M=0.5)  # documented singular handling

    def test_calendar_vcov_none_fallback_and_finite_effect_filter(self, panel):
        """R3: a vcov-less container takes the WARNED diagonal fallback on
        HonestDiD (the relative container path's message), and a non-finite
        coefficient with a positive SE is filtered like the native MPD
        branch - an interior NaN effect breaks the consecutive grid and
        fails loud, a leading one is dropped safely."""
        import dataclasses

        s, _ = self._surfaces(panel)
        bare = dataclasses.replace(s, vcov=None, vcov_index=None)
        with pytest.warns(UserWarning, match="no full covariance matrix"):
            hb = compute_honest_did(bare, M=0.5)
        assert np.isfinite(hb.ci_lb) and np.isfinite(hb.ci_ub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_pretrends_power(bare, M=0.5)

        row_of = {t: i for i, t in enumerate(s.event_time.tolist())}
        att_interior = np.asarray(s.att, dtype=float).copy()
        att_interior[row_of[1]] = np.nan  # interior pre-period (pre grid 0,1)
        with pytest.raises(ValueError, match="consecutive estimated horizons"):
            compute_honest_did(dataclasses.replace(s, att=att_interior), M=0.5)

        att_leading = np.asarray(s.att, dtype=float).copy()
        att_leading[row_of[0]] = np.nan  # leading pre-period: droppable
        h_lead = compute_honest_did(dataclasses.replace(s, att=att_leading), M=0.5)
        assert h_lead.pre_periods_used == [1]

    def test_honest_rejects_all_post_nan_surface(self, panel):
        """R4: a surface whose every post-period row carries withheld
        (NaN) inference passes the trailing-drop geometry but has no
        sensitivity target - both the calendar route and the guard's
        native sibling fail loud instead of handing num_post=0 to the
        restriction system."""
        import dataclasses

        s, _ = self._surfaces(panel)
        row_of = {t: i for i, t in enumerate(s.event_time.tolist())}
        arrays = {
            f: np.asarray(getattr(s, f), dtype=float).copy()
            for f in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper")
        }
        for p_ in s.post_periods:
            for f in arrays:
                arrays[f][row_of[p_]] = np.nan
        bad = dataclasses.replace(s, **arrays)
        with pytest.raises(ValueError, match="No post-period effects with finite estimates"):
            compute_honest_did(bad, M=0.5)

    def test_diagnostic_and_business_report_reject_surface(self, panel):
        from diff_diff.business_report import BusinessReport
        from diff_diff.diagnostic_report import DiagnosticReport

        s, _ = self._surfaces(panel)
        with pytest.raises(TypeError, match="DiagnosticReport does not yet support"):
            DiagnosticReport(s)
        with pytest.raises(TypeError, match="BusinessReport does not yet support"):
            BusinessReport(s)

    def test_plot_partition_aware_shading_both_renderers(self, panel):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from diff_diff.visualization import plot_event_study
        from diff_diff.visualization._event_study import _pre_shading_runs

        assert _pre_shading_runs([0, 1, 3]) == [(0, 1), (3, 3)]
        s = TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=[2, 5],
            reference_period=4,
        )
        # pre = {0, 1, 3} (positions 0, 1, 3); post period 2 (position 2)
        # must NOT be covered by any shaded span on either renderer.
        plot_event_study(s)
        ax = plt.gcf().axes[0]
        spans = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches)
        assert spans == [(-0.5, 1.5), (2.5, 3.5)]
        plt.close("all")
        plotly = pytest.importorskip("plotly")
        del plotly
        fig = plot_event_study(s, backend="plotly")
        vrects = sorted(
            (sh.x0, sh.x1) for sh in fig.layout.shapes if getattr(sh, "type", "") == "rect"
        )
        assert (-0.5, 1.5) in vrects and (2.5, 3.5) in vrects
        assert not any(x0 < 2.0 < x1 for x0, x1 in vrects)
