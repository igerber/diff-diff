"""Merge tests: TWFE event-study mode, the MultiPeriodDiD removal, time->post.

``test_ref`` for ledger rows M-010 (TwoWayFixedEffects absorbs
MultiPeriodDiD), M-082 (static ``fit(time=)`` -> ``post=``) and M-083 (static
``time=`` raises since 4.0), covering the
v4-design section 4.1 gate triple, the mode/rename validation surfaces, the
day-one auto-cluster carve-outs (user decision 2026-08-07), the 4.0 removal
pins (M-010/M-011/M-012/M-060: the names are gone; M-007: the legacy
reference-period warning is retired; M-016: ``period_effects`` is a
FutureWarning view on the surface), the unified surface contract, warning
attribution, and the consumer ports. Through 3.x the MultiPeriodDiD class was
the live twin of every lane here; those parity pins moved to the committed
pre-refactor oracle in ``tests/test_twfe_event_study_decoupling.py``.

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
from event_study_surface_builders import calendar_surface

import diff_diff
from diff_diff import (
    CallawaySantAnna,
    DifferenceInDifferences,
    SurveyDesign,
    TwoWayFixedEffects,
    compute_honest_did,
    generate_staggered_data,
)
from diff_diff.pretrends import compute_pretrends_power
from diff_diff.results_base import (
    EventStudyResults,
    _PeriodEffectView,
    build_event_study_surface,
)
from diff_diff.visualization import plot_event_study, plot_honest_event_study

# ---------------------------------------------------------------------------
# Pinned messages (the deprecation/validation contract)
# ---------------------------------------------------------------------------

PERIOD_EFFECTS_MSG = (
    "EventStudyResults.period_effects is deprecated and will be removed in "
    "5.0; use to_dataframe() (one row per event_time, reference row included) "
    "or the columnar att/se/t_stat/p_value/conf_int_lower/conf_int_upper "
    "arrays instead."
)
REMOVED_NAMES = ("MultiPeriodDiD", "MultiPeriodDiDResults", "PeriodEffect", "EventStudy")
STATIC_TIME_MSG = (
    "TwoWayFixedEffects.fit: in static mode (event_study=False) time= is "
    "the event-study calendar column only (4.0, row M-083); pass the 0/1 "
    "post-treatment indicator as post=."
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
    "and the pre-4.0 midpoint default (last half of the calendar) is "
    "deliberately not carried into the merged mode."
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
        post_periods= fails loud instead of inheriting the pre-4.0 midpoint
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

    def test_time_in_static_mode_raises_naming_post(self, panel):
        """M-083 (4.0): in static mode `time=` is the event-study calendar
        column only; passing it raises and points at `post=` instead of being
        reinterpreted as the post dummy (the 3.x M-082 FutureWarning shim)."""
        df = _static_df(panel)
        with pytest.raises(ValueError, match=re.escape(STATIC_TIME_MSG)) as exc:
            TwoWayFixedEffects().fit(df, "y", "treated", time="post", unit="unit")
        assert "post=" in str(exc.value)
        # The canonical keyword still fits, warning-free.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = TwoWayFixedEffects().fit(df, "y", "treated", post="post", unit="unit")
        assert np.isfinite(r.att)

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

    def test_conley_carve_out_drops_auto_cluster(self):
        """ES pooled + unit + conley + cluster=None drops the auto-cluster
        (no implicit spatial x unit product kernel; through 3.x this lane was
        pinned bit-exact to the MultiPeriodDiD conley fit, now the committed
        oracle's within_conley lane); an explicit cluster= legitimately
        combines and differs."""
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
        assert np.isfinite(np.asarray(r.se)[~r.is_reference]).all()
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
        """ES pooled + unit + no-PSU survey keeps the implicit per-observation
        PSUs (the auto-cluster is never injected); explicit cluster= injects
        the PSU and changes the SEs. Both lanes use a survey design WITHOUT
        its own PSU: when survey_design.psu is set, the shared resolver gives
        the PSU precedence over any cluster on every class."""
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
        assert np.isfinite(np.asarray(r.se)[~r.is_reference]).all()
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
# The 4.0 removal (M-010 / M-011 / M-012 / M-060) and the retired M-007 warning
# ---------------------------------------------------------------------------


class TestGateERemoval:
    """The removed names are absent everywhere a user could reach them."""

    @pytest.mark.parametrize("name", REMOVED_NAMES)
    def test_name_is_absent(self, name):
        assert not hasattr(diff_diff, name)
        assert name not in diff_diff.__all__
        assert name not in {str(n) for n in dir(diff_diff)}
        with pytest.raises(ImportError):
            exec(f"from diff_diff import {name}", {})  # noqa: S102

    def test_internal_adapters_are_gone(self):
        import diff_diff.estimators as est
        import diff_diff.results as res
        import diff_diff.results_base as rb

        assert not hasattr(rb, "_from_mpd")
        assert not hasattr(est, "_mpd_results_from_payload")
        assert not hasattr(est, "_MPD_DEPRECATION_MSG")
        assert not hasattr(res, "MultiPeriodDiDResults") and not hasattr(res, "PeriodEffect")

    def test_successors_do_not_warn(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            TwoWayFixedEffects()
            DifferenceInDifferences()
        assert not [w for w in rec if issubclass(w.category, FutureWarning)]

    def test_core_requires_post_periods_even_when_called_directly(self, panel):
        """The core's own fail-closed raise (the pre-4.0 midpoint default is
        gone; the front door raises earlier, so only a direct call reaches it)."""
        est = TwoWayFixedEffects()
        with pytest.raises(ValueError, match="requires an explicit, non-empty post_periods="):
            est._fit_event_study_core(
                panel,
                "y",
                "treated",
                "period",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "analytical",
                include_treatment_main=True,
                cluster_override=None,
            )

    def test_hc2_bm_contrast_batch_keeps_post_average_anchor(self, panel, monkeypatch):
        """The clustered CR2 Bell-McCaffrey DOF guard is batch-relative, so the
        core keeps the (otherwise unused) post-period-average contrast column as
        the batch anchor (user decision 2026-10-25): structural pin that the
        contrast matrix handed to linalg has n_kept + 1 columns whose last is
        1/n_post on the post-period interaction rows and 0 elsewhere."""
        import diff_diff.linalg as linalg

        seen = {}
        real = linalg._compute_cr2_bm_vcov_and_dof

        def spy(X, cluster_ids, bread, contrasts, **kw):
            seen["contrasts"] = np.asarray(contrasts).copy()
            return real(X, cluster_ids, bread, contrasts, **kw)

        monkeypatch.setattr(linalg, "_compute_cr2_bm_vcov_and_dof", spy)
        s = TwoWayFixedEffects(cluster="unit", vcov_type="hc2_bm").fit(
            panel,
            "y",
            "treated",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )
        c = seen["contrasts"]
        n_kept = c.shape[0]
        assert c.shape == (n_kept, n_kept + 1)
        np.testing.assert_array_equal(c[:, :n_kept], np.eye(n_kept))
        anchor = c[:, -1]
        n_post = len(_post())
        post_rows = [i for i, lab in enumerate(s.vcov_index.tolist()) if lab in _post()]
        assert len(post_rows) == n_post
        # the surface's vcov_index rows are the interaction columns in label
        # order; locate them in the kept design by their anchor weight
        assert np.isclose(anchor.sum(), 1.0)
        assert np.count_nonzero(anchor) == n_post
        assert np.allclose(anchor[np.nonzero(anchor)], 1.0 / n_post)


class TestPeriodEffectsView:
    """M-016: ``period_effects`` is a FutureWarning view on calendar surfaces."""

    def _fit(self, panel, alpha=0.05):
        return TwoWayFixedEffects(alpha=alpha).fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            time="period",
            post_periods=_post(),
        )

    def test_is_a_property_not_a_field(self):
        import dataclasses
        import inspect

        assert "period_effects" not in {f.name for f in dataclasses.fields(EventStudyResults)}
        assert isinstance(inspect.getattr_static(EventStudyResults, "period_effects"), property)

    def test_warns_once_and_matches_the_surface_rowwise(self, panel):
        s = self._fit(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pe = s.period_effects
        fw = [w for w in rec if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1 and str(fw[0].message) == PERIOD_EFFECTS_MSG
        labels = [lab for lab, ref in zip(s.event_time.tolist(), s.is_reference) if not ref]
        assert list(pe) == labels
        for i, lab in enumerate(s.event_time.tolist()):
            if s.is_reference[i]:
                assert lab not in pe
                continue
            rec_ = pe[lab]
            assert isinstance(rec_, _PeriodEffectView)
            assert rec_.period == lab
            assert rec_.effect == float(s.att[i]) and rec_.se == float(s.se[i])
            assert rec_.t_stat == float(s.t_stat[i]) and rec_.p_value == float(s.p_value[i])
            assert rec_.conf_int == (float(s.conf_int_lower[i]), float(s.conf_int_upper[i]))
            assert rec_.is_significant == bool(s.p_value[i] < s.alpha)
        # the reference row is omitted, as the pre-4.0 dict did
        assert len(pe) == int((~s.is_reference).sum())

    def test_is_significant_uses_the_surface_alpha(self, panel):
        s10 = self._fit(panel, alpha=0.10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pe = s10.period_effects
        ps = {lab: r.p_value for lab, r in pe.items()}
        for lab, r in pe.items():
            assert r.is_significant == (ps[lab] < 0.10)
        # a row between 0.05 and 0.10 is significant at the surface's alpha
        # but would not have been at the removed record's hard-coded 0.05 -
        # the pin is on the rule, so just assert the rule held for every row
        assert all(r.is_significant == (r.p_value < 0.10) for r in pe.values())

    def test_significance_stars_thresholds(self):
        rows = {1: (1.0, 0.1), 2: (1.0, 0.2), 4: (1.0, 0.5), 5: (0.001, 1.0), 6: (0.05, 1.0)}
        s = calendar_surface(rows, reference=3, post_periods=[4, 5, 6])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pe = s.period_effects
        from diff_diff.results_base import _significance_stars

        for lab, r in pe.items():
            assert r.significance_stars == _significance_stars(r.p_value)
        assert _significance_stars(np.nan) == ""
        assert _significance_stars(0.0005) == "***"
        assert _significance_stars(0.005) == "**"
        assert _significance_stars(0.03) == "*"
        assert _significance_stars(0.07) == "."
        assert _significance_stars(0.5) == ""

    def test_nan_rows_stay_nan_records(self):
        rows = {1: (0.1, 0.2), 2: (np.nan, np.nan), 4: (0.5, 0.1)}
        s = calendar_surface(rows, reference=3, post_periods=[4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            r = s.period_effects[2]
        assert np.isnan(r.effect) and np.isnan(r.p_value)
        assert r.is_significant is False and r.significance_stars == ""

    def test_datetime_labels_survive_as_keys(self):
        import dataclasses

        rows = {1: (0.1, 0.2), 2: (0.2, 0.2), 4: (0.5, 0.1)}
        base = calendar_surface(rows, reference=3, post_periods=[4])
        # object-dtype Timestamps are what a datetime time column produces
        # (the surface keeps the native dtype; a datetime64[ns] array would
        # be flattened to integer nanoseconds by .tolist(), which the view
        # avoids by iterating the array itself)
        stamps = np.array([pd.Timestamp(f"2020-0{m}-01") for m in (1, 2, 3, 4)], dtype=object)
        s = dataclasses.replace(
            base,
            event_time=stamps,
            vcov=None,
            vcov_index=None,
            reference_period=stamps[2],
            post_periods=(stamps[3],),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pe = s.period_effects
        assert set(pe) == {stamps[0], stamps[1], stamps[3]}
        assert all(isinstance(k, pd.Timestamp) for k in pe)
        assert pe[pd.Timestamp("2020-04-01")].effect == 0.5

    def test_duplicate_labels_fail_closed(self):
        import dataclasses

        rows = {1: (0.1, 0.2), 2: (0.2, 0.2), 4: (0.5, 0.1)}
        base = calendar_surface(rows, reference=3, post_periods=[4])
        dup = dataclasses.replace(
            base, event_time=np.array([1, 1, 3, 4]), vcov=None, vcov_index=None
        )
        with pytest.raises(ValueError, match="duplicate event_time labels"):
            dup.period_effects
        # a label shared by the reference row and an estimated row is rejected
        # too (it would surface the estimate under the reference's key)
        ref_dup = dataclasses.replace(
            base, event_time=np.array([1, 3, 3, 4]), vcov=None, vcov_index=None
        )
        with pytest.raises(ValueError, match="duplicate event_time labels"):
            ref_dup.period_effects

    def test_relative_surfaces_do_not_carry_the_view(self):
        """Gated to calendar surfaces: a relative-time aggregate('event_study')
        container never had period_effects, so hasattr is False and nothing
        warns there."""
        data = generate_staggered_data(n_units=40, n_periods=6, seed=3)
        cs = CallawaySantAnna().fit(
            data, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        es = cs.aggregate("event_study")
        assert isinstance(es, EventStudyResults) and es.time_scale == "relative"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert not hasattr(es, "period_effects")
            with pytest.raises(AttributeError, match="time_scale='calendar'"):
                es.period_effects
        assert not [w for w in rec if issubclass(w.category, FutureWarning)]

    def test_hasattr_fires_the_view_on_calendar_surfaces(self, panel):
        """Documents the hazard: any new duck sniff must guard with isinstance
        first, because hasattr runs the getter (and warns)."""
        s = self._fit(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert hasattr(s, "period_effects")
        assert [w for w in rec if "period_effects" in str(w.message)]

    def test_builder_passthrough_never_fires_the_view(self, panel):
        s = self._fit(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert build_event_study_surface(s) is s
        assert not [w for w in rec if "period_effects" in str(w.message)]

    def test_no_consumer_path_fires_the_view(self, panel):
        """The view-never-fires sweep: every surface-consuming path stays
        duck-sniff free (scoped to the view's own FutureWarning, so legitimate
        UserWarnings or matplotlib/scipy warnings do not fail it)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = self._fit(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            s.to_dataframe()
            s.to_dict()
            s.summary()
            build_event_study_surface(s)
            ax = plot_event_study(s)
            plt.close("all")
            h = compute_honest_did(s, M=1.0)
            compute_pretrends_power(s, M=0.3, pretest_form="wald")
            plot_honest_event_study(h)  # original_results rides on h
            plt.close("all")
        assert ax is not None
        offenders = [
            w
            for w in rec
            if issubclass(w.category, FutureWarning) and "period_effects" in str(w.message)
        ]
        assert offenders == []


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
        """M-007 retired at 4.0: no event-study fit emits the legacy
        reference-period transition warning (it left with the removed class;
        the mode was born on e=-1, v4-design section 4.1)."""
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
        # the default itself is the last pre-period (e=-1), silently
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            s = TwoWayFixedEffects().fit(
                panel,
                "y",
                "treated",
                unit="unit",
                event_study=True,
                time="period",
                post_periods=_post(),
            )
        assert s.reference_period == _post()[0] - 1
        assert not [w for w in rec2 if issubclass(w.category, FutureWarning)]

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
        removed class."""
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
        shift); the within spec inherits the class behavior EXACTLY (through
        3.x pinned bit-exact to the same-data MultiPeriodDiD absorb fit).
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
        """The A1 attribution pins on the TWFE event-study path: the inline
        staggered advisory and the snap warning attribute to the USER file
        (the _frame_offset contract the core kept through the removal)."""
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
        inline = [w for w in rec if "simultaneous adoption" in str(w.message)]
        assert inline and inline[0].filename.endswith("test_v4_merge_mpd.py")
        # snap (class-(i)-via-helper): unit-constant covariate under the
        # absorbed unit FE
        df2 = panel.assign(ucov=(panel["unit"] % 3).astype(float))
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
    def _surfaces(self, panel):
        return TwoWayFixedEffects(cluster="unit").fit(
            panel,
            "y",
            "treated",
            unit="unit",
            event_study=True,
            spec="pooled",
            time="period",
            post_periods=_post(),
        )

    def test_honest_did_survey_df_scalar_threads(self, panel):
        """Survey-backed lane: the df_survey scalar that moves FLCI critical
        values threads through the calendar route - a hand-built twin of the
        surface with the same df_survey reproduces the FLCI exactly, and one
        without it differs."""
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
        assert s.df_survey is not None and np.isfinite(s.df_survey)
        rows = {
            lab: (
                float(s.att[i]),
                float(s.se[i]),
                float(s.t_stat[i]),
                float(s.p_value[i]),
                (float(s.conf_int_lower[i]), float(s.conf_int_upper[i])),
            )
            for i, lab in enumerate(s.event_time.tolist())
            if not s.is_reference[i]
        }
        twin = calendar_surface(
            rows,
            reference=s.reference_period,
            post_periods=list(s.post_periods),
            vcov=s.vcov,
            vcov_index=list(s.vcov_index),
            df_survey=s.df_survey,
        )
        no_df = calendar_surface(
            rows,
            reference=s.reference_period,
            post_periods=list(s.post_periods),
            vcov=s.vcov,
            vcov_index=list(s.vcov_index),
            df_survey=None,
        )
        hs = compute_honest_did(s, M=1.0)
        ht = compute_honest_did(twin, M=1.0)
        hn = compute_honest_did(no_df, M=1.0)
        assert hs.ci_lb == ht.ci_lb and hs.ci_ub == ht.ci_ub
        assert (hs.ci_lb, hs.ci_ub) != (hn.ci_lb, hn.ci_ub)

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
        s = self._surfaces(panel)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            compute_honest_did(s, M=1.0)
        assert not [w for w in rec if "provenance" in str(w.message)]

    def test_pretrends_explicit_pre_periods_validated(self, panel):
        """R8: an explicit pre_periods= selection on the calendar route
        is VALIDATED and chronologically ordered (the relative container
        route's contract) - unknown labels, the reference row,
        duplicates all fail loud, and a reversed-but-valid selection
        gives the identical positional analysis (last_period targets the
        chronological grid, not argument order)."""
        s = self._surfaces(panel)
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
        shifting the positional l_vec target); the fit itself applied the
        same sorted() rule."""
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
        with pytest.warns(UserWarning, match="STRING calendar labels"):
            hs = compute_honest_did(s, M=1.0)
        assert np.isfinite(hs.ci_lb) and np.isfinite(hs.ci_ub)

    def test_calendar_vcov_integrity_rejections(self, panel):
        """R3 hardening: the calendar routes share the relative container
        path's vcov integrity contract instead of silently degrading to
        diag(se^2) - duplicate/incomplete vcov_index and malformed
        sub-blocks fail loud; HonestDiD additionally rejects singular
        blocks (allow_singular=False) while PreTrendsPower keeps its
        documented singular handling."""
        import dataclasses

        s = self._surfaces(panel)
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

        s = self._surfaces(panel)
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

        s = self._surfaces(panel)
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

        s = self._surfaces(panel)
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
