"""EventStudyResults consumer gates (2(b) PR-1, rows M-092/M-093 pre-cut half).

``compute_honest_did``, ``compute_pretrends_power`` and ``plot_event_study`` /
``plot_honest_event_study`` accept the unified event-study container produced
by ``CallawaySantAnnaResults.aggregate('event_study')``. The gates:

- END-TO-END (the TODO row's acceptance criteria): HonestDiD on a
  ``base_period='universal'`` container; PreTrendsPower on an
  ``anticipation=1`` container.
- ROUTE PARITY: for the same fit, the container route reproduces the
  native route. HonestDiD outputs are deterministic and compared at
  equality; PreTrendsPower's extraction tuple is compared bit-exactly,
  while its end-to-end power gets a STOCHASTIC tolerance - scipy's Genz
  multivariate-normal CDF is internally randomized (two native calls on
  identical inputs differ at ~1e-5), so power equality at 1e-14 is not a
  property even of the native route.
- SOURCE-SCOPED ADMISSION: honest/pretrends accept CS- and Stacked-sourced
  containers (the latter widened with row M-024; ``kappa_pre >= 2``
  required); dCDH l1 containers, calendar containers, other-producer/e0
  containers and hand-built source=None containers fail closed. The
  plotters take no source guard (label-faithful rendering).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, compute_honest_did, compute_pretrends_power
from diff_diff.pretrends import PreTrendsPower
from diff_diff.results_base import EventStudyResults

FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")


def _panel(seed=11, n_units=80, n_periods=8):
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_units):
        g = 4 if u < n_units // 3 else (6 if u < 2 * n_units // 3 else 0)
        ui = rng.randn() * 2
        for t in range(1, n_periods + 1):
            post = 1 if (g > 0 and t >= g) else 0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": g,
                    "y": ui + 0.3 * t + 2.0 * post + rng.randn() * 0.5,
                    "cluster_col": u % 20,
                    "survey_weights": 1.0 + 0.1 * (u % 5),
                    "strata": u % 4,
                    "psu": u,
                }
            )
    return pd.DataFrame(rows)


def _fit_cs(data, **cs_kw):
    # The deprecated fit-time aggregate= populates the NATIVE surface the
    # route-parity tests compare against; the container side re-aggregates
    # from the kit either way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CallawaySantAnna(**cs_kw).fit(data, aggregate="event_study", **FIT_KW)


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def cs_universal(panel):
    return _fit_cs(panel, base_period="universal")


@pytest.fixture(scope="module")
def cs_varying(panel):
    return _fit_cs(panel)


def _tiny_container(**overrides):
    """Hand-built 4-row relative container (one reference row at -1)."""
    kwargs = dict(
        event_time=np.array([-2, -1, 0, 1]),
        att=np.array([0.1, 0.0, 1.9, 2.1]),
        se=np.array([0.1, np.nan, 0.12, 0.13]),
        t_stat=np.array([1.0, np.nan, 15.8, 16.2]),
        p_value=np.array([0.3, np.nan, 0.0, 0.0]),
        conf_int_lower=np.array([-0.1, np.nan, 1.66, 1.85]),
        conf_int_upper=np.array([0.3, np.nan, 2.14, 2.35]),
        is_reference=np.array([False, True, False, False]),
        n=np.array([10.0, np.nan, 10.0, 10.0]),
        source="CallawaySantAnnaResults",
    )
    kwargs.update(overrides)
    return EventStudyResults(**kwargs)


# --------------------------------------------------------------------------- #
# End-to-end acceptance (the TODO row's gates)
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_honest_did_on_universal_container(self, cs_universal):
        surface = cs_universal.aggregate("event_study")
        h = compute_honest_did(surface, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)
        assert np.isfinite(h.ci_lb) and np.isfinite(h.ci_ub)

    def test_pretrends_power_on_anticipation_container(self, panel):
        res = _fit_cs(panel, anticipation=1)
        surface = res.aggregate("event_study")
        assert surface.anticipation == 1
        p = compute_pretrends_power(surface, M=0.1)
        assert np.isfinite(p.power)


# --------------------------------------------------------------------------- #
# Route parity: container route == native route on the same fit
# --------------------------------------------------------------------------- #

_HONEST_FIELDS = ("lb", "ub", "ci_lb", "ci_ub", "original_estimate", "original_se", "df_survey")


def _assert_honest_parity(res):
    surface = res.aggregate("event_study")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_native = compute_honest_did(res, M=0.5)
        h_container = compute_honest_did(surface, M=0.5)
    for attr in _HONEST_FIELDS:
        a, b = getattr(h_native, attr), getattr(h_container, attr)
        if a is None or b is None:
            assert a is b, attr
        else:
            np.testing.assert_allclose(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                atol=1e-14,
                rtol=1e-14,
                equal_nan=True,
                err_msg=attr,
            )
    # Documented divergence: the container carries no survey-metadata
    # object, so the stored field is None on the container route (its only
    # inferential consumer is the df extraction, replaced by df_survey).
    assert h_container.survey_metadata is None
    return h_native, h_container


class TestHonestRouteParity:
    def test_universal(self, cs_universal):
        _assert_honest_parity(cs_universal)

    def test_varying(self, cs_varying):
        _assert_honest_parity(cs_varying)

    def test_anticipation(self, panel):
        _assert_honest_parity(_fit_cs(panel, anticipation=1))

    def test_anticipation_window_is_post_not_pre(self, panel):
        # REGISTRY anticipation contract: with anticipation=k the window
        # [e=-k, -1] carries anticipated TREATMENT effects, so the clean
        # pre-trend set is e < -k and beta_post starts at -k. Splitting
        # at 0 misclassified e=-1 as a pre-trend coefficient IDENTICALLY
        # on both routes - the parity gate alone could not catch it, so
        # this pins the semantics directly.
        res = _fit_cs(panel, anticipation=1)
        surface = res.aggregate("event_study")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_native = compute_honest_did(res, M=0.5)
            h_container = compute_honest_did(surface, M=0.5)
        for h in (h_native, h_container):
            assert -1 not in h.pre_periods_used
            assert h.post_periods_used[0] == -1

    def test_bare_cluster_df_threads(self, panel):
        res = _fit_cs(panel, cluster="cluster_col")
        h_native, h_container = _assert_honest_parity(res)
        # bare-cluster fits carry df_inference -> finite scalar df on BOTH routes
        assert h_container.df_survey is not None
        assert np.isfinite(float(h_container.df_survey))

    def test_survey_df_threads(self, panel):
        from diff_diff.survey import SurveyDesign

        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna().fit(
                _panel(), survey_design=sd, aggregate="event_study", **FIT_KW
            )
        h_native, h_container = _assert_honest_parity(res)
        assert h_container.df_survey is not None
        assert np.isfinite(float(h_container.df_survey))

    def test_zero_se_rows_dropped_on_both_routes(self, panel):
        # A zero-SE row carries undefined inference (safe_inference NaNs
        # its t/p/CI); admitting it would launder that into finite honest
        # bounds. Both routes drop it identically. Container side:
        surface = _tiny_container(
            base_period="universal",
            se=np.array([0.0, np.nan, 0.12, 0.13]),
            t_stat=np.array([np.nan, np.nan, 15.8, 16.2]),
            p_value=np.array([np.nan, np.nan, 0.0, 0.0]),
            conf_int_lower=np.array([np.nan, np.nan, 1.66, 1.85]),
            conf_int_upper=np.array([np.nan, np.nan, 2.14, 2.35]),
        )
        # The only pre-period row has se == 0 -> dropped -> no pre periods.
        with pytest.raises(ValueError, match="pre-period"):
            compute_honest_did(surface, M=0.5)
        # Native side: inject a zero-SE pre row into a real fit's surface
        # and assert the same drop (the row disappears from the retained
        # pre set rather than entering beta with sigma 0).
        res = _fit_cs(panel, base_period="universal")
        from diff_diff.honest_did import _extract_event_study_params

        pre_key = min(k for k in res.event_study_effects if k < -1)
        res.event_study_effects[pre_key] = dict(
            res.event_study_effects[pre_key], se=0.0, p_value=np.nan
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = _extract_event_study_params(res)
            except ValueError:
                out = None  # grid gap after the drop - also a valid fail-closed
        if out is not None:
            assert pre_key not in out[4]

    def test_misaligned_container_vcov_raises(self):
        # A SUPPLIED covariance whose index omits a retained horizon is
        # inconsistent: fail loud, never silently degrade to diagonal
        # (diag fallback is reserved for vcov=None).
        surface = _tiny_container(
            base_period="universal",
            vcov=np.eye(2),
            vcov_index=np.array([-2, 0]),  # omits retained horizon 1
        )
        with pytest.raises(ValueError, match="vcov_index is missing"):
            compute_honest_did(surface, M=0.5)

    def test_replicate_undefined_sentinel_relays(self):
        # The container's df_survey=0.0 sentinel (replicate design with an
        # undefined df) passes through to HonestDiDResults.df_survey exactly
        # as the fit-time branch's sentinel does.
        surface = _tiny_container(base_period="universal", df_survey=0.0)
        h = compute_honest_did(surface, M=0.5)
        assert h.df_survey == 0.0

    @pytest.mark.parametrize("method", ["smoothness", "relative_magnitude"])
    def test_replicate_undefined_df_fails_closed_to_nan_ci(self, method):
        # df_survey=0.0 means UNDEFINED inference: every FLCI path must
        # yield NaN CI endpoints (never a silent normal-theory fallback) -
        # incl. the optimal smoothness-FLCI with a full covariance, whose
        # _cv_alpha/_flci_solve guards fail closed on a provided df <= 0.
        surface = _tiny_container(
            base_period="universal",
            df_survey=0.0,
            vcov=np.diag([0.01, 0.0144, 0.0169]),
            vcov_index=np.array([-2, 0, 1]),
        )
        h = compute_honest_did(surface, method=method, M=0.5)
        assert np.isnan(h.ci_lb) and np.isnan(h.ci_ub)


class TestPretrendsRouteParity:
    def test_extraction_bit_exact(self, panel):
        res = _fit_cs(panel, anticipation=1)
        surface = res.aggregate("event_study")
        pt = PreTrendsPower()
        e1, s1, v1, n1, r1, src1 = pt._extract_pre_period_params(res)
        e2, s2, v2, n2, r2, src2 = pt._extract_pre_period_params(surface)
        assert src1 == src2 == "full_pre_period_vcov"
        assert n1 == n2
        np.testing.assert_array_equal(r1, r2)
        # bit-exact: the container relays stored values verbatim
        assert np.array_equal(e1, e2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(v1, v2)

    def test_power_within_stochastic_tolerance(self, cs_varying):
        surface = cs_varying.aggregate("event_study")
        p_native = compute_pretrends_power(cs_varying, M=0.1)
        p_container = compute_pretrends_power(surface, M=0.1)
        # scipy's MVN CDF is internally randomized: identical inputs differ
        # at ~1e-5 across calls, so this is a smoke bound, not 1e-14.
        assert abs(p_native.power - p_container.power) < 1e-3

    def test_explicit_pre_periods_honored_on_both_routes(self, panel):
        # An explicitly requested pre-period subset must subset effects/
        # SEs/VCV on BOTH routes - never be silently ignored.
        res = _fit_cs(panel)
        surface = res.aggregate("event_study")
        pt = PreTrendsPower()
        full = pt._extract_pre_period_params(surface)
        subset_labels = [int(t) for t in full[4][:2]]
        e1, s1, v1, n1, r1, src1 = pt._extract_pre_period_params(res, subset_labels)
        e2, s2, v2, n2, r2, src2 = pt._extract_pre_period_params(surface, subset_labels)
        assert n1 == n2 == len(subset_labels)
        np.testing.assert_array_equal(r1, r2)
        assert np.array_equal(e1, e2) and np.array_equal(s1, s2) and np.array_equal(v1, v2)
        assert v2.shape == (len(subset_labels), len(subset_labels))

    def test_invalid_explicit_pre_periods_raise(self, panel):
        res = _fit_cs(panel)
        surface = res.aggregate("event_study")
        pt = PreTrendsPower()
        with pytest.raises(ValueError, match="not eligible"):
            pt._extract_pre_period_params(surface, [999])
        with pytest.raises(ValueError, match="not eligible"):
            pt._extract_pre_period_params(res, [999])

    def test_empty_explicit_pre_periods_raise(self, panel):
        # pre_periods=[] passes the per-label eligibility check vacuously;
        # without a post-subset guard it reached zero-dimensional matrix
        # logic downstream (opaque reshape error). All three explicit-
        # subset paths reject it with the user-facing message.
        from diff_diff import SunAbraham

        res = _fit_cs(panel)
        surface = res.aggregate("event_study")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sa = SunAbraham().fit(panel, **FIT_KW)
        pt = PreTrendsPower()
        for target in (res, surface, sa):
            with pytest.raises(ValueError, match="at least one pre-period"):
                pt._extract_pre_period_params(target, [])


# --------------------------------------------------------------------------- #
# Universal-base warning (fail-safe on missing provenance)
# --------------------------------------------------------------------------- #


class TestUniversalBaseWarning:
    def test_varying_container_warns(self, cs_varying):
        surface = cs_varying.aggregate("event_study")
        assert surface.base_period == "varying"
        with pytest.warns(UserWarning, match="base_period='universal'"):
            compute_honest_did(surface, M=0.5)

    def test_missing_provenance_warns(self):
        surface = _tiny_container(base_period=None)
        with pytest.warns(UserWarning, match="no base_period provenance"):
            compute_honest_did(surface, M=0.5)

    def test_universal_container_silent(self, cs_universal):
        surface = cs_universal.aggregate("event_study")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_honest_did(surface, M=0.5)
        assert not [w for w in caught if "base_period" in str(w.message)]


class TestCommonReferenceGuard:
    """Cohort-level normalization-base provenance (reference_event_times).

    CS base_period='universal' on a GAPPED grid selects cohort-specific
    positional bases. In the OVERLAP layout ({1,2,3,5}, cohorts {2,3,5})
    cohort 5's base (period 3, e=-2) coincides with cohort 3's estimated
    pre-trend horizon, so the aggregated e=-2 row is a real estimate and
    NO reference-only row marks that anchor - is_reference-based guards
    cannot see it. The reference_event_times provenance field is the
    authoritative signal: more than one distinct entry means the
    coefficients were normalized against different bases, and HonestDiD /
    PreTrendsPower fail closed on BOTH routes.
    """

    @staticmethod
    def _fit_gapped(periods, cohorts, **fit_kw):
        rng = np.random.RandomState(7)
        rows = []
        coh = list(cohorts) + [0]
        for u in range(160):
            g = coh[u % len(coh)]
            ufe = rng.randn() * 2
            for t in periods:
                post = 1 if (g > 0 and t >= g) else 0
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": g,
                        "y": ufe + 0.3 * t + 2.0 * post + rng.randn() * 0.5,
                    }
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CallawaySantAnna(n_bootstrap=0, base_period="universal").fit(
                pd.DataFrame(rows), aggregate="event_study", **fit_kw, **FIT_KW
            )

    def test_overlap_layout_provenance(self):
        res = self._fit_gapped((1, 2, 3, 5), (2, 3, 5))
        # Cohorts 2,3 base at e=-1; cohort 5's base (period 3) at e=-2.
        assert tuple(int(e) for e in res.reference_event_times) == (-2, -1)
        surface = res.aggregate("event_study")
        assert tuple(int(e) for e in surface.reference_event_times) == (-2, -1)
        # The overlapped anchor is INVISIBLE to is_reference: only e=-1 is
        # a reference-only row; e=-2 aggregates cohort 3's real estimate.
        marked = sorted(int(t) for t in surface.event_time[surface.is_reference])
        assert marked == [-1]
        assert surface.to_dict()["reference_event_times"] == [-2, -1]

    def test_overlap_fails_closed_on_all_four_routes(self):
        res = self._fit_gapped((1, 2, 3, 5), (2, 3, 5))
        surface = res.aggregate("event_study")
        for consumer, target in (
            (compute_honest_did, res),
            (compute_honest_did, surface),
            (compute_pretrends_power, res),
            (compute_pretrends_power, surface),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="common reference"):
                    consumer(target, M=0.5)

    def test_non_overlap_gapped_fails_closed(self):
        # The {1,3,6} layout materializes every anchor as its own
        # reference-only row; the provenance guard still fires first with
        # the actionable common-reference message on both routes.
        res = self._fit_gapped((1, 3, 6), (3, 6))
        assert tuple(int(e) for e in res.reference_event_times) == (-3, -2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="common reference"):
                compute_honest_did(res, M=0.5)
            with pytest.raises(ValueError, match="common reference"):
                compute_honest_did(res.aggregate("event_study"), M=0.5)

    def test_regular_universal_singleton_passes(self, cs_universal):
        assert tuple(int(e) for e in cs_universal.reference_event_times) == (-1,)
        surface = cs_universal.aggregate("event_study")
        assert tuple(int(e) for e in surface.reference_event_times) == (-1,)
        h = compute_honest_did(surface, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)

    def test_varying_fit_carries_none(self, cs_varying):
        # Varying base has no constant per-cohort reference: the field is
        # None (unknown/NA), never invented - the varying-base WARNINGS
        # cover that regime instead.
        assert cs_varying.reference_event_times is None
        assert cs_varying.aggregate("event_study").reference_event_times is None

    def test_balance_e_recomputes_provenance_over_retained_cohorts(self):
        # The FIT-level tuple is fit-wide; the CONTAINER's must reflect
        # the cohorts the aggregation actually retained. balance_e can
        # drop the cohort responsible for the second base - a stale tuple
        # would reject a balanced surface whose remaining cohorts share
        # one reference.
        res = self._fit_gapped((1, 2, 3, 4, 6), (3, 6))
        # Cohort 3 base at e=-1 (period 2); cohort 6 base at e=-2 (period
        # 4, the positional neighbor on the gapped grid).
        assert tuple(int(e) for e in res.reference_event_times) == (-2, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unbalanced = res.aggregate("event_study")
            balanced = res.aggregate("event_study", balance_e=1)

        # Unbalanced surface: both cohorts retained -> both bases -> the
        # common-reference guard fires.
        assert tuple(int(e) for e in unbalanced.reference_event_times) == (-2, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="common reference"):
                compute_pretrends_power(unbalanced, M=0.1)

        # balance_e=1 retains only cohort 3 (the only cohort with an
        # effect at e=1): the surface-faithful provenance is the single
        # remaining base, and the consumer accepts the surface.
        assert tuple(int(e) for e in balanced.reference_event_times) == (-1,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = compute_pretrends_power(balanced, M=0.1)
        assert np.isfinite(p.power)

    def test_missing_native_provenance_derives_from_reference_cells(self):
        # A provenance-less universal result (pre-3.9 pickle or
        # replace()-stripped copy) must not FAIL OPEN: the cohort bases
        # are re-derived from the materialized reference cells, so the
        # mixed-base layout still fails closed on both consumers.
        import dataclasses

        res = self._fit_gapped((1, 2, 3, 5), (2, 3, 5))
        stripped = dataclasses.replace(res, reference_event_times=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="common reference"):
                compute_honest_did(stripped, M=0.5)
            with pytest.raises(ValueError, match="common reference"):
                compute_pretrends_power(stripped, M=0.1)

    def test_missing_container_provenance_warns(self):
        # A hand-built universal container without the field cannot be
        # verified (no cells to derive from): warn fail-safe, never fail
        # open silently. CS-produced containers always record the field.
        surface = _tiny_container(base_period="universal")
        assert surface.reference_event_times is None
        with pytest.warns(UserWarning, match="no reference_event_times provenance"):
            compute_honest_did(surface, M=0.5)
        with pytest.warns(UserWarning, match="no reference_event_times provenance"):
            compute_pretrends_power(surface, M=0.1)

    def test_fit_time_balance_e_provenance_matches_surface(self):
        # The deprecated fit-time aggregate="event_study" + balance_e
        # stores a RESTRICTED surface: its provenance must describe the
        # retained cohorts too, or the native route would reject a fit
        # whose equivalent post-fit container is accepted (route parity).
        res = self._fit_gapped((1, 2, 3, 4, 6), (3, 6), balance_e=1)
        assert tuple(int(e) for e in res.reference_event_times) == (-1,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = compute_pretrends_power(res, M=0.1)
        assert np.isfinite(p.power)

    def test_to_dict_reference_event_times_json_safe(self):
        # CS period arithmetic yields numpy scalars; to_dict must emit
        # JSON-serializable labels.
        import json

        res = self._fit_gapped((1, 3, 6), (3, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surface = res.aggregate("event_study")
        d = surface.to_dict()
        assert d["reference_event_times"] == [-3, -2]
        json.dumps(d)  # must not raise on numpy-labeled provenance


class TestContainerIntegrity:
    """Hand-built containers with malformed rows/covariance fail closed.

    Containers are publicly constructible: consumers subset by explicit
    [sorted pre; sorted post] label order (row order is not trusted) and
    validate covariance integrity at the boundary.
    """

    @staticmethod
    def _container(order, vcov=None, vcov_index=None, se_override=None):
        data = {
            -3: (0.10, 0.10, -0.10, 0.30),
            -2: (0.05, 0.10, -0.15, 0.25),
            -1: (0.0, np.nan, np.nan, np.nan),
            0: (1.9, 0.12, 1.66, 2.14),
            1: (2.1, 0.13, 1.85, 2.35),
        }
        rows = [data[t] for t in order]
        se = np.array([r[1] for r in rows])
        if se_override is not None:
            se = se_override
        return EventStudyResults(
            event_time=np.array(order),
            att=np.array([r[0] for r in rows]),
            se=se,
            t_stat=np.array([np.nan] * len(order)),
            p_value=np.array([np.nan] * len(order)),
            conf_int_lower=np.array([r[2] for r in rows]),
            conf_int_upper=np.array([r[3] for r in rows]),
            is_reference=np.array([t == -1 for t in order]),
            n=np.array([10.0] * len(order)),
            source="CallawaySantAnnaResults",
            base_period="universal",
            reference_event_times=(-1,),
            vcov=vcov,
            vcov_index=vcov_index,
        )

    def test_permuted_rows_produce_identical_bounds(self):
        # Interleaved rows must yield the SAME bounds as the sorted
        # container: beta_hat/sigma are subset in [sorted pre; sorted
        # post] order, never row order (the fit-side split takes the
        # first num_pre entries as beta_pre).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_sorted = compute_honest_did(self._container([-3, -2, -1, 0, 1]), M=0.5)
            h_perm = compute_honest_did(self._container([-3, 0, -2, -1, 1]), M=0.5)
        assert h_perm.lb == h_sorted.lb and h_perm.ub == h_sorted.ub
        assert h_perm.pre_periods_used == h_sorted.pre_periods_used == [-3, -2]
        assert h_perm.post_periods_used == h_sorted.post_periods_used == [0, 1]
        # Pretrends is label-aligned elementwise: power invariant too.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_sorted = compute_pretrends_power(self._container([-3, -2, -1, 0, 1]), M=0.1)
            p_perm = compute_pretrends_power(self._container([-3, 0, -2, -1, 1]), M=0.1)
        assert abs(p_sorted.power - p_perm.power) < 1e-3  # MVN-CDF jitter

    def test_reversed_rows_last_period_violation_invariant(self):
        # Positional violation patterns (last_period assigns weights[-1]
        # to the FINAL entry) require chronological pre-period order, not
        # row order: a reversed hand-built container must produce the
        # same power as the sorted one.
        pt_kwargs = dict(M=0.5, violation_type="last_period")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_sorted = compute_pretrends_power(self._container([-3, -2, -1, 0, 1]), **pt_kwargs)
            p_rev = compute_pretrends_power(self._container([-2, -3, -1, 0, 1]), **pt_kwargs)
        assert abs(p_sorted.power - p_rev.power) < 1e-3  # MVN-CDF jitter
        # Extraction-level exactness: chronological labels either way.
        pt = PreTrendsPower()
        rel_sorted = pt._extract_pre_period_params(self._container([-3, -2, -1, 0, 1]))[4]
        rel_rev = pt._extract_pre_period_params(self._container([-2, -3, -1, 0, 1]))[4]
        np.testing.assert_array_equal(rel_sorted, rel_rev)

    def test_duplicate_event_time_labels_rejected(self):
        surface = self._container([-3, -2, -1, 0, 0])
        for consumer in (compute_honest_did, compute_pretrends_power):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="duplicate event_time"):
                    consumer(surface, M=0.5)

    @pytest.mark.parametrize(
        "corruption, match",
        [
            ("nonfinite", "non-finite"),
            ("asymmetric", "not symmetric"),
            ("indefinite", "indefinite"),
            ("diag_mismatch", "inconsistent with the stored standard errors"),
            ("dup_index", "duplicate\\s+labels|carries duplicate"),
        ],
    )
    def test_malformed_covariance_rejected(self, corruption, match):
        order = [-3, -2, -1, 0, 1]
        ses = np.array([0.10, 0.10, 0.12, 0.13])  # retained rows, sorted
        vcov = np.diag(ses**2)
        vcov_index = np.array([-3, -2, 0, 1])
        if corruption == "nonfinite":
            vcov = vcov.copy()
            vcov[0, 1] = np.nan
            vcov[1, 0] = np.nan
        elif corruption == "asymmetric":
            vcov = vcov.copy()
            vcov[0, 1] = 0.005  # not mirrored
        elif corruption == "indefinite":
            vcov = vcov.copy()
            # off-diagonal larger than the diagonal product -> negative eig
            vcov[0, 1] = vcov[1, 0] = 0.02
        elif corruption == "diag_mismatch":
            vcov = vcov.copy()
            vcov[0, 0] = 0.5  # != se**2
        elif corruption == "dup_index":
            vcov_index = np.array([-3, -3, 0, 1])
        surface = self._container(order, vcov=vcov, vcov_index=vcov_index)
        for consumer in (compute_honest_did, compute_pretrends_power):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match=match):
                    consumer(surface, M=0.5)

    def test_low_scale_indefinite_rejected(self):
        # Tolerances are RELATIVE to the matrix scale: a uniformly tiny
        # indefinite matrix (diag 1e-10, eigenvalues [-1e-10, 3e-10])
        # must not slip under an absolute floor.
        vcov = np.diag(np.full(4, 1e-10))
        vcov[0, 1] = vcov[1, 0] = 2e-10
        se_override = np.array([1e-5, 1e-5, np.nan, 1e-5, 1e-5])
        surface = self._container(
            [-3, -2, -1, 0, 1],
            vcov=vcov,
            vcov_index=np.array([-3, -2, 0, 1]),
            se_override=se_override,
        )
        for consumer in (compute_honest_did, compute_pretrends_power):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="indefinite"):
                    consumer(surface, M=0.5)

    def test_singular_covariance_honest_rejects_pretrends_accepts(self):
        # Perfectly-correlated pre-rows: PSD but SINGULAR. HonestDiD
        # rejects (Rambachan-Roth assumes covariance eigenvalues bounded
        # away from zero); PreTrendsPower keeps its documented
        # singular-covariance handling.
        ses = np.array([0.1, 0.1, 0.12, 0.13])
        vcov = np.diag(ses**2)
        vcov[0, 1] = vcov[1, 0] = 0.01  # corr = 1 between the pre rows
        surface = self._container(
            [-3, -2, -1, 0, 1], vcov=vcov, vcov_index=np.array([-3, -2, 0, 1])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="singular"):
                compute_honest_did(surface, M=0.5)
            p = compute_pretrends_power(surface, M=0.1)
        assert p is not None

    def test_valid_covariance_still_accepted(self):
        ses = np.array([0.10, 0.10, 0.12, 0.13])
        vcov = np.diag(ses**2)
        surface = self._container(
            [-3, -2, -1, 0, 1], vcov=vcov, vcov_index=np.array([-3, -2, 0, 1])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = compute_honest_did(surface, M=0.5)
            p = compute_pretrends_power(surface, M=0.1)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)
        assert np.isfinite(p.power)


class TestLinearViolationAnchoring:
    """Roth's linear violation is anchored at the omitted reference.

    Roth labels the omitted period t=0, so delta = gamma*t vanishes there
    by construction; translated to estimator-native labels the threaded
    relative times are t - t_ref, NOT raw treatment-relative labels
    (which overstate the violation by the reference offset). MPD already
    anchored via _coerce_relative_times_from_reference; these pin the
    CS-universal, SunAbraham and container routes.
    """

    def test_universal_weights_are_reference_relative(self, panel):
        res = _fit_cs(panel, base_period="universal")
        surface = res.aggregate("event_study")
        keep = (
            (~surface.is_reference)
            & np.isfinite(surface.se)
            & (surface.se > 0)
            & (surface.event_time < 0)
        )
        labels = surface.event_time[keep].astype(float)
        pt = PreTrendsPower()
        rel_native = pt._extract_pre_period_params(res)[4]
        rel_container = pt._extract_pre_period_params(surface)[4]
        expected = labels - (-1.0)  # anchored at the e=-1 reference
        np.testing.assert_array_equal(rel_native, expected)
        np.testing.assert_array_equal(rel_container, expected)

    def test_universal_anticipation_anchor_and_weights(self, panel):
        # anticipation=1: reference at e=-2; pre labels t < -1 anchor
        # there, and the hand-calculated weight vector is |t - t_ref|.
        res = _fit_cs(panel, base_period="universal", anticipation=1)
        surface = res.aggregate("event_study")
        assert surface.reference_period == -2
        keep = (
            (~surface.is_reference)
            & np.isfinite(surface.se)
            & (surface.se > 0)
            & (surface.event_time < -1)
        )
        expected = surface.event_time[keep].astype(float) + 2.0
        pt = PreTrendsPower()
        rel = pt._extract_pre_period_params(surface)[4]
        np.testing.assert_array_equal(rel, expected)
        w = pt._get_violation_weights(len(rel), relative_times=rel)
        np.testing.assert_array_equal(w, np.abs(expected))

    def test_anchor_invariant_to_label_origin(self):
        # Containers identical up to a label SHIFT (reference at -1 vs 0)
        # extract IDENTICAL reference-relative times - the violation is
        # anchored at the reference, not at the label origin.
        common = dict(
            att=np.array([0.1, 0.05, 0.0, 1.9]),
            se=np.array([0.1, 0.1, np.nan, 0.12]),
            t_stat=np.array([1.0, 0.5, np.nan, 15.8]),
            p_value=np.array([0.3, 0.6, np.nan, 0.0]),
            conf_int_lower=np.array([-0.1, -0.15, np.nan, 1.66]),
            conf_int_upper=np.array([0.3, 0.25, np.nan, 2.14]),
            is_reference=np.array([False, False, True, False]),
            n=np.array([10.0, 10.0, np.nan, 10.0]),
            source="CallawaySantAnnaResults",
            base_period="universal",
        )
        a = EventStudyResults(
            event_time=np.array([-3, -2, -1, 0]), reference_event_times=(-1,), **common
        )
        b = EventStudyResults(
            event_time=np.array([-2, -1, 0, 1]), reference_event_times=(0,), **common
        )
        pt = PreTrendsPower()
        rel_a = pt._extract_pre_period_params(a)[4]
        rel_b = pt._extract_pre_period_params(b)[4]
        np.testing.assert_array_equal(rel_a, rel_b)
        np.testing.assert_array_equal(rel_a, np.array([-2.0, -1.0]))

    def test_sun_abraham_anchor(self, panel):
        from diff_diff import SunAbraham

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sa = SunAbraham().fit(panel, **FIT_KW)
        assert sa.reference_period == -1
        pt = PreTrendsPower()
        effects, _, _, n_pre, rel, _ = pt._extract_pre_period_params(sa)
        # Anchored at the omitted e = -1 - anticipation = -1.
        labels = sorted(
            t
            for t, d in sa.event_study_effects.items()
            if t < 0 and np.isfinite(d.get("se", np.nan)) and float(d.get("se", 0.0)) > 0
        )
        np.testing.assert_array_equal(rel, np.asarray(labels, dtype=float) + 1.0)


class TestVaryingBasePretrendsWarning:
    """Twin of HonestDiD's universal-base warning, on both pretrends routes.

    The built-in ``linear`` violation constructs delta as a slope on
    relative time (level coefficients against one common reference); CS
    varying-base pre-treatment effects are consecutive-period comparisons,
    so linear power/MDV target a different violation shape (REGISTRY
    PreTrendsPower Note; full fix tracked in TODO.md).
    """

    def test_varying_native_warns(self, cs_varying):
        with pytest.warns(UserWarning, match="base_period='universal'"):
            compute_pretrends_power(cs_varying, M=0.1)

    def test_varying_container_warns(self, cs_varying):
        surface = cs_varying.aggregate("event_study")
        with pytest.warns(UserWarning, match="base_period='universal'"):
            compute_pretrends_power(surface, M=0.1)

    def test_missing_provenance_warns(self):
        surface = _tiny_container(base_period=None)
        with pytest.warns(UserWarning, match="no base_period provenance"):
            compute_pretrends_power(surface, M=0.1)

    def test_non_linear_violation_does_not_warn(self, cs_varying):
        # The warning concerns the built-in LINEAR construction only;
        # constant/last_period/custom vectors are user-specified in
        # coefficient space.
        surface = cs_varying.aggregate("event_study")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_pretrends_power(cs_varying, M=0.1, violation_type="constant")
            compute_pretrends_power(surface, M=0.1, violation_type="constant")
        assert not [w for w in caught if "base_period" in str(w.message)]

    def test_universal_silent_on_both_routes(self, panel):
        res = _fit_cs(panel, base_period="universal")
        surface = res.aggregate("event_study")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_pretrends_power(res, M=0.1)
            compute_pretrends_power(surface, M=0.1)
        assert not [w for w in caught if "base_period" in str(w.message)]


# --------------------------------------------------------------------------- #
# Provenance threading (incl. the requested-but-empty path)
# --------------------------------------------------------------------------- #


class TestProvenanceThreading:
    def test_container_carries_fit_provenance(self, panel):
        res = _fit_cs(panel, base_period="universal", anticipation=1)
        surface = res.aggregate("event_study")
        assert surface.base_period == "universal"
        assert surface.anticipation == 1
        assert surface.df_survey is None  # no survey design, no cluster df

    def test_bare_cluster_container_df(self, panel):
        res = _fit_cs(panel, cluster="cluster_col")
        surface = res.aggregate("event_study")
        assert surface.df_survey is not None and np.isfinite(surface.df_survey)


# --------------------------------------------------------------------------- #
# Source-scoped admission (fail-closed)
# --------------------------------------------------------------------------- #


def _dcdh_container():
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

    rng = np.random.RandomState(5)
    rows = []
    for u in range(30):
        s_t = 4 if u < 15 else 10**6
        for t in range(1, 7):
            d = 1 if t >= s_t else 0
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "outcome": u / 10 + 0.2 * t + 1.5 * d + rng.randn() * 0.3,
                    "treat": d,
                }
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ChaisemartinDHaultfoeuille().fit(
            pd.DataFrame(rows), outcome="outcome", unit="unit", time="period", treatment="treat"
        )
    return res.aggregate("event_study")


class TestSourceScopedAdmission:
    def test_dcdh_container_rejected_by_honest(self):
        surface = _dcdh_container()
        assert surface.event_time_convention == "l1_first_switch"
        with pytest.raises(TypeError, match="CallawaySantAnnaResults.aggregate"):
            compute_honest_did(surface, M=0.5)

    def test_dcdh_container_rejected_by_pretrends_without_dead_route(self):
        surface = _dcdh_container()
        with pytest.raises(TypeError) as exc_info:
            compute_pretrends_power(surface, M=0.1)
        msg = str(exc_info.value)
        # pretrends' native accepted set has NO dCDH branch - the message
        # must name ITS OWN natives, never point dCDH at a dead route.
        assert "SunAbrahamResults" in msg
        assert "natively" not in msg or "ChaisemartinDHaultfoeuille" not in msg

    def test_hand_built_source_none_rejected(self):
        surface = _tiny_container(source=None)
        with pytest.raises(TypeError, match="source=None"):
            compute_honest_did(surface, M=0.5)
        with pytest.raises(TypeError, match="source=None"):
            compute_pretrends_power(surface, M=0.1)

    def test_non_cs_e0_source_rejected(self):
        surface = _tiny_container(source="ImputationDiDResults")
        with pytest.raises(TypeError, match="ImputationDiDResults"):
            compute_honest_did(surface, M=0.5)
        with pytest.raises(TypeError, match="ImputationDiDResults"):
            compute_pretrends_power(surface, M=0.1)

    def test_calendar_scale_rejected(self):
        # Belt-and-suspenders: even a CS-sourced container is rejected on a
        # calendar time scale (CS never emits calendar). Since the M-010
        # merge, calendar surfaces route to the TWFE calendar branch, whose
        # source gate rejects everything but the TWFE event-study producer -
        # the rejection survives with the calendar-route message.
        surface = _tiny_container(
            event_time=np.array(["2018", "2019", "2020", "2021"], dtype=object),
            time_scale="calendar",
        )
        with pytest.raises(TypeError, match="TwoWayFixedEffects event-study mode"):
            compute_honest_did(surface, M=0.5)
        with pytest.raises(TypeError, match="TwoWayFixedEffects event-study mode"):
            compute_pretrends_power(surface, M=0.1)

    def test_multiple_reference_rows_fail_closed_in_honest(self):
        # DELIBERATE deviation from the fit-time branch, which silently
        # splits around the FIRST n_groups==0 marker in dict order: the
        # container branch refuses - the consecutive-grid contract is
        # defined around a single omitted reference.
        surface = _tiny_container(
            is_reference=np.array([True, True, False, False]),
            att=np.array([0.0, 0.0, 1.9, 2.1]),
            se=np.array([np.nan, np.nan, 0.12, 0.13]),
            t_stat=np.array([np.nan, np.nan, 15.8, 16.2]),
            p_value=np.array([np.nan, np.nan, 0.0, 0.0]),
            conf_int_lower=np.array([np.nan, np.nan, 1.66, 1.85]),
            conf_int_upper=np.array([np.nan, np.nan, 2.14, 2.35]),
            n=np.array([np.nan, np.nan, 10.0, 10.0]),
            base_period="universal",
        )
        with pytest.raises(ValueError, match="multiple reference rows") as exc_info:
            compute_honest_did(surface, M=0.5)
        # Message-level pin: the native-results route fails its own
        # consecutive-grid validation on the same gapped layout, so the
        # error must NOT recommend it - it recommends re-estimation on a
        # consecutive grid instead.
        msg = str(exc_info.value)
        assert "native" not in msg
        assert "consecutive" in msg and "re-estimate" in msg


# --------------------------------------------------------------------------- #
# Plotting (no source guard: label-faithful for any producer)
# --------------------------------------------------------------------------- #


class TestPlotting:
    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_cs_container_plots(self, cs_universal):
        from diff_diff.visualization import plot_event_study

        surface = cs_universal.aggregate("event_study")
        ax = plot_event_study(surface, show=False)
        assert ax is not None

    def test_dcdh_l1_container_plots(self):
        from diff_diff.visualization import plot_event_study

        surface = _dcdh_container()
        ax = plot_event_study(surface, show=False)
        assert ax is not None

    @staticmethod
    def _multi_ref_container():
        return _tiny_container(
            is_reference=np.array([True, True, False, False]),
            att=np.array([0.0, 0.0, 1.9, 2.1]),
            se=np.array([np.nan, np.nan, 0.12, 0.13]),
            t_stat=np.array([np.nan, np.nan, 15.8, 16.2]),
            p_value=np.array([np.nan, np.nan, 0.0, 0.0]),
            conf_int_lower=np.array([np.nan, np.nan, 1.66, 1.85]),
            conf_int_upper=np.array([np.nan, np.nan, 2.14, 2.35]),
            n=np.array([np.nan, np.nan, 10.0, 10.0]),
        )

    @staticmethod
    def _hollow_marker_xs(ax):
        """Positional x of single-point markers drawn hollow (white face)."""
        out = set()
        for line in ax.lines:
            xs = line.get_xdata()
            if len(xs) == 1 and line.get_markerfacecolor() == "white":
                out.add(float(xs[0]))
        return out

    def test_multi_reference_rows_render_hollow(self):
        # Several is_reference rows (the CS gapped-grid universal case) are
        # carried ROW-ALIGNED via reference_marks: every normalization
        # anchor renders hollow at 0 - never silently dropped, never
        # presented as a filled estimate. The scalar reference stays None.
        from diff_diff.visualization import plot_event_study
        from diff_diff.visualization._event_study import _extract_plot_data

        surface = self._multi_ref_container()
        out = _extract_plot_data(surface, None, None, None, None)
        effects, _, periods, _, _, ref, ref_inferred, *_rest, marks = out
        assert ref is None and ref_inferred is False
        assert periods == [-2, -1, 0, 1]
        assert effects[-2] == 0.0 and effects[-1] == 0.0
        assert marks == {-2, -1}

        ax = plot_event_study(surface, show=False)
        # Periods -2, -1 sit at positional x 0, 1.
        assert self._hollow_marker_xs(ax) == {0.0, 1.0}

    def test_multi_reference_explicit_renormalization(self):
        # Re-basing a multi-reference surface around a period that is NOT
        # one of its anchors is undefined (each anchor constrains its own
        # cohort base) -> fail closed. Choosing one of the anchors is a
        # no-op shift and keeps every anchor hollow.
        from diff_diff.visualization import plot_event_study

        surface = self._multi_ref_container()
        with pytest.raises(ValueError, match="multiple reference rows"):
            plot_event_study(surface, reference_period=1, show=False)
        ax = plot_event_study(surface, reference_period=-1, show=False)
        assert self._hollow_marker_xs(ax) == {0.0, 1.0}

    def test_anticipation_window_shaded_as_post(self, panel):
        # REGISTRY anticipation contract, mirrored from the HonestDiD/
        # pretrends boundary: with anticipation=k the window [e=-k, -1]
        # carries anticipated TREATMENT effects, so the pre-treatment
        # shading covers e < -k only - on the container route AND the
        # native fit-time dict route.
        from diff_diff.visualization._event_study import _extract_plot_data

        res = _fit_cs(panel, anticipation=1)
        surface = res.aggregate("event_study")
        for target in (surface, res):
            out = _extract_plot_data(target, None, None, None, None)
            pre, post = out[3], out[4]
            assert -1 in post and -1 not in pre
            assert all(p < -1 for p in pre)

    def test_plotly_off_center_intervals_render_endpoints(self):
        # Twin of the matplotlib endpoint-based pin: the plotly renderers
        # draw the stored interval endpoints verbatim (filled band /
        # segment traces, no estimate-centered arithmetic), so intervals
        # wholly above the estimate render exactly and never raise.
        plotly = pytest.importorskip("plotly")
        assert plotly is not None
        from diff_diff.visualization import plot_event_study, plot_honest_event_study

        surface = _tiny_container(
            # wholly ABOVE the estimates
            conf_int_lower=np.array([0.3, np.nan, 2.2, 2.5]),
            conf_int_upper=np.array([0.5, np.nan, 2.6, 2.9]),
            base_period="universal",
            reference_event_times=(-1,),
        )
        fig = plot_event_study(surface, show=False, backend="plotly")
        band_y = [round(float(v), 6) for v in fig.data[0].y]
        # upper endpoints forward, lower endpoints reversed - verbatim.
        assert band_y == [0.5, 2.6, 2.9, 2.5, 2.2, 0.3]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = compute_honest_did(surface, M=0.5)
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) + 0.2, "ci_ub": float(a) + 0.6}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig2 = plot_honest_event_study(h, show=False, backend="plotly")
        assert fig2 is not None

    def test_off_center_stored_intervals_render(self):
        # Percentile/bootstrap intervals need not contain the point
        # estimate; estimate-centered yerr went negative and crashed
        # matplotlib. Endpoint-based bars draw exactly [lower, upper].
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(
            # interval wholly ABOVE the estimate at -2, wholly BELOW at 1
            conf_int_lower=np.array([0.3, np.nan, 1.66, 1.5]),
            conf_int_upper=np.array([0.5, np.nan, 2.14, 1.9]),
        )
        ax = plot_event_study(surface, show=False)
        assert self._interval_set(ax) == {(0.3, 0.5), (1.66, 2.14), (1.5, 1.9)}

    def test_container_all_post_rows_undefined_fails_closed(self):
        # Pin: an all-invalid post block never reaches zero-dimensional
        # optimization - HonestDiD.fit's centralized num_post check
        # rejects it with a clear message (verified for the container
        # route; the MPD twin is pinned in test_honest_did.py).
        surface = _tiny_container(
            se=np.array([0.1, np.nan, 0.0, np.nan]),
            t_stat=np.array([1.0, np.nan, np.nan, np.nan]),
            p_value=np.array([0.3, np.nan, np.nan, np.nan]),
            conf_int_lower=np.array([-0.1, np.nan, np.nan, np.nan]),
            conf_int_upper=np.array([0.3, np.nan, np.nan, np.nan]),
            base_period="universal",
            reference_event_times=(-1,),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No post-period effects"):
                compute_honest_did(surface, M=0.5)

    def test_honest_plot_skips_rows_excluded_by_honest(self):
        # A zero-SE TRAILING row survives the consecutive-grid check but
        # is excluded from beta_hat; the honest plotter must not paint it
        # with the aggregate honest interval (scalar-bounds path), must
        # accept per-period bounds keyed on the RETAINED set, and must
        # reject an explicit request for the excluded row.
        from diff_diff.visualization import plot_honest_event_study

        surface = _tiny_container(
            se=np.array([0.1, np.nan, 0.12, 0.0]),
            t_stat=np.array([1.0, np.nan, 15.8, np.nan]),
            p_value=np.array([0.3, np.nan, 0.0, np.nan]),
            conf_int_lower=np.array([-0.1, np.nan, 1.66, np.nan]),
            conf_int_upper=np.array([0.3, np.nan, 2.14, np.nan]),
            base_period="universal",
            reference_event_times=(-1,),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = compute_honest_did(surface, M=0.5)
        assert h.post_periods_used == [0]  # row 1 excluded (zero SE)

        # Scalar-bounds path: only retained rows are plotted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plot_honest_event_study(h, show=False)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "1" not in tick_labels and "-2" in tick_labels and "0" in tick_labels

        # Per-period bounds keyed on the retained set render fine.
        h.event_study_bounds = {-2: {"ci_lb": -0.4, "ci_ub": 0.6}, 0: {"ci_lb": 1.6, "ci_ub": 2.4}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert plot_honest_event_study(h, show=False) is not None

        # An explicit request for the excluded row fails with a clear
        # message, never a KeyError or a fabricated honest interval.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="not retained by HonestDiD"):
                plot_honest_event_study(h, periods=[-2, 0, 1], show=False)

    def test_honest_plot_reference_anchor(self, cs_universal):
        # The container's single reference renders as a hollow
        # normalization anchor at 0 with NO standard or honest interval:
        # inferred by default, accepted in explicit periods=, on both
        # backends.
        from diff_diff.visualization import plot_honest_event_study

        surface = cs_universal.aggregate("event_study")
        ref = surface.reference_period
        h = compute_honest_did(surface, M=0.5)
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) - 0.5, "ci_ub": float(a) + 0.5}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        ax = plot_honest_event_study(h, show=False)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert str(ref) in labels
        ref_x = float(labels.index(str(ref)))
        hollow = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 1 and line.get_markerfacecolor() == "white"
        ]
        assert len(hollow) == 1
        assert float(hollow[0].get_xdata()[0]) == ref_x
        # No interval segment (standard or honest) at the reference.
        for c in ax.containers:
            if hasattr(c, "lines"):
                for lc in c.lines[2]:
                    for s in lc.get_segments():
                        if len(s) >= 2:
                            assert abs(s[0][0] - ref_x) > 1e-9
        # Explicit periods= including the reference are accepted.
        all_periods = sorted(int(t) for t in surface.event_time)
        assert plot_honest_event_study(h, periods=all_periods, show=False) is not None
        plotly = pytest.importorskip("plotly")
        assert plotly is not None
        fig = plot_honest_event_study(h, show=False, backend="plotly")
        assert fig is not None

    def test_honest_plot_off_center_intervals_render(self, cs_universal):
        # Honest bounds need not bracket the effects either; the honest
        # plotter draws endpoint-based bars for both interval layers.
        from diff_diff.visualization import plot_honest_event_study

        surface = cs_universal.aggregate("event_study")
        h = compute_honest_did(surface, M=0.5)
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) + 0.2, "ci_ub": float(a) + 0.6}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        fig = plot_honest_event_study(h, show=False)
        assert fig is not None

    def test_container_alpha_mismatch_warns(self):
        # Stored container intervals are at the FIT's level; the plot's
        # ``alpha`` cannot re-level bootstrap/t-based intervals from the
        # SE, so a mismatch is named, never silently relabeled.
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(alpha=0.10)
        with pytest.warns(UserWarning, match="does not apply to an EventStudyResults"):
            plot_event_study(surface, show=False)

    def test_container_alpha_match_silent(self):
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container()  # alpha=0.05 == plot default
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_event_study(surface, show=False)
        assert not [w for w in caught if "does not apply" in str(w.message)]

    def test_explicit_normalization_recomputes_at_requested_alpha(self):
        # The explicit-reference path discards the stored overrides and
        # recomputes pointwise intervals at the requested alpha, so no
        # stored-level mismatch remains to warn about.
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(alpha=0.10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_event_study(surface, reference_period=-1, show=False)
        assert not [w for w in caught if "does not apply" in str(w.message)]

    def test_honest_plot_alpha_mismatch_warns(self, cs_universal):
        from diff_diff.visualization import plot_honest_event_study

        surface = cs_universal.aggregate("event_study")  # alpha=0.05
        h = compute_honest_did(surface, M=0.5, alpha=0.10)
        # Bracketing per-period bounds (the established fixture pattern) so
        # the honest-CI overlay renders.
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) - 0.5, "ci_ub": float(a) + 0.5}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        with pytest.warns(UserWarning, match="container's stored intervals at"):
            plot_honest_event_study(h, show=False)

    def test_gapped_cs_container_multi_reference_plot(self):
        # End-to-end on a REAL CS universal fit over a gapped grid
        # ({1,3,6}, cohorts {3,6}): aggregate("event_study") carries >=2
        # reference rows; automatic plotting hollows every anchor and
        # explicit renormalization around a non-anchor fails closed.
        from diff_diff.visualization import plot_event_study

        rng = np.random.RandomState(3)
        rows = []
        for u in range(40):
            g = 3 if u < 15 else (6 if u < 30 else 0)
            ufe = rng.randn() * 2
            for t in (1, 3, 6):
                post = 1 if (g > 0 and t >= g) else 0
                rows.append(
                    {
                        "unit": u,
                        "period": t,
                        "outcome": ufe + 0.3 * t + 2.0 * post + rng.randn() * 0.5,
                        "first_treat": g,
                    }
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = CallawaySantAnna(n_bootstrap=0, base_period="universal").fit(
                pd.DataFrame(rows),
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        surface = cs.aggregate("event_study")
        keys = surface.event_time.tolist()
        refs = sorted(surface.event_time[surface.is_reference].tolist())
        assert len(refs) >= 2

        ax = plot_event_study(surface, show=False)
        assert self._hollow_marker_xs(ax) == {float(keys.index(r)) for r in refs}
        with pytest.raises(ValueError, match="multiple reference rows"):
            plot_event_study(surface, reference_period=0, show=False)

    def test_explicit_normalization_discards_stale_bands(self):
        # REGISTRY (Event Study Plotting): explicit reference_period=
        # normalization recomputes CIs from normalized effects + original
        # SEs and NaNs the reference CI - simultaneous bands computed
        # around the UN-normalized effects must be discarded, not drawn.
        import matplotlib.pyplot as plt

        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(
            att=np.array([0.1, 0.05, 2.0, 2.1]),
            se=np.array([0.1, 0.1, 0.12, 0.13]),
            t_stat=np.array([1.0, 0.5, 15.0, 16.0]),
            p_value=np.array([0.3, 0.6, 0.0, 0.0]),
            conf_int_lower=np.array([-0.1, -0.15, 1.76, 1.85]),
            conf_int_upper=np.array([0.3, 0.25, 2.24, 2.35]),
            is_reference=np.array([False, False, False, False]),
            n=np.array([10.0, 10.0, 10.0, 10.0]),
            cband_lower=np.array([-0.15, -0.2, 1.7, 1.8]),
            cband_upper=np.array([0.35, 0.3, 2.3, 2.4]),
        )
        ax = plot_event_study(surface, reference_period=-1, show=False)
        # Recovered plotted CIs: reference row has NO error bar (NaN CI);
        # other rows are recentered pointwise intervals, not the stale
        # bands (which bracket the un-normalized effects). Periods map to
        # POSITIONAL x (period -1 sits at x=1 for the [-2,-1,0,1] order).
        ref_x = float([-2, -1, 0, 1].index(-1))
        containers = [c for c in ax.containers if hasattr(c, "lines")]
        assert containers, "expected errorbar containers"
        segs = [seg for c in containers for lc in c.lines[2] for seg in lc.get_segments()]
        assert segs, "expected drawn interval segments"
        for seg in segs:
            (x0, y0), (_, y1) = seg
            if abs(x0 - ref_x) < 1e-9:
                raise AssertionError(
                    f"reference row at positional x={ref_x} must carry no "
                    f"error bar, got {(y0, y1)}"
                )
        # The drawn intervals are the RECOMPUTED pointwise ones around the
        # normalized effects (shifted by the ref effect 0.05), never the
        # stale bands.
        from scipy import stats as scipy_stats

        z = scipy_stats.norm.ppf(0.975)
        expected = set()
        for eff, se_v in ((0.1, 0.1), (2.0, 0.12), (2.1, 0.13)):
            norm_eff = eff - 0.05
            expected.add((round(norm_eff - z * se_v, 6), round(norm_eff + z * se_v, 6)))
        assert self._interval_set(ax) == expected
        plt.close("all")

    def test_calendar_container_positional_split(self):
        # Calendar labels (str/Timestamp) break numeric p<0 splitting; the
        # container branch splits positionally around the reference row.
        from diff_diff.visualization._event_study import _extract_plot_data

        surface = _tiny_container(
            event_time=np.array(["2018", "2019", "2020", "2021"], dtype=object),
            time_scale="calendar",
            is_reference=np.array([False, True, False, False]),
        )
        out = _extract_plot_data(surface, None, None, None, None)
        _, _, _, pre, post, ref, _, _, _, _, _, _ = out
        assert ref == "2019"
        assert pre == ["2018"]
        assert post == ["2020", "2021"]

    def test_stored_pointwise_intervals_preserved(self):
        # The container carries the producer's ACTUAL intervals (bootstrap
        # percentile / survey-t / Bell-McCaffrey differ from att +/- z*se);
        # the plot branch must relay them, not reconstruct normal-style
        # intervals from the SE.
        from diff_diff.visualization._event_study import _extract_plot_data

        surface = _tiny_container(
            # deliberately asymmetric, non +/-1.96*se intervals
            conf_int_lower=np.array([-0.4, np.nan, 1.2, 1.4]),
            conf_int_upper=np.array([0.2, np.nan, 2.2, 2.5]),
        )
        out = _extract_plot_data(surface, None, None, None, None)
        _, _, _, _, _, _, _, band_lo, band_hi, pw_lo, pw_hi, _ = out
        # No cbands on this surface: the band channel falls back per-row to
        # the stored interval; the pointwise channel is the stored interval
        # verbatim. Both preserve the reference row's NaN (undefined
        # inference is never recomputed from the SE).
        for clo, chi in ((band_lo, band_hi), (pw_lo, pw_hi)):
            assert clo is not None and chi is not None
            assert clo[-2] == -0.4 and chi[-2] == 0.2
            assert clo[0] == 1.2 and chi[0] == 2.2
            assert np.isnan(clo[-1]) and np.isnan(chi[-1])

    @staticmethod
    def _vertical_segments(ax):
        segs = []
        for c in ax.containers:
            if not hasattr(c, "lines"):
                continue
            for lc in c.lines[2]:
                segs.extend(lc.get_segments())
        return segs

    @classmethod
    def _interval_set(cls, ax):
        """Drawn vertical error intervals as a set of rounded (lo, hi)."""
        out = set()
        for s in cls._vertical_segments(ax):
            if len(s) < 2:  # NaN rows can yield empty segments
                continue
            ys = sorted((s[0][1], s[1][1]))
            if np.isfinite(ys[0]) and np.isfinite(ys[1]):
                out.add((round(ys[0], 6), round(ys[1], 6)))
        return out

    def test_rendered_intervals_match_stored_ci(self):
        # Survey-t / bootstrap-style stored intervals differ from
        # att +/- 1.96*se; the RENDERED error bars must be the stored
        # endpoints, not normal reconstructions.
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(
            conf_int_lower=np.array([-0.4, np.nan, 1.2, 1.4]),
            conf_int_upper=np.array([0.2, np.nan, 2.2, 2.5]),
        )
        ax = plot_event_study(surface, show=False)
        drawn = self._interval_set(ax)
        # every drawn interval is a STORED one; the z-reconstruction for
        # period 0 (~1.665..2.135) must NOT appear
        assert (1.2, 2.2) in drawn and (-0.4, 0.2) in drawn and (1.4, 2.5) in drawn
        assert not any(abs(lo - 1.665) < 0.01 for lo, _ in drawn)

    def test_use_cband_false_keeps_stored_intervals(self):
        # use_cband=False must select the POINTWISE stored channel, not
        # clear all overrides into normal reconstruction (dCDH bootstrap
        # percentile intervals are asymmetric).
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(
            conf_int_lower=np.array([-0.4, np.nan, 1.2, 1.4]),
            conf_int_upper=np.array([0.2, np.nan, 2.2, 2.5]),
            cband_lower=np.array([-0.6, np.nan, 1.0, 1.2]),
            cband_upper=np.array([0.4, np.nan, 2.4, 2.7]),
        )
        ax = plot_event_study(surface, use_cband=False, show=False)
        drawn = self._interval_set(ax)
        assert (1.2, 2.2) in drawn  # stored pointwise, NOT the band
        assert (1.0, 2.4) not in drawn
        ax2 = plot_event_study(surface, use_cband=True, show=False)
        drawn2 = self._interval_set(ax2)
        assert (1.0, 2.4) in drawn2  # the band channel
        assert (1.2, 2.2) not in drawn2

    def test_zero_se_row_renders_without_interval(self):
        # A non-reference row with se == 0 and NaN stored inference must
        # NOT get a zero-width normal interval: its stored NaN bounds are
        # preserved, so no error bar is drawn at that period.
        from diff_diff.visualization import plot_event_study

        surface = _tiny_container(
            se=np.array([0.0, np.nan, 0.12, 0.13]),
            t_stat=np.array([np.nan, np.nan, 15.8, 16.2]),
            p_value=np.array([np.nan, np.nan, 0.0, 0.0]),
            conf_int_lower=np.array([np.nan, np.nan, 1.66, 1.85]),
            conf_int_upper=np.array([np.nan, np.nan, 2.14, 2.35]),
        )
        ax = plot_event_study(surface, show=False)
        drawn = self._interval_set(ax)
        # only the two rows with defined stored inference draw intervals;
        # the zero-SE row contributes NO zero-width normal interval
        assert drawn == {(1.66, 2.14), (1.85, 2.35)}

    def test_plottable_results_membership(self):
        from diff_diff.visualization._event_study import PlottableResults

        assert "EventStudyResults" in str(PlottableResults)

    def test_plot_honest_event_study_container_route(self, cs_universal):
        # Before this PR the honest plotter's own re-extraction probe fell
        # through to TypeError("Cannot extract event study data from
        # original_results") for container-route HonestDiD results.
        from diff_diff.visualization._event_study import plot_honest_event_study

        surface = cs_universal.aggregate("event_study")
        h = compute_honest_did(surface, M=0.5)
        # event_study_bounds is populated by the per-period bounds path;
        # attach a minimal dict (the established fixture pattern) so the
        # plotter's honest-CI overlay has content.
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) - 0.5, "ci_ub": float(a) + 0.5}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        fig = plot_honest_event_study(h, show=False)
        assert fig is not None

    def test_plot_honest_uses_stored_container_intervals(self):
        # The honest plotter's "original" intervals on the container route
        # must be the container's STORED intervals (survey-t / bootstrap
        # inference), never z-reconstructions; stored NaN (undefined
        # inference, e.g. the replicate df_survey=0 sentinel case) is
        # preserved rather than replaced with finite bounds.
        from diff_diff.visualization._event_study import plot_honest_event_study

        surface = _tiny_container(
            conf_int_lower=np.array([np.nan, np.nan, 1.2, 1.4]),
            conf_int_upper=np.array([np.nan, np.nan, 2.2, 2.5]),
            base_period="universal",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = compute_honest_did(surface, M=0.5)
        h.event_study_bounds = {
            int(t): {"ci_lb": float(a) - 0.5, "ci_ub": float(a) + 0.5}
            for t, a, r in zip(surface.event_time, surface.att, surface.is_reference)
            if not r
        }
        ax = plot_honest_event_study(h, show=False)
        drawn = TestPlotting._interval_set(ax)
        # the stored (not z-reconstructed) original interval appears; the
        # z-reconstruction for period 0 (~1.665..2.135) does not
        assert (1.2, 2.2) in drawn, drawn
        assert not any(abs(lo - 1.665) < 0.01 for lo, _ in drawn)


# --------------------------------------------------------------------------- #
# StackedDiD-sourced container admission (row M-024, second M-093 pre-cut)
# --------------------------------------------------------------------------- #

STACKED_FIT_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _stacked_panel(seed=42, n_units=120, n_periods=12, cohorts=(4, 6, 8)):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=list(cohorts), seed=seed
    )


def _fit_stacked(data, **est_kw):
    from diff_diff import StackedDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StackedDiD(**est_kw).fit(data, **STACKED_FIT_KW)


@pytest.fixture(scope="module")
def stacked_surface():
    """kappa_pre=3 surface: two estimated pre-periods for honest/pretrends."""
    res = _fit_stacked(_stacked_panel(), kappa_pre=3, kappa_post=2)
    return res.aggregate("event_study")


class TestStackedContainerAdmission:
    def test_honest_end_to_end(self, stacked_surface):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h = compute_honest_did(stacked_surface, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)
        assert np.isfinite(h.ci_lb) and np.isfinite(h.ci_ub)
        assert h.ci_lb <= h.lb <= h.ub <= h.ci_ub

    def test_honest_anticipation_fit_behavior(self):
        # Behavior pin, not a threading-mechanism test: honest's container
        # split runs through the MATERIALIZED reference row (numerically the
        # same partition as the anticipation cutoff since ref = -1 - k).
        res = _fit_stacked(_stacked_panel(), kappa_pre=3, kappa_post=2, anticipation=1)
        surf = res.aggregate("event_study")
        assert surf.reference_event_times == (-2,)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h = compute_honest_did(surf, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)

    def test_pretrends_end_to_end_full_vcov(self, stacked_surface):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = compute_pretrends_power(stacked_surface)
        assert np.isfinite(p.power) and 0.0 < p.power <= 1.0
        assert np.isfinite(p.mdv) and p.mdv > 0
        # StackedDiD persists its ES VCV in every inference mode, so the
        # container always takes the full-covariance tier.
        assert p.covariance_source == "full_pre_period_vcov"

    def test_pretrends_anticipation_threading_mechanism(self):
        # THIS is where the surface.anticipation channel is exercised:
        # pretrends reads it directly for the pre cutoff (t < -k), so with
        # anticipation=1 the e=-1 anticipation row must be excluded from
        # the pre set while e=-4,-3 stay.
        res = _fit_stacked(_stacked_panel(), kappa_pre=3, kappa_post=2, anticipation=1)
        surf = res.aggregate("event_study")
        assert surf.anticipation == 1
        pt = PreTrendsPower()
        effects, ses, vcov, n_pre, rel_times, cov_src = pt._extract_pre_period_params(surf)
        # n_pre == 2 proves the e=-1 anticipation row was excluded by the
        # t < -k cutoff (with cutoff 0 it would be 3). rel_times are the
        # ROTH-ANCHORED offsets t - t_ref (#744 convention): raw pre labels
        # {-4, -3} anchored at the ref -2 give [-2, -1].
        assert n_pre == 2
        assert sorted(rel_times.tolist()) == [-2.0, -1.0]
        assert cov_src == "full_pre_period_vcov"

    def test_pretrends_last_period_positional_selection(self):
        # kappa_pre >= 3 so the eligible pre set has TWO horizons and the
        # last_period pattern actually selects one (at kappa_pre=2 the
        # weight vector degenerates to [1.] and the test would be vacuous).
        res = _fit_stacked(_stacked_panel(), kappa_pre=3, kappa_post=2)
        surf = res.aggregate("event_study")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_last = compute_pretrends_power(surf, violation_type="last_period")
        pt = PreTrendsPower(violation_type="last_period")
        effects, ses, vcov, n_pre, rel_times, _ = pt._extract_pre_period_params(surf)
        assert n_pre == 2
        w = pt._get_violation_weights(n_pre, rel_times)
        # chronologically sorted pre rows [-3, -2]: the LAST pre-period
        # (closest to treatment, e=-2) carries the violation mass.
        assert w.tolist() == [0.0, 1.0]
        assert np.isfinite(p_last.power)

    def test_kappa_pre_1_default_rejected(self):
        res = _fit_stacked(_stacked_panel(), kappa_pre=1, kappa_post=2)
        surf = res.aggregate("event_study")
        with pytest.raises(ValueError, match="No pre-period effects"):
            compute_honest_did(surf, M=0.5)
        with pytest.raises(ValueError, match="No pre-treatment periods"):
            compute_pretrends_power(surf)

    # ---- withheld-inference warning (uniform shape-based predicate) ---- #

    def test_bm_failure_fit_warns_in_both_consumers(self, monkeypatch):
        # MANDATORY real-shape fixture: monkeypatched BM-DOF failure on an
        # hc2_bm fit produces the joint-NaN rows (t/p/CI all NaN, finite
        # se) the estimator actually emits - not a hand-mutated container.
        import diff_diff.linalg as dl

        monkeypatch.setattr(
            dl,
            "_compute_cr2_bm_contrast_dof",
            lambda X, cluster_ids, bread_matrix, contrasts, weights=None: np.full(
                contrasts.shape[1], np.nan
            ),
        )
        res = _fit_stacked(
            _stacked_panel(),
            kappa_pre=3,
            kappa_post=2,
            vcov_type="hc2_bm",
            cluster="unit",
        )
        eff = res.event_study_effects
        assert any(
            np.isnan(v["t_stat"])
            and np.isnan(v["p_value"])
            and np.isfinite(v["se"])
            and np.isnan(v["conf_int"][0])
            and np.isnan(v["conf_int"][1])
            for h, v in eff.items()
            if v["n_obs"] > 0
        )
        surf = res.aggregate("event_study")
        with pytest.warns(UserWarning, match="withheld/undefined"):
            h = compute_honest_did(surf, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)
        with pytest.warns(UserWarning, match="withheld/undefined"):
            p = compute_pretrends_power(surf)
        assert np.isfinite(p.power)

    def test_replicate_undefined_state_warns_in_both(self, stacked_surface):
        # df_survey=0.0 (replicate-undefined sentinel) + withheld rows:
        # BOTH consumers warn (uniform predicate - no df conjunct), and
        # honest's identified-set bounds stay FINITE while its FLCI CI
        # endpoints are NaN (the executed-verified output split).
        import dataclasses

        pv = stacked_surface.p_value.copy()
        pv[~stacked_surface.is_reference] = np.nan
        ts = stacked_surface.t_stat.copy()
        ts[~stacked_surface.is_reference] = np.nan
        cl = stacked_surface.conf_int_lower.copy()
        cu = stacked_surface.conf_int_upper.copy()
        cl[~stacked_surface.is_reference] = np.nan
        cu[~stacked_surface.is_reference] = np.nan
        surf = dataclasses.replace(
            stacked_surface,
            p_value=pv,
            t_stat=ts,
            conf_int_lower=cl,
            conf_int_upper=cu,
            df_survey=0.0,
        )
        with pytest.warns(UserWarning, match="withheld/undefined"):
            h = compute_honest_did(surf, M=0.5)
        assert np.isfinite(h.lb) and np.isfinite(h.ub)
        assert np.isnan(h.ci_lb) and np.isnan(h.ci_ub)
        with pytest.warns(UserWarning, match="withheld/undefined"):
            p = compute_pretrends_power(surf)
        assert np.isfinite(p.power)

    def test_positive_finite_df_survey_still_warns(self, stacked_surface):
        # Third state of the "every df_survey state warns" contract - a
        # regression guard against re-introducing a df-conditioned
        # predicate (superseded round-5 draft).
        import dataclasses

        pv = stacked_surface.p_value.copy()
        pv[1] = np.nan
        surf = dataclasses.replace(stacked_surface, p_value=pv, df_survey=25.0)
        with pytest.warns(UserWarning, match="withheld/undefined"):
            compute_honest_did(surf, M=0.5)
        with pytest.warns(UserWarning, match="withheld/undefined"):
            compute_pretrends_power(surf)

    def test_cs_twin_does_not_get_withheld_warning(self, stacked_surface):
        import dataclasses

        pv = stacked_surface.p_value.copy()
        pv[1] = np.nan
        surf = dataclasses.replace(stacked_surface, p_value=pv, source="CallawaySantAnnaResults")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_honest_did(surf, M=0.5)
            compute_pretrends_power(surf)
        assert not any("withheld" in str(w.message) for w in caught)

    # ---- source-flip parity (admission is a pure gate) ---- #

    def test_source_flip_parity(self, stacked_surface):
        import dataclasses

        cs_surface = dataclasses.replace(stacked_surface, source="CallawaySantAnnaResults")
        h_st = compute_honest_did(stacked_surface, M=0.5)
        h_cs = compute_honest_did(cs_surface, M=0.5)
        np.testing.assert_allclose(
            [h_st.lb, h_st.ub, h_st.ci_lb, h_st.ci_ub],
            [h_cs.lb, h_cs.ub, h_cs.ci_lb, h_cs.ci_ub],
            rtol=1e-12,
        )
        pt = PreTrendsPower()
        ext_st = pt._extract_pre_period_params(stacked_surface)
        ext_cs = pt._extract_pre_period_params(cs_surface)
        for a, b in zip(ext_st[:5], ext_cs[:5]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert ext_st[5] == ext_cs[5]
        # end-to-end power at the suite's stochastic tolerance (module
        # docstring: scipy's Genz MVN CDF is internally randomized)
        p_st = compute_pretrends_power(stacked_surface, M=0.1)
        p_cs = compute_pretrends_power(cs_surface, M=0.1)
        assert abs(p_st.power - p_cs.power) < 1e-3

    # ---- guard-message reach tests (seven producer-derived sites) ---- #

    def test_honest_missing_provenance_warning_names_producer(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(stacked_surface, reference_event_times=None)
        assert surf.base_period == "universal"
        with pytest.warns(UserWarning, match="StackedDiD event-study container carries no"):
            compute_honest_did(surf, M=0.5)

    def test_honest_multi_reference_times_error_names_producer(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(stacked_surface, reference_event_times=(-3, -1))
        with pytest.raises(ValueError, match="StackedDiD container records DISTINCT"):
            compute_honest_did(surf, M=0.5)

    def test_honest_varying_base_warning_producer_conditional_remedy(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(
            stacked_surface, base_period="varying", reference_event_times=None
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_honest_did(surf, M=0.5)
        msgs = " ".join(str(w.message) for w in caught)
        assert "Rebuild the container" in msgs
        assert "StackedDiD results" in msgs
        assert "CallawaySantAnna(base_period='universal')" not in msgs

    def test_honest_multi_reference_rows_error(self, stacked_surface):
        import dataclasses

        is_ref = stacked_surface.is_reference.copy()
        # mark the earliest pre row as a second reference (att 0, se 0 so
        # the zero-count reference convention holds shape-wise)
        att = stacked_surface.att.copy()
        se = stacked_surface.se.copy()
        att[0], se[0] = 0.0, 0.0
        is_ref[0] = True
        surf = dataclasses.replace(
            stacked_surface,
            is_reference=is_ref,
            att=att,
            se=se,
            reference_event_times=(-1,),
        )
        with pytest.raises(ValueError, match="multiple reference rows"):
            compute_honest_did(surf, M=0.5)

    def test_pretrends_missing_provenance_warning_names_producer(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(stacked_surface, reference_event_times=None)
        with pytest.warns(UserWarning, match="StackedDiD event-study container carries no"):
            compute_pretrends_power(surf)

    def test_pretrends_multi_reference_times_error_names_producer(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(stacked_surface, reference_event_times=(-3, -1))
        with pytest.raises(ValueError, match="StackedDiD container records DISTINCT"):
            compute_pretrends_power(surf)

    def test_pretrends_varying_base_warning_producer_conditional_remedy(self, stacked_surface):
        import dataclasses

        surf = dataclasses.replace(
            stacked_surface, base_period="varying", reference_event_times=None
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_pretrends_power(surf)
        msgs = " ".join(str(w.message) for w in caught)
        assert "Rebuild the container" in msgs
        assert "PreTrendsPower on a StackedDiD event-study container" in msgs

    # ---- fail-closed structural paths ---- #

    def test_rank_drop_shape_gap_fail_closed_producer_neutral(self, stacked_surface):
        # FIRST pin of honest's retained-grid gap ValueError anywhere in
        # the suite: a NaN-se INTERIOR pre row (the rank-drop shape) drops
        # from the retained set and the grid gains a hole. The message must
        # carry the producer-neutral Stacked remedy, not balance_e (which
        # Stacked's aggregate() does not have).
        import dataclasses

        se = stacked_surface.se.copy()
        # interior pre row (e=-2 on the kappa_pre=3 grid [-3..-2] pre set
        # around ref -1): NaN it so pre keeps only -3 next to post 0..2 -
        # wait, the gap forms between -3 and the reference split; the
        # retained PRE block {-3} is contiguous, so NaN a POST interior
        # row (e=1) instead: post block {0, 2} has a gap.
        idx = stacked_surface.event_time.tolist().index(1)
        se[idx] = np.nan
        surf = dataclasses.replace(stacked_surface, se=se)
        with pytest.raises(ValueError, match="rank-dropped event-time column"):
            compute_honest_did(surf, M=0.5)

    def test_singular_pre_covariance_honest_rejects_pretrends_accepts(self):
        # Low-G / high-kappa_pre: the pre-period covariance sub-block is
        # singular (vcov rank <= G). Honest validates with
        # allow_singular=False (Rambachan-Roth eigenvalues bounded away
        # from zero) and REJECTS; pretrends keeps its documented singular
        # support and succeeds.
        panel = _stacked_panel(n_units=12, n_periods=30, cohorts=(17, 19))
        res = _fit_stacked(panel, kappa_pre=12, kappa_post=2, cluster="unit")
        surf = res.aggregate("event_study")
        with pytest.raises(ValueError, match="singular"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compute_honest_did(surf, M=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = compute_pretrends_power(surf)
        assert np.isfinite(p.power)

    # ---- provenance pins ---- #

    def test_provenance_ladder(self, stacked_surface):
        from diff_diff.survey import SurveyDesign

        # analytical -> df_survey None (deliberate: the resolver reads
        # df_inference, a name Stacked does not use - decision 3; the
        # TODO adapter-naming row must preserve this)
        assert stacked_surface.base_period == "universal"
        assert stacked_surface.reference_event_times == (-1,)
        assert stacked_surface.df_survey is None
        from diff_diff.results_base import _resolve_scalar_df_survey

        panel = _stacked_panel()
        res = _fit_stacked(panel, kappa_pre=3, kappa_post=2)
        assert _resolve_scalar_df_survey(res) is None
        # survey TSL -> the survey df threads
        spanel = panel.copy()
        spanel["w"] = 1.0 + 0.1 * (spanel["unit"] % 5)
        spanel["strata"] = spanel["unit"] % 4
        spanel["psu"] = spanel["unit"]
        from diff_diff import StackedDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sres = StackedDiD(kappa_pre=3, kappa_post=2).fit(
                spanel,
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
                **STACKED_FIT_KW,
            )
        ssurf = sres.aggregate("event_study")
        assert ssurf.df_survey == float(sres.survey_metadata.df_survey)


# --------------------------------------------------------------------------- #
# EfficientDiD containers (rows M-023/M-093): REJECTED BY DESIGN
# --------------------------------------------------------------------------- #


class TestEfficientContainerRejection:
    """A REAL EfficientDiD post-fit container is rejected by both consumers.

    The pinned hand-built rejection (test_non_cs_e0_source_rejected) cannot
    detect a missing message edit - "EfficientDiDResults" already matches
    today's messages via the got source={...!r} interpolation - so this test
    pins the NEW by-design clause text on a genuine aggregate() container.
    """

    def test_real_efficient_container_rejected_by_design(self):
        from diff_diff import EfficientDiD
        from diff_diff.prep_dgp import generate_staggered_data

        d = generate_staggered_data(n_units=80, n_periods=8, cohort_periods=[4, 6], seed=9)
        res = EfficientDiD().fit(
            d,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        surface = res.aggregate("event_study")
        assert surface.source == "EfficientDiDResults"
        with pytest.raises(TypeError, match="rejected BY DESIGN"):
            compute_honest_did(surface, M=1.0)
        with pytest.raises(TypeError, match="rejected BY DESIGN"):
            compute_pretrends_power(surface, M=1.0)


class TestImputationContainerRejection:
    """A REAL ImputationDiD post-fit container is rejected BY DESIGN.

    The match string pins the NEW clause text, not the class name (the
    got source={...!r} interpolation would match "ImputationDiDResults"
    even without the message edit).
    """

    def test_real_imputation_container_rejected_by_design(self):
        from diff_diff import ImputationDiD
        from diff_diff.prep_dgp import generate_staggered_data

        d = generate_staggered_data(n_units=80, n_periods=8, cohort_periods=[4, 6], seed=9)
        res = ImputationDiD().fit(
            d,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        surface = res.aggregate("event_study")
        assert surface.source == "ImputationDiDResults"
        with pytest.raises(TypeError, match="rejected BY DESIGN"):
            compute_honest_did(surface, M=1.0)
        with pytest.raises(TypeError, match="rejected BY DESIGN"):
            compute_pretrends_power(surface, M=1.0)


class TestTwoStageContainerRejection:
    """A REAL TwoStageDiD post-fit container is rejected with the DEFERRED
    clause (not the by-design clause: analytical fits DO carry the joint
    Gardner-GMM covariance, but the pre-period coefficients are stage-1
    residual means, not reference-normalized contrasts - admission awaits a
    normalization derivation; see the REGISTRY TwoStageDiD Note and the
    DEFERRED.md paper-gated row). A pretrends=True fit - the strongest
    admission candidate (estimated pre-periods + joint vcov) - is still
    rejected.
    """

    def test_real_twostage_container_rejected_deferred(self):
        import warnings as _warnings

        from diff_diff import TwoStageDiD
        from diff_diff.prep_dgp import generate_staggered_data

        d = generate_staggered_data(n_units=80, n_periods=8, cohort_periods=[4, 6], seed=9)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            res = TwoStageDiD(pretrends=True).fit(
                d,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        surface = res.aggregate("event_study")
        assert surface.source == "TwoStageDiDResults"
        assert surface.vcov is not None  # the joint GMM covariance IS present
        with pytest.raises(TypeError, match="DEFERRED pending a normalization"):
            compute_honest_did(surface, M=1.0)
        with pytest.raises(TypeError, match="DEFERRED pending a normalization"):
            compute_pretrends_power(surface, M=1.0)


class TestContinuousContainerRejection:
    """A REAL ContinuousDiD post-fit container is rejected BY DESIGN.

    Two independent grounds, both named in the clause: no joint event-study
    covariance (per-bin IF SEs only), and the binarized bins carry NO
    reference-period normalization at all (no reference row exists). The
    match strings pin the NEW clause text, not the class name (the
    got source={...!r} interpolation would match "ContinuousDiDResults"
    even without the message edit).
    """

    def test_real_continuous_container_rejected_by_design(self):
        from diff_diff import ContinuousDiD, generate_continuous_did_data

        d = generate_continuous_did_data(n_units=90, n_periods=6, cohort_periods=[3, 4], seed=17)
        res = ContinuousDiD().fit(
            d,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            dose="dose",
        )
        surface = res.aggregate("event_study")
        assert surface.source == "ContinuousDiDResults"
        assert surface.vcov is None  # no joint ES covariance exists
        with pytest.raises(TypeError, match="no reference-period[\\s\\n ]*normalization"):
            compute_honest_did(surface, M=1.0)
        with pytest.raises(TypeError, match="no reference-period[\\s\\n ]*normalization"):
            compute_pretrends_power(surface, M=1.0)


class TestHADContainerRejection:
    """A REAL HAD post-fit container is rejected - DEFERRED, not by-design.

    Two independent grounds, both named in the clause with the corrected
    wording: no joint cross-horizon covariance (per-horizon independent
    sandwiches; the DEFERRED.md row), and the anchor ROW e = -1 is omitted
    from the container - a metadata gap, NOT missing normalization (the
    coefficients ARE reference-normalized against the F-1 anchor). The
    match strings pin the clause text, not the class name (the
    got source={...!r} interpolation would match the class name even
    without the message edit).
    """

    def test_real_had_container_rejected_deferred(self):
        import warnings

        import numpy as np
        import pandas as pd

        from diff_diff import HeterogeneousAdoptionDiD

        rng = np.random.default_rng(29)
        rows = []
        for i in range(150):
            dose = rng.uniform(0.1, 2.0)
            for t in range(5):
                d_it = dose if t >= 2 else 0.0
                rows.append((i, t, d_it, 1.2 * d_it + rng.normal(), 2))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "ft"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = HeterogeneousAdoptionDiD().fit(
                panel,
                outcome="outcome",
                dose="dose",
                time="period",
                unit="unit",
                first_treat="ft",
            )
        surface = res.aggregate("event_study")
        assert surface.source == "HeterogeneousAdoptionDiDEventStudyResults"
        assert surface.vcov is None  # per-horizon independent sandwiches only
        assert not surface.is_reference.any()  # the anchor row is omitted
        with pytest.raises(TypeError, match="coefficients ARE reference-normalized"):
            compute_honest_did(surface, M=1.0)
        with pytest.raises(TypeError, match="coefficients ARE reference-normalized"):
            compute_pretrends_power(surface, M=1.0)
