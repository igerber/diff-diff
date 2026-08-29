"""survey_metadata raw-scale provenance pins for the unit-level recompute.

``SurveyDesign.resolve()`` rescales pweights to mean 1, so any metadata
recompute that feeds the RESOLVED weights back into
``compute_survey_metadata`` misreports the two scale-dependent fields
(``sum_weights``, ``weight_range``) while leaving the scale-invariant
fields (``effective_n``, ``design_effect``, ``df_survey``, ...) and all
estimates/inference untouched. DMLDiD received the raw-capture pattern in
its survey PR (``tests/test_survey_dml.py::TestSurveyMetadataRawScale``);
these pins cover the four remaining families with a unit-level recompute:
CallawaySantAnna (panel + RC lanes), the staggered DDD engine,
ContinuousDiD (analytic branch), and EfficientDiD.

All fixtures use deliberately non-unit-scale weights (uniform(0.5, 2.0) /
1 + exponential) so a mean-1 rescale is detectable.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ContinuousDiD,
    EfficientDiD,
    StaggeredTripleDifference,
)
from diff_diff.survey import SurveyDesign

# ---------------------------------------------------------------------------
# DGPs (weights constant within unit; strata + PSU present)
# ---------------------------------------------------------------------------


def _make_staggered_panel(n_units=60, n_periods=8, seed=42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 2.0, n_units)
    strata = np.arange(n_units) // 12
    psu = np.arange(n_units) // 5  # crosses strata; designs pass nest=True
    rows = []
    for u in range(n_units):
        ft = 4 if u < 20 else (6 if u < 40 else 0)
        for t in range(1, n_periods + 1):
            y = 10 + 0.05 * u + 0.2 * t + (2.0 if ft > 0 and t >= ft else 0.0)
            y += rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": weights[u],
                    "stratum": strata[u],
                    "psu": psu[u],
                }
            )
    return pd.DataFrame(rows)


def _make_rc_data(n=400, seed=42):
    """One row per unit: CallawaySantAnna(panel=False) rejects duplicates."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "unit": np.arange(n),
            "time": rng.choice([1, 2, 3, 4], n),
            "weight": rng.uniform(0.5, 2.0, n),
            "stratum": rng.choice(3, n),
        }
    )
    df["psu"] = df["stratum"] * 4 + rng.choice(4, n)
    df["first_treat"] = rng.choice([0, 3], n)
    df["outcome"] = rng.normal(0, 1, n) + 1.5 * (
        (df["first_treat"] > 0) & (df["time"] >= df["first_treat"])
    )
    return df


def _make_sddd_data(n_units=200, n_periods=6, seed=42):
    rng = np.random.default_rng(seed)
    cohorts = rng.choice([3, 4, 0], size=n_units, p=[0.4, 0.3, 0.3])
    elig = rng.binomial(1, 0.5, size=n_units)
    weights = 1.0 + rng.exponential(0.5, size=n_units)
    strata = rng.choice(4, size=n_units)
    psu = strata * 2 + rng.choice(2, size=n_units)  # globally unique labels
    rows = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            te = 2.0 if (cohorts[i] > 0 and t >= cohorts[i] and elig[i] == 1) else 0.0
            y = rng.normal(0, 1) + 0.5 * t + te + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "first_treat": cohorts[i],
                    "eligibility": elig[i],
                    "weight": weights[i],
                    "stratum": strata[i],
                    "psu": psu[i],
                }
            )
    return pd.DataFrame(rows)


def _make_continuous_data(n_u=80, n_t=4, seed=42):
    """Units 0-4 are TREATED with dose 0: the drop_units dose filter fires,
    so ``len(df) < len(data)`` inside fit and the survey re-resolve runs on
    the FILTERED frame — expected metadata below must be computed on the
    filtered frame (a raw capture reading the unfiltered input would fail)."""
    rng = np.random.default_rng(seed)
    units = np.repeat(range(n_u), n_t)
    times = np.tile(range(1, n_t + 1), n_u)
    ft_unit = np.where(np.arange(n_u) < 40, 3, 0)
    dose_unit = np.where(np.arange(n_u) < 40, rng.uniform(0.5, 2.0, n_u), 0.0)
    dose_unit[:5] = 0.0
    ft = np.repeat(ft_unit, n_t)
    dose = np.repeat(dose_unit, n_t)
    y = rng.normal(size=len(units)) + 0.5 * dose * (times >= ft) * (ft > 0)
    w = np.repeat(rng.uniform(0.5, 2.0, n_u), n_t)
    strata = np.repeat(np.where(np.arange(n_u) < 40, 1, 2), n_t)
    psu_unit = np.arange(n_u) // 10
    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft,
            "dose": dose,
            "outcome": y,
            "weight": w,
            "stratum": strata,
            "psu": np.repeat(psu_unit, n_t),
        }
    )


_DESIGN = SurveyDesign(weights="weight", strata="stratum", psu="psu", nest=True)
_DESIGN_NO_W = SurveyDesign(strata="stratum", psu="psu", nest=True)


def _assert_raw_scale(md, raw_w):
    """The two scale-dependent fields match the raw weights; the two
    weight-derived scale-invariant ratios match too (raw == normalized
    mathematically; rtol absorbs the constant-rescale last-ULP wiggle)."""
    raw_w = np.asarray(raw_w, dtype=np.float64)
    np.testing.assert_allclose(md.sum_weights, raw_w.sum(), rtol=1e-12)
    np.testing.assert_allclose(md.weight_range, (raw_w.min(), raw_w.max()), rtol=1e-12)
    n = len(raw_w)
    sum_w, sum_w2 = raw_w.sum(), (raw_w**2).sum()
    np.testing.assert_allclose(md.effective_n, sum_w**2 / sum_w2, rtol=1e-12)
    np.testing.assert_allclose(md.design_effect, n * sum_w2 / sum_w**2, rtol=1e-12)


def _fit_quiet(est, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# CallawaySantAnna
# ---------------------------------------------------------------------------


class TestCSRawScale:
    @pytest.fixture(scope="class")
    def panel_df(self):
        return _make_staggered_panel()

    def test_panel_metadata_uses_raw_unit_weights(self, panel_df):
        res = _fit_quiet(
            CallawaySantAnna(),
            panel_df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=_DESIGN,
        )
        _assert_raw_scale(res.survey_metadata, panel_df.groupby("unit")["weight"].first())

    def test_rc_metadata_uses_raw_obs_weights(self):
        rc = _make_rc_data()
        res = _fit_quiet(
            CallawaySantAnna(panel=False),
            rc,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=_DESIGN,
        )
        _assert_raw_scale(res.survey_metadata, rc["weight"])

    def test_injected_cluster_metadata_uses_raw_unit_weights(self, panel_df):
        # PSU-less design + cluster=: metadata is computed on the inject
        # path and recomputed at unit level — final values must still be
        # the raw unit-level scale.
        res = _fit_quiet(
            CallawaySantAnna(cluster="psu"),
            panel_df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="weight", strata="stratum", nest=True),
        )
        _assert_raw_scale(res.survey_metadata, panel_df.groupby("unit")["weight"].first())

    def test_weights_none_ones_fallback(self, panel_df):
        res = _fit_quiet(
            CallawaySantAnna(),
            panel_df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=_DESIGN_NO_W,
        )
        md = res.survey_metadata
        n_units = panel_df["unit"].nunique()
        assert md.sum_weights == n_units
        assert md.weight_range == (1.0, 1.0)
        # Length guard: a wrong-length ones array shifts effective_n.
        assert md.effective_n == n_units

    def test_bare_cluster_never_reaches_metadata_recompute(self, panel_df):
        # Bare cluster= (no survey_design) synthesizes an internal design but
        # leaves survey_metadata None, so the recompute's
        # `assert survey_design is not None` is unreachable on this route
        # (both lanes; also pinned upstream by
        # test_bare_cluster_does_not_set_survey_metadata).
        res = _fit_quiet(
            CallawaySantAnna(cluster="psu"),
            panel_df,
            "outcome",
            "unit",
            "time",
            "first_treat",
        )
        assert np.isfinite(res.overall_att)
        assert res.survey_metadata is None
        rc = _make_rc_data()
        rc["cluster_col"] = rc["unit"] // 20
        res_rc = _fit_quiet(
            CallawaySantAnna(panel=False, cluster="cluster_col"),
            rc,
            "outcome",
            "unit",
            "time",
            "first_treat",
        )
        assert np.isfinite(res_rc.overall_att)
        assert res_rc.survey_metadata is None


# ---------------------------------------------------------------------------
# Empty-string weight-column name (falsy but valid: resolve() checks
# `is not None`, so "" names a real column — truthiness checks would
# silently substitute ones)
# ---------------------------------------------------------------------------


class TestEmptyStringWeightColumn:
    def test_cs_rc_lane(self):
        rc = _make_rc_data().rename(columns={"weight": ""})
        res = _fit_quiet(
            CallawaySantAnna(panel=False),
            rc,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="", strata="stratum", psu="psu", nest=True),
        )
        _assert_raw_scale(res.survey_metadata, rc[""])

    def test_efficient_did(self):
        panel = _make_staggered_panel().rename(columns={"weight": ""})
        res = _fit_quiet(
            EfficientDiD(n_bootstrap=0),
            panel,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="", strata="stratum", psu="psu", nest=True),
        )
        _assert_raw_scale(res.survey_metadata, panel.groupby("unit")[""].first())

    def test_sun_abraham_parity_with_named_duplicate(self):
        # SunAbraham's cohort/event-time aggregation reads the weight COLUMN
        # NAME; a truthiness check treated "" as no-weights and silently fell
        # back to unweighted cohort mass (behavioral: moved att AND se).
        # A column named "" must give bit-identical results to the same
        # values under a normal name.
        from diff_diff import SunAbraham

        panel = _make_staggered_panel()
        panel[""] = panel["weight"]

        def fit(col):
            return _fit_quiet(
                SunAbraham(),
                panel,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=SurveyDesign(weights=col),
            )

        a, b = fit(""), fit("weight")
        assert a.overall_att == b.overall_att
        assert a.overall_se == b.overall_se
        for e in a.event_study_effects:
            assert a.event_study_effects[e]["effect"] == b.event_study_effects[e]["effect"]
            assert a.event_study_effects[e]["se"] == b.event_study_effects[e]["se"]

    def test_wooldridge_zero_weight_group_parity(self):
        # WooldridgeDiD's pre-exclusion zero-weight-group validation reads
        # the weight column name; a truthiness check skipped it for "" so a
        # zero-weight comparison cell passed validation under "" but was
        # rejected under a normal name. Both aliases must behave identically.
        from diff_diff import WooldridgeDiD

        rng = np.random.default_rng(11)
        n_u, n_t = 30, 4
        rows = []
        w = rng.uniform(0.5, 2.0, n_u)
        for u in range(n_u):
            ft = 3 if u < 15 else 0
            for t in range(1, n_t + 1):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": ft,
                        "outcome": rng.normal() + (1.0 if ft and t >= ft else 0.0),
                        "w": w[u],
                    }
                )
        df = pd.DataFrame(rows)
        # Zero out one comparison unit's weights entirely so the weighted
        # within-transform's 0/0 guard has something to reject.
        df.loc[df["unit"] == 20, "w"] = 0.0
        df[""] = df["w"]

        def outcome_of(col):
            try:
                r = _fit_quiet(
                    WooldridgeDiD(),
                    df,
                    "outcome",
                    "unit",
                    "time",
                    "first_treat",
                    survey_design=SurveyDesign(weights=col),
                )
                return ("fit", r.overall_att, r.overall_se)
            except (ValueError, NotImplementedError) as exc:
                # Normalize only the quoted column name, so messages that
                # embed it still compare equal across the two aliases.
                return ("raise", type(exc).__name__, str(exc).replace(f"'{col}'", "'<w>'"))

        assert outcome_of("") == outcome_of("w")

    def test_continuous_did_analytical(self):
        cont = _make_continuous_data().rename(columns={"weight": ""})
        info = cont.groupby("unit").first()[["first_treat", "dose"]]
        drop = info[(info["first_treat"] > 0) & (info["dose"] == 0)].index
        kept = cont[~cont["unit"].isin(drop)]
        res = _fit_quiet(
            ContinuousDiD(n_bootstrap=0),
            cont,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=SurveyDesign(weights="", strata="stratum", psu="psu", nest=True),
        )
        _assert_raw_scale(res.survey_metadata, kept.groupby("unit")[""].first())


# ---------------------------------------------------------------------------
# Staggered DDD engine
# ---------------------------------------------------------------------------


class TestStaggeredDDDRawScale:
    @pytest.fixture(scope="class")
    def sddd_df(self):
        return _make_sddd_data()

    def _fit(self, df, design):
        return _fit_quiet(
            StaggeredTripleDifference(),
            df,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=design,
        )

    def test_metadata_uses_raw_unit_weights(self, sddd_df):
        res = self._fit(sddd_df, _DESIGN)
        _assert_raw_scale(res.survey_metadata, sddd_df.groupby("unit")["weight"].first())

    def test_weights_none_ones_fallback(self, sddd_df):
        res = self._fit(sddd_df, _DESIGN_NO_W)
        md = res.survey_metadata
        n_units = sddd_df["unit"].nunique()
        assert md.sum_weights == n_units
        assert md.weight_range == (1.0, 1.0)
        assert md.effective_n == n_units

    def test_bare_cluster_never_reaches_metadata_recompute(self, sddd_df):
        # Same unreachability pin as the CS variant, on the DDD engine.
        df = sddd_df.assign(cluster_col=sddd_df["unit"] // 10)
        res = _fit_quiet(
            StaggeredTripleDifference(cluster="cluster_col"),
            df,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
        )
        assert np.isfinite(res.overall_att)
        assert res.survey_metadata is None


# ---------------------------------------------------------------------------
# ContinuousDiD
# ---------------------------------------------------------------------------


class TestContinuousDiDRawScale:
    @pytest.fixture(scope="class")
    def cont_df(self):
        return _make_continuous_data()

    @pytest.fixture(scope="class")
    def cont_filtered(self, cont_df):
        # Mirror the estimator's dose filter: drop treated units with dose 0.
        info = cont_df.groupby("unit").first()[["first_treat", "dose"]]
        drop = info[(info["first_treat"] > 0) & (info["dose"] == 0)].index
        assert len(drop) > 0  # the fixture must actually exercise the filter
        return cont_df[~cont_df["unit"].isin(drop)]

    def _fit(self, df, design, **kw):
        return _fit_quiet(
            ContinuousDiD(**kw),
            df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=design,
        )

    def test_analytic_metadata_uses_raw_unit_weights_of_filtered_frame(
        self, cont_df, cont_filtered
    ):
        res = self._fit(cont_df, _DESIGN, n_bootstrap=0)
        _assert_raw_scale(res.survey_metadata, cont_filtered.groupby("unit")["weight"].first())

    def test_bootstrap_metadata_stays_obs_level_raw(self, cont_df, cont_filtered):
        # Characterization pin for the deferred analytic-vs-bootstrap
        # granularity divergence (TODO.md): the bootstrap branch keeps the
        # OBS-level raw metadata from the (re-)resolve on the filtered frame.
        res = self._fit(cont_df, _DESIGN, n_bootstrap=20, seed=42)
        w = cont_filtered["weight"]
        md = res.survey_metadata
        np.testing.assert_allclose(md.sum_weights, w.sum(), rtol=1e-12)
        np.testing.assert_allclose(md.weight_range, (w.min(), w.max()), rtol=1e-12)

    def test_weight_column_aliasing_dose_reports_original_values(self):
        # weights == dose: the never-treated nonzero-dose coercion zeroes the
        # dose column on the working frame BEFORE metadata construction; the
        # metadata must still report the user's ORIGINAL column values
        # (snapshotted pre-mutation), not the coerced ones.
        rng = np.random.default_rng(3)
        n_u, n_t = 60, 4
        units = np.repeat(range(n_u), n_t)
        times = np.tile(range(1, n_t + 1), n_u)
        ft = np.repeat(np.where(np.arange(n_u) < 30, 3, 0), n_t)
        dose = np.repeat(rng.uniform(0.5, 2.0, n_u), n_t)  # NT rows nonzero
        y = rng.normal(size=len(units)) + 0.5 * dose * (times >= ft) * (ft > 0)
        df = pd.DataFrame(
            {"unit": units, "time": times, "first_treat": ft, "dose": dose, "outcome": y}
        )
        res = _fit_quiet(
            ContinuousDiD(n_bootstrap=0),
            df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=SurveyDesign(weights="dose"),
        )
        _assert_raw_scale(res.survey_metadata, df.groupby("unit")["dose"].first())

    def test_filtered_aliased_weights_match_immutable_duplicate(self):
        # COMBINED interaction: zero-dose TREATED units (the drop_units
        # filter fires -> the survey is re-resolved on the filtered rows)
        # AND weights == dose (the never-treated nonzero-dose coercion
        # mutates that column on the working frame). The re-resolve must
        # read PRISTINE data rows: resolving on the mutated frame
        # zero-weighted every never-treated unit (previously "No valid
        # (g,t) cells"). The aliased fit must match a fit on an immutable
        # duplicate of the original column on estimates, SEs, and every
        # metadata field.
        rng = np.random.default_rng(3)
        n_u, n_t = 60, 4
        units = np.repeat(range(n_u), n_t)
        times = np.tile(range(1, n_t + 1), n_u)
        ft = np.repeat(np.where(np.arange(n_u) < 30, 3, 0), n_t)
        dose_u = rng.uniform(0.5, 2.0, n_u)
        dose_u[:5] = 0.0  # treated zero-dose: filter fires
        dose = np.repeat(dose_u, n_t)  # never-treated rows nonzero: coercion fires
        y = rng.normal(size=len(units)) + 0.5 * dose * (times >= ft) * (ft > 0)
        df = pd.DataFrame(
            {"unit": units, "time": times, "first_treat": ft, "dose": dose, "outcome": y}
        )
        df["w_dup"] = df["dose"]  # immutable duplicate of the ORIGINAL values

        def fit_with(wcol):
            return _fit_quiet(
                ContinuousDiD(n_bootstrap=0),
                df,
                "outcome",
                "unit",
                "time",
                "first_treat",
                "dose",
                survey_design=SurveyDesign(weights=wcol),
            )

        r_alias, r_dup = fit_with("dose"), fit_with("w_dup")
        assert r_alias.overall_att == r_dup.overall_att
        assert r_alias.overall_att_se == r_dup.overall_att_se
        ma, md_ = r_alias.survey_metadata, r_dup.survey_metadata
        for field in (
            "sum_weights",
            "weight_range",
            "effective_n",
            "design_effect",
            "df_survey",
            "n_strata",
            "n_psu",
            "weight_type",
        ):
            assert getattr(ma, field) == getattr(md_, field), field
        # And both report the raw scale of the surviving (filtered) units.
        kept = df[~df["unit"].isin(range(5))]
        _assert_raw_scale(ma, kept.groupby("unit")["dose"].first())

    def test_weights_none_ones_fallback(self, cont_df, cont_filtered):
        res = self._fit(cont_df, _DESIGN_NO_W, n_bootstrap=0)
        md = res.survey_metadata
        n_units = cont_filtered["unit"].nunique()
        assert md.sum_weights == n_units
        assert md.weight_range == (1.0, 1.0)
        assert md.effective_n == n_units


# ---------------------------------------------------------------------------
# EfficientDiD
# ---------------------------------------------------------------------------


class TestEfficientDiDRawScale:
    @pytest.fixture(scope="class")
    def panel_df(self):
        return _make_staggered_panel()

    def _fit(self, df, design):
        return _fit_quiet(
            EfficientDiD(n_bootstrap=0),
            df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=design,
        )

    def test_metadata_uses_raw_unit_weights(self, panel_df):
        res = self._fit(panel_df, _DESIGN)
        _assert_raw_scale(res.survey_metadata, panel_df.groupby("unit")["weight"].first())

    def test_weights_none_ones_fallback(self, panel_df):
        res = self._fit(panel_df, _DESIGN_NO_W)
        md = res.survey_metadata
        n_units = panel_df["unit"].nunique()
        assert md.sum_weights == n_units
        assert md.weight_range == (1.0, 1.0)
        assert md.effective_n == n_units

    def test_shuffled_rows_last_cohort_alignment(self, panel_df):
        # Hardening for the positional data/_unit_first_panel_row alignment:
        # shuffled input rows + all-eventually-treated panel routed through
        # control_group="last_cohort" (unit_info reclassification, trimmed
        # period list — but no df row drops) must still surface one raw
        # weight per unit.
        df = panel_df.copy()
        df.loc[df["first_treat"] == 0, "first_treat"] = 7  # all eventually treated
        df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        res = _fit_quiet(
            EfficientDiD(n_bootstrap=0, control_group="last_cohort"),
            df,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=_DESIGN,
        )
        _assert_raw_scale(res.survey_metadata, df.groupby("unit")["weight"].first())
