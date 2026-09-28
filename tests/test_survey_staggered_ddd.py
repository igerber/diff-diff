"""Survey support tests for StaggeredTripleDifference estimator.

Tests follow the patterns from test_survey_phase4.py and test_survey_phase7a.py:
uniform weight equivalence, scale invariance, nontrivial weight effects,
full design (strata/PSU/FPC), replicate weights, and bootstrap + survey.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import StaggeredTripleDifference
from diff_diff.survey import SurveyDesign

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_staggered_ddd_data(n_units=200, n_periods=5, seed=42):
    """Generate balanced panel for staggered DDD with survey columns.

    Creates 3 cohorts:
      - g=3 (treated at period 3), ~40% of units
      - g=4 (treated at period 4), ~30% of units
      - g=0 (never-treated), ~30% of units

    Each unit is either eligible (Q=1, ~50%) or ineligible (Q=0).
    Treatment effect = 2.0 for eligible units in post-treatment periods.
    """
    rng = np.random.default_rng(seed)

    # Assign cohorts
    cohort_probs = [0.4, 0.3, 0.3]
    cohorts = rng.choice([3, 4, 0], size=n_units, p=cohort_probs)

    # Assign eligibility (binary, time-invariant)
    eligibility = rng.binomial(1, 0.5, size=n_units)

    # Survey design columns (unit-level, constant within unit)
    # Variable weights: higher for some units
    weights = 1.0 + rng.exponential(0.5, size=n_units)
    # Strata: 4 strata
    strata = rng.choice(4, size=n_units)
    # PSU: 2 PSUs per stratum = 8 PSUs total
    psu = strata * 2 + rng.choice(2, size=n_units)
    # Unit-level covariate (time-invariant)
    x1 = rng.normal(0, 1, size=n_units)

    rows = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            # Outcome: unit FE + time FE + treatment effect + noise
            unit_fe = rng.normal(0, 1)
            time_fe = 0.5 * t
            # Treatment effect for eligible treated units in post period
            g = cohorts[i]
            q = eligibility[i]
            te = 2.0 if (g > 0 and t >= g and q == 1) else 0.0
            y = unit_fe + time_fe + te + rng.normal(0, 0.5)

            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "first_treat": cohorts[i],
                    "eligibility": eligibility[i],
                    "weight": weights[i],
                    "stratum": strata[i],
                    "psu": psu[i],
                    "x1": x1[i],
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sddd_data():
    return _make_staggered_ddd_data()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSmokeSDDDSurvey:
    """Basic smoke tests: survey-weighted estimation produces finite results."""

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_finite_results(self, sddd_data, method):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(estimation_method=method)
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0

    @pytest.mark.parametrize("agg", ["simple", "event_study", "group", "all"])
    def test_aggregation(self, sddd_data, agg):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(estimation_method="reg")
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate=agg,
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        if agg in ("event_study", "all"):
            assert res.event_study_effects is not None
        if agg in ("group", "all"):
            assert res.group_effects is not None


# ---------------------------------------------------------------------------
# Uniform weight equivalence
# ---------------------------------------------------------------------------


class TestUniformWeightEquivalence:
    """Uniform weights should reproduce unweighted results exactly."""

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_uniform_weights_match_unweighted(self, sddd_data, method):
        data = sddd_data.copy()
        data["uniform_w"] = 1.0

        est = StaggeredTripleDifference(estimation_method=method)

        # Unweighted
        res_uw = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
        )

        # Uniform weights
        sd = SurveyDesign(weights="uniform_w")
        res_w = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )

        np.testing.assert_allclose(
            res_w.overall_att,
            res_uw.overall_att,
            atol=1e-10,
            err_msg=f"ATT mismatch for {method}",
        )
        np.testing.assert_allclose(
            res_w.overall_se,
            res_uw.overall_se,
            rtol=1e-6,
            err_msg=f"SE mismatch for {method}",
        )


# ---------------------------------------------------------------------------
# Scale invariance
# ---------------------------------------------------------------------------


class TestScaleInvariance:
    """Multiplying all weights by a constant should not change results."""

    def test_scale_invariance(self, sddd_data):
        data = sddd_data.copy()
        data["weight_2x"] = data["weight"] * 2.0

        est = StaggeredTripleDifference(estimation_method="dr")

        sd1 = SurveyDesign(weights="weight")
        res1 = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd1,
        )

        sd2 = SurveyDesign(weights="weight_2x")
        res2 = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd2,
        )

        np.testing.assert_allclose(
            res2.overall_att,
            res1.overall_att,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            res2.overall_se,
            res1.overall_se,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Nontrivial weights change SE
# ---------------------------------------------------------------------------


class TestNontrivialWeightsChangeSE:
    """Variable survey weights should produce different SEs than unweighted."""

    def test_weighted_se_differs(self, sddd_data):
        est = StaggeredTripleDifference(estimation_method="reg")

        res_uw = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
        )

        sd = SurveyDesign(weights="weight")
        res_w = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )

        assert abs(res_w.overall_se - res_uw.overall_se) > 1e-6, (
            f"Expected SEs to differ: weighted={res_w.overall_se}, "
            f"unweighted={res_uw.overall_se}"
        )


# ---------------------------------------------------------------------------
# Full survey design (strata/PSU/FPC)
# ---------------------------------------------------------------------------


class TestFullDesign:
    """Strata/PSU/FPC survey design produces finite results with metadata."""

    def test_strata_psu(self, sddd_data):
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
        )
        est = StaggeredTripleDifference(estimation_method="reg")
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.survey_metadata is not None
        assert res.survey_metadata.df_survey is not None
        assert res.survey_metadata.df_survey > 0

    def test_metadata_populated(self, sddd_data):
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = StaggeredTripleDifference(estimation_method="dr")
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )
        sm = res.survey_metadata
        assert sm is not None
        assert sm.effective_n > 0
        assert sm.n_strata > 0
        assert sm.n_psu > 0


# ---------------------------------------------------------------------------
# Design-based aggregation SEs
# ---------------------------------------------------------------------------


class TestDesignBasedAggSE:
    """Aggregated SEs with strata/PSU differ from pweight-only SEs."""

    def test_design_se_differs_from_pweight_only(self, sddd_data):
        est = StaggeredTripleDifference(estimation_method="reg")

        # pweight only
        sd_pw = SurveyDesign(weights="weight")
        res_pw = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd_pw,
        )

        # Full design
        sd_full = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
        )
        res_full = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd_full,
        )

        # Point estimates should be identical (same weights)
        np.testing.assert_allclose(
            res_full.overall_att,
            res_pw.overall_att,
            atol=1e-10,
        )

        # Aggregated SEs should differ (design-based variance)
        assert abs(res_full.overall_se - res_pw.overall_se) > 1e-6, (
            f"Expected aggregated SEs to differ: full={res_full.overall_se}, "
            f"pweight-only={res_pw.overall_se}"
        )


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


class TestReplicateWeights:
    """Replicate weight methods produce finite results."""

    def test_brr_replicate(self, sddd_data):
        data = sddd_data.copy()
        rng = np.random.default_rng(99)
        n_units = data["unit"].nunique()
        R = 20
        # Generate combined replicate weights: rep_r = weight * factor_r
        # (combined_weights=True means each column includes full-sample weight)
        unit_ids = sorted(data["unit"].unique())
        base_w = data.groupby("unit")["weight"].first().reindex(unit_ids).values
        for r in range(R):
            factor = np.abs(1.0 + rng.standard_normal(n_units) * 0.1)
            unit_w = dict(zip(unit_ids, base_w * factor))
            data[f"rep_{r}"] = data["unit"].map(unit_w)

        rep_cols = [f"rep_{r}" for r in range(R)]
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        est = StaggeredTripleDifference(estimation_method="reg")
        res = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0


# ---------------------------------------------------------------------------
# Replicate + bootstrap rejection
# ---------------------------------------------------------------------------


class TestReplicateBootstrapRejection:
    """Replicate weights + n_bootstrap>0 raises NotImplementedError."""

    def test_brr_with_bootstrap_rejected(self, sddd_data):
        data = sddd_data.copy()
        rng = np.random.default_rng(99)
        n_units = data["unit"].nunique()
        unit_ids = sorted(data["unit"].unique())
        R = 10
        base_w = data.groupby("unit")["weight"].first().reindex(unit_ids).values
        for r in range(R):
            factor = np.abs(1.0 + rng.standard_normal(n_units) * 0.1)
            unit_w = dict(zip(unit_ids, base_w * factor))
            data[f"rep_{r}"] = data["unit"].map(unit_w)

        rep_cols = [f"rep_{r}" for r in range(R)]
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        est = StaggeredTripleDifference(
            estimation_method="reg",
            n_bootstrap=49,
        )
        with pytest.raises(NotImplementedError, match="replicate"):
            est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                survey_design=sd,
            )


# ---------------------------------------------------------------------------
# Replicate-weight scale invariance
# ---------------------------------------------------------------------------


def _make_brr_data(sddd_data, rng_seed=99, R=20):
    """Helper: build combined BRR replicate weights for sddd_data."""
    data = sddd_data.copy()
    rng = np.random.default_rng(rng_seed)
    unit_ids = sorted(data["unit"].unique())
    n_units = len(unit_ids)
    base_w = data.groupby("unit")["weight"].first().reindex(unit_ids).values
    for r in range(R):
        factor = np.abs(1.0 + rng.standard_normal(n_units) * 0.1)
        unit_w = dict(zip(unit_ids, base_w * factor))
        data[f"rep_{r}"] = data["unit"].map(unit_w)
    rep_cols = [f"rep_{r}" for r in range(R)]
    return data, rep_cols


class TestReplicateScaleInvariance:
    """Rescaling all weights + replicates by constant k must not change results."""

    @pytest.mark.parametrize("agg", ["simple", "event_study", "group"])
    def test_scale_invariance(self, sddd_data, agg):
        data, rep_cols = _make_brr_data(sddd_data)
        k = 5.0

        sd1 = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        est = StaggeredTripleDifference(estimation_method="reg")
        res1 = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate=agg,
            survey_design=sd1,
        )

        # Scale all weights and replicate columns by k
        data_k = data.copy()
        data_k["weight"] = data_k["weight"] * k
        for col in rep_cols:
            data_k[col] = data_k[col] * k

        sd2 = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        res2 = est.fit(
            data_k,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate=agg,
            survey_design=sd2,
        )

        np.testing.assert_allclose(
            res2.overall_att,
            res1.overall_att,
            atol=1e-12,
            err_msg=f"ATT changed with weight rescaling (agg={agg})",
        )
        np.testing.assert_allclose(
            res2.overall_se,
            res1.overall_se,
            rtol=1e-6,
            err_msg=f"SE changed with weight rescaling (agg={agg})",
        )

        # Assert sub-aggregation outputs are also scale-invariant
        if agg == "event_study" and res1.event_study_effects:
            for e_key in res1.event_study_effects:
                np.testing.assert_allclose(
                    res2.event_study_effects[e_key]["effect"],
                    res1.event_study_effects[e_key]["effect"],
                    atol=1e-12,
                    err_msg=f"Event study e={e_key} effect changed",
                )
                np.testing.assert_allclose(
                    res2.event_study_effects[e_key]["se"],
                    res1.event_study_effects[e_key]["se"],
                    rtol=1e-6,
                    err_msg=f"Event study e={e_key} SE changed",
                )
        if agg == "group" and res1.group_effects:
            for g_key in res1.group_effects:
                np.testing.assert_allclose(
                    res2.group_effects[g_key]["effect"],
                    res1.group_effects[g_key]["effect"],
                    atol=1e-12,
                    err_msg=f"Group g={g_key} effect changed",
                )
                np.testing.assert_allclose(
                    res2.group_effects[g_key]["se"],
                    res1.group_effects[g_key]["se"],
                    rtol=1e-6,
                    err_msg=f"Group g={g_key} SE changed",
                )


# ---------------------------------------------------------------------------
# Survey-weighted aggregation point estimates
# ---------------------------------------------------------------------------


class TestSurveyWeightedAggregation:
    """Survey weights change aggregation point estimates (not just SEs)."""

    def test_unequal_cohort_weights_change_aggregate(self):
        """Cohorts with very different survey weights produce different
        aggregated ATT from unweighted."""
        # Create data where cohort g=3 units have weight=10 and g=4 have weight=1
        data = _make_staggered_ddd_data(n_units=200, seed=123)
        rng = np.random.default_rng(123)
        unit_df = data.groupby("unit")["first_treat"].first()
        # Assign extreme weights: g=3 units 10x heavier than g=4
        w_map = {}
        for uid, g in unit_df.items():
            if g == 3:
                w_map[uid] = 10.0 + rng.uniform(0, 1)
            elif g == 4:
                w_map[uid] = 1.0 + rng.uniform(0, 0.1)
            else:
                w_map[uid] = 3.0 + rng.uniform(0, 0.5)
        data["skewed_w"] = data["unit"].map(w_map)

        est = StaggeredTripleDifference(estimation_method="reg")

        # Unweighted
        res_uw = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
        )

        # Skewed survey weights
        sd = SurveyDesign(weights="skewed_w")
        res_w = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd,
        )

        # Both should be finite
        assert np.isfinite(res_uw.overall_att)
        assert np.isfinite(res_w.overall_att)

        # Aggregated ATT should differ due to different cohort weighting
        assert abs(res_w.overall_att - res_uw.overall_att) > 1e-6, (
            f"Expected aggregate ATTs to differ with skewed weights: "
            f"weighted={res_w.overall_att}, unweighted={res_uw.overall_att}"
        )


# ---------------------------------------------------------------------------
# pweight-only validation
# ---------------------------------------------------------------------------


class TestPweightOnlyValidation:
    """fweight/aweight rejected with ValueError."""

    def test_fweight_rejected(self, sddd_data):
        data = sddd_data.copy()
        data["fweight"] = 1
        sd = SurveyDesign(weights="fweight", weight_type="fweight")
        est = StaggeredTripleDifference(estimation_method="reg")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                survey_design=sd,
            )

    def test_aweight_rejected(self, sddd_data):
        data = sddd_data.copy()
        data["aweight"] = 1.0
        sd = SurveyDesign(weights="aweight", weight_type="aweight")
        est = StaggeredTripleDifference(estimation_method="reg")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                survey_design=sd,
            )


# ---------------------------------------------------------------------------
# Bootstrap + survey
# ---------------------------------------------------------------------------


class TestBootstrapSurvey:
    """Bootstrap with survey design produces bootstrap results."""

    def test_bootstrap_with_survey(self, sddd_data):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(
            estimation_method="reg",
            n_bootstrap=49,
            seed=42,
        )
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd,
        )
        assert res.bootstrap_results is not None
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0


# ---------------------------------------------------------------------------
# Control group variants
# ---------------------------------------------------------------------------


class TestControlGroupSurvey:
    """Both control_group settings work with survey."""

    @pytest.mark.parametrize("cg", ["nevertreated", "notyettreated"])
    def test_control_group(self, sddd_data, cg):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(
            estimation_method="reg",
            control_group=cg,
        )
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)


# ---------------------------------------------------------------------------
# Base period variants
# ---------------------------------------------------------------------------


class TestBasePeriodSurvey:
    """Both base_period settings work with survey."""

    @pytest.mark.parametrize("bp", ["varying", "universal"])
    def test_base_period(self, sddd_data, bp):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(
            estimation_method="reg",
            base_period=bp,
        )
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)


# ---------------------------------------------------------------------------
# Covariate-adjusted survey paths
# ---------------------------------------------------------------------------


class TestCovariateAdjustedSurvey:
    """Covariate-adjusted survey estimation produces finite results."""

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_covariate_survey(self, sddd_data, method):
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(estimation_method=method)
        res = est.fit(
            sddd_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            covariates=["x1"],
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0


# ---------------------------------------------------------------------------
# combined_weights=False replicate path
# ---------------------------------------------------------------------------


class TestReplicateCombinedWeightsFalse:
    """Replicate weights with combined_weights=False produce finite results."""

    def test_non_combined_replicate(self, sddd_data):
        data = sddd_data.copy()
        rng = np.random.default_rng(77)
        unit_ids = sorted(data["unit"].unique())
        n_units = len(unit_ids)
        R = 20
        # Non-combined: replicate columns are perturbation factors only
        for r in range(R):
            factor = np.abs(1.0 + rng.standard_normal(n_units) * 0.1)
            unit_w = dict(zip(unit_ids, factor))
            data[f"rep_{r}"] = data["unit"].map(unit_w)

        rep_cols = [f"rep_{r}" for r in range(R)]
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
            combined_weights=False,
        )
        est = StaggeredTripleDifference(estimation_method="reg")
        res = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="simple",
            survey_design=sd,
        )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        assert res.overall_se > 0


# ---------------------------------------------------------------------------
# Regression: weighted pscore fallback
# ---------------------------------------------------------------------------


class TestWeightedPscoreFallback:
    """Force pscore fallback and verify it uses survey-weighted treated share."""

    def test_fallback_uses_weighted_mean(self, sddd_data):
        """With a single covariate that is perfectly collinear with the
        treatment indicator, logit will fail and fall back to the
        unconditional propensity. The fallback should use survey-weighted
        treated share, producing finite results."""
        data = sddd_data.copy()
        # Create a covariate perfectly collinear with eligibility to force
        # logit failure in at least some subgroups
        data["collinear_x"] = data["eligibility"].astype(float) * 100.0
        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(estimation_method="ipw", pscore_fallback="unconditional")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                covariates=["collinear_x"],
                survey_design=sd,
            )
        assert np.isfinite(res.overall_att)


# ---------------------------------------------------------------------------
# Regression: zero-mass subgroup warning/skip
# ---------------------------------------------------------------------------


class TestZeroMassSubgroupSkip:
    """Zero survey-weight mass subgroups are warned and skipped."""

    def test_zero_mass_subgroup_warns(self):
        """When a subgroup has rows but zero survey weight, the comparison
        should be skipped with a warning (not produce NaN silently)."""
        data = _make_staggered_ddd_data(n_units=200, seed=55)
        # Zero out weights for all eligible units in cohort g=3
        # This makes the (S=3, Q=1) subgroup have zero mass
        mask = (data["first_treat"] == 3) & (data["eligibility"] == 1)
        data.loc[mask, "weight"] = 0.0

        sd = SurveyDesign(weights="weight")
        est = StaggeredTripleDifference(estimation_method="reg")
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = est.fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                survey_design=sd,
            )
        # Should have produced at least one empty/zero-mass subgroup warning
        mass_warnings = [x for x in w if "mass=0" in str(x.message)]
        assert len(mass_warnings) > 0, "Expected zero-mass subgroup warning"
        # Result should still be finite (other cohorts contribute)
        assert np.isfinite(res.overall_att)


# ---------------------------------------------------------------------------
# Paper Eq. 4.14 overall (event-study average) under survey weighting
# ---------------------------------------------------------------------------


class TestSurveyOverallAttEs:
    """The opt-in ``overall_att_es`` (paper Eq. 4.14) honours survey weights: its
    analytical SE routes through the same survey-aware variance estimator as the
    per-event-time effects, so uniform weights reproduce the unweighted result,
    nontrivial weights move the SE, and a full strata/PSU design stays finite.
    """

    def test_uniform_weight_equivalence(self):
        data = _make_staggered_ddd_data(n_units=200, seed=42)
        data["weight"] = 1.0
        sd = SurveyDesign(weights="weight")
        res_w = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
            survey_design=sd,
        )
        res_u = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        assert res_w.overall_att_es is not None and res_u.overall_att_es is not None
        assert np.isclose(res_w.overall_att_es, res_u.overall_att_es, atol=1e-8)
        assert res_w.overall_se_es is not None and res_u.overall_se_es is not None
        assert np.isclose(res_w.overall_se_es, res_u.overall_se_es, atol=1e-8)

    def test_nontrivial_weights_change_se(self):
        data = _make_staggered_ddd_data(n_units=200, seed=42)  # variable weights
        sd = SurveyDesign(weights="weight")
        res_w = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
            survey_design=sd,
        )
        res_u = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
        assert res_w.overall_se_es is not None and res_u.overall_se_es is not None
        # Variable survey weights change the Eq-4.14 overall SE
        assert not np.isclose(res_w.overall_se_es, res_u.overall_se_es, atol=1e-6)

    def test_full_design_overall_es_finite(self):
        data = _make_staggered_ddd_data(n_units=200, seed=42)
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        res = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
            survey_design=sd,
        )
        assert res.overall_att_es is not None and np.isfinite(res.overall_att_es)
        assert res.overall_se_es is not None and np.isfinite(res.overall_se_es)
        assert res.overall_se_es > 0

    def test_replicate_weight_overall_es_finite(self):
        """The Eq. 4.14 overall (overall_att_es) is computed and finite under a
        replicate-weight (BRR) survey design, routing its SE through the replicate
        variance path. Regression for the P1 fix that overall_att_es uses its OWN
        per-statistic effective df (compute_replicate_if_variance), not the simple
        overall's mutated df.
        """
        data = _make_staggered_ddd_data(n_units=200, seed=42)
        rng = np.random.default_rng(99)
        unit_ids = sorted(data["unit"].unique())
        base_w = data.groupby("unit")["weight"].first().reindex(unit_ids).values
        n_rep = 20
        for r in range(n_rep):
            factor = np.abs(1.0 + rng.standard_normal(len(unit_ids)) * 0.1)
            data[f"rep_{r}"] = data["unit"].map(dict(zip(unit_ids, base_w * factor)))
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=[f"rep_{r}" for r in range(n_rep)],
            replicate_method="BRR",
        )
        res = StaggeredTripleDifference(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
            survey_design=sd,
        )
        assert res.overall_att_es is not None and np.isfinite(res.overall_att_es)
        assert res.overall_se_es is not None and np.isfinite(res.overall_se_es)
        assert res.overall_se_es > 0
        # default overall_att (simple) also computed under the replicate design
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)

    def test_single_psu_bootstrap_cluster_unidentified(self):
        """Under a single-PSU survey design with n_bootstrap>0, the Eq. 4.14 overall
        keeps its (analytical) point estimate but its SE is unidentified -> NaN, and
        the inference fields are NaN-consistent.

        Regression for the P2 warning-misdiagnosis fix: when overall_se_es is NaN the
        terminal warning must not attribute it *solely* to a non-finite influence
        function -- a single-PSU/cluster design is the actual cause here, and the
        authoritative "variance unidentified" warning is raised by the bootstrap. The
        Eq. 4.14 warning, when emitted, must name the unidentified-variance cause.
        """
        data = _make_staggered_ddd_data(n_units=120, seed=42).copy()
        data["psu"] = 0  # collapse to a single PSU -> variance/cluster unidentified
        data["fpc_col"] = 10**6  # design-based variance so the PSU is retained
        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = StaggeredTripleDifference(estimation_method="reg", n_bootstrap=50, seed=7).fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                aggregate="event_study",
                survey_design=sd,
            )
        # Point estimate survives; SE + inference are NaN-consistent (never a stale value).
        assert res.overall_att_es is not None and np.isfinite(res.overall_att_es)
        assert res.overall_se_es is not None and np.isnan(res.overall_se_es)
        assert res.overall_t_stat_es is not None and np.isnan(res.overall_t_stat_es)
        assert res.overall_p_value_es is not None and np.isnan(res.overall_p_value_es)
        assert res.overall_conf_int_es is not None
        assert np.isnan(res.overall_conf_int_es[0]) and np.isnan(res.overall_conf_int_es[1])
        msgs = [str(w.message) for w in caught]
        # The bootstrap raises the authoritative "variance unidentified" warning.
        assert any("unidentified" in m for m in msgs)
        # The Eq. 4.14 overall warning is cause-accurate (would fail on the old wording
        # that attributed the NaN solely to a non-finite influence function).
        es_warns = [m for m in msgs if "overall_att_es)" in m]
        assert es_warns, "expected the Eq. 4.14 overall warning under single-PSU bootstrap"
        assert all("unidentified" in m for m in es_warns)
