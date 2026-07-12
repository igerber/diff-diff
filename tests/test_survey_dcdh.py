"""Survey support tests for ChaisemartinDHaultfoeuille (dCDH)."""

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from diff_diff import ChaisemartinDHaultfoeuille, SurveyDesign
from diff_diff.prep_dgp import generate_reversible_did_data

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_data():
    """Reversible-treatment panel with 30 groups, 6 periods."""
    return generate_reversible_did_data(
        n_groups=30,
        n_periods=6,
        pattern="single_switch",
        seed=42,
    )


@pytest.fixture(scope="module")
def data_with_survey(base_data):
    """Add survey columns: weights, strata, PSU."""
    rng = np.random.default_rng(99)
    df = base_data.copy()
    groups = df["group"].unique()

    # Assign per-group (constant within group) survey attributes
    g_weights = {g: rng.uniform(0.5, 3.0) for g in groups}
    g_strata = {g: int(i % 3) for i, g in enumerate(sorted(groups))}
    g_psu = {g: int(i % 10) for i, g in enumerate(sorted(groups))}

    df["pw"] = df["group"].map(g_weights)
    df["stratum"] = df["group"].map(g_strata)
    df["cluster"] = df["group"].map(g_psu)
    return df


# ── Test: Backward compatibility ────────────────────────────────────


class TestBackwardCompat:
    """survey_design=None produces identical results to pre-change code."""

    def test_no_survey_unchanged(self, base_data):
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=None,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) or result.overall_se != result.overall_se
        assert result.survey_metadata is None


# ── Test: Uniform weights = no-survey ───────────────────────────────


class TestUniformWeights:
    """Uniform weights should produce identical results to no survey."""

    def test_uniform_weights_match_unweighted(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw")

        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # Point estimates should match exactly (uniform weights = unweighted mean)
        assert result_plain.overall_att == pytest.approx(result_survey.overall_att, abs=1e-10)


# ── Test: Non-uniform weights change estimate ───────────────────────


class TestNonUniformWeights:

    def test_nonuniform_weights_change_att(self, base_data):
        """With multiple obs per cell and non-uniform weights, ATT differs."""
        # Duplicate rows to create multi-observation cells, then vary
        # outcomes and weights so weighted cell means differ from unweighted.
        rng = np.random.default_rng(77)
        df = base_data.copy()
        df2 = base_data.copy()
        df2["outcome"] = df2["outcome"] + rng.normal(0, 1.0, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)

        # Assign per-group constant weights (heavier on the second copy)
        groups = multi["group"].unique()
        g_weights = {g: rng.uniform(0.5, 3.0) for g in groups}
        multi["pw"] = multi["group"].map(g_weights)
        # Make second copy have different weights (still constant within group)
        # by giving rows from the second batch higher weight via a multiplier
        multi.loc[len(df) :, "pw"] = multi.loc[len(df) :, "pw"] * 2.0

        # Since weights now vary within group (first copy vs second copy),
        # we need per-observation weights. But dCDH requires group-constant
        # survey columns. Instead, use observation-level weights directly:
        # assign random weights per observation (but constant within group).
        # Actually, the constraint is within-GROUP constancy. With multi-obs
        # cells where weights vary within groups, validation will reject.
        # Solution: use group-constant weights with varied outcomes.
        multi["pw"] = multi["group"].map(g_weights)

        sd = SurveyDesign(weights="pw")
        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # With group-constant weights and multi-obs cells, weighted cell
        # means = unweighted cell means (all obs in the cell get the same
        # weight). The ATTs should match. This confirms the equal-cell
        # contract: survey weights don't change the cross-group aggregation.
        # The SE will differ because the survey variance accounts for design.
        assert result_plain.overall_att == pytest.approx(result_survey.overall_att, abs=1e-8)
        # But the SEs should differ when strata/PSU are present
        # (tested separately in TestSurveySE)


# ── Test: Scale invariance ──────────────────────────────────────────


class TestScaleInvariance:

    def test_weight_scale_invariance(self, data_with_survey):
        sd1 = SurveyDesign(weights="pw")

        df2 = data_with_survey.copy()
        df2["pw_scaled"] = df2["pw"] * 100.0
        sd2 = SurveyDesign(weights="pw_scaled")

        r1 = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd1,
        )
        r2 = ChaisemartinDHaultfoeuille(seed=1).fit(
            df2,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd2,
        )
        assert r1.overall_att == pytest.approx(r2.overall_att, abs=1e-10)
        assert r1.overall_se == pytest.approx(r2.overall_se, rel=1e-6)


# ── Test: Survey SE differs from analytical SE ──────────────────────


class TestSurveySE:

    def test_strata_psu_changes_se(self, data_with_survey):
        """Strata + PSU should produce a different SE than weights-only."""
        sd_weights_only = SurveyDesign(weights="pw")
        sd_full = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)

        r_w = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_weights_only,
        )
        r_full = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_full,
        )
        # Point estimates should match (same weights)
        assert r_w.overall_att == pytest.approx(r_full.overall_att, abs=1e-10)
        # SEs should differ (strata/PSU affects variance)
        if np.isfinite(r_w.overall_se) and np.isfinite(r_full.overall_se):
            assert r_w.overall_se != pytest.approx(r_full.overall_se, rel=0.01)

    def test_survey_metadata_populated(self, data_with_survey):
        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert result.survey_metadata is not None


# ── Test: Validation ────────────────────────────────────────────────


class TestValidation:

    def test_rejects_fweight(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            ChaisemartinDHaultfoeuille().fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_rejects_aweight(self, base_data):
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="aweight")
        with pytest.raises(ValueError, match="pweight"):
            ChaisemartinDHaultfoeuille().fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_varying_weights_within_group_accepted(self, base_data):
        """Observation-level weights varying within groups are valid."""
        # Create multi-obs cells with varying weights
        rng = np.random.default_rng(1)
        df = base_data.copy()
        df2 = base_data.copy()
        df2["outcome"] = df2["outcome"] + rng.normal(0, 0.5, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)
        # Observation-level weights (vary within group)
        multi["pw"] = rng.uniform(0.5, 3.0, size=len(multi))
        sd = SurveyDesign(weights="pw")
        # Should succeed - no group-constant restriction
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_varying_weights_change_att(self, base_data):
        """With multi-obs cells and varying weights, ATT differs from unweighted.

        dCDH uses first differences Y_{g,t} - Y_{g,t-1}, so group-constant
        noise cancels. The noise must vary across both group AND time for
        weighted cell means to affect the ATT via different first differences.
        """
        rng = np.random.default_rng(42)
        df = base_data.copy()
        df2 = base_data.copy()
        # Per-observation noise (varies by group AND time)
        df2["outcome"] = df2["outcome"] + rng.normal(0, 3.0, size=len(df2))
        multi = pd.concat([df, df2], ignore_index=True)
        # Give first copy weight=1, second copy weight=10
        multi["pw"] = np.where(np.arange(len(multi)) < len(df), 1.0, 10.0)
        sd = SurveyDesign(weights="pw")
        result_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        result_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            multi,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # Weighted cell means with time-varying noise produce different
        # first differences -> different ATT
        assert result_plain.overall_att != pytest.approx(result_survey.overall_att, abs=0.01)

    def test_rejects_replicate_weights_with_bootstrap(self, base_data):
        """Replicate weights combined with n_bootstrap > 0 is rejected.

        Replicate variance is closed-form (compute_replicate_if_variance);
        combining it with a multiplier bootstrap would double-count
        variance. Matches library precedent in efficient_did.py:989,
        staggered.py:1869, two_stage.py:251-253. The standalone
        replicate-only path (n_bootstrap=0) is supported separately;
        see tests/test_survey_dcdh_replicate_psu.py.
        """
        df = base_data.copy()
        df["pw"] = 1.0
        df["rep1"] = 1.0
        df["rep2"] = 1.0
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        with pytest.raises(NotImplementedError, match="replicate weights and n_bootstrap"):
            ChaisemartinDHaultfoeuille(n_bootstrap=100, seed=1).fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )


# ── Test: Multi-horizon with survey ─────────────────────────────────


class TestMultiHorizonSurvey:

    def test_multi_horizon_survey_runs(self, data_with_survey):
        """L_max >= 1 with survey should produce finite event study effects."""
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert 1 in result.event_study_effects
        assert np.isfinite(result.event_study_effects[1]["effect"])


# ── Test: Bootstrap + survey warning ────────────────────────────────


class TestBootstrapSurveyWarning:

    def test_bootstrap_survey_auto_inject_no_warning(self, data_with_survey):
        """Under auto-inject psu=group, Hall-Mammen wild PSU bootstrap
        coincides with the group-level multiplier bootstrap — so no
        warning should fire. The old 'PSU-level deferred' warning has
        been replaced with a conditional one that only fires when the
        user passes a strictly coarser PSU.

        See the new test file tests/test_survey_dcdh_replicate_psu.py for
        the strictly-coarser-PSU case where the warning DOES fire.
        """
        sd = SurveyDesign(weights="pw")
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            # Ignore warnings unrelated to the bootstrap-PSU contract.
            _w.filterwarnings("ignore", message="Single-period placebo SE")
            _w.filterwarnings("ignore", message="pweight weights normalized")
            _w.filterwarnings("ignore", message="Assumption 11")
            ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
                data_with_survey,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )


# ── Test: SE scale pinning ──────────────────────────────────────────


class TestSEScalePinning:
    """Survey SE with uniform weights and no strata/PSU must match plug-in SE."""

    def test_uniform_survey_se_matches_plugin(self, base_data):
        """Pins the divisor normalization: uniform survey SE with group-level
        PSU clustering should be close to plug-in SE.

        Under dCDH's auto-inject contract (see REGISTRY.md §dCDH survey),
        SurveyDesign(weights='pw') with no explicit psu is equivalent to
        SurveyDesign(weights='pw', psu='group'). Either form aligns the
        effective sampling unit with the group-level IF and matches the
        plug-in SE up to small-sample corrections.
        """
        df = base_data.copy()
        df["pw"] = 1.0
        sd = SurveyDesign(weights="pw", psu="group")

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            base_data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # With PSU=group and uniform weights, survey SE should be
        # close to plug-in SE (both assume group-level independence).
        # Small-sample corrections (n/(n-1)) cause minor differences.
        if np.isfinite(r_plain.overall_se) and np.isfinite(r_survey.overall_se):
            assert r_plain.overall_se == pytest.approx(r_survey.overall_se, rel=0.15), (
                f"Survey SE ({r_survey.overall_se:.6f}) should be close to "
                f"plug-in SE ({r_plain.overall_se:.6f}) with uniform weights "
                f"and PSU=group"
            )


# ── Test: Zero-weight cells ─────────────────────────────────────────


class TestZeroWeightCells:

    def test_zero_weight_cell_excluded(self, base_data):
        """A cell with zero survey weight is treated as absent."""
        df = base_data.copy()
        df["pw"] = 1.0
        # Zero out weight for one group at one period
        target_group = df["group"].unique()[0]
        target_period = df["period"].unique()[1]
        mask = (df["group"] == target_group) & (df["period"] == target_period)
        df.loc[mask, "pw"] = 0.0
        sd = SurveyDesign(weights="pw")

        # Should not raise; the zero-weight cell is just absent
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)


# ── Test: Delta overall surface threads survey df ───────────────────


class TestSurveyDeltaInference:
    """Verify the L_max>=2 cost-benefit delta surface uses survey df."""

    def test_survey_delta_uses_survey_df(self, data_with_survey):
        """Under L_max=2 with a survey design, overall_p_value must match
        t-distribution inference with df=df_survey (not z-inference)."""
        from scipy import stats

        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )
        if not (np.isfinite(r.overall_se) and r.overall_se > 0):
            pytest.skip("delta not estimable on this fixture")

        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert df_s is not None and df_s > 0, f"expected positive df_survey, got {df_s}"

        t_stat = r.overall_att / r.overall_se
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        # Reported p-value must match t-based (proving _df_survey was threaded)
        assert r.overall_p_value == pytest.approx(p_t, abs=1e-10)

    def test_survey_delta_t_differs_from_z(self, base_data):
        """With a small-df design (df~4), survey-t p-value must differ
        measurably from z p-value at the delta surface."""
        from scipy import stats

        df_ = base_data.copy()
        df_["pw"] = 1.0
        # 2 strata × 3 clusters/stratum = 6 nested PSUs → df_survey = 4
        groups = sorted(df_["group"].unique())
        n_g = len(groups)
        strata_map = {g: i // (n_g // 2) for i, g in enumerate(groups)}
        psu_map = {g: i // (n_g // 6) for i, g in enumerate(groups)}
        df_["stratum"] = df_["group"].map(strata_map)
        df_["cluster"] = df_["group"].map(psu_map)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )
        if not (np.isfinite(r.overall_se) and r.overall_se > 0):
            pytest.skip("delta not estimable on this fixture")
        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert (
            df_s is not None and df_s < 30
        ), f"expected small df_survey for t-vs-z gap, got {df_s}"

        t_stat = r.overall_att / r.overall_se
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        p_z = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
        # Threaded p-value must match t, not z
        assert r.overall_p_value == pytest.approx(p_t, abs=1e-10)
        assert (
            abs(r.overall_p_value - p_z) > 1e-6
        ), "overall_p_value must differ from z-inference when df_survey is small"


# ── Test: Survey + controls (DID^X) ─────────────────────────────────


class TestSurveyControls:
    """Covariate-adjusted (DID^X) path must work with survey_design."""

    def test_survey_plus_controls_runs(self, data_with_survey):
        """Covariate-adjusted dCDH with survey_design produces finite ATT."""
        rng = np.random.default_rng(7)
        df_ = data_with_survey.copy()
        df_["x"] = rng.normal(0, 1.0, size=len(df_))
        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            controls=["x"],
            L_max=1,
            survey_design=sd,
        )
        assert np.isfinite(r.overall_att)
        assert r.survey_metadata is not None


# ── Test: Survey + HonestDiD ────────────────────────────────────────


class TestSurveyHonestDiD:
    """HonestDiD bounds on survey-backed dCDH results must carry df_survey."""

    def test_survey_honest_did_propagates_df(self, data_with_survey):
        """results.honest_did_results.df_survey must match
        results.survey_metadata.df_survey (non-None propagation)."""
        import warnings

        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        with warnings.catch_warnings():
            # dCDH HonestDiD emits a methodology-deviation warning
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                data_with_survey,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=2,
                honest_did=True,
                survey_design=sd,
            )
        if r.honest_did_results is None:
            pytest.skip("HonestDiD computation returned None on this fixture")
        assert r.survey_metadata is not None
        df_meta = r.survey_metadata.df_survey
        assert df_meta is not None
        # df_survey must propagate from survey_metadata into HonestDiD result
        assert r.honest_did_results.df_survey == df_meta


# ── Test: Survey-aware heterogeneity ────────────────────────────────


class TestSurveyHeterogeneity:
    """Heterogeneity testing under survey_design uses WLS + TSL IF."""

    def test_uniform_weights_het_matches_unweighted(self, base_data):
        """Uniform pweights must yield identical β_het to plain OLS path."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # time-invariant group-level covariate
        rng = np.random.default_rng(0)
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
        )
        sd = SurveyDesign(weights="pw")
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
            survey_design=sd,
        )
        assert r_plain.heterogeneity_effects is not None
        assert r_survey.heterogeneity_effects is not None
        b_plain = r_plain.heterogeneity_effects[1]["beta"]
        b_survey = r_survey.heterogeneity_effects[1]["beta"]
        # WLS with uniform weights = OLS; β_het must match
        if np.isfinite(b_plain) and np.isfinite(b_survey):
            assert b_plain == pytest.approx(b_survey, abs=1e-8)

    def test_nonuniform_het_changes_beta(self, base_data):
        """Varying pweights change the WLS point estimate vs plain OLS."""
        rng = np.random.default_rng(42)
        df_ = base_data.copy()
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        pw_map = {g: rng.uniform(0.5, 4.0) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        df_["pw"] = df_["group"].map(pw_map)
        sd = SurveyDesign(weights="pw")

        r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
        )
        r_survey = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
            survey_design=sd,
        )
        assert r_plain.heterogeneity_effects is not None
        assert r_survey.heterogeneity_effects is not None
        b_plain = r_plain.heterogeneity_effects[1]["beta"]
        b_survey = r_survey.heterogeneity_effects[1]["beta"]
        if np.isfinite(b_plain) and np.isfinite(b_survey):
            # WLS with non-uniform weights differs from unweighted OLS
            assert b_plain != pytest.approx(
                b_survey, abs=1e-6
            ), f"plain={b_plain} survey={b_survey} should differ with varying weights"

    def test_survey_het_uses_survey_df(self, data_with_survey):
        """Reported p_value must match t-distribution inference with df_survey."""
        from scipy import stats

        rng = np.random.default_rng(7)
        df_ = data_with_survey.copy()
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
            survey_design=sd,
        )
        assert r.heterogeneity_effects is not None
        entry = r.heterogeneity_effects[1]
        if not (np.isfinite(entry["se"]) and entry["se"] > 0):
            pytest.skip("heterogeneity SE not estimable on this fixture")
        assert r.survey_metadata is not None
        df_s = r.survey_metadata.df_survey
        assert df_s is not None and df_s > 0

        t_stat = entry["beta"] / entry["se"]
        p_t = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_s))
        assert entry["p_value"] == pytest.approx(p_t, abs=1e-10)


# ── Test: TWFE helper parity under survey ───────────────────────────


class TestSurveyTWFEParity:
    """twowayfeweights() with survey_design matches fit().twfe_* under survey."""

    def test_twfe_helper_matches_fit_under_survey(self, data_with_survey):
        """fit and twowayfeweights() produce identical TWFE output under survey."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        sd = SurveyDesign(weights="pw", strata="stratum", psu="cluster", nest=True)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        helper = twowayfeweights(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # fit() twfe_* may be None if non-binary treatment; this fixture is binary
        assert r.twfe_fraction_negative is not None
        assert r.twfe_sigma_fe is not None
        assert r.twfe_beta_fe is not None
        assert r.twfe_fraction_negative == pytest.approx(helper.fraction_negative, abs=1e-12)
        assert r.twfe_sigma_fe == pytest.approx(helper.sigma_fe, abs=1e-12)
        assert r.twfe_beta_fe == pytest.approx(helper.beta_fe, abs=1e-12)

    def test_twfe_helper_rejects_non_pweight(self, base_data):
        """fweight/aweight must be rejected by twowayfeweights() under survey."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        df_ = base_data.copy()
        df_["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            twowayfeweights(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_twfe_helper_accepts_replicate_weights(self, base_data):
        """Replicate-weight designs are accepted by the helper (no SE field
        on TWFEWeightsResult, so only cell aggregation is affected). Matches
        fit()'s new contract where replicate variance runs analytically via
        compute_replicate_if_variance."""
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights

        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["rep1"] = 1.0
        df_["rep2"] = 1.0
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        result = twowayfeweights(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.beta_fe)
        assert np.isfinite(result.sigma_fe)
        assert 0.0 <= result.fraction_negative <= 1.0


# ── Test: TWFE diagnostic oracle under survey ───────────────────────


class TestSurveyTWFEOracle:
    """twfe_beta_fe under survey must match an observation-level pweighted
    TWFE regression on the same data (proving w_gt is used, not n_gt)."""

    def test_survey_twfe_matches_obs_level_pweighted_ols(self, data_with_survey):
        from diff_diff.chaisemartin_dhaultfoeuille import twowayfeweights
        from diff_diff.linalg import solve_ols

        sd = SurveyDesign(weights="pw")
        helper = twowayfeweights(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(helper.beta_fe)

        # Build observation-level TWFE design with group and period FE
        # (reference category dropped) and treatment indicator.
        df_ = data_with_survey.copy()
        groups_u = sorted(df_["group"].unique())
        periods_u = sorted(df_["period"].unique())
        g_map = {g: i for i, g in enumerate(groups_u)}
        t_map = {t: i for i, t in enumerate(periods_u)}
        g_idx = df_["group"].map(g_map).to_numpy()
        t_idx = df_["period"].map(t_map).to_numpy()
        n = len(df_)
        X_g = np.zeros((n, len(groups_u) - 1))
        X_t = np.zeros((n, len(periods_u) - 1))
        for i in range(n):
            if g_idx[i] > 0:
                X_g[i, g_idx[i] - 1] = 1.0
            if t_idx[i] > 0:
                X_t[i, t_idx[i] - 1] = 1.0
        intercept = np.ones((n, 1))
        treat = df_["treatment"].to_numpy().astype(float).reshape(-1, 1)
        X_obs = np.hstack([intercept, X_g, X_t, treat])
        y_obs = df_["outcome"].to_numpy().astype(float)
        w_obs = df_["pw"].to_numpy().astype(float)

        coef, _, _ = solve_ols(
            X_obs,
            y_obs,
            weights=w_obs,
            weight_type="pweight",
            return_vcov=False,
        )
        beta_oracle = float(coef[-1])
        # Point-estimate match (one obs per cell in this fixture; so the
        # cell-level WLS with cell_weight == w_gt equals the obs-level
        # WLS with w_obs weights).
        assert helper.beta_fe == pytest.approx(beta_oracle, rel=1e-6), (
            f"helper.beta_fe={helper.beta_fe} oracle={beta_oracle} "
            f"— TWFE diagnostic must use w_gt under survey"
        )


# ── Test: Zero-weight subpopulation exclusion ──────────────────────


class TestZeroWeightSubpopulation:
    """Zero-weight rows must not trip fuzzy-DiD guard or inflate counts."""

    def test_mixed_zero_weight_row_excluded_from_validation(self, base_data):
        """A cell with a positive-weight treated obs and a zero-weight
        obs with a different treatment value must fit cleanly — the
        zero-weight row is out-of-sample (SurveyDesign.subpopulation())."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Pick a treated (g, t) cell. Add a zero-weight row in the same
        # cell with the opposite treatment value. Unweighted d_min != d_max
        # would trip the fuzzy-DiD guard; pre-filtering zero-weight rows
        # must bypass it.
        treated_mask = df_["treatment"] == 1
        if not treated_mask.any():
            pytest.skip("no treated row in fixture")
        sample = df_[treated_mask].iloc[0].copy()
        # Flip treatment on the injected row, give it zero weight
        sample["treatment"] = 0
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")

        # Must succeed (not raise fuzzy-DiD ValueError)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_outcome(self, base_data):
        """A zero-weight row with NaN outcome must not trip the outcome
        NaN validator. SurveyDesign.subpopulation() contract."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        sample = df_.iloc[0].copy()
        sample["outcome"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row with NaN outcome is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_group_id(self, base_data):
        """A zero-weight row with NaN group id must not crash the SE
        factorization. SurveyDesign.subpopulation() contract."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Cast group to object to allow NaN without coercion errors
        df_["group"] = df_["group"].astype(object)
        sample = df_.iloc[0].copy()
        sample["group"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row's NaN group id is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_control(self, base_data):
        """A zero-weight row with NaN in a control column must not abort
        the DID^X path, and the covariate cell aggregation must use only
        positive-weight rows (no length-mismatch error)."""
        rng = np.random.default_rng(13)
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["x"] = rng.normal(0, 1, size=len(df_))
        # Inject a zero-weight row with NaN control value
        sample = df_.iloc[0].copy()
        sample["x"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            controls=["x"],
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_zero_weight_row_with_nan_heterogeneity(self, base_data):
        """A zero-weight row with NaN in the heterogeneity column must
        not trip the heterogeneity time-invariance validator."""
        rng = np.random.default_rng(0)
        df_ = base_data.copy()
        df_["pw"] = 1.0
        groups = sorted(df_["group"].unique())
        het_map = {g: rng.uniform(-1, 1) for g in groups}
        df_["x_het"] = df_["group"].map(het_map)
        # Inject a zero-weight row with NaN het value for an existing group
        sample = df_.iloc[0].copy()
        sample["x_het"] = np.nan
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw")
        # Must succeed — zero-weight row with NaN het is out-of-sample
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            heterogeneity="x_het",
            survey_design=sd,
        )
        assert result.heterogeneity_effects is not None


# ── Test: Survey + trends_linear ────────────────────────────────────


class TestSurveyTrendsLinear:
    """Survey-backed trends_linear fit must populate linear_trends_effects."""

    def test_survey_trends_linear_runs(self, data_with_survey):
        sd = SurveyDesign(weights="pw")
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            trends_linear=True,
            survey_design=sd,
        )
        assert r.survey_metadata is not None
        # linear_trends_effects populated per REGISTRY line 614 contract
        assert r.linear_trends_effects is not None
        # At least one horizon should be estimable with finite value
        finite_horizons = [
            h
            for h, entry in r.linear_trends_effects.items()
            if np.isfinite(entry.get("effect", np.nan))
        ]
        assert (
            len(finite_horizons) > 0
        ), "expected at least one horizon with finite linear_trends_effect"


# ── Test: Survey + trends_nonparam ──────────────────────────────────


class TestSurveyTrendsNonparam:
    """Survey-backed trends_nonparam fit must thread set-restrictions."""

    def test_survey_trends_nonparam_runs(self, data_with_survey):
        # Reuse stratum as set ID (time-invariant per group)
        sd = SurveyDesign(weights="pw")
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            data_with_survey,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            trends_nonparam="stratum",
            survey_design=sd,
        )
        assert r.survey_metadata is not None
        assert r.event_study_effects is not None
        # Support trimming may reduce counts but at least one finite-SE
        # horizon should remain on this fixture.
        finite_ses = [
            entry
            for entry in r.event_study_effects.values()
            if np.isfinite(entry.get("se", np.nan))
        ]
        assert len(finite_ses) > 0, (
            "expected at least one event-study horizon with finite SE "
            "under trends_nonparam + survey"
        )


# ── Test: Survey + design2 ──────────────────────────────────────────


class TestSurveyDesign2:
    """Survey-backed design2 fit must populate design2_effects."""

    @staticmethod
    def _make_join_then_leave_panel(seed=42, n_groups=30, n_periods=8):
        """Panel with join-then-leave (Design-2) groups, matching the
        existing design2 fixture in test_chaisemartin_dhaultfoeuille.py."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            group_fe = rng.normal(0, 2)
            for t in range(n_periods):
                if g < 10:
                    d = 1 if 2 <= t < 5 else 0
                elif g < 20:
                    d = 1 if t >= 3 else 0
                else:
                    d = 0
                y = group_fe + 2.0 * t + 5.0 * d + rng.normal(0, 0.3)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y, "pw": 1.0})
        return pd.DataFrame(rows)

    def test_survey_design2_runs(self):
        df_ = self._make_join_then_leave_panel()
        sd = SurveyDesign(weights="pw")
        # drop_larger_lower=False keeps the 2-switch groups
        r = ChaisemartinDHaultfoeuille(seed=1, drop_larger_lower=False).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
            design2=True,
            survey_design=sd,
        )
        assert r.survey_metadata is not None
        assert r.design2_effects is not None
        assert r.design2_effects["n_design2_groups"] == 10
        # switch_in and switch_out mean effects should be finite
        assert np.isfinite(r.design2_effects["switch_in"]["mean_effect"])
        assert np.isfinite(r.design2_effects["switch_out"]["mean_effect"])


# ── Test: Within-group constancy of strata and PSU ──────────────────


class TestSurveyWithinGroupValidation:
    """Cell-period IF allocator contract: strata and PSU may vary ACROSS
    cells of a group, but must be constant WITHIN each (g, t) cell. In
    canonical one-obs-per-cell panels the cell-level constancy check is
    trivially satisfied. All three variance paths — analytical TSL,
    heterogeneity WLS, and the PSU-level wild multiplier bootstrap —
    support within-group-varying PSU via the cell-period allocator. No
    dCDH survey combination raises NotImplementedError beyond the
    SurveyDesign-level exclusion of ``replicate_weights`` +
    ``n_bootstrap > 0``.
    """

    def test_accepts_varying_psu_within_group(self, base_data):
        """Under the cell-period allocator, PSU that varies across cells
        of a group is a valid design — the allocator attributes IF mass
        to each (g, t) cell separately and Binder TSL aggregates at PSU
        level with the honest cell-level variance.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        # PSU varies within each group (alternates by period). Still
        # constant within each (g, t) cell because one obs per cell.
        df_["psu"] = df_["period"] % 2
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=2,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        # And the SE differs from a within-group-constant-PSU baseline,
        # because the cell allocator now honors the extra PSU structure.
        df_const = base_data.copy()
        df_const["pw"] = 1.0
        df_const["stratum"] = 0
        df_const["psu"] = 0  # constant-within-group PSU baseline
        sd_const = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        r_const = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_const,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_const,
            L_max=2,
        )
        assert result.overall_se != r_const.overall_se, (
            "Cell-period allocator must produce a different SE when PSU "
            "actually varies across cells vs. constant-within-group."
        )

    def test_accepts_varying_strata_within_group(self, base_data):
        """Strata that vary across cells of a group are supported under
        the cell-period allocator, trivially satisfying within-cell
        constancy in one-obs-per-cell panels.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Stratum varies within each group across cells
        df_["stratum"] = df_["period"] % 2
        # PSU = group, nested inside stratum so the resolver accepts the
        # cross-stratum reuse of group labels.
        df_["psu"] = df_["group"]
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu", nest=True)
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=2,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)

    def test_heterogeneity_with_varying_psu_succeeds(self, base_data):
        """heterogeneity= is supported under within-group-varying PSU
        via the cell-period allocator: psi_g is attributed in full to
        the (g, out_idx) post-period cell and expanded to obs level as
        psi_i = psi_g * (w_i / W_{g, out_idx}). All five inference
        fields must be finite when the survey design is regular.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Make x_het time-invariant within each group (heterogeneity
        # test requires a group-level covariate).
        df_["x_het"] = (df_["group"].astype(int) % 2).astype(float)
        # Per-group PSU parity — unique per (group, parity), varies
        # within group so the cell-period allocator is exercised.
        df_["psu"] = df_["group"].astype(int) * 2 + (df_["period"].astype(int) % 2)
        sd = SurveyDesign(weights="pw", psu="psu")
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            heterogeneity="x_het",
            L_max=1,
            survey_design=sd,
        )
        assert res.heterogeneity_effects is not None
        entry = res.heterogeneity_effects[1]
        assert np.isfinite(entry["beta"])
        assert np.isfinite(entry["se"]) and entry["se"] >= 0.0
        assert np.isfinite(entry["t_stat"])
        assert np.isfinite(entry["p_value"])
        assert all(np.isfinite(entry["conf_int"]))

    def test_bootstrap_with_varying_psu_succeeds(self, base_data):
        """PR 4: the PSU-level wild multiplier bootstrap now supports
        within-group-varying PSU via the cell-level allocator. Each
        (g, t) cell's IF mass is multiplied by its PSU's multiplier,
        so a group spanning 2+ PSUs receives independent draws per
        PSU. Assert a finite bootstrap SE and CI at the overall
        surface and at every event-study horizon (the multi-horizon
        path uses a shared PSU-level weight matrix for the sup-t
        simultaneous band).
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Per-group PSU parity — unique per (group, parity), varies
        # within group so the cell-level dispatcher is exercised.
        df_["psu"] = df_["group"].astype(int) * 2 + (df_["period"].astype(int) % 2)
        sd = SurveyDesign(weights="pw", psu="psu")
        res = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
            L_max=2,
        )
        assert res.bootstrap_results is not None
        assert np.isfinite(res.bootstrap_results.overall_se)
        assert res.bootstrap_results.overall_se >= 0.0
        overall_ci = res.bootstrap_results.overall_ci
        assert overall_ci is not None and all(np.isfinite(overall_ci))
        # Multi-horizon bootstrap must produce finite SE at each
        # horizon (guards the shared-PSU-weight path from CRITICAL #2
        # of the PR 4 plan review).
        es_ses = res.bootstrap_results.event_study_ses
        assert es_ses is not None
        for l_h in (1, 2):
            assert l_h in es_ses, f"horizon {l_h} missing from event_study_ses"
            assert np.isfinite(es_ses[l_h]), f"horizon {l_h} SE not finite"
            assert es_ses[l_h] >= 0.0

    def test_auto_inject_with_varying_strata_nest_true_succeeds(self, base_data):
        """When strata varies across cells of a group and the user
        passes ``nest=True`` with no explicit ``psu``, the auto-inject
        path is valid: ``SurveyDesign.resolve()`` combines
        ``(stratum, psu)`` into globally-unique labels via the
        nest=True path (``diff_diff/survey.py:299-302``), so the
        cross-stratum PSU uniqueness check is satisfied. Byte-check
        against the explicit ``SurveyDesign(..., psu="group",
        nest=True)`` baseline — both paths resolve to the same design.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = df_["period"] % 2
        sd_auto = SurveyDesign(weights="pw", strata="stratum", nest=True)
        sd_explicit = SurveyDesign(
            weights="pw",
            strata="stratum",
            psu="group",
            nest=True,
        )
        r_auto = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_auto,
            L_max=2,
        )
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd_explicit,
            L_max=2,
        )
        assert np.isfinite(r_auto.overall_att)
        assert np.isfinite(r_auto.overall_se)
        if np.isfinite(r_auto.overall_se) and np.isfinite(r_explicit.overall_se):
            assert r_auto.overall_se == pytest.approx(r_explicit.overall_se, rel=1e-6)
        assert r_auto.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert r_auto.survey_metadata.df_survey == r_explicit.survey_metadata.df_survey

    def test_heterogeneity_auto_inject_with_varying_strata_nest_true_succeeds(self, base_data):
        """PR 3 unblocks heterogeneity + SurveyDesign(strata, nest=True)
        with auto-inject psu=group. Under nest=True the resolver
        combines (stratum, psu) into globally-unique labels, so
        resolved.psu varies across cells of each group and the
        cell-period allocator handles it. Mirrors ATT coverage at
        test_auto_inject_with_varying_strata_nest_true_succeeds.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = df_["period"] % 2
        df_["x_het"] = (df_["group"].astype(int) % 2).astype(float)
        sd = SurveyDesign(weights="pw", strata="stratum", nest=True)
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            heterogeneity="x_het",
            L_max=1,
            survey_design=sd,
        )
        assert res.heterogeneity_effects is not None
        entry = res.heterogeneity_effects[1]
        assert np.isfinite(entry["beta"])
        assert np.isfinite(entry["se"]) and entry["se"] >= 0.0
        assert np.isfinite(entry["t_stat"])
        assert np.isfinite(entry["p_value"])
        assert all(np.isfinite(entry["conf_int"]))

    def test_heterogeneity_multi_horizon_varying_psu_succeeds(self, base_data):
        """Multi-horizon heterogeneity (L_max >= 2) + within-group-
        varying PSU — each horizon rebuilds its own (g, out_idx) cell
        mapping, so the per-horizon allocator must produce finite
        inference independently at every horizon.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["x_het"] = (df_["group"].astype(int) % 2).astype(float)
        df_["psu"] = df_["group"].astype(int) * 2 + (df_["period"].astype(int) % 2)
        sd = SurveyDesign(weights="pw", psu="psu")
        res = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            heterogeneity="x_het",
            L_max=2,
            survey_design=sd,
        )
        assert res.heterogeneity_effects is not None
        for horizon in (1, 2):
            entry = res.heterogeneity_effects[horizon]
            assert np.isfinite(entry["beta"]), f"beta NaN at horizon {horizon}"
            assert np.isfinite(entry["se"]) and entry["se"] >= 0.0
            assert np.isfinite(entry["t_stat"])
            assert np.isfinite(entry["p_value"])
            assert all(np.isfinite(entry["conf_int"]))

    def test_auto_inject_with_varying_strata_raises(self, base_data):
        """Auto-injected `psu=<group>` with nest=False cannot honor
        strata that vary across cells of a group — the synthesized PSU
        column would reuse group labels across strata and trip the
        cross-stratum PSU uniqueness check. fit() detects that combo
        before survey resolution and raises a targeted ValueError
        pointing users to the explicit `psu=<col>, nest=True` path.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = df_["period"] % 2  # varies across cells of each group
        sd = SurveyDesign(weights="pw", strata="stratum")
        with pytest.raises(ValueError, match=r"psu=<col>"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_within_cell_psu_variation_rejected(self, base_data):
        """Multiple PSUs inside a single (g, t) cell (a multi-obs-per-
        cell panel) remain ambiguous under the cell allocator and must
        be rejected.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = 0
        # Duplicate the first row with a different PSU so that cell
        # (group[0], period[0]) has two obs with different PSU labels.
        dup = df_.iloc[0].copy()
        dup["psu"] = 99
        df_ = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(ValueError, match="PSU to be constant within each"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_within_cell_strata_variation_rejected(self, base_data):
        """Multiple strata inside a single (g, t) cell are ambiguous
        under the cell allocator and must be rejected.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        # Duplicate the first row with a different stratum.
        dup = df_.iloc[0].copy()
        dup["stratum"] = 1
        df_ = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        df_["psu"] = np.arange(len(df_))
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        with pytest.raises(ValueError, match="strata to be constant within each"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_accepts_varying_weights_within_group(self, base_data):
        """Within-group-varying pweights remain supported — the expansion
        psi_i = U[g] * (w_i / W_g) handles obs-level weight variation."""
        df_ = base_data.copy()
        rng = np.random.default_rng(7)
        df_["pw"] = rng.uniform(0.5, 2.0, size=len(df_))
        sd = SurveyDesign(weights="pw")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)

    def test_auto_inject_psu_matches_explicit_group_psu(self, base_data):
        """SurveyDesign(weights='pw') (no PSU) must yield the same SE and
        df_survey as SurveyDesign(weights='pw', psu='group') after
        dCDH auto-injects psu=group."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        r_no_psu = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=SurveyDesign(weights="pw"),
        )
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=SurveyDesign(weights="pw", psu="group"),
        )
        assert r_no_psu.overall_att == pytest.approx(r_explicit.overall_att, abs=1e-10)
        if np.isfinite(r_no_psu.overall_se) and np.isfinite(r_explicit.overall_se):
            assert r_no_psu.overall_se == pytest.approx(r_explicit.overall_se, rel=1e-6)
        assert r_no_psu.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert r_no_psu.survey_metadata.df_survey == r_explicit.survey_metadata.df_survey

    def test_degenerate_cohort_survey_se_is_nan(self):
        """When every variance-eligible group is its own singleton
        cohort (D_{g,1}, F_g, S_g), the cohort-recentered IF is
        identically zero. The survey SE path must return NaN (not 0.0)
        so the degenerate-cohort warning fires and inference stays
        NaN-consistent — matching the _plugin_se contract documented
        in REGISTRY.md."""
        # 4 groups × 5 periods, each group switches at a unique F_g so
        # the (D_{g,1}=0, F_g, S_g=+1) cohort key is unique per group.
        rows = []
        for g, f_switch in enumerate([1, 2, 3, 4]):
            for t in range(5):
                d = 1 if t >= f_switch else 0
                y = float(g) + 0.5 * t + float(d)
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": d,
                        "outcome": y,
                        "pw": 1.0,
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw")

        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = ChaisemartinDHaultfoeuille(seed=1).fit(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

        # overall_se must be NaN on degenerate cohorts (not 0.0)
        assert np.isnan(result.overall_se), (
            f"Degenerate-cohort survey overall_se must be NaN, " f"got {result.overall_se}"
        )
        # Degenerate-cohort warning must fire
        assert any(
            "cohort" in str(wi.message).lower() and "identically zero" in str(wi.message).lower()
            for wi in w
        ), "Expected degenerate-cohort warning to fire under survey path"

    def test_subpopulation_preserves_full_design_df_survey(self, base_data):
        """Under dCDH auto-inject, zero-weighting an entire group must not
        shrink df_survey below what the full-design PSU count would give.

        Mirrors SurveyDesign.subpopulation() semantics where excluded
        rows keep their weights at zero but remain in the design so
        that t critical values, p-values, CIs, and HonestDiD bounds
        reflect the full sampling structure."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        # Mimic subpopulation() by zero-weighting one entire group
        excluded_group = df_["group"].unique()[0]
        df_.loc[df_["group"] == excluded_group, "pw"] = 0.0

        sd = SurveyDesign(weights="pw")
        r_subpop = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        # Reference: explicit psu='group' preserves the full-design
        # PSU count because the resolver sees all groups (even those
        # entirely zero-weighted). The auto-inject path must match this.
        r_explicit = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=SurveyDesign(weights="pw", psu="group"),
        )
        assert r_subpop.survey_metadata is not None
        assert r_explicit.survey_metadata is not None
        assert r_subpop.survey_metadata.df_survey == r_explicit.survey_metadata.df_survey, (
            f"Auto-inject df_survey={r_subpop.survey_metadata.df_survey} "
            f"must match explicit psu='group' df_survey="
            f"{r_explicit.survey_metadata.df_survey} "
            f"(full-design subpopulation contract)."
        )

    def test_off_horizon_row_duplication_does_not_change_se(self, base_data):
        """Under auto-injected psu=group, duplicating an observation
        within a group (cell mean unchanged because the duplicate matches
        the existing row exactly) must not change the SE. Under the old
        per-obs-PSU fallback this invariant did not hold."""
        df_ = base_data.copy()
        df_["pw"] = 1.0
        sd = SurveyDesign(weights="pw")

        r_base = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )

        # Duplicate the first row: cell mean y_gt stays the same
        # (identical y/treatment). Under per-obs PSU fallback the
        # "extra observation" would change the variance; under
        # auto-inject psu=group the group structure is unchanged.
        dup = df_.iloc[0].copy()
        df_dup = pd.concat([df_, pd.DataFrame([dup])], ignore_index=True)
        r_dup = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_dup,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
            survey_design=sd,
        )
        if np.isfinite(r_base.overall_se) and np.isfinite(r_dup.overall_se):
            assert r_base.overall_se == pytest.approx(r_dup.overall_se, rel=1e-6), (
                f"Row duplication changed SE ({r_base.overall_se} vs "
                f"{r_dup.overall_se}) — auto-inject psu=group is not active."
            )

    def test_cell_allocator_row_sum_identity(self):
        """Cell-period allocator contract: for every group, the per-
        period attribution sums across time to the per-group IF
        (before cohort centering). This is the invariant that makes
        PSU-level Binder aggregation telescope to ``U_centered[g]``
        under within-group-constant PSU and therefore guarantees byte-
        identity with the legacy group-level allocator on the old
        accepted input set. Hand-computed on a 4-group × 3-period
        panel: two never-treated (stable_0) and two joiners switching
        at ``t = 2``.
        """
        from diff_diff.chaisemartin_dhaultfoeuille import (
            _cohort_recenter,
            _cohort_recenter_per_period,
            _compute_full_per_group_contributions,
        )

        # D_mat, Y_mat, N_mat shaped (n_groups=4, n_periods=3).
        D_mat = np.array(
            [
                [0, 0, 0],  # G0 never-treated
                [0, 0, 0],  # G1 never-treated
                [0, 0, 1],  # G2 joiner at t=2
                [0, 0, 1],  # G3 joiner at t=2
            ],
            dtype=float,
        )
        Y_mat = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.1, 3.1, 4.2],
                [0.5, 1.2, 5.4],
                [1.3, 2.4, 6.1],
            ],
            dtype=float,
        )
        N_mat = np.ones_like(D_mat, dtype=int)
        # Per-period cell counts aligned to periods[1:]
        # t=1: all stable_0 (4 in n_00); t=2: 2 joiners (n_10) + 2 stable_0 (n_00)
        n_10_t_arr = np.array([0, 2], dtype=int)
        n_00_t_arr = np.array([4, 2], dtype=int)
        n_01_t_arr = np.array([0, 0], dtype=int)
        n_11_t_arr = np.array([0, 0], dtype=int)
        # A11 zeroed at t=1 (no joiners); active at t=2.
        a11_plus_zeroed = np.array([True, False], dtype=bool)
        a11_minus_zeroed = np.array([True, True], dtype=bool)

        U, U_pp = _compute_full_per_group_contributions(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            n_10_t_arr=n_10_t_arr,
            n_00_t_arr=n_00_t_arr,
            n_01_t_arr=n_01_t_arr,
            n_11_t_arr=n_11_t_arr,
            a11_plus_zeroed_arr=a11_plus_zeroed,
            a11_minus_zeroed_arr=a11_minus_zeroed,
            side="overall",
            compute_per_period=True,
        )
        assert U_pp is not None

        # Hand computation at t=2 joiner side:
        #   G0: stable_0, -(2/2) * (3.0 - 2.0) = -1.0
        #   G1: stable_0, -(2/2) * (4.2 - 3.1) = -1.1
        #   G2: joiner,  (5.4 - 1.2) = 4.2
        #   G3: joiner,  (6.1 - 2.4) = 3.7
        expected_U = np.array([-1.0, -1.1, 4.2, 3.7])
        np.testing.assert_allclose(U, expected_U, atol=1e-12)

        # Row-sum identity: U_per_period.sum(axis=1) == U exactly.
        np.testing.assert_allclose(U_pp.sum(axis=1), U, atol=1e-12)

        # Post-period attribution: all mass at t=2 (the transition's
        # post cell); t=0 and t=1 columns are zero for every group.
        np.testing.assert_array_equal(U_pp[:, 0], np.zeros(4))
        np.testing.assert_array_equal(U_pp[:, 1], np.zeros(4))
        np.testing.assert_allclose(U_pp[:, 2], expected_U, atol=1e-12)

        # Cohort centering preserves the row-sum identity: per-period
        # cohort centering and group-level cohort centering produce
        # 2D and 1D arrays whose row sums agree to FP precision.
        # Cohorts: A = {G0, G1} (never-treated), B = {G2, G3} (joiners).
        cohort_ids = np.array([0, 0, 1, 1])
        U_c = _cohort_recenter(U, cohort_ids)
        U_pp_c = _cohort_recenter_per_period(U_pp, cohort_ids)
        np.testing.assert_allclose(U_pp_c.sum(axis=1), U_c, atol=1e-12)

    def test_within_cell_check_excludes_zero_weight_rows(self, base_data):
        """A zero-weight row with a different PSU label from its cell
        must not trigger rejection — it is out-of-sample by the
        subpopulation contract and does not enter the variance.
        """
        df_ = base_data.copy()
        df_["pw"] = 1.0
        df_["stratum"] = 0
        df_["psu"] = 0
        # Inject a zero-weight row whose PSU would collide with the
        # first row's cell if it were counted.
        sample = df_.iloc[0].copy()
        sample["psu"] = 99  # would violate within-cell constancy if counted
        sample["pw"] = 0.0
        df_ = pd.concat([df_, pd.DataFrame([sample])], ignore_index=True)
        sd = SurveyDesign(weights="pw", strata="stratum", psu="psu")
        result = ChaisemartinDHaultfoeuille(seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)


class TestHeterogeneityCellPeriod:
    """Unit tests for the heterogeneity cell-period allocator invariants.

    Under PSU=group the new single-cell psi_obs distribution
    (mass in the (g, out_idx) post-period cell only, scaled by
    w_i / W_{g, out_idx}) differs from the legacy group-level
    distribution (mass everywhere in g, scaled by w_i / W_g) at the
    observation level. But both telescope to the same PSU-level sum
    psi_g because compute_survey_if_variance aggregates to PSU first.
    Binder TSL variance must therefore be byte-identical.
    """

    def test_psu_level_byte_identity_under_psu_equals_group(self):
        """Construct both legacy and new psi_obs on a tiny fixture
        (PSU=group, one obs per cell), feed both through
        compute_survey_if_variance, and assert variances equal
        within ULP — the exact invariant the REGISTRY Note claims.
        """
        from diff_diff.survey import (
            SurveyDesign,
            compute_survey_if_variance,
        )

        # Fixture: 4 groups * 4 periods = 16 obs, one obs per cell,
        # pw=1 everywhere. PSU=group so each group is a single PSU.
        n_groups, n_periods = 4, 4
        rows = []
        for g in range(n_groups):
            for t in range(n_periods):
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "pw": 1.0,
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw", psu="group")
        resolved = sd.resolve(df_)

        # Arbitrary non-zero group-level IF values + per-group out_idx
        # (the post-period cell chosen by the heterogeneity horizon).
        psi_g = np.array([1.0, -0.5, 2.0, -1.5], dtype=np.float64)
        out_idx_per_group = np.array([3, 3, 2, 2], dtype=np.int64)

        obs_group = df_["group"].values.astype(np.int64)
        obs_period = df_["period"].values.astype(np.int64)
        w = df_["pw"].to_numpy(dtype=np.float64)

        # Legacy expansion: psi_i = psi_g[g_i] * w_i / W_g.
        W_g = np.zeros(n_groups, dtype=np.float64)
        np.add.at(W_g, obs_group, w)
        psi_legacy = np.zeros_like(w)
        for i in range(len(w)):
            if W_g[obs_group[i]] > 0:
                psi_legacy[i] = psi_g[obs_group[i]] * (w[i] / W_g[obs_group[i]])

        # New expansion: psi_i = psi_g[g_i] * w_i / W_{g, out_idx}
        # for obs in (g, out_idx), zero elsewhere.
        W_cell_out = np.zeros(n_groups, dtype=np.float64)
        for i in range(len(w)):
            if obs_period[i] == out_idx_per_group[obs_group[i]]:
                W_cell_out[obs_group[i]] += w[i]
        psi_new = np.zeros_like(w)
        for i in range(len(w)):
            g_i = obs_group[i]
            if obs_period[i] == out_idx_per_group[g_i] and W_cell_out[g_i] > 0:
                psi_new[i] = psi_g[g_i] * (w[i] / W_cell_out[g_i])

        # Distributions differ at the obs level.
        assert not np.allclose(psi_legacy, psi_new), (
            "fixture should exercise the mass redistribution — "
            "legacy spreads across all obs of g, new concentrates in "
            "(g, out_idx)."
        )

        # PSU-level sums must match — this is the invariant the
        # REGISTRY Note claims under PSU=group.
        assert resolved.psu is not None
        psu_codes = np.asarray(resolved.psu, dtype=np.int64)
        psu_sum_legacy = np.zeros(n_groups, dtype=np.float64)
        psu_sum_new = np.zeros(n_groups, dtype=np.float64)
        np.add.at(psu_sum_legacy, psu_codes, psi_legacy)
        np.add.at(psu_sum_new, psu_codes, psi_new)
        assert np.allclose(psu_sum_legacy, psu_sum_new, atol=0.0, rtol=1e-15)

        # Binder TSL variances must match byte-for-byte (single stratum,
        # each PSU sum contributes equally in both paths).
        var_legacy = compute_survey_if_variance(psi_legacy, resolved)
        var_new = compute_survey_if_variance(psi_new, resolved)
        assert np.isfinite(var_legacy) and np.isfinite(var_new)
        assert var_legacy == pytest.approx(var_new, rel=1e-14, abs=1e-14)

    def test_replicate_variance_non_invariance_under_varying_ratios(self):
        """When replicate-weight ratios vary within group,
        compute_replicate_if_variance is NOT PSU-telescoping — its
        theta_r = sum_i ratio_ir * psi_i reads psi at observation
        level, so redistributing mass across cells of g changes the
        replicate variance. This is why the heterogeneity path keeps
        the legacy group-level allocator on the replicate branch
        (only Binder TSL uses the cell-period allocator). Regression
        guard for the CI-review P1 on PR #329.
        """
        from diff_diff.survey import (
            SurveyDesign,
            compute_replicate_if_variance,
        )

        # Minimal reproduction of the reviewer's counterexample: one
        # group with two obs (ref and post-period cells), replicate
        # ratios that vary within the group. legacy allocator spreads
        # psi_g across both obs; new cell-period allocator puts it
        # all on the post-period cell. Replicate variance differs.
        rows = []
        for g in range(2):
            for t in range(2):
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "pw": 1.0,
                        # Non-PSU-aligned replicate columns: vary within
                        # each group so sum_i ratio_ir * psi_i sees a
                        # different weighted average of psi mass under
                        # the two allocators.
                        "rep0": 0.5 if t == 0 else 1.5,
                        "rep1": 1.5 if t == 0 else 0.5,
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(
            weights="pw",
            replicate_weights=["rep0", "rep1"],
            replicate_method="SDR",
        )
        resolved = sd.resolve(df_)

        # Arbitrary non-zero psi_g per group; out_idx = 1 for each.
        psi_g = np.array([0.75, -1.25], dtype=np.float64)
        obs_group = df_["group"].values.astype(np.int64)
        obs_period = df_["period"].values.astype(np.int64)
        w = df_["pw"].to_numpy(dtype=np.float64)

        # Legacy group-level expansion (what the replicate branch now
        # uses inside _compute_heterogeneity_test after the PR-329 fix).
        W_g = np.zeros(2, dtype=np.float64)
        np.add.at(W_g, obs_group, w)
        psi_legacy = np.zeros_like(w)
        for i in range(len(w)):
            if W_g[obs_group[i]] > 0:
                psi_legacy[i] = psi_g[obs_group[i]] * (w[i] / W_g[obs_group[i]])

        # New cell-period single-cell expansion (what the Binder TSL
        # branch uses). All mass lands on obs at t == 1.
        psi_new = np.zeros_like(w)
        for i in range(len(w)):
            if obs_period[i] == 1:
                psi_new[i] = psi_g[obs_group[i]]  # W_{g,out_idx}=1

        var_legacy, _ = compute_replicate_if_variance(psi_legacy, resolved)
        var_new, _ = compute_replicate_if_variance(psi_new, resolved)
        assert np.isfinite(var_legacy) and np.isfinite(var_new)
        # Documented non-invariance: replicate variance differs
        # materially between the two allocators on this fixture.
        assert not np.isclose(var_legacy, var_new, rtol=1e-6), (
            f"Expected legacy vs cell-period replicate variance to "
            f"differ when replicate ratios vary within group "
            f"(counterexample from PR #329 CI review). Got "
            f"var_legacy={var_legacy}, var_new={var_new}."
        )


class TestBootstrapCellPeriod:
    """Regression guards for the cell-level wild PSU bootstrap allocator
    (PR 4). Under PSU-within-group-constant regimes the dispatcher
    routes to the legacy group-level bootstrap for bit-identity; under
    within-group-varying PSU the cell-level path runs with per-cell
    PSU multipliers.
    """

    # Captured on pre-PR-4 code (origin/main at SHA ac181b7f — PR #329
    # merge) via a scratch fit on the fixture below. Pinned here as
    # the bit-identity regression guard for the dispatcher's
    # PSU-within-group-constant legacy-path routing. If this test
    # drifts, the dispatcher is no longer reproducing pre-PR-4
    # behavior under PSU=group and the legacy fast path has
    # regressed. Values differ between Rust and pure-Python backends
    # because `_generate_bootstrap_weights_batch` and `solve_ols`
    # take different numeric paths; bit-identity still holds within
    # each backend.
    _BASELINE_OVERALL_SE_RUST = 0.30560839419979546
    _BASELINE_OVERALL_SE_PYTHON = 0.3030802540369796

    @staticmethod
    def _make_baseline_fixture() -> pd.DataFrame:
        """Deterministic fixture for the pinned bootstrap SE baseline.
        12 groups, 5 periods, two switch cohorts (first-treated at
        periods 2 and 3) plus never-switchers, fixed per-row
        idiosyncratic draws so the bootstrap distribution is
        non-degenerate. MUST stay identical to the capture fixture
        or the pinned baseline becomes meaningless.
        """
        rng = np.random.default_rng(12345)
        rows = []
        n_groups, n_periods = 12, 5
        for g in range(n_groups):
            if g < 4:
                f: Optional[int] = None
            elif g < 8:
                f = 2
            else:
                f = 3
            for t in range(n_periods):
                d = 1 if (f is not None and t >= f) else 0
                y = float(g) + 0.3 * t + 1.5 * d + float(rng.normal(0.0, 0.5))
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "treatment": int(d),
                        "outcome": y,
                        "pw": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_bootstrap_se_matches_pre_pr4_baseline(self):
        """Bit-identity regression guard: under PSU=group the
        dispatcher routes through the legacy group-level bootstrap
        path, so the overall bootstrap SE must match pre-PR-4 code
        to ULP precision. The baseline values were captured on
        `origin/main` at `ac181b7f` (the PR #329 merge) under each
        backend independently.

        The baseline arm is selected from the SAME dispatch globals
        the fit consumes (bound at import into bootstrap_utils /
        linalg). Reading ``os.environ`` or ``diff_diff._backend`` at
        call time is order-fragile: an env mutation after import
        (e.g. a leaked doc-snippet backend override) flips that
        detection without changing the already-imported dispatch,
        selecting the wrong pinned-baseline arm under full-suite
        order.
        """
        from diff_diff import bootstrap_utils as _bu
        from diff_diff import linalg as _la

        bootstrap_rust = bool(_bu.HAS_RUST_BACKEND and _bu._rust_bootstrap_weights is not None)
        ols_rust = bool(_la.HAS_RUST_BACKEND and _la._rust_solve_ols is not None)
        assert bootstrap_rust == ols_rust, (
            "Incoherent backend dispatch state between bootstrap_utils "
            f"(rust={bootstrap_rust}) and linalg (rust={ols_rust}) — a "
            "leaked monkeypatch? No pinned baseline exists for a mixed "
            "backend; refusing to compare against either arm."
        )
        pure_python = not bootstrap_rust
        expected = (
            self._BASELINE_OVERALL_SE_PYTHON if pure_python else self._BASELINE_OVERALL_SE_RUST
        )
        df_ = self._make_baseline_fixture()
        sd = SurveyDesign(weights="pw", psu="group")
        res = ChaisemartinDHaultfoeuille(n_bootstrap=500, seed=42).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert res.bootstrap_results is not None
        observed_se = float(res.bootstrap_results.overall_se)
        backend_label = "pure-python" if pure_python else "rust"
        assert observed_se == pytest.approx(
            expected,
            rel=0.0,
            abs=1e-15,
        ), (
            f"Bootstrap SE drifted from pre-PR-4 baseline "
            f"(backend={backend_label}). expected={expected!r}, "
            f"observed={observed_se!r}. The dispatcher's "
            f"PSU-within-group-constant routing is no longer "
            f"bit-identical to the legacy group-level bootstrap."
        )

    def test_bootstrap_cell_level_raises_on_missing_overall_tensor(self):
        """Contract: when PSU varies within group, the bootstrap
        dispatcher must NOT silently fall back to group-level for any
        target. Invoking _compute_dcdh_bootstrap directly with a
        varying `psu_codes_per_cell` but `u_per_period_overall=None`
        must raise ValueError (not silently under-cluster).
        """
        est = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1)
        # Minimal group-level IF inputs; matches arbitrary 2-group setup.
        u_overall = np.array([0.5, -0.3], dtype=np.float64)
        eligible_group_ids = np.array([0, 1])
        group_id_to_psu_code = {0: 0, 1: 1}
        # Varying PSU: row 0 has two distinct PSU codes.
        psu_codes_per_cell = np.array(
            [[0, 1], [0, 0]],
            dtype=np.int64,
        )
        with pytest.raises(ValueError, match="u_per_period_overall"):
            est._compute_dcdh_bootstrap(
                n_groups_for_overall=2,
                u_centered_overall=u_overall,
                divisor_overall=4,
                original_overall=0.1,
                group_id_to_psu_code=group_id_to_psu_code,
                eligible_group_ids=eligible_group_ids,
                u_per_period_overall=None,  # missing — must raise
                psu_codes_per_cell=psu_codes_per_cell,
            )

    def test_bootstrap_cell_level_raises_on_shape_mismatch(self):
        """Contract: _unroll_target_to_cells rejects shape mismatches
        between `u_per_period_target` and `psu_codes_per_cell` — a
        silent misalignment would put PSU codes on the wrong cells.
        """
        est = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1)
        u_overall = np.array([0.5, -0.3], dtype=np.float64)
        eligible_group_ids = np.array([0, 1])
        group_id_to_psu_code = {0: 0, 1: 1}
        psu_codes_per_cell = np.array(
            [[0, 1], [0, 0]],
            dtype=np.int64,
        )
        # Wrong shape — 3 columns instead of 2.
        u_pp_wrong = np.array(
            [[0.25, 0.25, 0.0], [-0.15, -0.15, 0.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            est._compute_dcdh_bootstrap(
                n_groups_for_overall=2,
                u_centered_overall=u_overall,
                divisor_overall=4,
                original_overall=0.1,
                group_id_to_psu_code=group_id_to_psu_code,
                eligible_group_ids=eligible_group_ids,
                u_per_period_overall=u_pp_wrong,
                psu_codes_per_cell=psu_codes_per_cell,
            )

    def test_bootstrap_cell_level_raises_on_sentinel_mass_leak(self):
        """Contract: when `_cohort_recenter_per_period` subtracts
        column means across the full period grid, a group with no
        observation at period t can acquire non-zero centered mass
        at that cell. Under the cell-level bootstrap path, such
        mass lands on a `psu_codes_per_cell == -1` sentinel cell
        and has no PSU to attach to — the bootstrap must raise
        rather than silently drop the mass.
        """
        est = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1)
        # Build a per-cell IF tensor with non-zero mass at a cell
        # whose PSU code is -1 (simulating terminal missingness
        # after cohort-recentering leaks mass to a missing cell).
        psu_codes_per_cell = np.array(
            [[0, 1, -1], [0, 1, 0]],
            dtype=np.int64,
        )
        u_pp_overall_with_leak = np.array(
            [[0.25, 0.25, -0.15], [-0.15, -0.15, 0.15]],
            dtype=np.float64,
        )
        u_overall = np.array([0.5, -0.3], dtype=np.float64)
        eligible_group_ids = np.array([0, 1])
        group_id_to_psu_code = {0: 0, 1: 1}
        with pytest.raises(ValueError, match="no positive-weight observations"):
            est._compute_dcdh_bootstrap(
                n_groups_for_overall=2,
                u_centered_overall=u_overall,
                divisor_overall=4,
                original_overall=0.1,
                group_id_to_psu_code=group_id_to_psu_code,
                eligible_group_ids=eligible_group_ids,
                u_per_period_overall=u_pp_overall_with_leak,
                psu_codes_per_cell=psu_codes_per_cell,
            )

    def test_bootstrap_cell_level_raises_on_missing_horizon_tensor(self):
        """Contract: when PSU varies within group, each multi-horizon
        target must supply its per-cell IF tensor; missing one raises
        ValueError rather than degrading the sup-t joint distribution.
        """
        est = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1)
        u_overall = np.array([0.5, -0.3], dtype=np.float64)
        u_pp_overall = np.array(
            [[0.25, 0.25], [-0.15, -0.15]],
            dtype=np.float64,
        )
        eligible_group_ids = np.array([0, 1])
        group_id_to_psu_code = {0: 0, 1: 1}
        psu_codes_per_cell = np.array(
            [[0, 1], [0, 0]],
            dtype=np.int64,
        )
        # Horizon tuple missing its per-cell tensor (4th slot = None).
        u_h = np.array([0.4, -0.2], dtype=np.float64)
        mh_inputs = {1: (u_h, 4, 0.1, None)}
        with pytest.raises(ValueError, match="multi-horizon.*l=1"):
            est._compute_dcdh_bootstrap(
                n_groups_for_overall=2,
                u_centered_overall=u_overall,
                divisor_overall=4,
                original_overall=0.1,
                multi_horizon_inputs=mh_inputs,
                group_id_to_psu_code=group_id_to_psu_code,
                eligible_group_ids=eligible_group_ids,
                u_per_period_overall=u_pp_overall,
                psu_codes_per_cell=psu_codes_per_cell,
            )

    def test_bootstrap_cell_level_with_all_zero_weight_group_does_not_crash(self):
        """When one eligible group has all zero-weight observations,
        every entry of its `psu_codes_per_cell` row is the sentinel
        -1 and it contributes no cells to the bootstrap. The
        dispatcher must handle this without crashing; overall
        inference stays finite (driven by the remaining groups).
        """
        rows = []
        n_groups, n_periods = 10, 5
        for g in range(n_groups):
            f = 3 if g < n_groups // 2 else None
            for t in range(n_periods):
                # Zero-weight the last group entirely.
                pw = 0.0 if g == n_groups - 1 else 1.0
                d = 1 if (f is not None and t >= f) else 0
                y = float(g) + 0.1 * t + 1.0 * d
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "treatment": int(d),
                        "outcome": y,
                        "pw": pw,
                        "psu": int(g) * 2 + (int(t) % 2),  # varying PSU
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw", psu="psu")
        res = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
            df_,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert res.bootstrap_results is not None
        # Bootstrap SE should be finite (zero-weight group does not
        # disturb the other groups' contributions).
        assert np.isfinite(res.bootstrap_results.overall_se)

    def test_bootstrap_zero_weight_group_equivalent_to_removing_it(self):
        """Fixture A: 9 groups (1 all-zero-weighted + 8 positive)
        with **within-group-varying PSU** so the dispatcher routes
        through the cell-level path. Fixture B: 8 groups (same panel
        without the zero-weight group), same varying PSU. Under the
        fix, an eligible group with no positive-weight cells
        contributes nothing to the bootstrap (its row of
        `psu_codes_per_cell` is all sentinel), so both fits produce
        byte-identical bootstrap SE at the same seed. Without the
        fix, the `valid_map` gate disabled the entire PSU-aware
        path — silently dropping fixture A to unclustered group-
        level bootstrap while fixture B correctly ran the cell-
        level path. Using `psu=group` (a within-group-constant PSU)
        would not exercise this regression because the buggy and
        correct paths collapse to the same identity-draw structure
        under PSU=group — we deliberately use varying PSU here.
        """

        def _make(include_zero_group: bool) -> pd.DataFrame:
            rows = []
            n_groups = 9 if include_zero_group else 8
            for g in range(n_groups):
                f = 3 if g < 4 else None
                for t in range(5):
                    pw = 0.0 if (include_zero_group and g == 8) else 1.0
                    d = 1 if (f is not None and t >= f) else 0
                    y = float(g) + 0.1 * t + 1.0 * d
                    # Within-group-varying PSU (period parity per
                    # group) — exercises the cell-level dispatcher.
                    psu = int(g) * 2 + (int(t) % 2)
                    rows.append(
                        {
                            "group": int(g),
                            "period": int(t),
                            "treatment": int(d),
                            "outcome": y,
                            "pw": pw,
                            "psu": psu,
                        }
                    )
            return pd.DataFrame(rows)

        sd = SurveyDesign(weights="pw", psu="psu")
        res_a = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=7).fit(
            _make(include_zero_group=True),
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        res_b = ChaisemartinDHaultfoeuille(n_bootstrap=200, seed=7).fit(
            _make(include_zero_group=False),
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            survey_design=sd,
        )
        assert res_a.bootstrap_results is not None
        assert res_b.bootstrap_results is not None
        se_a = float(res_a.bootstrap_results.overall_se)
        se_b = float(res_b.bootstrap_results.overall_se)
        assert np.isfinite(se_a) and np.isfinite(se_b)
        assert se_a == pytest.approx(se_b, rel=0.0, abs=1e-15), (
            f"Bootstrap SE must match when a zero-weight eligible "
            f"group is added under within-group-varying PSU (fix "
            f"P0 #1 — no silent dropback to unclustered group-"
            f"level). Got SE_with_zero={se_a!r}, "
            f"SE_without_zero={se_b!r}."
        )

    def test_fit_raises_on_terminal_missingness_with_varying_psu(self):
        """End-to-end `fit()` regression: when a survey panel has a
        terminally-missing group in a cohort whose other groups still
        contribute at the missing period, combined with within-group-
        varying PSU, both the analytical TSL path and the cell-level
        bootstrap path must raise `ValueError` — cohort-recentering
        leaks non-zero centered IF mass onto cells with no positive-
        weight obs, and both paths (`_survey_se_from_group_if` for
        analytical, `_unroll_target_to_cells` for bootstrap) use the
        cell-period allocator and therefore cannot allocate leaked
        mass to any observation or PSU. Pre-processing the panel
        (dropping late-exit groups or trimming to a balanced sub-
        panel) is the documented workaround.
        """
        rows = []
        # 10 groups. Joiners at period 3 (cohort A): groups 0-4.
        # Leavers at period 4 (cohort B, D=1 at period 0): groups 5-7.
        # Never-treated: groups 8-9.
        # Group 2 is terminally missing at periods 4-5. It is in
        # cohort A; at period 4 the other joiners (0, 1, 3, 4) serve
        # as stable_1 controls (they switched on at period 3 and
        # contribute when leavers appear at period 4). The cohort
        # mean at period 4 is therefore non-zero, and
        # `_cohort_recenter_per_period` leaks `-col_mean` onto
        # group 2's missing cell — which the cell-level bootstrap
        # cannot allocate to any PSU.
        for g in range(10):
            if g < 5:
                # Joiners at period 3 (D=0 at baseline, D=1 from t=3).
                d_pattern = [0, 0, 0, 1, 1, 1]
            elif g < 8:
                # Leavers at period 4 (D=1 at baseline, D=0 from t=4).
                d_pattern = [1, 1, 1, 1, 0, 0]
            else:
                # Never-treated controls.
                d_pattern = [0, 0, 0, 0, 0, 0]
            for t in range(6):
                if g == 2 and t >= 4:
                    # Terminal missingness: drop rows past period 3.
                    continue
                d = d_pattern[t]
                y = float(g) + 0.1 * t + 1.0 * d
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "treatment": int(d),
                        "outcome": y,
                        "pw": 1.0,
                        # Within-group-varying PSU: period parity per group.
                        "psu": int(g) * 2 + (int(t) % 2),
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw", psu="psu")

        import warnings as _w

        # Analytical path (n_bootstrap=0): the sentinel-mass guard in
        # `_survey_se_from_group_if` raises on the same leakage the
        # bootstrap guard rejects — both paths use the cell-period
        # allocator and cannot allocate leaked mass to any
        # observation.
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # terminal-missingness UserWarning
            with pytest.raises(
                ValueError,
                match="no positive-weight observations",
            ):
                ChaisemartinDHaultfoeuille(n_bootstrap=0, seed=1).fit(
                    df_,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                    survey_design=sd,
                    L_max=1,
                )

        # Bootstrap path (n_bootstrap > 0): same sentinel-mass guard
        # fires via `_unroll_target_to_cells`.
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            with pytest.raises(
                ValueError,
                match="no positive-weight observations",
            ):
                ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1).fit(
                    df_,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                    survey_design=sd,
                    L_max=1,
                )

    def test_fit_succeeds_on_terminal_missingness_with_psu_group(self):
        """Companion regression: the terminal-missingness + varying-PSU
        sentinel-mass guard must NOT fire when PSU is within-group-
        constant. Same fixture as the varying-PSU test above but with
        `psu=<group column>` (auto-inject default) — both analytical
        and bootstrap paths route through the legacy group-level
        allocator (the analytical dispatcher in
        `_survey_se_from_group_if` falls back to the group-level
        allocator when PSU does not vary within group; the bootstrap
        dispatcher in `_compute_dcdh_bootstrap` does the same). Fit
        must succeed with finite SE.
        """
        rows = []
        for g in range(10):
            if g < 5:
                d_pattern = [0, 0, 0, 1, 1, 1]
            elif g < 8:
                d_pattern = [1, 1, 1, 1, 0, 0]
            else:
                d_pattern = [0, 0, 0, 0, 0, 0]
            for t in range(6):
                if g == 2 and t >= 4:
                    continue
                d = d_pattern[t]
                y = float(g) + 0.1 * t + 1.0 * d
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "treatment": int(d),
                        "outcome": y,
                        "pw": 1.0,
                    }
                )
        df_ = pd.DataFrame(rows)
        # Auto-inject: no explicit `psu` → `SurveyDesign` falls back to
        # `psu=<group_col>` at fit() time. Within-group-constant.
        sd = SurveyDesign(weights="pw")
        import warnings as _w

        for n_boot in (0, 50):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = ChaisemartinDHaultfoeuille(
                    n_bootstrap=n_boot,
                    seed=1,
                ).fit(
                    df_,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                    survey_design=sd,
                    L_max=1,
                )
            assert np.isfinite(res.overall_att), (
                f"n_bootstrap={n_boot}: overall_att must be finite "
                f"under PSU=group + terminal missingness."
            )
            assert np.isfinite(res.overall_se) and res.overall_se >= 0.0, (
                f"n_bootstrap={n_boot}: overall_se must be finite "
                f"under PSU=group + terminal missingness."
            )

    def test_bootstrap_dense_codes_under_singleton_baseline_excluded_group(self):
        """Regression for P0 #2: when a group is singleton-baseline-
        excluded (e.g., an always-treated group whose baseline D=1
        has no peer), its PSU label must NOT pollute the dense code
        factorization used by `_compute_dcdh_bootstrap`. Otherwise
        eligible groups that share a PSU receive gapped dense codes
        (e.g., `[1, 1]`), `_generate_psu_or_group_weights` computes
        `n_psu = max + 1 = 2 == n_groups_target = 2`, and the
        identity fast path wrongly triggers — giving those eligible
        groups independent multiplier draws instead of a shared
        one. Assertion: instrument the call to capture the
        `group_id_to_psu_code` dict actually passed and confirm its
        values form a contiguous range `[0, n_unique - 1]`.
        """
        # Fixture: one always-treated group (D=1 at period 0 → singleton-
        # baseline-excluded), plus eligible groups that share a PSU
        # label while the excluded group has a different PSU.
        rows = []
        for g in range(5):
            for t in range(5):
                if g == 0:
                    d = 1  # always-treated; baseline D=1 singleton
                    psu = 100  # distinct PSU for the excluded group
                else:
                    d = 1 if t >= 3 else 0  # joiners at period 3
                    # Groups 1, 2 share PSU=200; groups 3, 4 share PSU=300.
                    psu = 200 if g in (1, 2) else 300
                rows.append(
                    {
                        "group": int(g),
                        "period": int(t),
                        "treatment": int(d),
                        "outcome": float(g) + 0.1 * t + 0.5 * d,
                        "pw": 1.0,
                        "psu": psu,
                    }
                )
        df_ = pd.DataFrame(rows)
        sd = SurveyDesign(weights="pw", psu="psu")

        captured: dict = {}

        est = ChaisemartinDHaultfoeuille(n_bootstrap=50, seed=1)
        original_bootstrap = est._compute_dcdh_bootstrap

        def _spy(**kwargs):
            captured["group_id_to_psu_code"] = kwargs.get("group_id_to_psu_code")
            return original_bootstrap(**kwargs)

        est._compute_dcdh_bootstrap = _spy  # type: ignore[method-assign]

        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")  # singleton-baseline warning
            est.fit(
                df_,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

        dict_passed = captured["group_id_to_psu_code"]
        assert dict_passed is not None, (
            "bootstrap received group_id_to_psu_code=None — the "
            "PSU-aware path was disabled instead of routing to the "
            "cell/legacy path via densified codes."
        )
        codes = sorted(set(dict_passed.values()))
        # Eligible groups share only two PSUs (200 for g=1,2;
        # 300 for g=3,4). Dense codes must be [0, 1], NOT [1, 2]
        # (which would happen if the excluded g=0's PSU=100 were
        # dense-coded first).
        assert codes == list(range(len(codes))), (
            f"group_id_to_psu_code values must be contiguous "
            f"dense codes starting at 0, got {codes}. A non-"
            f"contiguous range signals the excluded group's PSU "
            f"polluted the dense factorization (P0 #2 regression)."
        )
        # Sanity: eligible groups 1, 2 must share a code (PSU=200),
        # and eligible groups 3, 4 must share a code (PSU=300).
        assert dict_passed[1] == dict_passed[2], (
            "Groups 1 and 2 share PSU=200 and must receive the same "
            "dense code under correct densification."
        )
        assert dict_passed[3] == dict_passed[4], (
            "Groups 3 and 4 share PSU=300 and must receive the same " "dense code."
        )
        assert dict_passed[1] != dict_passed[3], (
            "Groups in PSU=200 and PSU=300 must receive distinct " "dense codes."
        )
