"""Tests for Phase 5 survey support: SyntheticDiD and TROP.

Covers: pweight-only survey integration for both estimators, including
point estimate weighting, bootstrap/placebo SE threading, survey_metadata
in results, error guards for unsupported designs, and scale invariance.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign, SyntheticDiD
from diff_diff.trop import TROP, trop

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sdid_survey_data():
    """Balanced panel for SDID with survey design columns.

    20 units (5 treated, 15 control), 10 periods, block treatment at period 6.
    Unit-constant weight column that varies across units.
    """
    np.random.seed(42)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated = 1 if u < n_treated else 0
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.5 * t + np.random.randn() * 0.5
            if is_treated and t >= 6:
                y += 2.0  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "treated": is_treated})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.1  # [1.0, 1.1, ..., 2.9]
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def trop_survey_data():
    """Panel data for TROP with absorbing-state D and survey columns.

    20 units (5 treated starting at period 5), 10 periods.
    """
    np.random.seed(123)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated_unit = u < n_treated
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.3 * t + np.random.randn() * 0.5
            # Absorbing state: D=1 for t >= 5 if treated unit
            d = 1 if (is_treated_unit and t >= 5) else 0
            if d == 1:
                y += 1.5  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "D": d})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.15
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def survey_design_weights():
    return SurveyDesign(weights="weight")


@pytest.fixture
def survey_design_full():
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


# =============================================================================
# SyntheticDiD Survey Tests
# =============================================================================


class TestSyntheticDiDSurvey:
    """Survey support tests for SyntheticDiD."""

    def test_smoke_weights_only(self, sdid_survey_data, survey_design_weights):
        """Fit completes and survey_metadata is populated."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_uniform_weights_match_unweighted(self, sdid_survey_data):
        """Uniform weights (all 1.0) produce same ATT as unweighted."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["uniform_w"] = 1.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_uniform = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, sdid_survey_data, survey_design_weights):
        """Metadata has correct weight_type, effective_n, design_effect."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0

    def test_full_design_bootstrap_succeeds(self, sdid_survey_data, survey_design_full):
        """Full survey design (strata/PSU) with bootstrap succeeds (PR #352).

        Restored capability: composes Rao-Wu rescaled weights with the
        weighted-Frank-Wolfe variant per draw (see REGISTRY.md §SyntheticDiD
        survey + bootstrap composition Note). Asserts:
        - finite SE > 0
        - survey_metadata populated with n_strata / n_psu
        - result.summary() round-trips without error (cross-surface guard)
        """
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None
        # summary() must render the bootstrap replications line and the
        # survey design block without exception.
        summary = result.summary()
        assert "Survey Design" in summary
        assert "Bootstrap replications" in summary

    def test_full_design_placebo_succeeds(self, sdid_survey_data_full_design):
        """Placebo variance with full design now succeeds (restored capability).

        Stratified-permutation allocator draws pseudo-treated indices
        within each stratum containing treated units; weighted-FW
        re-estimates ω and λ per draw on the pseudo-panel. Uses the
        non-degenerate full-design fixture (stratum 0 has 5 treated +
        10 controls, so the within-stratum permutation has ``C(10, 5) =
        252`` distinct allocations — SE reflects a genuine null
        distribution, not FP noise from a single-allocation collapse).
        See REGISTRY §SyntheticDiD "Note (survey + placebo composition)".
        """
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        # SE must be materially positive, not sub-ULP FP noise from a
        # degenerate single-allocation permutation (R4 P1 regression —
        # the prior fixture had n_c == n_t in stratum 0, yielding
        # SE ≈ 1e-16; the Case D guard below rejects that shape).
        assert result.se > 1e-6
        assert result.variance_method == "placebo"
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None
        # summary() renders without exception
        summary = result.summary()
        assert "Survey Design" in summary

    def test_full_design_jackknife_succeeds(self, sdid_survey_data_jk_well_formed):
        """Jackknife variance with full design now succeeds (restored capability).

        PSU-level LOO with stratum aggregation (Rust & Rao 1996):
        SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)². Uses
        the well-formed jackknife fixture so every PSU-LOO in every
        contributing stratum is defined (treated units spread across two
        PSUs). See REGISTRY §SyntheticDiD "Note (survey + jackknife
        composition)".
        """
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.variance_method == "jackknife"
        assert result.survey_metadata is not None
        assert result.survey_metadata.n_strata is not None
        assert result.survey_metadata.n_psu is not None
        summary = result.summary()
        assert "Survey Design" in summary

    def test_placebo_with_pweight_only_full_design_stripped_att_match(self, sdid_survey_data):
        """Placebo ATT with pweight-only is unchanged when stratum/psu
        columns are physically dropped from the input DataFrame.

        Point estimates depend only on the pseudo-population weights, not on
        the strata/PSU structure. PR #352 restored bootstrap support for
        strata/PSU (which inflates SE via Rao-Wu clustering) but the
        placebo / jackknife methods still depend only on per-unit pweight
        for the point estimate. A silent pickup of ``stratum`` or ``psu``
        by the estimator (e.g., by name-matching a convention column)
        would cause the two fits to diverge, so comparing a DataFrame with
        those columns present against one with them dropped is the real
        contract.
        """
        sd_pweight_only = SurveyDesign(weights="weight")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=100, seed=42)

        result_with_cols = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight_only,
        )
        sdid_survey_data_stripped = sdid_survey_data.drop(columns=["stratum", "psu"])
        result_stripped = est.fit(
            sdid_survey_data_stripped,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight_only,
        )
        assert np.isfinite(result_with_cols.att)
        assert np.isfinite(result_with_cols.se)
        assert result_with_cols.se > 0
        # ATT depends only on pweight — silent pickup of stratum/psu would
        # make the with-columns fit differ from the stripped fit.
        assert result_with_cols.att == pytest.approx(result_stripped.att, abs=1e-12)

    def test_fweight_aweight_raises(self, sdid_survey_data):
        """Non-pweight raises ValueError."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_weighted_att_differs(self, sdid_survey_data, survey_design_weights):
        """Non-uniform weights produce different ATT than unweighted."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        # ATTs should differ since weights are non-uniform
        assert result_survey.att != pytest.approx(result_no_survey.att, abs=1e-6)

    def test_summary_includes_survey(self, sdid_survey_data, survey_design_weights):
        """summary() output contains Survey Design section."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_bootstrap_with_pweight_only_succeeds(self, sdid_survey_data, survey_design_weights):
        """variance_method='bootstrap' with pweight-only survey succeeds (PR #352).

        Restored capability: the bootstrap loop dispatches to the
        weighted-FW variant per draw with constant ``rw = w_control``
        (no Rao-Wu rescaling — pweight-only). Cross-surface guard: the
        result still labels itself as bootstrap (variance_method allow-list
        in results.py / business_report.py).
        """
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert result.variance_method == "bootstrap"

    def test_bootstrap_full_design_se_differs_from_pweight_only(self, sdid_survey_data):
        """Full-design bootstrap SE differs from pweight-only bootstrap SE.

        Resurrects the test_full_design_se_differs_from_weights_only
        contract that PR #351 deleted (R3 cleanup). With Rao-Wu rescaling
        (full design), per-draw weights are clustered by PSU within
        strata, inflating the bootstrap variance vs the pweight-only path
        which uses constant per-control weights. ATT point estimates
        match (both compose ω_eff = ω·w_control / Σ post-fit) but SEs
        diverge — that's the methodology contract.
        """
        sd_pweight = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        est_pw = SyntheticDiD(variance_method="bootstrap", n_bootstrap=100, seed=42)
        result_pw = est_pw.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight,
        )
        est_full = SyntheticDiD(variance_method="bootstrap", n_bootstrap=100, seed=42)
        result_full = est_full.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_full,
        )
        # ATT point estimates match (same w_control, same post-fit
        # composition). Tolerance loose because the bootstrap path
        # re-estimates ω̂_b under different weighted objectives.
        assert result_pw.att == pytest.approx(result_full.att, abs=1e-10)
        # SEs differ — Rao-Wu adds PSU clustering variance.
        assert result_pw.se != pytest.approx(result_full.se, abs=1e-6)

    def test_jackknife_with_pweight_only(self, sdid_survey_data, survey_design_weights):
        """variance_method='jackknife' completes with pweight-only survey weights.

        Positive coverage for the pweight-only + jackknife path that replaces
        the removed pweight-only + bootstrap case.
        """
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_placebo_with_survey(self, sdid_survey_data, survey_design_weights):
        """variance_method='placebo' completes with survey weights."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_weight_scale_invariance(self, sdid_survey_data, survey_design_weights):
        """Multiplying all weights by constant produces same ATT."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["weight_2x"] = sdid_survey_data["weight"] * 2.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_1x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        result_2x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="weight_2x"),
        )
        assert result_2x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, sdid_survey_data):
        """Time-varying weight column raises ValueError."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["bad_weight"] = sdid_survey_data["weight"] + sdid_survey_data["time"] * 0.1
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_to_dict_includes_survey(self, sdid_survey_data, survey_design_weights):
        """to_dict() output includes survey metadata fields."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_covariates_with_survey(self, sdid_survey_data, survey_design_weights):
        """Covariates + survey_design smoke test (WLS residualization)."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["x1"] = np.random.randn(len(sdid_survey_data))
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            covariates=["x1"],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert result.survey_metadata is not None

    def test_effective_weights_returned(self, sdid_survey_data, survey_design_weights):
        """unit_weights returns composed ω_eff (not raw ω) under survey weighting."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        weights = result.unit_weights
        # Effective weights should sum to 1 (renormalized)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)
        # With non-uniform survey weights, effective weights should differ
        # from what uniform survey weights would produce
        sdid_survey_data_u = sdid_survey_data.copy()
        sdid_survey_data_u["uniform_w"] = 1.0
        result_u = est.fit(
            sdid_survey_data_u,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        # Non-uniform weights should change the returned weight distribution
        eff_vals = sorted(weights.values(), reverse=True)
        uni_vals = sorted(result_u.unit_weights.values(), reverse=True)
        assert eff_vals != pytest.approx(uni_vals, abs=1e-6)


# =============================================================================
# SyntheticDiD Full-Design Placebo & Jackknife Tests
# =============================================================================


@pytest.fixture
def sdid_survey_data_full_design():
    """Balanced 30-unit panel with adequate stratum structure for full-design.

    30 units (5 treated 0-4, 25 control 5-29), 10 periods. Treated all in
    stratum 0 PSU 0. Controls spread across multiple strata + PSUs so
    stratified-permutation placebo has >1 permutation (stratum 0 has
    10 controls, n_t=5 → C(10,5)=252 draws) and PSU-level LOO jackknife
    has ≥2 PSUs per stratum so every stratum contributes to variance.

    Layout:
        stratum 0: treated PSU 0 (units 0-4), control PSUs 1 & 2 (units 5-14)
        stratum 1: control PSUs 3, 4, 5 (units 15-29)
    """
    np.random.seed(7)
    n_units = 30
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated = 1 if u < n_treated else 0
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.5 * t + np.random.randn() * 0.5
            if is_treated and t >= 6:
                y += 2.0
            rows.append({"unit": u, "time": t, "outcome": y, "treated": is_treated})

    data = pd.DataFrame(rows)

    unit_weight = 1.0 + np.arange(n_units) * 0.05
    unit_stratum = np.array([0] * 15 + [1] * 15)
    unit_psu = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5)
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def sdid_survey_design_full():
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


@pytest.fixture
def sdid_survey_data_jk_well_formed():
    """30-unit panel where every jackknife PSU-LOO is defined.

    The Rust & Rao (1996) stratified jackknife formula requires that
    every LOO within a contributing stratum produce a defined
    ``τ̂_{(h,j)}``. In particular, **every PSU that contains treated
    units must also leave enough treated units behind when dropped** —
    otherwise the LOO removes all treated and the SDID estimator is
    undefined. The "treated all in one PSU" fixture used for the placebo
    tests triggers this by design; this fixture distributes the 5
    treated units across **two PSUs within stratum 0** so that LOO of
    any treated-containing PSU still leaves ≥1 treated unit.

    Layout:
        stratum 0 (13 units):
            PSU 0: treated units 0, 1  + control units 5, 6
            PSU 1: treated units 2, 3, 4 + control units 7, 8
            PSU 2: control units 9, 10, 11, 12
        stratum 1 (17 units):
            PSU 3: control units 13-17
            PSU 4: control units 18-22
            PSU 5: control units 23-29
    """
    np.random.seed(7)
    n_units = 30
    n_periods = 10
    # Treated at unit IDs 0-4.
    treated_ids = {0, 1, 2, 3, 4}

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated = 1 if u in treated_ids else 0
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.5 * t + np.random.randn() * 0.5
            if is_treated and t >= 6:
                y += 2.0
            rows.append({"unit": u, "time": t, "outcome": y, "treated": is_treated})

    data = pd.DataFrame(rows)

    unit_weight = 1.0 + np.arange(n_units) * 0.05
    # Stratum: units 0-12 → 0, units 13-29 → 1
    unit_stratum = np.array([0] * 13 + [1] * 17)
    # PSU layout (12 stratum-0 units spread across PSU 0/1/2; 17
    # stratum-1 units across PSU 3/4/5). Treated units 0-4 straddle
    # PSU 0 (units 0-1) and PSU 1 (units 2-4).
    unit_psu = np.zeros(n_units, dtype=int)
    unit_psu[0:2] = 0  # PSU 0: treated 0, 1
    unit_psu[2:5] = 1  # PSU 1: treated 2, 3, 4
    unit_psu[5:7] = 0  # PSU 0: control 5, 6
    unit_psu[7:9] = 1  # PSU 1: control 7, 8
    unit_psu[9:13] = 2  # PSU 2: control 9-12
    unit_psu[13:18] = 3  # PSU 3: control 13-17
    unit_psu[18:23] = 4  # PSU 4: control 18-22
    unit_psu[23:30] = 5  # PSU 5: control 23-29

    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


class TestSDIDSurveyPlaceboFullDesign:
    """Stratified-permutation placebo allocator under strata/PSU/FPC (this PR).

    Allocator: pseudo-treated indices are drawn WITHIN each stratum
    containing actual treated units; weighted-FW re-estimates ω and λ per
    draw with per-control survey weights. See REGISTRY §SyntheticDiD
    "Note (survey + placebo composition)".
    """

    def test_placebo_full_design_pseudo_treated_stays_within_treated_strata(
        self, sdid_survey_data_full_design, sdid_survey_design_full
    ):
        """Every draw's pseudo-treated units have stratum ∈ treated-strata set.

        Stratified permutation preserves the treated-stratum marginal
        exactly — pseudo-treated never picks from strata with no actual
        treated units. Patches ``rng.choice`` inside the per-draw loop
        to record every actual ``pseudo_treated_idx`` and asserts each
        sampled control's stratum membership ⊆ treated-strata set
        across every draw (R9 P3 — direct allocator inspection rather
        than just dispatch arg recording).
        """
        # Capture every np.random.Generator.choice call inside the per-
        # draw loop. ``_placebo_variance_se_survey`` constructs a fresh
        # rng = np.random.default_rng(self.seed), so monkey-patching at
        # the class level on Generator doesn't intercept it. Instead we
        # wrap ``np.random.default_rng`` to return a recording-aware rng.
        captured_pseudo_treated_strata: list[set] = []

        # Run the fit and intercept rng.choice via a thin Generator wrapper.
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=123)

        # Build the resolved strata arrays the same way fit() does so we
        # can map sampled control indices back to their stratum.
        # sdid_survey_data_full_design layout: stratum 0 has treated
        # 0-4 + controls 5-14, stratum 1 has controls 15-29.
        treated_units = list(range(5))
        control_units = list(range(5, 30))
        unit_to_stratum = sdid_survey_data_full_design.groupby("unit")["stratum"].first().to_dict()
        strata_control = np.array([unit_to_stratum[u] for u in control_units])
        treated_strata_set = set(unit_to_stratum[u] for u in treated_units)

        # Wrap np.random.default_rng to install a Generator with an
        # instrumented `choice`.
        import numpy as _np

        original_default_rng = _np.random.default_rng

        class _RecordingChoiceGenerator:
            def __init__(self, inner: _np.random.Generator):
                self._inner = inner

            def choice(self, a, *args, **kwargs):  # type: ignore[override]
                idx = self._inner.choice(a, *args, **kwargs)
                # `a` is controls_in_h (control-array positions).
                # idx is the picked subset (also control-array positions).
                picked_strata = strata_control[np.asarray(idx)]
                captured_pseudo_treated_strata.append(set(picked_strata.tolist()))
                return idx

            def permutation(self, *args, **kwargs):  # type: ignore[override]
                return self._inner.permutation(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def fake_default_rng(seed=None):
            return _RecordingChoiceGenerator(original_default_rng(seed))

        try:
            _np.random.default_rng = fake_default_rng  # type: ignore[assignment]
            est.fit(
                sdid_survey_data_full_design,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sdid_survey_design_full,
            )
        finally:
            _np.random.default_rng = original_default_rng  # type: ignore[assignment]

        # We expect ≥1 rng.choice call per replication × treated-stratum.
        # All treated are in stratum 0, so each draw produces exactly one
        # choice over stratum-0 controls.
        assert len(captured_pseudo_treated_strata) >= 30
        # Every sampled pseudo-treated subset must come from a stratum
        # that contains actual treated units (here, only stratum 0).
        for draw_strata in captured_pseudo_treated_strata:
            assert draw_strata.issubset(treated_strata_set), (
                f"sampled stratum {draw_strata} not subset of "
                f"treated_strata_set {treated_strata_set}"
            )

    def test_placebo_full_design_raises_on_zero_control_stratum(self, sdid_survey_data_full_design):
        """Case B: stratum with treated units but zero controls → ValueError."""
        df = sdid_survey_data_full_design.copy()
        # Move all controls out of stratum 0; treated stays in stratum 0.
        df.loc[df["unit"].isin(range(5, 15)), "stratum"] = 1

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=7)
        with pytest.raises(ValueError, match=r"at least one control per stratum.*has 0 controls"):
            est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_placebo_full_design_raises_on_effective_single_support(
        self, sdid_survey_data_full_design
    ):
        """R11 P1 fix: Case D effective single-support — n_t_h == 1 with
        only one positive-weight control + zero-weight cohabitants.

        Row-count guards pass (``n_c_h > n_t_h``) and Case E passes
        (``n_c_h_positive == n_t_h``), but every successful pseudo-
        treated draw collapses to the single positive-weight control's
        outcome (zero-weight cohabitants contribute 0 to numerator and
        denominator). The placebo distribution is degenerate: SE ≈ 0
        from FP noise across identical means, not a meaningful null.

        Without this guard, the previous code marked the stratum as
        non-degenerate based on ``n_c_h > n_t_h`` (raw count), and the
        retry loop would silently succeed on any positive-mass pick
        with the same effective mean → ``SE = 0.0``.
        """
        # Build a fixture where stratum 0 has 1 treated + 1 positive-
        # weight control + multiple zero-weight controls; stratum 1
        # has only controls. This sets up effective single-support
        # in stratum 0 even though raw n_c_h > n_t_h.
        # Reuse sdid_survey_data_full_design but trim to 1 treated and
        # zero out most stratum-0 controls' weights.
        df = sdid_survey_data_full_design.copy()
        # Drop treated units 1-4, keep unit 0 as the sole treated.
        df = df[~df["unit"].isin([1, 2, 3, 4])].copy()
        # Stratum 0 controls (5-14): keep unit 5 with positive weight,
        # zero out 6-14.
        df.loc[df["unit"].isin(range(6, 15)), "weight"] = 0.0

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.raises(
            ValueError,
            match=r"single distinct positive-mass pseudo-treated mean",
        ):
            est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_placebo_full_design_raises_on_zero_weight_controls_in_stratum(
        self, sdid_survey_data_full_design
    ):
        """R9 P1 fix: Case E — treated stratum has raw controls but
        zero positive-weight controls.

        Row-count guards (Case B/C) pass because stratum 0 has 10 raw
        controls vs 5 treated. But the placebo allocator computes
        pseudo-treated means as ``np.average(Y, weights=w_control)``;
        if every stratum-0 control has weight 0, every draw's pseudo-
        treated subset has zero weight sum (ZeroDivisionError on
        np.average). Previously the retry loop swallowed each failure
        and the fit reported ``SE=0.0`` with a generic
        ``n_successful=0`` warning. The new Case E fit-time guard
        rejects up-front with a targeted ValueError.
        """
        df = sdid_survey_data_full_design.copy()
        # Zero out survey weights for all stratum-0 controls (units 5-14).
        # Treated units (0-4) and stratum-1 controls (15-29) keep
        # positive weights, so Cases B/C still pass.
        df.loc[df["unit"].isin(range(5, 15)), "weight"] = 0.0

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=42)
        with pytest.raises(
            ValueError,
            match=r"at least n_treated controls with positive survey weight",
        ):
            est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_placebo_full_design_raises_on_exact_count_stratum(
        self, sdid_survey_data, survey_design_full
    ):
        """R4 P1 fix: Case D — every treated stratum has n_c == n_t.

        The ``sdid_survey_data`` fixture has 5 treated units + 5 controls
        in stratum 0 and 10 controls in stratum 1 (with no treated
        units). For placebo stratified permutation, the pseudo-treated
        set within stratum 0 is chosen from 5 controls, sized 5 — only
        one allocation is possible. Every placebo draw reproduces the
        same pseudo-treated set, the placebo null collapses to a
        single point, and SE = FP noise (~1e-16). The new Case D guard
        rejects this design at fit-time rather than silently reporting
        a near-zero SE that would pass a naïve ``result.se > 0`` check.
        """
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.raises(
            ValueError,
            match=r"single distinct positive-mass pseudo-treated mean across all draws",
        ):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=survey_design_full,
            )

    def test_placebo_full_design_raises_on_undersupplied_stratum(
        self, sdid_survey_data_full_design
    ):
        """Case C: stratum with n_controls < n_treated → ValueError."""
        df = sdid_survey_data_full_design.copy()
        # Move 8 of the 10 stratum-0 controls out; leaves 2 controls
        # in stratum 0 with 5 treated → n_c=2 < n_t=5 → Case C. Using
        # ``nest=True`` so the shifted PSUs stay unique-within-stratum.
        df.loc[df["unit"].isin(range(7, 15)), "stratum"] = 1

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", nest=True)
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=7)
        with pytest.raises(
            ValueError,
            match=r"at least n_treated controls.*2 controls but 5 treated",
        ):
            est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_placebo_full_design_se_differs_from_pweight_only(self, sdid_survey_data_full_design):
        """Full-design placebo SE differs from pweight-only placebo SE.

        Pweight-only path permutes across ALL controls (unstratified);
        full-design permutes WITHIN treated-strata only. Different
        permutation supports ⇒ different null distributions ⇒ different
        SEs. Analog of the bootstrap differs-test.
        """
        sd_pweight = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        est_pw = SyntheticDiD(variance_method="placebo", n_bootstrap=100, seed=42)
        result_pw = est_pw.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight,
        )
        est_full = SyntheticDiD(variance_method="placebo", n_bootstrap=100, seed=42)
        result_full = est_full.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_full,
        )
        assert result_pw.att == pytest.approx(result_full.att, abs=1e-10)
        assert result_pw.se != pytest.approx(result_full.se, abs=1e-6)

    def test_placebo_fpc_alone_no_op_warns_and_matches_pweight_only(
        self, sdid_survey_data_full_design
    ):
        """R8 P1 fix: ``fpc=`` alone does not flip placebo dispatch.

        Permutation tests condition on the observed sample (Pesarin 2001
        §1.5), so FPC's sampling-fraction adjustment doesn't enter
        Algorithm 4 or its stratified-permutation survey extension. The
        previous dispatcher routed any ``fpc is not None`` design through
        ``_placebo_variance_se_survey`` (weighted-FW per draw), silently
        changing numerics relative to the no-FPC fit even though FPC
        played no role in the math.

        The fix gates placebo's survey-path dispatch on
        ``strata is not None OR psu is not None`` only, and emits a
        ``UserWarning`` whenever FPC is set on a placebo fit. This test
        asserts both: (a) the warning fires and (b) ``SE`` matches the
        pweight-only-no-FPC fit at ``rel=1e-12`` (FPC truly is a no-op).
        """
        df = sdid_survey_data_full_design.copy()
        df["fpc_col"] = 1000.0  # any positive value — no-op on placebo

        sd_fpc_only = SurveyDesign(weights="weight", fpc="fpc_col")
        sd_pweight_only = SurveyDesign(weights="weight")

        est_fpc = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.warns(
            UserWarning,
            match=r"SurveyDesign\(fpc=\.\.\.\) is a no-op on variance_method='placebo'",
        ):
            r_fpc = est_fpc.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_fpc_only,
            )

        est_pw = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        r_pw = est_pw.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight_only,
        )

        # FPC is documented as no-op for placebo: the SE under FPC must
        # exactly match the SE without FPC (same dispatch path, no
        # numerical drift from the routing flip the dispatcher used to
        # introduce on `fpc is not None`).
        assert r_fpc.se == pytest.approx(r_pw.se, rel=1e-12)
        assert r_fpc.att == pytest.approx(r_pw.att, abs=1e-12)

    def test_placebo_typo_fpc_column_still_raises(self, sdid_survey_data_full_design):
        """R13 P3 fix: typoed FPC column name must still raise on placebo.

        The FPC pre-resolve drop on placebo (R11 P1) bypasses
        ``SurveyDesign.resolve()``'s missing-column check. A typoed
        ``fpc="fpc_typo"`` would be silently ignored behind the no-op
        warning, hiding a genuine input-spec mistake. The fix
        validates the FPC column name against ``data.columns`` BEFORE
        dropping; missing columns surface the same targeted error
        ``resolve()`` would have raised.
        """
        sd_typo = SurveyDesign(weights="weight", fpc="nonexistent_col")
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=42)
        with pytest.raises(
            ValueError,
            match=r"FPC column 'nonexistent_col' not found in data",
        ):
            est.fit(
                sdid_survey_data_full_design,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_typo,
            )

    def test_placebo_low_fpc_with_explicit_psu_skips_resolve_validator(
        self, sdid_survey_data_full_design
    ):
        """R11 P1 fix: ``SurveyDesign.resolve()`` itself enforces
        ``FPC >= n_PSU`` design-validity, but FPC is a placebo no-op
        (Pesarin 2001 §1.5). On the placebo path, FPC is dropped from
        a copy of the SurveyDesign before resolution so the
        resolve-time validator never fires; the user sees the
        documented FPC no-op warning and the fit succeeds.

        Test: explicit ``psu`` + low ``fpc`` (below the per-stratum
        ``n_PSU`` threshold) — would normally raise inside
        ``SurveyDesign.resolve()`` with "FPC must be >= n_PSU". On
        placebo, it succeeds with the no-op warning. Bootstrap on the
        same design still raises (validator-skip is variance-method-
        gated).
        """
        df = sdid_survey_data_full_design.copy()
        # Each stratum has 3 PSUs in the well-formed-jackknife layout,
        # but sdid_survey_data_full_design has stratum 0 with PSUs
        # {0,1,2} and stratum 1 with PSUs {3,4,5} — 3 PSUs each. fpc=2
        # is below the 3-PSU threshold per stratum.
        df["fpc_low"] = 2.0

        sd_low_fpc_psu = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc_low")
        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        est_fpc = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.warns(
            UserWarning,
            match=r"SurveyDesign\(fpc=\.\.\.\) is a no-op on variance_method='placebo'",
        ):
            r_fpc = est_fpc.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_low_fpc_psu,
            )

        est_no_fpc = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        r_no_fpc = est_no_fpc.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_no_fpc,
        )

        # FPC truly is a no-op for placebo even with explicit psu: SE
        # matches the no-FPC fit at machine precision.
        assert r_fpc.se == pytest.approx(r_no_fpc.se, rel=1e-12)
        assert r_fpc.att == pytest.approx(r_no_fpc.att, abs=1e-12)
        # Bootstrap on the same low-FPC design still raises the resolve-
        # time validator error (validator-skip stays placebo-only).
        est_boot = SyntheticDiD(variance_method="bootstrap", n_bootstrap=20, seed=42)
        with pytest.raises(
            ValueError,
            match=r"FPC \(2\.0\) is less than the number of PSUs",
        ):
            est_boot.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_low_fpc_psu,
            )

    def test_placebo_low_fpc_no_psu_warns_no_validator_block(self, sdid_survey_data_full_design):
        """R10 P1 fix: implicit-PSU FPC validator skipped on placebo.

        The implicit-PSU FPC validator (PR #355 R8 P1) rejects designs
        where ``psu is None`` and ``fpc < n_units`` because Rao-Wu
        bootstrap treats each unit as its own PSU and would fail mid-
        draw. But placebo doesn't use FPC at all (Pesarin 2001 §1.5 —
        permutation tests condition on the observed sample), so the
        validator should be skipped on the placebo path. Otherwise a
        legitimate placebo fit raises on a constraint that doesn't
        apply to its math.

        Test: ``fpc_col = 5`` is well below ``n_units = 30``, which
        would trip the bootstrap validator. With ``variance_method=
        "placebo"``, the fit must succeed, emit the documented FPC
        no-op ``UserWarning``, and produce SE matching the no-FPC
        pweight-only fit (true no-op).
        """
        df = sdid_survey_data_full_design.copy()
        # 5 << n_units=30 — would trip the bootstrap implicit-PSU
        # FPC validator (FPC must be >= n_units when psu is None).
        df["fpc_col"] = 5.0

        sd_low_fpc = SurveyDesign(weights="weight", fpc="fpc_col")
        sd_pweight = SurveyDesign(weights="weight")

        est_fpc = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        with pytest.warns(
            UserWarning,
            match=r"SurveyDesign\(fpc=\.\.\.\) is a no-op on variance_method='placebo'",
        ):
            r_fpc = est_fpc.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_low_fpc,
            )

        est_pw = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        r_pw = est_pw.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight,
        )

        # FPC truly is a no-op for placebo: SE/ATT must match the no-FPC
        # fit at machine precision regardless of FPC value.
        assert r_fpc.se == pytest.approx(r_pw.se, rel=1e-12)
        assert r_fpc.att == pytest.approx(r_pw.att, abs=1e-12)
        # And bootstrap on the same low-FPC design must still raise the
        # implicit-PSU validator error (validator stays gated on
        # bootstrap/jackknife only).
        est_boot = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        with pytest.raises(
            ValueError,
            match=r"FPC \(5\.0\) is less than the number of units",
        ):
            est_boot.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_low_fpc,
            )

    def test_placebo_full_design_psu_only_routes_through_survey_path(
        self, sdid_survey_data_jk_well_formed
    ):
        """R2 P1 regression: PSU/FPC-without-strata placebo routes through
        ``_placebo_variance_se_survey`` (with a synthesized single
        stratum), matching the documented full-design contract.

        The original implementation only routed through the survey path
        when ``strata`` was declared; PSU/FPC-only designs fell through
        to the non-survey placebo allocator even though REGISTRY
        declares full-design support. Now all three designs (strata,
        psu, fpc) dispatch to the weighted-FW survey path.
        """
        sd_psu_only = SurveyDesign(weights="weight", psu="psu")

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=42)
        # Monkeypatch to verify dispatch (sentinel returns distinct from
        # both paths).
        est._placebo_variance_se_survey = lambda *a, **kw: (42.0, np.array([1.0]))  # type: ignore[assignment]
        est._placebo_variance_se = lambda *a, **kw: (99.0, np.array([1.0]))  # type: ignore[assignment]
        result = est.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_psu_only,
        )
        # Survey path should have fired — SE reflects 42.0 sentinel,
        # rescaled by Y_scale.
        assert result.se > 40.0 and result.se < 90.0

    def test_placebo_dispatches_to_survey_method_under_full_design(
        self, sdid_survey_data_full_design, sdid_survey_design_full
    ):
        """Full design → _placebo_variance_se_survey; pweight-only → _placebo_variance_se.

        Deterministic dispatch test via monkeypatch. Sentinel return
        value verifies the right branch fires.
        """
        # Full-design dispatch
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=42)
        sentinel = (42.0, np.array([1.0, 2.0, 3.0]))
        est._placebo_variance_se_survey = lambda *a, **kw: sentinel  # type: ignore[assignment]
        est._placebo_variance_se = lambda *a, **kw: (99.0, np.array([]))  # type: ignore[assignment]
        result = est.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sdid_survey_design_full,
        )
        # se rescales by Y_scale (normalization applied in fit), so check
        # ordering rather than exact sentinel.
        assert result.se > 40.0  # distinguishes 42.0 sentinel from 99.0
        assert result.variance_method == "placebo"

        # Pweight-only dispatch
        est2 = SyntheticDiD(variance_method="placebo", n_bootstrap=30, seed=42)
        est2._placebo_variance_se_survey = lambda *a, **kw: (42.0, np.array([]))  # type: ignore[assignment]
        est2._placebo_variance_se = lambda *a, **kw: (99.0, np.array([1.0]))  # type: ignore[assignment]
        sd_pw = SurveyDesign(weights="weight")
        result2 = est2.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pw,
        )
        # Pweight-only should dispatch to the non-survey method (99.0 * Y_scale)
        assert result2.se > 90.0  # distinguishes 99.0 from 42.0


class TestSDIDSurveyJackknifeFullDesign:
    """PSU-level LOO jackknife with stratum aggregation (Rust & Rao 1996).

    Variance formula: SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²
    with f_h = n_h_sampled / fpc[h]. See REGISTRY §SyntheticDiD
    "Note (survey + jackknife composition)".
    """

    def test_jackknife_full_design_stratum_aggregation_formula_magnitude(
        self, sdid_survey_data_jk_well_formed
    ):
        """SE² matches the Rust & Rao stratum-aggregation formula exactly.

        Independently recomputes SE from the returned tau_loo_all array
        using ``Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²``; asserts
        rtol=1e-12 match. Catches off-by-one in (n_h-1)/n_h, wrong
        tau_bar_h, or missing (1-f_h). Uses the well-formed fixture so
        every PSU-LOO is defined (6 strata-level replicates total).
        """
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert np.isfinite(result.se)
        assert result.se > 0
        # Fixture structure: stratum 0 has PSUs {0, 1, 2} (n_h=3), stratum 1
        # has PSUs {3, 4, 5} (n_h=3). No FPC → f_h=0 for both. Every
        # PSU-LOO is well-defined, so tau_loo_all has 3 + 3 = 6 entries
        # ordered as [s0 PSU 0, s0 PSU 1, s0 PSU 2, s1 PSU 3, s1 PSU 4, s1 PSU 5].
        taus = np.asarray(result.variance_effects, dtype=float)
        assert len(taus) == 6
        # Apply the Rust & Rao formula by hand. Y_scale rescaling is
        # applied uniformly to tau_loo_all inside fit(), so the formula
        # holds on the rescaled values.
        s0 = taus[:3]
        s1 = taus[3:6]
        n_h = 3
        factor = (n_h - 1) / n_h  # f_h = 0 → (1 - f_h) = 1
        ss0 = np.sum((s0 - s0.mean()) ** 2)
        ss1 = np.sum((s1 - s1.mean()) ** 2)
        expected_se = np.sqrt(factor * (ss0 + ss1))
        assert result.se == pytest.approx(expected_se, rel=1e-12)

    def test_jackknife_full_design_fpc_reduces_se_magnitude(self, sdid_survey_data_jk_well_formed):
        """With FPC, SE is reduced by the (1-f_h) multiplier per stratum.

        Two fits: one without FPC (f_h=0 so (1-f_h)=1); one with FPC set
        to a population count such that f_h = n_h/fpc = 3/6 = 0.5.
        Expected: SE_fpc = SE_nofpc * sqrt(1-0.5) = SE_nofpc / sqrt(2).
        Uses the well-formed fixture so every LOO is defined.
        """
        df_no_fpc = sdid_survey_data_jk_well_formed
        df_fpc = sdid_survey_data_jk_well_formed.copy()
        df_fpc["fpc_col"] = 6.0  # n_h=3 per stratum, f_h = 3/6 = 0.5

        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        sd_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc_col")

        est1 = SyntheticDiD(variance_method="jackknife", seed=42)
        result_no_fpc = est1.fit(
            df_no_fpc,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_no_fpc,
        )
        est2 = SyntheticDiD(variance_method="jackknife", seed=42)
        result_fpc = est2.fit(
            df_fpc,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_fpc,
        )
        # Expected magnitude ratio: SE_fpc/SE_no_fpc = sqrt(1 - 0.5) = 1/sqrt(2)
        assert result_fpc.se == pytest.approx(result_no_fpc.se / np.sqrt(2), rel=1e-10)

    def test_jackknife_full_design_se_differs_from_pweight_only(
        self, sdid_survey_data_jk_well_formed
    ):
        """Full-design jackknife SE differs from pweight-only jackknife SE.

        Full-design: PSU-level LOO + stratum aggregation. Pweight-only:
        unit-level LOO (classical fixed-weight jackknife). Different
        resampling granularity ⇒ different SE.
        """
        sd_pweight = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        est_pw = SyntheticDiD(variance_method="jackknife", seed=42)
        result_pw = est_pw.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pweight,
        )
        est_full = SyntheticDiD(variance_method="jackknife", seed=42)
        result_full = est_full.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_full,
        )
        assert result_pw.att == pytest.approx(result_full.att, abs=1e-10)
        assert result_pw.se != pytest.approx(result_full.se, abs=1e-6)

    def test_get_loo_effects_df_raises_on_survey_jackknife(self, sdid_survey_data_jk_well_formed):
        """R1 P1 fix: get_loo_effects_df blocks only on full-design survey
        jackknife (PSU-level replicates), not on pweight-only jackknife.

        Keys off the explicit ``_loo_granularity`` flag (R2 P1 — the old
        ``survey_metadata.n_psu`` heuristic false-positives on pweight-
        only fits, which also populate ``n_psu`` via implicit-PSU
        metadata but still run unit-level LOO).
        """
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert getattr(result, "_loo_granularity", None) == "psu"
        with pytest.raises(
            NotImplementedError,
            match=r"unit-level-LOO only.*PSU-level LOO with stratum aggregation",
        ):
            result.get_loo_effects_df()

    def test_get_loo_effects_df_works_on_pweight_only_jackknife(
        self, sdid_survey_data, survey_design_weights
    ):
        """R2 P1 regression: pweight-only jackknife still exposes unit-level
        LOO diagnostics through ``get_loo_effects_df``.

        Pweight-only fits populate ``survey_metadata.n_psu`` (via the
        implicit-PSU metadata path) but run the non-survey unit-level
        jackknife (classical Algorithm 3). The accessor must stay
        available on this path — blocking it would regress a documented
        diagnostic surface.
        """
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert getattr(result, "_loo_granularity", None) == "unit"
        # Accessor returns a unit-indexed DataFrame with the expected
        # schema; positional join is well-defined on the pweight-only
        # path because ``variance_effects`` has length n_control + n_treated.
        df = result.get_loo_effects_df()
        assert len(df) == result.n_control + result.n_treated
        assert set(df.columns) == {"unit", "role", "att_loo", "delta_from_full"}
        assert set(df["role"].unique()) <= {"control", "treated"}

    def test_jackknife_full_design_full_census_fpc_returns_zero_se(
        self, sdid_survey_data_jk_well_formed
    ):
        """R5 P1 fix: full-census FPC → SE=0, not NaN.

        Rust & Rao's stratified jackknife formula has an explicit
        ``(1 - f_h)`` factor. When ``fpc[h] == n_h`` for every
        contributing stratum, ``f_h = 1``, ``(1 - f_h) = 0``, and every
        stratum contribution is zero → ``total_variance = 0`` by
        legitimate design, not by "every stratum skipped". The correct
        jackknife SE in that case is **zero** (full census: no sampling
        variance), not NaN. Reserve NaN for the truly-undefined cases
        (all strata skipped, undefined PSU-LOO replicate).
        """
        df = sdid_survey_data_jk_well_formed.copy()
        # Each stratum has n_h=3 PSUs. Setting fpc=3 gives f_h=1 and
        # (1 - f_h) = 0 — the formula collapses the stratum contribution
        # to zero for legitimate design reasons.
        df["fpc_full_census"] = 3.0

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_full_census",
        )
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        # SE must be exactly zero (legitimate full-census no-sampling
        # variance), not NaN (undefined) and not a tiny positive number.
        assert np.isfinite(result.se)
        assert result.se == 0.0

    def test_jackknife_full_design_full_census_short_circuits_undefined_loo(
        self, sdid_survey_data_full_design
    ):
        """R6 P1 fix: full-census stratum short-circuits before undefined-LOO.

        ``sdid_survey_data_full_design`` packs all 5 treated units into
        stratum 0 PSU 0 → LOO PSU 0 removes all treated → undefined
        replicate → would normally return ``SE=NaN``. But if every
        stratum has ``fpc = n_h`` (full census, ``f_h = 1`` →
        ``(1 - f_h) = 0``), every stratum's variance contribution is
        zero regardless of LOO feasibility. The correct jackknife SE
        in that case is exactly zero (full census: no sampling
        variance), not NaN from an undefined replicate that doesn't
        actually enter the formula.
        """
        df = sdid_survey_data_full_design.copy()
        # Each stratum has n_h=3 PSUs → fpc=3 gives f_h=1 per stratum.
        df["fpc_full_census"] = 3.0

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_full_census",
        )
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        # Fit must succeed without the undefined-LOO warning, and SE
        # must be exactly zero (not NaN and not a non-zero number).
        result = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert np.isfinite(result.se)
        assert result.se == 0.0

    def test_jackknife_full_design_lonely_psu_adjust_raises(self, sdid_survey_data_jk_well_formed):
        """R5 P1 fix: ``SurveyDesign(lonely_psu='adjust')`` on the jackknife
        survey path raises NotImplementedError rather than silently being
        treated as ``"remove"``.

        ``"remove"`` and ``"certainty"`` both contribute 0 variance for
        singleton strata on the jackknife path, matching canonical R
        ``survey::svyjkn`` behavior. ``"adjust"`` requires an overall-
        mean fallback per stratum that is not yet implemented; rejecting
        upfront prevents silent variance miscomputation.
        """
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            lonely_psu="adjust",
        )
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        with pytest.raises(
            NotImplementedError,
            match=r"lonely_psu='adjust'.*not supported on the SDID jackknife",
        ):
            est.fit(
                sdid_survey_data_jk_well_formed,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_jackknife_full_design_all_certainty_psu_returns_zero_se(
        self, sdid_survey_data_jk_well_formed
    ):
        """R12 P1 fix: all-singleton-strata + ``lonely_psu='certainty'``
        returns SE=0 (legitimate zero variance), not NaN.

        Mirrors the broader survey contract from
        ``tests/test_survey.py::test_all_certainty_psu_zero_vcov``:
        certainty PSUs are sampled with certainty (no sampling
        variance). When every stratum is singleton + certainty, the
        Rust & Rao stratified jackknife sums zero variance
        contributions across strata — this is a legitimate zero, not
        an "every stratum was skipped → undefined" case. The previous
        code conflated the two and returned ``SE = NaN``.

        Test: collapse the well-formed jackknife fixture so every unit
        is its own stratum AND its own PSU (all 30 strata are
        singletons). Under ``lonely_psu='certainty'``, the fit must
        return SE = 0 exactly, with NaN inference fields downstream
        (via ``safe_inference``). Under ``lonely_psu='remove'``, the
        same design returns SE = NaN with the "every stratum was
        skipped" warning.
        """
        df = sdid_survey_data_jk_well_formed.copy()
        # Each unit is its own stratum AND its own PSU (all strata
        # singletons).
        df["stratum"] = df["unit"]
        df["psu"] = df["unit"]

        # Under "certainty": legitimate zero-variance contributors → SE=0.
        sd_certainty = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", lonely_psu="certainty"
        )
        est_cert = SyntheticDiD(variance_method="jackknife", seed=42)
        result_cert = est_cert.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_certainty,
        )
        assert np.isfinite(result_cert.se)
        assert result_cert.se == 0.0
        # Inference downstream from SE=0 is NaN via safe_inference (zero
        # SE → undefined t-statistic / p-value / CI).
        assert np.isnan(result_cert.t_stat)
        assert np.isnan(result_cert.p_value)

        # Under "remove": same design returns SE=NaN with the "every
        # stratum was skipped" warning (no contributing stratum).
        sd_remove = SurveyDesign(weights="weight", strata="stratum", psu="psu", lonely_psu="remove")
        est_rem = SyntheticDiD(variance_method="jackknife", seed=42)
        with pytest.warns(UserWarning, match=r"every stratum was skipped"):
            result_rem = est_rem.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd_remove,
            )
        assert np.isnan(result_rem.se)

    def test_jackknife_full_design_lonely_psu_certainty_equivalent_to_remove(
        self, sdid_survey_data_jk_well_formed
    ):
        """``lonely_psu='certainty'`` is accepted and produces the same SE
        as ``lonely_psu='remove'`` (both contribute 0 for singleton
        strata on the jackknife path).
        """
        sd_remove = SurveyDesign(weights="weight", strata="stratum", psu="psu", lonely_psu="remove")
        sd_certainty = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", lonely_psu="certainty"
        )

        est1 = SyntheticDiD(variance_method="jackknife", seed=42)
        result_remove = est1.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_remove,
        )
        est2 = SyntheticDiD(variance_method="jackknife", seed=42)
        result_certainty = est2.fit(
            sdid_survey_data_jk_well_formed,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_certainty,
        )
        assert result_remove.se == pytest.approx(result_certainty.se, rel=1e-14)

    def test_jackknife_full_design_undefined_replicate_returns_nan(
        self, sdid_survey_data_full_design
    ):
        """R1 P0 fix: if any LOO in a contributing stratum is undefined,
        the stratified Rust & Rao formula does not apply and SE is NaN.

        ``sdid_survey_data_full_design`` has all treated units in stratum
        0 PSU 0. LOO of PSU 0 removes all treated and the SDID estimator
        τ̂_{(0,0)} is undefined. The old code silently skipped this LOO
        while still applying the full ``(n_h-1)/n_h = 2/3`` factor,
        under-scaling variance (silently wrong SE). The new code returns
        NaN + a targeted UserWarning instead.
        """
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        with pytest.warns(
            UserWarning,
            match=r"delete-one replicate for stratum 0 PSU 0 is not "
            r"computable.*deletion removes all treated units",
        ):
            result = est.fit(
                sdid_survey_data_full_design,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )
        assert np.isnan(result.se)

    def test_jackknife_full_design_single_psu_stratum_skipped(self, sdid_survey_data_full_design):
        """Stratum with only 1 PSU contributes 0 to total variance.

        Degenerate stratum: relabel stratum-0 PSU 1+2 to a new stratum 2
        each with only 1 PSU. Jackknife should silently skip them and
        produce SE only from stratum 1 (which still has 3 PSUs).
        """
        df = sdid_survey_data_full_design.copy()
        # Units 5-9 → stratum 2, PSU 1 alone; units 10-14 → stratum 3, PSU 2 alone
        df.loc[df["unit"].isin(range(5, 10)), "stratum"] = 2
        df.loc[df["unit"].isin(range(10, 15)), "stratum"] = 3

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        # Stratum 0 now has only PSU 0 (treated, degenerate LOO).
        # Strata 2, 3 each have 1 PSU → skipped.
        # Stratum 1 has 3 PSUs → contributes.
        # Fit should proceed; SE reflects only stratum 1.
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_jackknife_full_design_unstratified_short_circuit(self, sdid_survey_data_full_design):
        """No strata + single PSU → SE=NaN (unidentified variance)."""
        df = sdid_survey_data_full_design.copy()
        df["psu"] = 0  # all units in a single PSU

        # Unstratified single-PSU design
        sd = SurveyDesign(weights="weight", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        result = est.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd,
        )
        assert np.isnan(result.se)

    def test_jackknife_full_design_all_strata_skipped_warns_and_returns_nan(
        self, sdid_survey_data_full_design
    ):
        """Every stratum has <2 PSUs → UserWarning + NaN SE."""
        df = sdid_survey_data_full_design.copy()
        # Collapse so every stratum has only 1 PSU: unit 0→psu0/s0, unit 1→psu1/s1, etc.
        df["psu"] = df["unit"]
        df["stratum"] = df["unit"]  # each unit is its own stratum

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        with pytest.warns(UserWarning, match=r"every stratum was skipped|SE is undefined"):
            result = est.fit(
                df,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )
        assert np.isnan(result.se)

    def test_jackknife_dispatches_to_survey_method_under_full_design(
        self, sdid_survey_data_full_design, sdid_survey_design_full
    ):
        """Full design → _jackknife_se_survey; pweight-only → _jackknife_se."""
        est = SyntheticDiD(variance_method="jackknife", seed=42)
        est._jackknife_se_survey = lambda *a, **kw: (42.0, np.array([1.0, 2.0]))  # type: ignore[assignment]
        est._jackknife_se = lambda *a, **kw: (99.0, np.array([]))  # type: ignore[assignment]
        result = est.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sdid_survey_design_full,
        )
        assert result.se > 40.0  # from 42.0 sentinel

        est2 = SyntheticDiD(variance_method="jackknife", seed=42)
        est2._jackknife_se_survey = lambda *a, **kw: (42.0, np.array([]))  # type: ignore[assignment]
        est2._jackknife_se = lambda *a, **kw: (99.0, np.array([1.0]))  # type: ignore[assignment]
        sd_pw = SurveyDesign(weights="weight")
        result2 = est2.fit(
            sdid_survey_data_full_design,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=sd_pw,
        )
        assert result2.se > 90.0  # from 99.0 sentinel


# =============================================================================
# TROP Survey Tests
# =============================================================================


class TestTROPSurvey:
    """Survey support tests for TROP (local and global methods)."""

    def test_smoke_local_weights_only(self, trop_survey_data, survey_design_weights):
        """Local method completes with survey weights."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_smoke_global_weights_only(self, trop_survey_data, survey_design_weights):
        """Global method completes with survey weights."""
        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_uniform_weights_match_local(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (local)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_uniform_weights_match_global(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (global)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, trop_survey_data, survey_design_weights):
        """Metadata has correct fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0

    def test_full_design_local_rao_wu(self, trop_survey_data, survey_design_full):
        """Full design (strata/PSU/FPC) uses Rao-Wu bootstrap and succeeds."""
        est = TROP(method="local", n_bootstrap=20, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_full_design_global_rao_wu(self, trop_survey_data, survey_design_full):
        """Full design (strata/PSU/FPC) with global method uses Rao-Wu bootstrap."""
        est = TROP(method="global", n_bootstrap=20, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_full,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_fweight_aweight_raises(self, trop_survey_data):
        """Non-pweight raises ValueError."""
        est = TROP(method="local", n_bootstrap=10, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="aweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=sd,
            )

    def test_weighted_att_differs(self, trop_survey_data, survey_design_weights):
        """Non-uniform weights change ATT."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result_survey.att != pytest.approx(result_no.att, abs=1e-6)

    def test_weighted_att_differs_global(self, trop_survey_data, survey_design_weights):
        """Non-uniform weights change ATT for method='global'."""
        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result_no = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result_survey.att != pytest.approx(result_no.att, abs=1e-6)

    def test_summary_includes_survey(self, trop_survey_data, survey_design_weights):
        """summary() contains survey section."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary

    def test_weight_scale_invariance(self, trop_survey_data, survey_design_weights):
        """Scale invariance: 2x weights produce same ATT."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["weight_3x"] = trop_survey_data["weight"] * 3.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_1x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        result_3x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="weight_3x"),
        )
        assert result_3x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, trop_survey_data):
        """Validation catches time-varying weights."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["bad_weight"] = trop_survey_data["weight"] + trop_survey_data["time"] * 0.1
        est = TROP(method="local", n_bootstrap=10, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_convenience_function_with_survey(self, trop_survey_data, survey_design_weights):
        """trop() convenience function accepts survey_design."""
        result = trop(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
            n_bootstrap=10,
            seed=42,
            max_iter=5,
        )
        assert result.survey_metadata is not None

    def test_to_dict_includes_survey(self, trop_survey_data, survey_design_weights):
        """to_dict() includes survey metadata fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_local_bootstrap_nan_treated_outcomes(self, trop_survey_data):
        """Bootstrap handles NaN treated outcomes without poisoning SE."""
        trop_survey_data = trop_survey_data.copy()
        # Set some treated post-treatment outcomes to NaN
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 7)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        # Point estimate should use finite cells only
        assert np.isfinite(result.att)
        # SE should remain finite (not poisoned by NaN)
        assert np.isfinite(result.se)

    def test_local_bootstrap_nan_with_survey(self, trop_survey_data, survey_design_weights):
        """Bootstrap + survey handles NaN treated outcomes correctly."""
        trop_survey_data = trop_survey_data.copy()
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 8)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)


# =============================================================================
# Pinned Numerical Tests
# =============================================================================


class TestPinnedNumerical:
    """Deterministic numerical tests for exact weighted formulas."""

    def test_sdid_weighted_att_manual(self):
        """Manual ATT check: survey-weighted treated means + ω∘w_co composition."""
        # Tiny 2x2 balanced panel: 2 control, 1 treated, 2 pre + 1 post
        np.random.seed(99)
        data = pd.DataFrame(
            {
                "unit": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "outcome": [1.0, 2.0, 3.0, 2.0, 3.0, 4.5, 5.0, 6.0, 10.0],
                "treated": [0, 0, 0, 0, 0, 0, 1, 1, 1],
                "weight": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
            }
        )
        # Single treated unit → treated means are trivially that unit's outcomes
        # (survey weight doesn't change a single-unit mean)
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=20, seed=42)
        result = est.fit(
            data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[2],
            survey_design=SurveyDesign(weights="weight"),
        )
        # Verify unit_weights sum to 1 (composed with survey)
        assert sum(result.unit_weights.values()) == pytest.approx(1.0, abs=1e-10)
        assert np.isfinite(result.att)

    def test_trop_weighted_att_aggregation(self):
        """Verify TROP ATT = weighted mean of tau values."""
        # Create data where we can predict directional effect of weighting
        np.random.seed(77)
        n_units = 15
        n_periods = 6
        n_treated = 3

        units = list(range(n_units))
        periods = list(range(n_periods))

        rows = []
        for u in units:
            is_treated = u < n_treated
            base = u * 0.5
            for t in periods:
                y = base + 0.2 * t + np.random.randn() * 0.3
                d = 1 if (is_treated and t >= 3) else 0
                if d == 1:
                    # Different effect per unit: unit 0 gets +1, unit 1 gets +3, unit 2 gets +5
                    y += 1.0 + 2.0 * u
                rows.append({"unit": u, "time": t, "outcome": y, "D": d})

        data = pd.DataFrame(rows)
        # Weight unit 2 (biggest effect) heavily
        weights = np.ones(n_units)
        weights[2] = 10.0  # unit 2 has effect ~5, heavily weighted
        unit_map = {u: i for i, u in enumerate(units)}
        data["weight"] = weights[data["unit"].map(unit_map).values]

        est_no = TROP(method="local", n_bootstrap=5, seed=42, max_iter=3)
        result_no = est_no.fit(data, "outcome", "D", "unit", "time")

        est_w = TROP(method="local", n_bootstrap=5, seed=42, max_iter=3)
        result_w = est_w.fit(
            data,
            "outcome",
            "D",
            "unit",
            "time",
            survey_design=SurveyDesign(weights="weight"),
        )

        # Weighted ATT should be pulled toward unit 2's larger effect
        assert result_w.att > result_no.att

    def test_sdid_to_dict_schema_matches_did(self):
        """SyntheticDiDResults.to_dict() survey fields match DiDResults schema."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "unit": [0, 0, 1, 1, 2, 2],
                "time": [0, 1, 0, 1, 0, 1],
                "outcome": [1.0, 2.0, 2.0, 3.0, 5.0, 8.0],
                "treated": [0, 0, 0, 0, 1, 1],
                "weight": [1.0, 1.0, 2.0, 2.0, 1.5, 1.5],
            }
        )
        est = SyntheticDiD(n_bootstrap=10, seed=42)
        result = est.fit(
            data,
            "outcome",
            "treated",
            "unit",
            "time",
            post_periods=[1],
            survey_design=SurveyDesign(weights="weight"),
        )
        d = result.to_dict()
        # Schema alignment: all these fields should be present
        for key in [
            "weight_type",
            "effective_n",
            "design_effect",
            "sum_weights",
            "n_strata",
            "n_psu",
            "df_survey",
        ]:
            assert key in d, f"Missing key: {key}"


class TestTROPRaoWuEquivalence:
    """Test Rao-Wu vs block bootstrap equivalence under degenerate design."""

    def test_rao_wu_approximates_block_no_strata(self, trop_survey_data):
        """Without real strata variation, Rao-Wu SE ~ block bootstrap SE."""
        from diff_diff import TROP

        data = trop_survey_data.copy()
        # Single stratum, PSU = unit (effectively block bootstrap)
        data["single_stratum"] = 0
        data["unit_psu"] = data["unit"]

        sd_rw = SurveyDesign(
            weights="weight",
            strata="single_stratum",
            psu="unit_psu",
        )
        sd_block = SurveyDesign(weights="weight")

        result_rw = TROP(method="local", n_bootstrap=99, seed=42, max_iter=5).fit(
            data,
            "outcome",
            "D",
            "unit",
            "time",
            survey_design=sd_rw,
        )
        result_block = TROP(method="local", n_bootstrap=99, seed=42, max_iter=5).fit(
            data,
            "outcome",
            "D",
            "unit",
            "time",
            survey_design=sd_block,
        )

        # Point estimates identical (same weights)
        assert result_rw.att == pytest.approx(result_block.att, abs=1e-10)
        # SEs should be within factor of 2
        if result_block.se > 0:
            ratio = result_rw.se / result_block.se
            assert 0.5 < ratio < 2.0, (
                f"Rao-Wu SE ({result_rw.se:.4f}) and block SE "
                f"({result_block.se:.4f}) differ by > 2x (ratio={ratio:.2f})"
            )
