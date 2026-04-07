"""Tests for Phase 6 survey support: Subpopulation, DEFF, Replicate Weights."""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DEFFDiagnostics,
    DifferenceInDifferences,
    SurveyDesign,
    compute_deff_diagnostics,
)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def basic_did_data():
    """Simple 2x2 DiD panel with survey design columns."""
    np.random.seed(42)
    n_units = 80
    n_periods = 4
    rows = []
    for unit in range(n_units):
        treated = unit < 40
        ft = 3 if treated else 0
        stratum = unit // 20  # 4 strata
        psu = unit // 10  # 8 PSUs
        wt = 1.0 + 0.5 * (unit % 5)

        for t in range(1, n_periods + 1):
            y = 10.0 + unit * 0.05 + t * 0.3
            if treated and t >= 3:
                y += 2.0
            y += np.random.normal(0, 0.5)
            rows.append({
                "unit": unit, "time": t, "first_treat": ft,
                "outcome": y, "stratum": stratum, "psu": psu,
                "weight": wt, "fpc": 200.0,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Subpopulation Analysis
# =============================================================================


class TestSubpopulationAnalysis:
    """Tests for SurveyDesign.subpopulation()."""

    def test_basic_subpopulation(self, basic_did_data):
        """Subpopulation returns a new SurveyDesign and DataFrame."""
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        mask = basic_did_data["unit"] < 60  # Exclude last 20 units
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        assert new_sd.weights == "_subpop_weight"
        assert new_sd.strata == "stratum"
        assert new_sd.psu == "psu"
        assert "_subpop_weight" in new_data.columns

        # Excluded obs have zero weight
        excluded = new_data[~mask.values]
        assert (excluded["_subpop_weight"] == 0.0).all()

        # Included obs have original weights
        included = new_data[mask.values]
        np.testing.assert_array_equal(
            included["_subpop_weight"].values,
            basic_did_data.loc[mask.values, "weight"].values,
        )

    def test_subpopulation_preserves_design_structure(self, basic_did_data):
        """PSU/strata structure is preserved even when some PSUs are zeroed."""
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        # Exclude all units in stratum 0
        mask = basic_did_data["stratum"] != 0
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        # Should resolve without error — stratum 0 PSUs still exist in design
        resolved = new_sd.resolve(new_data)
        assert resolved.n_strata == 4  # All strata still present
        assert resolved.n_psu == 8  # All PSUs still present

    def test_subpopulation_with_did(self, basic_did_data):
        """Subpopulation works end-to-end with DifferenceInDifferences."""
        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        mask = basic_did_data["unit"] < 60
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        # Add treated/post columns for the estimator
        new_data["treated"] = (new_data["first_treat"] > 0).astype(int)
        new_data["post"] = (new_data["time"] >= 3).astype(int)
        est = DifferenceInDifferences()
        result = est.fit(
            new_data, outcome="outcome", treatment="treated", time="post",
            survey_design=new_sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_subpopulation_preserves_more_psus_than_subset(self, basic_did_data):
        """Subpopulation retains all PSUs in the design; subset drops them."""
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        mask = basic_did_data["unit"] < 60

        # Subpopulation: preserves full design
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)
        resolved_subpop = new_sd.resolve(new_data)

        # Naive subset: loses PSUs in excluded observations
        subset = basic_did_data[mask.values].copy()
        resolved_subset = sd.resolve(subset)

        # Subpopulation preserves the full PSU count
        assert resolved_subpop.n_psu >= resolved_subset.n_psu

    def test_subpopulation_mask_as_column_name(self, basic_did_data):
        """Mask can be a column name string."""
        sd = SurveyDesign(weights="weight")
        data = basic_did_data.copy()
        data["include"] = data["unit"] < 60
        new_sd, new_data = sd.subpopulation(data, "include")
        assert (new_data.loc[~data["include"], "_subpop_weight"] == 0.0).all()

    def test_subpopulation_mask_as_callable(self, basic_did_data):
        """Mask can be a callable."""
        sd = SurveyDesign(weights="weight")
        new_sd, new_data = sd.subpopulation(
            basic_did_data, lambda df: df["unit"] < 60,
        )
        excluded = new_data[basic_did_data["unit"] >= 60]
        assert (excluded["_subpop_weight"] == 0.0).all()

    def test_subpopulation_no_weights_uses_uniform(self, basic_did_data):
        """When SurveyDesign has no weights, subpopulation uses 1.0."""
        sd = SurveyDesign(strata="stratum")
        mask = basic_did_data["unit"] < 60
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)

        included = new_data[mask.values]
        assert (included["_subpop_weight"] == 1.0).all()

    def test_subpopulation_all_excluded_raises(self, basic_did_data):
        """Excluding all obs should raise ValueError."""
        sd = SurveyDesign(weights="weight")
        mask = np.zeros(len(basic_did_data), dtype=bool)
        with pytest.raises(ValueError, match="excludes all"):
            sd.subpopulation(basic_did_data, mask)

    def test_subpopulation_none_excluded_identity(self, basic_did_data):
        """Including all obs should return weights unchanged."""
        sd = SurveyDesign(weights="weight")
        mask = np.ones(len(basic_did_data), dtype=bool)
        new_sd, new_data = sd.subpopulation(basic_did_data, mask)
        np.testing.assert_array_equal(
            new_data["_subpop_weight"].values,
            basic_did_data["weight"].values,
        )

    def test_subpopulation_mask_length_mismatch_raises(self, basic_did_data):
        """Mask length must match data length."""
        sd = SurveyDesign(weights="weight")
        mask = np.ones(10, dtype=bool)
        with pytest.raises(ValueError, match="length"):
            sd.subpopulation(basic_did_data, mask)

    def test_zero_weights_allowed_in_resolve(self):
        """resolve() should accept zero weights (no longer strictly positive)."""
        data = pd.DataFrame({
            "w": [1.0, 2.0, 0.0, 1.5],
            "y": [1, 2, 3, 4],
        })
        sd = SurveyDesign(weights="w")
        resolved = sd.resolve(data)
        assert resolved.weights[2] == 0.0  # Zero preserved after normalization

    def test_negative_weights_rejected(self):
        """Negative weights should still be rejected."""
        data = pd.DataFrame({"w": [1.0, -1.0, 2.0], "y": [1, 2, 3]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="non-negative"):
            sd.resolve(data)


# =============================================================================
# DEFF Diagnostics
# =============================================================================


class TestDEFFDiagnostics:
    """Tests for per-coefficient design effect diagnostics."""

    @staticmethod
    def _fit_lr_with_survey(data, sd):
        """Fit a LinearRegression with survey design for testing DEFF."""
        from diff_diff.linalg import LinearRegression
        from diff_diff.survey import _resolve_survey_for_fit

        resolved, sw, swt, sm = _resolve_survey_for_fit(sd, data, "analytical")
        y = data["outcome"].values
        X = np.column_stack([
            np.ones(len(data)),
            data["treated"].values,
            data["post"].values,
            (data["treated"] * data["post"]).values,
        ])
        reg = LinearRegression(
            include_intercept=False, weights=sw,
            weight_type=swt, survey_design=resolved,
        )
        reg.fit(X, y)
        return reg

    def test_uniform_weights_deff_near_one(self, basic_did_data):
        """With uniform weights and no clustering, DEFF should be ~1."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)
        data["uniform_wt"] = 1.0

        sd = SurveyDesign(weights="uniform_wt")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        assert isinstance(deff, DEFFDiagnostics)
        # With uniform weights, DEFF should be close to 1
        for d in deff.deff:
            if np.isfinite(d):
                assert d == pytest.approx(1.0, abs=0.15)

    def test_clustered_design_deff_gt_one(self, basic_did_data):
        """With clustered design, DEFF should be > 1 for at least some coefs."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        # At least one coefficient should have DEFF > 1 (design effect)
        assert np.any(deff.deff[np.isfinite(deff.deff)] > 1.0)

    def test_deff_srs_se_matches_robust_vcov(self, basic_did_data):
        """SRS SE in DEFF diagnostics should match compute_robust_vcov."""
        from diff_diff.linalg import compute_robust_vcov

        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()

        # Manually compute SRS SE
        srs_vcov = compute_robust_vcov(
            reg._X, reg.residuals_,
            cluster_ids=None, weights=reg.weights,
            weight_type=reg.weight_type,
        )
        expected_srs_se = np.sqrt(np.diag(srs_vcov))
        np.testing.assert_allclose(deff.srs_se, expected_srs_se, rtol=1e-10)

    def test_standalone_compute_deff_diagnostics(self, basic_did_data):
        """compute_deff_diagnostics() works as a standalone function."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = compute_deff_diagnostics(
            reg._X, reg.residuals_, reg.vcov_,
            reg.weights, weight_type=reg.weight_type,
            coefficient_names=["intercept", "treated", "post", "treated:post"],
        )
        assert deff.coefficient_names is not None
        assert len(deff.coefficient_names) == 4
        assert len(deff.deff) == 4

    def test_deff_effective_n(self, basic_did_data):
        """effective_n should be n / DEFF."""
        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        reg = self._fit_lr_with_survey(data, sd)
        deff = reg.compute_deff()
        n = reg.n_obs_
        for i in range(len(deff.deff)):
            if np.isfinite(deff.deff[i]) and deff.deff[i] > 0:
                assert deff.effective_n[i] == pytest.approx(
                    n / deff.deff[i], rel=1e-10
                )

    def test_compute_deff_no_survey_raises(self, basic_did_data):
        """compute_deff() without survey should raise."""
        from diff_diff.linalg import LinearRegression

        data = basic_did_data.copy()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)

        y = data["outcome"].values
        X = np.column_stack([np.ones(len(data)), data["treated"].values])
        reg = LinearRegression(include_intercept=False)
        reg.fit(X, y)
        with pytest.raises(ValueError, match="survey design"):
            reg.compute_deff()


# =============================================================================
# Replicate Weight Variance
# =============================================================================


@pytest.fixture
def replicate_data():
    """Data with full-sample + replicate weight columns for testing."""
    np.random.seed(123)
    n = 200
    k = 20  # number of replicates

    # Simple regression: y = 1 + 2*x + eps
    x = np.random.randn(n)
    eps = np.random.randn(n) * 0.5
    y = 1.0 + 2.0 * x + eps

    # Full-sample weights (non-uniform)
    w = 1.0 + np.random.exponential(0.5, n)

    data = pd.DataFrame({"x": x, "y": y, "weight": w})

    # Generate JK1 replicate weights (delete-1 jackknife on PSU clusters)
    # Each replicate drops one cluster and rescales remaining weights
    cluster_size = n // k
    for r in range(k):
        w_r = w.copy()
        start = r * cluster_size
        end = min((r + 1) * cluster_size, n)
        w_r[start:end] = 0.0
        # Rescale remaining: multiply by k/(k-1)
        w_r[w_r > 0] *= k / (k - 1)
        data[f"rep_{r}"] = w_r

    return data, [f"rep_{r}" for r in range(k)]


class TestReplicateWeightVariance:
    """Tests for replicate weight variance estimation."""

    def test_brr_validation(self):
        """BRR validation: method + weights required together."""
        with pytest.raises(ValueError, match="replicate_method must be provided"):
            SurveyDesign(weights="w", replicate_weights=["r1", "r2"])
        with pytest.raises(ValueError, match="replicate_weights must be provided"):
            SurveyDesign(weights="w", replicate_method="BRR")

    def test_fay_rho_validation(self):
        """Fay requires rho in (0, 1)."""
        with pytest.raises(ValueError, match="fay_rho"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="Fay", fay_rho=0.0,
            )
        with pytest.raises(ValueError, match="fay_rho"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="BRR", fay_rho=0.5,
            )

    def test_replicate_strata_mutual_exclusion(self):
        """Replicate weights cannot be combined with strata/psu/fpc."""
        with pytest.raises(ValueError, match="cannot be combined"):
            SurveyDesign(
                weights="w", strata="s",
                replicate_weights=["r1", "r2"], replicate_method="BRR",
            )

    def test_replicate_requires_weights(self):
        """Full-sample weights required alongside replicates."""
        with pytest.raises(ValueError, match="Full-sample weights"):
            SurveyDesign(
                replicate_weights=["r1", "r2"], replicate_method="BRR",
            )

    def test_jk1_resolve(self, replicate_data):
        """JK1 replicates resolve correctly."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        assert resolved.uses_replicate_variance
        assert resolved.n_replicates == len(rep_cols)
        assert resolved.replicate_weights.shape == (len(data), len(rep_cols))
        assert resolved.strata is None  # Short-circuited
        assert resolved.df_survey == len(rep_cols) - 1

    def test_jk1_vcov_finite(self, replicate_data):
        """JK1 replicate vcov produces finite results."""
        from diff_diff.linalg import LinearRegression

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        reg = LinearRegression(
            include_intercept=False, weights=resolved.weights,
            weight_type="pweight", survey_design=resolved,
        )
        reg.fit(X, y)

        assert np.all(np.isfinite(reg.coefficients_))
        assert np.all(np.isfinite(np.diag(reg.vcov_)))
        # SE should be positive
        se = np.sqrt(np.diag(reg.vcov_))
        assert np.all(se > 0)

    def test_brr_vcov(self, replicate_data):
        """BRR uses 1/R factor."""
        from diff_diff.survey import compute_replicate_vcov

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        from diff_diff.linalg import solve_ols
        coef, _, _ = solve_ols(
            X, y, weights=resolved.weights, weight_type="pweight",
        )

        vcov, _nv = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))

    def test_fay_inflates_over_brr(self, replicate_data):
        """Fay's BRR with rho > 0 produces larger variance than BRR."""
        from diff_diff.survey import compute_replicate_vcov
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_brr = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_brr.weights)
        vcov_brr, _nv = compute_replicate_vcov(X, y, coef, resolved_brr)

        sd_fay = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="Fay", fay_rho=0.5,
        )
        resolved_fay = sd_fay.resolve(data)
        vcov_fay, _nv = compute_replicate_vcov(X, y, coef, resolved_fay)

        # Fay variance = BRR variance / (1-rho)^2 > BRR variance
        assert np.all(np.diag(vcov_fay) > np.diag(vcov_brr))

    def test_replicate_metadata(self, replicate_data):
        """SurveyMetadata should include replicate info."""
        from diff_diff.survey import compute_survey_metadata

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)
        sm = compute_survey_metadata(resolved, data["weight"].values)
        assert sm.replicate_method == "JK1"
        assert sm.n_replicates == len(rep_cols)
        assert sm.df_survey == len(rep_cols) - 1

    def test_replicate_rejected_by_base_did(self, replicate_data):
        """DifferenceInDifferences rejects replicate-weight designs."""
        data, rep_cols = replicate_data
        n = len(data)
        data["treated"] = (np.arange(n) < n // 2).astype(int)
        data["post"] = (np.arange(n) % 4 >= 2).astype(int)
        data["outcome"] = data["y"] + 1.5 * data["treated"] * data["post"]

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="DifferenceInDifferences"):
            DifferenceInDifferences().fit(
                data, outcome="outcome", treatment="treated", time="post",
                survey_design=sd,
            )

    def test_replicate_if_variance(self, replicate_data):
        """IF-based replicate variance produces finite results."""
        from diff_diff.survey import compute_replicate_if_variance

        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        resolved = sd.resolve(data)

        # Synthetic influence function
        psi = np.random.randn(len(data)) * 0.1
        var, _nv = compute_replicate_if_variance(psi, resolved)
        assert np.isfinite(var)
        assert var >= 0

    def test_min_2_replicates(self, replicate_data):
        """Need at least 2 replicate columns."""
        data, _ = replicate_data
        sd = SurveyDesign(
            weights="weight", replicate_weights=["rep_0"],
            replicate_method="BRR",
        )
        with pytest.raises(ValueError, match="At least 2"):
            sd.resolve(data)

    def test_jkn_requires_replicate_strata(self):
        """JKn must have replicate_strata."""
        with pytest.raises(ValueError, match="replicate_strata is required"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JKn",
            )

    def test_jkn_replicate_strata_length_mismatch(self):
        """replicate_strata length must match replicate_weights."""
        with pytest.raises(ValueError, match="length"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JKn", replicate_strata=[0],
            )

    def test_jkn_variance(self, replicate_data):
        """JKn with per-stratum scaling produces finite results."""
        from diff_diff.survey import compute_replicate_vcov
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        # Assign first half of replicates to stratum 0, second half to stratum 1
        n_rep = len(rep_cols)
        strata = [0] * (n_rep // 2) + [1] * (n_rep - n_rep // 2)

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JKn", replicate_strata=strata,
        )
        resolved = sd.resolve(data)
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        coef, _, _ = solve_ols(X, y, weights=resolved.weights)
        vcov, _nv = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))
        assert np.all(np.diag(vcov) > 0)

    def test_replicate_if_scale_matches_analytical(self):
        """Replicate IF variance should match analytical sum(psi^2) for SRS."""
        from diff_diff.survey import compute_replicate_if_variance

        np.random.seed(42)
        n = 50
        psi = np.random.randn(n) * 0.1

        # Build JK1 replicates: delete-1 jackknife
        w_full = np.ones(n)
        rep_cols = []
        data = pd.DataFrame({"w": w_full})
        for r in range(n):
            w_r = np.ones(n) * n / (n - 1)
            w_r[r] = 0.0
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="w", replicate_weights=rep_cols, replicate_method="JK1",
        )
        resolved = sd.resolve(data)

        v_rep, _nv = compute_replicate_if_variance(psi, resolved)
        v_analytical = float(np.sum(psi**2))

        # JK1 gives (n-1)/n * sum(...) which should approximate sum(psi^2)
        assert v_rep == pytest.approx(v_analytical, rel=0.05), (
            f"Replicate IF variance {v_rep:.6f} does not match analytical "
            f"{v_analytical:.6f}"
        )

    def test_replicate_if_matches_survey_if_variance(self):
        """Replicate IF variance should approximate compute_survey_if_variance
        when JK1 replicates match the PSU structure."""
        from diff_diff.survey import (
            compute_replicate_if_variance,
            compute_survey_if_variance,
            ResolvedSurveyDesign,
        )

        np.random.seed(123)
        n = 60
        n_psu = 15
        psu = np.repeat(np.arange(n_psu), n // n_psu)
        psi = np.random.randn(n) * 0.1
        weights = np.ones(n)

        # TSL variance with PSU clustering
        resolved_tsl = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=psu, fpc=None,
            n_strata=0, n_psu=n_psu, lonely_psu="remove",
        )
        v_tsl = compute_survey_if_variance(psi, resolved_tsl)

        # JK1 replicates: delete one PSU at a time, rescale remaining
        rep_arr = np.zeros((n, n_psu))
        for r in range(n_psu):
            w_r = weights.copy()
            w_r[psu == r] = 0.0
            w_r[psu != r] *= n_psu / (n_psu - 1)
            rep_arr[:, r] = w_r
        # Normalize each column to sum=n (matching resolve() normalization)
        for c in range(n_psu):
            cs = rep_arr[:, c].sum()
            if cs > 0:
                rep_arr[:, c] *= n / cs

        resolved_rep = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1",
            n_replicates=n_psu,
        )
        v_rep, _nv = compute_replicate_if_variance(psi, resolved_rep)

        # Should be in the same ballpark (within 50% — different estimators
        # of the same quantity)
        assert v_rep > 0 and v_tsl > 0
        ratio = v_rep / v_tsl
        assert 0.5 < ratio < 2.0, (
            f"Replicate IF var ({v_rep:.6f}) vs TSL IF var ({v_tsl:.6f}) "
            f"ratio={ratio:.2f}, expected within [0.5, 2.0]"
        )

    def test_all_zero_weights_rejected_by_solve_ols(self):
        """solve_ols should reject all-zero weight vectors."""
        from diff_diff.linalg import solve_ols

        X = np.column_stack([np.ones(10), np.random.randn(10)])
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="sum to zero"):
            solve_ols(X, y, weights=np.zeros(10))

    def test_solve_logit_rejects_single_class_positive_weight(self):
        """solve_logit should reject when positive-weight obs have one class."""
        from diff_diff.linalg import solve_logit

        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.array([1] * 10 + [0] * 10, dtype=float)
        w = np.ones(n)
        # Zero out all class-0 obs — only class 1 remains
        w[10:] = 0.0
        with pytest.raises(ValueError, match="outcome class"):
            solve_logit(X, y, weights=w)

    def test_solve_logit_rejects_too_few_positive_weight_obs(self):
        """solve_logit should reject when too few positive-weight obs."""
        from diff_diff.linalg import solve_logit

        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.array([1] * 10 + [0] * 10, dtype=float)
        w = np.zeros(n)
        # Only 2 positive-weight obs (one per class), but 2 params
        w[0] = 1.0  # class 1
        w[10] = 1.0  # class 0
        with pytest.raises(ValueError, match="Cannot identify"):
            solve_logit(X, y, weights=w)

    def test_solve_logit_rank_deficient_positive_weight_subset_error_mode(self):
        """solve_logit with rank_deficient_action='error' should reject when
        positive-weight subset is rank-deficient."""
        from diff_diff.linalg import solve_logit

        n = 20
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([x1, x2])
        y = np.array([1] * 10 + [0] * 10, dtype=float)
        w = np.zeros(n)
        for i in [0, 1, 2, 10, 11, 12]:
            w[i] = 1.0
            X[i, 1] = 5.0  # x2 constant → rank deficient in subset
        with pytest.raises(ValueError, match="rank-deficient"):
            solve_logit(X, y, weights=w, rank_deficient_action="error")

    def test_solve_logit_rank_deficient_positive_weight_subset_warn_mode(self):
        """solve_logit with rank_deficient_action='warn' should warn but not
        error on rank-deficient positive-weight subset."""
        from diff_diff.linalg import solve_logit

        n = 20
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([x1, x2])
        y = np.array([1] * 10 + [0] * 10, dtype=float)
        w = np.zeros(n)
        for i in [0, 1, 2, 10, 11, 12]:
            w[i] = 1.0
            X[i, 1] = 5.0
        with pytest.warns(UserWarning, match="rank-deficient"):
            beta, probs = solve_logit(X, y, weights=w, rank_deficient_action="warn")
        # Key assertion: beta must have original p+1 length (intercept + 2 covariates)
        assert len(beta) == 3, (
            f"Expected beta length 3 (p+1), got {len(beta)}. "
            f"Column dropping broke the coefficient vector shape."
        )

    def test_subpopulation_string_mask_rejected(self, basic_did_data):
        """Subpopulation mask with string values should be rejected."""
        sd = SurveyDesign(weights="weight")
        mask = ["yes"] * len(basic_did_data)
        mask[0] = "no"
        with pytest.raises(ValueError, match="string"):
            sd.subpopulation(basic_did_data, mask)

    def test_nonbinary_numeric_mask_rejected(self, basic_did_data):
        """Subpopulation mask with non-binary numeric codes should be rejected."""
        sd = SurveyDesign(weights="weight")
        # Coded domain column {1, 2} — not boolean, should fail
        mask = np.ones(len(basic_did_data), dtype=int)
        mask[:10] = 2
        with pytest.raises(ValueError, match="non-binary"):
            sd.subpopulation(basic_did_data, mask)

    def test_scale_invariance_combined_weights(self):
        """Multiplying all replicate columns by a constant should not change SE."""
        from diff_diff.survey import compute_replicate_if_variance, ResolvedSurveyDesign

        np.random.seed(42)
        n = 40
        R = 10
        psi = np.random.randn(n) * 0.1
        weights = 1.0 + np.random.exponential(0.3, n)

        # Build replicate weights (combined: include full-sample weight)
        rep_arr = np.zeros((n, R))
        cluster_size = n // R
        for r in range(R):
            w_r = weights.copy()
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n)
            w_r[start:end] = 0.0
            w_r[w_r > 0] *= R / (R - 1)
            rep_arr[:, r] = w_r

        resolved1 = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
            combined_weights=True,
        )
        v1, _ = compute_replicate_if_variance(psi, resolved1)

        # Scale all replicate columns by 3x
        resolved2 = ResolvedSurveyDesign(
            weights=weights * 3, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr * 3,
            replicate_method="JK1", n_replicates=R,
            combined_weights=True,
        )
        v2, _ = compute_replicate_if_variance(psi, resolved2)

        assert v1 == pytest.approx(v2, rel=1e-10), (
            f"Scale invariance violated: v1={v1:.8f} vs v2={v2:.8f}"
        )

    def test_combined_weights_false(self):
        """combined_weights=False treats replicate cols as perturbation factors."""
        sd = SurveyDesign(
            weights="w", replicate_weights=["r1", "r2"],
            replicate_method="BRR", combined_weights=False,
        )
        assert sd.combined_weights is False

    def test_custom_scale(self):
        """Custom replicate_scale overrides default factor."""
        sd = SurveyDesign(
            weights="w", replicate_weights=["r1", "r2"],
            replicate_method="BRR", replicate_scale=0.5,
        )
        assert sd.replicate_scale == 0.5

    def test_scale_rscales_multiplicative(self):
        """scale and rscales are applied multiplicatively for JK1."""
        from diff_diff.survey import compute_replicate_vcov, ResolvedSurveyDesign
        from diff_diff.linalg import solve_ols

        np.random.seed(42)
        n, R = 100, 5
        x = np.random.randn(n)
        y = 1.0 + 2.0 * x + np.random.randn(n) * 0.5
        X = np.column_stack([np.ones(n), x])
        w = np.ones(n)

        cluster_size = n // R
        rep_arr = np.ones((n, R))
        for r in range(R):
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n)
            rep_arr[start:end, r] = 0.0
            rep_arr[~((np.arange(n) >= start) & (np.arange(n) < end)), r] *= R / (R - 1)

        coef, _, _ = solve_ols(X, y, weights=w)

        # rscales only
        rscales = np.array([0.5, 1.0, 1.5, 1.0, 0.5])
        resolved_rscales = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
            replicate_rscales=rscales,
        )
        vcov_rscales, _ = compute_replicate_vcov(X, y, coef, resolved_rscales)

        # scale * rscales should multiply
        scale_val = 2.0
        resolved_both = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
            replicate_scale=scale_val, replicate_rscales=rscales,
        )
        vcov_both, _ = compute_replicate_vcov(X, y, coef, resolved_both)

        # V(scale+rscales) == scale * V(rscales)
        assert np.allclose(np.diag(vcov_both), scale_val * np.diag(vcov_rscales), rtol=1e-10)

    def test_brr_ignores_custom_scaling(self):
        """BRR ignores custom scale/rscales with a warning."""
        import warnings
        from diff_diff.survey import compute_replicate_vcov, ResolvedSurveyDesign
        from diff_diff.linalg import solve_ols

        np.random.seed(42)
        n, R = 100, 5
        x = np.random.randn(n)
        y = 1.0 + 2.0 * x + np.random.randn(n) * 0.5
        X = np.column_stack([np.ones(n), x])
        w = np.ones(n)
        rep_arr = np.random.uniform(0.8, 1.2, (n, R))

        coef, _, _ = solve_ols(X, y, weights=w)

        # Default BRR
        resolved_default = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="BRR", n_replicates=R,
        )
        vcov_default, _ = compute_replicate_vcov(X, y, coef, resolved_default)

        # BRR with custom scale (should be ignored with warning)
        resolved_custom = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="BRR", n_replicates=R,
            replicate_scale=99.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vcov_custom, _ = compute_replicate_vcov(X, y, coef, resolved_custom)
        assert any("ignored" in str(w.message) for w in caught)
        assert np.allclose(np.diag(vcov_default), np.diag(vcov_custom), rtol=1e-10)

    def test_replicate_if_no_divide_by_zero_warning(self):
        """compute_replicate_if_variance should not warn on zero weights."""
        from diff_diff.survey import compute_replicate_if_variance, ResolvedSurveyDesign

        n = 20
        psi = np.random.randn(n)
        weights = np.ones(n)
        weights[:5] = 0.0  # Some zero full-sample weights (subpopulation)

        rep_arr = np.ones((n, 5))
        rep_arr[:5, :] = 0.0  # Match zero pattern
        for r in range(5):
            cs = rep_arr[:, r].sum()
            if cs > 0:
                rep_arr[:, r] *= n / cs

        resolved = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=5,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should NOT raise RuntimeWarning for divide by zero
            v, _nv = compute_replicate_if_variance(psi, resolved)
            assert np.isfinite(v)


class TestReplicateEdgeCases:
    """Regression tests for analysis-weight df and rscales centering."""

    def test_df_survey_combined_weights_false(self):
        """df_survey uses analysis-weight rank when combined_weights=False."""
        from diff_diff.survey import ResolvedSurveyDesign

        np.random.seed(42)
        n = 50
        R = 5
        weights = 1.0 + np.random.exponential(0.5, n)
        # Perturbation factors (not full weights)
        rep_factors = np.random.uniform(0.8, 1.2, (n, R))

        resolved = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_factors,
            replicate_method="BRR", n_replicates=R,
            combined_weights=False,
        )
        # df should match pivoted QR rank of analysis weights
        analysis_weights = rep_factors * weights[:, np.newaxis]
        from scipy.linalg import qr as scipy_qr
        _, R_mat, _ = scipy_qr(analysis_weights, pivoting=True, mode='economic')
        diag_abs = np.abs(np.diag(R_mat))
        expected_rank = int(np.sum(diag_abs > 1e-5 * diag_abs.max()))
        expected_df = expected_rank - 1 if expected_rank > 1 else None
        assert resolved.df_survey == expected_df

        # Verify it differs from raw perturbation-factor rank when weights
        # cause a rank reduction (e.g., zero full-sample weights)
        weights_with_zeros = weights.copy()
        weights_with_zeros[:10] = 0.0  # subpopulation-zeroed
        resolved2 = ResolvedSurveyDesign(
            weights=weights_with_zeros, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_factors,
            replicate_method="BRR", n_replicates=R,
            combined_weights=False,
        )
        analysis_z = rep_factors * weights_with_zeros[:, np.newaxis]
        _, R_z, _ = scipy_qr(analysis_z, pivoting=True, mode='economic')
        diag_z = np.abs(np.diag(R_z))
        z_rank = int(np.sum(diag_z > 1e-5 * diag_z.max())) if diag_z.max() > 0 else 0
        z_df = z_rank - 1 if z_rank > 1 else None
        assert resolved2.df_survey == z_df

    def test_rscales_zero_centering_vcov(self):
        """mse=False with zero rscales: center only on rscales > 0 replicates."""
        from diff_diff.survey import compute_replicate_vcov, ResolvedSurveyDesign
        from diff_diff.linalg import solve_ols

        np.random.seed(42)
        n = 100
        R = 6
        x = np.random.randn(n)
        y = 1.0 + 2.0 * x + np.random.randn(n) * 0.5
        X = np.column_stack([np.ones(n), x])
        w = np.ones(n)

        # Build JK1-style replicates
        cluster_size = n // R
        rep_arr = np.ones((n, R))
        for r in range(R):
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n)
            rep_arr[start:end, :] = 0.0
            # Correct column r only
            rep_arr[:, r] = np.where(
                (np.arange(n) >= start) & (np.arange(n) < end), 0.0,
                R / (R - 1)
            )

        # rscales with one zero entry
        rscales = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

        coef, _, _ = solve_ols(X, y, weights=w)

        resolved = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
            replicate_rscales=rscales, mse=False,
        )
        vcov, _nv = compute_replicate_vcov(X, y, coef, resolved)

        # Manual computation: center only on replicates with rscales > 0
        coef_reps = []
        for r in range(R):
            c_r, _, _ = solve_ols(X, y, weights=rep_arr[:, r])
            coef_reps.append(c_r)
        coef_reps = np.array(coef_reps)
        pos_mask = rscales > 0
        center = np.mean(coef_reps[pos_mask], axis=0)
        diffs = coef_reps - center[np.newaxis, :]
        V_manual = np.zeros((2, 2))
        for r in range(R):
            V_manual += rscales[r] * np.outer(diffs[r], diffs[r])

        assert np.allclose(np.diag(vcov), np.diag(V_manual), rtol=1e-10)

    def test_rscales_zero_centering_if(self):
        """mse=False with zero rscales: IF path centers only on rscales > 0."""
        from diff_diff.survey import compute_replicate_if_variance, ResolvedSurveyDesign

        np.random.seed(42)
        n = 50
        R = 5
        psi = np.random.randn(n) * 0.1
        w = np.ones(n)

        # Build simple replicates
        rep_arr = np.ones((n, R))
        for r in range(R):
            start = r * (n // R)
            end = min((r + 1) * (n // R), n)
            rep_arr[start:end, r] = 0.0
            rep_arr[:, r] = np.where(
                (np.arange(n) >= start) & (np.arange(n) < end), 0.0,
                R / (R - 1)
            )

        rscales = np.array([1.0, 0.0, 1.0, 1.0, 1.0])

        resolved = ResolvedSurveyDesign(
            weights=w, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
            replicate_rscales=rscales, mse=False,
        )
        var, _nv = compute_replicate_if_variance(psi, resolved)

        # Manual: theta_r = sum((w_r/w) * psi), center on rscales > 0 only
        theta_full = float(np.sum(psi))
        theta_reps = np.array([
            float(np.sum(np.divide(rep_arr[:, r], w, out=np.zeros(n), where=w > 0) * psi))
            for r in range(R)
        ])
        pos_mask = rscales > 0
        center = float(np.mean(theta_reps[pos_mask]))
        diffs = theta_reps - center
        var_manual = float(np.sum(rscales * diffs**2))

        assert var == pytest.approx(var_manual, rel=1e-10)

    def test_rank_one_replicate_nan_inference(self):
        """Rank-1 replicate design yields df_survey=None and NaN inference."""
        from diff_diff.survey import ResolvedSurveyDesign
        from diff_diff.utils import safe_inference

        np.random.seed(42)
        n = 50
        R = 3
        weights = np.ones(n)
        # All replicate columns identical → rank 1
        col = 1.0 + np.random.randn(n) * 0.1
        rep_arr = np.column_stack([col, col, col])

        resolved = ResolvedSurveyDesign(
            weights=weights, weight_type="pweight",
            strata=None, psu=None, fpc=None,
            n_strata=0, n_psu=0, lonely_psu="remove",
            replicate_weights=rep_arr,
            replicate_method="JK1", n_replicates=R,
        )
        # Rank-1 → df_survey should be None
        assert resolved.df_survey is None

        # Passing df=0 (the guard value) should produce NaN p-value and CI
        _, p_val, ci = safe_inference(1.0, 0.5, df=0)
        assert np.isnan(p_val)
        assert np.isnan(ci[0]) and np.isnan(ci[1])


# =============================================================================
# Estimator-Level Replicate Weight Tests
# =============================================================================


class TestEstimatorReplicateWeights:
    """End-to-end tests for replicate weights with each estimator."""

    @staticmethod
    def _make_staggered_replicate_data():
        """Create staggered panel with replicate weight columns."""
        np.random.seed(42)
        n_units, n_periods = 60, 6
        n_rep = 15
        rows = []
        for unit in range(n_units):
            if unit < 20:
                ft = 3
            elif unit < 40:
                ft = 5
            else:
                ft = 0
            wt = 1.0 + 0.3 * (unit % 5)
            for t in range(1, n_periods + 1):
                y = 10.0 + unit * 0.05 + t * 0.2
                if ft > 0 and t >= ft:
                    y += 2.0
                y += np.random.normal(0, 0.5)
                rows.append({
                    "unit": unit, "time": t, "first_treat": ft,
                    "outcome": y, "weight": wt,
                })
        data = pd.DataFrame(rows)
        # Generate JK1 replicates (delete-cluster jackknife)
        cluster_size = n_units // n_rep
        rep_cols = []
        for r in range(n_rep):
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n_units)
            w_r = data["weight"].values.copy()
            mask = (data["unit"] >= start) & (data["unit"] < end)
            w_r[mask] = 0.0
            w_r[~mask] *= n_rep / (n_rep - 1)
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)
        return data, rep_cols

    def test_callaway_santanna_replicate(self):
        """CallawaySantAnna with replicate weights produces finite results."""
        from diff_diff import CallawaySantAnna

        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = CallawaySantAnna(estimation_method="reg", n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_efficient_did_replicate(self):
        """EfficientDiD with replicate weights produces finite results."""
        from diff_diff import EfficientDiD

        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = EfficientDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_subpopulation_preserves_replicate_metadata(self):
        """subpopulation() should preserve replicate design fields."""
        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        mask = data["unit"] < 40
        new_sd, new_data = sd.subpopulation(data, mask)

        assert new_sd.replicate_method == "JK1"
        assert new_sd.replicate_weights == rep_cols
        # Excluded obs have zero replicate weights
        excluded = new_data[~mask.values]
        for col in rep_cols:
            assert (excluded[col] == 0.0).all()

    def test_continuous_did_replicate_bootstrap_rejected(self):
        """ContinuousDiD should reject replicate weights + n_bootstrap > 0."""
        from diff_diff import ContinuousDiD

        data, rep_cols = self._make_staggered_replicate_data()
        data["dose"] = np.where(data["first_treat"] > 0, 1.0 + 0.5 * data["unit"] % 3, 0.0)
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            ContinuousDiD(n_bootstrap=30, seed=42).fit(
                data, "outcome", "unit", "time", "first_treat", "dose",
                survey_design=sd,
            )

    @pytest.mark.parametrize("est_method", ["reg", "ipw", "dr"])
    def test_triple_diff_replicate_all_methods(self, est_method):
        """TripleDifference with replicate weights works for reg/ipw/dr."""
        from diff_diff import TripleDifference

        np.random.seed(42)
        n = 200
        n_rep = 10
        d1 = np.repeat([0, 1], n // 2)
        d2 = np.tile([0, 1], n // 2)
        post = np.random.choice([0, 1], n)
        y = 1.0 + 0.5 * d1 + 0.3 * d2 + 2.0 * d1 * d2 * post + np.random.randn(n) * 0.5
        w = 1.0 + np.random.exponential(0.3, n)

        data = pd.DataFrame({
            "y": y, "d1": d1, "d2": d2, "post": post, "weight": w,
        })
        # Build JK1 replicates
        cluster_size = n // n_rep
        rep_cols = []
        for r in range(n_rep):
            w_r = w.copy()
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n)
            w_r[start:end] = 0.0
            w_r[w_r > 0] *= n_rep / (n_rep - 1)
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = TripleDifference(estimation_method=est_method).fit(
            data, outcome="y", group="d1", partition="d2",
            time="post", survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_sun_abraham_replicate_rejected(self):
        """SunAbraham should reject replicate-weight survey designs."""
        from diff_diff import SunAbraham

        data, rep_cols = self._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="SunAbraham"):
            SunAbraham(n_bootstrap=0).fit(
                data, "outcome", "unit", "time", "first_treat",
                survey_design=sd,
            )


class TestSubpopulationMaskValidation:
    """Tests for subpopulation mask edge cases."""

    def test_nan_mask_rejected(self, basic_did_data):
        """Subpopulation mask with NaN should be rejected."""
        sd = SurveyDesign(weights="weight")
        nan_mask = np.ones(len(basic_did_data))
        nan_mask[0] = np.nan
        with pytest.raises(ValueError, match="NA|NaN|missing"):
            sd.subpopulation(basic_did_data, nan_mask)

    def test_none_mask_rejected(self, basic_did_data):
        """Subpopulation mask with None should be rejected."""
        sd = SurveyDesign(weights="weight")
        mask = [True] * len(basic_did_data)
        mask[0] = None
        with pytest.raises(ValueError, match="None|NA|missing"):
            sd.subpopulation(basic_did_data, mask)

    def test_pd_na_mask_rejected(self, basic_did_data):
        """Subpopulation mask with pd.NA should be rejected."""
        sd = SurveyDesign(weights="weight")
        mask = pd.array([True] * len(basic_did_data), dtype=pd.BooleanDtype())
        mask[0] = pd.NA
        with pytest.raises(ValueError, match="NA|missing"):
            sd.subpopulation(basic_did_data, mask)

    def test_efficient_did_replicate_bootstrap_rejected(self):
        """EfficientDiD should reject replicate weights + n_bootstrap > 0."""
        from diff_diff import EfficientDiD

        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            EfficientDiD(n_bootstrap=30, seed=42).fit(
                data, "outcome", "unit", "time", "first_treat",
                survey_design=sd,
            )

    def test_callaway_santanna_replicate_bootstrap_rejected(self):
        """CallawaySantAnna should reject replicate weights + n_bootstrap > 0."""
        from diff_diff import CallawaySantAnna

        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            CallawaySantAnna(estimation_method="reg", n_bootstrap=30, seed=42).fit(
                data, "outcome", "unit", "time", "first_treat",
                survey_design=sd,
            )

    def test_twfe_replicate_rejected(self):
        """TwoWayFixedEffects should reject replicate-weight designs."""
        from diff_diff.twfe import TwoWayFixedEffects

        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="TwoWayFixedEffects"):
            TwoWayFixedEffects().fit(
                data, outcome="outcome", treatment="first_treat",
                unit="unit", time="time", survey_design=sd,
            )

    def test_stacked_did_replicate_rejected(self):
        """StackedDiD should reject replicate-weight designs."""
        from diff_diff import StackedDiD

        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        with pytest.raises(NotImplementedError, match="StackedDiD"):
            StackedDiD().fit(
                data, outcome="outcome", unit="unit", time="time",
                first_treat="first_treat", survey_design=sd,
            )

    def test_invalid_replicate_scale_rejected(self):
        """Negative or zero replicate_scale should be rejected."""
        with pytest.raises(ValueError, match="positive finite"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JK1", replicate_scale=-1.0,
            )
        with pytest.raises(ValueError, match="positive finite"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JK1", replicate_scale=0.0,
            )

    def test_invalid_replicate_rscales_rejected(self):
        """Negative replicate_rscales should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SurveyDesign(
                weights="w", replicate_weights=["r1", "r2"],
                replicate_method="JK1", replicate_rscales=[-1.0, 1.0],
            )

    def _replicate_sd_and_data(self):
        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        return data, sd

    def test_multi_period_did_replicate_rejected(self):
        """MultiPeriodDiD rejects replicate-weight designs."""
        from diff_diff.estimators import MultiPeriodDiD
        data, sd = self._replicate_sd_and_data()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        data["post"] = (data["time"] >= 3).astype(int)
        with pytest.raises(NotImplementedError):
            MultiPeriodDiD().fit(
                data, outcome="outcome", treatment="treated",
                time="post", survey_design=sd,
            )

    def test_imputation_did_replicate_rejected(self):
        """ImputationDiD rejects replicate-weight designs."""
        from diff_diff.imputation import ImputationDiD
        data, sd = self._replicate_sd_and_data()
        with pytest.raises(NotImplementedError):
            ImputationDiD().fit(
                data, outcome="outcome", unit="unit", time="time",
                first_treat="first_treat", survey_design=sd,
            )

    def test_two_stage_did_replicate_rejected(self):
        """TwoStageDiD rejects replicate-weight designs."""
        from diff_diff.two_stage import TwoStageDiD
        data, sd = self._replicate_sd_and_data()
        with pytest.raises(NotImplementedError):
            TwoStageDiD().fit(
                data, outcome="outcome", unit="unit", time="time",
                first_treat="first_treat", survey_design=sd,
            )

    def test_bacon_replicate_rejected(self):
        """BaconDecomposition rejects replicate-weight designs."""
        from diff_diff.bacon import BaconDecomposition
        data, sd = self._replicate_sd_and_data()
        with pytest.raises(NotImplementedError):
            BaconDecomposition().fit(
                data, outcome="outcome", unit="unit", time="time",
                first_treat="first_treat", survey_design=sd,
            )

    def test_synthetic_did_replicate_rejected(self):
        """SyntheticDiD rejects replicate-weight designs."""
        from diff_diff.synthetic_did import SyntheticDiD
        data, sd = self._replicate_sd_and_data()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        with pytest.raises(NotImplementedError):
            SyntheticDiD().fit(
                data, outcome="outcome", unit="unit", time="time",
                treatment="treated", survey_design=sd,
            )

    @pytest.mark.slow
    def test_trop_replicate_rejected(self):
        """TROP rejects replicate-weight designs."""
        from diff_diff.trop import TROP
        data, sd = self._replicate_sd_and_data()
        data["treated"] = (data["first_treat"] > 0).astype(int)
        with pytest.raises(NotImplementedError):
            TROP().fit(
                data, outcome="outcome", unit="unit", time="time",
                treatment="treated", survey_design=sd,
            )


class TestCSReplicateMethodCoverage:
    """Test that CS replicate weights work for all supported estimation methods."""

    @staticmethod
    def _make_cs_replicate_data():
        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        return data, sd

    @pytest.mark.parametrize("method", ["ipw", "dr"])
    def test_cs_replicate_ipw_dr_no_covariates(self, method):
        """CS replicate weights work for ipw/dr without covariates."""
        from diff_diff import CallawaySantAnna
        data, sd = self._make_cs_replicate_data()
        result = CallawaySantAnna(
            estimation_method=method, n_bootstrap=0,
        ).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None


# =============================================================================
# Effective-sample and d.f. consistency tests
# =============================================================================


class TestEffectiveSampleAndDfConsistency:
    """Tests for P1 effective-sample gates and P2 d.f. metadata consistency."""

    def test_continuous_did_positive_weight_gate(self):
        """ContinuousDiD skips cells where positive-weight count < n_basis."""
        import warnings
        from diff_diff import ContinuousDiD

        np.random.seed(42)
        n_units, n_periods = 60, 6
        n_rep = 10
        rows = []
        for unit in range(n_units):
            if unit < 20:
                ft = 3
            elif unit < 40:
                ft = 5
            else:
                ft = 0
            dose = float(unit % 3 + 1) if ft > 0 else 0.0
            wt = 1.0 + 0.3 * (unit % 5)
            for t in range(1, n_periods + 1):
                y = 10.0 + unit * 0.05 + t * 0.2
                if ft > 0 and t >= ft:
                    y += 2.0 * dose
                y += np.random.normal(0, 0.5)
                rows.append({
                    "unit": unit, "time": t, "first_treat": ft,
                    "outcome": y, "weight": wt, "dose": dose,
                })
        data = pd.DataFrame(rows)

        # Build replicate columns
        cluster_size = n_units // n_rep
        rep_cols = []
        for r in range(n_rep):
            w_r = data["weight"].values.copy()
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n_units)
            mask = (data["unit"] >= start) & (data["unit"] < end)
            w_r[mask] = 0.0
            w_r[~mask] *= n_rep / (n_rep - 1)
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )

        # Zero weights for most of cohort ft=5 (units 20-39) while keeping
        # cohort ft=3 (units 0-19) and controls intact. This leaves too few
        # positive-weight treated units in ft=5 cells, triggering the warning,
        # but ft=3 cells remain valid so the estimator still produces a result.
        subpop_mask = (data["first_treat"] != 5) | (data["unit"] < 22)
        new_sd, new_data = sd.subpopulation(data, subpop_mask)

        # Should emit warning about not enough positive-weight treated units
        # for the cohort-5 cells, but still produce a result from cohort-3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ContinuousDiD(n_bootstrap=0, num_knots=2).fit(
                new_data, "outcome", "unit", "time", "first_treat", "dose",
                survey_design=new_sd,
            )
        weight_warnings = [
            w for w in caught
            if "positive-weight" in str(w.message)
        ]
        assert len(weight_warnings) > 0, "Expected warning about insufficient positive-weight units"
        # Result should still be finite (from the valid cohort-3 cells)
        assert result is not None
        assert np.isfinite(result.overall_att)

    def test_triple_diff_positive_weight_gate(self):
        """TripleDifference falls back to weighted mean when n_eff < n_cols."""
        from diff_diff.triple_diff import TripleDifference

        np.random.seed(42)
        n = 200
        d1 = np.repeat([0, 1], n // 2)
        d2 = np.tile([0, 1], n // 2)
        post = np.tile([0, 0, 1, 1], n // 4)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 1.0 + 0.5 * d1 + 0.3 * d2 + 2.0 * d1 * d2 * post + np.random.randn(n) * 0.5
        w = 1.0 + np.random.exponential(0.3, n)

        # Create a cell (d1=1, d2=1, post=1) where most obs have zero weight
        # but each cell still has SOME positive weight for cell means.
        # With 3 covariates + intercept = 4 columns, n_eff=2 < 4 triggers fallback
        cell_target = (d1 == 1) & (d2 == 1) & (post == 1)
        cell_indices = np.where(cell_target)[0]
        # Zero all but 2 obs in this cell
        for idx in cell_indices[2:]:
            w[idx] = 0.0

        data = pd.DataFrame({
            "y": y, "d1": d1, "d2": d2, "post": post,
            "weight": w, "x1": x1, "x2": x2, "x3": x3,
        })

        sd = SurveyDesign(weights="weight")

        # Should not crash — falls back to weighted mean for the thin cell
        result = TripleDifference(estimation_method="reg").fit(
            data, outcome="y", group="d1", partition="d2",
            time="post", covariates=["x1", "x2", "x3"], survey_design=sd,
        )
        assert result is not None
        assert np.isfinite(result.att)

    def test_cs_replicate_df_metadata_consistency(self):
        """CS survey_metadata.df_survey matches the d.f. used for inference."""
        from diff_diff import CallawaySantAnna

        data, rep_cols = TestEstimatorReplicateWeights._make_staggered_replicate_data()
        n_rep = len(rep_cols)
        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = CallawaySantAnna(estimation_method="reg", n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.df_survey is not None
        assert sm.df_survey == n_rep - 1

        # summary() should include the df line
        summary_str = result.summary()
        assert "Survey d.f." in summary_str

    def test_triple_diff_replicate_df_metadata_consistency(self):
        """TripleDifference survey_metadata.df_survey matches inference d.f."""
        from diff_diff import TripleDifference

        np.random.seed(42)
        n = 200
        n_rep = 10
        d1 = np.repeat([0, 1], n // 2)
        d2 = np.tile([0, 1], n // 2)
        post = np.random.choice([0, 1], n)
        y = 1.0 + 0.5 * d1 + 0.3 * d2 + 2.0 * d1 * d2 * post + np.random.randn(n) * 0.5
        w = 1.0 + np.random.exponential(0.3, n)

        data = pd.DataFrame({
            "y": y, "d1": d1, "d2": d2, "post": post, "weight": w,
        })
        cluster_size = n // n_rep
        rep_cols = []
        for r in range(n_rep):
            w_r = w.copy()
            start = r * cluster_size
            end = min((r + 1) * cluster_size, n)
            w_r[start:end] = 0.0
            w_r[w_r > 0] *= n_rep / (n_rep - 1)
            col = f"rep_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight", replicate_weights=rep_cols,
            replicate_method="JK1",
        )
        result = TripleDifference(estimation_method="reg").fit(
            data, outcome="y", group="d1", partition="d2",
            time="post", survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.df_survey is not None
        assert sm.df_survey == n_rep - 1

        d = result.to_dict()
        assert d["df_survey"] == sm.df_survey
