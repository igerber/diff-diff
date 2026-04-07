"""
Validation tests for EfficientDiD against Chen, Sant'Anna & Xie (2025).

Path 1: HRS empirical replication (Table 6 of the paper)
Path 2: Compustat MC simulations (Tables 4 & 5 patterns)

These tests validate the estimator against published results from
"Efficient Difference-in-Differences and Event Study Estimators."
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, EfficientDiD
from edid_dgp import make_compustat_dgp, true_es_avg, true_overall_att

# =============================================================================
# Data Loaders & Helpers
# =============================================================================

_HRS_FIXTURE = Path(__file__).parent / "data" / "hrs_edid_validation.csv"

# Paper Table 6 reference values: (point_estimate, cluster_robust_se)
TABLE6_EDID = {
    (8, 8): (3072, 806),
    (8, 9): (1112, 637),
    (8, 10): (1038, 817),
    (9, 9): (3063, 690),
    (9, 10): (90, 641),
    (10, 10): (2908, 894),
}
TABLE6_ES = {
    0: (3024, 486),
    1: (692, 471),
    2: (1038, 816),
}
TABLE6_ES_AVG = (1585, 521)

TABLE6_CS_SA = {
    (8, 8): 2826,
    (8, 9): 825,
    (8, 10): 800,
    (9, 9): 3031,
    (9, 10): 107,
    (10, 10): 3092,
}


def _load_hrs_data():
    """Load the committed HRS test fixture."""
    df = pd.read_csv(_HRS_FIXTURE)
    # Ensure correct types
    df["unit"] = df["unit"].astype(int)
    df["time"] = df["time"].astype(int)
    df["first_treat"] = df["first_treat"].astype(float)
    return df


def _get_effect(effects_dict, g, t):
    """Look up ATT(g,t) handling potential float/int key mismatch."""
    for key, val in effects_dict.items():
        if int(key[0]) == g and int(key[1]) == t:
            return val
    raise KeyError(f"ATT({g},{t}) not found in results")


def _assert_close(actual, expected, label, se=None, se_frac=0.1):
    """Assert actual is close to expected, tolerance based on published SE.

    Default tolerance is 0.1 * SE (10% of one standard error). Our actual
    diffs are all < 0.03 SE, so this catches real drift while absorbing the
    4-individual sample difference (656 vs paper's 652).
    """
    if se is not None:
        tol = se_frac * se
    else:
        tol = max(0.05 * abs(expected), 50)
    diff = abs(actual - expected)
    assert diff < tol, (
        f"{label}: expected {expected}, got {actual:.1f} "
        f"(diff={diff:.1f}, tol={tol:.1f})"
    )


def _compute_es_avg(result):
    """Compute ES_avg (Eq 2.3): uniform average over post-treatment horizons."""
    if result.event_study_effects is None:
        raise ValueError("No event study effects; use aggregate='all'")
    es = {
        int(e): d["effect"]
        for e, d in result.event_study_effects.items()
        if int(e) >= 0
    }
    return np.mean(list(es.values()))


_TRUE_ES_AVG_COMPUSTAT = true_es_avg()


def _run_mc_simulation(n_sims, rho, seed=1000, also_cs=False):
    """Run MC simulation and return estimates."""
    edid_estimates = []
    edid_overall_att = []
    edid_overall_ci = []
    edid_overall_se = []
    cs_estimates_list = []

    for i in range(n_sims):
        data = make_compustat_dgp(rho=rho, seed=seed + i)

        edid = EfficientDiD(pt_assumption="all")
        res = edid.fit(
            data, outcome="y", unit="unit", time="time",
            first_treat="first_treat", aggregate="all",
        )
        edid_estimates.append(_compute_es_avg(res))
        edid_overall_att.append(res.overall_att)
        edid_overall_se.append(res.overall_se)
        edid_overall_ci.append(res.overall_conf_int)

        if also_cs:
            cs = CallawaySantAnna(control_group="never_treated")
            cs_res = cs.fit(
                data, outcome="y", unit="unit", time="time",
                first_treat="first_treat", aggregate="event_study",
            )
            cs_estimates_list.append(_compute_es_avg(cs_res))

    return {
        "edid_estimates": np.array(edid_estimates),
        "edid_overall_att": np.array(edid_overall_att),
        "edid_overall_se": np.array(edid_overall_se),
        "edid_overall_ci": np.array(edid_overall_ci),
        "cs_estimates": np.array(cs_estimates_list) if also_cs else None,
    }


# =============================================================================
# Path 1: HRS Empirical Replication (Table 6)
# =============================================================================


@pytest.fixture(scope="module")
def hrs_data():
    """Load HRS validation fixture."""
    if not _HRS_FIXTURE.exists():
        pytest.skip(f"HRS fixture not found at {_HRS_FIXTURE}")
    return _load_hrs_data()


@pytest.fixture(scope="module")
def edid_hrs_result(hrs_data):
    """Fit EDiD on HRS data (shared across tests)."""
    edid = EfficientDiD(pt_assumption="all")
    return edid.fit(
        hrs_data, outcome="outcome", unit="unit", time="time",
        first_treat="first_treat", aggregate="all",
    )


class TestHRSReplication:
    """Validate EDiD against Table 6 of Chen, Sant'Anna & Xie (2025)."""

    def test_sample_selection_yields_expected_counts(self, hrs_data):
        # Fixture is deterministic — assert exact counts
        n_units = hrs_data["unit"].nunique()
        assert n_units == 656, f"Expected 656 units, got {n_units}"

        groups = hrs_data.groupby("unit")["first_treat"].first()

        finite_groups = sorted(g for g in groups.unique() if np.isfinite(g))
        assert finite_groups == [8, 9, 10], f"Expected groups [8,9,10], got {finite_groups}"
        assert any(np.isinf(g) for g in groups.unique()), "Missing never-treated group"

        expected_sizes = {8: 252, 9: 176, 10: 163}
        for g, expected in expected_sizes.items():
            actual = (groups == g).sum()
            assert actual == expected, f"G={g}: expected {expected}, got {actual}"
        n_inf = groups.apply(np.isinf).sum()
        assert n_inf == 65, f"G=inf: expected 65, got {n_inf}"

        assert sorted(hrs_data["time"].unique()) == [7, 8, 9, 10], (
            f"Expected waves [7,8,9,10], got {sorted(hrs_data['time'].unique())}"
        )

    def test_group_time_effects_match_table6(self, edid_hrs_result):
        for (g, t), (expected_effect, se) in TABLE6_EDID.items():
            info = _get_effect(edid_hrs_result.group_time_effects, g, t)
            _assert_close(info["effect"], expected_effect, f"ATT({g},{t})", se=se)

    def test_event_study_effects_match_table6(self, edid_hrs_result):
        for e, (expected_effect, se) in TABLE6_ES.items():
            found = False
            for rel_time, info in edid_hrs_result.event_study_effects.items():
                if int(rel_time) == e:
                    _assert_close(info["effect"], expected_effect, f"ES({e})", se=se)
                    found = True
                    break
            assert found, f"ES({e}) not found in event study effects"

    def test_es_avg_matches_table6(self, edid_hrs_result):
        es_avg = _compute_es_avg(edid_hrs_result)
        _assert_close(es_avg, TABLE6_ES_AVG[0], "ES_avg", se=TABLE6_ES_AVG[1])

    def test_se_diagnostic_comparison(self, edid_hrs_result):
        """Log and sanity-check analytical vs cluster-robust SEs."""
        for (g, t), (_, cluster_se) in TABLE6_EDID.items():
            info = _get_effect(edid_hrs_result.group_time_effects, g, t)
            analytical_se = info["se"]
            assert np.isfinite(analytical_se) and analytical_se > 0, (
                f"ATT({g},{t}): analytical SE should be finite positive, got {analytical_se}"
            )
            ratio = analytical_se / cluster_se
            assert 0.3 < ratio < 3.0, (
                f"ATT({g},{t}): SE ratio (analytical/cluster) = {ratio:.2f} "
                f"outside (0.3, 3.0). Analytical={analytical_se:.1f}, "
                f"cluster={cluster_se}"
            )

    def test_cs_cross_validation(self, hrs_data):
        """Cross-validate data loading using CallawaySantAnna."""
        cs = CallawaySantAnna(control_group="never_treated")
        cs_result = cs.fit(
            hrs_data, outcome="outcome", unit="unit", time="time",
            first_treat="first_treat",
        )
        # CS-SA paper SEs from Table 6
        cs_ses = {(8,8): 1035, (8,9): 909, (8,10): 1008,
                  (9,9): 702, (9,10): 651, (10,10): 995}
        for (g, t), expected_effect in TABLE6_CS_SA.items():
            info = _get_effect(cs_result.group_time_effects, g, t)
            _assert_close(
                info["effect"], expected_effect,
                f"CS ATT({g},{t})", se=cs_ses[(g, t)],
            )

    def test_pretreatment_effects_near_zero(self, edid_hrs_result):
        """Check pre-treatment effects are small (parallel trends plausibility)."""
        pre_effects = []
        post_effects = []
        for (g, t), info in edid_hrs_result.group_time_effects.items():
            g_int, t_int = int(g), int(t)
            if t_int < g_int:
                pre_effects.append(abs(info["effect"]))
            else:
                post_effects.append(abs(info["effect"]))

        if not pre_effects:
            pytest.skip("No pre-treatment effects to check")

        mean_post = np.mean(post_effects)
        for i, pre_eff in enumerate(pre_effects):
            assert pre_eff < 0.5 * mean_post, (
                f"Pre-treatment effect [{i}] = {pre_eff:.1f} is too large "
                f"relative to mean post-treatment ({mean_post:.1f})"
            )


# =============================================================================
# Path 2: Compustat MC Simulations (Tables 4 & 5 patterns)
# =============================================================================


@pytest.mark.slow
class TestCompustatMCValidation:
    """Validate MC properties against Tables 4 & 5 patterns."""

    @pytest.mark.parametrize("rho", [0, 0.5, -0.5])
    def test_unbiasedness(self, ci_params, rho):
        n_sims = ci_params.bootstrap(200, min_n=49)
        mc = _run_mc_simulation(n_sims, rho=rho, seed=2000 + int(rho * 100))

        mean_est = np.mean(mc["edid_estimates"])
        mcse = np.std(mc["edid_estimates"]) / np.sqrt(n_sims)
        bias = abs(mean_est - _TRUE_ES_AVG_COMPUSTAT)

        assert bias < 3 * mcse + 0.05, (
            f"rho={rho}: bias={bias:.4f}, 3*MCSE={3*mcse:.4f}, "
            f"mean={mean_est:.4f}, true={_TRUE_ES_AVG_COMPUSTAT:.4f}"
        )

    @pytest.mark.parametrize("rho", [0, -0.5])
    def test_edid_has_lower_rmse_than_cs(self, ci_params, rho):
        n_sims = ci_params.bootstrap(150, min_n=49)
        mc = _run_mc_simulation(
            n_sims, rho=rho, seed=3000 + int(rho * 100), also_cs=True,
        )

        rmse_edid = np.sqrt(
            np.mean((mc["edid_estimates"] - _TRUE_ES_AVG_COMPUSTAT) ** 2)
        )
        rmse_cs = np.sqrt(
            np.mean((mc["cs_estimates"] - _TRUE_ES_AVG_COMPUSTAT) ** 2)
        )

        # EDiD should not be meaningfully worse than CS
        assert rmse_edid <= rmse_cs * 1.15, (
            f"rho={rho}: RMSE_edid={rmse_edid:.4f} > RMSE_cs={rmse_cs:.4f} * 1.15"
        )

        # For negative rho, efficiency gain should be clear
        if rho == -0.5:
            assert rmse_edid < rmse_cs, (
                f"rho={rho}: Expected RMSE_edid < RMSE_cs, "
                f"got {rmse_edid:.4f} >= {rmse_cs:.4f}"
            )

    def test_efficiency_gain_increases_with_serial_correlation(self, ci_params):
        n_sims = ci_params.bootstrap(150, min_n=49)
        mc_zero = _run_mc_simulation(n_sims, rho=0, seed=4000, also_cs=True)
        mc_neg = _run_mc_simulation(n_sims, rho=-0.5, seed=4500, also_cs=True)

        def rel_rmse(mc):
            rmse_e = np.sqrt(
                np.mean((mc["edid_estimates"] - _TRUE_ES_AVG_COMPUSTAT) ** 2)
            )
            rmse_c = np.sqrt(
                np.mean((mc["cs_estimates"] - _TRUE_ES_AVG_COMPUSTAT) ** 2)
            )
            return rmse_c / rmse_e if rmse_e > 0 else 1.0

        rel_zero = rel_rmse(mc_zero)
        rel_neg = rel_rmse(mc_neg)

        assert rel_neg > rel_zero, (
            f"Expected larger efficiency gain at rho=-0.5 ({rel_neg:.2f}) "
            f"than rho=0 ({rel_zero:.2f})"
        )

    def test_coverage_approximately_correct(self, ci_params):
        n_sims = ci_params.bootstrap(200, min_n=49)
        mc = _run_mc_simulation(n_sims, rho=0, seed=5000)

        true_overall = true_overall_att()
        covered = sum(
            ci[0] <= true_overall <= ci[1]
            for ci in mc["edid_overall_ci"]
        )
        coverage = covered / n_sims

        if n_sims >= 200:
            assert 0.88 <= coverage <= 0.99, (
                f"Coverage={coverage:.2f}, expected 0.88-0.99 (n_sims={n_sims})"
            )
        else:
            assert 0.80 <= coverage <= 1.00, (
                f"Coverage={coverage:.2f}, expected 0.80-1.00 (n_sims={n_sims})"
            )

    def test_analytical_se_calibration(self, ci_params):
        n_sims = ci_params.bootstrap(200, min_n=49)
        mc = _run_mc_simulation(n_sims, rho=0, seed=6000)

        mean_se = np.mean(mc["edid_overall_se"])
        mc_sd = np.std(mc["edid_overall_att"])

        ratio = mean_se / mc_sd if mc_sd > 0 else float("inf")
        assert 0.7 < ratio < 1.4, (
            f"SE calibration ratio={ratio:.2f} (mean_analytical={mean_se:.4f}, "
            f"mc_sd={mc_sd:.4f}), expected 0.7-1.4"
        )
