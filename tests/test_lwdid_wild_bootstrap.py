"""Tests for lwdid_wild_bootstrap module (house-engine wrapper API).

Rewritten in the LWDiD fix wave (WS4): wild_cluster_bootstrap now delegates
to the house WCR engine ``diff_diff.utils.wild_bootstrap_se``. The former
module-local implementation carried three execution-verified defects
(1-ULP tie handling below the attainable p floor, an intercept-only
restricted model that dropped controls from the null DGP, and a G=2
zero-SE roundoff escape reporting t~5e15 with p=0.25).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.lwdid_wild_bootstrap import (
    WildClusterBootstrapResult,
    wild_cluster_bootstrap,
)
from diff_diff.utils import wild_bootstrap_se

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cross_section_data():
    rng = np.random.default_rng(42)
    n = 100
    y = np.concatenate([rng.normal(2, 0.5, 30), rng.normal(0, 0.5, 70)])
    treatment = np.array([1.0] * 30 + [0.0] * 70)
    cluster_ids = np.repeat(np.arange(20), 5)
    controls = rng.normal(0, 1, (n, 2))
    return y, treatment, cluster_ids, controls


# ---------------------------------------------------------------------------
# Result schema (house-aligned)
# ---------------------------------------------------------------------------


class TestResultSchema:
    def test_schema_fields(self, cross_section_data):
        y, d, cl, _ = cross_section_data
        r = wild_cluster_bootstrap(y, d, cl, seed=1, n_bootstrap=99)
        assert isinstance(r, WildClusterBootstrapResult)
        assert np.isfinite(r.att)
        assert np.isfinite(r.se) and r.se > 0
        assert np.isfinite(r.t_stat_original)
        assert 0.0 <= r.p_value <= 1.0
        assert r.ci_lower <= r.ci_upper
        assert r.n_clusters == 20
        assert r.n_bootstrap >= 99
        assert r.weight_type == "rademacher"
        assert r.alpha == 0.05
        assert r.bootstrap_distribution is not None
        assert len(r.bootstrap_distribution) <= r.n_bootstrap
        assert r.n_dropped == 0
        # Retired fields are gone (API break, LWDiD unreleased)
        assert not hasattr(r, "se_bootstrap")
        assert not hasattr(r, "pvalue")
        assert not hasattr(r, "n_reps")
        assert not hasattr(r, "t_stats")

    def test_summary_returns_string(self, cross_section_data):
        y, d, cl, _ = cross_section_data
        r = wild_cluster_bootstrap(y, d, cl, seed=1, n_bootstrap=99)
        s = r.summary()
        assert isinstance(s, str)
        assert "Wild Cluster Bootstrap" in s
        assert "CR1" in s

    def test_matches_house_engine_exactly(self, cross_section_data):
        # The wrapper is a thin adapter: same X construction, same engine,
        # same numbers as calling wild_bootstrap_se directly.
        y, d, cl, controls = cross_section_data
        r = wild_cluster_bootstrap(y, d, cl, controls, seed=7, n_bootstrap=199)
        X = np.column_stack([np.ones(len(y)), d, controls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        house = wild_bootstrap_se(
            X, y, y - X @ beta, cl, 1, n_bootstrap=199, seed=7, return_distribution=True
        )
        np.testing.assert_allclose(r.se, house.se, rtol=0, atol=0)
        np.testing.assert_allclose(r.p_value, house.p_value, rtol=0, atol=0)
        np.testing.assert_allclose(r.ci_lower, house.ci_lower, rtol=0, atol=0)
        np.testing.assert_allclose(r.ci_upper, house.ci_upper, rtol=0, atol=0)
        np.testing.assert_allclose(r.att, beta[1], rtol=1e-12)


class TestWeightTypes:
    @pytest.mark.parametrize("wt", ["rademacher", "mammen", "webb"])
    def test_weight_types_run(self, cross_section_data, wt):
        y, d, cl, _ = cross_section_data
        r = wild_cluster_bootstrap(y, d, cl, weight_type=wt, seed=3, n_bootstrap=99)
        assert 0.0 <= r.p_value <= 1.0
        assert r.weight_type == wt

    def test_invalid_weight_type_raises(self, cross_section_data):
        y, d, cl, _ = cross_section_data
        with pytest.raises(ValueError, match="weight_type"):
            wild_cluster_bootstrap(y, d, cl, weight_type="gaussian")


class TestStatisticalProperties:
    def test_null_imposition_keeps_controls(self):
        # Campaign finding: the old restricted model was intercept-only,
        # dumping covariate signal into the bootstrap residuals (Monte
        # Carlo size 12.5% vs nominal 5% with a treatment-correlated
        # control). The house engine drops ONLY the treatment column; on a
        # null DGP with a strong treatment-correlated control the test must
        # not over-reject.
        rng = np.random.default_rng(0)
        rejections = 0
        n_sims = 40
        for _ in range(n_sims):
            G = 12
            cl = np.repeat(np.arange(G), 10)
            d = (cl < 4).astype(float)
            x = 2.0 * d + rng.normal(size=cl.size)
            y = 1.0 + 1.5 * x + rng.normal(size=cl.size)  # no treatment effect
            r = wild_cluster_bootstrap(
                y, d, cl, controls=x.reshape(-1, 1), n_bootstrap=199, seed=int(rng.integers(1e6))
            )
            rejections += int(r.p_value < 0.05)
        # Binomial(40, 0.05): P(X >= 9) < 1e-4
        assert rejections <= 8, rejections

    def test_full_enumeration_deterministic(self):
        rng = np.random.default_rng(5)
        G = 8
        cl = np.repeat(np.arange(G), 6)
        d = (cl < 3).astype(float)
        y = 0.5 * d + rng.normal(size=cl.size)
        r1 = wild_cluster_bootstrap(y, d, cl, n_bootstrap=999, seed=1)
        r2 = wild_cluster_bootstrap(y, d, cl, n_bootstrap=999, seed=2)
        # 2**8 = 256 <= 999 -> full enumeration, independent of the seed
        assert r1.n_bootstrap == 256 and r2.n_bootstrap == 256
        assert r1.p_value == r2.p_value

    def test_enumeration_p_is_exact_atom(self):
        # The campaign's 1-ULP tie finding (reported p below the attainable
        # floor of the OLD percentile-t enumeration) is resolved by
        # ADOPTION of the house WCR convention, whose tie handling is
        # pinned by the house R-parity goldens (tests/test_wild_bootstrap).
        # Contract here: under enumeration the p-value is an exact atom
        # k/2**G of the deterministic distribution.
        rng = np.random.default_rng(9)
        G = 4
        cl = np.repeat(np.arange(G), 8)
        d = (cl < 2).astype(float)
        y = 3.0 * d + rng.normal(scale=0.2, size=cl.size)
        with pytest.warns(UserWarning, match="fewer than 5 clusters"):
            r = wild_cluster_bootstrap(y, d, cl, n_bootstrap=999, seed=11)
        assert r.n_bootstrap == 16
        k = r.p_value * 16
        np.testing.assert_allclose(k, round(k), atol=1e-12)


class TestDegenerateDesigns:
    def test_g2_exactly_identified_fails_closed(self):
        # Campaign finding: the canonical two-cluster design (cluster-
        # invariant treatment) has cluster scores exactly ~0; BLAS roundoff
        # gave a tiny-positive SE, t ~ 5e15, and p = 0.25 (below the G=2
        # attainable floor of 0.5). Point retained; inference NaN.
        rng = np.random.default_rng(2)
        cl = np.repeat([0, 1], 12)
        d = (cl == 0).astype(float)
        y = 1.0 + 0.8 * d + rng.normal(scale=0.5, size=cl.size)
        with pytest.warns(UserWarning, match="not identified"):
            r = wild_cluster_bootstrap(y, d, cl, n_bootstrap=99, seed=4)
        assert np.isfinite(r.att)
        assert np.isnan(r.se) and np.isnan(r.p_value)
        assert np.isnan(r.ci_lower) and np.isnan(r.ci_upper)
        assert r.bootstrap_distribution is None

    def test_single_cluster_rejected(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=20)
        d = np.r_[np.ones(10), np.zeros(10)]
        cl = np.zeros(20)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            wild_cluster_bootstrap(y, d, cl)


class TestInputContracts:
    def test_nonfinite_y_dropped_with_warning_and_counted(self):
        rng = np.random.default_rng(6)
        G = 10
        cl = np.repeat(np.arange(G), 8)
        d = (cl < 4).astype(float)
        y = 1.0 * d + rng.normal(size=cl.size)
        y[3] = np.nan
        y[40] = np.inf
        with pytest.warns(UserWarning, match="dropped 2 observation"):
            r = wild_cluster_bootstrap(y, d, cl, n_bootstrap=99, seed=8)
        assert r.n_dropped == 2
        assert np.isfinite(r.p_value)

    def test_nonfinite_controls_raise(self, cross_section_data):
        y, d, cl, controls = cross_section_data
        controls = controls.copy()
        controls[0, 0] = np.nan
        with pytest.raises(ValueError, match="controls contains non-finite"):
            wild_cluster_bootstrap(y, d, cl, controls)

    def test_retired_parameters_rejected(self, cross_section_data):
        y, d, cl, _ = cross_section_data
        with pytest.raises(TypeError):
            wild_cluster_bootstrap(y, d, cl, impose_null=False)
        with pytest.raises(TypeError):
            wild_cluster_bootstrap(y, d, cl, full_enumeration=True)
        with pytest.raises(TypeError):
            wild_cluster_bootstrap(y, d, cl, n_reps=99)
        with pytest.raises(TypeError):
            wild_cluster_bootstrap(y, d, cl, ci_level=0.9)


class TestResultsConvenienceMethods:
    """LWDiDResults.wild_cluster_bootstrap() / .randomization_test().

    Round-5 review: these REPLAY the fitted estimation sample and RA
    design (no data arguments) and assert their observed statistic equals
    ``.att`` before caching - previously they accepted arbitrary caller
    arrays and a non-interacted design, so the cached p-values could
    describe a different estimand than the fitted ATT.
    """

    @staticmethod
    def _fitted_results(cluster=None, covariate=False):
        from diff_diff import LWDiD

        rng = np.random.default_rng(42)
        records = []
        for i in range(60):
            d = int(i < 20)
            x = float(i % 4) + (1.5 if d else 0.0)  # treatment-unbalanced
            for t in range(1, 7):
                y = 1.0 + 0.1 * t + 0.4 * x + rng.normal(0, 0.3)
                if d and t > 3:
                    y += 2.0 + 0.5 * x
                records.append(
                    {"unit": i, "time": t, "y": y, "treat": d * int(t > 3), "x": x, "cl": i % 12}
                )
        df = pd.DataFrame(records)
        est = LWDiD(cluster=cluster)
        return est.fit(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treatment="treat",
            covariates=["x"] if covariate else None,
        )

    def test_results_wild_cluster_bootstrap_replays_fit(self):
        res = self._fitted_results(cluster="cl", covariate=True)
        wcb = res.wild_cluster_bootstrap(n_bootstrap=99, seed=42)
        # coherence: the replayed observed ATT IS the fitted ATT
        np.testing.assert_allclose(wcb.att, res.att, rtol=1e-10)
        assert 0 <= wcb.p_value <= 1
        assert res.bootstrap_pvalue == wcb.p_value

    def test_results_wcb_requires_clustered_fit(self):
        res = self._fitted_results(cluster=None)
        with pytest.raises(ValueError, match="requires a clustered fit"):
            res.wild_cluster_bootstrap(n_bootstrap=99, seed=42)

    def test_results_randomization_test_replays_fit(self):
        res = self._fitted_results(covariate=True)
        ri = res.randomization_test(n_reps=199, seed=42)
        np.testing.assert_allclose(ri.att_observed, res.att, rtol=1e-10)
        assert 0 <= ri.pvalue <= 1
        assert res.ri_pvalue == ri.pvalue

    def test_results_methods_reject_non_reg_fits(self):
        from diff_diff import LWDiD

        rng = np.random.default_rng(0)
        records = []
        for i in range(40):
            d = int(i < 20)
            x = float(i % 5)
            for t in range(1, 7):
                y = 1.0 + 0.2 * x + rng.normal(0, 0.3) + (2.0 if d and t > 3 else 0.0)
                records.append({"unit": i, "time": t, "y": y, "treat": d * int(t > 3), "x": x})
        df = pd.DataFrame(records)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = LWDiD(estimation_method="ipw").fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treatment="treat",
                covariates=["x"],
            )
        with pytest.raises(ValueError, match="only\\s+defined for estimation_method='reg'"):
            res.randomization_test(n_reps=99, seed=0)
