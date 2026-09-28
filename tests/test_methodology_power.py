"""
PowerAnalysis methodology test file — Bloom (1995) + Burlig, Preonas & Woerman
(2020) walkthrough.

Companion to ``tests/test_power.py`` (basic unit-test surface): this file
validates the *analytical* power path against the specific paper equations, with
paper-equation-numbered assertions. Mirrors the structure of
``tests/test_methodology_pretrends.py``.

Bloom, H. S. (1995). Minimum Detectable Effects: A Simple Way to Report the
    Statistical Power of Experimental Designs. *Evaluation Review*, 19(5),
    547-556. https://doi.org/10.1177/0193841X9501900504
Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental
    Design. *Journal of Development Economics*, 144, 102458.
    https://doi.org/10.1016/j.jdeveco.2020.102458

Paper reviews on file: ``docs/methodology/papers/bloom-1995-review.md``,
``docs/methodology/papers/burlig-preonas-woerman-2020-review.md``.

Two methodology decisions locked in the PR-B review and pinned here:

- **D1 (z, not t):** the MDE multiplier uses the NORMAL distribution
  ``M = z_{1-alpha/2} + z_{power}`` (Bloom 1995), a large-sample approximation to
  Burlig Eq. 1's t-based multiplier. ``TestBloomMDEMultiplier`` reproduces Bloom
  Table 1.
- **D4 (panel variance = Burlig Eq. 2, equicorrelated):** the panel-DiD variance
  is the within-unit equicorrelated special case of Burlig Eq. 2,
  ``Var = sigma^2 (1/n_T + 1/n_C) (1/m + 1/r) (1 - rho)``, so within-unit
  correlation LOWERS the MDE. ``TestPanelVarianceBurlig`` pins the closed form
  and ``TestPanelVarianceMonteCarlo`` validates it against a literal
  equicorrelated DiD simulation.

Class structure:

- ``TestBloomMDEMultiplier`` — Bloom Table 1 normal (z) multipliers (D1).
- ``TestBasicDiDVariance`` — 2x2 DiD variance ``2 sigma^2 (1/n_T+1/n_C)`` and
  closed-form MDE / power.
- ``TestPanelVarianceBurlig`` — Burlig Eq. 2 equicorrelated closed form;
  rho-direction; pre/post period separation; continuity with the 2x2 branch.
- ``TestPanelVarianceMonteCarlo`` — literal equicorrelated DiD simulation
  confirms the closed-form variance is the correct DiD variance (D4).
- ``TestSampleSizeRoundTrip`` — sample_size -> mde recovers the target effect;
  allocation factor f(1-f) and 50/50 optimality.
- ``TestInputValidation`` — input guards for ALL designs (rho>=1, rho<-1/(T-1),
  n_pre/n_post==0) raise ValueError, including the T<=2 basic_did path and the
  compute_* wrappers; valid boundary accepted.
- ``TestPowerParityR`` — base-R ``qnorm`` parity at
  ``benchmarks/data/r_power_golden.json``.
"""

import json
import os

import numpy as np
import pytest
from scipy import stats

from diff_diff.power import PowerAnalysis

# =============================================================================
# TestBloomMDEMultiplier — Bloom (1995) Table 1, normal (z) multiplier (D1)
# =============================================================================


class TestBloomMDEMultiplier:
    """The MDE multiplier ``M = z_{1-alpha(/2)} + z_{power}`` (Bloom 1995, p.549).

    ``_compute_mde_from_se(se=1.0)`` returns the multiplier ``M`` directly. Bloom
    builds ``M`` from the NORMAL distribution (e.g. one-sided .05 / 80% power:
    1.645 + 0.84 = 2.49, p.549), which is exactly what the library implements via
    ``stats.norm.ppf`` (D1: deliberate normal approximation to Burlig's t).
    """

    @pytest.mark.parametrize(
        "power, stated",
        [(0.90, 2.93), (0.80, 2.49), (0.70, 2.17)],  # Bloom Table 1, one-sided .05
    )
    def test_one_sided_p05_table1(self, power, stated):
        pa = PowerAnalysis(alpha=0.05, power=power, alternative="greater")
        m = pa._compute_mde_from_se(1.0)
        # Matches the exact z-based multiplier ...
        assert np.isclose(m, stats.norm.ppf(0.95) + stats.norm.ppf(power), atol=1e-12)
        # ... and Bloom's stated 2-dp value.
        assert np.isclose(m, stated, atol=5e-3)

    def test_two_sided_textbook(self):
        # Two-sided .05 / 80% power: 1.96 + 0.84 = 2.80 (the textbook DiD value).
        pa = PowerAnalysis(alpha=0.05, power=0.80, alternative="two-sided")
        assert np.isclose(pa._compute_mde_from_se(1.0), 2.8016, atol=1e-3)


# =============================================================================
# TestBasicDiDVariance — 2x2 DiD variance (Bloom Eq. 1 DiD analog)
# =============================================================================


class TestBasicDiDVariance:
    """2x2 DiD variance ``2 sigma^2 (1/n_T + 1/n_C)``.

    The DiD analog of Bloom (1995) Eq. 1 (two independent measurement occasions
    -> factor of 2; no covariate ``R^2`` term in the library's analytical path).
    """

    @pytest.mark.parametrize("n_t, n_c, sigma", [(50, 50, 1.0), (30, 90, 2.0), (10, 10, 3.5)])
    def test_basic_variance_closed_form(self, n_t, n_c, sigma):
        pa = PowerAnalysis()
        v = pa._compute_variance(n_t, n_c, 1, 1, sigma, 0.0, design="basic_did")
        assert np.isclose(v, 2 * sigma**2 * (1 / n_t + 1 / n_c), atol=1e-12)

    def test_mde_uses_basic_for_two_periods(self):
        # n_pre = n_post = 1 -> T = 2 -> basic_did router branch.
        pa = PowerAnalysis(alpha=0.05, power=0.80)
        res = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=1, n_post=1)
        assert res.design == "basic_did"
        se = np.sqrt(2 * 1.0**2 * (1 / 50 + 1 / 50))
        mult = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
        assert np.isclose(res.mde, mult * se, atol=1e-12)

    def test_power_two_tail_normal(self):
        # The power function is the EXACT two-tail normal (not the one-tail
        # approximation): Power = 1 - Phi(z - d/se) + Phi(-z - d/se).
        pa = PowerAnalysis(alpha=0.05, power=0.80, alternative="two-sided")
        res = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
        se = np.sqrt(2 * (1 / 50 + 1 / 50))
        z = stats.norm.ppf(0.975)
        expected = 1 - stats.norm.cdf(z - 0.5 / se) + stats.norm.cdf(-z - 0.5 / se)
        assert np.isclose(res.power, expected, atol=1e-12)

    def test_rho_applies_at_two_periods(self):
        # The 2x2 path is the m=r=1 equicorrelated case (Burlig footnote 11), so
        # a nonzero rho is NOT silently ignored: Var = 2 sigma^2 (1/n_T+1/n_C)(1-rho).
        pa = PowerAnalysis()
        v = pa._compute_variance(50, 50, 1, 1, 1.0, 0.5, design="basic_did")
        assert np.isclose(v, 2 * (1 / 50 + 1 / 50) * (1 - 0.5), atol=1e-12)
        # ... and it flows through to a smaller MDE than rho=0.
        m0 = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=1, n_post=1, rho=0.0).mde
        m5 = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=1, n_post=1, rho=0.5).mde
        assert m5 < m0


# =============================================================================
# TestPanelVarianceBurlig — Burlig Eq. 2, equicorrelated case (D4)
# =============================================================================


class TestPanelVarianceBurlig:
    """Panel variance ``sigma^2 (1/n_T+1/n_C)(1/m+1/r)(1-rho)`` (Burlig Eq. 2).

    The within-unit equicorrelated special case of Burlig, Preonas & Woerman
    (2020) Eq. 2 (psi^B = psi^A = psi^X = rho*sigma^2). Key properties (all
    opposite to the pre-fix Moulton ``(1+(T-1)rho)/T`` factor):
    (a) cross-period correlation LOWERS the DiD variance; (b) periods enter as
    ``(1/m + 1/r)``, separating pre (m) and post (r); (c) continuous with the 2x2
    branch at m = r = 1.
    """

    @staticmethod
    def _closed_form(n_t, n_c, m, r, sigma, rho):
        return sigma**2 * (1 / n_t + 1 / n_c) * (1 / m + 1 / r) * (1 - rho)

    @pytest.mark.parametrize(
        "n_t, n_c, m, r, sigma, rho",
        [
            (50, 50, 3, 3, 1.0, 0.0),
            (50, 50, 3, 3, 1.0, 0.3),
            (50, 50, 5, 5, 1.0, 0.5),
            (60, 40, 2, 5, 1.2, 0.4),
            (50, 50, 3, 3, 1.0, -0.1),  # valid negative (> -1/(T-1) = -0.2)
        ],
    )
    def test_panel_variance_closed_form(self, n_t, n_c, m, r, sigma, rho):
        pa = PowerAnalysis()
        v = pa._compute_variance(n_t, n_c, m, r, sigma, rho, design="panel")
        assert np.isclose(v, self._closed_form(n_t, n_c, m, r, sigma, rho), atol=1e-12)

    def test_period_separation_not_pooled(self):
        # (1/m + 1/r), NOT 1/(m+r): asymmetric m != r must distinguish them.
        pa = PowerAnalysis()
        v = pa._compute_variance(50, 50, 1, 5, 1.0, 0.0, design="panel")
        assert np.isclose(v, 1.0 * (1 / 50 + 1 / 50) * (1 / 1 + 1 / 5), atol=1e-12)
        # A pooled 1/(m+r) factor would give a different (smaller) value.
        pooled = 1.0 * (1 / 50 + 1 / 50) * (1 / (1 + 5))
        assert not np.isclose(v, pooled)

    def test_rho_lowers_mde(self):
        # Burlig's core DiD result: higher within-unit correlation -> lower MDE.
        pa = PowerAnalysis(power=0.80)
        mdes = [
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=rho).mde
            for rho in (0.0, 0.3, 0.6)
        ]
        assert mdes[0] > mdes[1] > mdes[2]

    def test_more_periods_lower_variance_at_rho0(self):
        pa = PowerAnalysis()
        v_short = pa._compute_variance(50, 50, 2, 2, 1.0, 0.0, design="panel")
        v_long = pa._compute_variance(50, 50, 5, 5, 1.0, 0.0, design="panel")
        assert v_long < v_short

    def test_continuity_with_basic_at_m_r_1(self):
        # At m = r = 1, rho = 0 the panel branch reduces to the 2x2 branch:
        # (1/m + 1/r) = 2. Called directly because the router never evaluates
        # the panel branch at T = 2.
        pa = PowerAnalysis()
        v_panel = pa._compute_variance(50, 50, 1, 1, 1.0, 0.0, design="panel")
        v_basic = pa._compute_variance(50, 50, 1, 1, 1.0, 0.0, design="basic_did")
        assert np.isclose(v_panel, v_basic, atol=1e-12)


# =============================================================================
# TestPanelVarianceMonteCarlo — empirical DiD variance under equicorrelation
# =============================================================================


class TestPanelVarianceMonteCarlo:
    """The closed form IS the variance of the DiD estimator under equicorrelation.

    Draws literal within-unit equicorrelated Gaussian errors (common-shock
    construction: ``e = sigma(sqrt(1-rho) Z + sqrt(rho) U)``, giving unit
    variance sigma^2 and all within-unit pair covariances rho*sigma^2), forms the
    canonical DiD estimator ``(mean_T - mean_C)`` of unit post-minus-pre
    differences under the null, and checks the empirical variance against
    ``sigma^2 (1/n_T+1/n_C)(1/m+1/r)(1-rho)``. This validates the *model*, not
    just cross-language arithmetic (which TestPowerParityR covers).
    """

    @pytest.mark.parametrize(
        "n_t, n_c, m, r, sigma, rho",
        [
            (50, 50, 3, 3, 1.0, 0.0),
            (50, 50, 3, 3, 1.0, 0.3),
            (40, 60, 2, 4, 1.5, 0.6),
        ],
    )
    def test_empirical_matches_closed_form(self, n_t, n_c, m, r, sigma, rho):
        n_rep = 6000
        rng = np.random.default_rng(20260531)
        T = m + r

        def draw_diffs(n_units):
            z = rng.standard_normal((n_rep, n_units, T))
            u = rng.standard_normal((n_rep, n_units))
            e = sigma * (np.sqrt(1 - rho) * z + np.sqrt(rho) * u[..., None])
            return e[..., m:].mean(-1) - e[..., :m].mean(-1)  # (n_rep, n_units)

        att = draw_diffs(n_t).mean(-1) - draw_diffs(n_c).mean(-1)  # (n_rep,)
        emp_var = att.var(ddof=1)
        closed = sigma**2 * (1 / n_t + 1 / n_c) * (1 / m + 1 / r) * (1 - rho)
        # ~2% relative SE on the variance estimate at n_rep=6000; 6% (3 SE) is safe.
        assert np.isclose(emp_var, closed, rtol=0.06)


# =============================================================================
# TestSampleSizeRoundTrip — sample_size <-> mde consistency + allocation
# =============================================================================


class TestSampleSizeRoundTrip:
    """``sample_size`` inverts ``mde``; allocation factor ``f(1-f)`` is applied."""

    @pytest.mark.parametrize(
        "n_pre, n_post, rho",
        [(1, 1, 0.0), (3, 3, 0.3), (2, 5, 0.5)],
    )
    def test_round_trip(self, n_pre, n_post, rho):
        pa = PowerAnalysis(alpha=0.05, power=0.80)
        effect = 0.5
        ss = pa.sample_size(effect_size=effect, sigma=1.0, n_pre=n_pre, n_post=n_post, rho=rho)
        # Required N rounds up, so the achieved MDE is at or just below the target
        # effect, and power at that N meets the target.
        assert ss.mde <= effect + 1e-9
        assert ss.mde > 0.9 * effect  # not wildly over-powered
        pw = pa.power(
            effect_size=effect,
            n_treated=ss.n_treated,
            n_control=ss.n_control,
            sigma=1.0,
            n_pre=n_pre,
            n_post=n_post,
            rho=rho,
        )
        assert pw.power >= 0.80 - 1e-9

    def test_allocation_5050_most_efficient(self):
        pa = PowerAnalysis(power=0.80)
        n_balanced = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.5).required_n
        n_skewed = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.2).required_n
        assert n_balanced < n_skewed

    def test_allocation_factor_present(self):
        # Raw basic_did N carries 1/(f(1-f)); at f=0.5 the factor is 4, at f=0.2
        # it is 1/0.16 = 6.25, so n(0.2)/n(0.5) ~ 0.25/0.16 = 1.5625.
        pa = PowerAnalysis(power=0.80)
        n50 = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.5).required_n
        n20 = pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.2).required_n
        assert np.isclose(n20 / n50, 0.25 / 0.16, rtol=0.02)


# =============================================================================
# TestPanelValidation — panel-scoped input guards
# =============================================================================


class TestInputValidation:
    """Input guards apply to ALL designs (validation runs *before* the router).

    n_pre>=1, n_post>=1, rho in [-1/(T-1), 1). Critically the T<=2 (basic_did)
    path is validated too, so invalid 2-period shapes and out-of-range rho cannot
    silently fall through to basic_did and return a number.
    """

    # --- panel (T > 2) ---
    def test_rho_ge_one_raises(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="rho"):
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=1.0)

    def test_rho_below_equicorrelation_bound_raises(self):
        # T = 6 -> rho_min = -1/5 = -0.2; rho = -0.5 is invalid.
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="rho"):
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=-0.5)

    def test_valid_negative_rho_accepted(self):
        # rho = -0.15 is inside [-0.2, 1) for T = 6 -> no raise.
        pa = PowerAnalysis()
        res = pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=3, n_post=3, rho=-0.15)
        assert res.mde > 0

    def test_zero_pre_periods_raises(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="n_pre"):
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=0, n_post=3, rho=0.0)

    def test_zero_post_periods_raises(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="n_post"):
            pa.power(
                effect_size=0.5,
                n_treated=50,
                n_control=50,
                sigma=1.0,
                n_pre=3,
                n_post=0,
                rho=0.0,
            )

    # --- T <= 2 (basic_did): validation must fire here too (regression for the
    #     pre-fix silent fall-through to the basic_did branch) ---
    def test_zero_pre_at_two_periods_raises(self):
        # n_pre=0, n_post=2 -> T=2 -> routes to basic_did, must still raise.
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="n_pre"):
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=0, n_post=2, rho=0.0)

    def test_zero_post_at_two_periods_raises(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="n_post"):
            pa.sample_size(effect_size=0.5, sigma=1.0, n_pre=2, n_post=0, rho=0.0)

    def test_rho_out_of_range_at_two_periods_raises(self):
        # n_pre=n_post=1 -> T=2, rho range [-1, 1); rho=1.5 must raise rather than
        # be silently ignored by basic_did.
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="rho"):
            pa.mde(n_treated=50, n_control=50, sigma=1.0, n_pre=1, n_post=1, rho=1.5)

    def test_convenience_wrappers_validate(self):
        # The compute_* wrappers delegate to the class methods, so they inherit
        # the same guards (the public API codex flagged).
        from diff_diff import compute_mde, compute_power, compute_sample_size

        with pytest.raises(ValueError, match="n_pre"):
            compute_mde(n_treated=50, n_control=50, sigma=1.0, n_pre=0, n_post=2)
        with pytest.raises(ValueError, match="rho"):
            compute_power(
                effect_size=0.5,
                n_treated=50,
                n_control=50,
                sigma=1.0,
                n_pre=1,
                n_post=1,
                rho=1.5,
            )
        with pytest.raises(ValueError, match="n_post"):
            compute_sample_size(effect_size=0.5, sigma=1.0, n_pre=2, n_post=0)

    # --- non-design inputs (sigma, group counts, treat_frac) ---
    def test_negative_sigma_raises(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="sigma"):
            pa.mde(n_treated=50, n_control=50, sigma=-1.0)

    def test_nonpositive_group_counts_raise(self):
        pa = PowerAnalysis()
        with pytest.raises(ValueError, match="n_treated"):
            pa.mde(n_treated=0, n_control=50, sigma=1.0)
        with pytest.raises(ValueError, match="n_control"):
            pa.power(effect_size=0.5, n_treated=50, n_control=0, sigma=1.0)

    def test_treat_frac_out_of_range_raises(self):
        pa = PowerAnalysis()
        for tf in (0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="treat_frac"):
                pa.sample_size(effect_size=0.5, sigma=1.0, treat_frac=tf)
        from diff_diff import compute_sample_size

        with pytest.raises(ValueError, match="treat_frac"):
            compute_sample_size(effect_size=0.5, sigma=1.0, treat_frac=0.0)


# =============================================================================
# TestPowerParityR — base-R qnorm parity (benchmarks/data/r_power_golden.json)
# =============================================================================

_GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "data",
    "r_power_golden.json",
)


@pytest.mark.skipif(
    not os.path.exists(_GOLDEN_PATH),
    reason="R power parity goldens not present (partial checkout)",
)
class TestPowerParityR:
    """Parity with base-R ``qnorm``/``pnorm`` exact closed forms.

    Goldens at ``benchmarks/data/r_power_golden.json`` (generated by
    ``benchmarks/R/generate_power_golden.R``). The goldens are EXACT closed forms
    computed independently in R, so the realistic ceiling is ``atol=1e-9`` on the
    continuous quantities (qnorm/pnorm vs scipy.stats.norm agree to ~1e-15) and
    exact equality on the integer ``required_n``. The MDE multiplier is
    normal-based (D1), so the parity reference is ``qnorm`` -- NOT
    ``pwr::pwr.t.test()`` (noncentral-t), which would not match the library.
    """

    @staticmethod
    def _load():
        with open(_GOLDEN_PATH) as f:
            return json.load(f)

    def test_bloom_table1_multipliers(self):
        data = self._load()
        for row in data["bloom_table1_one_sided_p05"]:
            pa = PowerAnalysis(alpha=0.05, power=row["power"], alternative="greater")
            assert np.isclose(pa._compute_mde_from_se(1.0), row["multiplier"], atol=1e-12)

    def test_fixture_parity(self):
        data = self._load()
        for fx in data["fixtures"]:
            exp = fx["expected"]
            pa = PowerAnalysis(alpha=fx["alpha"], power=fx["power"], alternative=fx["alternative"])
            # variance (validates the 2x2 / panel formula directly)
            v = pa._compute_variance(
                fx["n_treated"],
                fx["n_control"],
                fx["n_pre"],
                fx["n_post"],
                fx["sigma"],
                fx["rho"],
                design=exp["design"],
            )
            assert np.isclose(v, exp["variance"], atol=1e-9), fx["name"]
            assert np.isclose(np.sqrt(v), exp["se"], atol=1e-9), fx["name"]

            # mde (integrated path: design routing + multiplier + variance)
            mde_res = pa.mde(
                fx["n_treated"],
                fx["n_control"],
                fx["sigma"],
                fx["n_pre"],
                fx["n_post"],
                fx["rho"],
            )
            assert mde_res.design == exp["design"], fx["name"]
            assert np.isclose(mde_res.mde, exp["mde"], atol=1e-9), fx["name"]

            # power
            pw = pa.power(
                effect_size=fx["effect_size"],
                n_treated=fx["n_treated"],
                n_control=fx["n_control"],
                sigma=fx["sigma"],
                n_pre=fx["n_pre"],
                n_post=fx["n_post"],
                rho=fx["rho"],
            )
            assert np.isclose(pw.power, exp["power"], atol=1e-9), fx["name"]

            # required_n (exact integer parity; R replicates the rounding)
            ss = pa.sample_size(
                effect_size=fx["effect_size"],
                sigma=fx["sigma"],
                n_pre=fx["n_pre"],
                n_post=fx["n_post"],
                rho=fx["rho"],
                treat_frac=fx["treat_frac"],
            )
            assert ss.required_n == exp["required_n"], fx["name"]
