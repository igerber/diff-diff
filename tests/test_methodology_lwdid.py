"""Independent methodology-validation suite for the LWDiD estimator (PR #588).

This module is the maintainer-side acceptance suite for the third-party LWDiD
(Lee & Wooldridge rolling-transformation DiD) contribution. It is merged to
main BEFORE the estimator exists: the module-level ``pytest.importorskip``
makes it skip cleanly until ``diff_diff.lwdid`` lands, at which point every
test activates automatically on the estimator branch.

This module verifies that the LWDiD implementation matches:

1. The published replication targets of Lee & Wooldridge (2026), Tables 3, 4
   and A1 (California Prop 99, three donor pools) including exact-inference
   and randomization-inference p-values.
2. The castle-doctrine staggered application of LW (2026) Section 7.2
   (tau_omega via the composite-outcome regression (7.18)/(7.19)).
3. The event-study replication targets of Lee & Wooldridge (2025), Appendix F
   Tables A4/A5 (Walmart entry), via the normative event-study API specified
   in ``TestEventStudySpec`` (xfail until PR #588's Appendix D work lands).
4. Estimator-independent properties: translation invariance of SEs,
   cross-estimator equivalences (plain DiD; per-period panel-DiD identity of
   LW 2026 eq. (2.20); the T=3 detrending closed form of LW 2025 eq. (5.7)),
   from-scratch reference implementations of Procedures 2.1/3.1 and the
   staggered per-cohort demeaning, exact small-sample t inference
   (T_{N-2} / T_{N-K-2}), and the Monte Carlo bias ordering of LW 2026
   Section 5 under heterogeneous trends.
5. REGISTRY.md edge cases: minimum pre-treatment periods (>= 1 demeaning,
   >= 2 detrending).

xfail semantics (first use of xfail in this codebase): an ``xfail`` marker
encodes an agreed, outstanding work item of PR #588 - the marker's ``reason``
names the item. ``strict=True`` markers MUST be removed by the commit that
fixes the item (the test then passing turns XPASS into a hard failure,
forcing explicit acceptance). ``strict=False`` is used only where numerical
fragility across environments is plausible.

Data sources:

- Prop 99 / Walmart: ``load_prop99()`` / ``load_walmart()`` (authors' SSC
  ancillary data; checksum-pinned). Tests skip unless
  ``df.attrs["source"] == "lwdid_ssc_ancillary"`` (i.e. real data, not the
  synthetic offline fallback).
- Castle doctrine: ``benchmarks/data/real/castle_lw_subset.csv``, the real
  Cheng & Hoekstra (2013) panel as packaged by Cunningham (2021) and
  distributed with PR #588 (extracted from commit 8c5cccea; columns state,
  year, effyear, lhomicide, homicide, population). Originally committed
  because the ``load_castle_doctrine`` upstream source was dead; that source
  is now pinned and verified, but this subset stays as the pinned artifact
  these goldens were captured against. Consolidating onto the loader is a
  separate decision, not a consequence of the source being reachable again.
- Walmart event-study goldens:
  ``benchmarks/data/lwdid_walmart_eventstudy_golden.json`` (Tables A4/A5,
  provenance embedded in the file).

References:

- Lee, S.J., & Wooldridge, J.M. (2025). A Simple Transformation Approach to
  Difference-in-Differences Estimation for Panel Data. SSRN No. 4516518
  (revision of June 8, 2026). https://ssrn.com/abstract=4516518
- Lee, S.J., & Wooldridge, J.M. (2026). Simple Approaches to Inference with
  Difference-in-Differences Estimators with Small Cross-Sectional Sample
  Sizes. SSRN No. 5325686 (cover date February 3, 2026).
  https://ssrn.com/abstract=5325686
- docs/methodology/REGISTRY.md, section "LWDiD".
- docs/methodology/papers/lee-wooldridge-{2025,2026}-review.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytest.importorskip(
    "diff_diff.lwdid",
    reason="LWDiD estimator not yet on main (arrives via PR #588)",
)

from diff_diff.lwdid import LWDiD  # noqa: E402

from diff_diff import (  # noqa: E402
    DifferenceInDifferences,  # noqa: E402
    load_prop99,
    load_walmart,
)

# ---------------------------------------------------------------------------
# Published replication targets (LW 2026; see module docstring for provenance)
# ---------------------------------------------------------------------------

# Table 3 (38-state donor pool): {procedure: (ATT, SE)}
TABLE3_AVERAGE = {"demean": (-0.422, 0.121), "detrend": (-0.227, 0.094)}
TABLE3_PER_PERIOD = {
    "demean": {1989: (-0.168, 0.096), 1995: (-0.484, 0.137), 2000: (-0.667, 0.164)},
    "detrend": {1989: (-0.043, 0.059), 1995: (-0.282, 0.112), 2000: (-0.403, 0.152)},
}
TABLE3_DETREND_EXACT_P = 0.021
TABLE3_DETREND_RI_P = 0.020

# Table 4 (Southern pool) and Table A1 (Midwestern pool): Average rows
SOUTHERN_POOL = ["Alabama", "Arkansas", "Louisiana", "Mississippi"]
TABLE4_AVERAGE = {"demean": (-0.556, 0.080), "detrend": (-0.215, 0.039)}
MIDWEST_POOL = ["Illinois", "Iowa", "Minnesota", "Ohio"]
TABLEA1_AVERAGE = {"demean": (-0.413, 0.118), "detrend": (-0.198, 0.079)}

# Castle-laws application (Section 7.2): tau_omega targets
CASTLE_COHORTS = {2005: 1, 2006: 13, 2007: 4, 2008: 2, 2009: 1}
CASTLE_TAU_DEMEAN = (0.092, 0.057)  # (tau_omega, usual OLS SE)
CASTLE_TAU_DETREND = 0.067

# Printed-precision tolerance for 3-decimal published tables
PRINTED_ATOL = 1e-3

_CASTLE_CSV = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "real" / "castle_lw_subset.csv"
)
_WALMART_ES_GOLDEN = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "data"
    / "lwdid_walmart_eventstudy_golden.json"
)

XFAIL_IPW_CENTERING = pytest.mark.xfail(
    strict=True,
    reason="PR #588 step-2 item 1: IPW influence function is un-centered, "
    "making the IPW SE translation-variant. Remove this marker in the "
    "commit that centers the IPW IF.",
)
XFAIL_EVENT_STUDY = pytest.mark.xfail(
    strict=True,
    reason="PR #588 Option A: Appendix D event study + Algorithm 1 "
    "multiplier bootstrap not yet implemented. Remove this marker in the "
    "commit that implements the event study (deterministic spec tests).",
)
XFAIL_EVENT_STUDY_GOLDENS = pytest.mark.xfail(
    strict=False,
    reason="PR #588 Option A: Appendix D event study + Algorithm 1 not yet "
    "implemented. Non-strict (numerical fragility): the golden SEs are the "
    "paper's printed B=999 multiplier-bootstrap draws; a re-seeded bootstrap "
    "can sit near the printed-precision tolerance boundary across "
    "platforms. Re-calibrate the SE tolerance when the event study lands, "
    "then remove the marker.",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_prop99():
    """Load the real Prop 99 panel or skip (offline / synthetic fallback).

    Skips are VISIBLE (not silent): CI runners have network and the loaders
    are SHA-256-pinned, so these tests run in practice; a dedicated
    real-data canary lane is tracked in TODO.md alongside the legacy-loader
    fallback repair.
    """
    df = load_prop99()
    if df.attrs.get("source") != "lwdid_ssc_ancillary":
        pytest.skip("real Prop 99 data unavailable (synthetic fallback in use)")
    return df


def _real_walmart():
    """Load the real Walmart panel or skip (offline / synthetic fallback)."""
    df = load_walmart()
    if df.attrs.get("source") != "lwdid_ssc_ancillary":
        pytest.skip("real Walmart data unavailable (synthetic fallback in use)")
    return df


def _fit_prop99(df, rolling, **kwargs):
    est = LWDiD(rolling=rolling, estimator="ra", vce="classical", **kwargs)
    return est.fit(df, outcome="lcigsale", unit="state", time="year", treatment="treated")


def _demean_reference(df, unit, time, outcome, pre_end):
    """From-scratch Procedure 2.1: per-unit pre-mean, post-average residual."""
    out = {}
    for u, g in df.groupby(unit):
        g = g.sort_values(time)
        pre = g.loc[g[time] <= pre_end, outcome].to_numpy()
        post = g.loc[g[time] > pre_end, outcome].to_numpy()
        out[u] = post.mean() - pre.mean()
    return pd.Series(out)


def _detrend_reference(df, unit, time, outcome, pre_end):
    """From-scratch Procedure 3.1: per-unit pre-period OLS on (1, t),
    out-of-sample residuals for post periods, averaged."""
    out = {}
    for u, g in df.groupby(unit):
        g = g.sort_values(time)
        pre = g[g[time] <= pre_end]
        post = g[g[time] > pre_end]
        X = np.column_stack([np.ones(len(pre)), pre[time].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X, pre[outcome].to_numpy(dtype=float), rcond=None)
        yhat = beta[0] + beta[1] * post[time].to_numpy(dtype=float)
        out[u] = float((post[outcome].to_numpy(dtype=float) - yhat).mean())
    return pd.Series(out)


def _cross_section_did(ybar, treated):
    """OLS of the collapsed outcome on (1, D): returns (tau, classical SE, df)."""
    y = ybar.to_numpy(dtype=float)
    d = treated.to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(d), d])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(beta[1]), float(np.sqrt(cov[1, 1])), dof


def _synthetic_common_timing(n_treat=60, n_control=140, t_max=8, s=5, effect=1.5, seed=11):
    """Seeded synthetic common-timing panel with one covariate."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treat + n_control):
        is_treated = i < n_treat
        a = rng.normal(0, 1)
        x = rng.normal(0, 1)
        for t in range(1, t_max + 1):
            post = t >= s
            y = (
                a
                + 0.4 * x
                + 0.2 * t
                + rng.normal(0, 0.6)
                + (effect if is_treated and post else 0.0)
            )
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "x": x,
                    "treat": int(is_treated and post),
                    "treated_group": int(is_treated),
                    "post": int(post),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_staggered(n_units=120, t_max=10, cohorts=(5, 7), nt_share=0.4, seed=23):
    """Seeded synthetic staggered panel: cohorts + never-treated (first_year=0)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        u = rng.uniform()
        if u < nt_share:
            g = 0
        else:
            g = cohorts[int(rng.integers(0, len(cohorts)))]
        a = rng.normal(0, 1)
        for t in range(1, t_max + 1):
            te = 1.0 + 0.1 * (t - g) if (g > 0 and t >= g) else 0.0
            y = a + 0.15 * t + rng.normal(0, 0.5) + te
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "first_year": g,
                    "treat": int(g > 0 and t >= g),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Prop 99, Table 3 (38-state donor pool)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
class TestProp99Table3Goldens:
    """LW (2026) Table 3: the authors' Stata `lwdid` output, frozen in print."""

    @pytest.fixture(scope="class")
    def prop99(self):
        return _real_prop99()

    @pytest.mark.parametrize("rolling", ["demean", "detrend"])
    def test_average_att_and_se(self, prop99, rolling):
        res = _fit_prop99(prop99, rolling)
        att, se = TABLE3_AVERAGE[rolling]
        np.testing.assert_allclose(res.att, att, atol=PRINTED_ATOL)
        np.testing.assert_allclose(res.se, se, atol=PRINTED_ATOL)

    @pytest.mark.parametrize("rolling", ["demean", "detrend"])
    def test_per_period_atts(self, prop99, rolling):
        res = _fit_prop99(prop99, rolling, period_specific=True)
        assert res.period_effects, "period_specific=True should populate period_effects"
        for year, (att, se) in TABLE3_PER_PERIOD[rolling].items():
            eff = res.period_effects[year]
            np.testing.assert_allclose(eff["att"], att, atol=PRINTED_ATOL)
            np.testing.assert_allclose(eff["se"], se, atol=PRINTED_ATOL)

    def test_detrend_exact_inference_p_value(self, prop99):
        res = _fit_prop99(prop99, "detrend")
        np.testing.assert_allclose(res.p_value, TABLE3_DETREND_EXACT_P, atol=PRINTED_ATOL)

    @pytest.mark.xfail(
        strict=False,
        reason="PR #588 step-2 discussion: RI p-value convention diverges "
        "from LW 2026 Table 3 Note 2 (implementation gives the seed-stable "
        "~2/39 two-sided exact permutation atom for N1=1 among 39 states - "
        "arguably the standard exact answer - vs the paper's 0.020, whose "
        "permutation scheme is under-documented; see the maintainer review "
        "doc Gaps section). Reconcile against the authors' Stata "
        "`lwdid, ri` behavior.",
    )
    def test_detrend_randomization_inference_p_value(self, prop99):
        from diff_diff.lwdid_randomization import randomization_inference

        ybar = _detrend_reference(prop99, "state", "year", "lcigsale", pre_end=1988)
        treated = prop99.groupby("state")["first_year"].first().loc[ybar.index] > 0
        ri = randomization_inference(
            ybar.to_numpy(), treated.to_numpy(dtype=float), n_reps=1000, seed=2026
        )
        # Paper: RI p = 0.020 from 1,000 permutations; tolerance covers ~3
        # binomial standard errors at p ~= 0.02 with 1,000 replications.
        np.testing.assert_allclose(ri.pvalue, TABLE3_DETREND_RI_P, atol=0.015)


# ---------------------------------------------------------------------------
# 2. Prop 99, alternative donor pools (Tables 4 and A1)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
class TestDonorPoolVariants:
    """LW (2026) Tables 4 / A1: Southern and Midwestern donor pools."""

    @pytest.fixture(scope="class")
    def prop99(self):
        return _real_prop99()

    @pytest.mark.parametrize(
        "pool,targets",
        [(SOUTHERN_POOL, TABLE4_AVERAGE), (MIDWEST_POOL, TABLEA1_AVERAGE)],
        ids=["table4_southern", "tableA1_midwest"],
    )
    @pytest.mark.parametrize("rolling", ["demean", "detrend"])
    def test_average_att_and_se(self, prop99, pool, targets, rolling):
        sub = prop99[prop99["state"].isin(pool + ["California"])]
        assert sub["state"].nunique() == 5
        res = _fit_prop99(sub, rolling)
        att, se = targets[rolling]
        np.testing.assert_allclose(res.att, att, atol=PRINTED_ATOL)
        np.testing.assert_allclose(res.se, se, atol=PRINTED_ATOL)


# ---------------------------------------------------------------------------
# 3. Castle doctrine (Section 7.2): the aggregation adjudicator
# ---------------------------------------------------------------------------


@pytest.mark.realdata
class TestCastleTauOmegaAdjudicator:
    """LW (2026) Section 7.2: staggered tau_omega via composite regression.

    This class adjudicates PR #588's staggered aggregation: the paper's
    tau_omega comes from the single composite-outcome regression
    (7.18)/(7.19); a numerically different aggregation cannot reproduce
    the paper's point estimate and OLS SE simultaneously.
    """

    @pytest.fixture(scope="class")
    def castle(self):
        if not _CASTLE_CSV.exists():
            pytest.skip(f"{_CASTLE_CSV.name} not committed (partial checkout)")
        df = pd.read_csv(_CASTLE_CSV)
        # Data-integrity assertion: this must be the paper's exact sample.
        fy = df.groupby("state")["effyear"].first()
        counts = fy.dropna().astype(int).value_counts().sort_index().to_dict()
        assert counts == CASTLE_COHORTS, f"castle cohorts {counts} != paper {CASTLE_COHORTS}"
        assert df["state"].nunique() == 50
        assert int(fy.isna().sum()) == 29
        df = df.copy()
        df["first_year"] = df["effyear"].fillna(0).astype(int)
        df["treat"] = ((df["first_year"] > 0) & (df["year"] >= df["first_year"])).astype(int)
        return df

    @staticmethod
    def _fit(castle, rolling):
        # The paper's per-cohort effects (7.10) use never-treated controls;
        # PR #588's default control pool is not-yet-treated, which moves the
        # castle point estimate from 0.092 to 0.074 (calibrated 2026-07-13).
        est = LWDiD(rolling=rolling, estimator="ra", vce="classical", control_group="never_treated")
        return est.fit(
            castle,
            outcome="lhomicide",
            unit="state",
            time="year",
            treatment="treat",
            cohort="first_year",
        )

    @staticmethod
    def _composite_reference(castle, rolling):
        """From-scratch LW 2026 (7.18)/(7.19): composite outcome + single
        cross-sectional regression. Reproduces the paper's printed targets
        (verified: demean 0.0917/SE 0.0571; detrend 0.0666)."""
        fy = castle.groupby("state")["first_year"].first()
        cohorts = sorted(set(fy[fy > 0]))
        n_treat = int((fy > 0).sum())
        transform = _demean_reference if rolling == "demean" else _detrend_reference
        ydot = {g: transform(castle, "state", "year", "lhomicide", pre_end=g - 1) for g in cohorts}
        y, d = [], []
        for s in fy.index:
            g = fy[s]
            if g > 0:
                y.append(ydot[g][s])
                d.append(1.0)
            else:
                y.append(sum((fy[fy == gg].size / n_treat) * ydot[gg][s] for gg in cohorts))
                d.append(0.0)
        tau, se, dof = _cross_section_did(pd.Series(y), pd.Series(d))
        return tau, se

    def test_composite_regression_reference_reproduces_paper(self, castle):
        """The (7.18)/(7.19) reference implementation hits the printed targets -
        proving the targets are reproducible and the sample construction is
        correct, independent of the PR #588 estimator."""
        tau_dm, se_dm = self._composite_reference(castle, "demean")
        np.testing.assert_allclose(tau_dm, CASTLE_TAU_DEMEAN[0], atol=PRINTED_ATOL)
        np.testing.assert_allclose(se_dm, CASTLE_TAU_DEMEAN[1], atol=PRINTED_ATOL)
        tau_dt, _ = self._composite_reference(castle, "detrend")
        np.testing.assert_allclose(tau_dt, CASTLE_TAU_DETREND, atol=PRINTED_ATOL)

    def test_demean_tau_omega_point(self, castle):
        res = self._fit(castle, "demean")
        np.testing.assert_allclose(res.att, CASTLE_TAU_DEMEAN[0], atol=PRINTED_ATOL)

    @pytest.mark.xfail(
        strict=True,
        reason="PR #588 step-2 aggregation: the overall SE must come from the "
        "composite-outcome regression (7.18)/(7.19) (paper OLS SE 0.057; "
        "implementation's independence-across-cohorts SE gives 0.051). "
        "Remove this marker in the commit that adopts the composite "
        "regression.",
    )
    def test_demean_tau_omega_ols_se(self, castle):
        res = self._fit(castle, "demean")
        np.testing.assert_allclose(res.se, CASTLE_TAU_DEMEAN[1], atol=PRINTED_ATOL)

    def test_detrend_tau_omega_point(self, castle):
        res = self._fit(castle, "detrend")
        np.testing.assert_allclose(res.att, CASTLE_TAU_DETREND, atol=PRINTED_ATOL)


# ---------------------------------------------------------------------------
# 4. Translation invariance of SEs (property test)
# ---------------------------------------------------------------------------


class TestTranslationInvariance:
    """A constant added to all post-period outcomes shifts every transformed
    outcome equally: the ATT is invariant, and any correct SE is invariant."""

    SHIFT = 100.0

    def _fit_pair(self, estimator):
        df = _synthetic_common_timing()
        df2 = df.copy()
        df2["y"] = df2["y"] + self.SHIFT * df2["post"]
        kw = dict(outcome="y", unit="unit", time="time", treatment="treat", controls=["x"])
        r1 = LWDiD(rolling="demean", estimator=estimator, vce="hc1").fit(df, **kw)
        r2 = LWDiD(rolling="demean", estimator=estimator, vce="hc1").fit(df2, **kw)
        return r1, r2

    @pytest.mark.parametrize("estimator", ["ra", "ipwra", "ipw"])
    def test_att_translation_invariant(self, estimator):
        r1, r2 = self._fit_pair(estimator)
        np.testing.assert_allclose(r1.att, r2.att, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("estimator", ["ra", "ipwra"])
    def test_se_translation_invariant(self, estimator):
        r1, r2 = self._fit_pair(estimator)
        np.testing.assert_allclose(r1.se, r2.se, rtol=0, atol=1e-10)

    @XFAIL_IPW_CENTERING
    def test_ipw_se_translation_invariant(self):
        r1, r2 = self._fit_pair("ipw")
        np.testing.assert_allclose(r1.se, r2.se, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Cross-estimator equivalences
# ---------------------------------------------------------------------------


class TestCrossEstimatorEquivalence:
    """Theory-mandated numerical identities against already-validated code."""

    def test_demean_ra_equals_plain_did(self):
        """Common timing, no covariates: rolling demeaning + RA reproduces the
        standard DiD estimator exactly (LW 2026, eq. (2.5) / Section 9)."""
        df = _synthetic_common_timing()
        lw = LWDiD(rolling="demean", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        dd = DifferenceInDifferences().fit(df, outcome="y", treatment="treated_group", post="post")
        np.testing.assert_allclose(lw.att, dd.att, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("r", [5, 7])
    def test_per_period_equals_subset_panel_did(self, r):
        """LW 2026 eq. (2.20): the per-period effect tau_hat_{t,DD} is
        numerically identical to a standard panel DiD run on periods
        {1, ..., S-1, t}."""
        df = _synthetic_common_timing(t_max=8, s=5)
        res = LWDiD(rolling="demean", estimator="ra", vce="classical", period_specific=True).fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        sub = df[(df["time"] < 5) | (df["time"] == r)]
        dd = DifferenceInDifferences().fit(sub, outcome="y", treatment="treated_group", post="post")
        np.testing.assert_allclose(res.period_effects[r]["att"], dd.att, rtol=0, atol=1e-10)

    def test_detrend_t3_closed_form(self):
        """LW 2025 eq. (5.7): with T=3, S=3, no covariates, the detrending
        estimator equals the difference-in-difference-in-differences of
        period means."""
        rng = np.random.default_rng(7)
        rows = []
        for i in range(80):
            treated = i < 30
            a, b = rng.normal(0, 1), rng.normal(0.1, 0.05)
            for t in (1, 2, 3):
                y = a + b * t + rng.normal(0, 0.3) + (0.8 if treated and t == 3 else 0.0)
                rows.append({"unit": i, "time": t, "y": y, "treat": int(treated and t == 3)})
        df = pd.DataFrame(rows)
        res = LWDiD(rolling="detrend", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        m = df.assign(g=(df["unit"] < 30).astype(int)).groupby(["g", "time"])["y"].mean()
        closed = ((m[1, 3] - m[1, 2]) - (m[0, 3] - m[0, 2])) - (
            (m[1, 2] - m[1, 1]) - (m[0, 2] - m[0, 1])
        )
        np.testing.assert_allclose(res.att, closed, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. From-scratch reference implementations
# ---------------------------------------------------------------------------


class TestFromScratchReference:
    """Pure-numpy reimplementation of the procedures, blind to the estimator."""

    def test_procedure_2_1_demeaning(self):
        df = _synthetic_common_timing(t_max=8, s=5)
        ybar = _demean_reference(df, "unit", "time", "y", pre_end=4)
        treated = df.groupby("unit")["treated_group"].first().loc[ybar.index]
        tau, se, dof = _cross_section_did(ybar, treated)
        res = LWDiD(rolling="demean", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        np.testing.assert_allclose(res.att, tau, rtol=0, atol=1e-10)
        np.testing.assert_allclose(res.se, se, rtol=1e-8)

    def test_procedure_3_1_detrending(self):
        df = _synthetic_common_timing(t_max=8, s=5)
        ybar = _detrend_reference(df, "unit", "time", "y", pre_end=4)
        treated = df.groupby("unit")["treated_group"].first().loc[ybar.index]
        tau, se, dof = _cross_section_did(ybar, treated)
        res = LWDiD(rolling="detrend", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        np.testing.assert_allclose(res.att, tau, rtol=0, atol=1e-10)
        np.testing.assert_allclose(res.se, se, rtol=1e-8)

    def test_staggered_per_cohort_demeaning(self):
        """LW 2026 (7.4)/(7.9)/(7.10): per-cohort post-average demeaned
        outcome, regression on the cohort + never-treated subsample
        (control_group='never_treated' matches the (7.10) sample)."""
        df = _synthetic_staggered()
        res = LWDiD(
            rolling="demean", estimator="ra", vce="classical", control_group="never_treated"
        ).fit(df, outcome="y", unit="unit", time="time", treatment="treat", cohort="first_year")
        assert res.cohort_effects, "staggered fit should populate cohort_effects"
        fy = df.groupby("unit")["first_year"].first()
        for g in sorted(set(fy[fy > 0])):
            members = fy.index[(fy == g) | (fy == 0)]
            sub = df[df["unit"].isin(members)]
            ybar = _demean_reference(sub, "unit", "time", "y", pre_end=g - 1)
            treated = (fy.loc[ybar.index] == g).astype(int)
            tau, se, dof = _cross_section_did(ybar, treated)
            eff = res.cohort_effects[g]
            np.testing.assert_allclose(eff["att"], tau, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Exact small-sample inference (LW 2026, Section 2)
# ---------------------------------------------------------------------------


class TestExactSmallSampleInference:
    """The collapsed cross-sectional regression carries exact t inference."""

    def test_classical_p_value_uses_t_n_minus_2(self):
        df = _synthetic_common_timing(n_treat=6, n_control=10, t_max=6, s=4)
        res = LWDiD(rolling="demean", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        n = df["unit"].nunique()
        p_expected = 2 * stats.t.sf(abs(res.t_stat), n - 2)
        np.testing.assert_allclose(res.p_value, p_expected, rtol=1e-10)

    def test_single_treated_unit_inference_is_finite(self):
        """N1 = 1: exact inference remains valid (studentized-residual t)."""
        df = _synthetic_common_timing(n_treat=1, n_control=12, t_max=6, s=4)
        res = LWDiD(rolling="demean", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se) and res.se > 0
        assert np.isfinite(res.p_value)
        n = df["unit"].nunique()
        p_expected = 2 * stats.t.sf(abs(res.t_stat), n - 2)
        np.testing.assert_allclose(res.p_value, p_expected, rtol=1e-10)

    @pytest.mark.xfail(
        strict=True,
        reason="PR #588 step-2: no N_infinity >= 2 guard exists for the "
        "never-treated-only staggered control strategy (LW 2026, p26).",
    )
    def test_never_treated_pool_of_one_is_rejected(self):
        df = _synthetic_staggered(n_units=30, nt_share=0.0, seed=5)
        # Force exactly one never-treated unit
        first_unit = df["unit"] == 0
        df.loc[first_unit, "first_year"] = 0
        df.loc[first_unit, "treat"] = 0
        with pytest.raises(ValueError, match="[Nn]ever[- ]treated"):
            # The N_infinity >= 2 requirement applies to the NT-only control
            # strategy (LW 2026, p26); NYT controls are exempt.
            LWDiD(
                rolling="demean",
                estimator="ra",
                vce="classical",
                control_group="never_treated",
            ).fit(df, outcome="y", unit="unit", time="time", treatment="treat", cohort="first_year")


# ---------------------------------------------------------------------------
# 8. Event-study specification (normative API; Option A work)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
class TestEventStudySpec:
    """Normative event-study API for LWDiD (maintainer-specified).

    Invocation: ``fit(..., aggregate="event_study")`` (CallawaySantAnna
    precedent). Results must expose ``event_study_effects: Dict[int, Dict]``
    keyed by relative period r, each with keys ``effect, se, t_stat,
    p_value, conf_int`` and (when simultaneous bands are computed via
    Algorithm 1) ``cband_conf_int``; result-level metadata ``cband_method,
    cband_crit_value, cband_n_bootstrap``. Anchor periods are excluded:
    r = -1 (demeaning); r = -2, -1 (detrending).

    All tests xfail until PR #588's Appendix D + Algorithm 1 work lands.
    """

    @pytest.fixture(scope="class")
    def walmart(self):
        return _real_walmart()

    @pytest.fixture(scope="class")
    def golden(self):
        if not _WALMART_ES_GOLDEN.exists():
            pytest.skip(f"{_WALMART_ES_GOLDEN.name} not committed (partial checkout)")
        return json.loads(_WALMART_ES_GOLDEN.read_text())

    def _fit_es(self, walmart, rolling, estimator, outcome="log_retail_emp"):
        # The golden SEs are Algorithm 1 multiplier-bootstrap SEs (B = 999):
        # the spec requires the bootstrap path, not analytical vce.
        est = LWDiD(rolling=rolling, estimator=estimator, n_bootstrap=999, bootstrap_seed=42)
        return est.fit(
            walmart,
            outcome=outcome,
            unit="cid",
            time="year",
            treatment="treated",
            cohort="first_year",
            aggregate="event_study",
        )

    @XFAIL_EVENT_STUDY
    @pytest.mark.parametrize(
        "outcome,table_key",
        [
            ("log_retail_emp", "table_a4_log_retail"),
            ("log_wholesale_emp", "table_a5_log_wholesale"),
        ],
        ids=["a4_retail", "a5_wholesale"],
    )
    @pytest.mark.parametrize(
        "rolling,estimator,column",
        [
            ("detrend", "ra", "rolling_ra_detrend"),
            ("detrend", "ipwra", "rolling_ipwra_detrend"),
            ("demean", "ipwra", "rolling_ipwra_demean"),
        ],
    )
    def test_walmart_eventstudy_point_goldens(
        self, walmart, golden, rolling, estimator, column, outcome, table_key
    ):
        """Deterministic WATT(r) point estimates vs Tables A4/A5 (strict)."""
        res = self._fit_es(walmart, rolling, estimator, outcome=outcome)
        table = golden[table_key]
        for r_str, cols in table.items():
            r = int(r_str)
            att, _se = cols[column]
            eff = res.event_study_effects[r]
            np.testing.assert_allclose(eff["effect"], att, atol=PRINTED_ATOL)

    @XFAIL_EVENT_STUDY_GOLDENS
    @pytest.mark.parametrize(
        "outcome,table_key",
        [
            ("log_retail_emp", "table_a4_log_retail"),
            ("log_wholesale_emp", "table_a5_log_wholesale"),
        ],
        ids=["a4_retail", "a5_wholesale"],
    )
    @pytest.mark.parametrize(
        "rolling,estimator,column",
        [
            ("detrend", "ra", "rolling_ra_detrend"),
            ("detrend", "ipwra", "rolling_ipwra_detrend"),
            ("demean", "ipwra", "rolling_ipwra_demean"),
        ],
    )
    def test_walmart_eventstudy_se_goldens(
        self, walmart, golden, rolling, estimator, column, outcome, table_key
    ):
        """Bootstrap SEs vs the paper's printed B=999 draws (non-strict:
        re-seeded bootstrap noise can sit near printed precision)."""
        res = self._fit_es(walmart, rolling, estimator, outcome=outcome)
        table = golden[table_key]
        for r_str, cols in table.items():
            r = int(r_str)
            _att, se = cols[column]
            eff = res.event_study_effects[r]
            np.testing.assert_allclose(eff["se"], se, atol=PRINTED_ATOL)

    @XFAIL_EVENT_STUDY
    def test_anchor_periods_excluded(self, walmart):
        res_dm = self._fit_es(walmart, "demean", "ra")
        assert -1 not in res_dm.event_study_effects
        res_dt = self._fit_es(walmart, "detrend", "ra")
        assert -1 not in res_dt.event_study_effects
        assert -2 not in res_dt.event_study_effects

    def test_detrend_insample_residuals_sum_to_zero(self):
        """LW 2026 (pp20-21): per-unit detrended residuals sum to zero over
        the fitted pre-window - a property of OLS-with-intercept residuals
        IN SAMPLE. NOTE: this identity does NOT transfer to the reported
        staggered placebo coefficients (anchor-excluded, D.3-pooled,
        cohort-weighted contrasts) - do not assert a sum-to-zero on
        ``event_study_effects``; a correct implementation need not satisfy
        it there. Verified here on the reference transformation."""
        df = _synthetic_common_timing(t_max=8, s=5)
        for _, g in df.groupby("unit"):
            g = g.sort_values("time")
            pre = g[g["time"] <= 4]
            X = np.column_stack([np.ones(len(pre)), pre["time"].to_numpy(dtype=float)])
            beta, *_ = np.linalg.lstsq(X, pre["y"].to_numpy(dtype=float), rcond=None)
            resid = pre["y"].to_numpy(dtype=float) - X @ beta
            np.testing.assert_allclose(resid.sum(), 0.0, atol=1e-9)

    @XFAIL_EVENT_STUDY
    def test_simultaneous_band_metadata(self, walmart):
        res = self._fit_es(walmart, "detrend", "ra")
        assert res.cband_method is not None
        assert res.cband_n_bootstrap >= 999
        any_r = next(iter(res.event_study_effects))
        assert "cband_conf_int" in res.event_study_effects[any_r]


# ---------------------------------------------------------------------------
# 9. Monte Carlo bias ordering (LW 2026, Section 5)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMonteCarloBiasOrdering:
    """Under the paper's heterogeneous-trend DGP (Table 1, Scenario 1),
    demeaning is badly biased while detrending is nearly unbiased
    (Table 2: bias 1.914 vs 0.009)."""

    N, T, S = 20, 20, 11
    LAMBDAS = np.array(
        [0, 0, 0, 0, 0.2, 0.6, 0.7, 0.8, 0.6, 0.9, 0.9, 1, 1.1, 1.3, 1.2, 1.5, 0.6, 1.4, 1.8, 1.9]
    )
    DELTAS = np.array([1, 2, 3, 3, 3, 2, 2, 2, 1, 1], dtype=float)

    def _one_rep(self, rng):
        n, t_max, s = self.N, self.T, self.S
        c = rng.normal(0, 2, n)
        g = rng.normal(1, 1, n)
        a0, a1, a2 = -1.0, -1.0 / 3.0, 0.25
        d = (a0 - a1 * c + a2 * g + rng.logistic(0, 1, n) > 0).astype(int)
        if d.sum() in (0, n):  # degenerate assignment; caller redraws
            return None
        u = np.zeros((n, t_max))
        u[:, 0] = rng.normal(0, np.sqrt(2 / (1 - 0.75**2)), n)
        for t in range(1, t_max):
            u[:, t] = 0.75 * u[:, t - 1] + rng.normal(0, np.sqrt(2), n)
        ts = np.arange(1, t_max + 1)
        y0 = self.LAMBDAS[None, :] - c[:, None] + g[:, None] * ts[None, :] + u
        y = y0.copy()
        post = ts >= s
        delta_full = np.zeros(t_max)
        delta_full[s - 1 :] = self.DELTAS
        y[d == 1] = y0[d == 1] + delta_full[None, :] + rng.normal(0, np.sqrt(2), (d.sum(), t_max))
        sample_att = delta_full[post].mean()
        df = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n), t_max),
                "time": np.tile(ts, n),
                "y": y.ravel(),
                "treat": (np.repeat(d, t_max) * np.tile(post.astype(int), n)),
            }
        )
        out = {}
        for rolling in ("demean", "detrend"):
            res = LWDiD(rolling=rolling, estimator="ra", vce="classical").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
            out[rolling] = res.att - sample_att
        return out

    def test_bias_ordering_under_heterogeneous_trends(self):
        rng = np.random.default_rng(20260713)
        errs = {"demean": [], "detrend": []}
        reps = 0
        while reps < 200:
            rep = self._one_rep(rng)
            if rep is None:
                continue
            errs["demean"].append(rep["demean"])
            errs["detrend"].append(rep["detrend"])
            reps += 1
        bias_dm = abs(float(np.mean(errs["demean"])))
        bias_dt = abs(float(np.mean(errs["detrend"])))
        assert bias_dt < 0.5, f"detrending bias {bias_dt:.3f} unexpectedly large"
        assert bias_dm > 3 * max(
            bias_dt, 0.15
        ), f"bias ordering violated: demean {bias_dm:.3f} vs detrend {bias_dt:.3f}"


# ---------------------------------------------------------------------------
# 10. Minimum pre-treatment periods (REGISTRY edge case)
# ---------------------------------------------------------------------------


class TestMinimumPrePeriods:
    """Demeaning requires >= 1 pre-period; detrending >= 2 (rank condition,
    LW 2025 Appendix B)."""

    @staticmethod
    def _panel(first_period_treated):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(30):
            treated = i < 10
            for t in range(1, 6):
                post = t >= first_period_treated
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "y": rng.normal(0, 1) + (0.5 if treated and post else 0),
                        "treat": int(treated and post),
                    }
                )
        return pd.DataFrame(rows)

    def test_demeaning_with_single_pre_period_works(self):
        df = self._panel(first_period_treated=2)  # exactly one pre-period
        res = LWDiD(rolling="demean", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert np.isfinite(res.att)

    def test_detrending_with_single_pre_period_warns_and_nans(self):
        """One pre-period is rank-deficient for detrending. Calibrated
        behavior (2026-07-13): the implementation warns per unit and returns
        NaN inference - loud, house-compatible (warn + NaN, never silent)."""
        df = self._panel(first_period_treated=2)  # one pre-period: rank-deficient
        with pytest.warns(UserWarning, match="at least 2 pre-treatment periods"):
            res = LWDiD(rolling="detrend", estimator="ra", vce="classical").fit(
                df, outcome="y", unit="unit", time="time", treatment="treat"
            )
        assert np.isnan(res.att)
        assert np.isnan(res.se)
        # The FULL inference tuple must be NaN together (house NaN contract)
        assert np.isnan(res.t_stat)
        assert np.isnan(res.p_value)
        assert np.isnan(res.conf_int[0]) and np.isnan(res.conf_int[1])

    def test_detrending_with_two_pre_periods_works(self):
        df = self._panel(first_period_treated=3)  # two pre-periods: minimum
        res = LWDiD(rolling="detrend", estimator="ra", vce="classical").fit(
            df, outcome="y", unit="unit", time="time", treatment="treat"
        )
        assert np.isfinite(res.att)
