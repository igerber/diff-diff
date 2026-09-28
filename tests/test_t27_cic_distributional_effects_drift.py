"""Drift detection for Tutorial 27
(``docs/tutorials/27_cic_distributional_effects.ipynb``).

The tutorial's narrative rests on five quantitative claims:

1. **The mean hides the action.** Mean DiD on spend reads ~$0.22 (p ~ 0.90)
   while the true mean effect is $3.01 by construction - and CiC's ATT
   (~$3.05) lands on the truth. The gap is bias, not just noise: the market
   trend is multiplicative, so dollar trends are not parallel.
2. **The bands split at the median.** Sup-t uniform bands exclude zero for
   exactly tau = 0.05..0.50 and nowhere above - the tutorial's joint
   "bottom half moved" claim.
3. **The guardrail is loud.** The mid-market-control variant fires the
   Assumption-3.4 support warning plus the interior-range warning, reports
   interior range (0.1698, 0.9956), and NaNs inference at tau = 0.05-0.15
   while keeping point estimates.
4. **Scale-equivariance (the centerpiece).** The unconditional CiC
   counterfactual quantiles from the levels fit equal exp() of the log-fit
   ones. The property is proven exactly (private-helper route) by
   ``test_cic_scale_invariance_nonlinear_dgp`` in
   ``tests/test_methodology_changes_in_changes.py``; here we assert the
   tutorial's PUBLIC-API construction (type-1 ``inverted_cdf`` treated-post
   quantiles minus the reported QTE - exact because the cell size 901 is
   coprime to the quantile-grid denominator 20) at rtol 1e-14, not bit
   equality: the exp/log round-trip is libm-dependent across the CI OS
   matrix. QDiD has no such property: its levels-vs-logs gap grows toward
   the top (~-0.24 / -2.10 / -5.65 at tau = 0.50 / 0.75 / 0.90) and
   QDiD-in-levels reports ~-$8.48 at tau = 0.90 where the truth is ~0.
5. **Covariates fix composition confounding.** On the tenure design (ported
   from the calibrated ``_make_shift_dgp`` in the methodology tests) the
   unconditional ATT is ~9.38 with a truth-excluding CI; ``covariates=
   ['tenure']`` gives ~6.53 with a truth-covering CI (truth 6.0).

``pytest --nbmake`` only checks that cells execute; ``nbsphinx_execute =
"never"`` means RTD renders the committed outputs. These tests re-derive the
load-bearing numbers from the same public API the notebook uses and
cross-check the rendered surface, so library drift or notebook-only edits
fail loudly.

Structure (T24/T25 split): deterministic single-fit pins run unmarked; the
covariate quantile-regression bootstrap re-derivation (~1 min of LP solves,
not ``ci_params``-scalable - it must reproduce the notebook's exact
seed/n_bootstrap numbers) is ``@pytest.mark.slow`` and runs in the ``-m ''``
CI legs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import ChangesInChanges, DifferenceInDifferences, practitioner_next_steps
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/27_cic_distributional_effects.ipynb"

# Locked design - must stay in sync with the notebook's DGP cells
# (cross-checked by ``test_notebook_constants_match``).
SEED = 27
N_CELL = 901  # 17*53 - coprime to 20 (quantile grid) and 100 (QR tau grid)
MU_LOG, SIGMA_LOG = 3.4, 0.75
GAMMA, ALPHA = 1.06, -0.17
U_LO, U_HI = 0.05, 0.90
BETA_A, BETA_B = 1.2, 2.2
LIFT_MAX, LIFT_MID, LIFT_SCALE = 0.65, 0.22, 0.07

COV_SEED, COV_N = 272, 299
COV_EFFECT = 6.0
COV_SHIFT, COV_TREND, COV_LEVEL, COV_SCALE = 0.8, 0.6, 0.15, 8.0


# --------------------------------------------------------------------------- #
# Faithful copies of the notebook's DGP helpers
# --------------------------------------------------------------------------- #
def h_pre(u):
    return np.exp(MU_LOG + SIGMA_LOG * stats.norm.ppf(u))


def h_post(u):
    return np.exp(ALPHA + GAMMA * (MU_LOG + SIGMA_LOG * stats.norm.ppf(u)))


def lift(u):
    return LIFT_MAX / (1.0 + np.exp((u - LIFT_MID) / LIFT_SCALE))


def make_spend_data(seed=SEED, n=N_CELL):
    rng = np.random.default_rng(seed)
    u00 = rng.uniform(0.0005, 0.9995, n)
    u01 = rng.uniform(0.0005, 0.9995, n)
    u10 = U_LO + (U_HI - U_LO) * rng.beta(BETA_A, BETA_B, n)
    u11 = U_LO + (U_HI - U_LO) * rng.beta(BETA_A, BETA_B, n)
    frames = []
    for g, t, y in (
        (0, 0, h_pre(u00)),
        (0, 1, h_post(u01)),
        (1, 0, h_pre(u10)),
        (1, 1, h_post(u11) * (1.0 + lift(u11))),
    ):
        frames.append(pd.DataFrame({"treated": g, "post": t, "spend": y}))
    return pd.concat(frames, ignore_index=True)


def make_midmarket_variant(seed=SEED, n=N_CELL):
    rng = np.random.default_rng(seed + 1)
    u00 = rng.uniform(0.15, 0.85, n)
    u01 = rng.uniform(0.15, 0.85, n)
    u10 = U_LO + (U_HI - U_LO) * rng.beta(BETA_A, BETA_B, n)
    u11 = U_LO + (U_HI - U_LO) * rng.beta(BETA_A, BETA_B, n)
    frames = []
    for g, t, y in (
        (0, 0, h_pre(u00)),
        (0, 1, h_post(u01)),
        (1, 0, h_pre(u10)),
        (1, 1, h_post(u11) * (1.0 + lift(u11))),
    ):
        frames.append(pd.DataFrame({"treated": g, "post": t, "spend": y}))
    return pd.concat(frames, ignore_index=True)


def make_engagement_data(seed=COV_SEED, n=COV_N):
    rng = np.random.default_rng(seed)
    frames = []
    for g in (0, 1):
        for t in (0, 1):
            x = rng.uniform(0, 2, n) + COV_SHIFT * g
            noise = rng.normal(0, 0.4, n)
            y = COV_SCALE * (0.5 + COV_LEVEL * x + COV_TREND * x * t + noise) + COV_EFFECT * g * t
            frames.append(pd.DataFrame({"treated": g, "post": t, "tenure": x, "engagement": y}))
    return pd.concat(frames, ignore_index=True)


def true_mean_effect(n_mc=2_000_000, seed=123456):
    rng = np.random.default_rng(seed)
    u = U_LO + (U_HI - U_LO) * rng.beta(BETA_A, BETA_B, n_mc)
    return float(np.mean(h_post(u) * lift(u)))


# --------------------------------------------------------------------------- #
# Shared fits (module-scoped: each is seconds or less)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def df():
    return make_spend_data()


@pytest.fixture(scope="module")
def df_log(df):
    return df.assign(log_spend=np.log(df["spend"]))


@pytest.fixture(scope="module")
def cic(df):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ChangesInChanges(n_bootstrap=999, seed=SEED).fit(
            df, outcome="spend", treatment="treated", time="post"
        )


@pytest.fixture(scope="module")
def did(df):
    return DifferenceInDifferences().fit(df, outcome="spend", treatment="treated", post="post")


class TestMainStory:
    def test_cell_sizes_coprime_to_grids(self):
        # Structural precondition of the invariance demo (no n*tau knife
        # edges for the type-1 / inverted_cdf agreement) and of the QR
        # vertex-degeneracy avoidance in the covariate section.
        assert math.gcd(N_CELL, 20) == 1
        assert math.gcd(N_CELL, 100) == 1
        assert math.gcd(COV_N, 100) == 1

    def test_main_fits_warning_free_and_support_clean(self, df, df_log, cic):
        y00 = df.query("treated == 0 and post == 0")["spend"].to_numpy()
        y10 = df.query("treated == 1 and post == 0")["spend"].to_numpy()
        assert y00.min() < y10.min() and y10.max() < y00.max()
        assert np.isfinite(cic.att)  # fixture itself fit under simplefilter("error")
        # cic fixture already fit under simplefilter("error"); the other
        # main fits must be warning-free too (incl. QDiD monotonicity on
        # BOTH scales - the prose claims it stays monotone).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DifferenceInDifferences().fit(df, outcome="spend", treatment="treated", post="post")
            ChangesInChanges(n_bootstrap=0).fit(
                df_log, outcome="log_spend", treatment="treated", time="post"
            )
            ChangesInChanges(method="qdid", n_bootstrap=0).fit(
                df, outcome="spend", treatment="treated", time="post"
            )
            ChangesInChanges(method="qdid", n_bootstrap=0).fit(
                df_log, outcome="log_spend", treatment="treated", time="post"
            )

    def test_mean_did_verdict(self, did):
        # "$0.22, p = 0.90" - insignificant AND biased (truth is ~$3.01).
        assert abs(did.att - 0.22) < 0.01
        assert abs(did.p_value - 0.900) < 0.005
        truth = true_mean_effect()
        assert abs(truth - 3.01) < 0.01
        assert abs(did.att - truth) > 2.5

    def test_cic_att_matches_truth(self, cic, did):
        assert abs(cic.att - 3.05) < 0.01
        assert cic.p_value < 0.01
        assert abs(cic.att - true_mean_effect()) < 0.1
        assert abs(cic.att - did.att) > 2.5

    def test_qte_profile_pins(self, cic):
        qe = cic.quantile_effects
        for tau, expected in ((0.05, 4.91), (0.25, 5.44), (0.50, 3.86), (0.75, 0.59)):
            got = float(qe.loc[np.isclose(qe["quantile"], tau), "qte"].iloc[0])
            assert abs(got - expected) < 0.01, f"tau={tau}: {got} vs quoted {expected}"

    def test_uniform_bands_split_at_median(self, cic):
        bands = cic.uniform_bands()
        excluded = ((bands["band_low"] > 0) | (bands["band_high"] < 0)).to_numpy()
        # The joint claim: zero excluded for EXACTLY tau = 0.05..0.50.
        expected = cic.quantiles <= 0.5
        np.testing.assert_array_equal(excluded, expected)

    def test_main_interior_range_trivial(self, cic):
        assert cic.q_lower == 0.0
        assert cic.q_upper == 1.0


class TestGuardrailVariant:
    def test_variant_warns_and_nans_the_tail(self):
        with pytest.warns(UserWarning, match="Assumption 3.4"):
            with pytest.warns(UserWarning, match="outside the point-identified interior"):
                cic_v = ChangesInChanges(n_bootstrap=999, seed=SEED).fit(
                    make_midmarket_variant(), outcome="spend", treatment="treated", time="post"
                )
        assert abs(cic_v.q_lower - 0.1698) < 0.001
        assert abs(cic_v.q_upper - 0.9956) < 0.001
        qe = cic_v.quantile_effects
        nan_taus = qe.loc[qe["se"].isna(), "quantile"].to_numpy()
        np.testing.assert_allclose(nan_taus, [0.05, 0.10, 0.15], atol=1e-12)
        # Point estimates are kept (qte parity) even where inference is NaN.
        assert np.isfinite(qe.loc[qe["se"].isna(), "qte"]).all()


class TestScaleEquivariance:
    def test_cic_counterfactual_quantiles_equivariant(self, df, df_log, cic):
        cic_log = ChangesInChanges(n_bootstrap=0).fit(
            df_log, outcome="log_spend", treatment="treated", time="post"
        )
        y11 = df.query("treated == 1 and post == 1")["spend"].to_numpy()
        grid = cic.quantiles
        q11_lvl = np.quantile(y11, grid, method="inverted_cdf")
        q11_log = np.quantile(np.log(y11), grid, method="inverted_cdf")
        cf_levels = q11_lvl - cic.quantile_effects["qte"].to_numpy()
        cf_logs = np.exp(q11_log - cic_log.quantile_effects["qte"].to_numpy())
        # rtol 1e-14, not bit equality: the exp/log round-trip is the only
        # daylight and its rounding is libm-dependent across the CI OS
        # matrix. The exact-identity property itself is proven via the
        # module's own type-1 helpers in test_methodology_changes_in_changes
        # .py::test_cic_scale_invariance_nonlinear_dgp.
        np.testing.assert_allclose(cf_levels, cf_logs, rtol=1e-14, atol=0.0)

    def test_did_flips_verdict_across_scales(self, df_log, did):
        did_log = DifferenceInDifferences().fit(
            df_log, outcome="log_spend", treatment="treated", time="post"
        )
        assert did.p_value > 0.5  # levels: "no effect"
        assert did_log.p_value < 0.005  # logs: "big effect"
        assert abs(did_log.att - 0.1307) < 0.001
        assert abs(100 * (np.exp(did_log.att) - 1) - 14.0) < 0.1

    def test_qdid_scale_gap_grows_toward_top(self, df, df_log):
        qdid = ChangesInChanges(method="qdid", n_bootstrap=0).fit(
            df, outcome="spend", treatment="treated", time="post"
        )
        qdid_log = ChangesInChanges(method="qdid", n_bootstrap=0).fit(
            df_log, outcome="log_spend", treatment="treated", time="post"
        )
        y11 = df.query("treated == 1 and post == 1")["spend"].to_numpy()
        grid = qdid.quantiles
        q11_t7_log = np.quantile(np.log(y11), grid, method="linear")
        backmapped = np.exp(q11_t7_log) - np.exp(
            q11_t7_log - qdid_log.quantile_effects["qte"].to_numpy()
        )
        gap = qdid.quantile_effects["qte"].to_numpy() - backmapped
        pins = {0.50: -0.24, 0.75: -2.10, 0.90: -5.65}
        for tau, expected in pins.items():
            got = float(gap[np.isclose(grid, tau)][0])
            assert abs(got - expected) < 0.02, f"tau={tau}: gap {got} vs quoted {expected}"
        # Direction + growth toward the top (the prose claim), and the
        # spurious levels-scale "loss" at tau=0.90 where the truth is ~0.
        assert abs(gap[np.isclose(grid, 0.90)][0]) > abs(gap[np.isclose(grid, 0.75)][0])
        assert abs(gap[np.isclose(grid, 0.75)][0]) > abs(gap[np.isclose(grid, 0.50)][0])
        qd90 = float(qdid.quantile_effects["qte"].to_numpy()[np.isclose(grid, 0.90)][0])
        assert abs(qd90 - (-8.48)) < 0.02


class TestCovariateSection:
    def test_unconditional_is_confidently_wrong(self):
        eng = make_engagement_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            unc = ChangesInChanges(n_bootstrap=999, seed=SEED).fit(
                eng, outcome="engagement", treatment="treated", time="post"
            )
        assert abs(unc.att - 9.38) < 0.02
        # CI endpoints as quoted in the prose ("CI [8.32, 10.45]").
        assert abs(unc.conf_int[0] - 8.32) < 0.02
        assert abs(unc.conf_int[1] - 10.45) < 0.02
        # CI excludes the truth - the bias is not a power problem.
        assert unc.conf_int[0] > COV_EFFECT

    @pytest.mark.slow
    def test_conditional_recovers_truth(self):
        # ~1 min of quantile-regression LP solves (2 cells x 99 taus x 100
        # fits); must reproduce the notebook's exact seed/n_bootstrap, so it
        # cannot be ci_params-scaled. Runs in the -m '' CI legs.
        eng = make_engagement_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cov = ChangesInChanges(n_bootstrap=99, seed=SEED).fit(
                eng,
                outcome="engagement",
                treatment="treated",
                time="post",
                covariates=["tenure"],
            )
        assert abs(cov.att - 6.53) < 0.1
        # CI endpoints as quoted in the prose ("CI [5.35, 7.70]") - looser
        # tolerance than the unconditional pins: quantile-regression LP
        # vertex selection can shift bootstrap replicates slightly across
        # platforms (see the parity-test header for the tie mechanics).
        assert abs(cov.conf_int[0] - 5.35) < 0.1
        assert abs(cov.conf_int[1] - 7.70) < 0.1
        assert cov.conf_int[0] < COV_EFFECT < cov.conf_int[1]
        assert abs(cov.att - COV_EFFECT) < 0.6


class TestCloseAndSurface:
    def test_practitioner_close_is_clean(self, cic):
        out = practitioner_next_steps(cic, verbose=False)
        assert out["estimator"] == "ChangesInChanges (CiC)"
        assert out["warnings"] == []
        labels = [s["label"] for s in out["next_steps"]]
        assert any("Placebo ChangesInChanges" in lbl for lbl in labels)

    def test_notebook_has_no_asserts_or_warning_filters(self):
        # The tutorial deliberately lets the guardrail-variant warnings
        # render in the committed output rather than filtering them, and
        # per the T19 rule all numerical guards live HERE, not in cells.
        import ast
        import json
        from pathlib import Path

        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip(f"Notebook not found at {nb_path}; surface checks are full-checkout only.")
        cells = [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in json.loads(nb_path.read_text())["cells"]
            if c["cell_type"] == "code"
        ]
        src = "\n".join(cells)
        assert "import warnings" not in src
        assert "filterwarnings" not in src
        assert "simplefilter" not in src
        # AST-based: catches indented asserts a substring check would miss.
        for cell_src in cells:
            tree = ast.parse(cell_src)
            asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
            assert not asserts, f"notebook cell contains assert statements:\n{cell_src[:200]}"

    def test_notebook_constants_match(self):
        import json
        from pathlib import Path

        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip(f"Notebook not found at {nb_path}; sync guard is full-checkout only.")
        src = "\n".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in json.loads(nb_path.read_text())["cells"]
            if c["cell_type"] == "code"
        )
        for needle in (
            "SEED = 27",
            "N_CELL = 901",
            "MU_LOG, SIGMA_LOG = 3.4, 0.75",
            "GAMMA, ALPHA = 1.06, -0.17",
            "U_LO, U_HI = 0.05, 0.90",
            "BETA_A, BETA_B = 1.2, 2.2",
            "LIFT_MAX, LIFT_MID, LIFT_SCALE = 0.65, 0.22, 0.07",
            "COV_SEED, COV_N = 272, 299",
            "COV_EFFECT = 6.0",
            "COV_SHIFT, COV_TREND, COV_LEVEL, COV_SCALE = 0.8, 0.6, 0.15, 8.0",
            "h_post(u11) * (1.0 + lift(u11))",
            "rng.uniform(0.15, 0.85",
            "COV_SCALE * (0.5 + COV_LEVEL * x + COV_TREND * x * t + noise)",
            "ChangesInChanges(n_bootstrap=999, seed=SEED)",
            "ChangesInChanges(n_bootstrap=99, seed=SEED)",
            'covariates=["tenure"]',
            'method="inverted_cdf"',
        ):
            assert needle in src, f"notebook drifted from locked config: {needle!r}"

    def test_rendered_surface_quotes(self):
        # Markdown prose quotes (reader-facing claims)...
        assert_quotes_in_rendered(
            NB,
            [
                "p = 0.90",
                "(0.17, 0.996)",
                "+14.0%",
                "bottom half",
                "and for none above",  # bands are silent, not exonerating,
                "absence of evidence",  # above the median (no null-acceptance)
                "p = 0.044",  # the pointwise blip the joint bands protect against
                "p. 447",
                "**6.0 points**",
            ],
            surface="markdown",
        )
        # ...and the executed-output numbers they round from.
        assert_quotes_in_rendered(
            NB,
            [
                "3.0476",  # CiC ATT
                "4.9087",  # QTE at tau=0.05
                "0.9001",  # mean-DiD p-value
                "0.1698",  # variant interior range lower bound
                "9.38",  # unconditional covariate-section ATT
                "6.53",  # conditional covariate-section ATT
                "CI [8.32, 10.45]",  # unconditional CI (truth-excluding)
                "CI [5.35, 7.70]",  # conditional CI (truth-covering)
            ],
            surface="output",
        )

    def test_no_local_paths_in_committed_outputs(self):
        # The guardrail-variant warnings render in the committed output by
        # design, but the machine-specific checkout prefix is normalized
        # away before committing - rendered docs must not leak local
        # usernames/paths, whichever platform executed the notebook.
        from tests._tutorial_drift import notebook_output_text

        text = notebook_output_text(NB)
        for prefix in ("/Users/", "/home/", "/private/", "C:\\Users", "/tmp/"):
            assert prefix not in text, f"committed output leaks a local path: {prefix!r}"

    def test_headline_claims_present_in_prose(self):
        md = notebook_markdown(NB)
        assert "When the Average Hides the Action" in md
        assert "uniform band" in md or "uniform bands" in md
        assert "interior range" in md
        assert "equivarian" in md  # equivariant / equivariance
        assert "CallawaySantAnna" in md  # the when-to-use routing
