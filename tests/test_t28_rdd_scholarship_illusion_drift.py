"""Drift detection for Tutorial 28
(``docs/tutorials/28_rdd_scholarship_illusion.ipynb``).

The tutorial's narrative rests on these quantitative claims:

1. **The illusion.** The naive above-vs-below-cutoff earnings gap is
   ~$14,779, versus a true offer ITT of $2,640 by construction
   ((0.72 - 0.06) x $4,000).
2. **Sharp RD nails the ITT.** The robust bias-corrected offer effect is
   ~$2,629 with CI [$1,601, $3,657], at the mserd-selected h ~ 10.62.
3. **The validity toolkit is quiet.** Balance jumps for parental income
   (~$20, p ~ 0.99) and GPA (~-0.02, p ~ 0.27); placebo cutoffs at the
   within-side medians find nothing (p ~ 0.33 / 0.74); the manual-h sweep
   {3, 5, 8, 12, 15} stays between ~$2.5k and ~$3.9k with every CI
   excluding zero.
4. **Fuzzy RD recovers the enrollment effect.** First stage ~0.701;
   complier LATE ~$3,765 with CI [$1,952, $5,579], covering the $4,000
   truth.
5. **Covariates buy precision.** The adjusted fit reports ~$2,581 with a
   ~27% shorter CI than the unadjusted sharp fit.

``nbsphinx_execute = "never"`` means RTD renders the committed outputs, so
CI cannot detect drift by re-executing the notebook. These tests re-derive
the load-bearing numbers from the same public API the notebook uses and
cross-check the rendered surface (markdown + committed outputs), so library
drift or notebook-only edits fail loudly.

All fits here are single deterministic local-polynomial fits (no
bootstrap), so nothing needs ``ci_params`` scaling or a slow marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import RDPlot, RegressionDiscontinuity
from tests._tutorial_drift import assert_quotes_in_rendered

NB = "docs/tutorials/28_rdd_scholarship_illusion.ipynb"

# Locked design - must stay in sync with the notebook's DGP cell
# (cross-checked by ``test_notebook_constants_match``).
SEED = 25
N = 8_000
CUTOFF = 65.0
TRUE_EFFECT = 4_000.0
P_ENROLL_IF_OFFERED = 0.72
P_ALWAYS = 0.06
TRUE_ITT = (P_ENROLL_IF_OFFERED - P_ALWAYS) * TRUE_EFFECT  # 2640.0

H_GRID = (3, 5, 8, 12, 15)

# Single-fit local-polynomial estimates: cross-OS BLAS variation is
# ULP-scale (see .claude/memory.md tolerance notes), so dollar-scale pins
# use abs=0.01 and probability-scale pins abs=1e-6.
DOLLAR_TOL = 0.01
P_TOL = 1e-6


def make_cohort() -> pd.DataFrame:
    """Faithful copy of the notebook's DGP cell."""
    rng = np.random.default_rng(SEED)
    ability = rng.normal(0.0, 1.0, N)
    score = np.clip(64 + 11 * ability + rng.normal(0, 4, N), 30, 100)
    parental_income = np.clip(45_000 + 600 * (score - 60) + rng.normal(0, 15_000, N), 8_000, None)
    hs_gpa = np.clip(2.9 + 0.016 * (score - 60) + rng.normal(0, 0.30, N), 0.0, 4.0)
    offer = (score >= CUTOFF).astype(int)
    u = rng.uniform(size=N)
    enrolled = np.where(offer == 1, (u < P_ENROLL_IF_OFFERED), (u < P_ALWAYS)).astype(int)
    base = 24_000 + 320 * score + 7.0 * (score - 60) ** 2
    earnings = (
        base
        + 0.35 * (parental_income - 45_000)
        + 4_000 * (hs_gpa - 2.9)
        + TRUE_EFFECT * enrolled
        + rng.normal(0, 4_500, N)
    )
    return pd.DataFrame(
        {
            "score": score,
            "offer": offer,
            "enrolled": enrolled,
            "earnings": earnings,
            "parental_income": parental_income,
            "hs_gpa": hs_gpa,
        }
    )


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_cohort()


@pytest.fixture(scope="module")
def sharp(df):
    return RegressionDiscontinuity(cutoff=CUTOFF).fit(df, outcome="earnings", running="score")


class TestTutorial28Drift:
    def test_naive_gap_is_the_illusion(self, df):
        naive = df.loc[df.offer == 1, "earnings"].mean() - df.loc[df.offer == 0, "earnings"].mean()
        assert naive == pytest.approx(14778.9046, abs=DOLLAR_TOL)
        # the story: the naive gap overstates the true ITT >5x
        assert naive > 5 * TRUE_ITT

    def test_sharp_rd_pins(self, sharp):
        assert sharp.att == pytest.approx(2628.7444, abs=DOLLAR_TOL)
        assert sharp.se == pytest.approx(524.4613, abs=DOLLAR_TOL)
        assert sharp.conf_int[0] == pytest.approx(1600.8193, abs=DOLLAR_TOL)
        assert sharp.conf_int[1] == pytest.approx(3656.6696, abs=DOLLAR_TOL)
        assert sharp.att_conventional == pytest.approx(2616.1764, abs=DOLLAR_TOL)
        assert sharp.h_left == pytest.approx(10.6211, abs=1e-4)
        # the CI covers the designed truth
        assert sharp.conf_int[0] < TRUE_ITT < sharp.conf_int[1]

    def test_balance_pins(self, df):
        rd = RegressionDiscontinuity(cutoff=CUTOFF)
        income = rd.fit(df, outcome="parental_income", running="score")
        assert income.att == pytest.approx(20.1542, abs=DOLLAR_TOL)
        assert income.p_value == pytest.approx(0.986901, abs=P_TOL)
        gpa = rd.fit(df, outcome="hs_gpa", running="score")
        assert gpa.att == pytest.approx(-0.023270, abs=1e-5)
        assert gpa.p_value == pytest.approx(0.272935, abs=P_TOL)

    def test_placebo_pins(self, df):
        left = df[df.score < CUTOFF]
        right = df[df.score >= CUTOFF]
        pins = [
            (left, 56.9195, 1161.7591, 0.333126),
            (right, 72.6014, -349.1510, 0.742107),
        ]
        for side, fake_median, att, p in pins:
            fake = float(side.score.median())
            assert fake == pytest.approx(fake_median, abs=1e-4)
            fit = RegressionDiscontinuity(cutoff=fake).fit(
                side, outcome="earnings", running="score"
            )
            assert fit.att == pytest.approx(att, abs=DOLLAR_TOL)
            assert fit.p_value == pytest.approx(p, abs=P_TOL)

    def test_bandwidth_sweep_pins(self, df):
        pins = {
            3: (3905.1486, 1394.9944, 6415.3028),
            5: (3341.4404, 1441.2065, 5241.6743),
            8: (3005.2456, 1526.5811, 4483.9102),
            12: (2533.9585, 1335.5907, 3732.3262),
            15: (2606.9459, 1528.4022, 3685.4897),
        }
        for h in H_GRID:
            att, lo, hi = pins[h]
            fit = RegressionDiscontinuity(cutoff=CUTOFF, h=h).fit(
                df, outcome="earnings", running="score"
            )
            assert fit.att == pytest.approx(att, abs=DOLLAR_TOL)
            assert fit.conf_int[0] == pytest.approx(lo, abs=DOLLAR_TOL)
            assert fit.conf_int[1] == pytest.approx(hi, abs=DOLLAR_TOL)
            # the prose claims every interval excludes zero AND covers the
            # designed truth
            assert fit.conf_int[0] > 0
            assert fit.conf_int[0] < TRUE_ITT < fit.conf_int[1]

    def test_rdplot_acts_pins(self, df):
        # Act 2's two fits: full support (defaults) and the p=2 zoom with
        # per-bin CIs, exactly as configured in the notebook.
        full = RDPlot(cutoff=CUTOFF).fit(df, outcome="earnings", running="score")
        assert full.J == (123.0, 163.0)
        assert not full.ci_requested

        window = df[df.score.between(CUTOFF - 10, CUTOFF + 10)]
        assert len(window) == 4890
        zoom_est = RDPlot(cutoff=CUTOFF, p=2, ci=95)
        assert zoom_est.p == 2
        assert zoom_est.ci == 95
        zoom = zoom_est.fit(window, outcome="earnings", running="score")
        assert zoom.J == (72.0, 76.0)
        assert zoom.ci_requested

        # populated bin rows: the full fit drops empty tail bins (276 of the
        # 123+163 selected), the zoom window populates every bin (72+76)
        for res, n_rows in ((full, 276), (zoom, 148)):
            vb = res.vars_bins
            means = np.asarray(vb["rdplot_mean_y"], dtype=float)
            ci_l = np.asarray(vb["rdplot_ci_l"], dtype=float)
            ci_r = np.asarray(vb["rdplot_ci_r"], dtype=float)
            assert len(means) == n_rows
            assert np.isfinite(means).all()
            assert np.isfinite(ci_l).all() and np.isfinite(ci_r).all()

    def test_rdplot_renders_noninteractive(self, df):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        zoom = RDPlot(cutoff=CUTOFF, p=2, ci=95).fit(
            df[df.score.between(CUTOFF - 10, CUTOFF + 10)],
            outcome="earnings",
            running="score",
        )
        ax = zoom.plot(title="smoke", xlabel="score", ylabel="earnings")
        assert ax.get_title() == "smoke"
        plt.close(ax.figure)

    def test_fuzzy_pins(self, df, sharp):
        fuzzy = RegressionDiscontinuity(cutoff=CUTOFF).fit(
            df, outcome="earnings", running="score", takeup="enrolled"
        )
        assert fuzzy.estimand == "fuzzy (LATE for compliers at the cutoff)"
        assert fuzzy.first_stage is not None
        assert fuzzy.first_stage == pytest.approx(0.700885, abs=P_TOL)
        assert fuzzy.att == pytest.approx(3765.2237, abs=DOLLAR_TOL)
        assert fuzzy.conf_int[0] == pytest.approx(1951.6648, abs=DOLLAR_TOL)
        assert fuzzy.conf_int[1] == pytest.approx(5578.7826, abs=DOLLAR_TOL)
        # covers the designed enrollment effect
        assert fuzzy.conf_int[0] < TRUE_EFFECT < fuzzy.conf_int[1]
        # the tutorial's IV-arithmetic claim: approximate, not exact
        # (different bandwidths + linearized ratio bias correction)
        product = fuzzy.first_stage * fuzzy.att
        assert product == pytest.approx(sharp.att, rel=0.01)
        assert abs(product - sharp.att) > 1.0

    def test_covariate_adjustment_pins(self, df, sharp):
        adj = RegressionDiscontinuity(cutoff=CUTOFF).fit(
            df,
            outcome="earnings",
            running="score",
            covariates=["parental_income", "hs_gpa"],
        )
        assert adj.att == pytest.approx(2581.2134, abs=DOLLAR_TOL)
        assert adj.conf_int[0] == pytest.approx(1833.8958, abs=DOLLAR_TOL)
        assert adj.conf_int[1] == pytest.approx(3328.5311, abs=DOLLAR_TOL)
        shrink = 1 - (adj.conf_int[1] - adj.conf_int[0]) / (sharp.conf_int[1] - sharp.conf_int[0])
        # the "27% shorter CI in this simulation" claim
        assert round(shrink * 100) == 27

    @staticmethod
    def _notebook_code_cells() -> list:
        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip(
                f"Notebook {NB!r} not available in this CI environment "
                "(isolated-install job copies only tests/, not docs/)."
            )
        nb = json.loads(nb_path.read_text())
        return [
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"]
            if c["cell_type"] == "code"
        ]

    def test_notebook_dgp_cell_reproduces_test_cohort(self, df, capsys):
        """Full-DGP synchronization: execute the notebook's actual DGP cell
        and require its DataFrame to equal ``make_cohort()`` exactly.

        This locks EVERY term of the simulation (score parameters, baseline
        slope, earnings coefficients, ...), not a fragment allowlist - an
        edit to either copy that changes the data fails here.
        """
        cells = self._notebook_code_cells()
        dgp_cells = [c for c in cells if "SEED = 25" in c]
        assert len(dgp_cells) == 1, "expected exactly one DGP cell"
        ns: dict = {"np": np, "pd": pd}
        exec(dgp_cells[0], ns)  # noqa: S102 - trusted repo-committed notebook
        capsys.readouterr()  # swallow the cell's print() output
        pd.testing.assert_frame_equal(ns["df"], df)
        assert ns["true_itt"] == TRUE_ITT

    def test_notebook_fit_configs_match(self):
        """The fit-configuration surface outside the DGP cell."""
        src = "\n".join(self._notebook_code_cells())
        for needle in (
            "h_grid = [3, 5, 8, 12, 15]",
            'takeup="enrolled"',
            'covariates=["parental_income", "hs_gpa"]',
            "RDPlot(cutoff=CUTOFF)",
            "RDPlot(cutoff=CUTOFF, p=2, ci=95)",
            "df[df.score.between(CUTOFF - 10, CUTOFF + 10)]",
        ):
            assert needle in src, f"notebook drifted from locked config: {needle!r}"

    def test_rendered_surface_quotes(self):
        # Markdown prose quotes (reader-facing claims)...
        assert_quotes_in_rendered(
            NB,
            [
                "**\\$2,640**",  # the designed offer ITT
                "\\$2,629",  # sharp robust estimate in prose
                "**\\$3,765, CI [\\$1,952, \\$5,579]**",  # fuzzy LATE claim
                "70 percentage points",  # first-stage reading
                "p = 0.99",  # balance: parental income
                "p = 0.33 and 0.74",  # placebos
                "27% shorter",  # covariate payoff, scoped to the simulation
                "no balance test can prove continuity",  # falsification framing
                "mechanically ordered",  # fuzzy-CI width caveat
                "asymptotically valid",  # RBC coverage claim (not "honest")
                "testable sufficient condition",  # balance role (not necessity)
                "for this illustration",  # placebo medians are illustrative
                "funded enrollment",  # the treatment is enrollment, not
                # scholarship receipt (CI review)
                "`b = h`",  # h-only sweep moves the bias bandwidth too
                "first-order linearization of the ratio",  # RBC ratio caveat
            ],
            surface="markdown",
        )
        # ...and the executed-output numbers they round from.
        assert_quotes_in_rendered(
            NB,
            [
                "naive above-vs-below gap: $14,779",
                "offer effect (robust bias-corrected): $2,629",
                "95% CI: [$1,601, $3,657]",
                "jump at cutoff =   20.154",  # balance: parental income
                "p = 0.987",
                "placebo cutoff 56.9",
                "placebo cutoff 72.6",
                "h =  3: att = $ 3,905",
                "h = 15: att = $ 2,607",
                "0.7009",  # first stage (summary table)
                "3765.2237",  # fuzzy robust estimate (summary table)
                "CI shortened by 27%",
            ],
            surface="output",
        )
