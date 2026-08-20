"""Drift detection for Tutorial 31 (``docs/tutorials/31_lwdid.ipynb``).

The tutorial narrative quotes locked, seed-specific numbers (synthetic
TWFE-vs-LWDiD estimates, the exact-t small-N inference, the Prop 99 Table 3
replication, the RI p-value under the package convention, the Walmart WATT
replication with its near/far lead placebo maxima, and the wild-bootstrap
CI). ``pytest --nbmake`` only checks that cells *execute*; it does not check
the prose or the committed outputs. Two layers here:

1. ``assert_quotes_in_rendered`` pins the load-bearing quoted values against
   the committed rendered surface (markdown + outputs) - this is the guard
   that catches a notebook re-execution drifting away from the locked
   narrative numbers (the review-flagged failure mode: prose said "below
   0.006" while the output printed 0.0329).
2. Synthetic-DGP re-derivation: the section 1 panel is rebuilt from the
   locked seed and the TWFE/LWDiD estimates re-checked against the quoted
   values, so a library numerics change surfaces here even without
   re-executing the notebook. The empirical (Prop 99 / Walmart) numbers are
   already pinned end-to-end by ``tests/test_methodology_lwdid.py``; this
   file only guards that the notebook *quotes* them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import BaconDecomposition, LWDiD

from ._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/31_lwdid.ipynb"

# sha256[:16] of EVERY code cell's normalized source, in notebook order -
# the complete stale-output contract (see test_all_code_cells_hash_pinned)
ALL_CODE_CELL_HASHES = [
    "a60b92dc146a2807",
    "884416416c58746a",
    "79c16d470d0f3af8",
    "b6d66a5c23b7b155",
    "4d23491db4d23ab6",
    "2e512cad57e56aae",
    "8be925643ca3ca5c",
    "9b3e29cef6ffae9f",
    "4caee5075de23aa2",
    "cdbd8e20af36d565",
    "f1f705f29117ec03",
    "eaef5129556956fc",
    "ed972adaff1bee0d",
]


def _code_cell_hashes():
    import hashlib

    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip("notebook not available in this CI environment")
    nb = json.loads(nb_path.read_text())
    hashes = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        normalized = "\n".join(ln.rstrip() for ln in src.strip().splitlines())
        hashes.append(hashlib.sha256(normalized.encode()).hexdigest()[:16])
    return hashes


class TestRenderedSurface:
    def test_section1_synthetic_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "static TWFE estimate: 2.001",
                "LWDiD overall ATT: 2.953",
                "later_vs_earlier",
                "2.870",  # true overall quoted in prose and outputs
            ],
        )

    def test_section1_bacon_decomposition_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "treated_vs_never     weight  66.3%   avg 2x2 effect    2.794",
                "later_vs_earlier     weight  21.0%   avg 2x2 effect   -0.865",
                "cohort g=4: 4.019",
                "cohort g=8: 1.845",
            ],
            surface="output",
        )

    def test_section3_detrend_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "demean  : ATT 3.376 (SE 0.099)",
                "detrend : ATT 1.098 (SE 0.202)",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "**3.38**" in md and "**1.10**" in md

    def test_prop99_trajectory_and_diagnostics_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "tau(1989, r=0): -0.0423 (SE 0.0593)",
                "tau(2000, r=11): -0.4029 (SE 0.1525)",
                "units: 39/39 valid (0 dropped)",
                "pre-trend slopes: mean -0.0038, sd 0.0082",
            ],
            surface="output",
        )

    def test_walmart_overall_and_watt_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "overall ATT: 0.0199 (SE 0.0090)",
                "WATT(0):  0.0072",
                "WATT(5):  0.0164",
                "1277 counties x 23 years; 14 cohorts (1986-1999); 391 never-treated",
            ],
            surface="output",
        )

    def test_section2_exact_inference_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "ATT: 2.315 (SE 0.212)",
                "df = 10",
                "[1.843, 2.786]",
            ],
            surface="output",
        )

    def test_prop99_table3_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "ATT -0.4222 (SE 0.1208)",
                "ATT -0.2270 (SE 0.0941)",
                "exact p = 0.0209 (df = 37)",
                "randomization inference p-value: 0.0540",
                "lwdid_ssc_ancillary",
            ],
            surface="output",
        )

    def test_walmart_quotes_and_lead_reconciliation(self):
        # The near-lead and full-lead maxima must BOTH be rendered, and the
        # prose must quote the same two values the output computes - the
        # exact contradiction class the tutorial review flagged.
        assert_quotes_in_rendered(
            NB,
            [
                "WATT(1):  0.0322 (SE 0.0051)",
                "near leads (r = -7..-3):  max |WATT(r)| = 0.0059",
                "all leads (r = -22..-3):  max |WATT(r)| = 0.0329 (at r = -22)",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "below **0.006**" in md and "**0.033** at $r = -22$" in md
        # support-based (not extrapolation-based) far-lead explanation
        assert "only the 1999 cohort" in md

    def test_wild_bootstrap_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "hc1, clustering ignored: ATT 1.048 (SE 0.131)",
                "ATT 1.048 (CR1 SE 0.237), G = 12",
                "p = 0.0025, 95% CI [0.503, 1.614]",
            ],
            surface="output",
        )

    def test_estimand_labels_are_control_pool_accurate(self):
        """The fits use the default not-yet-treated pool, so the rendered
        surface must not label their ``.att`` as the never-treated
        eq. 7.18 tau_omega composite (review round 1 P1)."""
        md = notebook_markdown(NB)
        assert 'control_group="not_yet_treated"' in md
        assert "overall ATT (tau_omega)" not in md

    def test_ri_sharp_null_named(self):
        """RI must be presented as a test of Fisher's sharp null (round-4
        review): a generic 'treatment-effect null' reading invites
        misinterpretation as the weak zero-ATT hypothesis."""
        md = notebook_markdown(NB)
        assert "sharp null" in md

    def test_source_cells_match_rederived_dgps(self):
        """Close the stale-output blind spot: the quote pins read committed
        OUTPUTS and the rederivation tests duplicate the DGPs, so a source
        cell edited without re-execution could leave both layers green.
        Pin the load-bearing source fragments (seeds + DGP structure) so a
        source change that diverges from the rederived DGPs fails here."""
        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip("notebook not available in this CI environment")
        nb = json.loads(nb_path.read_text())
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for fragment in [
            "default_rng(42)",  # section 1 DGP seed
            "default_rng(7)",  # section 2 DGP seed
            "default_rng(5)",  # section 6 DGP seed
            "tau = d * (1.0 + 0.4 * (t - g)) * (1.5 if g == 4 else 1.0)",
            "outcome=alpha + 0.5 * t + 2.0 * d + rng.normal(0, 0.6)",
            "region_post_shock[region] * int(t >= 6)",
            "wild_cluster_bootstrap(n_bootstrap=1999, seed=42)",
            "randomization_test(n_reps=9999, seed=42)",
            'assert prop99.attrs["source"] == "lwdid_ssc_ancillary"',
            'assert walmart.attrs["source"] == "lwdid_ssc_ancillary"',
        ]:
            assert fragment in src, f"source fragment missing: {fragment!r}"

    def test_all_code_cells_hash_pinned(self):
        """Complete source/output contract (round-8 review): EVERY code
        cell's normalized source is hash-pinned, so ANY source edit -
        estimator arguments, control pools, fit configs, not just the
        synthetic DGPs - fails here until the pins are re-locked together
        with a fresh execution and updated rederivation constants. The
        committed outputs the quote pins read can therefore never coexist
        with silently changed source."""
        cells = _code_cell_hashes()
        assert cells == ALL_CODE_CELL_HASHES, (
            "notebook code cells changed - re-execute the notebook and "
            "re-lock ALL_CODE_CELL_HASHES plus any affected rederivation "
            f"constants. Got: {cells}"
        )

    def test_hash_guard_detects_mutation(self):
        """Negative control for the guard mechanism: a one-character
        estimator-argument mutation must change the cell hash."""
        import hashlib

        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip("notebook not available in this CI environment")
        nb = json.loads(nb_path.read_text())
        src = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and 'rolling="detrend"' in "".join(c["source"])
        )
        mutated = src.replace('rolling="detrend"', 'rolling="demean"', 1)
        norm = lambda x: "\n".join(ln.rstrip() for ln in x.strip().splitlines())  # noqa: E731
        h = lambda x: hashlib.sha256(norm(x).encode()).hexdigest()[:16]  # noqa: E731
        assert h(mutated) != h(src)

    def test_prose_quotes_load_bearing_values(self):
        """Every rounded value the NARRATIVE quotes must appear in the
        markdown surface - the outputs are pinned separately, so this is
        the guard against prose/output divergence (round-5 review)."""
        assert_quotes_in_rendered(
            NB,
            [
                # section 2 prose
                "**2.32\n(SE 0.21)**",
                "**[1.84, 2.79]**",
                # Prop 99 prose
                "**-0.422 (SE\n0.121)**",
                "**-0.227 (SE 0.094)**",
                "**0.021**",
                "**p = 0.054**",
                # section 6 prose
                "**0.131**",
                "**0.237**",
                "**[0.50, 1.61]**",
            ],
            surface="markdown",
        )

    def test_methodological_wording_pins(self):
        """Semantic pins for the round-7 corrected claims: parallel trends
        fails on treatment/cohort-RELATED slope differences (not any slope
        heterogeneity); the eligible classical tau_omega composite keeps
        exact t while multi-cell IF aggregates are normal-referenced; CR1
        is a t_{G-1} reference, never described as a normal approximation;
        the ETWFE impact-cell equivalence is calendar t=g / event r=0."""
        md = notebook_markdown(NB)
        assert "systematically by treatment status or cohort" in md
        assert "*sensitivity to the trend specification*" in md
        assert "not\nas a test that unit-specific trends exist" in md
        assert "multi-cell influence-function aggregates use a normal reference" in md
        assert "t_{G-1}" in md
        assert "CR1 normal" not in md
        assert "calendar time $t = g$, event\ntime $r = 0$" in md

    def test_paper_titles_match_registry_artifacts(self):
        assert_quotes_in_rendered(
            NB,
            [
                "A Simple Transformation Approach",
                "Cross-Sectional Sample Sizes",
                "ssrn.com/abstract=4516518",
                "ssrn.com/abstract=5325686",
            ],
            surface="markdown",
        )

    def test_notebook_hygiene(self):
        nb_path = Path(__file__).resolve().parents[1] / NB
        if not nb_path.exists():
            pytest.skip("notebook not available in this CI environment")
        nb = json.loads(nb_path.read_text())
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        errors = [
            out
            for cell in nb["cells"]
            if cell["cell_type"] == "code"
            for out in cell.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        assert not errors


@pytest.fixture(scope="module")
def syn():
    rng = np.random.default_rng(42)
    rows = []
    for u in range(80):
        g = [0, 4, 8][u % 3]
        alpha = rng.normal(0, 2)
        for t in range(1, 13):
            d = int(g > 0 and t >= g)
            tau = d * (1.0 + 0.4 * (t - g)) * (1.5 if g == 4 else 1.0)
            rows.append(
                dict(
                    unit=u,
                    period=t,
                    treated=d,
                    first_treat=g,
                    outcome=alpha + 0.3 * t + tau + rng.normal(0, 0.5),
                )
            )
    return pd.DataFrame(rows)


class TestSyntheticRederivation:
    """Rebuild the section 1 panel from the locked seed and re-check the
    quoted estimates against the library - catches numerics drift without
    notebook re-execution. DGP mirrors the notebook cell exactly."""

    def test_twfe_and_lwdid_estimates(self, syn):
        bacon = BaconDecomposition().fit(
            syn, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        np.testing.assert_allclose(bacon.twfe_estimate, 2.001, atol=5e-4)
        lw = LWDiD(rolling="demean").fit(
            syn,
            outcome="outcome",
            unit="unit",
            time="period",
            treatment="treated",
            first_treat="first_treat",
        )
        np.testing.assert_allclose(lw.att, 2.953, atol=5e-4)
        np.testing.assert_allclose(lw.se, 0.060, atol=5e-4)
        cohorts = {int(g): eff["att"] for g, eff in lw.cohort_effects.items()}
        np.testing.assert_allclose(cohorts[4], 4.019, atol=5e-4)
        np.testing.assert_allclose(cohorts[8], 1.845, atol=5e-4)

    def test_small_panel_exact_inference(self):
        """Re-derive the section 2 exact-t numbers (quoted ATT/SE/df/CI)."""
        rng = np.random.default_rng(7)
        rows = []
        for u in range(12):
            treated_u = int(u < 4)
            alpha = rng.normal(0, 3)
            for t in range(1, 9):
                d = treated_u * int(t >= 6)
                rows.append(
                    dict(
                        unit=u,
                        period=t,
                        treated=d,
                        outcome=alpha + 0.5 * t + 2.0 * d + rng.normal(0, 0.6),
                    )
                )
        small = pd.DataFrame(rows)
        fit = LWDiD(rolling="demean", vcov_type="classical").fit(
            small, outcome="outcome", unit="unit", time="period", treatment="treated"
        )
        np.testing.assert_allclose(fit.att, 2.315, atol=5e-4)
        np.testing.assert_allclose(fit.se, 0.212, atol=5e-4)
        assert fit.df_inference == 10
        np.testing.assert_allclose(fit.conf_int, (1.843, 2.786), atol=5e-4)

    def test_clustered_panel_and_wild_bootstrap(self):
        """Re-derive section 6: the naive-vs-CR1 SE contrast and the WCB
        replay CI (the region-post shock must survive demeaning)."""
        rng = np.random.default_rng(5)
        rows = []
        region_post_shock = rng.normal(0, 0.4, 12)
        for u in range(60):
            region = u % 12
            treated_u = int(region < 5)
            alpha = rng.normal(0, 1.5)
            for t in range(1, 9):
                d = treated_u * int(t >= 6)
                shock = region_post_shock[region] * int(t >= 6)
                rows.append(
                    dict(
                        unit=u,
                        region=region,
                        period=t,
                        treated=d,
                        outcome=alpha + 0.4 * t + 1.2 * d + shock + rng.normal(0, 0.5),
                    )
                )
        clustered = pd.DataFrame(rows)
        naive = LWDiD(rolling="demean").fit(
            clustered, outcome="outcome", unit="unit", time="period", treatment="treated"
        )
        np.testing.assert_allclose(naive.se, 0.131, atol=5e-4)
        lw_cl = LWDiD(rolling="demean", cluster="region").fit(
            clustered, outcome="outcome", unit="unit", time="period", treatment="treated"
        )
        np.testing.assert_allclose(lw_cl.att, 1.048, atol=5e-4)
        np.testing.assert_allclose(lw_cl.se, 0.237, atol=5e-4)
        assert lw_cl.n_clusters == 12
        assert lw_cl.se > 1.5 * naive.se  # the shock survives demeaning
        wcb = lw_cl.wild_cluster_bootstrap(n_bootstrap=1999, seed=42)
        np.testing.assert_allclose(wcb.p_value, 0.0025, atol=1e-4)
        np.testing.assert_allclose((wcb.ci_lower, wcb.ci_upper), (0.503, 1.614), atol=5e-4)

    def test_detrend_panel_rederivation(self):
        """Re-derive the section 3 demean-vs-detrend contrast."""
        rng = np.random.default_rng(11)
        rows = []
        for u in range(40):
            treated_u = int(u < 15)
            alpha = rng.normal(0, 2)
            slope = 0.6 if treated_u else 0.2
            for t in range(1, 13):
                d = treated_u * int(t >= 9)
                rows.append(
                    dict(
                        unit=u,
                        period=t,
                        treated=d,
                        outcome=alpha + slope * t + 1.0 * d + rng.normal(0, 0.5),
                    )
                )
        trendy = pd.DataFrame(rows)
        fits = {
            rolling: LWDiD(rolling=rolling).fit(
                trendy, outcome="outcome", unit="unit", time="period", treatment="treated"
            )
            for rolling in ("demean", "detrend")
        }
        np.testing.assert_allclose(fits["demean"].att, 3.376, atol=5e-4)
        np.testing.assert_allclose(fits["detrend"].att, 1.098, atol=5e-4)
