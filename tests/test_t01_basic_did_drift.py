"""Drift detection for Tutorial 01 (``docs/tutorials/01_basic_did.ipynb``).

The tutorial's narrative rests on these claims:

1. **The 2x2 estimate recovers the truth.** On the simulated two-period panel
   (true ATT 5.0) the unit-clustered ``DifferenceInDifferences`` estimate is
   about 5.03 and its 95% interval covers 5.0; the formula interface returns
   the same estimate; two covariates that are independent of the outcome
   barely move it.
2. **Fixed effects change the standard error, not the estimate.** On the
   balanced four-period panel (true ATT 4.0) the plain, state-indicator,
   state-absorbed and two-way fixed effects fits share one point estimate.
3. **Clustering matters, and its direction is design-specific.** On the
   20-cluster data the cluster-robust SE is far *smaller* than HC1 (ratio
   about 0.14), because the persistent cluster component cancels out of the
   DiD estimate but stays in the residuals. Clustered inference uses t(19).
4. **The wild cluster bootstrap p-value shown is its floor**, 1/(999 + 1).

``pytest --nbmake`` only checks that cells execute, and ``nbsphinx_execute =
"never"`` means Read the Docs renders the committed outputs. These tests
re-derive the load-bearing numbers through the same public API the notebook
uses and cross-check the rendered surface, so library drift or a notebook-only
edit fails loudly. Tolerances are loose enough for both backends (outputs are
committed from the Rust backend; CI also runs pure Python).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences, TwoWayFixedEffects, generate_did_data
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/01_basic_did.ipynb"

TRUE_ATT = 5.0
TRUE_ATT_PANEL = 4.0
TRUE_ATT_CLUSTERED = 2.5
FIT = dict(outcome="outcome", treatment="treated", post="post")


def make_two_period():
    return generate_did_data(
        n_units=100,
        n_periods=2,
        treatment_effect=TRUE_ATT,
        treatment_fraction=0.5,
        treatment_period=1,
        noise_sd=1.0,
        seed=42,
    )


def make_panel():
    rng = np.random.default_rng(7)
    rows = []
    for unit in range(50):
        treated = int(unit < 25)
        state = unit % 5
        unit_effect = rng.normal(0, 2)
        for period in range(4):
            post = int(period >= 2)
            outcome = (
                10.0
                + unit_effect
                + 0.5 * period
                + 1.5 * state
                + TRUE_ATT_PANEL * treated * post
                + rng.normal(0, 0.5)
            )
            rows.append(
                {
                    "unit": unit,
                    "state": f"state_{state}",
                    "period": period,
                    "treated": treated,
                    "post": post,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def make_clustered():
    rng = np.random.default_rng(11)
    rows = []
    for cluster in range(20):
        treated = int(cluster < 10)
        cluster_effect = rng.normal(0, 2)
        for member in range(10):
            for period in (0, 1):
                outcome = (
                    10.0
                    + cluster_effect
                    + 3.0 * period
                    + TRUE_ATT_CLUSTERED * treated * period
                    + rng.normal(0, 0.5)
                )
                rows.append(
                    {
                        "cluster": cluster,
                        "member": member,
                        "treated": treated,
                        "post": period,
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    return make_two_period()


@pytest.fixture(scope="module")
def panel_df():
    return make_panel()


@pytest.fixture(scope="module")
def clustered_df():
    return make_clustered()


class TestTwoByTwo:
    def test_estimate_recovers_truth(self, data):
        res = DifferenceInDifferences(cluster="unit").fit(data, **FIT)
        assert res.att == pytest.approx(5.0318, abs=5e-4)
        assert res.se == pytest.approx(0.2821, abs=5e-4)
        lower, upper = res.conf_int
        assert (lower, upper) == pytest.approx((4.4721, 5.5915), abs=5e-4)
        assert lower <= TRUE_ATT <= upper
        assert res.n_clusters == 100
        assert_quotes_in_rendered(
            NB,
            [
                "Estimated ATT: 5.0318",
                "95% CI:        [4.4721, 5.5915]",
                "CI covers the true ATT: True",
                "CR1 cluster-robust at unit, G=100",
            ],
            surface="output",
        )

    def test_formula_interface_matches(self, data):
        by_name = DifferenceInDifferences(cluster="unit").fit(data, **FIT)
        by_formula = DifferenceInDifferences(cluster="unit").fit(
            data, formula="outcome ~ treated * post"
        )
        np.testing.assert_allclose(by_formula.att, by_name.att, rtol=1e-12)
        assert_quotes_in_rendered(NB, ["Same estimate:   True"], surface="output")

    def test_independent_covariates_barely_move_the_estimate(self, data):
        df = data.copy()
        rng = np.random.default_rng(2024)
        unit_size = pd.Series(rng.normal(100, 20, df["unit"].nunique()), index=df["unit"].unique())
        df["size"] = df["unit"].map(unit_size)
        df["age"] = rng.normal(10, 3, len(df))
        base = DifferenceInDifferences(cluster="unit").fit(df, **FIT)
        cov = DifferenceInDifferences(cluster="unit").fit(df, covariates=["size", "age"], **FIT)
        assert cov.att == pytest.approx(5.0143, abs=5e-4)
        assert cov.se == pytest.approx(0.2798, abs=5e-4)
        # "barely move": well inside one standard error, SE within a few percent
        assert abs(cov.att - base.att) < 0.25 * base.se
        assert abs(cov.se / base.se - 1.0) < 0.05
        assert_quotes_in_rendered(
            NB, ["With covariates:    ATT = 5.0143 (SE 0.2798)"], surface="output"
        )


class TestFixedEffects:
    def test_specifications_share_the_point_estimate(self, panel_df):
        plain = DifferenceInDifferences(cluster="unit").fit(panel_df, **FIT)
        dummies = DifferenceInDifferences(cluster="unit").fit(
            panel_df, fixed_effects=["state"], **FIT
        )
        absorbed = DifferenceInDifferences(cluster="unit").fit(panel_df, absorb=["state"], **FIT)
        twfe = TwoWayFixedEffects().fit(panel_df, unit="unit", **FIT)

        for res in (plain, dummies, absorbed, twfe):
            assert res.att == pytest.approx(4.0888, abs=5e-4)
            np.testing.assert_allclose(res.att, plain.att, rtol=1e-9)
        # indicator variables and the within transformation are the same model
        np.testing.assert_allclose(absorbed.se, dummies.se, rtol=1e-8)
        assert [round(r.se, 4) for r in (plain, dummies, absorbed, twfe)] == pytest.approx(
            [0.1351, 0.1365, 0.1365, 0.1348], abs=5e-4
        )
        lower, upper = twfe.conf_int
        assert lower <= TRUE_ATT_PANEL <= upper
        assert_quotes_in_rendered(
            NB,
            [
                "fixed_effects=['state']: ATT = 4.0888 (SE 0.1365)",
                "absorb=['state']:        ATT = 4.0888 (SE 0.1365)",
                "two-way fixed effects  4.0888  0.1348",
                "TWFE 95% CI [3.818, 4.360] covers the true ATT: True",
            ],
            surface="output",
        )

    def test_twfe_clusters_at_unit_by_default(self, panel_df):
        twfe = TwoWayFixedEffects().fit(panel_df, unit="unit", **FIT)
        assert twfe.cluster_name == "unit"
        assert twfe.n_clusters == 50
        assert_quotes_in_rendered(NB, ["CR1 cluster-robust at unit, G=50"], surface="output")


class TestClustering:
    def test_cluster_se_is_smaller_than_hc1_here(self, clustered_df):
        hc1 = DifferenceInDifferences().fit(clustered_df, **FIT)
        clustered = DifferenceInDifferences(cluster="cluster").fit(clustered_df, **FIT)
        np.testing.assert_allclose(clustered.att, hc1.att, rtol=1e-12)
        assert clustered.att == pytest.approx(2.4537, abs=5e-4)
        assert hc1.se == pytest.approx(0.4416, abs=5e-4)
        assert clustered.se == pytest.approx(0.0639, abs=5e-4)
        assert clustered.se / hc1.se == pytest.approx(0.14, abs=0.01)
        # the degrees of freedom the prose quotes
        assert clustered.inference_df == 19
        lower, upper = clustered.conf_int
        assert lower <= TRUE_ATT_CLUSTERED <= upper
        assert_quotes_in_rendered(
            NB,
            [
                "HC1 SE, rows treated as independent: 0.4416",
                "Cluster-robust SE:                   0.0639",
                "Ratio, cluster-robust / HC1:         0.14",
            ],
            surface="output",
        )
        assert_quotes_in_rendered(
            NB,
            [
                "the cluster-robust one is the *smaller*",
                'Do not read this as "clustering shrinks standard errors"',
                "(number of clusters - 1) degrees of freedom, 19 here",
            ],
            surface="markdown",
        )

    def test_wild_bootstrap_p_value_is_the_floor(self, clustered_df):
        analytical = DifferenceInDifferences(cluster="cluster").fit(clustered_df, **FIT)
        boot = DifferenceInDifferences(
            cluster="cluster",
            inference="wild_bootstrap",
            n_bootstrap=999,
            bootstrap_weights="rademacher",
            seed=42,
        ).fit(clustered_df, **FIT)
        # floored at 1 / (n_valid + 1)
        assert boot.p_value == pytest.approx(1.0 / 1000.0, rel=1e-12)
        # the bootstrap replaces p-value and interval, not the standard error
        np.testing.assert_allclose(boot.se, analytical.se, rtol=1e-12)
        lower, upper = boot.conf_int
        assert lower <= TRUE_ATT_CLUSTERED <= upper
        # the interval is a test inversion over sampled draws: pin loosely
        assert (lower, upper) == pytest.approx((2.3132, 2.5885), abs=0.01)
        assert_quotes_in_rendered(
            NB, ["wild cluster bootstrap    0.001    2.3132    2.5885"], surface="output"
        )
        assert_quotes_in_rendered(
            NB,
            ["with 999 draws it cannot be smaller than 0.001", "it does not produce a different"],
            surface="markdown",
        )


class TestNotebookSurface:
    def test_notebook_constants_match(self):
        """The DGP constants duplicated in this file are the notebook's."""
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"]
            if c["cell_type"] == "code"
        )
        for needle in (
            "TRUE_ATT = 5.0",
            "TRUE_ATT_PANEL = 4.0",
            "TRUE_ATT_CLUSTERED = 2.5",
            "seed=42,",
            "np.random.default_rng(2024)",
            "np.random.default_rng(7)",
            "np.random.default_rng(11)",
            "n_bootstrap=999,",
        ):
            assert needle in source, f"notebook code no longer contains {needle!r}"

    def test_states_its_assumptions(self):
        md = notebook_markdown(NB)
        assert "**Parallel trends.**" in md
        assert "**No anticipation.**" in md
        assert "cluster at the level where treatment is\nassigned" in md
