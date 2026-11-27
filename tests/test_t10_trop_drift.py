"""Drift detection for Tutorial 10 (``docs/tutorials/10_trop.ipynb``).

The tutorial fits ``TROP`` on a factor-model panel with a true effect of 2.0
and compares it with DiD and Synthetic DiD. Its claims:

1. **Truth check.** Cross-validated TROP gives 2.07, 95% CI [1.52, 2.62],
   which contains 2.0, with no convergence warning at ``max_iter=500``,
   ``tol=1e-4`` (the default ``max_iter=100`` does warn on this data).
2. **On one dataset nothing separates the estimators**: DiD 2.19, Synthetic
   DiD 2.03 and TROP 2.07 all cover the truth.
3. **Across 12 panels DiD is unreliable and TROP is not better than Synthetic
   DiD**: RMSE 0.89 against 0.33 and 0.33. The notebook does not claim TROP
   wins.
4. **The factor penalty controls the rank**: about two factors at 0.1, about
   one at 1.0, none at 10.

TROP is slow (tens of seconds per cross-validated fit), so every TROP
re-derivation is ``@pytest.mark.slow``; the cheap DiD and Synthetic DiD numbers
and the rendered surface run unmarked. Bootstrap standard errors are
backend-specific and are pinned on the rendered surface only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from diff_diff import (
    HAS_RUST_BACKEND,
    TROP,
    DifferenceInDifferences,
    SyntheticDiD,
    generate_factor_data,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/10_trop.ipynb"

TRUE_ATT = 2.0
N_PRE, N_POST = 6, 3
DESIGN = dict(
    n_units=30,
    n_pre=N_PRE,
    n_post=N_POST,
    n_treated=6,
    n_factors=2,
    treatment_effect=TRUE_ATT,
    factor_strength=1.5,
    treated_loading_shift=0.5,
    unit_fe_sd=1.0,
    noise_sd=0.5,
)
FIT = dict(outcome="outcome", treatment="treated", unit="unit", time="period")
SOLVER = dict(max_iter=500, tol=1e-4)
GRIDS = dict(
    lambda_time_grid=[0.0, 0.3, 1.0],
    lambda_unit_grid=[0.0, 0.3, 1.0],
    lambda_nn_grid=[0.1, 1.0],
)


def _panel(seed):
    panel = generate_factor_data(seed=seed, **DESIGN)
    panel["post"] = (panel["period"] >= N_PRE).astype(int)
    return panel


def _did(panel):
    return DifferenceInDifferences(cluster="unit").fit(
        panel, outcome="outcome", treatment="treat", post="post"
    )


def _sdid(panel):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pre-treatment fit is poor", category=UserWarning)
        return SyntheticDiD(variance_method="jackknife", seed=42).fit(
            panel,
            outcome="outcome",
            treatment="treat",
            unit="unit",
            time="period",
            post_periods=list(range(N_PRE, N_PRE + N_POST)),
        )


@pytest.fixture(scope="module")
def data():
    return _panel(42)


class TestFastEstimators:
    def test_design(self, data):
        assert data["unit"].nunique() == 30 and data["period"].nunique() == 9
        assert data.groupby("unit")["treat"].max().sum() == 6
        np.testing.assert_allclose(data.loc[data["treated"] == 1, "true_effect"], TRUE_ATT)

    def test_did_and_sdid_on_the_tutorial_panel(self, data):
        did = _did(data)
        sdid = _sdid(data)
        assert did.att == pytest.approx(2.191, abs=2e-3)
        assert did.conf_int[0] == pytest.approx(1.652, abs=5e-3)
        assert did.conf_int[1] == pytest.approx(2.731, abs=5e-3)
        assert sdid.att == pytest.approx(2.034, abs=5e-3)
        for fit in (did, sdid):
            assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]

    def test_did_is_unreliable_across_panels(self):
        did = np.array([_did(_panel(seed)).att for seed in range(12)])
        sdid = np.array([_sdid(_panel(seed)).att for seed in range(12)])
        did_rmse = np.sqrt(((did - TRUE_ATT) ** 2).mean())
        sdid_rmse = np.sqrt(((sdid - TRUE_ATT) ** 2).mean())
        assert did_rmse == pytest.approx(0.891, abs=0.01)
        assert sdid_rmse == pytest.approx(0.325, abs=0.02)
        assert did_rmse > 2.5 * sdid_rmse


@pytest.mark.slow
class TestTROP:
    def test_cross_validated_fit_covers_truth_without_warning(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if not HAS_RUST_BACKEND:
                # The pure-Python cross-validation reports a candidate penalty
                # (not the selected one) whose leave-one-out fits hit the
                # iteration cap; the Rust search the notebook is committed from
                # does not report it. The selected penalties and the estimate
                # are the same on both backends, and the final fit must still
                # be warning-free.
                warnings.filterwarnings("ignore", message="TROP local LOOCV", category=UserWarning)
            results = TROP(n_bootstrap=200, seed=42, **GRIDS, **SOLVER).fit(data, **FIT)
        assert results.att == pytest.approx(2.072, abs=5e-3)
        assert (results.lambda_time, results.lambda_unit, results.lambda_nn) == (0.3, 1.0, 1.0)
        assert results.effective_rank == pytest.approx(1.18, abs=0.03)
        assert results.conf_int[0] <= TRUE_ATT <= results.conf_int[1]
        assert results.factor_matrix.shape == (9, 30)

    def test_default_iteration_cap_warns_on_this_data(self, data):
        with pytest.warns(UserWarning, match="did not converge"):
            TROP(
                lambda_time_grid=[1.0],
                lambda_unit_grid=[1.0],
                lambda_nn_grid=[0.1],
                n_bootstrap=2,
                seed=42,
            ).fit(data, **FIT)

    def test_factor_penalty_controls_the_rank(self, data):
        ranks, atts = [], []
        for lambda_nn in (0.1, 1.0, 10.0):
            fit = TROP(
                lambda_time_grid=[0.3],
                lambda_unit_grid=[1.0],
                lambda_nn_grid=[lambda_nn],
                n_bootstrap=2,
                seed=42,
                **SOLVER,
            ).fit(data, **FIT)
            ranks.append(fit.effective_rank)
            atts.append(fit.att)
        np.testing.assert_allclose(ranks, [2.13, 1.18, 0.0], atol=0.05)
        np.testing.assert_allclose(atts, [2.040, 2.072, 2.047], atol=5e-3)

    def test_trop_is_not_better_than_sdid_across_panels(self):
        trop = np.array(
            [
                TROP(n_bootstrap=2, seed=42, **GRIDS, **SOLVER).fit(_panel(seed), **FIT).att
                for seed in range(12)
            ]
        )
        sdid = np.array([_sdid(_panel(seed)).att for seed in range(12)])
        did = np.array([_did(_panel(seed)).att for seed in range(12)])
        rmse = lambda x: float(np.sqrt(((x - TRUE_ATT) ** 2).mean()))  # noqa: E731
        assert rmse(trop) == pytest.approx(0.332, abs=0.03)
        assert rmse(trop) < 0.5 * rmse(did)
        assert abs(rmse(trop) - rmse(sdid)) < 0.08  # indistinguishable, not a win

    def test_global_method(self, data):
        fit = TROP(method="global", n_bootstrap=200, seed=42, **GRIDS, **SOLVER).fit(data, **FIT)
        assert fit.att == pytest.approx(2.036, abs=5e-3)
        assert fit.conf_int[0] <= TRUE_ATT <= fit.conf_int[1]


class TestNotebookSurface:
    def test_rendered_numbers(self):
        assert_quotes_in_rendered(
            NB,
            [
                "Estimate: 2.072  (SE 0.260)",
                "95% CI:   [1.523, 2.620]",
                "Selected lambda_time: 0.3",
                "DiD                2.191   0.264     [1.652, 2.731]        True",
                "DiD                1.623   0.843   0.891",
                "Synthetic DiD      1.948   0.335   0.325",
                "TROP               1.946   0.343   0.332",
                "global     2.036   0.207     [1.599, 2.473]",
            ],
            surface="output",
        )

    def test_no_convergence_warning_in_outputs(self):
        path = Path(__file__).resolve().parents[1] / NB
        if not path.exists():
            pytest.skip("notebook not available in this environment")
        nb = json.loads(path.read_text(encoding="utf-8"))
        stderr = "".join(
            "".join(o.get("text", ""))
            for c in nb["cells"]
            if c["cell_type"] == "code"
            for o in c.get("outputs", [])
            if o.get("name") == "stderr"
        )
        assert "did not converge" not in stderr
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in (
            "TRUE_ATT = 2.0",
            "treatment_effect=TRUE_ATT,",
            "solver = dict(max_iter=500, tol=1e-4)",
            'message="Pre-treatment fit is poor", category=UserWarning',
            "for seed in range(12):",
        ):
            assert needle in code, needle
        assert "time.time()" not in code  # no wall-clock numbers in committed output

    def test_prose_does_not_overclaim(self):
        md = notebook_markdown(NB)
        for needle in (
            "**DiD is unreliable here.**",
            "**TROP and Synthetic DiD have similar observed errors here**",
            "It is not a guaranteed improvement over\nSynthetic DiD.",
            "raise\n  `max_iter` before reading the results",
            "a staggered design is rejected",
            "It does not try every combination",
            "The bias is small if any one of the three is small.",
            # the bound is for a single block and fixed weights; on/off treatment needs more
            "weights that are fixed in advance and not chosen from the data",
            "No anticipation; no spillovers, so that one unit's\ntreatment does not change another unit's outcome",
            "That option adds a requirement of no carryover",
            "For a\n  single treated block and fixed weights",
            # time weights decay around the target period, not around adoption
            "the weights are centred on `t`, not on the adoption date",
            "it does not\n  take a TROP fit",
            "over the periods, other than `t` itself,\n  in which both units are untreated",
        ):
            assert needle in md, needle
        assert md.count("## Summary") == 1
        for stale in (
            "TROP should recover",
            "have been removed",
            "Reduced from",
            "the periods just before treatment",
        ):
            assert stale not in md, stale
