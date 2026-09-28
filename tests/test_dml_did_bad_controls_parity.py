"""Black-box parity of the DMLDiD bad-control lane against R ``badcontrols``.

Caetano, Callaway, Payne & Sant'Anna (2026) ship their estimator as the
GPL-3 R package ``badcontrols`` (1.0.0, GitHub commit ``651ccc92``; no
tags or releases exist). The Python lane is derived from the paper alone
(``docs/methodology/papers/caetano-2026-review.md``); the R package is only
EXECUTED as an oracle by ``benchmarks/R/generate_badcontrols_golden.R`` -
its source is never read (license posture; a direct port was rejected).

Why tolerance-based, not bit-exact: ``didbc()`` cross-fits with its own fold
draw (``nfolds = 1`` errors), so no shared fold assignment exists. The
2026-09-05 spike measured single-seed fold noise of ~0.1-0.25 analytical SE
on most cells and up to ~0.5 SE on the smallest cells (n = 500, W = lagged
outcome); ``att_seed_sd`` in the fixture records it per cell. Each R run is
therefore recorded as the per-cell mean over ``n_seeds`` seeds (per-seed
rows kept) and the Python side averages the same number of its own seeds,
which shrinks the fold noise of the DIFFERENCE of means to ~0.1-0.2 SE.

Assertion rules (fixed up front; a failing cell is a defect to investigate,
never a tolerance to loosen):

- Runs 1-2 (W = [W]; not-yet-treated / never-treated) on every cell, and run
  3 (W = [Y], Remark 5) on POST cells: per cell
  ``|mean_att_py - mean_att_R| < 0.5 * SE_R`` with ``SE_R = sqrt(V_R / n_R)``,
  ``|mean_se_py / SE_R - 1| < 0.3``, and the run's mean absolute gap
  ``< 2 * mean(SE_R) / sqrt(n_cells)``.
- Run 3 PRE cells: only ``|mean_att_py - mean_att_R| < 1.0 * SE_R`` (the
  period at which R evaluates the lagged-outcome W on pre-period cells is
  not established by the black-box spike; the library reads ``Y_{t-1}``).
- Run 4 (no bad control): a CHARACTERIZATION, not a same-estimator
  comparison - R's ``dr_ml`` path still regresses the propensity odds on Z
  for ``omega_0`` whereas DMLDiD uses ``ps/(1-ps)`` pointwise - asserted at
  ``|mean_att_py - mean_att_R| < 1.0 * SE_R`` per cell only.

Fixture: ``benchmarks/data/badcontrols_golden.json`` (+ the shared panel
``benchmarks/data/badcontrols_panel.csv``). Per ``feedback_golden_file_pytest_skip``
a missing fixture skips (CI isolated-install jobs copy ``tests/`` only).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import DMLDiD

DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "data"
FIXTURE_PATH = DATA_DIR / "badcontrols_golden.json"
PANEL_PATH = DATA_DIR / "badcontrols_panel.csv"

BADCONTROLS_VERSION = "1.0.0"
BADCONTROLS_SHA = "651ccc925776125bb9233d76867c862107ea0ba5"
PTETOOLS_VERSION = "1.0.0"

POINT_TOL_SE = 0.5
POINT_TOL_SE_CHARACTERIZATION = 1.0
SE_RATIO_TOL = 0.3

RUN_SPECS = {
    "w_nyt": dict(control_group="not_yet_treated", bad_control="X", bad_control_covariates=["W"]),
    "w_nt": dict(control_group="never_treated", bad_control="X", bad_control_covariates=["W"]),
    "ylag_nyt": dict(
        control_group="not_yet_treated", bad_control="X", bad_control_covariates=["Y"]
    ),
    "plain_nyt": dict(control_group="not_yet_treated"),
}


def _load_fixture():
    if not FIXTURE_PATH.exists() or not PANEL_PATH.exists():
        pytest.skip(
            f"Golden fixture {FIXTURE_PATH} / {PANEL_PATH} missing - regenerate via "
            "`Rscript benchmarks/R/generate_badcontrols_golden.R`."
        )
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fixture():
    return _load_fixture()


@pytest.fixture(scope="module")
def panel(fixture):
    return pd.read_csv(PANEL_PATH)


def _python_seed_mean(panel, spec, n_seeds):
    """Per-cell seed-mean of (att, se) over ``n_seeds`` Python fold draws."""
    cg = spec["control_group"]
    kw = {k: v for k, v in spec.items() if k != "control_group"}
    acc: dict = {}
    for seed in range(n_seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(control_group=cg, base_period="varying", pscore_trim=1e-6, seed=seed).fit(
                panel,
                outcome="Y",
                unit="id",
                time="period",
                first_treat="G",
                covariates=["Z"],
                **kw,
            )
        for (g, t), e in res.group_time_effects.items():
            assert e.get("skip_reason") is None, (g, t, e)
            acc.setdefault((g, t), []).append((e["effect"], e["se"]))
    return {
        k: (float(np.mean([a for a, _ in v])), float(np.mean([s for _, s in v])))
        for k, v in acc.items()
    }


def _r_rows(fixture, key):
    rows = fixture["runs"][key]["att_gt_mean"]
    return {(int(r["group"]), int(r["t"])): r for r in rows}


def _compare(py, r_rows, *, cells, point_tol_se, se_ratio_tol=None, mean_gap=False):
    gaps = []
    se_rs = []
    for cell in cells:
        r = r_rows[cell]
        att_r = r["att"]
        se_r = float(np.sqrt(r["V_analytical"] / r["n"]))
        att_py, se_py = py[cell]
        assert abs(att_py - att_r) < point_tol_se * se_r, (
            f"cell {cell}: |{att_py:.5f} - {att_r:.5f}| = {abs(att_py - att_r):.5f} "
            f">= {point_tol_se} * SE_R ({se_r:.5f}); R seed sd {r['att_seed_sd']:.5f}"
        )
        if se_ratio_tol is not None:
            assert (
                abs(se_py / se_r - 1.0) < se_ratio_tol
            ), f"cell {cell}: se_py/SE_R = {se_py / se_r:.4f}"
        gaps.append(abs(att_py - att_r))
        se_rs.append(se_r)
    if mean_gap:
        bound = 2.0 * float(np.mean(se_rs)) / np.sqrt(len(cells))
        assert float(np.mean(gaps)) < bound, (float(np.mean(gaps)), bound)


class TestBadControlsParity:
    def test_metadata_versions_match(self, fixture):
        meta = fixture["meta"]
        assert meta["badcontrols_version"] == BADCONTROLS_VERSION
        assert meta["badcontrols_remote_sha"] == BADCONTROLS_SHA
        assert meta["ptetools_version"] == PTETOOLS_VERSION
        assert meta["common_args"]["overlap_threshold"] == 1
        assert meta["common_args"]["nuisance_method"] == "parametric"
        assert int(meta["n_seeds"]) >= 5
        for key in RUN_SPECS:
            assert len(fixture["runs"][key]["seeds"]) == int(meta["n_seeds"])

    def test_panel_matches_metadata(self, fixture, panel):
        meta = fixture["meta"]
        assert panel["id"].nunique() == meta["n"]
        assert sorted(panel["period"].unique()) == list(range(1, meta["T_max"] + 1))
        assert set(panel.columns) >= {"id", "period", "G", "Y", "X", "Z", "W"}

    @pytest.mark.parametrize("key", ["w_nyt", "w_nt"])
    def test_bad_control_runs_all_cells(self, fixture, panel, key):
        r_rows = _r_rows(fixture, key)
        py = _python_seed_mean(panel, RUN_SPECS[key], int(fixture["meta"]["n_seeds"]))
        assert set(py) == set(r_rows)
        _compare(
            py,
            r_rows,
            cells=sorted(r_rows),
            point_tol_se=POINT_TOL_SE,
            se_ratio_tol=SE_RATIO_TOL,
            mean_gap=True,
        )

    def test_lagged_outcome_w_post_cells(self, fixture, panel):
        r_rows = _r_rows(fixture, "ylag_nyt")
        py = _python_seed_mean(panel, RUN_SPECS["ylag_nyt"], int(fixture["meta"]["n_seeds"]))
        post = sorted(c for c in r_rows if c[1] >= c[0])
        assert post
        _compare(
            py,
            r_rows,
            cells=post,
            point_tol_se=POINT_TOL_SE,
            se_ratio_tol=SE_RATIO_TOL,
            mean_gap=True,
        )

    def test_lagged_outcome_w_pre_cells_characterization(self, fixture, panel):
        r_rows = _r_rows(fixture, "ylag_nyt")
        py = _python_seed_mean(panel, RUN_SPECS["ylag_nyt"], int(fixture["meta"]["n_seeds"]))
        pre = sorted(c for c in r_rows if c[1] < c[0])
        assert pre
        _compare(py, r_rows, cells=pre, point_tol_se=POINT_TOL_SE_CHARACTERIZATION)

    def test_no_bad_control_characterization(self, fixture, panel):
        # NOT a same-estimator comparison (see module docstring).
        r_rows = _r_rows(fixture, "plain_nyt")
        py = _python_seed_mean(panel, RUN_SPECS["plain_nyt"], int(fixture["meta"]["n_seeds"]))
        _compare(py, r_rows, cells=sorted(r_rows), point_tol_se=POINT_TOL_SE_CHARACTERIZATION)

    def test_true_att_recovered_by_post_cells(self, fixture, panel):
        # The simulated truth is the paper's own DGP target; a coarse sanity
        # anchor independent of R (3 SE on the seed-mean).
        truth = {(int(r["g"]), int(r["t"])): r["att"] for r in fixture["true_att_gt"]}
        py = _python_seed_mean(panel, RUN_SPECS["w_nyt"], 3)
        for cell, att_true in truth.items():
            att_py, se_py = py[cell]
            assert abs(att_py - att_true) < 3.0 * se_py, (cell, att_py, att_true, se_py)
