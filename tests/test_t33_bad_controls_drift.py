"""Drift detection for Tutorial 33 (``docs/tutorials/33_bad_controls.ipynb``).

The tutorial narrative quotes locked, seed-specific numbers (the naive TWFE
bias, the Approach-1 fits with and without ``W``, the bad-control lane and its
``W`` variants, the six ``ATT_X(g, t)`` rows and the event study). ``pytest
--nbmake`` only checks that cells *execute*; it does not check the prose or
the committed outputs (``nbsphinx_execute = "never"`` renders the committed
outputs verbatim). Three layers here:

1. ``assert_quotes_in_rendered`` pins the load-bearing quoted values and the
   pre/post ``ATT_X`` reading against the committed rendered surface.
2. Full re-derivation: the DGP is rebuilt from the locked seed and every
   quoted estimate re-checked at ``atol=5e-4``; the pre-period placebo rows of
   the correctly specified fits are guarded at ``|t| < 2`` (the narrative
   rests on them being unremarkable).
3. ``ALL_CODE_CELL_HASHES`` pins every code cell's normalized source, and
   source-fragment pins keep the duplicated DGP below in sync with the
   notebook cell it mirrors.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, DMLDiD, LinearRegression
from diff_diff.utils import within_transform

from ._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/33_bad_controls.ipynb"

# sha256[:16] of EVERY code cell's normalized source, in notebook order -
# the complete stale-output contract (see test_all_code_cells_hash_pinned)
ALL_CODE_CELL_HASHES = [
    "9e484fe6603ca81c",
    "65244e784d2b85f8",
    "b44219f7b25a0eae",
    "dae3beed78d843ae",
    "9b2c8d153cfa8488",
    "a6d67e8cfd91946a",
    "0c8cf0d2120a3941",
    "2c6701475db185fe",
    "9ee221cd97948c35",
]

FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")
CG = dict(control_group="not_yet_treated")
ATOL = 5e-4

# Locked numbers (seed 5). Every value is quoted in the notebook prose or
# rendered in its outputs.
TWFE_INCLUDE = (0.4886, 0.0142)
CS_A1 = (1.0540, 0.0235)
CS_A1_W = (0.9754, 0.0274)
LANE_W = (0.9766, 0.0260)
LANE_Y = (1.0169, 0.0259)
LANE_NOW = (1.0591, 0.0235)
LANE_RIDGE = (0.9770, 0.0261)
ATT_X_ROWS = {  # (g, t): (att_x, se_x)
    (3, 2): (0.0019, 0.0166),
    (3, 3): (0.5137, 0.0156),
    (3, 4): (0.5055, 0.0299),
    (4, 2): (0.0120, 0.0169),
    (4, 3): (0.0061, 0.0190),
    (4, 4): (0.4764, 0.0263),
}
EVENT_STUDY = {
    -2: (0.0128, 0.0281),
    -1: (-0.0044, 0.0208),
    0: (0.9837, 0.0226),
    1: (0.9629, 0.0412),
}
COHORTS = {0: 974, 3: 531, 4: 495}

# Fragments of the notebook's DGP cell that the mirror below reproduces.
DGP_SOURCE_FRAGMENTS = [
    "rng = np.random.default_rng(5)",
    "W = 0.8 * eta + 0.3 * Z + 0.2 * rng.standard_normal(n)",
    "D = (0.2 * Z + 0.4 * W + 0.3 * eta + rng.standard_normal(n)) > 0",
    "g = np.where(D, rng.choice([3, 4], size=n), 0)",
    "X0[:, 1] = 0.5 * eta + 0.4 * Z + 0.3 * rng.standard_normal(n)",
    "X0[:, t] = 0.7 * X0[:, t - 1] + 0.3 * Z + 0.2 * W + 0.15 + 0.3 * rng.standard_normal(n)",
    "x_t = X0[:, t] + 0.5 * post",
    "y_t = 0.3 * t + 0.5 * eta + 0.3 * Z + X0[:, t] + 0.3 * rng.standard_normal(n) + post * (0.5 + 0.5)",
]


def _load_nb():
    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip("notebook not available in this CI environment")
    return json.loads(nb_path.read_text())


def _norm(src: str) -> str:
    return "\n".join(ln.rstrip() for ln in src.strip().splitlines())


def _code_cell_hashes():
    hashes = []
    for c in _load_nb()["cells"]:
        if c["cell_type"] != "code":
            continue
        hashes.append(hashlib.sha256(_norm("".join(c["source"])).encode()).hexdigest()[:16])
    return hashes


def make_panel() -> pd.DataFrame:
    """Mirror of the notebook's DGP cell (staggered variant of the paper's DGP 1)."""
    n, T = 2000, 4
    rng = np.random.default_rng(5)
    Z = rng.standard_normal(n)
    eta = rng.standard_normal(n)
    W = 0.8 * eta + 0.3 * Z + 0.2 * rng.standard_normal(n)
    D = (0.2 * Z + 0.4 * W + 0.3 * eta + rng.standard_normal(n)) > 0
    g = np.where(D, rng.choice([3, 4], size=n), 0)
    X0 = np.empty((n, T + 1))
    X0[:, 1] = 0.5 * eta + 0.4 * Z + 0.3 * rng.standard_normal(n)
    for t in range(2, T + 1):
        X0[:, t] = 0.7 * X0[:, t - 1] + 0.3 * Z + 0.2 * W + 0.15 + 0.3 * rng.standard_normal(n)
    frames = []
    for t in range(1, T + 1):
        post = (g > 0) & (t >= g)
        x_t = X0[:, t] + 0.5 * post
        y_t = (
            0.3 * t
            + 0.5 * eta
            + 0.3 * Z
            + X0[:, t]
            + 0.3 * rng.standard_normal(n)
            + post * (0.5 + 0.5)
        )
        frames.append(
            pd.DataFrame(
                {
                    "unit": np.arange(n),
                    "time": t,
                    "first_treat": g,
                    "y": y_t,
                    "x": x_t,
                    "z": Z,
                    "w": W,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["post"] = ((df["first_treat"] > 0) & (df["time"] >= df["first_treat"])).astype(float)
    return df


def _twfe(df, cols):
    d = within_transform(df, variables=["y", "post", *cols], unit="unit", time="time")
    X = d[[f"{c}_demeaned" for c in ["post", *cols]]].to_numpy()
    y = d["y_demeaned"].to_numpy()
    lr = LinearRegression(include_intercept=False, cluster_ids=df["unit"].to_numpy()).fit(
        X, y, cluster_k_adjustment=4
    )
    return float(lr.coefficients_[0]), float(np.sqrt(lr.vcov_[0, 0]))


def _lane(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMLDiD(seed=0, **CG, **kw.pop("ctor", {})).fit(
            df, **FIT_KW, covariates=["z"], bad_control="x", **kw
        )


@pytest.fixture(scope="module")
def panel_df():
    return make_panel()


@pytest.fixture(scope="module")
def lane_fit(panel_df):
    return _lane(panel_df, bad_control_covariates=["w"])


class TestRenderedSurface:
    def test_comparison_table_quotes(self):
        assert_quotes_in_rendered(
            NB,
            ["0.4886", "1.0540", "0.9754", "0.9766", "1.0169", "1.0591", "0.9770"],
            surface="output",
        )
        md = notebook_markdown(NB)
        for quoted in (
            "**0.4886 ± 0.0142**",
            "**1.0540 ± 0.0235**",
            "**0.9754 ± 0.0274**",
            "**0.9766 ± 0.0260**",
            "**1.0169 ± 0.0259**",
            "**1.0591 ± 0.0235**",
            "**0.9770 ± 0.0261**",
            "**974**",
            "**531**",
            "**495**",
        ):
            assert quoted in md, f"prose quote missing: {quoted}"

    def test_att_x_and_event_study_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "0.0019",
                "0.0120",
                "0.0061",
                "0.5137",
                "0.5055",
                "0.4764",
                "0.9837",
                "0.9629",
                "-0.0044",
                "0.0128",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        for quoted in (
            "**0.0019**",
            "**0.0120**",
            "**0.0061**",
            "**0.5137**",
            "**0.5055**",
            "**0.4764**",
        ):
            assert quoted in md, f"prose quote missing: {quoted}"

    def test_pre_post_att_x_reading(self):
        """The Remark 6 reading (PR-B CI-review correction): pre rows pre-test
        the identifying assumptions and should be zero; post rows check that
        treatment moves the covariate; a zero does not prove the converse."""
        md = " ".join(notebook_markdown(NB).split())  # whitespace-normalized prose
        assert "They should be zero" in md
        assert "possible violation" in md
        assert "does not establish the converse" in md
        assert "*not* evidence that the covariate is a bad control" in md

    def test_restrictions_and_warnings_wording(self):
        md = notebook_markdown(NB)
        assert "must not appear in `covariates`" in md
        assert_quotes_in_rendered(
            NB,
            [
                "were clipped",
                "will be trimmed",
                "nested_stage: {'split_half'}",
                "warnings raised by the fit",
            ],
            surface="output",
        )
        assert "no exclude arm" in md or 'does not run a "drop X" arm' in md
        assert "seed" in md.lower()
        # The overlap caveat (MP-6 has no uniform lower bound under Gaussian
        # tails) must stay next to the "hold by construction" claim.
        flat = " ".join(md.split())
        assert "does *not* hold uniformly" in flat
        # MP-6 is the never-treated side (propensity bounded away from ONE), and
        # trimming is regularization, not identification (round-6 review).
        assert "bounded away from zero" in flat and "bounded away from one" in flat
        assert "not a repair of MP-6" in flat
        assert "no fixed positive lower bound" not in flat
        # Approach 1 is not immune to future-informed columns.
        assert "impossible by construction" not in flat
        assert "user's responsibility" in flat
        # learner-sensitivity claim scoped to this refit and draw
        assert "insensitive to this linear-to-ridge refit on this draw" in flat
        assert "does not depend on the outcome learner" not in flat
        assert "every identifying assumption holds" not in flat.lower()

    def test_identification_caveats_present(self):
        """The two identification caveats the CI review asked for: MP-7 is named
        for both staggered routes (endpoint-only history), and the W=[y] fit is
        labelled misspecified with its closeness to the truth disclaimed."""
        flat = " ".join(notebook_markdown(NB).split())
        # Approach 1's assumption map (Theorem 1 / Proposition 1 / Proposition 3):
        # the maintained conditional parallel trends is Assumption 2 / MP-4;
        # Assumptions 4 / 5 (MP-8 / MP-9) are unconfoundedness / redundancy.
        assert "Assumption 2 / MP-4" in flat
        assert "Assumption 4 / MP-8" in flat and "Assumption 5 / MP-9" in flat
        assert "MP-1 to MP-4, MP-6, MP-7 and MP-8 or MP-9" in flat
        assert "Assumptions 1-3" in flat
        assert "expanded Approach-1 covariate set" in flat
        # the phrase survives only inside the disclaimer that it is NOT a label
        assert flat.count("parallel trends given $X_{g-1}$") == 1
        assert 'is a "parallel trends given $X_{g-1}$" assumption' in flat
        assert flat.count("MP-7") >= 3
        assert "only through" in flat and "two endpoints" in flat
        assert 'MP-5 does **not** hold with `W=["y"]` here' in flat
        assert "not an identification diagnostic" in flat
        assert "deliberately misspecified sensitivity fit" in flat
        assert "practical default" not in flat
        # Both approaches permit treatment to affect X; Approach 2 is distinguished
        # by W-conditional path unconfoundedness + the nested score (round-5 review).
        assert "Both approaches allow treatment to" in flat
        assert "omitted-confounder demonstration" not in flat
        assert "out of scope here, not harmless" in flat

    def test_source_cells_match_rederived_dgp(self):
        nb = _load_nb()
        dgp_cells = [
            c
            for c in nb["cells"]
            if c["cell_type"] == "code" and "rng = np.random.default_rng(5)" in "".join(c["source"])
        ]
        assert len(dgp_cells) == 1
        dgp_src = _norm("".join(dgp_cells[0]["source"]))
        for frag in DGP_SOURCE_FRAGMENTS:
            assert frag in dgp_src, f"DGP cell drifted from the test mirror: {frag}"

    def test_all_code_cells_hash_pinned(self):
        cells = _code_cell_hashes()
        assert cells == ALL_CODE_CELL_HASHES, (
            "notebook code cells changed - re-execute the notebook and re-lock "
            f"ALL_CODE_CELL_HASHES plus any affected rederivation constants. Got: {cells}"
        )

    def test_hash_guard_detects_mutation(self):
        nb = _load_nb()
        src = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and 'bad_control_covariates=["w"]' in "".join(c["source"])
        )
        mutated = src.replace('bad_control_covariates=["w"]', 'bad_control_covariates=["y"]', 1)
        h = lambda x: hashlib.sha256(_norm(x).encode()).hexdigest()[:16]  # noqa: E731
        assert h(src) != h(mutated)
        assert h(src) in ALL_CODE_CELL_HASHES

    def test_practitioner_banner_names_the_lane(self):
        assert_quotes_in_rendered(
            NB, ["Practitioner Guidance — DMLDiD (CCPS 2026 bad-control score)"], surface="output"
        )
        from ._tutorial_drift import notebook_output_text

        assert "DMLDiD (Chang 2020 double/debiased ML)" not in notebook_output_text(NB)

    def test_paper_reference_present(self):
        md = notebook_markdown(NB)
        assert "arXiv:2608.03881" in md
        assert "Caetano" in md and "Sant'Anna" in md

    def test_notebook_hygiene(self):
        nb = _load_nb()
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        errors = [
            out
            for cell in nb["cells"]
            if cell["cell_type"] == "code"
            for out in cell.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        assert not errors


class TestRederivation:
    def test_cohort_split(self, panel_df):
        counts = panel_df.groupby("unit")["first_treat"].first().value_counts().to_dict()
        assert {int(k): int(v) for k, v in counts.items()} == COHORTS

    def test_twfe_include_bias(self, panel_df):
        att, se = _twfe(panel_df, ["x"])
        np.testing.assert_allclose([att, se], TWFE_INCLUDE, atol=ATOL)
        assert att < 0.6  # the paper's -0.5 bias

    def test_approach_one(self, panel_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a1 = CallawaySantAnna(**CG).fit(panel_df, **FIT_KW, covariates=["x", "z"])
            a1w = CallawaySantAnna(**CG).fit(panel_df, **FIT_KW, covariates=["x", "z", "w"])
        np.testing.assert_allclose([a1.att, a1.se], CS_A1, atol=ATOL)
        np.testing.assert_allclose([a1w.att, a1w.se], CS_A1_W, atol=ATOL)
        assert abs(a1.att - 1.0) > 2 * a1.se  # biased without W
        assert abs(a1w.att - 1.0) < 2 * a1w.se
        # narrative guard: the correctly specified lane's pre rows are unremarkable
        for (g, t), cell in a1w.group_time_effects.items():
            if t < g and cell.get("skip_reason") is None and np.isfinite(cell["se"]):
                assert abs(cell["effect"] / cell["se"]) < 2, (g, t, cell)

    def test_bad_control_lane(self, panel_df, lane_fit):
        res = lane_fit
        np.testing.assert_allclose([res.att, res.se], LANE_W, atol=ATOL)
        assert res.bad_control == "x" and res.bad_control_covariates == ("w",)
        for (g, t), d in res.cross_fit_diagnostics.items():
            assert d.get("skip_reason") is None
            assert d["n_clipped_omega"] > 0
            assert d["nested_stage"] == "in_sample"
        for (g, t), cell in res.group_time_effects.items():
            if t < g:
                assert abs(cell["effect"] / cell["se"]) < 2, (g, t, cell)

    def test_w_variants(self, panel_df):
        res_y = _lane(panel_df, bad_control_covariates=["y"])
        res_now = _lane(panel_df)
        np.testing.assert_allclose([res_y.att, res_y.se], LANE_Y, atol=ATOL)
        np.testing.assert_allclose([res_now.att, res_now.se], LANE_NOW, atol=ATOL)
        assert abs(res_now.att - 1.0) > 2 * res_now.se  # no-W mirrors Approach 1 without W

    def test_ridge_refit_split_half(self, panel_df):
        res_ridge = _lane(
            panel_df, bad_control_covariates=["w"], ctor=dict(outcome_learner="ridge")
        )
        np.testing.assert_allclose([res_ridge.att, res_ridge.se], LANE_RIDGE, atol=ATOL)
        assert {d["nested_stage"] for d in res_ridge.cross_fit_diagnostics.values()} == {
            "split_half"
        }

    def test_att_x_rows(self, lane_fit):
        tab = lane_fit.bad_control_summary()
        assert len(tab) == 6
        assert {(int(r.group), int(r.time)) for r in tab.itertuples()} == set(ATT_X_ROWS)
        for _, row in tab.iterrows():
            key = (int(row["group"]), int(row["time"]))
            np.testing.assert_allclose([row["att_x"], row["se_x"]], ATT_X_ROWS[key], atol=ATOL)
            if row["post"]:
                assert abs(row["att_x"] - 0.5) < 3 * row["se_x"]
            else:
                assert abs(row["att_x"] / row["se_x"]) < 2  # narrative guard

    def test_event_study_rows(self, lane_fit):
        es = lane_fit.aggregate("event_study").to_dataframe()
        assert set(es["event_time"].astype(int)) == set(EVENT_STUDY)
        for _, row in es.iterrows():
            e = int(row["event_time"])
            np.testing.assert_allclose([row["att"], row["se"]], EVENT_STUDY[e], atol=ATOL)
            if e < 0:
                assert abs(row["att"] / row["se"]) < 2  # narrative guard
