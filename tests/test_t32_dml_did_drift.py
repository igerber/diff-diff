"""Drift detection for Tutorial 32 (``docs/tutorials/32_dml_did.ipynb``).

The tutorial narrative quotes locked, seed-specific numbers (the DGP-implied
truth, the learner-comparison bias table, the HonestDiD robust CI, bootstrap
sup-t bands, and the RCS/survey fits). ``pytest --nbmake`` only checks that
cells *execute*; it does not check the prose or the committed outputs
(``nbsphinx_execute = "never"`` renders the committed outputs verbatim).
Three layers here:

1. ``assert_quotes_in_rendered`` pins the load-bearing quoted values against
   the committed rendered surface (markdown + outputs).
2. Full re-derivation: the panel and RCS DGPs are rebuilt from the locked
   seeds and the section 3/4/6/7 estimates re-checked against the quoted
   values - sections 6-7 exercise the survey/cluster lane, so a library
   numerics change surfaces here even without re-executing the notebook.
3. ``ALL_CODE_CELL_HASHES`` pins every code cell's normalized source, and
   source-fragment pins keep the duplicated DGP/learner definitions below in
   sync with the notebook cells they mirror.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff import DMLDiD, SurveyDesign, compute_honest_did

from ._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/32_dml_did.ipynb"

# sha256[:16] of EVERY code cell's normalized source, in notebook order -
# the complete stale-output contract (see test_all_code_cells_hash_pinned)
ALL_CODE_CELL_HASHES = [
    "9405e25ece47ce8b",
    "68f3f1e50a2a3c60",
    "128c08960b3dfd45",
    "c48f19ab5c5b76ad",
    "d4eed65cbf569b74",
    "8ecbf97a1a817414",
    "98695413c281f252",
    "84d167bbdd9af3c8",
    "9efefccaf9379ba7",
    "c3573fc2030e099a",
    "bbd6bc001d063ff8",
    "ffe1220f7721cb2f",
    "6b746ea09fd9e3da",
    "ed87cf8c4a4cd42f",
    "9ee221cd97948c35",
]

FIT_KW = dict(
    outcome="y", unit="unit", time="time", first_treat="first_treat", covariates=["x1", "x2"]
)


def _load_nb():
    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip("notebook not available in this CI environment")
    return json.loads(nb_path.read_text())


def _code_cell_hashes():
    import hashlib

    hashes = []
    for c in _load_nb()["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        normalized = "\n".join(ln.rstrip() for ln in src.strip().splitlines())
        hashes.append(hashlib.sha256(normalized.encode()).hexdigest()[:16])
    return hashes


class TestRenderedSurface:
    def test_dgp_and_first_fit_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "DGP-implied overall ATT: 2.2388",
                "DMLDiD (sieve) estimate: 2.2804 +/- 0.0438",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "**2.2804 ± 0.0438**" in md and "**2.2388**" in md
        assert "354 never-treated units" in md

    def test_learner_table_quotes(self):
        assert_quotes_in_rendered(
            NB,
            ["2.5909", "2.5898", "2.2804", "2.2818"],
            surface="output",
        )
        md = notebook_markdown(NB)
        for quoted in (
            "**linear 2.5909**",
            "**ridge 2.5898**",
            "**sieve 2.2804**",
            "**PolynomialRidge 2.2818**",
        ):
            assert quoted in md, f"prose quote missing: {quoted}"

    def test_score_family_and_rm_units_wording(self):
        """Review-round pins: Chang's score is a DISTINCT family from the
        Sant'Anna-Zhao DR score (REGISTRY DR-score families note), the
        relative-magnitude restriction is stated in consecutive
        first-difference units, and the downstream HonestDiD/sup-t claims
        carry the no-nominal-coverage qualification."""
        md = notebook_markdown(NB)
        assert "distinct score family" in md
        assert "same doubly-robust moment" not in md
        assert "consecutive first difference" in md
        assert "coverage guarantee" in md
        assert "deliberately violates" in md  # sup-t nominal-coverage caveat

    def test_dr_inference_caveat_present(self):
        """The bias demo deliberately misspecifies the propensity in every
        arm, so the prose MUST carry the Theorem-1 both-nuisances caveat
        (plan-review round 3): SEs/CIs in the table are illustrative."""
        md = notebook_markdown(NB)
        assert "double robustness" in md.lower()
        assert "cannot compensate" in md
        assert "illustrative rather than theory-backed" in md

    def test_diagnostics_quotes(self):
        assert_quotes_in_rendered(NB, ["0.2547", "0.2610"], surface="output")
        md = notebook_markdown(NB)
        assert "fold losses do not verify that" in md

    def test_aggregation_and_honest_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "Identified set:                [2.2291, 2.4779]",
                "95% Robust CI:                 [2.1393, 2.5677]",
                "analytical ATT 2.2804 | bootstrap ATT 2.2804",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "**[2.14, 2.57]**" in md
        assert "**2.04, 2.30, 2.71**" in md
        assert "**2.38**" in md and "**2.14**" in md

    def test_rcs_and_survey_quotes(self):
        assert_quotes_in_rendered(
            NB,
            [
                "RCS ATT: 2.4428 +/- 0.2373",
                "survey ATT: 2.2886 +/- 0.1509",
                "design df:  16 (20 PSUs - 4 strata)",
            ],
            surface="output",
        )
        md = notebook_markdown(NB)
        assert "**2.44 ± 0.24**" in md and "**2.29 ± 0.15**" in md

    def test_seed_quotes(self):
        assert_quotes_in_rendered(
            NB,
            ["seed=1:       2.271637", "seed=2:       2.273391", "seed=1 again: 2.271637"],
            surface="output",
        )

    def test_source_cells_match_rederived_dgps(self):
        """The quote pins read committed OUTPUTS and the rederivation tests
        below duplicate the DGPs and the PolynomialRidge learner, so a
        source cell edited without re-execution could leave both layers
        green. Pin the load-bearing source fragments (seeds + DGP structure
        + learner math) so a source change that diverges from the
        rederivations fails here."""
        nb = _load_nb()
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for fragment in [
            "default_rng(7)",  # panel DGP seed
            "default_rng(11)",  # RCS DGP seed
            "0.9 * (x1**2 - 1.0) + 0.7 * x1 * x2",  # nonlinear assignment score
            "1.6 * (x1**2 - 1.0) + 1.0 * x1 * x2",  # nonlinear trend
            "2.0 + 0.3 * (t - cohort[i])",  # true dynamic effect
            "np.clip(1.0 / (1.0 + np.exp(-score)), 0.05, 0.60)",
            'base_period="universal"',
            "n_bootstrap=199, cband=True",
            "A[0, 0] -= self.alpha",  # PolynomialRidge intercept exemption
            'SurveyDesign(weights="w", strata="stratum", psu="psu")',
            "seed=42,",
        ]:
            assert fragment in src, f"source fragment missing: {fragment!r}"

    def test_all_code_cells_hash_pinned(self):
        """Complete source/output contract: EVERY code cell's normalized
        source is hash-pinned, so ANY source edit fails here until the pins
        are re-locked together with a fresh execution and updated
        rederivation constants."""
        cells = _code_cell_hashes()
        assert cells == ALL_CODE_CELL_HASHES, (
            "notebook code cells changed - re-execute the notebook and "
            "re-lock ALL_CODE_CELL_HASHES plus any affected rederivation "
            f"constants. Got: {cells}"
        )

    def test_hash_guard_detects_mutation(self):
        """Negative control: a one-character estimator-argument mutation
        must change the cell hash."""
        import hashlib

        nb = _load_nb()
        src = next(
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and 'outcome_learner="sieve"' in "".join(c["source"])
        )
        mutated = src.replace('outcome_learner="sieve"', 'outcome_learner="ridge"', 1)
        norm = lambda x: "\n".join(ln.rstrip() for ln in x.strip().splitlines())  # noqa: E731
        h = lambda x: hashlib.sha256(norm(x).encode()).hexdigest()[:16]  # noqa: E731
        assert h(mutated) != h(src)

    def test_paper_reference_present(self):
        assert_quotes_in_rendered(
            NB,
            ["The Econometrics Journal", "10.1093/ectj/utaa001"],
            surface="markdown",
        )

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


# ---------------------------------------------------------------------------
# Re-derivation (DGPs mirror the notebook cells exactly; source-fragment
# pins above keep the copies honest)
# ---------------------------------------------------------------------------


class _PolynomialRidge:
    """Mirror of the notebook's custom learner (degree-2 features + ridge)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _expand(self, X):
        n, d = X.shape
        cols = [np.ones(n)] + [X[:, j] for j in range(d)]
        for j in range(d):
            for k in range(j, d):
                cols.append(X[:, j] * X[:, k])
        return np.column_stack(cols)

    def fit(self, X, y, sample_weight=None):
        Z = self._expand(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        ZtW = Z.T * w
        A = ZtW @ Z + self.alpha * np.eye(Z.shape[1])
        A[0, 0] -= self.alpha
        self.coef_ = np.linalg.solve(A, ZtW @ y)
        return self

    def predict(self, X):
        return self._expand(np.asarray(X, dtype=np.float64)) @ self.coef_


@pytest.fixture(scope="module")
def panel_df():
    rng = np.random.default_rng(7)
    n_units, periods = 600, [1, 2, 3, 4, 5, 6]
    x1 = rng.normal(size=n_units)
    x2 = rng.normal(size=n_units)
    score = 0.9 * (x1**2 - 1.0) + 0.7 * x1 * x2
    p_any = np.clip(1.0 / (1.0 + np.exp(-score)), 0.05, 0.60)
    treated = rng.uniform(size=n_units) < p_any
    early = rng.uniform(size=n_units) < 0.5
    cohort = np.where(treated, np.where(early, 4, 5), 0)
    f_nl = 1.6 * (x1**2 - 1.0) + 1.0 * x1 * x2
    rows = []
    for i in range(n_units):
        alpha_i = 0.5 * x1[i] - 0.3 * x2[i] + rng.normal(scale=0.3)
        for t in periods:
            y = 1.0 + 0.25 * t + alpha_i + f_nl[i] * (t / 6) + rng.normal(scale=0.5)
            if cohort[i] > 0 and t >= cohort[i]:
                y += 2.0 + 0.3 * (t - cohort[i])
            rows.append((i, t, y, cohort[i], x1[i], x2[i]))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "first_treat", "x1", "x2"])


@pytest.fixture(scope="module")
def sieve_fit(panel_df):
    return DMLDiD(
        propensity_learner="logit",
        outcome_learner="sieve",
        n_folds=5,
        seed=42,
        base_period="universal",
    ).fit(panel_df, **FIT_KW)


class TestPanelRederivation:
    def test_cohort_split(self, panel_df):
        counts = panel_df.groupby("unit")["first_treat"].first().value_counts().to_dict()
        assert counts == {0: 354, 5: 125, 4: 121}

    def test_truth_and_sieve_fit(self, sieve_fit):
        num = den = 0.0
        for (g, t), cell in sieve_fit.group_time_effects.items():
            if t < g or cell.get("is_reference"):
                continue
            w = cell["n_treated"]
            num += w * (2.0 + 0.3 * (t - g))
            den += w
        np.testing.assert_allclose(num / den, 2.2388, atol=5e-4)
        np.testing.assert_allclose(sieve_fit.overall_att, 2.2804, atol=5e-4)
        np.testing.assert_allclose(sieve_fit.overall_se, 0.0438, atol=5e-4)

    def test_learner_table(self, panel_df):
        expected = {
            "linear": (2.5909, 0.0743),
            "ridge": (2.5898, 0.0736),
            "PolynomialRidge": (2.2818, 0.0438),
        }
        for name, (att, se) in expected.items():
            learner = _PolynomialRidge(alpha=1.0) if name == "PolynomialRidge" else name
            r = DMLDiD(outcome_learner=learner, n_folds=5, seed=42, base_period="universal").fit(
                panel_df, **FIT_KW
            )
            np.testing.assert_allclose(r.overall_att, att, atol=5e-4, err_msg=name)
            np.testing.assert_allclose(r.overall_se, se, atol=5e-4, err_msg=name)

    def test_honest_bounds(self, sieve_fit):
        honest = compute_honest_did(
            sieve_fit.aggregate("event_study"), method="relative_magnitude", M=1.0
        )
        np.testing.assert_allclose(honest.lb, 2.2291, atol=1e-3)
        np.testing.assert_allclose(honest.ub, 2.4779, atol=1e-3)
        np.testing.assert_allclose(honest.ci_lb, 2.1393, atol=1e-3)
        np.testing.assert_allclose(honest.ci_ub, 2.5677, atol=1e-3)

    def test_bootstrap_supt_bands(self, panel_df, sieve_fit, monkeypatch):
        # The notebook was executed on the NumPy weight backend, and the two
        # backends draw DIFFERENT (equally valid) multiplier matrices from
        # the same seed (REGISTRY weight-backend identity Note) - so force
        # the NumPy generator here to match the committed cband goldens on
        # Rust-enabled installs (pattern: test_bootstrap_chunking.py).
        from diff_diff import bootstrap_chunking

        monkeypatch.setattr(bootstrap_chunking, "_rust_bootstrap_weights", None)
        assert bootstrap_chunking.effective_weight_backend() == "numpy"
        r = DMLDiD(outcome_learner="sieve", n_folds=5, seed=42, n_bootstrap=199, cband=True).fit(
            panel_df, **FIT_KW
        )
        # bootstrap overrides inference only; point estimate matches the
        # analytical fit across base_period (post cells share the g-1 base)
        np.testing.assert_allclose(r.overall_att, sieve_fit.overall_att, rtol=1e-12)
        es = r.aggregate("event_study").to_dataframe()
        row0 = es[es["event_time"] == 0].iloc[0]
        np.testing.assert_allclose(row0["cband_lower"], 1.9245, atol=5e-4)
        np.testing.assert_allclose(row0["cband_upper"], 2.1622, atol=5e-4)


@pytest.fixture(scope="module")
def rcs_df():
    rng2 = np.random.default_rng(11)
    n_rows = 5000
    rx1 = rng2.normal(size=n_rows)
    rx2 = rng2.normal(size=n_rows)
    rscore = 0.9 * (rx1**2 - 1.0) + 0.7 * rx1 * rx2
    rp = np.clip(1.0 / (1.0 + np.exp(-rscore)), 0.05, 0.60)
    rtreated = rng2.uniform(size=n_rows) < rp
    rearly = rng2.uniform(size=n_rows) < 0.5
    rcohort = np.where(rtreated, np.where(rearly, 3, 4), 0)
    rt = rng2.integers(1, 6, size=n_rows)
    rf = 1.6 * (rx1**2 - 1.0) + 1.0 * rx1 * rx2
    ry = (
        1.0
        + 0.25 * rt
        + 0.5 * rx1
        - 0.3 * rx2
        + rf * (rt / 5.0)
        + rng2.normal(scale=0.6, size=n_rows)
    )
    rpost = (rcohort > 0) & (rt >= rcohort)
    ry = ry + np.where(rpost, 2.0 + 0.3 * (rt - rcohort), 0.0)
    rpsu = rng2.integers(0, 20, size=n_rows)
    rw = rng2.uniform(0.5, 1.5, size=n_rows)
    return pd.DataFrame(
        {
            "unit": np.arange(n_rows),
            "time": rt,
            "y": ry,
            "first_treat": rcohort,
            "x1": rx1,
            "x2": rx2,
            "psu": rpsu,
            "stratum": rpsu % 4,
            "w": rw / rw.mean(),
        }
    )


class TestRCSRederivation:
    @staticmethod
    def _fit_expecting_only_a23(est, rcs_df, **kw):
        """Record everything; the ONLY tolerated warning is the deliberate
        Assumption 2.3 lane warning - anything else (clipping, degenerate
        cells, fold reduction, lonely PSU, weight normalization) fails."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = est.fit(rcs_df, **kw)
        expected = [
            w for w in caught if w.category is UserWarning and "Assumption 2.3" in str(w.message)
        ]
        unexpected = [f"{w.category.__name__}: {w.message}" for w in caught if w not in expected]
        assert not unexpected, f"unexpected warnings: {unexpected}"
        # ... and the documented lane UserWarning must actually FIRE (the
        # lane contract, not just be tolerated).
        assert len(expected) == 1, [str(w.message) for w in caught]
        return r

    def test_rcs_fit(self, rcs_df):
        r = self._fit_expecting_only_a23(
            DMLDiD(outcome_learner="sieve", n_folds=5, seed=42, panel=False), rcs_df, **FIT_KW
        )
        np.testing.assert_allclose(r.overall_att, 2.4428, atol=5e-4)
        np.testing.assert_allclose(r.overall_se, 0.2373, atol=5e-4)

    def test_survey_fit(self, rcs_df):
        r = self._fit_expecting_only_a23(
            DMLDiD(outcome_learner="sieve", n_folds=5, seed=42, panel=False),
            rcs_df,
            **FIT_KW,
            survey_design=SurveyDesign(weights="w", strata="stratum", psu="psu"),
        )
        np.testing.assert_allclose(r.overall_att, 2.2886, atol=5e-4)
        np.testing.assert_allclose(r.overall_se, 0.1509, atol=5e-4)
        assert r.survey_metadata.df_survey == 16
        assert r.survey_metadata.n_psu == 20
        assert r.effective_n_folds is None  # 20 PSUs >= 5 folds: no reduction
