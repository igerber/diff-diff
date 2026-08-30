"""DMLDiD survey-design support (both lanes): TSL + replicate weights +
survey bootstrap + cluster=.

Survey support is a documented library extension of Chang (2020), which
assumes i.i.d. sampling (Assumption 2.3) — no external oracle exists
(DoubleML has no survey support; R ``did::`` is survey-naive), so the
evidence is the library's standard survey invariant battery (mirroring
``tests/test_survey_phase4.py``'s CS coverage) plus direct kernel
cross-checks against ``compute_survey_if_variance`` (TSL lane) and
``compute_replicate_if_variance`` (replicate lane).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import DMLDiD
from diff_diff.staggered_aggregation import fixed_cohort_agg_weights
from diff_diff.survey import (
    SurveyDesign,
    compute_replicate_if_variance,
    compute_survey_if_variance,
)
from diff_diff.utils import safe_inference
from tests.conftest import assert_nan_inference

FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="g")


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _make_panel(n_units=120, n_periods=4, seed=7, weight_fn=None, drop_one=False):
    rng = np.random.RandomState(seed)
    units = np.arange(n_units)
    psu = units // 6  # 20 PSUs, strictly coarser than the unit
    stratum = psu % 2
    g = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
    w = weight_fn(rng, n_units) if weight_fn is not None else rng.uniform(0.5, 2.0, n_units)
    rows = []
    for u in units:
        for t in range(1, n_periods + 1):
            y = 1.0 + 0.4 * t + 0.2 * rng.randn() + (1.5 if g[u] > 0 and t >= g[u] else 0.0)
            rows.append((u, t, y, g[u], rng.randn(), psu[u], stratum[u], w[u]))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "g", "x1", "psu", "stratum", "w"])
    if drop_one:
        # One treated unit's post outcome goes non-finite -> incomplete panel.
        victim = df.index[(df["g"] == 3) & (df["time"] == 3)][0]
        df.loc[victim, "y"] = np.nan
    return df


def _make_rcs(n_obs=900, seed=11, weight_fn=None):
    rng = np.random.RandomState(seed)
    obs = np.arange(n_obs)
    psu = obs // 45  # 20 PSUs
    stratum = psu % 2
    g = rng.choice([0, 3, 4], size=n_obs, p=[0.5, 0.25, 0.25])
    t = rng.choice([1, 2, 3, 4], size=n_obs)
    w = weight_fn(rng, n_obs) if weight_fn is not None else rng.uniform(0.5, 2.0, n_obs)
    y = 1.0 + 0.4 * t + 0.2 * rng.randn(n_obs) + np.where((g > 0) & (t >= g), 1.5, 0.0)
    return pd.DataFrame(
        {
            "unit": obs,
            "time": t,
            "y": y,
            "g": g,
            "x1": rng.randn(n_obs),
            "psu": psu,
            "stratum": stratum,
            "w": w,
        }
    )


def _fit(df, *, panel=True, survey=None, cluster=None, seed=42, ignore_warnings=True, **kw):
    est = DMLDiD(seed=seed, panel=panel, cluster=cluster, **kw)
    if not ignore_warnings:
        return est.fit(df, covariates=["x1"], survey_design=survey, **FIT_KW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, covariates=["x1"], survey_design=survey, **FIT_KW)


_DESIGN = SurveyDesign(weights="w", strata="stratum", psu="psu")


def _attach_jk1(df, psu_col="psu", w_col="w"):
    """Delete-cluster JK1 replicate columns over the frame's PSUs (the
    test_survey_phase6.py construction). Returns (df_copy, rep_cols)."""
    df = df.copy()
    psus = np.unique(df[psu_col])
    n_rep = len(psus)
    rep_cols = []
    for r, p in enumerate(psus):
        w_r = df[w_col].to_numpy(dtype=np.float64).copy()
        mask = (df[psu_col] == p).to_numpy()
        w_r[mask] = 0.0
        w_r[~mask] *= n_rep / (n_rep - 1)
        col = f"rep_{r}"
        df[col] = w_r
        rep_cols.append(col)
    return df, rep_cols


def _attach_brr(df, psu_col="psu", w_col="w", n_rep=8):
    """Half-sample BRR replicate columns (double one half, zero the other)."""
    df = df.copy()
    psu = df[psu_col].to_numpy()
    rep_cols = []
    for r in range(n_rep):
        half = ((psu + r) % 2) == 0
        w_r = df[w_col].to_numpy(dtype=np.float64).copy()
        w_r[half] *= 2.0
        w_r[~half] = 0.0
        col = f"brr_{r}"
        df[col] = w_r
        rep_cols.append(col)
    return df, rep_cols


def _jk1_design(rep_cols, **kw):
    return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1", **kw)


@pytest.fixture(scope="module")
def panel_replicate_df(panel_df):
    return _attach_jk1(panel_df)


@pytest.fixture(scope="module")
def rcs_replicate_df(rcs_df):
    return _attach_jk1(rcs_df)


@pytest.fixture(scope="module")
def panel_replicate(panel_replicate_df):
    df, rep_cols = panel_replicate_df
    return _fit(df, survey=_jk1_design(rep_cols))


@pytest.fixture(scope="module")
def rcs_replicate(rcs_replicate_df):
    df, rep_cols = rcs_replicate_df
    return _fit(df, panel=False, survey=_jk1_design(rep_cols))


@pytest.fixture(scope="module")
def panel_df():
    return _make_panel()


@pytest.fixture(scope="module")
def rcs_df():
    return _make_rcs()


@pytest.fixture(scope="module")
def panel_plain(panel_df):
    return _fit(panel_df)


@pytest.fixture(scope="module")
def panel_survey(panel_df):
    return _fit(panel_df, survey=_DESIGN)


@pytest.fixture(scope="module")
def rcs_plain(rcs_df):
    return _fit(rcs_df, panel=False)


@pytest.fixture(scope="module")
def rcs_survey(rcs_df):
    return _fit(rcs_df, panel=False, survey=_DESIGN)


@pytest.fixture(scope="module")
def rcs_cluster(rcs_df):
    return _fit(rcs_df, panel=False, cluster="psu")


# ---------------------------------------------------------------------------
# Duck learners for the capability-gate and weight-threading pins
# ---------------------------------------------------------------------------


class _XYOnlyRegressor:
    """(X, y)-only fit: valid on no-design fits, rejected under survey."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


class _WeightedRegressor:
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.mean_ = float(np.mean(y))
        else:
            self.mean_ = float(np.average(y, weights=np.where(sample_weight > 0, sample_weight, 0)))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


class _KwargsRegressor:
    def fit(self, X, y, **kwargs):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


class _SpyRegressor:
    """Records every sample_weight its fit() receives.

    The log lives on the CLASS: cross_fit_predict deep-copies the learner
    template per fold, and class attributes stay shared across copies.
    """

    log: list = []

    def fit(self, X, y, sample_weight=None):
        type(self).log.append(None if sample_weight is None else np.asarray(sample_weight).copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


class _SpyClassifier:
    log: list = []

    def fit(self, X, y, sample_weight=None):
        type(self).log.append(None if sample_weight is None else np.asarray(sample_weight).copy())
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = min(max(self.p_, 0.05), 0.95)
        return np.column_stack([np.full(X.shape[0], 1 - p), np.full(X.shape[0], p)])


# ---------------------------------------------------------------------------
# 1. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("panel", [True, False])
    def test_uniform_weights_match_unweighted(
        self, panel, panel_df, rcs_df, panel_plain, rcs_plain
    ):
        df = (panel_df if panel else rcs_df).copy()
        df["w1"] = 1.0
        # Weights-only design: no PSU, no strata -> stratified folds retained,
        # weighted kernels with w == 1 (1e-12, not bit-level: the
        # sample_weight path through the native solvers is a different
        # numerical route than the unweighted one).
        res_w = _fit(df, panel=panel, survey=SurveyDesign(weights="w1"))
        base = panel_plain if panel else rcs_plain
        np.testing.assert_allclose(res_w.overall_att, base.overall_att, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res_w.overall_se, base.overall_se, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("panel", [True, False])
    def test_weight_scale_invariance(self, panel, panel_df, rcs_df):
        df = (panel_df if panel else rcs_df).copy()
        df["w3"] = df["w"] * 3.0
        r1 = _fit(df, panel=panel, survey=_DESIGN)
        r3 = _fit(df, panel=panel, survey=SurveyDesign(weights="w3", strata="stratum", psu="psu"))
        np.testing.assert_allclose(r3.overall_att, r1.overall_att, rtol=1e-10)
        np.testing.assert_allclose(r3.overall_se, r1.overall_se, rtol=1e-10)

    @pytest.mark.parametrize("panel", [True, False])
    def test_weights_move_the_point_estimate(
        self, panel, panel_survey, rcs_survey, panel_plain, rcs_plain
    ):
        res, base = (panel_survey, panel_plain) if panel else (rcs_survey, rcs_plain)
        assert np.isfinite(res.overall_att)
        assert res.overall_att != base.overall_att

    @pytest.mark.parametrize("panel", [True, False])
    def test_full_design_smoke(self, panel, panel_survey, rcs_survey):
        res = panel_survey if panel else rcs_survey
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        assert res.survey_metadata is not None
        assert res.survey_metadata.n_strata == 2
        assert res.survey_metadata.n_psu == 20
        assert res.survey_metadata.df_survey == 18  # n_psu - n_strata
        assert res.cluster_name == "psu"
        assert res.n_clusters == 20

    def test_fpc_shrinks_se(self, panel_df):
        df = panel_df.copy()
        # Census-like FPC: each stratum's PSU population barely exceeds the
        # sample -> (1 - f_h) shrinks every variance contribution.
        df["fpc"] = 11
        r_nofpc = _fit(df, survey=_DESIGN)
        r_fpc = _fit(df, survey=SurveyDesign(weights="w", strata="stratum", psu="psu", fpc="fpc"))
        assert r_fpc.overall_se < r_nofpc.overall_se


# ---------------------------------------------------------------------------
# 2. Per-cell SE
# ---------------------------------------------------------------------------


def _reconstruct_cell_psi(kit, gt_key):
    info = kit.influence[gt_key]
    resolved = kit.bookkeeping["resolved_survey_unit"]
    n = len(resolved.weights)
    psi = np.zeros(n)
    psi[info["treated_idx"]] = info["treated_inf"]
    psi[info["control_idx"]] = info["control_inf"]
    return psi, resolved


class TestPerCellSE:
    @pytest.mark.parametrize("fixture", ["panel_survey", "rcs_survey"])
    def test_psu_design_matches_compute_survey_if_variance(self, fixture, request):
        res = request.getfixturevalue(fixture)
        kit = res._aggregation_kit
        checked = 0
        for gt_key, data in res.group_time_effects.items():
            if data.get("skip_reason") is not None or data.get("is_reference"):
                continue
            if gt_key not in kit.influence:
                continue
            psi, resolved = _reconstruct_cell_psi(kit, gt_key)
            expected = compute_survey_if_variance(psi, resolved)
            np.testing.assert_allclose(data["se"], np.sqrt(expected), rtol=1e-12)
            checked += 1
        assert checked > 0

    def test_strata_only_design_uses_weighted_sqrt_sum(self, panel_df):
        res = _fit(panel_df, survey=SurveyDesign(weights="w", strata="stratum"))
        kit = res._aggregation_kit
        checked = 0
        for gt_key, data in res.group_time_effects.items():
            if data.get("skip_reason") is not None or data.get("is_reference"):
                continue
            info = kit.influence.get(gt_key)
            if info is None:
                continue
            ssq = float(np.sum(info["treated_inf"] ** 2) + np.sum(info["control_inf"] ** 2))
            np.testing.assert_allclose(data["se"], np.sqrt(ssq), rtol=1e-12)
            checked += 1
        assert checked > 0

    def test_single_psu_retained_cells_nan_inference(self, panel_df):
        # ONE global PSU: cluster-cohesive folds impossible -> stratified
        # fallback fits the points; the clustered variance is unidentified
        # -> retained cells with NaN-consistent inference, NOT skips.
        df = panel_df.copy()
        df["one_psu"] = 0
        with pytest.warns(UserWarning, match="PSU"):
            res = _fit(df, survey=SurveyDesign(weights="w", psu="one_psu"), ignore_warnings=False)
        assert np.isfinite(res.overall_att)
        found = False
        for data in res.group_time_effects.values():
            if data.get("is_reference") or data.get("skip_reason") is not None:
                continue
            assert np.isfinite(data["effect"])
            assert_nan_inference(data)
            found = True
        assert found

    def test_all_lonely_psu_remove_nan_inference(self, panel_df):
        # >= 2 PSUs but every stratum is a singleton under lonely_psu
        # "remove": the meat collapses -> same retained-cell NaN contract.
        df = panel_df.copy()
        df["lone_stratum"] = df["unit"] % 4
        df["lone_psu"] = df["unit"] % 4
        with pytest.warns(UserWarning, match="lonely_psu|PSU"):
            res = _fit(
                df,
                survey=SurveyDesign(
                    weights="w", strata="lone_stratum", psu="lone_psu", lonely_psu="remove"
                ),
                ignore_warnings=False,
            )
        for data in res.group_time_effects.values():
            if data.get("is_reference") or data.get("skip_reason") is not None:
                continue
            assert np.isfinite(data["effect"])
            assert_nan_inference(data)


# ---------------------------------------------------------------------------
# 3. cluster= wiring
# ---------------------------------------------------------------------------


class TestClusterWiring:
    def test_bare_cluster_synthesizes_design(self, panel_df):
        res = _fit(panel_df, cluster="psu")
        assert res.survey_metadata is None  # declared-survey marker stays off
        assert res.cluster_name == "psu"
        assert res.n_clusters == 20
        assert res.df_inference == 19.0  # n_psu - 1

    def test_bare_cluster_matches_explicit_psu_design(self, panel_df):
        r_cluster = _fit(panel_df, cluster="psu")
        r_design = _fit(panel_df, survey=SurveyDesign(psu="psu"))
        np.testing.assert_allclose(r_cluster.overall_att, r_design.overall_att, rtol=1e-12)
        np.testing.assert_allclose(r_cluster.overall_se, r_design.overall_se, rtol=1e-12)

    def test_identity_psu_bare_cluster_bit_identical_on_complete_panel(self, panel_df, panel_plain):
        # Identity PSU (cluster == unit) on a fully complete panel: folds
        # stay stratified (predicate's coarser-than-unit conjunct), kernels
        # stay unweighted, aggregation masses coincide -> bit-identical.
        res = _fit(panel_df, cluster="unit")
        assert res.overall_att == panel_plain.overall_att
        # The SE deliberately moves: identity-PSU CR1 carries the
        # per-PSU centering + Bessel factor the plain sqrt(sum(if^2))
        # does not (that IS the clustering request).
        assert res.overall_se != panel_plain.overall_se
        assert np.isfinite(res.overall_se)

    def test_incomplete_panel_bare_cluster_moves_overall_att(self):
        # CS-parity divergence (staggered_results.py bare-cluster note): the
        # synthesized all-ones survey_weights switch aggregation masses from
        # per-cell complete-case n_treated to full cohort mass, so on an
        # INCOMPLETE panel the overall ATT moves. Expected, documented.
        df = _make_panel(drop_one=True)
        base = _fit(df)
        res = _fit(df, cluster="unit")
        assert np.isfinite(res.overall_att)
        assert res.overall_att != base.overall_att

    def test_xy_only_learner_works_under_bare_cluster(self, panel_df):
        res = _fit(panel_df, cluster="psu", outcome_learner=_XYOnlyRegressor())
        assert np.isfinite(res.overall_att)

    def test_design_psu_wins_over_cluster_with_warning(self, panel_df):
        df = panel_df.copy()
        df["other"] = df["unit"] % 7
        est = DMLDiD(seed=42, cluster="other")
        with pytest.warns(UserWarning, match="PSU|cluster"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = est.fit(df, covariates=["x1"], survey_design=_DESIGN, **FIT_KW)
        assert res.cluster_name == "psu"  # design PSU wins

    def test_design_without_psu_injects_cluster(self, panel_df):
        res = _fit(panel_df, survey=SurveyDesign(weights="w", strata="stratum"), cluster="psu")
        assert res.survey_metadata is not None
        assert res.survey_metadata.n_psu == 20
        assert res.cluster_name == "psu"

    def test_missing_cluster_column_raises(self, panel_df):
        with pytest.raises(ValueError, match="cluster column"):
            _fit(panel_df, cluster="nope")

    def test_nan_cluster_raises(self, panel_df):
        df = panel_df.copy()
        df.loc[df.index[0], "psu"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            _fit(df, cluster="psu")

    def test_cluster_constructor_rejects_non_str(self):
        with pytest.raises(ValueError, match="cluster must be"):
            DMLDiD(cluster=123)

    def test_cluster_set_params_transactional(self):
        est = DMLDiD()
        with pytest.raises(ValueError, match="cluster must be"):
            est.set_params(cluster=3.5)
        assert est.cluster is None  # probe re-init validated before mutating

    def test_cluster_direct_mutation_revalidated_at_fit(self, panel_df):
        est = DMLDiD(seed=42)
        est.cluster = 123  # bypasses __init__ validation
        with pytest.raises(ValueError, match="cluster must be"):
            est.fit(panel_df, covariates=["x1"], **FIT_KW)


class TestClusterReportingProvenance:
    """Bare cluster= carries df_inference but is NOT a survey fit — the
    reporting and practitioner surfaces must not describe it as one."""

    def test_bare_cluster_rcs_headline_uses_fixed_row_masses(self, rcs_cluster, rcs_survey):
        from diff_diff._reporting_helpers import describe_target_parameter

        assert rcs_cluster.survey_metadata is None
        assert rcs_cluster.df_inference is not None
        bare = describe_target_parameter(rcs_cluster)
        assert "survey" not in bare["name"]
        assert "SURVEY" not in bare["definition"]
        assert "FIXED cohort row masses" in bare["definition"]
        surv = describe_target_parameter(rcs_survey)
        assert "survey-cohort-mass-weighted" in surv["name"]

    def test_bare_cluster_panel_headline_keeps_cohort_mass_wording(self, panel_df):
        # Panel bare cluster= DOES switch to cohort masses (all-ones survey
        # weights replace per-cell complete-case n_treated) — pin that the
        # provenance split keeps this branch intact.
        from diff_diff._reporting_helpers import describe_target_parameter

        res = _fit(panel_df, cluster="psu")
        desc = describe_target_parameter(res)
        assert "cohort-mass-weighted" in desc["name"]
        assert "bare ``cluster=``" in desc["definition"]

    def test_rcs_dropped_warning_wording_by_provenance(self, rcs_df):
        df = rcs_df.copy()
        victim = df.index[(df["g"] == 3) & (df["time"] == 3)][0]
        df.loc[victim, "y"] = np.nan
        with pytest.warns(UserWarning, match="fixed cohort row masses"):
            _fit(df, panel=False, cluster="psu", ignore_warnings=False)
        with pytest.warns(UserWarning, match="survey cohort masses"):
            _fit(df, panel=False, survey=_DESIGN, ignore_warnings=False)

    def test_practitioner_refit_snippet_preserves_cluster(self, panel_df, rcs_cluster):
        from diff_diff import practitioner_next_steps

        # Injected cluster on a PSU-less design must survive into the
        # learner-sensitivity refit (it carries the PSU-cohesive folds and
        # clustered inference that isolate the learner comparison).
        res = _fit(panel_df, survey=SurveyDesign(weights="w", strata="stratum"), cluster="psu")
        text = str(practitioner_next_steps(res))
        assert "survey_design=" in text
        assert "cluster='psu'" in text
        # Bare cluster= keeps its cluster too.
        assert "cluster='psu'" in str(practitioner_next_steps(rcs_cluster))

    def test_practitioner_refit_snippet_no_cluster_on_plain(self, panel_plain):
        from diff_diff import practitioner_next_steps

        assert "cluster=" not in str(practitioner_next_steps(panel_plain))


class TestSurveyMetadataRawScale:
    """survey_metadata provenance must be on the RAW weight scale: resolve()
    rescales pweights to mean 1, so recomputing at the estimation index
    level with the resolved weights would misreport sum_weights and
    weight_range (the module fixtures' uniform(0.5, 2.0) weights are
    deliberately non-unit-scale)."""

    def test_panel_metadata_uses_raw_unit_weights(self, panel_df, panel_survey):
        md = panel_survey.survey_metadata
        unit_w = panel_df.groupby("unit")["w"].first()
        np.testing.assert_allclose(md.sum_weights, unit_w.sum(), rtol=1e-12)
        np.testing.assert_allclose(md.weight_range, (unit_w.min(), unit_w.max()), rtol=1e-12)

    def test_rcs_metadata_uses_raw_obs_weights(self, rcs_df, rcs_survey):
        md = rcs_survey.survey_metadata
        w = rcs_df["w"]
        np.testing.assert_allclose(md.sum_weights, w.sum(), rtol=1e-12)
        np.testing.assert_allclose(md.weight_range, (w.min(), w.max()), rtol=1e-12)

    def test_injected_cluster_metadata_uses_raw_unit_weights(self, panel_df):
        # PSU-less design + cluster=: metadata is recomputed on the inject
        # path AND again at unit level — the final values must still be the
        # raw unit-level scale.
        res = _fit(panel_df, survey=SurveyDesign(weights="w", strata="stratum"), cluster="psu")
        md = res.survey_metadata
        unit_w = panel_df.groupby("unit")["w"].first()
        np.testing.assert_allclose(md.sum_weights, unit_w.sum(), rtol=1e-12)
        np.testing.assert_allclose(md.weight_range, (unit_w.min(), unit_w.max()), rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Folds
# ---------------------------------------------------------------------------


class TestFolds:
    def test_psu_folds_diagnostic_true_under_coarse_design(self, panel_survey):
        diags = [d for d in panel_survey.cross_fit_diagnostics.values() if "psu_folds" in d]
        assert diags and all(d["psu_folds"] is True for d in diags)

    def test_identity_psu_keeps_stratified_folds(self, panel_df):
        res = _fit(panel_df, cluster="unit")
        diags = [d for d in res.cross_fit_diagnostics.values() if "psu_folds" in d]
        assert diags and all(d["psu_folds"] is False for d in diags)

    @pytest.mark.parametrize("panel", [True, False])
    def test_few_psus_reduce_fold_count_keep_cohesion(self, panel, panel_df, rcs_df):
        # 2 <= n_psu_global < n_folds: with >= 2 PSUs the clustered variance
        # is IDENTIFIED, so silently reverting to unit folds would legitimize
        # nuisances trained with within-PSU leakage (review R1 P1). Instead
        # the effective fold count is reduced to n_psu with a warning,
        # preserving PSU cohesion — on BOTH lanes.
        df = (panel_df if panel else rcs_df).copy()
        df["psu3"] = df["unit"] % 3
        with pytest.warns(UserWarning, match="fold count reduced to preserve cluster"):
            res = _fit(
                df,
                panel=panel,
                survey=SurveyDesign(weights="w", psu="psu3"),
                n_folds=5,
                ignore_warnings=False,
            )
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        diags = [d for d in res.cross_fit_diagnostics.values() if d.get("propensity")]
        assert diags and all(d["psu_folds"] is True for d in diags)
        # Effective fold count == n_psu (3), visible in the per-fold counts.
        assert all(len(d["propensity"]["n_fit_per_fold"]) == 3 for d in diags)
        # Provenance: requested vs realized fold counts both serialized.
        assert res.n_folds == 5
        assert res.effective_n_folds == 3
        d = res.to_dict()
        assert d["n_folds"] == 5 and d["effective_n_folds"] == 3
        assert "Effective folds (PSU-reduced):" in res.summary()
        assert "n_folds=5 (effective 3)" in repr(res)

    @pytest.mark.parametrize("panel", [True, False])
    def test_exact_psu_fold_match_uses_psu_folds_unreduced(self, panel, panel_df, rcs_df):
        # n_psu == n_folds: PSU folds at the requested count, no reduction.
        df = (panel_df if panel else rcs_df).copy()
        df["psu3"] = df["unit"] % 3
        res = _fit(df, panel=panel, survey=SurveyDesign(weights="w", psu="psu3"), n_folds=3)
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)
        diags = [d for d in res.cross_fit_diagnostics.values() if d.get("propensity")]
        assert diags and all(d["psu_folds"] is True for d in diags)
        assert all(len(d["propensity"]["n_fit_per_fold"]) == 3 for d in diags)
        assert res.effective_n_folds is None
        assert res.to_dict()["effective_n_folds"] is None
        assert "Effective folds (PSU-reduced):" not in res.summary()
        assert "effective" not in repr(res)

    def test_rcs_composition_guard_skips_cell(self):
        # Coarse PSU folds on the RCS lane where whole periods live inside
        # single PSUs: some training complement loses one period's controls
        # entirely -> the 6a guard converts the cell to a
        # cross_fit_degenerate skip instead of a finite-but-invalid fit.
        rng = np.random.RandomState(3)
        n = 240
        t = np.repeat([1, 2, 3, 4], n // 4)
        # PSU == period block: any fold built from whole PSUs drops periods.
        psu = t.copy()
        g = rng.choice([0, 3], size=n, p=[0.6, 0.4])
        df = pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": t,
                "y": 1.0 + 0.3 * t + rng.randn(n) * 0.2 + np.where((g > 0) & (t >= g), 1.0, 0.0),
                "g": g,
                "x1": rng.randn(n),
                "psu": psu,
                "w": np.ones(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = DMLDiD(seed=0, panel=False, n_folds=2)
            try:
                res = est.fit(
                    df,
                    covariates=["x1"],
                    survey_design=SurveyDesign(weights="w", psu="psu"),
                    **FIT_KW,
                )
            except ValueError as exc:
                # Every cell degenerate is also an acceptable outcome here.
                assert "Could not estimate any group-time effects" in str(exc)
                return
        skips = {
            d.get("skip_reason") for d in res.cross_fit_diagnostics.values() if d.get("skip_reason")
        }
        assert "cross_fit_degenerate" in skips

    def test_spy_learner_receives_weights_on_both_nuisances(self, panel_df):
        _SpyClassifier.log = []
        _SpyRegressor.log = []
        _fit(
            panel_df,
            survey=_DESIGN,
            propensity_learner=_SpyClassifier(),
            outcome_learner=_SpyRegressor(),
        )
        ps_log, or_log = _SpyClassifier.log, _SpyRegressor.log
        assert ps_log and or_log
        assert all(w is not None for w in ps_log)
        assert all(w is not None for w in or_log)
        # Every received vector is a per-fold slice of the RESOLVED design
        # weights (resolve() normalizes pweights to sum(w) == n_obs; the
        # panel collapse then takes each unit's first row).
        raw = panel_df["w"].to_numpy()
        norm = raw * (len(raw) / raw.sum())
        unit_w = pd.Series(norm, index=panel_df.index).groupby(panel_df["unit"]).first()
        w_all = set(np.round(unit_w.to_numpy(), 9))
        for rec in ps_log + or_log:
            assert set(np.round(rec, 9)).issubset(w_all)

    def test_spy_learner_receives_no_weights_without_design(self, panel_df):
        _SpyClassifier.log = []
        _SpyRegressor.log = []
        _fit(
            panel_df,
            propensity_learner=_SpyClassifier(),
            outcome_learner=_SpyRegressor(),
        )
        ps_log, or_log = _SpyClassifier.log, _SpyRegressor.log
        assert ps_log and or_log
        assert all(w is None for w in ps_log + or_log)


# ---------------------------------------------------------------------------
# 5. Skips
# ---------------------------------------------------------------------------


class TestZeroWeightMass:
    def test_panel_zero_treated_mass_skips(self, panel_df):
        df = panel_df.copy()
        df.loc[df["g"] == 3, "w"] = 0.0  # cohort present, zero survey mass
        with pytest.warns(UserWarning, match="zero_weight_mass"):
            res = _fit(df, survey=_DESIGN, ignore_warnings=False)
        reasons = {
            d.get("skip_reason") for d in res.group_time_effects.values() if d.get("skip_reason")
        }
        assert "zero_weight_mass" in reasons

    def test_rcs_zero_group_mass_skips(self, rcs_df):
        df = rcs_df.copy()
        df.loc[(df["g"] == 3) & (df["time"] >= 3), "w"] = 0.0
        with pytest.warns(UserWarning, match="zero_weight_mass"):
            res = _fit(df, panel=False, survey=_DESIGN, ignore_warnings=False)
        reasons = {
            d.get("skip_reason") for d in res.group_time_effects.values() if d.get("skip_reason")
        }
        assert "zero_weight_mass" in reasons


# ---------------------------------------------------------------------------
# 6. Bootstrap
# ---------------------------------------------------------------------------


class TestSurveyBootstrap:
    @pytest.mark.parametrize("panel", [True, False])
    def test_survey_bootstrap_smoke(self, panel, panel_df, rcs_df, ci_params):
        df = panel_df if panel else rcs_df
        res = _fit(df, panel=panel, survey=_DESIGN, n_bootstrap=ci_params.bootstrap(49))
        assert np.isfinite(res.overall_att)
        assert np.isfinite(res.overall_se)

    def test_single_psu_bootstrap_nan_with_dml_label(self, panel_df, ci_params):
        df = panel_df.copy()
        df["one_psu"] = 0
        with pytest.warns(UserWarning, match="DMLDiD bootstrap with survey/cluster design"):
            res = _fit(
                df,
                survey=SurveyDesign(weights="w", psu="one_psu"),
                n_bootstrap=ci_params.bootstrap(29),
                ignore_warnings=False,
            )
        assert np.isnan(res.overall_se)

    def test_replay_label_and_legacy_kit_fallback(self, panel_df, ci_params):
        df = panel_df.copy()
        df["one_psu"] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _fit(
                df,
                survey=SurveyDesign(weights="w", psu="one_psu"),
                n_bootstrap=ci_params.bootstrap(29),
            )
        # Post-fit replay re-emits the <2-PSU warning branded DMLDiD (the
        # kit label fix; the replay host used to hardcode CallawaySantAnna).
        with pytest.warns(UserWarning, match="DMLDiD bootstrap with survey/cluster design"):
            res.aggregate("event_study")
        # Legacy kits without the key fall back to the CS default, no crash.
        res._aggregation_kit.bookkeeping.pop("bootstrap_label")
        with pytest.warns(UserWarning, match="CallawaySantAnna bootstrap with survey/cluster"):
            res.aggregate("event_study")

    def test_survey_bootstrap_replay_matches_fit(self, panel_df, ci_params):
        res = _fit(panel_df, survey=_DESIGN, n_bootstrap=ci_params.bootstrap(49))
        agg = res.aggregate("simple")
        np.testing.assert_allclose(float(agg.att[0]), res.overall_att, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. Rejections
# ---------------------------------------------------------------------------


class TestRejections:
    def test_cluster_plus_replicate_hits_targeted_message(self, panel_replicate_df):
        df, rep_cols = panel_replicate_df
        with pytest.raises(NotImplementedError, match="cluster.*replicate-weight"):
            _fit(df, survey=_jk1_design(rep_cols), cluster="psu")

    def test_bogus_cluster_on_replicate_fit_raises_value_error_first(self, panel_replicate_df):
        # CS ordering parity: the cluster column-existence check runs BEFORE
        # the replicate + cluster= rejection, so a bogus name is ValueError.
        df, rep_cols = panel_replicate_df
        with pytest.raises(ValueError, match="cluster column 'nope' not found"):
            _fit(df, survey=_jk1_design(rep_cols), cluster="nope")

    def test_bootstrap_plus_replicate_rejected(self, panel_replicate_df):
        df, rep_cols = panel_replicate_df
        with pytest.raises(NotImplementedError, match="bootstrap.*replicate-weight"):
            _fit(df, survey=_jk1_design(rep_cols), n_bootstrap=29)

    def test_non_pweight_rejected(self, panel_df):
        with pytest.raises(ValueError, match="pweight"):
            _fit(panel_df, survey=SurveyDesign(weights="w", weight_type="fweight"))

    def test_total_fails_closed_on_survey_fit(self, panel_survey):
        with pytest.raises(NotImplementedError):
            panel_survey.aggregate("total")

    def test_total_admitted_on_complete_panel_bare_cluster(self, panel_df):
        res = _fit(panel_df, cluster="psu")
        total = res.aggregate("total")
        assert np.isfinite(float(total.att[0]))

    def test_panel_mover_survey_column_rejected(self, panel_df):
        df = panel_df.copy()
        # weight varies WITHIN a unit -> panel-lane unit-constancy rejection
        df.loc[df["time"] == 1, "w"] = df.loc[df["time"] == 1, "w"] * 2.0
        with pytest.raises(ValueError):
            _fit(df, survey=_DESIGN)

    def test_xy_only_learner_rejected_up_front_under_survey(self, panel_df):
        with pytest.raises(TypeError, match="sample_weight"):
            _fit(panel_df, survey=_DESIGN, outcome_learner=_XYOnlyRegressor())

    def test_weighted_and_kwargs_learners_accepted_under_survey(self, panel_df):
        r1 = _fit(panel_df, survey=_DESIGN, outcome_learner=_WeightedRegressor())
        assert np.isfinite(r1.overall_att)
        r2 = _fit(panel_df, survey=_DESIGN, outcome_learner=_KwargsRegressor())
        assert np.isfinite(r2.overall_att)


# ---------------------------------------------------------------------------
# 8. df threading + rendering
# ---------------------------------------------------------------------------


class TestDfThreading:
    def test_per_cell_inference_uses_survey_df(self, panel_survey):
        df_survey = panel_survey.survey_metadata.df_survey
        for data in panel_survey.group_time_effects.values():
            if data.get("skip_reason") is not None or data.get("is_reference"):
                continue
            t_ref, p_ref, ci_ref = safe_inference(
                data["effect"], data["se"], alpha=0.05, df=df_survey
            )
            np.testing.assert_allclose(data["p_value"], p_ref, rtol=1e-12)
            np.testing.assert_allclose(data["conf_int"], ci_ref, rtol=1e-12)

    def test_summary_keeps_t_labels_under_survey(self, panel_survey):
        text = panel_survey.summary()
        assert "t-stat" in text and "z-stat" not in text
        assert "Survey Design" in text or "survey" in text.lower()

    def test_summary_keeps_t_labels_under_bare_cluster(self, panel_df):
        res = _fit(panel_df, cluster="psu")
        text = res.summary()
        assert "t-stat" in text and "z-stat" not in text

    def test_summary_keeps_z_labels_without_design(self, panel_plain):
        text = panel_plain.summary()
        assert "z-stat" in text

    def test_to_dict_carries_cluster_fields(self, panel_survey):
        d = panel_survey.to_dict()
        assert d.get("cluster_name") == "psu"
        assert d.get("n_clusters") == 20

    def test_event_study_df_threads_kit_df_survey(self, panel_survey):
        es = panel_survey.aggregate("event_study")
        assert es.df is not None
        df_vals = np.asarray(es.df, dtype=float)
        finite = df_vals[np.isfinite(df_vals)]
        assert finite.size > 0
        assert np.all(finite == panel_survey.survey_metadata.df_survey)

    def test_bootstrap_replay_scalar_persists_per_row_cleared(self, panel_df, ci_params):
        # Two-channel contract on the bootstrap REPLAY (DMLDiD has no
        # fit-time ES surface): the per-row df column is all-NaN beside
        # percentile inference (M-027) while the df_survey scalar - the
        # fit's resolved scalar inference df - persists.
        boot = _fit(panel_df, survey=_DESIGN, n_bootstrap=ci_params.bootstrap(29))
        es = boot.aggregate("event_study")
        assert np.all(np.isnan(np.asarray(es.df, dtype=float)))
        assert es.df_survey is not None and np.isfinite(es.df_survey)
        assert es.df_survey == boot.survey_metadata.df_survey

    def test_describe_target_parameter_names_cohort_masses(self, panel_survey, panel_plain):
        from diff_diff._reporting_helpers import describe_target_parameter

        surv = describe_target_parameter(panel_survey)
        assert "cohort-mass-weighted" in surv["name"]
        plain = describe_target_parameter(panel_plain)
        assert "valid-treated-count-weighted" in plain["name"]

    def test_design_effect_diagnostic_live_on_survey_fit(self, panel_survey):
        from diff_diff.diagnostic_report import DiagnosticReport

        report = DiagnosticReport(panel_survey).to_dict()
        block = report.get("design_effect")
        assert block is not None
        assert "not applicable" not in str(block).lower()


# ---------------------------------------------------------------------------
# 9. Aggregation masses
# ---------------------------------------------------------------------------


class TestAggregationMasses:
    @pytest.mark.parametrize("fixture", ["panel_survey", "rcs_survey"])
    def test_survey_cohort_masses_drive_aggregations(self, fixture, request):
        res = request.getfixturevalue(fixture)
        for level in ("simple", "event_study", "group"):
            agg = res.aggregate(level)
            assert np.all(np.isfinite(np.asarray(agg.att, dtype=float)))

    def test_rcs_masses_equal_hand_computed_weighted_masses(self, rcs_df, rcs_survey):
        bk = rcs_survey._aggregation_kit.bookkeeping
        masses = fixed_cohort_agg_weights(bk)
        raw = rcs_df["w"].to_numpy()
        norm_factor = len(raw) / raw.sum()  # resolve() normalizes to sum == n
        for g in (3, 4):
            expected = float(rcs_df.loc[rcs_df["g"] == g, "w"].sum()) * norm_factor
            np.testing.assert_allclose(masses[g], expected, rtol=1e-10)

    def test_survey_mass_dict_no_int64_collision(self):
        # Regression for the >2**53 float-key collision the RCS precompute
        # deliberately avoids: the survey mass branch keys by NATIVE cohort
        # values, so two int64 cohorts colliding as float64 stay distinct.
        g1 = 2**53
        g2 = 2**53 + 1
        assert float(g1) == float(g2)  # they WOULD collide as float keys
        unit_cohorts = np.array([g1, g1, g2, 0], dtype=np.int64)
        sw = np.array([1.0, 2.0, 5.0, 7.0])
        masses = fixed_cohort_agg_weights({"survey_weights": sw, "unit_cohorts": unit_cohorts})
        assert masses[g1] == 3.0
        assert masses[g2] == 5.0


# ---------------------------------------------------------------------------
# 10. Replicate-weight designs (IF-reweighting; per-cell + aggregate)
# ---------------------------------------------------------------------------


class TestReplicateDesigns:
    @pytest.mark.parametrize("fixture", ["panel_replicate", "rcs_replicate"])
    def test_acceptance_both_lanes(self, fixture, request):
        res = request.getfixturevalue(fixture)
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        assert res.survey_metadata is not None
        # Full-rank JK1 over 20 PSUs: QR-rank df = R - 1.
        assert res.survey_metadata.df_survey == 19
        finite_cells = [
            v
            for v in res.group_time_effects.values()
            if v.get("skip_reason") is None and not v.get("is_reference")
        ]
        assert finite_cells
        assert all(np.isfinite(v["se"]) and v["se"] > 0 for v in finite_cells)

    def test_brr_method_accepted(self, panel_df):
        df, rep_cols = _attach_brr(panel_df)
        res = _fit(
            df,
            survey=SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="BRR"),
        )
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        assert res.survey_metadata is not None

    def test_combined_weights_false_branch(self, panel_replicate_df, panel_replicate):
        # combined_weights=False takes the ratio = w_r branch (not w_r/w) and
        # a different df_survey analysis-weight construction; the SE must
        # DIFFER from the combined fit on the same frame (a regression that
        # routed it through the combined branch would reproduce it exactly).
        df, rep_cols = panel_replicate_df
        res = _fit(df, survey=_jk1_design(rep_cols, combined_weights=False))
        assert np.isfinite(res.overall_att) and np.isfinite(res.overall_se)
        np.testing.assert_allclose(res.overall_att, panel_replicate.overall_att, rtol=1e-12)
        assert not np.isclose(res.overall_se, panel_replicate.overall_se, rtol=1e-6)

    @pytest.mark.parametrize("fixture", ["panel_replicate", "rcs_replicate"])
    def test_per_cell_se_matches_compute_replicate_if_variance(self, fixture, request):
        res = request.getfixturevalue(fixture)
        kit = res._aggregation_kit
        checked = 0
        for gt_key, data in res.group_time_effects.items():
            if data.get("skip_reason") is not None or data.get("is_reference"):
                continue
            if gt_key not in kit.influence:
                continue
            psi, resolved = _reconstruct_cell_psi(kit, gt_key)
            variance, _n_valid = compute_replicate_if_variance(psi, resolved)
            np.testing.assert_allclose(data["se"], np.sqrt(variance), rtol=1e-12)
            checked += 1
        assert checked > 0

    @pytest.mark.parametrize("learner", ["linear", "sieve"])
    def test_scale_invariance_linear_sieve(self, panel_replicate_df, learner):
        # Replicate designs skip mean-1 normalization, so the raw weight
        # scale reaches the learners' sample_weight verbatim; linear/sieve
        # solves and the Hajek moments are scale-invariant (allclose, not
        # exact equality: BLAS summation order varies by backend/OS).
        df, rep_cols = panel_replicate_df
        base = _fit(df, survey=_jk1_design(rep_cols), outcome_learner=learner)
        df_s = df.copy()
        for c in ["w"] + rep_cols:
            df_s[c] = df_s[c] * 100.0
        scaled = _fit(df_s, survey=_jk1_design(rep_cols), outcome_learner=learner)
        np.testing.assert_allclose(scaled.overall_att, base.overall_att, rtol=1e-14)
        np.testing.assert_allclose(scaled.overall_se, base.overall_se, rtol=1e-14)

    def test_ridge_scale_dependence_bounded(self, panel_replicate_df):
        # RidgeLearner is weight-SCALE-sensitive by documented solve_ridge
        # behavior (unnormalized weighted loss vs a fixed penalty; REGISTRY
        # DMLDiD Note) — no equality pin, only a sanity ceiling on the drift.
        from diff_diff._learners import RidgeLearner

        df, rep_cols = panel_replicate_df
        base = _fit(df, survey=_jk1_design(rep_cols), outcome_learner=RidgeLearner(alpha=1.0))
        df_s = df.copy()
        for c in ["w"] + rep_cols:
            df_s[c] = df_s[c] * 100.0
        scaled = _fit(df_s, survey=_jk1_design(rep_cols), outcome_learner=RidgeLearner(alpha=1.0))
        assert np.isfinite(scaled.overall_att)
        assert abs(scaled.overall_att - base.overall_att) < 1e-2

    def test_df_tightening_spy_reaches_all_surfaces(self, panel_replicate_df, monkeypatch):
        # dCDH two-pass convention: reduce n_valid on EVERY call; the
        # min(df_survey, n_valid - 1) rule must reach per-cell inference,
        # the overall, AND survey_metadata.df_survey.
        from scipy import stats as _stats

        from diff_diff import survey as _survey_mod

        df, rep_cols = panel_replicate_df
        reduced_n_valid = 7  # (7 - 1) < R - 1 = 19 -> the cap binds
        original = _survey_mod.compute_replicate_if_variance

        def reduce_n_valid(psi, resolved_arg):
            var, _n_valid = original(psi, resolved_arg)
            return var, reduced_n_valid

        monkeypatch.setattr(_survey_mod, "compute_replicate_if_variance", reduce_n_valid)
        res = _fit(df, survey=_jk1_design(rep_cols))
        expected_df = reduced_n_valid - 1
        assert res.survey_metadata.df_survey == expected_df
        # Overall: min-cap picked the reduced value (19 -> 6).
        t_overall = res.overall_att / res.overall_se
        assert res.overall_p_value == pytest.approx(
            2 * _stats.t.sf(abs(t_overall), df=expected_df), rel=1e-10
        )
        # Per-cell: every finite cell's p-value uses the reduced df.
        checked = 0
        for v in res.group_time_effects.values():
            if v.get("skip_reason") is not None or v.get("is_reference"):
                continue
            if not (np.isfinite(v["se"]) and v["se"] > 0):
                continue
            t_cell = v["effect"] / v["se"]
            assert v["p_value"] == pytest.approx(
                2 * _stats.t.sf(abs(t_cell), df=expected_df), rel=1e-10
            )
            checked += 1
        assert checked > 0

    def test_overall_df_caps_never_replaces(self, panel_replicate_df, monkeypatch):
        # The CAP-vs-REPLACE discriminator: a rank-deficient-but-usable
        # design (duplicated replicate columns -> QR rank 5, df_survey = 4)
        # plus a spy n_valid with n_valid - 1 = 6 > 4. CS's replace
        # convention would report df = 6; DMLDiD's min-cap must keep 4
        # (deliberate documented divergence; REGISTRY DMLDiD Note).
        from scipy import stats as _stats

        from diff_diff import survey as _survey_mod

        df, rep_cols = panel_replicate_df
        keep = rep_cols[:5]
        r8_cols = []
        for i in range(8):
            src = keep[min(i, 4)]  # cols 5..7 duplicate col 4 -> rank 5
            col = f"r8_{i}"
            df = df.assign(**{col: df[src]})
            r8_cols.append(col)
        design = _jk1_design(r8_cols)
        assert design.resolve(df).df_survey == 4

        original = _survey_mod.compute_replicate_if_variance

        def reduce_n_valid(psi, resolved_arg):
            var, _n_valid = original(psi, resolved_arg)
            return var, 7  # < R = 8 -> the overall relay activates with 6

        monkeypatch.setattr(_survey_mod, "compute_replicate_if_variance", reduce_n_valid)
        res = _fit(df, survey=design)
        assert res.survey_metadata.df_survey == 4
        t_overall = res.overall_att / res.overall_se
        assert res.overall_p_value == pytest.approx(
            2 * _stats.t.sf(abs(t_overall), df=4), rel=1e-10
        )

    def test_rank_one_replicate_matrix_nan_inference(self, panel_df):
        # All replicate columns identical to the full weights: QR rank 1 ->
        # df_survey None AND zero replicate contrast -> per-cell degenerate
        # guard (se == NaN, not the clamped 0.0) + NaN inference on cell and
        # overall surfaces; survey_metadata.df_survey stays None (the df=0
        # sentinel is local to safe_inference, never leaked to metadata).
        df = panel_df.copy()
        rep_cols = []
        for r in range(6):
            col = f"same_{r}"
            df[col] = df["w"]
            rep_cols.append(col)
        res = _fit(df, survey=_jk1_design(rep_cols))
        assert res.survey_metadata.df_survey is None
        checked = 0
        for v in res.group_time_effects.values():
            if v.get("skip_reason") is not None or v.get("is_reference"):
                continue
            assert np.isnan(v["se"])
            assert_nan_inference(v)
            checked += 1
        assert checked > 0
        assert_nan_inference(
            {
                "se": res.overall_se,
                "t_stat": res.overall_t_stat,
                "p_value": res.overall_p_value,
                "conf_int": res.overall_conf_int,
            }
        )

    @pytest.mark.parametrize("degenerate", [(0.0, 20), (float("nan"), 1)])
    def test_degenerate_cell_fails_closed_to_nan(self, panel_replicate_df, monkeypatch, degenerate):
        # Zero variance (a cell no replicate column perturbs) and the
        # n_valid < 2 helper contract (variance NaN) both fail closed
        # PER-CELL: se must be NaN EXPLICITLY (assert_nan_inference alone
        # cannot discriminate NaN from a clamped 0.0 — safe_inference NaNs
        # t/p/CI at se=0 either way). Spy hits the FIRST per-cell call only;
        # every other cell stays finite.
        from diff_diff import survey as _survey_mod

        df, rep_cols = panel_replicate_df
        original = _survey_mod.compute_replicate_if_variance
        counter = {"n": 0}

        def degrade_first_cell(psi, resolved_arg):
            counter["n"] += 1
            if counter["n"] == 1:
                return degenerate
            return original(psi, resolved_arg)

        monkeypatch.setattr(_survey_mod, "compute_replicate_if_variance", degrade_first_cell)
        res = _fit(df, survey=_jk1_design(rep_cols))
        cells = [
            v
            for v in res.group_time_effects.values()
            if v.get("skip_reason") is None and not v.get("is_reference")
        ]
        nan_cells = [v for v in cells if np.isnan(v["se"])]
        finite_cells = [v for v in cells if np.isfinite(v["se"]) and v["se"] > 0]
        assert len(nan_cells) == 1
        assert_nan_inference(nan_cells[0])
        assert len(finite_cells) == len(cells) - 1
        # The degenerate cell is RETAINED (NaN-consistent inference), never
        # silently skipped or given an analytical fallback.
        assert nan_cells[0].get("skip_reason") is None

    def test_aggregations_finite_es_vcov_none_total_closed(self, panel_replicate):
        es = panel_replicate.aggregate("event_study")
        se_vals = np.asarray(es.se, dtype=float)
        ref = np.asarray(es.is_reference, dtype=bool)
        assert np.all(np.isfinite(se_vals[~ref]))
        assert es.vcov is None  # deliberate under replicates (HonestDiD diagonal)
        gr = panel_replicate.aggregate("group")
        assert np.all(np.isfinite(np.asarray(gr.se, dtype=float)))
        with pytest.raises(NotImplementedError):
            panel_replicate.aggregate("total")

    def test_es_df_pins_inherited_replace_convention(self, panel_replicate_df, monkeypatch):
        # The SHARED event-study aggregation path REPLACES the design df
        # with min(non_none effective dfs) (staggered_aggregation.py) —
        # the inherited CS convention, NOT the fit-level min-cap. To
        # discriminate, the design must be rank-deficient (df_survey = 4)
        # with spy n_valid - 1 = 6 > 4: replace reports 6, a cap would
        # report 4. Pinned at 6 so the follow-up flip to min-cap (TODO.md
        # CS-parity row: all five replace-style relay sites) is detectable;
        # latent in practice (n_valid == R on well-formed designs).
        from diff_diff import survey as _survey_mod

        df, rep_cols = panel_replicate_df
        keep = rep_cols[:5]
        r8_cols = []
        for i in range(8):
            col = f"esr8_{i}"
            df = df.assign(**{col: df[keep[min(i, 4)]]})
            r8_cols.append(col)
        design = _jk1_design(r8_cols)
        assert design.resolve(df).df_survey == 4
        res = _fit(df, survey=design)
        original = _survey_mod.compute_replicate_if_variance

        def reduce_n_valid(psi, resolved_arg):
            var, _n_valid = original(psi, resolved_arg)
            return var, 7  # < R = 8 -> effective df 6 at the shared site

        monkeypatch.setattr(_survey_mod, "compute_replicate_if_variance", reduce_n_valid)
        es = res.aggregate("event_study")
        df_vals = np.asarray(es.df, dtype=float)
        finite = df_vals[np.isfinite(df_vals)]
        assert finite.size > 0
        assert np.all(finite == 6.0)  # inherited replace; a min-cap would give 4

    def test_replicate_fit_keeps_stratified_folds(self, panel_replicate):
        # Replicate designs have psu None by mutual exclusion -> the
        # PSU-cohesive fold gate is structurally inert.
        diags = [d for d in panel_replicate.cross_fit_diagnostics.values() if "psu_folds" in d]
        assert diags and all(d["psu_folds"] is False for d in diags)
