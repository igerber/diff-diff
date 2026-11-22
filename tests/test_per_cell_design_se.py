"""Per-cell ATT(g,t) SEs honor replicate and PSU survey designs.

Covers CallawaySantAnna (replicate lane) and the staggered TripleDifference
engine (replicate and PSU lanes). The invariant under test: under a replicate
or PSU design a per-cell SE comes from the same influence function and the
same design variance routine as the aggregate SEs of the same fit.

Before this change the cell SE ignored the design (one-cell panels, where the
cell IS the overall ATT): CS + JK1 reported cell SE 0.156789 against an overall
SE of 0.342755; staggered DDD reported 0.217289 against 0.325258 under both
JK1 replicates and ``SurveyDesign(psu=...)``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import diff_diff.staggered as staggered_mod
import diff_diff.survey as survey_mod
from diff_diff import CallawaySantAnna, SurveyDesign, TripleDifference
from diff_diff.prep_dgp import generate_staggered_ddd_data
from diff_diff.staggered import (
    _cluster_robust_se_from_per_gt_if,
    _design_se_for_cell,
    _replicate_se_from_per_gt_if,
)
from diff_diff.survey import (
    cap_replicate_df,
    compute_replicate_if_variance,
    compute_survey_if_variance,
)
from tests.conftest import assert_nan_inference

# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _add_jk1(df, cluster_col="cl", weight_col="w"):
    """Delete-one-cluster JK1 replicate columns (combined weights)."""
    clusters = np.sort(df[cluster_col].unique())
    n_cl = len(clusters)
    cols = {
        f"rep_{r}": np.where(df[cluster_col] == c, 0.0, df[weight_col] * n_cl / (n_cl - 1))
        for r, c in enumerate(clusters)
    }
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1), list(cols)


def _cs_panel(seed=7, n_clusters=20, per_cluster=6, n_periods=4):
    """Multi-cohort panel: cohorts {3, 4, never}, cluster-level period shocks."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        g = 3 if c < 6 else (4 if c < 12 else 0)
        shocks = rng.normal(0.0, 0.8, n_periods)
        for j in range(per_cluster):
            u = c * per_cluster + j
            a, x, w = rng.normal(), rng.normal(), rng.uniform(0.5, 2.0)
            for t in range(1, n_periods + 1):
                effect = 1.0 + 0.2 * (t - g) if (g and t >= g) else 0.0
                y = a + 0.3 * t + 0.4 * x * t + shocks[t - 1] + effect + rng.normal(0.0, 0.5)
                rows.append((u, c, c % 2, t, g, y, x, w))
    df = pd.DataFrame(rows, columns=["unit", "cl", "stratum", "t", "g", "y", "x0", "w"])
    return _add_jk1(df)


def _cs_rcs(seed=11, n_obs=900):
    """Genuine repeated cross-section: one row per unit, 20 PSUs."""
    rng = np.random.default_rng(seed)
    obs = np.arange(n_obs)
    cl = obs // 45
    g = rng.choice([0, 3, 4], size=n_obs, p=[0.5, 0.25, 0.25])
    t = rng.choice([1, 2, 3, 4], size=n_obs)
    shock = rng.normal(0.0, 0.5, (20, 5))[cl, t]
    y = 1.0 + 0.4 * t + shock + np.where((g > 0) & (t >= g), 1.5, 0.0) + rng.normal(0, 0.4, n_obs)
    df = pd.DataFrame(
        {
            "unit": obs,
            "cl": cl,
            "stratum": cl % 2,
            "t": t,
            "g": g,
            "y": y,
            "w": rng.uniform(0.5, 2.0, n_obs),
        }
    )
    return _add_jk1(df)


def _cs_one_cell():
    """Single cohort, two periods: the one cell IS the overall ATT."""
    rng = np.random.default_rng(7)
    n_cl, per = 30, 8
    rows = []
    for c in range(n_cl):
        treated = c < 12
        shock = rng.normal(0, 1.0)
        for j in range(per):
            u = c * per + j
            a = rng.normal(0, 1)
            w = rng.uniform(0.5, 2.0)
            for t in (1, 2):
                y = (
                    a
                    + 0.3 * t
                    + (shock if t == 2 else 0)
                    + (1.0 if (treated and t == 2) else 0)
                    + rng.normal(0, 0.5)
                )
                rows.append((u, c, t, 2 if treated else 0, y, w))
    df = pd.DataFrame(rows, columns=["unit", "cl", "t", "g", "y", "w"])
    return _add_jk1(df)


def _ddd_frame(seed, n_periods, cohort_periods, never_enabled_frac, shock_seed, all_periods):
    d = generate_staggered_ddd_data(
        n_units=240,
        n_periods=n_periods,
        cohort_periods=cohort_periods,
        never_enabled_frac=never_enabled_frac,
        seed=seed,
    )
    units = np.sort(d["unit"].unique())
    rng = np.random.default_rng(shock_seed)
    n_cl = 30
    ft = d.groupby("unit")["first_treat"].first().sort_values(kind="stable")
    cl_of = pd.Series(np.arange(len(ft)) // (len(ft) // n_cl), index=ft.index).clip(upper=n_cl - 1)
    d["cl"] = d["unit"].map(cl_of)
    d["stratum"] = d["cl"] % 2
    last = d["period"].max()
    if all_periods:
        periods = np.sort(d["period"].unique())
        shock = pd.DataFrame(
            rng.normal(0, 0.6, (n_cl, len(periods))), index=range(n_cl), columns=periods
        )
        per_row = shock.to_numpy()[
            d["cl"].to_numpy(), np.searchsorted(periods, d["period"].to_numpy())
        ]
        d["outcome"] = d["outcome"] + np.where(d["eligibility"] == 1, per_row, 0.0)
    else:
        shock = pd.Series(rng.normal(0, 1.0, n_cl))
        d["outcome"] = d["outcome"] + np.where(
            (d["period"] == last) & (d["eligibility"] == 1), d["cl"].map(shock), 0.0
        )
    wu = pd.Series(rng.uniform(0.5, 2.0, len(units)), index=units)
    d["w"] = d["unit"].map(wu)
    return _add_jk1(d)


def _ddd_one_cell():
    return _ddd_frame(
        seed=3,
        n_periods=2,
        cohort_periods=[2],
        never_enabled_frac=0.5,
        shock_seed=11,
        all_periods=False,
    )


def _ddd_panel():
    return _ddd_frame(
        seed=5,
        n_periods=5,
        cohort_periods=[3, 4],
        never_enabled_frac=0.3,
        shock_seed=13,
        all_periods=True,
    )


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------


def _jk1(rep_cols):
    return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")


def _fit_cs(df, survey=None, method="reg", covariates=None, panel=True, time="t"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CallawaySantAnna(estimation_method=method, panel=panel).fit(
            df,
            outcome="y",
            unit="unit",
            time=time,
            first_treat="g",
            covariates=covariates,
            survey_design=survey,
        )


def _fit_ddd(df, survey=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TripleDifference().fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            partition="eligibility",
            survey_design=survey,
        )


def _estimable_cells(res):
    out = {}
    for key, cell in res.group_time_effects.items():
        if cell.get("skip_reason") is not None or cell.get("is_reference"):
            continue
        if not np.isfinite(cell["effect"]):
            continue
        out[(float(key[0]), float(key[1]))] = cell
    return out


def _cell_psi(res, key):
    kit = res._aggregation_kit
    resolved = kit.bookkeeping["resolved_survey_unit"]
    info = kit.influence[key]
    psi = np.zeros(len(resolved.weights))
    psi[info["treated_idx"]] = info["treated_inf"]
    psi[info["control_idx"]] = info["control_inf"]
    return psi, resolved


def _kit_cells(res):
    kit = res._aggregation_kit
    for key, cell in res.group_time_effects.items():
        if cell.get("skip_reason") is not None or cell.get("is_reference"):
            continue
        if key not in kit.influence or len(kit.influence[key]["treated_idx"]) == 0:
            continue
        yield key, cell


# Cell-count-preserving lane parametrization: each id names the per-cell
# producer the fit exercises under a survey design.
CS_LANES = {
    "vectorized_reg": dict(method="reg", covariates=None),
    "general_reg_cov": dict(method="reg", covariates=["x0"]),
    "general_ipw": dict(method="ipw", covariates=["x0"]),
    "general_dr": dict(method="dr", covariates=["x0"]),
}


@pytest.fixture(scope="module")
def cs_panel():
    return _cs_panel()


@pytest.fixture(scope="module")
def cs_rcs():
    return _cs_rcs()


@pytest.fixture(scope="module")
def ddd_panel():
    return _ddd_panel()


# ---------------------------------------------------------------------------
# 1. Exact match to the design variance routine
# ---------------------------------------------------------------------------


class TestExactMatchToVarianceRoutine:
    @pytest.mark.parametrize("lane", sorted(CS_LANES))
    def test_cs_replicate_panel(self, cs_panel, lane):
        df, rep_cols = cs_panel
        res = _fit_cs(df, survey=_jk1(rep_cols), **CS_LANES[lane])
        checked = 0
        for key, cell in _kit_cells(res):
            psi, resolved = _cell_psi(res, key)
            variance, _ = compute_replicate_if_variance(psi, resolved)
            np.testing.assert_allclose(cell["se"], np.sqrt(variance), rtol=1e-12)
            checked += 1
        assert checked >= 4

    def test_cs_replicate_rcs(self, cs_rcs):
        df, rep_cols = cs_rcs
        res = _fit_cs(df, survey=_jk1(rep_cols), panel=False)
        checked = 0
        for key, cell in _kit_cells(res):
            psi, resolved = _cell_psi(res, key)
            variance, _ = compute_replicate_if_variance(psi, resolved)
            np.testing.assert_allclose(cell["se"], np.sqrt(variance), rtol=1e-12)
            checked += 1
        assert checked >= 4

    def test_ddd_replicate(self, ddd_panel):
        df, rep_cols = ddd_panel
        res = _fit_ddd(df, survey=_jk1(rep_cols))
        checked = 0
        for key, cell in _kit_cells(res):
            psi, resolved = _cell_psi(res, key)
            variance, _ = compute_replicate_if_variance(psi, resolved)
            np.testing.assert_allclose(cell["se"], np.sqrt(variance), rtol=1e-12)
            checked += 1
        assert checked >= 4

    def test_ddd_psu(self, ddd_panel):
        df, _ = ddd_panel
        res = _fit_ddd(df, survey=SurveyDesign(weights="w", psu="cl"))
        checked = 0
        for key, cell in _kit_cells(res):
            psi, resolved = _cell_psi(res, key)
            variance = compute_survey_if_variance(psi, resolved)
            np.testing.assert_allclose(cell["se"], np.sqrt(variance), rtol=1e-12)
            checked += 1
        assert checked >= 4


# ---------------------------------------------------------------------------
# 2. One-cell coherence: the cell IS the overall ATT
# ---------------------------------------------------------------------------


class TestOneCellCoherence:
    def test_cs_replicate(self):
        df, rep_cols = _cs_one_cell()
        res = _fit_cs(df, survey=_jk1(rep_cols))
        weights_only = _fit_cs(df, survey=SurveyDesign(weights="w"))
        (cell,) = _estimable_cells(res).values()
        (cell_w,) = _estimable_cells(weights_only).values()
        np.testing.assert_allclose(cell["effect"], res.att, rtol=1e-12)
        np.testing.assert_allclose(cell["se"], res.se, rtol=1e-10)
        # Measured: 0.342755 (design) vs 0.156789 (weights-only, the pre-fix
        # cell SE under the replicate design).
        np.testing.assert_allclose(cell["se"], 0.342755, rtol=1e-5)
        np.testing.assert_allclose(cell_w["se"], 0.156789, rtol=1e-5)
        np.testing.assert_allclose(cell["p_value"], res.p_value, rtol=1e-8)

    @pytest.mark.parametrize("design", ["replicate", "psu"])
    def test_ddd(self, design):
        df, rep_cols = _ddd_one_cell()
        survey = _jk1(rep_cols) if design == "replicate" else SurveyDesign(weights="w", psu="cl")
        res = _fit_ddd(df, survey=survey)
        weights_only = _fit_ddd(df, survey=SurveyDesign(weights="w"))
        (cell,) = _estimable_cells(res).values()
        (cell_w,) = _estimable_cells(weights_only).values()
        np.testing.assert_allclose(cell["effect"], res.att, rtol=1e-12)
        np.testing.assert_allclose(cell["se"], res.se, rtol=1e-10)
        # Measured: 0.325258 (design) vs 0.217289 (the pre-fix cell SE under
        # both designs).
        np.testing.assert_allclose(cell["se"], 0.325258, rtol=1e-5)
        np.testing.assert_allclose(cell_w["se"], 0.217289, rtol=1e-5)
        np.testing.assert_allclose(cell["p_value"], res.p_value, rtol=1e-8)


# ---------------------------------------------------------------------------
# 3. JK1 (delete-one-PSU) == CR1 on the same PSUs
# ---------------------------------------------------------------------------


class TestJK1EqualsCR1:
    """Algebraic identity: theta_r - theta = -G (psi_r - psi_bar) / (G - 1), so
    the JK1 variance (G-1)/G * sum(...)^2 equals CR1's G/(G-1) * sum((psi_r -
    psi_bar)^2) on the same cell influence function."""

    @pytest.mark.parametrize("lane", sorted(CS_LANES))
    def test_cs_panel(self, cs_panel, lane):
        df, rep_cols = cs_panel
        rep = _estimable_cells(_fit_cs(df, survey=_jk1(rep_cols), **CS_LANES[lane]))
        psu = _estimable_cells(
            _fit_cs(df, survey=SurveyDesign(weights="w", psu="cl"), **CS_LANES[lane])
        )
        assert rep.keys() == psu.keys() and len(rep) >= 4
        for key in rep:
            np.testing.assert_allclose(rep[key]["effect"], psu[key]["effect"], rtol=1e-10)
            np.testing.assert_allclose(rep[key]["se"], psu[key]["se"], rtol=1e-10)

    def test_cs_rcs(self, cs_rcs):
        df, rep_cols = cs_rcs
        rep = _estimable_cells(_fit_cs(df, survey=_jk1(rep_cols), panel=False))
        psu = _estimable_cells(_fit_cs(df, survey=SurveyDesign(weights="w", psu="cl"), panel=False))
        assert rep.keys() == psu.keys() and len(rep) >= 4
        for key in rep:
            np.testing.assert_allclose(rep[key]["se"], psu[key]["se"], rtol=1e-10)

    def test_ddd(self, ddd_panel):
        df, rep_cols = ddd_panel
        res_rep = _fit_ddd(df, survey=_jk1(rep_cols))
        res_psu = _fit_ddd(df, survey=SurveyDesign(weights="w", psu="cl"))
        rep, psu = _estimable_cells(res_rep), _estimable_cells(res_psu)
        assert rep.keys() == psu.keys() and len(rep) >= 4
        # The identity must be exercised on GMM-combined (multi-comparison-group)
        # cells too, not only on single-comparison-group ones.
        counts = set(res_rep.comparison_group_counts.values())
        assert 1 in counts and max(counts) > 1
        for key in rep:
            np.testing.assert_allclose(rep[key]["effect"], psu[key]["effect"], rtol=1e-10)
            np.testing.assert_allclose(rep[key]["se"], psu[key]["se"], rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Cell-local df: min(design df, n_valid - 1), never written back
# ---------------------------------------------------------------------------


def _spy_cell_n_valid(monkeypatch, n_valid):
    """Rewrite n_valid on PER-CELL replicate calls only.

    Patching ``survey.compute_replicate_if_variance`` would also reach the
    aggregate ``_se_from_psi`` path and move ``survey_metadata.df_survey``
    through the overall relay; wrapping the per-cell helper isolates the cell
    df from the aggregate df.
    """
    real = staggered_mod._replicate_se_from_per_gt_if

    def wrapped(inf_info, resolved_survey):
        se, _ = real(inf_info, resolved_survey)
        return se, n_valid

    monkeypatch.setattr(staggered_mod, "_replicate_se_from_per_gt_if", wrapped)


def _rank5(df, rep_cols):
    """20 replicate columns of rank 5 (four copies of five distinct columns)."""
    out = df.copy()
    for i, col in enumerate(rep_cols):
        out[col] = df[rep_cols[i % 5]]
    return out


class TestCellLocalDf:
    def _assert_cells_use_df(self, res, df_expected):
        checked = 0
        for cell in _estimable_cells(res).values():
            expected = 2.0 * stats.t.sf(abs(cell["effect"] / cell["se"]), df_expected)
            np.testing.assert_allclose(cell["p_value"], expected, rtol=1e-10)
            checked += 1
        assert checked >= 4

    @pytest.mark.parametrize("lane", ["vectorized_reg", "general_dr"])
    def test_cs_cap_slack_uses_n_valid(self, cs_panel, lane, monkeypatch):
        df, rep_cols = cs_panel
        _spy_cell_n_valid(monkeypatch, 7)
        res = _fit_cs(df, survey=_jk1(rep_cols), **CS_LANES[lane])
        # Full-rank JK1 over 20 PSUs: design df 19, cell df min(19, 7 - 1) = 6.
        self._assert_cells_use_df(res, 6)
        assert res.survey_metadata.df_survey == 19

    @pytest.mark.parametrize("lane", ["vectorized_reg", "general_dr"])
    def test_cs_cap_binds_at_design_df(self, cs_panel, lane, monkeypatch):
        df, rep_cols = cs_panel
        _spy_cell_n_valid(monkeypatch, 7)
        res = _fit_cs(_rank5(df, rep_cols), survey=_jk1(rep_cols), **CS_LANES[lane])
        # Rank 5 -> design df 4; n_valid - 1 = 6 must NOT replace it.
        self._assert_cells_use_df(res, 4)
        assert res.survey_metadata.df_survey == 4

    def test_ddd_cap(self, ddd_panel, monkeypatch):
        df, rep_cols = ddd_panel
        _spy_cell_n_valid(monkeypatch, 7)
        res = _fit_ddd(df, survey=_jk1(rep_cols))
        self._assert_cells_use_df(res, 6)
        assert res.survey_metadata.df_survey == len(rep_cols) - 1
        res5 = _fit_ddd(_rank5(df, rep_cols), survey=_jk1(rep_cols))
        self._assert_cells_use_df(res5, 4)
        assert res5.survey_metadata.df_survey == 4

    def test_rank_one_design_nan_inference(self, cs_panel, ddd_panel):
        df, rep_cols = cs_panel
        flat = df.copy()
        for col in rep_cols:
            flat[col] = df["w"]
        res = _fit_cs(flat, survey=_jk1(rep_cols))
        cells = _estimable_cells(res)
        assert len(cells) >= 4
        for cell in cells.values():
            assert_nan_inference(cell)

        ddf, drep = ddd_panel
        dflat = ddf.copy()
        for col in drep:
            dflat[col] = ddf["w"]
        dres = _fit_ddd(dflat, survey=_jk1(drep))
        dcells = _estimable_cells(dres)
        assert len(dcells) >= 4
        for cell in dcells.values():
            assert_nan_inference(cell)


# ---------------------------------------------------------------------------
# 5/6. Degenerate cells fail closed and never contaminate other cells
# ---------------------------------------------------------------------------


def _spy_one_cell(monkeypatch, call_index, variance):
    """Force ``variance`` on the ``call_index``-th per-cell replicate call."""
    real = staggered_mod._replicate_se_from_per_gt_if
    real_var = survey_mod.compute_replicate_if_variance
    state = {"n": 0}

    def wrapped(inf_info, resolved_survey):
        hit = state["n"] == call_index
        state["n"] += 1
        if not hit:
            return real(inf_info, resolved_survey)
        with monkeypatch.context() as m:
            m.setattr(
                survey_mod,
                "compute_replicate_if_variance",
                lambda psi, resolved: (variance, real_var(psi, resolved)[1]),
            )
            return real(inf_info, resolved_survey)

    monkeypatch.setattr(staggered_mod, "_replicate_se_from_per_gt_if", wrapped)


class TestDegenerateCells:
    @pytest.mark.parametrize("lane", ["vectorized_reg", "general_dr"])
    @pytest.mark.parametrize("variance", [0.0, np.nan, -1e-30])
    def test_cs_degenerate_cell_is_nan_and_isolated(self, cs_panel, lane, variance, monkeypatch):
        df, rep_cols = cs_panel
        base = _estimable_cells(_fit_cs(df, survey=_jk1(rep_cols), **CS_LANES[lane]))
        _spy_one_cell(monkeypatch, 0, variance)
        res = _fit_cs(df, survey=_jk1(rep_cols), **CS_LANES[lane])
        cells = _estimable_cells(res)
        # The cell is retained (finite effect, no skip reason) with NaN inference.
        assert cells.keys() == base.keys()
        bad = [k for k, c in cells.items() if not np.isfinite(c["se"])]
        assert len(bad) == 1
        assert_nan_inference(cells[bad[0]])
        assert res.group_time_effects  # still a completed fit
        for key in cells:
            if key == bad[0]:
                continue
            np.testing.assert_array_equal(cells[key]["se"], base[key]["se"])
            np.testing.assert_array_equal(cells[key]["p_value"], base[key]["p_value"])
            np.testing.assert_array_equal(cells[key]["conf_int"], base[key]["conf_int"])

    def test_ddd_degenerate_cell_is_nan_and_isolated(self, ddd_panel, monkeypatch):
        df, rep_cols = ddd_panel
        base = _estimable_cells(_fit_ddd(df, survey=_jk1(rep_cols)))
        _spy_one_cell(monkeypatch, 0, 0.0)
        cells = _estimable_cells(_fit_ddd(df, survey=_jk1(rep_cols)))
        assert cells.keys() == base.keys()
        bad = [k for k, c in cells.items() if not np.isfinite(c["se"])]
        assert len(bad) == 1
        assert_nan_inference(cells[bad[0]])
        for key in cells:
            if key != bad[0]:
                np.testing.assert_array_equal(cells[key]["se"], base[key]["se"])
                np.testing.assert_array_equal(cells[key]["p_value"], base[key]["p_value"])

    def test_ddd_single_psu_propagates_nan(self, ddd_panel):
        df, _ = ddd_panel
        one = df.assign(one_psu=0)
        res = _fit_ddd(one, survey=SurveyDesign(weights="w", psu="one_psu"))
        cells = _estimable_cells(res)
        assert len(cells) >= 4
        for cell in cells.values():
            # Unidentified clustered variance: NaN, never the unit-level SE.
            assert_nan_inference(cell)


# ---------------------------------------------------------------------------
# 7. Unchanged lanes (literals captured from the unmodified tree)
# ---------------------------------------------------------------------------

# Captured on the tree BEFORE the per-cell dispatch landed (rtol 1e-10: lane
# misrouting moves these by more than 1e-2 relative; 1e-10 absorbs backend /
# BLAS rounding, so one literal set serves both backends).
_CS_WEIGHTED = {
    (3.0, 2.0): 0.22824161596588807,
    (3.0, 3.0): 0.3248858376303261,
    (3.0, 4.0): 0.33702332799162515,
    (4.0, 2.0): 0.28234200951533633,
    (4.0, 3.0): 0.2798227207932187,
    (4.0, 4.0): 0.38146007170016494,
}
UNCHANGED_CS = {
    "none": {
        (3.0, 2.0): 0.2287237602551264,
        (3.0, 3.0): 0.27449135225445065,
        (3.0, 4.0): 0.2845982302625901,
        (4.0, 2.0): 0.26226113435289605,
        (4.0, 3.0): 0.2488707412488824,
        (4.0, 4.0): 0.33529556326941945,
    },
    "weights": _CS_WEIGHTED,
    # Strata-only: the per-cell SE is the weights-only one (documented
    # convention - the full design enters the aggregate SEs only).
    "strata": _CS_WEIGHTED,
}
_DDD_WEIGHTED = {
    (3.0, 2.0): 0.24466409297412722,
    (3.0, 3.0): 0.2848199127512167,
    (3.0, 4.0): 0.32746875872510617,
    (3.0, 5.0): 0.32042936473851963,
    (4.0, 2.0): 0.27174596394847605,
    (4.0, 3.0): 0.3503229202031343,
    (4.0, 4.0): 0.3296421409836883,
    (4.0, 5.0): 0.2940868008043224,
}
UNCHANGED_DDD = {
    "none": {
        (3.0, 2.0): 0.23861592308214064,
        (3.0, 3.0): 0.2569313134463256,
        (3.0, 4.0): 0.31485553592229476,
        (3.0, 5.0): 0.31078126391929545,
        (4.0, 2.0): 0.2558390475671242,
        (4.0, 3.0): 0.3241892351948469,
        (4.0, 4.0): 0.29590977121252926,
        (4.0, 5.0): 0.27887271825120685,
    },
    "weights": _DDD_WEIGHTED,
    "strata": _DDD_WEIGHTED,
}


def _lane_fits(kind, df):
    fit = _fit_cs if kind == "cs" else _fit_ddd
    return {
        "none": fit(df),
        "weights": fit(df, survey=SurveyDesign(weights="w")),
        "strata": fit(df, survey=SurveyDesign(weights="w", strata="stratum")),
    }


class TestUnchangedLanes:
    @pytest.mark.parametrize("design", ["none", "weights", "strata"])
    def test_cs(self, cs_panel, design):
        df, _ = cs_panel
        res = _lane_fits("cs", df)[design]
        cells = _estimable_cells(res)
        assert cells.keys() == UNCHANGED_CS[design].keys()
        for key, cell in cells.items():
            np.testing.assert_allclose(cell["se"], UNCHANGED_CS[design][key], rtol=1e-10)

    def test_cs_strata_only_is_weighted_sqrt_sum(self, cs_panel):
        df, _ = cs_panel
        res = _fit_cs(df, survey=SurveyDesign(weights="w", strata="stratum"))
        checked = 0
        for key, cell in _kit_cells(res):
            psi, _ = _cell_psi(res, key)
            np.testing.assert_allclose(cell["se"], np.sqrt(np.sum(psi**2)), rtol=1e-12)
            checked += 1
        assert checked >= 4

    @pytest.mark.parametrize("design", ["none", "weights", "strata"])
    def test_ddd(self, ddd_panel, design):
        df, _ = ddd_panel
        res = _lane_fits("ddd", df)[design]
        cells = _estimable_cells(res)
        assert cells.keys() == UNCHANGED_DDD[design].keys()
        for key, cell in cells.items():
            np.testing.assert_allclose(cell["se"], UNCHANGED_DDD[design][key], rtol=1e-10)

    def test_ddd_strata_only_single_gc_is_sqrt_sum_multi_gc_is_gmm(self, ddd_panel):
        df, _ = ddd_panel
        res = _fit_ddd(df, survey=SurveyDesign(weights="w", strata="stratum"))
        single = multi = 0
        for key, cell in _kit_cells(res):
            psi, _ = _cell_psi(res, key)
            sqrt_sum = np.sqrt(np.sum(psi**2))
            if res.comparison_group_counts[key] == 1:
                np.testing.assert_allclose(cell["se"], sqrt_sum, rtol=1e-12)
                single += 1
            else:
                # Multi-comparison-group cells keep R's GMM formula
                # sqrt(1 / (n * 1'Omega^-1 1)), which is NOT the IF sqrt-sum.
                assert not np.isclose(cell["se"], sqrt_sum, rtol=1e-6)
                multi += 1
        assert single >= 1 and multi >= 1


# ---------------------------------------------------------------------------
# 8. Helper unit tests
# ---------------------------------------------------------------------------


class TestCapReplicateDf:
    @pytest.mark.parametrize(
        "base, n_valid, expected",
        [
            (None, 20, 0),  # undefined design df (QR rank <= 1) -> NaN sentinel
            (0, 20, 0),  # an already-sentinelled base stays 0
            (19, 20, 19),  # cap slack: full replicate count
            (19, 7, 6),  # n_valid - 1 binds
            (4, 7, 4),  # design df binds (never raised to n_valid - 1)
            (19, 1, 0),
            (19, 0, 0),
            (19.0, 7, 6),
        ],
    )
    def test_table(self, base, n_valid, expected):
        out = cap_replicate_df(base, n_valid)
        assert out == expected and isinstance(out, int)


def _hand_design(kind):
    """10 units; cell payload on units 0-4, bystanders 5-9."""
    df = pd.DataFrame({"w": np.ones(10), "psu": np.repeat(np.arange(5), 2)})
    if kind == "psu":
        return SurveyDesign(weights="w", psu="psu").resolve(df)
    if kind == "weights":
        return SurveyDesign(weights="w").resolve(df)
    if kind == "replicate_bystander":
        # Four columns perturbing only bystander units: the cell's replicate
        # variance is exactly zero while the replicate matrix has rank > 1.
        for r in range(4):
            col = np.ones(10)
            col[5 + r] = 0.0
            df[f"r{r}"] = col
    else:
        for r in range(5):
            df[f"r{r}"] = np.where(df["psu"] == r, 0.0, 5.0 / 4.0)
    cols = [c for c in df.columns if c.startswith("r")]
    return SurveyDesign(weights="w", replicate_weights=cols, replicate_method="JK1").resolve(df)


def _payload(treated_inf=(0.3, -0.1), control_inf=(-0.2, 0.05, -0.05)):
    return {
        "treated_idx": np.array([0, 1]),
        "control_idx": np.array([2, 3, 4]),
        "treated_inf": np.array(treated_inf, dtype=float),
        "control_inf": np.array(control_inf, dtype=float),
    }


class TestDesignSeForCell:
    def test_no_design_passthrough(self):
        assert _design_se_for_cell(_payload(), None, 0.25, None) == (0.25, None)
        assert _design_se_for_cell(_payload(), _hand_design("weights"), 0.25, 9) == (0.25, 9)

    def test_psu_branch_uses_cr1_and_keeps_df(self):
        resolved = _hand_design("psu")
        se, df = _design_se_for_cell(_payload(), resolved, 0.25, 4)
        psi = np.zeros(10)
        psi[:5] = [0.3, -0.1, -0.2, 0.05, -0.05]
        np.testing.assert_allclose(se, np.sqrt(compute_survey_if_variance(psi, resolved)))
        assert df == 4 and se != 0.25

    def test_psu_malformed_payload_keeps_unit_se(self):
        resolved = _hand_design("psu")
        assert _design_se_for_cell({"treated_idx": np.array([0])}, resolved, 0.25, 4) == (0.25, 4)

    def test_replicate_branch(self):
        resolved = _hand_design("replicate")
        se, df = _design_se_for_cell(_payload(), resolved, 0.25, resolved.df_survey)
        psi = np.zeros(10)
        psi[:5] = [0.3, -0.1, -0.2, 0.05, -0.05]
        variance, n_valid = compute_replicate_if_variance(psi, resolved)
        np.testing.assert_allclose(se, np.sqrt(variance), rtol=1e-14)
        assert n_valid == 5 and df == min(resolved.df_survey, 4)

    def test_replicate_malformed_payload_fails_closed(self):
        resolved = _hand_design("replicate")
        se, df = _design_se_for_cell({"treated_idx": np.array([0])}, resolved, 0.25, 4)
        # Never the unit-level SE under a replicate design.
        assert np.isnan(se) and df == 0
        se, n_valid = _replicate_se_from_per_gt_if(None, resolved)
        assert np.isnan(se) and n_valid == 0

    def test_replicate_zero_variance_is_nan_not_zero(self):
        resolved = _hand_design("replicate_bystander")
        assert resolved.df_survey is not None and resolved.df_survey >= 1  # rank > 1
        psi = np.zeros(10)
        psi[:5] = [0.3, -0.1, -0.2, 0.05, -0.05]
        variance, n_valid = compute_replicate_if_variance(psi, resolved)
        assert variance == 0.0 and n_valid == 4
        se, df = _design_se_for_cell(_payload(), resolved, 0.25, resolved.df_survey)
        assert np.isnan(se)

    @pytest.mark.parametrize("kind", ["psu", "replicate"])
    def test_non_finite_influence_value_is_nan(self, kind):
        resolved = _hand_design(kind)
        bad = _payload(treated_inf=(np.nan, -0.1))
        se, _ = _design_se_for_cell(bad, resolved, 0.25, 4)
        # The PSU variance routine sums per PSU with NaN-skipping, so without
        # the guard this cell would report a FINITE clustered SE.
        assert np.isnan(se)

    def test_cluster_helper_finite_guard_direct(self):
        resolved = _hand_design("psu")
        assert np.isnan(
            _cluster_robust_se_from_per_gt_if(_payload(treated_inf=(np.inf, -0.1)), resolved)
        )
        # Contract unchanged for a finite payload and for a malformed one.
        assert np.isfinite(_cluster_robust_se_from_per_gt_if(_payload(), resolved))
        assert _cluster_robust_se_from_per_gt_if(None, resolved) is None
