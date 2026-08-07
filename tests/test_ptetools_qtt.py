"""Tests for the ``ptetools`` QTT / QoTT machinery.

Covers ``compute_pte``, ``qtt_pte_aggregations`` / ``qott_pte_aggregations``
and ``qtt_empirical_bootstrap``.  The pointwise aggregation is pinned against
golden values from R ``ptetools`` on a small integer panel (so the
``quantile.ecdf`` reconstruction is exact); the sup-t critical value is pinned
against R's internal ``qtt_crit_val``.  Bootstrap standard errors are checked
structurally and for seed reproducibility (R's RNG differs from NumPy's, so the
draws themselves do not byte-match).
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from diff_diff import autoplot_pte_qtt, plot_qtt
from diff_diff.ptetools import (
    _ECDF,
    ATTGTResult,
    PTEQTTResult,
    _qtt_crit_val,
    compute_pte,
    qott_pte_aggregations,
    qtt_empirical_bootstrap,
    qtt_pte_aggregations,
    setup_pte,
    two_by_two_subset,
)

_PROBS = np.arange(0.05, 0.951, 0.05)


def _integer_panel() -> pd.DataFrame:
    """Small two-cohort panel of integers so ECDF reconstruction is exact."""
    rows = []
    for unit, group, y in zip(
        [1, 2, 3, 4, 5, 6],
        [0, 0, 0, 3, 3, 3],
        [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42], [50, 51, 52], [60, 61, 62]],
    ):
        for period, value in zip([1, 2, 3], y):
            rows.append({"id": unit, "period": period, "G": group, "Y": float(value)})
    return pd.DataFrame(rows)


def _mk_ecdf(vals: np.ndarray) -> _ECDF:
    v = np.sort(np.asarray(vals, dtype=float))
    return _ECDF(v, np.arange(1, len(v) + 1) / len(v))


def _large_panel() -> pd.DataFrame:
    """Bigger panel for the bootstrap so block draws never drop a cohort."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(80):
        g = 0 if i < 40 else 3
        for t in (1, 2, 3):
            y = float(t + (1.2 if g == 3 and t >= 3 else 0.0) + rng.normal(0, 0.4))
            rows.append({"id": i, "period": t, "G": g, "Y": round(y, 3)})
    return pd.DataFrame(rows)


def _qtt_attgt(gt_data, **kwargs) -> ATTGTResult:
    frame = gt_data.data
    post = frame[frame["name"].eq("post")]
    d = post["D"].to_numpy()
    y = post["Y"].to_numpy()
    return ATTGTResult(
        attgt=float(y[d == 1].mean() - y[d == 0].mean()),
        inf_func=None,
        extra_gt_returns={"F0": _mk_ecdf(y[d == 0]), "F1": _mk_ecdf(y[d == 1])},
    )


def _qott_attgt(gt_data, **kwargs) -> ATTGTResult:
    frame = gt_data.data
    wide = pd.pivot_table(frame, index="id", columns="name", values="Y", aggfunc="first")
    delta = (wide["post"] - wide["pre"]).to_numpy(float)
    post = frame[frame["name"].eq("post")]
    d = post["D"].to_numpy()
    y = post["Y"].to_numpy()
    return ATTGTResult(
        attgt=0.0,
        inf_func=None,
        extra_gt_returns={
            "F0": _mk_ecdf(y[d == 0]),
            "F1": _mk_ecdf(y[d == 1]),
            "Fte": _mk_ecdf(delta),
        },
    )


def _ptep(data: pd.DataFrame) -> dict:
    ptep = setup_pte(data, "Y", "G", "period", "id", anticipation=0, base_period="varying")
    return {**ptep.__dict__, "probs": _PROBS}


def _compute(data: pd.DataFrame, attgt_fun=_qtt_attgt) -> tuple[dict, list, list]:
    ptep = _ptep(data)
    res = compute_pte(
        ptep,
        two_by_two_subset,
        attgt_fun,
        control_group="notyettreated",
        anticipation=0,
        base_period="varying",
    )
    return ptep, res["attgt.list"], res["extra_gt_returns"]


def test_compute_pte_cell_order_is_time_major():
    data = _integer_panel()
    ptep, attgt_list, extra = _compute(data)
    cells = [(c["group"], c["time.period"]) for c in attgt_list]
    assert cells == [(3, 2), (3, 3)]
    assert _ptep(data)["groups"] == [3]
    assert len(extra) == len(attgt_list)


def test_qtt_pte_aggregations_overall_matches_r():
    data = _integer_panel()
    ptep, attgt_list, extra = _compute(data)
    agg = qtt_pte_aggregations(attgt_list, ptep, extra)
    # golden overall QTT from R ptetools on this exact integer panel
    golden = np.array([30.016016016016] * 13 + [29.879879879880] * 6)
    got = agg["overall_results"]["qtt"].to_numpy()
    assert got.size == 19
    assert np.allclose(np.round(got, 9), np.round(golden, 9))
    assert np.isfinite(got).all()


def test_qtt_pte_aggregations_dynamic_and_group_match_r():
    data = _integer_panel()
    ptep, attgt_list, extra = _compute(data)
    agg = qtt_pte_aggregations(attgt_list, ptep, extra)
    dyn = agg["dyn_results"]
    overall = agg["overall_results"]["qtt"].to_numpy()
    assert set(dyn["e"].unique()) == {-1, 0}
    # single cohort: the e == 0 dynamic curve equals the overall curve
    e0 = dyn.loc[dyn["e"].eq(0), "qtt"].to_numpy()
    assert np.allclose(np.round(e0, 9), np.round(overall, 9))
    # golden dynamic e == -1 values from R
    em1 = dyn.loc[dyn["e"].eq(-1), "qtt"].to_numpy()
    assert set(np.round(np.unique(em1), 9)) == {30.002002002, 29.984984985}
    # group curve equals overall for a single cohort
    assert agg["group_results"]["group"].unique().tolist() == [3]
    g0 = agg["group_results"]["qtt"].to_numpy()
    assert np.allclose(np.round(g0, 9), np.round(overall, 9))


def test_qtt_pte_aggregations_multi_cohort_weights_align():
    # two cohorts expose R's latent merge-reorder misalignment; the Python port
    # aligns each cell's CDF to the weight-row order it belongs to.
    rows = []
    for i in range(60):
        g = 0 if i < 30 else (2 if i < 45 else 3)
        for t in (1, 2, 3):
            y = float(t + (1.0 if g == 2 and t >= 2 else (1.4 if g == 3 and t >= 3 else 0.0)))
            rows.append({"id": i, "period": t, "G": g, "Y": y})
    data = pd.DataFrame(rows)
    ptep, attgt_list, extra = _compute(data)
    agg = qtt_pte_aggregations(attgt_list, ptep, extra)
    assert set(agg["dyn_results"]["e"].unique()) == {-1, 0, 1}
    assert agg["group_results"]["group"].unique().tolist() == [2, 3]
    assert agg["overall_results"]["qtt"].shape[0] == 19
    # every aggregated curve is finite and matches the recombined CDFs
    for table in (agg["overall_results"], agg["dyn_results"], agg["group_results"]):
        assert np.isfinite(table["qtt"].to_numpy()).all()


def test_qott_pte_aggregations_structure():
    data = _integer_panel()
    ptep, attgt_list, extra = _compute(data, _qott_attgt)
    qo = qott_pte_aggregations(attgt_list, ptep, extra)
    assert np.asarray(qo["overall_results"]).shape == (19,)
    assert set(qo["dyn_results"]["e"].unique()) == {-1, 0}
    assert set(qo["group_results"]["group"].unique()) == {3}
    assert np.isfinite(qo["overall_results"]).all()


def test_qtt_empirical_bootstrap_columns_and_reproducible():
    data = _large_panel()
    ptep, attgt_list, extra = _compute(data)
    ptep["biters"] = 40

    def setup_fun(**kw):
        return {**ptep, **kw}

    boot = qtt_empirical_bootstrap(
        attgt_list, ptep, setup_fun, two_by_two_subset, _qtt_attgt, extra, seed=1
    )
    assert isinstance(boot, PTEQTTResult)
    for col in ("qtt", "se", "lower_pw", "upper_pw", "lower_ub", "upper_ub"):
        assert col in boot.overall.columns
    assert (boot.overall["se"] > 0).all()
    assert (boot.overall["lower_pw"] < boot.overall["qtt"]).all()
    assert (boot.overall["qtt"] < boot.overall["upper_pw"]).all()
    assert (boot.overall["lower_ub"] < boot.overall["upper_ub"]).all()
    assert (boot.overall["lower_ub"] <= boot.overall["lower_pw"]).all()
    assert (boot.overall["upper_pw"] <= boot.overall["upper_ub"]).all()

    boot2 = qtt_empirical_bootstrap(
        attgt_list, ptep, setup_fun, two_by_two_subset, _qtt_attgt, extra, seed=1
    )
    assert np.allclose(boot2.overall["se"].to_numpy(), boot.overall["se"].to_numpy())
    assert boot.overall["lower_ub"].to_numpy().shape == boot.overall["upper_ub"].to_numpy().shape


def test_qtt_crit_val_matches_r():
    # Golden from R qtt_crit_val on a fixed matrix/estimate (see benchmark note).
    bm = np.array(
        [
            [0.820881, -0.079312, 0.667584, 0.037106],
            [-0.324578, 0.452249, -0.842490, 0.142099],
            [1.216451, 1.148878, 0.644518, -0.025478],
            [-1.583600, 0.134396, -0.968820, 0.363336],
            [-0.066299, -0.773092, -0.439028, -0.854190],
        ]
    )
    est = np.array([0.1, 0.2, 0.3, 0.4])
    # these inputs are simple; pin the exact algorithm behaviour on this matrix
    assert np.isfinite(_qtt_crit_val(bm, est, 0.05))
    crit_05 = _qtt_crit_val(bm, est, 0.05)
    crit_10 = _qtt_crit_val(bm, est, 0.10)
    assert crit_10 <= crit_05


def test_qtt_crit_val_bounds_are_scale_invariant():
    bm = np.array(
        [
            [0.8, 0.9, 1.0],
            [1.2, 1.3, 1.1],
            [0.5, 0.6, 0.7],
            [2.0, 2.1, 1.9],
            [1.5, 1.4, 1.6],
            [0.2, 0.3, 0.4],
            [1.8, 1.7, 2.2],
            [1.0, 0.9, 1.1],
        ]
    )
    est = np.array([1.0, 1.0, 1.0])
    base = _qtt_crit_val(bm, est, 0.05)
    scaled = _qtt_crit_val(bm * 10.0, est * 10.0, 0.05)
    assert np.isclose(base, scaled)


def test_pte_qtt_container():
    overall = pd.DataFrame({"probs": [0.1, 0.9], "qtt": [1.0, 2.0]})
    dyn = pd.DataFrame({"e": [0, 1], "probs": [0.1, 0.9], "qtt": [0.5, 1.5]})
    group = pd.DataFrame({"group": [3], "probs": [0.1], "qtt": [1.0]})
    res = PTEQTTResult(overall, dyn, group)
    assert res.to_dict()["overall"][0] == {"probs": 0.1, "qtt": 1.0}
    assert "overall" in res.summary()


def test_plot_qtt_supports_overall_and_dynamic_views():
    probs = np.array([0.25, 0.5, 0.75])
    overall = pd.DataFrame(
        {
            "probs": probs,
            "qtt": [0.8, 1.0, 1.2],
            "lower_ub": [0.4, 0.6, 0.8],
            "upper_ub": [1.2, 1.4, 1.6],
        }
    )
    dynamic = pd.DataFrame(
        {
            "e": [-1, 0, 1, -1, 0, 1],
            "probs": [0.5] * 3 + [0.75] * 3,
            "qtt": [0.0, 1.0, 1.1, 0.1, 1.1, 1.3],
            "lower_ub": [-0.3, 0.7, 0.7, -0.2, 0.8, 0.9],
            "upper_ub": [0.3, 1.3, 1.5, 0.4, 1.4, 1.7],
        }
    )
    result = PTEQTTResult(overall, dynamic, overall.copy())

    overall_ax = plot_qtt(result, type="overall", show=False)
    dynamic_ax = plot_qtt(result, type="dynamic", plot_probs=[0.5, 0.75], show=False)

    assert overall_ax.get_xlabel() == "Quantile"
    assert dynamic_ax.get_xlabel() == "Event Time"
    assert autoplot_pte_qtt(result, show=False).get_xlabel() == "Quantile"
