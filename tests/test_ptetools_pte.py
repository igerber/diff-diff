import matplotlib
import numpy as np

matplotlib.use("Agg")

from diff_diff import autoplot_pte_results, did_attgt, ggpte, pte, two_by_two_subset


def _panel():
    import pandas as pd

    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(4), 3),
            "period": np.tile([1, 2, 3], 4),
            "G": np.repeat([0, 0, 2, 3], 3),
            "Y": [0, 1, 2, 0, 0, 1, 0, 2, 4, 0, 0, 3],
        }
    )


def test_pte_runs_group_time_loop_and_returns_results():
    result = pte(_panel(), yname="Y", gname="G", tname="period", idname="id")
    assert set(result.att_gt.columns) == {"group", "time", "attgt", "se"}
    assert len(result.att_gt) == 4
    assert np.isfinite(result.overall_att)
    assert (
        autoplot_pte_results(result, show=False).get_title() == "Treatment Effects Over Event Time"
    )
    assert result.to_dataframe().equals(result.att_gt)


def test_pte_aggregate_exposes_inference_and_level_tables():
    result = pte(_panel(), yname="Y", gname="G", tname="period", idname="id")
    dynamic = result.aggregate("dynamic")

    assert np.isfinite(dynamic.standard_error)
    assert dynamic.conf_int[0] <= dynamic.estimate <= dynamic.conf_int[1]
    dynamic_table = result.to_dataframe("dynamic")
    assert {"event_time", "estimate", "se", "conf_int_lower", "conf_int_upper"}.issubset(
        dynamic_table.columns
    )
    assert dynamic_table["event_time"].is_unique


def test_pte_dynamic_aggregate_multiplier_bootstrap_bands():
    result = pte(_panel(), yname="Y", gname="G", tname="period", idname="id")
    aggregate = result.aggregate("dynamic", bstrap=True, biters=40, seed=11)
    table = aggregate.to_dataframe()

    assert aggregate.bootstrap_distribution is not None
    assert aggregate.bootstrap_distribution.shape == (40, len(table))
    assert {"lower_pw", "upper_pw", "lower_ub", "upper_ub"}.issubset(table.columns)


def test_ggpte_adapts_dynamic_results_to_event_study_plot():
    result = pte(_panel(), yname="Y", gname="G", tname="period", idname="id")
    ax = ggpte(result, show=False)

    assert ax.get_title() == "Treatment Effects Over Event Time"
    assert ax.get_xlabel() == "Period Relative to Treatment"
    tick_labels = {label.get_text() for label in ax.get_xticklabels()}
    assert {"-1", "0", "1"}.issubset(tick_labels)


def test_pte_accepts_pre_period_covariates():
    panel = _panel()
    panel["Z"] = np.repeat([0.0, 1.0, 0.5, 1.5], 3)
    result = pte(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        covariates=["Z"],
    )
    assert np.isfinite(result.att_gt["attgt"].dropna()).all()


def test_pte_accepts_custom_subset_and_attgt_callbacks():
    def subset_fun(data, group, time):
        return two_by_two_subset(data, group, time)

    def attgt_fun(gt_data):
        result = did_attgt(gt_data)
        return {"attgt": result.attgt, "inf_func": result.inf_func}

    result = pte(
        _panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        subset_fun=subset_fun,
        attgt_fun=attgt_fun,
    )
    assert len(result.att_gt) == 4
    assert np.isfinite(result.overall_att)


def test_pte_empirical_bootstrap_is_seed_reproducible():
    kwargs = {
        "yname": "Y",
        "gname": "G",
        "tname": "period",
        "idname": "id",
        "bstrap": True,
        "biters": 9,
        "seed": 42,
    }
    first = pte(_panel(), **kwargs)
    second = pte(_panel(), **kwargs)
    assert np.isfinite(first.overall_se)
    assert np.isclose(first.overall_se, second.overall_se)
    assert first.bootstrap_distribution is not None
    assert len(first.bootstrap_distribution) == 9
    assert "overall_att" in first.to_dict()
    assert first.overall_conf_int[0] <= first.overall_conf_int[1]
    assert "PTEResults" in first.summary()


def test_pte_influence_surface_zero_pads_off_support_and_scales_by_n_over_n1():
    """Mirror R's compute.pte influence surface (pte.R:137-141): off-support
    units get a zero influence entry and the cell influence function is scaled
    by (n / n1) to account for the overall-vs-cell sample sizes."""
    panel = _panel()
    n_units = panel["id"].nunique()
    result = pte(panel, yname="Y", gname="G", tname="period", idname="id")

    influence = result.influence_functions
    assert influence is not None
    assert influence.shape == (n_units, len(result.att_gt))
    assert not np.isnan(influence).any(), "off-support entries must be zero, not NaN"

    for row, gtp in result.att_gt.iterrows():
        subset = two_by_two_subset(
            panel, gtp["group"], gtp["time"], gname="G", tname="period", idname="id"
        )
        if subset.n1 == 0:
            continue
        cell_att = did_attgt(subset.gt_data)
        placed = influence[subset.disidx, row]
        expected = (n_units / subset.n1) * np.asarray(cell_att.inf_func, dtype=float)
        assert np.allclose(placed, expected)
        assert np.all(influence[~subset.disidx, row] == 0.0)


def test_pte_supports_repeated_cross_sections():
    import pandas as pd

    data = pd.DataFrame(
        {
            "period": [1, 1, 2, 2],
            "G": [0, 2, 0, 2],
            "Y": [0.0, 1.0, 1.0, 4.0],
        }
    )
    result = pte(data, yname="Y", gname="G", tname="period", panel=False)
    assert len(result.att_gt) == 1
    assert np.isclose(result.overall_att, 2.0)


def test_rcs_pte_bootstrap_is_reproducible():
    import pandas as pd

    data = pd.DataFrame(
        {
            "period": [1, 1, 2, 2],
            "G": [0, 2, 0, 2],
            "Y": [0.0, 1.0, 1.0, 4.0],
        }
    )
    first = pte(
        data, yname="Y", gname="G", tname="period", panel=False, bstrap=True, biters=5, seed=3
    )
    second = pte(
        data, yname="Y", gname="G", tname="period", panel=False, bstrap=True, biters=5, seed=3
    )
    assert np.isfinite(first.overall_se)
    assert np.isclose(first.overall_se, second.overall_se)
