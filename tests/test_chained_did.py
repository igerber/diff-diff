"""Behavioural tests for the native incomplete-panel estimator."""

import pickle

import numpy as np
import pandas as pd
import pytest

import diff_diff


def rotating_panel(windows=((0, 1), (1, 2), (2, 3))):
    """Known effects 3 and 7, with a non-zero earlier trend differential."""
    rows = []
    for cohort in (0, 2):
        for window in windows:
            for j, slope in enumerate((-1.5, -0.5, 0.5, 1.5)):
                uid = f"{cohort}-{window}-{j}"
                for t in window:
                    effect = {2: 3, 3: 7}.get(t, 0) if cohort else 0
                    y = 10 * bool(cohort) + j + 3 * t + slope * t
                    y += (2 * (t >= 1) + effect) if cohort else 0
                    rows.append((uid, t, cohort, y))
    data = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    for t in sorted(set(range(4)) - set(data.t)):
        data = pd.concat([data, data.iloc[[0]].assign(t=t, y=np.nan)], ignore_index=True)
    return data


def fit(data, **kwargs):
    assert hasattr(diff_diff, "ChainedDiD"), "ChainedDiD must be publicly exported"
    return diff_diff.ChainedDiD(**kwargs).fit(
        data, outcome="y", unit="id", time="t", first_treat="g"
    )


def test_rotating_windows_recover_long_effect_without_long_panel():
    data = rotating_panel()
    assert data.groupby("id").size().eq(2).all()
    result = fit(data)
    assert result.group_time_effects[2, 2].effect == pytest.approx(3)
    assert result.group_time_effects[2, 3].effect == pytest.approx(7)
    assert result.att == pytest.approx(5)
    assert result.group_time_effects[2, 2].se ** 2 == pytest.approx(0.625)
    assert result.group_time_effects[2, 3].se ** 2 == pytest.approx(1.25)
    assert result.n_units == data.id.nunique()


def test_earlier_differential_is_reported_directly_and_not_restricted():
    result = fit(rotating_panel(windows=((0, 1), (0, 3))))
    assert result.group_time_effects[2, 0].effect == pytest.approx(-2)
    assert result.group_time_effects[2, 3].effect == pytest.approx(7)
    assert result.group_time_effects[2, 2].skip_reason == "unsupported_target"
    assert np.isnan(result.att), "overall must not silently drop the missing horizon"


def test_nonadjacent_pair_retained_and_missing_calendar_endpoint_preserved():
    data = rotating_panel(windows=((1, 3),))
    result = fit(data)
    assert result.group_time_effects[2, 3].effect == pytest.approx(7)
    assert np.isnan(result.group_time_effects[2, 2].effect)
    assert result.diagnostics["catalogue"] == [(2, 1, 3)]


def test_disconnected_effects_are_unavailable_without_fabricated_coordinates():
    result = fit(rotating_panel(windows=((0, 1), (2, 3))))
    assert np.isnan(result.group_time_effects[2, 2].effect)
    assert np.isnan(result.group_time_effects[2, 3].effect)
    assert np.isnan(result.att)
    assert result.group_time_effects[2, 0].effect == pytest.approx(-2)


def test_row_order_and_arbitrary_unit_labels_do_not_change_estimates_or_covariance():
    data = rotating_panel()
    before = data.copy(deep=True)
    a = fit(data)
    b = fit(data.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(data, before)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=1e-12, equal_nan=True)
    assert a.att == pytest.approx(b.att)


@pytest.mark.parametrize("fault", ["duplicate", "infinite", "missing_id", "changing_cohort"])
def test_invalid_panel_rejected(fault):
    data = rotating_panel()
    if fault == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif fault == "infinite":
        data.loc[0, "y"] = np.inf
    elif fault == "missing_id":
        data.loc[0, "id"] = None
    else:
        data.loc[0, "g"] = 2
    with pytest.raises(ValueError):
        fit(data)


def test_parameter_contract_and_canonical_results():
    assert hasattr(diff_diff, "ChainedDiD"), "ChainedDiD must be publicly exported"
    estimator = diff_diff.ChainedDiD(alpha=0.1)
    assert estimator.get_params() == {"alpha": 0.1, "cluster": None}
    with pytest.raises(ValueError):
        estimator.set_params(alpha=2)
    assert estimator.alpha == 0.1
    result = estimator.fit(rotating_panel(), "y", "id", "t", "g")
    assert result.se > 0
    assert 0 <= result.p_value <= 1
    assert result.conf_int[0] < result.att < result.conf_int[1]
    assert "Chained" in result.summary()
    assert result.to_dict()["att"] == result.att
    restored = pickle.loads(pickle.dumps(result))
    assert restored.aggregate("simple").att[0] == pytest.approx(result.att)


def test_event_study_uses_canonical_container_with_joint_covariance():
    result = fit(rotating_panel())
    event = result.aggregate("event_study")
    assert isinstance(event, diff_diff.EventStudyResults)
    assert event.att[event.event_time == 0][0] == pytest.approx(3)
    assert event.att[event.event_time == 1][0] == pytest.approx(7)
    assert event.is_reference[event.event_time == -1][0]
    assert event.vcov is not None
    with pytest.raises(ValueError):
        result.aggregate("event_study", balance_e=1)
    with pytest.raises(ValueError):
        result.aggregate("simple", weights="equal")


def test_single_cluster_withholds_all_inference_but_preserves_estimate():
    data = rotating_panel().assign(cluster="only")
    result = fit(data, cluster="cluster")
    assert result.att == pytest.approx(5)
    for value in (result.se, result.t_stat, result.p_value, *result.conf_int):
        assert np.isnan(value)


@pytest.mark.parametrize("alpha", [True, None, "0.05", np.nan, np.inf, 0, 1, -0.1])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        diff_diff.ChainedDiD(alpha=alpha)


def test_failed_refit_clears_estimator_state_and_existing_results_are_detached():
    estimator = diff_diff.ChainedDiD()
    result = estimator.fit(rotating_panel(), "y", "id", "t", "g")
    estimator.set_params(alpha=0.1)
    assert result.alpha == 0.05
    assert result.aggregate("simple").att[0] == pytest.approx(5)
    with pytest.raises(ValueError):
        estimator.fit(rotating_panel(), "absent", "id", "t", "g")
    assert not estimator.is_fitted_
    assert estimator.results_ is None


def test_plotting_accepts_retained_canonical_event_study():
    import matplotlib.pyplot as plt

    result = fit(rotating_panel())
    ax = diff_diff.plot_event_study(result.aggregate("event_study"), show=False)
    assert ax is not None
    plt.close("all")


def test_irregular_numeric_calendar_preserves_actual_horizons():
    data = rotating_panel()
    labels = {0: 2000, 1: 2002, 2: 2005, 3: 2009}
    data["t"] = data.t.map(labels)
    data["g"] = data.g.map({0: 0, 2: 2005})
    result = fit(data)
    event = result.aggregate("event_study")
    assert event.event_time.tolist() == [-5, -3, 0, 4]
    assert event.reference_period == -3
    assert result.group_time_effects[2005, 2009].effect == pytest.approx(7)


def test_inconsistent_or_missing_cluster_metadata_rejected():
    data = rotating_panel().assign(cluster="a")
    data.loc[0, "cluster"] = "b"
    with pytest.raises(ValueError, match="constant"):
        fit(data, cluster="cluster")
    data.loc[0, "cluster"] = None
    with pytest.raises(ValueError, match="complete"):
        fit(data, cluster="cluster")


@pytest.mark.parametrize("column", ["g", "id", "x"])
def test_cluster_can_share_a_static_metadata_column(column):
    from tests.test_methodology_chained_did import binary_covariate_panel

    data, *_ = binary_covariate_panel()
    data["independent_cluster_name"] = data[column]
    a = diff_diff.ChainedDiD(cluster=column).fit(data, "y", "id", "t", "g", covariates=["x"])
    b = diff_diff.ChainedDiD(cluster="independent_cluster_name").fit(
        data, "y", "id", "t", "g", covariates=["x"]
    )
    np.testing.assert_allclose(a.vcov, b.vcov)


def test_empty_string_cluster_name_is_used_and_false_remains_explicitly_unclustered():
    data = rotating_panel()
    data[""] = np.arange(len(data)) // 8
    data["named"] = data[""]
    a, b = fit(data, cluster=""), fit(data, cluster="named")
    np.testing.assert_allclose(a.vcov, b.vcov)
    off = diff_diff.ChainedDiD(cluster=False)
    assert off.get_params()["cluster"] is False
    np.testing.assert_allclose(off.fit(data, "y", "id", "t", "g").vcov, fit(data).vcov)


@pytest.mark.parametrize("parameter,value", [("alpha", 2), ("alpha", True), ("cluster", True)])
def test_fit_revalidates_directly_mutated_parameters_and_clears_state(parameter, value):
    estimator = diff_diff.ChainedDiD()
    estimator.fit(rotating_panel(), "y", "id", "t", "g")
    setattr(estimator, parameter, value)
    with pytest.raises(ValueError, match=parameter):
        estimator.fit(rotating_panel(), "y", "id", "t", "g")
    assert not estimator.is_fitted_
    assert estimator.results_ is None


def test_simple_aggregate_reports_its_defined_weight_and_summary_refuses_relabelling():
    result = fit(rotating_panel())
    simple = result.aggregate("simple")
    np.testing.assert_array_equal(simple.weight, [1.0])
    with pytest.raises(ValueError, match="refit|re-fit") as exc:
        result.summary(alpha=0.1)
    assert "bootstrap" not in str(exc.value)


@pytest.mark.parametrize("arm", [0, 2])
def test_singleton_pair_is_recorded_and_cannot_distort_connected_post_effects(arm):
    data = rotating_panel()
    ids = data.loc[(data.g == arm) & (data.t == 0), "id"].unique()
    data = data.loc[~data.id.isin(ids[1:])].copy()
    with pytest.warns(UserWarning, match="singleton pairs excluded"):
        result = fit(data)
    assert (2, 0, 1) not in result.diagnostics["catalogue"]
    excluded = next(
        row for row in result.diagnostics["excluded_links"] if row["s"] == 0 and row["t"] == 1
    )
    assert excluded["reason"] == "insufficient_pair_replication"
    assert excluded["n_treated" if arm else "n_control"] == 1
    assert result.diagnostics["pair_support"]["minimum_per_arm"] == 2
    assert result.group_time_effects[2, 0].skip_reason == "unsupported_direct_pre_comparison"
    assert result.att == pytest.approx(5)
    changed = data.copy()
    changed.loc[changed.t == 0, "y"] += 1000
    with pytest.warns(UserWarning):
        other = fit(changed)
    assert other.att == pytest.approx(result.att)
    np.testing.assert_allclose(other.vcov, result.vcov, equal_nan=True)


def test_excluding_only_singleton_bridge_withholds_target_and_aggregate():
    data = rotating_panel(windows=((1, 3),))
    ids = data.loc[data.g == 2, "id"].unique()
    data = data.loc[~data.id.isin(ids[1:])]
    with pytest.warns(UserWarning, match="singleton pairs excluded"):
        result = fit(data)
    assert result.diagnostics["catalogue"] == []
    cell = result.group_time_effects[2, 3]
    assert cell.skip_reason == "unsupported_target"
    assert np.isnan(cell.effect) and np.isnan(cell.se)
    assert np.isnan(result.att) and np.isnan(result.se)
    assert result.n_units == 5


def test_low_count_warning_retains_replicated_pairs_with_diagnostics():
    with pytest.warns(UserWarning, match="retained pairs have low support"):
        result = fit(rotating_panel())
    assert len(result.diagnostics["catalogue"]) == 3
    assert all(row["low_support"] for row in result.diagnostics["links"])
    assert result.diagnostics["pair_support"]["warning_threshold"] == 20
    assert result.att == pytest.approx(5)


def test_control_effective_sample_size_warns_despite_large_raw_arms():
    x = np.r_[np.ones(95), np.zeros(5), np.ones(2), np.zeros(98)]
    g = np.r_[np.full(100, 2), np.zeros(100)]
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(200), 2),
            "t": np.tile([1, 2], 200),
            "g": np.repeat(g, 2),
            "x": np.repeat(x, 2),
        }
    )
    data["y"] = data.t * (data.x + (data.g == 2))
    with pytest.warns(UserWarning, match="retained pairs have low support"):
        result = diff_diff.ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])
    link = result.diagnostics["links"][0]
    assert link["n_treated"] == link["n_control"] == 100
    assert link["control_ess"] < 20
    assert link["low_support"]


def test_mixed_reference_event_row_uses_only_estimated_diagnostics_and_their_weights():
    rows = []
    for g, n in [(0, 30), (3, 20), (5, 10)]:
        for i in range(n):
            for t in (1, 3, 4, 5):
                y = t + i * t / n + (4 * (t >= 3) if g == 5 else 0)
                rows.append((f"{g}-{i}", t, g, y))
    result = fit(pd.DataFrame(rows, columns=["id", "t", "g", "y"]))
    event = result.aggregate("event_study")
    k = np.flatnonzero(event.event_time == -2)[0]
    diagnostic = result.group_time_effects[5, 3]
    assert event.att[k] == pytest.approx(diagnostic.effect)
    assert event.se[k] == pytest.approx(diagnostic.se)
    assert event.n[k] == 10
    assert not event.is_reference[k]
    assert event.reference_event_times == (-2, -1)
    assert event.reference_periods == [-1]
    weights = result.aggregation_weights("event_study").query("label == -2")
    assert weights[["group", "time"]].to_numpy().tolist() == [[5, 3]]
    np.testing.assert_allclose(weights.weight, [1])
    assert result.aggregate("event_study").att[k] == event.att[k]


def test_constituent_weights_explain_simple_and_event_aggregates_without_renormalising():
    result = fit(rotating_panel())
    weights = result.aggregation_weights()
    assert weights.label.tolist() == ["overall", "overall"]
    np.testing.assert_allclose(weights.weight, [0.5, 0.5])
    assert weights.available.all()
    assert sum(
        row.weight * result.group_time_effects[row.group, row.time].effect
        for row in weights.itertuples()
    ) == pytest.approx(result.att)
    weights.loc[:, "weight"] = 0
    np.testing.assert_allclose(result.aggregation_weights().weight, [0.5, 0.5])
    unavailable = fit(rotating_panel(windows=((1, 3),))).aggregation_weights()
    np.testing.assert_allclose(unavailable.weight, [0.5, 0.5])
    assert unavailable.available.tolist() == [False, True]
    with pytest.raises(ValueError, match="simple.*event_study"):
        result.aggregation_weights("group")
