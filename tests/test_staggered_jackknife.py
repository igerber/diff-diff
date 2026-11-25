"""Cluster jackknife (CV3) inference for CallawaySantAnna.

Karim, Nielsen, MacKinnon & Webb (2026), "Improved Inference for CSDID Using
the Cluster Jackknife" (arXiv:2602.12043), equation (20). The reference for
every number here is a delete-one loop written out by hand: the same
estimator, ``vcov_type="hc1"``, the same ``cluster=``, refitted on each
delete-one frame.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import (
    CallawaySantAnna,
    DiagnosticReport,
    DMLDiD,
    HonestDiD,
    SurveyDesign,
    diagnose_design,
)
from tests.conftest import assert_nan_inference

FIT_KW = {"outcome": "y", "unit": "unit", "time": "t", "first_treat": "g"}
GOLDEN = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "cs_jackknife_golden.json"


def make_panel(cohorts, n_clusters=10, units_per=2, n_periods=8, seed=0, effect=0.7):
    """A panel of ``n_clusters`` clusters; ``cohorts`` maps cluster -> first period."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        shock = rng.normal(size=n_periods + 1).cumsum() * 0.3
        g = cohorts.get(c, 0)
        for j in range(units_per):
            u = c * units_per + j
            a = rng.normal()
            for t in range(1, n_periods + 1):
                y = a + shock[t] + rng.normal() + (effect if g and t >= g else 0.0)
                rows.append((u, f"s{c}", t, g, y, rng.normal()))
    return pd.DataFrame(rows, columns=["unit", "state", "t", "g", "y", "x"])


TWO_COHORTS = {0: 4, 1: 4, 2: 6, 3: 6}


def quiet_fit(estimator, data, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(data, **{**FIT_KW, **overrides})


def eq20(full, reps):
    reps = np.asarray(reps, dtype=float)
    h = len(reps)
    return np.sqrt((h - 1) / h * np.sum((reps - full) ** 2, axis=0))


def by_hand(data, column, labels, fit_overrides=None, **params):
    """Full-sample ``hc1`` fit and the delete-one refits, in ``labels`` order."""
    fit_overrides = fit_overrides or {}
    full = quiet_fit(CallawaySantAnna(**params), data, **fit_overrides)
    reps = [
        quiet_fit(CallawaySantAnna(**params), data[data[column] != label], **fit_overrides)
        for label in labels
    ]
    return full, reps


def es_frame(results, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return results.aggregate("event_study", **kwargs).to_dataframe().set_index("event_time")


def group_frame(results):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return results.aggregate("group").to_dataframe().set_index("label")


@pytest.fixture(scope="module")
def panel():
    return make_panel(TWO_COHORTS, seed=11)


@pytest.fixture(scope="module")
def clustered(panel):
    """Jackknife fit clustered on ``state`` with its by-hand reference."""
    jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
    full, reps = by_hand(panel, "state", list(panel["state"].unique()), cluster="state")
    return jk, full, reps


class TestEquation20:
    """The jackknife equals the by-hand delete-one loop, for every quantity."""

    def test_overall(self, clustered):
        jk, full, reps = clustered
        assert jk.att == full.att
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)

    def test_every_cell(self, clustered):
        jk, full, reps = clustered
        for gt, cell in jk.group_time_effects.items():
            expected = eq20(
                full.group_time_effects[gt]["effect"],
                [r.group_time_effects[gt]["effect"] for r in reps],
            )
            np.testing.assert_allclose(cell["se"], expected, rtol=1e-12, err_msg=str(gt))

    @pytest.mark.parametrize("balance_e", [None, 1])
    def test_event_study(self, clustered, balance_e):
        jk, full, reps = clustered
        got = es_frame(jk, balance_e=balance_e)
        ref = es_frame(full, balance_e=balance_e)
        rep_att = np.array(
            [es_frame(r, balance_e=balance_e)["att"].reindex(got.index).to_numpy() for r in reps]
        )
        np.testing.assert_allclose(got["att"], ref["att"], rtol=1e-14)
        np.testing.assert_allclose(got["se"], eq20(ref["att"].to_numpy(), rep_att), rtol=1e-12)

    def test_group(self, clustered):
        jk, full, reps = clustered
        got = group_frame(jk)
        ref = group_frame(full)
        rep_att = np.array([group_frame(r)["att"].reindex(got.index).to_numpy() for r in reps])
        np.testing.assert_allclose(got["se"], eq20(ref["att"].to_numpy(), rep_att), rtol=1e-12)

    @pytest.mark.parametrize("cluster", [None, False])
    def test_unit_level(self, panel, cluster):
        """No cluster column (or an explicit ``False``): the unit is deleted."""
        jk = quiet_fit(CallawaySantAnna(cluster=cluster, vcov_type="jackknife"), panel)
        full, reps = by_hand(panel, "unit", sorted(panel["unit"].unique()))
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        assert jk.df_inference == panel["unit"].nunique() - 1
        got = es_frame(jk)
        rep_att = np.array([es_frame(r)["att"].reindex(got.index).to_numpy() for r in reps])
        np.testing.assert_allclose(
            got["se"], eq20(es_frame(full)["att"].to_numpy(), rep_att), rtol=1e-12
        )

    def test_t_reference_with_h_minus_one_df(self, clustered):
        jk, _, reps = clustered
        h = len(reps)
        t_stat = jk.att / jk.se
        crit = stats.t.ppf(0.975, h - 1)
        np.testing.assert_allclose(jk.t_stat, t_stat, rtol=1e-12)
        np.testing.assert_allclose(jk.p_value, 2 * stats.t.sf(abs(t_stat), h - 1), rtol=1e-10)
        np.testing.assert_allclose(
            jk.conf_int, (jk.att - crit * jk.se, jk.att + crit * jk.se), rtol=1e-10
        )
        cell = next(iter(jk.group_time_effects.values()))
        np.testing.assert_allclose(
            cell["p_value"], 2 * stats.t.sf(abs(cell["effect"] / cell["se"]), h - 1), rtol=1e-10
        )
        es = jk.aggregate("event_study")
        np.testing.assert_allclose(
            es.conf_int_upper - es.att, crit * es.se, rtol=1e-10, equal_nan=True
        )


class TestConfigurations:
    """Full re-estimation: every point-estimate configuration is covered."""

    @pytest.mark.parametrize(
        "params, fit_overrides",
        [
            ({"estimation_method": "reg"}, {"covariates": ["x"]}),
            ({"estimation_method": "dr"}, {"covariates": ["x"]}),
            ({"estimation_method": "ipw"}, {"covariates": ["x"]}),
            ({"control_group": "not_yet_treated"}, {}),
            ({"anticipation": 1}, {}),
            ({"base_period": "universal"}, {}),
        ],
    )
    def test_matches_by_hand(self, panel, params, fit_overrides):
        jk = quiet_fit(
            CallawaySantAnna(cluster="state", vcov_type="jackknife", **params),
            panel,
            **fit_overrides,
        )
        full, reps = by_hand(
            panel, "state", list(panel["state"].unique()), fit_overrides, cluster="state", **params
        )
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        got = es_frame(jk)
        rep_att = np.array([es_frame(r)["att"].reindex(got.index).to_numpy() for r in reps])
        live = ~got["is_reference"].to_numpy()
        np.testing.assert_allclose(
            got["se"].to_numpy()[live],
            eq20(es_frame(full)["att"].to_numpy(), rep_att)[live],
            rtol=1e-12,
        )

    def test_repeated_cross_sections(self):
        """``panel=False``: the cluster column defines the deletion units."""
        rng = np.random.default_rng(5)
        rows = []
        uid = 0
        for c in range(10):
            g = TWO_COHORTS.get(c, 0)
            level = rng.normal()
            for t in range(1, 7):
                for _ in range(6):
                    y = level + 0.1 * t + rng.normal() + (0.6 if g and t >= g else 0.0)
                    rows.append((uid, f"s{c}", t, g, y))
                    uid += 1
        data = pd.DataFrame(rows, columns=["unit", "state", "t", "g", "y"])
        jk = quiet_fit(CallawaySantAnna(panel=False, cluster="state", vcov_type="jackknife"), data)
        full, reps = by_hand(
            data, "state", list(data["state"].unique()), panel=False, cluster="state"
        )
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        assert jk.df_inference == 9
        assert jk.get_loo_effects_df()["cluster"].map(type).eq(str).all()

    @pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
    def test_repeated_cross_sections_with_covariates(self, method):
        rng = np.random.default_rng(6)
        rows = []
        uid = 0
        for c in range(12):
            g = TWO_COHORTS.get(c, 0)
            level = rng.normal()
            for t in range(1, 7):
                for _ in range(8):
                    x = rng.normal() + 0.3 * (g > 0)
                    y = level + 0.1 * t + 0.5 * x + rng.normal() + (0.6 if g and t >= g else 0.0)
                    rows.append((uid, f"s{c}", t, g, y, x))
                    uid += 1
        data = pd.DataFrame(rows, columns=["unit", "state", "t", "g", "y", "x"])
        params = dict(panel=False, cluster="state", estimation_method=method)
        overrides = {"covariates": ["x"]}
        jk = quiet_fit(CallawaySantAnna(vcov_type="jackknife", **params), data, **overrides)
        full, reps = by_hand(data, "state", list(data["state"].unique()), overrides, **params)
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        for gt, cell in jk.group_time_effects.items():
            expected = eq20(
                full.group_time_effects[gt]["effect"],
                [r.group_time_effects[gt]["effect"] for r in reps],
            )
            np.testing.assert_allclose(cell["se"], expected, rtol=1e-12, err_msg=str(gt))
        got = es_frame(jk)
        rep_att = np.array([es_frame(r)["att"].reindex(got.index).to_numpy() for r in reps])
        np.testing.assert_allclose(
            got["se"], eq20(es_frame(full)["att"].to_numpy(), rep_att), rtol=1e-12
        )
        groups = group_frame(jk)
        rep_g = np.array([group_frame(r)["att"].reindex(groups.index).to_numpy() for r in reps])
        np.testing.assert_allclose(
            groups["se"], eq20(group_frame(full)["att"].to_numpy(), rep_g), rtol=1e-12
        )


class TestWeightingRoute:
    """A refit must keep ``cluster=``: it selects the aggregation weights."""

    def test_refits_keep_the_cluster_route(self):
        # Unequal cohort sizes plus missing treated outcomes: with a cluster
        # column the aggregation weights are full cohort masses, without one
        # they are each cell's complete-case count, and the two disagree.
        data = make_panel({0: 4, 1: 4, 2: 4, 3: 6}, seed=21)
        holes = (data["state"].isin(["s0", "s1"])) & (data["t"] >= 7)
        data = data.loc[~holes].reset_index(drop=True)
        labels = list(data["state"].unique())
        jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), data)
        full, reps = by_hand(data, "state", labels, cluster="state")
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)

        full_none, reps_none = by_hand(data, "state", labels)
        wrong = eq20(full_none.att, [r.att for r in reps_none])
        assert abs(full_none.att - full.att) > 1e-6, "panel no longer separates the two routes"
        assert abs(wrong - jk.se) > 1e-6


class TestEstimationSample:
    """``H`` counts the clusters that enter the estimation, nothing else."""

    def test_rows_with_a_missing_unit_id_form_no_cluster(self, panel):
        extra = panel.head(3).copy()
        extra["unit"] = np.nan
        extra["state"] = "ghost"
        dirty = pd.concat([panel.astype({"unit": float}), extra], ignore_index=True)
        clean = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
        got = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), dirty)
        np.testing.assert_allclose(got.se, clean.se, rtol=1e-12)
        assert got.df_inference == clean.df_inference == 9
        assert got.n_clusters == clean.n_clusters == 10
        assert "ghost" not in set(got.get_loo_effects_df()["cluster"])

    def test_always_treated_cluster_is_not_a_deletion_unit(self):
        data = make_panel({**TWO_COHORTS, 9: 1}, seed=13)
        jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), data)
        labels = [s for s in data["state"].unique() if s != "s9"]
        full, reps = by_hand(data, "state", labels, cluster="state")
        # Deleting the always-treated cluster changes nothing, so it is no replicate.
        dropped = quiet_fit(CallawaySantAnna(cluster="state"), data[data["state"] != "s9"])
        assert dropped.att == full.att
        assert jk.df_inference == 8
        assert jk.n_clusters == 9
        assert "s9" not in set(jk.get_loo_effects_df()["cluster"])
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)

    def test_cluster_that_only_carries_cohort_mass_is_a_deletion_unit(self):
        # With a cluster column the aggregations weight by whole-cohort
        # masses. Cluster s10 belongs to cohort 2 and has no outcome at all,
        # so it enters no cell, yet deleting it changes the weights: the
        # overall ATT moves from 17/11 to 5/3.
        rows = []
        for c, g in enumerate([2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 2]):
            for j in range(2):
                for t in (1, 2, 3):
                    effect = 1.0 if (g == 2 and t >= 2) else 3.0 if (g == 3 and t >= 3) else 0.0
                    rows.append((c * 2 + j, f"s{c:02d}", t, g, np.nan if c == 10 else effect))
        data = pd.DataFrame(rows, columns=["unit", "state", "t", "g", "y"])
        labels = list(data["state"].unique())
        full, reps = by_hand(data, "state", labels, cluster="state")
        np.testing.assert_allclose(full.att, 17 / 11, rtol=1e-12)
        np.testing.assert_allclose(reps[-1].att, 5 / 3, rtol=1e-12)
        jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), data)
        assert jk.df_inference == 10 and jk.n_clusters == 11
        assert set(jk.get_loo_effects_df()["cluster"]) == set(labels)
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        got = es_frame(jk)
        rep_att = np.array([es_frame(r)["att"].reindex(got.index).to_numpy() for r in reps])
        np.testing.assert_allclose(
            got["se"], eq20(es_frame(full)["att"].to_numpy(), rep_att), rtol=1e-12, atol=1e-15
        )
        # Without a cluster column cells are weighted by their own counts,
        # the outcome-less units enter nothing, and they are not deleted.
        unit_jk = quiet_fit(CallawaySantAnna(vcov_type="jackknife"), data)
        assert unit_jk.df_inference == 20 - 1

    def test_reference_cell_mass_makes_a_cohort_unit_a_deletion_unit(self):
        # Universal base on a gapped grid: horizon -2 mixes cohort 3's
        # estimate with cohort 5's reference cell, weighted by cohort 5's
        # whole mass. A cohort-5 unit with no outcome enters no cell but is
        # in that mass, so deleting it moves the horizon's estimate.
        data = make_panel({0: 3, 1: 3, 2: 5, 3: 5}, n_clusters=9, n_periods=7, seed=43)
        data = data[data["t"] != 4].reset_index(drop=True)
        ghost = data["unit"] == data.loc[data["state"] == "s3", "unit"].iloc[0]
        data.loc[ghost, "y"] = np.nan
        units = sorted(data["unit"].unique())
        ghost_unit = int(data.loc[ghost, "unit"].iloc[0])
        full, reps = by_hand(data, "unit", units, base_period="universal")
        dropped = reps[units.index(ghost_unit)]
        assert abs(es_frame(dropped).loc[-2, "att"] - es_frame(full).loc[-2, "att"]) > 1e-8
        jk = quiet_fit(CallawaySantAnna(vcov_type="jackknife", base_period="universal"), data)
        assert jk.df_inference == len(units) - 1
        got = es_frame(jk)
        rep_e = [es_frame(r).loc[-2, "att"] for r in reps]
        np.testing.assert_allclose(
            got.loc[-2, "se"], eq20(es_frame(full).loc[-2, "att"], rep_e), rtol=1e-12
        )

    def test_a_cluster_the_rule_misses_is_still_deleted(self, panel, monkeypatch):
        """The structural rule is checked on every fit, not trusted."""
        import diff_diff.staggered_jackknife as module

        real = module.resolve_deletion_units

        def forgets_one_cluster(*args, **kwargs):
            units = real(*args, **kwargs)
            assert units.spare_values == ()
            keep = slice(None, -1)
            return module.DeletionUnits(
                column=units.column,
                values=units.values[keep],
                labels=units.labels[keep],
                unit_name=units.unit_name,
                labels_are_codes=units.labels_are_codes,
                roles=units.roles[keep],
                cell_treated=units.cell_treated,
                spare_values=units.values[-1:],
                spare_labels=units.labels[-1:],
                spare_roles=units.roles[-1:],
            )

        expected = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
        monkeypatch.setattr(module, "resolve_deletion_units", forgets_one_cluster)
        got = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
        assert got.df_inference == expected.df_inference == 9
        np.testing.assert_allclose(got.se, expected.se, rtol=1e-12)
        assert set(got.get_loo_effects_df()["cluster"]) == set(panel["state"].unique())

    def test_mixed_type_cluster_labels(self, panel):
        data = panel.copy()
        mapping = {f"s{c}": (c if c % 2 else f"s{c}") for c in range(10)}
        data["state"] = data["state"].map(mapping).astype(object)
        jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), data)
        ref = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
        np.testing.assert_allclose(jk.se, ref.se, rtol=1e-12)
        assert set(jk.get_loo_effects_df()["cluster"]) == set(mapping.values())

    def test_two_clusters(self):
        """``H = 2``: each refit has a single cluster and still gives its estimate."""
        data = make_panel({0: 4, 1: 4, 2: 6}, n_clusters=8, units_per=2, seed=3)
        # Two regions, each holding treated and never-treated units.
        data["region"] = np.where(data["unit"] % 2 == 0, "east", "west")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jk = CallawaySantAnna(cluster="region", vcov_type="jackknife").fit(data, **FIT_KW)
        assert not any("PSU" in str(w.message) for w in caught)
        assert jk.df_inference == 1 and jk.n_clusters == 2
        full, reps = by_hand(data, "region", ["east", "west"], cluster="region")
        assert np.isfinite(jk.se) and jk.se > 0
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)


class TestDeletionLabels:
    """Deletion labels come from the caller's frame, not the fit's recoded copy."""

    @pytest.mark.parametrize("never_treated", [np.inf, "0"])
    def test_cluster_column_that_the_fit_recodes(self, never_treated):
        # Clustering on the first-treatment column itself: the fit recodes
        # ``inf`` to 0 and coerces strings in its working copy, so a label
        # read from there would match no row of the caller's frame and the
        # comparison cluster would never be deleted.
        data = make_panel(TWO_COHORTS, seed=37)
        if isinstance(never_treated, str):
            data["g"] = data["g"].astype(str)
        else:
            data["g"] = data["g"].astype(float).replace(0.0, never_treated)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jk = CallawaySantAnna(cluster="g", vcov_type="jackknife").fit(data, **FIT_KW)
        loo = jk.get_loo_effects_df()
        assert len(loo) == 3 and jk.df_inference == 2
        # Deleting the comparison cluster leaves no estimate: undefined, loudly.
        assert loo["att_loo"].isna().sum() == 1
        assert loo.loc[loo["att_loo"].isna(), "role"].tolist() == ["control"]
        assert any("leaves no estimate" in str(w.message) for w in caught)
        assert_nan_inference(
            {"se": jk.se, "t_stat": jk.t_stat, "p_value": jk.p_value, "conf_int": jk.conf_int}
        )

    def test_a_deletion_that_removes_nothing_is_an_error(self, panel, monkeypatch):
        import diff_diff.staggered_jackknife as module

        real = module.resolve_deletion_units

        def stale_labels(*args, **kwargs):
            units = real(*args, **kwargs)
            return module.DeletionUnits(
                column=units.column,
                values=("not-a-state",) + units.values[1:],
                labels=units.labels,
                unit_name=units.unit_name,
                labels_are_codes=units.labels_are_codes,
                roles=units.roles,
                cell_treated=units.cell_treated,
            )

        monkeypatch.setattr(module, "resolve_deletion_units", stale_labels)
        with pytest.raises(RuntimeError, match="removed no rows"):
            quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)


class TestReferenceCells:
    def test_horizon_that_turns_reference_only_keeps_its_value(self):
        """Universal base on a gapped grid: one horizon mixes an estimate and a reference."""
        data = make_panel({0: 3, 1: 5, 2: 5}, n_clusters=8, n_periods=7, seed=41)
        data = data[data["t"] != 4].reset_index(drop=True)
        params = dict(cluster="state", base_period="universal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jk = CallawaySantAnna(vcov_type="jackknife", **params).fit(data, **FIT_KW)
            got = jk.aggregate("event_study").to_dataframe().set_index("event_time")
        full, reps = by_hand(data, "state", list(data["state"].unique()), **params)
        # Cohort 5's base is period 3 (event time -2), where cohort 3 has an estimate.
        cells = full.group_time_effects
        assert cells[(5, 3)].get("is_reference") and not cells[(3, 1)].get("is_reference")
        assert not got.loc[-2, "is_reference"]
        # Without cohort 3's cluster the horizon is reference-only: its value is 0.
        without_s0 = es_frame(reps[0])
        assert without_s0.loc[-2, "att"] == 0.0 and without_s0.loc[-2, "is_reference"]
        rep_att = [es_frame(r).loc[-2, "att"] for r in reps]
        np.testing.assert_allclose(
            got.loc[-2, "se"], eq20(es_frame(full).loc[-2, "att"], rep_att), rtol=1e-12
        )
        assert np.isfinite(got.loc[-2, "se"]) and got.loc[-2, "se"] > 0

    def test_universal_base_reference_cells_keep_nan_se(self, panel):
        jk = quiet_fit(
            CallawaySantAnna(cluster="state", vcov_type="jackknife", base_period="universal"),
            panel,
        )
        refs = [c for c in jk.group_time_effects.values() if c.get("is_reference")]
        assert refs, "universal base no longer materializes reference cells"
        for cell in refs:
            assert cell["effect"] == 0.0
            assert_nan_inference(cell)
        es = jk.aggregate("event_study")
        ref_times = set(es.event_time[es.is_reference].tolist())
        assert ref_times and not ref_times & set(np.asarray(es.vcov_index).tolist())
        assert np.all(np.isnan(es.se[es.is_reference]))


class TestLostCells:
    """What a deletion that removes cells does (paper, Section 4 caveats)."""

    def test_single_cluster_cohort(self):
        data = make_panel({0: 4, 1: 6, 2: 6}, n_clusters=8, seed=17)
        with pytest.warns(UserWarning, match=r"deleting 1 of 8 state .*\('s0'\)"):
            jk = CallawaySantAnna(cluster="state", vcov_type="jackknife").fit(data, **FIT_KW)
        full, reps = by_hand(data, "state", list(data["state"].unique()), cluster="state")
        # Paper rule: the replicate built from the surviving cohort enters the sum.
        assert np.isfinite(jk.se)
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        for (g, _t), cell in jk.group_time_effects.items():
            if g == 4:
                assert_nan_inference(cell)
            else:
                assert np.isfinite(cell["se"]) and cell["se"] > 0
        with pytest.warns(UserWarning, match="the group aggregates average different cohorts"):
            groups = jk.aggregate("group").to_dataframe().set_index("label")
        assert np.isnan(groups.loc[4, "se"]) and np.isfinite(groups.loc[6, "se"])
        with pytest.warns(UserWarning, match="the event-study estimates average different cohorts"):
            es = jk.aggregate("event_study").to_dataframe().set_index("event_time")
        # Event time 0 is shared with cohort 6, so it survives every deletion.
        assert np.isfinite(es.loc[0, "se"])
        rep_e0 = [es_frame(r).loc[0, "att"] for r in reps]
        np.testing.assert_allclose(es.loc[0, "se"], eq20(es_frame(full).loc[0, "att"], rep_e0))
        # Event time 4 exists for cohort 4 only: lost when its cluster is deleted.
        assert np.isnan(es.loc[4, "se"])

    def test_cell_that_keeps_its_key_with_a_nan_effect(self):
        # Cohort 4 has two clusters; one of them is unobserved in the last
        # period, so deleting the other leaves (4, 8) with no treated support.
        data = make_panel({0: 4, 1: 4, 2: 6, 3: 6}, seed=23)
        data = data.loc[~((data["state"] == "s1") & (data["t"] == 8))].reset_index(drop=True)
        with pytest.warns(UserWarning, match=r"changes which ATT\(g,t\) cells are estimable"):
            jk = CallawaySantAnna(cluster="state", vcov_type="jackknife").fit(data, **FIT_KW)
        full, reps = by_hand(data, "state", list(data["state"].unique()), cluster="state")
        s0 = reps[0].group_time_effects.get((4, 8))
        assert s0 is None or np.isnan(s0["effect"])
        assert np.isfinite(full.group_time_effects[(4, 8)]["effect"])
        assert_nan_inference(jk.group_time_effects[(4, 8)])
        np.testing.assert_allclose(jk.se, eq20(full.att, [r.att for r in reps]), rtol=1e-12)
        loo = jk.get_loo_effects_df().set_index("cluster")
        assert loo.loc["s0", "n_cells"] < loo.loc["s5", "n_cells"]

    def test_deleting_the_only_comparison_cluster_is_undefined(self):
        data = make_panel({0: 4, 1: 4, 2: 6, 3: 6, 4: 6}, n_clusters=6, seed=19)
        with pytest.warns(UserWarning, match=r"'s5' leaves no estimate .*standard error is NaN"):
            jk = CallawaySantAnna(cluster="state", vcov_type="jackknife").fit(data, **FIT_KW)
        assert np.isfinite(jk.att)
        assert_nan_inference(
            {"se": jk.se, "t_stat": jk.t_stat, "p_value": jk.p_value, "conf_int": jk.conf_int}
        )
        for cell in jk.group_time_effects.values():
            assert_nan_inference(cell)
        es = es_frame(jk)
        assert es["se"].isna().all() and es["p_value"].isna().all()
        assert jk.aggregate("event_study").vcov is None
        assert group_frame(jk)["se"].isna().all()
        loo = jk.get_loo_effects_df()
        assert np.isnan(loo.set_index("cluster").loc["s5", "att_loo"])
        assert loo["cluster"].iloc[-1] == "s5"


class TestFewTreatedClusters:
    """A number resting on fewer than three treated clusters draws a warning."""

    @staticmethod
    def messages(caught):
        return [str(w.message) for w in caught if "treated clusters" in str(w.message)]

    def fit_and_aggregate(self, cohorts, n_clusters):
        data = make_panel(cohorts, n_clusters=n_clusters, seed=31)
        with warnings.catch_warnings(record=True) as at_fit:
            warnings.simplefilter("always")
            jk = CallawaySantAnna(cluster="state", vcov_type="jackknife").fit(data, **FIT_KW)
        with warnings.catch_warnings(record=True) as at_es:
            warnings.simplefilter("always")
            jk.aggregate("event_study")
        with warnings.catch_warnings(record=True) as at_group:
            warnings.simplefilter("always")
            jk.aggregate("group")
        return self.messages(at_fit), self.messages(at_es), self.messages(at_group)

    def test_two_treated_clusters_per_cohort(self):
        # Four treated clusters overall, two behind each cohort's own cells.
        fit, es, group = self.fit_and_aggregate(TWO_COHORTS, 10)
        assert len(fit) == 1 and "14 of 14 ATT(g,t) cells rest on fewer than 3" in fit[0]
        assert "overall ATT" not in fit[0]
        # Event times 0 to 2 pool both cohorts (four clusters); the rest do not.
        assert len(es) == 1 and "event time(s) -4, -3, 3, 4 rest" in es[0]
        assert len(group) == 1 and "cohort(s) 4, 6 rest" in group[0]

    def test_one_treated_cluster_per_cohort_also_names_the_overall_att(self):
        fit, _, _ = self.fit_and_aggregate({0: 4, 1: 6}, 8)
        assert len(fit) == 1 and "the overall ATT rests on 2 and" in fit[0]

    def test_three_treated_clusters_per_cohort_is_silent(self):
        fit, es, group = self.fit_and_aggregate({0: 4, 1: 4, 2: 4, 3: 6, 4: 6, 5: 6}, 12)
        assert fit == [] and es == [] and group == []

    def test_the_default_variance_never_warns(self):
        data = make_panel(TWO_COHORTS, seed=31)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hc1 = CallawaySantAnna(cluster="state").fit(data, **FIT_KW)
            hc1.aggregate("event_study")
        assert self.messages(caught) == []

    def test_many_deletion_units_warn_about_the_cost(self, panel, monkeypatch):
        import diff_diff.staggered_jackknife as module

        monkeypatch.setattr(module, "_MANY_REPLICATES", 15)
        with pytest.warns(UserWarning, match=r"20 deletion units, and fit\(\) ran 21 extra"):
            CallawaySantAnna(vcov_type="jackknife").fit(panel, **FIT_KW)

    def test_the_cost_counts_audited_clusters_too(self, monkeypatch):
        import diff_diff.staggered_jackknife as module

        # Two always-treated clusters are refitted to check they are inert,
        # then dropped: 9 deletion units, 11 delete-one refits, 1 baseline.
        data = make_panel({**TWO_COHORTS, 8: 1, 9: 1}, n_clusters=11, seed=13)
        monkeypatch.setattr(module, "_MANY_REPLICATES", 10)
        with pytest.warns(UserWarning, match=r"9 deletion units, and fit\(\) ran 12 extra"):
            jk = CallawaySantAnna(cluster="state", vcov_type="jackknife").fit(data, **FIT_KW)
        assert jk.df_inference == 8


class TestRefusals:
    def test_bootstrap_at_construction(self):
        with pytest.raises(ValueError, match="two inference methods"):
            CallawaySantAnna(vcov_type="jackknife", n_bootstrap=99)

    def test_bootstrap_at_set_params(self):
        est = CallawaySantAnna(vcov_type="jackknife")
        with pytest.raises(ValueError, match="two inference methods"):
            est.set_params(n_bootstrap=99)
        assert est.n_bootstrap == 0
        boot = CallawaySantAnna(n_bootstrap=99)
        with pytest.raises(ValueError, match="two inference methods"):
            boot.set_params(vcov_type="jackknife")
        assert boot.vcov_type == "hc1"

    def test_bootstrap_after_direct_mutation(self, panel):
        est = CallawaySantAnna(vcov_type="jackknife")
        est.n_bootstrap = 99
        with pytest.raises(ValueError, match="two inference methods"):
            est.fit(panel, **FIT_KW)

    def test_survey_design(self, panel):
        data = panel.assign(w=1.0)
        with pytest.raises(ValueError, match="not supported with survey_design"):
            CallawaySantAnna(vcov_type="jackknife").fit(
                data, survey_design=SurveyDesign(weights="w"), **FIT_KW
            )

    def test_allow_unbalanced_panel(self, panel):
        with pytest.raises(ValueError, match="allow_unbalanced_panel=True"):
            CallawaySantAnna(vcov_type="jackknife", allow_unbalanced_panel=True).fit(
                panel, **FIT_KW
            )

    def test_repeated_cross_sections_need_a_cluster(self, panel):
        data = panel.assign(unit=np.arange(len(panel)))
        with pytest.raises(ValueError, match="panel=False needs cluster"):
            quiet_fit(CallawaySantAnna(vcov_type="jackknife", panel=False), data)

    def test_one_cluster(self, panel):
        data = panel.assign(state="only")
        with pytest.raises(ValueError, match="at least two clusters"):
            quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), data)

    @pytest.mark.parametrize("bad", ["classical", "hc2", "hc2_bm", "conley", "hc3", "cr3"])
    def test_other_values_are_still_rejected(self, bad):
        with pytest.raises(ValueError, match="vcov_type"):
            CallawaySantAnna(vcov_type=bad)


class TestResults:
    def test_metadata(self, panel, clustered):
        jk, full, _ = clustered
        assert jk.vcov_type == "jackknife"
        assert jk.df_inference == 9
        assert (jk.cluster_name, jk.n_clusters) == (full.cluster_name, full.n_clusters)
        assert jk.bootstrap_results is None
        unit_jk = quiet_fit(CallawaySantAnna(vcov_type="jackknife"), panel)
        unit_hc1 = quiet_fit(CallawaySantAnna(), panel)
        assert (unit_jk.cluster_name, unit_jk.n_clusters) == (None, None)
        assert (unit_hc1.cluster_name, unit_hc1.n_clusters) == (None, None)
        assert unit_jk.df_inference == panel["unit"].nunique() - 1

    def test_event_study_covariance(self, clustered):
        jk, full, reps = clustered
        es = jk.aggregate("event_study")
        index = list(np.asarray(es.vcov_index).tolist())
        frame = es.to_dataframe().set_index("event_time")
        rep_att = np.array([es_frame(r)["att"].reindex(index).to_numpy() for r in reps])
        deviations = rep_att - es_frame(full)["att"].reindex(index).to_numpy()
        h = len(reps)
        # Element-wise, off-diagonals included: the outer-product form of eq. 20.
        np.testing.assert_allclose(
            es.vcov, (h - 1) / h * deviations.T @ deviations, rtol=1e-11, atol=1e-15
        )
        np.testing.assert_allclose(np.diag(es.vcov), frame.loc[index, "se"] ** 2, rtol=1e-12)
        assert np.linalg.eigvalsh(es.vcov).min() > -1e-12
        assert not np.allclose(es.vcov, np.diag(np.diag(es.vcov)))
        np.testing.assert_array_equal(np.asarray(es.df, dtype=float), np.full(len(es.att), 9.0))
        assert es.cband_crit_value is None
        # The results fields stay None on a CS fit: the container carries them.
        assert jk.event_study_vcov is None and jk.event_study_vcov_index is None

    def test_group_table_records_its_df(self, clustered):
        jk, _, _ = clustered
        np.testing.assert_array_equal(np.asarray(jk.aggregate("group").df, dtype=float), 9.0)

    def test_simple_and_total_relay_the_jackknife(self, clustered):
        jk, _, _ = clustered
        simple = jk.aggregate("simple")
        assert simple.se[0] == jk.se and simple.df[0] == 9
        total = jk.aggregate("total")
        scale = total.att[0] / jk.att
        np.testing.assert_allclose(total.se[0], scale * jk.se, rtol=1e-12)
        np.testing.assert_allclose(total.p_value[0], jk.p_value, rtol=1e-12)

    def test_summary_names_the_variance(self, panel, clustered):
        jk, full, _ = clustered
        line = "Variance: cluster jackknife (CV3), delete-one state, H=10, t(9)"
        assert line in jk.summary().splitlines()
        assert "jackknife" not in full.summary()
        unit_jk = quiet_fit(CallawaySantAnna(vcov_type="jackknife"), panel)
        assert "delete-one unit, H=20, t(19)" in unit_jk.summary()

    def test_pickle_round_trip_then_aggregate(self, clustered):
        jk, _, _ = clustered
        clone = pickle.loads(pickle.dumps(jk))
        pd.testing.assert_frame_equal(es_frame(clone), es_frame(jk))
        pd.testing.assert_frame_equal(clone.get_loo_effects_df(), jk.get_loo_effects_df())

    def test_fit_is_idempotent_and_leaves_params_alone(self, panel):
        est = CallawaySantAnna(cluster="state", vcov_type="jackknife")
        before = est.get_params()
        first = quiet_fit(est, panel)
        second = quiet_fit(est, panel)
        assert est.get_params() == before
        assert first.se == second.se
        pd.testing.assert_frame_equal(first.get_loo_effects_df(), second.get_loo_effects_df())

    def test_a_kit_without_the_field_still_aggregates(self, panel):
        """An artifact pickled before the kit carried replicates has no such attribute."""
        hc1 = quiet_fit(CallawaySantAnna(cluster="state"), panel)
        expected = es_frame(hc1)
        object.__delattr__(hc1._aggregation_kit, "jackknife")
        assert "jackknife" not in vars(hc1._aggregation_kit)
        pd.testing.assert_frame_equal(es_frame(hc1), expected)


class TestLeaveOneOutTable:
    def test_values_and_layout(self, clustered):
        jk, full, reps = clustered
        loo = jk.get_loo_effects_df()
        assert list(loo.columns) == ["cluster", "role", "att_loo", "delta_from_full", "n_cells"]
        by_label = loo.set_index("cluster")
        for label, rep in zip(full_labels(reps, by_label), reps):
            np.testing.assert_allclose(by_label.loc[label, "att_loo"], rep.att, rtol=1e-14)
        np.testing.assert_allclose(loo["delta_from_full"], loo["att_loo"] - jk.att)
        assert loo["delta_from_full"].abs().is_monotonic_decreasing
        assert set(by_label.index[by_label["role"] == "treated"]) == {"s0", "s1", "s2", "s3"}
        assert (loo["n_cells"] == len(jk.group_time_effects)).all()

    @pytest.mark.parametrize("cluster", [None, "unit", "unit_alias"])
    def test_unit_level_tables_carry_codes_not_labels(self, panel, cluster):
        """No raw unit identifier is retained, however the unit is named."""
        data = panel.copy()
        data["unit"] = "store-" + data["unit"].astype(str).str.zfill(3)
        data["unit_alias"] = "alias-" + data["unit"]
        jk = quiet_fit(CallawaySantAnna(cluster=cluster, vcov_type="jackknife"), data)
        loo = jk.get_loo_effects_df()
        assert sorted(loo["cluster"]) == list(range(20))
        # Code i is the i-th unit in sorted order.
        units = np.sort(data["unit"].unique())
        full, reps = by_hand(data, "unit", list(units), cluster=cluster)
        np.testing.assert_allclose(
            loo.set_index("cluster").loc[list(range(20)), "att_loo"],
            [r.att for r in reps],
            rtol=1e-12,
        )
        blob = pickle.dumps(jk)
        assert b"store-" not in blob and b"alias-" not in blob
        assert "delete-one unit" in jk.summary()

    def test_refused_without_the_jackknife(self, panel, clustered):
        _, full, _ = clustered
        with pytest.raises(ValueError, match="requires vcov_type='jackknife'"):
            full.get_loo_effects_df()

    def test_refused_on_dml_did(self):
        from diff_diff import generate_staggered_data

        data = generate_staggered_data(n_units=120, n_periods=5, seed=4)
        data["x"] = np.random.default_rng(0).normal(size=len(data))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0).fit(
                data, "outcome", "unit", "period", "first_treat", covariates=["x"]
            )
        with pytest.raises(ValueError, match="requires vcov_type='jackknife'"):
            res.get_loo_effects_df()


def full_labels(reps, by_label):
    """Deletion labels in the by-hand order (order of first appearance)."""
    return [f"s{c}" for c in range(len(reps))]


class TestConsumers:
    def test_honest_did_uses_the_jackknife_df(self):
        data = make_panel(TWO_COHORTS, n_clusters=24, n_periods=6, seed=29)
        jk = quiet_fit(
            CallawaySantAnna(cluster="state", vcov_type="jackknife", base_period="universal"),
            data,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            honest = HonestDiD(method="relative_magnitude", M=1.0).fit(jk.aggregate("event_study"))
        width = honest.ci_ub - honest.ci_lb
        assert np.isfinite(width)
        # t(23) critical value, not 1.96.
        assert width > 2 * stats.t.ppf(0.975, 23) * honest.original_se * (1 - 1e-9)

    def test_honest_did_refuses_a_singular_jackknife_covariance(self, panel):
        """Ten clusters cannot identify a covariance over nine horizons."""
        jk = quiet_fit(
            CallawaySantAnna(cluster="state", vcov_type="jackknife", base_period="universal"),
            panel,
        )
        es = jk.aggregate("event_study")
        assert len(es.vcov_index) >= jk.df_inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="singular or near-singular"):
                HonestDiD(method="relative_magnitude", M=1.0).fit(es)

    def test_diagnostic_report_policy(self, panel):
        flags = dict(
            run_sensitivity=False,
            run_bacon=False,
            run_heterogeneity=False,
            run_design_effect=False,
            run_epv=False,
        )
        jk = quiet_fit(CallawaySantAnna(cluster="state", vcov_type="jackknife"), panel)
        hc1 = quiet_fit(CallawaySantAnna(cluster="state"), panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jk_report = DiagnosticReport(jk, **flags).to_dict()
            hc1_report = DiagnosticReport(hc1, **flags).to_dict()
        pt = jk_report["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "bonferroni"
        pre = es_frame(jk)
        pre = pre[(pre.index < 0) & pre["p_value"].notna()]
        np.testing.assert_allclose(
            pt["joint_p_value"], min(1.0, len(pre) * pre["p_value"].min()), rtol=1e-12
        )
        assert jk_report["pretrends_power"]["status"] == "skipped"
        assert "cluster-jackknife" in jk_report["pretrends_power"]["reason"]
        # The default variance keeps its covariance-based test and its power section.
        assert hc1_report["parallel_trends"]["method"].startswith("joint_wald")
        assert hc1_report["pretrends_power"]["status"] == "ran"

    def test_design_diagnosis_accepts_a_jackknife_candidate(self):
        rng = np.random.default_rng(8)
        history = pd.DataFrame(
            [
                (u, t, level + rng.normal())
                for u, level in enumerate(rng.normal(size=16))
                for t in range(1, 9)
            ],
            columns=["unit", "t", "y"],
        )
        candidates = {
            "cs": CallawaySantAnna(),
            "cs_jackknife": CallawaySantAnna(vcov_type="jackknife"),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diagnosis = diagnose_design(
                history,
                outcome="y",
                unit="unit",
                time="t",
                estimators=candidates,
                cohort_periods=[6],
                n_treated=6,
                n_simulations=10,
                seed=1,
                progress=False,
            )
            lock = diagnosis.lock("cs_jackknife", data=history)
        frame = diagnosis.to_dataframe().set_index("label")
        assert set(frame.index) == {"cs", "cs_jackknife"}
        assert frame.loc["cs_jackknife", "n_simulation_failures"] == 0
        assert frame.loc["cs_jackknife", "n_nan_inference"] == 0
        # Same placebo draws, same estimates; only the standard errors differ.
        np.testing.assert_allclose(
            frame.loc["cs_jackknife", "mean_estimate"], frame.loc["cs", "mean_estimate"]
        )
        assert frame.loc["cs_jackknife", "mean_se"] != frame.loc["cs", "mean_se"]
        rebuilt = lock.to_estimator()
        assert isinstance(rebuilt, CallawaySantAnna) and rebuilt.vcov_type == "jackknife"


SIZE_EVIDENCE = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "cs_jackknife_size.json"
)


@pytest.fixture(scope="module")
def points():
    if not SIZE_EVIDENCE.exists():
        pytest.skip("size simulation results not present")
    return json.loads(SIZE_EVIDENCE.read_text())["design_points"]


class TestSizeEvidence:
    """The rejection rates the registry quotes are the ones the simulation produced.

    ``benchmarks/python/cs_jackknife_size.py`` writes the file; the registry's
    "Measured size" note (CallawaySantAnna, Cluster jackknife) and the
    reporting note quote the ranges below, in whole percent.
    """

    SINGLE_COHORT = ("es_3", "es_4")
    SHARED = ("es_0", "es_1", "es_2")

    @staticmethod
    def span(points, method, quantities, treated):
        rates = [
            p["results"][method][q]["rejection_rate"]
            for p in points
            if p["n_treated_per_cohort"] in treated
            for q in quantities
        ]
        rates = [r for r in rates if np.isfinite(r)]
        if not rates:
            return None
        return (int(f"{100 * min(rates):.0f}"), int(f"{100 * max(rates):.0f}"))

    @pytest.mark.parametrize(
        "treated, hc1, bootstrap, jackknife, hc1_single, jackknife_single",
        [
            ((1,), (15, 35), (26, 39), (8, 18), (42, 72), None),
            ((2,), (5, 15), (14, 19), (3, 10), (14, 29), (6, 18)),
            ((3,), (7, 11), (12, 21), (5, 8), (12, 19), (6, 12)),
            ((4,), (6, 9), (9, 12), (4, 7), (7, 14), (4, 9)),
            ((6, 8, 10, 12), (4, 7), (7, 10), (3, 6), (6, 10), (4, 7)),
        ],
    )
    def test_registry_table(
        self, points, treated, hc1, bootstrap, jackknife, hc1_single, jackknife_single
    ):
        assert self.span(points, "hc1", ["overall"], treated) == hc1
        assert self.span(points, "bootstrap", ["overall"], treated) == bootstrap
        assert self.span(points, "jackknife", ["overall"], treated) == jackknife
        assert self.span(points, "hc1", self.SINGLE_COHORT, treated) == hc1_single
        assert self.span(points, "jackknife", self.SINGLE_COHORT, treated) == jackknife_single

    def test_jackknife_rejects_less_than_the_default_everywhere(self, points):
        for p in points:
            res = p["results"]
            assert (
                res["jackknife"]["overall"]["rejection_rate"]
                < res["hc1"]["overall"]["rejection_rate"]
            ), (p["n_clusters"], p["n_treated_per_cohort"], p["units_per_cluster"])
            assert sum(p["fit_failures"].values()) == 0

    def test_equal_treated_clusters_give_like_rejection(self, points):
        """Two treated clusters: overall at J=1 and a single-cohort horizon at J=2."""
        assert self.span(points, "jackknife", ["overall"], (1,)) == (8, 18)
        assert self.span(points, "jackknife", self.SINGLE_COHORT, (2,)) == (6, 18)
        assert self.span(points, "jackknife", ["overall"], (2,)) == (3, 10)
        assert self.span(points, "jackknife", self.SINGLE_COHORT, (4,)) == (4, 9)

    def test_pretrend_checks(self, points):
        bonferroni = ["pretrend_bonferroni"]
        assert self.span(points, "jackknife", bonferroni, (2,)) == (8, 29)
        assert self.span(points, "jackknife", bonferroni, (3,)) == (6, 14)
        assert self.span(points, "jackknife", bonferroni, (4,)) == (5, 12)
        assert self.span(points, "jackknife", bonferroni, (6, 8, 10, 12)) == (4, 7)
        wald = ["pretrend_wald_chi2"]
        assert self.span(points, "jackknife", wald, (2,)) == (49, 71)
        # The default fit's chi-square joint test (tracked in TODO.md).
        assert self.span(points, "hc1", wald, (2,)) == (62, 78)
        assert self.span(points, "hc1", wald, (6, 8, 10, 12)) == (13, 22)

    def test_single_cohort_quantities_are_undefined_with_one_cluster_per_cohort(self, points):
        for p in points:
            if p["n_treated_per_cohort"] != 1:
                continue
            jk = p["results"]["jackknife"]
            for name in (*self.SINGLE_COHORT, "pretrend_bonferroni"):
                assert jk[name]["share_undefined"] == 1.0
            for name in ("overall", *self.SHARED):
                assert jk[name]["share_undefined"] == 0.0


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("R golden not generated (benchmarks/R/generate_cs_jackknife_golden.R)")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def golden_data(golden):
    return pd.DataFrame(golden["data"])


class TestRParity:
    """R ``did`` on the same delete-one samples (the ``didjack`` procedure)."""

    @pytest.mark.parametrize(
        "name",
        [
            "reg_nevertreated_varying",
            "dr_nevertreated_varying",
            "reg_notyettreated_varying",
            "dr_nevertreated_universal",
        ],
    )
    def test_configuration(self, golden, golden_data, name):
        cfg = golden["configurations"][name]
        jk = quiet_fit(
            CallawaySantAnna(
                cluster="state",
                vcov_type="jackknife",
                estimation_method=cfg["est_method"],
                control_group=cfg["control_group"],
                base_period=cfg["base_period"],
            ),
            golden_data,
            outcome="y",
            unit="unit",
            time="t",
            first_treat="g",
        )
        np.testing.assert_allclose(jk.att, cfg["overall_att"], rtol=1e-8)
        np.testing.assert_allclose(jk.se, cfg["overall_se"], rtol=1e-8)
        np.testing.assert_allclose(jk.p_value, cfg["overall_p"], rtol=1e-7)
        np.testing.assert_allclose(jk.conf_int, (cfg["overall_lci"], cfg["overall_uci"]), rtol=1e-7)
        assert jk.df_inference == cfg["df"]
        loo = jk.get_loo_effects_df().set_index("cluster")["att_loo"]
        np.testing.assert_allclose(
            loo.reindex(golden["clusters"]).to_numpy(), cfg["overall_att_loo"], rtol=1e-8
        )
        groups = group_frame(jk)
        np.testing.assert_allclose(
            groups.loc[cfg["group_labels"], "se"], cfg["group_se"], rtol=1e-8
        )
        es = es_frame(jk)
        np.testing.assert_allclose(
            es.loc[cfg["event_times"], "se"], cfg["event_study_se"], rtol=1e-8
        )
