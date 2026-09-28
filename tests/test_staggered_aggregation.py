"""
Frozen-copy parity and contract tests for the CS aggregation IF fast path.

The O(n_units) rewrite of ``_compute_combined_influence_function`` (per-fit
cohort tables + closed-form WIF) replaces the dense (n_units x n_gt) WIF
matrices, the per-group full-DataFrame scans, and the per-unit Python loops
of the pre-rewrite implementation. ``_frozen_combined_influence_function``
below is a byte-copy of that pre-rewrite implementation (self removed), kept
as a drift-bound guard: the fast path must agree with it at rtol=0,
atol=1e-9. The closed form is algebraically identical to the dense
``wif_matrix @ effects``; only floating-point accumulation order differs
(bincount cohort masses vs per-group mask-sums, table-gather vs matmul), so
observed drift is ~1e-16 - the 1e-9 bound mirrors the frozen-copy convention
in tests/test_utils.py.

One intentional value difference: for units whose cohort is not among the
keeper (g,t) groups, the old dense form realizes an exact algebraic zero
through cancelling terms (~1e-16 residue in floating point) while the closed
form returns exactly 0.0 - parity therefore holds at atol, not bitwise.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, StaggeredTripleDifference, SurveyDesign
from diff_diff.staggered_aggregation import CallawaySantAnnaAggregationMixin


def _frozen_combined_influence_function(
    gt_pairs,
    weights,
    effects,
    groups_for_gt,
    influence_func_info,
    df,
    unit,
    precomputed=None,
    global_unit_to_idx=None,
    n_global_units=None,
):
    """Byte-copy of the pre-rewrite ``_compute_combined_influence_function``
    (dense-WIF implementation, ``self`` removed - the body never used it)."""
    if not influence_func_info:
        if n_global_units is not None:
            return np.zeros(n_global_units), None
        return np.zeros(0), None

    # Detect RCS mode via explicit flag. In RCS, obs indices ARE array positions.
    _is_rcs = precomputed is not None and not precomputed.get("is_panel", True)

    # Build unit index mapping (local or global)
    if _is_rcs and n_global_units is not None:
        # RCS: direct indexing — obs indices are the array positions
        n_units = n_global_units
        all_units = None
    elif global_unit_to_idx is not None and n_global_units is not None:
        n_units = n_global_units
        all_units = None  # caller already has the unit list
    else:
        all_units_set = set()
        for g, t in gt_pairs:
            if (g, t) in influence_func_info:
                info = influence_func_info[(g, t)]
                all_units_set.update(info["treated_units"])
                all_units_set.update(info["control_units"])

        if not all_units_set:
            return np.zeros(0), []

        all_units = sorted(all_units_set)
        n_units = len(all_units)

    # Get unique groups and their information
    unique_groups = sorted(set(groups_for_gt))
    unique_groups_set = set(unique_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    # Check for survey weights in precomputed data
    survey_w = precomputed.get("survey_weights") if precomputed is not None else None

    # Compute group-level probabilities matching R's formula
    group_sizes = {}
    if survey_w is not None:
        precomputed_cohorts = precomputed["unit_cohorts"]
        for g in unique_groups:
            mask_g = precomputed_cohorts == g
            group_sizes[g] = float(np.sum(survey_w[mask_g]))
        total_weight = float(np.sum(survey_w))
    elif _is_rcs:
        precomputed_cohorts = precomputed["unit_cohorts"]
        for g in unique_groups:
            group_sizes[g] = int(np.sum(precomputed_cohorts == g))
        total_weight = float(n_units)
    else:
        for g in unique_groups:
            treated_in_g = df[df["first_treat"] == g][unit].nunique()
            group_sizes[g] = treated_in_g
        total_weight = float(n_units)

    pg_by_group = np.array([group_sizes[g] / total_weight for g in unique_groups])
    pg_keepers = np.array([pg_by_group[group_to_idx[g]] for g in groups_for_gt])
    sum_pg_keepers = np.sum(pg_keepers)

    if sum_pg_keepers == 0:
        return np.zeros(n_units), all_units

    psi_standard = np.zeros(n_units)

    for j, (g, t) in enumerate(gt_pairs):
        if (g, t) not in influence_func_info:
            continue

        info = influence_func_info[(g, t)]
        w = weights[j]

        treated_idx = info["treated_idx"]
        if len(treated_idx) > 0:
            np.add.at(psi_standard, treated_idx, w * info["treated_inf"])

        control_idx = info["control_idx"]
        if len(control_idx) > 0:
            np.add.at(psi_standard, control_idx, w * info["control_inf"])

    unit_groups_array = np.full(n_units, -1, dtype=np.float64)

    if _is_rcs:
        precomputed_cohorts = precomputed["unit_cohorts"]
        for g in unique_groups:
            mask_g = precomputed_cohorts == g
            unit_groups_array[mask_g] = g
    elif global_unit_to_idx is not None:
        idx_uid_pairs = [(idx, uid) for uid, idx in global_unit_to_idx.items()]

        if precomputed is not None:
            precomputed_cohorts = precomputed["unit_cohorts"]
            precomputed_unit_to_idx = precomputed["unit_to_idx"]
            for idx, uid in idx_uid_pairs:
                if uid in precomputed_unit_to_idx:
                    cohort = precomputed_cohorts[precomputed_unit_to_idx[uid]]
                    if cohort in unique_groups_set:
                        unit_groups_array[idx] = cohort
        else:
            for idx, uid in idx_uid_pairs:
                unit_first_treat = df[df[unit] == uid]["first_treat"].iloc[0]
                if unit_first_treat in unique_groups_set:
                    unit_groups_array[idx] = unit_first_treat
    else:
        idx_uid_pairs = list(enumerate(all_units))
        for idx, uid in idx_uid_pairs:
            unit_first_treat = df[df[unit] == uid]["first_treat"].iloc[0]
            if unit_first_treat in unique_groups_set:
                unit_groups_array[idx] = unit_first_treat

    groups_for_gt_array = np.array(groups_for_gt)
    indicator_matrix = (
        unit_groups_array[:, np.newaxis] == groups_for_gt_array[np.newaxis, :]
    ).astype(np.float64)

    if survey_w is not None:
        if _is_rcs and precomputed is not None:
            unit_sw = survey_w
        elif global_unit_to_idx is not None and precomputed is not None:
            unit_sw = np.zeros(n_units)
            precomputed_unit_to_idx_local = precomputed["unit_to_idx"]
            idx_uid_pairs_sw = [(idx, uid) for uid, idx in global_unit_to_idx.items()]
            for idx, uid in idx_uid_pairs_sw:
                if uid in precomputed_unit_to_idx_local:
                    pc_idx = precomputed_unit_to_idx_local[uid]
                    unit_sw[idx] = survey_w[pc_idx]
        else:
            unit_sw = np.ones(n_units)

        weighted_indicator = indicator_matrix * unit_sw[:, np.newaxis]
        indicator_diff = weighted_indicator - pg_keepers
        indicator_sum_w = np.sum(indicator_diff, axis=1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if1_matrix = indicator_diff / sum_pg_keepers
            if2_matrix = np.outer(indicator_sum_w, pg_keepers) / (sum_pg_keepers**2)
            wif_matrix = if1_matrix - if2_matrix
            wif_contrib = wif_matrix @ effects
    else:
        indicator_sum = np.sum(indicator_matrix - pg_keepers, axis=1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if1_matrix = (indicator_matrix - pg_keepers) / sum_pg_keepers
            if2_matrix = np.outer(indicator_sum, pg_keepers) / (sum_pg_keepers**2)
            wif_matrix = if1_matrix - if2_matrix
            wif_contrib = wif_matrix @ effects

    if not np.all(np.isfinite(wif_contrib)):
        return np.full(n_units, np.nan), all_units

    psi_wif = wif_contrib / total_weight
    psi_total = psi_standard + psi_wif

    return psi_total, all_units


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(
    n_units=300,
    n_periods=10,
    n_cohorts=3,
    n_cov=0,
    never_frac=0.3,
    drop_frac=0.0,
    survey=False,
    inf_coded=False,
    seed=42,
):
    """Small staggered panel with covariate-selected cohorts."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_units, max(n_cov, 1)))
    treat_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    logits = 0.4 * x[:, 0][:, None] + rng.normal(0, 1, size=(n_units, n_cohorts))
    never = rng.random(n_units) < never_frac
    g = np.where(never, 0, treat_periods[logits.argmax(axis=1)])

    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)
    g_long = np.repeat(g, n_periods)
    treated = (g_long > 0) & (t >= g_long)
    y = (
        np.repeat(rng.normal(0, 1, n_units), n_periods)
        + 0.05 * t
        + 2.0 * treated
        + rng.normal(0, 1, n_units * n_periods)
    )
    df = pd.DataFrame({"unit": unit, "time": t, "outcome": y, "first_treat": g_long})
    if inf_coded:
        # staggered.py accepts np.inf for never-treated; exercises the
        # input-normalization path in front of the cohort tables.
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] == 0, "first_treat"] = np.inf
    for j in range(n_cov):
        df[f"x{j + 1}"] = np.repeat(x[:, j], n_periods)
    if survey:
        df["pw"] = np.repeat(np.exp(rng.normal(0, 0.4, n_units)), n_periods)
    if drop_frac > 0:
        df = df[rng.random(len(df)) >= drop_frac].reset_index(drop=True)
    return df


def _make_rcs(n_obs=20_000, n_periods=10, n_cohorts=3, n_cov=2, seed=42):
    """Repeated cross-sections: unique unit id per row (panel=False)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, max(n_cov, 1)))
    treat_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    logits = 0.4 * x[:, 0][:, None] + rng.normal(0, 1, size=(n_obs, n_cohorts))
    never = rng.random(n_obs) < 0.3
    g = np.where(never, 0, treat_periods[logits.argmax(axis=1)])
    t = rng.integers(1, n_periods + 1, size=n_obs)
    treated = (g > 0) & (t >= g)
    y = 0.05 * t + 2.0 * treated + rng.normal(0, 1, n_obs)
    df = pd.DataFrame({"unit": np.arange(n_obs), "time": t, "outcome": y, "first_treat": g})
    for j in range(n_cov):
        df[f"x{j + 1}"] = x[:, j]
    return df


def _make_ddd(n_units=400, n_periods=8, n_cohorts=2, seed=42):
    """Staggered triple-diff panel: unit-level eligibility partition."""
    rng = np.random.default_rng(seed)
    treat_periods = np.linspace(3, n_periods - 2, n_cohorts).astype(int)
    never = rng.random(n_units) < 0.4
    g = np.where(never, 0, treat_periods[rng.integers(0, n_cohorts, size=n_units)])
    elig = (rng.random(n_units) < 0.5).astype(int)

    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)
    g_long = np.repeat(g, n_periods)
    e_long = np.repeat(elig, n_periods)
    treated = (g_long > 0) & (t >= g_long) & (e_long == 1)
    y = (
        np.repeat(rng.normal(0, 1, n_units), n_periods)
        + 0.05 * t
        + 0.3 * e_long
        + 2.0 * treated
        + rng.normal(0, 1, n_units * n_periods)
    )
    return pd.DataFrame(
        {
            "unit": unit,
            "period": t,
            "outcome": y,
            "first_treat": g_long,
            "eligibility": e_long,
        }
    )


class _ParityRecorder:
    """Monkeypatch wrapper: on every combined-IF call, compare the live
    implementation against the frozen pre-rewrite copy on identical inputs."""

    def __init__(self, monkeypatch, atol=1e-9):
        self.n_calls = 0
        self.atol = atol
        orig = CallawaySantAnnaAggregationMixin._compute_combined_influence_function

        recorder = self

        def wrapper(mixin_self, *args, **kwargs):
            new_psi, new_units = orig(mixin_self, *args, **kwargs)
            old_psi, _ = _frozen_combined_influence_function(*args, **kwargs)
            assert new_psi.shape == old_psi.shape
            np.testing.assert_allclose(
                new_psi,
                old_psi,
                rtol=0,
                atol=recorder.atol,
                equal_nan=True,
                err_msg="fast-path psi diverged from frozen pre-rewrite copy",
            )
            recorder.n_calls += 1
            return new_psi, new_units

        monkeypatch.setattr(
            CallawaySantAnnaAggregationMixin,
            "_compute_combined_influence_function",
            wrapper,
        )


class TestCombinedIFFrozenParity:
    """Old-vs-new psi_total parity across data families and aggregation
    targets. aggregate='all' exercises the multi-cohort overall target,
    every per-horizon keeper subset (incl. single-keeper horizons), and
    single-cohort group targets in one fit."""

    def _fit_cs(self, df, covs=None, survey=False, rcs=False, boot=0):
        cs = CallawaySantAnna(
            estimation_method="dr",
            n_bootstrap=boot,
            panel=not rcs,
            seed=7,
        )
        cs.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=covs,
            aggregate="all",
            survey_design=SurveyDesign(weights="pw") if survey else None,
        )
        return cs

    def test_plain_panel(self, monkeypatch):
        rec = _ParityRecorder(monkeypatch)
        self._fit_cs(_make_panel(n_cov=2), covs=["x1", "x2"])
        assert rec.n_calls > 0

    def test_survey_pweights(self, monkeypatch):
        rec = _ParityRecorder(monkeypatch)
        self._fit_cs(_make_panel(n_cov=2, survey=True), covs=["x1", "x2"], survey=True)
        assert rec.n_calls > 0

    def test_rcs(self, monkeypatch):
        rec = _ParityRecorder(monkeypatch)
        with pytest.warns(UserWarning, match="stationary"):
            self._fit_cs(_make_rcs(), covs=["x1", "x2"], rcs=True)
        assert rec.n_calls > 0

    def test_unbalanced_panel(self, monkeypatch):
        rec = _ParityRecorder(monkeypatch)
        self._fit_cs(_make_panel(n_cov=2, drop_frac=0.2), covs=["x1", "x2"])
        assert rec.n_calls > 0

    def test_inf_coded_never_treated(self, monkeypatch):
        rec = _ParityRecorder(monkeypatch)
        self._fit_cs(_make_panel(inf_coded=True))
        assert rec.n_calls > 0

    def test_bootstrap_prep_path(self, monkeypatch):
        """Bootstrap call sites (overall + per-horizon prep) go through the
        same parity wrapper."""
        rec = _ParityRecorder(monkeypatch)
        with pytest.warns(UserWarning, match="n_bootstrap"):
            self._fit_cs(_make_panel(), boot=19)
        assert rec.n_calls > 0

    def test_triple_diff_zeroed_cohorts(self, monkeypatch):
        """StaggeredTripleDifference aggregates through a shallow copy of
        precomputed with an eligibility-zeroed unit_cohorts - the fourth
        consumer of the mixin."""
        rec = _ParityRecorder(monkeypatch)
        std = StaggeredTripleDifference()
        std.fit(
            _make_ddd(),
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="all",
        )
        assert rec.n_calls > 0


class TestWifZeroContract:
    """Units whose cohort is not among the keeper groups get a wif
    contribution of exactly 0.0 on the fast path (the old dense form left an
    ~eps residue from cancelling terms)."""

    def test_non_keeper_cohort_exact_zero(self):
        n_units = 12
        # cohorts: 0 = never treated (not a keeper), 3 and 5 = keepers
        unit_cohorts = np.array([0, 0, 0, 0, 3, 3, 3, 5, 5, 5, 0, 3])
        precomputed = {
            "unit_cohorts": unit_cohorts,
            "is_panel": True,
            "unit_to_idx": {i: i for i in range(n_units)},
        }
        gt_pairs = [(3, 4), (5, 6)]
        # Zero-valued IFs isolate psi_wif: psi_total == wif / total_weight
        influence_func_info = {
            gt: {
                "treated_idx": np.where(unit_cohorts == gt[0])[0],
                "control_idx": np.where(unit_cohorts == 0)[0],
                "treated_inf": np.zeros(np.sum(unit_cohorts == gt[0])),
                "control_inf": np.zeros(np.sum(unit_cohorts == 0)),
            }
            for gt in gt_pairs
        }
        mixin = CallawaySantAnna()
        psi, _ = mixin._compute_combined_influence_function(
            gt_pairs,
            weights=np.array([0.5, 0.5]),
            effects=np.array([1.7, -0.9]),
            groups_for_gt=np.array([3, 5]),
            influence_func_info=influence_func_info,
            df=None,
            unit="unit",
            precomputed=precomputed,
            global_unit_to_idx=precomputed["unit_to_idx"],
            n_global_units=n_units,
        )
        non_keeper = unit_cohorts == 0
        assert np.all(psi[non_keeper] == 0.0)
        assert np.all(psi[~non_keeper] != 0.0)


class TestAggCache:
    """The per-fit cohort-table cache is validated by array identity and
    rebuilt into a fresh dict - the StaggeredTripleDifference shallow-copy
    pattern must never see stale tables."""

    def _precomputed(self, cohorts, sw=None):
        return {
            "unit_cohorts": cohorts,
            "survey_weights": sw,
            "is_panel": True,
            "unit_to_idx": {i: i for i in range(len(cohorts))},
        }

    def test_cache_hit_on_same_arrays(self):
        p = self._precomputed(np.array([0, 3, 3, 5]))
        c1 = CallawaySantAnnaAggregationMixin._get_agg_cache(p)
        c2 = CallawaySantAnnaAggregationMixin._get_agg_cache(p)
        assert c1 is c2
        assert c1["cohorts_ref"] is p["unit_cohorts"]

    def test_shallow_copy_with_swapped_cohorts_rebuilds(self):
        p = self._precomputed(np.array([0, 3, 3, 5]))
        c1 = CallawaySantAnnaAggregationMixin._get_agg_cache(p)
        # StaggeredTripleDifference pattern: dict() copy + replaced cohorts
        p2 = dict(p)
        p2["unit_cohorts"] = np.array([0, 0, 3, 5])  # eligibility-zeroed
        c2 = CallawaySantAnnaAggregationMixin._get_agg_cache(p2)
        assert c2 is not c1  # fresh dict, not in-place mutation
        assert c2["cohorts_ref"] is p2["unit_cohorts"]
        # original cache untouched and still valid for the original dict
        assert p["_agg_cache"] is c1
        assert CallawaySantAnnaAggregationMixin._get_agg_cache(p) is c1
        # masses reflect the respective cohort arrays
        assert not np.array_equal(c1["cohort_masses"], c2["cohort_masses"])

    def test_survey_weight_identity_checked(self):
        cohorts = np.array([0, 3, 3, 5])
        p = self._precomputed(cohorts, sw=np.array([1.0, 2.0, 1.0, 0.5]))
        c1 = CallawaySantAnnaAggregationMixin._get_agg_cache(p)
        p["survey_weights"] = np.array([1.0, 1.0, 1.0, 1.0])
        c2 = CallawaySantAnnaAggregationMixin._get_agg_cache(p)
        assert c2 is not c1
        assert c2["sw_ref"] is p["survey_weights"]


class TestFastPathTaken:
    """A broken dispatch guard would silently fall back to the general path
    and revert the entire optimization while every value-level test stays
    green - so pin that the fast path actually runs (and resolves) for the
    canonical panel fit."""

    def test_fast_path_taken_on_plain_panel_fit(self, monkeypatch):
        calls = {"resolved": 0, "fallback": 0}
        seen = {}
        orig = CallawaySantAnnaAggregationMixin._combined_if_fast

        def spy(
            self,
            gt_pairs,
            weights,
            effects,
            groups_for_gt,
            influence_func_info,
            precomputed,
            n_units,
        ):
            out = orig(
                self,
                gt_pairs,
                weights,
                effects,
                groups_for_gt,
                influence_func_info,
                precomputed,
                n_units,
            )
            calls["resolved" if out is not None else "fallback"] += 1
            seen["precomputed"] = precomputed
            return out

        monkeypatch.setattr(CallawaySantAnnaAggregationMixin, "_combined_if_fast", spy)
        cs = CallawaySantAnna(estimation_method="dr", seed=7)
        cs.fit(
            _make_panel(),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="all",
        )
        # every aggregation target routed through the fast path, none fell
        # back to the general (DataFrame-scanning) branch
        assert calls["resolved"] > 0
        assert calls["fallback"] == 0
        # the per-fit cache exists and pins the exact arrays it was built on
        cache = seen["precomputed"]["_agg_cache"]
        assert cache["cohorts_ref"] is seen["precomputed"]["unit_cohorts"]

    def _spy(self, monkeypatch):
        """Count fast-path dispatch outcomes (resolved vs fell-back)."""
        calls = {"resolved": 0, "fallback": 0}
        orig = CallawaySantAnnaAggregationMixin._combined_if_fast

        def spy(mixin_self, *args):
            out = orig(mixin_self, *args)
            calls["resolved" if out is not None else "fallback"] += 1
            return out

        monkeypatch.setattr(CallawaySantAnnaAggregationMixin, "_combined_if_fast", spy)
        return calls

    def test_rcs_fast_path_taken(self, monkeypatch):
        calls = self._spy(monkeypatch)
        cs = CallawaySantAnna(estimation_method="dr", panel=False, seed=7)
        with pytest.warns(UserWarning, match="stationary"):
            cs.fit(
                _make_rcs(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1", "x2"],
                aggregate="all",
            )
        assert calls["resolved"] > 0
        assert calls["fallback"] == 0

    def test_survey_fast_path_taken(self, monkeypatch):
        """Survey-weighted fits must dispatch through the fast path too - a
        silent fallback would pass every parity test while reverting the
        optimization for the whole survey family."""
        calls = self._spy(monkeypatch)
        cs = CallawaySantAnna(estimation_method="dr", seed=7)
        cs.fit(
            _make_panel(n_cov=2, survey=True),
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1", "x2"],
            aggregate="all",
            survey_design=SurveyDesign(weights="pw"),
        )
        assert calls["resolved"] > 0
        assert calls["fallback"] == 0

    def test_bootstrap_prep_fast_path_taken(self, monkeypatch):
        """Bootstrap prep (overall + per-horizon combined-IF calls) must
        dispatch through the fast path."""
        calls = self._spy(monkeypatch)
        cs = CallawaySantAnna(estimation_method="dr", n_bootstrap=19, seed=7)
        with pytest.warns(UserWarning, match="n_bootstrap"):
            cs.fit(
                _make_panel(),
                outcome="outcome",
                unit="unit",
                time="time",
                first_treat="first_treat",
                aggregate="all",
            )
        assert calls["resolved"] > 0
        assert calls["fallback"] == 0

    def test_triple_diff_fast_path_taken(self, monkeypatch):
        """StaggeredTripleDifference's shallow-copied precomputed (zeroed
        cohorts, no canonical_size key) must still resolve the fast path via
        the .get defaults."""
        calls = self._spy(monkeypatch)
        std = StaggeredTripleDifference()
        std.fit(
            _make_ddd(),
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="all",
        )
        assert calls["resolved"] > 0
        assert calls["fallback"] == 0
