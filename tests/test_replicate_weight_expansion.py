"""Tests for replicate weight support expansion to 7 additional estimators."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    ImputationDiD,
    MultiPeriodDiD,
    StackedDiD,
    SunAbraham,
    TwoStageDiD,
    TwoWayFixedEffects,
)
from diff_diff.survey import SurveyDesign

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_panel():
    """2-period panel for DiD/MultiPeriodDiD (treatment/post binary columns)."""
    np.random.seed(123)
    n_units = 40
    rows = []
    for i in range(n_units):
        treated = 1 if i < 20 else 0
        wt = 1.0 + 0.2 * (i % 5)
        for t in [0, 1]:
            y = 5.0 + 0.5 * treated + 1.0 * t
            if treated and t == 1:
                y += 2.0  # ATT = 2
            y += np.random.normal(0, 0.3)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "treated": treated,
                    "post": t,
                    "outcome": y,
                    "weight": wt,
                }
            )
    data = pd.DataFrame(rows)
    return data


def _make_staggered_panel():
    """Multi-period staggered panel for TWFE/SA/Stacked/Imputation/TwoStage."""
    np.random.seed(456)
    n_units, n_periods = 50, 8
    rows = []
    for i in range(n_units):
        if i < 15:
            ft = 4  # cohort 1
        elif i < 30:
            ft = 6  # cohort 2
        else:
            ft = 0  # never-treated
        wt = 1.0 + 0.3 * (i % 5)
        for t in range(1, n_periods + 1):
            y = 10.0 + i * 0.03 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0
            y += np.random.normal(0, 0.4)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": wt,
                    "treated": 1 if ft > 0 else 0,
                    "post": 1 if ft > 0 and t >= ft else 0,
                }
            )
    data = pd.DataFrame(rows)
    return data


def _add_jk1_replicates(data, n_rep=15, unit_col="unit"):
    """Add JK1 (delete-cluster jackknife) replicate weight columns."""
    units = sorted(data[unit_col].unique())
    cluster_size = max(1, len(units) // n_rep)
    rep_cols = []
    for r in range(n_rep):
        start = r * cluster_size
        end = min((r + 1) * cluster_size, len(units))
        deleted_units = set(units[start:end])
        w_r = data["weight"].values.copy()
        mask = data[unit_col].isin(deleted_units).values
        w_r[mask] = 0.0
        w_r[~mask] *= n_rep / (n_rep - 1)
        col = f"rep_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return rep_cols


def _add_brr_replicates(data, n_rep=16, unit_col="unit"):
    """Add BRR replicate weight columns (random sign perturbation)."""
    rng = np.random.RandomState(789)
    units = sorted(data[unit_col].unique())
    rep_cols = []
    for r in range(n_rep):
        signs = rng.choice([-1, 1], size=len(units))
        sign_map = dict(zip(units, signs))
        perturbation = data[unit_col].map(sign_map).values.astype(float)
        # BRR: w_r = w * (1 + epsilon) / 2, where epsilon in {-1, 1}
        # Simplified: w_r = w * (1 + perturbation) (combined_weights=True style)
        w_r = data["weight"].values * (1.0 + 0.5 * perturbation)
        w_r = np.maximum(w_r, 0.0)
        col = f"brr_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return rep_cols


# ---------------------------------------------------------------------------
# Smoke tests — each estimator × {JK1, BRR}
# ---------------------------------------------------------------------------


class TestDiDReplicate:
    """DifferenceInDifferences with replicate weights."""

    def test_did_jk1(self):
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = DifferenceInDifferences().fit(
            data,
            "outcome",
            "treated",
            "post",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_did_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = DifferenceInDifferences().fit(
            data,
            "outcome",
            "treated",
            "post",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0

    def test_did_wild_bootstrap_rejected(self):
        """Wild bootstrap + survey is rejected before replicate check."""
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises((ValueError, NotImplementedError)):
            DifferenceInDifferences(inference="wild_bootstrap", cluster="unit").fit(
                data,
                "outcome",
                "treated",
                "post",
                survey_design=sd,
            )


class TestDiDAbsorbReplicate:
    """DiD absorb path with replicate weights."""

    def test_did_absorb_brr(self):
        """Absorb path should produce finite replicate SE."""
        data = _make_simple_panel()
        # Add a group column for absorb
        data["group"] = (data["unit"] % 4).astype(str)
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = DifferenceInDifferences().fit(
            data,
            "outcome",
            "treated",
            "post",
            absorb=["group"],
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0


class TestMultiPeriodDiDReplicate:
    """MultiPeriodDiD with replicate weights."""

    def test_multiperiod_jk1(self):
        data = _make_simple_panel()
        rep_cols = _add_jk1_replicates(data, n_rep=10)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = MultiPeriodDiD().fit(
            data,
            "outcome",
            "treated",
            "time",
            post_periods=[1],
            survey_design=sd,
        )
        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se) and result.avg_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_multiperiod_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = MultiPeriodDiD().fit(
            data,
            "outcome",
            "treated",
            "time",
            post_periods=[1],
            survey_design=sd,
        )
        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se) and result.avg_se > 0


class TestTWFEReplicate:
    """TwoWayFixedEffects with replicate weights."""

    @staticmethod
    def _make_twfe_panel():
        """Balanced 2-period panel with variation in treatment timing."""
        np.random.seed(321)
        n_units = 40
        rows = []
        for i in range(n_units):
            treated = 1 if i < 20 else 0
            wt = 1.0 + 0.2 * (i % 5)
            for t in [0, 1]:
                y = 5.0 + i * 0.05 + t * 1.0
                if treated and t == 1:
                    y += 2.0
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "treated": treated,
                        "post": t,
                        "outcome": y,
                        "weight": wt,
                    }
                )
        return pd.DataFrame(rows)

    def test_twfe_brr(self):
        """BRR works well with TWFE: perturbation doesn't zero out units."""
        data = self._make_twfe_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = TwoWayFixedEffects().fit(
            data,
            "outcome",
            "treated",
            "post",
            "unit",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_twfe_brr_larger(self):
        """Second BRR test with different seed."""
        data = self._make_twfe_panel()
        rep_cols = _add_brr_replicates(data, n_rep=20)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = TwoWayFixedEffects().fit(
            data,
            "outcome",
            "treated",
            "post",
            "unit",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se) and result.se > 0


class TestSunAbrahamReplicate:
    """SunAbraham with replicate weights."""

    def test_sun_abraham_brr(self):
        """BRR replicates are less aggressive than JK1 for SunAbraham."""
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = SunAbraham(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_sun_abraham_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        with pytest.raises(ValueError, match="n_bootstrap"):
            SunAbraham(n_bootstrap=100).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )


class TestStackedDiDReplicate:
    """StackedDiD with replicate weights."""

    def test_stacked_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = StackedDiD().fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_stacked_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = StackedDiD().fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0


class TestImputationDiDReplicate:
    """ImputationDiD with replicate weights."""

    def test_imputation_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = ImputationDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_imputation_event_study_replicate(self):
        """Event-study with replicate weights: overall ATT SE must be finite,
        and at least some per-period SEs should be finite."""
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = ImputationDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.event_study_effects is not None
        # At least some identified periods should have finite SE
        finite_ses = [
            e
            for e, eff in result.event_study_effects.items()
            if np.isfinite(eff["effect"]) and np.isfinite(eff["se"]) and eff["se"] > 0
        ]
        assert len(finite_ses) > 0, "No event-study periods have finite replicate SE"

    def test_imputation_group_replicate(self):
        """Group SEs should use replicate variance."""
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = ImputationDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="group",
            survey_design=sd,
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"group {g}: SE not finite"

    def test_imputation_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises(ValueError, match="n_bootstrap"):
            ImputationDiD(n_bootstrap=100).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )


class TestTwoStageDiDReplicate:
    """TwoStageDiD with replicate weights."""

    def test_two_stage_jk1(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = TwoStageDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se) and result.overall_se > 0
        assert result.survey_metadata is not None
        result.summary()

    def test_two_stage_event_study_replicate(self):
        """Event-study SEs should use replicate variance, not GMM SE."""
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = TwoStageDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="event_study",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        non_ref = {e: eff for e, eff in result.event_study_effects.items() if eff["effect"] != 0.0}
        assert len(non_ref) > 0, "No non-reference event-study effects"
        for e, eff in non_ref.items():
            assert np.isfinite(eff["se"]) and eff["se"] > 0, f"period {e}: SE not finite"

    def test_two_stage_always_treated(self):
        """Replicate weights subsetted to post-always-treated-drop sample.

        Wave E.3 parity: the main fit retains full-domain `resolved_survey`
        but subsets `survey_weights` to the post-drop OLS sample. The
        replicate refit callback receives FULL-DOMAIN replicate weights and
        must apply the SAME `keep_mask` subsetting before threading into
        stage-1 / stage-2. Without the subset, `solve_ols` rejects the
        length mismatch and `compute_replicate_refit_variance` swallows the
        ValueError so replicate inference NaNs out. This test exercises the
        full replicate variance pipeline (not just the point estimate) under
        the always-treated drop to lock the parity contract end-to-end.
        """
        # Seeded RNG for deterministic always-treated outcomes (regression
        # tests should not depend on numpy global RNG state at import time).
        rng = np.random.default_rng(101)
        data = _make_staggered_panel()
        # Add always-treated units (first_treat <= min time)
        for i in range(50, 55):
            for t in range(1, 9):
                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            [
                                {
                                    "unit": i,
                                    "time": t,
                                    "first_treat": 1,
                                    "outcome": 12.0 + rng.normal(0, 0.3),
                                    "weight": 1.5,
                                    "treated": 1,
                                    "post": 1,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        rep_cols = _add_jk1_replicates(data, n_rep=10, unit_col="unit")
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        # Should not crash despite always-treated unit exclusion
        result = TwoStageDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        # ATT comes from the main fit (always finite once always-treated drop runs)
        assert np.isfinite(result.overall_att)
        # SE comes from the replicate refit variance: requires the refit
        # callback to align replicate weights with the post-drop sample.
        # Pre-Wave-E.3-parity-fix, replicate refits raised ValueError on
        # length mismatch, `compute_replicate_refit_variance` swallowed
        # them, and `overall_se` came out NaN with all replicate-based
        # inference fields NaN.
        assert np.isfinite(result.overall_se), (
            "Replicate SE must be finite under always-treated drop. "
            "If NaN, the replicate refit callback is failing to align "
            "weights with the post-drop sample — Wave E.3 parity bug."
        )
        assert np.isfinite(result.overall_p_value)
        assert result.overall_conf_int is not None
        assert np.all(np.isfinite(result.overall_conf_int))

    def test_two_stage_always_treated_event_study_and_group_replicate(self):
        """Replicate refit covers event-study + group surfaces under
        always-treated drop. Companion to ``test_two_stage_always_treated``
        which asserts the overall ATT surface only; the Wave E.3 parity
        fix to ``_refit_ts`` aligns ``w_r`` with ``keep_mask`` for ALL
        three stage-2 surfaces (``_stage2_static`` / ``_stage2_event_study``
        / ``_stage2_group``), so this test exercises the event-study +
        group replicate refit branches end-to-end with the same
        always-treated fixture."""
        # Seeded RNG for deterministic always-treated outcomes.
        rng = np.random.default_rng(202)
        data = _make_staggered_panel()
        for i in range(50, 55):
            for t in range(1, 9):
                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            [
                                {
                                    "unit": i,
                                    "time": t,
                                    "first_treat": 1,
                                    "outcome": 12.0 + rng.normal(0, 0.3),
                                    "weight": 1.5,
                                    "treated": 1,
                                    "post": 1,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        rep_cols = _add_jk1_replicates(data, n_rep=10, unit_col="unit")
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        result = TwoStageDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            aggregate="all",
            survey_design=sd,
        )
        # Event-study surface: at least one non-reference horizon must have
        # replicate-derived finite SE / p_value / conf_int (the replicate
        # override path at `two_stage.py` updates SE + t_stat + p_value +
        # conf_int separately for each non-reference horizon via the same
        # _refit_ts callback, so all four fields must be locked).
        assert result.event_study_effects is not None
        non_ref_es = {
            e: eff
            for e, eff in result.event_study_effects.items()
            if eff["effect"] != 0.0 and np.isfinite(eff["effect"])
        }
        assert len(non_ref_es) > 0, "no non-reference event-study effects"
        for e, eff in non_ref_es.items():
            assert np.isfinite(eff["se"]) and eff["se"] > 0, (
                f"event-study horizon {e}: replicate SE must be finite "
                f"under always-treated drop"
            )
            assert np.isfinite(
                eff["p_value"]
            ), f"event-study horizon {e}: replicate p_value must be finite"
            assert eff["conf_int"] is not None and np.all(
                np.isfinite(eff["conf_int"])
            ), f"event-study horizon {e}: replicate conf_int bounds must be finite"
        # Group surface: at least one cohort with finite replicate SE /
        # p_value / conf_int (same override path applies to group effects).
        assert result.group_effects is not None
        finite_groups = {
            g: eff for g, eff in result.group_effects.items() if np.isfinite(eff["effect"])
        }
        assert len(finite_groups) > 0, "no finite group effects"
        for g, eff in finite_groups.items():
            assert np.isfinite(eff["se"]) and eff["se"] > 0, (
                f"cohort {g}: replicate SE must be finite under " f"always-treated drop"
            )
            assert np.isfinite(eff["p_value"]), f"cohort {g}: replicate p_value must be finite"
            assert eff["conf_int"] is not None and np.all(
                np.isfinite(eff["conf_int"])
            ), f"cohort {g}: replicate conf_int bounds must be finite"

    def test_two_stage_bootstrap_rejected(self):
        data = _make_staggered_panel()
        rep_cols = _add_jk1_replicates(data)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="JK1")
        with pytest.raises(ValueError, match="n_bootstrap"):
            TwoStageDiD(n_bootstrap=100).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )


class TestSunAbrahamCohortSEs:
    """SunAbraham cohort-level SEs should be consistent with replicate vcov."""

    def test_cohort_ses_finite(self):
        """Cohort SEs should be recomputed from replicate vcov, not stale."""
        data = _make_staggered_panel()
        rep_cols = _add_brr_replicates(data, n_rep=16)
        sd = SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")
        result = SunAbraham(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.cohort_effects is not None
        for key, eff in result.cohort_effects.items():
            se = eff["se"]
            assert np.isfinite(se), f"cohort {key}: SE is {se}"


class TestReplicateVcovTypeWarn:
    """Explicit non-hc1 vcov_type + replicate design warns and proceeds with
    replicate variance, bit-identical to the hc1 request (the analytical
    vcov family cannot influence any number: per-replicate refits return
    point estimates only). Previously DiD silently ignored the kwarg and
    TWFE raised NotImplementedError — the warn-and-remap contract unifies
    the twins."""

    @staticmethod
    def _sd(rep_cols):
        return SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")

    def test_did_explicit_hc2_warns_and_matches_hc1(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=8)
        with pytest.warns(UserWarning, match="has no effect with replicate-weight"):
            res = DifferenceInDifferences(vcov_type="hc2").fit(
                data, "outcome", "treated", "post", survey_design=self._sd(rep_cols)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            base = DifferenceInDifferences(vcov_type="hc1").fit(
                data, "outcome", "treated", "post", survey_design=self._sd(rep_cols)
            )
        assert res.att == base.att and res.se == base.se

    def test_did_default_vcov_stays_silent(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = DifferenceInDifferences().fit(
                data, "outcome", "treated", "post", survey_design=self._sd(rep_cols)
            )
        assert np.isfinite(res.se) and res.se > 0

    def test_multiperiod_explicit_hc2_warns_and_matches_hc1(self):
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=8)
        with pytest.warns(UserWarning, match="has no effect with replicate-weight"):
            res = MultiPeriodDiD(vcov_type="hc2").fit(
                data,
                "outcome",
                "treated",
                "time",
                post_periods=[1],
                survey_design=self._sd(rep_cols),
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            base = MultiPeriodDiD(vcov_type="hc1").fit(
                data,
                "outcome",
                "treated",
                "time",
                post_periods=[1],
                survey_design=self._sd(rep_cols),
            )
        # explicit hc1 must NOT trigger the replicate-override warning
        # (the fixture legitimately emits an unrelated pre-period warning).
        assert not any("has no effect with replicate-weight" in str(x.message) for x in w)
        assert res.avg_att == base.avg_att and res.avg_se == base.avg_se

    def test_conley_keeps_own_survey_contract(self):
        """conley is excluded from the warn-and-remap: its survey-design
        validators keep firing unchanged (no misleading 'has no effect'
        warning followed by a conley rejection)."""
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=8)
        rng = np.random.default_rng(0)
        data["lat"] = rng.uniform(0, 10, len(data))
        data["lon"] = rng.uniform(0, 10, len(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ValueError):
                DifferenceInDifferences(
                    vcov_type="conley",
                    conley_coords=("lat", "lon"),
                    conley_cutoff_km=100,
                ).fit(data, "outcome", "treated", "post", survey_design=self._sd(rep_cols))
        assert not any("has no effect with replicate-weight" in str(x.message) for x in w)

    def test_did_absorb_replicate_hc2_matches_hc1_surface_and_numbers(self):
        """CI-review P1 regression: under a replicate design the remap must
        also disable the absorb->full-dummy swap, or explicit hc2 would still
        change the result surface (full-dummy coefficient vector vs absorbed
        reduced fit) despite the 'has no effect' warning."""
        data = _make_simple_panel()
        data["group"] = (data["unit"] % 4).astype(str)
        rep_cols = _add_brr_replicates(data, n_rep=8)

        def _fit(vc):
            return DifferenceInDifferences(vcov_type=vc).fit(
                data,
                "outcome",
                "treated",
                "post",
                absorb=["group"],
                survey_design=self._sd(rep_cols),
            )

        with pytest.warns(UserWarning, match="has no effect with replicate-weight"):
            res = _fit("hc2")
        base = _fit("hc1")
        assert res.att == base.att and res.se == base.se
        # Same result surface: absorbed reduced fit, not the full-dummy swap
        # (the swap would expose the group-dummy coefficients).
        assert len(res.coefficients) == len(base.coefficients)

    def test_wild_bootstrap_rejection_fires_before_vcov_warning(self):
        """CI-review P3 regression: the wild_bootstrap x replicate rejection
        must fire without first emitting a 'proceeding with replicate
        variance' warning that the subsequent raise contradicts."""
        data = _make_simple_panel()
        rep_cols = _add_brr_replicates(data, n_rep=8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # The survey resolver rejects wild_bootstrap x survey even before
            # the estimator-level replicate raise; either way the contract is
            # a clean rejection with NO preceding 'proceeding with replicate
            # variance' warning.
            with pytest.raises((ValueError, NotImplementedError), match="ild bootstrap"):
                DifferenceInDifferences(vcov_type="hc2", inference="wild_bootstrap").fit(
                    data, "outcome", "treated", "post", survey_design=self._sd(rep_cols)
                )
        assert not any("has no effect with replicate-weight" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# TRUE half-sample BRR (Hadamard-balanced) — per-estimator-family regressions
# ---------------------------------------------------------------------------


def _add_true_brr_replicates(data, unit_col="unit"):
    """Add TRUE half-sample BRR replicate columns (Hadamard-balanced).

    Pairs consecutive units into 2-PSU pseudo-strata and assigns
    half-samples from a Sylvester-Hadamard matrix: within each pair the
    selected PSU gets ``w*2`` and the other ``0`` (Wolter 2007 ch. 3; the
    R ``survey::brrweights`` full-BRR convention with
    ``combined_weights=True``). Unlike :func:`_add_brr_replicates`'
    Fay-like 0.5/1.5 perturbation — under which every unit keeps positive
    weight — every replicate here is a genuine half-sample: half the
    paired PSUs carry exactly zero weight, exercising the zero-weight-PSU
    code paths (FE identification drops, zero-mass cells) through each
    estimator's replicate refit. An odd trailing unit (if any) is kept at
    its base weight in every replicate (certainty PSU).
    """
    from scipy.linalg import hadamard

    units = sorted(data[unit_col].unique())
    n_paired = len(units) - (len(units) % 2)
    pairs = [(units[i], units[i + 1]) for i in range(0, n_paired, 2)]
    n_strata = len(pairs)
    # Sylvester-Hadamard column 0 is all +1, which would leave the first
    # pseudo-stratum permanently unbalanced (the same PSU selected in every
    # replicate); survey::brrweights skips the constant column, so strata
    # map to columns 1..n_strata — hence R >= n_strata + 1.
    n_rep = 4
    while n_rep < n_strata + 1:
        n_rep *= 2
    H = hadamard(n_rep)
    base = data["weight"].to_numpy(dtype=float)
    unit_vals = data[unit_col].to_numpy()
    rep_cols = []
    for r in range(n_rep):
        w_r = base.copy()
        for h, (u1, u2) in enumerate(pairs):
            selected, dropped = (u1, u2) if H[r, h + 1] == 1 else (u2, u1)
            w_r[unit_vals == selected] *= 2.0
            w_r[unit_vals == dropped] = 0.0
        col = f"tbrr_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return rep_cols


class TestTrueBRRHalfSample:
    """TRUE half-sample BRR per estimator family (TODO row: the smoke tests
    above use Fay-like 0.5/1.5 perturbations, which never zero a unit;
    ``test_survey_phase6.py`` covers true BRR only at the vcov-helper level).

    Each test asserts (a) finite positive replicate SE under genuine
    half-samples — half the paired PSUs at weight 0 per replicate — and
    (b) the point estimate is IDENTICAL to the same fit
    without replicate columns: replicate weights drive only the variance,
    never the point estimate (base-weights invariance contract).
    """

    @staticmethod
    def _sd(rep_cols):
        return SurveyDesign(weights="weight", replicate_weights=rep_cols, replicate_method="BRR")

    def test_construction_is_true_half_sample(self):
        data = _make_simple_panel()
        rep_cols = _add_true_brr_replicates(data)
        units = sorted(data["unit"].unique())
        n_paired = len(units) - (len(units) % 2)
        base_per_unit = data.groupby("unit")["weight"].first()
        for col in rep_cols:
            per_unit = data.groupby("unit")[col].first()
            zeroed = int((per_unit.loc[units[:n_paired]] == 0.0).sum())
            assert zeroed == n_paired // 2, f"{col}: {zeroed} != {n_paired // 2}"
            mult = per_unit.loc[units[:n_paired]] / base_per_unit.loc[units[:n_paired]]
            assert set(np.round(mult.unique(), 12)) == {0.0, 2.0}
        # Hadamard BALANCE: every paired PSU is selected (w*2) in exactly
        # half the replicates — the property the constant Hadamard column
        # would break for the first pseudo-stratum (each column 1..H of a
        # Sylvester matrix has zero sum, so mean multiplier is exactly 1).
        rep_mat = np.column_stack(
            [data.groupby("unit")[c].first().loc[units[:n_paired]] for c in rep_cols]
        )
        base_vec = base_per_unit.loc[units[:n_paired]].to_numpy()
        mean_mult = rep_mat.mean(axis=1) / base_vec
        np.testing.assert_allclose(mean_mult, 1.0, rtol=0, atol=1e-12)

    @staticmethod
    def _assert_point_invariance(att_rep, att_base):
        np.testing.assert_allclose(att_rep, att_base, rtol=0, atol=1e-12)

    def test_did_true_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = DifferenceInDifferences().fit(
            data, "outcome", "treated", "post", survey_design=self._sd(rep_cols)
        )
        base = DifferenceInDifferences().fit(
            data, "outcome", "treated", "post", survey_design=SurveyDesign(weights="weight")
        )
        assert np.isfinite(res.se) and res.se > 0
        self._assert_point_invariance(res.att, base.att)

    def test_did_absorb_true_brr(self):
        data = _make_simple_panel()
        data["group"] = (data["unit"] % 4).astype(str)
        rep_cols = _add_true_brr_replicates(data)
        res = DifferenceInDifferences().fit(
            data,
            "outcome",
            "treated",
            "post",
            absorb=["group"],
            survey_design=self._sd(rep_cols),
        )
        base = DifferenceInDifferences().fit(
            data,
            "outcome",
            "treated",
            "post",
            absorb=["group"],
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.se) and res.se > 0
        self._assert_point_invariance(res.att, base.att)

    def test_multiperiod_true_brr(self):
        data = _make_simple_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = MultiPeriodDiD().fit(
            data,
            "outcome",
            "treated",
            "time",
            post_periods=[1],
            survey_design=self._sd(rep_cols),
        )
        base = MultiPeriodDiD().fit(
            data,
            "outcome",
            "treated",
            "time",
            post_periods=[1],
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.avg_se) and res.avg_se > 0
        self._assert_point_invariance(res.avg_att, base.avg_att)

    def test_twfe_true_brr(self):
        # The family's dedicated 2-period panel (the staggered fixture's
        # binary treated x post parameterization can lose identification
        # inside a genuine half-sample replicate refit — a real behavior
        # of true BRR that the Fay-like perturbation never exercised;
        # TWFE fails loudly there rather than silently degrading).
        data = TestTWFEReplicate._make_twfe_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = TwoWayFixedEffects().fit(
            data, "outcome", "treated", "post", "unit", survey_design=self._sd(rep_cols)
        )
        base = TwoWayFixedEffects().fit(
            data,
            "outcome",
            "treated",
            "post",
            "unit",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.se) and res.se > 0
        self._assert_point_invariance(res.att, base.att)

    def test_sun_abraham_true_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = SunAbraham(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=self._sd(rep_cols)
        )
        base = SunAbraham(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        self._assert_point_invariance(res.overall_att, base.overall_att)

    def test_stacked_true_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = StackedDiD().fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=self._sd(rep_cols)
        )
        base = StackedDiD().fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        self._assert_point_invariance(res.overall_att, base.overall_att)

    def test_imputation_true_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = ImputationDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=self._sd(rep_cols)
        )
        base = ImputationDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        self._assert_point_invariance(res.overall_att, base.overall_att)

    def test_two_stage_true_brr(self):
        data = _make_staggered_panel()
        rep_cols = _add_true_brr_replicates(data)
        res = TwoStageDiD(n_bootstrap=0).fit(
            data, "outcome", "unit", "time", "first_treat", survey_design=self._sd(rep_cols)
        )
        base = TwoStageDiD(n_bootstrap=0).fit(
            data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=SurveyDesign(weights="weight"),
        )
        assert np.isfinite(res.overall_se) and res.overall_se > 0
        self._assert_point_invariance(res.overall_att, base.overall_att)
