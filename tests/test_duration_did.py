"""API, validation, inference-contract and results-surface tests for DurationDiD.

Covers constructor validation, the sklearn-style parameter surface, fit
input validation (panel hygiene, binary/absorbing outcome, time grid,
fitting selectors), the joint-NaN inference contract, bootstrap seeding and
the per-family draw-failure policy, the curve-validity gate, every warning,
the results API (summary/to_dict/to_dataframe/aggregate), and the reporting
consumers' contract. Paper identities live in
``tests/test_methodology_duration_did.py``.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff.duration_did as dd_module
from diff_diff import (
    BusinessReport,
    DiagnosticReport,
    DurationDiD,
    DurationDiDPretestResults,
    DurationDiDResults,
)
from diff_diff.practitioner import practitioner_next_steps
from diff_diff.results_base import EventStudyResults
from tests.conftest import assert_nan_inference

# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def build_from_survivors(n1, s1, n2, s2, times=None):
    """Long absorbing panel from per-date survivor COUNTS (treated, control).

    Unit ``i`` of a group survives date ``t`` iff ``i < survivors[t]`` (so
    counts must be nonincreasing). ``n`` above the first count means
    baseline absorption.
    """
    n_periods = len(s1)
    times = np.arange(1, n_periods + 1) if times is None else np.asarray(times)
    rows = []
    uid = 0
    for g, n, surv in ((1, n1, s1), (0, n2, s2)):
        for i in range(n):
            for t in range(n_periods):
                rows.append((uid, times[t], g, int(i >= surv[t])))
            uid += 1
    return pd.DataFrame(rows, columns=["unit", "time", "treated", "exited"])


def micro_panel(times=None):
    """The hand-computed micro panel: CD holds exactly with exp(-c) = 2/3."""
    return build_from_survivors(30, [18, 9, 4, 1, 0], 10, [8, 6, 4, 3, 2], times=times)


def simulate_panel(n=400, n_periods=8, tstar_idx=3, c=0.05, beta=0.3, seed=0, method="cd"):
    """Exact-population absorbing DGP (one uniform per unit)."""
    rng = np.random.default_rng(seed)
    s = np.arange(2, n_periods + 1)
    lam2 = 0.15 + 0.02 * s
    lam1 = lam2 + c if method == "cd" else c * lam2
    lam1_fact = lam1.copy()
    lam1_fact[tstar_idx:] += beta
    S2 = 0.8 * np.exp(-np.concatenate([[0.0], np.cumsum(lam2)]))
    S1 = 0.6 * np.exp(-np.concatenate([[0.0], np.cumsum(lam1_fact)]))
    u = rng.uniform(size=2 * n)
    g = np.repeat([1, 0], n)
    S_fact = np.where(g[:, None] == 1, S1[None, :], S2[None, :])
    Y = (u[:, None] > S_fact).astype(int)
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(2 * n), n_periods),
            "time": np.tile(np.arange(1, n_periods + 1), 2 * n),
            "treated": np.repeat(g, n_periods),
            "exited": Y.ravel(),
        }
    )


FIT_KW = dict(outcome="exited", unit="unit", time="time", treatment="treated")


def fit_quiet(est, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, **FIT_KW, **kw)


@pytest.fixture(scope="module")
def sim_df():
    return simulate_panel()


@pytest.fixture(scope="module")
def fitted(sim_df):
    return fit_quiet(DurationDiD(n_bootstrap=60, seed=3), sim_df, last_pre_period=4)


# ---------------------------------------------------------------------------
# Constructor validation and parameter surface
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    @pytest.mark.parametrize("bad", ["not_a_method", "CD", 1, None])
    def test_method_rejected(self, bad):
        with pytest.raises(ValueError, match="method must be 'cd' or 'ph'"):
            DurationDiD(method=bad)

    @pytest.mark.parametrize("bad", [-1, 2.5, None, True])
    def test_n_bootstrap_shared_message(self, bad):
        with pytest.raises(ValueError, match="n_bootstrap must be a non-negative integer"):
            DurationDiD(n_bootstrap=bad)

    def test_n_bootstrap_one_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            DurationDiD(n_bootstrap=1)

    def test_n_bootstrap_zero_and_two_legal(self):
        assert DurationDiD(n_bootstrap=0).n_bootstrap == 0
        assert DurationDiD(n_bootstrap=2).n_bootstrap == 2

    @pytest.mark.parametrize("bad", [0.0, 1.0, 5.0, "0.05", True])
    def test_alpha_rejected(self, bad):
        with pytest.raises(ValueError, match="alpha"):
            DurationDiD(alpha=bad)

    @pytest.mark.parametrize("bad", [-1, 1.5, "7", True])
    def test_seed_rejected(self, bad):
        with pytest.raises(ValueError, match="seed"):
            DurationDiD(seed=bad)


class TestParamSurface:
    def test_get_params_round_trip(self):
        est = DurationDiD(method="ph", n_bootstrap=10, alpha=0.1, seed=5)
        params = est.get_params()
        assert params == {"method": "ph", "n_bootstrap": 10, "alpha": 0.1, "seed": 5}
        assert DurationDiD(**params).get_params() == params

    def test_set_params_transactional(self):
        est = DurationDiD()
        with pytest.raises(ValueError):
            est.set_params(method="bogus", alpha=0.2)
        assert est.method == "cd" and est.alpha == 0.05
        est.set_params(method="ph")
        assert est.method == "ph"

    @pytest.mark.parametrize(
        "attr, bad, msg",
        [
            ("method", "typo", "method must be 'cd' or 'ph'"),
            ("method", "PH", "method must be 'cd' or 'ph'"),
            ("n_bootstrap", 1, "at least 2"),
            ("n_bootstrap", -5, "n_bootstrap must be a non-negative integer"),
            ("alpha", 1.5, "alpha"),
            ("alpha", "0.05", "alpha"),
            ("seed", -1, "seed"),
            ("seed", 2.5, "seed"),
        ],
    )
    def test_direct_attribute_mutation_rejected_at_fit(self, attr, bad, msg):
        # Bypassing set_params must not reach an estimation branch: an unknown
        # method would otherwise fall through to PH.
        est = DurationDiD(n_bootstrap=0)
        setattr(est, attr, bad)
        with pytest.raises(ValueError, match=msg):
            fit_quiet(est, micro_panel(), last_pre_period=3)

    def test_mutation_is_checked_before_any_data_work(self):
        # The configuration error wins over a data error, proving the check
        # runs before the panel is read.
        est = DurationDiD(n_bootstrap=0)
        est.method = "typo"
        with pytest.raises(ValueError, match="method must be 'cd' or 'ph'"):
            est.fit(pd.DataFrame(), "exited", "unit", "time", "treated", last_pre_period=3)
        est.method = "cd"
        with pytest.raises(ValueError, match="outcome column 'exited' not found"):
            est.fit(pd.DataFrame(), "exited", "unit", "time", "treated", last_pre_period=3)

    def test_selectors_are_fit_time_not_params(self):
        assert "pre_periods" not in DurationDiD().get_params()
        assert "last_pre_period" not in DurationDiD().get_params()


# ---------------------------------------------------------------------------
# Fit validation
# ---------------------------------------------------------------------------


class TestFitValidation:
    def test_missing_column(self):
        df = micro_panel()
        with pytest.raises(ValueError, match="outcome column 'nope' not found"):
            DurationDiD(n_bootstrap=0).fit(df, "nope", "unit", "time", "treated", last_pre_period=3)

    def test_missing_unit_identifier(self):
        df = micro_panel().astype({"unit": float})
        df.loc[df.index[:2], "unit"] = np.nan
        with pytest.raises(ValueError, match="unit column 'unit' contains missing values"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_datetime_time_rejected(self):
        df = micro_panel()
        df["time"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df["time"], unit="D")
        with pytest.raises(ValueError, match="must be numeric; convert datetime"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_string_time_rejected(self):
        df = micro_panel()
        df["time"] = df["time"].astype(str)
        with pytest.raises(ValueError, match="time column 'time' must be numeric"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    @pytest.mark.parametrize("n_dates", [0, 1, 2])
    def test_fewer_than_three_dates(self, n_dates):
        df = micro_panel()
        df = df[df["time"] <= n_dates] if n_dates else df.iloc[0:0]
        with pytest.raises(ValueError, match="at least three distinct time periods"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=1)

    def test_duplicate_cells(self):
        df = pd.concat([micro_panel(), micro_panel().iloc[:1]])
        with pytest.raises(ValueError, match="exactly one row per"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_unbalanced_panel(self):
        df = micro_panel().iloc[1:]
        with pytest.raises(ValueError, match="Unbalanced panel"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_unequal_spacing(self):
        df = micro_panel(times=[1, 2, 3, 5, 6])
        with pytest.raises(ValueError, match="equally spaced"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_small_magnitude_unequal_spacing_rejected(self):
        # Relative check with atol=0: numpy's default atol=1e-8 would accept
        # a 100% spacing difference on a denormal-scale grid.
        df = micro_panel(times=[0.0, 1e-12, 3e-12, 4e-12, 5e-12])
        with pytest.raises(ValueError, match="equally spaced"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3e-12)

    def test_string_outcome_rejected(self):
        df = micro_panel()
        df["exited"] = df["exited"].map({0: "no", 1: "yes"})
        with pytest.raises(ValueError, match="outcome column 'exited' must be a numeric 0/1"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_string_treatment_rejected(self):
        df = micro_panel()
        df["treated"] = df["treated"].map({0: "control", 1: "treated"})
        with pytest.raises(ValueError, match="treatment column 'treated' must be a numeric"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    @pytest.mark.parametrize("col", ["exited", "treated"])
    def test_numeric_string_columns_rejected(self, col):
        # Object-dtype "0"/"1" is never coerced (REGISTRY: non-numeric binary
        # columns raise).
        df = micro_panel()
        df[col] = df[col].astype(str)
        with pytest.raises(ValueError, match=f"column '{col}' must be a numeric 0/1 column"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_bool_columns_accepted(self):
        df = micro_panel()
        df["exited"] = df["exited"].astype(bool)
        df["treated"] = df["treated"].astype(bool)
        r = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)
        assert r.coefficient == pytest.approx(math.log(1.5))

    def test_nan_outcome_cell_rejected(self):
        # utils.validate_binary alone strips NaN before its membership test.
        df = micro_panel().astype({"exited": float})
        df.loc[df.index[7], "exited"] = np.nan
        with pytest.raises(ValueError, match="outcome column 'exited' contains 1 missing"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_nan_treatment_rejected(self):
        df = micro_panel().astype({"treated": float})
        df.loc[df.index[0], "treated"] = np.nan
        with pytest.raises(ValueError, match="treatment column 'treated' contains 1 missing"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_nonbinary_outcome(self):
        df = micro_panel()
        df.loc[df.index[0], "exited"] = 2
        with pytest.raises(ValueError, match="outcome must be binary"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_absorption_reversal(self):
        df = micro_panel()
        # unit 29 (treated, absorbed at date 1) -> revive at date 3
        df.loc[(df["unit"] == 29) & (df["time"] == 3), "exited"] = 0
        with pytest.raises(ValueError, match="absorbing"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_treatment_varies_within_unit(self):
        df = micro_panel()
        df.loc[(df["unit"] == 0) & (df["time"] == 5), "treated"] = 0
        with pytest.raises(ValueError, match="fixed 0/1 group indicator"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_absent_group(self):
        df = micro_panel()
        df["treated"] = 1
        with pytest.raises(ValueError, match="both groups are required"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    @pytest.mark.parametrize("bad", [1, 5, 2.5, "3"])
    def test_last_pre_period_bad(self, bad):
        with pytest.raises(ValueError, match="last_pre_period"):
            fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=bad)

    def test_baseline_zero_survival(self):
        df = build_from_survivors(5, [0, 0, 0, 0, 0], 10, [8, 6, 4, 3, 2])
        with pytest.raises(ValueError, match="zero survival at the baseline"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_treated_exhausted_before_treatment(self):
        df = build_from_survivors(10, [8, 4, 0, 0, 0], 10, [8, 6, 4, 3, 2])
        with pytest.raises(ValueError, match="treated group is fully absorbed"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    def test_control_exhausted_before_treatment(self):
        df = build_from_survivors(10, [8, 6, 4, 3, 2], 10, [8, 4, 0, 0, 0])
        with pytest.raises(ValueError, match="control group is fully absorbed"):
            fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)

    @pytest.mark.parametrize(
        "pre, msg",
        [
            ([9], "Pre-period '9' not found"),
            ([1], "strictly after the baseline"),
            ([4], "strictly after the baseline"),
            ([2, 2], "duplicate"),
            ([], "at least one"),
        ],
    )
    def test_pre_periods_rejected(self, pre, msg):
        with pytest.raises(ValueError, match=msg):
            fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3, pre_periods=pre)

    @pytest.mark.parametrize("bad", [3, 2.0, "23", b"23", {"a": 1}])
    def test_pre_periods_scalar_or_string_rejected(self, bad):
        # A string would otherwise be split character-wise into different dates.
        with pytest.raises(ValueError, match="pre_periods must be a list of"):
            fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3, pre_periods=bad)

    @pytest.mark.parametrize("bad", [1.0, 2, "11"])
    def test_pre_period_weights_scalar_or_string_rejected(self, bad):
        with pytest.raises(ValueError, match="pre_period_weights must be a list of"):
            fit_quiet(
                DurationDiD(n_bootstrap=0),
                micro_panel(),
                last_pre_period=3,
                pre_periods=[2, 3],
                pre_period_weights=bad,
            )

    def test_pre_periods_set_rejected(self):
        # Sets have no positional order to align with pre_period_weights.
        with pytest.raises(ValueError, match="ordered list"):
            fit_quiet(
                DurationDiD(n_bootstrap=0),
                micro_panel(),
                last_pre_period=3,
                pre_periods={3, 2},
                pre_period_weights=[1, 3],
            )
        with pytest.raises(ValueError, match="ordered list"):
            fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3, pre_periods={2})

    def test_pre_periods_non_numeric_entries_rejected(self):
        with pytest.raises(ValueError, match="pre_periods must be a list of numeric"):
            fit_quiet(
                DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3, pre_periods=["a"]
            )

    @pytest.mark.parametrize(
        "pre, w, msg",
        [
            (None, [1.0], "requires pre_periods"),
            ([2, 3], [1.0], "one entry per pre_periods"),
            ([2, 3], [1.0, -1.0], "nonnegative"),
            ([2, 3], [1.0, np.nan], "finite"),
            ([2, 3], [0.0, 0.0], "not all be zero"),
        ],
    )
    def test_pre_period_weights_rejected(self, pre, w, msg):
        with pytest.raises(ValueError, match=msg):
            fit_quiet(
                DurationDiD(n_bootstrap=0),
                micro_panel(),
                last_pre_period=3,
                pre_periods=pre,
                pre_period_weights=w,
            )


# ---------------------------------------------------------------------------
# Fitting dates and weights
# ---------------------------------------------------------------------------


class TestFittingPeriods:
    def test_default_equals_explicit_equal_weights(self):
        df = micro_panel()
        r0 = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=3)
        r1 = fit_quiet(
            DurationDiD(n_bootstrap=0),
            df,
            last_pre_period=3,
            pre_periods=[2, 3],
            pre_period_weights=[1, 1],
        )
        assert r0.coefficient == r1.coefficient
        np.testing.assert_array_equal(r0.att_by_period, r1.att_by_period)
        np.testing.assert_array_equal(r0.pre_periods, [2, 3])
        np.testing.assert_allclose(r0.pre_period_weights, [0.5, 0.5])

    def test_unequal_weights_change_coefficient_by_hand(self):
        df = micro_panel()
        r = fit_quiet(
            DurationDiD(method="ph", n_bootstrap=0),
            df,
            last_pre_period=3,
            pre_periods=[2, 3],
            pre_period_weights=[3, 1],
        )
        ratios = np.array([2.4094208, 2.1699250])
        assert r.coefficient == pytest.approx(0.75 * ratios[0] + 0.25 * ratios[1], abs=1e-6)
        np.testing.assert_allclose(r.pre_period_weights, [0.75, 0.25])

    def test_zero_weight_date_excluded(self):
        df = micro_panel()
        r = fit_quiet(
            DurationDiD(n_bootstrap=0),
            df,
            last_pre_period=3,
            pre_periods=[2, 3],
            pre_period_weights=[0, 1],
        )
        np.testing.assert_array_equal(r.pre_periods, [3])
        assert r.excluded_pre_periods == {2: "zero_weight"}
        # Fitting on the anchor alone: c = H_13 - H_23 at date 3.
        assert r.coefficient == pytest.approx(math.log(1.5))

    def test_weight_normalization_is_scale_invariant(self, sim_df):
        # Weights near the float64 limit must not overflow the sum to inf and
        # silently normalize to zeros.
        kw = dict(last_pre_period=4, pre_periods=[2, 3, 4])
        a = fit_quiet(
            DurationDiD(n_bootstrap=20, seed=1), sim_df, **kw, pre_period_weights=[1, 1, 1]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            b = DurationDiD(n_bootstrap=20, seed=1).fit(
                sim_df, **FIT_KW, **kw, pre_period_weights=[1e308, 1e308, 1e308]
            )
        np.testing.assert_array_equal(a.pre_periods, b.pre_periods)
        np.testing.assert_array_equal(a.pre_period_weights, b.pre_period_weights)
        assert a.coefficient == b.coefficient
        np.testing.assert_array_equal(a.att_by_period, b.att_by_period)
        assert a.se == b.se
        assert b.pre_period_weights.sum() == pytest.approx(1.0)

    def test_role_columns_named_like_internals(self):
        # Column names that collide with internal temporaries are fine.
        df = micro_panel().rename(
            columns={"unit": "_y", "time": "_g", "exited": "y", "treated": "g"}
        )
        r = DurationDiD(n_bootstrap=0).fit(
            df, outcome="y", unit="_y", time="_g", treatment="g", last_pre_period=3
        )
        assert r.coefficient == pytest.approx(math.log(1.5))

    def test_last_k_window(self):
        df = simulate_panel(n_periods=10, tstar_idx=5)
        r = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=6, pre_periods=[5, 6])
        np.testing.assert_array_equal(r.pre_periods, [5, 6])
        # Pretest scope is unconditional: every interior pre-date is tested.
        np.testing.assert_array_equal(r.pretest.periods, [2, 3, 4, 5])

    def test_ph_ineligible_date_excluded_and_renormalized(self):
        # Control has no exit between dates 1 and 2: PH cannot use date 2.
        df = build_from_survivors(30, [18, 9, 4, 1, 0], 10, [8, 8, 4, 3, 2])
        with pytest.warns(UserWarning, match="renormalized"):
            r = DurationDiD(method="ph", n_bootstrap=0).fit(
                df, **FIT_KW, last_pre_period=3, pre_periods=[2, 3], pre_period_weights=[1, 1]
            )
        assert r.excluded_pre_periods == {2: "zero_control_increment"}
        np.testing.assert_array_equal(r.pre_periods, [3])
        np.testing.assert_allclose(r.pre_period_weights, [1.0])

    def test_no_eligible_period_raises(self):
        df = build_from_survivors(30, [18, 9, 4, 1, 0], 10, [8, 8, 8, 3, 2])
        with pytest.raises(ValueError, match="no eligible fitting period"):
            fit_quiet(DurationDiD(method="ph", n_bootstrap=0), df, last_pre_period=3)

    def test_realized_selectors_echoed_in_to_dict(self):
        r = fit_quiet(
            DurationDiD(n_bootstrap=0),
            micro_panel(),
            last_pre_period=3,
            pre_periods=[3],
            pre_period_weights=[2.0],
        )
        d = r.to_dict()
        assert d["pre_periods"] == [3]
        assert d["pre_period_weights"] == [1.0]


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------


class TestInferenceContract:
    def test_n_bootstrap_zero_joint_nan(self):
        r = fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3)
        assert r.inference_status == "disabled"
        assert np.isfinite(r.att)
        assert_nan_inference(
            {"se": r.se, "t_stat": r.t_stat, "p_value": r.p_value, "conf_int": r.conf_int}
        )
        assert np.all(np.isnan(r.se_by_period)) and np.all(np.isnan(r.p_value_by_period))
        assert np.all(np.isnan(r.pointwise_crit_values))
        assert np.isnan(r.cband_crit_value) and np.isnan(r.joint_p_value)
        assert r.vcov is None and r.bootstrap_effects is None
        assert r.pretest.status == "disabled"
        assert np.all(np.isfinite(r.pretest.contrast))
        assert r.pretest.reject is None
        assert r.is_significant is False and r.significance_stars == ""

    def test_seed_determinism(self, sim_df):
        a = fit_quiet(DurationDiD(n_bootstrap=30, seed=7), sim_df, last_pre_period=4)
        b = fit_quiet(DurationDiD(n_bootstrap=30, seed=7), sim_df, last_pre_period=4)
        c = fit_quiet(DurationDiD(n_bootstrap=30, seed=8), sim_df, last_pre_period=4)
        assert a.se == b.se and a.p_value == b.p_value
        np.testing.assert_array_equal(a.bootstrap_effects, b.bootstrap_effects)
        assert a.se != c.se

    def test_available_family_is_coherent(self, fitted):
        r = fitted
        assert r.inference_status == "ok"
        assert r.n_bootstrap_valid == r.n_bootstrap
        assert np.all(np.isfinite(r.se_by_period)) and np.all(r.se_by_period > 0)
        assert r.vcov.shape == (len(r.post_periods), len(r.post_periods))
        assert np.array_equal(r.se_by_period, np.sqrt(np.diag(r.vcov)))
        assert np.all(
            r.cband_upper - r.cband_lower >= r.conf_int_by_period[:, 1] - r.conf_int_by_period[:, 0]
        )
        assert r.cband_crit_value >= np.max(r.pointwise_crit_values)
        # Band rule <=> p <= alpha (headline and every period)
        assert r.is_significant == (not (r.conf_int[0] <= 0 <= r.conf_int[1]))
        for k in range(len(r.post_periods)):
            excl = not (r.conf_int_by_period[k, 0] <= 0 <= r.conf_int_by_period[k, 1])
            assert excl == (r.p_value_by_period[k] <= r.alpha)
        assert r.n_obs == r.n_units * r.n_periods
        assert r.n_units == r.n_treated + r.n_control

    def test_single_post_date(self):
        df = simulate_panel(n=300, n_periods=5, tstar_idx=3)
        r = fit_quiet(DurationDiD(n_bootstrap=40, seed=1), df, last_pre_period=4)
        assert r.vcov.shape == (1, 1)
        np.testing.assert_array_equal(r.cband_lower, r.conf_int_by_period[:, 0])
        assert r.cband_crit_value == r.pointwise_crit_values[0]
        es = r.aggregate("event_study")
        assert es.vcov.shape == (1, 1)

    def test_deterministic_per_family_masks(self, monkeypatch):
        df = (
            micro_panel()
        )  # treated 0..29 (tstar survivors 0..3), control 30..39 (horizon survivors 30, 31)

        def fake_draws(rng, n, size):
            base = np.arange(n)
            d2 = base.copy()
            d2[[30, 31]] = [32, 33]  # no control horizon survivor; tstar survivors intact
            d4 = base.copy()
            d4[:4] = 4  # no treated tstar survivor: both families fail
            return np.stack([base, d2, base.copy(), d4])[:size]

        monkeypatch.setattr(dd_module, "_draw_indices", fake_draws)
        with pytest.warns(UserWarning, match="Bootstrap draws failed"):
            r = DurationDiD(n_bootstrap=4, seed=0).fit(df, **FIT_KW, last_pre_period=3)
        assert r.inference_status == "unavailable_failed_draws"
        assert r.bootstrap_failure_reasons == {
            "post": {"control_survival_zero": 1, "zero_survival_last_pre": 1},
            "pretest": {"zero_survival_last_pre": 1},
        }
        assert r.n_bootstrap_valid == 2 and r.n_bootstrap_valid_pretest == 3
        assert r.pretest.status == "unavailable_failed_draws"
        assert np.isfinite(r.att) and np.all(np.isnan(r.se_by_period))
        assert r.vcov is None and np.all(np.isnan(r.pointwise_crit_values))
        # Failed rows are NaN; the identity draws reproduce the point estimate.
        assert np.all(np.isnan(r.bootstrap_effects[1]))
        np.testing.assert_array_equal(r.bootstrap_effects[0], r.att_by_period)

    def test_pretest_family_survives_post_failure(self, monkeypatch):
        df = micro_panel()

        def fake_draws(rng, n, size):
            base = np.arange(n)
            d2 = base.copy()
            d2[[30, 31]] = [32, 33]
            return np.stack([base, d2, base.copy()])[:size]

        monkeypatch.setattr(dd_module, "_draw_indices", fake_draws)
        r = fit_quiet(DurationDiD(n_bootstrap=3, seed=0), df, last_pre_period=3)
        assert r.inference_status == "unavailable_failed_draws"
        assert r.n_bootstrap_valid_pretest == 3
        # Identity draws give zero pretest SD -> zero-SE gate for that family.
        assert r.pretest.status == "unavailable_zero_se"

    def test_whole_histories_travel_together(self, monkeypatch):
        df = micro_panel()
        seen = {}

        def fake_draws(rng, n, size):
            idx = np.full((size, n), 5)  # every draw = 40 copies of treated unit 5 ... group_empty
            seen["shape"] = idx.shape
            return idx

        monkeypatch.setattr(dd_module, "_draw_indices", fake_draws)
        r = fit_quiet(DurationDiD(n_bootstrap=2, seed=0), df, last_pre_period=3)
        assert seen["shape"] == (2, 40)
        assert r.bootstrap_failure_reasons["post"] == {"group_empty": 2}

    def test_nonfinite_coefficient_draw_is_a_failure(self, monkeypatch):
        # Denormal-scale grid passes the relative spacing check; a draw with a
        # positive treated increment makes H = D/elapsed overflow -> c = +inf,
        # S0 = 0 and a FINITE artifact tau; the predicate must still fail it.
        # Treated increment log(10) at date 2 over elapsed 1e-308 overflows.
        df = build_from_survivors(
            30, [30, 3, 1, 1, 0], 10, [8, 6, 4, 3, 2], times=[0.0, 1e-308, 2e-308, 3e-308, 4e-308]
        )
        calls = {"n": 0}

        def fake_draws(rng, n, size):
            calls["n"] += 1
            return np.tile(np.arange(n), (size, 1))

        monkeypatch.setattr(dd_module, "_draw_indices", fake_draws)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            r = DurationDiD(n_bootstrap=2, seed=0).fit(df, **FIT_KW, last_pre_period=2e-308)
        assert not np.isfinite(r.coefficient)
        assert "counterfactual_nonfinite" in r.curve_status
        assert r.inference_status == "unavailable_invalid_periods"
        assert r.bootstrap_failure_reasons["post"] == {"nonfinite_counterfactual": 2}
        assert r.pretest.status == "unavailable_nonfinite_moments"

    @pytest.mark.parametrize("method", ["cd", "ph"])
    @pytest.mark.parametrize("n_boot", [3, 1000])
    @pytest.mark.parametrize("panel", ["micro", "simulation"])
    def test_zero_se_gate_every_column(self, monkeypatch, method, n_boot, panel):
        df = micro_panel() if panel == "micro" else simulate_panel(n=800)
        last_pre = 3 if panel == "micro" else 4
        monkeypatch.setattr(
            dd_module, "_draw_indices", lambda rng, n, size: np.tile(np.arange(n), (size, 1))
        )
        r = fit_quiet(
            DurationDiD(method=method, n_bootstrap=n_boot, seed=0), df, last_pre_period=last_pre
        )
        assert np.all(r.bootstrap_effects == r.bootstrap_effects[0])
        assert r.n_bootstrap_valid == n_boot and r.n_bootstrap_valid_pretest == n_boot
        assert r.inference_status == "unavailable_zero_se"
        assert np.all(np.isnan(r.se_by_period)) and np.all(np.isnan(r.p_value_by_period))
        assert np.all(np.isnan(r.t_stat_by_period)) and np.all(np.isnan(r.conf_int_by_period))
        assert np.all(np.isnan(r.pointwise_crit_values)) and r.vcov is None
        assert np.all(np.isnan(r.cband_lower)) and np.isnan(r.cband_crit_value)
        assert np.all(np.isnan(r.cband_upper)) and np.isnan(r.joint_p_value)
        assert_nan_inference(
            {"se": r.se, "t_stat": r.t_stat, "p_value": r.p_value, "conf_int": r.conf_int}
        )
        assert r.pretest.status == "unavailable_zero_se"
        assert np.all(np.isnan(r.pretest.se))
        assert np.all(np.isnan(r.pretest.band_lower))
        assert np.all(np.isnan(r.pretest.band_upper))
        assert np.isnan(r.pretest.statistic) and np.isnan(r.pretest.p_value)
        assert np.isnan(r.pretest.crit_value) and r.pretest.reject is None

    def test_seeded_constant_draws_withhold_inference(self):
        # Real pooled resamples, with no mocking: all three estimates have the
        # same non-binary-exact value. Mean roundoff previously admitted an
        # SE around 7e-17, a zero p-value and a point CI as valid inference.
        df = build_from_survivors(3, [3, 2, 0], 3, [3, 2, 1])
        r = fit_quiet(DurationDiD(n_bootstrap=3, seed=11461), df, last_pre_period=2)
        assert r.n_bootstrap_valid == 3
        assert np.all(r.bootstrap_effects == r.bootstrap_effects[0])
        assert r.bootstrap_effects[0, 0] == pytest.approx(1.0 / 3.0)
        assert r.inference_status == "unavailable_zero_se"
        assert r.vcov is None and np.all(np.isnan(r.se_by_period))
        assert np.all(np.isnan(r.p_value_by_period))
        assert np.all(np.isnan(r.conf_int_by_period))
        assert_nan_inference(
            {"se": r.se, "t_stat": r.t_stat, "p_value": r.p_value, "conf_int": r.conf_int}
        )

    @pytest.mark.parametrize("constant_statistic", ["period", "headline", "pretest"])
    def test_constant_statistic_withholds_only_its_family(self, monkeypatch, constant_statistic):
        # Inject controlled replicate statistics after otherwise valid pooled
        # resampling, isolating each gate while the other family still varies.
        original_bootstrap = dd_module._run_bootstrap

        def controlled_bootstrap(*args, **kwargs):
            boot = original_bootstrap(*args, **kwargs)
            assert boot["ok_post"].all() and boot["ok_pretest"].all()
            if constant_statistic == "pretest":
                boot["delta_star"][:, 0] = 1.0 / 3.0
            else:
                if constant_statistic == "period":
                    boot["tau_star"][:, 0] = 1.0 / 3.0
                else:
                    offset = np.arange(1, 4) / 64.0
                    boot["tau_star"] = np.column_stack([1.0 / 3.0 - offset, 1.0 / 3.0 + offset])
                    assert np.all(np.ptp(boot["tau_star"], axis=0) > 0)
                boot["head_star"] = boot["tau_star"].mean(axis=1)
                if constant_statistic == "headline":
                    assert np.all(boot["head_star"] == boot["head_star"][0])
            return boot

        monkeypatch.setattr(dd_module, "_run_bootstrap", controlled_bootstrap)
        r = fit_quiet(
            DurationDiD(n_bootstrap=3, seed=3),
            simulate_panel(n=800, n_periods=6),
            last_pre_period=4,
        )
        if constant_statistic == "pretest":
            assert r.inference_status == "ok" and np.isfinite(r.se)
            assert r.pretest.status == "unavailable_zero_se"
            assert np.all(np.isnan(r.pretest.se)) and np.isnan(r.pretest.p_value)
        else:
            assert r.pretest.status == "ok" and np.isfinite(r.pretest.p_value)
            assert r.inference_status == "unavailable_zero_se"
            assert r.vcov is None and np.all(np.isnan(r.se_by_period))
            assert_nan_inference(
                {"se": r.se, "t_stat": r.t_stat, "p_value": r.p_value, "conf_int": r.conf_int}
            )


class TestCurveValidity:
    @pytest.mark.parametrize("n_boot", [0, 200])
    def test_boundary_is_time_label_invariant(self, n_boot):
        # Single fitting date with no treated pre-period exits: R0 at the
        # fitting date is exactly zero mathematically; roundoff must not flag
        # it, and a relabelled grid must give the same fit.
        s1, s2 = [200, 200, 160], [200, 160, 100]
        a = fit_quiet(
            DurationDiD(n_bootstrap=n_boot, seed=4),
            build_from_survivors(200, s1, 200, s2, times=[0, 1, 2]),
            last_pre_period=1,
        )
        b = fit_quiet(
            DurationDiD(n_bootstrap=n_boot, seed=4),
            build_from_survivors(200, s1, 200, s2, times=[0, 1.3, 2.6]),
            last_pre_period=1.3,
        )
        for r in (a, b):
            assert r.curve_status == ["ok"] * 3
            assert r.inference_status == ("ok" if n_boot else "disabled")
            assert r.counterfactual_survival[1] == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(a.att_by_period, b.att_by_period, rtol=0, atol=1e-12)
        assert a.att == pytest.approx(b.att, abs=1e-12)
        if n_boot:
            assert a.se == pytest.approx(b.se, abs=1e-12)
            np.testing.assert_allclose(a.se_by_period, b.se_by_period, rtol=0, atol=1e-12)
            assert a.n_draws_invalid_counterfactual == b.n_draws_invalid_counterfactual == 0

    def test_negative_gap_cd_nonmonotone(self):
        df = build_from_survivors(30, [18, 18, 18, 18, 18], 10, [8, 6, 4, 4, 4])
        with pytest.warns(UserWarning, match="Invalid imputed counterfactual curve"):
            r = DurationDiD(n_bootstrap=20, seed=0).fit(df, **FIT_KW, last_pre_period=3)
        assert r.coefficient < 0
        assert "counterfactual_nonmonotone" in r.period_status
        assert r.inference_status == "unavailable_invalid_periods"
        assert np.isnan(r.att)
        assert np.all(np.isnan(r.se_by_period)) and np.all(np.isnan(r.p_value_by_period))
        assert np.all(np.isnan(r.pointwise_crit_values)) and np.all(np.isnan(r.cband_lower))
        assert np.isnan(r.cband_crit_value) and np.isnan(r.joint_p_value) and r.vcov is None
        # Raw finite extrapolations are retained on the flagged dates.
        assert np.all(np.isfinite(r.att_by_period))
        assert r.pretest.status == "ok"
        assert r.bootstrap_effects is not None

    def test_pre_date_out_of_domain_flags_whole_path(self):
        # Fitted pre-date survival 1.5 (R0 < 0) with every post step positive.
        n = 1000
        df = build_from_survivors(
            n, [1000, 1000, 900, 800, 700], n, [1000, 200, 100, 10, 1], times=np.arange(5)
        )
        with pytest.warns(UserWarning, match="Invalid imputed counterfactual"):
            r = DurationDiD(n_bootstrap=0).fit(df, **FIT_KW, last_pre_period=2)
        assert r.curve_status[2] == "counterfactual_survival_above_one"
        assert r.period_status == ["ok", "ok"]
        assert r.counterfactual_survival[2] == pytest.approx(1.5)
        assert np.isnan(r.att)

    def test_ph_counterexample_is_valid(self):
        # D1=[0,1,6,7], D2=[0,1,2,2.01]: mean-ratio c=2 puts the fitted R0 at the
        # first post date (4.02) below the OBSERVED R_1,tstar (6); the gate is on
        # the imputed curve, so every status is "ok".
        big = 200_000
        s1 = [big] + [int(round(big * math.exp(-d))) for d in (1, 6, 7)]
        s2 = [big] + [int(round(big * math.exp(-d))) for d in (1, 2, 2.01)]
        df = build_from_survivors(big, s1, big, s2)
        r = fit_quiet(DurationDiD(method="ph", n_bootstrap=0), df, last_pre_period=3)
        assert r.coefficient == pytest.approx(2.0, abs=1e-3)
        assert r.curve_status == ["ok"] * 4
        assert -math.log(r.counterfactual_survival[3]) < -math.log(r.survival_treated[2])

    @pytest.mark.parametrize("seed", range(6))
    def test_ph_never_flags(self, seed):
        df = simulate_panel(n=150, seed=seed, method="ph", c=1.5)
        r = fit_quiet(DurationDiD(method="ph", n_bootstrap=0), df, last_pre_period=4)
        assert r.curve_status == ["ok"] * r.n_periods

    def test_control_survival_zero_masks_att_and_curve(self):
        df = build_from_survivors(30, [18, 9, 4, 2, 1], 10, [8, 6, 4, 2, 0])
        with pytest.warns(UserWarning, match="control_survival_zero"):
            r = DurationDiD(n_bootstrap=10, seed=0).fit(df, **FIT_KW, last_pre_period=3)
        assert r.period_status == ["ok", "control_survival_zero"]
        assert np.isfinite(r.att_by_period[0]) and np.isnan(r.att_by_period[1])
        assert np.isnan(r.counterfactual_survival[4])
        assert r.inference_status == "unavailable_invalid_periods"
        assert r.n_control_survivors_at_horizon == 0

    def test_extreme_negative_cd_overflow_is_clean(self):
        # Tiny control hazard, large negative gap: exp(-R0) overflows to inf.
        n1, n2 = 400, 400
        s1 = [400, 400, 400, 400, 400]
        s2 = [400, 1, 1, 1, 1]
        df = build_from_survivors(n1, s1, n2, s2, times=[0, 1000, 2000, 3000, 4000])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.warns(UserWarning, match="Invalid imputed counterfactual"):
                r = DurationDiD(n_bootstrap=0).fit(df, **FIT_KW, last_pre_period=2000)
        assert "counterfactual_survival_above_one" in r.curve_status
        flagged = [
            i for i, s in enumerate(r.curve_status) if s == "counterfactual_survival_above_one"
        ]
        assert all(
            np.isnan(r.counterfactual_survival[i]) or r.counterfactual_survival[i] > 1
            for i in flagged
        )
        assert not np.any(np.isinf(r.att_by_period))
        assert not np.any(np.isinf(r.counterfactual_survival))


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestInvalidCurveMessage:
    def _message(self, df, last_pre, method="cd"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = DurationDiD(method=method, n_bootstrap=0).fit(
                df, **FIT_KW, last_pre_period=last_pre
            )
        msgs = [
            str(w.message) for w in rec if "Invalid imputed counterfactual curve" in str(w.message)
        ]
        assert len(msgs) == 1, [str(w.message) for w in rec]
        return r, msgs[0]

    def test_all_post_control_zero_keeps_prefix(self):
        df = build_from_survivors(30, [18, 9, 4, 2, 1], 10, [8, 6, 4, 0, 0])
        r, msg = self._message(df, 3)
        assert msg.startswith(
            "Invalid imputed counterfactual curve at 4 (control_survival_zero), 5"
        )
        assert "first post-treatment date is already invalid" in msg
        assert "at or before" not in msg
        assert r.summary().count("Invalid imputed counterfactual curve") == 1

    def test_post_violation_recommends_effective_horizon(self):
        # Control survival first reaches zero at date 5: the recommended
        # horizon is the last date strictly before it (4), which keeps one
        # valid post date.
        df = build_from_survivors(30, [18, 9, 4, 2, 1], 10, [8, 6, 4, 2, 0])
        r, msg = self._message(df, 3)
        assert "subset the data to dates at or before 4 and refit" in msg
        assert "cannot repair" not in msg
        refit = fit_quiet(DurationDiD(n_bootstrap=0), df[df["time"] <= 4], last_pre_period=3)
        assert refit.curve_status == ["ok"] * 4

    def test_pre_date_violation_says_change_the_fit(self):
        n = 1000
        df = build_from_survivors(
            n, [1000, 1000, 900, 800, 700], n, [1000, 200, 100, 10, 1], times=np.arange(5)
        )
        r, msg = self._message(df, 2)
        assert msg.startswith(
            "Invalid imputed counterfactual curve at 2 (counterfactual_survival_above_one)"
        )
        assert "cannot repair it: change the fit" in msg
        assert "at or before" not in msg
        assert "cannot repair" in r.summary()

    def test_negative_gap_mixed_violation(self):
        df = build_from_survivors(30, [18, 18, 18, 18, 18], 10, [8, 6, 4, 4, 4])
        r, msg = self._message(df, 3)
        assert "cannot repair it: change the fit" in msg
        assert "first post-treatment date is already invalid" in msg
        assert "at or before" not in msg


class TestWarnings:
    def test_ph_support_pretest_unavailable_with_finite_att(self):
        # PH with a zero control increment at an interior pre-date: the point
        # estimate stands on the remaining fitting date, the diagnostic is withheld.
        df = build_from_survivors(30, [18, 9, 4, 1, 0], 10, [8, 8, 4, 3, 2])
        r = fit_quiet(DurationDiD(method="ph", n_bootstrap=20, seed=0), df, last_pre_period=3)
        assert r.pretest.status == "unavailable_ph_support"
        assert np.all(np.isnan(r.pretest.contrast)) and r.pretest.reject is None
        assert np.isfinite(r.att) and np.all(np.isfinite(r.att_by_period))

    def test_two_pre_dates_warns_and_pretest_unavailable(self):
        df = build_from_survivors(30, [18, 9, 4, 1], 10, [8, 6, 4, 3])
        with pytest.warns(UserWarning, match="Only two pre-treatment dates"):
            r = DurationDiD(n_bootstrap=10, seed=0).fit(df, **FIT_KW, last_pre_period=2)
        assert r.pretest.status == "unavailable_insufficient_pre_periods"
        assert len(r.pretest.periods) == 0
        assert np.isfinite(r.att)

    def test_fit_omits_anchor_warns(self):
        with pytest.warns(UserWarning, match="omit last_pre_period"):
            DurationDiD(n_bootstrap=0).fit(
                micro_panel(), **FIT_KW, last_pre_period=3, pre_periods=[2]
            )

    def test_ph_boundary_warns(self):
        df = build_from_survivors(30, [18, 18, 18, 10, 5], 10, [8, 6, 4, 3, 2])
        with pytest.warns(UserWarning, match="PH ratio is exactly zero"):
            r = DurationDiD(method="ph", n_bootstrap=0).fit(df, **FIT_KW, last_pre_period=3)
        assert r.ph_ratio_boundary is True and r.coefficient == 0.0

    def test_weak_support_warns_on_few_survivors(self):
        # Count rule: the micro panel's control group has 2 survivors at the
        # horizon, below the 5-survivor floor, at every sample size.
        assert dd_module._WEAK_SUPPORT_MIN_SURVIVORS == 5
        with pytest.warns(UserWarning, match="fewer than 5 survivors \\(minimum 2\\)"):
            r = DurationDiD(n_bootstrap=0).fit(micro_panel(), **FIT_KW, last_pre_period=3)
        assert r.n_control_survivors_at_horizon == 2  # warning only, nothing changes

    def test_weak_support_silent_with_ample_survivors(self, sim_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            DurationDiD(n_bootstrap=0).fit(sim_df, **FIT_KW, last_pre_period=4)


# ---------------------------------------------------------------------------
# Results API
# ---------------------------------------------------------------------------


class TestResultsAPI:
    def test_summary_reports_out_of_domain_draws(self, fitted):
        line = [ln for ln in fitted.summary().splitlines() if "Out-of-domain imputed curves" in ln]
        assert len(line) == 1
        assert f"{fitted.n_draws_invalid_counterfactual}/{fitted.n_bootstrap_valid}" in line[0]
        r0 = fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3)
        assert "n/a (no complete draws)" in r0.summary()

    def test_types(self, fitted):
        assert isinstance(fitted, DurationDiDResults)
        assert isinstance(fitted.pretest, DurationDiDPretestResults)

    def test_summary_alpha_guard(self, fitted):
        text = fitted.summary()
        assert "Duration Difference-in-Differences" in text and "95% CI" in text
        assert "specification test" in text
        with pytest.raises(ValueError, match="never recomputes"):
            fitted.summary(alpha=0.10)
        assert fitted.summary(alpha=0.05) == text

    def test_to_dict_json(self, fitted):
        d = fitted.to_dict()
        assert {"att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"} <= set(d)
        assert not [k for k in d if k.startswith("overall_") or k.startswith("avg_")]
        json.dumps(d)
        assert d["conf_int_lower"] == fitted.conf_int[0]
        assert d["n_units"] == fitted.n_units and d["n_obs"] == fitted.n_obs
        assert d["pretest"]["status"] == "ok"

    def test_to_dataframe_levels(self, fitted):
        periods = fitted.to_dataframe()
        assert list(periods["period"]) == list(fitted.post_periods)
        assert {"att", "se", "cband_lower", "cband_upper", "status"} <= set(periods.columns)
        att = fitted.to_dataframe(level="att")
        assert att.shape[0] == 1 and att.loc[0, "att"] == fitted.att
        with pytest.raises(ValueError, match="level must be"):
            fitted.to_dataframe(level="bogus")

    def test_pretest_surface(self, fitted):
        p = fitted.pretest
        frame = p.to_dataframe()
        assert list(frame.columns) == ["period", "contrast", "se", "band_lower", "band_upper"]
        assert "Algorithm 2" in p.summary()
        json.dumps(p.to_dict())
        assert p.reject == (p.statistic > p.crit_value) == (p.p_value <= p.alpha)
        assert not hasattr(p, "att") and not hasattr(p, "conf_int")

    def test_event_study_aggregate(self, fitted):
        es = fitted.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        n_post = len(fitted.post_periods)
        np.testing.assert_array_equal(es.event_time, np.arange(-1, n_post))
        assert es.is_reference[0] and not es.is_reference[1:].any()
        assert es.att[0] == 0.0 and np.isnan(es.se[0])
        assert es.n.shape == (n_post + 1,) and es.n_kind == "units"
        assert np.all(es.n[1:] == fitted.n_units)
        assert es.event_time_convention == "e0_first_treated"
        assert es.source == "DurationDiDResults"
        np.testing.assert_array_equal(es.att[1:], fitted.att_by_period)
        assert np.allclose(np.diag(es.vcov), es.se[1:] ** 2, rtol=1e-6, atol=0.0)
        assert es.cband_crit_value == fitted.cband_crit_value
        with pytest.raises(ValueError, match="balance_e"):
            fitted.aggregate("event_study", balance_e=1)
        with pytest.raises(ValueError, match="weights"):
            fitted.aggregate("event_study", weights="cell")
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            fitted.aggregate("simple")

    def test_event_study_without_inference(self):
        r = fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3)
        es = r.aggregate("event_study")
        assert es.vcov is None and es.cband_lower is None and es.cband_crit_value is None
        assert np.all(np.isnan(es.se))


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


class TestConsumers:
    def test_diagnostic_report_rejects(self, fitted, sim_df):
        with pytest.raises(TypeError, match="DurationDiDResults"):
            DiagnosticReport(fitted, data=sim_df)

    def test_business_report_rejects(self, fitted):
        with pytest.raises(TypeError, match="DurationDiDResults"):
            BusinessReport(fitted)

    def test_practitioner_handler(self, fitted):
        out = practitioner_next_steps(fitted, verbose=False)
        steps = out["next_steps"]
        text = "\n".join(s["why"] + "\n" + s["code"] for s in steps)
        assert "parallel trends variant" not in text
        assert "compute_honest_did" not in text
        assert any("pretest" in s["code"] for s in steps)
        assumptions = [s for s in steps if s["baker_step"] == 2]
        assert assumptions and "hazard" in assumptions[0]["why"]
        placebo = [s for s in steps if "last_pre_period" in s["code"] and "<=" in s["code"]]
        assert placebo, "anticipation placebo must truncate the frame before refitting"
        assert out["estimator"].startswith("DurationDiD")

    def test_practitioner_placebo_not_applicable_on_two_pre_dates(self):
        df = build_from_survivors(30, [18, 9, 4, 1], 10, [8, 6, 4, 3])
        r = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=2)
        steps = practitioner_next_steps(r, verbose=False)["next_steps"]
        placebo = [s for s in steps if s["baker_step"] == 6 and "placebo" in s["label"].lower()]
        assert placebo and "third pre-date" in placebo[0]["why"]

    def test_practitioner_placebo_exact_anchor_on_decimal_grid(self):
        df = micro_panel(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        r = fit_quiet(DurationDiD(n_bootstrap=0), df, last_pre_period=0.3)
        steps = practitioner_next_steps(r, verbose=False)["next_steps"]
        code = [s["code"] for s in steps if "last_pre_period" in s["code"] and "<=" in s["code"]][0]
        assert "last_pre_period=0.2)" in code
        ns = {"data": df, "DurationDiD": DurationDiD, "np": np}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code, ns)
        assert ns["placebo"].last_pre_period == 0.2

    def test_practitioner_placebo_code_runs(self, sim_df, fitted):
        steps = practitioner_next_steps(fitted, verbose=False)["next_steps"]
        code = [s["code"] for s in steps if "last_pre_period" in s["code"] and "<=" in s["code"]][0]
        ns = {"data": sim_df, "DurationDiD": DurationDiD, "np": np}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code, ns)
        assert "placebo" in ns


# ---------------------------------------------------------------------------
# Numeric clock identity, real-input guards and downstream date consumers
# ---------------------------------------------------------------------------


def _relabel_panel(data, clock, dtype=None):
    """Assign the full typed array so pandas row construction cannot hide its dtype."""
    result = data.copy()
    values = np.tile(clock, len(data) // len(clock))
    result["time"] = pd.array(values, dtype=dtype) if dtype is not None else values
    return result


class TestNumericClock:
    @pytest.mark.parametrize("method", ["cd", "ph"])
    @pytest.mark.parametrize(
        "dtype, origin",
        [
            ("int64", 2**53),
            ("int64", 2**53 + 1),
            ("int64", -(2**63)),
            ("int64", 2**63 - 15),
            ("uint64", 2**64 - 15),
            ("Int64", 2**53 + 1),
            ("UInt64", 2**64 - 15),
        ],
    )
    def test_integer_translation_preserves_full_inference(self, method, dtype, origin):
        data = simulate_panel(n=600, method=method, c=0.05 if method == "cd" else 1.3)
        small = np.arange(8) * 2
        clock = np.array([origin + int(t) for t in small], dtype=dtype.lower())
        original = fit_quiet(
            DurationDiD(method=method, n_bootstrap=40, seed=7),
            _relabel_panel(data, small),
            last_pre_period=6,
            pre_periods=[2, 6],
            pre_period_weights=[1, 3],
        )
        translated = fit_quiet(
            DurationDiD(method=method, n_bootstrap=40, seed=7),
            _relabel_panel(data, clock, dtype),
            last_pre_period=int(clock[3]),
            pre_periods=[clock[1], int(clock[3])],
            pre_period_weights=[1, 3],
        )
        assert original.inference_status == translated.inference_status == "ok"
        assert original.pretest.status == translated.pretest.status == "ok"
        for field in (
            "coefficient",
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int",
            "att_by_period",
            "counterfactual_survival",
            "se_by_period",
            "conf_int_by_period",
            "cband_lower",
            "cband_upper",
            "cband_crit_value",
            "joint_p_value",
            "vcov",
            "bootstrap_effects",
        ):
            np.testing.assert_array_equal(getattr(translated, field), getattr(original, field))
        for field in (
            "contrast",
            "se",
            "band_lower",
            "band_upper",
            "statistic",
            "p_value",
            "reject",
        ):
            np.testing.assert_array_equal(
                getattr(translated.pretest, field), getattr(original.pretest, field)
            )
        assert translated.curve_status == original.curve_status
        assert translated.period_status == original.period_status
        assert translated.bootstrap_failure_reasons == original.bootstrap_failure_reasons
        assert translated.n_draws_invalid_counterfactual == original.n_draws_invalid_counterfactual
        expected = [origin + int(t) for t in small]
        assert translated.periods.tolist() == expected
        labels = translated.to_dict()
        assert labels["periods"] == expected
        assert all(type(p) is int for p in labels["periods"])
        assert labels["last_pre_period"] == expected[3]
        assert json.loads(json.dumps(labels))["periods"] == expected

    @pytest.mark.parametrize("method", ["cd", "ph"])
    def test_native_unequal_large_origin_rejected(self, method):
        clock = [2**53, 2**53 + 2, 2**53 + 3]
        data = build_from_survivors(100, [100, 80, 50], 100, [100, 90, 70], times=clock)
        with pytest.raises(ValueError, match="equally spaced"):
            DurationDiD(method=method, n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=clock[1])

    @pytest.mark.parametrize("origin", [0, 2**53 + 1])
    @pytest.mark.parametrize(
        "deviation, accepted", [(-11, False), (-9, True), (9, True), (11, False)]
    )
    def test_spacing_tolerance_uses_first_native_step(self, origin, deviation, accepted):
        clock = [origin, origin + 10**9, origin + 2 * 10**9 + deviation]
        data = build_from_survivors(100, [100, 80, 50], 100, [100, 90, 70], times=clock)
        if accepted:
            result = fit_quiet(DurationDiD(n_bootstrap=0), data, last_pre_period=clock[1])
            # Keep the actual duration even when its spacing is within tolerance.
            c = (math.log(100 / 80) - math.log(100 / 90)) / 10**9
            expected = math.exp(-math.log(100 / 70) - (2 * 10**9 + deviation) * c) - 0.5
            assert result.att == pytest.approx(expected, abs=1e-14)
        else:
            with pytest.raises(ValueError, match="equally spaced"):
                DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=clock[1])

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("margin, accepted", [(-1, True), (1, False)])
    def test_large_spacing_comparison_precedes_float_rounding(self, sign, margin, accepted):
        deviation = sign * (10**10 + margin)
        clock = [0, 10**18, 2 * 10**18 + deviation]
        data = build_from_survivors(100, [100, 80, 50], 100, [100, 90, 70], times=clock)
        assert (abs(deviation) <= 10**10) == accepted
        if accepted:
            result = fit_quiet(DurationDiD(n_bootstrap=0), data, last_pre_period=clock[1])
            assert np.isfinite(result.att)
        else:
            with pytest.raises(ValueError, match="equally spaced"):
                DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=clock[1])

    @pytest.mark.parametrize(
        "delta, accepted", [(9e-9, True), (11e-9, False), (-9e-9, True), (-11e-9, False)]
    )
    def test_float_spacing_boundary(self, delta, accepted):
        data = build_from_survivors(
            100, [100, 80, 50], 100, [100, 90, 70], times=[0.0, 1.0, 2.0 + delta]
        )
        if accepted:
            assert np.isfinite(fit_quiet(DurationDiD(n_bootstrap=0), data, last_pre_period=1.0).att)
        else:
            with pytest.raises(ValueError, match="equally spaced"):
                DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=1.0)

    def test_elapsed_offsets_do_not_wrap_signed_integer_range(self):
        step = 2**62 + 1
        clock = np.array([-(2**63) + k * step for k in range(4)], dtype=np.int64)
        data = build_from_survivors(100, [100, 80, 60, 40], 100, [100, 90, 80, 70], times=clock)
        result = fit_quiet(
            DurationDiD(n_bootstrap=0),
            data,
            last_pre_period=int(clock[2]),
            pre_period_weights=[1, 3],
            pre_periods=[int(clock[1]), int(clock[2])],
        )
        elapsed = np.array([int(t) - int(clock[0]) for t in clock], dtype=float)
        reference = sum(
            w * (math.log(100 / s1) - math.log(100 / s2)) / e
            for w, s1, s2, e in zip([0.25, 0.75], [80, 60], [90, 80], elapsed[1:3])
        )
        np.testing.assert_allclose(result.coefficient, reference, rtol=1e-14, atol=0)
        arranged = dd_module._validate_and_arrange(data, **FIT_KW, last_pre_period=int(clock[2]))
        np.testing.assert_array_equal(arranged["elapsed"], elapsed)
        assert int(clock[2]) - int(clock[0]) > np.iinfo(np.int64).max

    @pytest.mark.parametrize("clock", [[1, 2, 3], [1.0, 2.0, 3.0], [10**400 + k for k in range(3)]])
    def test_numeric_object_clock_remains_unsupported(self, clock):
        data = build_from_survivors(100, [100, 80, 50], 100, [100, 90, 70])
        data["time"] = pd.Series(
            np.tile(np.array(clock, dtype=object), len(data) // 3), dtype=object
        )
        with pytest.raises(ValueError, match="time column 'time' must be numeric.*object"):
            DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=clock[1])

    @pytest.mark.parametrize("clock", [[-1.7e308, 0.0, 1.7e308], [-1.7e308, -1.6e308, 1.7e308]])
    def test_unrepresentable_elapsed_clock_rejected(self, clock):
        data = build_from_survivors(100, [100, 80, 50], 100, [100, 90, 70], times=clock)
        with pytest.raises(ValueError, match="finite"):
            DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=clock[1])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_clock_rejected_before_panel_arrangement(self, value):
        data = micro_panel()
        data["time"] = data["time"].astype(float)
        data.loc[0, "time"] = value
        with pytest.raises(ValueError, match="time column 'time'.*finite"):
            DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
    def test_representable_float_clock_and_json(self, dtype):
        clock = np.array([0.125, 0.25, 0.375, 0.5, 0.625], dtype=dtype)
        data = _relabel_panel(micro_panel(), clock)
        assert data.time.to_numpy().dtype.type is dtype
        result = fit_quiet(
            DurationDiD(n_bootstrap=0),
            data,
            last_pre_period=dtype(0.375),
            pre_periods=[dtype(0.25), dtype(0.375)],
            pre_period_weights=[0, 1],
        )
        assert result.periods.dtype.type is dtype
        assert type(result.last_pre_period) is float
        assert type(result.pretest.anchor_period) is float
        assert list(result.excluded_pre_periods) == [0.25]
        assert type(next(iter(result.excluded_pre_periods))) is float
        d = json.loads(json.dumps(result.to_dict()))
        p = json.loads(json.dumps(result.pretest.to_dict()))
        assert d["periods"] == [0.125, 0.25, 0.375, 0.5, 0.625]
        assert d["excluded_pre_periods"] == {"0.25": "zero_weight"}
        assert p["anchor_period"] == d["last_pre_period"] == 0.375
        assert p["periods"] == [0.25]
        assert "np." not in result.summary() + result.pretest.summary()
        for frame in [result.to_dataframe(), result.pretest.to_dataframe()]:
            assert np.isfinite(frame["period"].to_numpy()).all()
        steps = practitioner_next_steps(result, verbose=False)["next_steps"]
        code = next(s["code"] for s in steps if "placebo_data =" in s["code"])
        namespace = {"data": data, "DurationDiD": DurationDiD}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code, namespace)
        assert namespace["placebo"].last_pre_period == 0.25

    @pytest.mark.skipif(
        np.finfo(np.longdouble).nmant == np.finfo(float).nmant,
        reason="longdouble has no extra mantissa bits on this platform",
    )
    @pytest.mark.parametrize("field", ["time", "last_pre_period", "pre_periods"])
    def test_lossy_wide_float_rejected(self, field):
        extra = np.longdouble(1) + np.finfo(np.longdouble).eps
        data = micro_panel()
        kw = dict(last_pre_period=3)
        if field == "time":
            clock = np.arange(1, 6, dtype=np.longdouble) + np.finfo(np.longdouble).eps
            data = _relabel_panel(data, clock)
            assert data.time.to_numpy().dtype.type is np.longdouble
        else:
            kw[field] = extra if field == "last_pre_period" else [extra]
        with pytest.raises(ValueError, match="exactly representable as float64"):
            DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, **kw)


class TestExactDateSelectors:
    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    @pytest.mark.parametrize(
        "value", [True, False, np.bool_(True), np.bool_(False), "1", 1 + 0j, np.complex128(1)]
    )
    def test_non_real_date_rejected(self, field, value):
        data = micro_panel(times=[-1, 0, 1, 2, 3])
        kw = dict(last_pre_period=1)
        kw[field] = value if field == "last_pre_period" else [value]
        with pytest.raises(ValueError, match="numeric"):
            DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, **kw)

    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float32("nan"),
            np.float64("inf"),
            np.float64("-inf"),
        ],
    )
    def test_nonfinite_date_rejected_before_lookup(self, field, value):
        kw = dict(last_pre_period=3)
        kw[field] = value if field == "last_pre_period" else [value]
        with pytest.raises(ValueError, match="finite numeric time"):
            DurationDiD(n_bootstrap=0).fit(micro_panel(), **FIT_KW, **kw)

    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    def test_oversized_absent_integer_is_reported_exactly(self, field):
        absent = 10**400
        kw = dict(last_pre_period=3)
        kw[field] = absent if field == "last_pre_period" else [absent]
        with pytest.raises(ValueError) as err:
            DurationDiD(n_bootstrap=0).fit(micro_panel(), **FIT_KW, **kw)
        assert str(absent) in str(err.value)
        assert "not " in str(err.value)

    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    def test_rounded_float_cannot_select_odd_integer(self, field):
        clock = np.array([2**53 + 1 + 2 * k for k in range(5)], dtype=np.int64)
        kw = dict(last_pre_period=int(clock[2]))
        value = float(int(clock[2]))
        assert value != int(clock[2])
        kw[field] = value if field == "last_pre_period" else [value]
        with pytest.raises(ValueError, match="not (a value|found)"):
            DurationDiD(n_bootstrap=0).fit(_relabel_panel(micro_panel(), clock), **FIT_KW, **kw)

    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    @pytest.mark.parametrize("scalar_type", [int, np.int64])
    def test_absent_integer_neighbor_does_not_match(self, field, scalar_type):
        clock = np.array([2**53 + 1 + 2 * k for k in range(5)], dtype=np.int64)
        absent = scalar_type(2**53 + 4)
        kw = dict(last_pre_period=int(clock[2]))
        kw[field] = absent if field == "last_pre_period" else [absent]
        with pytest.raises(ValueError, match="not (a value|found)"):
            DurationDiD(n_bootstrap=0).fit(_relabel_panel(micro_panel(), clock), **FIT_KW, **kw)

    @pytest.mark.parametrize("field", ["last_pre_period", "pre_periods"])
    def test_integer_cannot_alias_neighboring_float_grid(self, field):
        clock = np.array([2**53 + 2 * k for k in range(5)], dtype=float)
        absent = 2**53 + 3
        kw = dict(last_pre_period=float(clock[2]))
        kw[field] = absent if field == "last_pre_period" else [absent]
        with pytest.raises(ValueError, match="not (a value|found)"):
            DurationDiD(n_bootstrap=0).fit(_relabel_panel(micro_panel(), clock), **FIT_KW, **kw)

    def test_mixed_selector_list_preserves_adjacent_large_integers(self):
        origin = 2**53
        clock = np.array([origin + k for k in range(5)], dtype=np.int64)
        result = fit_quiet(
            DurationDiD(n_bootstrap=0),
            _relabel_panel(micro_panel(), clock),
            last_pre_period=np.int64(origin + 3),
            pre_periods=[origin + 1, float(origin + 2), np.int64(origin + 3)],
            pre_period_weights=[0, 1, 3],
        )
        assert result.pre_periods.tolist() == [origin + 2, origin + 3]
        assert result.last_pre_period == origin + 3
        assert result.excluded_pre_periods == {origin + 1: "zero_weight"}
        assert result.to_dict()["excluded_pre_periods"] == {str(origin + 1): "zero_weight"}
        np.testing.assert_array_equal(result.pre_period_weights, [0.25, 0.75])

    @pytest.mark.parametrize("shape", ["2d", "generator"])
    def test_ordered_selector_flattening_preserves_alignment(self, shape):
        periods = [[3, 2]] if shape == "2d" else (p for p in [3, 2])
        weights = [[3, 1]] if shape == "2d" else (w for w in [3, 1])
        result = fit_quiet(
            DurationDiD(n_bootstrap=0),
            micro_panel(),
            last_pre_period=3.0,
            pre_periods=periods,
            pre_period_weights=weights,
        )
        reference = fit_quiet(
            DurationDiD(n_bootstrap=0),
            micro_panel(),
            last_pre_period=3,
            pre_periods=[3.0, 2.0],
            pre_period_weights=[3, 1],
        )
        np.testing.assert_array_equal(result.pre_periods, reference.pre_periods)
        np.testing.assert_array_equal(result.pre_period_weights, reference.pre_period_weights)
        assert result.coefficient == reference.coefficient


class TestRealInputs:
    @pytest.mark.parametrize(
        "column, message", [("exited", "outcome"), ("treated", "treatment"), ("time", "time")]
    )
    @pytest.mark.parametrize("imaginary", [0, 1])
    def test_complex_columns_rejected_before_coercion(self, column, message, imaginary):
        data = micro_panel()
        data[column] = data[column].astype(complex) + imaginary * 1j
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=message):
                DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=3)

    @pytest.mark.parametrize(
        "weights",
        [
            np.array([1 + 1j, 3 + 2j]),
            np.array([1, 3 + 0j], dtype=object),
            [1, np.complex64(3 + 1j)],
        ],
    )
    def test_complex_weights_rejected_before_coercion(self, weights):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="pre_period_weights.*real"):
                DurationDiD(n_bootstrap=0).fit(
                    micro_panel(),
                    **FIT_KW,
                    last_pre_period=3,
                    pre_periods=[2, 3],
                    pre_period_weights=weights,
                )

    @pytest.mark.parametrize("dtype", [bool, int, float])
    def test_real_binary_inputs_keep_the_same_fit(self, dtype):
        data = micro_panel()
        data[["exited", "treated"]] = data[["exited", "treated"]].astype(dtype)
        reference = fit_quiet(DurationDiD(n_bootstrap=0), micro_panel(), last_pre_period=3)
        result = fit_quiet(DurationDiD(n_bootstrap=0), data, last_pre_period=3)
        assert result.att == reference.att
        assert result.coefficient == reference.coefficient


class TestTranslatedDateConsumers:
    @pytest.mark.parametrize("anchor_idx", [1, 2, 3])
    def test_placebo_eligibility_and_executable_anchor(self, anchor_idx):
        origin = 2**53
        clock = np.array([origin + k for k in range(5)], dtype=np.int64)
        data = _relabel_panel(micro_panel(), clock)
        result = fit_quiet(DurationDiD(n_bootstrap=0), data, last_pre_period=int(clock[anchor_idx]))
        steps = practitioner_next_steps(result, verbose=False)["next_steps"]
        step = next(s for s in steps if "placebo" in s["label"].lower())
        if anchor_idx == 1:
            assert "not applicable" in step["label"]
        else:
            namespace = {"data": data, "DurationDiD": DurationDiD}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(step["code"], namespace)
            assert namespace["placebo"].last_pre_period == int(clock[anchor_idx - 1])
            assert namespace["placebo"].post_periods.tolist() == [int(clock[anchor_idx])]

    @pytest.mark.parametrize("pre_violation", [False, True])
    def test_invalid_curve_message_keeps_exact_pre_post_boundary(self, pre_violation):
        origin = 2**53 + 1
        clock = np.array([origin + k for k in range(5)], dtype=np.int64)
        if pre_violation:
            data = build_from_survivors(
                1000, [1000, 1000, 900, 800, 700], 1000, [1000, 200, 100, 10, 1]
            )
        else:
            data = build_from_survivors(30, [18, 9, 4, 2, 1], 10, [8, 6, 4, 2, 0])
        data = _relabel_panel(data, clock)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = DurationDiD(n_bootstrap=0).fit(data, **FIT_KW, last_pre_period=int(clock[2]))
        message = next(str(w.message) for w in captured if "Invalid imputed" in str(w.message))
        if pre_violation:
            assert f"at {int(clock[2])} (counterfactual_survival_above_one)" in message
            assert "cannot repair it: change the fit" in message
            assert "at or before" not in message
        else:
            assert f"at or before {int(clock[3])} and refit" in message
            assert "cannot repair" not in message
        assert message in result.summary()
