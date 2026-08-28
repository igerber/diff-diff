"""DMLDiD core suite (PR-B1): construction, validation, payload, aggregation,
bootstrap, interop. Methodology recovery/parity lives in
tests/test_methodology_dml_did.py; DoubleML golden literals in
TestDoubleMLGoldenParity there.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, DMLDiD, SieveLearner
from diff_diff.staggered import (
    select_base_period as _select_base_period,
)
from diff_diff.staggered import (
    valid_periods_for_group as _valid_periods_for_group,
)


def make_staggered_dml_data(
    n_units=120,
    periods=(2000, 2001, 2002, 2003),
    cohorts=(0, 2001, 2002),
    seed=42,
    effect=2.0,
):
    rng = np.random.default_rng(seed)
    cohort_arr = rng.choice(
        cohorts, size=n_units, p=[0.4] + [0.6 / (len(cohorts) - 1)] * (len(cohorts) - 1)
    )
    rows = []
    for i in range(n_units):
        x1, x2 = rng.normal(), rng.normal()
        for t in periods:
            y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.2 * t + rng.normal(scale=0.5)
            if cohort_arr[i] > 0 and t >= cohort_arr[i]:
                y += effect
            rows.append((i, t, cohort_arr[i], y, x1, x2))
    return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1", "x2"])


FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")
COV = {"covariates": ["x1", "x2"]}


@pytest.fixture(scope="module")
def data():
    return make_staggered_dml_data()


@pytest.fixture(scope="module")
def fitted(data):
    return DMLDiD(seed=0).fit(data, **FIT_KW, **COV)


# ===========================================================================
# Construction / validation matrix
# ===========================================================================


class TestConstructionValidation:
    def test_defaults_construct(self):
        est = DMLDiD()
        assert est.n_folds == 5
        assert est.bootstrap_weights == "rademacher"  # resolved from None
        assert est.results_ is None and est.is_fitted_ is False

    @pytest.mark.parametrize("bad", [0, 1, -1, True, 2.5, "5"])
    def test_n_folds_rejected(self, bad):
        with pytest.raises(ValueError, match="n_folds"):
            DMLDiD(n_folds=bad)

    def test_n_folds_numpy_int_coerced(self):
        est = DMLDiD(n_folds=np.int64(5))
        assert type(est.n_folds) is int

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, True, "0.05", np.nan])
    def test_alpha_bounds_both_sides(self, bad):
        with pytest.raises(ValueError, match="alpha"):
            DMLDiD(alpha=bad)

    @pytest.mark.parametrize("bad", [0.0, 0.5, -0.01, 0.7, True, None, "0.01", np.nan])
    def test_pscore_trim_rejected(self, bad):
        with pytest.raises(ValueError, match="pscore_trim"):
            DMLDiD(pscore_trim=bad)

    def test_pscore_trim_one_element_array_rejected(self):
        # A bare 0 < x < 0.5 silently ACCEPTS a 1-element ndarray.
        with pytest.raises(ValueError, match="pscore_trim"):
            DMLDiD(pscore_trim=np.array([0.01]))

    def test_pscore_trim_numpy_float_coerced(self):
        assert type(DMLDiD(pscore_trim=np.float32(0.01)).pscore_trim) is float

    @pytest.mark.parametrize("bad", [-1, True, 2.5, "0"])
    def test_seed_rejected(self, bad):
        with pytest.raises(ValueError, match="seed"):
            DMLDiD(seed=bad)

    def test_seed_numpy_int_coerced(self):
        assert type(DMLDiD(seed=np.int64(3)).seed) is int

    def test_enum_params_rejected(self):
        with pytest.raises(ValueError, match="control_group"):
            DMLDiD(control_group="nevertreated")
        with pytest.raises(ValueError, match="base_period"):
            DMLDiD(base_period="always")
        with pytest.raises(ValueError, match="bootstrap_weights"):
            DMLDiD(bootstrap_weights="gaussian")

    def test_learner_spec_errors_name_the_param(self):
        with pytest.raises(ValueError, match="propensity_learner"):
            DMLDiD(propensity_learner="forest")
        with pytest.raises(ValueError, match="outcome_learner"):
            DMLDiD(outcome_learner="logit")  # classifier name on regressor slot
        with pytest.raises(TypeError, match="outcome_learner"):
            DMLDiD(outcome_learner=object())
        with pytest.raises(TypeError, match="propensity_learner"):
            DMLDiD(propensity_learner=object())

    def test_get_set_params_roundtrip_learner_object_identity(self):
        learner = SieveLearner(k_max=2)
        est = DMLDiD(outcome_learner=learner, seed=3)
        params = est.get_params()
        assert params["outcome_learner"] is learner  # verbatim-spec contract
        est2 = DMLDiD(**params)
        assert est2.outcome_learner is learner
        est2.set_params(n_folds=7)
        assert est2.n_folds == 7 and est2.outcome_learner is learner

    def test_fit_path_revalidates_mutated_config(self, data):
        est = DMLDiD()
        est.pscore_trim = 0.9
        with pytest.raises(ValueError, match="pscore_trim"):
            est.fit(data, **FIT_KW, **COV)
        est = DMLDiD()
        est.n_folds = 1
        with pytest.raises(ValueError, match="n_folds"):
            est.fit(data, **FIT_KW, **COV)


# ===========================================================================
# Input validation (fit path)
# ===========================================================================


class TestInputValidation:
    def test_covariates_none_routes_to_cs(self, data):
        with pytest.raises(ValueError, match="CallawaySantAnna"):
            DMLDiD().fit(data, **FIT_KW)
        with pytest.raises(ValueError, match="CallawaySantAnna"):
            DMLDiD().fit(data, **FIT_KW, covariates=[])

    def test_bare_string_covariates(self, data):
        with pytest.raises(ValueError, match="bare"):
            DMLDiD().fit(data, **FIT_KW, covariates="x1")

    def test_duplicate_covariates_rejected(self, data):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            DMLDiD().fit(data, **FIT_KW, covariates=["x1", "x1"])

    def test_tuple_covariates_accepted(self, data):
        res = DMLDiD(seed=0).fit(data, **FIT_KW, covariates=("x1", "x2"))
        assert np.isfinite(res.att)

    def test_generator_covariates_equal_list(self, data):
        # One-shot iterables are materialized EXACTLY ONCE: a generator must
        # produce the identical fit as the equivalent list (the pre-fix
        # behavior consumed the generator during validation and silently fit
        # intercept-only nuisances), and an empty generator must raise the
        # required-covariates error.
        a = DMLDiD(seed=0).fit(data, **FIT_KW, covariates=["x1", "x2"])
        b = DMLDiD(seed=0).fit(data, **FIT_KW, covariates=(c for c in ["x1", "x2"]))
        assert a.att == b.att and a.se == b.se
        for k in a.group_time_effects:
            assert a.group_time_effects[k]["effect"] == b.group_time_effects[k]["effect"]
        with pytest.raises(ValueError, match="CallawaySantAnna"):
            DMLDiD().fit(data, **FIT_KW, covariates=(c for c in []))

    def test_cband_strict_bool(self):
        for bad in ("False", 0, 1, None):
            with pytest.raises(ValueError, match="cband"):
                DMLDiD(cband=bad)

    def test_results_store_learner_repr_not_object(self, data):
        import pickle

        res = DMLDiD(outcome_learner=SieveLearner(k_max=2), seed=0).fit(data, **FIT_KW, **COV)
        assert isinstance(res.outcome_learner, str)
        assert res.outcome_learner == "SieveLearner(k_max=2, criterion='bic')"
        pickle.dumps(res)  # no learner objects retained on the results

    def test_role_columns_distinct(self, data):
        with pytest.raises(ValueError, match="distinct"):
            DMLDiD().fit(
                data,
                outcome="time",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )
        with pytest.raises(ValueError, match="covariate|reserved|unit"):
            DMLDiD().fit(data, **FIT_KW, covariates=["unit"])

    def test_missing_columns(self, data):
        with pytest.raises(ValueError, match="Missing columns"):
            DMLDiD().fit(
                data,
                outcome="nope",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )

    def test_nan_identifiers(self, data):
        df = data.copy()
        df.loc[0, "unit"] = np.nan
        with pytest.raises(ValueError, match="unit"):
            DMLDiD().fit(df, **FIT_KW, **COV)
        df = data.copy().astype({"time": float})
        df.loc[0, "time"] = np.nan
        with pytest.raises(ValueError, match="time"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_inf_time_rejected(self, data):
        df = data.copy().astype({"time": float})
        df.loc[0, "time"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_duplicate_unit_time_rows(self, data):
        df = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_first_treat_nan_varying_negative(self, data):
        df = data.copy().astype({"first_treat": float})
        df.loc[0, "first_treat"] = np.nan
        with pytest.raises(ValueError, match="first_treat"):
            DMLDiD().fit(df, **FIT_KW, **COV)
        df = data.copy()
        df.loc[df.index[0], "first_treat"] = 2003  # varies within unit 0
        with pytest.raises(ValueError, match="varies within"):
            DMLDiD().fit(df, **FIT_KW, **COV)
        df = data.copy()
        df.loc[df["unit"] == 0, "first_treat"] = -1
        with pytest.raises(ValueError, match="negative"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_inf_first_treat_recodes_with_warning(self, data):
        df = data.copy().astype({"first_treat": float})
        never = df.groupby("unit")["first_treat"].first() == 0
        never_units = never[never].index
        df.loc[df["unit"].isin(never_units), "first_treat"] = np.inf
        with pytest.warns(UserWarning, match="first_treat=inf"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_non_numeric_outcome_and_covariate(self, data):
        df = data.copy()
        df["y"] = "not-a-number"
        with pytest.raises(ValueError, match="'y'"):
            DMLDiD().fit(df, **FIT_KW, **COV)
        df = data.copy()
        df["x1"] = "abc"
        with pytest.raises(ValueError, match="'x1'"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_complex_columns_rejected(self, data):
        df = data.copy()
        df["y"] = df["y"].astype(complex)
        with pytest.raises(ValueError, match="complex"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_bool_outcome_fits_finite(self, data):
        # The float64 cast is load-bearing: pd.to_numeric preserves bool, and
        # a bool outcome would TypeError in the subtraction.
        df = data.copy()
        df["y"] = df["y"] > df["y"].median()
        res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_int64_extreme_outcome_no_silent_wrap(self, data):
        # int64 opposite-sign extremes: after the float64 cast the difference
        # is FINITE (~1.8e19) and correct — proves wrap-prevention, not
        # overflow (the overflow case is the float64 test below).
        df = data.copy()
        df["y"] = df["y"].astype(np.int64)
        u0 = df["unit"] == 0
        df.loc[u0 & (df["time"] == 2000), "y"] = np.iinfo(np.int64).min + 2
        df.loc[u0 & (df["time"] == 2003), "y"] = np.iinfo(np.int64).max - 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_float64_extreme_outcome_dY_overflow_drops_with_warning(self, data):
        # Two FINITE opposite-sign float64 extremes overflow dY to inf in the
        # cells pairing them; the unit drops from those cells and the
        # consolidated unbalanced-input warning is the user-visible trace
        # (errstate silences numpy). A NEVER-TREATED victim joins control
        # masks, so the base-2000/t-2003 cells see the overflow; sibling
        # cells with one huge-but-finite endpoint die as non_finite_score
        # (also warned) — the fit still survives on the clean cells.
        df = data.copy()
        victim = int(df.loc[df["first_treat"] == 0, "unit"].iloc[0])
        vmask = df["unit"] == victim
        df.loc[vmask & (df["time"] == 2000), "y"] = -1e308
        df.loc[vmask & (df["time"] == 2003), "y"] = 1e308
        with pytest.warns(UserWarning, match="excluded from at least one"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_missing_outcome_complete_case_per_cell(self, data):
        df = data.copy()
        victim = int(df.loc[df["first_treat"] == 2001, "unit"].iloc[0])
        df.loc[(df["unit"] == victim) & (df["time"] == 2003), "y"] = np.nan
        with pytest.warns(UserWarning, match="excluded from at least one"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = DMLDiD(seed=0).fit(data, **FIT_KW, **COV)
        # The unit drops ONLY from cells touching 2003.
        for (g, t), entry in res.group_time_effects.items():
            base_entry = base.group_time_effects[(g, t)]
            if g == 2001 and t == 2003:
                assert entry["n_treated"] == base_entry["n_treated"] - 1
            elif t != 2003:
                assert entry["n_treated"] == base_entry["n_treated"]

    def test_genuinely_unbalanced_frame(self, data):
        # An ABSENT (unit, time) row, not a stored NaN: pivot generates the
        # missing cell, n_obs counts actual rows, warning fires.
        df = data.copy()
        victim = int(df.loc[df["first_treat"] == 2001, "unit"].iloc[0])
        df = df[~((df["unit"] == victim) & (df["time"] == 2003))]
        with pytest.warns(UserWarning, match="excluded from at least one"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert res.n_obs == len(df)
        assert np.isfinite(res.att)

    def test_nonfinite_covariate_per_cell_exclusion(self, data):
        df = data.copy()
        victim = int(df.loc[df["first_treat"] == 0, "unit"].iloc[0])
        df.loc[df["unit"] == victim, "x1"] = np.nan
        with pytest.warns(UserWarning, match="excluded from at least one"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_balanced_input_no_unbalanced_warning(self, data):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DMLDiD(seed=0).fit(data, **FIT_KW, **COV)
        assert not any("excluded from at least one" in str(w.message) for w in rec)

    def test_no_never_treated_error_and_nested_cohort_guard(self, data):
        df = data[data["first_treat"] > 0].copy()
        with pytest.raises(ValueError, match="not_yet_treated"):
            DMLDiD().fit(df, **FIT_KW, **COV)
        one_cohort = df[df["first_treat"] == 2001].copy()
        with pytest.raises(ValueError, match="2 treatment"):
            DMLDiD(control_group="not_yet_treated").fit(one_cohort, **FIT_KW, **COV)
        # With never-treated present a ONE-cohort not_yet_treated fit is valid.
        one_plus_never = data[data["first_treat"].isin([0, 2001])].copy()
        res = DMLDiD(control_group="not_yet_treated", seed=0).fit(one_plus_never, **FIT_KW, **COV)
        assert np.isfinite(res.att)


class TestLabelNormalization:
    def test_int_labels_render_int64(self, fitted):
        assert all(type(g).__name__ == "int64" for g in fitted.groups)
        d = fitted.to_dict()
        cf = d["cross_fit_diagnostics"]
        assert all("." not in k.split(",")[0] for k in cf)  # "g=2001,t=..." no .0

    def test_fractional_labels_take_float_lane(self, data):
        df = data.copy().astype({"time": float, "first_treat": float})
        df["time"] = df["time"] + 0.5
        df.loc[df["first_treat"] > 0, "first_treat"] = df["first_treat"] + 0.5
        res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_string_numeric_labels_fit(self, data):
        df = data.copy()
        df["time"] = df["time"].astype(str)
        res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_magnitude_bound_uint64_above_2_63(self, data):
        df = data.copy()
        # uint64 above 2^63: the int64 round-trip double-wraps and compares
        # equal — only the magnitude bound can catch it.
        mapping = {2000: 2**63 + 16, 2001: 2**63 + 32, 2002: 2**63 + 48, 2003: 2**63 + 64}
        df["time"] = df["time"].map(mapping).astype(np.uint64)
        df["first_treat"] = (
            df["first_treat"].map({0: 0, 2001: 2**63 + 32, 2002: 2**63 + 48}).astype(np.uint64)
        )
        with pytest.raises(ValueError, match="2\\*\\*62"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_magnitude_bound_int64_above_2_62(self, data):
        df = data.copy()
        df["time"] = df["time"].astype(np.int64) - 2000 + 2**62 + 16
        ft = df["first_treat"].astype(np.int64)
        df["first_treat"] = np.where(ft > 0, ft - 2000 + 2**62 + 16, 0)
        with pytest.raises(ValueError, match="2\\*\\*62"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_composite_bound_labels_plus_anticipation(self, data):
        # Labels individually admissible; max|label| + anticipation crosses.
        df = data.copy()
        shift = 2**62 - 2010  # puts max label just below the bound
        df["time"] = df["time"].astype(np.int64) + shift
        ft = df["first_treat"].astype(np.int64)
        df["first_treat"] = np.where(ft > 0, ft + shift, 0)
        with pytest.raises(ValueError, match="anticipation"):
            DMLDiD(anticipation=10**6).fit(df, **FIT_KW, **COV)

    def test_object_column_merge_and_shift_rejected(self):
        base = make_staggered_dml_data(n_units=40, periods=(1, 2, 3, 4), cohorts=(0, 2, 3))
        for raw_labels in (
            [2**60, 2**60 + 1, 0.5, 5],  # pd.to_numeric MERGES the two ints
            [2**60 + 1, 2**60 + 257, 0.5, 5],  # SHIFTS both, nunique kept
        ):
            df = base.copy()
            time_map = dict(zip([1, 2, 3, 4], raw_labels))
            ft_map = {0: 0, 2: raw_labels[1], 3: raw_labels[2]}
            # dtype=object is load-bearing: a plain .map would infer float64
            # and collapse the >2^53 ints in the FIXTURE, before DMLDiD ever
            # sees distinct labels.
            df["time"] = pd.Series([time_map[v] for v in df["time"]], index=df.index, dtype=object)
            df["first_treat"] = pd.Series(
                [ft_map[v] for v in df["first_treat"]], index=df.index, dtype=object
            )
            with pytest.raises(ValueError, match="not exactly representable|precision"):
                DMLDiD().fit(df, **FIT_KW, **COV)

    def test_nan_first_treat_rejected_for_nan_not_elementwise(self):
        base = make_staggered_dml_data(n_units=30)
        df = base.copy().astype({"first_treat": float})
        df.loc[df.index[:4], "first_treat"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            DMLDiD().fit(df, **FIT_KW, **COV)


# ===========================================================================
# Payload contract + determinism + degenerate handling
# ===========================================================================


class TestPayloadContract:
    def test_if_payload_matches_cell_se(self, fitted):
        kit = fitted._aggregation_kit
        assert kit is not None
        ifi = kit.influence
        for (g, t), entry in fitted.group_time_effects.items():
            if entry.get("is_reference") or entry.get("skip_reason") is not None:
                continue
            info = ifi[(g, t)]
            for key in ("treated_idx", "control_idx"):
                assert info[key].dtype == np.int64
                assert len(np.unique(info[key])) == len(info[key])
            assert not set(info["treated_idx"]) & set(info["control_idx"])
            total = float(np.sum(info["treated_inf"] ** 2) + np.sum(info["control_inf"] ** 2))
            np.testing.assert_allclose(np.sqrt(total), entry["se"], atol=1e-15, rtol=1e-12)

    def test_determinism_and_seed_none_entropy(self, data, fitted):
        res2 = DMLDiD(seed=0).fit(data, **FIT_KW, **COV)
        assert res2.att == fitted.att and res2.se == fitted.se
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DMLDiD().fit(data, **FIT_KW, **COV)
            b = DMLDiD().fit(data, **FIT_KW, **COV)
        ent_a = next(iter(a.cross_fit_diagnostics.values()))["fold_seed"]["entropy"]
        ent_b = next(iter(b.cross_fit_diagnostics.values()))["fold_seed"]["entropy"]
        assert ent_a != ent_b  # deterministic assertion, unlike estimates

    def test_aggregate_replay_inert(self, fitted):
        before = fitted.att
        fitted.aggregate("event_study")
        fitted.aggregate("group")
        assert fitted.att == before

    def test_cross_fit_diagnostics_schema(self, fitted):
        for entry in fitted.cross_fit_diagnostics.values():
            assert set(entry) >= {"propensity", "outcome", "p_hat", "n_clipped_ps", "fold_seed"}
            assert isinstance(entry["p_hat"], float)
            for stage in ("propensity", "outcome"):
                sub = entry[stage]
                assert sub is not None and set(sub) == {"fold_losses", "n_fit_per_fold"}


class TestDegenerateHandling:
    def test_single_treated_cohort_cell_nans_and_aggregate_survives(self):
        df = make_staggered_dml_data(n_units=60, seed=3)
        # Shrink cohort 2001 to ONE unit -> singleton D-stratum in its cells.
        units_2001 = df.loc[df["first_treat"] == 2001, "unit"].unique()
        keep = set(units_2001[:1])
        df.loc[df["unit"].isin(set(units_2001) - keep), "first_treat"] = 0
        with pytest.warns(UserWarning, match="could not be estimated"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        skips = {
            (g, t): e["skip_reason"]
            for (g, t), e in res.group_time_effects.items()
            if e["skip_reason"] is not None
        }
        assert all(v == "cross_fit_degenerate" for v in skips.values())
        assert any(g == 2001 for (g, t) in skips)
        assert np.isfinite(res.att)
        # NaN cells carry no IF entry.
        kit = res._aggregation_kit
        assert all(k not in kit.influence for k in skips)

    def test_all_degenerate_raises_before_reference_cells(self):
        # Every cohort has one unit -> every cell degenerate; universal base
        # must RAISE, not return reference-only results.
        df = make_staggered_dml_data(n_units=8, seed=1)
        first = {u: c for u, c in df.groupby("unit")["first_treat"].first().items()}
        treated_units = [u for u, c in first.items() if c > 0]
        for u in treated_units[2:]:
            df.loc[df["unit"] == u, "first_treat"] = 0
        for keep_cohort, u in zip((2001, 2002), treated_units[:2]):
            df.loc[df["unit"] == u, "first_treat"] = keep_cohort
        with pytest.raises(ValueError, match="Could not estimate any"):
            DMLDiD(base_period="universal", seed=0).fit(df, **FIT_KW, **COV)

    def test_non_finite_score_cell(self, data):
        # Oracle learner injecting predictions that overflow the summand to
        # inf (finite inputs; taxonomy (b) — a ValueError from chang_*).
        class ExplodingRegressor:
            # Class-level counter shared across the per-fold deep copies:
            # explode during the FIRST cell's folds only, so sibling cells
            # survive and the consolidated warning (not the all-degenerate
            # ValueError) is the observable.
            calls = [0]

            def fit(self, X, y, sample_weight=None):
                return self

            def predict(self, X):
                self.calls[0] += 1
                if self.calls[0] <= 5:  # n_folds predictions of cell 1
                    return np.full(len(X), -1.7e308)
                return np.zeros(len(X))

        with pytest.warns(UserWarning, match="non_finite_score"):
            res = DMLDiD(outcome_learner=ExplodingRegressor(), seed=0).fit(data, **FIT_KW, **COV)
        skipped = [
            (g, t)
            for (g, t), e in res.group_time_effects.items()
            if e["skip_reason"] == "non_finite_score"
        ]
        assert skipped
        kit = res._aggregation_kit
        assert all(k not in kit.influence for k in skipped)

    def test_rank_deficient_covariate_kills_cell_loudly(self):
        # A covariate CONSTANT at one cell's base period: the fail-closed
        # default learners raise, cross_fit_predict converts to
        # DegenerateFoldError, the cell NaNs as cross_fit_degenerate with the
        # chained learner message quoted; other cells survive. Remedy route
        # is outcome_learner="ridge"/"sieve" (user-facing names).
        df = make_staggered_dml_data(n_units=100, seed=9)
        df["x1"] = np.where(df["time"] <= 2000, 3.0, df["x1"])  # constant at the 2000 base
        with pytest.warns(UserWarning, match="cross_fit_degenerate"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert any(
            e["skip_reason"] == "cross_fit_degenerate" for e in res.group_time_effects.values()
        )
        assert np.isfinite(res.att)
        # ridge remedy: penalized fit proceeds.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_r = DMLDiD(outcome_learner="ridge", seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res_r.att)

    def test_all_pre_treatment_cells_nan_headline(self):
        # Only pre-treatment cells estimable: passes the >=1-finite gate but
        # _aggregate_simple's empty-post-treatment branch NaNs the headline
        # behind its own UserWarning; per-cell pre-treatment estimates stay
        # finite and reported.
        df = make_staggered_dml_data(n_units=80, periods=(2000, 2001, 2002, 2003), seed=5)
        # Make the post-treatment outcomes missing for treated cohorts' cells:
        # drop every treated unit's outcomes at t >= its cohort.
        for u, c in df.groupby("unit")["first_treat"].first().items():
            if c > 0:
                df.loc[(df["unit"] == u) & (df["time"] >= c), "y"] = np.nan
        with pytest.warns(UserWarning):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isnan(res.att)
        from tests.conftest import assert_nan_inference

        assert_nan_inference(
            {
                "se": res.se,
                "t_stat": res.t_stat,
                "p_value": res.p_value,
                "conf_int": res.overall_conf_int,
            }
        )
        assert any(
            np.isfinite(e["effect"]) for (g, t), e in res.group_time_effects.items() if t < g
        )

    def test_finite_zero_se_cell(self):
        # Oracle outcome learner predicting dY exactly => psi_bar == 0 on a
        # no-effect DGP cell? Simpler: constant outcome => dY == 0 for all,
        # oracle m_hat == 0 => summand == 0 => theta == 0, se == 0.
        df = make_staggered_dml_data(n_units=60, seed=7, effect=0.0)
        df["y"] = 5.0  # constant outcome: dY identically zero

        class ZeroRegressor:
            def fit(self, X, y, sample_weight=None):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner=ZeroRegressor(), seed=0).fit(df, **FIT_KW, **COV)
        entries = [
            e
            for e in res.group_time_effects.values()
            if e["skip_reason"] is None and not e.get("is_reference")
        ]
        assert entries and all(e["se"] == 0.0 and e["effect"] == 0.0 for e in entries)
        for e in entries:
            assert np.isnan(e["t_stat"]) and np.isnan(e["p_value"])
            assert np.isnan(e["conf_int"][0]) and np.isnan(e["conf_int"][1])
        # Empty control_idx tolerated by aggregation (inf_full == 0 everywhere).
        assert np.isfinite(res.att) or np.isnan(res.att)


# ===========================================================================
# Semantics: control group, anticipation, base periods
# ===========================================================================


class TestSemantics:
    def test_control_group_semantics_differ(self, data, fitted):
        res_nyt = DMLDiD(control_group="not_yet_treated", seed=0).fit(data, **FIT_KW, **COV)
        # 2001's 2001-cell gains 2002-cohort controls under not_yet_treated
        # only when 2002 > max(t, base) — here (t=2001, base=2000):
        e_never = fitted.group_time_effects[(2001, 2001)]
        e_nyt = res_nyt.group_time_effects[(2001, 2001)]
        assert e_nyt["n_control"] > e_never["n_control"]

    def test_anticipation_shifts_base(self, data):
        res = DMLDiD(anticipation=1, seed=0).fit(data, **FIT_KW, **COV)
        assert type(res.anticipation) is int and res.anticipation == 1
        assert np.isfinite(res.att)

    def test_base_period_helper_parity_vs_cs_on_gapped_grid(self):
        # Post-extraction this pins the DELEGATION: the CS methods and the
        # shared module-level helpers (which DMLDiD consumes) must agree.
        cs = CallawaySantAnna()
        gapped = [1, 2, 4, 7, 8]
        for base_period in ("varying", "universal"):
            for anticipation in (0, 1):
                cs.base_period = base_period
                cs.anticipation = anticipation
                for g in (4, 7):
                    for t in gapped:
                        assert _select_base_period(
                            base_period, anticipation, g, t, gapped
                        ) == cs._select_base_period(g, t, gapped), (base_period, anticipation, g, t)
                    assert _valid_periods_for_group(
                        base_period, anticipation, g, gapped, gapped
                    ) == cs._valid_periods_for_group(g, gapped, gapped)

    def test_universal_reference_rows_and_reference_event_times(self, data):
        res = DMLDiD(base_period="universal", seed=0).fit(data, **FIT_KW, **COV)
        refs = {(g, t): e for (g, t), e in res.group_time_effects.items() if e.get("is_reference")}
        assert refs
        for (g, t), e in refs.items():
            assert e["effect"] == 0.0 and np.isnan(e["se"])
            assert e["n_treated"] > 0 and e["n_control"] == 0
        kit = res._aggregation_kit
        for k in refs:
            info = kit.influence[k]
            assert len(info["treated_idx"]) == 0 and len(info["treated_inf"]) == 0
        # Direct assertion on the raw container (serialization fidelity).
        assert res.reference_event_times == (-1,)
        assert res.to_dict()["reference_event_times"] == [-1]

    def test_varying_base_reference_event_times_none(self, fitted):
        assert fitted.reference_event_times is None
        assert "reference_event_times" not in fitted.to_dict()

    def test_two_period_degenerate_single_cell(self):
        df = make_staggered_dml_data(n_units=100, periods=(0, 1), cohorts=(0, 1), seed=11)
        res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        cells = [
            e
            for e in res.group_time_effects.values()
            if e["skip_reason"] is None and not e.get("is_reference")
        ]
        assert len(cells) == 1
        assert res.att == cells[0]["effect"]
        assert res.se == cells[0]["se"]


# ===========================================================================
# Aggregation + bootstrap
# ===========================================================================


class TestAggregationBootstrap:
    def test_event_study_and_group_aggregations(self, fitted):
        es = fitted.aggregate("event_study")
        df_es = es.to_dataframe()
        assert len(df_es) > 1
        grp = fitted.aggregate("group").to_dataframe()
        assert {int(v) for v in grp["label"]} == {int(g) for g in fitted.groups}
        tot = fitted.aggregate("total").to_dataframe()
        assert np.isfinite(tot["att"].iloc[0])

    def test_total_mass_matches_kit_cell_selection(self, data):
        # Behavioral total-mass test on a DGP with per-cell missingness:
        # aggregate('total') treated mass == sum of per-cell valid-mask
        # n_treated over exactly the cells _cs_total_mass selects —
        # post-anticipation (t >= g - anticipation) AND finite-effect.
        df = data.copy()
        victim = int(df.loc[df["first_treat"] == 2001, "unit"].iloc[0])
        df.loc[(df["unit"] == victim) & (df["time"] == 2003), "y"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        tot = res.aggregate("total").to_dataframe()
        expected_mass = sum(
            e["n_treated"]
            for (g, t), e in res.group_time_effects.items()
            if t >= g - res.anticipation and np.isfinite(e["effect"])
        )
        np.testing.assert_allclose(tot["att"].iloc[0] / res.att, expected_mass, rtol=1e-10)

    def test_bootstrap_override_wired(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analytical = DMLDiD(seed=0).fit(data, **FIT_KW, **COV)
            boot = DMLDiD(seed=0, n_bootstrap=99).fit(data, **FIT_KW, **COV)
        # Point estimates EQUAL (the override replaces inference only).
        assert boot.att == analytical.att
        for k, e in boot.group_time_effects.items():
            assert e["effect"] == analytical.group_time_effects[k]["effect"]
        # Inference fields DIFFER: overall quartet + per-cell quartet.
        assert boot.se != analytical.se
        assert boot.overall_conf_int != analytical.overall_conf_int
        assert boot.overall_p_value != analytical.overall_p_value
        changed = [
            k
            for k, e in boot.group_time_effects.items()
            if e["skip_reason"] is None and e["se"] != analytical.group_time_effects[k]["se"]
        ]
        assert changed
        assert boot.bootstrap_results is not None
        assert boot.cband_crit_value is None  # structurally None at fit

    def test_sup_t_bands_on_replay(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(seed=0, n_bootstrap=99, cband=True).fit(data, **FIT_KW, **COV)
        es = boot.aggregate("event_study")
        df_es = es.to_dataframe()
        cband_cols = [c for c in df_es.columns if "cband" in c]
        assert cband_cols, f"no sup-t band columns on the replay output: {list(df_es.columns)}"

    def test_sieve_outcome_learner_smoke(self, data):
        res = DMLDiD(outcome_learner="sieve", seed=0).fit(data, **FIT_KW, **COV)
        assert np.isfinite(res.att)
        res2 = DMLDiD(outcome_learner=SieveLearner(k_max=2), seed=0).fit(data, **FIT_KW, **COV)
        assert np.isfinite(res2.att)


# ===========================================================================
# Results surface
# ===========================================================================


class TestResultsSurface:
    def test_summary_banner_and_content(self, fitted):
        s = fitted.summary()
        assert "DML DiD (Chang 2020)" in s
        assert "Callaway-Sant'Anna Staggered" not in s
        assert "Propensity learner:" in s and "'logit'" in s
        assert "Cross-fitting folds (K):" in s
        assert "Seed:" in s  # unconditional — seed moves point estimates
        assert "Bootstrap iterations:" not in s  # analytical fit

    def test_summary_bootstrap_lines(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(seed=0, n_bootstrap=49).fit(data, **FIT_KW, **COV)
        s = boot.summary()
        assert "Bootstrap iterations:" in s and "rademacher" in s

    def test_to_dict_json_roundtrip(self, fitted):
        d = fitted.to_dict()
        restored = json.loads(json.dumps(d))
        assert restored["vcov_type"] == "hc1"
        assert restored["n_folds"] == 5
        assert restored["seed"] == 0
        assert restored["pscore_trim"] == 0.01
        assert restored["n_bootstrap"] == 0
        assert restored["cband"] is True
        assert restored["propensity_learner"] == "logit"
        cf = restored["cross_fit_diagnostics"]
        assert all(k.startswith("g=") for k in cf)

    def test_to_dict_json_roundtrip_numpy_config(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=np.int64(1), n_bootstrap=np.int64(19), n_folds=np.int64(5)).fit(
                data, **FIT_KW, **COV
            )
        json.dumps(res.to_dict())

    def test_group_effects_stays_none(self, fitted):
        fitted.aggregate("group")
        assert fitted.group_effects is None  # aggregate never mutates self

    def test_repr_of_learner_objects(self, data):
        res = DMLDiD(outcome_learner=SieveLearner(k_max=2), seed=0).fit(data, **FIT_KW, **COV)
        d = res.to_dict()
        assert d["outcome_learner"] == "SieveLearner(k_max=2, criterion='bic')"


# ===========================================================================
# Interop: HonestDiD / PreTrendsPower / reports / visualization
# ===========================================================================


class TestInterop:
    def test_native_routes_raise_class_aware(self, fitted):
        from diff_diff import compute_honest_did, compute_pretrends_power

        with pytest.raises(ValueError, match="DMLDiDResults"):
            compute_honest_did(fitted)
        with pytest.raises(ValueError, match="DMLDiDResults"):
            compute_pretrends_power(fitted)

    def test_container_route_varying_warns(self, fitted):
        from diff_diff import compute_honest_did, compute_pretrends_power

        es = fitted.aggregate("event_study")
        with pytest.warns(UserWarning, match="DMLDiD\\(base_period='universal'\\)"):
            compute_honest_did(es, method="relative_magnitude", M=0.5)
        with pytest.warns(UserWarning, match="DMLDiD\\(base_period='universal'\\)"):
            compute_pretrends_power(es, M=0.1)  # linear default gates the warning

    def test_container_route_universal_clean(self, data):
        from diff_diff import compute_honest_did

        res = DMLDiD(base_period="universal", seed=0).fit(data, **FIT_KW, **COV)
        es = res.aggregate("event_study")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                honest = compute_honest_did(es, method="relative_magnitude", M=0.5)
            except UserWarning as w:  # pragma: no cover - diagnostic clarity
                pytest.fail(f"unexpected warning on universal-base container: {w}")
        assert honest is not None

    def test_diagnostic_report_renders(self, fitted):
        from diff_diff import DiagnosticReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = DiagnosticReport(fitted).run_all()
        schema = rep.schema
        assert "parallel_trends" in rep.applicable_checks
        # heterogeneity RESOLVES via the direct group_time_effects read.
        assert "heterogeneity" in rep.applicable_checks
        assert "heterogeneity" not in rep.skipped_checks
        het = schema.get("heterogeneity")
        assert het is not None
        # No sensitivity narration on a varying-base fit.
        assert "sensitivity" in rep.skipped_checks
        assert "universal" in rep.skipped_checks["sensitivity"]

    def test_business_report_renders(self, fitted):
        from diff_diff import BusinessReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            br = BusinessReport(fitted, outcome_label="revenue")
            text = br.full_report()
        assert "Chang" in text or "DML" in text

    def test_business_report_rejects_varying_base_sensitivity(self, fitted):
        from diff_diff import BusinessReport

        with pytest.raises(ValueError, match="DMLDiDResults"):
            BusinessReport(fitted, precomputed={"sensitivity": {"anything": 1}})

    def test_diagnostic_report_rejects_varying_base_precomputed(self, fitted):
        from diff_diff import DiagnosticReport

        with pytest.raises(ValueError, match="DMLDiDResults"):
            DiagnosticReport(fitted, precomputed={"sensitivity": {"anything": 1}})

    def test_never_treated_count_contract(self, data):
        from diff_diff import BusinessReport

        res = DMLDiD(control_group="not_yet_treated", seed=0).fit(data, **FIT_KW, **COV)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = BusinessReport(res).full_report()
        assert "never" in text.lower()

    def test_visualization_routes(self, fitted):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff import plot_event_study, plot_group_effects

        with pytest.raises(TypeError, match="aggregate"):
            plot_event_study(fitted)
        es = fitted.aggregate("event_study")
        ax = plot_event_study(es)
        assert ax is not None
        ax2 = plot_group_effects(fitted)
        assert ax2 is not None

    def test_practitioner_next_steps(self, fitted):
        from diff_diff import practitioner_next_steps

        steps = practitioner_next_steps(fitted)
        text = str(steps)
        assert "aggregate('event_study')" in text
        assert "covariates=None" not in text  # the CS covariates step is excluded

    def test_business_report_anticipation_aware(self, data):
        # The shared _apply_anticipation_to_assumption helper must cover the
        # DML identification branch: anticipation>0 flips no_anticipation off
        # and REPLACES the strict wording (no contradictory prose).
        from diff_diff import BusinessReport, DMLDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(seed=0, anticipation=1).fit(data, **FIT_KW, **COV)
            d = BusinessReport(res).to_dict()

        found = []

        def _walk(obj):
            if isinstance(obj, dict):
                if "no_anticipation" in obj:
                    found.append(obj)
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(d)
        assert found
        blk = found[0]
        assert blk["no_anticipation"] is False
        assert "Anticipation is allowed" in blk["description"]
        assert "plus no anticipation." not in blk["description"]


class TestAnticipationInteractions:
    def test_nyt_anticipation_exact_control_membership(self):
        # control_group='not_yet_treated', anticipation=1: controls for cell
        # (g, t) are never-treated PLUS cohorts with
        # cohort > max(t, base) + anticipation (and != g). Pin the exact
        # membership via per-cell n_control on a hand-countable DGP.
        df = make_staggered_dml_data(
            n_units=120, periods=(2000, 2001, 2002, 2003), cohorts=(0, 2002, 2003), seed=13
        )
        res = DMLDiD(control_group="not_yet_treated", anticipation=1, seed=0).fit(
            df, **FIT_KW, **COV
        )
        counts = df.groupby("unit")["first_treat"].first().value_counts()
        n_never, n_2003 = int(counts.get(0, 0)), int(counts.get(2003, 0))
        # Cell (2002, 2002): anticipation=1 -> base = largest observed <
        # g - 1 = 2001 -> base=2000; nyt_threshold = max(2002, 2000) + 1 =
        # 2003 -> cohorts > 2003: none. Controls = never-treated only.
        e = res.group_time_effects[(2002, 2002)]
        assert e["n_control"] == n_never
        # Cell (2002, 2001): pre period (t < g), base = 2000; threshold =
        # max(2001, 2000) + 1 = 2002 -> cohort 2003 qualifies (> 2002).
        e_pre = res.group_time_effects[(2002, 2001)]
        assert e_pre["n_control"] == n_never + n_2003

    def test_anticipation_aggregation_includes_t_g_minus_1(self):
        # With anticipation=1, simple/total aggregation includes cells from
        # t = g - 1 (t >= g - anticipation), and the bootstrap replay's
        # cell selection matches.
        df = make_staggered_dml_data(seed=17)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(anticipation=1, seed=0, n_bootstrap=49).fit(df, **FIT_KW, **COV)
        included = {
            (g, t)
            for (g, t), e in res.group_time_effects.items()
            if t >= g - 1 and np.isfinite(e["effect"])
        }
        assert any(t == g - 1 for (g, t) in included)  # the anticipation cell
        tot = res.aggregate("total").to_dataframe()
        expected_mass = sum(res.group_time_effects[k]["n_treated"] for k in included)
        np.testing.assert_allclose(tot["att"].iloc[0] / res.att, expected_mass, rtol=1e-10)
        # Replay on the bootstrapped fit works and stays inert.
        es = res.aggregate("event_study").to_dataframe()
        assert (es["event_time"] == -1).any() if "event_time" in es.columns else len(es) > 1


class TestUnweightedLearnerSignature:
    def test_fit_x_y_only_learners_through_public_fit(self, data):
        # The advertised duck-typed contract is fit/predict(_proba); on
        # no-survey and bare-cluster fits DMLDiD passes no sample weights,
        # so learners whose fit signature is only (X, y) must work end to
        # end through the public fit(). (Declared survey_design= fits DO
        # pass sample_weight and gate on it up front —
        # tests/test_survey_dml.py.)
        class XYOnlyRegressor:
            def fit(self, X, y):
                import numpy as _np

                Xa = _np.column_stack([_np.ones(len(X)), X])
                self.beta, *_ = _np.linalg.lstsq(Xa, y, rcond=None)
                return self

            def predict(self, X):
                import numpy as _np

                Xa = _np.column_stack([_np.ones(len(X)), X])
                return Xa @ self.beta

        class XYOnlyClassifier:
            def fit(self, X, y):
                self.p = float(np.mean(y))
                return self

            def predict_proba(self, X):
                p = np.full(len(X), min(max(self.p, 0.05), 0.95))
                return np.column_stack([1 - p, p])

        res = DMLDiD(
            propensity_learner=XYOnlyClassifier(),
            outcome_learner=XYOnlyRegressor(),
            seed=0,
        ).fit(data, **FIT_KW, **COV)
        assert np.isfinite(res.att)


class TestFullConfigMutationDefense:
    @pytest.mark.parametrize(
        "attr,bad",
        [
            ("control_group", "typo"),
            ("base_period", "typo"),
            ("alpha", 1.5),
            ("n_bootstrap", -1),
            ("bootstrap_weights", "gaussian"),
            ("seed", -3),
            ("propensity_learner", "forest"),
            ("outcome_learner", "logit"),
            ("pscore_trim", 0.9),
            ("n_folds", 1),
            ("cluster", 123),
        ],
    )
    def test_mutated_config_raises_before_any_cell(self, data, attr, bad):
        # EVERY estimate/inference-moving param is re-validated at the start
        # of fit(): a mutated control_group/base_period would otherwise fall
        # through an else branch and silently select a valid-but-unintended
        # methodology while reporting the invalid label.
        est = DMLDiD()
        setattr(est, attr, bad)
        with pytest.raises((ValueError, TypeError)):
            est.fit(data, **FIT_KW, **COV)


class TestReportingWeightingLabel:
    def test_target_parameter_names_complete_case_weighting(self, fitted):
        from diff_diff._reporting_helpers import describe_target_parameter

        desc = describe_target_parameter(fitted)
        assert "valid-treated-count" in desc["name"]
        assert "complete-case" in desc["definition"]
        assert "cohort-size-weighted" not in desc["definition"]

    def test_summary_surfaces_entropy_when_seed_none(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD().fit(data, **FIT_KW, **COV)
        s = res.summary()
        assert "entropy" in s  # OS-drawn fold entropy surfaced for audit


class TestDecimalLabelPrecision:
    def test_decimal_labels_shift_rejected(self):
        # decimal.Decimal is a registered numbers.Number: the elementwise
        # exact check covers it, so >2^53 Decimal labels that pd.to_numeric
        # SHIFTS without merging (cardinality preserved — the fallback lane
        # cannot see it) raise the targeted precision error.
        from decimal import Decimal

        base = make_staggered_dml_data(n_units=40, periods=(1, 2, 3, 4), cohorts=(0, 2, 3))
        df = base.copy()
        big = 2**53
        tmap = {1: Decimal(big + 1), 2: Decimal(big + 3), 3: Decimal(big + 5), 4: Decimal(big + 7)}
        df["time"] = pd.Series([tmap[v] for v in df["time"]], index=df.index, dtype=object)
        df["first_treat"] = pd.Series(
            [Decimal(0) if v == 0 else tmap[v] for v in df["first_treat"]],
            index=df.index,
            dtype=object,
        )
        with pytest.raises(ValueError, match="precision|not exactly representable"):
            DMLDiD().fit(df, **FIT_KW, **COV)


class TestNumericStringPrecision:
    def test_shifted_decimal_string_cohorts_rejected(self):
        # "9007199254740992.5" parses lossily under float() (the same loss
        # as pd.to_numeric), which would silently MERGE it with the
        # "9007199254740992" cohort; the exact Decimal parse rejects it.
        base = make_staggered_dml_data(n_units=60, periods=(1, 2), cohorts=(0, 2), seed=1)
        df = base.copy()
        big = "9007199254740992"
        tmap = {1: "9007199254740991", 2: big}
        df["time"] = pd.Series([tmap[v] for v in df["time"]], index=df.index, dtype=object)
        units = df.groupby("unit")["first_treat"].first()
        treated = [u for u, c in units.items() if c > 0]
        half = set(treated[: len(treated) // 2])

        def label(u):
            if units[u] == 0:
                return "0"
            return big if u in half else big + ".5"

        df["first_treat"] = pd.Series([label(u) for u in df["unit"]], index=df.index, dtype=object)
        with pytest.raises(ValueError, match="precision|not exactly representable"):
            DMLDiD().fit(df, **FIT_KW, **COV)


class TestConversionCreatedInfinity:
    """Finite oversized raw labels whose float64 conversion overflows to inf
    must be REJECTED, never recoded as the +inf never-treated sentinel (the
    raw-verification guard masks recoded positions, so the recode is the one
    door around the exact-label certificate)."""

    def _with_oversized_cohort(self, label):
        base = make_staggered_dml_data(n_units=40, periods=(1, 2, 3), cohorts=(0, 2), seed=3)
        df = base.copy()
        units = df.groupby("unit")["first_treat"].first()
        victims = [u for u, c in units.items() if c == 0][:3]
        df["first_treat"] = df["first_treat"].astype(object)
        df.loc[df["unit"].isin(victims), "first_treat"] = label
        return df

    def test_oversized_decimal_first_treat_rejected(self):
        from decimal import Decimal

        df = self._with_oversized_cohort(Decimal("1e400"))
        with pytest.raises(ValueError, match="overflows float64|2\\*\\*62"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_oversized_numeric_string_first_treat_rejected(self):
        # Newer pandas parses "1e400" to inf (rejected by the
        # conversion-created-infinity guard); the py3.9-floor pandas'
        # to_numeric raises its own "Unable to parse" instead, wrapped in
        # the targeted numeric-castable error. Both lanes reject loudly.
        df = self._with_oversized_cohort("1e400")
        with pytest.raises(ValueError, match="overflows float64|2\\*\\*62|numeric-castable"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_oversized_decimal_time_rejected(self):
        from decimal import Decimal

        base = make_staggered_dml_data(n_units=40, periods=(1, 2, 3), cohorts=(0, 2), seed=3)
        df = base.copy()
        df["time"] = df["time"].astype(object)
        df.loc[df["time"] == 3, "time"] = Decimal("1e400")
        with pytest.raises(ValueError, match="overflows float64|2\\*\\*62"):
            DMLDiD().fit(df, **FIT_KW, **COV)

    def test_genuine_decimal_infinity_still_recodes(self):
        # A raw label that IS +inf (Decimal("Infinity") here, np.inf covered
        # in TestInputValidation) keeps the CS-parity recode-with-warning.
        from decimal import Decimal

        df = self._with_oversized_cohort(Decimal("Infinity"))
        with pytest.warns(UserWarning, match="first_treat=inf"):
            res = DMLDiD(seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_negative_oversized_decimal_first_treat_rejected(self):
        from decimal import Decimal

        df = self._with_oversized_cohort(Decimal("-1e400"))
        with pytest.raises(ValueError, match="overflows float64|2\\*\\*62"):
            DMLDiD().fit(df, **FIT_KW, **COV)


class TestFloatLaneAnticipationExactness:
    """Float-lane labels must support EXACT +/- anticipation arithmetic:
    labels that are multiples of 512 near 2**62 pass the magnitude and
    round-trip certificates, yet max(t, base) + anticipation can round ONTO
    a later cohort and silently flip its not-yet-treated eligibility."""

    def _mixed_label_frame(self):
        # Reviewer's construction: t = 2**62 - 2048, later cohort at
        # 2**62 - 1536, anticipation 257 -> exact threshold is 255 below the
        # cohort but float64 rounds it onto the cohort exactly. A fractional
        # early period forces the float64 lane.
        rng = np.random.default_rng(5)
        t_lo, t_mid, t_hi = 0.5, float(2**62 - 2048), float(2**62 - 1536)
        rows = []
        for u in range(45):
            if u < 12:
                ft = t_mid  # cohort treated at t_mid
            elif u < 24:
                ft = t_hi  # later cohort: the eligibility-flip victim
            else:
                ft = 0.0
            for tt in (t_lo, t_mid, t_hi):
                rows.append((u, tt, ft, rng.normal(), rng.normal()))
        return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "y", "x1"])

    def test_inexact_anticipation_rejected(self):
        df = self._mixed_label_frame()
        with pytest.raises(ValueError, match="exact anticipation arithmetic"):
            DMLDiD(anticipation=257, control_group="not_yet_treated", seed=0).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )

    def test_exact_float_lane_still_fits(self, data):
        # Half-integer labels with small anticipation are exact in float64
        # and must keep fitting on the float lane.
        df = data.copy()
        df["time"] = df["time"].astype(float) + 0.5
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] > 0, "first_treat"] += 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(anticipation=1, seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)
        # Distinct horizons stay distinct on the exact float lane: the
        # half-integer t - g differences are exact, so the event-study
        # carries one row per horizon (no silent merge).
        es = res.aggregate("event_study")
        es_times = list(es.to_dataframe()["event_time"])
        assert len(es_times) == len(set(es_times))
        assert len(es_times) > 1

    def test_anticipation_zero_mixed_magnitudes_rejected(self):
        # Even at anticipation=0 the mixed-magnitude frame is rejected by the
        # event-time subtraction certificate: with g = 2**62 - 2048, the
        # periods 0.5 / 2**62-2048 / 2**62-1536 produce t - g values that
        # ROUND (0.5 - g collapses onto -g), silently merging distinct
        # event-study horizons in the analytical and bootstrap aggregations.
        df = self._mixed_label_frame()
        with pytest.raises(ValueError, match="event-time arithmetic"):
            DMLDiD(seed=0, n_folds=2).fit(
                df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
                covariates=["x1"],
            )


class TestNativeLearnerTrustBoundary:
    """The trusted-repr / verbatim-error path admits EXACT native learner
    types only: a user-defined subclass carries arbitrary code and must take
    the foreign path (class-name label, withheld exception text)."""

    def test_native_subclass_repr_not_published(self, data):
        from diff_diff._learners import LinearLearner

        class SubclassedLinear(LinearLearner):
            def __repr__(self):
                return "token=SECRET-REPR /home/user/private.csv"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner=SubclassedLinear(), seed=0).fit(data, **FIT_KW, **COV)
        import json

        blob = json.dumps(res.to_dict())
        assert "SECRET-REPR" not in blob
        assert "SECRET-REPR" not in res.summary()
        assert "SubclassedLinear" in res.outcome_learner  # class-name label

    def test_native_subclass_error_text_withheld(self, data):
        from diff_diff._learners import LinearLearner

        class RaisingSubclass(LinearLearner):
            def fit(self, X, y, sample_weight=None):
                raise ValueError("token=SECRET-SUBEXC /home/user/private.csv")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                DMLDiD(outcome_learner=RaisingSubclass(), seed=0).fit(data, **FIT_KW, **COV)
        # Partial-failure route persists diagnostics: assert the withheld form.

        class RaisingOnceSubclass(LinearLearner):
            calls = [0]

            def fit(self, X, y, sample_weight=None):
                self.calls[0] += 1
                if self.calls[0] <= 5:
                    raise ValueError("token=SECRET-SUBEXC")
                self.m = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner=RaisingOnceSubclass(), seed=0).fit(data, **FIT_KW, **COV)
        import json

        assert "SECRET-SUBEXC" not in json.dumps(res.to_dict())
        assert any(
            "withheld" in str(e.get("error", "")) for e in res.cross_fit_diagnostics.values()
        )

    def test_secret_bearing_config_value_not_published(self, data):
        # A float SUBCLASS as a native config value fires its own __repr__
        # inside the native configuration repr — the trust predicate must
        # demote the learner to the foreign path (class-name label).
        from diff_diff._learners import RidgeLearner

        class SecretFloat(float):
            def __repr__(self):
                return "token=SECRET-NESTED /home/user/private.csv"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner=RidgeLearner(alpha=SecretFloat(2.5)), seed=0).fit(
                data, **FIT_KW, **COV
            )
        import json

        blob = json.dumps(res.to_dict())
        assert "SECRET-NESTED" not in blob
        assert "SECRET-NESTED" not in res.summary()
        assert "RidgeLearner" in res.outcome_learner  # class-name label

    def test_secret_bearing_config_error_text_withheld(self, data):
        # An INVALID str-subclass config makes the native learner raise with
        # the value interpolated into library error text; the demoted-trust
        # path must withhold that text from warnings and diagnostics.
        from diff_diff._learners import RidgeLearner

        class SecretStr(str):
            def __repr__(self):
                return "token=SECRET-CFGERR /home/user/private.csv"

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError):
                # every cell degenerate -> the no-estimable-cells ValueError
                DMLDiD(outcome_learner=RidgeLearner(alpha=SecretStr("bogus")), seed=0).fit(
                    data, **FIT_KW, **COV
                )
        assert not any("SECRET-CFGERR" in str(w.message) for w in rec)

    def test_exact_native_objects_store_config_reprs(self, data):
        from diff_diff._learners import LogitLearner, RidgeLearner

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(
                propensity_learner=LogitLearner(max_iter=30),
                outcome_learner=RidgeLearner(alpha=2.5),
                seed=0,
            ).fit(data, **FIT_KW, **COV)
        assert res.propensity_learner == "LogitLearner(max_iter=30, tol=1e-08)"
        assert res.outcome_learner == "RidgeLearner(alpha=2.5)"
        assert "0x" not in res.propensity_learner  # deterministic, no address


class TestEmpiricalOverlapWarning:
    def test_sparse_treated_share_warns_per_cell(self):
        # Two treated units among many controls: min(p_hat, 1-p_hat) <
        # pscore_trim must warn even when the fitted propensities are
        # non-extreme (oracle constant-0.5 classifier).
        df = make_staggered_dml_data(n_units=300, periods=(0, 1), cohorts=(0, 1), seed=3)
        units = df.groupby("unit")["first_treat"].first()
        treated = [u for u, c in units.items() if c > 0]
        df.loc[df["unit"].isin(treated[2:]), "first_treat"] = 0  # keep 2 treated

        class HalfClassifier:
            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                p = np.full(len(X), 0.5)
                return np.column_stack([1 - p, p])

        with pytest.warns(UserWarning, match="empirical treated share"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                DMLDiD(propensity_learner=HalfClassifier(), seed=0, n_folds=2).fit(
                    df, **FIT_KW, **COV
                )


class TestBootstrapPlotUsesStoredCI:
    def test_group_plot_yerr_matches_percentile_ci(self, data):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff import plot_group_effects

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(seed=0, n_bootstrap=99).fit(data, **FIT_KW, **COV)
        ax = plot_group_effects(boot, show=False)
        # Collect plotted error-bar extents and compare to the STORED
        # percentile CIs (asymmetric; a z*SE reconstruction would differ).
        stored = {
            (g, t): e["conf_int"]
            for (g, t), e in boot.group_time_effects.items()
            if e["skip_reason"] is None
        }
        segs = []
        for coll in ax.collections:
            segs.extend(coll.get_segments() if hasattr(coll, "get_segments") else [])
        plotted_bounds = {
            (round(float(s[0][0]), 6), round(float(s[0][1]), 10), round(float(s[1][1]), 10))
            for s in segs
            if len(s) == 2
        }
        assert plotted_bounds
        matched = 0
        for (g, t), (lo, hi) in stored.items():
            for _x, plo, phi in plotted_bounds:
                if abs(plo - lo) < 1e-9 and abs(phi - hi) < 1e-9:
                    matched += 1
                    break
        assert matched >= len(stored) * 0.8  # stored CIs drawn, not z*SE


class TestPlotAlternateAlphaZeroSE:
    """A zero/non-finite-SE cell has all-NaN stored inference; the
    alternate-alpha z*SE reconstruction must NOT resurrect it as a finite
    zero-width interval (all-or-nothing inference)."""

    class _FakeResults:
        alpha = 0.05
        bootstrap_results = None
        survey_metadata = None
        df_inference = None

        def __init__(self):
            nan = float("nan")
            self.group_time_effects = {
                (2, 2): {
                    "effect": 1.5,
                    "se": 0.0,
                    "t_stat": nan,
                    "p_value": nan,
                    "conf_int": (nan, nan),
                    "n_treated": 5,
                    "n_control": 5,
                    "skip_reason": None,
                },
                (2, 3): {
                    "effect": 1.0,
                    "se": 0.5,
                    "t_stat": 2.0,
                    "p_value": 0.045,
                    "conf_int": (0.02, 1.98),
                    "n_treated": 5,
                    "n_control": 5,
                    "skip_reason": None,
                },
            }

    def _yerr_devs(self, ax):
        segs = []
        for coll in ax.collections:
            if hasattr(coll, "get_segments"):
                segs.extend(coll.get_segments())
        return segs

    @pytest.mark.parametrize("bootstrap_flag", [False, True])
    def test_zero_se_cell_nan_gated_mpl(self, bootstrap_flag):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff.visualization import plot_group_effects

        res = self._FakeResults()
        if bootstrap_flag:
            res.bootstrap_results = object()  # triggers the reconstruction warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plot_group_effects(res, alpha=0.10, show=False)
        # Only the finite-SE cell may contribute a drawn error bar; the
        # zero-SE cell's reconstructed bounds are NaN (nothing at width 0).
        for seg in self._yerr_devs(ax):
            if len(seg) == 2 and abs(float(seg[0][0]) - 2.0) < 1e-9:  # x == t of zero-SE cell
                lo_y, hi_y = float(seg[0][1]), float(seg[1][1])
                assert not (
                    abs(lo_y - 1.5) < 1e-12 and abs(hi_y - 1.5) < 1e-12
                ), "zero-SE cell drawn with a finite zero-width CI"

    def test_zero_se_cell_nan_gated_plotly(self):
        pytest.importorskip("plotly")
        from diff_diff.visualization import plot_group_effects

        res = self._FakeResults()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plot_group_effects(res, alpha=0.10, backend="plotly", show=False)
        checked = 0
        for trace in fig.data:
            err = getattr(trace, "error_y", None)
            if err is None or err.array is None:
                continue
            for x, dev in zip(trace.x, err.array):
                if abs(float(x) - 2.0) < 1e-9:  # t of the zero-SE cell
                    checked += 1
                    # dev must be NaN, never a finite zero-width bar
                    assert dev is None or not np.isfinite(float(dev))
        assert checked > 0


class TestForeignLearnerErrorSanitized:
    def test_foreign_learner_error_text_not_persisted(self, data):
        class LeakyRegressor:
            def fit(self, X, y):
                raise ValueError("token=SECRET-XYZ /home/user/private.csv")

            def predict(self, X):  # pragma: no cover
                return np.zeros(len(X))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                # every cell degenerate -> all-NaN ValueError; the point is
                # what got PERSISTED before the raise
                DMLDiD(outcome_learner=LeakyRegressor(), seed=0).fit(data, **FIT_KW, **COV)

        # Partial-failure route: one cell survives via a counter learner.
        class LeakyOnce:
            calls = [0]

            def fit(self, X, y):
                self.calls[0] += 1
                if self.calls[0] <= 5:
                    raise ValueError("token=SECRET-XYZ")
                self.m = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(outcome_learner=LeakyOnce(), seed=0).fit(data, **FIT_KW, **COV)
        d = res.to_dict()
        import json

        blob = json.dumps(d)
        assert "SECRET-XYZ" not in blob
        assert any(
            "withheld" in str(e.get("error", "")) for e in res.cross_fit_diagnostics.values()
        )


class TestSummaryAlphaContract:
    def test_summary_rejects_non_fit_alpha(self, fitted, data):
        with pytest.raises(ValueError, match="never recomputes"):
            fitted.summary(alpha=0.10)
        fitted.summary(alpha=fitted.alpha)  # fit alpha accepted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(seed=0, n_bootstrap=49).fit(data, **FIT_KW, **COV)
        with pytest.raises(ValueError, match="percentile"):
            boot.summary(alpha=0.10)

    def test_summary_uses_z_labels(self, fitted):
        s = fitted.summary()
        assert "z-stat" in s and "P>|z|" in s
        assert "t-stat" not in s and "P>|t|" not in s

    def test_deepcopy_failure_message_not_leaked(self, data):
        # A foreign learner whose __deepcopy__ raises with sensitive text:
        # the reuse warning names only the exception CLASS.
        class LeakyDeepcopy:
            def __deepcopy__(self, memo):
                raise ValueError("token=SECRET-DCOPY /home/user/x.csv")

            def fit(self, X, y):
                self.m = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.m)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = DMLDiD(outcome_learner=LeakyDeepcopy(), seed=0).fit(data, **FIT_KW, **COV)
        texts = [str(w.message) for w in rec]
        assert not any("SECRET-DCOPY" in s for s in texts)
        assert any("could not deep-copy" in s and "ValueError" in s for s in texts)
        import json

        assert "SECRET-DCOPY" not in json.dumps(res.to_dict())
        assert "SECRET-DCOPY" not in res.summary()

    def test_bootstrap_summary_labels_percentile_p(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(seed=0, n_bootstrap=49).fit(data, **FIT_KW, **COV)
        s = boot.summary()
        assert "Boot. p" in s and "P>|z|" not in s
        assert "PERCENTILE" in s


# ===========================================================================
# Repeated cross sections (Chang Case 2, panel=False)
# ===========================================================================


def make_rcs_dml_data(n_rows=1500, periods=(1, 2, 3), cohorts=(0, 2, 3), seed=7, effect=2.0):
    """Declared-RCS frame: one observation per row, row-unique unit IDs."""
    rng = np.random.default_rng(seed)
    cohort_arr = rng.choice(
        cohorts, size=n_rows, p=[0.5] + [0.5 / (len(cohorts) - 1)] * (len(cohorts) - 1)
    )
    tt = rng.choice(periods, size=n_rows)
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.2 * tt + rng.normal(scale=0.5, size=n_rows)
    post = (cohort_arr > 0) & (tt >= cohort_arr)
    y = y + effect * post
    return pd.DataFrame(
        {
            "unit": np.arange(n_rows),
            "time": tt,
            "first_treat": cohort_arr,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


@pytest.fixture(scope="module")
def rcs_data():
    return make_rcs_dml_data()


@pytest.fixture(scope="module")
def rcs_fitted(rcs_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMLDiD(panel=False, seed=0).fit(rcs_data, **FIT_KW, **COV)


class TestRCSConstruction:
    @pytest.mark.parametrize("bad", ["False", 0.0, 1, None, [True]])
    def test_panel_strict_bool(self, bad):
        with pytest.raises(ValueError, match="panel must be a bool"):
            DMLDiD(panel=bad)

    def test_get_set_params_roundtrip(self):
        est = DMLDiD(panel=False, seed=3)
        params = est.get_params()
        assert params["panel"] is False
        est2 = DMLDiD().set_params(**params)
        assert est2.panel is False

    def test_mutated_panel_rejected_at_fit(self, rcs_data):
        est = DMLDiD(panel=False, seed=0)
        est.panel = "nope"
        with pytest.raises(ValueError, match="panel must be a bool"):
            est.fit(rcs_data, **FIT_KW, **COV)


class TestRCSInputValidation:
    def test_duplicate_unit_ids_rejected(self, rcs_data):
        df = rcs_data.copy()
        df.loc[df.index[1], "unit"] = df.loc[df.index[0], "unit"]
        with pytest.raises(ValueError, match="unique unit IDs"):
            DMLDiD(panel=False).fit(df, **FIT_KW, **COV)

    def test_panel_frame_under_rcs_hits_unique_id_error(self, data):
        # The panel duplicate-(unit,time) fixture frame has repeated unit
        # IDs — under panel=False it hits the unique-ID error, not the
        # duplicate-(unit,time) one.
        with pytest.raises(ValueError, match="unique unit IDs"):
            DMLDiD(panel=False).fit(data, **FIT_KW, **COV)

    def test_stationarity_warning_fires_only_under_rcs(self, rcs_data, data):
        # The warning must state the CORRECT Assumption 2.3 interpretation:
        # stable (D, X) wave composition with period-specific potential
        # outcomes — NOT a stable observed-outcome distribution (trends and
        # treatment effects are expected, not violations).
        with pytest.warns(UserWarning, match="composition of .D, X. is stable"):
            DMLDiD(panel=False, seed=0).fit(rcs_data, **FIT_KW, **COV)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DMLDiD(seed=0).fit(data, **FIT_KW, **COV)
        assert not any("stationary cross-sectional" in str(w.message) for w in rec)

    def test_covariates_still_required(self, rcs_data):
        with pytest.raises(ValueError, match="CallawaySantAnna"):
            DMLDiD(panel=False).fit(rcs_data, **FIT_KW, covariates=None)

    def test_label_pipeline_witness_string_labels(self, rcs_data):
        # The label pipeline is mode-independent — one witness on RCS.
        df = rcs_data.copy()
        df["time"] = df["time"].astype(str)
        df["first_treat"] = df["first_treat"].astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)


class TestRCSEstimation:
    def test_finite_att_se_and_recovery(self, rcs_fitted):
        assert np.isfinite(rcs_fitted.att) and np.isfinite(rcs_fitted.se)
        assert abs(rcs_fitted.att - 2.0) < 0.6  # DGP effect = 2.0

    def test_determinism_same_seed(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DMLDiD(panel=False, seed=5).fit(rcs_data, **FIT_KW, **COV)
            b = DMLDiD(panel=False, seed=5).fit(rcs_data, **FIT_KW, **COV)
        assert a.att == b.att and a.se == b.se

    def test_seed_none_entropy_surfaced(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DMLDiD(panel=False).fit(rcs_data, **FIT_KW, **COV)
            b = DMLDiD(panel=False).fit(rcs_data, **FIT_KW, **COV)
        ent_a = {e["fold_seed"]["entropy"] for e in a.cross_fit_diagnostics.values()}
        ent_b = {e["fold_seed"]["entropy"] for e in b.cross_fit_diagnostics.values()}
        assert ent_a and ent_b and ent_a != ent_b


class TestRCSPayloadContract:
    def test_if_payload_matches_cell_se(self, rcs_fitted):
        kit = rcs_fitted._aggregation_kit
        checked = 0
        for (g, t), entry in rcs_fitted.group_time_effects.items():
            if entry["skip_reason"] is not None or entry.get("is_reference"):
                continue
            ii = kit.influence[(g, t)]
            se = np.sqrt(np.sum(ii["treated_inf"] ** 2) + np.sum(ii["control_inf"] ** 2))
            np.testing.assert_allclose(se, entry["se"], rtol=1e-12, atol=0)
            checked += 1
        assert checked > 0

    def test_index_arrays_disjoint_increasing_both_periods(self, rcs_fitted, rcs_data):
        kit = rcs_fitted._aggregation_kit
        obs_time = rcs_data["time"].to_numpy()
        for (g, t), ii in kit.influence.items():
            ti, ci = ii["treated_idx"], ii["control_idx"]
            if len(ti) == 0 and len(ci) == 0:
                continue  # universal reference cells
            assert np.all(np.diff(ti) > 0) and np.all(np.diff(ci) > 0)
            assert len(np.intersect1d(ti, ci)) == 0
            union_times = set(obs_time[np.concatenate([ti, ci])].tolist())
            assert len(union_times) == 2  # rows from BOTH periods
            # A base-period treated row IS in treated_idx.
            assert len(set(obs_time[ti].tolist())) == 2

    def test_aggregate_replay_leaves_payload_bit_identical(self, rcs_fitted):
        kit = rcs_fitted._aggregation_kit
        before = {
            k: {kk: np.array(vv, copy=True) for kk, vv in v.items()}
            for k, v in kit.influence.items()
        }
        rcs_fitted.aggregate("event_study")
        for k, v in kit.influence.items():
            for kk in v:
                np.testing.assert_array_equal(v[kk], before[k][kk])


class TestRCSDegenerateHandling:
    def test_empty_four_group_zero_treated_control(self):
        from tests.conftest import assert_nan_inference

        # Cohort 3 has NO control rows at its base period 2 under
        # never_treated?? — construct directly: remove every control row in
        # period 1 so any cell with base 1 has an empty control-base group.
        df = make_rcs_dml_data(n_rows=800, seed=9)
        drop = (df["first_treat"] == 0) & (df["time"] == 1)
        df = df[~drop].reset_index(drop=True)
        df["unit"] = np.arange(len(df))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        skipped = [
            (k, e)
            for k, e in res.group_time_effects.items()
            if e["skip_reason"] == "zero_treated_control"
        ]
        assert skipped
        for _k, e in skipped:
            assert_nan_inference(e)
        assert any("could not be estimated" in str(w.message) for w in rec)

    def test_singleton_stratum_cross_fit_degenerate(self):
        from tests.conftest import assert_nan_inference

        # Exactly ONE treated row of cohort 2 in the base period 1: the
        # D x T stratum is a singleton -> assign_folds raises -> skip.
        df = make_rcs_dml_data(n_rows=600, seed=11)
        mask = (df["first_treat"] == 2) & (df["time"] == 1)
        keep_one = df.index[mask][:1]
        df = df[~mask | df.index.isin(keep_one)].reset_index(drop=True)
        df["unit"] = np.arange(len(df))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        degen = [
            e for e in res.group_time_effects.values() if e["skip_reason"] == "cross_fit_degenerate"
        ]
        assert degen
        for e in degen:
            assert_nan_inference(e)
        assert any("could not be estimated" in str(w.message) for w in rec)

    def test_learner_failure_maps_to_cross_fit_degenerate(self):
        from tests.conftest import assert_nan_inference

        # A covariate CONSTANT within one cell's rows: the fail-closed
        # LinearLearner raises rank-deficiency -> DegenerateFoldError ->
        # cross_fit_degenerate in the RCS branch.
        df = make_rcs_dml_data(n_rows=900, seed=13)
        cell_rows = df["time"].isin([1, 2])
        df.loc[cell_rows, "x2"] = 1.0
        df.loc[cell_rows, "x1"] = 1.0
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        degen = [
            e for e in res.group_time_effects.values() if e["skip_reason"] == "cross_fit_degenerate"
        ]
        assert degen
        for e in degen:
            assert_nan_inference(e)
        assert any("could not be estimated" in str(w.message) for w in rec)

    def test_injected_overflow_non_finite_score(self, rcs_data):
        from tests.conftest import assert_nan_inference

        class OverflowRegressor:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.full(len(X), 1e308)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                res = DMLDiD(panel=False, outcome_learner=OverflowRegressor(), seed=0).fit(
                    rcs_data, **FIT_KW, **COV
                )
            except ValueError:
                return  # all cells degenerate: loud failure is acceptable
        nf = [e for e in res.group_time_effects.values() if e["skip_reason"] == "non_finite_score"]
        assert nf
        for e in nf:
            assert_nan_inference(e)
        assert any("non_finite_score" in str(w.message) for w in rec)

    def test_all_degenerate_raises_before_reference_cells(self):
        df = make_rcs_dml_data(n_rows=40, seed=15)
        # Too few rows for any cell to cross-fit with K=5 strata.
        df = df.iloc[:16].reset_index(drop=True)
        df["unit"] = np.arange(len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="Could not estimate any"):
                DMLDiD(panel=False, base_period="universal", seed=0).fit(df, **FIT_KW, **COV)

    def test_lam_hat_extremeness_warning(self):
        # Lopsided periods: keep exactly two period-1 rows per cohort, so
        # every base-1 cell has lam_hat far above 1 - pscore_trim while the
        # four-group guard still passes.
        df = make_rcs_dml_data(n_rows=3000, seed=17)
        keep = np.zeros(len(df), dtype=bool)
        at1 = df["time"] == 1
        keep[~at1.to_numpy()] = True
        for cohort in (0, 2, 3):
            idx = df.index[at1 & (df["first_treat"] == cohort)][:2]
            keep[idx] = True
        df = df[keep].reset_index(drop=True)
        df["unit"] = np.arange(len(df))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
            except ValueError:
                pass
        assert any("lam_hat" in str(w.message) for w in rec)


class TestRCSAggregationBootstrap:
    def test_event_study_group_simple(self, rcs_fitted):
        es = rcs_fitted.aggregate("event_study")
        assert len(es.to_dataframe()) > 1
        gr = rcs_fitted.aggregate("group")
        assert len(gr.to_dataframe()) >= 1
        si = rcs_fitted.aggregate("simple")
        np.testing.assert_allclose(si.to_dataframe()["att"].iloc[0], rcs_fitted.att)

    def test_total_fails_closed(self, rcs_fitted):
        with pytest.raises(NotImplementedError, match="repeated-cross-section"):
            rcs_fitted.aggregate("total")

    def test_bootstrap_runs_with_per_row_weights(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = DMLDiD(panel=False, seed=0, n_bootstrap=49).fit(rcs_data, **FIT_KW, **COV)
        assert np.isfinite(boot.att)
        assert boot.bootstrap_results is not None
        es = boot.aggregate("event_study")
        df_es = es.to_dataframe()
        assert any("cband" in c for c in df_es.columns) or es.cband_crit_value is not None


class TestRCSAggWeights:
    def test_weights_are_fixed_cohort_row_masses(self, rcs_fitted, rcs_data):
        # aggregate('simple') equals the hand-computed agg_weight-weighted
        # combination of post-treatment finite cells (WIF-consistency
        # decision: fixed cohort row masses, never per-cell counts).
        cohort_mass = rcs_data.groupby("first_treat").size().to_dict()
        num = 0.0
        den = 0.0
        for (g, t), e in rcs_fitted.group_time_effects.items():
            if e["skip_reason"] is not None or e.get("is_reference"):
                continue
            if t < g - rcs_fitted.anticipation:
                continue
            if not np.isfinite(e["effect"]):
                continue
            w = e["agg_weight"]
            assert w == cohort_mass[g]
            num += w * e["effect"]
            den += w
        np.testing.assert_allclose(rcs_fitted.att, num / den, rtol=1e-12)

    def test_large_int64_cohort_labels_aggregate_and_replay(self):
        # >2**53 int64 cohort labels: the float-key hazard that motivated
        # leaving agg_cohort_masses unset stays off the shipped path —
        # aggregation and bootstrap replay run end-to-end.
        big = 2**60
        df = make_rcs_dml_data(n_rows=1200, seed=19)
        tmap = {1: big + 1, 2: big + 2, 3: big + 3}
        df["time"] = df["time"].map(tmap)
        df["first_treat"] = df["first_treat"].map({0: 0, 2: big + 2, 3: big + 3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(panel=False, seed=0, n_bootstrap=29).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)
        assert sorted(res.groups) == [big + 2, big + 3]
        es = res.aggregate("event_study")
        ets = list(es.to_dataframe()["event_time"])
        assert len(ets) == len(set(ets)) and len(ets) > 1


class TestRCSResultsSurface:
    def test_panel_flag_and_summary(self, rcs_fitted, fitted):
        assert rcs_fitted.panel is False
        s = rcs_fitted.summary()
        assert "repeated cross sections" in s
        assert "obs:" in s
        assert "repeated cross sections" not in fitted.summary()

    def test_to_dict_json_roundtrip(self, rcs_fitted):
        d = rcs_fitted.to_dict()
        blob = json.dumps(d)
        assert d["panel"] is False
        parsed = json.loads(blob)
        any_cell = next(iter(parsed["cross_fit_diagnostics"].values()))
        assert "lam_hat" in any_cell and "g2_lambda" in any_cell

    def test_business_report_rcs_semantics(self, rcs_fitted, rcs_data):
        from diff_diff import BusinessReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = BusinessReport(
                rcs_fitted,
                data=rcs_data,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            ).full_report()
        assert "observations" in rep
        assert "stationary" in rep.lower()
        # Correct Assumption 2.3 interpretation (stable wave composition,
        # not a stable observed-Y distribution).
        assert "composition of (D, X) is stable" in rep
        assert "period-specific potential" in rep

    def test_target_parameter_design_aware(self, rcs_fitted, fitted):
        from diff_diff._reporting_helpers import describe_target_parameter

        rcs_block = describe_target_parameter(rcs_fitted)
        assert "cohort-mass-weighted" in rcs_block["name"]
        panel_block = describe_target_parameter(fitted)
        assert "valid-treated-count-weighted" in panel_block["name"]

    def test_diagnostic_report_bacon_skipped_on_rcs(self, rcs_fitted, rcs_data):
        from diff_diff import DiagnosticReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dr = DiagnosticReport(
                rcs_fitted,
                data=rcs_data,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            )
            rep = dr.run_all()
        assert "bacon" in rep.skipped_checks
        assert "requires panel data" in rep.skipped_checks["bacon"]
        # Panel fits keep running the decomposition (no skip).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            panel_df = make_staggered_dml_data()
            panel_res = DMLDiD(seed=0).fit(panel_df, **FIT_KW, **COV)
            panel_rep = DiagnosticReport(
                panel_res,
                data=panel_df,
                outcome="y",
                unit="unit",
                time="time",
                first_treat="first_treat",
            ).run_all()
        assert "bacon" not in panel_rep.skipped_checks

    def test_practitioner_snippet_carries_panel_false(self, rcs_fitted, fitted):
        from diff_diff import practitioner_next_steps

        steps = practitioner_next_steps(rcs_fitted)
        text = str(steps)
        assert "panel=False" in text
        assert "panel=False" not in str(practitioner_next_steps(fitted))

    def test_plot_smoke(self, rcs_fitted):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from diff_diff import plot_event_study, plot_group_effects

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_group_effects(rcs_fitted, show=False)
            plot_event_study(rcs_fitted.aggregate("event_study"), show=False)


class TestRCSSemantics:
    def test_not_yet_treated_differs(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nt = DMLDiD(panel=False, seed=0).fit(rcs_data, **FIT_KW, **COV)
            nyt = DMLDiD(panel=False, seed=0, control_group="not_yet_treated").fit(
                rcs_data, **FIT_KW, **COV
            )
        assert nt.att != nyt.att

    def test_anticipation_shifts_base(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a0 = DMLDiD(panel=False, seed=0).fit(rcs_data, **FIT_KW, **COV)
            a1 = DMLDiD(panel=False, seed=0, anticipation=1).fit(rcs_data, **FIT_KW, **COV)
        assert set(a0.group_time_effects) != set(a1.group_time_effects)

    def test_universal_base_reference_cells(self, rcs_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = DMLDiD(panel=False, seed=0, base_period="universal").fit(
                rcs_data, **FIT_KW, **COV
            )
        refs = [e for e in res.group_time_effects.values() if e.get("is_reference")]
        assert refs
        for e in refs:
            assert e["effect"] == 0.0 and "agg_weight" in e
        assert res.reference_event_times is not None


class TestRCSCompleteCases:
    def test_missing_outcome_row_drops_from_cell_only(self):
        df = make_rcs_dml_data(n_rows=1200, seed=21)
        victim = df.index[(df["first_treat"] == 2) & (df["time"] == 2)][0]
        df.loc[victim, "y"] = np.nan
        with pytest.warns(UserWarning, match="observation.s. were excluded"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_non_finite_covariate_row_excluded(self):
        df = make_rcs_dml_data(n_rows=1200, seed=23)
        victim = df.index[(df["first_treat"] == 0) & (df["time"] == 3)][0]
        df.loc[victim, "x1"] = np.inf
        with pytest.warns(UserWarning, match="observation.s. were excluded"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_inf_outcome_handled(self):
        df = make_rcs_dml_data(n_rows=1200, seed=25)
        victim = df.index[(df["first_treat"] == 3) & (df["time"] == 3)][0]
        df.loc[victim, "y"] = -np.inf
        with pytest.warns(UserWarning, match="observation.s. were excluded"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                res = DMLDiD(panel=False, seed=0).fit(df, **FIT_KW, **COV)
        assert np.isfinite(res.att)

    def test_no_warning_on_clean_input(self, rcs_data):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DMLDiD(panel=False, seed=0).fit(rcs_data, **FIT_KW, **COV)
        assert not any("were excluded" in str(w.message) for w in rec)
