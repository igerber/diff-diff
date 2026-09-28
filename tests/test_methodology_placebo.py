"""Methodology verification for PlaceboTests (``diff_diff/diagnostics.py``).

Anchored to Bertrand, Duflo & Mullainathan (2004), "How Much Should We Trust
Differences-in-Differences Estimates?", QJE 119(1):249-275 — the placebo-law /
randomization-inference framework. Paper review on file:
``docs/methodology/papers/bertrand-duflo-mullainathan-2004-review.md``.

These tests COMPLEMENT ``tests/test_diagnostics.py`` (which covers functional
behavior, dispatch routing, and the zero-SE / <2-LOO NaN-inference edge cases).
Here we verify the methodology itself:

- the randomization-inference p-value convention ``(1 + count)/(B + 1)``
  (Phipson & Smyth 2010; BDM fn 11), converging to the exact full enumeration,
- R parity via exact enumeration + deterministic leave-one-out / fake-group,
- the fake-timing pre-trends logic,
- the never-treated ``treatment`` filter on ``placebo_group_test``,
- the deliberate permutation NaN-decoupling contract (a valid RI p-value even
  when the permutation SE is degenerate; BDM fn 12).
"""

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diff_diff.diagnostics import (
    leave_one_out_test,
    permutation_test,
    placebo_group_test,
    placebo_timing_test,
    run_placebo_test,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "data"
_GOLDEN_PATH = _DATA_DIR / "placebo_golden.json"
_PANEL_PATH = _DATA_DIR / "placebo_test_panel.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fixed_panel() -> pd.DataFrame:
    """The committed 8-unit x 2-period golden panel, hardcoded (no fixture dep)."""
    y = [
        -1.639137, -0.623634, 0.051834, 1.622805, 0.261434, 0.82986,
        0.337559, 1.580412, -1.055892, -1.067745, 1.062855, 1.478681,
        0.139217, 0.8575, -1.253286, -0.560034,
    ]  # fmt: skip
    return pd.DataFrame(
        {
            "unit": np.repeat(range(8), 2),
            "t": list(range(2)) * 8,
            "y": y,
            "treatment": np.repeat([1, 1, 1, 0, 0, 0, 0, 0], 2),
        }
    )


def _did_att(df, treated, outcome="y", unit="unit", time="t") -> float:
    """Closed-form 2-period 2x2 DiD ATT (double difference of group means)."""
    is_t = df[unit].isin(treated)
    post = df[time] == 1
    return float(
        (df.loc[is_t & post, outcome].mean() - df.loc[is_t & ~post, outcome].mean())
        - (df.loc[~is_t & post, outcome].mean() - df.loc[~is_t & ~post, outcome].mean())
    )


def _exact_ri(df, treated, n_treated, unit="unit"):
    """Exact RI p-value over ALL C(N, n_treated) assignments (observed included).

    This is the ground truth the sampled ``(1 + count)/(B + 1)`` converges to;
    here the observed assignment is one of the enumerated assignments, so the
    convention is ``count / total`` (min ``1/total``).
    """
    att_obs = _did_att(df, treated)
    units = sorted(df[unit].unique())
    atts = np.array([_did_att(df, list(c)) for c in itertools.combinations(units, n_treated)])
    count = int(np.sum(np.abs(atts) >= np.abs(att_obs) - 1e-12))
    return att_obs, count, len(atts), count / len(atts)


def _make_timing_panel(violated: bool, seed: int = 7, n_per_group: int = 15):
    """4-period panel; ``violated`` adds a differential pre-trend to treated units."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(2 * n_per_group):
        treated = u < n_per_group
        a = rng.normal(0, 1.0)
        for t in range(4):
            y = a + 0.4 * t + rng.normal(0, 0.1)
            if violated and treated:
                y += 1.2 * t  # treated trend up faster, even pre-treatment
            rows.append({"unit": u, "t": t, "y": y, "treatment": int(treated)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. Randomization-inference p-value convention (no R required)
# --------------------------------------------------------------------------- #
class TestPlaceboRandomizationInference:
    """The sampled p-value is the Phipson-Smyth (2010) RI value ``(1 + count)/(B + 1)``.

    BDM (2004) fn 11 frames placebo laws as randomization inference; the p-value
    convention is Phipson & Smyth (2010). Assignments are sampled with replacement,
    so ``(1 + count)/(B + 1)`` is a *valid but slightly conservative* Monte-Carlo
    p-value (not an exact finite-sample value); it converges to the exact
    full-enumeration value ``count/total`` (see ``TestPlaceboParityR``).
    """

    def test_sampled_pvalue_matches_phipson_smyth_formula(self):
        panel = _fixed_panel()
        res = permutation_test(
            panel,
            outcome="y",
            treatment="treatment",
            post="t",
            unit="unit",
            n_permutations=200,
            seed=123,
        )
        dist = np.asarray(res.permutation_distribution)
        b = len(dist)
        assert res.original_effect is not None
        count = int(np.sum(np.abs(dist) >= np.abs(res.original_effect)))
        expected = (1 + count) / (b + 1)
        assert res.p_value == pytest.approx(expected, abs=1e-12)
        # With-replacement Monte Carlo: (1+count)/(B+1) is conservative vs the
        # naive count/B (the +1/+1 only ever raises the p-value), and never zero.
        assert res.p_value >= count / b - 1e-12
        assert res.p_value > 0.0

    def test_with_replacement_sampling_tolerates_duplicate_assignments(self):
        """Assignments are drawn independently (with replacement), so duplicates
        (incl. the observed assignment) can recur across the B draws; the p-value
        stays valid. This documents the conservative with-replacement contract."""
        # Tiny universe: C(4, 2) = 6 assignments, so B=50 draws must repeat some.
        rows = [
            {
                "unit": u,
                "t": t,
                "y": float(u) + 0.5 * t + 0.3 * (u % 2) * t,
                "treatment": int(u < 2),
            }
            for u in range(4)
            for t in range(2)
        ]
        panel = pd.DataFrame(rows)
        res = permutation_test(
            panel,
            outcome="y",
            treatment="treatment",
            post="t",
            unit="unit",
            n_permutations=50,
            seed=7,
        )
        dist = np.asarray(res.permutation_distribution)
        # duplicates present (6 distinct assignments, 50 draws) -> with replacement
        assert len(np.unique(np.round(dist, 9))) < len(dist)
        assert 0.0 < res.p_value <= 1.0

    def test_pvalue_bounded_and_floored(self):
        panel = _fixed_panel()
        res = permutation_test(
            panel,
            outcome="y",
            treatment="treatment",
            post="t",
            unit="unit",
            n_permutations=200,
            seed=1,
        )
        b = len(np.asarray(res.permutation_distribution))
        assert 0.0 < res.p_value <= 1.0
        assert res.p_value >= 1.0 / (b + 1) - 1e-12

    def test_exact_enumeration_is_count_over_total(self):
        """From-scratch exhaustive enumeration uses ``count/total`` (observed incl.)."""
        panel = _fixed_panel()
        _, count, total, p_exact = _exact_ri(panel, [0, 1, 2], 3)
        assert total == 56  # C(8, 3)
        assert p_exact == pytest.approx(count / total, abs=1e-15)
        # the observed assignment is always at least as extreme as itself
        assert count >= 1

    @pytest.mark.slow
    def test_sampled_converges_to_exact(self, ci_params):
        panel = _fixed_panel()
        _, _, _, p_exact = _exact_ri(panel, [0, 1, 2], 3)
        b = ci_params.bootstrap(20000, min_n=2000)
        res = permutation_test(
            panel,
            outcome="y",
            treatment="treatment",
            post="t",
            unit="unit",
            n_permutations=b,
            seed=20240101,
        )
        assert res.p_value == pytest.approx(p_exact, abs=0.05)


# --------------------------------------------------------------------------- #
# 2. R parity (skip if golden JSON not committed)
# --------------------------------------------------------------------------- #
def _load_golden() -> dict:
    if not _GOLDEN_PATH.exists() or not _PANEL_PATH.exists():
        pytest.skip(
            f"Placebo R-parity goldens missing at {_DATA_DIR}. To regenerate: "
            "`Rscript benchmarks/R/generate_placebo_golden.R`. The goldens are "
            "committed by default; this skip covers partial checkouts."
        )
    return json.loads(_GOLDEN_PATH.read_text())


class TestPlaceboParityR:
    """Parity with the R reference (base-R ``combn`` exact enumeration)."""

    @pytest.fixture
    def golden(self) -> dict:
        return _load_golden()

    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        _load_golden()  # trigger skip if missing
        return pd.read_csv(_PANEL_PATH)

    def test_exact_enumeration_matches_r(self, golden, panel):
        att_obs, count, total, p_exact = _exact_ri(
            panel, golden["real_treated"], golden["n_treated"]
        )
        assert att_obs == pytest.approx(golden["observed_att"], abs=1e-12)
        assert count == golden["permutation"]["count"]
        assert total == golden["permutation"]["total"]
        assert p_exact == pytest.approx(golden["permutation"]["p_exact"], abs=1e-12)

    def test_leave_one_out_matches_r(self, golden, panel):
        res = leave_one_out_test(panel, outcome="y", treatment="treatment", post="t", unit="unit")
        gl = golden["leave_one_out"]
        assert res.placebo_effect == pytest.approx(gl["mean"], abs=1e-10)
        assert res.se == pytest.approx(gl["se"], abs=1e-10)
        assert res.t_stat == pytest.approx(gl["t_stat"], abs=1e-10)
        assert res.p_value == pytest.approx(gl["p_value"], abs=1e-10)
        assert res.conf_int[0] == pytest.approx(gl["ci_lower"], abs=1e-9)
        assert res.conf_int[1] == pytest.approx(gl["ci_upper"], abs=1e-9)
        assert res.leave_one_out_effects is not None
        for u, att in res.leave_one_out_effects.items():
            assert att == pytest.approx(gl["per_drop_att"][str(int(u))], abs=1e-10)
        # SE-audit C3: pin the LOO t-test degrees of freedom (= n_valid - 1).
        # The df is baked into t_stat / p_value / conf_int (asserted above) but
        # is not surfaced as a result field; recover it from the public
        # leave_one_out_effects (dict unit->effect, NaN for a failed drop).
        n_valid = sum(1 for a in res.leave_one_out_effects.values() if np.isfinite(a))
        assert n_valid - 1 == gl["df"]

    def test_fake_group_matches_r(self, golden, panel):
        fg = golden["fake_group"]
        res = placebo_group_test(
            panel,
            outcome="y",
            time="t",
            unit="unit",
            fake_treated_units=fg["fake_treated_units"],
            treatment="treatment",
        )
        assert res.placebo_effect == pytest.approx(fg["att"], abs=1e-10)

    @pytest.mark.slow
    def test_sampled_permutation_matches_r_exact(self, golden, panel, ci_params):
        b = ci_params.bootstrap(20000, min_n=2000)
        res = permutation_test(
            panel,
            outcome="y",
            treatment="treatment",
            post="t",
            unit="unit",
            n_permutations=b,
            seed=99,
        )
        assert res.p_value == pytest.approx(golden["permutation"]["p_exact"], abs=0.05)


# --------------------------------------------------------------------------- #
# 3. Fake-timing pre-trends logic
# --------------------------------------------------------------------------- #
class TestPlaceboFakeTiming:
    """``placebo_timing_test`` detects differential pre-trends (BDM placebo-law)."""

    def test_null_under_parallel_trends(self):
        panel = _make_timing_panel(violated=False)
        res = placebo_timing_test(
            panel,
            outcome="y",
            treatment="treatment",
            time="t",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )
        assert not res.is_significant
        assert abs(res.placebo_effect) < 0.25

    def test_detects_violated_pre_trends(self):
        panel = _make_timing_panel(violated=True)
        res = placebo_timing_test(
            panel,
            outcome="y",
            treatment="treatment",
            time="t",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )
        assert res.is_significant
        assert res.placebo_effect > 0.8  # treated rose faster pre-treatment

    def test_uses_only_pre_treatment_data(self):
        panel = _make_timing_panel(violated=False)
        res = placebo_timing_test(
            panel,
            outcome="y",
            treatment="treatment",
            time="t",
            fake_treatment_period=1,
            post_periods=[2, 3],
        )
        # pre-periods are {0, 1}: 30 units x 2 periods = 60 rows
        assert res.n_obs == 60

    def test_post_period_fake_date_rejected(self):
        panel = _make_timing_panel(violated=False)
        with pytest.raises(ValueError, match="pre-treatment"):
            placebo_timing_test(
                panel,
                outcome="y",
                treatment="treatment",
                time="t",
                fake_treatment_period=3,
                post_periods=[2, 3],
            )


# --------------------------------------------------------------------------- #
# 4. Fake-group test + never-treated filter
# --------------------------------------------------------------------------- #
class TestPlaceboFakeGroup:
    """``placebo_group_test`` and its optional never-treated ``treatment`` filter."""

    def _control_panel(self, seed=3, n=10):
        rng = np.random.default_rng(seed)
        rows = []
        for u in range(n):
            a = rng.normal(0, 1.0)
            for t in range(2):
                rows.append({"unit": u, "t": t, "y": a + 0.3 * t + rng.normal(0, 0.1)})
        return pd.DataFrame(rows)

    def test_null_on_valid_control_only_design(self):
        panel = self._control_panel()
        res = placebo_group_test(
            panel, outcome="y", time="t", unit="unit", fake_treated_units=[0, 1, 2, 3, 4]
        )
        assert not res.is_significant

    def test_treatment_filter_drops_ever_treated(self):
        panel = _fixed_panel()  # units 0,1,2 are real-treated
        res = placebo_group_test(
            panel,
            outcome="y",
            time="t",
            unit="unit",
            fake_treated_units=[3, 4],
            treatment="treatment",
        )
        # 5 never-treated units x 2 periods = 10 rows (3 real-treated dropped)
        assert res.n_obs == 10
        # fake_group records the units actually used, not just the requested list
        assert res.fake_group == [3, 4]

    def test_dispatcher_threads_treatment_into_fake_group(self):
        """run_placebo_test(test_type='fake_group') filters ever-treated by default."""
        panel = _fixed_panel()  # units 0,1,2 are real-treated
        res = run_placebo_test(
            panel,
            outcome="y",
            treatment="treatment",
            time="t",
            unit="unit",
            test_type="fake_group",
            fake_treatment_group=[3, 4],
        )
        # dispatcher passes treatment -> ever-treated dropped -> 5 controls x 2 = 10 rows
        assert res.n_obs == 10
        assert res.fake_group == [3, 4]

    def test_backward_compatible_without_treatment(self):
        panel = _fixed_panel()
        res = placebo_group_test(
            panel, outcome="y", time="t", unit="unit", fake_treated_units=[3, 4]
        )
        assert res.n_obs == 16  # all units retained (old behavior)

    def test_degenerate_filter_raises_valueerror(self):
        panel = _fixed_panel()
        # all requested fake-treated units are themselves real-treated -> dropped
        # (the misuse UserWarning is intrinsic here and asserted separately below)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="No fake-treated observations remain"):
                placebo_group_test(
                    panel,
                    outcome="y",
                    time="t",
                    unit="unit",
                    fake_treated_units=[0, 1],
                    treatment="treatment",
                )

    def test_treatment_filter_rejects_non_binary(self):
        """Fail closed: a non-0/1 treatment column must not silently skip filtering."""
        panel = _fixed_panel()
        panel["treatment"] = panel["treatment"] * 2  # {0, 2} -> non-binary
        with pytest.raises(ValueError, match="binary"):
            placebo_group_test(
                panel,
                outcome="y",
                time="t",
                unit="unit",
                fake_treated_units=[3, 4],
                treatment="treatment",
            )

    def test_treatment_filter_rejects_missing_column(self):
        panel = _fixed_panel()
        with pytest.raises(ValueError, match="not found"):
            placebo_group_test(
                panel,
                outcome="y",
                time="t",
                unit="unit",
                fake_treated_units=[3, 4],
                treatment="nonexistent",
            )

    def test_treatment_filter_rejects_missing_values(self):
        """Fail closed: NaN treatment values (which groupby().max() silently skips)."""
        panel = _fixed_panel()
        panel.loc[0, "treatment"] = np.nan
        with pytest.raises(ValueError, match="contains missing values"):
            placebo_group_test(
                panel,
                outcome="y",
                time="t",
                unit="unit",
                fake_treated_units=[3, 4],
                treatment="treatment",
            )

    def test_misuse_warns_when_fake_unit_is_real_treated(self):
        panel = _fixed_panel()
        with pytest.warns(UserWarning, match="real-treated"):
            placebo_group_test(
                panel,
                outcome="y",
                time="t",
                unit="unit",
                fake_treated_units=[2, 3, 4],
                treatment="treatment",
            )


# --------------------------------------------------------------------------- #
# 5. Inference-honesty contracts
# --------------------------------------------------------------------------- #
class TestPlaceboInferenceContracts:
    """The deliberate permutation NaN-decoupling + fail-closed RuntimeErrors."""

    def test_permutation_nan_decoupling(self):
        """n_valid == 1 -> se is NaN and t_stat NaN, but the RI p-value stays finite.

        BDM fn 12: the RI p-value is count-based and valid even when the
        permutation SE is degenerate, intentionally departing from the
        bootstrap-NaN contract (non-finite SE -> full NaN tuple). With a single
        permutation, ``se = std([x], ddof=1)`` is NaN, so ``t_stat`` is NaN, yet
        the count-based p-value remains finite.
        """
        panel = _fixed_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # ddof=1 on a 1-element array
            res = permutation_test(
                panel,
                outcome="y",
                treatment="treatment",
                post="t",
                unit="unit",
                n_permutations=1,
                seed=5,
            )
        assert not np.isfinite(res.se)  # NaN at n_valid == 1
        assert np.isnan(res.t_stat)
        assert np.isfinite(res.p_value)  # the contract: p-value survives

    def test_leave_one_out_all_fail_raises(self):
        """A single treated unit -> no LOO estimate possible -> fail-closed."""
        rows = [
            {"unit": u, "t": t, "y": float(u) + 0.5 * t + 0.1 * u * t, "treatment": int(u == 0)}
            for u in range(5)
            for t in range(2)
        ]
        panel = pd.DataFrame(rows)
        with pytest.raises(RuntimeError, match="leave-one-out"):
            leave_one_out_test(panel, outcome="y", treatment="treatment", post="t", unit="unit")
