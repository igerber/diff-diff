"""Wrapper-deprecation shims (rows M-070..M-077, 2(d) PR-A).

The eight module-level convenience wrappers are deprecated in 3.9 and
removed in 4.0; classes are the single canonical construction surface.
This suite is the rows' shared ``test_ref``. Per wrapper it pins:

1. WARNING: one legacy call emits the uniform FutureWarning, matched on
   the FULL message via ``re.escape`` (never a bare template - the
   message is dense in regex metacharacters). ``bacon_decompose`` uses
   diagnostic wording (``BaconDecompositionResults`` subclasses the
   ``Diagnostic`` marker).
2. EQUIVALENCE (THE GATE): the wrapper is a pure construct+fit relay,
   so its result equals the explicit class path BIT-EXACTLY -
   ``assert_allclose(rtol=0, atol=0, equal_nan=True)`` over the full
   inference quintet (att/se/t_stat/p_value/conf_int). NaN-by-design
   fields compare NaN==NaN (SyntheticControl's quintet is all-NaN
   beside a finite att; additionally validated via
   ``assert_nan_inference``). ``BaconDecompositionResults`` has no
   att/se - its gate compares the enumerated diagnostic field list
   plus the ``comparisons`` payload via ``to_dataframe()``.
   Determinism: only TROP and SyntheticControl draw entropy - both
   sides share an explicit ``seed=`` (TROP additionally runs a
   CI-scaled ``n_bootstrap`` and a narrowed regularization grid);
   the other six are deterministic at their defaults.
3. SENTINEL FORWARDING (imputation_did / two_stage_did / stacked_did):
   a plain wrapper call fires EXACTLY ONE FutureWarning - the wrapper
   deprecation - never the fit-time aggregate warning; an explicit
   ``aggregate=`` fires both (wrapper first). dCDH is safe by
   construction (``**fit_kwargs`` - a plain call never passes
   aggregate).
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from diff_diff import (
    TROP,
    BaconDecomposition,
    ChaisemartinDHaultfoeuille,
    ImputationDiD,
    StackedDiD,
    SyntheticControl,
    TripleDifference,
    TwoStageDiD,
    bacon_decompose,
    chaisemartin_dhaultfoeuille,
    imputation_did,
    stacked_did,
    synthetic_control,
    triple_difference,
    trop,
    two_stage_did,
)

from .conftest import assert_nan_inference

QUINTET = ("att", "se", "t_stat", "p_value", "conf_int")


def _wrapper_message(name: str, cls: str, kind: str = "estimator") -> str:
    return (
        f"{name}() is deprecated and will be removed in 4.0; "
        f"construct the {kind} instead: {cls}(...).fit(data, ...)."
    )


def _assert_quintet_bit_equal(wrapped, direct):
    """THE GATE: full inference quintet, zero tolerance, NaN==NaN."""
    for field in QUINTET:
        w, d = getattr(wrapped, field), getattr(direct, field)
        assert_allclose(
            np.asarray(w, dtype=float),
            np.asarray(d, dtype=float),
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=f"wrapper/class divergence on {field}",
        )


@pytest.fixture(scope="module")
def staggered_panel():
    rng = np.random.default_rng(42)
    n_units, periods = 40, [1, 2, 3, 4, 5]
    rows = []
    for u in range(n_units):
        first = 0 if u >= 24 else (3 if u < 12 else 4)
        for t in periods:
            treated = int(first > 0 and t >= first)
            y = 1.0 + 0.3 * u / n_units + 0.2 * t + 0.8 * treated + rng.normal(0, 0.4)
            rows.append((u, t, first, treated, y))
    return pd.DataFrame(rows, columns=["unit", "time", "first_treat", "treated", "y"])


@pytest.fixture(scope="module")
def ddd_panel():
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame(
        {
            "group": np.repeat([0, 1], n // 2 * 4)[: n * 4],
            "partition": np.tile(np.repeat([0, 1], 2), n),
            "post": np.tile([0, 1], n * 2),
        }
    )
    df["y"] = (
        0.5
        + 0.2 * df.group
        + 0.1 * df.partition
        + 0.15 * df.post
        + 0.9 * df.group * df.partition * df.post
        + rng.normal(0, 0.3, len(df))
    )
    return df


@pytest.fixture(scope="module")
def scm_panel():
    rng = np.random.default_rng(3)
    units, periods = list(range(8)), list(range(1, 9))
    rows = []
    for u in units:
        base = 1.0 + 0.2 * u
        for t in periods:
            treated = int(u == 0 and t >= 6)
            y = base + 0.1 * t + 1.5 * treated + rng.normal(0, 0.1)
            rows.append((u, t, treated, y))
    return pd.DataFrame(rows, columns=["unit", "time", "treated", "y"])


@pytest.fixture(scope="module")
def absorbing_panel():
    rng = np.random.default_rng(11)
    n_units, periods = 12, list(range(1, 7))
    rows = []
    for u in range(n_units):
        first = 4 if u < 6 else 0
        for t in periods:
            d = int(first > 0 and t >= first)
            y = 0.5 + 0.1 * u + 0.2 * t + 1.0 * d + rng.normal(0, 0.3)
            rows.append((u, t, d, y))
    return pd.DataFrame(rows, columns=["unit", "time", "d", "y"])


class TestWrapperWarnings:
    """Pin 1: the uniform full-message FutureWarning, per wrapper."""

    def test_imputation_did_warns(self, staggered_panel):
        with pytest.warns(
            FutureWarning, match=re.escape(_wrapper_message("imputation_did", "ImputationDiD"))
        ):
            imputation_did(staggered_panel, "y", "unit", "time", "first_treat")

    def test_two_stage_did_warns(self, staggered_panel):
        with pytest.warns(
            FutureWarning, match=re.escape(_wrapper_message("two_stage_did", "TwoStageDiD"))
        ):
            two_stage_did(staggered_panel, "y", "unit", "time", "first_treat")

    def test_stacked_did_warns(self, staggered_panel):
        with pytest.warns(
            FutureWarning, match=re.escape(_wrapper_message("stacked_did", "StackedDiD"))
        ):
            stacked_did(staggered_panel, "y", "unit", "time", "first_treat")

    def test_trop_warns(self, absorbing_panel, ci_params):
        with pytest.warns(FutureWarning, match=re.escape(_wrapper_message("trop", "TROP"))):
            trop(
                absorbing_panel,
                "y",
                "d",
                "unit",
                "time",
                n_bootstrap=ci_params.bootstrap(19),
                lambda_time_grid=[0.1],
                lambda_unit_grid=[0.1],
                lambda_nn_grid=[0.1],
                seed=42,
            )

    def test_synthetic_control_warns(self, scm_panel):
        with pytest.warns(
            FutureWarning,
            match=re.escape(_wrapper_message("synthetic_control", "SyntheticControl")),
        ):
            synthetic_control(scm_panel, "y", "treated", "unit", "time", seed=1)

    def test_triple_difference_warns(self, ddd_panel):
        with pytest.warns(
            FutureWarning,
            match=re.escape(_wrapper_message("triple_difference", "TripleDifference")),
        ):
            triple_difference(
                ddd_panel, outcome="y", group="group", partition="partition", time="post"
            )

    def test_bacon_decompose_warns_diagnostic_wording(self, staggered_panel):
        # The Bacon carve-out: BaconDecompositionResults subclasses the
        # Diagnostic marker, so the message says "diagnostic", not
        # "estimator".
        with pytest.warns(
            FutureWarning,
            match=re.escape(
                _wrapper_message("bacon_decompose", "BaconDecomposition", kind="diagnostic")
            ),
        ):
            bacon_decompose(
                staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
            )

    def test_chaisemartin_dhaultfoeuille_warns(self, staggered_panel):
        with pytest.warns(
            FutureWarning,
            match=re.escape(
                _wrapper_message("chaisemartin_dhaultfoeuille", "ChaisemartinDHaultfoeuille")
            ),
        ):
            chaisemartin_dhaultfoeuille(
                staggered_panel, outcome="y", group="unit", time="time", treatment="treated"
            )


class TestWrapperEquivalence:
    """Pin 2 (THE GATE): wrapper == class path, zero tolerance."""

    def _wrapped(self, fn, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return fn(*args, **kwargs)

    def test_imputation_did_equivalent(self, staggered_panel):
        wrapped = self._wrapped(imputation_did, staggered_panel, "y", "unit", "time", "first_treat")
        direct = ImputationDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        _assert_quintet_bit_equal(wrapped, direct)

    def test_two_stage_did_equivalent(self, staggered_panel):
        wrapped = self._wrapped(two_stage_did, staggered_panel, "y", "unit", "time", "first_treat")
        direct = TwoStageDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        _assert_quintet_bit_equal(wrapped, direct)

    def test_stacked_did_equivalent(self, staggered_panel):
        wrapped = self._wrapped(stacked_did, staggered_panel, "y", "unit", "time", "first_treat")
        direct = StackedDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        _assert_quintet_bit_equal(wrapped, direct)

    @pytest.mark.slow
    def test_trop_equivalent(self, absorbing_panel, ci_params):
        # TROP draws bootstrap entropy: identical explicit seed on both
        # sides, CI-scaled replicate count, and a narrowed
        # regularization grid (the 200-replicate default over full
        # grids, run twice, is the trop-heavy-tests hazard).
        kw = dict(
            n_bootstrap=ci_params.bootstrap(19),
            lambda_time_grid=[0.1],
            lambda_unit_grid=[0.1],
            lambda_nn_grid=[0.1],
            seed=42,
        )
        wrapped = self._wrapped(trop, absorbing_panel, "y", "d", "unit", "time", **kw)
        direct = TROP(**kw).fit(absorbing_panel, "y", "d", "unit", "time")
        _assert_quintet_bit_equal(wrapped, direct)

    def test_synthetic_control_equivalent(self, scm_panel):
        # SyntheticControl draws optimizer-start entropy under
        # seed=None: identical explicit seed on both sides. Its
        # analytical quintet is all-NaN beside a finite att by design.
        wrapped = self._wrapped(
            synthetic_control, scm_panel, "y", "treated", "unit", "time", seed=1
        )
        direct = SyntheticControl(seed=1).fit(scm_panel, "y", "treated", "unit", "time")
        _assert_quintet_bit_equal(wrapped, direct)
        assert np.isfinite(wrapped.att)
        assert_nan_inference(
            {
                "se": wrapped.se,
                "t_stat": wrapped.t_stat,
                "p_value": wrapped.p_value,
                "conf_int": wrapped.conf_int,
            }
        )

    def test_triple_difference_equivalent(self, ddd_panel):
        wrapped = self._wrapped(
            triple_difference,
            ddd_panel,
            outcome="y",
            group="group",
            partition="partition",
            time="post",
        )
        direct = TripleDifference().fit(
            data=ddd_panel, outcome="y", group="group", partition="partition", post="post"
        )
        _assert_quintet_bit_equal(wrapped, direct)

    def test_bacon_decompose_equivalent(self, staggered_panel):
        # BaconDecompositionResults is a Diagnostic with NO att/se: the
        # gate compares the enumerated diagnostic field list, zero
        # tolerance, plus the comparisons payload via to_dataframe().
        wrapped = self._wrapped(
            bacon_decompose,
            staggered_panel,
            outcome="y",
            unit="unit",
            time="time",
            first_treat="first_treat",
        )
        direct = BaconDecomposition().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        for field in (
            "twfe_estimate",
            "total_weight_treated_vs_never",
            "total_weight_earlier_vs_later",
            "total_weight_later_vs_earlier",
            "weighted_avg_treated_vs_never",
            "weighted_avg_earlier_vs_later",
            "weighted_avg_later_vs_earlier",
            "decomposition_error",
        ):
            assert_allclose(
                getattr(wrapped, field),
                getattr(direct, field),
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=f"wrapper/class divergence on {field}",
            )
        for field in ("n_timing_groups", "n_never_treated", "n_obs", "n_always_treated_remapped"):
            assert getattr(wrapped, field) == getattr(direct, field), field
        pd.testing.assert_frame_equal(wrapped.to_dataframe(), direct.to_dataframe())

    def test_chaisemartin_dhaultfoeuille_equivalent(self, staggered_panel):
        wrapped = self._wrapped(
            chaisemartin_dhaultfoeuille,
            staggered_panel,
            outcome="y",
            group="unit",
            time="time",
            treatment="treated",
        )
        direct = ChaisemartinDHaultfoeuille().fit(
            staggered_panel, outcome="y", unit="unit", time="time", treatment="treated"
        )
        _assert_quintet_bit_equal(wrapped, direct)


class TestSentinelForwarding:
    """Pin 3: the NOT_SUPPLIED forwarding contract on the three wrappers
    that carry a deprecated ``aggregate`` sentinel (M-021/M-022/M-024).
    """

    @pytest.mark.parametrize(
        "fn",
        [imputation_did, two_stage_did, stacked_did],
        ids=["imputation_did", "two_stage_did", "stacked_did"],
    )
    def test_plain_wrapper_call_fires_exactly_one_warning(self, fn, staggered_panel):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(staggered_panel, "y", "unit", "time", "first_treat")
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1, [str(w.message) for w in fw]
        assert "is deprecated and will be removed in 4.0; construct the" in str(fw[0].message)

    @pytest.mark.parametrize(
        "fn,inner_match",
        [
            (imputation_did, "ImputationDiD.fit"),
            (two_stage_did, "TwoStageDiD.fit"),
            (stacked_did, "StackedDiD.fit"),
        ],
        ids=["imputation_did", "two_stage_did", "stacked_did"],
    )
    def test_explicit_aggregate_fires_both_warnings(self, fn, inner_match, staggered_panel):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(staggered_panel, "y", "unit", "time", "first_treat", aggregate="event_study")
        fw = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 2, fw
        # Wrapper deprecation first, then the fit-time aggregate shim.
        assert "construct the" in fw[0]
        assert inner_match in fw[1]
