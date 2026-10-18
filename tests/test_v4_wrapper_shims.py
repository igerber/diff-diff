"""Wrapper removal pins (rows M-070..M-077, removed at the 4.0 cut).

The eight module-level convenience wrappers (``imputation_did``,
``two_stage_did``, ``stacked_did``, ``trop``, ``synthetic_control``,
``triple_difference``, ``bacon_decompose``, ``chaisemartin_dhaultfoeuille``)
were deprecated in 3.9 and deleted in 4.0; classes are the single canonical
construction surface. This suite is the rows' shared ``test_ref``. Per
wrapper it pins:

1. SMOKE: the explicit class path (the wrapper's documented replacement)
   fits the same shape of call the wrapper used to take.
2. ABSENCE: the name is not in ``diff_diff.__all__``. For the four names
   that do NOT collide with a submodule (``imputation_did``,
   ``two_stage_did``, ``triple_difference``, ``bacon_decompose``)
   ``from diff_diff import <name>`` raises ``ImportError`` and
   ``diff_diff.<name>`` raises ``AttributeError``. The other four
   (``trop``, ``synthetic_control``, ``stacked_did``,
   ``chaisemartin_dhaultfoeuille``) share their SUBMODULE's name, so the
   from-import succeeds and yields the module object (as
   ``diff_diff.bacon`` / ``diff_diff.imputation`` always did); for those
   the pin is ``inspect.ismodule(diff_diff.<name>)`` - a resurrected
   function there would fail it.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import (
    TROP,
    BaconDecomposition,
    ChaisemartinDHaultfoeuille,
    ImputationDiD,
    StackedDiD,
    SyntheticControl,
    TripleDifference,
    TwoStageDiD,
)

NON_COLLIDING = ("imputation_did", "two_stage_did", "triple_difference", "bacon_decompose")
MODULE_COLLIDING = ("trop", "synthetic_control", "stacked_did", "chaisemartin_dhaultfoeuille")


def _assert_removed_non_colliding(name: str) -> None:
    assert name not in diff_diff.__all__
    with pytest.raises(AttributeError):
        getattr(diff_diff, name)
    with pytest.raises(ImportError):
        exec(f"from diff_diff import {name}", {})  # noqa: S102


def _assert_removed_module_colliding(name: str) -> None:
    assert name not in diff_diff.__all__
    obj = getattr(diff_diff, name)
    assert inspect.ismodule(obj), f"diff_diff.{name} must be the submodule, got {obj!r}"
    assert obj.__name__ == f"diff_diff.{name}"
    ns: dict = {}
    exec(f"from diff_diff import {name}", ns)  # noqa: S102
    assert ns[name] is obj


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


class TestImputationDidRemoved:
    """M-070."""

    def test_class_path_smoke(self, staggered_panel):
        res = ImputationDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.att)

    def test_wrapper_absent(self):
        _assert_removed_non_colliding("imputation_did")


class TestTwoStageDidRemoved:
    """M-071."""

    def test_class_path_smoke(self, staggered_panel):
        res = TwoStageDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.att)

    def test_wrapper_absent(self):
        _assert_removed_non_colliding("two_stage_did")


class TestStackedDidRemoved:
    """M-072 (name collides with ``diff_diff.stacked_did``, the submodule)."""

    def test_class_path_smoke(self, staggered_panel):
        res = StackedDiD().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.att)

    def test_wrapper_absent_name_is_submodule(self):
        _assert_removed_module_colliding("stacked_did")
        assert diff_diff.stacked_did.StackedDiD is StackedDiD


class TestTropRemoved:
    """M-073 (name collides with ``diff_diff.trop``, the submodule)."""

    def test_class_path_smoke(self, absorbing_panel, ci_params):
        # Narrowed regularization grid + CI-scaled replicate count (the
        # trop-heavy-tests hazard): this is a construction smoke, not a
        # methodology test.
        res = TROP(
            n_bootstrap=ci_params.bootstrap(19),
            lambda_time_grid=[0.1],
            lambda_unit_grid=[0.1],
            lambda_nn_grid=[0.1],
            seed=42,
        ).fit(absorbing_panel, "y", "d", "unit", "time")
        assert np.isfinite(res.att)

    def test_wrapper_absent_name_is_submodule(self):
        _assert_removed_module_colliding("trop")
        assert diff_diff.trop.TROP is TROP


class TestSyntheticControlRemoved:
    """M-074 (name collides with ``diff_diff.synthetic_control``, the submodule)."""

    def test_class_path_smoke(self, scm_panel):
        res = SyntheticControl(seed=1).fit(scm_panel, "y", "treated", "unit", "time")
        assert np.isfinite(res.att)

    def test_wrapper_absent_name_is_submodule(self):
        _assert_removed_module_colliding("synthetic_control")
        assert diff_diff.synthetic_control.SyntheticControl is SyntheticControl


class TestTripleDifferenceRemoved:
    """M-075."""

    def test_class_path_smoke(self, ddd_panel):
        res = TripleDifference().fit(
            data=ddd_panel, outcome="y", group="group", partition="partition", post="post"
        )
        assert np.isfinite(res.att)

    def test_wrapper_absent(self):
        _assert_removed_non_colliding("triple_difference")


class TestBaconDecomposeRemoved:
    """M-076."""

    def test_class_path_smoke(self, staggered_panel):
        res = BaconDecomposition().fit(
            staggered_panel, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        assert np.isfinite(res.twfe_estimate)
        assert not res.to_dataframe().empty

    def test_wrapper_absent(self):
        _assert_removed_non_colliding("bacon_decompose")


class TestChaisemartinDhaultfoeuilleRemoved:
    """M-077 (name collides with ``diff_diff.chaisemartin_dhaultfoeuille``, the submodule)."""

    def test_class_path_smoke(self, staggered_panel):
        res = ChaisemartinDHaultfoeuille().fit(
            staggered_panel, outcome="y", unit="unit", time="time", treatment="treated"
        )
        assert np.isfinite(res.att)

    def test_wrapper_absent_name_is_submodule(self):
        _assert_removed_module_colliding("chaisemartin_dhaultfoeuille")
        assert (
            diff_diff.chaisemartin_dhaultfoeuille.ChaisemartinDHaultfoeuille
            is ChaisemartinDHaultfoeuille
        )


class TestWrapperRosterComplete:
    """The two absence shapes together cover exactly the eight M-070..M-077 names."""

    def test_roster(self):
        assert len(NON_COLLIDING) + len(MODULE_COLLIDING) == 8
        for name in NON_COLLIDING + MODULE_COLLIDING:
            assert name not in diff_diff.__all__, name
