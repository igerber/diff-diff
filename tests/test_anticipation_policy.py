"""Family-wide ``anticipation`` domain policy (ledger row M-144).

The contract: every anticipation-taking estimator validates the param via
the shared ``diff_diff.utils.validate_anticipation`` at ``__init__``
(``set_params`` inherits transactionally via the BaseEstimator probe
re-init) AND re-checks it on the fit path — the uniform direct-mutation
defense, using the ASSIGNMENT form ``self.anticipation =
validate_anticipation(self.anticipation)`` so a mutated numpy scalar is
normalized to a Python ``int`` before any ``g - 1 - anticipation``-style
arithmetic can overflow on an unsigned scalar.

Roster notes:

- SpilloverDiD's fit-path pins live in ``tests/test_spillover.py`` (its
  keyword-only XOR ``treatment``/``first_treat`` interface doesn't fit
  this suite's shared invocation); the other adopters are pinned HERE in
  the mutation lanes.
- The deprecated ``StaggeredTripleDifference`` stays
  construction-permissive BY DESIGN (frozen 3.x API shape; the shared
  staggered engine validates at fit) — pinned by
  test_v4_merge_ddd.py::test_deprecated_sibling_also_fails_closed_on_anticipation,
  and deliberately NOT in this roster.

Message pins match the FULL text via ``re.escape`` (repo convention),
except numpy-scalar values whose repr is NEP-51-dependent — those use the
shared prefix, matching the existing test_v4_merge_ddd pin.
"""

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import (
    CallawaySantAnna,
    ContinuousDiD,
    EfficientDiD,
    ImputationDiD,
    SpilloverDiD,
    StackedDiD,
    StaggeredTripleDifference,
    SunAbraham,
    TripleDifference,
    TwoStageDiD,
    WooldridgeDiD,
    generate_staggered_data,
)
from diff_diff._base import BaseEstimator

# ===========================================================================
# Constants
# ===========================================================================

ANTICIPATION_MSG_PREFIX = "anticipation must be a non-negative integer"

# The nine sweep adopters + TripleDifference (validated since birth).
VALIDATED_CLASSES = [
    CallawaySantAnna,
    SunAbraham,
    ImputationDiD,
    TwoStageDiD,
    StackedDiD,
    ContinuousDiD,
    EfficientDiD,
    WooldridgeDiD,
    SpilloverDiD,
    TripleDifference,
]

_CTOR_KWARGS = {SpilloverDiD: {"rings": [0.0, 100.0]}}

# The full nine-value set from test_v4_merge_ddd.py's staggered-DDD pin.
BAD_VALUES = [-1, -5, 1.5, 0.5, "1", None, True, False, np.float64(2.0)]


def _make(cls, **overrides):
    return cls(**{**_CTOR_KWARGS.get(cls, {}), **overrides})


def _expected_message(value):
    """The validator's full message for ``value``, per its branch rule.

    ONLY non-bool negative ints take the NEGATIVE branch (no type suffix);
    everything else — bools included, because the ``isinstance(..., bool)``
    guard precedes the integer check — takes the TYPE branch.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        return f"{ANTICIPATION_MSG_PREFIX}; got {value!r}."
    return f"{ANTICIPATION_MSG_PREFIX}; got {value!r} (type {type(value).__name__})."


def _match_for(value):
    """re pattern for ``value``: full text, except NEP-51-variable reprs.

    ``np.float64(2.0)`` reprs as ``np.float64(2.0)`` on numpy>=2 but ``2.0``
    on the declared 1.20 floor, so numpy scalars pin the PREFIX only (the
    same reason the existing test_v4_merge_ddd pin is prefix-matched).
    """
    if isinstance(value, np.generic):
        return re.escape(ANTICIPATION_MSG_PREFIX)
    return re.escape(_expected_message(value))


_ids = [f"{v!r}" for v in BAD_VALUES]


# ===========================================================================
# Roster guard
# ===========================================================================


class TestAnticipationRoster:
    def test_anticipation_exposed_by_exactly_the_validated_roster(self):
        discovered, seen = [], set()
        for name in diff_diff.__all__:
            obj = getattr(diff_diff, name)
            if not isinstance(obj, type) or id(obj) in seen:
                continue
            seen.add(id(obj))
            if issubclass(obj, BaseEstimator):
                params = inspect.signature(obj.__init__).parameters
                if "anticipation" in params:
                    discovered.append(obj)
        assert set(discovered) == set(VALIDATED_CLASSES) | {StaggeredTripleDifference}, (
            "The `anticipation` roster changed. A future estimator exposing "
            "`anticipation` must join the family-wide validation policy "
            "(ledger row M-144): validate via utils.validate_anticipation at "
            "__init__ AND re-check on the fit path via the assignment form. "
            "The deprecated StaggeredTripleDifference is the one documented "
            "construction-permissive exception (fit-validated via the shared "
            "engine)."
        )


# ===========================================================================
# Lane 1: bad values raise at __init__
# ===========================================================================


class TestConstructorValidation:
    @pytest.mark.parametrize("cls", VALIDATED_CLASSES)
    @pytest.mark.parametrize("value", BAD_VALUES, ids=_ids)
    def test_bad_value_raises_at_init(self, cls, value):
        with pytest.raises(ValueError, match=_match_for(value)):
            _make(cls, anticipation=value)


# ===========================================================================
# Lane 2: boundary / accepted values normalize to Python int
# ===========================================================================


class TestAcceptedValuesNormalize:
    @pytest.mark.parametrize("cls", VALIDATED_CLASSES)
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0), (1, 1), (np.int64(2), 2), (np.uint64(2), 2)],
        ids=["0", "1", "np.int64(2)", "np.uint64(2)"],
    )
    def test_accepted_value_round_trips_as_int(self, cls, value, expected):
        est = _make(cls, anticipation=value)
        # `type is int` is the real pin: `np.uint64(2) == 2` is True even
        # without normalization, so a bare equality check would be vacuous.
        assert type(est.anticipation) is int
        assert est.anticipation == expected
        assert type(est.get_params()["anticipation"]) is int


# ===========================================================================
# Lane 3: set_params raises and rolls back
# ===========================================================================


class TestSetParamsTransactional:
    @pytest.mark.parametrize("cls", VALIDATED_CLASSES)
    def test_set_params_bad_value_rolls_back(self, cls):
        est = _make(cls, anticipation=1)
        before = est.get_params()
        with pytest.raises(ValueError, match=re.escape(ANTICIPATION_MSG_PREFIX)):
            est.set_params(anticipation=-1)
        assert est.get_params() == before


# ===========================================================================
# Lane 4: fit-time direct-mutation defense
# ===========================================================================

# The nine minus SpilloverDiD (whose fit-path pins live in
# tests/test_spillover.py) and TripleDifference (staggered fit needs a
# partition column; covered in lane 4b instead).
_FIT_MUTATION_CLASSES = [
    CallawaySantAnna,
    SunAbraham,
    ImputationDiD,
    TwoStageDiD,
    StackedDiD,
    ContinuousDiD,
    EfficientDiD,
    WooldridgeDiD,
]


def _fit_kwargs_for(cls):
    kwargs = dict(outcome="y", unit="u", time="t", first_treat="g")
    if cls is ContinuousDiD:
        kwargs["dose"] = "d"
    return kwargs


class TestFitMutationDefense:
    @pytest.mark.parametrize("cls", _FIT_MUTATION_CLASSES)
    def test_mutated_anticipation_raises_at_fit(self, cls):
        """``True`` is the discriminating mutation: bool is an int subclass
        and not negative, so any legacy ``< 0``-style check accepts it —
        only the shared helper rejects it. Each estimator's re-check runs
        before any column handling, so a bare empty DataFrame suffices."""
        est = _make(cls)
        est.anticipation = True
        with pytest.raises(ValueError, match=re.escape(ANTICIPATION_MSG_PREFIX)):
            est.fit(pd.DataFrame(), **_fit_kwargs_for(cls))


# ===========================================================================
# Lane 4b: fit-path re-check NORMALIZES (assignment form, not validate-only)
# ===========================================================================


@pytest.fixture(scope="module")
def small_panel():
    """Public DGP: columns unit / period / outcome / first_treat."""
    return generate_staggered_data(n_units=30, n_periods=4, seed=1)


_PANEL_FIT_KWARGS = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


class TestFitNormalization:
    @pytest.mark.parametrize(
        "cls",
        [
            CallawaySantAnna,
            SunAbraham,
            ImputationDiD,
            TwoStageDiD,
            StackedDiD,
            ContinuousDiD,
            EfficientDiD,
            WooldridgeDiD,
        ],
    )
    def test_mutated_numpy_scalar_normalized_by_fit(self, cls, small_panel):
        """A mutated ``np.uint64`` must be re-assigned as a Python int by
        the fit-path re-check BEFORE any ``g - 1 - anticipation`` arithmetic
        (which raises OverflowError on unsigned scalars on SunAbraham /
        TwoStageDiD / StackedDiD, measured on numpy 2.4.5). A validate-only
        call that discards the helper's return fails this lane."""
        df = small_panel
        if cls is ContinuousDiD:
            df = df.copy()
            rng = np.random.default_rng(0)
            dose_map = {
                u: (rng.uniform(0.5, 2.0) if ft > 0 else 0.0)
                for u, ft in df.groupby("unit")["first_treat"].first().items()
            }
            df["dose"] = df["unit"].map(dose_map)
        est = _make(cls)
        est.anticipation = np.uint64(1)
        fit_kwargs = dict(_PANEL_FIT_KWARGS)
        if cls is ContinuousDiD:
            fit_kwargs["dose"] = "dose"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(df, **fit_kwargs)
        assert type(est.anticipation) is int
        assert est.anticipation == 1

    def test_triple_difference_engine_normalizes(self, small_panel):
        """Engine coverage: TripleDifference's staggered fit routes through
        the shared ``_staggered_triple_diff_engine`` re-check, whose
        assignment form must normalize a mutated numpy scalar too."""
        df = small_panel.copy()
        df["eligible"] = (df["unit"] % 2 == 0).astype(int)
        est = TripleDifference()
        est.anticipation = np.uint64(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(df, partition="eligible", **_PANEL_FIT_KWARGS)
        assert type(est.anticipation) is int
        assert est.anticipation == 1


# ===========================================================================
# Lane 4c: hausman_pretest classmethod normalizes its own argument
# ===========================================================================


class TestHausmanPretestNormalization:
    def test_uint64_matches_int_result(self, small_panel):
        """``EfficientDiD.hausman_pretest`` uses ``anticipation`` in its OWN
        event-time arithmetic (``e < -ant``). Without normalization,
        ``-np.uint64(1)`` silently WRAPS to 2**64-1 (no OverflowError) and
        the pretest degrades to an all-NaN inconclusive result — so the pin
        must compare against the Python-int result, not just assert no
        exception."""
        kwargs = dict(_PANEL_FIT_KWARGS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_int = EfficientDiD.hausman_pretest(small_panel, anticipation=1, **kwargs)
            r_u64 = EfficientDiD.hausman_pretest(small_panel, anticipation=np.uint64(1), **kwargs)
        assert np.isfinite(r_int.statistic) and np.isfinite(r_int.p_value)
        assert r_u64.statistic == r_int.statistic
        assert r_u64.p_value == r_int.p_value
        assert r_u64.df == r_int.df
