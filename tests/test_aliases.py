"""Tests for estimator short aliases.

Covers the kept aliases (silent identity), the M-062 SCM introduction,
and the M-132..M-134 alias diet: ``CDiD`` / ``Gardner`` / ``Stacked`` left
module globals in 3.9 (served by the M-135 module ``__getattr__`` with a
FutureWarning) and were removed outright at the 4.0 cut - the
``__getattr__`` mechanism was retired with them, so the names now raise
``AttributeError`` / ``ImportError`` and are absent from ``__all__``.
"""

import re
import warnings

import pytest

import diff_diff

# The three dieted aliases (M-132..M-134) and the class each one named.
DIETED_ALIASES = {
    "CDiD": "ContinuousDiD",
    "Gardner": "TwoStageDiD",
    "Stacked": "StackedDiD",
}


def test_alias_identity():
    """Each kept alias is the same class object as the full name (silent)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert diff_diff.DiD is diff_diff.DifferenceInDifferences
        assert diff_diff.TWFE is diff_diff.TwoWayFixedEffects
        assert diff_diff.EventStudy is diff_diff.MultiPeriodDiD
        assert diff_diff.SDiD is diff_diff.SyntheticDiD
        assert diff_diff.CS is diff_diff.CallawaySantAnna
        assert diff_diff.SA is diff_diff.SunAbraham
        assert diff_diff.BJS is diff_diff.ImputationDiD
        assert diff_diff.DDD is diff_diff.TripleDifference
        assert diff_diff.Bacon is diff_diff.BaconDecomposition
        assert diff_diff.CiC is diff_diff.ChangesInChanges
        assert diff_diff.SCM is diff_diff.SyntheticControl


def test_aliases_in_all():
    """All kept aliases are listed in __all__."""
    aliases = [
        "DiD",
        "TWFE",
        "EventStudy",
        "SDiD",
        "CS",
        "SA",
        "BJS",
        "DDD",
        "SCM",
        "Bacon",
        "CiC",
    ]
    for alias in aliases:
        assert alias in diff_diff.__all__, f"{alias} missing from __all__"


def test_alias_instantiation():
    """Instantiating via alias produces the correct type."""
    model = diff_diff.DiD()
    assert isinstance(model, diff_diff.DifferenceInDifferences)


class TestAliasDiet:
    """M-132..M-134 removal pins (and the retired M-135 ``__getattr__``)."""

    @pytest.mark.parametrize("name", sorted(DIETED_ALIASES))
    def test_getattr_raises_attribute_error(self, name):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(diff_diff, name)

    @pytest.mark.parametrize("name", sorted(DIETED_ALIASES))
    def test_from_import_raises_import_error(self, name):
        with pytest.raises(ImportError):
            exec(f"from diff_diff import {name}", {})  # noqa: S102

    def test_dieted_aliases_not_in_all(self):
        for name in DIETED_ALIASES:
            assert name not in diff_diff.__all__, f"{name} must be absent from __all__ at 4.0"

    @pytest.mark.parametrize("target", sorted(DIETED_ALIASES.values()))
    def test_surviving_targets_still_exported(self, target):
        assert target in diff_diff.__all__
        assert isinstance(getattr(diff_diff, target), type)

    def test_dieted_aliases_left_module_globals(self):
        """The names are absent from dir()/vars(); SCM is present."""
        dir_names = {str(n) for n in dir(diff_diff)}
        vars_names = set(vars(diff_diff))
        for name in DIETED_ALIASES:
            assert name not in dir_names, f"{name} still in dir(diff_diff)"
            assert name not in vars_names, f"{name} still in vars(diff_diff)"
        assert "SCM" in dir_names and "SCM" in vars_names

    def test_module_getattr_mechanism_retired(self):
        """M-135: no module-level ``__getattr__`` serves aliases any more."""
        assert "__getattr__" not in vars(diff_diff)
        assert "_DEPRECATED_ALIASES" not in vars(diff_diff)

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(
            AttributeError, match=re.escape("module 'diff_diff' has no attribute 'NotAnAlias'")
        ):
            diff_diff.NotAnAlias

    def test_star_import_silent_and_does_not_bind_dieted_aliases(self):
        """Star-import records ZERO FutureWarnings and binds none of the three."""
        ns: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec("from diff_diff import *", ns)  # noqa: S102
        fw = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        assert fw == [], fw
        for name, target in DIETED_ALIASES.items():
            assert name not in ns, f"{name} must not be star-importable at 4.0"
            assert ns[target] is getattr(diff_diff, target)

    def test_scm_alias_is_silent(self):
        """M-062 introduce-only alias: no warning on access."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert diff_diff.SCM is diff_diff.SyntheticControl
