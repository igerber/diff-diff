"""Tests for estimator short aliases.

Covers the kept aliases (silent identity), the M-062 SCM introduction,
and the M-132..M-134 alias diet served by the M-135 module
``__getattr__`` (FutureWarning naming the surviving class; names stay
in ``__all__`` through 3.9 but leave module globals / ``dir()``).
"""

import re
import warnings

import pytest

import diff_diff

# The three dieted aliases (M-132..M-134): removed from module globals in
# 3.9, served by the M-135 module __getattr__ with a FutureWarning until
# the 4.0 removal.
DIETED_ALIASES = {
    "CDiD": "ContinuousDiD",
    "Gardner": "TwoStageDiD",
    "Stacked": "StackedDiD",
}


def _alias_message(name: str, target: str) -> str:
    return f"diff_diff.{name} is deprecated and will be removed in 4.0; use diff_diff.{target}."


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
    """All aliases are listed in __all__ (the dieted three stay through 3.9)."""
    aliases = [
        "DiD",
        "TWFE",
        "EventStudy",
        "SDiD",
        "CS",
        "CDiD",
        "SA",
        "BJS",
        "Gardner",
        "DDD",
        "SCM",
        "Stacked",
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
    """M-132..M-134 dieted aliases + the M-135 __getattr__ mechanism."""

    @pytest.mark.parametrize("name,target", sorted(DIETED_ALIASES.items()))
    def test_getattr_route_warns_once_and_resolves(self, name, target):
        """Plain attribute access: ONE __getattr__ hit -> one warning."""
        with pytest.warns(FutureWarning, match=re.escape(_alias_message(name, target))):
            obj = getattr(diff_diff, name)
        assert obj is getattr(diff_diff, target)

    @pytest.mark.parametrize("name,target", sorted(DIETED_ALIASES.items()))
    def test_from_import_route_warns_and_resolves(self, name, target):
        """``from diff_diff import <alias>`` still works and warns.

        Asserted SET-WISE, not by count: for a PACKAGE, the import
        protocol resolves each fromlist name twice (``_handle_fromlist``
        hasattr-probes the name before the IMPORT_FROM opcode getattrs
        it), so a single from-import records TWO copies of the same
        FutureWarning under ``simplefilter("always")``.
        """
        ns: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(f"from diff_diff import {name}", ns)  # noqa: S102
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert fw, "from-import fired no FutureWarning"
        assert {str(w.message) for w in fw} == {_alias_message(name, target)}
        assert ns[name] is getattr(diff_diff, target)

    def test_dieted_aliases_stay_in_all(self):
        for name in DIETED_ALIASES:
            assert name in diff_diff.__all__, f"{name} must stay in __all__ through 3.9"

    def test_dieted_aliases_left_module_globals(self):
        """The diet removes the names from dir()/vars(); SCM is present."""
        dir_names = {str(n) for n in dir(diff_diff)}
        vars_names = set(vars(diff_diff))
        for name in DIETED_ALIASES:
            assert name not in dir_names, f"{name} still in dir(diff_diff)"
            assert name not in vars_names, f"{name} still in vars(diff_diff)"
        assert "SCM" in dir_names and "SCM" in vars_names

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(
            AttributeError, match=re.escape("module 'diff_diff' has no attribute 'NotAnAlias'")
        ):
            diff_diff.NotAnAlias

    def test_star_import_fires_alias_warnings_setwise(self):
        """Star-import serves the three dieted aliases via __getattr__.

        Asserted SET-WISE, not by count: diff_diff is a PACKAGE, so
        ``_handle_fromlist`` hasattr-probes every missing ``__all__``
        name (deciding whether it names a submodule) before
        ``import_all_from`` getattrs it — two ``__getattr__`` hits per
        alias, i.e. 6 recorded warnings for 3 aliases under
        ``simplefilter("always")``. The exact count is an importlib
        implementation detail; the contract is the message SET.
        """
        expected = {_alias_message(n, t) for n, t in DIETED_ALIASES.items()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec("from diff_diff import *", {})  # noqa: S102
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        got = {str(w.message) for w in fw}
        assert got == expected, f"unexpected FutureWarning set: {got ^ expected}"
        # Every recorded FutureWarning is one of the three alias messages.
        assert all(str(w.message) in expected for w in fw)

    def test_star_import_binds_dieted_aliases(self):
        """The names remain star-importable and identity holds."""
        ns: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            exec("from diff_diff import *", ns)  # noqa: S102
        for name, target in DIETED_ALIASES.items():
            assert ns[name] is getattr(diff_diff, target), name

    def test_scm_alias_is_silent(self):
        """M-062 introduce-only alias: no warning on access."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert diff_diff.SCM is diff_diff.SyntheticControl
