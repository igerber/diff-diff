"""Tests for short estimator aliases."""

import diff_diff


# Each (alias, canonical) pair to verify.
_ALIAS_PAIRS = [
    ("DiD", "DifferenceInDifferences"),
    ("TWFE", "TwoWayFixedEffects"),
    ("MPDiD", "MultiPeriodDiD"),
    ("SDiD", "SyntheticDiD"),
    ("CSA", "CallawaySantAnna"),
    ("SA", "SunAbraham"),
    ("IDiD", "ImputationDiD"),
    ("TSDiD", "TwoStageDiD"),
    ("DDD", "TripleDifference"),
    ("Bacon", "BaconDecomposition"),
]


class TestAliases:
    """Verify short aliases point to the same objects as the full names."""

    def test_aliases_are_identical(self):
        """Each alias must be the exact same object (``is``) as its canonical name."""
        for alias, canonical in _ALIAS_PAIRS:
            alias_obj = getattr(diff_diff, alias)
            canonical_obj = getattr(diff_diff, canonical)
            assert alias_obj is canonical_obj, (
                f"{alias} is not the same object as {canonical}"
            )

    def test_aliases_in_all(self):
        """All aliases must appear in ``__all__``."""
        for alias, _ in _ALIAS_PAIRS:
            assert alias in diff_diff.__all__, f"{alias} missing from __all__"

    def test_canonical_names_in_all(self):
        """Canonical names must still appear in ``__all__``."""
        for _, canonical in _ALIAS_PAIRS:
            assert canonical in diff_diff.__all__, (
                f"{canonical} missing from __all__"
            )

    def test_alias_import(self):
        """Aliases can be imported directly from the package."""
        from diff_diff import (
            DiD,
            TWFE,
            MPDiD,
            SDiD,
            CSA,
            SA,
            IDiD,
            TSDiD,
            DDD,
            Bacon,
        )
        # Smoke-check: all are callable classes
        for cls in (DiD, TWFE, MPDiD, SDiD, CSA, SA, IDiD, TSDiD, DDD, Bacon):
            assert callable(cls)
