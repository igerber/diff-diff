"""Tests for estimator short aliases."""

import diff_diff


def test_alias_identity():
    """Each alias is the same class object as the full name."""
    assert diff_diff.DiD is diff_diff.DifferenceInDifferences
    assert diff_diff.TWFE is diff_diff.TwoWayFixedEffects
    assert diff_diff.EventStudy is diff_diff.MultiPeriodDiD
    assert diff_diff.SDiD is diff_diff.SyntheticDiD
    assert diff_diff.CS is diff_diff.CallawaySantAnna
    assert diff_diff.CDiD is diff_diff.ContinuousDiD
    assert diff_diff.SA is diff_diff.SunAbraham
    assert diff_diff.BJS is diff_diff.ImputationDiD
    assert diff_diff.Gardner is diff_diff.TwoStageDiD
    assert diff_diff.DDD is diff_diff.TripleDifference
    assert diff_diff.Stacked is diff_diff.StackedDiD
    assert diff_diff.Bacon is diff_diff.BaconDecomposition
    assert diff_diff.CiC is diff_diff.ChangesInChanges


def test_aliases_in_all():
    """All aliases are listed in __all__."""
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
