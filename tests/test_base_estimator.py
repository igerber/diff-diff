"""Cross-estimator contract suite for the shared BaseEstimator mixin.

Every estimator's ``get_params``/``set_params`` comes from
``diff_diff._base.BaseEstimator`` (v4-design section 7, 2(c)-i). This suite
pins the shared contract ONCE, parametrized over a DYNAMICALLY discovered
roster (never a hand-listed set, so a future estimator cannot silently skip
the contract):

- roster completeness: every ``diff_diff.__all__`` class with a public
  ``fit`` method must have ``BaseEstimator`` in its MRO (explicit exclusion
  tuple below, reasons attached);
- init-signature <-> get_params sync;
- ``cls(**est.get_params())`` re-instantiation round-trip;
- strict unknown-key rejection (method names and private attrs included)
  with batch atomicity;
- value-level transactional rollback (constructor-validated classes);
- ``deep=`` triple-equality; returns-self;
- ``set_params`` == fresh-construction equivalence, stated POST the
  ``_normalize_set_params`` hook (so DiD's robust-alone re-derivation is
  inside the contract, not an exemption);
- fitted-state attrs are never touched by ``set_params``;
- ``sklearn.base.clone`` round-trip under ``importorskip`` (config-equality
  is the always-running contract; clone identity is opportunistic local
  coverage - scikit-learn is deliberately not a dev dependency).
"""

import inspect
import warnings

import pytest

import diff_diff
from diff_diff._base import BaseEstimator

# ---------------------------------------------------------------------------
# Dynamic roster discovery
# ---------------------------------------------------------------------------

# __all__ classes with a public `fit` that deliberately do NOT adopt the
# estimator param contract. Reasons required.
NON_PARTICIPANTS = {
    # Internal OLS engine exported for power users; predates the estimator
    # contract and carries fit-time (not constructor) configuration. Adding
    # get_params/set_params is a NEW surface tracked in TODO.md, not part of
    # the 2(c)-i refactor.
    "LinearRegression",
}


def _discover():
    mixin, fit_classes, seen = [], [], set()
    for name in diff_diff.__all__:
        obj = getattr(diff_diff, name)
        if not inspect.isclass(obj) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if issubclass(obj, BaseEstimator):
            mixin.append(obj)
        fit = inspect.getattr_static(obj, "fit", None)
        if fit is not None and (
            inspect.isfunction(fit) or isinstance(fit, (staticmethod, classmethod))
        ):
            fit_classes.append(obj)
    return mixin, fit_classes


MIXIN_CLASSES, FIT_CLASSES = _discover()

# Required constructor kwargs for classes that cannot default-construct.
DEFAULT_KWARGS = {
    "SpilloverDiD": {"rings": [0.0, 50.0, 100.0]},
}

# One known constructor-rejected value per class WITH constructor validation
# (value-level rollback lane). Classes absent here are still covered by the
# unknown-key atomicity lane.
BAD_VALUES = {
    "DifferenceInDifferences": {"vcov_type": "hc99"},
    "TwoWayFixedEffects": {"vcov_type": "hc99"},
    "MultiPeriodDiD": {"vcov_type": "hc99"},
    "SyntheticDiD": {"variance_method": "not_a_method"},
    "CallawaySantAnna": {"vcov_type": "hc4"},
    "SunAbraham": {"vcov_type": "hc4"},
    "ImputationDiD": {"vcov_type": "hc4"},
    "TwoStageDiD": {"vcov_type": "hc4"},
    "TripleDifference": {"vcov_type": "hc4"},
    "EfficientDiD": {"vcov_type": "hc4"},
    "ContinuousDiD": {"n_bootstrap": -3},
    "StackedDiD": {"control_group": "not_a_mode"},
    "LPDiD": {"alpha": 5.0},
    "ChangesInChanges": {"alpha": 5.0},
    "QDiD": {"alpha": 5.0},
    "HeterogeneousAdoptionDiD": {"design": "not_a_design"},
    "RegressionDiscontinuity": {"kernel": "not_a_kernel"},
    "ChaisemartinDHaultfoeuille": {"cluster": "unit"},  # NotImplementedError gate
    "TROP": {"method": "not_a_method"},
    "PreTrendsPower": {"alpha": 5.0},
    "WooldridgeDiD": {"method": "not_a_method"},
    "SpilloverDiD": {"rank_deficient_action": "not_an_action"},
    "StaggeredTripleDifference": {"estimation_method": "not_a_method"},
    "BaconDecomposition": {"weights": "not_a_weighting"},
}

# A safe single-param mutation per class for the equivalence/returns-self
# lanes. Defaults to alpha/seed when present.
MUTATIONS = {
    "BaconDecomposition": {"weights": "approximate"},
}


def _make(cls):
    return cls(**DEFAULT_KWARGS.get(cls.__name__, {}))


def _mutation_for(cls, params):
    if cls.__name__ in MUTATIONS:
        return MUTATIONS[cls.__name__]
    if "alpha" in params:
        return {"alpha": 0.11}
    if "seed" in params:
        return {"seed": 1234}
    return None


def _params_id(cls):
    return cls.__name__


estimators = pytest.mark.parametrize("cls", MIXIN_CLASSES, ids=_params_id)


# ---------------------------------------------------------------------------
# Roster completeness
# ---------------------------------------------------------------------------


def test_roster_discovered_nontrivially():
    # 25 defining classes + TwoWayFixedEffects/MultiPeriodDiD inheritors.
    assert len(MIXIN_CLASSES) >= 27


def test_every_fit_class_is_a_base_estimator():
    missing = [
        cls.__name__
        for cls in FIT_CLASSES
        if cls.__name__ not in NON_PARTICIPANTS and not issubclass(cls, BaseEstimator)
    ]
    assert missing == [], (
        f"exported fit-classes without BaseEstimator in their MRO: {missing}; "
        "either mix it in or add a reasoned NON_PARTICIPANTS entry"
    )


def test_non_participant_entries_are_live():
    # A dead exclusion reads as load-bearing; every entry must name an
    # exported fit-class that really lacks the mixin.
    fit_names = {cls.__name__ for cls in FIT_CLASSES}
    for name in NON_PARTICIPANTS:
        assert name in fit_names, f"stale NON_PARTICIPANTS entry: {name}"
        assert not issubclass(getattr(diff_diff, name), BaseEstimator)


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------


@estimators
def test_init_signature_matches_get_params(cls):
    est = _make(est_cls := cls)
    sig = set(inspect.signature(est_cls.__init__).parameters) - {"self"}
    assert set(est.get_params()) == sig


# Classes whose CONSTRUCTION deliberately warns during a deprecation window
# (3.9 class merges, v4-design section 4.1). The round-trip test's
# warnings-as-errors filter exists to catch UNEXPECTED warnings from a
# re-init; these messages are expected by contract and are ignored inside
# the error filter. Forward home for the remaining phase-3 sibling (QDiD).
DEPRECATED_CLASS_WARNINGS = {
    "MultiPeriodDiD": r"MultiPeriodDiD is deprecated",
    "StaggeredTripleDifference": r"StaggeredTripleDifference is deprecated",
}


@estimators
def test_reinstantiation_round_trip(cls):
    est = _make(cls)
    params = est.get_params()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _expected = DEPRECATED_CLASS_WARNINGS.get(cls.__name__)
        if _expected is not None:
            warnings.filterwarnings("ignore", message=_expected, category=FutureWarning)
        clone = cls(**params)
    assert clone.get_params() == params


@estimators
def test_deep_triple_equality(cls):
    est = _make(cls)
    assert est.get_params() == est.get_params(deep=True) == est.get_params(deep=False)


@estimators
def test_unknown_key_raises_and_batch_is_atomic(cls):
    est = _make(cls)
    before = est.get_params()
    first_key = next(iter(before))
    with pytest.raises(ValueError, match="Unknown parameter"):
        est.set_params(**{first_key: before[first_key], "definitely_not_a_param": 1})
    assert est.get_params() == before


@estimators
def test_method_names_and_private_attrs_rejected(cls):
    est = _make(cls)
    with pytest.raises(ValueError, match="Unknown parameter"):
        est.set_params(fit=lambda: None)
    with pytest.raises(ValueError, match="Unknown parameter"):
        est.set_params(_private_attr=42)


@estimators
def test_bad_value_rolls_back(cls):
    bad = BAD_VALUES.get(cls.__name__)
    if bad is None:
        pytest.skip("no constructor-validated bad value catalogued")
    est = _make(cls)
    before = est.get_params()
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        est.set_params(**bad)
    assert est.get_params() == before


@estimators
def test_set_params_returns_self_and_matches_reinit(cls):
    est = _make(cls)
    params = est.get_params()
    mutation = _mutation_for(cls, params)
    if mutation is None:
        pytest.skip("no safe single-param mutation catalogued")
    result = est.set_params(**mutation)
    assert result is est
    # Post-normalization-hook equivalence: config state equals a fresh
    # construction from the merged params.
    normalized = type(est)._normalize_set_params(dict(mutation))
    expected = cls(**{**params, **normalized})
    assert est.get_params() == expected.get_params()
    for attr in type(est)._DERIVED_CONFIG_ATTRS:
        assert getattr(est, attr) == getattr(expected, attr), attr


@estimators
def test_fitted_state_untouched_by_set_params(cls):
    est = _make(cls)
    sentinels = {}
    # `results_` is a read-only property on some classes (WooldridgeDiD keeps
    # fitted state at `_results`); only instance-dict attrs can host a sentinel.
    for attr in ("is_fitted_", "results_", "_results"):
        if attr in vars(est):
            sentinels[attr] = object()
            setattr(est, attr, sentinels[attr])
    if not sentinels:
        pytest.skip("class holds no pre-declared fitted-state attrs")
    mutation = _mutation_for(cls, est.get_params()) or {}
    est.set_params(**mutation)
    for attr, sentinel in sentinels.items():
        assert getattr(est, attr) is sentinel, attr


@estimators
def test_sklearn_clone_round_trip_if_available(cls):
    base = pytest.importorskip("sklearn.base")
    est = _make(cls)
    cloned = base.clone(est)
    assert cloned is not est
    assert type(cloned) is type(est)
    assert cloned.get_params() == est.get_params()


# ---------------------------------------------------------------------------
# DiD robust-alone normalization hook (H1) - the one non-identity hook
# ---------------------------------------------------------------------------


def test_did_robust_alone_rederives_vcov_type():
    est = diff_diff.DifferenceInDifferences(vcov_type="hc2")
    est.set_params(robust=False)
    assert est.vcov_type == "classical"
    assert est._vcov_type_arg is None
    assert est._vcov_type_explicit is False
