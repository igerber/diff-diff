"""Inference-surface policies (rows M-081 + M-096, 2(d) PR-B).

This suite is both rows' shared ``test_ref``.

M-081 (``n_bootstrap`` semantic unification): the shared
``diff_diff.utils.validate_n_bootstrap`` (promoted from ChangesInChanges'
local validator) rejects bool/None/float/negative at ``__init__`` across
every estimator whose ``n_bootstrap`` was previously unvalidated; ``0``
stays legal at construction and still means bootstrap off on every
``> 0``-gated analytical lane. Transactional ``set_params`` inherits the
validation via the BaseEstimator probe re-init.

M-096 (fail-closed ``inference`` selector): the accepted value set is
exactly ``("analytical", "wild_bootstrap")`` (string-typed) at
``__init__`` and set_params; at fit, DiD wild-bootstrap without
``cluster=`` raises, and DiD/TWFE wild with ``n_bootstrap < 2`` raise
(``n_bootstrap ∈ {0, 1}`` previously ran WCR with too few draws and
returned a wild-labeled all-NaN inference tuple). The survey and Conley
front doors keep precedence (their ``NotImplementedError`` rejections fire
before the coherence checks). TWFE's unit auto-cluster stays;
MultiPeriodDiD's warn-and-analytical-fallback stays (n_bootstrap-
independent). The roster guard pins ``inference`` exposure in
``get_params()`` to exactly {DifferenceInDifferences, MultiPeriodDiD,
TwoWayFixedEffects}.

Message pins match the FULL text via ``re.escape`` (repo convention).
"""

import re

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import (
    CallawaySantAnna,
    ChangesInChanges,
    ContinuousDiD,
    DifferenceInDifferences,
    EfficientDiD,
    ImputationDiD,
    MultiPeriodDiD,
    QDiD,
    StaggeredTripleDifference,
    SunAbraham,
    TwoStageDiD,
    TwoWayFixedEffects,
    WooldridgeDiD,
)
from diff_diff._base import BaseEstimator
from diff_diff.survey import SurveyDesign
from tests.test_base_estimator import _make

# ---------------------------------------------------------------------------
# Pinned messages
# ---------------------------------------------------------------------------

N_BOOTSTRAP_MSG_PREFIX = "n_bootstrap must be a non-negative integer"

INFERENCE_MSG = "inference must be one of ('analytical', 'wild_bootstrap'), got {value!r}"

CLUSTER_REQUIRED_MSG = (
    "inference='wild_bootstrap' requires cluster=. The wild cluster "
    "bootstrap resamples at the cluster level; pass cluster= or use "
    "inference='analytical'."
)


def _floor_msg(n: int) -> str:
    return (
        f"inference='wild_bootstrap' requires n_bootstrap >= 2 "
        f"(got {n}). At least 2 replications are needed "
        f"for bootstrap inference; use inference='analytical' for "
        f"analytical SEs."
    )


MPD_FALLBACK_MSG = (
    "Wild bootstrap inference is not yet supported for MultiPeriodDiD. "
    "Using analytical inference instead."
)

# The M-081 sweep roster (9 previously-unvalidated classes) plus CiC/QDiD,
# whose local validator was the promotion source and now routes through the
# shared helper.
VALIDATED_CLASSES = [
    DifferenceInDifferences,
    MultiPeriodDiD,
    TwoWayFixedEffects,
    CallawaySantAnna,
    SunAbraham,
    EfficientDiD,
    ImputationDiD,
    TwoStageDiD,
    WooldridgeDiD,
    ContinuousDiD,
    StaggeredTripleDifference,
    ChangesInChanges,
    QDiD,
]

SELECTOR_CLASSES = [DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects]


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clustered_panel():
    """Two-period clustered DiD panel (8 clusters, healthy effect)."""
    rng = np.random.default_rng(0)
    n_units, periods = 40, 2
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), periods),
            "post": np.tile([0, 1], n_units),
            "cluster": np.repeat(np.arange(8), periods * 5),
        }
    )
    df["treated"] = (df["unit"] < 20).astype(int)
    df["w"] = 1.0
    df["y"] = 1.0 + 0.5 * df["treated"] * df["post"] + rng.normal(0, 1, len(df))
    return df


@pytest.fixture(scope="module")
def conley_panel(clustered_panel):
    df = clustered_panel.copy()
    rng = np.random.default_rng(1)
    df["lat"] = rng.uniform(-30, 30, len(df))
    df["lon"] = rng.uniform(-100, 100, len(df))
    return df


@pytest.fixture(scope="module")
def multi_period_panel():
    rng = np.random.default_rng(2)
    n_units, periods = 30, 4
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), periods),
            "time": np.tile(np.arange(periods), n_units),
        }
    )
    df["treated"] = (df["unit"] < 15).astype(int)
    df["y"] = 1.0 + 0.3 * df["treated"] * (df["time"] >= 2) + rng.normal(0, 1, len(df))
    return df


def _fit_did(est, df, **kw):
    return est.fit(df, outcome="y", treatment="treated", post="post", **kw)


def _fit_twfe(est, df, **kw):
    return est.fit(df, outcome="y", treatment="treated", time="post", unit="unit", **kw)


def _assert_finite_quintet(results):
    """The full inference tuple is finite together (never field-by-field)."""
    lo, hi = results.conf_int
    assert np.all(np.isfinite([results.att, results.se, results.t_stat, results.p_value, lo, hi]))


# ===========================================================================
# M-081: shared n_bootstrap validation
# ===========================================================================


class TestNBootstrapValidation:
    @pytest.mark.parametrize("cls", VALIDATED_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("bad", [-3, 1.5, True, None], ids=repr)
    def test_bad_value_raises_at_init(self, cls, bad):
        with pytest.raises(ValueError, match=re.escape(N_BOOTSTRAP_MSG_PREFIX)):
            cls(n_bootstrap=bad)

    def test_bad_value_message_echoes_value(self):
        with pytest.raises(ValueError, match=re.escape("got '-3'")):
            DifferenceInDifferences(n_bootstrap=-3)

    @pytest.mark.parametrize("cls", VALIDATED_CLASSES, ids=lambda c: c.__name__)
    def test_zero_stays_legal_at_construction(self, cls):
        est = cls(n_bootstrap=0)
        assert est.get_params()["n_bootstrap"] == 0

    @pytest.mark.parametrize("cls", VALIDATED_CLASSES, ids=lambda c: c.__name__)
    def test_numpy_integer_accepted(self, cls):
        est = cls(n_bootstrap=np.int64(5))
        assert est.get_params()["n_bootstrap"] == 5

    @pytest.mark.parametrize("cls", VALIDATED_CLASSES, ids=lambda c: c.__name__)
    def test_set_params_rejects_and_rolls_back(self, cls):
        est = cls()
        before = est.get_params()
        with pytest.raises(ValueError, match=re.escape(N_BOOTSTRAP_MSG_PREFIX)):
            est.set_params(n_bootstrap=-1)
        assert est.get_params() == before


# ===========================================================================
# M-096: fail-closed inference selector
# ===========================================================================


class TestInferenceSelector:
    @pytest.mark.parametrize("cls", SELECTOR_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("value", ["analytical", "wild_bootstrap"])
    def test_accepted_values_construct(self, cls, value):
        assert cls(inference=value).get_params()["inference"] == value

    @pytest.mark.parametrize("cls", SELECTOR_CLASSES, ids=lambda c: c.__name__)
    def test_invalid_spelling_raises_at_init(self, cls):
        with pytest.raises(ValueError, match=re.escape(INFERENCE_MSG.format(value="banana"))):
            cls(inference="banana")

    @pytest.mark.parametrize("cls", SELECTOR_CLASSES, ids=lambda c: c.__name__)
    def test_non_string_value_raises_at_init(self, cls):
        # Bare tuple membership admits a one-element ndarray via elementwise
        # __eq__; the isinstance guard must reject it.
        with pytest.raises(ValueError, match="inference must be one of"):
            cls(inference=np.array(["wild_bootstrap"]))

    def test_set_params_rejects_and_rolls_back(self):
        did = DifferenceInDifferences()
        before = did.get_params()
        with pytest.raises(ValueError, match=re.escape(INFERENCE_MSG.format(value="banana"))):
            did.set_params(inference="banana")
        assert did.get_params() == before

    # -- fit-level coherence: cluster requirement (DiD only) ----------------

    def test_did_wild_without_cluster_raises(self, clustered_panel):
        did = DifferenceInDifferences(inference="wild_bootstrap", n_bootstrap=99, seed=42)
        with pytest.raises(ValueError, match=re.escape(CLUSTER_REQUIRED_MSG)):
            _fit_did(did, clustered_panel)

    def test_did_wild_with_cluster_fits(self, clustered_panel, ci_params):
        n_boot = ci_params.bootstrap(99)
        did = DifferenceInDifferences(
            inference="wild_bootstrap", cluster="cluster", n_bootstrap=n_boot, seed=42
        )
        results = _fit_did(did, clustered_panel)
        assert results.inference_method == "wild_bootstrap"
        assert results.n_bootstrap == n_boot

    def test_twfe_auto_cluster_under_wild_fits(self, clustered_panel, ci_params):
        # TWFE has NO cluster-required check: cluster=None auto-clusters at
        # unit under wild bootstrap.
        twfe = TwoWayFixedEffects(
            cluster=None,
            inference="wild_bootstrap",
            n_bootstrap=ci_params.bootstrap(99),
            seed=42,
        )
        results = _fit_twfe(twfe, clustered_panel)
        assert results.inference_method == "wild_bootstrap"
        assert np.isfinite(results.se)

    # -- fit-level coherence: the n_bootstrap >= 2 floor --------------------

    @pytest.mark.parametrize("n", [0, 1])
    def test_did_wild_below_floor_raises(self, clustered_panel, n):
        # cluster= supplied so the raise comes from the FLOOR guard, not the
        # cluster guard (B2 checks cluster first).
        did = DifferenceInDifferences(
            inference="wild_bootstrap", cluster="cluster", n_bootstrap=n, seed=42
        )
        with pytest.raises(ValueError, match=re.escape(_floor_msg(n))):
            _fit_did(did, clustered_panel)

    @pytest.mark.parametrize("n", [0, 1])
    def test_twfe_wild_below_floor_raises(self, clustered_panel, n):
        twfe = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=n, seed=42)
        with pytest.raises(ValueError, match=re.escape(_floor_msg(n))):
            _fit_twfe(twfe, clustered_panel)

    def test_boundary_n_bootstrap_2_accepted_did(self, clustered_panel):
        # Boundary acceptance catches an erroneous `<= 2`. The label alone
        # derives from `_bootstrap_results is not None` (which a degenerate
        # run also satisfies), so the full quintet must be finite too.
        did = DifferenceInDifferences(
            inference="wild_bootstrap", cluster="cluster", n_bootstrap=2, seed=42
        )
        results = _fit_did(did, clustered_panel)
        assert results.inference_method == "wild_bootstrap"
        _assert_finite_quintet(results)

    def test_boundary_n_bootstrap_2_accepted_twfe(self, clustered_panel):
        twfe = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=2, seed=42)
        results = _fit_twfe(twfe, clustered_panel)
        assert results.inference_method == "wild_bootstrap"
        _assert_finite_quintet(results)

    # -- negative control: the floor must live inside the wild branch -------

    def test_did_analytical_with_zero_n_bootstrap_fits(self, clustered_panel):
        # M-081 keeps n_bootstrap=0 legal; a guard accidentally hoisted out
        # of the `inference == "wild_bootstrap"` block would break this.
        results = _fit_did(
            DifferenceInDifferences(inference="analytical", n_bootstrap=0), clustered_panel
        )
        assert results.inference_method == "analytical"
        _assert_finite_quintet(results)

    def test_twfe_analytical_with_zero_n_bootstrap_fits(self, clustered_panel):
        results = _fit_twfe(
            TwoWayFixedEffects(inference="analytical", n_bootstrap=0), clustered_panel
        )
        assert results.inference_method == "analytical"
        _assert_finite_quintet(results)

    # -- precedence: survey / Conley front doors fire before the floor ------

    def test_did_wild_survey_precedence_at_sub_floor_count(self, clustered_panel):
        did = DifferenceInDifferences(inference="wild_bootstrap", n_bootstrap=0)
        sd = SurveyDesign(weights="w", weight_type="pweight")
        with pytest.raises(NotImplementedError, match="Wild bootstrap"):
            _fit_did(did, clustered_panel, survey_design=sd)

    def test_twfe_wild_survey_precedence_at_sub_floor_count(self, clustered_panel):
        # The survey resolver rejects wild x survey for ANY design BEFORE the
        # estimator-level replicate/floor checks (survey.py), so the
        # exception TYPE is the pin.
        twfe = TwoWayFixedEffects(inference="wild_bootstrap", n_bootstrap=0)
        sd = SurveyDesign(weights="w", weight_type="pweight")
        with pytest.raises(NotImplementedError, match="Wild bootstrap"):
            _fit_twfe(twfe, clustered_panel, survey_design=sd)

    def test_did_wild_conley_precedence_at_sub_floor_count(self, conley_panel):
        did = DifferenceInDifferences(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=1000.0,
            conley_lag_cutoff=0,
            inference="wild_bootstrap",
            n_bootstrap=1,
        )
        with pytest.raises(NotImplementedError, match=r"(?i)wild.bootstrap|conley"):
            did.fit(conley_panel, outcome="y", treatment="treated", post="post", unit="unit")

    def test_twfe_wild_conley_precedence_at_sub_floor_count(self, conley_panel):
        twfe = TwoWayFixedEffects(
            vcov_type="conley",
            conley_coords=("lat", "lon"),
            conley_cutoff_km=1000.0,
            conley_lag_cutoff=0,
            inference="wild_bootstrap",
            n_bootstrap=1,
        )
        with pytest.raises(NotImplementedError, match=r"(?i)wild.bootstrap|conley"):
            _fit_twfe(twfe, conley_panel)

    # -- MPD carve-out: warn + analytical fallback, n_bootstrap-independent -

    @pytest.mark.parametrize("n", [0, 1, 999])
    def test_mpd_wild_falls_back_never_raises(self, multi_period_panel, n):
        mpd = MultiPeriodDiD(inference="wild_bootstrap", n_bootstrap=n)
        with pytest.warns(UserWarning, match=re.escape(MPD_FALLBACK_MSG)):
            results = mpd.fit(
                multi_period_panel,
                outcome="y",
                treatment="treated",
                time="time",
                post_periods=[2, 3],
            )
        assert np.isfinite(results.avg_att)

    # -- per-fit bootstrap-state reset (refit transitions) ------------------

    def test_did_refit_transition_clears_bootstrap_metadata(self, clustered_panel, ci_params):
        did = DifferenceInDifferences(
            inference="wild_bootstrap",
            cluster="cluster",
            n_bootstrap=ci_params.bootstrap(99),
            seed=42,
        )
        wild = _fit_did(did, clustered_panel)
        assert wild.inference_method == "wild_bootstrap"

        did.set_params(inference="analytical")
        analytical = _fit_did(did, clustered_panel)
        # All four metadata fields flow from the one _bootstrap_results
        # conditional - pin the full set.
        assert analytical.inference_method == "analytical"
        assert analytical.n_bootstrap is None
        assert analytical.n_clusters is None
        assert analytical.p_val_type is None
        _assert_finite_quintet(analytical)

        did.set_params(inference="wild_bootstrap")
        rewild = _fit_did(did, clustered_panel)
        assert rewild.inference_method == "wild_bootstrap"

    def test_twfe_refit_transition_clears_bootstrap_metadata(self, clustered_panel, ci_params):
        twfe = TwoWayFixedEffects(
            inference="wild_bootstrap", n_bootstrap=ci_params.bootstrap(99), seed=42
        )
        wild = _fit_twfe(twfe, clustered_panel)
        assert wild.inference_method == "wild_bootstrap"

        twfe.set_params(inference="analytical")
        analytical = _fit_twfe(twfe, clustered_panel)
        assert analytical.inference_method == "analytical"
        assert analytical.n_bootstrap is None
        assert analytical.n_clusters is None
        assert analytical.p_val_type is None

        twfe.set_params(inference="wild_bootstrap")
        rewild = _fit_twfe(twfe, clustered_panel)
        assert rewild.inference_method == "wild_bootstrap"


# ===========================================================================
# Roster guard
# ===========================================================================


class TestInferenceRoster:
    def test_inference_exposed_by_exactly_the_wcr_roster(self):
        discovered, seen = [], set()
        for name in diff_diff.__all__:
            obj = getattr(diff_diff, name)
            if not isinstance(obj, type) or id(obj) in seen:
                continue
            seen.add(id(obj))
            if issubclass(obj, BaseEstimator):
                discovered.append(obj)
        exposing = {cls for cls in discovered if "inference" in _make(cls).get_params()}
        assert exposing == {DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects}, (
            "The `inference` selector roster changed. A future estimator "
            "gaining wild cluster bootstrap must adopt the fail-closed "
            "selector contract (M-096) or land its own ledger row; one whose "
            "bootstrap IS its inference method must keep n_bootstrap as "
            "documented domain vocabulary instead (v4-design section 7)."
        )
