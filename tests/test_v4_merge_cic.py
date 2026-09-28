"""Phase 3(c): ChangesInChanges absorbs QDiD - rows M-015 / M-143.

TOLERANCE DOCTRINE. Every merge-parity gate is BIT-EXACT
(``assert_array_equal``): both surfaces call the same untouched
``_fit_distributional`` in the same process, so a needed tolerance IS the
finding. The single exception is Gate B's committed oracle, captured in a
different process (and possibly a different backend), asserted with
``assert_allclose(rtol=1e-9, atol=1e-12)``.

ORACLE PROVENANCE (Gate B). Literals captured by
``tests/_capture_v4_merge_cic_oracles.py`` on the UNMODIFIED pre-merge tree:

    commit  c2941caa2a7865c9458c6092359e75238fdbabb1  (origin/main, pre-3(c))
    command DIFF_DIFF_BACKEND=python python3 tests/_capture_v4_merge_cic_oracles.py

The oracle covers the UNCONDITIONAL arm only, and omits ``quantile_effects``.
Both are deliberate: the covariate quantile-regression path is
tie-selection-bounded with BLAS-dependent tie flips (see
``tests/test_changes_in_changes_parity.py``: COV_ATT_ATOL = 0.04,
COV_QTE_ATOL = 0.25) and CI runs ubuntu/macos/windows/arm, so tight committed
covariate literals would be platform-fragile; and ``benchmarks/data/qte_golden.json``
is git-tracked, so ``test_point_parity`` never skips and already pins
``quantile_effects["qte"]`` against the R ``qte`` 1.3.1 golden at
``atol=1e-10, rtol=0`` for cic/qdid x panel/rcs - a stronger absolute pin than
anything this file would add. Covariate coverage here comes from Gate A's
in-process bit-exact parity, which cannot be platform-dependent.
"""

import pickle
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from diff_diff import (
    ChangesInChanges,
    ChangesInChangesResults,
    QDiD,
    QDiDResults,
    practitioner_next_steps,
)
from diff_diff.changes_in_changes import _validate_all_params

from ._capture_v4_merge_cic_oracles import make_2x2

# --------------------------------------------------------------------------
# Pinned messages
# --------------------------------------------------------------------------
QDID_DEPRECATION_RE = re.escape("QDiD is deprecated and will be removed in 4.0")
METHOD_VALUE_RE = re.escape("method must be 'cic' or 'qdid', got ")
FIELD_DEPRECATION_RE = re.escape("ChangesInChangesResults.estimator is deprecated")

FIT_KW = dict(outcome="y", treatment="treated", time="post")


def _qdid(**kw):
    """Construct the dying class without its (expected) FutureWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=QDID_DEPRECATION_RE, category=FutureWarning)
        return QDiD(**kw)


def _fit_quiet(est, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(df, **FIT_KW, **kw)


@pytest.fixture(scope="module")
def df():
    return make_2x2()


@pytest.fixture(scope="module")
def cov_df():
    """Small covariate frame - the conditional QR path, exercised in-process only."""
    rng = np.random.default_rng(11)
    base = make_2x2(n_treated=25, n_control=25, seed=3)
    base = base.copy()
    base["x1"] = rng.normal(0, 1, len(base))
    return base


# ==========================================================================
# Gate A - merged mode is bit-exact against the dying class
# ==========================================================================
class TestGateAParity:
    _FIELDS = ("att", "se", "t_stat", "p_value", "q_lower", "q_upper", "sup_t_crit")

    def _assert_same(self, merged, dying):
        for f in self._FIELDS:
            assert_array_equal(
                np.asarray(getattr(merged, f), dtype=float),
                np.asarray(getattr(dying, f), dtype=float),
                err_msg=f"field {f!r} diverged between the merged and dying surfaces",
            )
        assert_array_equal(np.asarray(merged.conf_int, float), np.asarray(dying.conf_int, float))
        assert_array_equal(
            merged.quantile_effects.to_numpy(dtype=float),
            dying.quantile_effects.to_numpy(dtype=float),
        )
        assert list(merged.quantile_effects.columns) == list(dying.quantile_effects.columns)
        assert merged.cell_sizes == dying.cell_sizes
        assert merged.n_obs == dying.n_obs
        assert merged.n_bootstrap_valid == dying.n_bootstrap_valid
        assert merged.method == dying.method == "qdid"
        assert type(merged).__name__ == type(dying).__name__

    @pytest.mark.parametrize("panel", [False, True], ids=["rcs", "panel"])
    def test_point_parity(self, df, panel):
        kw = {"unit": "id"} if panel else {}
        merged = _fit_quiet(ChangesInChanges(n_bootstrap=0, panel=panel, method="qdid"), df, **kw)
        dying = _fit_quiet(_qdid(n_bootstrap=0, panel=panel), df, **kw)
        self._assert_same(merged, dying)

    def test_seeded_bootstrap_parity(self, df):
        merged = _fit_quiet(ChangesInChanges(n_bootstrap=49, seed=7, method="qdid"), df)
        dying = _fit_quiet(_qdid(n_bootstrap=49, seed=7), df)
        self._assert_same(merged, dying)

    def test_covariate_parity(self, cov_df):
        """The conditional QR path - in-process, so bit-exact is legitimate here."""
        merged = _fit_quiet(
            ChangesInChanges(n_bootstrap=0, method="qdid"), cov_df, covariates=["x1"]
        )
        dying = _fit_quiet(_qdid(n_bootstrap=0), cov_df, covariates=["x1"])
        self._assert_same(merged, dying)

    def test_cic_default_unchanged(self, df):
        """method='cic' is the default, and must match a bare ChangesInChanges."""
        a = _fit_quiet(ChangesInChanges(n_bootstrap=49, seed=7), df)
        b = _fit_quiet(ChangesInChanges(n_bootstrap=49, seed=7, method="cic"), df)
        assert_array_equal(np.asarray(a.att), np.asarray(b.att))
        assert_array_equal(np.asarray(a.se), np.asarray(b.se))
        assert a.method == b.method == "cic"

    def test_fit_time_userwarning_sets_match(self, df):
        """Scoped to fit-time UserWarnings ONLY.

        The whole warning sets differ BY CONSTRUCTION - Gate D requires QDiD()
        to emit a FutureWarning the merged surface must not - so comparing them
        wholesale would contradict this file's own deprecation gate.
        """

        def _user_warnings(make):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                make()
            return sorted(str(x.message) for x in w if issubclass(x.category, UserWarning))

        merged = _user_warnings(
            lambda: ChangesInChanges(n_bootstrap=0, method="qdid").fit(df, **FIT_KW)
        )
        dying = _user_warnings(lambda: _qdid(n_bootstrap=0).fit(df, **FIT_KW))
        assert merged == dying
        assert any("non-monotone" in m for m in merged), (
            "the DGP is expected to trip QDiD's footnote-21 non-monotonicity warning; "
            "without it this parity assertion would be vacuous"
        )


# ==========================================================================
# Gate B - committed pre-merge oracle
# ==========================================================================
_NAN = float("nan")
ORACLES = {
    "cic_panel0_nb0": {
        "att": 0.9704717894944577,
        "se": _NAN,
        "t_stat": _NAN,
        "p_value": _NAN,
        "conf_int_lower": _NAN,
        "conf_int_upper": _NAN,
        "q_lower": 0.0,
        "q_upper": 1.0,
        "sup_t_crit": _NAN,
        "n_obs": 280,
        "n_bootstrap_valid": 0,
    },
    "cic_panel1_nb0": {
        "att": 0.9704717894944577,
        "se": _NAN,
        "t_stat": _NAN,
        "p_value": _NAN,
        "conf_int_lower": _NAN,
        "conf_int_upper": _NAN,
        "q_lower": 0.0,
        "q_upper": 1.0,
        "sup_t_crit": _NAN,
        "n_obs": 280,
        "n_bootstrap_valid": 0,
    },
    "cic_panel0_nb49": {
        "att": 0.9704717894944577,
        "se": 0.19313483048560523,
        "t_stat": 5.02484086922265,
        "p_value": 5.03850140083979e-07,
        "conf_int_lower": 0.5919344775824229,
        "conf_int_upper": 1.3490091014064924,
        "q_lower": 0.0,
        "q_upper": 1.0,
        "sup_t_crit": 3.4366340739481993,
        "n_obs": 280,
        "n_bootstrap_valid": 49,
    },
    "qdid_panel0_nb0": {
        "att": 0.9562069949948577,
        "se": _NAN,
        "t_stat": _NAN,
        "p_value": _NAN,
        "conf_int_lower": _NAN,
        "conf_int_upper": _NAN,
        "q_lower": _NAN,
        "q_upper": _NAN,
        "sup_t_crit": _NAN,
        "n_obs": 280,
        "n_bootstrap_valid": 0,
    },
    "qdid_panel1_nb0": {
        "att": 0.9562069949948577,
        "se": _NAN,
        "t_stat": _NAN,
        "p_value": _NAN,
        "conf_int_lower": _NAN,
        "conf_int_upper": _NAN,
        "q_lower": _NAN,
        "q_upper": _NAN,
        "sup_t_crit": _NAN,
        "n_obs": 280,
        "n_bootstrap_valid": 0,
    },
    "qdid_panel0_nb49": {
        "att": 0.9562069949948577,
        "se": 0.1922323026627172,
        "t_stat": 4.9742264008176535,
        "p_value": 6.55087129855928e-07,
        "conf_int_lower": 0.5794386051107289,
        "conf_int_upper": 1.3329753848789867,
        "q_lower": _NAN,
        "q_upper": _NAN,
        "sup_t_crit": 3.6662009990828244,
        "n_obs": 280,
        "n_bootstrap_valid": 49,
    },
}
_CELLS = {"control_post": 80, "control_pre": 80, "treated_post": 60, "treated_pre": 60}


class TestGateBOracle:
    @pytest.mark.parametrize("key", sorted(ORACLES))
    def test_matches_pre_merge_capture(self, df, key):
        method, panel_tag, boot_tag = key.split("_")
        panel = panel_tag == "panel1"
        n_boot = int(boot_tag[2:])
        est = ChangesInChanges(
            n_bootstrap=n_boot, panel=panel, seed=7 if n_boot else None, method=method
        )
        res = _fit_quiet(est, df, **({"unit": "id"} if panel else {}))

        exp = ORACLES[key]
        lo, hi = res.conf_int
        got = {
            "att": res.att,
            "se": res.se,
            "t_stat": res.t_stat,
            "p_value": res.p_value,
            "conf_int_lower": lo,
            "conf_int_upper": hi,
            "q_lower": res.q_lower,
            "q_upper": res.q_upper,
            "sup_t_crit": res.sup_t_crit,
        }
        for field, want in got.items():
            assert_allclose(
                want,
                exp[field],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{key}.{field} drifted from the pre-merge capture",
            )
        assert res.n_obs == exp["n_obs"]
        assert res.n_bootstrap_valid == exp["n_bootstrap_valid"]
        assert res.cell_sizes == _CELLS

    def test_methods_are_not_interchangeable(self):
        """Guards the oracle itself: if cic and qdid agreed, it could not discriminate."""
        assert ORACLES["cic_panel0_nb0"]["att"] != ORACLES["qdid_panel0_nb0"]["att"]


# ==========================================================================
# Gate C - the new parameter's contract
# ==========================================================================
class TestGateCValidation:
    @pytest.mark.parametrize("bad", ["bogus", None, "CiC", "QDiD", "", 1, 0.5])
    def test_rejects_at_construction(self, bad):
        """Construction-time, not just fit-time: the regression test for
        ``method`` being in __init__'s hand-built validation dict. With only
        _validate_all_params' ``.get`` default this would construct silently."""
        with pytest.raises(ValueError, match=METHOD_VALUE_RE):
            ChangesInChanges(method=bad)

    @pytest.mark.parametrize(
        "arr",
        [np.array(["cic"]), np.array(["cic", "qdid"]), np.array([["cic"]])],
        ids=["one-element", "multi-element", "2d"],
    )
    def test_rejects_ndarray_lookalikes(self, arr):
        """A bare membership test would accept the 1-element case.

        ``np.array(["cic"]) in ("cic", "qdid")`` compares elementwise and
        bool()s a 1-element result to True, so the ARRAY would be stored as
        ``self.method`` - an unhashable tag that breaks the _ESTIMATOR_TITLES
        lookup in summary() and serializes as an array rather than the
        documented string. The multi-element cases must raise the same clean
        message, not numpy's ambiguous-truth error.
        """
        with pytest.raises(ValueError, match=METHOD_VALUE_RE):
            ChangesInChanges(method=arr)

    @pytest.mark.parametrize("good", ["cic", "qdid"])
    def test_accepts_vocabulary(self, good):
        assert ChangesInChanges(method=good).method == good

    def test_keyword_only(self):
        import inspect

        kind = inspect.signature(ChangesInChanges).parameters["method"].kind
        assert kind is inspect.Parameter.KEYWORD_ONLY
        with pytest.raises(TypeError):
            ChangesInChanges(None, 200, 0.05, False, None, "qdid")  # 6th positional

    def test_set_params_success_flips_behaviour(self, df):
        est = ChangesInChanges(n_bootstrap=0)
        est.set_params(method="qdid")
        assert est.method == "qdid"
        assert_array_equal(
            np.asarray(_fit_quiet(est, df).att),
            np.asarray(ORACLES["qdid_panel0_nb0"]["att"]),
        )

    def test_set_params_failure_is_transactional(self):
        est = ChangesInChanges(n_bootstrap=0, method="cic")
        with pytest.raises(ValueError, match=METHOD_VALUE_RE):
            est.set_params(method="bogus")
        assert est.method == "cic"

    def test_get_params_key_sets(self):
        merged = set(ChangesInChanges().get_params())
        assert merged == {"quantiles", "n_bootstrap", "alpha", "panel", "seed", "method"}
        # The dying class's five-param contract is FROZEN through removal.
        assert set(_qdid().get_params()) == merged - {"method"}

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_reinstantiation_round_trip(self, method):
        est = ChangesInChanges(n_bootstrap=3, alpha=0.1, method=method)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = ChangesInChanges(**est.get_params())
        assert clone.get_params() == est.get_params()

    def test_dying_class_round_trip_needs_targeted_ignore(self):
        """QDiD construction MUST warn, so the error filter needs the per-class
        ignore (the tests/test_base_estimator.py:188-193 idiom)."""
        est = _qdid(n_bootstrap=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message=QDID_DEPRECATION_RE, category=FutureWarning)
            clone = QDiD(**est.get_params())
        assert clone.get_params() == est.get_params()


# ==========================================================================
# Gate D - deprecation choreography
# ==========================================================================
class TestGateDDeprecation:
    def test_construction_warns_once(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QDiD(n_bootstrap=0)
        fw = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(fw) == 1
        assert re.search(QDID_DEPRECATION_RE, str(fw[0].message))
        assert "ChangesInChanges(method='qdid')" in str(fw[0].message)

    def test_warns_but_still_works(self, df):
        res = _fit_quiet(_qdid(n_bootstrap=0), df)
        assert np.isfinite(res.att)

    def test_stacklevel_attributes_to_caller(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QDiD(n_bootstrap=0)
        assert w[0].filename == __file__, "stacklevel=2 must blame the user frame"

    def test_merged_surface_is_silent(self):
        """The METHOD is not deprecated - only the class spelling."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            ChangesInChanges(n_bootstrap=0, method="qdid")
            ChangesInChanges(n_bootstrap=0, method="cic")

    def test_set_params_re_emits(self):
        """Documented side effect of BaseEstimator's transactional probe re-init."""
        est = _qdid(n_bootstrap=0)
        with pytest.warns(FutureWarning, match=QDID_DEPRECATION_RE):
            est.set_params(n_bootstrap=5)

    def test_results_alias_identity(self):
        assert QDiDResults is ChangesInChangesResults


# ==========================================================================
# Gate E - the renamed results field (M-143)
# ==========================================================================
class TestGateEFieldShim:
    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_method_field(self, df, method):
        assert _fit_quiet(ChangesInChanges(n_bootstrap=0, method=method), df).method == method

    def test_old_name_reads_and_warns(self, df):
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        with pytest.warns(FutureWarning, match=FIELD_DEPRECATION_RE):
            assert res.estimator == "cic"

    def test_old_name_is_not_a_constructor_keyword(self, df):
        """Pins the DOCUMENTED scope of M-143's deprecation window.

        The window covers the READ path (the property below) and pickle
        migration - not construction. `ChangesInChangesResults(estimator=...)`
        raises immediately rather than warning until 4.0, matching M-094
        (`treatment_col` -> `takeup`) and M-114 (`groups` -> `units`), neither
        of which kept a deprecated constructor keyword. Asserted so the
        REGISTRY note is an enforced contract rather than prose: a later
        "helpful" constructor shim would have to change this test deliberately.
        """
        good = _fit_quiet(ChangesInChanges(n_bootstrap=0), df).to_dict()
        with pytest.raises(TypeError, match="estimator"):
            ChangesInChangesResults(  # type: ignore[call-arg]
                att=good["att"],
                se=good["se"],
                t_stat=good["t_stat"],
                p_value=good["p_value"],
                conf_int=(good["conf_int_lower"], good["conf_int_upper"]),
                quantile_effects=None,
                q_lower=good["q_lower"],
                q_upper=good["q_upper"],
                sup_t_crit=good["sup_t_crit"],
                n_obs=good["n_obs"],
                cell_sizes=good["cell_sizes"],
                n_bootstrap=good["n_bootstrap"],
                n_bootstrap_valid=good["n_bootstrap_valid"],
                panel=good["panel"],
                estimator="cic",
                quantiles=None,
            )

    def test_old_name_is_read_only(self, df):
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        with pytest.raises(AttributeError):
            res.estimator = "qdid"

    def test_to_dict_carries_both_keys(self, df):
        """3.9 dual-key window (the M-094 twin). Both read the SAME attribute."""
        d = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df).to_dict()
        assert d["method"] == "qdid"
        assert d["estimator"] == d["method"]

    def test_repr_label_flipped(self, df):
        r = repr(_fit_quiet(ChangesInChanges(n_bootstrap=0), df))
        assert "method='cic'" in r
        assert "estimator=" not in r

    def test_live_pickle_round_trip(self, df):
        """__setstate__ runs for CURRENT objects too - a defective one breaks them."""
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df)
        clone = pickle.loads(pickle.dumps(res))
        assert clone.method == "qdid"
        assert_array_equal(np.asarray(clone.att), np.asarray(res.att))

    def test_legacy_state_migration(self, df):
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df)
        legacy = dict(res.__dict__)
        legacy["estimator"] = legacy.pop("method")
        revived = ChangesInChangesResults.__new__(ChangesInChangesResults)
        revived.__setstate__(legacy)
        assert revived.method == "qdid"
        with pytest.warns(FutureWarning, match=FIELD_DEPRECATION_RE):
            assert revived.estimator == "qdid"

    def test_summary_interior_range_branch(self, df):
        """Inspects summary() TEXT, not the field.

        ``q_lower``/``q_upper`` are computed by the engine either way, so
        asserting on their values would pass even with a broken ``self.method``
        read. Only the rendered line proves the branch saw the right value.
        """
        cic = _fit_quiet(ChangesInChanges(n_bootstrap=0), df).summary()
        qdid = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df).summary()
        assert "interior quantile range" in cic
        assert "interior quantile range" not in qdid
        assert "Changes-in-Changes" in cic
        assert "Quantile Difference-in-Differences" in qdid

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_canonical_surface_emits_no_deprecation(self, df, method):
        """The ONLY thing that catches an unmigrated internal ``self.estimator``.

        The shim returns the right value, so behaviour looks correct and every
        value assertion still passes - while every user gets a FutureWarning on
        each summary()/repr() call, which the blanket pyproject filter hides.
        """
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method=method), df)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            res.summary()
            res.to_dict()
            repr(res)
            res.to_dataframe()


# ==========================================================================
# Gate F - dispatch guards inside the shared engine
# ==========================================================================
class TestGateFEngineGuards:
    def test_qdid_fit_survives_revalidation(self, df):
        """QDiD's get_params() has no 'method' key and _fit_distributional
        re-validates through _validate_all_params on every fit."""
        assert np.isfinite(_fit_quiet(_qdid(n_bootstrap=0), df).att)

    def test_validate_all_params_tolerates_missing_method(self):
        """The behavioural pin for ``params.get("method", "cic")``.

        A ``params["method"]`` read would KeyError here - which is exactly what
        every QDiD fit would hit. Asserted directly rather than by mutating
        source, so nothing dirties the worktree.
        """
        _validate_all_params(
            {"quantiles": None, "n_bootstrap": 0, "alpha": 0.05, "panel": False, "seed": None}
        )

    def test_direct_attribute_mutation_revalidated_at_fit(self, df):
        """The one route that bypasses BOTH __init__ and the set_params probe."""
        est = ChangesInChanges(n_bootstrap=0)
        est.method = "bogus"
        with pytest.raises(ValueError, match=METHOD_VALUE_RE):
            est.fit(df, **FIT_KW)

    @pytest.mark.parametrize(
        "factory,expected",
        [
            (lambda: ChangesInChanges(n_bootstrap=0, method="qdid"), "ChangesInChanges"),
            (lambda: _qdid(n_bootstrap=0), "QDiD"),
        ],
        ids=["merged", "dying"],
    )
    def test_errors_name_the_class_the_user_built(self, cov_df, factory, expected):
        """Derived from type(est).__name__, so a method='qdid' fit never names
        a class the caller never constructed."""
        bad = cov_df.copy()
        bad["x_str"] = "a"
        with pytest.raises(ValueError) as exc:
            factory().fit(bad, **FIT_KW, covariates=["x_str"])
        assert expected in str(exc.value)


# ==========================================================================
# Gate G - consumers
# ==========================================================================
def _steps(res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return practitioner_next_steps(res)["next_steps"]


class TestGateGConsumers:
    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_no_snippet_teaches_the_dying_class(self, df, method):
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method=method), df)
        blob = "\n".join(s.get("code", "") for s in _steps(res))
        assert not re.search(r"\bQDiD\s*\(", blob), "emitted a deprecated QDiD(...) constructor"
        assert not re.search(r"import[^\n]*\bQDiD\b", blob), "emitted a deprecated QDiD import"

    def test_qdid_snippet_uses_the_merged_surface(self, df):
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        blob = "\n".join(s.get("code", "") for s in _steps(res))
        assert "ChangesInChanges(method='qdid'" in blob

    def test_cic_snippets_carry_no_method_argument(self, df):
        """method= must not leak into snippets for a CiC refit."""
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        for s in _steps(res):
            code = s.get("code", "")
            if "cic_results =" in code or "results_nocov =" in code:
                assert "method=" not in code

    def test_step8_snippet_executes(self, df):
        """EXEC, not compile: a missing import is a run-time NameError that
        compile() cannot see - exactly how the Step-8 two-name import
        (ChangesInChanges + DifferenceInDifferences) would slip through.

        The fixture is deliberately unconditional and panel=False: the helper
        emits a placeholder ``unit='unit_id'`` for panel fits, and
        ``covariates="same"`` would copy a covariate list into this block and
        drag the slow QR path into the test.
        """
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        blocks = [s["code"] for s in _steps(res) if "qdid_results" in s.get("code", "")]
        assert blocks, "the CiC branch must emit the Step-8 QDiD comparison snippet"
        ns = {"data": df, "results": res}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(blocks[0], ns)  # noqa: S102 - executing our own emitted guidance is the point
        assert "qdid_results" in ns and "did_results" in ns

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_guidance_emits_no_deprecation_warning(self, df, method):
        """Gate E's doctrine applied to the consumer surface: an old-name-first
        or unmigrated read would keep the output identical while warning users,
        and the blanket pyproject filter would hide it."""
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method=method), df)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            practitioner_next_steps(res)

    def test_duck_typed_legacy_results_still_route(self):
        """Exercises the __dict__ fallback in _distributional_kind.

        Cannot use the _mock_cic idiom: ``estimator`` is now a setter-less
        property, so ``setattr`` on a real ChangesInChangesResults raises.
        Dispatch keys on type(...).__name__, so a separate class sharing the
        name is the supported way to build a pre-rename duck type.
        """

        class ChangesInChangesResults:  # noqa: F811 - deliberate name shadow
            def __init__(self):
                self.estimator = "qdid"
                self.att = 0.5
                self.covariates = None
                self.n_bootstrap = 200
                self.n_bootstrap_valid = 200
                self.panel = False

        out = _steps(ChangesInChangesResults())
        assert any("QDiD" in s.get("label", "") + s.get("why", "") for s in out), (
            "a legacy duck-typed result carrying only `estimator` must still reach the "
            "QDiD branch - otherwise the __dict__ fallback is dead code"
        )
