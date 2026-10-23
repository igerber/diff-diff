"""Merge gates: ChangesInChanges absorbed QDiD (phase 3(c), rows M-015 / M-143);
the QDiD class and the ``QDiDResults`` alias (M-061) were removed at the 4.0 cut
(PR 4a). ``ChangesInChanges(method="qdid")`` is the one surviving surface.

TOLERANCE DOCTRINE. Through 3.x the merge-parity gates compared the two
surfaces BIT-EXACTLY (``assert_array_equal``) because both called the same
``_fit_distributional`` in one process. With the class gone, the absolute pin
is Gate B's committed oracle, captured in a different process (and possibly a
different backend), asserted with ``assert_allclose(rtol=1e-9, atol=1e-12)``.

ORACLE PROVENANCE (Gate B). Literals captured by
``tests/_capture_v4_merge_cic_oracles.py`` on the UNMODIFIED pre-merge tree:

    commit  c2941caa2a7865c9458c6092359e75238fdbabb1  (origin/main, pre-3(c))
    command DIFF_DIFF_BACKEND=python python3 tests/_capture_v4_merge_cic_oracles.py

That script drives the removed class and is HISTORICAL (runnable only at that
commit); it is kept as the oracle's provenance record.

The oracle covers the UNCONDITIONAL arm only, and omits ``quantile_effects``.
Both are deliberate: the covariate quantile-regression path is
tie-selection-bounded with BLAS-dependent tie flips (see
``tests/test_changes_in_changes_parity.py``: COV_ATT_ATOL = 0.04,
COV_QTE_ATOL = 0.25) and CI runs ubuntu/macos/windows/arm, so tight committed
covariate literals would be platform-fragile; and ``benchmarks/data/qte_golden.json``
is git-tracked, so ``tests/test_changes_in_changes_parity.py::test_point_parity``
never skips and already pins ``quantile_effects["qte"]`` against the R ``qte``
1.3.1 golden at ``atol=1e-10, rtol=0`` for cic/qdid x panel/rcs - a stronger
absolute pin than anything this file would add. Covariate coverage lives in
that parity suite's covariate lanes (platform-tolerant by design).
"""

import pickle
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from diff_diff import (
    ChangesInChanges,
    ChangesInChangesResults,
    practitioner_next_steps,
)

# --------------------------------------------------------------------------
# Pinned messages
# --------------------------------------------------------------------------
METHOD_VALUE_RE = re.escape("method must be 'cic' or 'qdid', got ")

FIT_KW = dict(outcome="y", treatment="treated", time="post")


def make_2x2(n_treated=60, n_control=80, seed=0, effect=1.0):
    """Full-overlap continuous 2x2 panel (long format, one row per unit-period).

    The oracle DGP: a copy of the helper the HISTORICAL capture script
    ``tests/_capture_v4_merge_cic_oracles.py`` carries (that script imports the
    removed QDiD class and no longer runs on this tree). The ``id`` column
    repeats across periods, so the same frame serves both ``panel=False``
    (pooled row resample) and ``panel=True`` (unit-block resample, ``unit="id"``).
    """
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    treat = np.repeat([1, 0], [n_treated, n_control])
    u = rng.normal(0, 1, n)
    y_pre = u + rng.normal(0, 0.3, n)
    y_post = u + 0.5 + rng.normal(0, 0.3, n) + treat * effect
    return pd.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "post": np.repeat([0, 1], n),
            "treated": np.tile(treat, 2),
            "y": np.concatenate([y_pre, y_post]),
        }
    )


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
class TestGateAMergedSurface:
    """Merged-surface pins that the 3.x old-vs-new parity lanes carried."""

    def test_cic_default_unchanged(self, df):
        """method='cic' is the default, and must match a bare ChangesInChanges."""
        a = _fit_quiet(ChangesInChanges(n_bootstrap=49, seed=7), df)
        b = _fit_quiet(ChangesInChanges(n_bootstrap=49, seed=7, method="cic"), df)
        assert_array_equal(np.asarray(a.att), np.asarray(b.att))
        assert_array_equal(np.asarray(a.se), np.asarray(b.se))
        assert a.method == b.method == "cic"

    def test_qdid_fit_emits_the_footnote_21_warning(self, df):
        """The Athey-Imbens footnote-21 non-monotonicity UserWarning fires on this
        DGP through the merged surface (through 3.x this was asserted as a
        warning-set parity against the removed class; it is now a direct pin).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChangesInChanges(n_bootstrap=0, method="qdid").fit(df, **FIT_KW)
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any("non-monotone" in m for m in msgs), msgs
        assert not [x for x in w if issubclass(x.category, FutureWarning)]


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
        ``method`` being in __init__'s hand-built validation dict (__init__ does
        not forward get_params(), so a missing key would construct silently)."""
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

    def test_get_params_key_set(self):
        merged = set(ChangesInChanges().get_params())
        assert merged == {"quantiles", "n_bootstrap", "alpha", "panel", "seed", "method"}

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_reinstantiation_round_trip(self, method):
        est = ChangesInChanges(n_bootstrap=3, alpha=0.1, method=method)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clone = ChangesInChanges(**est.get_params())
        assert clone.get_params() == est.get_params()


# ==========================================================================
# Gate D - removal pins (M-015 / M-061; 4.0 cut, PR 4a)
# ==========================================================================
class TestGateDRemoval:
    """The QDiD class and the QDiDResults alias are gone; the merged surface
    constructs silently with either method."""

    @pytest.mark.parametrize("name", ["QDiD", "QDiDResults"])
    def test_name_is_absent(self, name):
        import diff_diff

        assert not hasattr(diff_diff, name)
        assert name not in diff_diff.__all__
        assert name not in dir(diff_diff)
        with pytest.raises(ImportError):
            exec(f"from diff_diff import {name}")

    def test_results_module_has_no_alias(self):
        import diff_diff.changes_in_changes_results as mod

        assert not hasattr(mod, "QDiDResults")
        assert mod.ChangesInChangesResults is ChangesInChangesResults

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_merged_surface_is_silent(self, method):
        """The METHOD was never deprecated - only the removed class spelling."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            est = ChangesInChanges(n_bootstrap=0, method=method)
            est.set_params(n_bootstrap=5)
        assert est.method == method


# ==========================================================================
# Gate E - the renamed results field (M-143, alias REMOVED at the 4.0 cut)
# ==========================================================================
class TestGateEFieldShim:
    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_method_field(self, df, method):
        assert _fit_quiet(ChangesInChanges(n_bootstrap=0, method=method), df).method == method

    def test_old_name_removed(self, df):
        """M-143 removal pin: the 3.9 read-only ``estimator`` alias is gone."""
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0), df)
        assert "estimator" not in ChangesInChangesResults.__dataclass_fields__
        with pytest.raises(AttributeError):
            res.estimator

    def test_old_name_is_not_a_constructor_keyword(self, df):
        """`ChangesInChangesResults(estimator=...)` raises - it never was a
        constructor keyword (M-143's 3.9 window covered the READ path and
        pickle migration only, matching M-094 and M-114), and at 4.0 the
        read alias is gone too. Asserted so a later "helpful" constructor
        shim would have to change this test deliberately.
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

    def test_to_dict_carries_method_only(self, df):
        """The 3.9 dual-key window (the M-094 twin) closed at 4.0: canonical key only."""
        d = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df).to_dict()
        assert d["method"] == "qdid"
        assert "estimator" not in d

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
        """``__setstate__`` keeps mapping the pre-M-143 ``estimator`` key (kept at 4.0)."""
        res = _fit_quiet(ChangesInChanges(n_bootstrap=0, method="qdid"), df)
        legacy = dict(res.__dict__)
        legacy["estimator"] = legacy.pop("method")
        revived = ChangesInChangesResults.__new__(ChangesInChangesResults)
        revived.__setstate__(legacy)
        assert revived.method == "qdid"
        assert "estimator" not in revived.__dict__
        with pytest.raises(AttributeError):
            revived.estimator

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
        """The canonical surface renders without any deprecation noise.

        Through 3.13 this was the ONLY thing that caught an unmigrated
        internal ``self.estimator`` read (the shim returned the right value
        while warning on every summary()/repr() call); at 4.0 such a read is
        an AttributeError, and this stays as the family-wide silence pin.
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
    def test_direct_attribute_mutation_revalidated_at_fit(self, df):
        """The one route that bypasses BOTH __init__ and the set_params probe."""
        est = ChangesInChanges(n_bootstrap=0)
        est.method = "bogus"
        with pytest.raises(ValueError, match=METHOD_VALUE_RE):
            est.fit(df, **FIT_KW)

    @pytest.mark.parametrize("method", ["cic", "qdid"])
    def test_errors_name_the_class_the_user_built(self, cov_df, method):
        """Derived from type(est).__name__, so a method='qdid' fit names
        ChangesInChanges, never the removed QDiD class."""
        bad = cov_df.copy()
        bad["x_str"] = "a"
        with pytest.raises(ValueError) as exc:
            ChangesInChanges(n_bootstrap=0, method=method).fit(bad, **FIT_KW, covariates=["x_str"])
        assert "ChangesInChanges" in str(exc.value)
        assert "QDiD(" not in str(exc.value)


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
