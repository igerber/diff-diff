"""Tests for `vcov_type` threading through DifferenceInDifferences.

Covers the Phase 1a commitments in the approved plan:
- `robust=True` aliases `vcov_type="hc1"`.
- `robust=False` aliases `vcov_type="classical"` (backward compat for the 7
  existing test files that pass `robust=False`).
- Explicit `vcov_type` values validate against {classical, hc1, hc2, hc2_bm}.
- `robust=False` + explicit non-classical `vcov_type` raises at `__init__`.
- `MultiPeriodDiD` and `TwoWayFixedEffects` inherit through `get_params`.
- HC2+BM produces a wider CI than HC1 on the same data (property of the DOF
  correction).
- `get_params` / `set_params` round-trip preserves `vcov_type`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import SunAbraham, SurveyDesign
from diff_diff.estimators import DifferenceInDifferences, MultiPeriodDiD
from diff_diff.linalg import _absorbed_fe_vcov_scale
from diff_diff.twfe import TwoWayFixedEffects


def _make_did_panel(n_units: int = 30, seed: int = 20260420) -> pd.DataFrame:
    """Deterministic two-period DiD panel with a treatment effect of 1.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        treated = int(i >= n_units // 2)
        for t in (0, 1):
            y = rng.normal(0.0, 1.0) + 0.5 * treated + 1.0 * treated * t
            rows.append({"unit": i, "time": t, "treated": treated, "y": y})
    return pd.DataFrame(rows)


# =============================================================================
# robust <-> vcov_type alias resolution
# =============================================================================


class TestRobustAliasing:
    def test_robust_true_aliases_hc1(self):
        est = DifferenceInDifferences(robust=True)
        assert est.vcov_type == "hc1"

    def test_robust_false_aliases_classical(self):
        est = DifferenceInDifferences(robust=False)
        assert est.vcov_type == "classical"

    def test_explicit_vcov_type_wins_when_robust_default(self):
        """When `robust` is the default (True) and vcov_type is explicit, vcov_type wins."""
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        assert est.vcov_type == "hc2_bm"

    def test_robust_false_and_classical_coexist(self):
        """robust=False + vcov_type='classical' is redundant but not an error."""
        est = DifferenceInDifferences(robust=False, vcov_type="classical")
        assert est.vcov_type == "classical"
        assert est.robust is False

    def test_robust_false_explicit_hc1_raises(self):
        """robust=False + vcov_type='hc1' is inconsistent -> ValueError."""
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            DifferenceInDifferences(robust=False, vcov_type="hc1")

    def test_robust_false_explicit_hc2_raises(self):
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            DifferenceInDifferences(robust=False, vcov_type="hc2")

    def test_unknown_vcov_type_raises(self):
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            DifferenceInDifferences(vcov_type="hc3")

    def test_hc0_not_accepted(self):
        for bad in ("hc0", "HC1", "CR2", "cr1", "hc2+bm"):
            with pytest.raises(ValueError, match="vcov_type must be one of"):
                DifferenceInDifferences(vcov_type=bad)


# =============================================================================
# get_params / set_params round-trip
# =============================================================================


class TestParamsRoundTrip:
    def test_get_params_includes_vcov_type(self):
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        params = est.get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc2_bm"

    def test_get_params_default_vcov_type(self):
        """Default construction returns the raw alias-derived None from
        get_params() so clones preserve the implicit remap behavior.
        The resolved value (hc1) is on self.vcov_type.
        """
        est = DifferenceInDifferences()
        assert est.get_params()["vcov_type"] is None
        assert est.vcov_type == "hc1"
        # Explicit construction round-trips the exact value.
        assert DifferenceInDifferences(vcov_type="hc1").get_params()["vcov_type"] == "hc1"

    def test_set_params_preserves_vcov_type(self):
        est = DifferenceInDifferences()
        est.set_params(vcov_type="hc2")
        assert est.vcov_type == "hc2"

    def test_set_params_rejects_conflict_robust_false_hc2(self):
        """set_params must re-validate robust/vcov_type consistency."""
        est = DifferenceInDifferences()
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            est.set_params(robust=False, vcov_type="hc2")

    def test_set_params_robust_only_rederives_vcov_type(self):
        """Setting robust= alone after init re-derives vcov_type from the alias.

        When only ``robust`` is passed to ``set_params``, the new ``robust`` value
        overrides the previously-set ``vcov_type`` via the alias rule:
        ``robust=False`` -> ``"classical"``. This keeps the pair internally
        consistent rather than leaving the estimator with ``robust=False,
        vcov_type="hc2_bm"`` (a state that ``__init__`` forbids).
        """
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        est.set_params(robust=False)
        assert est.vcov_type == "classical"

    def test_set_params_invalid_vcov_type_rejected(self):
        est = DifferenceInDifferences()
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            est.set_params(vcov_type="hc3")

    def test_set_params_robust_true_then_back_to_hc1(self):
        """robust=True after construction restores hc1 when no explicit vcov_type."""
        est = DifferenceInDifferences(robust=False)
        assert est.vcov_type == "classical"
        est.set_params(robust=True)
        assert est.vcov_type == "hc1"

    def test_set_params_multi_period_inherits(self):
        est = MultiPeriodDiD(vcov_type="hc2_bm")
        params = est.get_params()
        assert params["vcov_type"] == "hc2_bm"

    def test_set_params_twfe_inherits(self):
        est = TwoWayFixedEffects(vcov_type="hc2")
        assert est.vcov_type == "hc2"

    def test_set_params_conflict_leaves_estimator_unchanged(self):
        """A rejected set_params() call must leave the estimator unchanged.

        Previously `set_params` mutated attributes via `setattr` BEFORE
        re-validating the robust/vcov_type pair. A failing call left the
        estimator in exactly the half-configured state the alias/conflict
        check is supposed to prevent, which defeats callers that catch
        `ValueError` and try to keep using the object. This test pins the
        atomic behavior: on failure, no attribute moves.
        """
        est = DifferenceInDifferences(
            robust=True,
            vcov_type="hc1",
            cluster=None,
            alpha=0.05,
        )
        before_robust = est.robust
        before_vcov = est.vcov_type
        before_cluster = est.cluster
        before_alpha = est.alpha
        with pytest.raises(ValueError, match="robust=False conflicts with"):
            # Conflict: robust=False + vcov_type="hc2". The side-effect here is
            # the regression target — set_params must NOT apply cluster=/alpha=
            # (or anything else in the batch) when validation fails.
            est.set_params(robust=False, vcov_type="hc2", cluster="unit", alpha=0.1)
        assert est.robust == before_robust
        assert est.vcov_type == before_vcov
        assert est.cluster == before_cluster
        assert est.alpha == before_alpha

    def test_set_params_unknown_key_leaves_estimator_unchanged(self):
        """Unknown-key rejections must be atomic too, not partial.

        Regression guard for the first-pass validator: when one key in the
        params batch is unknown, no keys in the batch should have been
        applied by the time we raise.
        """
        est = DifferenceInDifferences(vcov_type="hc1", alpha=0.05)
        with pytest.raises(ValueError, match="Unknown parameter"):
            # vcov_type is valid but `not_a_real_param` is not — reject the
            # whole batch and leave vcov_type at "hc1".
            est.set_params(vcov_type="hc2_bm", not_a_real_param=1)
        assert est.vcov_type == "hc1"
        assert est.alpha == 0.05


# =============================================================================
# End-to-end fit() behavior
# =============================================================================


class TestFitBehavior:
    def test_robust_false_with_cluster_preserves_cr1(self):
        """Legacy alias backward-compat: `robust=False` + `cluster=...` must
        still produce CR1 cluster-robust SEs, not raise on `classical + cluster`.

        Previously (pre-vcov_type), the cluster structure silently overrode
        the non-robust flag. The vcov_type threading made `robust=False`
        eagerly resolve to `"classical"`, which the linalg validator rejects
        alongside `cluster_ids`. Fix: track `_vcov_type_explicit` and remap
        implicit `"classical"` + cluster to `"hc1"` (CR1) at fit time with a
        UserWarning.
        """
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences(robust=False, cluster="unit")
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        # The effective vcov_type in the result reflects the remap.
        assert res.vcov_type == "hc1"
        # The stored value on the estimator is unchanged (it tracks what the
        # user configured).
        assert est.vcov_type == "classical"
        assert "CR1 cluster-robust at unit" in res.summary()

    def test_explicit_classical_with_cluster_still_raises(self):
        """When the user explicitly asks for `vcov_type="classical"` with a
        cluster, the validator should still reject. The remap only applies
        when vcov_type was implicit (alias-derived).
        """
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences(vcov_type="classical", cluster="unit")
        assert est._vcov_type_explicit is True
        with pytest.raises(ValueError, match="classical SEs are one-way only"):
            est.fit(data, outcome="y", treatment="treated", time="time")

    def test_twfe_robust_false_preserves_cr1_via_autocluster(self):
        """TWFE auto-clusters at unit; `robust=False` on TWFE historically
        produced CR1 at unit. Same implicit-alias remap must apply.
        """
        data = _make_did_panel(n_units=20)
        est = TwoWayFixedEffects(robust=False)
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res = est.fit(data, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.att) and np.isfinite(res.se)
        assert res.vcov_type == "hc1"
        assert "CR1 cluster-robust at unit" in res.summary()

    def test_multi_period_robust_false_with_cluster_preserves_cr1(self):
        """MultiPeriodDiD(robust=False, cluster=...) must also preserve CR1."""
        rng = np.random.default_rng(20260420)
        n_units, n_time = 30, 4
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(n_time):
                y = rng.normal(0.0, 1.0) + 0.3 * treated + 0.5 * treated * (t >= 2)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        est = MultiPeriodDiD(robust=False, cluster="unit")
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res = est.fit(data, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.avg_att) and np.isfinite(res.avg_se)
        assert res.vcov_type == "hc1"

    def test_linear_regression_robust_false_with_cluster_preserves_cr1(self):
        """Direct LinearRegression API: constructor-time cluster remap
        produces CR1 inference WITHOUT mutating self.vcov_type.

        Configured state (``self.vcov_type``) is preserved as
        ``"classical"``; the fit-time effective family is recorded on
        the fitted attribute ``self._fit_vcov_type_``. This makes
        repeated fits idempotent on configuration.
        """
        from diff_diff.linalg import LinearRegression

        rng = np.random.default_rng(1)
        n = 100
        X = rng.normal(size=(n, 1))
        y = 1.0 + 0.5 * X[:, 0] + rng.normal(scale=0.3, size=n)
        cluster_ids = np.repeat(np.arange(10), 10)

        with pytest.warns(UserWarning, match="historically produced CR1"):
            reg = LinearRegression(robust=False, cluster_ids=cluster_ids).fit(X, y)
        # Configured state unchanged; effective state on fitted attr.
        assert reg.vcov_type == "classical"
        assert reg._fit_vcov_type_ == "hc1"
        assert reg.coefficients_ is not None
        inf = reg.get_inference(1)
        assert np.isfinite(inf.se) and inf.se > 0

    def test_linear_regression_robust_false_fit_time_cluster_preserves_cr1(self):
        """LinearRegression(robust=False).fit(cluster_ids=...) override path.

        Same invariant as the constructor-time test: configured state is
        preserved; effective vcov_type lands on ``_fit_vcov_type_``.
        """
        from diff_diff.linalg import LinearRegression

        rng = np.random.default_rng(2)
        n = 100
        X = rng.normal(size=(n, 1))
        y = 1.0 + 0.5 * X[:, 0] + rng.normal(scale=0.3, size=n)
        cluster_ids = np.repeat(np.arange(10), 10)

        reg = LinearRegression(robust=False)
        assert reg.vcov_type == "classical"  # constructor-resolved alias

        with pytest.warns(UserWarning, match="historically produced CR1"):
            reg.fit(X, y, cluster_ids=cluster_ids)
        # Configured state unchanged; effective state on fitted attr.
        assert reg.vcov_type == "classical"
        assert reg._fit_vcov_type_ == "hc1"
        assert reg.coefficients_ is not None
        inf = reg.get_inference(1)
        assert np.isfinite(inf.se) and inf.se > 0

    def test_linear_regression_repeat_fit_clustered_then_unclustered(self):
        """Repeat-fit idempotence regression guard.

        Fit once with cluster_ids (which triggers the legacy remap), then
        fit again WITHOUT cluster_ids. The second fit must use classical
        SEs — not silently inherit the remapped hc1 from the first fit.
        This pins the "fit() does not mutate configured state" invariant.
        """
        from diff_diff.linalg import LinearRegression

        rng = np.random.default_rng(3)
        n = 100
        X = rng.normal(size=(n, 1))
        y = 1.0 + 0.5 * X[:, 0] + rng.normal(scale=0.3, size=n)
        cluster_ids = np.repeat(np.arange(10), 10)

        reg = LinearRegression(robust=False)
        with pytest.warns(UserWarning, match="historically produced CR1"):
            reg.fit(X, y, cluster_ids=cluster_ids)
        assert reg._fit_vcov_type_ == "hc1"
        assert reg.vcov_type == "classical"  # configured unchanged

        # Second fit WITHOUT cluster: must use classical (not hc1 from prior fit)
        reg.fit(X, y)
        assert reg._fit_vcov_type_ == "classical"
        assert reg.vcov_type == "classical"

    def test_robust_false_without_cluster_stays_classical(self):
        """No remap when no cluster is present: `robust=False` without cluster
        should still produce classical non-robust SEs."""
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences(robust=False)
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert res.vcov_type == "classical"
        assert "Classical OLS" in res.summary()

    def test_get_params_round_trip_preserves_implicit_classical(self):
        """Clone round-trip regression guard.

        ``DifferenceInDifferences(robust=False, cluster="unit")`` originally
        has ``_vcov_type_explicit=False`` and remaps to CR1 at fit time.
        A clone via ``__init__(**orig.get_params())`` must ALSO be implicit
        and remap the same way. If ``get_params`` serialized the
        alias-resolved ``"classical"`` instead of the raw ``None``, the
        clone would mark it explicit and raise on cluster fit. This pins
        that sklearn-style clone preserves backward-compat behavior.
        """
        orig = DifferenceInDifferences(robust=False, cluster="unit")
        assert orig._vcov_type_explicit is False
        params = orig.get_params()
        # get_params must return None for implicit alias path.
        assert params["vcov_type"] is None
        clone = DifferenceInDifferences(**params)
        assert clone._vcov_type_explicit is False
        # Fit both: should behave identically (CR1 via remap, with warning).
        data = _make_did_panel(n_units=20)
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res_orig = orig.fit(data, outcome="y", treatment="treated", time="time")
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res_clone = clone.fit(data, outcome="y", treatment="treated", time="time")
        assert res_orig.vcov_type == res_clone.vcov_type == "hc1"
        # Point estimate and SE identical.
        assert res_orig.att == pytest.approx(res_clone.att, abs=1e-12)
        assert res_orig.se == pytest.approx(res_clone.se, abs=1e-12)

    def test_get_params_round_trip_preserves_explicit_vcov_type(self):
        """Round-trip for explicitly-set vcov_type: raw arg round-trips."""
        orig = DifferenceInDifferences(vcov_type="hc2_bm")
        assert orig._vcov_type_explicit is True
        params = orig.get_params()
        assert params["vcov_type"] == "hc2_bm"
        clone = DifferenceInDifferences(**params)
        assert clone._vcov_type_explicit is True
        assert clone.vcov_type == "hc2_bm"

    def test_set_params_robust_false_then_cluster_preserves_cr1(self):
        """set_params path: after `est.set_params(robust=False)` the flag is
        cleared to False, so a subsequent cluster-bearing fit remaps."""
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences()
        est.set_params(robust=False, cluster="unit")
        assert est._vcov_type_explicit is False  # robust= only, no vcov_type
        with pytest.warns(UserWarning, match="robust=False with cluster"):
            res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert res.vcov_type == "hc1"

    def test_hc1_fit_and_summary_contain_expected_fields(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert np.isfinite(res.conf_int[0])
        assert np.isfinite(res.conf_int[1])

    def test_hc1_and_hc2_bm_both_fit(self):
        """HC1 and HC2_BM produce the same point estimate; may share SE on a
        saturated balanced DiD but must still fit cleanly.

        For a saturated 2x2 DiD with balanced cells, h_ii = k/n is constant and
        both HC1 adjustment n/(n-k) and HC2's 1/(1-h_ii) cancel into the same
        vcov. The per-coefficient BM DOF for the saturated interaction happens
        to equal n-k exactly, so CIs match too. This test pins the point
        estimate equivalence, which is the guarantee users can rely on.
        """
        data = _make_did_panel()
        est_hc1 = DifferenceInDifferences(vcov_type="hc1")
        est_hc2bm = DifferenceInDifferences(vcov_type="hc2_bm")
        r_hc1 = est_hc1.fit(data, outcome="y", treatment="treated", time="time")
        r_hc2bm = est_hc2bm.fit(data, outcome="y", treatment="treated", time="time")
        # Point estimate unaffected by vcov choice.
        assert r_hc1.att == pytest.approx(r_hc2bm.att, abs=1e-10)
        # Both produce finite SEs and CIs.
        assert np.isfinite(r_hc1.se)
        assert np.isfinite(r_hc2bm.se)
        assert np.isfinite(r_hc1.conf_int[0]) and np.isfinite(r_hc1.conf_int[1])
        assert np.isfinite(r_hc2bm.conf_int[0]) and np.isfinite(r_hc2bm.conf_int[1])

    def test_classical_via_robust_false(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(robust=False)
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_classical_via_explicit_vcov_type(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="classical")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.se)

    def test_summary_includes_vcov_label_hc1(self):
        """`summary()` output includes an HC1 label in the Variance line."""
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "HC1 heteroskedasticity-robust" in summary

    def test_summary_includes_vcov_label_hc2_bm(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "HC2 + Bell-McCaffrey" in summary

    def test_summary_includes_vcov_label_classical(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="classical")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "Classical OLS SEs" in summary

    def test_summary_includes_vcov_label_cr1(self):
        """CR1 cluster-robust (HC1 + cluster) labels with the cluster name."""
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1", cluster="unit")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "CR1 cluster-robust at unit" in summary

    def test_multi_period_fit_honors_classical(self):
        """MultiPeriodDiD.fit with vcov_type='classical' produces non-robust SEs.

        Regression test for the CI review finding: `MultiPeriodDiD` inherits
        `vcov_type` from the base class via get_params but its `fit()` path
        used to ignore the knob. Here we compare classical vs hc1 SEs on the
        same data and assert they differ (i.e. the parameter actually took).
        """
        rng = np.random.default_rng(20260419)
        n_units = 40
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(4):
                post = int(t >= 2)
                y = rng.normal(0.0, 1.0) + 0.3 * treated + 0.8 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        r_hc1 = MultiPeriodDiD(vcov_type="hc1").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        r_classical = MultiPeriodDiD(vcov_type="classical").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        # Point estimates identical.
        assert r_hc1.avg_att == pytest.approx(r_classical.avg_att, abs=1e-10)
        # SEs must differ — vcov_type actually changed the variance family.
        assert r_hc1.avg_se != pytest.approx(r_classical.avg_se, abs=1e-10)

    def test_multi_period_cluster_plus_hc2_bm_produces_finite_inference(self):
        """MultiPeriodDiD(cluster=..., vcov_type='hc2_bm') is now supported.

        The cluster-aware CR2 Bell-McCaffrey contrast DOF for the
        post-period-average ATT is implemented via the new
        `_compute_cr2_bm_contrast_dof` helper that generalizes the
        per-coefficient loop in `_compute_cr2_bm` to arbitrary linear
        combinations of coefficients. End-to-end smoke test: assert
        finite avg_att inference (was `NotImplementedError` pre-PR).
        """
        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in range(3):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * (t >= 1)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        res = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        # Headline contract: finite avg_att inference under cluster+hc2_bm.
        assert np.isfinite(res.avg_att)
        assert np.isfinite(res.avg_se)
        assert np.isfinite(res.avg_t_stat)
        assert np.isfinite(res.avg_p_value)
        assert np.isfinite(res.avg_conf_int[0])
        assert np.isfinite(res.avg_conf_int[1])
        # Per-period inference should also be finite for the post period.
        post_pe = res.period_effects[2]
        assert np.isfinite(post_pe.effect)
        assert np.isfinite(post_pe.se)
        assert np.isfinite(post_pe.p_value)

    def test_multi_period_hc2_bm_nonfinite_bm_dof_fails_closed(self, monkeypatch):
        """A non-finite (guard-suppressed) Bell-McCaffrey DOF must fail closed to
        NaN inference on user-facing MPD contrasts, not fall back to the shared
        residual df. The `_cr2_bm_dof_inner` guard can NaN a per-coefficient DOF
        (high-leverage / collinear column); reporting finite t/p/CI for such a
        coefficient via the residual df would violate the joint-NaN inference
        contract. Monkeypatches every BM DOF to NaN and asserts SEs stay finite
        (vcov is unaffected) while all t/p/CI fields are NaN.
        """
        import numpy as np

        import diff_diff.linalg as _linalg

        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in range(3):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * (t >= 1)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        _orig = _linalg._compute_cr2_bm_vcov_and_dof

        def _nan_dof(*args, **kwargs):
            vcov, dof = _orig(*args, **kwargs)
            return vcov, np.full(np.asarray(dof).shape, np.nan)

        # MPD's hc2_bm+cluster path imports this from linalg inside fit(), so the
        # local import binds to the patched attribute at call time.
        monkeypatch.setattr(_linalg, "_compute_cr2_bm_vcov_and_dof", _nan_dof)

        res = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        for p, pe in res.period_effects.items():
            assert np.isfinite(pe.se), f"period {p}: SE should remain finite"
            assert np.isnan(pe.t_stat), f"period {p}: t_stat should be NaN"
            assert np.isnan(pe.p_value), f"period {p}: p_value should be NaN"
            assert np.isnan(pe.conf_int[0]) and np.isnan(pe.conf_int[1])
        # The post-period-average contrast fails closed too.
        assert np.isfinite(res.avg_se), "avg SE should remain finite"
        assert np.isnan(res.avg_t_stat)
        assert np.isnan(res.avg_p_value)
        assert np.isnan(res.avg_conf_int[0]) and np.isnan(res.avg_conf_int[1])

    def test_multi_period_cluster_hc2_bm_avg_att_uses_clubsandwich_dof(self):
        """MPD(cluster=..., hc2_bm) `avg_att` inference uses the new
        cluster-aware contrast Satterthwaite DOF, not the shared n-k fallback.

        Pins the implied DOF from `avg_p_value` against the R `dof_avg` target
        on the `mpd_clustered_avg_att_dof` fixture. The R-side compound contrast
        DOF (from `Wald_test(test="HTZ")$df_denom` on a 1-row constraint matrix
        — equivalent to the Satterthwaite t-test DOF) is the parity target.
        Recovers the DOF by inverting `avg_p_value = 2 * (1 - t.cdf(|t|, df))`.
        """
        import json
        from pathlib import Path

        from scipy import stats

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden JSON not present; run "
                "`Rscript benchmarks/R/generate_clubsandwich_golden.R` first."
            )
        with open(golden_path) as f:
            golden = json.load(f)
        if "mpd_clustered_avg_att_dof" not in golden:
            pytest.skip("Golden JSON missing `mpd_clustered_avg_att_dof` scenario.")
        d = golden["mpd_clustered_avg_att_dof"]
        dof_avg_r = float(d["dof_avg"])

        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "y": d["y"],
            }
        )
        # MPD parameterization with `fixed_effects=["unit"]` matches the R
        # generator's `factor(unit)` term (the cluster column is the unit).
        # Derive `post_periods` from the R `post_interaction_names` so the
        # contrast we compare to is the SAME `c_avg = (1/n_post) Σ e_p` that
        # the R generator builds. Without this, MPD defaults to the
        # last-half-of-periods rule and computes avg_att over [3, 4] on this
        # 4-period panel, but the R fixture's `c_avg` is over [2, 3, 4] —
        # the DOFs happen to coincide here but the avg_att estimands differ.
        post_periods = [int(name.rsplit("_", 1)[1]) for name in d["post_interaction_names"]]
        res = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
            data,
            outcome="y",
            treatment="treated",
            time="period",
            fixed_effects=["unit"],
            reference_period=int(d["reference_period"]),
            post_periods=post_periods,
            unit="unit",
        )
        assert np.isfinite(res.avg_att) and np.isfinite(res.avg_se)
        # SE-audit C2: pin the average-effect SE (was only checked finite). The
        # R golden carries the contrast `c_avg` and the CR2 vcov, so the expected
        # avg SE is sqrt(c_avg' Vcr2 c_avg).
        c_avg = np.asarray(d["c_avg"], dtype=np.float64)
        vcov_cr2 = np.asarray(d["vcov_cr2"]).reshape(d["vcov_cr2_shape"], order="F")
        expected_avg_se = float(np.sqrt(c_avg @ vcov_cr2 @ c_avg))
        np.testing.assert_allclose(res.avg_se, expected_avg_se, atol=1e-8, rtol=0)
        # Recover the implied DOF from the reported p_value:
        # avg_p_value = 2 * (1 - t.cdf(|t|, df))  ->  df = root of
        # `t.sf(|t|, df) * 2 - p` (Satterthwaite-bounded scalar bisection
        # via scipy's brentq on a sane interval).
        t_stat = abs(res.avg_t_stat)
        p_target = res.avg_p_value
        # Sanity: BM Satterthwaite DOF is bounded in [1, n-k]. With 60 obs and
        # ~21 coefficients (per the fixture), DOF is in [1, 39].
        from scipy.optimize import brentq

        def _residual(df):
            return 2.0 * stats.t.sf(t_stat, df) - p_target

        implied_dof = brentq(_residual, 1.0, 100.0, xtol=1e-8)
        # The implied DOF should match the R golden at 1e-6 (small tolerance
        # accounts for the t.cdf evaluation roundoff, not the DOF computation).
        np.testing.assert_allclose(implied_dof, dof_avg_r, atol=1e-6)
        # Pin that the new path is in use, not the n-k fallback: dof_avg_r
        # is well below n-k for this fixture (60 obs - 21 coefs = 39 > 8.1).
        assert implied_dof < 30, (
            f"Implied DOF {implied_dof:.2f} is suspiciously large; expected "
            f"~{dof_avg_r:.2f} (Satterthwaite-corrected) and the n-k fallback "
            "would be ~39, so the contrast-DOF helper may not be wired."
        )

        # SE-audit C2: lock the per-period BM Satterthwaite DOF (`dof_per_coef`)
        # via CI-inversion. Each event-study period effect maps to R's
        # `treated_period_p` coefficient; reconstructing that period's CI from
        # the golden per-coef DOF and asserting equality with the reported
        # `conf_int` pins the DOF (effect+se already fixed → the CI matches iff
        # the DOF matches). On this fixture the post-period entries coincide
        # with `dof_avg`, so this completes the per-PERIOD path coverage (a
        # distinct code path from the avg-contrast DOF pinned above), not a new
        # numeric target.
        name_to_dof = dict(zip(d["finite_coef_names"], d["dof_per_coef"]))
        for p in post_periods:
            pe = res.period_effects[p]
            gold_dof = name_to_dof.get(f"treated_period_{p}")
            assert gold_dof is not None, f"golden dof_per_coef missing treated_period_{p}"
            t_crit = float(stats.t.ppf(1.0 - res.alpha / 2.0, gold_dof))
            expected_ci = (pe.effect - t_crit * pe.se, pe.effect + t_crit * pe.se)
            np.testing.assert_allclose(pe.conf_int, expected_ci, atol=1e-8, rtol=0)

    def test_multi_period_fit_honors_hc2_bm(self):
        """MultiPeriodDiD.fit with vcov_type='hc2_bm' uses Bell-McCaffrey DOF.

        Checks two things: (a) fit completes without error on the hc2_bm path
        for the period-effect loop, and (b) the BM Satterthwaite DOF produces
        a CI for avg_att with a finite width (non-degenerate case).
        """
        rng = np.random.default_rng(1919)
        n_units = 50
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(5):
                post = int(t >= 3)
                y = rng.normal(0.0, 1.0) + 0.2 * treated + 0.6 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        r_hc2bm = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        assert np.isfinite(r_hc2bm.avg_att)
        assert np.isfinite(r_hc2bm.avg_se)
        assert np.isfinite(r_hc2bm.avg_conf_int[0])
        assert np.isfinite(r_hc2bm.avg_conf_int[1])
        # CI width is finite and positive.
        ci_width = r_hc2bm.avg_conf_int[1] - r_hc2bm.avg_conf_int[0]
        assert ci_width > 0

    def test_twfe_hc2_and_hc2_bm_produce_finite_inference(self):
        """TWFE with vcov_type in {hc2, hc2_bm} now produces finite inference
        via the inline full-dummy build (Gate 1 lift).

        FWL preserves coefficients and residuals but NOT the hat matrix, so
        HC2 leverage and CR2-BM DOF must compute on the full FE projection.
        TWFE.fit bypasses the within-transform on these vcov_types and stacks
        [intercept, treated*post, covariates, unit_dummies, time_dummies]
        explicitly.
        """
        data = _make_did_panel(n_units=20)
        for vcov in ("hc2", "hc2_bm"):
            res = TwoWayFixedEffects(vcov_type=vcov).fit(
                data,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
            )
            assert np.isfinite(res.att), f"{vcov}: ATT not finite"
            assert np.isfinite(res.se), f"{vcov}: SE not finite"
            assert res.se > 0, f"{vcov}: SE not positive"
            assert np.isfinite(res.p_value), f"{vcov}: p-value not finite"
            ci = res.conf_int
            assert np.isfinite(ci[0]) and np.isfinite(ci[1]), f"{vcov}: CI not finite"

    def test_twfe_hc2_matches_did_fixed_effects_full_dummy(self):
        """TWFE(vcov_type='hc2') is bit-equal to DifferenceInDifferences with
        fixed_effects=[unit, time] (same full-dummy algebra under the hood).

        Compares only .att and .se — the full .coefficients dict may differ
        because pd.get_dummies(drop_first=True) reference-category ordering
        is not guaranteed identical between TWFE's inline build and DiD's
        fixed_effects= branch.
        """
        data = _make_did_panel(n_units=20)
        res_twfe = TwoWayFixedEffects(vcov_type="hc2").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        res_did = DifferenceInDifferences(vcov_type="hc2").fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
        )
        np.testing.assert_allclose(res_twfe.att, res_did.att, atol=1e-12)
        np.testing.assert_allclose(res_twfe.se, res_did.se, atol=1e-12)

    def test_twfe_hc2_bm_matches_did_fixed_effects_full_dummy(self):
        """Same refactor-regression check as the hc2 variant, for hc2_bm.

        Note: TWFE's hc2_bm path auto-clusters at unit (preserved), while DiD
        does NOT auto-cluster — so we explicitly pass cluster='unit' to DiD
        to align the inference paths.
        """
        data = _make_did_panel(n_units=20)
        res_twfe = TwoWayFixedEffects(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        res_did = DifferenceInDifferences(vcov_type="hc2_bm", cluster="unit").fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
        )
        np.testing.assert_allclose(res_twfe.att, res_did.att, atol=1e-12)
        np.testing.assert_allclose(res_twfe.se, res_did.se, atol=1e-12)

    def test_twfe_hc2_bm_auto_clusters_at_unit(self):
        """TWFE(vcov_type='hc2_bm') with no explicit cluster routes to CR2-BM
        at unit (auto-cluster default preserved on the hc2_bm path).

        Two-pronged verification, both required to distinguish CR2-BM-at-unit
        from one-way HC2-BM:

        (1) **Equivalence check against a reference path**:
            DifferenceInDifferences(vcov_type='hc2_bm', cluster='unit',
            fixed_effects=[unit, time]). Both paths share the full-dummy
            design and the same CR2-BM Satterthwaite DOF at unit, so ATT
            and SE match bit-equally at atol=1e-12.

        (2) **Inequality check against one-way HC2-BM on the same X**:
            on the shared 20×4 multi-period fixture, CR2-BM-at-unit and
            one-way HC2-BM produce numerically different SEs (ratio ~1.22).
            Without this check, the test would pass even if TWFE silently
            fell through to one-way HC2-BM (on a 2-period panel the two
            paths happen to coincide numerically, defeating the equivalence
            check above). The 4-period fixture separates them.
        """
        # Multi-period panel: cluster blocks of size 4 do NOT coincide with
        # the unit FE structure in the same way 2-obs clusters would.
        rng = np.random.default_rng(20260420)
        n_units, n_periods = 20, 4
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(n_periods):
                post = int(t >= n_periods // 2)
                y = rng.normal(0.0, 1.0) + 0.5 * treated + 1.0 * treated * post
                rows.append({"unit": i, "time": post, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        res_twfe = TwoWayFixedEffects(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        # Auto-cluster fires; result reports unit as the cluster name.
        assert res_twfe.cluster_name == "unit"
        assert np.isfinite(res_twfe.se) and res_twfe.se > 0

        # (1) Reference path: explicit CR2-BM at unit via DiD's fixed_effects=
        # branch. TWFE's auto-cluster should land on the same algebra at
        # machine precision.
        res_did = DifferenceInDifferences(vcov_type="hc2_bm", cluster="unit").fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
        )
        np.testing.assert_allclose(res_twfe.att, res_did.att, atol=1e-12)
        np.testing.assert_allclose(res_twfe.se, res_did.se, atol=1e-12)

        # (2) Sanity: the auto-clustered SE must NOT equal the one-way
        # HC2-BM SE on the same full-dummy X. If it did, a regression where
        # TWFE silently dropped the auto-cluster (one-way fall-through) would
        # slip through the equivalence check above.
        from diff_diff.linalg import solve_ols

        df_local = data.copy()
        df_local["_tp"] = df_local["treated"] * df_local["time"]
        unit_dummies = pd.get_dummies(
            df_local["unit"], prefix="_fe_unit", drop_first=True
        ).values.astype(np.float64)
        time_dummies = pd.get_dummies(
            df_local["time"], prefix="_fe_time", drop_first=True
        ).values.astype(np.float64)
        X = np.column_stack(
            [
                np.ones(len(df_local)),
                df_local["_tp"].values.astype(np.float64),
                unit_dummies,
                time_dummies,
            ]
        )
        y = df_local["y"].values.astype(np.float64)
        _, _, vcov_one_way = solve_ols(X, y, vcov_type="hc2_bm")
        se_one_way_att = float(np.sqrt(vcov_one_way[1, 1]))
        # Use a meaningful tolerance: on this fixture the two SEs differ by
        # ~22%; require at least 1% gap to lock in the distinction.
        assert abs(res_twfe.se - se_one_way_att) / se_one_way_att > 0.01, (
            f"auto-cluster CR2-BM SE ({res_twfe.se}) coincides with one-way "
            f"HC2-BM SE ({se_one_way_att}); the test cannot distinguish "
            "the two paths on this fixture, so a regression where TWFE "
            "silently drops the unit cluster would not be caught."
        )

    def test_twfe_hc2_explicit_no_auto_cluster_analytical(self):
        """Explicit `vcov_type='hc2'` + analytical inference drops the unit
        auto-cluster (one-way HC2; the linalg validator rejects hc2 + cluster).
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(vcov_type="hc2", inference="analytical").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        # No auto-cluster on explicit one-way hc2 + analytical.
        assert res.cluster_name is None

    def test_twfe_hc2_wild_bootstrap_survives_rank_deficient_full_dummy(self):
        """TWFE(vcov_type='hc2', inference='wild_bootstrap') stays finite when
        the full-dummy design has a rank-deficient nuisance column.

        Regression for a P1 bug in `wild_bootstrap_se()`: it previously built
        `y_star = X @ beta_restricted`, which propagates NaN through every
        observation whenever solve_ols dropped a nuisance column (e.g. a
        time-invariant covariate collinear with the unit FE). The ATT was
        analytically identified, but the bootstrap crashed because every
        `y_star` was all-NaN. Reachable on the new TWFE HC2 full-dummy path
        (the within-transform path absorbed time-invariant covariates so
        the issue was hidden pre-PR).

        Fix: `wild_bootstrap_se()` now uses solve_ols's kept-columns
        `fitted_restricted` instead of `X @ beta_restricted`, so dropped
        nuisance columns no longer poison `y_star`.
        """
        data = _make_did_panel(n_units=20).copy()
        # x_invariant is time-invariant (only varies across units),
        # so it's collinear with the unit fixed effect on the
        # full-dummy design and gets dropped by solve_ols.
        rng = np.random.default_rng(99)
        unit_to_x = {u: rng.normal() for u in data["unit"].unique()}
        data["x_invariant"] = data["unit"].map(unit_to_x).astype(float)
        with warnings.catch_warnings():
            # The expected rank-deficient column drop emits a UserWarning;
            # we accept it as part of the documented full-dummy path.
            warnings.simplefilter("ignore", UserWarning)
            res = TwoWayFixedEffects(
                vcov_type="hc2",
                inference="wild_bootstrap",
                n_bootstrap=50,
                seed=1,
            ).fit(
                data,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                covariates=["x_invariant"],
            )
        # ATT remains identified despite the dropped nuisance column.
        assert np.isfinite(res.att), "ATT should remain finite despite rank deficiency"
        assert np.isfinite(res.se), (
            "Bootstrap SE should be finite — if NaN, wild_bootstrap_se's "
            "y_star construction is propagating NaN from beta_restricted."
        )
        assert res.se > 0
        assert np.isfinite(res.p_value)
        assert np.isfinite(res.conf_int[0]) and np.isfinite(res.conf_int[1])

    def test_twfe_hc2_wild_bootstrap_keeps_auto_cluster(self):
        """Wild-bootstrap inference on TWFE(vcov_type='hc2') must keep the
        unit auto-cluster (bootstrap resampling uses the cluster structure).

        Regression for the auto-cluster sub-guard: omitting the
        `inference == "analytical"` companion would crash wild_bootstrap
        with `np.unique(None)` TypeError.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(
            vcov_type="hc2",
            inference="wild_bootstrap",
            n_bootstrap=50,
            seed=1,
        ).fit(data, outcome="y", treatment="treated", time="time", unit="unit")
        assert np.isfinite(res.se)
        assert res.se > 0
        # Bootstrap consumed unit-level clusters.
        assert res.n_clusters == 20

    @pytest.mark.parametrize("vcov", ["hc2", "hc2_bm"])
    def test_twfe_rejects_replicate_weights_under_hc2(self, vcov):
        """TWFE + hc2/hc2_bm + replicate-weight survey design raises
        NotImplementedError.

        The replicate path re-demeans per replicate (re-demeaning depends
        on the per-replicate weight vector), which doesn't compose with
        the full-dummy build. Documented scope limit; tracked in TODO.md.
        """
        data = _make_did_panel(n_units=20).copy()
        # Attach full-sample weight + 4 BRR replicate-weight columns.
        rng = np.random.default_rng(0)
        data["weight"] = 1.0
        rep_cols = [f"rep{r}" for r in range(4)]
        for col in rep_cols:
            data[col] = rng.choice([0.5, 1.5], size=len(data))
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
            weight_type="pweight",
        )
        with pytest.raises(
            NotImplementedError,
            match=r"replicate-weight.*not yet supported",
        ):
            TwoWayFixedEffects(vcov_type=vcov).fit(
                data,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                survey_design=sd,
            )

    def test_twfe_hc2_always_treated_unit_finite_att(self):
        """Always-treated unit (D=1 in all periods) doesn't poison the ATT
        on the full-dummy HC2 path.

        The plan's footgun was theoretical (always-treated unit × treat_post
        could be collinear with the unit dummy). In practice, on a 2-period
        DiD with at least one switching cohort, the design retains full rank.
        Pivoted-QR in solve_ols would cleanly drop any column that DID
        become rank-deficient on a more degenerate design.
        """
        data = _make_did_panel(n_units=20)
        # Make unit 0 always-treated (treated=1 in both periods).
        data = data.copy()
        data.loc[data["unit"] == 0, "treated"] = 1
        # Recompute treat * time for the always-treated rows.
        # (TWFE.fit builds _treatment_post internally from data[treatment] *
        # data[time], so we just need data["treated"] and data["time"] right.)
        res = TwoWayFixedEffects(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert res.se > 0

    @pytest.mark.parametrize("vcov", ["hc2", "hc2_bm"])
    def test_twfe_hc2_coefficients_align_with_vcov(self, vcov):
        """Under the full-dummy HC2/HC2-BM path, `result.coefficients` must
        carry one entry per `result.vcov` column (no duplicates, no
        collapsing).

        Mirrors the MPD invariant at
        ``test_absorb_hc2_bm_coefficients_align_with_vcov`` (line 1611)
        and the REGISTRY/CHANGELOG promise that the full-dummy fit
        exposes the FE-dummy entries alongside the ATT key.
        """
        from collections import Counter

        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(vcov_type=vcov).fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert res.vcov is not None
        assert res.vcov.shape[0] == res.vcov.shape[1]
        assert len(res.coefficients) == res.vcov.shape[0], (
            f"[{vcov}] coefficients dict length ({len(res.coefficients)}) "
            f"must match vcov rank ({res.vcov.shape[0]}); duplicate var_names "
            "or hardcoded {'ATT': ...} would break this invariant."
        )
        dups = {k: v for k, v in Counter(res.coefficients.keys()).items() if v > 1}
        assert not dups, f"[{vcov}] duplicate names in coefficients: {dups}"
        # Backward-compat: ATT key still resolves to the ATT coefficient.
        assert "ATT" in res.coefficients
        assert np.isclose(res.coefficients["ATT"], res.att, atol=1e-12)

    @pytest.mark.parametrize("vcov", ["hc2", "hc2_bm"])
    def test_twfe_hc2_full_surface_matches_did_fixed_effects(self, vcov):
        """Under the HC2/HC2-BM full-dummy path, the entire `DiDResults`
        surface (residuals, fitted_values, r_squared) reflects the
        full-dummy fit and matches DiD(fixed_effects=[unit, time]) bit-
        equally, not just ATT/SE.

        Regression for the REGISTRY/CHANGELOG disclosure that under
        `vcov_type in {"hc2","hc2_bm"}`, `result.residuals`,
        `result.fitted_values`, and `result.r_squared` reflect the
        un-demeaned full-dummy fit (matching DiD-absorb / MPD-absorb
        auto-route behavior).
        """
        data = _make_did_panel(n_units=20)
        res_twfe = TwoWayFixedEffects(vcov_type=vcov).fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        cluster_kwarg = "unit" if vcov == "hc2_bm" else None
        res_did = DifferenceInDifferences(vcov_type=vcov, cluster=cluster_kwarg).fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
        )
        assert res_twfe.residuals is not None and res_did.residuals is not None
        assert res_twfe.fitted_values is not None and res_did.fitted_values is not None
        np.testing.assert_allclose(res_twfe.residuals, res_did.residuals, atol=1e-12)
        np.testing.assert_allclose(res_twfe.fitted_values, res_did.fitted_values, atol=1e-12)
        np.testing.assert_allclose(res_twfe.r_squared, res_did.r_squared, atol=1e-12)

    @pytest.mark.parametrize("vcov", ["hc2", "hc2_bm"])
    def test_twfe_hc2_with_survey_weights_matches_did_fixed_effects(self, vcov):
        """TWFE(vcov_type in {'hc2','hc2_bm'}) with a non-replicate
        SurveyDesign(weights=...) routes through the full-dummy build,
        with survey TSL variance taking precedence over the analytical
        HC2/HC2-BM sandwich (per the documented survey-design scope).

        End-to-end consistency check: TWFE's auto-route on the full-dummy
        design under survey weights must match
        DifferenceInDifferences(fixed_effects=[unit, time]) with the same
        survey design and cluster. Both paths feed the survey-resolved
        design to LinearRegression's compute_survey_vcov (TSL) on an
        identical full-dummy X, so ATT and SE match bit-equally at
        atol=1e-12. Regression for the concern that the survey path could
        revert to the within-transform branch or mishandle PSU injection
        under the new FE route.

        Note: `cluster='unit'` is passed EXPLICITLY to TWFE on the
        `hc2_bm` branch to align with DiD's explicit-cluster + survey
        PSU-injection convention. Without explicit cluster, TWFE's
        survey-design scope rule (twfe.py:_resolve_effective_cluster
        branch) drops the auto-cluster from PSU injection — that's
        intentional but causes the path to diverge from DiD here. The
        explicit-cluster form is the documented user-facing way to
        invoke clustered survey-aware HC2-BM on TWFE.
        """
        data = _make_did_panel(n_units=20).copy()
        rng = np.random.default_rng(7)
        data["w"] = rng.uniform(0.5, 2.0, size=len(data))
        sd = SurveyDesign(weights="w")
        # Explicit cluster on both paths so PSU injection matches.
        cluster_kwarg = "unit" if vcov == "hc2_bm" else None
        res_twfe = TwoWayFixedEffects(vcov_type=vcov, cluster=cluster_kwarg).fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            survey_design=sd,
        )
        res_did = DifferenceInDifferences(vcov_type=vcov, cluster=cluster_kwarg).fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
            survey_design=sd,
        )
        np.testing.assert_allclose(res_twfe.att, res_did.att, atol=1e-12)
        np.testing.assert_allclose(res_twfe.se, res_did.se, atol=1e-12)

    @pytest.mark.parametrize("vcov", ["hc2", "hc2_bm"])
    def test_twfe_hc2_with_survey_strata_psu_matches_did_fixed_effects(self, vcov):
        """TWFE(vcov_type in {'hc2','hc2_bm'}) with a full SurveyDesign
        (weights + strata + psu) routes through the full-dummy build, with
        survey TSL variance (including stratified-design adjustments)
        taking precedence over the analytical sandwich.

        Extends the weights-only regression with a multi-stage survey
        design (strata + PSU). Verifies that TWFE's full-dummy route
        threads strata / PSU columns to LinearRegression's survey
        variance path identically to DiD's fixed_effects= branch — so
        ATT and SE match bit-equally at atol=1e-12 under non-trivial
        survey design metadata.
        """
        data = _make_did_panel(n_units=20).copy()
        rng = np.random.default_rng(11)
        data["w"] = rng.uniform(0.5, 2.0, size=len(data))
        # Stratum = unit cohort (treated vs control); PSU = unit. Both
        # constant within each unit, satisfying typical survey-design
        # constraints. Globally unique PSU ids per SurveyDesign convention.
        data["stratum"] = data["treated"].astype(int)
        data["psu"] = data["unit"].astype(int)
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu")
        # Explicit cluster='unit' on both paths so PSU injection matches
        # under hc2_bm; hc2 paths drop the cluster as one-way.
        cluster_kwarg = "unit" if vcov == "hc2_bm" else None
        res_twfe = TwoWayFixedEffects(vcov_type=vcov, cluster=cluster_kwarg).fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            survey_design=sd,
        )
        res_did = DifferenceInDifferences(vcov_type=vcov, cluster=cluster_kwarg).fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit", "time"],
            survey_design=sd,
        )
        np.testing.assert_allclose(res_twfe.att, res_did.att, atol=1e-12)
        np.testing.assert_allclose(res_twfe.se, res_did.se, atol=1e-12)

    def test_twfe_results_record_cluster_name(self):
        """TWFE results should label the auto-clustered SE with the unit column."""
        rng = np.random.default_rng(1)
        n_units = 20
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(3):
                post = int(t >= 1)
                y = rng.normal(0.0, 1.0) + 0.5 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        res = TwoWayFixedEffects(vcov_type="hc1").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        summary = res.summary()
        # TWFE auto-clusters at the unit column when cluster=None.
        assert "CR1 cluster-robust at unit" in summary

    def test_twfe_honors_classical_without_autocluster(self):
        """TWFE with vcov_type='classical' must skip its unit auto-cluster.

        Classical SEs are one-way only and would be rejected by the linalg
        validator if TWFE still injected unit-level clustering. The fix
        drops the auto-cluster when the user explicitly asks for a one-way
        family.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(vcov_type="classical").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert res.se > 0
        assert res.vcov_type == "classical"
        # Without an explicit cluster and with a one-way family, TWFE should
        # NOT have injected unit as the auto-cluster.
        assert res.cluster_name is None
        summary = res.summary()
        # Summary must label the one-way family, not CR1 cluster-robust.
        assert "Classical OLS" in summary
        assert "CR1 cluster-robust" not in summary

    def test_twfe_explicit_classical_without_autocluster(self):
        """`vcov_type="classical"` EXPLICIT on TWFE disables the auto-cluster
        (the user is deliberately asking for one-way non-robust SEs). The
        implicit ``robust=False`` path instead preserves CR1 at unit via the
        backward-compat remap — covered by
        ``test_twfe_robust_false_preserves_cr1_via_autocluster``.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(vcov_type="classical").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert res.vcov_type == "classical"
        assert res.cluster_name is None
        assert "CR1 cluster-robust" not in res.summary()

    def test_twfe_wild_bootstrap_preserves_auto_cluster(self):
        """Wild-bootstrap inference on TWFE with no explicit cluster must
        keep the unit auto-cluster, even under vcov_type='classical'.

        Regression guard for a bug where the one-way-family auto-cluster
        bypass also applied under wild_bootstrap, silently dropping the
        cluster structure the bootstrap was supposed to consume. The fix
        gates the bypass on inference=='analytical'.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(
            vcov_type="classical",
            inference="wild_bootstrap",
            n_bootstrap=50,
            seed=1,
        ).fit(data, outcome="y", treatment="treated", time="time", unit="unit")
        # Bootstrap must have succeeded with a finite SE.
        assert np.isfinite(res.se)
        assert res.se > 0
        # Bootstrap consumed a unit-level cluster (20 clusters).
        assert res.n_clusters == 20

    def test_did_absorb_hc2_and_hc2_bm_auto_route(self):
        """DifferenceInDifferences with absorb= + HC2/HC2-BM now auto-routes to
        fixed_effects= internally.

        FWL preserves coefficients but not the hat matrix; HC2/CR2-BM leverage
        corrections require the FULL FE hat matrix. Rather than reject, we
        internally promote absorb= to fixed_effects= so the existing full-
        dummy design path computes the algebraically correct vcov.

        Verifies: (1) fit succeeds (no NotImplementedError); (2) ATT matches
        between absorb-routed and explicit fixed_effects= paths; (3) SE
        matches between the two paths (bit-equal — same algebra under the
        hood).
        """
        rng = np.random.default_rng(20260420)
        n_units, n_time = 30, 3
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(n_time):
                post = int(t >= 1)
                y = rng.normal(0.0, 1.0) + 0.5 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "post": post, "y": y})
        data = pd.DataFrame(rows)

        for vcov in ("hc2", "hc2_bm"):
            res_absorb = DifferenceInDifferences(vcov_type=vcov).fit(
                data,
                outcome="y",
                treatment="treated",
                time="post",
                absorb=["unit"],
            )
            res_fe = DifferenceInDifferences(vcov_type=vcov).fit(
                data,
                outcome="y",
                treatment="treated",
                time="post",
                fixed_effects=["unit"],
            )
            assert np.isfinite(res_absorb.att)
            assert np.isfinite(res_absorb.se)
            # Auto-route should be bit-equal to explicit fixed_effects= path.
            np.testing.assert_allclose(res_absorb.att, res_fe.att, atol=1e-12)
            np.testing.assert_allclose(res_absorb.se, res_fe.se, atol=1e-12)

    def test_did_fixed_effects_dummies_still_accept_hc2_and_hc2_bm(self):
        """DifferenceInDifferences with fixed_effects= (dummy expansion) is
        NOT affected by the absorb-FE guard: the dummies appear in the full
        design matrix, so HC2 leverage is computed on the full projection.
        """
        rng = np.random.default_rng(20260420)
        n_units, n_time = 20, 2
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            stratum = i // 5  # categorical for fixed_effects= dummies
            for t in range(n_time):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * t
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "treated": treated,
                        "post": t,
                        "stratum": stratum,
                        "y": y,
                    }
                )
        data = pd.DataFrame(rows)

        # Neither call should raise.
        for good in ("hc2", "hc2_bm"):
            res = DifferenceInDifferences(vcov_type=good).fit(
                data,
                outcome="y",
                treatment="treated",
                time="post",
                fixed_effects=["stratum"],
            )
            assert np.isfinite(res.att)
            assert np.isfinite(res.se)

    def test_summary_suppresses_variance_line_under_wild_bootstrap(self):
        """When inference_method='wild_bootstrap', the Variance label is omitted.

        The wild-bootstrap path reports bootstrap SE/CI, not analytical. Printing
        an analytical family like 'HC1 heteroskedasticity-robust' under those
        numbers would be misleading.
        """
        rng = np.random.default_rng(42)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in (0, 1):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * t
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        est = DifferenceInDifferences(
            vcov_type="hc1",
            inference="wild_bootstrap",
            cluster="unit",
            n_bootstrap=50,
            seed=7,
        )
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        # The bootstrap path substitutes SE/CI from resampling; the Variance:
        # line (which labels the analytical family) must be suppressed so the
        # displayed inference is unambiguous.
        assert "Variance:" not in summary
        # But the inference method should still be visible.
        assert "wild_bootstrap" in summary

    def test_wild_bootstrap_preserves_vcov_type_no_error(self):
        """Wild-bootstrap inference path doesn't fight with vcov_type.

        The wild-bootstrap SE comes from resampling, not from the analytical
        sandwich. `vcov_type` has no effect on the bootstrap SE output, but
        the fit should still succeed without errors.
        """
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences(
            vcov_type="hc2_bm",
            inference="wild_bootstrap",
            n_bootstrap=50,
            seed=42,
        )
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.se)


# =============================================================================
# Survey-fit summary labeling (P2 fix from CI review on PR #327)
# =============================================================================


def _make_survey_panel(seed: int = 20260420) -> pd.DataFrame:
    """Two-period DiD panel with strata/PSU/weight columns for survey fits.

    40 units, 4 strata (10 units each), 8 PSUs nested within strata (2 PSUs
    per stratum, 5 units each). Treatment is 20 vs 20; PSU labels are
    globally unique so SurveyDesign.resolve does not raise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    n_units = 40
    for i in range(n_units):
        treated = int(i >= n_units // 2)
        stratum = i // 10  # 4 strata, 10 units each
        psu = i // 5  # 8 PSUs globally (2 per stratum)
        wt = 1.0 + 0.25 * stratum
        for t in (0, 1):
            y = rng.normal(0.0, 1.0) + 0.5 * treated + 1.0 * treated * t
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "treated": treated,
                    "stratum": stratum,
                    "psu": psu,
                    "weight": wt,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


class TestSummarySurveyLabeling:
    """When a SurveyDesign drives inference, the analytical `Variance:` line
    must be suppressed: the reported SEs come from Taylor linearization or
    replicate-weight variance, not from the analytical HC/CR sandwich. The
    survey inference block (weight_type, strata/PSU counts, replicate method)
    already surfaces the actual inference source; a parallel
    `Variance: HC1/...` line would mislabel what produced the SEs.

    These tests pin the P2 fix flagged by CI review on PR #327.
    """

    def test_survey_taylor_suppresses_analytical_variance_label(self):
        """SurveyDesign with PSU/strata (no replicate weights) uses Taylor
        linearization; the analytical `Variance:` line must not appear.
        """
        data = _make_survey_panel()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        # Explicit vcov_type="hc1" to make the regression meaningful: if the
        # suppression wasn't in place, the summary would print "HC1
        # heteroskedasticity-robust" even though the SE came from survey
        # Taylor linearization.
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        # The analytical Variance: label must not appear; the survey design
        # line(s) already surface the actual inference source.
        assert "Variance:" not in summary
        # And the summary must still show the survey design block so the
        # user can see where the SEs came from.
        assert (
            "pweight" in summary
            or "Weight type" in summary
            or "n_psu" in summary.lower()
            or "psu" in summary.lower()
        )

    def test_survey_replicate_weights_suppresses_analytical_variance_label(self):
        """SurveyDesign with replicate_weights (BRR) drives replicate-variance
        inference; the analytical `Variance:` line must not appear.
        """
        data = _make_survey_panel()
        # Attach 10 BRR replicate-weight columns.
        rng = np.random.default_rng(12345)
        rep_cols = [f"rep{r}" for r in range(10)]
        for col in rep_cols:
            data[col] = rng.choice([0.5, 1.5], size=len(data))

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
            weight_type="pweight",
        )
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        # The analytical HC2+BM label must not appear for a replicate-weight
        # fit: the actual SEs come from the BRR replicates.
        assert "Variance:" not in summary
        assert "HC2 + Bell-McCaffrey" not in summary
        # Survey metadata should surface the replicate method.
        assert "BRR" in summary or "replicate" in summary.lower()

    def test_multi_period_survey_taylor_suppresses_variance_label(self):
        """Same survey suppression holds for `MultiPeriodDiDResults.summary()`.

        MultiPeriodDiD has its own summary block and its own gating logic; the
        P2 fix applies there too.
        """
        data = _make_survey_panel()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        est = MultiPeriodDiD(vcov_type="hc1")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        assert "Variance:" not in summary

    def test_non_survey_fit_still_prints_variance_label(self):
        """Regression guard: the survey-only suppression must not break the
        non-survey path, which should still print the analytical Variance: line.
        """
        data = _make_did_panel(n_units=30)
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert res.survey_metadata is None
        summary = res.summary()
        assert "Variance:" in summary
        assert "HC1" in summary


class TestDiDAbsorbedFERParity:
    """R-parity for `DifferenceInDifferences(absorb=..., vcov_type in {hc2, hc2_bm})`.

    The auto-route promotes absorb= to fixed_effects= internally, building
    the full-dummy design that R's `lm(y ~ treat_post + factor(unit) +
    factor(period))` produces. HC2-BM unclustered is computed via
    clubSandwich's singleton-cluster CR2 trick; CR2 clustered by unit uses
    `vcovCR(..., cluster=d$unit, type="CR2")`. Parity tolerance 1e-6
    (empirically matches at ≤ 1e-10 in the local smoke test).
    """

    def _load_golden(self):
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden JSON not present; run `Rscript "
                "benchmarks/R/generate_clubsandwich_golden.R` to generate."
            )
        with open(golden_path) as f:
            golden = json.load(f)
        if "absorbed_fe_did" not in golden:
            pytest.skip(
                "Golden JSON does not yet include `absorbed_fe_did` scenario; "
                "regenerate via the R script."
            )
        return golden["absorbed_fe_did"]

    def _fit_absorb(self, d, vcov_type):
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "post": d["post"],
                "y": d["y"],
            }
        )
        return DifferenceInDifferences(vcov_type=vcov_type).fit(
            data,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "period"],
            unit="unit",
        )

    def test_absorb_hc2_bm_matches_clubsandwich_singleton_cluster(self):
        """`absorb=` + `hc2_bm` matches `lm() + clubSandwich::vcovCR(cluster=1:n)`.

        Asserts on the treat_post slope SE only (the inference target);
        FE-dummy coefficient SEs are not the user-facing inference and
        can differ in higher decimal places due to absorbed-FE rank
        treatment.
        """
        d = self._load_golden()
        res = self._fit_absorb(d, "hc2_bm")
        coef_names = d["coef_names"]
        treat_post_idx = coef_names.index("treat_post")
        expected_vcov = np.asarray(d["vcov_hc2_bm"]).reshape(d["vcov_hc2_bm_shape"])
        expected_se_slope = float(np.sqrt(expected_vcov[treat_post_idx, treat_post_idx]))
        expected_dof_slope = float(d["dof_hc2_bm"][treat_post_idx])
        np.testing.assert_allclose(res.se, expected_se_slope, atol=1e-10)
        # ATT also bit-equal.
        np.testing.assert_allclose(res.att, float(d["coef"][treat_post_idx]), atol=1e-10)
        # SE-audit C2: CI-inversion lock of `dof_hc2_bm`. res.conf_int uses a
        # Satterthwaite t-dist with the BM DOF; with att+se already pinned above,
        # reconstructing the CI from the golden DOF pins the DOF itself (the CI
        # matches iff the estimator's DOF equals the golden DOF). Verified this
        # session: matches to ~2e-16, diverging ~0.03 under the wrong n-k DOF.
        from scipy import stats

        t_crit = float(stats.t.ppf(1.0 - res.alpha / 2.0, expected_dof_slope))
        expected_ci = (res.att - t_crit * res.se, res.att + t_crit * res.se)
        np.testing.assert_allclose(res.conf_int, expected_ci, atol=1e-9, rtol=0)

    def test_unweighted_cr2_bm_per_coef_dof_no_nonphysical(self):
        """Unweighted clustered CR2-BM per-coef DOF: physical or NaN, never garbage.

        Direct `_compute_cr2_bm_contrast_dof(..., weights=None)` on the full-dummy
        absorbed-FE design. The well-conditioned contrasts estimators actually
        consume (`treat_post`, the period dummies) match R clubSandwich; the
        high-leverage FE-dummy / intercept nuisance columns previously returned
        non-physical DOF (~1e61 from float64-noise `trace_B2`, and ~32.7 / ~16.3
        vs R's 6 / 3 from the simple `(tr B)²/tr(B²)` form). The noise-floor +
        cluster-count (`DOF <= G`) guards now NaN those instead of shipping them.
        """
        import numpy as np

        from diff_diff.linalg import _compute_cr2_bm_contrast_dof

        d = self._load_golden()
        names = d["coef_names"]
        n = len(d["y"])
        treated = np.asarray(d["treated"], dtype=float)
        post = np.asarray(d["post"], dtype=float)
        unit = np.asarray(d["unit"])
        period = np.asarray(d["period"])
        tp = treated * post

        def _col(nm):
            if nm == "(Intercept)":
                return np.ones(n)
            if nm == "treat_post":
                return tp
            if nm.startswith("period"):
                return (period == int(nm[len("period") :])).astype(float)
            if nm.startswith("unit"):
                return (unit == int(nm[len("unit") :])).astype(float)
            return np.zeros(n)

        X = np.column_stack([_col(nm) for nm in names])
        k = X.shape[1]
        G = len(np.unique(unit))
        gold = d["dof_cr2"]

        with pytest.warns(UserWarning, match="noise floor|cluster-count"):
            dof = _compute_cr2_bm_contrast_dof(X, unit, X.T @ X, np.eye(k), weights=None)

        # No finite DOF exceeds the cluster count (rigorous Satterthwaite bound).
        finite = np.isfinite(dof)
        assert np.all(dof[finite] <= G + 1e-6), (
            f"non-physical DOF (> G={G}): "
            f"{[(names[i], dof[i]) for i in range(k) if finite[i] and dof[i] > G + 1e-6]}"
        )
        # The user-facing / well-conditioned contrasts still match R clubSandwich.
        for nm in ["treat_post", "period2", "period3", "period4"]:
            i = names.index(nm)
            assert dof[i] == pytest.approx(float(gold[i]), abs=1e-6), f"{nm} DOF vs R"
        # The high-leverage nuisance columns are suppressed to NaN, not garbage.
        assert np.isnan(dof[names.index("unit2")]), "collinear unit dummy DOF should be NaN"

    def test_unweighted_cr2_bm_dof_scale_invariant_batch(self):
        """The batch-relative noise-floor guard is contrast-scale invariant.

        Satterthwaite DOF is scale-invariant, but the guard statistic `max|B|`
        scales as ‖c‖². The batch-relative floor divides by ‖c‖² so the same
        contrast passed at two scales in one batch yields identical finite DOF -
        without the normalization the smaller-scale copy would be spuriously
        NaN'd (its `max|B|` sits `scale²` below the larger copy's).
        """
        import numpy as np

        from diff_diff.linalg import _compute_cr2_bm_contrast_dof

        d = self._load_golden()
        names = d["coef_names"]
        n = len(d["y"])
        treated = np.asarray(d["treated"], dtype=float)
        post = np.asarray(d["post"], dtype=float)
        unit = np.asarray(d["unit"])
        X = np.column_stack([np.ones(n) if nm == "(Intercept)" else np.zeros(n) for nm in names])
        # Rebuild the full design (same as the sibling test).
        period = np.asarray(d["period"])
        tp = treated * post
        for j, nm in enumerate(names):
            if nm == "treat_post":
                X[:, j] = tp
            elif nm.startswith("period"):
                X[:, j] = (period == int(nm[len("period") :])).astype(float)
            elif nm.startswith("unit"):
                X[:, j] = (unit == int(nm[len("unit") :])).astype(float)
        k = X.shape[1]
        c = np.zeros(k)
        c[names.index("treat_post")] = 1.0
        # Same contrast direction at scale 1 and 1e6 in a single batch. Both are
        # the well-conditioned treatment contrast, so neither is degenerate and no
        # warning fires (the guard must not spuriously flag the smaller-scale copy).
        contrasts = np.column_stack([c, 1e6 * c])
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any noise-floor warning here is a bug
            dof = _compute_cr2_bm_contrast_dof(X, unit, X.T @ X, contrasts, weights=None)
        assert np.all(np.isfinite(dof)), f"scale-only difference should not NaN: {dof}"
        assert dof[0] == pytest.approx(dof[1], rel=1e-12), "DOF must be scale-invariant"

    def test_absorb_hc2_matches_sandwich_vcovhc(self):
        """`absorb=` + `hc2` matches `lm() + sandwich::vcovHC(type="HC2")`.

        Pins the HC2 SE on the treat_post slope against an external R target
        (the R generator computes `sandwich::vcovHC(fit_did, type="HC2")` on
        the full-dummy design and stores `vcov_hc2`).
        """
        d = self._load_golden()
        if "vcov_hc2" not in d:
            pytest.skip(
                "Golden JSON does not yet include `vcov_hc2` for absorbed_fe_did; "
                "regenerate via the R script."
            )
        res = self._fit_absorb(d, "hc2")
        coef_names = d["coef_names"]
        treat_post_idx = coef_names.index("treat_post")
        expected_vcov = np.asarray(d["vcov_hc2"]).reshape(d["vcov_hc2_shape"])
        expected_se_slope = float(np.sqrt(expected_vcov[treat_post_idx, treat_post_idx]))
        np.testing.assert_allclose(res.se, expected_se_slope, atol=1e-10)
        np.testing.assert_allclose(res.att, float(d["coef"][treat_post_idx]), atol=1e-10)

    def test_absorb_hc2_bm_clustered_matches_clubsandwich(self):
        """`absorb=` + `hc2_bm` + `cluster=unit` matches clubSandwich's CR2.

        Exercises the cluster-aware CR2 BM path that the R generator pins
        via `vcovCR(fit_did, cluster=d_did$unit, type="CR2")`. Without this
        test, the new auto-route would have an unverified clustered-CR2
        lane.
        """
        d = self._load_golden()
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "post": d["post"],
                "y": d["y"],
            }
        )
        res = DifferenceInDifferences(vcov_type="hc2_bm", cluster="unit").fit(
            data,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "period"],
            unit="unit",
        )
        coef_names = d["coef_names"]
        treat_post_idx = coef_names.index("treat_post")
        expected_vcov = np.asarray(d["vcov_cr2"]).reshape(d["vcov_cr2_shape"])
        expected_se_slope = float(np.sqrt(expected_vcov[treat_post_idx, treat_post_idx]))
        np.testing.assert_allclose(res.se, expected_se_slope, atol=1e-10)
        np.testing.assert_allclose(res.att, float(d["coef"][treat_post_idx]), atol=1e-10)

    def test_absorb_plus_fixed_effects_still_rejected_under_hc2_bm(self):
        """Mutual-exclusion of `absorb=` and `fixed_effects=` is preserved.

        R4 review caught that the auto-route initially merged the two
        arguments silently on the HC2/HC2-BM path. The public API has
        always treated this combination as invalid; the rejection must
        fire regardless of `vcov_type`.
        """
        d = self._load_golden()
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "post": d["post"],
                "y": d["y"],
            }
        )
        for vcov in ("hc1", "hc2", "hc2_bm"):
            with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
                DifferenceInDifferences(vcov_type=vcov).fit(
                    data,
                    outcome="y",
                    treatment="treated",
                    time="post",
                    absorb=["unit"],
                    fixed_effects=["period"],
                    unit="unit",
                )

    def test_absorb_hc2_bm_survey_multi_absorb_auto_routes(self):
        """Survey-weighted multi-absorb + HC2-BM should auto-route, not reject.

        The legacy guard at `estimators.py` rejects `survey_design` paired with
        `len(absorb) > 1` because single-pass demeaning is not the correct
        weighted FWL projection for multiple absorbed dimensions. But when the
        auto-route fires (hc2/hc2_bm), absorb is swapped for fixed_effects=
        BEFORE the survey guard sees it, so the demeaning rationale doesn't
        apply. R2 review caught the scope mismatch: REGISTRY said "SUPPORTED"
        but the survey guard fired first on weighted multi-absorb. This test
        pins the new placement.
        """
        from diff_diff import SurveyDesign

        d = self._load_golden()
        rng = np.random.default_rng(20260420)
        n = len(d["y"])
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "post": d["post"],
                "y": d["y"],
                "weight": rng.uniform(0.5, 2.0, size=n),
            }
        )
        sd = SurveyDesign(weights="weight", weight_type="aweight")
        # Multi-absorb (`unit` + `period`) + survey-weighted + hc2_bm: should
        # auto-route to fixed_effects= and succeed.
        res = DifferenceInDifferences(vcov_type="hc2_bm").fit(
            data,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "period"],
            unit="unit",
            survey_design=sd,
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_absorb_hc2_bm_df_sensitive_inference(self):
        """Bell-McCaffrey Satterthwaite DOF must propagate to `p_value` / `conf_int`.

        HC2-BM differs from HC2 only in the DOF used for inference (Satterthwaite
        ratio rather than n-k). If the auto-routed fit silently used n-k for the
        BM path, `p_value` and `conf_int` would be wrong even though `se` looked
        right. This test asserts that:

        (1) HC2 and HC2-BM give the same `se` on the same data (HC2 meat is shared);
        (2) HC2 and HC2-BM produce DIFFERENT `p_value` and `conf_int` because the
            critical-value DOF differs (HC2-BM uses Satterthwaite DOF < n-k, so
            t-critical is larger → wider CI, larger p-value).

        This is the df-sensitive regression guard the R1 reviewer asked for.
        """
        d = self._load_golden()
        res_hc2 = self._fit_absorb(d, "hc2")
        res_hc2_bm = self._fit_absorb(d, "hc2_bm")
        # Same point estimate.
        np.testing.assert_allclose(res_hc2.att, res_hc2_bm.att, atol=1e-12)
        # Same SE (the meat is the same; only the DOF differs for inference).
        np.testing.assert_allclose(res_hc2.se, res_hc2_bm.se, atol=1e-12)
        # DIFFERENT p_value and conf_int (DOF differs).
        assert res_hc2.p_value != res_hc2_bm.p_value, (
            "HC2 and HC2-BM should have different p_values "
            "because the BM Satterthwaite DOF differs from n-k. "
            "Same p_value indicates the DOF was not propagated to inference."
        )
        ci_hc2 = res_hc2.conf_int
        ci_hc2_bm = res_hc2_bm.conf_int
        # The BM CI should be WIDER than the HC2 CI (smaller DOF → larger
        # t-critical → wider interval).
        width_hc2 = float(ci_hc2[1] - ci_hc2[0])
        width_hc2_bm = float(ci_hc2_bm[1] - ci_hc2_bm[0])
        assert width_hc2_bm > width_hc2, (
            f"HC2-BM CI width ({width_hc2_bm:.6f}) should exceed "
            f"HC2 CI width ({width_hc2:.6f}) — BM Satterthwaite DOF is "
            "smaller than n-k, so the critical value is larger."
        )


class TestMPDAbsorbedFERParity:
    """R-parity for `MultiPeriodDiD(absorb=..., vcov_type in {hc2, hc2_bm})`.

    Mirrors `TestDiDAbsorbedFERParity`. The auto-route promotes `absorb=` to
    `fixed_effects=` internally; MPD's existing `fixed_effects=` code path
    builds the full-dummy design that R's `lm()` produces.

    Collinearity note: MPD's `treated` is a time-invariant ever-treated
    indicator, so it lies in the span of the intercept and the
    post-auto-route unit FE dummies (under `pd.get_dummies(drop_first=True)`
    the dropped reference unit is folded into the intercept; the exact
    alias relation depends on the omitted category and is NOT simply
    "the sum of treated-cohort unit dummies"). `solve_ols` resolves this
    by dropping one column from the collinear set under R-style
    rank-deficiency handling. In the shipped parity fixture the dropped
    column is a unit dummy from the never-treated cohort (`unit_25`) and
    the `treated` main effect remains finite, but the specific column
    dropped is pivot-order and dummy-coding dependent. Tests therefore
    pin parity on a per-period interaction (`treated:period_4`) which is
    identified independent of that choice, exposed as
    `result.period_effects[4]`.

    Time-FE skip note: when the routed (or explicit) `fixed_effects` list
    contains the `time` column, MPD silently skips emitting `<time>_<X>`
    dummies for that entry because the design already absorbs the time
    dimension via the non-reference period dummies. The
    `test_absorb_hc2_result_surface_invariants_multi_absorb` test pins
    that the resulting `coefficients` dict cardinality matches `vcov`
    rank and has no duplicate names.
    """

    def _load_golden(self):
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden JSON not present; run `Rscript "
                "benchmarks/R/generate_clubsandwich_golden.R` to generate."
            )
        with open(golden_path) as f:
            golden = json.load(f)
        if "mpd_absorbed_fe_did" not in golden:
            pytest.skip(
                "Golden JSON does not yet include `mpd_absorbed_fe_did` scenario; "
                "regenerate via the R script."
            )
        return golden["mpd_absorbed_fe_did"]

    def _make_data(self, d):
        return pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "y": d["y"],
            }
        )

    def _fit_absorb(self, d, vcov_type, absorb_cols=("unit",)):
        return MultiPeriodDiD(vcov_type=vcov_type).fit(
            self._make_data(d),
            outcome="y",
            treatment="treated",
            time="period",
            absorb=list(absorb_cols),
            reference_period=int(d["reference_period"]),
            unit="unit",
        )

    def _fit_fixed_effects(self, d, vcov_type, fe_cols=("unit",)):
        return MultiPeriodDiD(vcov_type=vcov_type).fit(
            self._make_data(d),
            outcome="y",
            treatment="treated",
            time="period",
            fixed_effects=list(fe_cols),
            reference_period=int(d["reference_period"]),
            unit="unit",
        )

    def test_absorb_hc2_matches_fixed_effects_dummies_single_absorb(self):
        """`absorb=["unit"]` + hc2 produces the same per-period SE as
        `fixed_effects=["unit"]` + hc2 (auto-route invariant)."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        res_a = self._fit_absorb(d, "hc2", absorb_cols=("unit",))
        res_f = self._fit_fixed_effects(d, "hc2", fe_cols=("unit",))
        pe_a = res_a.period_effects[target_period]
        pe_f = res_f.period_effects[target_period]
        np.testing.assert_allclose(pe_a.effect, pe_f.effect, atol=1e-12)
        np.testing.assert_allclose(pe_a.se, pe_f.se, atol=1e-12)

    def test_absorb_hc2_result_surface_invariants_multi_absorb(self):
        """Result-surface contract on the multi-absorb auto-route: the
        returned `MultiPeriodDiDResults.coefficients` dict must remain
        complete (one entry per fitted column), uniquely named, and aligned
        in cardinality with `vcov`.

        Regression for the duplicate-name collision the auto-route would
        otherwise expose: MPD already includes period dummies in its
        event-study design, so adding the `time` column as a fixed-effect
        dummy via `pd.get_dummies(prefix="period", drop_first=True)` would
        produce a second `period_<X>` block. Under `var_names`-keyed
        `coef_dict` construction, the duplicates silently collapse and the
        original event-study period coefficients are overwritten by the
        rank-deficient NaN drops on the redundant FE block. The fix at
        `estimators.py:fit()` (skip `fe == time` in the fixed_effects
        loop) eliminates the duplicate columns entirely. Test pins both
        the auto-route and the explicit `fixed_effects=` paths.
        """
        d = self._load_golden()
        target_period = int(d["target_period"])
        from collections import Counter

        for kwarg in ("absorb", "fixed_effects"):
            res = MultiPeriodDiD(vcov_type="hc2").fit(
                self._make_data(d),
                outcome="y",
                treatment="treated",
                time="period",
                **{kwarg: ["unit", "period"]},
                reference_period=int(d["reference_period"]),
                unit="unit",
            )
            assert res.vcov is not None
            assert len(res.coefficients) == res.vcov.shape[0], (
                f"[{kwarg}=] coefficients dict length ({len(res.coefficients)}) "
                f"must match vcov rank ({res.vcov.shape[0]}); duplicate var_names "
                "collapsed the dict and broke coefficients-vs-vcov alignment."
            )
            assert res.vcov.shape[0] == res.vcov.shape[1]
            dups = {k: v for k, v in Counter(res.coefficients.keys()).items() if v > 1}
            assert not dups, f"[{kwarg}=] duplicate names in coefficients: {dups}"
            # Sanity: the event-study period coefficients should be finite
            # (they are MPD's own non-reference period dummies, NOT the
            # redundant FE-block that was skipped).
            pe_name = f"period_{target_period}"
            assert pe_name in res.coefficients
            assert np.isfinite(res.coefficients[pe_name]), (
                f"[{kwarg}=] event-study {pe_name!r} should remain finite "
                "after the time-FE-skip fix (the duplicate FE-block that "
                "would have NaN'd it is no longer emitted)."
            )

    def test_absorb_hc2_matches_fixed_effects_dummies_multi_absorb(self):
        """`absorb=["unit","time"]` invariant: with both unit and time
        FE auto-routed, the period dummies collide with time-FE dummies;
        `solve_ols` handles rank deficiency, slope SE on the
        target per-period interaction stays well-defined and matches the
        explicit `fixed_effects=` path."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        # Use "period" as the time/FE column name to match the data column.
        res_a = MultiPeriodDiD(vcov_type="hc2").fit(
            self._make_data(d),
            outcome="y",
            treatment="treated",
            time="period",
            absorb=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
        )
        res_f = MultiPeriodDiD(vcov_type="hc2").fit(
            self._make_data(d),
            outcome="y",
            treatment="treated",
            time="period",
            fixed_effects=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
        )
        pe_a = res_a.period_effects[target_period]
        pe_f = res_f.period_effects[target_period]
        assert np.isfinite(pe_a.se)
        np.testing.assert_allclose(pe_a.effect, pe_f.effect, atol=1e-12)
        np.testing.assert_allclose(pe_a.se, pe_f.se, atol=1e-12)

    def test_absorb_hc2_bm_matches_fixed_effects_dummies(self):
        """`absorb=` + hc2_bm equals `fixed_effects=` + hc2_bm bit-for-bit
        on both per-period SE and inference (DOF transfers identically)."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        res_a = self._fit_absorb(d, "hc2_bm", absorb_cols=("unit",))
        res_f = self._fit_fixed_effects(d, "hc2_bm", fe_cols=("unit",))
        pe_a = res_a.period_effects[target_period]
        pe_f = res_f.period_effects[target_period]
        np.testing.assert_allclose(pe_a.effect, pe_f.effect, atol=1e-12)
        np.testing.assert_allclose(pe_a.se, pe_f.se, atol=1e-12)
        np.testing.assert_allclose(pe_a.p_value, pe_f.p_value, atol=1e-12)

    def test_absorb_hc2_matches_sandwich_vcovhc(self):
        """`absorb=` + hc2 matches `lm() + sandwich::vcovHC(type="HC2")`
        at 1e-10 on the target per-period interaction."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        target_coef = f"treated_period_{target_period}"
        coef_names = d["coef_names"]
        idx = coef_names.index(target_coef)
        expected_vcov = np.asarray(d["vcov_hc2"]).reshape(d["vcov_hc2_shape"])
        expected_se = float(np.sqrt(expected_vcov[idx, idx]))
        expected_coef = float(d["coef"][idx])

        res = self._fit_absorb(d, "hc2", absorb_cols=("unit",))
        pe = res.period_effects[target_period]
        np.testing.assert_allclose(pe.effect, expected_coef, atol=1e-10)
        np.testing.assert_allclose(pe.se, expected_se, atol=1e-10)

    def test_absorb_hc2_bm_matches_clubsandwich_singleton_cluster(self):
        """`absorb=` + hc2_bm matches `clubSandwich::vcovCR(cluster=1:n, type="CR2")`
        at 1e-10 on the target per-period interaction (singleton-cluster
        CR2 = one-way HC2-BM by PT2018 §3.3)."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        target_coef = f"treated_period_{target_period}"
        coef_names = d["coef_names"]
        idx = coef_names.index(target_coef)
        expected_vcov = np.asarray(d["vcov_hc2_bm"]).reshape(d["vcov_hc2_bm_shape"])
        expected_se = float(np.sqrt(expected_vcov[idx, idx]))

        res = self._fit_absorb(d, "hc2_bm", absorb_cols=("unit",))
        pe = res.period_effects[target_period]
        np.testing.assert_allclose(pe.se, expected_se, atol=1e-10)

    def test_absorb_plus_fixed_effects_still_rejected_under_hc2_bm(self):
        """Mutual-exclusion of `absorb=` and `fixed_effects=` is preserved
        on MPD across all vcov_types (the auto-route does NOT silently merge)."""
        d = self._load_golden()
        for vcov in ("hc1", "hc2", "hc2_bm"):
            with pytest.raises(ValueError, match="Cannot use both absorb and fixed_effects"):
                MultiPeriodDiD(vcov_type=vcov).fit(
                    self._make_data(d),
                    outcome="y",
                    treatment="treated",
                    time="period",
                    absorb=["unit"],
                    fixed_effects=["period"],
                    reference_period=int(d["reference_period"]),
                    unit="unit",
                )

    def test_absorb_hc2_bm_df_sensitive_inference(self):
        """HC2 vs HC2-BM produce the same SE on the target per-period
        interaction but different `p_value` / `conf_int` because the BM
        Satterthwaite DOF differs from n-k. Guards against an unwired DOF
        path (R1 review on PR #458 caught the analogous gap on DiD)."""
        d = self._load_golden()
        target_period = int(d["target_period"])
        res_hc2 = self._fit_absorb(d, "hc2", absorb_cols=("unit",))
        res_hc2_bm = self._fit_absorb(d, "hc2_bm", absorb_cols=("unit",))
        pe_hc2 = res_hc2.period_effects[target_period]
        pe_hc2_bm = res_hc2_bm.period_effects[target_period]
        np.testing.assert_allclose(pe_hc2.effect, pe_hc2_bm.effect, atol=1e-12)
        np.testing.assert_allclose(pe_hc2.se, pe_hc2_bm.se, atol=1e-12)
        assert pe_hc2.p_value != pe_hc2_bm.p_value, (
            "HC2 and HC2-BM should have different p_values "
            "because the BM Satterthwaite DOF differs from n-k."
        )
        width_hc2 = float(pe_hc2.conf_int[1] - pe_hc2.conf_int[0])
        width_hc2_bm = float(pe_hc2_bm.conf_int[1] - pe_hc2_bm.conf_int[0])
        assert width_hc2_bm > width_hc2, (
            f"HC2-BM CI width ({width_hc2_bm:.6f}) should exceed "
            f"HC2 CI width ({width_hc2:.6f}) — BM Satterthwaite DOF is "
            "smaller than n-k, so the critical value is larger."
        )

    def test_absorb_hc2_bm_avg_att_df_sensitive_inference(self):
        """The post-period-average ATT (`avg_att`, the MPD-specific
        contrast that does NOT have a DiD analogue) must also reflect
        the Satterthwaite DOF under HC2-BM: HC2 and HC2-BM share
        `avg_se` but differ in `avg_p_value` and `avg_conf_int`. This is
        the MPD-specific inference pin that the DiD test class cannot
        cover."""
        d = self._load_golden()
        res_hc2 = self._fit_absorb(d, "hc2", absorb_cols=("unit",))
        res_hc2_bm = self._fit_absorb(d, "hc2_bm", absorb_cols=("unit",))
        np.testing.assert_allclose(res_hc2.avg_att, res_hc2_bm.avg_att, atol=1e-12)
        np.testing.assert_allclose(res_hc2.avg_se, res_hc2_bm.avg_se, atol=1e-12)
        assert res_hc2.avg_p_value != res_hc2_bm.avg_p_value, (
            "HC2 and HC2-BM should produce different `avg_p_value` because "
            "the BM Satterthwaite DOF on the post-period-average contrast "
            "differs from n-k. Same p_value indicates the DOF was not "
            "propagated to the avg_att inference path."
        )
        width_hc2 = float(res_hc2.avg_conf_int[1] - res_hc2.avg_conf_int[0])
        width_hc2_bm = float(res_hc2_bm.avg_conf_int[1] - res_hc2_bm.avg_conf_int[0])
        assert width_hc2_bm > width_hc2, (
            f"HC2-BM avg_att CI width ({width_hc2_bm:.6f}) should exceed "
            f"HC2 avg_att CI width ({width_hc2:.6f}) — BM Satterthwaite "
            "DOF is smaller than n-k, so the critical value is larger."
        )

    def test_absorb_hc2_bm_survey_multi_absorb_auto_routes(self):
        """Survey-weighted multi-absorb + HC2-BM should auto-route, not reject.

        Mirrors the DiD-class test of the same name: the legacy guard at
        `estimators.py:1505-1512` rejects `survey_design + len(absorb) > 1`
        because single-pass demeaning is not the correct weighted FWL
        projection for multiple absorbed dimensions. But when the auto-route
        fires (hc2/hc2_bm), absorb is swapped for fixed_effects= BEFORE the
        survey guard sees it, so the demeaning rationale doesn't apply. The
        auto-route placement is precisely tuned for this case; this test
        pins it on the MPD path."""
        from diff_diff import SurveyDesign

        d = self._load_golden()
        rng = np.random.default_rng(20260420)
        n = len(d["y"])
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "y": d["y"],
                "weight": rng.uniform(0.5, 2.0, size=n),
            }
        )
        sd = SurveyDesign(weights="weight", weight_type="aweight")
        # Multi-absorb (unit + period) + survey + hc2_bm: auto-route fires
        # and the multi-absorb-survey guard is bypassed cleanly.
        res = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data,
            outcome="y",
            treatment="treated",
            time="period",
            absorb=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
            survey_design=sd,
        )
        # Parity invariant: the explicit fixed_effects= path on the same
        # data must produce the same per-period SE.
        res_fe = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data,
            outcome="y",
            treatment="treated",
            time="period",
            fixed_effects=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
            survey_design=sd,
        )
        target_period = int(d["target_period"])
        pe_a = res.period_effects[target_period]
        pe_f = res_fe.period_effects[target_period]
        assert np.isfinite(pe_a.effect)
        assert np.isfinite(pe_a.se)
        np.testing.assert_allclose(pe_a.effect, pe_f.effect, atol=1e-12)
        np.testing.assert_allclose(pe_a.se, pe_f.se, atol=1e-12)
        np.testing.assert_allclose(res.avg_att, res_fe.avg_att, atol=1e-12)
        np.testing.assert_allclose(res.avg_se, res_fe.avg_se, atol=1e-12)

    def test_absorb_hc2_bm_replicate_weights_auto_routes(self):
        """Replicate-weight survey design + absorb + HC2-BM auto-routes
        through `compute_replicate_vcov` on the full-dummy design.

        The CHANGELOG/REGISTRY claim that, under the auto-route, the
        survey-replicate absorb-refit branch at `estimators.py:1693` is
        short-circuited (no per-replicate refit needed because the
        full-dummy design does not depend on replicate weights — the
        standard `compute_replicate_vcov` path applies directly). This
        test pins the parity invariant on a JK1 fixture: `absorb=`
        + replicate weights must produce the same `period_effects`
        and `avg_att` SEs as the explicit `fixed_effects=` form."""
        from diff_diff import SurveyDesign

        d = self._load_golden()
        rng = np.random.default_rng(20260420)
        n = len(d["y"])
        data = pd.DataFrame(
            {
                "unit": d["unit"],
                "period": d["period"],
                "treated": d["treated"],
                "y": d["y"],
                "weight": rng.uniform(0.5, 2.0, size=n),
            }
        )
        # 10 JK1 jackknife replicate-weight columns; weights drawn from
        # {0.5, 1.5} match the BRR pattern of the existing replicate
        # tests in this file.
        rep_cols = [f"rep{r}" for r in range(10)]
        for col in rep_cols:
            data[col] = rng.choice([0.5, 1.5], size=n)
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="JK1",
            replicate_scale=1.0,
        )
        res_absorb = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data,
            outcome="y",
            treatment="treated",
            time="period",
            absorb=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
            survey_design=sd,
        )
        res_fe = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data,
            outcome="y",
            treatment="treated",
            time="period",
            fixed_effects=["unit", "period"],
            reference_period=int(d["reference_period"]),
            unit="unit",
            survey_design=sd,
        )
        target_period = int(d["target_period"])
        pe_a = res_absorb.period_effects[target_period]
        pe_f = res_fe.period_effects[target_period]
        assert np.isfinite(pe_a.effect)
        assert np.isfinite(pe_a.se)
        # The auto-route short-circuits the absorb-refit branch and routes
        # both calls through the standard replicate-vcov path; SE parity
        # is therefore exact (bit-identical, not just to within 1e-10).
        np.testing.assert_allclose(pe_a.effect, pe_f.effect, atol=1e-12)
        np.testing.assert_allclose(pe_a.se, pe_f.se, atol=1e-12)
        np.testing.assert_allclose(res_absorb.avg_att, res_fe.avg_att, atol=1e-12)
        np.testing.assert_allclose(res_absorb.avg_se, res_fe.avg_se, atol=1e-12)


class TestTWFECovariateNameCollision:
    """PR3: TwoWayFixedEffects covariate-name collision guard.

    On the full-dummy HC2/HC2-BM path covariates are zipped into the coefficient
    dict alongside "const"/"ATT" and the unit/time dummies, so a colliding
    covariate would silently overwrite that coefficient. The within-transform
    (default HC1) path exposes only {"ATT": att}, but the covariate is still in
    X and a covariate named "_treatment_post" would clobber the internal
    interaction column, so the guard fires on ALL paths. Lives here next to the
    full-dummy ``len(coefficients) == vcov.shape[0]`` invariant.
    """

    @staticmethod
    def _panel_with(covname: str) -> pd.DataFrame:
        df = _make_did_panel(n_units=30, seed=20260420)
        # unit 0..29 / time {0,1}: get_dummies(drop_first=True) keeps
        # "_fe_unit_1".."_fe_unit_29" and "_fe_time_1".
        df[covname] = np.random.default_rng(99).normal(size=len(df))
        return df

    @pytest.mark.parametrize("vcov_type", ["hc1", "hc2"])
    @pytest.mark.parametrize(
        "name", ["const", "ATT", "_treatment_post", "_fe_unit_1", "_fe_time_1"]
    )
    def test_collision_raises_on_all_paths(self, vcov_type, name):
        df = self._panel_with(name)
        with pytest.raises(ValueError, match="collide"):
            TwoWayFixedEffects(vcov_type=vcov_type).fit(
                df,
                outcome="y",
                treatment="treated",
                time="time",
                unit="unit",
                covariates=[name],
            )

    def test_hc2_full_dummy_noncolliding_preserves_coefs(self):
        df = self._panel_with("x1")
        r = TwoWayFixedEffects(vcov_type="hc2").fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            covariates=["x1"],
        )
        ck = r.coefficients
        assert "ATT" in ck and "x1" in ck
        # No key overwritten: dict size matches the full-dummy vcov rank.
        assert len(ck) == r.vcov.shape[0]

    def test_within_transform_noncolliding_returns_att_only(self):
        df = self._panel_with("x1")
        r = TwoWayFixedEffects().fit(  # default hc1 -> within-transform
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            covariates=["x1"],
        )
        # The within-transform path exposes only the ATT coefficient by design;
        # the covariate is NOT a dict key there (so there is no overwrite surface).
        assert set(r.coefficients.keys()) == {"ATT"}

    def test_within_path_does_not_materialize_fe_dummies(self, monkeypatch):
        # Regression: the within-transform (default hc1) path must NOT build full
        # unit/time dummy matrices merely to reserve collision names — that would
        # defeat its high-cardinality scaling contract. Reserved names come from
        # fe_dummy_names (category levels only), so pd.get_dummies must never be
        # called on this path.
        df = self._panel_with("x1")

        def _boom(*args, **kwargs):
            raise AssertionError("pd.get_dummies must not be called on the within-transform path")

        monkeypatch.setattr(pd, "get_dummies", _boom)
        r = TwoWayFixedEffects().fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            covariates=["x1"],
        )
        assert set(r.coefficients.keys()) == {"ATT"}


class TestMPDClusterHC2BMSharedPrecompute:
    """`MultiPeriodDiD(cluster=..., vcov_type='hc2_bm')` builds the CR2
    Bell-McCaffrey precomputes ONCE, not twice.

    Mechanism guard for the perf dedup: vcov and the per-coefficient +
    post-period-average contrast DOF now come from a single
    `_compute_cr2_bm_vcov_and_dof` call, so the expensive per-cluster
    adjustment matrices (`_cr2_adjustment_matrix`) are built exactly once per
    cluster. Before the dedup, solve_ols's vcov path and the separate
    contrast-DOF call each built them, i.e. `2 * G`.

    (Absolute SE/DOF values are pinned independently by the R/clubSandwich
    goldens in `test_multi_period_cluster_hc2_bm_avg_att_uses_clubsandwich_dof`
    and `test_linalg_hc2_bm.py`; these tests guard the *new invariants* the
    refactor introduces.)
    """

    @staticmethod
    def _balanced_panel():
        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in range(3):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * (t >= 1)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        return pd.DataFrame(rows)

    @staticmethod
    def _unbalanced_panel():
        rng = np.random.default_rng(7)
        rows = []
        for i in range(24):
            treated = int(i >= 12)
            periods = [0, 1, 2, 3] if (i % 3 != 0) else [0, 2, 3]
            for t in periods:
                y = rng.normal(0.0, 1.0) + 0.4 * treated * (t >= 2)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("which", ["balanced", "unbalanced"])
    def test_cr2_precompute_built_once(self, which, monkeypatch):
        """`_cr2_adjustment_matrix` is called exactly `G` times (one precompute
        build), not `2 * G`, on the cluster+hc2_bm path."""
        import diff_diff.linalg as L

        data = self._balanced_panel() if which == "balanced" else self._unbalanced_panel()
        n_clusters = data["unit"].nunique()

        orig = L._cr2_adjustment_matrix
        calls = {"n": 0}

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(L, "_cr2_adjustment_matrix", _counting)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
                data, outcome="y", treatment="treated", time="time", unit="unit"
            )

        # One build: exactly one adjustment matrix per cluster (was 2 * G when
        # solve_ols's vcov and the contrast-DOF call each built the precomputes).
        assert calls["n"] == n_clusters, (
            f"Expected {n_clusters} _cr2_adjustment_matrix calls (one CR2 "
            f"precompute build), got {calls['n']} — the precompute is being "
            "built more than once."
        )
        # Inference is still finite (the dedup did not break the path).
        assert np.isfinite(res.avg_att) and np.isfinite(res.avg_se)
        assert np.isfinite(res.avg_p_value)

    def test_fit_is_reproducible(self):
        """Two independent fits (and a repeat fit of the same estimator) give
        identical avg-ATT and per-period inference — determinism + the
        fit-does-not-mutate-config contract on the bypass path."""
        data = self._balanced_panel()

        est = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = est.fit(data, outcome="y", treatment="treated", time="time", unit="unit")
            # Repeat fit of the SAME object (config must be unmutated).
            r2 = est.fit(data, outcome="y", treatment="treated", time="time", unit="unit")
            # Fresh object, same config.
            r3 = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit").fit(
                data, outcome="y", treatment="treated", time="time", unit="unit"
            )

        for r in (r2, r3):
            assert r.avg_att == r1.avg_att
            assert r.avg_se == r1.avg_se
            assert r.avg_t_stat == r1.avg_t_stat
            assert r.avg_p_value == r1.avg_p_value
            assert set(r.period_effects) == set(r1.period_effects)
            for p in r1.period_effects:
                assert r.period_effects[p].effect == r1.period_effects[p].effect
                assert r.period_effects[p].se == r1.period_effects[p].se


def _make_absorb_panel(seed: int = 11, n_units: int = 50, n_periods: int = 6) -> pd.DataFrame:
    """Heteroskedastic multi-period panel with a unit-level covariate.

    Heteroskedasticity makes hc1 differ from classical; absorbed unit + time FE
    give a nonzero ``df_adjustment`` so the full-K vcov rescale is exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated = int(u >= n_units // 2)
        ui = float(rng.normal())
        x = float(rng.normal())
        for t in range(n_periods):
            post = int(t >= n_periods // 2)
            sd = 0.5 + 1.2 * treated + 0.25 * t
            y = (
                ui
                + 0.4 * x
                + 0.5 * t
                + (0.8 if treated and post else 0.0)
                + float(rng.normal(0, sd))
            )
            rows.append({"unit": u, "time": t, "post": post, "treated": treated, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestAbsorbedFEFullKParity:
    """Absorbed-FE variance scale matches the full-dummy (fixest full-K) SE.

    Within-transform (``absorb=``) fits previously scaled the *non-clustered*
    classical / hc1 vcov by ``k_visible`` (absorbed FE excluded), sitting ~6.5%
    below ``fixest`` feols(vcov="iid"/"hetero") even though the reported t-df
    already used ``K_full``. The absorbed-FE vcov rescale
    (``linalg._absorbed_fe_vcov_scale``) aligns the SE's k with ``K_full`` so the
    absorb path equals the explicit full-dummy path -- the in-repo oracle,
    verified equal to ``fixest`` to 6 digits (iid 0.202158 / hetero 0.432290 on
    the audit smoke panels). Clustered SEs (fixest nested-FE convention) and
    hc2 / hc2_bm (leverage / Satterthwaite DOF) are intentionally unaffected.
    """

    @pytest.mark.parametrize("vcov", ["classical", "hc1"])
    def test_did_absorb_matches_full_dummy_oracle(self, vcov):
        df = _make_absorb_panel()
        ab = DifferenceInDifferences(vcov_type=vcov).fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "time"],
            covariates=["x"],
        )
        fe = DifferenceInDifferences(vcov_type=vcov).fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            fixed_effects=["unit", "time"],
            covariates=["x"],
        )
        assert np.isfinite(ab.se)
        np.testing.assert_allclose(ab.att, fe.att, rtol=1e-9)
        np.testing.assert_allclose(ab.se, fe.se, rtol=1e-9)

    @pytest.mark.parametrize("vcov", ["classical", "hc1"])
    def test_mpd_absorb_matches_full_dummy_oracle(self, vcov):
        df = _make_absorb_panel()
        ab = MultiPeriodDiD(vcov_type=vcov).fit(
            df, outcome="y", treatment="treated", time="time", absorb=["unit"], covariates=["x"]
        )
        fe = MultiPeriodDiD(vcov_type=vcov).fit(
            df,
            outcome="y",
            treatment="treated",
            time="time",
            fixed_effects=["unit"],
            covariates=["x"],
        )
        assert np.isfinite(ab.avg_se)
        np.testing.assert_allclose(ab.avg_att, fe.avg_att, rtol=1e-9)
        np.testing.assert_allclose(ab.avg_se, fe.avg_se, rtol=1e-9)

    def test_twfe_classical_matches_full_dummy_oracle(self):
        """TWFE always within-transforms; the equivalent explicit full-dummy
        DiD is the oracle. After the fix TWFE(classical) SE == that oracle."""
        df = _make_absorb_panel()
        tw = TwoWayFixedEffects(vcov_type="classical").fit(
            df, outcome="y", treatment="treated", time="post", unit="unit"
        )
        fe = DifferenceInDifferences(vcov_type="classical").fit(
            df, outcome="y", treatment="treated", time="post", fixed_effects=["unit", "post"]
        )
        np.testing.assert_allclose(tw.att, fe.att, rtol=1e-9)
        np.testing.assert_allclose(tw.se, fe.se, rtol=1e-9)

    def test_absorb_cluster_not_rescaled(self):
        """The absorbed-FE full-K rescale must NOT touch clustered SEs.

        The rescale is gated on ``cluster_ids is None``, so the cluster-absorb
        SE stays at ``k_visible`` and differs from the full-dummy path here.
        (Full fixest cluster parity is a *separate*, out-of-scope matter: fixest
        counts non-nested absorbed FE in the CR1 denominator, so for
        ``absorb=["unit","time"], cluster="unit"`` the non-nested time FE would
        need counting -- a documented pre-existing limitation, see REGISTRY;
        this test only pins that D4 does not rescale the cluster path.)
        """
        df = _make_absorb_panel()
        ab = DifferenceInDifferences(vcov_type="hc1", cluster="unit").fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "time"],
            covariates=["x"],
        )
        fe = DifferenceInDifferences(vcov_type="hc1", cluster="unit").fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            fixed_effects=["unit", "time"],
            covariates=["x"],
        )
        assert np.isfinite(ab.se)
        assert not np.isclose(ab.se, fe.se, rtol=1e-6), (
            "cluster-absorb SE must keep k_visible (nested-FE), not be rescaled to "
            "the full-dummy K_full value"
        )

    def test_absorb_hc2_bm_not_rescaled(self):
        """hc2_bm auto-routes absorb -> full-dummy and uses Satterthwaite DOF;
        the classical/hc1 rescale must not touch it (absorb == full-dummy)."""
        df = _make_absorb_panel()
        ab = DifferenceInDifferences(vcov_type="hc2_bm").fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            absorb=["unit", "time"],
            covariates=["x"],
        )
        fe = DifferenceInDifferences(vcov_type="hc2_bm").fit(
            df,
            outcome="y",
            treatment="treated",
            time="post",
            fixed_effects=["unit", "time"],
            covariates=["x"],
        )
        np.testing.assert_allclose(ab.se, fe.se, rtol=1e-9)

    def test_sunab_hc1_autoclusters_so_gate_skips(self):
        """SunAbraham hc1 auto-clusters at unit, so the D4 rescale (gated on
        ``cluster_ids is None``) never fires -> its documented hc1 deviation is
        preserved. Invariant: default hc1 == explicit unit cluster."""
        rng = np.random.default_rng(3)
        rows = []
        for u in range(40):
            cohort = 0 if u < 20 else (3 if u < 30 else 5)
            ui = float(rng.normal())
            for t in range(6):
                treated_now = cohort != 0 and t >= cohort
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": cohort,
                        "y": ui
                        + 0.5 * t
                        + (1.2 if treated_now else 0.0)
                        + float(rng.normal(0, 0.7)),
                    }
                )
        df = pd.DataFrame(rows)
        r_def = SunAbraham(vcov_type="hc1").fit(
            df, outcome="y", first_treat="first_treat", time="time", unit="unit"
        )
        r_cl = SunAbraham(vcov_type="hc1", cluster="unit").fit(
            df, outcome="y", first_treat="first_treat", time="time", unit="unit"
        )
        np.testing.assert_allclose(r_def.overall_se, r_cl.overall_se, rtol=1e-12)

    def test_absorbed_fe_vcov_scale_fail_closed(self):
        """The rescale helper is fail-closed on non-positive full-K residual dof."""
        # Normal: (n-k)/(n-k-adj) = 7/5.
        assert _absorbed_fe_vcov_scale(10, 3, 2) == pytest.approx(1.4)
        # No absorbed FE -> no-op scale.
        assert _absorbed_fe_vcov_scale(10, 3, 0) == 1.0
        # Fail-closed: full-K residual dof <= 0 -> NaN, so the caller voids the
        # vcov to NaN inference (rather than leaving a misleading k_visible SE).
        assert np.isnan(_absorbed_fe_vcov_scale(10, 3, 8))  # full-K denom = -1
        assert np.isnan(_absorbed_fe_vcov_scale(10, 2, 8))  # full-K denom = 0
        assert np.isnan(_absorbed_fe_vcov_scale(10, 12, 2))  # visible denom < 0

    def test_absorb_saturated_full_k_df_le_zero_nan_inference(self):
        """A saturated within-transform design (full-K residual dof <= 0) yields
        NaN SE/inference end-to-end, not a misleading finite k_visible SE.

        2 units x 5 periods with ``absorb=["unit"]`` drives K_full past n, so the
        classical full-K variance is undefined -> the rescale helper returns NaN
        and the vcov is voided (fail-closed), even though the point estimate is
        still computed.
        """
        rng = np.random.default_rng(0)
        rows = []
        for u in range(2):
            tr = int(u >= 1)
            for t in range(5):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": tr,
                        "y": float(rng.normal()) + 0.5 * tr * (t >= 1),
                    }
                )
        df = pd.DataFrame(rows)
        r = MultiPeriodDiD(vcov_type="classical").fit(
            df, outcome="y", treatment="treated", time="time", absorb=["unit"]
        )
        assert np.isnan(r.avg_se), "saturated full-K design must yield NaN SE (fail-closed)"
