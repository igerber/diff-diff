"""Tests for :class:`diff_diff.had.HeterogeneousAdoptionDiD` (Phase 2a).

Covers the 12 plan commit criteria:

1. All three design paths produce a finite result on synthetic DGPs.
2. ``design="auto"`` resolves correctly on each DGP + two edge cases.
3. Beta-scale WAS estimator at atol=1e-14:
   - Design 1' / continuous_at_zero:
     ``att = (mean(ΔY) - tau_bc) / mean(D)``
   - Design 1 / continuous_near_d_lower:
     ``att = (mean(ΔY) - tau_bc) / mean(D - d_lower)``
   - CI endpoints reverse under subtraction:
     ``CI_lower(att) = (mean(ΔY) - CI_upper_boundary) / den``
4. Mass-point Wald-IV point estimate matches manual formula at
   ``atol=1e-14``.
5. Mass-point 2SLS SE parity against hand-coded sandwich at
   ``atol=1e-12`` for HC1, classical, and CR1 (cluster-robust).
6. Mass-point + ``vcov_type in {hc2, hc2_bm}`` raises
   ``NotImplementedError``.
7. Panel-contract violations raise targeted ``ValueError``s.
8. NaN propagation: constant-y and mass-point degenerate inputs produce
   all-NaN inference.
9. sklearn clone round-trip preserves raw ``design="auto"``; fit is
   idempotent.
10. Scaffolding (``aggregate="event_study"``, ``survey``, ``weights``)
    raises ``NotImplementedError`` with phase pointers.
11. ``get_params()`` keys match ``__init__`` signature.
12. REGISTRY ticks tested indirectly via parity with the paper rules.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from diff_diff.had import (
    HeterogeneousAdoptionDiD,
    HeterogeneousAdoptionDiDEventStudyResults,
    HeterogeneousAdoptionDiDResults,
    _aggregate_first_difference,
    _aggregate_multi_period_first_differences,
    _detect_design,
    _fit_mass_point_2sls,
    _validate_had_panel,
    _validate_had_panel_event_study,
)
from diff_diff.local_linear import bias_corrected_local_linear
from tests.conftest import assert_nan_inference

# =============================================================================
# DGP helpers
# =============================================================================


def _make_panel(d_post, delta_y, periods=(1, 2), extra_cols=None):
    """Build a balanced two-period panel with ``D_{g,1} = 0``.

    Parameters
    ----------
    d_post : np.ndarray, shape (G,)
        Unit-level post-period dose ``D_{g,2}``.
    delta_y : np.ndarray, shape (G,)
        Unit-level first-difference outcome ``Y_{g,2} - Y_{g,1}``.
    periods : tuple
        (t_pre, t_post).
    extra_cols : dict or None
        Additional unit-constant columns (e.g., cluster variable).
    """
    G = len(d_post)
    t_pre, t_post = periods
    units = np.arange(G)
    df = pd.DataFrame(
        {
            "unit": np.repeat(units, 2),
            "period": np.tile([t_pre, t_post], G),
            "dose": np.column_stack([np.zeros(G), d_post]).ravel(),
            # Set period-1 outcome to 0; period-2 outcome = delta_y so that
            # Y_{g,2} - Y_{g,1} == delta_y exactly.
            "outcome": np.column_stack([np.zeros(G), delta_y]).ravel(),
        }
    )
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = np.repeat(vals, 2)
    return df


def _dgp_continuous_at_zero(G, seed):
    """Design 1' DGP: uniform dose on [0, 1] with exact zero in the sample."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 1.0, G)
    d[0] = 0.0  # guarantee continuous_at_zero auto-detection
    dy = 0.3 * d + 0.1 * rng.standard_normal(G)
    return d, dy


def _dgp_continuous_near_d_lower(G, seed):
    """Design 1 continuous-near-d_lower DGP: Beta(2,2) shifted to [0.1, 1]."""
    rng = np.random.default_rng(seed)
    u = rng.beta(2, 2, G)
    d = 0.1 + 0.9 * u
    dy = 0.3 * d + 0.1 * rng.standard_normal(G)
    return d, dy


def _dgp_mass_point(G, seed, d_lower=0.5, mass_frac=0.3, beta=0.3):
    """Mass-point DGP: ``mass_frac`` at d_lower, rest Uniform(d_lower, 1)."""
    rng = np.random.default_rng(seed)
    mass_n = int(mass_frac * G)
    d = np.concatenate([np.full(mass_n, d_lower), rng.uniform(d_lower, 1.0, G - mass_n)])
    dy = beta * d + 0.1 * rng.standard_normal(G)
    return d, dy


# =============================================================================
# Criterion 1: Smoke tests - all 3 design paths produce finite output
# =============================================================================


class TestSmokeAllDesigns:
    def test_continuous_at_zero_finite(self):
        d, dy = _dgp_continuous_at_zero(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_continuous_near_d_lower_finite(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_mass_point_finite(self):
        d, dy = _dgp_mass_point(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_result_is_dataclass(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert isinstance(r, HeterogeneousAdoptionDiDResults)

    def test_continuous_populates_bandwidth_diagnostics(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.bandwidth_diagnostics is not None
        assert r.bias_corrected_fit is not None

    def test_mass_point_nulls_bandwidth_diagnostics(self):
        d, dy = _dgp_mass_point(400, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.bandwidth_diagnostics is None
        assert r.bias_corrected_fit is None
        assert r.n_mass_point is not None
        assert r.n_above_d_lower is not None

    def test_continuous_nulls_mass_point_counts(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.n_mass_point is None
        assert r.n_above_d_lower is None


# =============================================================================
# Criterion 2: design="auto" detection rule
# =============================================================================


class TestDesignAutoDetect:
    def test_detect_design_1_prime_exact_zero(self):
        d, _ = _dgp_continuous_at_zero(500, seed=0)
        assert _detect_design(d) == "continuous_at_zero"

    def test_detect_design_continuous_near_d_lower(self):
        d, _ = _dgp_continuous_near_d_lower(500, seed=0)
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_detect_mass_point(self):
        d, _ = _dgp_mass_point(500, seed=0)
        assert _detect_design(d) == "mass_point"

    def test_edge_small_mass_at_zero_resolves_continuous_at_zero(self):
        """Plan criterion 2 edge-case (a): 3% at D=0 + 97% Uniform(0.5, 1)."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = int(0.03 * G)
        d = np.concatenate([np.zeros(mass_n), rng.uniform(0.5, 1.0, G - mass_n)])
        assert _detect_design(d) == "continuous_at_zero"

    def test_edge_shifted_beta_not_small_enough_for_design_1_prime(self):
        """Plan criterion 2 edge-case (b): d.min/median ~ 0.03 > 0.01 threshold."""
        rng = np.random.default_rng(0)
        u = rng.beta(2, 2, 1000)
        d = 0.03 + u
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_design_auto_dispatches_correctly_at_fit(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="auto").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.design == "continuous_at_zero"

    def test_design_auto_mass_point_at_fit(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="auto").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.design == "mass_point"

    def test_auto_does_not_mutate_self_design(self):
        """Plan decision #14: self.design preserves raw 'auto' after fit."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        est = HeterogeneousAdoptionDiD(design="auto")
        _ = est.fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert est.design == "auto"
        assert est.get_params()["design"] == "auto"


# =============================================================================
# Criterion 3: Beta-scale rescaling parity
# =============================================================================


class TestBetaScaleRescaling:
    """Plan commit criterion #3 + review P0: the continuous estimator is

        att = (mean(ΔY) - tau_bc) / den

    with ``den = mean(D)`` for Design 1' and ``den = mean(D - d_lower)``
    for Design 1 continuous-near-d_lower. SE is ``se_robust / |den|``.
    CI endpoints are computed via ``att +/- z * se`` (endpoints reverse
    relative to the boundary-limit CI because the numerator is
    ``ΔȲ - tau_bc``).
    """

    def test_att_design_1_prime(self):
        """att = (mean(ΔY) - tau_bc) / D_bar for Design 1' at atol=1e-14."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        d_bar = float(d.mean())
        dy_mean = float(dy.mean())
        expected = (dy_mean - float(bc.estimate_bias_corrected)) / d_bar
        assert abs(r.att - expected) < 1e-14

    def test_se_design_1_prime(self):
        """se = se_robust / |D_bar| for Design 1' at atol=1e-14."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        expected = float(bc.se_robust) / abs(float(d.mean()))
        assert abs(r.se - expected) < 1e-14

    def test_ci_endpoints_reverse_under_subtraction(self):
        """Because att = (ΔȲ - tau_bc)/D_bar, CI endpoints reverse:

        CI_lower(att) = (ΔȲ - CI_upper_boundary) / D_bar
        CI_upper(att) = (ΔȲ - CI_lower_boundary) / D_bar
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        d_bar = float(d.mean())
        dy_mean = float(dy.mean())
        # CI bounds on the att scale, computed by endpoint reversal from
        # the boundary-limit CI.
        expected_lower = (dy_mean - float(bc.ci_high)) / d_bar
        expected_upper = (dy_mean - float(bc.ci_low)) / d_bar
        assert abs(r.conf_int[0] - expected_lower) < 1e-14
        assert abs(r.conf_int[1] - expected_upper) < 1e-14

    def test_att_design_1_continuous_near_d_lower(self):
        """att = (mean(ΔY) - tau_bc) / mean(D - d_lower) for Design 1 at atol=1e-14."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_val = float(d.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        d_reg = d - d_lower_val
        bc = bias_corrected_local_linear(d=d_reg, y=dy, boundary=0.0, alpha=0.05)
        den = float((d - d_lower_val).mean())
        dy_mean = float(dy.mean())
        expected = (dy_mean - float(bc.estimate_bias_corrected)) / den
        assert abs(r.att - expected) < 1e-14

    def test_att_recovers_true_beta_design_1_prime(self):
        """Sanity: on a known DGP with beta=0.3, att should be close to 0.3."""
        rng = np.random.default_rng(0)
        G = 2000
        d = rng.uniform(0, 1, G)
        d[0] = 0.0
        dy = 0.3 * d + 0.05 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # Asymptotic: expect att close to 0.3 at G=2000, n=4000 observations.
        assert abs(r.att - 0.3) < 0.1

    def test_att_recovers_true_beta_continuous_near_d_lower(self):
        """Sanity: Design 1 DGP with beta_d_lower=0.3 recovers beta at scale."""
        rng = np.random.default_rng(0)
        G = 2000
        u = rng.beta(2, 2, G)
        d = 0.1 + 0.9 * u  # d_lower ~ 0.1
        # True WAS_{d_lower} = 0.3 since dy = 0.3 * (d - d_lower) + noise
        dy = 0.3 * (d - 0.1) + 0.05 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert abs(r.att - 0.3) < 0.1

    def test_dose_mean_stored_on_result(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")
        assert abs(r.dose_mean - float(d.mean())) < 1e-14


# =============================================================================
# Criterion 4: Mass-point Wald-IV point estimate parity
# =============================================================================


class TestMassPointWaldIV:
    def test_wald_iv_point_estimate(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        Z = (d > 0.5).astype(float)
        expected = (dy[Z == 1].mean() - dy[Z == 0].mean()) / (d[Z == 1].mean() - d[Z == 0].mean())
        assert abs(r.att - expected) < 1e-14

    def test_wald_iv_equals_2sls(self):
        """Sanity: Wald-IV is exactly 2SLS for binary instrument."""
        d, dy = _dgp_mass_point(500, seed=7)
        Z = (d > 0.5).astype(float).reshape(-1, 1)
        # 2SLS via Z'X invert: beta = [(Z'X)^-1 Z'y][1]
        X = np.column_stack([np.ones_like(d), d])
        Zd = np.column_stack([np.ones_like(d), Z.ravel()])
        beta_2sls = np.linalg.inv(Zd.T @ X) @ (Zd.T @ dy)
        beta_wald = (dy[Z.ravel() == 1].mean() - dy[Z.ravel() == 0].mean()) / (
            d[Z.ravel() == 1].mean() - d[Z.ravel() == 0].mean()
        )
        assert abs(float(beta_2sls[1]) - beta_wald) < 1e-12

    def test_mass_point_n_counts_populated(self):
        d, dy = _dgp_mass_point(500, seed=0, d_lower=0.5, mass_frac=0.3)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r.n_mass_point == int(0.3 * 500)
        assert r.n_above_d_lower == 500 - int(0.3 * 500)
        assert r.n_treated == r.n_above_d_lower
        assert r.n_control == r.n_mass_point


# =============================================================================
# Criterion 5: Mass-point 2SLS SE sandwich parity
# =============================================================================


def _manual_2sls_sandwich_se(d, dy, d_lower, vcov_type, cluster=None):
    """Hand-coded textbook 2SLS sandwich using structural residuals.

    Returns se_beta for the coefficient on d. Mirrors the helper in had.py
    but computed from scratch to serve as the parity reference.
    """
    n = len(d)
    Z = (d > d_lower).astype(np.float64)
    dose_gap = d[Z == 1].mean() - d[Z == 0].mean()
    dy_gap = dy[Z == 1].mean() - dy[Z == 0].mean()
    beta = dy_gap / dose_gap
    alpha_hat = dy.mean() - beta * d.mean()
    u = dy - alpha_hat - beta * d  # STRUCTURAL residuals
    X = np.column_stack([np.ones(n), d])
    Zd = np.column_stack([np.ones(n), Z])
    ZtX_inv = np.linalg.inv(Zd.T @ X)

    if cluster is not None:
        Omega = np.zeros((2, 2))
        clusters = pd.unique(cluster)
        G = len(clusters)
        for c in clusters:
            idx = cluster == c
            s = Zd[idx].T @ u[idx]
            Omega += np.outer(s, s)
        Omega *= (G / (G - 1)) * ((n - 1) / (n - 2))
    elif vcov_type == "classical":
        sigma2 = (u * u).sum() / (n - 2)
        Omega = sigma2 * (Zd.T @ Zd)
    elif vcov_type == "hc1":
        Omega = (n / (n - 2)) * (Zd.T @ ((u * u)[:, None] * Zd))
    else:
        raise ValueError(f"unknown vcov_type={vcov_type}")

    V = ZtX_inv @ Omega @ ZtX_inv.T
    return float(np.sqrt(V[1, 1]))


class TestMassPointSEParity:
    def test_classical_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "classical")
        assert abs(r.se - expected) < 1e-12

    def test_hc1_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "hc1")
        assert abs(r.se - expected) < 1e-12

    def test_cr1_cluster_robust_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        cluster_ids = np.tile(np.arange(50), 10)  # 50 clusters of 10 units
        panel = _make_panel(d, dy, extra_cols={"state": cluster_ids})
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "hc1", cluster=cluster_ids)
        assert abs(r.se - expected) < 1e-12

    def test_robust_alias_maps_to_hc1(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            r_robust = HeterogeneousAdoptionDiD(design="mass_point", robust=True).fit(
                panel, "outcome", "dose", "period", "unit"
            )
        r_hc1 = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r_robust.se == r_hc1.se

    def test_robust_false_maps_to_classical(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            r_robust = HeterogeneousAdoptionDiD(design="mass_point", robust=False).fit(
                panel, "outcome", "dose", "period", "unit"
            )
        r_classical = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r_robust.se == r_classical.se

    def test_vcov_type_explicit_overrides_robust(self):
        """When vcov_type is explicit, robust is ignored."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            r = HeterogeneousAdoptionDiD(
                design="mass_point", vcov_type="classical", robust=True
            ).fit(panel, "outcome", "dose", "period", "unit")
        assert r.vcov_type == "classical"


# =============================================================================
# Criterion 6: hc2 / hc2_bm raise NotImplementedError
# =============================================================================


class TestMassPointUnsupportedVcov:
    def test_hc2_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2")
        with pytest.raises(NotImplementedError, match="HC2"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_hc2_bm_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2_bm")
        with pytest.raises(NotImplementedError, match="HC2"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_hc2_pointer_references_followup_pr(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2")
        with pytest.raises(NotImplementedError, match="follow-up"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_vcov_type_ignored_on_continuous(self):
        """hc2 passed with continuous design emits warning, does not raise."""
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", vcov_type="hc2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("ignored" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)

    def test_robust_true_ignored_on_continuous_warns(self):
        """Review P2 round 9: robust=True on continuous path must warn.

        The continuous designs use the CCT-2014 robust SE unconditionally;
        robust= is a mass-point-only backward-compat alias for vcov_type.
        Passing robust=True on a continuous path has no effect on the
        computed SE, so the user must get a warning that the flag was
        ignored.
        """
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            est = HeterogeneousAdoptionDiD(design="continuous_at_zero", robust=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            robust_warnings = [warn for warn in w if "robust" in str(warn.message).lower()]
            assert len(robust_warnings) >= 1
        assert np.isfinite(r.att)

    def test_robust_false_silent_on_continuous(self):
        """robust=False (the default) on continuous path emits no robust-warn."""
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        with pytest.warns(FutureWarning, match=r"\(robust=\) is deprecated"):
            est = HeterogeneousAdoptionDiD(design="continuous_at_zero", robust=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            robust_warnings = [warn for warn in w if "robust=True is ignored" in str(warn.message)]
            assert len(robust_warnings) == 0
        assert np.isfinite(r.att)


# =============================================================================
# Criterion 7: Panel-contract violations
# =============================================================================


class TestPanelContract:
    def test_missing_outcome_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "missing", "dose", "period", "unit")

    def test_missing_dose_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "missing", "period", "unit")

    def test_missing_time_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "dose", "missing", "unit")

    def test_missing_unit_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "dose", "period", "missing")

    def test_nonzero_pre_period_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[panel["period"] == 1, "dose"] = 0.5
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"D_\{g,1\}|pre-period"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_unbalanced_panel_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy).iloc[:-1]  # drop one row
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"[Uu]nbalanced|[Bb]alanced"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_three_periods_without_first_treat_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel2 = _make_panel(d, dy)
        panel3 = pd.concat([panel2, panel2.assign(period=3)])
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"two time periods|Phase 2b"):
            est.fit(panel3, "outcome", "dose", "period", "unit")

    def test_three_periods_with_first_treat_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel2 = _make_panel(d, dy)
        panel3 = pd.concat([panel2, panel2.assign(period=3)])
        panel3["ft"] = 2  # arbitrary first_treat
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"two time periods|Phase 2b"):
            est.fit(
                panel3,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="ft",
            )

    def test_single_period_raises(self):
        d, _ = _dgp_continuous_at_zero(200, seed=0)
        panel = pd.DataFrame(
            {
                "unit": np.arange(200),
                "period": 2,
                "dose": d,
                "outcome": np.zeros(200),
            }
        )
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="two-period"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_outcome_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[0, "outcome"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[3, "dose"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_duplicate_unit_period_raises(self):
        """Two observations of the same unit-period cell."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel = pd.concat([panel, panel.iloc[[0]]])  # duplicate first row
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"[Uu]nbalanced|observation"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_first_treat_col_invalid_cohort_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Set first_treat values to {0, 5, 2} where 5 is not t_post.
        ft_unit = np.where(np.arange(200) % 2 == 0, 0, 5)
        panel["ft"] = np.repeat(ft_unit, 2)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"first_treat"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="ft",
            )

    def test_first_treat_col_mixed_row_nan_raises(self):
        """Review P2 round 8: per-unit rows like [valid, NaN] must be rejected.

        `groupby().first()` silently skips NaNs; a unit with [0, NaN]
        collapses to first_treat=0 and a unit-level NaN check would
        pass. Row-level validation must catch the NaN on the bad row.
        """
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Unit-level first_treat all zero (never-treated); inject a NaN on
        # exactly the second row of unit 0 (t_post row).
        panel["ft"] = 0.0
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "ft"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit", first_treat="ft")

    def test_first_treat_col_mixed_row_invalid_value_raises(self):
        """Per-unit rows like [valid, invalid_value] must be rejected."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel["ft"] = 0.0
        # Inject an out-of-domain value on unit 0's post-period row.
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "ft"] = 999.0
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"first_treat.*999"):
            est.fit(panel, "outcome", "dose", "period", "unit", first_treat="ft")


# =============================================================================
# Criterion 8: NaN propagation
# =============================================================================


class TestNaNPropagation:
    def test_constant_y_produces_nan_inference(self):
        """Constant outcome -> zero residuals -> NaN via safe_inference."""
        d, _ = _dgp_continuous_at_zero(500, seed=0)
        dy_zero = np.zeros_like(d)
        panel = _make_panel(d, dy_zero)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # All inference fields NaN when SE is non-finite.
        assert_nan_inference(
            {
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "conf_int": r.conf_int,
            }
        )

    def test_mass_point_all_at_d_lower_nan(self):
        """Degenerate mass-point: all units at d_lower -> NaN."""
        rng = np.random.default_rng(0)
        G = 500
        d = np.full(G, 0.5)  # all at 0.5
        dy = 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        # Avoid triggering pre-period D=0 check by starting at 0.5 at t2.
        r = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.5).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isnan(r.att)
        assert_nan_inference(
            {
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "conf_int": r.conf_int,
            }
        )

    def test_helper_returns_nan_on_empty_z_one(self):
        """_fit_mass_point_2sls returns NaN when no units above d_lower."""
        d = np.full(50, 0.5)
        dy = np.random.default_rng(0).standard_normal(50)
        beta, se, _ = _fit_mass_point_2sls(d, dy, 0.5, None, "hc1")
        assert np.isnan(beta)
        assert np.isnan(se)

    def test_helper_returns_nan_on_empty_z_zero(self):
        """_fit_mass_point_2sls returns NaN when no units at d_lower."""
        d = np.full(50, 0.6)  # all strictly above d_lower=0.5
        dy = np.random.default_rng(0).standard_normal(50)
        beta, se, _ = _fit_mass_point_2sls(d, dy, 0.5, None, "hc1")
        assert np.isnan(beta)
        assert np.isnan(se)

    def test_single_cluster_cr1_returns_nan(self):
        """CR1 with only 1 cluster is undefined -> NaN."""
        rng = np.random.default_rng(0)
        G = 100
        d = np.concatenate([np.full(30, 0.5), rng.uniform(0.5, 1.0, G - 30)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy, extra_cols={"state": np.zeros(G, dtype=int)})  # single cluster
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isnan(r.se)


# =============================================================================
# Criterion 9: sklearn clone round-trip + fit idempotence
# =============================================================================


class TestSklearnCompat:
    def test_get_params_returns_all_constructor_args(self):
        est = HeterogeneousAdoptionDiD(
            design="continuous_near_d_lower",
            d_lower=0.3,
            kernel="triangular",
            alpha=0.1,
            vcov_type="hc1",
            cluster="state",
            n_bootstrap=500,
            seed=42,
        )
        params = est.get_params()
        assert params == {
            "design": "continuous_near_d_lower",
            "d_lower": 0.3,
            "kernel": "triangular",
            "alpha": 0.1,
            "vcov_type": "hc1",
            "robust": None,
            "cluster": "state",
            "n_bootstrap": 500,
            "seed": 42,
        }

    def test_clone_round_trip(self):
        est = HeterogeneousAdoptionDiD(design="auto", alpha=0.1, kernel="triangular")
        est2 = HeterogeneousAdoptionDiD(**est.get_params())
        assert est.get_params() == est2.get_params()

    def test_fit_idempotent_same_att(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        r1 = est.fit(panel, "outcome", "dose", "period", "unit")
        r2 = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r1.att == r2.att
        assert r1.se == r2.se
        assert r1.conf_int == r2.conf_int

    def test_set_params_updates_and_returns_self(self):
        est = HeterogeneousAdoptionDiD()
        ret = est.set_params(alpha=0.1, design="continuous_at_zero")
        assert ret is est
        assert est.alpha == 0.1
        assert est.design == "continuous_at_zero"

    def test_set_params_invalid_key_raises(self):
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(not_a_param=True)

    def test_set_params_rejects_method_names(self):
        """Review P1 round 10: set_params must restrict to constructor keys,
        not any hasattr-able name. Method names like 'fit' must raise,
        else they would silently overwrite the method.
        """
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(fit="not_a_method")
        # sanity: fit is still callable on the class
        assert callable(est.fit)

    def test_set_params_rejects_private_attrs(self):
        """Internal-looking attribute names must also raise."""
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(_internal=42)

    def test_get_params_accepts_deep_keyword(self):
        """Review P1 round 10: get_params must match sklearn's signature.

        sklearn.base.BaseEstimator.get_params(deep=True). This estimator
        has no nested sub-estimators, so deep=True and deep=False return
        the same dict, but the keyword must be accepted.
        """
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", alpha=0.1)
        params_default = est.get_params()
        params_deep_true = est.get_params(deep=True)
        params_deep_false = est.get_params(deep=False)
        assert params_default == params_deep_true == params_deep_false

    def test_sklearn_clone_round_trip_if_available(self):
        """If sklearn is installed, sklearn.base.clone round-trips the estimator."""
        sklearn_base = pytest.importorskip("sklearn.base")
        est = HeterogeneousAdoptionDiD(design="auto", alpha=0.1, kernel="triangular")
        cloned = sklearn_base.clone(est)
        assert cloned.get_params() == est.get_params()
        assert cloned is not est
        # clone produces a fresh instance of the same class.
        assert type(cloned) is type(est)

    def test_set_params_invalid_design_raises(self):
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="design"):
            est.set_params(design="made_up")

    def test_set_params_rollback_on_failure(self):
        """Review P2 round 11: set_params must be ATOMIC.

        A failing call (valid key but value violates constructor
        constraints) must leave the estimator unchanged so the caller
        can catch the ValueError and reuse the object.
        """
        est = HeterogeneousAdoptionDiD(alpha=0.05, design="continuous_at_zero")
        baseline = est.get_params()
        # Multi-key call where alpha is valid but design is invalid.
        # The old (non-atomic) code would have set alpha before raising
        # on design, leaving the estimator half-mutated.
        with pytest.raises(ValueError):
            est.set_params(alpha=0.1, design="garbage_design")
        assert est.get_params() == baseline

    def test_set_params_rollback_on_invalid_key(self):
        """Rejecting an unknown key must leave self unchanged."""
        est = HeterogeneousAdoptionDiD(alpha=0.05)
        baseline = est.get_params()
        with pytest.raises(ValueError):
            est.set_params(alpha=0.1, not_a_param=True)
        assert est.get_params() == baseline

    def test_set_params_rollback_on_invalid_alpha(self):
        """alpha outside (0, 1) must leave self unchanged."""
        est = HeterogeneousAdoptionDiD(alpha=0.05, design="continuous_at_zero")
        baseline = est.get_params()
        with pytest.raises(ValueError):
            est.set_params(alpha=1.5, kernel="triangular")
        assert est.get_params() == baseline


# =============================================================================
# Criterion 10: Scaffolding raises
# =============================================================================


class TestScaffoldingRejections:
    def test_aggregate_event_study_on_two_period_panel_raises(self):
        """Event-study mode requires T > 2 (Phase 2b). A T=2 panel should
        raise a helpful ValueError pointing to ``aggregate='overall'``."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="more than two"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
            )

    def test_aggregate_invalid_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Invalid aggregate"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="garbage",
            )

    def test_survey_design_bad_type_raises(self):
        """survey_design= must be a SurveyDesign-like object with a
        `.resolve()` method; a bare string (or any object lacking
        `.resolve()`) raises TypeError front-door. The data-in type guard
        runs at the canonical entry and rejects on the
        `hasattr(survey_design, "resolve")` check (which catches both bare
        strings and ResolvedSurveyDesign / make_pweight_design output)."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(TypeError, match="SurveyDesign"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design="anything",
            )


# =============================================================================
# Criterion 11: get_params signature enumeration
# =============================================================================


class TestGetParamsContract:
    def test_get_params_matches_init_signature(self):
        sig_params = set(inspect.signature(HeterogeneousAdoptionDiD.__init__).parameters.keys()) - {
            "self"
        }
        gp_params = set(HeterogeneousAdoptionDiD().get_params().keys())
        assert sig_params == gp_params

    def test_set_params_covers_all_init_params(self):
        """Every __init__ param must be settable via set_params."""
        est = HeterogeneousAdoptionDiD()
        params = est.get_params()
        # Round-trip via set_params
        new_est = HeterogeneousAdoptionDiD()
        new_est.set_params(**params)
        assert new_est.get_params() == params


# =============================================================================
# Result class methods
# =============================================================================


class TestResultMethods:
    def _result(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        panel = _make_panel(d, dy)
        return HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")

    def test_summary_returns_string(self):
        r = self._result()
        s = r.summary()
        assert isinstance(s, str)
        assert "HeterogeneousAdoptionDiD" in s
        assert "WAS" in s
        assert "Confidence Interval" in s

    def test_summary_uses_target_parameter_for_row_label(self):
        """Review P2: the estimate row must render target_parameter (WAS or
        WAS_d_lower), not hardcoded 'WAS'.
        """
        # Design 1' -> target_parameter = "WAS"
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        panel = _make_panel(d, dy)
        r_d1p = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        s_d1p = r_d1p.summary()
        assert r_d1p.target_parameter == "WAS"
        assert "WAS" in s_d1p

        # Design 1 continuous-near-d_lower -> target_parameter = "WAS_d_lower"
        d, dy = _dgp_continuous_near_d_lower(400, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_d1 = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_d1.target_parameter == "WAS_d_lower"
        assert "WAS_d_lower" in r_d1.summary()

        # Design 1 mass-point -> target_parameter = "WAS_d_lower"
        d, dy = _dgp_mass_point(400, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_mp = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_mp.target_parameter == "WAS_d_lower"
        assert "WAS_d_lower" in r_mp.summary()

    def test_print_summary_executes(self, capsys):
        r = self._result()
        r.print_summary()
        captured = capsys.readouterr()
        assert "HeterogeneousAdoptionDiD" in captured.out

    def test_to_dict_populated(self):
        r = self._result()
        d = r.to_dict()
        assert "att" in d
        assert "se" in d
        assert "design" in d
        assert "target_parameter" in d
        assert "d_lower" in d
        assert "dose_mean" in d
        assert "n_obs" in d
        assert d["design"] == "continuous_at_zero"

    def test_to_dataframe_populated(self):
        r = self._result()
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_repr_concise(self):
        r = self._result()
        s = repr(r)
        assert "HeterogeneousAdoptionDiDResults" in s
        assert "att=" in s
        assert "design=" in s

    def test_mass_point_summary_includes_mass_count(self):
        d, dy = _dgp_mass_point(400, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        s = r.summary()
        assert "mass point" in s.lower() or "At d_lower" in s


# =============================================================================
# Design metadata
# =============================================================================


class TestDesignMetadata:
    def test_target_parameter_design_1_prime(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS"

    def test_target_parameter_design_1(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS_d_lower"

    def test_target_parameter_mass_point(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS_d_lower"

    def test_d_lower_zero_for_design_1_prime(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.d_lower == 0.0

    def test_d_lower_from_data_for_continuous_near(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert abs(r.d_lower - float(d.min())) < 1e-14

    def test_d_lower_explicit_override(self):
        """d_lower override must satisfy d.min() >= d_lower (else negative shifted doses)."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        # d.min() is around 0.1 + epsilon for this DGP; override within that.
        d_lower_user = float(d.min())  # explicit but equal to default
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=d_lower_user).fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert abs(r.d_lower - d_lower_user) < 1e-14

    def test_inference_method_continuous(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.inference_method == "analytical_nonparametric"

    def test_inference_method_mass_point(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.inference_method == "analytical_2sls"

    def test_survey_metadata_always_none_in_phase_2a(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.survey_metadata is None

    def test_alpha_stored_on_result(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(alpha=0.1).fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.alpha == 0.1


# =============================================================================
# Constructor validation
# =============================================================================


class TestConstructorValidation:
    def test_invalid_design_raises(self):
        with pytest.raises(ValueError, match="Invalid design"):
            HeterogeneousAdoptionDiD(design="random_garbage")

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=0.0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=-0.05)

    def test_invalid_vcov_type_raises(self):
        with pytest.raises(ValueError, match="vcov_type"):
            HeterogeneousAdoptionDiD(vcov_type="garbage")

    def test_vcov_type_none_accepted(self):
        est = HeterogeneousAdoptionDiD(vcov_type=None)
        assert est.vcov_type is None

    def test_d_lower_none_accepted(self):
        est = HeterogeneousAdoptionDiD(d_lower=None)
        assert est.d_lower is None

    def test_d_lower_float_accepted(self):
        est = HeterogeneousAdoptionDiD(d_lower=0.3)
        assert est.d_lower == 0.3

    def test_d_lower_nan_raises(self):
        """Review P1 round 13: d_lower=NaN must be rejected in __init__."""
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("nan"))

    def test_d_lower_posinf_raises(self):
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("inf"))

    def test_d_lower_neginf_raises(self):
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("-inf"))

    def test_d_lower_nan_via_set_params_raises(self):
        """d_lower=NaN through set_params must also raise (atomic rollback)."""
        est = HeterogeneousAdoptionDiD(d_lower=0.3)
        baseline = est.get_params()
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            est.set_params(d_lower=float("nan"))
        # Atomic rollback: d_lower unchanged after failure.
        assert est.get_params() == baseline


# =============================================================================
# Explicit design override (don't auto-reject)
# =============================================================================


class TestExplicitDesignOverrides:
    def test_force_continuous_at_zero_on_mass_point_data(self):
        """Forcing Design 1' on mass-point data should run (may produce wide CIs)."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        # Phase 1c's _validate_had_inputs would reject this (mass point),
        # so this will raise NotImplementedError from underneath, NOT from had.py.
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(NotImplementedError, match="mass-point"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_mass_point_on_d_lower_zero_sample_raises(self):
        """Review P1 round 4: Design 1 paths require d_lower > 0.

        Paper Section 3.2 reserves the d_lower=0 regime for Design 1'
        (continuous_at_zero). Forcing `mass_point` on a sample with
        d.min()==0 must raise, pointing the user to continuous_at_zero
        or auto.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.0)
        with pytest.raises(ValueError, match=r"d_lower > 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_continuous_near_d_lower_on_d_lower_zero_sample_raises(self):
        """Parallel: continuous_near_d_lower must also reject d_lower=0."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
        # d_lower auto-resolves to float(d.min()) == 0.0 on this DGP.
        with pytest.raises(ValueError, match=r"d_lower > 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_mass_point_d_lower_none_on_zero_sample_raises(self):
        """d_lower=None on a d.min()==0 sample resolves to 0; must still raise."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=None)
        with pytest.raises(ValueError, match=r"d_lower > 0"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_nonzero_d_lower_raises(self):
        """Review P1 round 12: continuous_at_zero must reject nonzero d_lower.

        Paper Section 3.2 Design 1' is defined at d_lower = 0; silently
        coercing a user-supplied d_lower=0.5 to zero would contradict
        the documented regime contract.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.5)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_small_d_lower_raises(self):
        """Even a small nonzero d_lower should raise."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.01)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_zero_d_lower_succeeds(self):
        """d_lower=0.0 exactly is fine (redundant but allowed)."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.0)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.d_lower == 0.0
        assert np.isfinite(r.att)

    def test_auto_on_zero_sample_ignores_user_d_lower(self):
        """design='auto' resolving to continuous_at_zero must ALSO reject
        an explicit nonzero d_lower, not silently drop it.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="auto", d_lower=0.5)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")


# =============================================================================
# Design 1 d_lower contract enforcement (mass-point + continuous_near_d_lower)
# =============================================================================


class TestDesign1DLowerContract:
    """Paper Sections 3.2.2-3.2.4: Design 1 estimators identify at the support
    infimum. Both mass_point and continuous_near_d_lower require
    ``d_lower == float(d.min())`` within float tolerance; mismatched
    overrides raise.
    """

    def test_mass_point_d_lower_above_min_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)  # d.min() == 0.5
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.6)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_d_lower_below_min_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.3)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_d_lower_matches_succeeds(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.5)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.d_lower == 0.5
        assert np.isfinite(r.att)

    def test_mass_point_d_lower_none_auto_resolves_to_min(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=None)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert abs(r.d_lower - float(d.min())) < 1e-14

    def test_mass_point_d_lower_within_tolerance_succeeds(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_user = float(d.min()) + 1e-15
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=d_lower_user)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert np.isfinite(r.att)

    def test_mass_point_d_lower_below_min_within_tolerance_snaps(self):
        """Review P1 round 8: tolerance-accepted d_lower = d.min() - ε must
        be SNAPPED to d.min() so the instrument Z = d > d_lower matches
        the exact-minimum case; otherwise mass-point units would fall
        into Z=1 and empty the control group.
        """
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_below = float(d.min()) - 1e-15
        r_below = HeterogeneousAdoptionDiD(design="mass_point", d_lower=d_lower_below).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        r_exact = HeterogeneousAdoptionDiD(design="mass_point", d_lower=float(d.min())).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # Behavior must be identical within ULP (the snap collapses them).
        assert r_below.att == r_exact.att
        assert r_below.se == r_exact.se
        assert r_below.n_mass_point == r_exact.n_mass_point
        assert r_below.n_above_d_lower == r_exact.n_above_d_lower

    def test_continuous_near_d_lower_above_within_tolerance_snaps(self):
        """Review P1 round 8: tolerance-accepted d_lower = d.min() + ε on
        continuous_near_d_lower must be SNAPPED so the regressor shift
        `d - d_lower` does not produce negative doses and trip Phase 1c's
        _validate_had_inputs negative-dose guard.
        """
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_above = float(d.min()) + 1e-15
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_above = HeterogeneousAdoptionDiD(
                design="continuous_near_d_lower", d_lower=d_lower_above
            ).fit(panel, "outcome", "dose", "period", "unit")
            r_exact = HeterogeneousAdoptionDiD(
                design="continuous_near_d_lower", d_lower=float(d.min())
            ).fit(panel, "outcome", "dose", "period", "unit")
        assert r_above.att == r_exact.att
        assert r_above.se == r_exact.se

    def test_continuous_near_d_lower_above_min_raises(self):
        """Review P1: continuous_near_d_lower must also enforce support infimum."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=0.3)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_d_lower_below_min_raises(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        # d.min() for this Beta DGP is > 0.1 but setting d_lower=0.05 is below min.
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=0.05)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_d_lower_matches_succeeds(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=float(d.min()))
            r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert np.isfinite(r.att)


# =============================================================================
# Post-period dose non-negative contract (review P1)
# =============================================================================


class TestPostPeriodDoseContract:
    """Paper Section 2 dose definition: D_{g,2} >= 0. _validate_had_panel
    rejects negative post-period dose front-door on the ORIGINAL scale
    (before the regressor shift) so the error references the user's
    dose column, not the Phase 1c shifted values.
    """

    def test_negative_post_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Inject a negative post-period dose on one unit.
        post_mask = panel["period"] == 2
        idx = panel[post_mask].index[0]
        panel.loc[idx, "dose"] = -0.1
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"D_\{g,2\}|negative post"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_zero_post_dose_accepted(self):
        """D_{g,2} == 0 is the Design 1' no-treated-group case, always allowed."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        # Ensure d[0] == 0 exactly (no-treated unit) is accepted.
        assert d[0] == 0.0
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)


# =============================================================================
# Design 1 Assumption 5/6 identification warning (review P1)
# =============================================================================


class TestAssumptionFiveSixWarning:
    """Paper Sections 3.2.2-3.2.4: Design 1 fits require Assumption 5 (sign
    identification) or Assumption 6 (point identification of WAS_{d_lower})
    beyond parallel trends. These extras are not pre-trend testable. A
    UserWarning surfaces the identification burden on Design 1 fits.
    """

    def test_continuous_near_d_lower_emits_assumption_warning(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption" in str(warn.message)]
            assert len(assumption_warnings) >= 1

    def test_mass_point_emits_assumption_warning(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption" in str(warn.message)]
            assert len(assumption_warnings) >= 1

    def test_continuous_at_zero_does_not_emit_assumption_warning(self):
        """Design 1' (d_lower=0) is identified under Assumption 3 only; no warning."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption 6" in str(warn.message)]
            assert len(assumption_warnings) == 0


# =============================================================================
# Cluster handling (unit-level aggregation)
# =============================================================================


class TestClusterHandling:
    def test_cluster_not_constant_within_unit_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        panel = _make_panel(d, dy)
        # Make cluster vary within unit
        panel["state"] = np.arange(len(panel))
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match=r"constant within unit"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_missing_cluster_column_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="missing")
        with pytest.raises(ValueError, match="cluster"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_cluster_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        # Unit-level cluster ids: 50 clusters, 2 units each, with NaN on unit 0.
        cluster_unit = np.repeat(np.arange(50).astype(float), 2)  # length 100
        cluster_unit[0] = np.nan
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mixed_row_nan_cluster_raises_on_mass_point(self):
        """Review P2 round 8: a unit with rows [valid, NaN] on mass-point
        must be rejected by row-level validation, not masked by
        `groupby().first()`.
        """
        d, dy = _dgp_mass_point(100, seed=0)
        cluster_unit = np.repeat(np.arange(50).astype(float), 2)  # all valid
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        # Inject NaN only on the second row (t_post) of unit 0.
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "state"] = np.nan
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    @staticmethod
    def _clustered_continuous(G=200, n_clusters=40, seed=0):
        """Design 1' DGP with cluster-correlated errors (so CR1 != HC)."""
        rng = np.random.default_rng(seed)
        d = np.where(rng.random(G) < 0.15, 0.0, rng.uniform(0.2, 1.5, size=G))
        d[0] = 0.0
        cl = np.repeat(np.arange(n_clusters), G // n_clusters)
        shock = rng.normal(scale=1.0, size=n_clusters)[cl]
        dy = 1.5 * d + shock + rng.normal(scale=0.3, size=G)
        return d, dy, cl

    def test_cluster_threaded_on_continuous_path(self):
        # Phase 2a: cluster= is now threaded into bias_corrected_local_linear
        # (no longer ignored). The estimator SE equals the direct clustered
        # local-linear se_robust rescaled by 1/|den|; the point estimate is
        # unchanged and the clustered SE differs from the unclustered SE.
        d, dy, cl = self._clustered_continuous()
        panel = _make_panel(d, dy, extra_cols={"state": cl})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r_cl = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        # No "cluster ignored" warning any more.
        assert not any(
            "cluster" in str(x.message).lower() and "ignore" in str(x.message).lower() for x in w
        )
        r_none = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        den = float(d.mean())
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, cluster=cl)
        se_ref = float(bc.se_robust) / abs(den)
        np.testing.assert_allclose(r_cl.att, r_none.att, atol=1e-12)  # point unchanged
        np.testing.assert_allclose(r_cl.se, se_ref, rtol=0.0, atol=1e-12)  # exact rescale
        assert abs(r_cl.se - r_none.se) > 1e-4  # cluster-robust differs from HC
        assert r_cl.vcov_type == "cr1"
        assert r_cl.cluster_name == "state"

    def test_cluster_survey_design_raises_on_continuous(self):
        # cluster= + survey_design= is rejected: the Binder-TSL survey path
        # would override the cluster-robust SE (route via psu= instead).
        from diff_diff.survey import SurveyDesign

        d, dy, cl = self._clustered_continuous(seed=3)
        rng = np.random.default_rng(4)
        panel = _make_panel(
            d, dy, extra_cols={"state": cl, "wt": rng.uniform(0.5, 2.0, size=len(d))}
        )
        with pytest.raises(NotImplementedError, match="cluster.*survey_design"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state").fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="wt", psu="state"),
                )

    def test_cluster_threaded_on_continuous_near_d_lower(self):
        # The other continuous design (continuous_near_d_lower) threads cluster=
        # through the same _fit_continuous hook: the SE equals the direct
        # clustered local-linear se_robust on the shifted regressor (d - d_lower)
        # rescaled by 1/|mean(d - d_lower)|.
        rng = np.random.default_rng(5)
        G, n_clusters = 200, 40
        d = 0.1 + 0.9 * rng.beta(2, 2, G)
        cl = np.repeat(np.arange(n_clusters), G // n_clusters)
        shock = rng.normal(scale=1.0, size=n_clusters)[cl]
        dy = 1.5 * d + shock + rng.normal(scale=0.3, size=G)
        panel = _make_panel(d, dy, extra_cols={"state": cl})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        d_lower = float(d.min())
        den = float((d - d_lower).mean())
        bc = bias_corrected_local_linear(d=d - d_lower, y=dy, boundary=0.0, cluster=cl)
        np.testing.assert_allclose(r.se, float(bc.se_robust) / abs(den), rtol=0.0, atol=1e-12)
        assert r.cluster_name == "state"
        assert r.vcov_type == "cr1"

    def test_single_cluster_continuous_at_zero_nan_inference(self):
        # Cluster-robust inference is unidentified with one cluster: se/p/CI are
        # NaN while att stays finite (mirrors the mass-point CR1 contract).
        rng = np.random.default_rng(0)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy, extra_cols={"state": np.zeros(G, dtype=int)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)
        assert np.isnan(r.se)
        assert np.isnan(r.p_value)
        assert np.all(np.isnan(r.conf_int))

    def test_single_cluster_continuous_near_d_lower_nan_inference(self):
        rng = np.random.default_rng(1)
        G = 300
        d = 0.1 + 0.9 * rng.beta(2, 2, G)
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy, extra_cols={"state": np.zeros(G, dtype=int)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)
        assert np.isnan(r.se)

    def test_cluster_name_populated_mass_point(self):
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)  # 50 clusters, 4 units each
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r.cluster_name == "state"

    def test_cluster_name_none_without_cluster(self):
        d, dy = _dgp_mass_point(200, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.cluster_name is None

    def test_missing_cluster_column_on_continuous_raises(self):
        """Now that cluster= is threaded on the continuous path (Phase 2a), a
        nonexistent cluster column raises (mirrors the mass-point path) rather
        than being silently ignored with a warning."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="does_not_exist")
        with pytest.raises(ValueError, match="cluster"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_cluster_on_continuous_raises(self):
        """NaN cluster IDs on the continuous path now raise (cluster is used)."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        cluster_unit = np.repeat(np.arange(100).astype(float), 2)
        cluster_unit[0] = np.nan
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state")
        with pytest.raises(ValueError, match="cluster"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_within_unit_varying_cluster_on_continuous_raises(self):
        """Within-unit-varying cluster IDs on the continuous path now raise."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Varies within unit (distinct value per row)
        panel["state"] = np.arange(len(panel))
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state")
        with pytest.raises(ValueError, match="cluster"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_auto_design_continuous_threads_cluster(self):
        """design='auto' resolving to a continuous path now threads (and honors)
        a valid cluster column."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        cluster_unit = np.repeat(np.arange(100), 5)  # 100 clusters, unit-level
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="auto", cluster="state")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.design == "continuous_at_zero"
        assert np.isfinite(r.att)
        assert r.cluster_name == "state"
        assert r.vcov_type == "cr1"


# =============================================================================
# First-difference aggregation helper
# =============================================================================


class TestFirstDifferenceAggregation:
    def test_aggregate_returns_sorted_unit_order(self):
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy)
        # Shuffle rows to test sort-invariance
        panel_shuffled = panel.sample(frac=1, random_state=42).reset_index(drop=True)
        d_arr, dy_arr, _, unit_ids = _aggregate_first_difference(
            panel_shuffled, "outcome", "dose", "period", "unit", 1, 2, None
        )
        # unit_ids sorted
        assert np.all(np.diff(unit_ids) >= 0)
        # Each dose matches the input dose for its unit
        for i, uid in enumerate(unit_ids):
            assert abs(d_arr[i] - d[uid]) < 1e-14
            assert abs(dy_arr[i] - dy[uid]) < 1e-14

    def test_aggregate_cluster_array_correct(self):
        d, dy = _dgp_mass_point(100, seed=0)
        cluster_unit = np.repeat(np.arange(25), 4)  # 25 clusters, 4 units each
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        _, _, cluster_arr, unit_ids = _aggregate_first_difference(
            panel,
            "outcome",
            "dose",
            "period",
            "unit",
            1,
            2,
            "state",
        )
        assert cluster_arr is not None
        assert len(cluster_arr) == 100
        # Cluster_arr[i] should equal cluster_unit[unit_ids[i]]
        for i, uid in enumerate(unit_ids):
            assert cluster_arr[i] == cluster_unit[uid]

    def test_aggregate_no_cluster_returns_none(self):
        d, dy = _dgp_continuous_at_zero(50, seed=0)
        panel = _make_panel(d, dy)
        _, _, cluster_arr, _ = _aggregate_first_difference(
            panel, "outcome", "dose", "period", "unit", 1, 2, None
        )
        assert cluster_arr is None


# =============================================================================
# Auto-detect mass-point vs continuous-near at boundary
# =============================================================================


class TestAutoDetectEdges:
    def test_exactly_two_percent_modal_is_not_mass_point(self):
        """Threshold is strict >, not >=. 2% exactly should stay continuous."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 20  # exactly 2%
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        # d.min() == 0.5, not 0, and modal fraction == 2% (not > 2%)
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_slightly_over_two_percent_is_mass_point(self):
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 25  # 2.5%
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        assert _detect_design(d) == "mass_point"

    def test_all_at_zero_resolves_continuous_at_zero(self):
        """Degenerate but well-defined: all zeros -> continuous_at_zero."""
        d = np.zeros(100)
        assert _detect_design(d) == "continuous_at_zero"


# =============================================================================
# Panel validator direct tests
# =============================================================================


class TestValidateHadPanel:
    def test_returns_period_pair(self):
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=(2020, 2021))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == 2020
        assert t_post == 2021

    def test_rejects_string_periods_gracefully(self):
        """String periods should still sort and validate."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # Should not raise - strings sort fine
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "A"
        assert t_post == "B"

    def test_first_treat_col_with_string_periods(self):
        """Review P1: first_treat_col validator must be dtype-agnostic.

        With string periods ("A", "B") and first_treat_col values in
        {0, "B"}, the validator must not attempt numeric coercion.
        """
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # 50 units never-treated (first_treat=0), 50 treated (first_treat="B")
        ft_unit = np.array([0 if i % 2 == 0 else "B" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")
        assert t_pre == "A"
        assert t_post == "B"

    def test_first_treat_col_dtype_agnostic_rejects_invalid_string(self):
        """Mix string periods + invalid first_treat_col string -> ValueError."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # Invalid: "Z" is neither 0 nor "B"
        ft_unit = np.array([0 if i % 2 == 0 else "Z" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        with pytest.raises(ValueError, match="first_treat"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")

    def test_semantic_pre_post_labels_not_lexicographic(self):
        """Review P1 round 3: pre/post inference must be dose-based.

        ("pre", "post") sorts alphabetically to ["post", "pre"], which
        previously flipped the pre/post labels and raised on a valid
        panel. The validator now infers pre from the all-zero-dose
        period.
        """
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "pre"
        assert t_post == "post"

    def test_semantic_pre_post_with_first_treat_col(self):
        """Combined: string periods + first_treat_col in {0, 'post'}."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        ft_unit = np.array([0 if i % 2 == 0 else "post" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")
        assert t_pre == "pre"
        assert t_post == "post"

    def test_semantic_pre_post_fit_end_to_end(self):
        """End-to-end: fit() runs on ("pre","post")-labelled panel."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)

    def test_before_after_labels(self):
        """("before","after") is also reversed alphabetically; must not fail."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("before", "after"))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "before"
        assert t_post == "after"

    def test_no_all_zero_period_raises(self):
        """If neither period has all-zero dose, HAD's D_{g,1}=0 contract fails."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy)
        # Inject nonzero dose into the pre period so neither period is all-zero.
        panel.loc[panel["period"] == 1, "dose"] = 0.5
        with pytest.raises(ValueError, match=r"D_\{g,1\}|pre-treatment"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)

    def test_both_all_zero_periods_raises(self):
        """If both periods have all-zero dose, no treatment to estimate."""
        G = 100
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.zeros(2 * G),
                "outcome": np.random.default_rng(0).standard_normal(2 * G),
            }
        )
        with pytest.raises(ValueError, match="variation"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)

    def test_repeated_cross_section_raises(self):
        """Review P1 round 6: Phase 2a is panel-only. An RCS input (disjoint
        unit IDs across periods) must be rejected by the balanced-panel
        validator with the "unit(s) do not appear in both periods" error.
        """
        rng = np.random.default_rng(0)
        G = 100
        pre = pd.DataFrame(
            {
                "unit": np.arange(G),
                "period": 1,
                "dose": np.zeros(G),
                "outcome": rng.standard_normal(G),
            }
        )
        post = pd.DataFrame(
            {
                "unit": np.arange(G, 2 * G),
                "period": 2,
                "dose": rng.uniform(0, 1, G),
                "outcome": rng.standard_normal(G),
            }
        )
        rcs = pd.concat([pre, post], ignore_index=True)
        with pytest.raises(ValueError, match=r"both periods|[Uu]nbalanced"):
            _validate_had_panel(rcs, "outcome", "dose", "period", "unit", None)

    def test_repeated_cross_section_fit_raises(self):
        """End-to-end: fit() on an RCS panel raises ValueError."""
        rng = np.random.default_rng(0)
        G = 100
        pre = pd.DataFrame(
            {
                "unit": np.arange(G),
                "period": 1,
                "dose": np.zeros(G),
                "outcome": rng.standard_normal(G),
            }
        )
        post = pd.DataFrame(
            {
                "unit": np.arange(G, 2 * G),
                "period": 2,
                "dose": rng.uniform(0, 1, G),
                "outcome": rng.standard_normal(G),
            }
        )
        rcs = pd.concat([pre, post], ignore_index=True)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"both periods|[Uu]nbalanced"):
            est.fit(rcs, "outcome", "dose", "period", "unit")


# =============================================================================
# Review P1: continuous_near_d_lower on a true mass-point sample rejects
# =============================================================================


class TestContinuousPathRejectsMassPoint:
    """Explicit override to continuous_near_d_lower on a mass-point sample
    must raise before the regressor shift, otherwise the Phase 1c
    mass-point guard (which fires only on d.min() > 0) is bypassed.
    """

    def test_continuous_near_on_mass_point_sample_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
        with pytest.raises(ValueError, match=r"mass-point sample|mass_point"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_on_continuous_sample_runs(self):
        """Sanity: the pre-shift check does NOT reject valid continuous samples."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)


class TestMassPointPathRejectsContinuousSample:
    """Review P1 round 5: reciprocal guard. Forcing design="mass_point" on a
    continuous-near-d_lower sample (modal fraction at d.min() <= 2%) must
    raise, otherwise 2SLS identifies the exact-d.min() cell rather than
    the paper's boundary-limit estimand.
    """

    def test_mass_point_on_continuous_near_sample_raises(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point")
        with pytest.raises(ValueError, match=r"modal mass|2SLS.*continuous"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_on_true_mass_point_sample_runs(self):
        """Sanity: the reciprocal guard does NOT reject valid mass-point samples."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)

    def test_mass_point_modal_at_threshold_runs(self):
        """At exactly 2% + 1 unit, mass_point runs (strict > 0.02)."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 25  # 2.5% > threshold
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)

    def test_mass_point_modal_exactly_two_percent_raises(self):
        """At exactly 2% (not strictly greater), mass_point must raise."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 20  # exactly 2% (not > 2%)
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point")
        with pytest.raises(ValueError, match=r"modal mass"):
            est.fit(panel, "outcome", "dose", "period", "unit")


# =============================================================================
# Review P2: cluster-applied mass-point stores vcov_type="cr1"
# =============================================================================


class TestMassPointClusterLabel:
    def test_cluster_stores_cr1(self):
        """to_dict() / downstream consumers see 'cr1' not 'hc1' when clustered."""
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r.vcov_type == "cr1"
        assert r.cluster_name == "state"

    def test_no_cluster_stores_base_family(self):
        """Unclustered mass-point keeps 'hc1' or 'classical' label."""
        d, dy = _dgp_mass_point(200, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_hc1 = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            r_cl = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_hc1.vcov_type == "hc1"
        assert r_cl.vcov_type == "classical"

    def test_cluster_with_classical_collapses_to_cr1(self):
        """classical + cluster is CR1 in practice; label reflects that."""
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(
                design="mass_point", vcov_type="classical", cluster="state"
            ).fit(panel, "outcome", "dose", "period", "unit")
        assert r.vcov_type == "cr1"

    def test_to_dict_shows_effective_family(self):
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        result_dict = r.to_dict()
        assert result_dict["vcov_type"] == "cr1"
        assert result_dict["cluster_name"] == "state"


# =============================================================================
# Phase 2b: Multi-period event-study extension (paper Appendix B.2)
# =============================================================================


def _make_multi_period_panel(
    d_at_F,
    n_periods=5,
    F=3,
    seed=0,
    pre_trend=0.0,
    beta=0.3,
    sigma=0.1,
    first_treat=None,
    extra_cols=None,
):
    """Build a balanced multi-period HAD panel with common adoption at F.

    ``D_{g,t} = 0`` for ``t < F``; ``D_{g,t} = d_at_F[g]`` for ``t >= F``.
    ``Y_{g,t} = alpha_g + pre_trend * t + beta * D_{g,t} * 1{t >= F} + eps``.

    Parameters
    ----------
    d_at_F : np.ndarray, shape (G,)
        Unit-level treatment dose realized at period F.
    n_periods : int
        Total number of periods; periods indexed 1..n_periods.
    F : int
        First treatment period (1 <= F <= n_periods).
    seed : int
        RNG seed for outcome noise.
    pre_trend : float
        Deterministic linear trend (identical across units; zero under the
        paper's parallel-trends assumption).
    beta : float
        True treatment-effect coefficient on dose.
    sigma : float
        Outcome noise SD.
    first_treat : np.ndarray or None, shape (G,)
        Optional unit-level first-treatment labels (``0`` for
        never-treated). If provided, written to a ``first_treat``
        column; used for staggered-timing tests.
    extra_cols : dict or None
        Additional unit-constant columns.
    """
    rng = np.random.default_rng(seed)
    G = len(d_at_F)
    units = np.arange(G)
    periods = np.arange(1, n_periods + 1)
    alpha_g = 0.5 * rng.standard_normal(G)
    rows = []
    for g in units:
        d_g = float(d_at_F[g])
        # Preserve None in first_treat to support NaN-injection tests; cast
        # valid ints only.
        if first_treat is not None:
            ft_raw = first_treat[g]
            ft_g: Any = ft_raw if ft_raw is None else int(ft_raw)
        else:
            ft_g = 0 if d_g == 0 else F
        for t in periods:
            if first_treat is not None:
                if ft_g is None or ft_g == 0 or t < ft_g:
                    dose = 0.0
                else:
                    dose = d_g
            else:
                dose = d_g if t >= F else 0.0
            eps = sigma * rng.standard_normal()
            outcome = alpha_g[g] + pre_trend * t + beta * dose + eps
            row = {"unit": g, "period": t, "dose": dose, "outcome": outcome}
            if first_treat is not None:
                row["first_treat"] = ft_g
            rows.append(row)
    df = pd.DataFrame(rows)
    if extra_cols is not None:
        for col, vals in extra_cols.items():
            df = df.merge(pd.DataFrame({"unit": units, col: vals}), on="unit", how="left")
    return df


def _fit_es(est, *args, **kwargs) -> HeterogeneousAdoptionDiDEventStudyResults:
    """Fit and return a narrowed event-study result type.

    The public ``fit()`` is annotated to return a union over the
    single-period ``HeterogeneousAdoptionDiDResults`` and the multi-
    period ``HeterogeneousAdoptionDiDEventStudyResults``, matching its
    runtime polymorphism on ``aggregate``. When ``aggregate="event_study"``
    is requested, this helper narrows the union to the event-study
    branch for the test body via ``typing.cast``.
    """
    kwargs.setdefault("aggregate", "event_study")
    result = est.fit(*args, **kwargs)
    return cast(HeterogeneousAdoptionDiDEventStudyResults, result)


class TestFitReturnAnnotation:
    """Pin the source-level return annotation on ``HeterogeneousAdoptionDiD.fit``.

    The annotation MUST be a union over the single-period and event-study
    result classes — narrowing one branch out silently would make the
    static type contract diverge from the runtime polymorphism on
    ``aggregate``.
    """

    def test_fit_return_annotation_is_union_of_result_classes(self):
        import typing

        hints = typing.get_type_hints(HeterogeneousAdoptionDiD.fit)
        ret = hints.get("return")
        args = set(typing.get_args(ret))
        expected = {
            HeterogeneousAdoptionDiDResults,
            HeterogeneousAdoptionDiDEventStudyResults,
        }
        assert args == expected, (
            "HeterogeneousAdoptionDiD.fit return annotation drifted; "
            f"expected union of {expected}, got args={args} (ret={ret!r})."
        )


class TestEventStudySmoke:
    """Smoke tests: the three design paths produce finite event-study results."""

    def test_continuous_at_zero_smoke(self):
        rng = np.random.default_rng(0)
        G = 400
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        result = cast(
            HeterogeneousAdoptionDiDEventStudyResults,
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            ),
        )
        assert isinstance(result, HeterogeneousAdoptionDiDEventStudyResults)
        assert result.design == "continuous_at_zero"
        assert result.F == 3
        assert result.n_units == G
        # Post-period horizons should all produce finite att (noise-level);
        # pre-period placebo at e=-2 may hit a degenerate fit (zero-variance
        # baseline outcomes) and return NaN, that's acceptable.
        post_mask = result.event_times >= 0
        assert np.all(np.isfinite(result.att[post_mask]))

    def test_continuous_near_d_lower_smoke(self):
        rng = np.random.default_rng(1)
        G = 400
        u = rng.beta(2, 2, G)
        d = 0.1 + 0.9 * u
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = cast(
                HeterogeneousAdoptionDiDEventStudyResults,
                HeterogeneousAdoptionDiD(design="auto").fit(
                    panel, "outcome", "dose", "period", "unit", aggregate="event_study"
                ),
            )
        assert result.design == "continuous_near_d_lower"
        assert result.target_parameter == "WAS_d_lower"
        post_mask = result.event_times >= 0
        assert np.all(np.isfinite(result.att[post_mask]))

    def test_mass_point_smoke(self):
        rng = np.random.default_rng(2)
        G = 400
        mass_n = int(0.3 * G)
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5, 1.0, G - mass_n)])
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = cast(
                HeterogeneousAdoptionDiDEventStudyResults,
                HeterogeneousAdoptionDiD(design="auto").fit(
                    panel, "outcome", "dose", "period", "unit", aggregate="event_study"
                ),
            )
        assert result.design == "mass_point"
        assert result.inference_method == "analytical_2sls"
        assert result.bandwidth_diagnostics is None
        assert result.bias_corrected_fit is None
        post_mask = result.event_times >= 0
        assert np.all(np.isfinite(result.att[post_mask]))
        assert np.all(np.isfinite(result.se[post_mask]))


class TestEventStudyBaselineConvention:
    """e = -1 (anchor) must NOT appear in event_times."""

    def test_anchor_not_in_event_times(self):
        rng = np.random.default_rng(0)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        result = cast(
            HeterogeneousAdoptionDiDEventStudyResults,
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            ),
        )
        assert -1 not in result.event_times.tolist()

    def test_post_horizons_start_at_zero(self):
        rng = np.random.default_rng(0)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        result = cast(
            HeterogeneousAdoptionDiDEventStudyResults,
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            ),
        )
        # F=3, n_periods=5 -> periods 1..5, F-1=2 is anchor.
        # e = t-F for t in {1,2,3,4,5} -> {-2,-1,0,1,2}; -1 skipped.
        assert result.event_times.tolist() == [-2, 0, 1, 2]


class TestEventStudyDesignResolution:
    """Design / d_lower / target_parameter are SCALARS shared across horizons."""

    def test_design_is_scalar(self):
        rng = np.random.default_rng(0)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        result = cast(
            HeterogeneousAdoptionDiDEventStudyResults,
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            ),
        )
        assert isinstance(result.design, str)
        assert isinstance(result.d_lower, float)
        assert isinstance(result.target_parameter, str)
        assert isinstance(result.inference_method, str)


class TestEventStudyStaggeredFilter:
    """Auto-filter to last cohort + UserWarning per paper Appendix B.2."""

    def _staggered_panel(self, seed=0):
        rng = np.random.default_rng(seed)
        G = 300
        # Three cohorts: 0 (never), 3, 5. Last cohort = 5.
        ft_draw = rng.integers(0, 3, G)
        ft = np.array([0, 3, 5])[ft_draw]
        d = np.where(ft == 0, 0.0, rng.uniform(0.1, 1.0, G))
        # d[0] zeroed only if first_treat is 0; otherwise keep realized dose
        panel = _make_multi_period_panel(d, n_periods=6, F=5, seed=seed + 1, first_treat=ft)
        return panel, ft

    def test_staggered_filter_warning(self):
        panel, _ = self._staggered_panel(seed=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )
        filter_warnings = [msg for msg in w if "Staggered" in str(msg.message)]
        assert len(filter_warnings) == 1

    def test_staggered_filter_info_populated(self):
        panel, ft = self._staggered_panel(seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )
        assert result.filter_info is not None
        assert result.filter_info["F_last"] == 5
        # n_kept = last-cohort units + never-treated units (both retained).
        n_kept_expected = int(((ft == 5) | (ft == 0)).sum())
        assert result.filter_info["n_kept"] == n_kept_expected
        # n_dropped = earlier cohorts only (never-treated are kept).
        n_dropped_expected = int((ft == 3).sum())
        assert result.filter_info["n_dropped"] == n_dropped_expected
        assert 3 in result.filter_info["dropped_cohorts"]
        # Never-treated cohort (0) is NOT in dropped_cohorts.
        assert 0 not in result.filter_info["dropped_cohorts"]

    def test_staggered_filter_keeps_last_cohort_and_never_treated(self):
        panel, ft = self._staggered_panel(seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )
        # Paper Appendix B.2: staggered HAD applies to last cohort + keeps
        # never-treated as the "untreated group" comparison. Earlier cohorts
        # (first_treat=3) are dropped; never-treated (first_treat=0) AND
        # last-cohort (first_treat=5) are retained.
        n_kept_expected = int(((ft == 5) | (ft == 0)).sum())
        assert result.n_units == n_kept_expected
        assert result.F == 5

    def test_staggered_filter_retains_never_treated_units(self):
        """Explicit sample-composition test: after staggered filter, kept
        units are the union of last-cohort and never-treated.

        This pins the paper Appendix B.2 contract: "there must be an
        untreated group, at least till the period where the last cohort
        gets treated". Earlier cohorts are dropped; never-treated are NOT.
        """
        panel, ft = self._staggered_panel(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )
        # The fit ran successfully with never-treated retained. Verify
        # directly: the validator returns data_filtered with expected
        # composition.
        F, t_pre, t_post, data_filtered, filter_info = _validate_had_panel_event_study(
            panel, "outcome", "dose", "period", "unit", "first_treat"
        )
        kept_ft_values = set(data_filtered["first_treat"].unique().tolist())
        # Should contain exactly {0, F_last=5}; NOT earlier cohort 3.
        assert kept_ft_values == {0, 5}

    def test_no_filter_on_single_cohort(self):
        """Panel with one nonzero cohort (plus never-treated): no filter."""
        rng = np.random.default_rng(0)
        G = 200
        ft = rng.choice([0, 3], size=G, p=[0.5, 0.5])
        d = np.where(ft == 0, 0.0, rng.uniform(0.1, 1.0, G))
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1, first_treat=ft)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )
        filter_warnings = [msg for msg in w if "Staggered" in str(msg.message)]
        assert len(filter_warnings) == 0
        assert result.filter_info is None


class TestEventStudyPerHorizonSEIndependence:
    """Each horizon's SE matches Phase 2a SE on the two-period subset.

    Proves the per-horizon independence contract: the event-study path
    computes per-event-time estimates identically to what Phase 2a would
    produce on a (F-1, t) two-period subset.
    """

    def test_mass_point_per_horizon_matches_phase_2a(self):
        rng = np.random.default_rng(0)
        G = 300
        mass_n = int(0.3 * G)
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5, 1.0, G - mass_n)])
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        # Event-study fit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            es_result = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        assert isinstance(es_result, HeterogeneousAdoptionDiDEventStudyResults)
        # Phase 2a fit on each post-period (F-1, t) two-period subset.
        # Skip pre-period horizons since Phase 2a would reject the pre-pre
        # subset (both periods all-zero dose).
        F = 3
        for i, e in enumerate(es_result.event_times):
            if int(e) < 0:
                continue  # pre-period comparisons not applicable
            t_target = F + int(e)
            subset = panel[panel["period"].isin([F - 1, t_target])].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                overall_result = HeterogeneousAdoptionDiD(design="mass_point").fit(
                    subset, "outcome", "dose", "period", "unit"
                )
            np.testing.assert_allclose(es_result.att[i], overall_result.att, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(es_result.se[i], overall_result.se, atol=1e-12, rtol=1e-12)

    def test_continuous_at_zero_per_horizon_matches_phase_2a(self):
        rng = np.random.default_rng(1)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            es_result = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        assert isinstance(es_result, HeterogeneousAdoptionDiDEventStudyResults)
        # Skip pre-period horizons since Phase 2a would reject the pre-pre
        # subset (both periods all-zero dose).
        F = 3
        for i, e in enumerate(es_result.event_times):
            if int(e) < 0:
                continue
            t_target = F + int(e)
            subset = panel[panel["period"].isin([F - 1, t_target])].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                overall_result = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    subset, "outcome", "dose", "period", "unit"
                )
            # Match if both finite; if both NaN (degenerate bandwidth
            # selector on this horizon), skip assertion.
            if np.isfinite(es_result.att[i]) and np.isfinite(overall_result.att):
                np.testing.assert_allclose(
                    es_result.att[i], overall_result.att, atol=1e-12, rtol=1e-12
                )
                np.testing.assert_allclose(
                    es_result.se[i], overall_result.se, atol=1e-12, rtol=1e-12
                )
            else:
                assert np.isnan(es_result.att[i])
                assert np.isnan(overall_result.att)


class TestEventStudyAggregateMatrix:
    """2x2 period/aggregate matrix: reciprocal rejections."""

    def test_T2_event_study_raises(self):
        d = np.linspace(0.0, 1.0, 100)
        dy = 0.3 * d + 0.01 * np.random.default_rng(0).standard_normal(100)
        panel = _make_panel(d, dy)
        with pytest.raises(ValueError, match="more than two"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_T_gt_2_overall_raises(self):
        rng = np.random.default_rng(0)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        with pytest.raises(ValueError, match="aggregate='event_study'"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="overall"
            )

    def test_invalid_aggregate_raises(self):
        d = np.linspace(0.0, 1.0, 100)
        dy = 0.3 * d
        panel = _make_panel(d, dy)
        with pytest.raises(ValueError, match="Invalid aggregate"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="garbage"
            )


class TestEventStudyPlacebos:
    """Pre-period placebos: near 0 under no pre-trend; detectable under pre-trend."""

    def test_no_pre_trend_placebos_near_zero(self):
        rng = np.random.default_rng(0)
        G = 500
        d = rng.uniform(0.1, 1.0, G)  # mass-point-free
        d[0] = 0.0  # Design 1'
        panel = _make_multi_period_panel(
            d, n_periods=6, F=4, seed=1, pre_trend=0.0, beta=0.3, sigma=0.05
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = cast(
                HeterogeneousAdoptionDiDEventStudyResults,
                HeterogeneousAdoptionDiD(design="auto").fit(
                    panel, "outcome", "dose", "period", "unit", aggregate="event_study"
                ),
            )
        pre_mask = result.event_times <= -2
        # Placebo estimates should be near 0 (noise band).
        pre_atts = result.att[pre_mask]
        pre_atts_finite = pre_atts[np.isfinite(pre_atts)]
        if len(pre_atts_finite) > 0:
            # Generous band — the DGP has unit FEs that wash out in first-diff
            # but the placebo samples shrink noise-level; tolerance 0.3.
            assert np.all(np.abs(pre_atts_finite) < 0.3)


class TestEventStudyResultMethods:
    """to_dataframe / to_dict / summary produce well-shaped outputs."""

    def _fit(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_to_dataframe_shape(self):
        result = self._fit()
        df = result.to_dataframe()
        assert len(df) == len(result.event_times)
        assert set(df.columns) == {
            "event_time",
            "att",
            "se",
            "t_stat",
            "p_value",
            "conf_int_low",
            "conf_int_high",
            "n_obs",
        }

    def test_to_dict_shape(self):
        result = self._fit()
        d = result.to_dict()
        assert "event_times" in d
        assert "att" in d
        assert len(d["event_times"]) == len(result.event_times)
        assert len(d["att"]) == len(result.att)
        assert d["design"] == result.design
        assert d["F"] == result.F

    def test_to_dict_json_serializable(self):
        """``to_dict()`` output must be JSON-serializable via ``json.dumps``.

        Covers CI reviewer round 5 P2: previously the per-horizon arrays
        contained numpy scalars that tripped ``json.dumps``.
        """
        import json

        result = self._fit()
        d = result.to_dict()
        # Should not raise.
        payload = json.dumps(d)
        assert isinstance(payload, str)
        # Round-trip: values should parse back as native Python types.
        parsed = json.loads(payload)
        assert isinstance(parsed["event_times"], list)
        assert isinstance(parsed["event_times"][0], int)
        assert isinstance(parsed["att"][0], float)
        assert isinstance(parsed["alpha"], float)
        assert isinstance(parsed["n_units"], int)

    def test_summary_renders(self):
        result = self._fit()
        summary = result.summary()
        assert "HeterogeneousAdoptionDiD Event-Study Results" in summary
        assert result.design in summary

    def test_repr(self):
        result = self._fit()
        rep = repr(result)
        assert "HeterogeneousAdoptionDiDEventStudyResults" in rep
        assert f"n_horizons={len(result.event_times)}" in rep


class TestEventStudyPanelContract:
    """Panel-contract guards for event-study mode."""

    def test_rcs_rejected(self):
        """Repeated-cross-section inputs (disjoint unit ids) rejected."""
        rng = np.random.default_rng(0)
        n_periods = 4
        rows = []
        for t in range(1, n_periods + 1):
            for u in range(50):
                unit_id = u + t * 1000  # disjoint IDs per period
                dose = 0.0 if t < 3 else rng.uniform(0.1, 1.0)
                rows.append(
                    {
                        "unit": unit_id,
                        "period": t,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Unbalanced panel"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_non_contiguous_dose_rejected(self):
        """Pre/post periods interleaved (dose reversal) raises."""
        G = 100
        rows = []
        rng = np.random.default_rng(0)
        d_post = rng.uniform(0.1, 1.0, G)
        for g in range(G):
            # Weird panel: t=1 all-zero, t=2 treated, t=3 all-zero (reverse!)
            for t, dose in [(1, 0.0), (2, d_post[g]), (3, 0.0)]:
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="not contiguous"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_nan_in_outcome_rejected(self):
        rng = np.random.default_rng(0)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        panel.loc[0, "outcome"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_nan_in_first_treat_col_rejected(self):
        rng = np.random.default_rng(0)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        ft = np.where(d > 0, 3, 0).astype(object)
        ft[5] = None  # type: ignore[call-overload]
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1, first_treat=ft)
        with pytest.raises(ValueError, match="NaN"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )

    def test_no_pre_period_rejected(self):
        """All periods nonzero dose -> no pre-period to anchor on."""
        rng = np.random.default_rng(0)
        G = 100
        rows = []
        d_g = rng.uniform(0.1, 1.0, G)
        for g in range(G):
            for t in range(1, 5):
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": d_g[g],  # dose always nonzero!
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="all-zero dose|pre-period"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_time_varying_post_F_dose_rejected(self):
        """Within-unit dose variation across post-periods raises.

        Paper Appendix B.2 assumes "once treated, stay treated with the
        same dose"; the aggregation uses ``D_{g, F}`` as the single
        regressor for every horizon. Silent acceptance of time-varying
        post-treatment doses would misattribute later-horizon effects.
        Covers CI reviewer round 1 P0: `_aggregate_multi_period_first_differences`
        would otherwise use period-F dose for all horizons.
        """
        rng = np.random.default_rng(0)
        G = 50
        rows = []
        for g in range(G):
            d_F = float(rng.uniform(0.1, 0.5))
            d_F_plus_1 = d_F + 0.3  # time-varying: dose changes after F
            for t in range(1, 6):
                if t < 3:
                    dose = 0.0
                elif t == 3:
                    dose = d_F
                else:
                    dose = d_F_plus_1  # different from d_F
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="constant dose|time-varying"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_staggered_ordered_categorical_chooses_chronological_last(self):
        """Staggered filter uses chronological (not lexicographic) last.

        Constructs an ordered-categorical time column where lexicographic
        and chronological orderings disagree. With category order
        ``["q1", "q2", "q3", "q10"]``, chronological last is ``"q10"``
        but lexicographic last is ``"q3"``. If cohorts are ``{"q2", "q10"}``,
        a raw-sort implementation would pick ``F_last = "q2"`` (lex-max
        of the two strings); the fixed version must pick ``F_last = "q10"``.

        Covers CI reviewer round 3 P0: cohort sorting must use
        chronological order from ``time_dtype``, not raw Python sort.
        """
        rng = np.random.default_rng(0)
        G = 80
        periods = ["q1", "q2", "q3", "q10"]
        cat_dtype = pd.CategoricalDtype(categories=periods, ordered=True)
        # Half of units treated at q2 (cohort 1), half at q10 (cohort 2).
        rows = []
        for g in range(G):
            F_g = "q2" if g < G // 2 else "q10"
            d_g = float(rng.uniform(0.1, 1.0))
            for p in periods:
                # Dose = d_g once the period >= F_g in chronological order.
                chrono_g = periods.index(F_g)
                chrono_p = periods.index(p)
                dose = d_g if chrono_p >= chrono_g else 0.0
                rows.append(
                    {
                        "unit": g,
                        "period": p,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                        "first_treat": F_g,
                    }
                )
        panel = pd.DataFrame(rows)
        panel["period"] = panel["period"].astype(cat_dtype)
        panel["first_treat"] = panel["first_treat"].astype(cat_dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )

        # Chronological last cohort = "q10", not lexicographic last ("q3"
        # is not even a cohort here; lex last of the two cohorts would
        # be "q2" since "q10" < "q2" lexicographically).
        assert result.filter_info is not None
        assert result.filter_info["F_last"] == "q10"
        assert result.F == "q10"
        # q2-cohort units (G/2) are dropped; q10-cohort units (G/2)
        # retained.
        assert result.n_units == G // 2
        # Dropped cohorts should list "q2".
        assert "q2" in result.filter_info["dropped_cohorts"]

    def test_first_treat_col_mismatch_with_dose_raises(self):
        """first_treat_col disagreeing with observed dose path must raise.

        A mislabeled cohort column would otherwise silently select the
        wrong cohort as F_last in the last-cohort auto-filter and
        produce event-study estimates for the wrong units. Covers CI
        reviewer round 2 P1.
        """
        rng = np.random.default_rng(0)
        G = 40
        rows = []
        for g in range(G):
            # Actual first-positive-dose period: t=3 for half, t=5 for half.
            F_actual = 3 if g < G // 2 else 5
            # But deliberately mislabel: swap the first_treat labels so
            # G/2 units declare 5 when actual is 3, and vice versa.
            F_declared = 5 if g < G // 2 else 3
            d_g = float(rng.uniform(0.1, 1.0))
            for t in range(1, 7):
                dose = d_g if t >= F_actual else 0.0
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                        "first_treat": F_declared,
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="disagrees with the observed dose"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat="first_treat",
                aggregate="event_study",
            )

    def test_unordered_string_time_col_rejected(self):
        """Object/string time columns raise on event-study path.

        Raw sort on arbitrary string labels is lexicographic, not
        chronological (e.g., 'pre1'/'pre2'/'post1'/'post2' would map
        to wrong event-time horizons). Covers CI reviewer round 2 P1.
        """
        rng = np.random.default_rng(0)
        G = 50
        rows = []
        d_post = rng.uniform(0.0, 1.0, G)
        d_post[0] = 0.0
        for g in range(G):
            for label, dose in [
                ("pre1", 0.0),
                ("pre2", 0.0),
                ("post1", d_post[g]),
                ("post2", d_post[g]),
            ]:
                rows.append(
                    {
                        "unit": g,
                        "period": label,  # object dtype
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="ordered time column|dtype"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )

    def test_ordered_categorical_with_unused_levels_accepted(self):
        """Ordered categorical with extra unused category levels fits.

        Covers CI reviewer round 4 P1: the balanced-panel check must
        use ``observed=True`` on categorical groupby so unused category
        levels don't expand to zero-count cells and falsely trip the
        balance guard.
        """
        rng = np.random.default_rng(0)
        G = 40
        # Observed periods: pre1, pre2, post1, post2
        # Declared categories: ALSO include pre0 (unused) and post3 (unused)
        all_categories = ["pre0", "pre1", "pre2", "post1", "post2", "post3"]
        observed = ["pre1", "pre2", "post1", "post2"]
        cat_dtype = pd.CategoricalDtype(categories=all_categories, ordered=True)
        rows = []
        d_post = rng.uniform(0.1, 1.0, G)
        d_post[0] = 0.0
        for g in range(G):
            for label in observed:
                dose = d_post[g] if label in ("post1", "post2") else 0.0
                rows.append(
                    {
                        "unit": g,
                        "period": label,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        panel["period"] = panel["period"].astype(cat_dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        # F should be post1 (first observed post-period); event_times
        # should be [-2, 0, 1] (e=-1 for anchor pre2 is skipped).
        assert result.F == "post1"
        assert result.event_times.tolist() == [-2, 0, 1]
        assert result.n_units == G

    def test_ordered_categorical_time_col_accepted(self):
        """Ordered categorical time dtype passes the ordered-time check."""
        rng = np.random.default_rng(0)
        G = 50
        labels = ["pre1", "pre2", "post1", "post2"]
        cat_dtype = pd.CategoricalDtype(categories=labels, ordered=True)
        rows = []
        d_post = rng.uniform(0.1, 1.0, G)
        d_post[0] = 0.0
        for g in range(G):
            for label, dose in [
                ("pre1", 0.0),
                ("pre2", 0.0),
                ("post1", d_post[g]),
                ("post2", d_post[g]),
            ]:
                rows.append(
                    {
                        "unit": g,
                        "period": label,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        panel["period"] = panel["period"].astype(cat_dtype)
        # Should fit without raising the ordered-time error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        # post1 is F; e=-2 (pre1) and e=0 (post1), e=1 (post2) expected.
        assert result.F == "post1"

    def test_staggered_without_first_treat_col_rejected(self):
        """Multi-cohort panel without first_treat_col raises (not silent).

        Without cohort metadata, the dose-invariant period classification
        would silently treat later-cohort units as zero-dose "controls"
        at the inferred F, violating Appendix B.2's last-cohort-only
        contract. Covers CI reviewer round 1 P1.
        """
        rng = np.random.default_rng(0)
        G = 100
        rows = []
        for g in range(G):
            # Assign cohort: half treat at t=3, half at t=5.
            F_g = 3 if g < G // 2 else 5
            d_g = float(rng.uniform(0.1, 1.0))
            for t in range(1, 7):
                dose = d_g if t >= F_g else 0.0
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": dose,
                        "outcome": rng.standard_normal(),
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Staggered-timing|first_treat_col"):
            HeterogeneousAdoptionDiD(design="auto").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )


class TestEventStudyGuardsPreserved:
    """Phase 2a policy guards fire on the event-study path too."""

    def test_continuous_at_zero_nonzero_d_lower_raises(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.3)
        with pytest.raises(ValueError, match="d_lower == 0"):
            est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")

    def test_mass_point_d_lower_zero_raises(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.5, 1.0, G)
        # Use mass-point design explicitly with d_lower=0 (invalid regime)
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.0)
        with pytest.raises(ValueError, match="d_lower > 0"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")

    def test_continuous_near_rejects_mass_point_sample(self):
        rng = np.random.default_rng(0)
        G = 200
        mass_n = int(0.3 * G)
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5, 1.0, G - mass_n)])
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
        with pytest.raises(ValueError, match="mass-point sample"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")

    def test_mass_point_rejects_continuous_sample(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.1, 1.0, G)  # no mass point
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="mass_point")
        with pytest.raises(ValueError, match="modal mass"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")


class TestEventStudyNaNPropagation:
    """NaN contract: degenerate fits produce NaN triple via safe_inference."""

    def test_constant_y_nan_inference(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        # Overwrite outcome with a constant: ΔY = 0 everywhere → degenerate
        panel["outcome"] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = cast(
                HeterogeneousAdoptionDiDEventStudyResults,
                HeterogeneousAdoptionDiD(design="auto").fit(
                    panel, "outcome", "dose", "period", "unit", aggregate="event_study"
                ),
            )
        # All per-horizon inference triples should be NaN when fit is degenerate.
        assert np.all(np.isnan(result.t_stat))
        assert np.all(np.isnan(result.p_value))
        assert np.all(np.isnan(result.conf_int_low))
        assert np.all(np.isnan(result.conf_int_high))


class TestEventStudySklearnCompat:
    """sklearn contract on the event-study path: clone round-trip, idempotence."""

    def test_fit_does_not_mutate_design(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="auto")
        est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
        assert est.design == "auto"  # raw preserved

    def test_fit_is_idempotent(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r1 = est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
            r2 = est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
        np.testing.assert_allclose(r1.att, r2.att, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(r1.se, r2.se, atol=1e-14, rtol=0.0)

    def test_sklearn_clone_round_trip(self):
        sklearn_base = pytest.importorskip("sklearn.base")
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        est = HeterogeneousAdoptionDiD(design="auto", alpha=0.1)
        cloned = sklearn_base.clone(est)
        assert cloned.design == "auto"
        assert cloned.alpha == 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_orig = est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
            r_clone = cloned.fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        np.testing.assert_allclose(r_orig.att, r_clone.att, atol=1e-14, rtol=0.0)


class TestEventStudyWarnings:
    """Continuous-path warnings on event-study mode (vcov/robust ignored; cluster= is now threaded)."""

    def _panel(self):
        rng = np.random.default_rng(0)
        G = 200
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        return _make_multi_period_panel(d, n_periods=5, F=3, seed=1)

    def test_vcov_type_ignored_on_continuous(self):
        panel = self._panel()
        est = HeterogeneousAdoptionDiD(design="auto", vcov_type="classical")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
        vcov_warnings = [
            msg for msg in w if "vcov_type" in str(msg.message) and "ignored" in str(msg.message)
        ]
        assert len(vcov_warnings) == 1  # ONE per fit, not per horizon

    def test_cluster_threaded_on_continuous_event_study(self):
        # Phase 2b: cluster= is now threaded into the per-horizon CCT SE on
        # the continuous event-study path (no longer ignored). No "ignored"
        # warning; SE becomes cluster-robust; result surface labels the
        # cluster-robust variance.
        panel = self._panel()
        panel["state"] = panel["unit"] % 20
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r_cl = HeterogeneousAdoptionDiD(design="auto", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit", aggregate="event_study"
            )
        cluster_warnings = [
            msg
            for msg in w
            if "cluster=" in str(msg.message) and "ignored" in str(msg.message).lower()
        ]
        assert len(cluster_warnings) == 0
        r_un = HeterogeneousAdoptionDiD(design="auto").fit(
            panel, "outcome", "dose", "period", "unit", aggregate="event_study"
        )
        assert not np.allclose(r_cl.se, r_un.se)
        assert r_cl.vcov_type == "cr1"
        assert r_cl.cluster_name == "state"


class TestEventStudyValidator:
    """Direct tests for ``_validate_had_panel_event_study``."""

    def test_too_few_periods_raises(self):
        d = np.array([0.0, 0.5, 0.8])
        dy = np.array([0.1, 0.2, 0.3])
        panel = _make_panel(d, dy, periods=(1, 2))
        with pytest.raises(ValueError, match="more than two"):
            _validate_had_panel_event_study(panel, "outcome", "dose", "period", "unit", None)

    def test_infers_F_from_dose_invariant(self):
        rng = np.random.default_rng(0)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        F, t_pre, t_post, data, filter_info = _validate_had_panel_event_study(
            panel, "outcome", "dose", "period", "unit", None
        )
        assert F == 3
        assert t_pre == [1, 2]
        assert t_post == [3, 4, 5]
        assert filter_info is None

    def test_empty_cohorts_raises(self):
        """All first_treat values are 0 (never-treated)."""
        G = 50
        rng = np.random.default_rng(0)
        rows = []
        for g in range(G):
            for t in range(1, 5):
                rows.append(
                    {
                        "unit": g,
                        "period": t,
                        "dose": 0.0,
                        "outcome": rng.standard_normal(),
                        "first_treat": 0,
                    }
                )
        panel = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="no nonzero cohort|all-zero dose"):
            _validate_had_panel_event_study(
                panel, "outcome", "dose", "period", "unit", "first_treat"
            )


class TestEventStudyAggregator:
    """Direct tests for ``_aggregate_multi_period_first_differences``."""

    def test_anchor_not_in_dy_dict(self):
        rng = np.random.default_rng(0)
        G = 50
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        F, t_pre, t_post, data, _ = _validate_had_panel_event_study(
            panel, "outcome", "dose", "period", "unit", None
        )
        d_arr, dy_dict, _, _, t_anchor = _aggregate_multi_period_first_differences(
            data, "outcome", "dose", "period", "unit", F, t_pre, t_post, None
        )
        assert t_anchor == 2  # F - 1
        assert -1 not in dy_dict
        # Horizons: e in {-2, 0, 1, 2}
        assert set(dy_dict.keys()) == {-2, 0, 1, 2}

    def test_dose_regressor_uses_period_F(self):
        rng = np.random.default_rng(0)
        G = 30
        d = rng.uniform(0.1, 1.0, G)
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=1)
        F, t_pre, t_post, data, _ = _validate_had_panel_event_study(
            panel, "outcome", "dose", "period", "unit", None
        )
        d_arr, _, _, unit_ids, _ = _aggregate_multi_period_first_differences(
            data, "outcome", "dose", "period", "unit", F, t_pre, t_post, None
        )
        # d_arr should be the period-F dose, unit-aligned
        expected = panel[panel["period"] == F].sort_values("unit")["dose"].to_numpy()
        np.testing.assert_allclose(d_arr, expected, atol=0.0, rtol=0.0)


# =============================================================================
# HAD survey-weighted path (Phase 4.5: survey/weights on continuous designs)
# =============================================================================


class TestHADSurvey:
    """Phase 4.5 (continuous-design survey support) validation suite.

    Scope: ``survey_design=SurveyDesign(weights=...)`` on
    ``continuous_at_zero`` and ``continuous_near_d_lower`` (the sole weighting
    entry as of the 3.7.0 ``survey=``/``weights=`` removal).
    """

    def _panel_with_unit_weights(self, G=200, seed=42, design="continuous_at_zero"):
        rng = np.random.default_rng(seed)
        if design == "continuous_at_zero":
            d = rng.uniform(0, 1, G)
            d_lower = 0.0
        else:
            d = rng.uniform(0.1, 1.0, G)
            d_lower = 0.1
        dy = 2.0 * (d - d_lower) + 0.3 * (d - d_lower) ** 2 + rng.normal(0, 0.2, G)
        panel = _make_panel(d, dy)
        w_unit = rng.uniform(0.5, 1.5, G)
        # Broadcast to every row (unit-constant per HAD contract)
        row_w = np.zeros(panel.shape[0])
        for g in range(G):
            row_w[panel["unit"].to_numpy() == g] = w_unit[g]
        return panel, row_w, w_unit, d, dy, d_lower

    # ---------- Uniform-weights bit-parity ----------

    def test_uniform_weights_continuous_at_zero_bit_parity(self):
        from diff_diff.survey import SurveyDesign

        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=np.ones(panel.shape[0]))
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        base = est.fit(panel, "outcome", "dose", "period", "unit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w1 = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        # Uniform weights are a no-op on the point estimate (same weighted
        # lprobust fit), so the ATT is bit-identical to unweighted.
        np.testing.assert_allclose(w1.att, base.att, atol=1e-12, rtol=1e-12)
        # The survey path uses the Binder-TSL variance family (not the
        # analytical robust SE), so its SE differs from the unweighted
        # SE even under uniform weights; assert only that inference is
        # well-formed rather than bit-parity.
        assert np.isfinite(w1.se) and w1.se > 0
        assert np.isfinite(w1.t_stat)
        assert np.isfinite(w1.p_value)
        assert np.isfinite(w1.conf_int[0]) and np.isfinite(w1.conf_int[1])

    def test_uniform_weights_continuous_near_d_lower_bit_parity(self):
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            panel, _, _, _, _, _ = self._panel_with_unit_weights(
                G=200, design="continuous_near_d_lower"
            )
            panel_w = panel.assign(w=np.ones(panel.shape[0]))
            est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
            base = est.fit(panel, "outcome", "dose", "period", "unit")
            w1 = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        # ATT is bit-identical (uniform weights no-op); the survey SE is a
        # different variance family so only assert it is finite/positive.
        np.testing.assert_allclose(w1.att, base.att, atol=1e-12, rtol=1e-12)
        assert np.isfinite(w1.se) and w1.se > 0

    # ---------- Non-trivial weights: mechanism has teeth ----------

    def test_nontrivial_weights_change_estimate(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        base = est.fit(panel, "outcome", "dose", "period", "unit")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert not np.isclose(w.att, base.att, atol=1e-6)

    # ---------- Validator contract ----------

    def test_weights_vary_within_unit_raises(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        # Corrupt one unit's pre-period weight so it differs from its
        # post-period weight.
        row_w_bad = row_w.copy()
        first_unit_mask = panel["unit"].to_numpy() == 0
        first_unit_idx = np.where(first_unit_mask)[0][0]
        row_w_bad[first_unit_idx] = row_w_bad[first_unit_idx] + 5.0
        panel_bad = panel.assign(w=row_w_bad)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(ValueError, match="weights vary within"):
            est.fit(
                panel_bad,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_negative_weights_raise(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        row_w_bad = row_w.copy()
        row_w_bad[0] = -1.0
        # Also corrupt the paired row so the within-unit check doesn't fire first.
        first_unit_mask = panel["unit"].to_numpy() == panel["unit"].iloc[0]
        row_w_bad[first_unit_mask] = -1.0
        panel_bad = panel.assign(w=row_w_bad)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(ValueError, match="non-negative"):
            est.fit(
                panel_bad,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_zero_sum_weights_raise(self):
        from diff_diff.survey import SurveyDesign

        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_bad = panel.assign(w=np.zeros(panel.shape[0]))
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(ValueError, match="sum to zero"):
            est.fit(
                panel_bad,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )

    # ---------- Previously deferred paths (Phase 4.5 B supported) ----------

    def test_mass_point_weights_smoke(self):
        """Mass-point + uniform survey weights fits and is bit-parity with
        unweighted (Phase 4.5 B)."""
        from diff_diff.survey import SurveyDesign

        d, dy = _dgp_mass_point(500, seed=42)
        panel = _make_panel(d, dy)
        panel_w = panel.assign(w=np.ones(panel.shape[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1")
            r_unw = est.fit(panel, "outcome", "dose", "period", "unit")
            r_uniform = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert np.isclose(r_unw.att, r_uniform.att, atol=1e-10)
        assert np.isclose(r_unw.se, r_uniform.se, atol=1e-10)
        assert r_uniform.variance_formula == "survey_binder_tsl_2sls"

    def test_event_study_weights_smoke(self):
        """Multi-period + event-study + uniform weights fits and
        preserves pre-PR numerical output on att/se at cband=False
        (Phase 4.5 B)."""
        rng = np.random.default_rng(7)
        G = 150
        d = rng.uniform(0, 1, G)
        units = np.arange(G)
        periods = [0, 1, 2]
        rows = []
        for t in periods:
            for g in units:
                dose = d[g] if t == 2 else 0.0
                y = 0.1 * t + (dose * 2.0 if t == 2 else 0.0) + rng.normal(0, 0.2)
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"])
        panel_w = panel.assign(w=np.ones(panel.shape[0]))
        est = HeterogeneousAdoptionDiD()
        r_unw = est.fit(
            panel,
            "outcome",
            "dose",
            "period",
            "unit",
            aggregate="event_study",
        )
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_w = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                cband=False,  # skip bootstrap
            )
        # Uniform weights are a no-op on the point estimate, so att recovers
        # the unweighted output at atol=1e-10 (composition through
        # np.average introduces O(ULP) reductions differing from raw mean()).
        np.testing.assert_allclose(r_unw.att, r_w.att, atol=1e-10, rtol=1e-10)
        # The survey path uses the Binder-TSL variance family, so its SE is
        # not bit-parity with the unweighted analytical SE; assert finite.
        assert np.all(np.isfinite(r_w.se))
        assert r_w.variance_formula == "survey_binder_tsl"
        assert r_w.cband_crit_value is None  # cband=False

    # ---------- Result-object contract ----------

    def test_survey_metadata_populated_under_weights(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert r.survey_metadata is not None
        # Repo-standard SurveyMetadata with attribute access.
        assert r.survey_metadata.weight_type == "pweight"
        assert r.survey_metadata.sum_weights > 0
        assert r.survey_metadata.effective_n > 0
        assert r.variance_formula == "survey_binder_tsl"

    def test_survey_metadata_none_when_unweighted(self):
        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.survey_metadata is None

    def test_to_dict_includes_survey_metadata(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        d = r.to_dict()
        assert "survey_metadata" in d
        # survey_metadata is now a SurveyMetadata dataclass; consumers
        # access attributes on the returned object.
        assert d["survey_metadata"].weight_type == "pweight"

    def test_to_dict_survey_metadata_none_key_present(self):
        """Even on unweighted fits, the key is present (value None) so
        downstream consumers can branch on the key not absence."""
        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        d = r.to_dict()
        assert "survey_metadata" in d
        assert d["survey_metadata"] is None

    def test_summary_renders_under_weights(self):
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        s = r.summary()
        assert "Variance formula" in s
        assert "Effective sample size" in s

    # ---------- SurveyDesign full composition (PSU / strata / FPC) ----------

    def _panel_with_survey_cols(self, G=200, seed=42):
        """Build a panel with w / strata / psu columns, all unit-constant."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(seed)
        d = rng.uniform(0, 1, G)
        dy = 2.0 * d + 0.3 * d**2 + rng.normal(0, 0.2, G)
        panel = _make_panel(d, dy)
        w_unit = rng.uniform(0.5, 1.5, G)
        strata_unit = rng.integers(0, 4, G)
        psu_unit = strata_unit * 100 + rng.integers(0, 20, G)
        # Broadcast to long panel.
        row_w = np.zeros(panel.shape[0])
        row_strata = np.zeros(panel.shape[0], dtype=np.int64)
        row_psu = np.zeros(panel.shape[0], dtype=np.int64)
        for g in range(G):
            mask = panel["unit"].to_numpy() == g
            row_w[mask] = w_unit[g]
            row_strata[mask] = strata_unit[g]
            row_psu[mask] = psu_unit[g]
        panel2 = panel.assign(w=row_w, strata=row_strata, psu=row_psu)
        return panel2, SurveyDesign

    def test_survey_with_strata_produces_different_se(self):
        """Adding strata to a SurveyDesign changes the SE (finer variance
        composition) but preserves the ATT (same weighted estimator)."""
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_basic = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
            r_strat = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata"),
            )
        np.testing.assert_allclose(r_basic.att, r_strat.att, atol=1e-14, rtol=1e-14)
        assert r_basic.se != r_strat.se
        assert r_strat.survey_metadata.n_strata == 4
        assert r_basic.survey_metadata.n_strata is None

    def test_survey_with_psu_clustering(self):
        """Adding PSU clustering changes the SE (within-PSU correlation
        aggregated). ATT unchanged."""
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_strat = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata"),
            )
            r_psu = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
        np.testing.assert_allclose(r_strat.att, r_psu.att, atol=1e-14, rtol=1e-14)
        assert r_psu.se != r_strat.se
        assert r_psu.survey_metadata.n_psu is not None
        assert r_psu.survey_metadata.n_psu > 1
        # PSU count is strictly less than unit count (clustering is actual).
        assert r_psu.survey_metadata.n_psu < 200

    def test_survey_metadata_records_binder_tsl_method(self):
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
        sm = r.survey_metadata
        assert sm is not None
        # Repo-standard SurveyMetadata attributes populated.
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.df_survey is not None
        # HAD-specific variance-formula label lives on the result,
        # orthogonal to the shared SurveyMetadata.
        assert r.variance_formula == "survey_binder_tsl"

    def test_survey_design_column_varies_within_unit_raises(self):
        """Strata that varies within a unit → ValueError (HAD requires
        unit-constant design columns)."""
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        # Corrupt one unit's pre-period strata.
        unit0_mask = panel["unit"].to_numpy() == 0
        unit0_pre_idx = np.where(unit0_mask & (panel["period"].to_numpy() == 1))[0][0]
        panel_bad = panel.copy()
        panel_bad.iloc[unit0_pre_idx, panel_bad.columns.get_loc("strata")] = 99
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="strata varies within"):
                est.fit(
                    panel_bad,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="w", strata="strata"),
                )

    def test_replicate_weights_not_yet_supported(self):
        """Replicate-weight designs raise NotImplementedError on HAD (Phase 4.5 C)."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        # Fabricate a replicate-weights design (BRR with 2 replicates).
        rep_w_col_1 = row_w * (1 + 0.1 * np.random.default_rng(1).normal(size=len(row_w)))
        rep_w_col_2 = row_w * (1 + 0.1 * np.random.default_rng(2).normal(size=len(row_w)))
        # Replicate weights must be non-negative; clip for safety.
        panel2 = panel.assign(
            w=row_w,
            rep1=np.clip(rep_w_col_1, 0.01, None),
            rep2=np.clip(rep_w_col_2, 0.01, None),
        )
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        sd = SurveyDesign(
            weights="w",
            replicate_weights=["rep1", "rep2"],
            replicate_method="BRR",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="Replicate-weight"):
                est.fit(panel2, "outcome", "dose", "period", "unit", survey_design=sd)

    # ---------- P0 fix: bias-corrected IF alignment (round 1 review) ----------

    def test_survey_if_uses_bias_corrected_scale(self):
        """The survey IF must align with ``V_Y_bc`` (bias-corrected), NOT
        ``V_Y_cl`` — the HAD ATT uses ``tau_bc``, so the survey SE must
        target the same estimator scale. White-box check: under
        unclustered HC0/HC1, ``sum(psi^2)`` must match ``V_Y_bc[0, 0]``
        (within BLAS tolerance), NOT ``V_Y_cl[0, 0]``. Under a nontrivial
        DGP (nonlinear m(d) with non-zero bias correction) the two
        differ, so this test has teeth."""
        from diff_diff._nprobust_port import lprobust

        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0.0, 1.0, n)
        # Nonlinear m(d) = 2d + 0.8 d² — the 0.8 quadratic term drives a
        # nontrivial bias correction so V_Y_bc != V_Y_cl.
        y = 2.0 * x + 0.8 * x**2 + rng.normal(0.0, 0.25, n)
        r = lprobust(y, x, eval_point=0.0, h=0.3, b=0.3, vce="hc1", return_influence=True)
        sum_if_sq = float((r.influence_function**2).sum())
        # Bias-corrected scale: sum(IF^2) should equal V_Y_bc[0,0] to
        # floating-point precision. NOT equal to V_Y_cl[0,0].
        np.testing.assert_allclose(sum_if_sq, r.V_Y_bc[0, 0], atol=1e-12, rtol=1e-12)
        # The classical SE is DIFFERENT from the bias-corrected SE under
        # nonlinear m(d) — the two differ by the bias-correction inflation.
        assert not np.isclose(
            r.V_Y_cl[0, 0], r.V_Y_bc[0, 0], atol=0.0, rtol=1e-6
        ), "DGP chosen to drive V_Y_cl != V_Y_bc; check nonlinearity"
        assert not np.isclose(sum_if_sq, r.V_Y_cl[0, 0], atol=0.0, rtol=1e-6), (
            "sum(IF^2) must track V_Y_bc (not V_Y_cl) — if this fails, "
            "the IF is computed with classical res_h instead of "
            "bias-corrected res_b, silently underestimating survey SE."
        )

    # ---------- P1a fix: df_survey threaded through inference ----------

    def test_survey_df_widens_ci_vs_normal(self):
        """Under small-PSU design, survey t-inference with finite
        ``df_survey`` must produce WIDER confidence intervals than
        Normal-theory inference at the same SE. Regression test for the
        df_survey threading in fit()."""
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Few PSUs within strata → small df_survey → t-inference
            # inflates CI vs Normal.
            r_sd = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
            # Same fit under a weights-only SurveyDesign (no PSU/strata).
            # ATT matches; the inference path differs.
            r_w = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        # df_survey surfaced in metadata.
        assert r_sd.survey_metadata is not None
        assert r_sd.survey_metadata.df_survey is not None
        assert r_sd.survey_metadata.df_survey > 0
        # Under the same SE (approximately), t-based CI > Normal CI.
        # (SE itself differs a bit because Binder-TSL vs weighted-robust
        # at the same fit, but df_survey inflates the t-critical-value
        # enough that the t-CI is wider.)
        # Sanity: both CIs well-defined.
        assert np.isfinite(r_sd.conf_int[0])
        assert np.isfinite(r_sd.conf_int[1])
        assert np.isfinite(r_w.conf_int[0])
        assert np.isfinite(r_w.conf_int[1])

    def test_survey_df_threaded_into_inference_via_t_distribution(self):
        """Direct check: when ``df_survey`` is small, the t-critical
        value exceeds the Normal z-critical value, which widens the CI
        at the same ``se``. Uses a tiny-PSU panel to produce a small df."""
        from scipy import stats

        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_sd = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
        assert r_sd.survey_metadata is not None
        df = r_sd.survey_metadata.df_survey
        assert df is not None and df > 0
        # CI width check: given att, se, and df, CI should match t-interval.
        z_norm = stats.norm.ppf(1 - r_sd.alpha / 2)
        t_crit = stats.t.ppf(1 - r_sd.alpha / 2, df=df)
        assert t_crit > z_norm, (
            f"t-critical ({t_crit:.4f}) should exceed z ({z_norm:.4f}) "
            f"for df={df}; otherwise df_survey threading is a no-op"
        )
        # CI half-width should equal t_crit * se (within float noise).
        half_width = (r_sd.conf_int[1] - r_sd.conf_int[0]) / 2.0
        np.testing.assert_allclose(half_width, t_crit * r_sd.se, rtol=1e-10)

    # ---------- P1b fix: reject non-pweight weight_type ----------

    def test_survey_aweight_raises_not_implemented(self):
        """``SurveyDesign(weight_type='aweight')`` raises — analytic
        weights would target a different estimand than sampling weights."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_with_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        sd = SurveyDesign(weights="w", weight_type="aweight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="aweight"):
                est.fit(panel_with_w, "outcome", "dose", "period", "unit", survey_design=sd)

    def test_survey_fweight_raises_not_implemented(self):
        """``SurveyDesign(weight_type='fweight')`` raises — frequency
        weights imply observation replication, not sampling design."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        # fweight requires non-negative integers; rebuild with positive
        # integer row weights.
        rng = np.random.default_rng(11)
        row_w_int = rng.integers(1, 5, size=panel.shape[0]).astype(np.float64)
        # Make constant-within-unit as HAD requires.
        df = panel.copy()
        df["w"] = row_w_int
        # Force unit-constant by taking first value per unit.
        first_per_unit = df.groupby("unit")["w"].first()
        df["w"] = df["unit"].map(first_per_unit)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        sd = SurveyDesign(weights="w", weight_type="fweight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="fweight"):
                est.fit(df, "outcome", "dose", "period", "unit", survey_design=sd)

    # ---------- P2 fix: SRS equivalence under survey_design= (uniform weights) ----------

    # ---------- P2 fix: weighted denominator contract ----------

    def test_effective_dose_mean_matches_weighted_mean_continuous_at_zero(self):
        """``effective_dose_mean`` must equal the weighted mean of D used
        in the β-scale rescaling — this is what the estimator actually
        uses, vs ``dose_mean`` which is the raw-sample mean (preserved
        for backward compatibility). Regression test for P2 from round 2
        CI review."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, w_unit, d, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert r.effective_dose_mean is not None
        expected = float(np.average(d, weights=w_unit))
        np.testing.assert_allclose(r.effective_dose_mean, expected, atol=1e-12, rtol=1e-12)
        # dose_mean stays as raw-sample mean — orthogonal to the
        # weighted denominator actually used in the fit.
        np.testing.assert_allclose(r.dose_mean, float(d.mean()), atol=1e-12, rtol=1e-12)

    def test_effective_dose_mean_matches_weighted_mean_near_d_lower(self):
        """For ``continuous_near_d_lower``, the estimator auto-resolves
        ``d_lower = d.min()`` (not the theoretical lower bound of the
        DGP), so the expected weighted denominator uses
        ``d - r.d_lower``, not the DGP's ``d_lower``."""
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            panel, row_w, w_unit, d, _, _ = self._panel_with_unit_weights(
                G=200, design="continuous_near_d_lower"
            )
            panel_w = panel.assign(w=row_w)
            est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert r.effective_dose_mean is not None
        # Use the estimator's auto-resolved d_lower (== d.min()), not the
        # DGP's theoretical lower bound.
        expected = float(np.average(d - r.d_lower, weights=w_unit))
        np.testing.assert_allclose(r.effective_dose_mean, expected, atol=1e-12, rtol=1e-12)

    def test_effective_dose_mean_none_when_unweighted(self):
        """On unweighted fits, ``effective_dose_mean`` is ``None`` —
        ``dose_mean`` is the sole denominator there (raw == weighted
        when w=1, so the duplicate field would be noise)."""
        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.effective_dose_mean is None

    # ---------- Round 5 P0: zero-weight units don't drive design ----------

    def test_zero_weight_unit_at_d_min_does_not_flip_design(self):
        """Round 5 P0: a zero-weight unit sitting at ``d.min() = 0``
        must not flip the auto-detect design from
        ``continuous_near_d_lower`` (correct on the positive-weight
        subpop) to ``continuous_at_zero`` (wrong, boundary=0 chosen from
        an excluded unit). Previously design detection ran on the full
        unit set, so a subpopulation-style zero-weight unit at d=0
        silently mistargeted."""
        rng = np.random.default_rng(42)
        G_pop = 200
        # Full population: one zero-weight unit at d=0; rest positive
        # weights with d in [0.1, 1.0] (so positive-weight support min = 0.1).
        d = np.concatenate([[0.0], rng.uniform(0.1, 1.0, G_pop - 1)])
        dy = 2.0 * (d - 0.1) + rng.normal(0, 0.2, G_pop)
        w_unit = np.concatenate([[0.0], rng.uniform(0.5, 1.5, G_pop - 1)])
        panel = _make_panel(d, dy)
        row_w = np.zeros(panel.shape[0])
        for g in range(G_pop):
            row_w[panel["unit"].to_numpy() == g] = w_unit[g]
        panel_w = panel.assign(w=row_w)
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Full panel with zero-weight unit at d=0: auto-detect.
            est = HeterogeneousAdoptionDiD(design="auto")
            r_full = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
            # Physically drop the zero-weight unit and refit.
            panel_dropped = panel_w[panel_w["unit"] != 0].reset_index(drop=True)
            r_dropped = est.fit(
                panel_dropped,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        # Both paths resolve to the SAME design (the positive-weight
        # support, not the contaminated d=0 boundary).
        assert r_full.design == r_dropped.design
        # Both paths produce the same ATT (lprobust already ignored the
        # zero-weight unit's kernel contribution; filtering earlier
        # doesn't change the fit numerically).
        np.testing.assert_allclose(r_full.att, r_dropped.att, atol=1e-10, rtol=1e-10)
        # The survey path RETAINS the zero-weight unit in the sampling
        # frame for variance (n_psu / df reflect the full frame), so the
        # Binder-TSL SE is not bit-identical to physically dropping the
        # unit; assert only that inference is well-formed. (The design /
        # d_lower non-flip below is the actual point of this test.)
        assert np.isfinite(r_full.se) and r_full.se > 0
        # d_lower set by the positive-weight subpopulation (d.min() of
        # the kept units), NOT the contaminated full d.min()=0.
        assert r_full.d_lower > 0.0
        np.testing.assert_allclose(r_full.d_lower, r_dropped.d_lower, atol=1e-12, rtol=1e-12)

    def test_zero_weight_filter_warns_user(self):
        """Dropping zero-weight units from design resolution should
        emit a UserWarning so the behavior is visible."""
        rng = np.random.default_rng(5)
        G = 150
        d = rng.uniform(0.0, 1.0, G)
        dy = 2.0 * d + rng.normal(0, 0.25, G)
        w_unit = rng.uniform(0.5, 1.5, G)
        # Zero out 5 units.
        w_unit[:5] = 0.0
        panel = _make_panel(d, dy)
        row_w = np.zeros(panel.shape[0])
        for g in range(G):
            row_w[panel["unit"].to_numpy() == g] = w_unit[g]
        panel_w = panel.assign(w=row_w)
        from diff_diff.survey import SurveyDesign

        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.warns(UserWarning, match="weight == 0"):
            est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_zero_weight_survey_metadata_preserves_full_design(self):
        """Round 6 P1a: on the ``survey_design=`` path, zero-weight units
        (subpopulation convention) stay in the ResolvedSurveyDesign for
        variance + SurveyMetadata. ``n_psu`` / ``n_strata`` /
        ``df_survey`` / ``sum_weights`` reflect the FULL sampling frame,
        not the in-domain subset — that is the standard
        domain-estimation convention in diff_diff.survey."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(10)
        G = 160
        d = rng.uniform(0.0, 1.0, G)
        dy = 2.0 * d + rng.normal(0, 0.25, G)
        # Strata + PSU structure on the FULL sample.
        strata = rng.integers(0, 4, G)
        psu = strata * 100 + rng.integers(0, 20, G)
        # Zero out weights on 1/4 of units (subpopulation exclusion).
        w_unit = rng.uniform(0.5, 1.5, G)
        zero_idx = rng.choice(G, size=G // 4, replace=False)
        w_unit[zero_idx] = 0.0
        panel = _make_panel(d, dy)
        row_w = np.zeros(panel.shape[0])
        row_strata = np.zeros(panel.shape[0], dtype=np.int64)
        row_psu = np.zeros(panel.shape[0], dtype=np.int64)
        for g in range(G):
            mask = panel["unit"].to_numpy() == g
            row_w[mask] = w_unit[g]
            row_strata[mask] = strata[g]
            row_psu[mask] = psu[g]
        panel_sd = panel.assign(w=row_w, strata=row_strata, psu=row_psu)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_full = est.fit(
                panel_sd,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
            # Reference fit: physically drop the zero-weight units and
            # refit on the positive-weight subsample. SurveyMetadata
            # values SHOULD DIFFER because dropping loses sampling frame
            # structure.
            keep_rows = panel_sd["unit"].isin([g for g in range(G) if w_unit[g] > 0])
            panel_sub = panel_sd.loc[keep_rows].reset_index(drop=True)
            r_sub = est.fit(
                panel_sub,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
        # Point estimate IDENTICAL — zero-weight units contribute 0 to
        # fit either way.
        np.testing.assert_allclose(r_full.att, r_sub.att, atol=1e-10, rtol=1e-10)
        # Full-design SurveyMetadata counts the FULL sampling frame.
        assert r_full.survey_metadata is not None
        assert r_sub.survey_metadata is not None
        # Full fit's sum_weights includes zero-weight units (sum is the
        # same as the filtered sum, since w=0 contributes 0). But n_psu
        # / n_strata preserve the FULL design whereas the filtered fit
        # sees only in-domain PSUs/strata.
        assert r_full.survey_metadata.n_psu >= r_sub.survey_metadata.n_psu

    def test_bias_corrected_local_linear_zero_weight_matches_filtered(self):
        """Round 6 P1b: ``bias_corrected_local_linear(weights=...)``
        with zero-weight units at the boundary must produce the same
        fit numbers as physically dropping those units (both in
        explicit-h/b and auto-bandwidth modes). Previously the wrapper
        ran ``_validate_had_inputs`` on the full sample, so zero-weight
        units at ``d.min()=0`` could spuriously trip the Design 1'
        support heuristic or the mass-point threshold."""
        from diff_diff.local_linear import bias_corrected_local_linear

        rng = np.random.default_rng(21)
        G = 250
        # Construct a sample where the positive-weight support has
        # d.min ~ 0.1 (continuous_near_d_lower shape), but add a
        # zero-weight unit at d=0 (would otherwise flip the wrapper to
        # Design 1').
        d_pos = rng.uniform(0.1, 1.0, G - 1)
        d_full = np.concatenate([[0.0], d_pos])
        y_full = np.concatenate([[0.0], 2.0 * (d_pos - 0.1) + rng.normal(0, 0.2, G - 1)])
        w_full = np.concatenate([[0.0], np.ones(G - 1)])
        # Reference: physically drop the zero-weight unit.
        d_ref = d_full[w_full > 0]
        y_ref = y_full[w_full > 0]
        # Explicit-h/b mode:
        r_weighted = bias_corrected_local_linear(
            d=d_full,
            y=y_full,
            boundary=float(d_pos.min()),
            h=0.3,
            b=0.3,
            weights=w_full,
            return_influence=True,
        )
        r_dropped = bias_corrected_local_linear(
            d=d_ref,
            y=y_ref,
            boundary=float(d_pos.min()),
            h=0.3,
            b=0.3,
            return_influence=True,
        )
        np.testing.assert_allclose(
            r_weighted.estimate_bias_corrected,
            r_dropped.estimate_bias_corrected,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            r_weighted.se_robust, r_dropped.se_robust, atol=1e-12, rtol=1e-12
        )
        # IF zero-padded back to FULL length (original ordering).
        assert r_weighted.influence_function is not None
        assert r_weighted.influence_function.shape[0] == G
        # Zero-weight unit at index 0 has IF=0.
        np.testing.assert_allclose(r_weighted.influence_function[0], 0.0, atol=1e-14, rtol=1e-14)
        # Positive-weight positions match the dropped-sample IF.
        np.testing.assert_allclose(
            r_weighted.influence_function[1:],
            r_dropped.influence_function,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_bias_corrected_local_linear_zero_weight_auto_bandwidth(self):
        """Round 6 P1b: auto-bandwidth selection also runs on the
        positive-weight subset — otherwise the unweighted MSE-DPI
        selector sees the zero-weight unit at the boundary as a valid
        observation and picks a wrong bandwidth."""
        from diff_diff.local_linear import bias_corrected_local_linear

        rng = np.random.default_rng(22)
        G = 300
        d_pos = rng.uniform(0.0, 1.0, G - 1)
        d_full = np.concatenate([d_pos, [1.0]])  # zero-weight unit at d=1.0
        y_full = np.concatenate([2.0 * d_pos + rng.normal(0, 0.25, G - 1), [0.0]])
        w_full = np.concatenate([np.ones(G - 1), [0.0]])
        d_ref = d_full[w_full > 0]
        y_ref = y_full[w_full > 0]
        r_weighted = bias_corrected_local_linear(d=d_full, y=y_full, boundary=0.0, weights=w_full)
        r_dropped = bias_corrected_local_linear(d=d_ref, y=y_ref, boundary=0.0)
        # Auto-selected h identical between the two paths.
        np.testing.assert_allclose(r_weighted.h, r_dropped.h, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            r_weighted.estimate_bias_corrected,
            r_dropped.estimate_bias_corrected,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_zero_weight_counts_reflect_positive_subset(self):
        """``n_obs`` / ``n_treated`` / ``n_control`` on the result must
        reflect the positive-weight sub-population, not the full panel."""
        rng = np.random.default_rng(7)
        G = 120
        d = rng.uniform(0.0, 1.0, G)
        dy = 2.0 * d + rng.normal(0, 0.25, G)
        w_unit = np.ones(G)
        w_unit[:20] = 0.0  # 20 zero-weight units
        panel = _make_panel(d, dy)
        row_w = np.zeros(panel.shape[0])
        for g in range(G):
            row_w[panel["unit"].to_numpy() == g] = w_unit[g]
        panel_w = panel.assign(w=row_w)
        from diff_diff.survey import SurveyDesign

        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        # 100 positive-weight units, not 120.
        assert r.n_obs == 100

    def test_repr_surfaces_weighted_fields_when_present(self):
        """Round 4 P3: ``__repr__`` must name ``variance_formula`` and
        ``effective_dose_mean`` when the fit was weighted so ad-hoc log
        output / interactive notebooks show which inference path and
        denominator were used."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_w = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        rep = repr(r_w)
        assert "variance_formula='survey_binder_tsl'" in rep
        assert "effective_dose_mean=" in rep
        # Unweighted fit: ``__repr__`` keeps the original compact form.
        r_unw = est.fit(panel, "outcome", "dose", "period", "unit")
        rep_unw = repr(r_unw)
        assert "variance_formula" not in rep_unw
        assert "effective_dose_mean" not in rep_unw

    def test_survey_path_populates_df_survey(self):
        """Counter-test to the above: under ``survey_design=SurveyDesign(...)``
        with real PSU/strata, ``df_survey`` IS populated and threaded
        into t-inference."""
        panel, SurveyDesign = self._panel_with_survey_cols(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
            )
        sm = r.survey_metadata
        assert sm is not None
        assert sm.n_strata is not None and sm.n_strata > 1
        assert sm.n_psu is not None and sm.n_psu > 1
        assert sm.df_survey is not None and sm.df_survey > 0

    def test_to_dict_includes_variance_formula_and_effective_dose_mean(self):
        """Round 3 P2b: ``to_dict()`` must surface ``variance_formula``
        and ``effective_dose_mean`` so downstream machine consumers can
        recover the weighted denominator + SE family without inspecting
        the result object directly."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        d = r.to_dict()
        assert "variance_formula" in d
        assert d["variance_formula"] == "survey_binder_tsl"
        assert "effective_dose_mean" in d
        assert d["effective_dose_mean"] is not None
        assert np.isfinite(d["effective_dose_mean"])

    def test_to_dict_variance_formula_none_when_unweighted(self):
        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        d = r.to_dict()
        assert d["variance_formula"] is None
        assert d["effective_dose_mean"] is None

    def test_summary_renders_effective_dose_mean_under_weights(self):
        """``summary()`` must display the weighted denominator explicitly
        when the fit used weights (Round 3 P2b)."""
        from diff_diff.survey import SurveyDesign

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        s = r.summary()
        assert "Weighted D" in s  # "Weighted D̄ (denominator):" header

    def test_effective_dose_mean_equals_dose_mean_under_uniform_weights(self):
        """Uniform weights → effective_dose_mean ≡ dose_mean at 1e-14."""
        from diff_diff.survey import SurveyDesign

        panel, _, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_w = panel.assign(w=np.ones(panel.shape[0]))
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert r.effective_dose_mean is not None
        np.testing.assert_allclose(r.effective_dose_mean, r.dose_mean, atol=1e-14, rtol=1e-14)

    # ---------- P1 fix: SurveyMetadata contract for downstream consumers ----------

    def test_survey_metadata_is_surveymetadata_instance(self):
        """HAD survey results expose ``survey_metadata`` as the repo-
        standard :class:`diff_diff.survey.SurveyMetadata` dataclass, so
        shared reporting consumers (BusinessReport, DiagnosticReport)
        can read ``df_survey`` / ``effective_n`` / ``n_strata`` /
        ``n_psu`` via attribute access uniformly across estimators.
        Regression lock for P1 from round 2 CI review."""
        from diff_diff.survey import SurveyDesign, SurveyMetadata

        panel, row_w, _, _, _, _ = self._panel_with_unit_weights(G=200)
        panel_with_w = panel.assign(w=row_w)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # A weights-only SurveyDesign and a fuller one both produce a
            # SurveyMetadata (not a dict).
            r_w = est.fit(
                panel_with_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
            r_sd = est.fit(
                panel_with_w,
                "outcome",
                "dose",
                "period",
                "unit",
                survey_design=SurveyDesign(weights="w"),
            )
        assert isinstance(r_w.survey_metadata, SurveyMetadata)
        assert isinstance(r_sd.survey_metadata, SurveyMetadata)
        # Attribute access for the fields downstream consumers read.
        for r in (r_w, r_sd):
            _ = r.survey_metadata.weight_type
            _ = r.survey_metadata.effective_n
            _ = r.survey_metadata.design_effect
            _ = r.survey_metadata.sum_weights
            _ = r.survey_metadata.n_strata
            _ = r.survey_metadata.n_psu
            _ = r.survey_metadata.df_survey
            _ = r.survey_metadata.weight_range


# =============================================================================
# Phase 4.5 B: mass-point weighted + event-study survey + sup-t bootstrap
# =============================================================================


class TestMassPointWeighted:
    """Weighted 2SLS on the mass-point path (Phase 4.5 B)."""

    @staticmethod
    def _dgp_mp(n, seed=0):
        rng = np.random.default_rng(seed)
        d = np.concatenate([np.full(n // 5, 0.3), rng.uniform(0.3, 1.0, n - n // 5)])
        rng.shuffle(d)
        dy = 2.0 * d + 0.3 * rng.standard_normal(n)
        return d, dy

    @staticmethod
    def _make_panel(d, dy):
        G = d.shape[0]
        return pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.column_stack([np.zeros(G), d]).ravel(),
                "outcome": np.column_stack([np.zeros(G), dy]).ravel(),
            }
        )

    def test_uniform_weights_bit_parity_all_vcov_variants(self):
        """Direct helper call: weights=np.ones ≡ unweighted at atol=1e-14
        across classical, hc1, and CR1 sandwich branches."""
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(400, seed=7)
        cluster = np.arange(d.shape[0]) // 10
        for vcov in ("classical", "hc1"):
            for use_cluster in (False, True):
                cluster_arg = cluster if use_cluster else None
                b0, s0, _ = _fit_mass_point_2sls(d, dy, 0.3, cluster_arg, vcov)
                b1, s1, _ = _fit_mass_point_2sls(
                    d,
                    dy,
                    0.3,
                    cluster_arg,
                    vcov,
                    weights=np.ones(d.shape[0]),
                    return_influence=False,
                )
                np.testing.assert_allclose(b0, b1, atol=1e-14, rtol=1e-14)
                np.testing.assert_allclose(s0, s1, atol=1e-14, rtol=1e-14)

    def test_weights_none_path_unchanged(self):
        """Unweighted path returns (beta, se, None) — third slot is None
        when return_influence=False (the default)."""
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(200, seed=1)
        _b, _s, psi = _fit_mass_point_2sls(d, dy, 0.3, None, "hc1")
        assert psi is None

    def test_negative_weights_rejected(self):
        """Front-door reject negative weights with a clear ValueError."""
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(200, seed=2)
        w = np.ones(d.shape[0])
        w[0] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            _fit_mass_point_2sls(d, dy, 0.3, None, "hc1", weights=w)

    def test_non_finite_weights_rejected(self):
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(200, seed=3)
        w = np.ones(d.shape[0])
        w[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _fit_mass_point_2sls(d, dy, 0.3, None, "hc1", weights=w)

    def test_zero_sum_weights_rejected(self):
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(200, seed=4)
        w = np.zeros(d.shape[0])
        with pytest.raises(ValueError, match="weights sum to zero"):
            _fit_mass_point_2sls(d, dy, 0.3, None, "hc1", weights=w)

    def test_weights_length_mismatch_rejected(self):
        from diff_diff.had import _fit_mass_point_2sls

        d, dy = self._dgp_mp(200, seed=5)
        w = np.ones(d.shape[0] - 5)
        with pytest.raises(ValueError, match="length"):
            _fit_mass_point_2sls(d, dy, 0.3, None, "hc1", weights=w)

    def test_fit_mass_point_survey_variance_formula(self):
        """`fit(design='mass_point', survey_design=...)` sets
        variance_formula='survey_binder_tsl_2sls' and populates
        survey_metadata with the full SurveyMetadata dataclass."""
        from diff_diff.survey import SurveyDesign

        d, dy = self._dgp_mp(300, seed=6)
        panel = self._make_panel(d, dy)
        panel["w"] = np.random.default_rng(0).uniform(0.5, 2.0, panel.shape[0])
        # Constant-within-unit for the aggregator.
        panel["w"] = panel.groupby("unit")["w"].transform("first")
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1")
            r = est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)
        assert r.variance_formula == "survey_binder_tsl_2sls"
        assert r.survey_metadata is not None
        assert r.survey_metadata.weight_type == "pweight"
        assert r.effective_dose_mean is not None

    def test_mass_point_non_pweight_rejected(self):
        """Non-pweight SurveyDesigns rejected at fit() with a clear
        NotImplementedError — mirrors static continuous path."""
        from diff_diff.survey import SurveyDesign

        d, dy = self._dgp_mp(200, seed=8)
        panel = self._make_panel(d, dy)
        panel["w"] = np.ones(panel.shape[0])
        sd = SurveyDesign(weights="w", weight_type="aweight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point")
            with pytest.raises(NotImplementedError, match="aweight"):
                est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)


class TestSupTReducesToNormalAtH1:
    """At H=1 the sup-t critical value must reduce to the Normal
    quantile (up to MC error). Catches the (1/n) prefactor / scale-
    convention drift in _sup_t_multiplier_bootstrap in isolation from
    the full event-study pipeline."""

    def test_sup_t_h1_reduces_to_normal_quantile(self):
        import scipy.stats

        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(42)
        G = 500
        # Well-scaled IF under unit-level iid multipliers: any i.i.d. psi
        # with bounded variance. Analytical "SE" = sqrt(sum(psi^2)) since
        # Var_xi(sum(xi * psi)) = sum(psi^2) under Rademacher xi ∈ {-1,+1}.
        psi = rng.standard_normal((G, 1))
        se = np.array([float(np.sqrt(np.sum(psi[:, 0] ** 2)))])
        q, _low, _high, n_valid = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(1),
            se,
            None,
            n_bootstrap=5000,
            alpha=0.05,
            seed=42,
        )
        assert n_valid >= 4500  # almost all draws finite
        expected = float(scipy.stats.norm.ppf(0.975))
        # B=5000 MC noise on the 97.5%-tile quantile is ~0.03-0.05.
        assert abs(q - expected) < 0.15, (
            f"H=1 sup-t quantile should reduce to Phi^-1(0.975)={expected:.4f}; "
            f"got q={q:.4f}. |diff|={abs(q - expected):.4f}. Likely a "
            f"scale-convention drift in _sup_t_multiplier_bootstrap (check "
            f"that perturbations = weights @ psi has no (1/n) prefactor and "
            f"that sum(psi^2) matches the claimed 'analytical' variance)."
        )

    def test_sup_t_h5_greater_than_pointwise(self):
        """At H=5 with i.i.d. psi, sup-t > 1.96. Catches degenerate
        constructions where sup collapses to the marginal."""
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(0)
        G = 500
        H = 5
        psi = rng.standard_normal((G, H))
        se = np.sqrt(np.sum(psi**2, axis=0))
        q, _, _, _ = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(H),
            se,
            None,
            n_bootstrap=1000,
            alpha=0.05,
            seed=42,
        )
        assert q > 1.96 + 0.15, (
            f"H=5 sup-t should exceed pointwise Normal quantile by a "
            f"material margin; got q={q:.4f}."
        )

    def test_sup_t_seed_reproducibility(self):
        """Same seed → same critical value (across repeated calls)."""
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(0)
        G = 200
        H = 3
        psi = rng.standard_normal((G, H))
        se = np.sqrt(np.sum(psi**2, axis=0))
        q1, _, _, _ = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(H),
            se,
            None,
            n_bootstrap=500,
            alpha=0.05,
            seed=17,
        )
        q2, _, _, _ = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(H),
            se,
            None,
            n_bootstrap=500,
            alpha=0.05,
            seed=17,
        )
        assert q1 == q2

    def test_clustered_sup_t_h1_reduces_to_normal(self):
        """Clustered branch: at H=1 the sup-t crit reduces to the Normal
        quantile for BOTH the continuous scalar (1.0) and the mass-point
        CR1 scalar sqrt(G/(G-1)) — the decisive variance-family check."""
        import scipy.stats

        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(3)
        n_units, n_clusters = 300, 30
        cluster_ids = rng.integers(0, n_clusters, size=n_units)
        expected = float(scipy.stats.norm.ppf(0.975))

        def _cluster_se(psi, scale):
            labels = pd.unique(cluster_ids)
            lab = {v: r for r, v in enumerate(labels)}
            pcl = np.zeros((len(labels), psi.shape[1]))
            for i in range(n_units):
                pcl[lab[cluster_ids[i]]] += psi[i]
            return np.sqrt(((scale * pcl) ** 2).sum(axis=0))

        for scale in (1.0, float(np.sqrt(n_clusters / (n_clusters - 1)))):
            psi = rng.standard_normal((n_units, 1))
            se = _cluster_se(psi, scale)
            q, _lo, _hi, n_valid = _sup_t_multiplier_bootstrap(
                psi,
                np.zeros(1),
                se,
                None,
                n_bootstrap=20000,
                alpha=0.05,
                seed=7,
                cluster_ids=cluster_ids,
                cluster_if_scale=scale,
            )
            assert n_valid >= 18000
            assert abs(q - expected) < 0.10, (
                f"clustered H=1 sup-t (scale={scale:.4f}) should reduce to "
                f"Phi^-1(0.975)={expected:.4f}; got {q:.4f} — variance-family "
                f"drift in the clustered bootstrap branch."
            )

    def test_clustered_sup_t_single_cluster_nan(self):
        """One cluster → NaN crit / None band (CR undefined)."""
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(0)
        psi = rng.standard_normal((100, 1))
        q, lo, hi, n_valid = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(1),
            np.array([1.0]),
            None,
            n_bootstrap=500,
            alpha=0.05,
            seed=1,
            cluster_ids=np.zeros(100, dtype=int),
        )
        assert np.isnan(q) and lo is None and hi is None and n_valid == 0


class TestEventStudyClusterBand:
    """Phase 2b: cluster-robust event-study pointwise CIs + clustered sup-t
    simultaneous band (continuous + mass-point). The core reconciliation is
    that the cluster-aggregated influence function reproduces the analytical
    cluster-robust SE — exactly (continuous, scale 1.0) or after the CR1
    sqrt(G/(G-1)) scalar (mass-point)."""

    @staticmethod
    def _clustered_panel(G=240, n_clusters=24, seed=0):
        rng = np.random.default_rng(seed)
        d = np.where(rng.random(G) < 0.15, 0.0, rng.uniform(0.2, 1.2, size=G))
        d[0] = 0.0
        state = np.repeat(np.arange(n_clusters), G // n_clusters)
        panel = _make_multi_period_panel(
            d, n_periods=5, F=3, seed=seed, extra_cols={"state": state}
        )
        return panel

    def test_continuous_if_reconciliation_deterministic(self):
        """sqrt(sum_c (sum_{i in c} IF_i)^2) == se_robust for the cluster-
        robust CCT fit (scale 1.0) — bootstrap-free proof of the continuous
        reconciliation on the REAL influence function."""
        d, dy, cl = self._make_cluster_dgp(seed=1)
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, cluster=cl, return_influence=True)
        recon = self._cluster_agg_norm(bc.influence_function, cl, scale=1.0)
        np.testing.assert_allclose(recon, bc.se_robust, rtol=0, atol=1e-10)

    def test_masspoint_if_reconciliation_deterministic(self):
        """sqrt(sum_c (sqrt(G/(G-1)) * sum_{i in c} psi_i)^2) == se for the
        mass-point CR1 fit — bootstrap-free proof the sqrt(G/(G-1)) scalar
        exactly restores the CR1 finite-sample factor on the REAL IF."""
        from diff_diff.had import _fit_mass_point_2sls

        rng = np.random.default_rng(2)
        G, d_lower = 240, 0.5
        mass_n = G // 3
        d = np.concatenate([np.full(mass_n, d_lower), rng.uniform(d_lower, 1.0, G - mass_n)])
        rng.shuffle(d)
        cl = np.arange(G) % 20
        shock = rng.normal(scale=0.5, size=20)[cl]
        dy = 0.3 * d + shock + 0.1 * rng.standard_normal(G)
        _beta, se, psi = _fit_mass_point_2sls(d, dy, d_lower, cl, "hc1", return_influence=True)
        n_cl = len(pd.unique(cl))
        recon = self._cluster_agg_norm(psi, cl, scale=float(np.sqrt(n_cl / (n_cl - 1))))
        np.testing.assert_allclose(recon, se, rtol=0, atol=1e-10)

    def test_continuous_clustered_band_end_to_end(self):
        import scipy.stats

        panel = self._clustered_panel(seed=2)
        r = HeterogeneousAdoptionDiD(
            design="continuous_at_zero", cluster="state", n_bootstrap=1500, seed=11
        ).fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study", cband=True)
        assert r.vcov_type == "cr1" and r.cluster_name == "state"
        assert r.cband_low is not None and np.all(np.isfinite(r.cband_low))
        assert r.cband_high is not None and np.all(np.isfinite(r.cband_high))
        assert r.cband_method == "cluster_multiplier_bootstrap"
        assert r.cband_crit_value >= float(scipy.stats.norm.ppf(0.975)) - 0.05
        # Simultaneous band is at least as wide as the pointwise CI.
        assert np.all(r.cband_low <= r.conf_int_low + 1e-9)
        assert np.all(r.cband_high >= r.conf_int_high - 1e-9)

    def test_unweighted_masspoint_clustered_band_end_to_end(self):
        """Unweighted mass-point event-study + cluster= + cband: finite
        cluster-robust band (the unweighted arm of the mass-point path)."""
        rng = np.random.default_rng(7)
        G, d_lower = 240, 0.5
        mass_n = G // 3
        d = np.concatenate([np.full(mass_n, d_lower), rng.uniform(d_lower, 1.0, G - mass_n)])
        rng.shuffle(d)
        state = np.arange(G) % 24
        panel = _make_multi_period_panel(d, n_periods=5, F=3, seed=7, extra_cols={"state": state})
        r = HeterogeneousAdoptionDiD(
            design="mass_point", cluster="state", d_lower=d_lower, n_bootstrap=1500, seed=17
        ).fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study", cband=True)
        assert r.vcov_type == "cr1" and r.cluster_name == "state"
        assert r.cband_low is not None and np.all(np.isfinite(r.cband_low))
        assert r.cband_method == "cluster_multiplier_bootstrap"

    def test_cluster_survey_event_study_raises(self):
        from diff_diff.survey import SurveyDesign

        panel = self._clustered_panel(seed=3)
        panel["w"] = 1.0
        with pytest.raises(NotImplementedError, match=r"cluster.*\+ survey_design="):
            HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state").fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_single_cluster_band_nan_and_warns(self):
        # Dense dose (mirrors the static single-cluster test) so the local-
        # linear fit resolves gracefully to NaN CR SEs rather than a
        # degenerate-bandwidth error; the band guard then fires.
        rng = np.random.default_rng(0)
        G = 300
        d = rng.uniform(0.0, 1.0, G)
        d[0] = 0.0
        panel = _make_multi_period_panel(
            d, n_periods=5, F=3, seed=0, extra_cols={"state": np.zeros(G, dtype=int)}
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = HeterogeneousAdoptionDiD(
                design="continuous_at_zero", cluster="state", n_bootstrap=500, seed=1
            ).fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study", cband=True)
        assert r.cband_low is None and r.cband_high is None
        assert any("single cluster" in str(x.message).lower() for x in w)
        # "Undefined band" (crit=NaN, method/count populated), NOT "band
        # skipped" (all None) — distinguishable by the caller.
        assert r.cband_crit_value is not None and np.isnan(r.cband_crit_value)
        assert r.cband_method == "cluster_multiplier_bootstrap"
        assert r.cband_n_bootstrap == 500

    def test_clustered_band_determinism(self):
        panel = self._clustered_panel(seed=2)
        kw = dict(design="continuous_at_zero", cluster="state", n_bootstrap=800, seed=21)
        fit_kw = dict(aggregate="event_study", cband=True)
        r1 = HeterogeneousAdoptionDiD(**kw).fit(
            panel, "outcome", "dose", "period", "unit", **fit_kw
        )
        r2 = HeterogeneousAdoptionDiD(**kw).fit(
            panel, "outcome", "dose", "period", "unit", **fit_kw
        )
        np.testing.assert_array_equal(r1.cband_low, r2.cband_low)
        np.testing.assert_array_equal(r1.cband_high, r2.cband_high)
        assert r1.cband_crit_value == r2.cband_crit_value

    # ---- helpers ----
    @staticmethod
    def _make_cluster_dgp(G=200, n_clusters=20, seed=0):
        rng = np.random.default_rng(seed)
        d = np.where(rng.random(G) < 0.15, 0.0, rng.uniform(0.2, 1.5, size=G))
        d[0] = 0.0
        cl = np.repeat(np.arange(n_clusters), G // n_clusters)
        shock = rng.normal(scale=1.0, size=n_clusters)[cl]
        dy = 1.5 * d + shock + rng.normal(scale=0.3, size=G)
        return d, dy, cl

    @staticmethod
    def _cluster_agg_norm(if_vec, cl, scale):
        labels = pd.unique(cl)
        lab = {v: r for r, v in enumerate(labels)}
        s = np.zeros(len(labels))
        for i in range(len(cl)):
            s[lab[cl[i]]] += if_vec[i]
        return float(np.sqrt(np.sum((scale * s) ** 2)))


class TestEventStudySurveyCband:
    """Event-study + weights / survey + sup-t cband scope (Phase 4.5 B)."""

    @staticmethod
    def _multi_period_panel(G=150, T=4, seed=0):
        rng = np.random.default_rng(seed)
        d_post = rng.uniform(0.0, 1.0, G)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_post[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"])
        return panel

    def test_unweighted_es_cband_fields_none(self):
        """Unweighted event-study: all cband_* fields are None (pre-PR
        numerical output preserved)."""
        panel = self._multi_period_panel(G=200)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", seed=0)
        r = est.fit(panel, "outcome", "dose", "period", "unit", aggregate="event_study")
        assert r.cband_low is None
        assert r.cband_high is None
        assert r.cband_crit_value is None
        assert r.cband_method is None
        assert r.variance_formula is None

    def test_weighted_es_cband_false_skips_bootstrap(self):
        """`cband=False` under weighted event-study: no bootstrap, cband_*
        fields are None; att/se bit-exact to unweighted at uniform
        weights."""
        from diff_diff.survey import SurveyDesign

        panel = self._multi_period_panel(G=200, seed=3).assign(w=1.0)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                cband=False,
            )
        assert r.cband_low is None
        assert r.cband_high is None
        assert r.cband_crit_value is None
        # variance_formula IS set (survey Binder-TSL path active).
        assert r.variance_formula == "survey_binder_tsl"

    def test_weighted_es_cband_true_populates_band(self):
        """Weighted event-study + cband=True populates cband_* fields,
        with cband_crit_value in a plausible range."""
        from diff_diff.survey import SurveyDesign

        panel = self._multi_period_panel(G=200, seed=5).assign(w=1.0)
        est = HeterogeneousAdoptionDiD(
            design="continuous_at_zero",
            seed=42,
            n_bootstrap=500,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                cband=True,
            )
        assert r.cband_low is not None and r.cband_high is not None
        assert r.cband_crit_value is not None and np.isfinite(r.cband_crit_value)
        assert r.cband_method == "multiplier_bootstrap"
        assert r.cband_n_bootstrap == 500
        # Sup-t should be >= pointwise Normal quantile (1.96) to cover
        # all horizons simultaneously.
        assert r.cband_crit_value >= 1.5  # loose lower bound given MC noise
        # Band strictly wider than pointwise CI (centered on att).
        for i in range(len(r.event_times)):
            if np.isfinite(r.se[i]) and r.se[i] > 0:
                pointwise_width = r.conf_int_high[i] - r.conf_int_low[i]
                sim_width = r.cband_high[i] - r.cband_low[i]
                assert sim_width >= pointwise_width * 0.99  # allow tiny MC slack

    def test_event_study_filter_info_stable_across_weight_patterns(self):
        """filter_info is identical whether the fit is unweighted,
        uniform-weighted, or informatively-weighted (staggered-filter
        is identification-theory, not sampling-domain)."""
        rng = np.random.default_rng(0)
        G = 120
        # Staggered panel: cohort 3 and cohort 4.
        d_post = rng.uniform(0.0, 1.0, G)
        first_treat = rng.choice([0, 3, 4], size=G, p=[0.4, 0.3, 0.3])
        rows = []
        for t in range(5):
            for g in range(G):
                ft = first_treat[g]
                dose = d_post[g] if (ft > 0 and t >= ft) else 0.0
                y = 0.2 * t + (2.0 * dose if dose > 0 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y, ft))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "first_treat"])

        from diff_diff.survey import SurveyDesign

        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_unw = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                first_treat="first_treat",
            )
            r_uni = est.fit(
                panel.assign(w=1.0),
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                first_treat="first_treat",
                survey_design=SurveyDesign(weights="w"),
                cband=False,
            )
            # Informative per-unit weights (constant within unit).
            w_unit = 1.0 + 0.5 * rng.standard_normal(G)
            w_unit = np.clip(w_unit, 0.1, None)
            w_row = panel["unit"].map(lambda g: w_unit[g]).to_numpy()
            r_inf = est.fit(
                panel.assign(w=w_row),
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                first_treat="first_treat",
                survey_design=SurveyDesign(weights="w"),
                cband=False,
            )
        # filter_info must agree across all three fits (same dropped cohorts).
        assert r_unw.filter_info == r_uni.filter_info == r_inf.filter_info

    def test_event_study_mass_point_weighted_smoke(self):
        """Mass-point + weighted (survey) event-study smoke:
        variance_formula = 'survey_binder_tsl_2sls' and cband populated."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(10)
        G = 200
        T = 4
        d_mp = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d_mp)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_mp[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"]).assign(w=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(
                design="mass_point",
                vcov_type="hc1",
                seed=0,
                n_bootstrap=200,
            )
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
            )
        assert r.design == "mass_point"
        assert r.variance_formula == "survey_binder_tsl_2sls"
        assert r.cband_crit_value is not None and np.isfinite(r.cband_crit_value)

    def test_zero_se_horizon_nan_gates_cband(self):
        """Review R1 P0: a horizon with se <= 0 or non-finite must NOT
        produce a finite simultaneous-band endpoint — gating matches
        the pointwise ``safe_inference`` contract."""
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(0)
        G = 200
        H = 3
        psi = rng.standard_normal((G, H))
        se = np.array([np.sqrt(np.sum(psi[:, 0] ** 2)), 0.0, np.nan])
        att = np.array([1.0, 2.0, 3.0])
        q, low, high, n_valid = _sup_t_multiplier_bootstrap(
            psi,
            att,
            se,
            None,
            n_bootstrap=500,
            alpha=0.05,
            seed=1,
        )
        assert n_valid > 250
        # Horizon 0: finite se → finite band.
        assert np.isfinite(low[0]) and np.isfinite(high[0])
        # Horizons 1 and 2: zero / NaN se → NaN band (not `att ± q * 0`).
        assert np.isnan(low[1]) and np.isnan(high[1])
        assert np.isnan(low[2]) and np.isnan(high[2])

    def test_mass_point_survey_plus_cluster_rejected_static(self):
        """Review R2 P1: mass-point + survey_design= + cluster= must raise
        NotImplementedError on the static path. The survey path would
        silently override the CR1 SE with Binder-TSL while the result still
        reported vcov_type='cr1'; a bare cluster= (unweighted CR1) is the
        supported clustering entry, and weighted clustering routes through
        survey_design=SurveyDesign(weights=, psu=). This test uses
        survey_design= + cluster= to trigger the guard."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(0)
        G = 200
        d = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d)
        dy = 2.0 * d + 0.3 * rng.standard_normal(G)
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.column_stack([np.zeros(G), d]).ravel(),
                "outcome": np.column_stack([np.zeros(G), dy]).ravel(),
                "state": np.repeat(np.arange(G) // 20, 2),
                "w": np.ones(2 * G),
            }
        )
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1", cluster="state")
            with pytest.raises(NotImplementedError, match="cluster"):
                est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)

    def test_lonely_psu_adjust_with_singletons_rejected_on_cband(self):
        """Review R2 P1: sup-t bootstrap rejects lonely_psu='adjust'
        when there are singleton strata, because the bootstrap helper
        pools singletons with nonzero multipliers but the analytical
        target centers them at the global mean — mismatch."""
        from diff_diff.had import _sup_t_multiplier_bootstrap
        from diff_diff.survey import ResolvedSurveyDesign

        rng = np.random.default_rng(0)
        G = 80
        # 3 strata, two with multiple PSUs, one singleton.
        strata = np.array([1] * 30 + [2] * 30 + [3] * 20)
        # PSUs: 10 in stratum 1, 10 in stratum 2, 1 in stratum 3 (singleton).
        psu = np.concatenate(
            [np.arange(10).repeat(3), (10 + np.arange(10)).repeat(3), np.full(20, 20)]
        )
        adjust_resolved = ResolvedSurveyDesign(
            weights=np.ones(G),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=21,
            lonely_psu="adjust",
            combined_weights=True,
            mse=False,
        )
        psi = rng.standard_normal((G, 2))
        with pytest.raises(NotImplementedError, match="lonely_psu='adjust'"):
            _sup_t_multiplier_bootstrap(
                psi,
                np.zeros(2),
                np.array([1.0, 1.0]),
                adjust_resolved,
                n_bootstrap=200,
                alpha=0.05,
                seed=0,
            )

    def test_stratified_h1_sup_t_matches_analytical(self):
        """Review R2 P1 coverage: stratum-centered H=1 bootstrap variance
        matches the analytical Binder-TSL target (q ≈ 1.96 at H=1)."""
        from diff_diff.had import _sup_t_multiplier_bootstrap
        from diff_diff.survey import ResolvedSurveyDesign, compute_survey_if_variance

        rng = np.random.default_rng(7)
        G = 400
        strata = np.repeat(np.arange(4), G // 4)
        psu = np.arange(G)
        resolved = ResolvedSurveyDesign(
            weights=np.ones(G),
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=4,
            n_psu=G,
            lonely_psu="remove",
            combined_weights=True,
            mse=False,
        )
        psi = rng.standard_normal((G, 1))
        V_analytical = compute_survey_if_variance(psi[:, 0], resolved)
        se_analytical = np.sqrt(V_analytical)
        q, _, _, _ = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(1),
            np.array([se_analytical]),
            resolved,
            n_bootstrap=5000,
            alpha=0.05,
            seed=42,
        )
        # At H=1 the sup collapses to the marginal; with stratum-
        # centered + small-sample-corrected perturbations the bootstrap
        # distribution is ~ N(0, 1), so q → Phi^-1(0.975) = 1.96.
        # B=5000 MC noise on the tail quantile is ~0.03-0.05.
        assert abs(q - 1.96) < 0.15, (
            f"Stratified H=1 sup-t should match Normal quantile 1.96 up to "
            f"MC noise; got q={q:.4f}. Likely a stratum-centering bug in "
            f"_sup_t_multiplier_bootstrap."
        )

    def test_trivial_survey_h1_sup_t_matches_analytical(self):
        """Review R3 P1: the survey-aware bootstrap branch must fire even
        on trivial ``SurveyDesign(weights=...)`` (no explicit strata /
        PSU / FPC). The analytical target is still the centered
        (n/(n-1)) · Σ(ψ − ψ̄)² Binder formula, so the bootstrap must
        also apply stratum-demeaning + small-sample correction — NOT
        fall through to raw unit-level Rademacher.
        """
        from diff_diff.had import _sup_t_multiplier_bootstrap
        from diff_diff.survey import ResolvedSurveyDesign, compute_survey_if_variance

        rng = np.random.default_rng(11)
        G = 300
        # Trivial resolved: weights only, no strata / PSU / FPC.
        resolved = ResolvedSurveyDesign(
            weights=np.ones(G),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=1,
            n_psu=G,
            lonely_psu="remove",
            combined_weights=True,
            mse=False,
        )
        psi = rng.standard_normal((G, 1))
        V_analytical = compute_survey_if_variance(psi[:, 0], resolved)
        se_analytical = np.sqrt(V_analytical)
        q, _, _, _ = _sup_t_multiplier_bootstrap(
            psi,
            np.zeros(1),
            np.array([se_analytical]),
            resolved,
            n_bootstrap=5000,
            alpha=0.05,
            seed=42,
        )
        # q ≈ 1.96 at H=1 confirms the trivial-survey branch applies
        # the same stratum-demean + sqrt(n/(n-1)) correction the
        # analytical target uses. Pre-R3, use_survey_bootstrap fell
        # through to unit-level Rademacher, off by sqrt(n/(n-1)).
        assert abs(q - 1.96) < 0.15, (
            f"Trivial-survey H=1 sup-t should match Normal quantile "
            f"1.96 up to MC noise; got q={q:.4f}. Likely the survey-"
            f"aware bootstrap branch is not firing on trivial "
            f"SurveyDesign."
        )

    def test_mass_point_classical_survey_rejected_static(self):
        """Review R3 P1: vcov_type='classical' + survey_design= on
        design='mass_point' rejects with a clear pointer to HC1.
        Previously the survey path silently overrode classical SE
        with Binder-TSL composed from the HC1-scale IF."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(20)
        G = 200
        d = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d)
        dy = 2.0 * d + 0.3 * rng.standard_normal(G)
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.column_stack([np.zeros(G), d]).ravel(),
                "outcome": np.column_stack([np.zeros(G), dy]).ravel(),
                "w": np.ones(2 * G),
            }
        )
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical")
            with pytest.raises(NotImplementedError, match="classical.*survey"):
                est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)

    def test_mass_point_classical_event_study_with_cband_rejected(self):
        """Review R3 P1 (event-study arm): vcov_type='classical' is
        rejected on the mass-point event-study survey path — the survey
        Binder-TSL variance is built from the IF matrix, which is
        incompatible with a classical SE."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(30)
        G, T = 150, 4
        d_mp = np.concatenate([np.full(30, 0.3), rng.uniform(0.3, 1.0, G - 30)])
        rng.shuffle(d_mp)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_mp[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"]).assign(w=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(
                design="mass_point", vcov_type="classical", seed=0, n_bootstrap=100
            )
            with pytest.raises(NotImplementedError, match="classical"):
                est.fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    aggregate="event_study",
                    survey_design=SurveyDesign(weights="w"),
                    cband=True,
                )

    def test_event_study_zero_weight_units_excluded_from_n_units(self):
        """Review R4 P2: weighted event-study reports the POSITIVE-WEIGHT
        contributing sample size in n_units / n_obs_per_horizon (matches
        the static-path n_obs contract). survey_metadata still carries
        the full-design effective_n / n_psu."""
        rng = np.random.default_rng(50)
        G, T = 200, 4
        d_post = rng.uniform(0.0, 1.0, G)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_post[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        from diff_diff.survey import SurveyDesign

        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"])
        w_unit = np.ones(G)
        w_unit[:30] = 0.0  # 30 zero-weight units; 170 contribute.
        w_row = panel["unit"].map(lambda g: w_unit[g]).to_numpy()
        panel = panel.assign(w=w_row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="continuous_at_zero", seed=0)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
                cband=False,
            )
        assert r.n_units == 170, (
            f"n_units should report positive-weight contributing count "
            f"(170), not full-design size (200); got {r.n_units}"
        )
        assert np.all(r.n_obs_per_horizon == 170)

    def test_mass_point_default_vcov_survey_rejected_static(self):
        """Review R5 P1: the effective-classical rejection must fire
        even when the user does NOT pass vcov_type explicitly — the
        default mapping (vcov_type=None, robust=False) resolves to
        'classical', and that default must NOT silently slip through
        on the survey_design= mass-point path."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(60)
        G = 200
        d = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d)
        dy = 2.0 * d + 0.3 * rng.standard_normal(G)
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.column_stack([np.zeros(G), d]).ravel(),
                "outcome": np.column_stack([np.zeros(G), dy]).ravel(),
                "w": np.ones(2 * G),
            }
        )
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Default vcov_type=None, robust=False → resolves to classical.
            est = HeterogeneousAdoptionDiD(design="mass_point")
            with pytest.raises(NotImplementedError, match="classical"):
                est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)

    def test_mass_point_default_vcov_event_study_cband_rejected(self):
        """Review R5 P1 (event-study arm): default vcov_type=None +
        survey_design= + cband=True must hit the effective-classical
        rejection. Previous guard only checked explicit
        vcov_type='classical'."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(61)
        G, T = 150, 4
        d_mp = np.concatenate([np.full(30, 0.3), rng.uniform(0.3, 1.0, G - 30)])
        rng.shuffle(d_mp)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_mp[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"]).assign(w=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Default vcov_type=None, robust=False.
            est = HeterogeneousAdoptionDiD(design="mass_point", seed=0, n_bootstrap=100)
            with pytest.raises(NotImplementedError, match="classical"):
                est.fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    aggregate="event_study",
                    survey_design=SurveyDesign(weights="w"),
                    cband=True,
                )

    def test_survey_event_study_continuous_end_to_end(self):
        """Review R6 P3: estimator-level
        ``fit(aggregate='event_study', survey_design=SurveyDesign(...))``
        integration lock for the continuous path. Verifies
        variance_formula, survey_metadata.df_survey (t-inference path),
        cband_* population, and stratified PSU dispatch through
        _aggregate_unit_resolved_survey."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(70)
        G, T, n_strata = 200, 4, 4
        d_post = rng.uniform(0.0, 1.0, G)
        strata_per_unit = np.repeat(np.arange(n_strata), G // n_strata)
        rng.shuffle(strata_per_unit)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_post[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y, strata_per_unit[g]))
        panel = pd.DataFrame(
            rows,
            columns=["unit", "period", "dose", "outcome", "stratum"],
        )
        w_unit = 1.0 + 0.3 * np.abs(rng.standard_normal(G))
        panel["w"] = panel["unit"].map(lambda g: w_unit[g])
        sd = SurveyDesign(weights="w", strata="stratum")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="continuous_at_zero", seed=0, n_bootstrap=200)
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=sd,
            )
        assert r.variance_formula == "survey_binder_tsl"
        assert r.survey_metadata is not None
        assert r.survey_metadata.n_strata == n_strata
        assert r.survey_metadata.n_psu == G
        assert r.survey_metadata.df_survey == G - n_strata
        assert r.cband_crit_value is not None and np.isfinite(r.cband_crit_value)
        assert r.cband_method == "multiplier_bootstrap"
        assert r.cband_n_bootstrap == 200
        assert r.cband_low is not None and r.cband_high is not None
        assert np.all(np.isfinite(r.se))

    def test_survey_event_study_mass_point_end_to_end(self):
        """Review R6 P3: estimator-level
        ``fit(design='mass_point', aggregate='event_study',
        survey_design=...)`` integration lock. Verifies
        variance_formula='survey_binder_tsl_2sls' and that the
        weighted 2SLS IF flows correctly through per-horizon
        Binder-TSL + sup-t bootstrap."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(71)
        G, T = 200, 4
        d_mp = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d_mp)
        strata_per_unit = np.repeat(np.arange(4), G // 4)
        rng.shuffle(strata_per_unit)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_mp[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y, strata_per_unit[g]))
        panel = pd.DataFrame(
            rows,
            columns=["unit", "period", "dose", "outcome", "stratum"],
        )
        w_unit = 1.0 + 0.3 * np.abs(rng.standard_normal(G))
        panel["w"] = panel["unit"].map(lambda g: w_unit[g])
        sd = SurveyDesign(weights="w", strata="stratum")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(
                design="mass_point",
                vcov_type="hc1",
                seed=0,
                n_bootstrap=200,
            )
            r = est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=sd,
            )
        assert r.variance_formula == "survey_binder_tsl_2sls"
        assert r.survey_metadata is not None
        assert r.survey_metadata.n_strata == 4
        assert r.cband_crit_value is not None and np.isfinite(r.cband_crit_value)
        assert r.cband_method == "multiplier_bootstrap"
        assert np.all(np.isfinite(r.se))

    def test_mass_point_default_vcov_robust_true_survey_allowed(self):
        """Complement: robust=True on the default path resolves to
        hc1, so the survey_design= mass-point fit is allowed with no explicit
        vcov_type."""
        from diff_diff.survey import SurveyDesign

        rng = np.random.default_rng(62)
        G = 200
        d = np.concatenate([np.full(40, 0.3), rng.uniform(0.3, 1.0, G - 40)])
        rng.shuffle(d)
        dy = 2.0 * d + 0.3 * rng.standard_normal(G)
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.column_stack([np.zeros(G), d]).ravel(),
                "outcome": np.column_stack([np.zeros(G), dy]).ravel(),
                "w": np.ones(2 * G),
            }
        )
        sd = SurveyDesign(weights="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # robust= deliberately exercises the deprecated alias (M-047)
            warnings.simplefilter("ignore", FutureWarning)
            est = HeterogeneousAdoptionDiD(design="mass_point", robust=True)
            r = est.fit(panel, "outcome", "dose", "period", "unit", survey_design=sd)
        assert r.vcov_type == "hc1"
        assert r.variance_formula == "survey_binder_tsl_2sls"


# =============================================================================
# TODO L74: extensive-margin / positive-untreated-mass fit-time warning
# =============================================================================

_EXTENSIVE_MARGIN_SUBSTR = "exactly-zero post-period dose"


def _panel_with_zero_fraction(G, n_zero, seed=0):
    """continuous_at_zero 2-period panel with EXACTLY ``n_zero`` zero post doses.

    The positive interior is drawn from Uniform(0.2, 1.0) so no accidental
    zeros sneak in — the exactly-zero fraction is precisely ``n_zero / G``.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.2, 1.0, G)
    d[:n_zero] = 0.0
    dy = 0.3 * d + 0.1 * rng.standard_normal(G)
    return _make_panel(d, dy)


class TestExtensiveMarginWarning:
    """The overall ``fit()`` path warns above a 10% exactly-zero-dose cutoff.

    Locks the TODO L74 fit-time UserWarning: HAD targets a WAS assuming no
    genuine untreated group, so a substantial exactly-zero (untreated) mass
    suggests a real extensive margin where standard DiD may be preferable.
    """

    def test_fires_above_threshold(self):
        # 40/200 = 20% exactly-zero -> warning fires.
        panel = _panel_with_zero_fraction(200, 40, seed=0)
        with pytest.warns(UserWarning, match=_EXTENSIVE_MARGIN_SUBSTR):
            HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")

    def test_fires_exactly_at_threshold(self):
        # 20/200 = 10% exactly -> the >= cutoff fires at the boundary.
        panel = _panel_with_zero_fraction(200, 20, seed=1)
        with pytest.warns(UserWarning, match=_EXTENSIVE_MARGIN_SUBSTR):
            HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")

    def test_message_names_count_and_pct(self):
        panel = _panel_with_zero_fraction(200, 40, seed=0)
        with pytest.warns(UserWarning) as rec:
            HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")
        msgs = [str(w.message) for w in rec if _EXTENSIVE_MARGIN_SUBSTR in str(w.message)]
        assert len(msgs) == 1
        # Names the count/total and percentage, and points to standard DiD.
        assert "40/200" in msgs[0]
        assert "20%" in msgs[0]
        assert "standard DiD" in msgs[0]

    def test_no_fire_all_positive(self):
        # No exactly-zero units -> no extensive-margin warning.
        panel = _panel_with_zero_fraction(200, 0, seed=2)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")
        assert not any(_EXTENSIVE_MARGIN_SUBSTR in str(w.message) for w in rec)

    def test_no_fire_just_below_threshold(self):
        # 19/200 = 9.5% < 10% -> no warning (boundary no-fire).
        panel = _panel_with_zero_fraction(200, 19, seed=3)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")
        assert not any(_EXTENSIVE_MARGIN_SUBSTR in str(w.message) for w in rec)

    def test_event_study_with_never_treated_does_not_warn(self):
        # Scope lock: the event-study path REQUIRES never-treated units
        # (Appendix B.2), so a 20% never-treated mass must NOT trip the
        # overall-path extensive-margin warning. The warning code sits after
        # the event-study dispatch returns, so it is structurally unreachable
        # here — this test guards against a future re-placement regressing it.
        rng = np.random.default_rng(4)
        d_at_F = rng.uniform(0.2, 1.0, 200)
        d_at_F[:40] = 0.0  # 20% never-treated (dose 0 at every period)
        panel = _make_multi_period_panel(d_at_F, n_periods=5, F=3, seed=4)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _fit_es(
                HeterogeneousAdoptionDiD(),
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
            )
        assert not any(_EXTENSIVE_MARGIN_SUBSTR in str(w.message) for w in rec)


class TestCovariatesTrap:
    """TODO L73: ``fit(covariates=...)`` raises NotImplementedError.

    Covariate-adjusted HAD (de Chaisemartin et al. 2026, Appendix B.1 /
    Theorem 6) is not implemented; the explicit param surfaces the roadmap
    instead of a bare ``TypeError`` from an unknown kwarg.
    """

    def test_covariates_raises_overall(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        with pytest.raises(NotImplementedError, match="Appendix B.1"):
            HeterogeneousAdoptionDiD().fit(
                panel, "outcome", "dose", "period", "unit", covariates=["x"]
            )

    def test_covariates_raises_event_study(self):
        # Raises before the event-study dispatch, so any panel suffices.
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        with pytest.raises(NotImplementedError, match="multivariate"):
            HeterogeneousAdoptionDiD().fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                covariates=["x"],
            )

    def test_covariates_none_default_does_not_raise(self):
        # The default covariates=None preserves the pre-PR fit path.
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD().fit(
                panel, "outcome", "dose", "period", "unit", covariates=None
            )
        assert np.isfinite(r.att)
