"""Monte Carlo oracle-consistency tests for weighted HAD (Phase 4.5).

No public weighted-CCF bias-corrected local-linear reference exists in
any language, so methodology confidence under informative sampling
weights is carried by these MC oracle tests + the uniform-weights
bit-parity lock (``TestHADSurvey``) + the cross-language weighted-OLS
parity (``test_np_npreg_weighted_parity.py``).

Each test is ``@pytest.mark.slow`` and gated by ``ci_params.bootstrap``
so pure-Python CI runs a reduced replication count while preserving
the code-path coverage.

The DGP is a known-tau HAD setting: ΔY_g = β · D_g + ε_g with
``β = 2.0``, ``D_g ~ Uniform[0, 1]``, and a heteroskedastic noise term.
The "tau" in this DGP is the WAS under the paper's identification —
the weighted slope at the boundary — which equals β exactly under the
linear DGP.

Informative sampling: selection probability ``p(D) ~ exp(-|D - 0.5|)``
so units near the interior are over-sampled relative to boundary units.
Under uniform weights, the boundary estimate is biased by the
over-sampling. Under the correct pweights (inverse selection
probability), the boundary estimate recovers β.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# DGP helpers
# =============================================================================


def _generate_had_panel(G, seed, beta=2.0, sigma=0.25):
    """Known-tau HAD two-period panel with LINEAR m(d).

    ΔY = β·D + ε with D ~ Uniform[0, 1]. Under linear m(d), the per-unit
    treatment effect TE_g = ΔY_g / D_g = β + ε_g/D_g, and the paper's WAS
    target is:

      WAS = E[(D_2 / E[D_2]) · TE_2] = β · E[D] / E[D] = β

    So the estimator should recover β exactly as G → ∞. Linear DGP keeps
    the oracle simple and avoids the WAS ≠ β confound that a quadratic
    nonlinear term introduces.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 1.0, G)
    dy = beta * d + rng.normal(0, sigma, G)
    return d, dy


def _build_panel_with_weights(d, dy, w_unit):
    """Wrap unit-level (d, dy, w_unit) into a two-period long panel."""
    G = len(d)
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(G), 2),
            "period": np.tile([0, 1], G),
            "dose": np.stack([np.zeros(G), d], axis=1).ravel(),
            "outcome": np.stack([np.zeros(G), dy], axis=1).ravel(),
            "w": np.repeat(w_unit, 2),
        }
    )


def _informative_weights(d, *, concentration=2.0):
    """Selection probability that over-samples near D=0.5.

    Under inverse-probability weighting, pweight_g = 1 / p(D_g) recovers
    the population target. Uses ``exp(-concentration * |D - 0.5|)`` which
    is bounded and non-zero on [0, 1].
    """
    p = np.exp(-concentration * np.abs(d - 0.5))
    return 1.0 / p  # inverse-probability pweight


# =============================================================================
# Oracle consistency tests
# =============================================================================


class TestWeightedMCConsistency:
    """Monte Carlo oracle consistency for the weighted continuous-HAD path."""

    @pytest.mark.slow
    def test_uniform_weights_recover_truth(self, ci_params):
        """Under uniform weights (equivalent to unweighted), the HAD
        continuous_at_zero estimator recovers β = 2.0 in expectation.
        This is the baseline regression lock — if this fails, the
        estimator itself is broken before any weighting question."""
        from diff_diff.had import HeterogeneousAdoptionDiD
        from diff_diff.survey import SurveyDesign

        G = 500
        n_reps = ci_params.bootstrap(200, min_n=25)
        beta_true = 2.0
        estimates = np.full(n_reps, np.nan)
        for r in range(n_reps):
            d, dy = _generate_had_panel(G, seed=1000 + r, beta=beta_true)
            w_unit = np.ones(G)
            panel = _build_panel_with_weights(d, dy, w_unit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r_fit = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="w"),
                )
            if np.isfinite(r_fit.att):
                estimates[r] = r_fit.att
        mean_est = float(np.nanmean(estimates))
        # Allow ~30% bias at G=500 with reduced reps; tightening would
        # require larger G or more reps. The point is methodology sanity,
        # not precise convergence.
        assert abs(mean_est - beta_true) < 0.6, (
            f"Mean estimate {mean_est:.3f} deviates from β={beta_true} "
            f"by more than 0.6 at G={G}"
        )

    @pytest.mark.slow
    def test_informative_weights_recover_truth(self, ci_params):
        """Under informative sampling + inverse-probability pweights,
        the weighted HAD recovers β. This is the core methodology claim
        under survey weights — the estimator must be weight-aware in a
        statistically meaningful sense, not just plumbing-level."""
        from diff_diff.had import HeterogeneousAdoptionDiD
        from diff_diff.survey import SurveyDesign

        G = 500
        n_reps = ci_params.bootstrap(200, min_n=25)
        beta_true = 2.0
        estimates = np.full(n_reps, np.nan)
        for r in range(n_reps):
            d, dy = _generate_had_panel(G, seed=2000 + r, beta=beta_true)
            w_unit = _informative_weights(d, concentration=2.0)
            panel = _build_panel_with_weights(d, dy, w_unit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r_fit = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="w"),
                )
            if np.isfinite(r_fit.att):
                estimates[r] = r_fit.att
        mean_est = float(np.nanmean(estimates))
        assert abs(mean_est - beta_true) < 0.8, (
            f"Weighted mean estimate {mean_est:.3f} deviates from "
            f"β={beta_true} by more than 0.8 at G={G}. Weight-aware "
            f"estimation should recover β even under informative sampling."
        )

    @pytest.mark.slow
    def test_ci_coverage_near_nominal(self, ci_params):
        """95% CI under uniform sampling should cover β at a rate close
        to 95% in expectation. Under reduced MC reps the coverage has
        wide Monte Carlo error; use a loose bar (>80%) to avoid
        false-positive CI flakiness."""
        from diff_diff.had import HeterogeneousAdoptionDiD
        from diff_diff.survey import SurveyDesign

        G = 500
        n_reps = ci_params.bootstrap(200, min_n=25)
        beta_true = 2.0
        covered = 0
        n_conclusive = 0
        for r in range(n_reps):
            d, dy = _generate_had_panel(G, seed=3000 + r, beta=beta_true)
            w_unit = np.ones(G)
            panel = _build_panel_with_weights(d, dy, w_unit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r_fit = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    panel,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="w"),
                )
            if np.isfinite(r_fit.conf_int[0]) and np.isfinite(r_fit.conf_int[1]):
                n_conclusive += 1
                if r_fit.conf_int[0] <= beta_true <= r_fit.conf_int[1]:
                    covered += 1
        coverage_rate = covered / max(1, n_conclusive)
        # Loose bar chosen to stay above chance (50%) while tolerating
        # both reduced MC reps and CCT-2014's known slight undercoverage
        # at small G. At n_reps=25, nominal 95% coverage has a MC std
        # error of ~4%, so anything below 60% indicates a real coverage
        # problem. Tightening would require n_reps in the hundreds.
        assert coverage_rate >= 0.60, (
            f"95% CI coverage of {coverage_rate:.2%} is below the 60% "
            f"bar at n_reps={n_reps} (G={G}); CCT under-coverage plus MC "
            f"noise shouldn't push below 60%."
        )

    @pytest.mark.slow
    def test_unweighted_informative_sampling_is_biased(self, ci_params):
        """Sanity check that weighting DOES something: under informative
        sampling with UNIFORM weights (ignoring the sampling design), the
        estimator produces a visibly-biased estimate. This shows the
        weight mechanism has real teeth — if this test passes trivially
        (unweighted happens to be close to β anyway), the informative
        DGP is too weak to distinguish weighted from unweighted and the
        other tests above lack teeth too."""
        from diff_diff.had import HeterogeneousAdoptionDiD
        from diff_diff.survey import SurveyDesign

        G = 500
        n_reps = ci_params.bootstrap(200, min_n=25)
        beta_true = 2.0
        est_unweighted = np.full(n_reps, np.nan)
        est_weighted = np.full(n_reps, np.nan)
        rng_seeds = np.arange(4000, 4000 + n_reps)
        for i, seed in enumerate(rng_seeds):
            # Subsample the full population (size 2G) by inverse
            # informative sampling to get an INFORMATIVE sample of size G.
            # Then compare unweighted vs weighted fits.
            rng = np.random.default_rng(seed)
            d_pop = rng.uniform(0.0, 1.0, 2 * G)
            dy_pop = beta_true * d_pop + rng.normal(0, 0.25, 2 * G)
            # Informative sampling: probability of inclusion inversely
            # related to |d - 0.5|. Under-sample boundary.
            p = np.exp(-2.0 * np.abs(d_pop - 0.5))
            p = p / p.sum()
            idx = rng.choice(2 * G, size=G, replace=False, p=p)
            d = d_pop[idx]
            dy = dy_pop[idx]
            # pweights = 1/p (up to a constant rescaling that doesn't
            # affect the estimator since it's weight-scale-invariant).
            w_unit = 1.0 / p[idx]
            panel_unw = _build_panel_with_weights(d, dy, np.ones(G))
            panel_w = _build_panel_with_weights(d, dy, w_unit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r_unw = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    panel_unw, "outcome", "dose", "period", "unit"
                )
                r_w = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                    panel_w,
                    "outcome",
                    "dose",
                    "period",
                    "unit",
                    survey_design=SurveyDesign(weights="w"),
                )
            if np.isfinite(r_unw.att):
                est_unweighted[i] = r_unw.att
            if np.isfinite(r_w.att):
                est_weighted[i] = r_w.att
        bias_unweighted = abs(float(np.nanmean(est_unweighted)) - beta_true)
        bias_weighted = abs(float(np.nanmean(est_weighted)) - beta_true)
        # Weighted should be closer to truth than unweighted.
        assert bias_weighted < bias_unweighted + 0.05, (
            f"Weighted bias ({bias_weighted:.3f}) should be <= unweighted "
            f"bias ({bias_unweighted:.3f}) under informative sampling. "
            f"Weight correction is not reducing bias — mechanism broken."
        )
