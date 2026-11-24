"""Methodology tests for diff_diff.design (diagnose_design).

``docs/methodology/REGISTRY.md`` ``## DesignDiagnosis`` is the contract. The
source papers evaluate their procedures on CPS data that is not reproduced
here; these tests reproduce the *shape* of their findings on simulated panels
and pin the library choices the registry documents.

Sources: Bertrand, Duflo & Mullainathan (2004); Mizushima & Powell (2025);
Blair, Cooper, Coppock & Humphreys (2019). Paper reviews:
``docs/methodology/papers/``.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from diff_diff import (
    CallawaySantAnna,
    DifferenceInDifferences,
    ImputationDiD,
    SunAbraham,
    TwoWayFixedEffects,
    diagnose_design,
)


def ar1_panel(n_units, n_periods, rho, seed, unit_sd=1.0):
    """Unit effects plus a stationary AR(1) error, no treatment anywhere."""
    rng = np.random.default_rng(seed)
    errors = np.empty((n_units, n_periods))
    state = rng.normal(size=n_units)
    for t in range(n_periods):
        state = rho * state + np.sqrt(1.0 - rho**2) * rng.normal(size=n_units)
        errors[:, t] = state
    y = rng.normal(scale=unit_sd, size=(n_units, 1)) + errors
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_periods),
            "period": np.tile(np.arange(1, n_periods + 1), n_units),
            "y": y.ravel(),
        }
    )


def diagnose(panel, estimators, **kwargs):
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("progress", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose_design(panel, "y", "unit", "period", estimators, **kwargs)


@pytest.fixture(scope="module")
def serial_correlation_result():
    """50 units x 21 periods, rho = 0.8: the Bertrand et al. AR(1) design."""
    panel = ar1_panel(50, 21, 0.8, seed=1)
    return diagnose(
        panel,
        {
            "twfe_classical": TwoWayFixedEffects(vcov_type="classical"),
            "twfe_clustered": TwoWayFixedEffects(),
        },
        cohort_periods=[11],
        treatment_fraction=0.5,
        effect_sizes=[0.25, 0.5, 1.0],
        n_simulations=300,
    )


class TestSerialCorrelationOverRejection:
    """Bertrand, Duflo & Mullainathan (2004), Table II Panel B, in shape.

    With serially correlated outcomes and an absorbing treatment, standard
    errors that ignore within-unit correlation reject a true null far above
    the nominal level; clustering on the unit restores it.
    """

    def test_unclustered_standard_errors_over_reject(self, serial_correlation_result):
        null = serial_correlation_result.results["twfe_classical"][0]
        assert null.true_effect == 0.0
        assert null.rejection_rate > 0.25
        assert serial_correlation_result.size_controlled["twfe_classical"] is False

    def test_unit_clustered_standard_errors_are_near_nominal(self, serial_correlation_result):
        null = serial_correlation_result.results["twfe_clustered"][0]
        assert null.rejection_rate < 0.12
        assert serial_correlation_result.size_controlled["twfe_clustered"] is True

    def test_both_variants_share_point_estimates(self, serial_correlation_result):
        """Same estimator, same placebo draws: only the inference differs."""
        cells = serial_correlation_result.results
        for i in range(4):
            assert cells["twfe_classical"][i].mean_estimate == pytest.approx(
                cells["twfe_clustered"][i].mean_estimate, abs=1e-10
            )

    def test_coverage_mirrors_size_under_the_null(self, serial_correlation_result):
        for cells in serial_correlation_result.results.values():
            null = cells[0]
            assert null.coverage == pytest.approx(1.0 - null.rejection_rate, abs=1e-12)


class TestEffectInjection:
    """Bertrand et al. (2004), pp.259, 261: add the effect to treated observations."""

    def test_power_rises_with_the_injected_effect(self, serial_correlation_result):
        cells = serial_correlation_result.results["twfe_clustered"]
        powers = [cell.power for cell in cells]
        assert powers[0] < powers[1] < powers[2] <= powers[3]
        assert powers[3] > 0.8

    def test_estimates_are_unbiased_for_the_injected_effect(self, serial_correlation_result):
        cells = serial_correlation_result.results["twfe_clustered"]
        for cell in cells:
            # Bias is the same at every effect: the placebo draws are shared.
            assert cell.bias == pytest.approx(cells[0].bias, abs=1e-10)
            assert abs(cell.bias) < 4.0 * cell.std_estimate / np.sqrt(cell.n_simulations)

    def test_over_rejecting_test_shows_inflated_power(self, serial_correlation_result):
        """The raw rejection rate of an over-sized test is not size-adjusted."""
        cells = serial_correlation_result.results
        assert cells["twfe_classical"][1].power > cells["twfe_clustered"][1].power


class TestFewTreatedUnits:
    """Mizushima & Powell (2025), in shape: asymptotic inference over-rejects
    when very few units are treated, and is near nominal when many are."""

    estimators = staticmethod(lambda: [CallawaySantAnna(), ImputationDiD(), SunAbraham()])

    def test_two_treated_units_over_reject(self):
        result = diagnose(
            ar1_panel(40, 12, 0.5, seed=2),
            self.estimators(),
            cohort_periods=[8],
            n_treated=2,
            n_simulations=300,
        )
        for label, cells in result.results.items():
            assert cells[0].rejection_rate > 0.15, label
            assert result.size_controlled[label] is False, label

    def test_twenty_treated_units_are_near_nominal(self):
        result = diagnose(
            ar1_panel(40, 12, 0.5, seed=2),
            self.estimators(),
            cohort_periods=[8],
            n_treated=20,
            n_simulations=300,
        )
        for label, cells in result.results.items():
            assert cells[0].rejection_rate < 0.12, label


class TestDocumentedChoices:
    """Library choices recorded as Notes in REGISTRY ``## DesignDiagnosis``."""

    def test_rmse_is_the_existing_sample_sd_form(self, serial_correlation_result):
        """Not Blair et al.'s root mean squared error: the variance term uses ddof=1."""
        cell = serial_correlation_result.results["twfe_clustered"][2]
        estimates = np.array([r["estimate"] for r in cell.simulation_results])
        bias = estimates.mean() - cell.true_effect
        assert cell.rmse == pytest.approx(np.sqrt(bias**2 + estimates.std(ddof=1) ** 2))
        root_mse = np.sqrt(np.mean((estimates - cell.true_effect) ** 2))
        n = len(estimates)
        assert cell.rmse**2 - root_mse**2 == pytest.approx(estimates.var(ddof=1) / n)

    def test_mde_is_close_to_the_normal_multiplier(self):
        """For a correctly sized test the MDE is about (z_{1-a/2} + z_power) standard errors."""
        panel = ar1_panel(60, 10, 0.0, seed=3)
        result = diagnose(
            panel,
            {"did": DifferenceInDifferences(cluster="unit")},
            cohort_periods=[6],
            treatment_fraction=0.5,
            n_simulations=300,
            power=0.8,
        )
        assert result.size_controlled["did"] is True
        spread = result.results["did"][0].std_estimate
        multiplier = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        assert result.mde["did"] == pytest.approx(multiplier * spread, rel=0.25)
        assert result.mde_results["did"].power_at_mde >= 0.8

    def test_no_mde_for_an_estimator_that_fails_the_size_check(self):
        result = diagnose(
            ar1_panel(50, 21, 0.8, seed=1),
            {"twfe_classical": TwoWayFixedEffects(vcov_type="classical")},
            cohort_periods=[11],
            treatment_fraction=0.5,
            n_simulations=200,
            power=0.8,
        )
        assert result.size_controlled["twfe_classical"] is False
        assert np.isnan(result.mde["twfe_classical"])
        assert result.mde_results == {}

    def test_fixed_panel_and_resampled_panel_both_run(self):
        """``resample_units=False`` reassigns treatment on the panel as observed."""
        panel = ar1_panel(30, 10, 0.3, seed=4)
        kwargs = dict(cohort_periods=[6], treatment_fraction=0.5, n_simulations=100)
        estimator = {"did": DifferenceInDifferences(cluster="unit")}
        resampled = diagnose(panel, estimator, **kwargs)
        fixed = diagnose(panel, estimator, resample_units=False, **kwargs)
        assert resampled.resample_units is True and fixed.resample_units is False
        assert fixed.results["did"][0].n_simulations == 100
        # Different designs, hence different draws.
        assert fixed.results["did"][0].mean_estimate != resampled.results["did"][0].mean_estimate

    def test_staggered_declaration_draws_each_unit_from_the_declared_periods(self):
        panel = ar1_panel(40, 12, 0.3, seed=5)
        result = diagnose(
            panel,
            [CallawaySantAnna(), ImputationDiD()],
            cohort_periods=[6, 8, 10],
            n_treated=18,
            effect_sizes=[1.0],
            n_simulations=100,
        )
        for label, cells in result.results.items():
            assert cells[1].mean_estimate == pytest.approx(1.0, abs=0.15), label
            assert cells[1].power > cells[0].power, label


@pytest.fixture(scope="module")
def locked():
    """The AR(1) comparison of two variants, and the lock of one of them."""
    panel = ar1_panel(50, 21, 0.8, seed=1)
    result = diagnose(
        panel,
        {
            "twfe_classical": TwoWayFixedEffects(vcov_type="classical"),
            "twfe_clustered": TwoWayFixedEffects(),
        },
        cohort_periods=[11],
        treatment_fraction=0.5,
        effect_sizes=[0.5],
        n_simulations=300,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lock = result.lock("twfe_clustered", data=panel, progress=False)
    return result, lock


class TestLockRerun:
    """Library choice (REGISTRY ``## DesignDiagnosis``, lock rerun).

    When several estimators were compared, the lock reruns the chosen one
    alone on the same panel with new placebo draws. The tests pin that the
    rerun is a correctly sized, separate run of the declared design. How much
    of the overstatement from choosing it removes is measured by
    ``benchmarks/python/design_lock_rerun.py`` (see ``TestLockRerunEvidence``).
    """

    def test_rerun_uses_the_diagnosed_panel(self, locked):
        result, lock = locked
        assert result.n_periods == lock.n_periods == 21
        assert lock.data_sha256 == result.data_sha256
        assert lock.time_periods == result.time_periods

    def test_rerun_of_a_correctly_sized_estimator_is_near_nominal(self, locked):
        _, lock = locked
        null = lock.rerun_results[0]
        assert null["true_effect"] == 0.0
        assert null["rejection_rate"] < 0.12
        assert lock.size_controlled is True

    def test_rerun_is_separate_from_the_comparison(self, locked):
        result, lock = locked
        compared = result.results["twfe_clustered"]
        for cell, row in zip(compared, lock.rerun_results):
            assert row["true_effect"] == cell.true_effect
            assert row["mean_estimate"] != cell.mean_estimate
            assert row["n_simulations"] == cell.n_simulations


# Simulation artifact (generated by benchmarks/python/design_lock_rerun.py)
EVIDENCE = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "design_lock_rerun.json"
LONG_WINDOW = ("S3_window", "S5_default_replicates")


@pytest.fixture(scope="module")
def evidence():
    # Skip, not fail, when the file is absent: the CI matrix runs the tests from an
    # isolated copy of tests/ that has no benchmarks/ directory (the repo convention
    # for every test reading benchmarks/data). The file is tracked, so runs from a
    # checkout read it.
    if not EVIDENCE.exists():
        pytest.skip(
            "benchmarks/data/design_lock_rerun.json is absent; regenerate it with "
            "`python benchmarks/python/design_lock_rerun.py --output "
            "benchmarks/data/design_lock_rerun.json`."
        )
    return json.loads(EVIDENCE.read_text())["scenarios"]


class TestLockRerunEvidence:
    """The numbers REGISTRY ``Note (lock rerun)`` quotes from the simulation."""

    def test_scenarios(self, evidence):
        assert set(evidence) == {
            "S1_typical",
            "S2_staggered",
            "S3_window",
            "S4_specification_search",
            "S5_default_replicates",
        }
        for scenario in evidence.values():
            assert scenario["n_panels"] == 400

    def test_holding_out_periods_moves_the_rerun_by_under_one_point(self, evidence):
        for name, scenario in evidence.items():
            assert abs(scenario["paired_difference"]) <= 0.01, name

    def test_selection_effect_is_small_on_the_same_panel(self, evidence):
        for name, scenario in evidence.items():
            arm = scenario["same_panel"]
            assert abs(arm["comparison_minus_reference"]) <= 0.03, name
            assert abs(arm["rerun_minus_reference"]) <= 0.02, name

    def test_shortened_comparison_understates_power_on_long_windows(self, evidence):
        for name in LONG_WINDOW:
            assert evidence[name]["held_out"]["comparison_minus_reference"] <= -0.10, name

    def test_lock_reports_the_rerun(self, evidence):
        recorded = [s["lock_matches_rerun"] for s in evidence.values() if s["lock_matches_rerun"]]
        assert len(recorded) == 4  # not the specification search: one estimator per call
        assert all(all(flags) for flags in recorded)
