"""Shared factories for the Phase 2 results-contract test files.

Two kinds of factories:

- ``make_constructed_diagnostics()`` - directly constructed instances of
  every class-backed diagnostic roster member EXCEPT
  ``BaconDecompositionResults`` (which is produced by a real fit in the
  test files - its container is populated by the decomposition and its
  ``summary()`` renders the comparison table). Synthetic values are
  internally consistent so ``summary()`` / ``to_dataframe()`` render.
- Small-fit panel builders shared by the event-study surface tests and
  the serialization tests (see ``staggered_panel()`` and friends).

Kept in ``tests/helpers`` (on ``sys.path`` via ``tests/conftest.py``) so
multiple test modules can import the same factories without duplicating
panel constructions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

import diff_diff
from diff_diff.diagnostic_report import DiagnosticReportResults


def make_constructed_diagnostics() -> Dict[str, Any]:
    """Return {class_name: instance} for the direct-construction roster."""
    rng_bins = pd.DataFrame(
        {
            "rdplot_id": [1, 2],
            "rdplot_mean_bin": [-0.5, 0.5],
            "rdplot_mean_x": [-0.5, 0.5],
            "rdplot_mean_y": [1.0, 2.0],
            "rdplot_N": [10, 10],
            "rdplot_ci_l": [0.8, 1.8],
            "rdplot_ci_r": [1.2, 2.2],
        }
    )
    poly = pd.DataFrame({"rdplot_x": [-1.0, 0.0, 1.0], "rdplot_y": [0.9, 1.4, 2.1]})
    coef = pd.DataFrame({"side": ["left", "right"], "coef_0": [1.0, 2.0]})

    qug = diff_diff.QUGTestResults(
        t_stat=1.2,
        p_value=0.23,
        reject=False,
        alpha=0.05,
        critical_value=1.96,
        n_obs=50,
        n_excluded_zero=0,
        d_order_1=0.1,
        d_order_2=0.2,
    )
    stute = diff_diff.StuteTestResults(
        cvm_stat=0.4,
        p_value=0.31,
        reject=False,
        alpha=0.05,
        n_bootstrap=99,
        n_obs=50,
        seed=42,
    )
    yatchew = diff_diff.YatchewTestResults(
        t_stat_hr=0.8,
        p_value=0.42,
        reject=False,
        alpha=0.05,
        critical_value=1.64,
        sigma2_lin=1.1,
        sigma2_diff=1.0,
        sigma2_W=0.9,
        n_obs=50,
    )

    instances: Dict[str, Any] = {
        "RDPlotResult": diff_diff.RDPlotResult(
            coef=coef,
            vars_bins=rng_bins,
            vars_poly=poly,
            J=(2.0, 2.0),
            J_IMSE=(2.0, 2.0),
            J_MV=(3.0, 3.0),
            scale=(1.0, 1.0),
            rscale=(1.0, 1.0),
            bin_avg=(5.0, 5.0),
            bin_med=(5.0, 5.0),
            p=4,
            cutoff=0.0,
            h=(1.0, 1.0),
            N=(10, 10),
            N_h=(10, 10),
            binselect="esmv",
            kernel_type="Uniform",
            ci_level=95.0,
            ci_requested=False,
        ),
        "RDDensityTestResult": diff_diff.RDDensityTestResult(
            t_stat=-0.5,
            p_value=0.62,
            f_left=0.35,
            f_right=0.33,
            f_diff=-0.02,
            se_left=0.04,
            se_right=0.04,
            se_diff=0.05,
            f_left_conventional=None,
            f_right_conventional=None,
            f_diff_conventional=None,
            se_left_conventional=None,
            se_right_conventional=None,
            se_diff_conventional=None,
            t_stat_conventional=None,
            p_value_conventional=None,
            n=1000,
            n_left=500,
            n_right=500,
            n_eff_left=200,
            n_eff_right=210,
            h_left=0.5,
            h_right=0.55,
            bandwidths=None,
            cutoff=0.0,
            p=2,
            q=3,
            fitselect="unrestricted",
            kernel="triangular",
            vcov_type="jackknife",
            bwselect="comb",
            bandwidth_method="estimated",
            masspoints="adjust",
            masspoints_adjusted=False,
            regularize=True,
            n_local_min=23,
            n_unique_min=23,
            report_all=False,
        ),
        "HonestDiDResults": diff_diff.HonestDiDResults(
            lb=0.1,
            ub=0.9,
            ci_lb=0.05,
            ci_ub=0.95,
            M=1.0,
            method="relative_magnitude",
            original_estimate=0.5,
            original_se=0.2,
        ),
        "SensitivityResults": diff_diff.SensitivityResults(
            M_values=np.array([0.5, 1.0]),
            bounds=[(0.2, 0.8), (0.1, 0.9)],
            robust_cis=[(0.1, 0.9), (0.0, 1.0)],
            breakdown_M=1.0,
            method="relative_magnitude",
            original_estimate=0.5,
            original_se=0.2,
        ),
        "PreTrendsPowerResults": diff_diff.PreTrendsPowerResults(
            power=0.62,
            mdv=0.8,
            violation_magnitude=0.5,
            violation_type="linear",
            alpha=0.05,
            target_power=0.8,
            n_pre_periods=3,
            test_statistic=2.1,
            critical_value=7.8,
            noncentrality=3.3,
            pre_period_effects=np.array([0.01, -0.02, 0.03]),
            pre_period_ses=np.array([0.05, 0.05, 0.05]),
            vcov=np.eye(3) * 0.0025,
        ),
        "PreTrendsPowerCurve": diff_diff.PreTrendsPowerCurve(
            M_values=np.array([0.0, 0.5, 1.0]),
            powers=np.array([0.05, 0.44, 0.91]),
            mdv=0.8,
            alpha=0.05,
            target_power=0.8,
            violation_type="linear",
        ),
        "PowerResults": diff_diff.PowerResults(
            power=0.8,
            mde=0.25,
            required_n=120,
            effect_size=0.3,
            alpha=0.05,
            alternative="two-sided",
            n_treated=60,
            n_control=60,
            n_pre=4,
            n_post=4,
            sigma=1.0,
        ),
        "SimulationPowerResults": diff_diff.SimulationPowerResults(
            power=0.78,
            power_se=0.04,
            power_ci=(0.7, 0.86),
            rejection_rate=0.78,
            mean_estimate=0.29,
            std_estimate=0.11,
            mean_se=0.1,
            coverage=0.94,
            n_simulations=100,
            effect_sizes=[0.3],
            powers=[0.78],
            true_effect=0.3,
            alpha=0.05,
            estimator_name="DifferenceInDifferences",
        ),
        "SimulationMDEResults": diff_diff.SimulationMDEResults(
            mde=0.31,
            power_at_mde=0.8,
            target_power=0.8,
            alpha=0.05,
            n_units=100,
            n_simulations_per_step=50,
            n_steps=6,
            search_path=[{"effect": 0.3, "power": 0.75}],
            estimator_name="DifferenceInDifferences",
        ),
        "SimulationSampleSizeResults": diff_diff.SimulationSampleSizeResults(
            required_n=140,
            power_at_n=0.81,
            target_power=0.8,
            alpha=0.05,
            effect_size=0.3,
            n_simulations_per_step=50,
            n_steps=6,
            search_path=[{"n": 140, "power": 0.81}],
            estimator_name="DifferenceInDifferences",
        ),
        "PlaceboTestResults": diff_diff.PlaceboTestResults(
            test_type="placebo_treatment",
            placebo_effect=0.02,
            se=0.05,
            t_stat=0.4,
            p_value=0.69,
            conf_int=(-0.08, 0.12),
            n_obs=200,
            is_significant=False,
        ),
        "QUGTestResults": qug,
        "StuteTestResults": stute,
        "YatchewTestResults": yatchew,
        "StuteJointResult": diff_diff.StuteJointResult(
            cvm_stat_joint=0.6,
            p_value=0.27,
            reject=False,
            alpha=0.05,
            horizon_labels=["e0", "e1"],
            per_horizon_stats={"e0": 0.3, "e1": 0.3},
            n_bootstrap=99,
            n_obs=50,
            n_horizons=2,
            seed=42,
            null_form="linear",
            exact_linear_short_circuited=False,
        ),
        "HADPretestReport": diff_diff.HADPretestReport(
            qug=qug,
            stute=stute,
            yatchew=yatchew,
            all_pass=True,
            verdict="pass",
            alpha=0.05,
            n_obs=50,
        ),
        "DiagnosticReportResults": DiagnosticReportResults(
            schema={"schema_version": "2.0", "estimator": "DiDResults"},
            interpretation="All applicable checks passed.",
            applicable_checks=("parallel_trends",),
        ),
    }
    return instances
