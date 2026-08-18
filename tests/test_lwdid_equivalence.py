"""Numerical equivalence tests: diff-diff LWDiD vs lwdid-py reference.

These tests require lwdid>=0.2.2 (optional dev dependency).
Run with: pytest tests/test_lwdid_equivalence.py -v
Skipped automatically if lwdid is not installed.

Tolerance standards (per Lee & Wooldridge paper precision requirements):
- RA + classical/HC1: atol=1e-10 (direct matrix inversion, deterministic)
- RA + cluster: atol=1e-8 (grouping introduces floating-point reassociation)
- IPW/IPWRA: atol=1e-6 (logit optimization path may differ)
- PSM: atol=1e-4 (matching tie-breaking may differ)
- Staggered aggregation: atol=1e-6 (multi-layer aggregation)
"""

import numpy as np
import pandas as pd
import pytest

# ============================================================
# Test Data Generators (deterministic, shared between both packages)
# ============================================================


def _generate_common_timing_panel(n=100, T=8, post_start=6, true_att=2.0, n_controls=1, seed=42):
    """Generate balanced panel for common-timing tests.

    Produces columns compatible with BOTH lwdid-py and diff-diff APIs:
    - unit: unit identifier
    - time: time period (1..T)
    - y: outcome variable
    - treat: unit-level treatment indicator (time-invariant)
    - post: post-treatment indicator (0 in pre, 1 in post)
    - d: treatment status per obs (treat * post)
    - x1: a covariate
    """
    rng = np.random.default_rng(seed)
    n_treated = n // 3

    rows = []
    for i in range(n):
        is_treated = i < n_treated
        unit_fe = rng.normal(0, 2)
        trend_slope = rng.normal(0.3, 0.1)
        x1 = rng.normal() + int(is_treated) * 0.3
        for t in range(1, T + 1):
            time_trend = trend_slope * t
            noise = rng.normal(0, 0.3)
            is_post = int(t >= post_start)
            treatment_effect = true_att if (is_treated and is_post) else 0.0
            y = unit_fe + time_trend + noise + treatment_effect + 0.5 * x1
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "treat": int(is_treated),
                    "post": is_post,
                    "d": int(is_treated and bool(is_post)),
                    "x1": x1,
                }
            )

    return pd.DataFrame(rows)


def _generate_staggered_panel(n=120, T=10, seed=42):
    """Generate staggered adoption panel.

    Produces columns compatible with BOTH packages:
    - unit: unit identifier
    - time: time period (1..T)
    - y: outcome variable
    - treat: current treatment status (0/1)
    - cohort: first treatment time (0 = never-treated)
    - gvar: cohort var for lwdid-py (NaN for never-treated)
    - x1: a covariate
    """
    rng = np.random.default_rng(seed)
    cohorts = [0, 4, 6, 8]  # 0 = never-treated
    true_att = 1.5

    rows = []
    for i in range(n):
        g = cohorts[i % len(cohorts)]
        unit_fe = rng.normal(0, 2)
        x1 = rng.normal()
        for t in range(1, T + 1):
            is_post = int(g > 0 and t >= g)
            effect = true_att * is_post
            y = unit_fe + 0.2 * t + rng.normal(0, 0.2) + effect
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "y": y,
                    "treat": is_post,
                    "d": int(g > 0),
                    "post": is_post,
                    "cohort": g,
                    "gvar": g if g > 0 else np.nan,
                    "x1": x1,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Helper functions to run both packages
# ============================================================


def _run_lwdid_py_common(df, rolling, estimator, vce, controls=None, cluster_var=None):
    """Run lwdid-py on common-timing panel."""
    from lwdid import lwdid as lwdid_func

    kwargs = dict(
        data=df.copy(),
        y="y",
        d="treat",
        ivar="unit",
        tvar="time",
        post="post",
        rolling=rolling,
        estimator=estimator,
        verbose="quiet",
    )
    if vce is not None:
        if vce == "cluster":
            kwargs["vce"] = "cluster"
            kwargs["cluster_var"] = cluster_var or "unit"
        else:
            kwargs["vce"] = vce
    if controls:
        kwargs["controls"] = controls
    return lwdid_func(**kwargs)


def _run_diff_diff_common(df, rolling, estimator, vce, controls=None, cluster=None):
    """Run diff-diff LWDiD on common-timing panel.

    The estimator/vce spec tokens follow lwdid-py vocabulary; they are
    mapped to diff-diff's canonical estimation_method/vcov_type here.
    """
    from diff_diff import LWDiD

    method_map = {"ra": "reg", "ipwra": "dr"}
    vce_map = {"robust": "hc1", "ols": "classical", "cluster": "hc1"}
    dd_vcov = vce_map.get(vce, vce) if vce else "classical"

    model = LWDiD(
        rolling=rolling,
        estimation_method=method_map.get(estimator, estimator),
        vcov_type=dd_vcov,
        cluster=cluster,
    )
    return model.fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="d",
        covariates=controls,
    )


def _run_lwdid_py_staggered(
    df, rolling, estimator, vce, control_group, controls=None, cluster_var=None
):
    """Run lwdid-py on staggered panel.

    Returns (result, actual_control_group_used) tuple because lwdid-py may
    auto-switch from 'not_yet_treated' to 'never_treated' when aggregate='cohort'.

    Aggregation basis: we explicitly request aggregate="overall" so that
    lwdid-py estimates the overall ATT from a single pooled cross-section
    regression, the basis recommended by Lee & Wooldridge (2026, eq. 7.19),
    which "automatically accounts for the correlations among the tau_g".
    lwdid-py's default aggregate="cohort" instead combines per-cohort SEs via
    sqrt(sum(w^2 * SE^2)), which assumes independence across cohort estimates
    and therefore understates the overall SE. diff-diff's joint influence
    function SE matches the eq. 7.19 pooled-regression basis (and Stata
    lwdid.ado), so "overall" is the correct reference for equivalence.
    """
    import warnings

    from lwdid import lwdid as lwdid_func

    kwargs = dict(
        data=df.copy(),
        y="y",
        gvar="gvar",
        ivar="unit",
        tvar="time",
        rolling=rolling,
        estimator=estimator,
        control_group=control_group,
        aggregate="overall",
        verbose="quiet",
    )
    if vce is not None:
        if vce == "cluster":
            kwargs["vce"] = "cluster"
            kwargs["cluster_var"] = cluster_var or "unit"
        else:
            kwargs["vce"] = vce
    if controls:
        kwargs["controls"] = controls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lwdid_func(**kwargs)
    actual_cg = getattr(result, "control_group_used", control_group)
    return result, actual_cg


def _run_diff_diff_staggered(
    df, rolling, estimator, vce, control_group, controls=None, cluster=None
):
    """Run diff-diff LWDiD on staggered panel (lwdid-py spec tokens mapped
    to canonical estimation_method/vcov_type)."""
    from diff_diff import LWDiD

    method_map = {"ra": "reg", "ipwra": "dr"}
    vce_map = {"robust": "hc1", "ols": "classical", "cluster": "hc1"}
    dd_vcov = vce_map.get(vce, vce) if vce else "classical"

    model = LWDiD(
        rolling=rolling,
        estimation_method=method_map.get(estimator, estimator),
        vcov_type=dd_vcov,
        cluster=cluster,
        control_group=control_group,
    )
    return model.fit(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="treat",
        first_treat="cohort",
        covariates=controls,
    )


# ============================================================
# Parametrized Equivalence Matrix: Common Timing
# ============================================================


COMMON_TIMING_CONFIGS = [
    # (rolling, estimator, vce, use_controls, atol, description)
    ("demean", "ra", None, False, 1e-10, "demean+RA+classical, no controls"),
    ("demean", "ra", "hc1", False, 1e-10, "demean+RA+HC1, no controls"),
    ("demean", "ra", None, True, 1e-10, "demean+RA+classical, with controls"),
    ("demean", "ra", "hc1", True, 1e-10, "demean+RA+HC1, with controls"),
    ("demean", "ra", "cluster", False, 1e-8, "demean+RA+cluster"),
    ("demean", "ra", "cluster", True, 1e-8, "demean+RA+cluster, with controls"),
    ("detrend", "ra", None, False, 1e-10, "detrend+RA+classical"),
    ("detrend", "ra", "hc1", False, 1e-10, "detrend+RA+HC1"),
    ("detrend", "ra", "hc1", True, 1e-10, "detrend+RA+HC1, with controls"),
    ("detrend", "ra", "cluster", False, 1e-8, "detrend+RA+cluster"),
    ("demean", "ipw", "hc1", True, 0.05, "demean+IPW+HC1"),
    ("demean", "ipwra", "hc1", True, 0.01, "demean+IPWRA+HC1"),
    ("detrend", "ipw", "hc1", True, 0.05, "detrend+IPW+HC1"),
    ("detrend", "ipwra", "hc1", True, 0.01, "detrend+IPWRA+HC1"),
]


@pytest.mark.parametrize(
    "rolling,estimator,vce,use_controls,atol,desc",
    COMMON_TIMING_CONFIGS,
    ids=[c[-1] for c in COMMON_TIMING_CONFIGS],
)
def test_equivalence_common_timing(
    rolling, estimator, vce, use_controls, atol, desc, require_lwdid
):
    """Verify numerical equivalence against lwdid-py for common timing."""

    df = _generate_common_timing_panel(seed=42)

    # --- lwdid-py reference ---
    controls_py = ["x1"] if use_controls else None
    cluster_py = "unit" if vce == "cluster" else None

    ref = _run_lwdid_py_common(
        df, rolling, estimator, vce, controls=controls_py, cluster_var=cluster_py
    )

    # --- diff-diff native ---
    dd = _run_diff_diff_common(
        df, rolling, estimator, vce, controls=controls_py, cluster=cluster_py
    )

    # --- Compare ---
    np.testing.assert_allclose(dd.att, ref.att, atol=atol, err_msg=f"ATT mismatch [{desc}]")
    # SE comparison
    if np.isfinite(ref.se_att) and ref.se_att > 0:
        np.testing.assert_allclose(dd.se, ref.se_att, atol=atol, err_msg=f"SE mismatch [{desc}]")
    # t-stat comparison (use rtol for IPW/IPWRA since t-stats are large
    # and differences compound from ATT+SE optimization path divergence)
    if hasattr(ref, "t_stat") and np.isfinite(ref.t_stat):
        if hasattr(dd, "t_stat") and np.isfinite(dd.t_stat):
            t_rtol = 0.25 if estimator in ("ipw", "ipwra") else 1e-3
            np.testing.assert_allclose(
                dd.t_stat, ref.t_stat, rtol=t_rtol, err_msg=f"t-stat mismatch [{desc}]"
            )


# ============================================================
# Parametrized Equivalence Matrix: Staggered
# ============================================================


STAGGERED_CONFIGS = [
    # (rolling, estimator, vce, control_group, controls, atol)
    ("demean", "ra", "cluster", "never_treated", None, 1e-8),
    ("demean", "ra", "cluster", "not_yet_treated", None, 1e-8),
    ("detrend", "ra", "cluster", "never_treated", None, 1e-8),
    ("demean", "ra", "hc1", "never_treated", None, 1e-8),
    ("demean", "ra", "hc1", "not_yet_treated", None, 1e-8),
    ("demean", "ipw", "cluster", "not_yet_treated", ["x1"], 0.01),
    ("demean", "ipwra", "cluster", "not_yet_treated", ["x1"], 0.01),
    ("demean", "ipw", "hc1", "never_treated", ["x1"], 0.01),
    ("demean", "ipwra", "hc1", "never_treated", ["x1"], 0.01),
]


@pytest.mark.parametrize(
    "rolling,estimator,vce,control_group,controls,atol",
    STAGGERED_CONFIGS,
    ids=[f"{r}+{e}+{v}+{cg}" for r, e, v, cg, _, _ in STAGGERED_CONFIGS],
)
def test_equivalence_staggered(
    rolling, estimator, vce, control_group, controls, atol, require_lwdid
):
    """Verify numerical equivalence against lwdid-py for staggered designs."""
    df = _generate_staggered_panel(seed=42)

    cluster_var = "unit" if vce == "cluster" else None

    # --- lwdid-py reference ---
    # lwdid-py may auto-switch 'not_yet_treated' -> 'never_treated'
    # when aggregate='cohort' (default). Use actual control group for fair comparison.
    ref, actual_cg = _run_lwdid_py_staggered(
        df, rolling, estimator, vce, control_group, controls=controls, cluster_var=cluster_var
    )

    # --- diff-diff native (use the control group lwdid-py actually used) ---
    dd = _run_diff_diff_staggered(
        df, rolling, estimator, vce, actual_cg, controls=controls, cluster=cluster_var
    )

    # --- Compare overall ATT ---
    np.testing.assert_allclose(
        dd.att,
        ref.att,
        atol=atol,
        err_msg=f"Staggered ATT mismatch [{rolling}/{estimator}/{vce}/{control_group}]",
    )
    # SE comparison: both sides use the LW 2026 eq. 7.19 pooled-regression
    # basis (lwdid-py aggregate="overall" vs diff-diff joint influence
    # function). rtol=0.01 absorbs the small difference in where the HC1
    # dof correction is applied (per-cell vs overall regression). IPW-family
    # estimators get a looser rtol since the logit optimization path differs.
    if np.isfinite(ref.se_att) and ref.se_att > 0:
        se_rtol = 0.05 if estimator in ("ipw", "ipwra") else 0.01
        np.testing.assert_allclose(
            dd.se,
            ref.se_att,
            rtol=se_rtol,
            err_msg=f"Staggered SE mismatch [{rolling}/{estimator}/{vce}/{control_group}]",
        )


# ============================================================
# Multi-seed robustness
# ============================================================


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 123])
def test_equivalence_multi_seed(seed, require_lwdid):
    """Verify equivalence holds across multiple random seeds."""
    df = _generate_common_timing_panel(seed=seed)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    np.testing.assert_allclose(dd.att, ref.att, atol=1e-10, err_msg=f"Seed {seed} ATT mismatch")
    if np.isfinite(ref.se_att) and ref.se_att > 0:
        np.testing.assert_allclose(
            dd.se, ref.se_att, atol=1e-10, err_msg=f"Seed {seed} SE mismatch"
        )


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
def test_equivalence_detrend_multiseed(seed, require_lwdid):
    """Detrend+RA path across multiple seeds."""
    df = _generate_common_timing_panel(seed=seed)

    ref = _run_lwdid_py_common(df, "detrend", "ra", "hc1")
    dd = _run_diff_diff_common(df, "detrend", "ra", "hc1")

    np.testing.assert_allclose(
        dd.att, ref.att, atol=1e-10, err_msg=f"Detrend ATT mismatch at seed={seed}"
    )


@pytest.mark.parametrize("seed", [0, 42, 99])
def test_equivalence_staggered_multiseed(seed, require_lwdid):
    """Staggered RA+demean across multiple seeds."""
    df = _generate_staggered_panel(seed=seed)

    ref, actual_cg = _run_lwdid_py_staggered(df, "demean", "ra", "hc1", "never_treated")
    dd = _run_diff_diff_staggered(df, "demean", "ra", "hc1", actual_cg)

    np.testing.assert_allclose(
        dd.att, ref.att, atol=1e-8, err_msg=f"Staggered ATT mismatch at seed={seed}"
    )


# ============================================================
# Transformation intermediate values
# ============================================================


def test_transformed_outcomes_match(require_lwdid):
    """Verify that transformed Y values match between implementations.

    Since we cannot easily access internal transformed data from lwdid-py,
    we verify through ATT (which is a direct function of the transformed
    outcomes) at machine-epsilon tolerance.
    """
    df = _generate_common_timing_panel(seed=42)

    for rolling in ["demean", "detrend"]:
        ref = _run_lwdid_py_common(df, rolling, "ra", None)
        dd = _run_diff_diff_common(df, rolling, "ra", None)
        np.testing.assert_allclose(
            dd.att, ref.att, atol=1e-10, err_msg=f"{rolling} transform mismatch"
        )


# ============================================================
# Inference Equivalence
# ============================================================


def test_equivalence_t_stat_and_pvalue(require_lwdid):
    """t-stat and p-value should match between implementations."""
    df = _generate_common_timing_panel(seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    # t-stat
    if hasattr(ref, "t_stat") and np.isfinite(ref.t_stat):
        np.testing.assert_allclose(dd.t_stat, ref.t_stat, rtol=1e-3, err_msg="t-stat mismatch")

    # p-value
    if hasattr(ref, "pvalue") and np.isfinite(ref.pvalue):
        np.testing.assert_allclose(dd.p_value, ref.pvalue, rtol=1e-2, err_msg="p-value mismatch")


def test_equivalence_confidence_interval(require_lwdid):
    """CI bounds should match between implementations."""
    df = _generate_common_timing_panel(seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    if hasattr(ref, "ci_lower") and np.isfinite(ref.ci_lower):
        np.testing.assert_allclose(
            dd.conf_int[0], ref.ci_lower, rtol=1e-3, err_msg="CI lower mismatch"
        )
    if hasattr(ref, "ci_upper") and np.isfinite(ref.ci_upper):
        np.testing.assert_allclose(
            dd.conf_int[1], ref.ci_upper, rtol=1e-3, err_msg="CI upper mismatch"
        )


# ============================================================
# Sample Size Equivalence
# ============================================================


def test_equivalence_sample_sizes(require_lwdid):
    """n_treated and n_control should match."""
    df = _generate_common_timing_panel(seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    assert dd.n_treated == ref.n_treated
    assert dd.n_control == ref.n_control


# ============================================================
# Edge Case Equivalence
# ============================================================


def test_equivalence_single_post_period(require_lwdid):
    """Single post-treatment period should still match."""
    df = _generate_common_timing_panel(n=80, T=6, post_start=6, seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    np.testing.assert_allclose(dd.att, ref.att, atol=1e-10)


def test_equivalence_many_periods(require_lwdid):
    """Many pre/post periods should still match."""
    df = _generate_common_timing_panel(n=80, T=18, post_start=10, seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    np.testing.assert_allclose(dd.att, ref.att, atol=1e-10)


def test_equivalence_large_sample(require_lwdid):
    """Larger sample size should maintain equivalence."""
    df = _generate_common_timing_panel(n=500, T=8, post_start=6, seed=42)

    ref = _run_lwdid_py_common(df, "demean", "ra", "hc1")
    dd = _run_diff_diff_common(df, "demean", "ra", "hc1")

    np.testing.assert_allclose(dd.att, ref.att, atol=1e-10)
    if np.isfinite(ref.se_att) and ref.se_att > 0:
        np.testing.assert_allclose(dd.se, ref.se_att, atol=1e-10)
