"""
Practitioner guidance for Difference-in-Differences analysis.

Implements Baker et al. (2025) "Difference-in-Differences Designs:
A Practitioner's Guide" as context-aware runtime guidance. Call
``practitioner_next_steps(results)`` after estimation to get a
structured set of recommended next steps.
"""

import math
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Valid step names (Baker et al. 8-step framework)
# ---------------------------------------------------------------------------
STEPS: Set[str] = {
    "target_parameter",
    "assumptions",
    "parallel_trends",
    "estimator_selection",
    "estimation",
    "sensitivity",
    "placebo",
    "heterogeneity",
    "robustness",
}

# ---------------------------------------------------------------------------
# Estimator name mapping
# ---------------------------------------------------------------------------
_ESTIMATOR_NAMES: Dict[str, str] = {
    "DiDResults": "DifferenceInDifferences",
    "MultiPeriodDiDResults": "MultiPeriodDiD (Event Study)",
    "CallawaySantAnnaResults": "CallawaySantAnna",
    "SunAbrahamResults": "SunAbraham",
    "ImputationDiDResults": "ImputationDiD (Borusyak-Jaravel-Spiess)",
    "TwoStageDiDResults": "TwoStageDiD (Gardner)",
    "StackedDiDResults": "StackedDiD",
    "SyntheticDiDResults": "SyntheticDiD",
    "TROPResults": "TROP",
    "SyntheticControlResults": "SyntheticControl",
    "EfficientDiDResults": "EfficientDiD",
    "ContinuousDiDResults": "ContinuousDiD",
    "TripleDifferenceResults": "TripleDifference (DDD)",
    "BaconDecompositionResults": "BaconDecomposition",
    "HeterogeneousAdoptionDiDResults": "HeterogeneousAdoptionDiD (HAD)",
    "HeterogeneousAdoptionDiDEventStudyResults": "HeterogeneousAdoptionDiD (Event Study)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def practitioner_next_steps(
    results: Any,
    *,
    completed_steps: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Context-aware practitioner guidance based on Baker et al. (2025).

    Inspects the type and attributes of *results* to recommend which
    Baker et al. steps remain. Returns a structured dict and optionally
    prints a human-readable summary.

    Parameters
    ----------
    results : Any
        A diff-diff results object (e.g. ``DiDResults``,
        ``CallawaySantAnnaResults``, etc.).
    completed_steps : list of str, optional
        Steps the caller has already completed. Valid names:
        ``"target_parameter"``, ``"assumptions"``, ``"parallel_trends"``,
        ``"estimator_selection"``, ``"estimation"``, ``"sensitivity"``,
        ``"placebo"``, ``"heterogeneity"``, ``"robustness"``.
    verbose : bool, default True
        If True, print a human-readable summary to stdout.

    Returns
    -------
    dict
        Keys: ``"estimator"`` (str), ``"completed"`` (list of str),
        ``"next_steps"`` (list of dict), ``"warnings"`` (list of str).
        Each next_step dict has: ``"baker_step"`` (int), ``"label"`` (str),
        ``"why"`` (str), ``"code"`` (str), ``"priority"`` (str).
    """
    completed = set(completed_steps or [])
    unknown = completed - STEPS
    if unknown:
        raise ValueError(f"Unknown step names: {unknown}. Valid names: {sorted(STEPS)}")

    # Estimation is always complete if we have a results object
    completed.add("estimation")

    type_name = type(results).__name__
    handler = _HANDLERS.get(type_name, _handle_generic)
    steps, warnings = handler(results)

    # Prepend Steps 1-2 (pre-estimation reasoning) to every handler's output.
    # These are always relevant and filterable via completed_steps.
    pre_estimation = [
        _step(
            baker_step=1,
            label="Define target parameter",
            why=(
                "State explicitly what causal effect you are estimating "
                "(ATT, ATT(g,t), weighted/unweighted) and what policy "
                "question it answers."
            ),
            code="# What is the target parameter? ATT? Weighted or unweighted?",
            priority="high",
            step_name="target_parameter",
        ),
        _step(
            baker_step=2,
            label="State identification assumptions",
            why=(
                "Name the parallel trends variant you are invoking "
                "(unconditional, conditional, PT-GT-NYT, etc.), the "
                "no-anticipation assumption, and any overlap conditions."
            ),
            code="# Which PT variant? No-anticipation? Overlap?",
            priority="high",
            step_name="assumptions",
        ),
    ]
    steps = pre_estimation + steps

    # Filter out completed steps
    steps = _filter_steps(steps, completed)

    output = {
        "estimator": _ESTIMATOR_NAMES.get(type_name, type_name),
        "completed": sorted(completed),
        "next_steps": steps,
        "warnings": warnings,
    }

    if verbose:
        _print_output(output)

    return output


# ---------------------------------------------------------------------------
# Step builder helper
# ---------------------------------------------------------------------------
def _step(
    baker_step: int,
    label: str,
    why: str,
    code: str,
    priority: str = "high",
    step_name: str = "",
) -> Dict[str, Any]:
    return {
        "baker_step": baker_step,
        "label": label,
        "why": why,
        "code": code,
        "priority": priority,
        "_step_name": step_name,
    }


# ---------------------------------------------------------------------------
# Common steps reused across handlers
# ---------------------------------------------------------------------------
def _parallel_trends_step(staggered: bool = False) -> Dict[str, Any]:
    if staggered:
        return _step(
            baker_step=3,
            label="Test parallel trends (event-study pre-periods)",
            why=(
                "For staggered designs, inspect event-study pre-period "
                "coefficients rather than the generic check_parallel_trends() "
                "which assumes a single binary treatment with universal "
                "pre-periods. Pre-treatment ATTs should be near zero. "
                "Use CS with aggregate='event_study' or check the estimator's "
                "event-study output directly."
            ),
            code=(
                "# Inspect pre-treatment event-study coefficients:\n"
                "# (available after fitting with event-study aggregation)\n"
                "# Pre-period effects should be near zero and insignificant."
            ),
            step_name="parallel_trends",
        )
    return _step(
        baker_step=3,
        label="Test parallel trends assumption",
        why=(
            "Parallel trends is the core identifying assumption. "
            "Insignificant pre-trends do NOT prove it holds. For "
            "MultiPeriodDiD or CS results, use HonestDiD to bound "
            "the impact of violations."
        ),
        code=(
            "from diff_diff import check_parallel_trends\n"
            "pt = check_parallel_trends(data, outcome='y', time='period',\n"
            "                           treatment_group='treated')"
        ),
        step_name="parallel_trends",
    )


def _honest_did_step() -> Dict[str, Any]:
    return _step(
        baker_step=6,
        label="Run HonestDiD sensitivity analysis",
        why=(
            "Bounds the treatment effect under plausible violations of "
            "parallel trends. Essential for assessing result robustness."
        ),
        code=(
            "from diff_diff import compute_honest_did\n"
            "honest = compute_honest_did(results, method='relative_magnitude', M=1.0)\n"
            "print(honest.summary())"
        ),
        step_name="sensitivity",
    )


def _placebo_step() -> Dict[str, Any]:
    """Placebo tests for simple 2x2 DiD designs only."""
    return _step(
        baker_step=6,
        label="Run placebo tests",
        why=(
            "Falsification tests using fake timing, permutation, and "
            "leave-one-out diagnostics to probe assumption validity."
        ),
        code=(
            "from diff_diff import run_all_placebo_tests\n"
            "# Requires binary time indicator (post=0/1), not multi-period:\n"
            "placebo = run_all_placebo_tests(\n"
            "    data, outcome='y', treatment='treated', time='post',\n"
            "    unit='unit_id', pre_periods=[0], post_periods=[1],\n"
            "    n_permutations=500, seed=42)"
        ),
        priority="medium",
        step_name="sensitivity",
    )


def _robustness_compare_step(alternatives: str) -> Dict[str, Any]:
    return _step(
        baker_step=8,
        label=f"Compare with alternative estimators ({alternatives})",
        why=(
            "Agreement across estimators with different assumptions "
            "strengthens conclusions. Disagreement reveals sensitivity."
        ),
        code=(
            f"# Re-estimate with {alternatives} and compare ATT, SE, CI\n"
            f"# If results agree, confidence increases.\n"
            f"# If they disagree, investigate which assumptions differ."
        ),
        step_name="robustness",
    )


def _covariates_step() -> Dict[str, Any]:
    return _step(
        baker_step=8,
        label="Report with and without covariates",
        why=(
            "Shows whether results are sensitive to covariate conditioning. "
            "Large shifts suggest covariates are driving identification."
        ),
        code=(
            "# Re-estimate without covariates and compare:\n"
            "result_no_cov = estimator.fit(data, ..., covariates=None)\n"
            "# Compare ATT with and without covariates.\n"
            "# Use .att (basic DiD; also a read-only flat-alias on staggered\n"
            "# classes) or .overall_att (canonical name on staggered results)."
        ),
        priority="medium",
        step_name="robustness",
    )


# ---------------------------------------------------------------------------
# Per-type handlers — each returns (steps, warnings)
# ---------------------------------------------------------------------------
def _handle_did(results: Any):
    steps = [
        _step(
            baker_step=3,
            label="Test parallel trends assumption",
            why=(
                "Parallel trends is the core identifying assumption. "
                "Insignificant pre-trends do NOT prove it holds."
            ),
            code=(
                "from diff_diff import check_parallel_trends\n"
                "pt = check_parallel_trends(data, outcome='y', time='period',\n"
                "                           treatment_group='treated')"
            ),
            step_name="parallel_trends",
        ),
        _placebo_step(),  # valid: basic 2x2 DiD with binary time
        _step(
            baker_step=4,
            label="Check if data is actually staggered",
            why=(
                "If treatment timing varies across units, basic DiD produces "
                "biased estimates. Use CallawaySantAnna or another "
                "heterogeneity-robust estimator instead."
            ),
            code=(
                "# Check if there are multiple treatment cohorts:\n"
                "print(data.groupby('unit')['treatment_date'].first().nunique())\n"
                "# If > 1 cohort, switch to CallawaySantAnna"
            ),
            step_name="estimator_selection",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_multi_period(results: Any):
    steps = [
        _parallel_trends_step(),
        _honest_did_step(),
        # Note: run_all_placebo_tests() requires binary time indicator,
        # which MultiPeriodDiD does not use. Omit placebo for this type.
        _robustness_compare_step("CS, SA, or BJS"),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_cs(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Run HonestDiD sensitivity analysis",
            why=(
                "Bounds the treatment effect under plausible violations of "
                "parallel trends. Requires event study effects — refit with "
                "aggregate='event_study' or 'all' if not already done."
            ),
            code=(
                "from diff_diff import compute_honest_did\n"
                "# CS results must have event_study_effects:\n"
                "results = cs.fit(data, ..., aggregate='event_study')\n"
                "honest = compute_honest_did(results, method='relative_magnitude', M=1.0)\n"
                "print(honest.summary())"
            ),
            step_name="sensitivity",
        ),
        _step(
            baker_step=7,
            label="Examine group and event study effects",
            why=(
                "Aggregate ATT may mask heterogeneity across cohorts or "
                "dynamic effects over time. Inspect group and event study "
                "aggregations."
            ),
            code=(
                "# Re-fit with aggregate='all' to get all aggregations:\n"
                "results = cs.fit(data, ..., aggregate='all')\n"
                "print(results.group_effects)       # Per-cohort ATTs\n"
                "print(results.event_study_effects)  # Dynamic effects"
            ),
            step_name="heterogeneity",
        ),
        _robustness_compare_step("SA, BJS, or Gardner"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_sa(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Specification-based falsification",
            why=(
                "Compare results across control group definitions "
                "(never_treated vs not_yet_treated) and anticipation "
                "settings to assess robustness."
            ),
            code=(
                "# Re-estimate with different control group / anticipation:\n"
                "# sa_alt = SunAbraham(control_group='not_yet_treated')"
            ),
            priority="medium",
            # DR's sensitivity section runs HonestDiD, not specification
            # variation; tagging this as ``sensitivity`` caused
            # ``_collect_next_steps`` to suppress it after HonestDiD ran.
            # Use ``specification_comparison`` so the recommendation
            # persists alongside a completed HonestDiD sensitivity check.
            step_name="specification_comparison",
        ),
        _step(
            baker_step=7,
            label="Examine event-study and cohort effects",
            why=(
                "SunAbraham results include event_study_effects (dynamic "
                "effects by relative period) and cohort_effects (per-cohort "
                "effects). Note: SA does not have an aggregate parameter — "
                "these are computed automatically during fit()."
            ),
            code=(
                "# SA event-study effects:\n"
                "sa_es_df = results.to_dataframe(level='event_study')\n"
                "# SA cohort effects:\n"
                "sa_cohort_df = results.to_dataframe(level='cohort')"
            ),
            step_name="heterogeneity",
        ),
        _robustness_compare_step("CS, BJS, or Gardner"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_imputation(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Specification-based falsification",
            why=(
                "ImputationDiD does not have a control_group parameter. "
                "Compare results with and without covariates, vary the "
                "sample (drop cohorts), and compare with CS/SA as "
                "falsification checks."
            ),
            code=(
                "# Compare with alternative estimators as robustness:\n"
                "# Leave-one-cohort-out sensitivity analysis"
            ),
            priority="medium",
            # See note on SA handler: DR completes ``sensitivity`` when
            # HonestDiD runs, which is unrelated to this specification-
            # variation recommendation. Tag separately.
            step_name="specification_comparison",
        ),
        _robustness_compare_step("CS, SA, or Gardner"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_two_stage(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Specification-based falsification",
            why=(
                "TwoStageDiD does not have a control_group parameter. "
                "Compare results with and without covariates, vary the "
                "sample (drop cohorts), and compare with CS/SA as "
                "falsification checks."
            ),
            code=(
                "# Compare with alternative estimators as robustness:\n"
                "# Leave-one-cohort-out sensitivity analysis"
            ),
            priority="medium",
            # See note on SA handler: DR completes ``sensitivity`` when
            # HonestDiD runs, which is unrelated to this specification-
            # variation recommendation. Tag separately.
            step_name="specification_comparison",
        ),
        _robustness_compare_step("CS, BJS, or SA"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_stacked(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Vary clean control definition",
            why=(
                "StackedDiD uses clean_control parameter (not control_group). "
                "Compare results with different clean control definitions "
                "and event window widths as falsification."
            ),
            code=(
                "# Re-estimate with different clean_control settings:\n"
                "# stacked_alt = StackedDiD(clean_control='not_yet_treated')"
            ),
            priority="medium",
            # See note on SA handler: DR completes ``sensitivity`` when
            # HonestDiD runs, which does not replay ``clean_control``
            # variation. Tag separately.
            step_name="specification_comparison",
        ),
        _step(
            baker_step=7,
            label="Check sub-experiment balance",
            why=(
                "Stacked DiD constructs sub-experiments for each cohort. "
                "Verify that each sub-experiment has sufficient controls."
            ),
            code="# Check results.n_sub_experiments and inspect results.stacked_data",
            priority="medium",
            step_name="heterogeneity",
        ),
        _robustness_compare_step("CS, SA, or BJS"),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_synthetic(results: Any):
    steps = [
        _step(
            baker_step=6,
            label="Check pre-treatment fit and weight concentration",
            why=(
                "Synthetic DiD relies on pre-treatment fit to construct "
                "weights. Poor fit or highly concentrated unit weights "
                "suggest the synthetic control may not approximate the "
                "counterfactual well."
            ),
            code=(
                "print(f'Pre-treatment fit (RMSE): {results.pre_treatment_fit:.4f}')\n"
                "concentration = results.get_weight_concentration()\n"
                "print(f\"Effective N: {concentration['effective_n']:.1f}\")\n"
                "print(f\"Top-5 weight share: {concentration['top_k_share']:.2%}\")"
            ),
            step_name="sensitivity",
        ),
        _step(
            baker_step=6,
            label="In-time placebo",
            why=(
                "Re-estimate on shifted fake treatment dates in the "
                "pre-period. A credible design yields near-zero placebo "
                "ATTs — departures signal that something is being picked "
                "up pre-treatment, weakening the causal interpretation."
            ),
            code=("placebo_df = results.in_time_placebo()\n" "print(placebo_df)"),
            priority="medium",
            step_name="sensitivity",
        ),
        _step(
            baker_step=6,
            label="Leave-one-out influence (jackknife)",
            why=(
                "If the estimate is driven by a single unit, robustness "
                "is weak. Fit with variance_method='jackknife' and inspect "
                "which units move the ATT the most."
            ),
            code=(
                "# Requires variance_method='jackknife' AND enough support for LOO\n"
                "# (n_treated >= 2 and >= 2 effective-weight controls).\n"
                "if getattr(results, '_loo_unit_ids', None) is not None:\n"
                "    loo_df = results.get_loo_effects_df()\n"
                "    print(loo_df.head(10))\n"
                "else:\n"
                "    print('LOO not available - re-fit with '\n"
                "          'variance_method=\"jackknife\" and ensure >=2 treated units '\n"
                "          'with positive effective support.')"
            ),
            priority="medium",
            # DR's SyntheticDiD native battery covers pre-treatment fit,
            # weight concentration, in-time placebo, and zeta-omega
            # sensitivity, but NOT the jackknife LOO workflow (which
            # requires a separate ``variance_method='jackknife'`` fit
            # via ``get_loo_effects_df``). Tagging this recommendation
            # as ``sensitivity`` caused ``_collect_next_steps`` to
            # suppress it as soon as the native block ran, even though
            # the jackknife was never executed. Round-24 P2 CI review
            # on PR #318; same class as round-20 Hausman mistag.
            step_name="loo_jackknife",
        ),
        _step(
            baker_step=6,
            label="Regularization sensitivity (zeta_omega)",
            why=(
                "The unit-weight regularization is auto-selected from "
                "data. Show whether the ATT moves materially across a "
                "grid of values to gauge robustness to this choice."
            ),
            code=("sens_df = results.sensitivity_to_zeta_omega()\n" "print(sens_df)"),
            priority="low",
            step_name="sensitivity",
        ),
        _step(
            baker_step=8,
            label="Compare with staggered estimators (CS, SA)",
            why=(
                "SyntheticDiD is for few treated units; compare with "
                "staggered estimators if applicable. Use TROP only if "
                "factor confounding is suspected (different use case)."
            ),
            code=(
                "from diff_diff import CallawaySantAnna\n"
                "cs = CallawaySantAnna()\n"
                "cs_result = cs.fit(data, ...)\n"
                "print(f'SDiD ATT: {results.att:.4f}, CS ATT: {cs_result.overall_att:.4f}')"
            ),
            step_name="robustness",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_trop(results: Any):
    steps = [
        _step(
            baker_step=6,
            label="Verify factor structure assumptions",
            why=(
                "TROP assumes an approximate factor model for untreated "
                "potential outcomes. If the factor structure is misspecified, "
                "estimates may be biased."
            ),
            code=(
                "# Check LOOCV-selected number of factors:\n"
                "# Compare with SyntheticDiD as a robustness check"
            ),
            step_name="sensitivity",
        ),
        _step(
            baker_step=6,
            label="In-time or in-space placebo",
            why=(
                "Test robustness by re-estimating on a placebo treatment "
                "period or dropping treated units one at a time. These "
                "are the natural falsification checks for factor-model "
                "panel estimators."
            ),
            code=(
                "# In-time placebo: re-estimate with a fake treatment date\n"
                "# Leave-one-out: drop each treated unit and re-estimate"
            ),
            priority="medium",
            # TROP's estimator-native diagnostics surface factor-model fit
            # metrics, not in-time or in-space placebos; DR does not run
            # placebos on TROP. Tag separately from ``sensitivity`` so the
            # recommendation persists after DR marks the TROP native
            # battery complete.
            step_name="placebo",
        ),
        _robustness_compare_step("SyntheticDiD or CS"),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_synthetic_control(results: Any):
    steps = [
        _step(
            baker_step=6,
            label="In-space placebo permutation inference",
            why=(
                "Classic SCM has no analytical standard error. Significance "
                "comes from the in-space placebo test (Abadie-Diamond-Hainmueller "
                "2010, Section 2.4): reassign treatment to each donor, refit, and "
                "rank the treated unit's post/pre RMSPE ratio "
                "(p = rank/(n_placebos+1), excluding non-converged placebos)."
            ),
            code=(
                "placebo_df = results.in_space_placebo()\n"
                "print(f'placebo p-value: {results.placebo_p_value:.3f} "
                "(n={results.n_placebos})')\n"
                "print(placebo_df)  # per-unit RMSPE-ratio table used for the rank"
            ),
            priority="high",
            # SCM's significance test IS the placebo; tag it "placebo" (not
            # "sensitivity") so it survives once the native diagnostics block runs,
            # mirroring _handle_trop.
            step_name="placebo",
        ),
        _step(
            baker_step=3,
            label="Demonstrate pre-treatment fit (SCM identification)",
            why=(
                "SCM's identifying assumption is design-enforced fit, not a "
                "parallel-trends test: it is only credible when the synthetic "
                "control reproduces the treated unit's pre-period path. Report "
                "the pre-RMSPE and predictor-balance table; a poor fit means do "
                "not use SCM (ADH 2010 p. 495)."
            ),
            code=(
                "print(f'pre-treatment RMSPE: {results.pre_rmspe:.4f}')\n"
                "print(results.predictor_balance)\n"
                "print(results.get_weights_df())  # donor weight concentration"
            ),
            priority="high",
            # Design-enforced fit IS SCM's parallel-trends analogue (mirrors the
            # DiagnosticReport ``scm_fit`` PT routing); tagging it "parallel_trends"
            # keeps it from being auto-suppressed as the completed estimation step.
            step_name="parallel_trends",
        ),
        _step(
            baker_step=4,
            label="Curate the donor pool",
            why=(
                "Donors exposed to the same/similar intervention or to large "
                "confounding shocks contaminate the comparison (ADH 2010 "
                "pp. 498-499). Restrict the donor pool to clean, comparable units."
            ),
            code=(
                "# Exclude contaminated donors explicitly:\n"
                "# synthetic_control(..., donor_pool=[clean, comparable, units])"
            ),
            priority="medium",
            step_name="estimator_selection",
        ),
        _step(
            baker_step=6,
            label="Leave-one-out donor robustness (ADH 2015)",
            why=(
                "Re-fit dropping each reportably-weighted donor (weight above the 1e-6 "
                "floor) in turn to confirm the "
                "estimate is not driven by a single donor (Abadie-Diamond-Hainmueller "
                "2015, Section 4); a large delta_att when one donor is removed flags "
                "single-donor dependence."
            ),
            code=(
                "loo_df = results.leave_one_out()\n"
                "print(loo_df)  # baseline + per-dropped-donor ATT and delta_att"
            ),
            priority="medium",
            # Not a standard STEPS tag, so a caller's completed_steps (validated
            # against STEPS) can never auto-suppress this opt-in recommendation.
            step_name="loo_jackknife",
        ),
        _step(
            baker_step=6,
            label="In-time (backdating) placebo (ADH 2015)",
            why=(
                "Reassign the intervention to an earlier pre-period and confirm no "
                "spurious gap appears before the true treatment date (Abadie-Diamond-"
                "Hainmueller 2015, Section 4, Figure 4)."
            ),
            code=(
                "itp_df = results.in_time_placebo()\n"
                "print(itp_df)  # per-backdated-date placebo ATT (should be ~0)"
            ),
            priority="medium",
            # Non-standard tag (not in STEPS) -> never auto-suppressed; deliberately
            # NOT "sensitivity" (a caller could mark that done and drop this step).
            step_name="in_time_placebo",
        ),
        _robustness_compare_step("SyntheticDiD or CS"),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_efficient(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Compare control group definitions",
            why=(
                "EfficientDiD supports never_treated and last_cohort "
                "control groups (not not_yet_treated). Compare results "
                "across both to assess robustness."
            ),
            code=(
                "# Re-estimate with alternative control group:\n"
                "# edid_alt = EfficientDiD(control_group='last_cohort')"
            ),
            priority="medium",
            # See note on SA handler: DR completes ``sensitivity`` when
            # HonestDiD runs, which does not re-estimate with an
            # alternative control_group. Tag separately so this
            # recommendation persists alongside a completed HonestDiD
            # block.
            step_name="specification_comparison",
        ),
        _step(
            baker_step=7,
            label="Run Hausman pretest (PT-All vs PT-Post)",
            why=(
                "EfficientDiD supports both PT-All and PT-Post assumptions. "
                "The Hausman pretest compares them — report which was selected."
            ),
            code=(
                "# Hausman pretest is a classmethod on the estimator:\n"
                "from diff_diff import EfficientDiD\n"
                "pretest = EfficientDiD.hausman_pretest(\n"
                "    data, outcome='y', unit='id', time='t', first_treat='g')"
            ),
            # The Hausman pretest is a parallel-trends diagnostic per
            # REGISTRY.md §EfficientDiD: it tests whether the stronger
            # PT-All regime is tenable relative to PT-Post. ``DiagnosticReport``
            # treats a ran Hausman block as ``parallel_trends`` completion
            # (``_check_pt_hausman``), so tagging this practitioner step as
            # ``parallel_trends`` keeps ``_collect_next_steps()`` from
            # recommending a check the report already executed. Round-20 P2
            # CI review on PR #318 flagged the earlier ``heterogeneity`` tag
            # as a mismatched-step-name bug.
            step_name="parallel_trends",
        ),
        _robustness_compare_step("CS, SA, or BJS"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_continuous(results: Any):
    steps = [
        _step(
            baker_step=3,
            label="Assess parallel trends for continuous treatment",
            why=(
                "ContinuousDiD has dose-specific parallel trends assumptions "
                "(PT/SPT) that differ from the binary treatment case. No "
                "built-in formal test exists; inspect dose-specific "
                "pre-treatment outcome trends across dose groups manually."
            ),
            code=(
                "# No built-in formal PT test for continuous treatment.\n"
                "# Inspect pre-treatment outcome trends by dose group."
            ),
            step_name="parallel_trends",
        ),
        _step(
            baker_step=4,
            label="Switch to HeterogeneousAdoptionDiD if no untreated units",
            why=(
                "ContinuousDiD's identification assumes a never-treated "
                "comparison group exists (units with dose = 0, encoded as "
                "either `first_treat = 0` or `first_treat = inf`; ContinuousDiD "
                "normalizes `inf -> 0` internally). When every unit is treated "
                "at some positive dose level - a universal rollout where "
                "treatment varies in intensity, not status - use "
                "HeterogeneousAdoptionDiD instead. HAD identifies a Weighted "
                "Average Slope (WAS) at the dose support boundary by leveraging "
                "dose variation across units. HAD's contract is panel-shape "
                "dependent: `aggregate='overall'` (the default) is two-period "
                "only and hard-rejects multi-period panels at fit time; "
                "multi-period panels MUST set `aggregate='event_study'`. "
                "Additionally, on staggered (multi-cohort) panels the event-"
                "study path auto-filters to the LAST treatment cohort + never-"
                "treated units (paper Appendix B.2) and the estimand becomes "
                "last-cohort-only WAS rather than a full multi-cohort average; "
                "use `ChaisemartinDHaultfoeuille` if full multi-cohort staggered "
                "support under continuous treatment is required."
            ),
            code=(
                "# HAD requires a REALIZED per-period dose column (zero\n"
                "# pre-switch, positive from switch onward) - DIFFERENT\n"
                "# from ContinuousDiD's time-invariant per-unit dose.\n"
                "# Re-prepare the panel into a HAD-shaped panel before\n"
                "# the fit; do NOT reuse the ContinuousDiD-shaped panel.\n"
                "# HAD's first_treat encoding is also stricter than\n"
                "# ContinuousDiD's: HAD requires never-treated units to\n"
                "# have first_treat = 0 (not inf or NaN); recode before\n"
                "# fit. HAD raises ValueError on any first_treat value\n"
                "# outside {0, t_post} for the two-period path.\n"
                "import numpy as np\n"
                "from diff_diff import HeterogeneousAdoptionDiD\n"
                "data_had_2p = data_had_2p.assign(\n"
                "    first_treat=data_had_2p['first_treat'].replace(\n"
                "        {np.inf: 0}))\n"
                "data_had_mp = data_had_mp.assign(\n"
                "    first_treat=data_had_mp['first_treat'].replace(\n"
                "        {np.inf: 0}))\n"
                "had = HeterogeneousAdoptionDiD()\n"
                "# Two-period panel (single cohort or 2 periods):\n"
                "had_results = had.fit(\n"
                "    data_had_2p, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d', first_treat_col='first_treat')\n"
                "\n"
                "# Multi-period panel: must set aggregate='event_study'\n"
                "# (on staggered panels this is auto-last-cohort-only WAS)\n"
                "had_es = had.fit(\n"
                "    data_had_mp, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d', first_treat_col='first_treat',\n"
                "    aggregate='event_study')"
            ),
            step_name="estimator_selection",
        ),
        _step(
            baker_step=7,
            label="Plot dose-response curve",
            why=(
                "Continuous DiD estimates treatment effects at each dose "
                "level. The dose-response curve reveals the functional form "
                "of the treatment-dose relationship."
            ),
            code=("from diff_diff import plot_dose_response\n" "plot_dose_response(results)"),
            step_name="heterogeneity",
        ),
        _step(
            baker_step=6,
            label="Check dose distribution",
            why=(
                "Sparse regions of the dose distribution produce imprecise "
                "estimates. Verify sufficient support across dose values."
            ),
            code="# Inspect the distribution of treatment doses in your data",
            priority="medium",
            step_name="sensitivity",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_triple(results: Any):
    steps = [
        _step(
            baker_step=3,
            label="Assess DDD identifying assumption",
            why=(
                "DDD identification is weaker than requiring separate "
                "parallel trends for two DiDs — it allows group-specific "
                "and partition-specific PT violations as long as they "
                "cancel in the triple difference. No built-in formal "
                "test exists; inspect pre-treatment outcome patterns "
                "across the treatment/eligibility/time cells."
            ),
            code=(
                "# No built-in formal DDD assumption test.\n"
                "# Inspect pre-treatment means across treatment x eligibility\n"
                "# cells to assess whether the DDD structure is plausible."
            ),
            step_name="parallel_trends",
        ),
        _step(
            baker_step=7,
            label="Test placebo group",
            why=(
                "Re-estimate using a placebo eligibility group to check "
                "whether the DDD result could be an artifact of the "
                "group structure rather than the treatment."
            ),
            code="# Re-estimate with a placebo eligibility group",
            step_name="heterogeneity",
        ),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_bacon(results: Any):
    steps = [
        _step(
            baker_step=4,
            label="Switch to heterogeneity-robust estimator",
            why=(
                "Bacon decomposition is diagnostic, not an estimator. "
                "If substantial weight falls on 'later vs earlier' "
                "comparisons, TWFE is biased. Use CS, SA, BJS, or another "
                "heterogeneity-robust estimator for causal estimates."
            ),
            code=(
                "from diff_diff import CallawaySantAnna\n"
                "cs = CallawaySantAnna(control_group='never_treated',\n"
                "                      estimation_method='dr')\n"
                "results = cs.fit(data, ...)"
            ),
            step_name="estimator_selection",
        ),
    ]
    warnings = []
    # Check for forbidden comparisons (later vs earlier treated)
    weight = getattr(results, "total_weight_later_vs_earlier", 0)
    if isinstance(weight, (int, float)) and weight > 0.01:
        warnings.append(
            f"Forbidden comparisons (later vs earlier treated) carry "
            f"{weight:.0%} of TWFE weight — TWFE estimate is contaminated. "
            f"Switch to a heterogeneity-robust estimator."
        )
    return steps, warnings


def _handle_had(results: Any):
    """HeterogeneousAdoptionDiD single-period guidance.

    Five Baker et al. steps (3, 4, 6, 7, 8). HAD's identifying
    comparison comes from dose VARIATION across units rather than
    from a never-treated holdout: treatment varies in intensity,
    not in status. Never-treated units may still be present in the
    panel and are retained as controls without breaking HAD's
    identification (REGISTRY HeterogeneousAdoptionDiD edge cases);
    the WAS-vs-ATT(d) distinction is the actual differentiator
    between HAD and ContinuousDiD, not untreated-unit presence
    alone.
    """
    steps = [
        _step(
            baker_step=3,
            label="Run the HAD pretest battery",
            why=(
                "On a two-period unweighted panel did_had_pretest_workflow "
                "runs paper Section 4.2 step 1 (QUG support-infimum test - "
                "decides Design 1' vs Design 1) and step 3 (Stute / "
                "Yatchew-HR Assumption 8 linearity tests). Step 2 "
                "(Assumption 7 pre-trends) is NOT covered on the overall "
                "path - a single pre-period cannot support the joint "
                "Stute variant - and the returned verdict explicitly "
                "flags that gap. To close step 2, refit on a multi-period "
                "panel with aggregate='event_study' AND verify the panel "
                "has at least one earlier placebo pre-period beyond F-1; "
                "if only the base pre-period F-1 is available, the "
                "workflow still sets pretrends_joint=None, all_pass=False, "
                "and a 'joint pre-trends skipped (no earlier pre-period)' "
                "verdict suffix - in that case step 2 stays uncovered "
                "even on the event-study path. On supported survey-weighted "
                "fits (pweight + PSU/FPC under survey_design= / survey= / "
                "weights=) the workflow skips QUG with a UserWarning "
                "(permanent Phase 4.5 C0 deferral - extreme order statistics "
                "are not smooth functionals of the empirical CDF) and returns "
                "a linearity-conditional verdict only - so step 1 coverage "
                "is unweighted-only and the reported verdict on supported "
                "weighted fits is conditional on QUG holding by assumption. "
                "Stratified (SurveyDesign(strata=...)) and replicate-weight "
                "(BRR/Fay/JK1/JKn/SDR) designs raise NotImplementedError on "
                "the linearity kernels and have no pretest workflow path "
                "yet - deferred to a follow-up. "
                "Assumptions 3 / 5 / 6 (uniform continuity at the "
                "boundary, Design 1 sign / WAS_d_lower identification) "
                "are NOT testable via pre-trends - the workflow vets only "
                "what can be vetted."
            ),
            code=(
                "from diff_diff import did_had_pretest_workflow\n"
                "report = did_had_pretest_workflow(\n"
                "    data, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d',\n"
                "    first_treat_col='first_treat')\n"
                "print(report.summary())\n"
                "# verdict explicitly flags the Assumption 7 gap on the\n"
                "# overall path; aggregate='event_study' on a multi-period\n"
                "# panel adds joint Stute pre-trends + joint homogeneity-linearity.\n"
                "# Passing survey_design= / weights= skips QUG (Phase 4.5 C0)\n"
                "# and returns a linearity-conditional verdict only."
            ),
            step_name="parallel_trends",
        ),
        _step(
            baker_step=4,
            label="Confirm WAS is the target estimand (vs ATT(d) for ContinuousDiD)",
            why=(
                "HAD targets WAS (Weighted Average Slope) at the dose "
                "support boundary. If you specifically want per-dose "
                "ATT(d) / ACRT(d) dose-response curves AND your panel "
                "has never-treated controls (units with dose == 0 "
                "throughout, encoded as either first_treat == 0 or "
                "first_treat == inf; ContinuousDiD normalizes inf -> 0 "
                "internally), ContinuousDiD is the alternative — "
                "different estimand; ContinuousDiD's default "
                "identification uses never-treated controls (or "
                "control_group='lowest_dose' for Remark 3.1 when "
                "P(D=0)=0). HAD itself remains "
                "valid even with a small share of never-treated units "
                "(paper compatibility; see REGISTRY § "
                "HeterogeneousAdoptionDiD edge cases — Garrett et al. "
                "2020 retained 12 untreated counties out of 2,954). The "
                "choice is about estimand, not about whether untreated "
                "units exist. NOTE: HAD and ContinuousDiD require "
                "DIFFERENT dose-column encodings — HAD uses the "
                "realized per-period dose (zero pre-switch, positive "
                "from switch onward) while ContinuousDiD requires a "
                "time-invariant per-unit dose column (the front-door "
                "check rejects within-unit dose variation). Re-prepare "
                "the panel into a unit-level dose summary before the "
                "ContinuousDiD fit; do NOT reuse the HAD-shaped panel "
                "directly."
            ),
            code=(
                "# HAD reports WAS at the dose support boundary.\n"
                "# If you instead want per-dose ATT(d)/ACRT(d) dose-response\n"
                "# curves AND the panel has never-treated controls:\n"
                "from diff_diff import ContinuousDiD\n"
                "# ContinuousDiD requires a TIME-INVARIANT per-unit dose; HAD\n"
                "# uses realized per-period dose. Re-prepare the panel\n"
                "# (e.g. collapse each unit's positive dose to one value) and\n"
                "# pass it as `data_cdid` with the time-invariant `dose_col`.\n"
                "cdid = ContinuousDiD()\n"
                "cdid_results = cdid.fit(\n"
                "    data_cdid, outcome='y', unit='unit', time='t',\n"
                "    first_treat='first_treat', dose='d',\n"
                "    aggregate='dose')"
            ),
            step_name="estimator_selection",
        ),
        _step(
            baker_step=6,
            label="Inspect bandwidth diagnostics (continuous designs)",
            why=(
                "Continuous-dose designs (continuous_at_zero / "
                "continuous_near_d_lower) use an MSE-DPI bandwidth selector "
                "for the bias-corrected local-linear estimator. Bandwidth "
                "choice affects WAS - verify the selector landed on a "
                "viable bandwidth (not boundary-clipped or near-degenerate). "
                "results.bandwidth_diagnostics is None on the mass_point "
                "design (parametric, no bandwidth)."
            ),
            code=(
                "# Inspect the auto-selected bandwidths:\n"
                "results.bandwidth_diagnostics  # None on mass_point"
            ),
            priority="medium",
            step_name="sensitivity",
        ),
        _step(
            baker_step=7,
            label="Re-fit with aggregate='event_study' for per-horizon WAS",
            why=(
                "On multi-period panels, the event-study aggregate returns "
                "per-event-time WAS estimates instead of a single scalar. "
                "Reveals whether dose response grows, decays, or stabilizes "
                "across post-treatment horizons. Pre-period placebos serve "
                "as a parallel-trends sanity check. NOTE: this handler is "
                "the single-period HAD handler, so the upstream fit was "
                "two-period-only (`aggregate='overall'` hard-rejects more "
                "than two periods at `had.py:980-987`). Use a distinct "
                "multi-period panel `data_mp` for this step - the same "
                "panel that the upstream fit ran on will not satisfy "
                "the event-study path's period-count requirement."
            ),
            code=(
                "from diff_diff import HeterogeneousAdoptionDiD\n"
                "# Requires a distinct multi-period panel - the upstream\n"
                "# two-period panel was already consumed by `aggregate='overall'`.\n"
                "est = HeterogeneousAdoptionDiD()\n"
                "es = est.fit(\n"
                "    data_mp, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d',\n"
                "    first_treat_col='first_treat',\n"
                "    aggregate='event_study')"
            ),
            priority="medium",
            step_name="heterogeneity",
        ),
        _step(
            baker_step=8,
            label="Verify design auto-detection with explicit design=",
            why=(
                "design='auto' picks one of {continuous_at_zero, "
                "continuous_near_d_lower, mass_point} from the dose "
                "support. Re-fit with an explicit design= to verify the "
                "auto-detection matched your panel structure - WAS vs "
                "WAS_d_lower target parameters, and the bias-corrected "
                "local-linear vs 2SLS estimation paths, differ in "
                "interpretation."
            ),
            code=(
                "# Refit with each candidate design and compare:\n"
                "from diff_diff import HeterogeneousAdoptionDiD\n"
                "for d in ['continuous_at_zero', 'continuous_near_d_lower',\n"
                "          'mass_point']:\n"
                "    try:\n"
                "        alt = HeterogeneousAdoptionDiD(design=d).fit(...)\n"
                "        print(d, alt.att, alt.target_parameter)\n"
                "    except Exception as e:\n"
                "        print(d, 'not applicable:', e)"
            ),
            priority="medium",
            step_name="robustness",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_had_event_study(results: Any):
    """HeterogeneousAdoptionDiD event-study guidance.

    Five Baker et al. steps (3, 4, 6, 7, 8). Same framing convention
    as _handle_had: identifying comparison comes from dose variation
    across units, treatment varies in intensity not status, and
    never-treated units may still be present in the panel without
    breaking HAD's identification - the WAS-vs-ATT(d) distinction
    is the actual differentiator between HAD and ContinuousDiD.
    """
    steps = [
        _step(
            baker_step=3,
            label="Run the HAD pretest battery (event-study mode)",
            why=(
                "On multi-period unweighted panels, did_had_pretest_workflow "
                "with aggregate='event_study' runs QUG plus joint Stute "
                "pre-trends plus joint homogeneity-linearity Stute. The "
                "joint Stute pre-trends variant closes the paper Section "
                "4.2 step-2 gap ONLY IF the panel carries at least one "
                "earlier placebo pre-period beyond the base F-1. With "
                "only the base F-1 pre-period present (e.g. a minimal "
                "valid 3-period event-study fit, or a 4-period fit under "
                "trends_lin=True where the consumed F-2 placebo gets "
                "dropped), pretrends_joint=None, all_pass=False, and the "
                "verdict carries 'joint pre-trends skipped (no earlier "
                "pre-period)' - step 2 stays uncovered. On supported "
                "survey-weighted fits (pweight + PSU/FPC under "
                "survey_design= / survey= / weights=) the workflow skips "
                "QUG with a UserWarning (permanent Phase 4.5 C0 deferral) "
                "and returns a linearity-conditional verdict only - so "
                "step 1 coverage is unweighted-only on the event-study "
                "path too, and the weighted verdict is conditional on QUG "
                "holding by assumption. Stratified (SurveyDesign("
                "strata=...)) and replicate-weight (BRR/Fay/JK1/JKn/SDR) "
                "designs raise NotImplementedError on the linearity "
                "kernels and have no pretest workflow path on the "
                "event-study path yet - deferred to a follow-up. "
                "The joint Stute pre-trends and joint "
                "homogeneity-linearity tests themselves remain available "
                "under supported survey weighting via PSU-level Mammen "
                "multiplier bootstrap."
            ),
            code=(
                "from diff_diff import did_had_pretest_workflow\n"
                "report = did_had_pretest_workflow(\n"
                "    data, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d',\n"
                "    first_treat_col='first_treat',\n"
                "    aggregate='event_study')\n"
                "print(report.summary())"
            ),
            step_name="parallel_trends",
        ),
        _step(
            baker_step=4,
            label="Confirm WAS is the target estimand (vs ATT(d) for ContinuousDiD)",
            why=(
                "HAD targets per-event-time WAS at the dose support "
                "boundary. If you instead want per-dose ATT(d) / ACRT(d) "
                "dose-response curves AND your panel has never-treated "
                "controls (units with dose == 0 throughout, encoded as "
                "either first_treat == 0 or first_treat == inf; "
                "ContinuousDiD normalizes inf -> 0 internally), "
                "ContinuousDiD is the alternative — different estimand, "
                "uses never-treated by default (or "
                "control_group='lowest_dose' for Remark 3.1 when "
                "P(D=0)=0). Two ContinuousDiD aggregation "
                "surfaces are relevant and distinct: `aggregate='dose'` "
                "(the default) produces the per-dose ATT(d) / ACRT(d) "
                "curves on `results.dose_response_att` / "
                "`results.dose_response_acrt`; `aggregate='eventstudy'` "
                "produces a binarized event-study of `att_glob` on "
                "`results.event_study_effects` (NOT per-dose by horizon). "
                "Pick the aggregation that matches the estimand you "
                "actually want. HAD itself remains valid even with a "
                "small share of never-treated units (paper compatibility); "
                "on staggered panels HAD's last-cohort filter explicitly "
                "RETAINS never-treated units as the untreated-group "
                "comparison (paper Appendix B.2). The choice between HAD "
                "and ContinuousDiD is about estimand. NOTE: HAD and "
                "ContinuousDiD require DIFFERENT dose-column encodings — "
                "HAD uses the realized per-period dose while ContinuousDiD "
                "requires a TIME-INVARIANT per-unit dose; re-prepare the "
                "panel into a unit-level dose summary before the "
                "ContinuousDiD fit, do NOT reuse the HAD-shaped panel."
            ),
            code=(
                "# HAD reports per-event-time WAS at the dose boundary.\n"
                "# For per-dose ATT(d)/ACRT(d) curves, use ContinuousDiD with\n"
                "# the DEFAULT aggregate='dose' (NOT 'eventstudy' - that gives\n"
                "# binarized event-study of att_glob, not per-dose curves).\n"
                "from diff_diff import ContinuousDiD\n"
                "# ContinuousDiD requires a TIME-INVARIANT per-unit dose.\n"
                "# Re-prepare the panel (e.g. collapse each unit's positive\n"
                "# dose to one value) and pass it as `data_cdid`.\n"
                "cdid = ContinuousDiD()\n"
                "cdid_res = cdid.fit(\n"
                "    data_cdid, outcome='y', unit='unit', time='t',\n"
                "    first_treat='first_treat', dose='d',\n"
                "    aggregate='dose')\n"
                "# Per-dose curves live here:\n"
                "#   cdid_res.dose_response_att / .dose_response_acrt"
            ),
            step_name="estimator_selection",
        ),
        _step(
            baker_step=6,
            label="Use simultaneous (sup-t) confidence bands when reading multiple horizons",
            why=(
                "Pointwise CIs over-reject when you read multiple horizons "
                "as a joint pattern. On weighted fits (survey_design= or "
                "weights=), fit(cband=True) constructs simultaneous (sup-t) "
                "bands across horizons via multiplier bootstrap. "
                "results.cband_low / results.cband_high give the band "
                "endpoints; results.cband_crit_value reports the sup-t "
                "critical value used."
            ),
            code=(
                "from diff_diff import HeterogeneousAdoptionDiD, SurveyDesign\n"
                "# Construct your survey design (adapt to your data):\n"
                "sd = SurveyDesign(weights='weight_col')\n"
                "# vcov_type='hc1' is REQUIRED on the mass-point design under\n"
                "# survey_design= (the default classical sandwich raises\n"
                "# NotImplementedError on the survey path because the\n"
                "# Binder-TSL composition consumes the HC1-scale IF -\n"
                "# see had.py:3495-3507). On the continuous designs the\n"
                "# vcov_type kwarg is unused (CCT-2014 robust SE is the\n"
                "# only formula), so passing vcov_type='hc1' is a no-op\n"
                "# there and a safe default for the survey-aware example.\n"
                "est = HeterogeneousAdoptionDiD(\n"
                "    n_bootstrap=999, seed=42, vcov_type='hc1')\n"
                "es = est.fit(\n"
                "    data, outcome_col='y', unit_col='unit',\n"
                "    time_col='t', dose_col='d',\n"
                "    first_treat_col='first_treat',\n"
                "    aggregate='event_study',\n"
                "    survey_design=sd, cband=True)\n"
                "es.cband_low, es.cband_high  # simultaneous band endpoints"
            ),
            priority="medium",
            step_name="sensitivity",
        ),
        _step(
            baker_step=7,
            label="Inspect per-horizon WAS arrays + pre-period placebos",
            why=(
                "Per-horizon WAS reveals adoption-effect dynamics. "
                "Pre-period placebo horizons (event_times <= -2) should be "
                "near zero - large pre-period coefficients flag a "
                "parallel-trends or anticipation problem. The anchor "
                "horizon e = -1 is excluded by construction."
            ),
            code=(
                "import numpy as np\n"
                "es.event_times, es.att, es.se  # per-horizon arrays\n"
                "# Pre-period placebos (should be near zero):\n"
                "pre_mask = es.event_times <= -2\n"
                "es.att[pre_mask], es.se[pre_mask]"
            ),
            step_name="heterogeneity",
        ),
        _step(
            baker_step=8,
            label="Report the last-cohort-only WAS framing on staggered panels",
            why=(
                "On staggered panels (multiple treatment cohorts), fit() "
                "auto-filters to the last treatment cohort plus "
                "never-treated units and emits a UserWarning naming "
                "kept/dropped counts (paper Appendix B.2). The resulting "
                "estimand is a last-cohort-only WAS, NOT a multi-cohort "
                "average - report it as such, and consider "
                "ChaisemartinDHaultfoeuille for full staggered support."
            ),
            code=(
                "# Inspect the kept/dropped cohort counts in the\n"
                "# UserWarning emitted at fit time.\n"
                "# For full multi-cohort support, see:\n"
                "# from diff_diff import ChaisemartinDHaultfoeuille"
            ),
            priority="medium",
            step_name="robustness",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_generic(results: Any):
    """Fallback for unknown result types."""
    steps = [
        _parallel_trends_step(),
        _step(
            baker_step=6,
            label="Run sensitivity analysis",
            why=(
                "Without sensitivity analysis, you cannot assess how "
                "robust results are to assumption violations."
            ),
            code=(
                "# Use compute_honest_did() if result type supports it,\n"
                "# or run_all_placebo_tests() for falsification."
            ),
            step_name="sensitivity",
        ),
        _step(
            baker_step=8,
            label="Compare with alternative estimators",
            why=(
                "Different estimators make different assumptions. "
                "Agreement strengthens conclusions."
            ),
            code="# Re-estimate with a different estimator and compare",
            step_name="robustness",
        ),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


# ---------------------------------------------------------------------------
# Handler registry — maps result type *names* (not classes) to avoid
# import-time circular dependencies
# ---------------------------------------------------------------------------
_HANDLERS = {
    "DiDResults": _handle_did,
    "MultiPeriodDiDResults": _handle_multi_period,
    "CallawaySantAnnaResults": _handle_cs,
    "SunAbrahamResults": _handle_sa,
    "ImputationDiDResults": _handle_imputation,
    "TwoStageDiDResults": _handle_two_stage,
    "StackedDiDResults": _handle_stacked,
    "SyntheticDiDResults": _handle_synthetic,
    "TROPResults": _handle_trop,
    "SyntheticControlResults": _handle_synthetic_control,
    "EfficientDiDResults": _handle_efficient,
    "ContinuousDiDResults": _handle_continuous,
    "TripleDifferenceResults": _handle_triple,
    "BaconDecompositionResults": _handle_bacon,
    "HeterogeneousAdoptionDiDResults": _handle_had,
    "HeterogeneousAdoptionDiDEventStudyResults": _handle_had_event_study,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_nan_att(results: Any) -> List[str]:
    """Return warnings if ATT is NaN.

    Scalar path executes byte-identically to the pre-Phase-5 helper for
    backcompat with the existing 12 untouched handlers. The ndarray
    branch is reached only when ``float(att)`` raises TypeError on a
    numpy array (HAD's event-study ``att`` field) and fires only when
    every horizon is NaN - partial-NaN arrays are legitimate event-study
    output (single-cluster collapse, degenerate horizon-specific design)
    and would over-fire if flagged. Falls through to ``_handle_generic``
    too: any future estimator returning ndarray ``att`` without a
    dedicated handler gets the same all-NaN warning shape.
    """
    # Check .att (DiDResults), .overall_att (staggered), .avg_att (MultiPeriod)
    att = getattr(results, "att", None)
    if att is None:
        att = getattr(results, "overall_att", None)
    if att is None:
        att = getattr(results, "avg_att", None)
    if att is None:
        return []
    try:
        scalar = float(att)
    except (TypeError, ValueError):
        # Ndarray path (HAD event-study, future ndarray-att estimators).
        # Use np.all (not np.any): partial-NaN arrays are legitimate.
        try:
            import numpy as np

            arr = np.asarray(att, dtype=float)
        except (TypeError, ValueError):
            return []
        if arr.size and bool(np.all(np.isnan(arr))):
            return [
                "All per-horizon estimates are NaN — check data "
                "preparation and model specification before proceeding "
                "with diagnostics."
            ]
        return []
    if math.isnan(scalar):
        return [
            "Estimation produced NaN ATT — check data preparation and "
            "model specification before proceeding with diagnostics."
        ]
    return []


def _filter_steps(steps: List[Dict[str, Any]], completed: Set[str]) -> List[Dict[str, Any]]:
    """Remove steps whose _step_name is in the completed set."""
    filtered = []
    for s in steps:
        step_name = s.get("_step_name", "")
        if step_name not in completed:
            # Remove internal field from output
            out = {k: v for k, v in s.items() if k != "_step_name"}
            filtered.append(out)
    return filtered


def _print_output(output: Dict[str, Any]) -> None:
    """Print human-readable guidance to stdout."""
    print(f"\n{'='*60}")
    print(f"Practitioner Guidance — {output['estimator']}")
    print("Baker et al. (2025) 8-Step Workflow")
    print(f"{'='*60}")

    if output["warnings"]:
        print("\nWARNINGS:")
        for w in output["warnings"]:
            print(f"  ! {w}")

    if output["next_steps"]:
        print(f"\nRecommended next steps ({len(output['next_steps'])} remaining):")
        for step in output["next_steps"]:
            priority = step.get("priority", "high")
            marker = "*" if priority == "high" else "-"
            print(
                f"\n  {marker} [{priority.upper()}] Step {step['baker_step']}: " f"{step['label']}"
            )
            print(f"    Why: {step['why']}")
            if step.get("code"):
                for line in step["code"].split("\n"):
                    print(f"    >>> {line}")
    else:
        print("\nAll Baker et al. steps completed!")

    print(f"\n{'='*60}\n")
