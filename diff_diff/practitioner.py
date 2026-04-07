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
    "EfficientDiDResults": "EfficientDiD",
    "ContinuousDiDResults": "ContinuousDiD",
    "TripleDifferenceResults": "TripleDifference (DDD)",
    "BaconDecompositionResults": "BaconDecomposition",
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
        ``"heterogeneity"``, ``"robustness"``.
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
        raise ValueError(
            f"Unknown step names: {unknown}. Valid names: {sorted(STEPS)}"
        )

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
            "# Use .att (basic DiD) or .overall_att (staggered estimators)."
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
            step_name="sensitivity",
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
            step_name="sensitivity",
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
            step_name="sensitivity",
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
            step_name="sensitivity",
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
            label="Check pre-treatment fit quality",
            why=(
                "Synthetic DiD relies on pre-treatment fit to construct "
                "weights. Poor fit suggests the synthetic control may not "
                "approximate the counterfactual well."
            ),
            code=(
                "# Check pre-treatment fit and unit weight concentration:\n"
                "print(f'Pre-treatment fit (RMSE): {results.pre_treatment_fit:.4f}')\n"
                "# Highly concentrated weights suggest fragile estimates"
            ),
            step_name="sensitivity",
        ),
        _step(
            baker_step=6,
            label="In-time or in-space placebo",
            why=(
                "Test robustness by re-estimating on a placebo treatment "
                "period (in-time) or excluding treated units one at a time "
                "(leave-one-out). These are the natural falsification "
                "checks for synthetic control methods."
            ),
            code=(
                "# In-time placebo: re-estimate with a fake treatment date\n"
                "# Leave-one-out: drop each treated unit and re-estimate"
            ),
            priority="medium",
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
            step_name="sensitivity",
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
            step_name="sensitivity",
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
            step_name="heterogeneity",
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
            baker_step=7,
            label="Plot dose-response curve",
            why=(
                "Continuous DiD estimates treatment effects at each dose "
                "level. The dose-response curve reveals the functional form "
                "of the treatment-dose relationship."
            ),
            code=(
                "from diff_diff import plot_dose_response\n"
                "plot_dose_response(results)"
            ),
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
    "EfficientDiDResults": _handle_efficient,
    "ContinuousDiDResults": _handle_continuous,
    "TripleDifferenceResults": _handle_triple,
    "BaconDecompositionResults": _handle_bacon,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_nan_att(results: Any) -> List[str]:
    """Return warnings if ATT is NaN."""
    # Check .att (DiDResults), .overall_att (staggered), .avg_att (MultiPeriod)
    att = getattr(results, "att", None)
    if att is None:
        att = getattr(results, "overall_att", None)
    if att is None:
        att = getattr(results, "avg_att", None)
    if att is not None:
        try:
            att = float(att)
        except (TypeError, ValueError):
            return []
    if att is not None and math.isnan(att):
        return [
            "Estimation produced NaN ATT — check data preparation and "
            "model specification before proceeding with diagnostics."
        ]
    return []


def _filter_steps(
    steps: List[Dict[str, Any]], completed: Set[str]
) -> List[Dict[str, Any]]:
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
            print(f"\n  {marker} [{priority.upper()}] Step {step['baker_step']}: "
                  f"{step['label']}")
            print(f"    Why: {step['why']}")
            if step.get("code"):
                for line in step["code"].split("\n"):
                    print(f"    >>> {line}")
    else:
        print("\nAll Baker et al. steps completed!")

    print(f"\n{'='*60}\n")
