"""
Practitioner guidance for Difference-in-Differences analysis.

Implements Baker et al. (2025) "Difference-in-Differences Designs:
A Practitioner's Guide" as context-aware runtime guidance. Call
``practitioner_next_steps(results)`` after estimation to get a
structured set of recommended next steps.
"""

import math
from typing import Any, Dict, List, Optional, Set

from diff_diff.results_base import Diagnostic

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
    "DMLDiDResults": "DMLDiD (Chang 2020 double/debiased ML)",
    "SunAbrahamResults": "SunAbraham",
    "ImputationDiDResults": "ImputationDiD (Borusyak-Jaravel-Spiess)",
    "TwoStageDiDResults": "TwoStageDiD",
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
    "ChangesInChangesResults": "ChangesInChanges / QDiD",
}


def _distributional_kind(results: Any) -> Any:
    """Read a ChangesInChangesResults' method tag, old field name included.

    Row M-143 renamed the field ``estimator`` -> ``method``. The fallback is
    NOT ``getattr(results, "estimator", None)``: on a real results object that
    name is now a deprecation property, so reading it emits a FutureWarning
    even when it returns the right value - which turns into an error under the
    warning-as-error assertions this module's own tests use. Reading the
    instance ``__dict__`` finds a duck-typed hand-built attribute without ever
    touching the descriptor. Real results resolve ``method`` first, so the
    fallback only ever fires for mocks built before the rename.
    """
    kind = getattr(results, "method", None)
    if kind is not None:
        return kind
    return getattr(results, "__dict__", {}).get("estimator")


def _estimator_display(type_name: str, results: Any) -> str:
    """Per-instance display name.

    ``ChangesInChangesResults`` is shared by CiC and QDiD (``QDiDResults``
    is an alias), so the static per-type map cannot distinguish them; the
    ``method`` field ("cic"/"qdid") does. Defensive: mock results may
    lack the field, in which case the static entry is the fallback.
    """
    if type_name == "ChangesInChangesResults":
        kind = _distributional_kind(results)
        if kind == "cic":
            return "ChangesInChanges (CiC)"
        if kind == "qdid":
            return "QDiD"
    return _ESTIMATOR_NAMES.get(type_name, type_name)


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

    type_name = type(results).__name__
    # Marked diagnostic results route through diagnostic-specific
    # handling (spec section 3.5, ledger row M-091) instead of the
    # unknown-result estimator fallback. Bacon stays on its name-keyed
    # handler with its existing framing.
    diagnostic_input = isinstance(results, Diagnostic) and type_name not in _HANDLERS

    if not diagnostic_input:
        # Estimation is always complete if we have an estimator results
        # object; a diagnostic input carries no estimation of its own.
        completed.add("estimation")

    handler = _HANDLERS.get(type_name)
    if handler is None:
        handler = _handle_diagnostic if diagnostic_input else _handle_generic
    steps, warnings = handler(results)

    # Prepend Steps 1-2 (pre-estimation reasoning) to every handler's output.
    # These are always relevant and filterable via completed_steps.
    # Diagnostic inputs skip the estimator framing entirely - Steps 1-2
    # define an estimation target the diagnostic does not carry.
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
    # ChangesInChangesResults: the generic Step 2 asks for a
    # parallel-trends variant, contradicting CiC/QDiD's distributional
    # identification - swap in the distributional assumptions statement
    # (same step_name, so completed_steps filtering is unchanged).
    if type_name == "ChangesInChangesResults":
        pre_estimation[1] = _cic_assumptions_step(results)

    if not diagnostic_input:
        steps = pre_estimation + steps

    # Filter out completed steps
    steps = _filter_steps(steps, completed)

    output = {
        "estimator": (
            f"{type_name} (diagnostic result)"
            if diagnostic_input
            else _estimator_display(type_name, results)
        ),
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
                "Use CS post-fit results.aggregate('event_study') or check the estimator's "
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
        _robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD"),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_cs(results: Any):
    # The post-fit RECOMPUTE levels work on bootstrapped fits too: they
    # REPLAY the fit-time multiplier bootstrap from the kit-retained RNG
    # state (percentile inference matching a fit-time aggregation), so the
    # advice is the same post-fit route on both regimes - only the caveats
    # differ. Static guidance with a refit remedy is this module's
    # convention; a legacy pickle without the replay state gets the refit
    # message from aggregate() itself.
    is_bootstrap = getattr(results, "bootstrap_results", None) is not None
    if is_bootstrap:
        sensitivity_why = (
            "Bounds the treatment effect under plausible violations of "
            "parallel trends. This fit is BOOTSTRAPPED: "
            "aggregate('event_study') replays the fit-time multiplier "
            "bootstrap post-fit (percentile inference; the container "
            "carries no joint covariance, so HonestDiD uses its diagonal "
            "approximation). A result predating the replay state asks for "
            "a refit."
        )
        sensitivity_code = (
            "from diff_diff import compute_honest_did\n"
            "# Bootstrapped fit: aggregate post-fit - the replay reproduces\n"
            "# the fit-time percentile inference (no refit needed):\n"
            "es = results.aggregate('event_study')\n"
            "honest = compute_honest_did(es, method='relative_magnitude', M=1.0)\n"
            "print(honest.summary())"
        )
        heterogeneity_code = (
            "# Bootstrapped fit: aggregate post-fit - no refit needed:\n"
            "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
            "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
        )
    else:
        sensitivity_why = (
            "Bounds the treatment effect under plausible violations of "
            "parallel trends. Aggregate the event study post-fit — no "
            "refit needed."
        )
        sensitivity_code = (
            "from diff_diff import compute_honest_did\n"
            "# Aggregate post-fit; the container feeds HonestDiD directly:\n"
            "es = results.aggregate('event_study')\n"
            "honest = compute_honest_did(es, method='relative_magnitude', M=1.0)\n"
            "print(honest.summary())"
        )
        heterogeneity_code = (
            "# Aggregate post-fit - no refit needed:\n"
            "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
            "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
        )
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Run HonestDiD sensitivity analysis",
            why=sensitivity_why,
            code=sensitivity_code,
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
            code=heterogeneity_code,
            step_name="heterogeneity",
        ),
        _robustness_compare_step("SunAbraham, ImputationDiD, or TwoStageDiD"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_dml_did(results: Any):
    # Mirrors _handle_cs's post-fit aggregation advice (the DML container
    # inherits the CS kit machinery) MINUS the covariates step: _handle_cs
    # closes by recommending a refit with covariates=None, a call DMLDiD
    # hard-rejects (covariates are required). Adds a learner-sensitivity
    # robustness step instead.
    is_bootstrap = getattr(results, "bootstrap_results", None) is not None
    if is_bootstrap:
        sensitivity_why = (
            "Bounds the treatment effect under plausible violations of "
            "(conditional) parallel trends. This fit is BOOTSTRAPPED: "
            "aggregate('event_study') replays the fit-time multiplier "
            "bootstrap post-fit (percentile inference; the container "
            "carries no joint covariance, so HonestDiD uses its diagonal "
            "approximation)."
        )
    else:
        sensitivity_why = (
            "Bounds the treatment effect under plausible violations of "
            "(conditional) parallel trends. Aggregate the event study "
            "post-fit — no refit needed."
        )
    sensitivity_code = (
        "from diff_diff import compute_honest_did\n"
        "# Aggregate post-fit; the container feeds HonestDiD directly:\n"
        "es = results.aggregate('event_study')\n"
        "honest = compute_honest_did(es, method='relative_magnitude', M=1.0)\n"
        "print(honest.summary())"
    )
    heterogeneity_code = (
        "# Aggregate post-fit - no refit needed:\n"
        "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
        "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
    )
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Run HonestDiD sensitivity analysis",
            why=sensitivity_why,
            code=sensitivity_code,
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
            code=heterogeneity_code,
            step_name="heterogeneity",
        ),
        _robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD"),
        _step(
            baker_step=8,
            label="Check learner sensitivity",
            why=(
                "DML estimates plug in cross-fitted nuisance functions; a "
                "conclusion that survives a different outcome learner "
                "(e.g. the adaptive polynomial sieve, or a penalized "
                "ridge) is more credible than one that depends on the "
                "default linear specification."
            ),
            code=(
                "# Refit with alternative nuisance learners:\n"
                "alt = DMLDiD(outcome_learner='sieve', seed=0"
                + (", panel=False" if getattr(results, "panel", True) is False else "")
                + ").fit(\n"
                "    df, outcome=..., unit=..., time=..., first_treat=...,\n"
                "    covariates=[...])\n"
                "print(alt.att, results.att)  # should be close"
            ),
            step_name="learner_sensitivity",
        ),
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
        _robustness_compare_step("CallawaySantAnna, ImputationDiD, or TwoStageDiD"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_imputation(results: Any):
    steps = [
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=7,
            label="Aggregate treatment-effect heterogeneity post-fit",
            why=(
                "ImputationDiD aggregates post-fit from its panel-backed kit "
                "(M-021) - no refit needed."
                if getattr(results, "bootstrap_results", None) is None
                else "This fit is BOOTSTRAPPED: the post-fit event-study/group "
                "recompute levels raise on bootstrap fits, while "
                "aggregate('simple') and, where supported, "
                "aggregate('total') relay the stored inference - "
                "refit with the deprecated fit-time aggregation (or "
                "n_bootstrap=0) to obtain the recomputed surfaces."
            ),
            code=(
                "# Aggregate post-fit - no refit needed:\n"
                "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
                "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
                if getattr(results, "bootstrap_results", None) is None
                else "# Bootstrap fit: aggregate at fit time (deprecated kwarg):\n"
                "results = imp.fit(data, ..., aggregate='all')\n"
                "print(results.group_effects)        # Per-cohort ATTs\n"
                "print(results.event_study_effects)  # Dynamic effects"
            ),
            priority="medium",
            # NON-STEPS key (the M-024 "sub_experiment_balance" lesson):
            # a STEPS-vocabulary name would let _filter_steps suppress
            # this guidance whenever a same-named DiagnosticReport check
            # completes, which never runs this aggregation.
            step_name="aggregation",
        ),
        _step(
            baker_step=6,
            label="Specification-based falsification",
            why=(
                "ImputationDiD does not have a control_group parameter. "
                "Compare results with and without covariates, vary the "
                "sample (drop cohorts), and compare with CallawaySantAnna/"
                "SunAbraham as falsification checks."
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
        _robustness_compare_step("CallawaySantAnna, SunAbraham, or TwoStageDiD"),
        _covariates_step(),
    ]
    warnings = _check_nan_att(results)
    return steps, warnings


def _handle_two_stage(results: Any):
    steps = [
        _step(
            baker_step=7,
            label="Aggregate treatment-effect heterogeneity post-fit",
            why=(
                "TwoStageDiD aggregates post-fit from its panel-backed kit "
                "(M-022) - no refit needed."
                if getattr(results, "bootstrap_results", None) is None
                else "This fit is BOOTSTRAPPED: the post-fit event-study/group "
                "recompute levels raise on bootstrap fits, while "
                "aggregate('simple') and, where supported, "
                "aggregate('total') relay the stored inference - "
                "refit with the deprecated fit-time aggregation (or "
                "n_bootstrap=0) to obtain the recomputed surfaces."
            ),
            code=(
                "# Aggregate post-fit - no refit needed:\n"
                "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
                "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
                if getattr(results, "bootstrap_results", None) is None
                else "# Bootstrap fit: aggregate at fit time (deprecated kwarg):\n"
                "results = ts.fit(data, ..., aggregate='all')\n"
                "print(results.group_effects)        # Per-cohort ATTs\n"
                "print(results.event_study_effects)  # Dynamic effects"
            ),
            priority="medium",
            # NON-STEPS key (the M-024 "sub_experiment_balance" lesson):
            # a STEPS-vocabulary name would let _filter_steps suppress
            # this guidance whenever a same-named DiagnosticReport check
            # completes, which never runs this aggregation.
            step_name="aggregation",
        ),
        _parallel_trends_step(staggered=True),
        _step(
            baker_step=6,
            label="Specification-based falsification",
            why=(
                "TwoStageDiD does not have a control_group parameter. "
                "Compare results with and without covariates, vary the "
                "sample (drop cohorts), and compare with CallawaySantAnna/"
                "SunAbraham as falsification checks."
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
        _robustness_compare_step("CallawaySantAnna, ImputationDiD, or SunAbraham"),
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
                "StackedDiD's control_group parameter selects the "
                "clean-control rule (not_yet_treated / strict / "
                "never_treated). Compare results with different clean "
                "control definitions and event window widths as "
                "falsification."
            ),
            code=(
                "# Re-estimate with different control_group settings:\n"
                "# stacked_alt = StackedDiD(control_group='not_yet_treated')"
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
            # 3.9 (row M-024): a DISTINCT key, deliberately not
            # "heterogeneity". Since the StackedDiD event-study surface is
            # always populated, DiagnosticReport's heterogeneity check now
            # runs on every plain fit and marks that step completed - which
            # used to silently drop this UNRELATED balance advice from
            # next_steps (the step_name collision). Like "loo_jackknife",
            # this key stays OUT of the STEPS completion vocabulary: no
            # diagnostic ever completes it, so the advice always survives.
            step_name="sub_experiment_balance",
        ),
        _robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD"),
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
            label="Compare with staggered estimators (CallawaySantAnna, SunAbraham)",
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
        _robustness_compare_step("SyntheticDiD or CallawaySantAnna"),
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
                "# SyntheticControl().fit(..., donor_pool=[clean, comparable, units])"
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
        _robustness_compare_step("SyntheticDiD or CallawaySantAnna"),
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
        _step(
            baker_step=7,
            label="Aggregate treatment-effect heterogeneity post-fit",
            why=(
                "EfficientDiD aggregates post-fit from retained EIFs " "(M-023) - no refit needed."
                if getattr(results, "bootstrap_results", None) is None
                else "This fit is BOOTSTRAPPED: the post-fit event-study/group "
                "recompute levels REPLAY the fit-time multiplier "
                "bootstrap from the retained RNG state (percentile "
                "inference, no refit needed), while aggregate('simple') "
                "and, where supported, aggregate('total') relay the "
                "stored inference."
            ),
            code=(
                "# Aggregate post-fit - no refit needed:\n"
                "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
                "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
                if getattr(results, "bootstrap_results", None) is None
                else "# Bootstrap fit: post-fit aggregation replays the fit-time bootstrap:\n"
                "print(results.aggregate('group').to_dataframe())        # Per-cohort ATTs\n"
                "print(results.aggregate('event_study').to_dataframe())  # Dynamic effects"
            ),
            priority="medium",
            # NON-STEPS key (the M-024 "sub_experiment_balance" lesson):
            # a STEPS-vocabulary name would let _filter_steps suppress
            # this guidance whenever a same-named DiagnosticReport check
            # completes, which never runs this aggregation.
            step_name="aggregation",
        ),
        _robustness_compare_step("CallawaySantAnna, SunAbraham, or ImputationDiD"),
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
                "dependent - and fit() selects the mode from it (M-027): the overall (single-period WAS) estimator is two-period "
                "only and hard-rejects multi-period panels at fit time; "
                "multi-period panels fit the per-horizon event-study estimator. "
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
                "    data_had_2p, outcome='y', unit='unit',\n"
                "    time='t', dose='d', first_treat='first_treat')\n"
                "\n"
                "# Multi-period panel: fit() selects the event-study mode\n"
                "# (on staggered panels this is auto-last-cohort-only WAS)\n"
                "had_es = had.fit(\n"
                "    data_had_mp, outcome='y', unit='unit',\n"
                "    time='t', dose='d', first_treat='first_treat')"
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
        _step(
            baker_step=7,
            label="Aggregate post-fit (event study / dose / simple)",
            why=(
                "The fit-time aggregate= kwarg is deprecated (row M-025; "
                "removed in 4.0). Aggregate as a post-fit step instead: "
                "aggregate('event_study') recomputes the binarized event "
                "study from the retained kit (analytical fits only - on a "
                "bootstrapped fit it raises; re-fit with n_bootstrap=0 or "
                "use the deprecated fit-time aggregate='eventstudy' until "
                "4.0), while aggregate('dose') and aggregate('simple') are "
                "views over the always-computed curves and overall "
                "ATT/ACRT, available on any fit."
            ),
            code=(
                "es = results.aggregate('event_study')  # analytical fits\n"
                "dose_table = results.aggregate('dose')\n"
                "overall = results.aggregate('simple')  # att + acrt rows"
            ),
            step_name="aggregation",
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
                "comparisons, TWFE is biased. Use CallawaySantAnna, SunAbraham, "
                "ImputationDiD, or another heterogeneity-robust estimator "
                "for causal estimates."
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
                "panel (the workflow then runs the event-study battery) AND verify the panel "
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
                "    data, outcome='y', unit='unit',\n"
                "    time='t', dose='d',\n"
                "    first_treat='first_treat')\n"
                "print(report.summary())\n"
                "# verdict explicitly flags the Assumption 7 gap on the\n"
                "# overall path; a multi-period panel (the event-study battery)\n"
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
                "# pass it as `data_cdid` with the time-invariant `dose` column.\n"
                "cdid = ContinuousDiD()\n"
                "cdid_results = cdid.fit(\n"
                "    data_cdid, outcome='y', unit='unit', time='t',\n"
                "    first_treat='first_treat', dose='d')\n"
                "# Dose-response curves are always computed:\n"
                "#   cdid_results.dose_response_att / .aggregate('dose')"
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
            label="Re-fit on a multi-period panel for per-horizon WAS",
            why=(
                "On multi-period panels, the event-study mode returns "
                "per-event-time WAS estimates instead of a single scalar. "
                "Reveals whether dose response grows, decays, or stabilizes "
                "across post-treatment horizons. Pre-period placebos serve "
                "as a parallel-trends sanity check. NOTE: this handler is "
                "the single-period HAD handler, so the upstream fit was "
                "two-period-only (the overall estimator hard-rejects more "
                "than two periods; fit() selects the mode from the panel "
                "shape, M-027). Use a distinct multi-period panel "
                "`data_mp` for this step - the same panel that the "
                "upstream fit ran on will not satisfy the event-study "
                "path's period-count requirement."
            ),
            code=(
                "from diff_diff import HeterogeneousAdoptionDiD\n"
                "# Requires a distinct multi-period panel - the upstream\n"
                "# two-period panel already fit the overall (single-period) mode.\n"
                "est = HeterogeneousAdoptionDiD()\n"
                "es = est.fit(\n"
                "    data_mp, outcome='y', unit='unit',\n"
                "    time='t', dose='d',\n"
                "    first_treat='first_treat')"
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
                "runs the event-study battery (selected from the panel "
                "shape, M-139): QUG plus joint Stute "
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
                "    data, outcome='y', unit='unit',\n"
                "    time='t', dose='d',\n"
                "    first_treat='first_treat')\n"
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
                "surfaces are relevant and distinct: the per-dose ATT(d) / "
                "ACRT(d) curves are ALWAYS computed by fit() (on "
                "`results.dose_response_att` / `results.dose_response_acrt`, "
                "or as a table via `results.aggregate('dose')`); the "
                "binarized event-study of `att_glob` comes from post-fit "
                "`results.aggregate('event_study')` (NOT per-dose by "
                "horizon). "
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
                "# For per-dose ATT(d)/ACRT(d) curves, use ContinuousDiD -\n"
                "# fit() always computes them (post-fit aggregate('event_study')\n"
                "# gives the binarized event-study of att_glob instead, which\n"
                "# is NOT per-dose).\n"
                "from diff_diff import ContinuousDiD\n"
                "# ContinuousDiD requires a TIME-INVARIANT per-unit dose.\n"
                "# Re-prepare the panel (e.g. collapse each unit's positive\n"
                "# dose to one value) and pass it as `data_cdid`.\n"
                "cdid = ContinuousDiD()\n"
                "cdid_res = cdid.fit(\n"
                "    data_cdid, outcome='y', unit='unit', time='t',\n"
                "    first_treat='first_treat', dose='d')\n"
                "# Per-dose curves live here:\n"
                "#   cdid_res.dose_response_att / .dose_response_acrt\n"
                "#   (tabular: cdid_res.aggregate('dose'))"
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
                "# see had.py:3646-3658). On the continuous designs the\n"
                "# vcov_type kwarg is unused (CCT-2014 robust SE is the\n"
                "# only formula), so passing vcov_type='hc1' is a no-op\n"
                "# there and a safe default for the survey-aware example.\n"
                "est = HeterogeneousAdoptionDiD(\n"
                "    n_bootstrap=999, seed=42, vcov_type='hc1')\n"
                "es = est.fit(\n"
                "    data, outcome='y', unit='unit',\n"
                "    time='t', dose='d',\n"
                "    first_treat='first_treat',\n"
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


def _cic_assumptions_step(results: Any) -> Dict[str, Any]:
    """Step-2 (assumptions) override for ``ChangesInChangesResults``.

    The generic Step 2 asks the user to name a parallel-trends variant,
    which contradicts the CiC/QDiD guidance that identification is
    distributional, not mean parallel trends. Same baker_step/step_name,
    so ``completed_steps`` filtering is unchanged.
    """
    if _distributional_kind(results) == "qdid":
        why = (
            "Name the distributional assumptions you are invoking - not a "
            "mean parallel-trends variant. QDiD's justifying model "
            "requires the scalar unobservable's distribution to be "
            "identical in all FOUR (group, period) cells (U independent "
            "of group AND period) and is not invariant to monotone "
            "transformations of the outcome (Athey-Imbens 2006, p. 447); "
            "add no-anticipation and the continuous-outcome scope."
        )
    else:
        why = (
            "Name the distributional assumptions you are invoking - not a "
            "mean parallel-trends variant. CiC (Athey-Imbens 2006, "
            "Assumptions 3.1-3.4) requires a monotone outcome model "
            "h(U, T) strictly increasing in a scalar unobservable U, "
            "time-invariance of U within groups (U independent of T "
            "given G), and support inclusion; add no-anticipation and "
            "the continuous-outcome scope."
        )
    return _step(
        baker_step=2,
        label="State identification assumptions (distributional)",
        why=why,
        code="# Which distributional assumptions? Monotonicity in U? Time-invariance? Support?",
        priority="high",
        step_name="assumptions",
    )


def _cic_fit_snippet(
    results: Any,
    var: str,
    data_var: str = "data",
    covariates: str = "same",
    *,
    method: Optional[str] = None,
) -> str:
    """Render a ChangesInChanges constructor+fit snippet preserving the fit's design.

    Always emits ``ChangesInChanges(...)``: row M-015 deprecates the QDiD
    CLASS, so guidance must never hand the reader a ``QDiD(...)`` call. Pass
    ``method="qdid"`` to select that estimator on the merged surface; any other
    value (including the default ``None``) emits no ``method=`` argument at
    all, keeping CiC snippets byte-identical to the pre-merge text.

    Refit snippets must mirror the original specification: ``panel=True``
    changes the bootstrap resampling scheme (unit-block vs pooled rows)
    and ``covariates`` selects the conditional QR estimator - silently
    dropping either would make copied guidance run a different
    specification with different SEs/bands. ``covariates`` modes:
    ``"same"`` mirrors the original fit's covariate list, ``"none"`` is
    an explicitly unconditional refit, ``"add"`` inserts placeholder
    covariate names. Reads are defensive (mock results may lack fields).

    Inference settings are deliberately NORMALIZED: ``n_bootstrap=200``
    is the constructor default, ``seed=42`` is illustrative (the default
    is ``seed=None``), and custom ``quantiles``/``alpha`` are not
    mirrored - a 19-value quantile grid inlined into guidance would be
    unreadable, and none of these settings changes the identifying
    specification. The emitted snippet says so on its last lines.
    """
    panel = bool(getattr(results, "panel", False))
    covs = list(getattr(results, "covariates", None) or []) if covariates == "same" else []
    ctor_args = "n_bootstrap=200, seed=42"
    if method == "qdid":
        # First: it is the identification choice, not an inference knob.
        ctor_args = "method='qdid', " + ctor_args
    if panel:
        ctor_args += ", panel=True"
    extras = []
    if covariates == "add":
        extras.append("covariates=['x1', 'x2']")
    elif covs:
        extras.append(f"covariates={covs!r}")
    if panel:
        extras.append("unit='unit_id'")
    body = f"    {data_var}, outcome='y', treatment='treated', time='post'"
    if extras:
        body += ",\n    " + ", ".join(extras) + ")"
    else:
        body += ")"
    snippet = f"{var} = ChangesInChanges({ctor_args}).fit(\n" + body
    if panel:
        snippet += (
            "\n# panel=True + unit= mirror the original unit-block bootstrap (use your unit column)"
        )
    snippet += (
        "\n# n_bootstrap=200 is the default; seed=42 is illustrative (default seed=None)"
        "\n# carry over quantiles=/alpha= if you customized them"
    )
    return snippet


def _did_anchor_snippet(results: Any) -> str:
    """Render the mean-DiD anchor fit, mirroring the fit's covariates.

    For covariate CiC/QDiD results the anchor is covariate-adjusted mean
    DiD (``DifferenceInDifferences`` accepts ``covariates=``) - anchoring
    a conditional fit against a raw unadjusted mean would compare across
    two specification changes at once.
    """
    covs = list(getattr(results, "covariates", None) or [])
    args = "    data, outcome='y', treatment='treated', post='post'"
    if covs:
        args += f",\n    covariates={covs!r}"
    return "did_results = DifferenceInDifferences().fit(\n" + args + ")"


def _handle_cic(results: Any):
    """ChangesInChanges / QDiD guidance (shared results class).

    CiC and QDiD share ``ChangesInChangesResults`` (``QDiDResults`` is an
    alias), so this single handler branches on the ``method`` field
    ("cic"/"qdid"; unknown or missing kinds fall to the CiC branch - the
    paper-primary, safe-voiced default) and on covariate status
    (truthiness, not ``is not None``: fit() normalizes ``covariates=[]``
    to None but hand-built results may not). HonestDiD is deliberately
    never recommended here - it requires event-study effects, which this
    results type does not carry. The conditional-envelope support step is
    CiC-covariate-only: the QDiD covariate path has no support diagnostic
    (``_check_conditional_support`` is invoked on the CiC path only).
    """
    is_qdid = _distributional_kind(results) == "qdid"
    has_cov = bool(getattr(results, "covariates", None))
    # Display name for step LABELS (still the estimator's own name - the METHOD
    # "QDiD" is not deprecated, only the class spelling is).
    est_name = "QDiD" if is_qdid else "ChangesInChanges"
    # Constructor argument for emitted SNIPPETS: every snippet now builds
    # ChangesInChanges, selecting the estimator via method= (row M-015).
    _snippet_method = "qdid" if is_qdid else None

    if is_qdid:
        s3_why = (
            "QDiD does not identify off mean parallel trends alone. Its "
            "justifying model is stronger than CiC's: the scalar "
            "unobservable's distribution must be identical in all FOUR "
            "(group, period) cells, and the model is not invariant to "
            "monotone transformations of the outcome (Athey-Imbens 2006, "
            "p. 447). Because that additive quantile model moves every "
            "quantile - and hence the cell means - additively (QDiD's "
            "mean effect equals standard DiD's ATT in population), a "
            "pre-period mean-trend break IS evidence against QDiD's "
            "model: with extra pre-periods in the source panel, "
            "check_parallel_trends() on pre-period MEANS is a meaningful "
            "screen, though passing it does not validate the "
            "distributional restrictions - the two-pre-period "
            "distributional placebo (see the Placebo step) is the "
            "sharper exercise. Beyond the counterfactual-monotonicity "
            "check the fit already runs on unconditional fits, none of "
            "this is directly testable in a 2x2 design."
        )
        s3_code = (
            "# Meaningful MEANS screen for QDiD's additive model. Needs\n"
            "# extra pre-periods in the SOURCE panel - the 2x2 itself\n"
            "# has none by definition:\n"
            "from diff_diff import check_parallel_trends\n"
            "pt = check_parallel_trends(source_panel, outcome='y',\n"
            "                           time='period',\n"
            "                           treatment_group='treated')"
        )
    else:
        s3_why = (
            "CiC does not identify off mean parallel trends - and does "
            "not require them. Identification (Athey-Imbens 2006, "
            "Assumptions 3.1-3.4) needs a monotone outcome model "
            "h(U, T) strictly increasing in a scalar unobservable U, "
            "time-invariance of U within groups (U independent of T "
            "given G), and support inclusion - none directly testable "
            "in a 2x2 design. Under a nonlinear h, group mean trends "
            "need not be parallel in a valid CiC design, so a "
            "pre-period mean-trend break is NOT by itself evidence "
            "against CiC; check_parallel_trends() on pre-period means "
            "is at most a descriptive mean-DiD anchor, and the relevant "
            "falsification exercise is the two-pre-period distributional "
            "placebo (see the Placebo step). Also note additive random "
            "group-time shocks bias CiC - unlike linear DiD, where they "
            "only complicate inference - and are undetectable in a 2x2 "
            "(p. 476)."
        )
        s3_code = (
            "# CiC does not require mean parallel trends - the relevant\n"
            "# falsification is the two-pre-period distributional placebo\n"
            "# (see the Placebo step). Optional DESCRIPTIVE mean anchor\n"
            "# (needs extra pre-periods in the SOURCE panel):\n"
            "from diff_diff import check_parallel_trends\n"
            "pt = check_parallel_trends(source_panel, outcome='y',\n"
            "                           time='period',\n"
            "                           treatment_group='treated')\n"
            "# a mean-trend break here is NOT by itself evidence against CiC"
        )

    steps = [
        _step(
            baker_step=3,
            label=(
                "Assess the distributional identifying assumptions " "(not mean parallel trends)"
            ),
            why=s3_why,
            code=s3_code,
            step_name="parallel_trends",
        ),
    ]

    if is_qdid:
        steps.append(
            _step(
                baker_step=4,
                label="Prefer ChangesInChanges over QDiD (Athey-Imbens 2006, p. 447)",
                why=(
                    "Athey-Imbens recommend CiC over QDiD: QDiD's "
                    "justifying model is not invariant to monotone "
                    "transformations of the outcome, forces identical "
                    "unobservable distributions across all four cells, and "
                    "places testable restrictions on the data - "
                    "unconditional fits warn when the implied "
                    "counterfactual quantile function is non-monotone "
                    "(footnote 21; with covariates the check is moot, "
                    "since the imputed counterfactual's quantile curve is "
                    "monotone by construction). Use QDiD as a comparison "
                    "estimator alongside CiC, not as the primary."
                ),
                code=(
                    "from diff_diff import ChangesInChanges\n"
                    + _cic_fit_snippet(results, "cic_results")
                    + "\nprint(cic_results.summary())"
                ),
                step_name="estimator_selection",
            )
        )
    else:
        steps.append(
            _step(
                baker_step=4,
                label="Confirm the 2x2 distributional design fits the question",
                why=(
                    "CiC in diff-diff is 2x2-only (the Athey-Imbens "
                    "Section 6 multi-group/multi-period extension is "
                    "deferred; REGISTRY ChangesInChanges). Collapsing a "
                    "staggered panel to a 2x2 discards timing variation - "
                    "for staggered mean effects use CallawaySantAnna or "
                    "another heterogeneity-robust estimator. If the fit "
                    "warned about heavy ties (>10% duplicate outcome "
                    "values within a cell), the outcome looks discrete: "
                    "the continuous machinery silently delivers one "
                    "endpoint of the Athey-Imbens Section 4 bounds, not a "
                    "point estimate (discrete-outcome bounds are "
                    "deferred) - interpret accordingly."
                ),
                code=(
                    "# 2x2-only. For staggered mean effects switch estimators:\n"
                    "# from diff_diff import CallawaySantAnna\n"
                    "# Ties warning at fit? Point estimates are one endpoint\n"
                    "# of the Athey-Imbens Section 4 bounds (discrete\n"
                    "# outcomes; deferred), not point identification."
                ),
                priority="medium",
                step_name="estimator_selection",
            )
        )

    if not is_qdid and not has_cov:
        steps.append(
            _step(
                baker_step=6,
                label="Respect the interior point-identification range (eq. 17)",
                why=(
                    "Unconditional CiC quantile effects are "
                    "point-identified only strictly inside the open "
                    "interval (q_lower, q_upper) (Athey-Imbens eq. 17 / "
                    "Theorem 5.3). Quantiles at or outside the bounds "
                    "keep their point estimates (qte parity) but report "
                    "NaN inference. If the fit also warned about support "
                    "(Assumption 3.4), the counterfactual distribution is "
                    "only partially identified (Corollary 3.1) and the "
                    "ATT involves extrapolation at the support edges. "
                    "Report the interior range (summary() prints it) and "
                    "read tail quantiles as partially identified."
                ),
                code=(
                    "print(f'interior range: ({results.q_lower:.3f}, '\n"
                    "      f'{results.q_upper:.3f})')\n"
                    "qe = results.quantile_effects\n"
                    "outside = qe[(qe['quantile'] <= results.q_lower) |\n"
                    "             (qe['quantile'] >= results.q_upper)]\n"
                    "print(outside)  # point estimates kept, inference NaN by design"
                ),
                priority="medium",
                step_name="sensitivity",
            )
        )
    if not is_qdid and has_cov:
        steps.append(
            _step(
                baker_step=6,
                label="Verify conditional support/overlap (envelope diagnostic)",
                why=(
                    "With covariates the eq. 17 unconditional bounds are "
                    "not the relevant objects (q_lower/q_upper are NaN by "
                    "design); the support check is the "
                    "conditional-envelope diagnostic, which warns at fit "
                    "when more than 10% of treated pre-period outcomes "
                    "fall outside the span of their 99 predicted control "
                    "pre-period grid quantiles at their own covariates "
                    "(Melly-Santangelo 2015, Assumption 4 - the covariate "
                    "analogue of Athey-Imbens Assumption 3.4). Roughly 2% "
                    "outside is expected under correct specification (the "
                    "envelope spans taus 0.01-0.99). If the warning "
                    "fired, those conditional ranks are extrapolated tail "
                    "plateaus and the counterfactual involves "
                    "out-of-support extrapolation: simplify the covariate "
                    "set, trim non-overlap regions, refit, and compare."
                ),
                code=(
                    "# The envelope diagnostic fires at fit time (UserWarning).\n"
                    "# If it fired: inspect covariate overlap between the\n"
                    "# treated-pre and control-pre cells, simplify or trim,\n"
                    "# then refit and compare the QTE profile."
                ),
                priority="medium",
                step_name="sensitivity",
            )
        )

    steps.extend(
        [
            _step(
                baker_step=6,
                label=f"Placebo {est_name} on two pre-periods",
                why=(
                    "The 2x2 design has no extra pre-periods by "
                    "definition, but if the source panel has two or more "
                    "pre-treatment periods, refit the same estimator on "
                    "two of them with the later relabeled as post. QTE "
                    "and ATT should be near zero - systematic placebo "
                    "'effects' flag a time-invariance violation. Note "
                    "run_all_placebo_tests() vets the MEAN DiD only; the "
                    "distributional placebo is this refit."
                ),
                code=(
                    "# Requires >= 2 pre-periods in the SOURCE panel:\n"
                    "from diff_diff import ChangesInChanges\n"
                    "pre = source_panel[source_panel['period'].isin([p0, p1])].copy()\n"
                    "pre['post'] = (pre['period'] == p1).astype(int)\n"
                    + _cic_fit_snippet(results, "placebo", data_var="pre", method=_snippet_method)
                    + "\nprint(placebo.summary())  # QTE/ATT should be ~ 0"
                ),
                priority="medium",
                step_name="placebo",
            ),
            _step(
                baker_step=7,
                label="Read the full QTE profile with uniform bands",
                why=(
                    "Distributional heterogeneity is the point of the "
                    "estimator - report quantile_effects, not just the "
                    "headline ATT. When reading the profile jointly "
                    "across quantiles, pointwise CIs over-reject; "
                    "uniform_bands() gives sup-t simultaneous bands over "
                    "the quantile grid at a FIXED 95% level (qte parity - "
                    "the band level does not follow alpha; the pointwise "
                    "CIs do). Rows with NaN se (no bootstrap, failed "
                    "replicate gate, or outside the interior range in an "
                    "unconditional CiC fit) get NaN bands."
                ),
                code=(
                    "print(results.quantile_effects)  # per-quantile QTE + "
                    "pointwise inference\n"
                    "print(results.uniform_bands())   # sup-t simultaneous "
                    "bands (fixed 95%)"
                ),
                step_name="heterogeneity",
            ),
        ]
    )

    if has_cov:
        if is_qdid:
            s8b_why = (
                "Shows whether conditioning drives the results. No "
                "interior-range guard applies either way (eq. 17 has no "
                "QDiD analogue), but the unconditional refit re-activates "
                "the footnote-21 counterfactual-monotonicity check, which "
                "is moot on the covariate path."
            )
        else:
            s8b_why = (
                "Shows whether conditioning drives the results. The "
                "comparison is not like-for-like in the tails: the "
                "unconditional refit re-enables the eq. 17 interior-range "
                "guard (quantiles at or outside (q_lower, q_upper) get "
                "NaN inference) and the unconditional support check, "
                "while the covariate fit reports NaN q_lower/q_upper. "
                "Compare point profiles everywhere; compare inference "
                "only on the shared interior."
            )
        steps.append(
            _step(
                baker_step=8,
                label="Report with and without covariates",
                why=s8b_why,
                code=(
                    "from diff_diff import ChangesInChanges\n"
                    "# Explicitly UNCONDITIONAL refit (covariates dropped by design):\n"
                    + _cic_fit_snippet(
                        results, "results_nocov", covariates="none", method=_snippet_method
                    )
                    + "\nprint(results.att, results_nocov.att)"
                ),
                priority="medium",
                step_name="robustness",
            )
        )
    else:
        steps.append(
            _step(
                baker_step=8,
                label="Re-estimate with covariates if composition changed",
                why=(
                    "In repeated cross-sections especially, composition "
                    "change across periods undermines the time-invariance "
                    "assumption; the conditional fit assumes invariance "
                    "conditional on the covariates instead. It ports "
                    "qte's xformla branch: per-cell linear quantile "
                    "regressions on a fixed internal 99-tau grid with "
                    "conditional ranks, integrating over treated "
                    "PRE-period covariates (qte parity; the full "
                    "Melly-Santangelo treated-post integration is a "
                    "documented deferral). Numeric covariates only - "
                    "dummy-encode categoricals first. Runtime note: every "
                    "bootstrap replicate refits every per-cell quantile "
                    "regression, so covariate fits cost tens of seconds "
                    "at moderate cell sizes."
                ),
                code=(
                    "from diff_diff import ChangesInChanges\n"
                    + _cic_fit_snippet(
                        results, "results_cov", covariates="add", method=_snippet_method
                    )
                    + "\nprint(results.att, results_cov.att)  # compare ATT + QTE profiles"
                ),
                priority="medium",
                step_name="robustness",
            )
        )

    if is_qdid and not has_cov:
        steps.append(
            _step(
                baker_step=8,
                label="Anchor against mean DiD (population equivalence)",
                why=(
                    "Unconditional QDiD's mean effect matches standard "
                    "DiD's ATT in population (Athey-Imbens 2006, p. 447; "
                    "the implemented qte finite-sample form deviates from "
                    "the paper's transformation - see the REGISTRY Note). "
                    "A large finite-sample gap between the QDiD ATT and "
                    "the linear-DiD ATT therefore flags small cells, "
                    "heavy ties, or specification problems rather than a "
                    "distributional discovery."
                ),
                code=(
                    "from diff_diff import DifferenceInDifferences\n"
                    + _did_anchor_snippet(results)
                    + "\nprint(results.att, did_results.att)  # population-equal "
                    "mean effect"
                ),
                priority="medium",
                step_name="robustness",
            )
        )
    elif is_qdid:
        steps.append(
            _step(
                baker_step=8,
                label="Anchor against covariate-adjusted mean DiD (descriptive)",
                why=(
                    "The p. 447 population equivalence between QDiD's "
                    "mean effect and standard DiD's ATT is established "
                    "for the unconditional estimator; the covariate "
                    "branch imputes via conditional ranks and per-cell "
                    "quantile regression, and no analogous equivalence is "
                    "documented for it. Compare against a "
                    "covariate-adjusted mean DiD as a descriptive anchor "
                    "only - the two adjust for covariates differently "
                    "(linear regression adjustment vs conditional-rank "
                    "QR), so a gap is not by itself evidence of a problem "
                    "or a discovery."
                ),
                code=(
                    "from diff_diff import DifferenceInDifferences\n"
                    + _did_anchor_snippet(results)
                    + "\nprint(results.att, did_results.att)  # descriptive anchor only"
                ),
                priority="medium",
                step_name="robustness",
            )
        )
    else:
        if has_cov:
            s8c_why_tail = (
                "The covariate-adjusted mean-DiD ATT is a useful "
                "descriptive anchor, with the caveat that the two adjust "
                "for covariates differently (linear regression adjustment "
                "vs conditional-rank QR) - report both, and read gaps as "
                "descriptive rather than diagnostic."
            )
        else:
            s8c_why_tail = (
                "The linear-DiD ATT is a useful anchor: CiC's ATT can "
                "differ from it when the outcome model is nonlinear, so a "
                "gap is informative about nonlinearity rather than a red "
                "flag on its own - report both."
            )
        steps.append(
            _step(
                baker_step=8,
                label="Compare with QDiD and mean DiD",
                why=(
                    "QDiD is the natural comparison estimator (same 2x2 "
                    "cells, different justifying model); broadly agreeing "
                    "QTE profiles strengthen the distributional "
                    "conclusions, with CiC remaining the recommended "
                    "primary (p. 447). " + s8c_why_tail
                ),
                code=(
                    "from diff_diff import ChangesInChanges, DifferenceInDifferences\n"
                    + _cic_fit_snippet(results, "qdid_results", method="qdid")
                    + "\n"
                    + _did_anchor_snippet(results)
                    + "\nprint(results.att, qdid_results.att, did_results.att)"
                ),
                priority="medium",
                step_name="robustness",
            )
        )

    warnings = _check_nan_att(results) + _cic_bootstrap_warnings(results)
    return steps, warnings


def _handle_diagnostic(results: Any):
    """Marked diagnostic results (``diff_diff.Diagnostic``).

    Diagnostic containers assess a design, an assumption, or robustness
    and carry no causal inference row, so the estimator-style fallback
    (parallel-trends test, sensitivity, estimator comparison) does not
    apply. Route the user to interpreting the diagnostic alongside the
    primary estimator fit.
    """
    name = type(results).__name__
    steps = [
        _step(
            baker_step=2,
            label="Interpret the diagnostic alongside the primary fit",
            why=(
                f"{name} is a diagnostic result - it assesses a design, an "
                "identifying assumption, or robustness, and carries no "
                "causal-effect estimate. Read its summary() next to the "
                "primary estimator's results before drawing conclusions."
            ),
            code="print(diagnostic.summary())\ndiagnostic.to_dataframe()",
            step_name="assumptions",
        ),
        _step(
            baker_step=8,
            label="Get next steps from the estimator result",
            why=(
                "practitioner_next_steps() tailors its checklist to the "
                "ESTIMATOR result; pass the fitted estimator's results "
                "object for estimator-specific guidance."
            ),
            code="practitioner_next_steps(results)  # the estimator's results",
            step_name="estimation",
        ),
    ]
    return steps, []


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
    "DMLDiDResults": _handle_dml_did,
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
    "ChangesInChangesResults": _handle_cic,
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


def _cic_bootstrap_warnings(results: Any) -> List[str]:
    """Bootstrap-health warnings for ``ChangesInChangesResults``.

    ``_check_nan_att`` misses the conditions flagged here: with
    ``n_bootstrap=0``, ``n_bootstrap=1``, or a failed replicate gate the
    point estimates stay finite while every inference field is NaN.
    Reads are defensive (mock results may lack the fields) and accept
    numpy integer/float scalars alongside Python numbers. The 5%
    materiality threshold mirrors the fit-time
    ``warn_bootstrap_failure_rate`` threshold so the two surfaces never
    disagree about what counts as a notable failure rate.
    """
    import numpy as np

    numeric = (int, float, np.integer, np.floating)
    nb = getattr(results, "n_bootstrap", None)
    nv = getattr(results, "n_bootstrap_valid", None)
    if not isinstance(nb, numeric):
        return []
    if nb == 0:
        return [
            "Inference is disabled (n_bootstrap=0): every SE/t/p/CI field "
            "and the uniform bands are NaN. Refit with n_bootstrap > 0 "
            "(default 200) and a seed for reproducible bootstrap inference."
        ]
    if 0 < nb < 2:
        return [
            "n_bootstrap=1 cannot produce inference: the SE gate requires "
            "at least 2 valid replicates, so every SE/t/p/CI field and the "
            "uniform bands are NaN. Refit with n_bootstrap >= 2 (default "
            "200)."
        ]
    if isinstance(nv, numeric) and 0 <= nv < nb and (nb - nv) / nb > 0.05:
        return [
            f"Only {int(nv)} of {int(nb)} bootstrap replicates were valid "
            f"({(nb - nv) / nb:.0%} failed). With fewer than half (minimum "
            "2) valid replicates, all SEs and the sup-t critical value are "
            "already NaN; above that gate, SEs rest on fewer replicates "
            "than requested. Replicates whose resample empties a (group, "
            "period) cell - or, under covariates, whose quantile "
            "regression fails - are invalid; investigate cell sizes before "
            "trusting the inference."
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
