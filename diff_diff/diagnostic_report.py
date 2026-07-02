"""
DiagnosticReport — unified, plain-English validity assessment for diff-diff results.

Orchestrates the library's existing diagnostic functions (parallel trends,
pre-trends power, HonestDiD sensitivity, Goodman-Bacon, design-effect
diagnostics, EPV, heterogeneity, and estimator-native checks for SDiD/TROP)
into a single report with a stable AI-legible schema.

Design principles:

- No hard pass/fail gates. Severity is conveyed by natural-language phrasing,
  not a traffic-light enum. See ``docs/methodology/REPORTING.md``.
- No estimator fitting and no variance re-derivation from raw data. Every
  effect, SE, p-value, CI, and sensitivity bound is either read from
  ``results`` or produced by an existing diff-diff utility. May call
  ``check_parallel_trends`` / ``bacon_decompose`` /
  ``EfficientDiD.hausman_pretest`` when the caller supplies the panel +
  column kwargs. Report-layer cross-period aggregations (joint-Wald /
  Bonferroni pre-trends p-value, heterogeneity dispersion over
  post-treatment effects) are enumerated in
  ``docs/methodology/REPORTING.md``.
- Lazy evaluation. ``DiagnosticReport(results, ...)`` is free; ``run_all()``
  triggers compute and caches.
- Never prove a null. Pre-trends phrasing uses power information from
  ``compute_pretrends_power`` to distinguish well-powered from underpowered
  non-violations.

The ``to_dict()`` surface is an AI-legible contract. See the schema reference
in ``docs/methodology/REPORTING.md`` and the ``DIAGNOSTIC_REPORT_SCHEMA_VERSION``
constant below. The schema is marked experimental in v3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._reporting_helpers import describe_target_parameter  # noqa: E402 (top-level import)

DIAGNOSTIC_REPORT_SCHEMA_VERSION = "2.0"

__all__ = [
    "DiagnosticReport",
    "DiagnosticReportResults",
    "DIAGNOSTIC_REPORT_SCHEMA_VERSION",
]


# ---------------------------------------------------------------------------
# Canonical check names and per-type applicability
# ---------------------------------------------------------------------------
# The set of check names that ``DiagnosticReport`` supports.
_CHECK_NAMES: Tuple[str, ...] = (
    "parallel_trends",
    "pretrends_power",
    "sensitivity",
    "bacon",
    "design_effect",
    "heterogeneity",
    "epv",
    "estimator_native",
    "placebo",
)

# Type-level applicability: which checks are *ever* applicable for each of the
# 16 result types. Instance-level applicability further filters by whether
# required attributes are present (e.g. ``survey_metadata`` for DEFF) and by
# whether the user disabled a check via ``run_*=False``.
# See ``docs/methodology/REPORTING.md`` for the full matrix and rationale.
#
# Implementation note: The keys are result-class names looked up via
# ``type(results).__name__``. This string-based dispatch mirrors the
# ``_HANDLERS`` pattern in ``diff_diff/practitioner.py`` and avoids circular
# imports across the 16 result modules. Renaming or aliasing any result class
# requires updating both this table and ``_PT_METHOD`` below; the
# applicability-matrix test parametrized over all result types serves as the
# regression guard.
# ``pretrends_power`` is restricted to the result families for which
# ``compute_pretrends_power`` has an explicit adapter — see
# ``diff_diff/pretrends.py`` around the result-type dispatch. Expanding
# beyond this set (Imputation / Stacked / TwoStage / EfficientDiD /
# StaggeredTripleDiff / Wooldridge / dCDH) would cause the helper to
# raise ``TypeError("Unsupported results type ...")`` and mark the check
# as ``error``, so the narrower set is the right contract.
#
# ``sensitivity`` is restricted to families with a ``HonestDiD``
# adapter: MultiPeriod, CS, dCDH (via ``placebo_event_study``). SDiD
# and TROP use their own native paths (``estimator_native``) instead
# of HonestDiD.
_APPLICABILITY: Dict[str, FrozenSet[str]] = {
    "DiDResults": frozenset({"parallel_trends", "design_effect"}),
    "MultiPeriodDiDResults": frozenset(
        {"parallel_trends", "pretrends_power", "sensitivity", "bacon", "design_effect"}
    ),
    "CallawaySantAnnaResults": frozenset(
        {
            "parallel_trends",
            "pretrends_power",
            "sensitivity",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
        }
    ),
    "SunAbrahamResults": frozenset(
        {
            "parallel_trends",
            "pretrends_power",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "ImputationDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "TwoStageDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "StackedDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "SyntheticDiDResults": frozenset(
        {"parallel_trends", "sensitivity", "design_effect", "estimator_native"}
    ),
    "TROPResults": frozenset(
        # TROP identification is factor-model-based, not parallel-trends-
        # based: the estimator native ``_pt_factor()`` handler returns
        # ``status="not_applicable"``, and REPORTING.md routes TROP PT
        # to factor-model diagnostics instead. Exposing PT in
        # ``applicable_checks`` advertised a handler that never runs —
        # round-28 P2 CI review on PR #318 flagged the contract mismatch
        # for callers who gate workflows on ``applicable_checks``.
        {
            "sensitivity",
            "design_effect",
            "heterogeneity",
            "estimator_native",
        }
    ),
    "SyntheticControlResults": frozenset(
        # Classic SCM expresses its identifying assumption as design-enforced
        # pre-treatment fit (the donor weights match the treated unit's
        # pre-period trajectory), so — like SDiD — ``parallel_trends`` routes to
        # the fit-based ``_pt_scm_fit`` (verdict ``design_enforced_pt``, NOT a PT
        # hypothesis test; see prose). It also exposes ``estimator_native`` (the
        # in-space placebo permutation inference + weight concentration).
        # ``sensitivity`` is omitted (no HonestDiD analogue — significance testing
        # is the native placebo); ``heterogeneity`` is omitted (one treated unit).
        {
            "parallel_trends",
            "design_effect",
            "estimator_native",
        }
    ),
    "EfficientDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
        }
    ),
    "ContinuousDiDResults": frozenset({"design_effect", "heterogeneity"}),
    "TripleDifferenceResults": frozenset({"design_effect", "epv"}),
    "StaggeredTripleDiffResults": frozenset({"parallel_trends", "design_effect"}),
    "WooldridgeDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "ChaisemartinDHaultfoeuilleResults": frozenset(
        {
            "parallel_trends",
            "sensitivity",
            "bacon",
            "design_effect",
        }
    ),
    "BaconDecompositionResults": frozenset({"bacon"}),
}

# Per-type parallel-trends method. The PT check dispatches internally on this.
# Values:
#   "two_x_two"      — uses utils.check_parallel_trends (requires ``data``)
#   "event_study"    — joint Wald on pre-period event-study coefficients
#   "hausman"        — EfficientDiD.hausman_pretest (native PT-All vs PT-Post)
#   "synthetic_fit"  — SDiD weighted pre-treatment fit (surfaces pre_treatment_fit)
#   "factor"         — TROP factor-model identification (no PT; renders "N/A" prose)
#   "scm_fit"        — classic SCM donor-weighted pre-treatment fit (surfaces pre_rmspe)
_PT_METHOD: Dict[str, str] = {
    "DiDResults": "two_x_two",
    "MultiPeriodDiDResults": "event_study",
    "CallawaySantAnnaResults": "event_study",
    "SunAbrahamResults": "event_study",
    "ImputationDiDResults": "event_study",
    "TwoStageDiDResults": "event_study",
    "StackedDiDResults": "event_study",
    "EfficientDiDResults": "hausman",
    "ContinuousDiDResults": "event_study",
    "StaggeredTripleDiffResults": "event_study",
    "WooldridgeDiDResults": "event_study",
    "ChaisemartinDHaultfoeuilleResults": "event_study",
    "SyntheticDiDResults": "synthetic_fit",
    "TROPResults": "factor",
    "SyntheticControlResults": "scm_fit",
}


@dataclass(frozen=True)
class DiagnosticReportResults:
    """Frozen container holding the outcome of a ``DiagnosticReport.run_all()`` call.

    Attributes
    ----------
    schema : dict
        The AI-legible structured schema (also returned by ``to_dict()``).
    interpretation : str
        The ``overall_interpretation`` paragraph synthesizing findings across
        checks.
    applicable_checks : tuple of str
        The names of checks that applied to this estimator + options.
    skipped_checks : dict of str -> str
        Mapping from skipped-check name to plain-English reason.
    warnings : tuple of str
        Warnings captured while running the underlying diagnostic functions.
    """

    schema: Dict[str, Any]
    interpretation: str
    applicable_checks: Tuple[str, ...]
    skipped_checks: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


class DiagnosticReport:
    """Run the standard diff-diff diagnostic battery on a fitted result.

    Parameters
    ----------
    results : Any
        A fitted diff-diff results object (e.g. ``CallawaySantAnnaResults``,
        ``DiDResults``, ``SyntheticDiDResults``). Any of the 16 result types
        in the library is accepted.
    data : pandas.DataFrame, optional
        The underlying panel. Required for checks that need raw data
        (2x2 parallel-trends check on ``DiDResults``; Bacon-from-scratch when
        ``results`` is not itself a Bacon fit; the opt-in placebo battery).
    outcome, treatment, time, unit, first_treat : str, optional
        Column names identifying the panel structure.
    pre_periods, post_periods : list, optional
        Explicit pre- and post-treatment period labels.
    run_parallel_trends, run_sensitivity, run_placebo, run_bacon, run_design_effect, run_heterogeneity, run_epv, run_pretrends_power : bool
        Per-check opt-in flags. ``run_placebo`` defaults to ``False`` — the generic
        placebo battery is not implemented in MVP, so the ``placebo`` key remains
        reserved as ``skipped`` in the schema. (Exception: ``SyntheticControl``'s
        in-space placebo permutation test IS implemented — run it via
        ``results.in_space_placebo()``; its result is surfaced under
        ``estimator_native_diagnostics.in_space_placebo``, not this generic section.)
        All other checks default to ``True`` and are further gated by estimator-type
        and instance-level applicability (see ``docs/methodology/REPORTING.md``).
    sensitivity_M_grid : tuple of float, default (0.5, 1.0, 1.5, 2.0)
        Grid of M values passed to ``HonestDiD.sensitivity``. Yields a
        ``SensitivityResults`` object with ``breakdown_M`` populated.
    sensitivity_method : str, default "relative_magnitude"
        HonestDiD restriction type.
    alpha : float, default 0.05
        Significance level used across checks.
    survey_design : SurveyDesign, optional
        The ``SurveyDesign`` object used to fit a survey-weighted
        estimator. Required for fit-faithful replay of Goodman-Bacon on a
        survey-backed fit; threaded to ``bacon_decompose(survey_design=...)``.
        When the fit carries ``survey_metadata`` but ``survey_design`` is
        not supplied, Bacon is skipped with an explicit reason rather than
        replaying an unweighted decomposition for a design that does not
        match the estimate. The simple 2x2 parallel-trends helper
        (``utils.check_parallel_trends``) has no survey-aware variant;
        on a survey-backed ``DiDResults`` it is skipped unconditionally
        regardless of ``survey_design``. Supply
        ``precomputed={'parallel_trends': ...}`` with a survey-aware
        pretest to opt in. See ``docs/methodology/REPORTING.md``.
    precomputed : dict, optional
        Map of check name to a pre-computed result object. Accepted keys
        (this is the full implemented list; unsupported keys raise
        ``ValueError``):

        - ``"parallel_trends"`` — a dict returned by
          ``utils.check_parallel_trends`` (adapted into the schema shape).
        - ``"sensitivity"`` — a ``SensitivityResults`` (grid) or
          ``HonestDiDResults`` (single-M) object; used verbatim and no
          ``HonestDiD.sensitivity_analysis`` call is made.
        - ``"pretrends_power"`` — a ``PreTrendsPowerResults`` object.
        - ``"bacon"`` — a ``BaconDecompositionResults`` object.

        Other sections (``design_effect``, ``heterogeneity``, ``epv``) are
        read directly from the fitted result object and do not currently
        accept precomputed values — there is no expensive call to bypass.
        ``placebo`` is reserved in the schema but opt-in / deferred in MVP for the
        generic battery; ``SyntheticControl`` surfaces its in-space placebo under
        ``estimator_native_diagnostics`` (run ``results.in_space_placebo()``).
    outcome_label, treatment_label : str, optional
        Plain-English labels used in prose rendering.
    """

    def __init__(
        self,
        results: Any,
        *,
        data: Optional[pd.DataFrame] = None,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        time: Optional[str] = None,
        unit: Optional[str] = None,
        first_treat: Optional[str] = None,
        pre_periods: Optional[List[Any]] = None,
        post_periods: Optional[List[Any]] = None,
        run_parallel_trends: bool = True,
        run_sensitivity: bool = True,
        run_placebo: bool = False,
        run_bacon: bool = True,
        run_design_effect: bool = True,
        run_heterogeneity: bool = True,
        run_epv: bool = True,
        run_pretrends_power: bool = True,
        sensitivity_M_grid: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
        sensitivity_method: str = "relative_magnitude",
        alpha: float = 0.05,
        survey_design: Optional[Any] = None,
        precomputed: Optional[Dict[str, Any]] = None,
        outcome_label: Optional[str] = None,
        treatment_label: Optional[str] = None,
    ):
        self._results = results
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._time = time
        self._unit = unit
        self._first_treat = first_treat
        self._pre_periods = pre_periods
        self._post_periods = post_periods
        self._run_flags: Dict[str, bool] = {
            "parallel_trends": run_parallel_trends,
            "pretrends_power": run_pretrends_power,
            "sensitivity": run_sensitivity,
            "bacon": run_bacon,
            "design_effect": run_design_effect,
            "heterogeneity": run_heterogeneity,
            "epv": run_epv,
            "placebo": run_placebo,
            "estimator_native": True,
        }
        self._sensitivity_M_grid = tuple(sensitivity_M_grid)
        self._sensitivity_method = sensitivity_method
        self._alpha = float(alpha)
        # Round-40 P1 CI review on PR #318: survey-backed fits need the
        # ``SurveyDesign`` object threaded through to ``bacon_decompose``
        # for a fit-faithful Goodman-Bacon replay, and the unweighted
        # 2x2 parallel-trends helper (``utils.check_parallel_trends``)
        # cannot be called on a survey-weighted DiDResults without
        # silently reporting an unweighted verdict for a weighted fit.
        # When the fit carries ``survey_metadata`` but the caller did
        # not supply ``survey_design``, both checks skip with an
        # explicit reason instead of replaying a different design than
        # the estimate. See REPORTING.md "Survey-backed fits".
        self._survey_design = survey_design
        self._precomputed = dict(precomputed or {})
        # Validate precomputed keys against the actually-implemented passthrough
        # set so advertised contracts do not silently diverge from behavior.
        _supported_precomputed = {"parallel_trends", "sensitivity", "pretrends_power", "bacon"}
        _unsupported = set(self._precomputed) - _supported_precomputed
        if _unsupported:
            raise ValueError(
                "precomputed= contains keys that are not implemented: "
                f"{sorted(_unsupported)}. Supported keys: "
                f"{sorted(_supported_precomputed)}. ``design_effect``, "
                "``heterogeneity``, and ``epv`` are read directly from the "
                "fitted result and do not accept precomputed overrides."
            )

        # Estimator-aware precomputed validation. SDiD / TROP route
        # robustness to ``estimator_native_diagnostics`` (SDiD: weighted
        # pre-treatment fit, in-time placebo, zeta-omega sensitivity;
        # TROP: factor-model fit metrics), and TROP PT is not applicable
        # (factor-model identification, not PT). Accepting generic
        # HonestDiD / parallel-trends precomputed inputs on these
        # estimators would surface methodology-incompatible diagnostics
        # through the generic report sections — the opposite of the
        # native-routing contract documented in REPORTING.md.
        # Round-21 P1 CI review on PR #318 flagged this bypass.
        _result_name = type(self._results).__name__
        _native_routed_names = {"SyntheticDiDResults", "TROPResults", "SyntheticControlResults"}
        if _result_name in _native_routed_names:
            _incompatible_keys = []
            if "sensitivity" in self._precomputed:
                _incompatible_keys.append("sensitivity")
            # All native-routed estimators reject a precomputed ``parallel_trends``
            # payload — their PT verdict is computed INTERNALLY (SDiD/SCM:
            # design-enforced pre-treatment fit; TROP: factor-model identification),
            # not supplied as a user p-value. SyntheticControl is no exception: its
            # ``scm_fit`` route reads ``pre_rmspe`` from the fit, and the generic
            # precomputed-PT adapter only understands p-value-style payloads, which
            # are methodology-incompatible with SCM (it has no PT p-value test).
            if "parallel_trends" in self._precomputed:
                _incompatible_keys.append("parallel_trends")
            # Round-32 P1 CI review on PR #318: ``pretrends_power`` is a
            # Roth-style power analysis on pre-period event-study
            # coefficients under the PT identifying contract. SDiD's PT
            # analogue is design-enforced pre-treatment fit and TROP uses
            # factor-model identification (PT not applicable); surfacing
            # a Roth-style power tier on either would bypass the native-
            # routing contract. Round-21's guard covered ``sensitivity``
            # and ``parallel_trends`` but not ``pretrends_power``, so the
            # round-31 ``_compute_applicable_checks`` broadening exposed
            # it.
            if "pretrends_power" in self._precomputed:
                _incompatible_keys.append("pretrends_power")
            if _incompatible_keys:
                raise ValueError(
                    f"{_result_name} routes robustness and pre-trends "
                    "diagnostics to ``estimator_native_diagnostics`` — "
                    "generic HonestDiD, parallel-trends, and pre-trends "
                    "power precomputed passthroughs are methodology-"
                    "incompatible with this estimator. Rejected "
                    f"precomputed keys: {sorted(_incompatible_keys)}. "
                    "Use the native diagnostics on the result object "
                    "(SDiD: ``in_time_placebo``, ``sensitivity_to_zeta_omega``, "
                    "``pre_treatment_fit``; TROP: ``effective_rank``, "
                    "``loocv_score``; SyntheticControl: ``in_space_placebo()``, "
                    "``pre_rmspe``, ``get_placebo_df()``) — DR surfaces these "
                    "automatically under ``estimator_native_diagnostics``."
                )

        # Round-44 P1 CI review on PR #318: mirror the SDiD/TROP
        # __init__ rejection pattern for ``CallawaySantAnna`` with
        # ``base_period != "universal"``. HonestDiD bounds are not
        # valid for interpretation on consecutive-comparison
        # (``base_period='varying'``) pre-period surfaces (REGISTRY.md
        # §CallawaySantAnna line 410 plus §HonestDiD line 2458).
        # ``precomputed["sensitivity"]`` would otherwise bypass the
        # applicability-gate guard (which already existed for the auto
        # path) and let BR/DR narrate the Rambachan-Roth bounds as
        # ordinary robustness on a displayed fit whose interpretation
        # does not match the bounds' provenance. Reject at
        # construction so users get the error up-front rather than a
        # late skip in the schema.
        if _result_name == "CallawaySantAnnaResults" and "sensitivity" in self._precomputed:
            _base_period = getattr(self._results, "base_period", "universal")
            if _base_period != "universal":
                raise ValueError(
                    "precomputed['sensitivity'] on "
                    "CallawaySantAnnaResults requires "
                    "``base_period='universal'`` on the displayed fit — "
                    "HonestDiD Rambachan-Roth bounds are not valid for "
                    "interpretation on the consecutive-comparison "
                    "pre-period surface produced by "
                    f"``base_period={_base_period!r}``. Narrating the "
                    "bounds as robustness alongside a varying-base fit "
                    "mixes provenance the bounds don't support. Re-fit "
                    "the main estimator with "
                    "``CallawaySantAnna(base_period='universal')`` "
                    "before passing precomputed sensitivity."
                )

        self._outcome_label = outcome_label
        self._treatment_label = treatment_label
        self._cached: Optional[DiagnosticReportResults] = None

    # -- Public API ---------------------------------------------------------

    def run_all(self) -> DiagnosticReportResults:
        """Run all applicable diagnostics. Idempotent; caches on first call."""
        if self._cached is None:
            self._cached = self._execute()
        return self._cached

    def to_dict(self) -> Dict[str, Any]:
        """Return the AI-legible structured schema."""
        return self.run_all().schema

    def summary(self) -> str:
        """Return a short plain-English paragraph."""
        return self.run_all().interpretation

    def full_report(self) -> str:
        """Return the multi-section markdown report."""
        return _render_dr_full_report(self.run_all())

    def export_markdown(self) -> str:
        """Alias for ``full_report()``."""
        return self.full_report()

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per check with status and headline metric."""
        schema = self.to_dict()
        rows = []
        for check in _CHECK_NAMES:
            section_key = "estimator_native_diagnostics" if check == "estimator_native" else check
            section = schema.get(section_key, {})
            rows.append(
                {
                    "check": check,
                    "status": section.get("status"),
                    "headline": _check_headline(check, section),
                    "reason": section.get("reason"),
                }
            )
        return pd.DataFrame(rows)

    @property
    def applicable_checks(self) -> Tuple[str, ...]:
        """Names of checks that will run, given estimator + instance + options.

        No compute is triggered; this reflects only the applicability matrix
        filtered by instance state (survey_metadata, epv_diagnostics, vcov)
        and the user's ``run_*`` flags.
        """
        return tuple(sorted(self._compute_applicable_checks()[0]))

    @property
    def skipped_checks(self) -> Dict[str, str]:
        """Mapping of skipped check -> plain-English reason. Requires ``run_all()``."""
        return dict(self.run_all().skipped_checks)

    # -- Implementation detail ---------------------------------------------

    def _compute_applicable_checks(self) -> Tuple[set, Dict[str, str]]:
        """Compute the applicable-check set + per-check skipped reasons.

        Returns
        -------
        applicable : set of str
            Checks that will run.
        skipped : dict
            Mapping from check name -> plain-English reason for any check
            that is type-applicable but skipped for this instance or by user
            opt-out. Checks that are not type-applicable for this estimator
            are omitted from both sets (not surfaced as "skipped").
        """
        type_name = type(self._results).__name__
        type_level = set(_APPLICABILITY.get(type_name, frozenset()))
        # A precomputed passthrough is a caller-supplied override, not
        # a claim about estimator-native applicability. Round-31 P1 CI
        # review on PR #318: when a caller passes
        # ``precomputed["sensitivity"] = ...`` on an estimator family
        # whose ``_APPLICABILITY`` row lacks ``"sensitivity"`` (SA,
        # Imputation, TwoStage, Stacked, EfficientDiD, Wooldridge,
        # TripleDifference, StaggeredTripleDiff, ContinuousDiD, plain
        # DiD), the gate previously filtered the section out silently
        # and the supplied result disappeared from the schema. SDiD
        # and TROP are still rejected up front in ``__init__``
        # (round-21) because their native-routing contract makes
        # HonestDiD methodology-incompatible; those never reach here.
        # For every other estimator, an explicit passthrough wins
        # over the default applicability matrix.
        type_level = type_level | set(self._precomputed)
        applicable: set = set()
        skipped: Dict[str, str] = {}

        for check in type_level:
            # Per-check user opt-out
            if not self._run_flags.get(check, True):
                skipped[check] = f"run_{check}=False (user opted out)"
                continue
            # Instance-level gating — skipped when the caller supplied
            # a precomputed override (the per-check ``_instance_skip_reason``
            # branches already return None for precomputed keys, but this
            # short-circuit makes the override contract explicit and
            # survives any future gate additions).
            if check in self._precomputed:
                applicable.add(check)
                continue
            reason = self._instance_skip_reason(check)
            if reason is not None:
                skipped[check] = reason
                continue
            applicable.add(check)

        # Placebo is reserved for every result type in MVP so the schema
        # shape is stable: ``schema["placebo"]["status"] == "skipped"``
        # always holds regardless of estimator. The opt-in execution path
        # is deferred to a follow-up; ``REPORTING.md`` documents this.
        # SyntheticControl is the exception — its in-space placebo permutation test
        # IS implemented (run via results.in_space_placebo()) and surfaced under
        # estimator_native_diagnostics, so its generic-section reason points there
        # rather than claiming placebo is unimplemented (``setdefault`` wins over
        # the generic message below).
        if type(self._results).__name__ == "SyntheticControlResults":
            skipped.setdefault(
                "placebo",
                "SyntheticControl's placebo battery is the in-space placebo "
                "permutation test — run results.in_space_placebo(); the result is "
                "surfaced under estimator_native_diagnostics.in_space_placebo, not "
                "this generic section.",
            )
        skipped.setdefault(
            "placebo",
            "Placebo battery runs on opt-in only; not yet implemented in MVP. "
            "Reserved in the schema for forward compatibility.",
        )

        return applicable, skipped

    def _instance_skip_reason(self, check: str) -> Optional[str]:
        """Return a plain-English reason this check cannot run on this instance, or None."""
        r = self._results
        name = type(r).__name__
        if check == "design_effect":
            if getattr(r, "survey_metadata", None) is None:
                return "No survey design attached to results.survey_metadata."
            return None
        if check == "epv":
            if getattr(r, "epv_diagnostics", None) is None:
                return "Estimator did not produce results.epv_diagnostics for this fit."
            return None
        if check == "parallel_trends":
            # Precomputed parallel-trends always unlocks this check. The
            # EfficientDiD Hausman skip message already points users at
            # ``precomputed={'parallel_trends': ...}`` when replay fails
            # (DR / survey fits), so applicability must honor the
            # override before the replay-gate below fires. Round-22 P1
            # CI review on PR #318 flagged that PT precomputed was
            # advertised but skipped before use.
            if "parallel_trends" in self._precomputed:
                return None
            method = _PT_METHOD.get(name)
            if method == "two_x_two":
                # Mirror the full argument contract of ``_pt_two_x_two``:
                # the runner needs ``data`` AND all three column names to
                # call ``check_parallel_trends``. Gating only on ``data``
                # (as before) left ``applicable_checks`` overstated when
                # one of the column kwargs was missing (round-11 CI
                # review on PR #318).
                two_x_two_missing = [
                    arg
                    for arg, val in (
                        ("data", self._data),
                        ("outcome", self._outcome),
                        ("time", self._time),
                        ("treatment", self._treatment),
                    )
                    if val is None
                ]
                if two_x_two_missing:
                    return (
                        "2x2 parallel-trends check needs raw panel data + "
                        "outcome / time / treatment column names. Missing: "
                        + ", ".join(two_x_two_missing)
                        + "."
                    )
                # Round-40 P1 CI review on PR #318: the simple 2x2 helper
                # ``utils.check_parallel_trends`` is unweighted — it has
                # no ``survey_design`` parameter and cannot faithfully
                # diagnose the pre-period trajectory of a survey-
                # weighted DiDResults. Rather than silently emitting
                # an unweighted verdict alongside the weighted estimate,
                # skip with an explicit reason. Users can supply
                # ``precomputed={'parallel_trends': ...}`` with a
                # survey-aware pretest result if they have one.
                if getattr(r, "survey_metadata", None) is not None:
                    return (
                        "Original fit used a survey design; the simple "
                        "2x2 parallel-trends check (``utils."
                        "check_parallel_trends``) is unweighted and "
                        "would diagnose a different design than the "
                        "weighted estimate. Supply a survey-aware "
                        "pretest via "
                        "``precomputed={'parallel_trends': ...}`` to "
                        "opt in."
                    )
            if method == "event_study":
                pre_coefs, n_dropped_undefined = _collect_pre_period_coefs(r)
                # Round-42 P1 CI review on PR #318: the all-undefined
                # pre-period case (every pre-row dropped for ``se <= 0``
                # / non-finite inference) is the twin of the partial-
                # undefined case from round-33. It must route to the
                # inconclusive runner rather than skip, so the explicit
                # ``method="inconclusive"`` / ``n_dropped_undefined``
                # provenance is surfaced through DR's schema and BR's
                # summary emits the "inconclusive" identifying-
                # assumption warning rather than silently dropping PT.
                if not pre_coefs and n_dropped_undefined == 0:
                    return (
                        "No pre-period event-study coefficients are exposed on "
                        "this fit. For staggered estimators, re-fit with "
                        "aggregate='event_study' to populate event-study output."
                    )
                # vcov is optional for the Bonferroni fallback.
            if method == "hausman":
                # EfficientDiD's Hausman pretest requires the raw panel
                # to refit under PT-All and PT-Post. Gate at applicability
                # rather than letting ``_pt_hausman`` skip at runtime, so
                # ``applicable_checks`` and ``completed_steps`` reflect
                # reality.
                hausman_missing = [
                    arg
                    for arg, val in (
                        ("data", self._data),
                        ("outcome", self._outcome),
                        ("unit", self._unit),
                        ("time", self._time),
                        ("first_treat", self._first_treat),
                    )
                    if val is None
                ]
                if hausman_missing:
                    return (
                        "EfficientDiD.hausman_pretest needs raw panel data; "
                        "pass data + outcome + unit + time + first_treat to "
                        "DiagnosticReport. Missing: " + ", ".join(hausman_missing) + "."
                    )
                # Fit-faithful guard: DR / survey fits cannot be replayed
                # under defaults, so skip with an explicit reason rather
                # than rerunning a different design.
                if getattr(r, "estimation_path", "nocov") != "nocov":
                    return (
                        "Original EfficientDiD fit used the doubly-robust "
                        "covariate path; ``covariates`` is not stored on "
                        "the result, so the Hausman pretest cannot be "
                        "faithfully replayed."
                    )
                if getattr(r, "survey_metadata", None) is not None:
                    return (
                        "Original EfficientDiD fit used a survey design; "
                        "replaying the Hausman pretest would require the "
                        "full ``SurveyDesign`` object."
                    )
            return None
        if check == "pretrends_power":
            # ``compute_pretrends_power`` handles CS / SA / ImputationDiD
            # event-study results by reading ``event_study_effects``
            # directly, so we accept either a top-level ``vcov`` OR a
            # populated event-study surface. Precomputed overrides also
            # bypass this gate.
            if "pretrends_power" in self._precomputed:
                return None
            has_vcov = getattr(r, "vcov", None) is not None
            has_event_vcov = getattr(r, "event_study_vcov", None) is not None
            has_event_es = getattr(r, "event_study_effects", None) is not None
            if not (has_vcov or has_event_vcov or has_event_es):
                return (
                    "Pre-trends power needs either results.vcov or "
                    "event_study_effects (from aggregate='event_study' on "
                    "staggered estimators); neither available."
                )
            pre_coefs, _ = _collect_pre_period_coefs(r)
            if len(pre_coefs) < 2:
                return "Pre-trends power needs >= 2 pre-treatment periods."
            return None
        if check == "sensitivity":
            # Native SDiD/TROP paths substitute for HonestDiD.
            if name in {"SyntheticDiDResults", "TROPResults"}:
                return None
            # Round-44 P1 CI review on PR #318: the CS varying-base
            # guard MUST fire before the precomputed early-return.
            # Previously, ``precomputed["sensitivity"]`` unlocked this
            # check unconditionally, letting BR/DR narrate the
            # Rambachan-Roth bounds as ordinary robustness even though
            # HonestDiD explicitly warns those bounds are not valid
            # for interpretation on consecutive-comparison
            # (``base_period='varying'``) pre-period surfaces
            # (REGISTRY.md §CallawaySantAnna line 410, §HonestDiD line
            # 2458). The previous skip message also mis-pointed users
            # at ``precomputed`` as the opt-in; that path now routes
            # through the same guard, so the correct remediation is to
            # re-fit the main estimator with ``base_period='universal'``
            # or to consult HonestDiD outside the report layer.
            if name == "CallawaySantAnnaResults":
                base_period = getattr(r, "base_period", "universal")
                if base_period != "universal":
                    return (
                        "HonestDiD on CallawaySantAnna requires "
                        "``base_period='universal'`` for valid interpretation "
                        "(Rambachan-Roth bounds are not comparable across the "
                        "consecutive pre-period comparisons produced by "
                        f"``base_period={base_period!r}``). Re-fit with "
                        "``CallawaySantAnna(base_period='universal')``; "
                        "``precomputed={'sensitivity': ...}`` is rejected here "
                        "because the precomputed bounds would be narrated as "
                        "robustness for a displayed fit whose pre-period "
                        "surface has a different interpretation than the one "
                        "the bounds were computed against."
                    )
            # Precomputed sensitivity unlocks this check for every
            # other estimator (SDiD/TROP were already rejected at DR
            # __init__; CS varying-base is gated above). The CS
            # guard above runs on the *displayed fit*, not on the
            # provenance of the precomputed bounds; it protects
            # against narrating bounds whose interpretation is
            # incompatible with the fit being summarized.
            if "sensitivity" in self._precomputed:
                return None
            # dCDH uses ``placebo_event_study`` as its pre-period surface,
            # which HonestDiD consumes via a dedicated branch. Accept the
            # fit when that attribute is populated.
            if name == "ChaisemartinDHaultfoeuilleResults":
                pes = getattr(r, "placebo_event_study", None)
                if pes is None:
                    return (
                        "HonestDiD on dCDH requires results.placebo_event_study "
                        "(re-fit with a placebo-producing configuration)."
                    )
                return None
            # MultiPeriod / CS path: ``HonestDiD.sensitivity_analysis``
            # consumes ``event_study_effects`` plus either ``vcov`` +
            # ``interaction_indices`` (MultiPeriod) or ``event_study_vcov``
            # + ``event_study_vcov_index`` (CS), with a per-SE diagonal
            # fallback otherwise.
            has_vcov = getattr(r, "vcov", None) is not None
            has_event_vcov = getattr(r, "event_study_vcov", None) is not None
            has_event_es = getattr(r, "event_study_effects", None) is not None
            if not (has_vcov or has_event_vcov or has_event_es):
                return (
                    "HonestDiD needs either results.vcov, event_study_vcov, "
                    "or event_study_effects; none available."
                )
            pre_coefs, _ = _collect_pre_period_coefs(r)
            if len(pre_coefs) < 1:
                return "HonestDiD requires at least one pre-period coefficient."
            return None
        if check == "bacon":
            # Precomputed Bacon always unlocks this check. Users with an
            # already-computed ``BaconDecompositionResults`` (e.g., run
            # separately against a stored panel that isn't available at
            # report time) need the passthrough to land on the Bacon
            # runner instead of being skipped for missing column kwargs.
            # Round-22 P1 CI review on PR #318 flagged that Bacon
            # precomputed was advertised but skipped before use.
            if "bacon" in self._precomputed:
                return None
            # ``BaconDecompositionResults`` carries the decomposition
            # directly; no data/column kwargs needed.
            if name == "BaconDecompositionResults":
                return None
            # Otherwise mirror the full argument contract of
            # ``_check_bacon`` / ``bacon_decompose``: the runner needs
            # ``data``, ``first_treat``, and the ``outcome`` / ``time`` /
            # ``unit`` column names. Gating on only ``data`` +
            # ``first_treat`` (as before) left ``applicable_checks``
            # overstated when a column kwarg was missing (round-11 CI
            # review on PR #318).
            bacon_missing = [
                arg
                for arg, val in (
                    ("data", self._data),
                    ("outcome", self._outcome),
                    ("time", self._time),
                    ("unit", self._unit),
                    ("first_treat", self._first_treat),
                )
                if val is None
            ]
            if bacon_missing:
                return (
                    "Bacon decomposition needs panel data + outcome / time "
                    "/ unit / first_treat column names. Missing: " + ", ".join(bacon_missing) + "."
                )
            # Round-40 P1 CI review on PR #318: ``bacon_decompose``
            # supports a ``survey_design`` kwarg for survey-weighted
            # decomposition. When the fitted result carries
            # ``survey_metadata`` but the caller did not supply a
            # ``survey_design`` object, replaying with defaults would
            # produce an unweighted decomposition for a different
            # design than the weighted estimate. Skip with an explicit
            # reason; users can pass ``survey_design=<design>`` on
            # ``DiagnosticReport`` / ``BusinessReport`` or supply
            # ``precomputed={'bacon': ...}`` with a survey-aware
            # decomposition.
            if getattr(r, "survey_metadata", None) is not None and self._survey_design is None:
                return (
                    "Original fit used a survey design; Goodman-Bacon "
                    "replay under defaults would produce an unweighted "
                    "decomposition for a different design than the "
                    "weighted estimate. Pass ``survey_design=<SurveyDesign>`` "
                    "on DiagnosticReport / BusinessReport, or supply "
                    "``precomputed={'bacon': ...}`` with a survey-aware "
                    "decomposition."
                )
            return None
        if check == "heterogeneity":
            # Needs multiple group or event-study effects. Use len() rather than
            # truthiness because some estimators expose these as DataFrames,
            # which raise on bool() conversion.
            for attr in (
                "group_effects",
                "event_study_effects",
                "treatment_effects",  # TROP per-(unit, time)
                "group_time_effects",  # CS default aggregation
                "period_effects",  # MultiPeriod
            ):
                val = getattr(r, attr, None)
                if val is None:
                    continue
                try:
                    if len(val) > 0:
                        return None
                except TypeError:
                    continue
            return "No group/event-study effects available to compute heterogeneity."
        if check == "estimator_native":
            if name not in {"SyntheticDiDResults", "TROPResults", "SyntheticControlResults"}:
                return f"{name} does not expose native validation methods."
            return None
        return None

    def _execute(self) -> DiagnosticReportResults:
        """Run the diagnostic battery and assemble the schema."""
        applicable, skipped = self._compute_applicable_checks()

        # Initialize all schema sections to either "ran"/"skipped"/"not_applicable".
        sections: Dict[str, Dict[str, Any]] = {}
        for check in _CHECK_NAMES:
            if check in applicable:
                sections[check] = {"status": "not_run", "reason": "pending implementation"}
            elif check in skipped:
                sections[check] = {"status": "skipped", "reason": skipped[check]}
            else:
                sections[check] = {
                    "status": "not_applicable",
                    "reason": f"{check} is not applicable to " f"{type(self._results).__name__}.",
                }

        # Run the checks that are applicable. Each returns a schema-section dict
        # that replaces the placeholder above.
        if "parallel_trends" in applicable:
            sections["parallel_trends"] = self._check_parallel_trends()
        if "pretrends_power" in applicable:
            sections["pretrends_power"] = self._check_pretrends_power()
        if "sensitivity" in applicable:
            sections["sensitivity"] = self._check_sensitivity()
        if "bacon" in applicable:
            sections["bacon"] = self._check_bacon()
        if "design_effect" in applicable:
            sections["design_effect"] = self._check_design_effect()
        if "heterogeneity" in applicable:
            sections["heterogeneity"] = self._check_heterogeneity()
        if "epv" in applicable:
            sections["epv"] = self._check_epv()
        if "estimator_native" in applicable:
            sections["estimator_native"] = self._check_estimator_native()

        # Estimator-native placeholder: SDiD/TROP diagnostics come in a later task.
        if "estimator_native" not in applicable and "estimator_native" not in skipped:
            sections["estimator_native"] = {
                "status": "not_applicable",
                "reason": f"{type(self._results).__name__} does not expose native "
                "validation methods beyond what's captured above.",
            }

        # Headline metric — best-effort across estimator types.
        # PR #347 R4 P1: the dCDH ``trends_linear=True`` + ``L_max>=2``
        # configuration does not produce a scalar headline by design
        # (``overall_att`` is intentionally NaN per
        # ``chaisemartin_dhaultfoeuille.py:2828-2834``). Route the
        # headline through a dedicated no-scalar block when the
        # target-parameter helper flags this case so prose does not
        # narrate it as an estimation failure.
        _tp_agg = describe_target_parameter(self._results).get("aggregation")
        if _tp_agg == "no_scalar_headline":
            # PR #347 R12 P1: distinguish populated vs empty per-horizon
            # surface. Pointing users at ``linear_trends_effects`` is
            # dead-end guidance when that dict is ``None``.
            _surface_empty = getattr(self._results, "linear_trends_effects", None) is None
            # PR #347 R14 P1: control-aware empty-surface label.
            _has_controls = getattr(self._results, "covariate_residuals", None) is not None
            _empty_surface_label = "DID^{X,fd}_l" if _has_controls else "DID^{fd}_l"
            if _surface_empty:
                headline_name = "no scalar headline (empty per-horizon surface)"
                headline_reason = (
                    "The fitted estimator intentionally does not produce a "
                    "scalar overall ATT on this configuration "
                    "(``trends_linear=True`` with ``L_max >= 2``), and on "
                    f"this fit no cumulated level effects ``{_empty_surface_label}`` "
                    "survived estimation — the per-horizon surface is "
                    "empty. Re-fit with a larger ``L_max`` or with "
                    "``trends_linear=False`` if you need a reportable "
                    "estimand."
                )
            else:
                headline_name = "no scalar headline (see linear_trends_effects)"
                headline_reason = (
                    "The fitted estimator intentionally does not produce a "
                    "scalar overall ATT on this configuration "
                    "(``trends_linear=True`` with ``L_max >= 2``). Per-horizon "
                    "cumulated level effects are on "
                    "``results.linear_trends_effects[l]``."
                )
            headline = {
                "status": "no_scalar_by_design",
                "name": headline_name,
                "value": None,
                "se": None,
                "p_value": None,
                "conf_int": (None, None),
                "alpha": self._alpha,
                "is_significant": False,
                "sign": "none",
                "reason": headline_reason,
            }
        else:
            headline = self._extract_headline_metric()

        # Pull suggested next steps from the practitioner workflow.
        next_steps = self._collect_next_steps(sections)

        # Populate schema-level warnings for every section that ended in "error",
        # so users and agents do not have to scan each section dict to discover
        # that a diagnostic failed. Preserves provenance per the "no silent
        # failures" convention.
        top_warnings: List[str] = []
        for check in _CHECK_NAMES:
            section_key = "estimator_native" if check == "estimator_native" else check
            section = sections.get(section_key, {})
            if section.get("status") == "error":
                reason = section.get("reason") or "diagnostic raised an exception"
                top_warnings.append(f"{check}: {reason}")
            # Surface non-fatal warnings captured by delegated diagnostics
            # (e.g., HonestDiD's "base_period='varying' is not valid for
            # interpretation" on CallawaySantAnna, or the diag-covariance
            # fallback on bootstrap-fitted CS). These rode up on each
            # section's ``warnings`` field and must not be swallowed.
            section_warnings = section.get("warnings")
            if isinstance(section_warnings, (list, tuple)):
                for msg in section_warnings:
                    if msg is None:
                        continue
                    top_warnings.append(f"{check}: {msg}")
            # Some sections (e.g., sensitivity skipped for varying-base CS)
            # also surface methodology-critical context via ``reason`` even
            # though ``status != "error"``. We do not duplicate those here
            # — the section's own status/reason is the authoritative record.

        schema: Dict[str, Any] = {
            "schema_version": DIAGNOSTIC_REPORT_SCHEMA_VERSION,
            "estimator": type(self._results).__name__,
            "headline_metric": headline,
            "target_parameter": describe_target_parameter(self._results),
            "parallel_trends": sections["parallel_trends"],
            "pretrends_power": sections["pretrends_power"],
            "sensitivity": sections["sensitivity"],
            "placebo": sections["placebo"],
            "bacon": sections["bacon"],
            "design_effect": sections["design_effect"],
            "heterogeneity": sections["heterogeneity"],
            "epv": sections["epv"],
            "estimator_native_diagnostics": sections["estimator_native"],
            "skipped": {k: v for k, v in skipped.items()},
            "warnings": top_warnings,
            "overall_interpretation": "",
            "next_steps": next_steps,
        }
        interpretation = _render_overall_interpretation(schema, self._context_labels())
        schema["overall_interpretation"] = interpretation

        return DiagnosticReportResults(
            schema=schema,
            interpretation=interpretation,
            applicable_checks=tuple(sorted(applicable)),
            skipped_checks=skipped,
            warnings=tuple(top_warnings),
        )

    def _context_labels(self) -> Dict[str, str]:
        """Return plain-English labels used in prose rendering."""
        return {
            "outcome_label": self._outcome_label or "the outcome",
            "treatment_label": self._treatment_label or "the treatment",
        }

    def _collect_next_steps(self, sections: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pull and filter practitioner_next_steps, marking DR-covered steps complete.

        A step is marked complete only when its DR section actually ran
        (``status == "ran"``). The previous implementation marked steps
        complete based on membership in the applicability set, which
        overstated completion for checks that were applicable but skipped
        at runtime (e.g., Hausman on a DR / survey fit; sensitivity on
        varying-base CS).
        """
        try:
            from diff_diff.practitioner import practitioner_next_steps

            def _ran(key: str) -> bool:
                return sections.get(key, {}).get("status") == "ran"

            completed = []
            if _ran("parallel_trends"):
                completed.append("parallel_trends")
            if _ran("sensitivity"):
                completed.append("sensitivity")
            # SDiD / TROP route their sensitivity analogue through
            # ``estimator_native_diagnostics`` rather than HonestDiD. When
            # that native block ran, the Baker step-6 sensitivity check
            # has effectively been performed; treating the sensitivity
            # section as not-run would have ``next_steps`` redundantly
            # recommend a check the report already executed (round-19
            # CI review on PR #318). SyntheticControl is deliberately NOT
            # in this set: its significance procedure is the OPT-IN in-space
            # placebo (``in_space_placebo()``), which ``_scm_native`` reports
            # but does not run, so the work is not complete merely because the
            # native block ran — the practitioner ``placebo`` step must persist.
            result_name = type(self._results).__name__
            if result_name in {
                "SyntheticDiDResults",
                "TROPResults",
            } and _ran("estimator_native"):
                if "sensitivity" not in completed:
                    completed.append("sensitivity")
            # SCM's significance step is the opt-in in-space placebo. Mark it
            # complete only when the caller has run it AND it produced a VALID
            # reference set (>=1 placebo + a finite p-value) — an attempted-but-
            # infeasible run (J<2, treated-fit failure, or all donors failed) leaves
            # placebo_p_value NaN and is NOT complete, so the recommendation persists.
            if (
                result_name == "SyntheticControlResults"
                and int(getattr(self._results, "n_placebos", 0) or 0) > 0
                and np.isfinite(getattr(self._results, "placebo_p_value", np.nan))
            ):
                completed.append("placebo")
            if _ran("heterogeneity"):
                completed.append("heterogeneity")
            ns = practitioner_next_steps(
                self._results,
                completed_steps=completed,
                verbose=False,
            )
            return [
                {
                    "label": s.get("label"),
                    "why": s.get("why"),
                    "code": s.get("code"),
                    "priority": s.get("priority"),
                    "baker_step": s.get("baker_step"),
                }
                for s in ns.get("next_steps", [])[:5]
            ]
        except Exception:  # noqa: BLE001
            return []

    # -- Per-check runners --------------------------------------------------

    def _check_parallel_trends(self) -> Dict[str, Any]:
        """Run the parallel-trends check. Dispatches on PT method for this type."""
        if "parallel_trends" in self._precomputed:
            return self._format_precomputed_pt(self._precomputed["parallel_trends"])

        method = _PT_METHOD.get(type(self._results).__name__)
        if method == "two_x_two":
            return self._pt_two_x_two()
        if method == "event_study":
            return self._pt_event_study()
        if method == "hausman":
            return self._pt_hausman()
        if method == "synthetic_fit":
            return self._pt_synthetic_fit()
        if method == "scm_fit":
            return self._pt_scm_fit()
        if method == "factor":
            return self._pt_factor()
        return {
            "status": "not_applicable",
            "reason": f"No parallel-trends method registered for "
            f"{type(self._results).__name__}.",
        }

    def _pt_two_x_two(self) -> Dict[str, Any]:
        """Simple two-period PT check via ``utils.check_parallel_trends``."""
        from diff_diff.utils import check_parallel_trends

        if self._data is None or self._outcome is None or self._time is None:
            return {
                "status": "skipped",
                "reason": "Requires data=, outcome=, time=, and a treatment-group "
                "column; not supplied.",
            }
        treatment_group = self._treatment
        if treatment_group is None:
            return {
                "status": "skipped",
                "reason": "Requires treatment=<column name> identifying the "
                "treated-group indicator; not supplied.",
            }
        # Round-40 P1 CI review on PR #318: defense-in-depth. The
        # instance-level applicability gate should have already returned
        # a skip reason when ``results.survey_metadata`` is non-None and
        # no precomputed PT was supplied, but ``_pt_two_x_two`` is also
        # reachable directly from ``_check_parallel_trends`` if future
        # callers add method dispatch overrides. Guard at the runner
        # too to prevent ``utils.check_parallel_trends`` from emitting
        # an unweighted verdict for a weighted fit.
        if getattr(self._results, "survey_metadata", None) is not None:
            return {
                "status": "skipped",
                "reason": (
                    "Original fit used a survey design; the simple 2x2 "
                    "parallel-trends helper (``utils.check_parallel_trends``) "
                    "is unweighted and cannot faithfully diagnose a "
                    "survey-weighted DiDResults. Supply a survey-aware "
                    "pretest via ``precomputed={'parallel_trends': ...}`` "
                    "to opt in."
                ),
            }
        try:
            raw = check_parallel_trends(
                self._data,
                outcome=self._outcome,
                time=self._time,
                treatment_group=treatment_group,
                pre_periods=self._pre_periods,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"check_parallel_trends raised {type(exc).__name__}: {exc}",
            }
        p_value = _to_python_float(raw.get("p_value"))
        return {
            "status": "ran",
            "method": "slope_difference",
            "joint_p_value": p_value,
            "treated_trend": _to_python_float(raw.get("treated_trend")),
            "control_trend": _to_python_float(raw.get("control_trend")),
            "trend_difference": _to_python_float(raw.get("trend_difference")),
            "t_statistic": _to_python_float(raw.get("t_statistic")),
            "verdict": _pt_verdict(p_value),
        }

    def _pt_event_study(self) -> Dict[str, Any]:
        """Event-study joint Wald (or Bonferroni fallback) on pre-period coefficients.

        Works with either ``pre_period_effects`` (``MultiPeriodDiDResults`` style,
        dict of ``PeriodEffect`` objects) or ``event_study_effects`` (CS / SA /
        ImputationDiD style, dict of dicts with ``effect``/``se``/``p_value`` keys).
        """
        r = self._results
        pre_coefs, n_dropped_undefined = _collect_pre_period_coefs(r)
        # Round-33 P0 / Round-42 P1 CI review on PR #318: undefined-
        # inference rows must drive an explicit ``inconclusive`` PT
        # result rather than either (a) silently shrinking the
        # Bonferroni family on the remaining subset and publishing a
        # finite joint p-value (R33, mixed-partial case), or (b)
        # routing through the empty-coefs ``skipped`` path when every
        # pre-row was rejected (R42, all-undefined case). Both violate
        # the ``safe_inference`` contract: ``se <= 0`` / non-finite
        # effect or SE yields NaN downstream per ``utils.py`` line
        # 175, REGISTRY.md line 197. The inconclusive block preserves
        # the undefined-row count on the schema so BR's summary can
        # quote it and stakeholders see an explicit "PT could not be
        # assessed" warning rather than a silent PT-absent narrative.
        if n_dropped_undefined > 0:
            return {
                "status": "ran",
                "method": "inconclusive",
                "joint_p_value": None,
                "test_statistic": None,
                "df": len(pre_coefs),
                "n_pre_periods": len(pre_coefs),
                "n_dropped_undefined": n_dropped_undefined,
                "verdict": "inconclusive",
                "reason": (
                    f"{n_dropped_undefined} pre-period coefficient(s) "
                    "have undefined inference (non-finite effect / SE or "
                    "SE <= 0). Per the safe-inference contract "
                    "(``utils.py`` line 175, REGISTRY.md line 197), this "
                    "yields NaN downstream; the joint PT test is "
                    "inconclusive on this fit. Re-fit with a different "
                    "variance method (bootstrap / cluster) if the "
                    "affected rows are a small number of cohorts, or "
                    "investigate why the per-period SE collapsed."
                ),
            }
        if not pre_coefs:
            return {
                "status": "skipped",
                "reason": "No pre-period event-study coefficients available.",
            }
        interaction_indices = getattr(r, "interaction_indices", None)
        vcov = getattr(r, "vcov", None)

        # pre_coefs is a sorted list of (key, effect, se, p_value) tuples.
        per_period = [
            {
                "period": _to_python_scalar(k),
                "coef": _to_python_float(eff),
                "se": _to_python_float(se),
                "p_value": _to_python_float(p),
            }
            for (k, eff, se, p) in pre_coefs
        ]

        joint_p: Optional[float] = None
        test_statistic: Optional[float] = None
        df = len(pre_coefs)
        method = "bonferroni"
        # Joint-Wald pathway is taken only when EVERY pre-period key is present
        # in the relevant index mapping (required len == df guard below). This
        # protects against estimators whose event-study keys use a different
        # namespace than the vcov indexing: if any key is missing, we fall back
        # to Bonferroni rather than risk indexing into the wrong vcov rows.
        # The schema's ``method`` field exposes which path ran so agents and
        # tests can distinguish the two unambiguously.
        #
        # Two covariance sources are supported:
        #   1. ``interaction_indices`` + ``vcov`` — the MultiPeriodDiDResults
        #      convention, where ``vcov`` is the full regression covariance
        #      matrix and ``interaction_indices`` maps period labels to rows.
        #   2. ``event_study_vcov_index`` + ``event_study_vcov`` — the
        #      CallawaySantAnnaResults convention, where the event-study
        #      covariance is stored separately from the full regression vcov.
        vcov_for_wald: Optional[Any] = None
        idx_map_for_wald: Optional[Any] = None
        vcov_method_tag = "joint_wald"
        if vcov is not None and interaction_indices is not None:
            vcov_for_wald = vcov
            idx_map_for_wald = interaction_indices
        else:
            es_vcov = getattr(r, "event_study_vcov", None)
            es_vcov_index = getattr(r, "event_study_vcov_index", None)
            if es_vcov is not None and es_vcov_index is not None:
                vcov_for_wald = es_vcov
                # ``event_study_vcov_index`` is an ordered list of relative-time
                # keys; convert it into a dict mapping key -> position.
                try:
                    idx_map_for_wald = {k: i for i, k in enumerate(es_vcov_index)}
                    vcov_method_tag = "joint_wald_event_study"
                except TypeError:
                    idx_map_for_wald = None
        df_denom: Optional[float] = None
        if vcov_for_wald is not None and idx_map_for_wald is not None and df > 0:
            try:
                keys_in_vcov = [k for (k, _, _, _) in pre_coefs if k in idx_map_for_wald]
                if len(keys_in_vcov) == df:
                    idx = [idx_map_for_wald[k] for k in keys_in_vcov]
                    beta_map = {k: eff for (k, eff, _, _) in pre_coefs}
                    beta = np.array([beta_map[k] for k in keys_in_vcov], dtype=float)
                    v_sub = np.asarray(vcov_for_wald)[np.ix_(idx, idx)]
                    stat = float(beta @ np.linalg.solve(v_sub, beta))

                    # Round-27 P1 CI review on PR #318: survey-backed
                    # fits carry a finite ``df_survey`` on
                    # ``survey_metadata``; using the chi-square reference
                    # distribution on those produces overconfident
                    # p-values because it ignores the finite-sample
                    # correction the design-based SE already reflects.
                    # When a finite denominator df is available, compute
                    # ``F = W / k`` (numerator df = k pre-periods) against
                    # an F(k, df_survey) reference. Reserve the chi-square
                    # path for fits with no finite-df information.
                    sm = getattr(r, "survey_metadata", None)
                    df_survey_raw = getattr(sm, "df_survey", None) if sm is not None else None
                    df_survey: Optional[float] = None
                    if df_survey_raw is not None:
                        try:
                            df_survey_val = float(df_survey_raw)
                            if np.isfinite(df_survey_val) and df_survey_val > 0:
                                df_survey = df_survey_val
                        except (TypeError, ValueError):
                            df_survey = None

                    if df_survey is not None:
                        from scipy.stats import f as f_dist

                        f_stat = stat / df
                        joint_p = float(1.0 - f_dist.cdf(f_stat, dfn=df, dfd=df_survey))
                        test_statistic = stat
                        method = f"{vcov_method_tag}_survey"
                        df_denom = df_survey
                    else:
                        from scipy.stats import chi2

                        joint_p = float(1.0 - chi2.cdf(stat, df=df))
                        test_statistic = stat
                        method = vcov_method_tag
            except Exception:  # noqa: BLE001
                joint_p = None
                test_statistic = None
                method = "bonferroni"

        if joint_p is None:
            # Bonferroni fallback is only valid when EVERY retained pre-
            # period contributes a finite p-value. Otherwise we would
            # silently shrink the test family (e.g., replicate-weight
            # survey fits where ``safe_inference`` returns NaN p-values
            # for rows whose effective survey df collapsed — the row's
            # ``effect`` / ``se`` is still finite, so the ``se > 0``
            # collector filter lets it through, but a Bonferroni
            # computed on the remaining subset publishes a finite joint
            # p-value that BR lifts into "consistent with parallel
            # trends" prose). Round-34 P0 CI review on PR #318 flagged
            # that the round-33 guard only caught the ``se <= 0`` case
            # and missed this.
            #
            # Strategy: if any retained pre-period has non-finite
            # ``p_value``, emit an explicit inconclusive PT block with
            # a visible count/reason. Otherwise run Bonferroni on the
            # full family as documented in REPORTING.md.
            nan_p_count = sum(
                1
                for p in per_period
                if not (isinstance(p["p_value"], (int, float)) and np.isfinite(p["p_value"]))
            )
            if nan_p_count > 0:
                return {
                    "status": "ran",
                    "method": "inconclusive",
                    "joint_p_value": None,
                    "test_statistic": None,
                    "df": len(pre_coefs),
                    "n_pre_periods": len(pre_coefs),
                    "n_dropped_undefined": nan_p_count,
                    "per_period": per_period,
                    "verdict": "inconclusive",
                    "reason": (
                        f"{nan_p_count} retained pre-period coefficient(s) "
                        "have non-finite per-period p-value (undefined "
                        "inference per the ``safe_inference`` contract — "
                        "e.g., replicate-weight survey fits where effective "
                        "df collapsed). Bonferroni on the remaining subset "
                        "would silently shrink the test family; the joint "
                        "PT test is inconclusive on this fit. Inspect the "
                        "per_period block for the undefined rows."
                    ),
                }
            ps = [p["p_value"] for p in per_period]
            if ps:
                joint_p = min(1.0, min(ps) * len(ps))

        out = {
            "status": "ran",
            "method": method,
            "joint_p_value": joint_p,
            "test_statistic": test_statistic,
            "df": df,
            "n_pre_periods": df,
            "per_period": per_period,
            "verdict": _pt_verdict(joint_p),
        }
        # Expose the denominator df when the survey F-path was used so
        # BR / DR prose can flag the finite-sample correction rather than
        # silently presenting a chi-square-style result.
        if df_denom is not None:
            out["df_denom"] = df_denom
        return out

    def _check_pretrends_power(self) -> Dict[str, Any]:
        """Compute pre-trends power (MDV) via ``compute_pretrends_power``.

        Feeds the ``mdv_share_of_att`` ratio used by ``BusinessReport`` to select
        the power-aware phrasing tier for the ``no_detected_violation`` verdict.
        """
        if "pretrends_power" in self._precomputed:
            return self._format_precomputed_pretrends_power(self._precomputed["pretrends_power"])

        from diff_diff.pretrends import compute_pretrends_power

        try:
            pp = compute_pretrends_power(
                self._results,
                alpha=self._alpha,
                target_power=0.80,
                violation_type="linear",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"compute_pretrends_power raised " f"{type(exc).__name__}: {exc}",
            }

        # Build the schema section and compute the level-scale max-pre-
        # violation / |ATT| ratio for BR tier classification. Post-PR-B
        # Step 4 the linear `mdv` is in Roth's γ units (a slope on
        # relative time), so the level-scale comparable quantity is
        # `max_abs_pre_violation = mdv * max(|violation_weights|)` —
        # the largest pre-period level deviation under the MDV. Using
        # raw `mdv` here would mix slope and level scales on irregular
        # grids and mis-tier well_powered / moderately_powered /
        # underpowered.
        headline_metric = self._extract_headline_metric()
        att = headline_metric.get("value") if headline_metric else None
        mdv = _to_python_float(getattr(pp, "mdv", None))
        max_abs_pre_violation = _to_python_float(getattr(pp, "max_abs_pre_violation", mdv))
        ratio: Optional[float] = None
        if (
            max_abs_pre_violation is not None
            and att is not None
            and np.isfinite(att)
            and abs(att) > 0
            and np.isfinite(max_abs_pre_violation)
        ):
            ratio = max_abs_pre_violation / abs(att)

        # Prefer the provenance label `pretrends.py` records on the result
        # itself (PR-B: `PreTrendsPowerResults.covariance_source` captures
        # which extraction path was actually taken — full Σ_22 sub-block
        # vs diag fallback). Fall back to type-based inference for legacy
        # serialized results pre-PR-B that lack the field.
        cov_source = getattr(pp, "covariance_source", "unknown")
        if cov_source == "unknown":
            cov_source = self._infer_cov_source(self._results)
        tier = _apply_diag_fallback_downgrade(_power_tier(ratio), cov_source)
        return {
            "status": "ran",
            "method": "compute_pretrends_power",
            "violation_type": getattr(pp, "violation_type", "linear"),
            "alpha": _to_python_float(getattr(pp, "alpha", self._alpha)),
            "target_power": _to_python_float(getattr(pp, "target_power", 0.80)),
            "mdv": mdv,
            "max_abs_pre_violation": max_abs_pre_violation,
            "mdv_share_of_att": ratio,
            # Power is reported at ``violation_magnitude`` — the M that
            # the helper actually evaluated (defaults to the MDV when
            # the caller passed ``M=None``). Schema consumers should
            # read ``violation_magnitude`` alongside the power value.
            "violation_magnitude": _to_python_float(getattr(pp, "violation_magnitude", None)),
            "power_at_violation_magnitude": _to_python_float(getattr(pp, "power", None)),
            "n_pre_periods": int(getattr(pp, "n_pre_periods", 0) or 0),
            "tier": tier,
            "covariance_source": cov_source,
        }

    def _format_precomputed_pretrends_power(self, obj: Any) -> Dict[str, Any]:
        """Adapt a pre-computed ``PreTrendsPowerResults`` to the schema shape.

        Round-20 P1 CI review on PR #318: this path must mirror the
        covariance-source annotation and diagonal-fallback downgrade that
        ``_check_pretrends_power`` applies on the default path. Otherwise
        the same fit passed through ``precomputed={"pretrends_power": ...}``
        can be labeled ``well_powered`` while the default path reports
        ``moderately_powered`` (per REPORTING.md's conservative deviation
        for CS / SA / ImputationDiD event-study fits with full
        ``event_study_vcov`` available but unused). Resolve the source
        fit via ``obj.original_results`` first (which ``compute_pretrends_power``
        populates at construction time), falling back to ``self._results``.
        """
        mdv = _to_python_float(getattr(obj, "mdv", None))
        # PR-B Step 4: use level-scale max_abs_pre_violation rather than
        # raw γ-unit mdv to tier (see ``_check_pretrends_power`` for the
        # rationale). Legacy precomputed PreTrendsPowerResults objects
        # without the property fall back to raw ``mdv``.
        max_abs_pre_violation = _to_python_float(getattr(obj, "max_abs_pre_violation", mdv))
        hm = self._extract_headline_metric()
        att = hm.get("value") if hm else None
        ratio: Optional[float] = None
        if (
            max_abs_pre_violation is not None
            and att is not None
            and np.isfinite(att)
            and abs(att) > 0
            and np.isfinite(max_abs_pre_violation)
        ):
            ratio = max_abs_pre_violation / abs(att)
        source_fit = getattr(obj, "original_results", None) or self._results
        # PR-B: prefer the provenance label `pretrends.py` records on the
        # precomputed result; fall back to type-based inference only for
        # legacy serialized results that lack the field.
        cov_source = getattr(obj, "covariance_source", "unknown")
        if cov_source == "unknown":
            cov_source = self._infer_cov_source(source_fit)
        tier = _apply_diag_fallback_downgrade(_power_tier(ratio), cov_source)
        return {
            "status": "ran",
            "method": "precomputed",
            "violation_type": getattr(obj, "violation_type", "linear"),
            "alpha": _to_python_float(getattr(obj, "alpha", self._alpha)),
            "target_power": _to_python_float(getattr(obj, "target_power", 0.80)),
            "mdv": mdv,
            "max_abs_pre_violation": max_abs_pre_violation,
            "mdv_share_of_att": ratio,
            "violation_magnitude": _to_python_float(getattr(obj, "violation_magnitude", None)),
            "power_at_violation_magnitude": _to_python_float(getattr(obj, "power", None)),
            "n_pre_periods": int(getattr(obj, "n_pre_periods", 0) or 0),
            "tier": tier,
            "covariance_source": cov_source,
            "precomputed": True,
        }

    @staticmethod
    def _infer_cov_source(source_fit: Any) -> str:
        """Classify whether ``compute_pretrends_power`` had access to the
        full pre-period covariance on ``source_fit``.

        Backwards-compatibility helper for legacy ``PreTrendsPowerResults``
        objects produced before PR-B (which records the actual extraction
        path on ``PreTrendsPowerResults.covariance_source`` at fit time).
        New fits read provenance directly off the result object; this
        fallback is only invoked when that field is missing or set to
        ``"unknown"`` (legacy-ambiguous).

        Classification rules:

        - ``"full_pre_period_vcov"`` — basic ``DiDResults`` and other
          non-event-study, non-MPD result types that historically expose
          the full pre-period covariance. ``MultiPeriodDiDResults`` is
          handled by an explicit branch below because its
          ``pretrends.py`` MPD branch only takes the full sub-block path
          when ``interaction_indices`` is populated, otherwise falling
          through to ``diag(ses**2)``.
        - ``"diag_fallback_available_full_vcov_unused"`` — event-study
          result types with populated ``event_study_vcov``. Under PR-B,
          new fits route through the full sub-block, but a legacy
          ``PreTrendsPowerResults`` lacking ``covariance_source`` may
          have been computed from ``diag(ses**2)`` even though the full
          matrix was attached on the source fit (PR-A behavior). Without
          the persisted provenance label we cannot distinguish the two,
          and the conservative default is to apply the PR-A downgrade.
          New PR-B fits set ``covariance_source`` directly and bypass
          this fallback entirely.
        - ``"diag_fallback"`` — event-study result types with
          ``event_study_vcov is None`` (bootstrap or replicate-weight
          CS / SA fits, plus ImputationDiD / Stacked / EfficientDiD /
          TwoStageDiD / etc. which don't yet expose ``event_study_vcov``);
          OR ``MultiPeriodDiDResults`` without ``interaction_indices``
          (genuine diag-only path inside ``pretrends.py:_extract_pre_period_params``,
          no "available but unused" concern, so no downgrade applies).
        """
        is_event_study_type = type(source_fit).__name__ in {
            "CallawaySantAnnaResults",
            "SunAbrahamResults",
            "ImputationDiDResults",
            "StackedDiDResults",
            "StaggeredTripleDiffResults",
            "WooldridgeDiDResults",
            "ChaisemartinDHaultfoeuilleResults",
            "EfficientDiDResults",
            "TwoStageDiDResults",
        }
        has_full_es_vcov = (
            getattr(source_fit, "event_study_vcov", None) is not None
            and getattr(source_fit, "event_study_vcov_index", None) is not None
        )
        if is_event_study_type and has_full_es_vcov:
            # Legacy-ambiguous: we don't know whether this serialized
            # result was computed pre- or post-PR-B; conservatively
            # downgrade. New PR-B fits will set covariance_source
            # explicitly on the result and never reach this branch.
            return "diag_fallback_available_full_vcov_unused"
        if is_event_study_type:
            return "diag_fallback"
        # Non-event-study path. MultiPeriodDiDResults takes the full
        # ``vcov[ix_]`` sub-block only when ``interaction_indices`` is
        # populated (pretrends.py MPD branch); otherwise it falls
        # through to ``diag(ses**2)`` and ships the diag-fallback path
        # — which is a normal (not "available but unused") fallback,
        # so no conservative downgrade applies. Legacy MPD result
        # objects without ``interaction_indices`` should be reported as
        # ``diag_fallback`` rather than overclaiming full-Σ_22.
        if type(source_fit).__name__ == "MultiPeriodDiDResults":
            mpd_has_full_vcov = (
                getattr(source_fit, "vcov", None) is not None
                and getattr(source_fit, "interaction_indices", None) is not None
            )
            return "full_pre_period_vcov" if mpd_has_full_vcov else "diag_fallback"
        # Other non-event-study types (basic DiDResults, TWFE, etc.)
        # historically expose the full covariance.
        return "full_pre_period_vcov"

    def _check_sensitivity(self) -> Dict[str, Any]:
        """Run HonestDiD over the M grid. Uses ``SensitivityResults.breakdown_M``.

        The standard path calls ``HonestDiD(method=..., M_grid=...).sensitivity_analysis()``.
        SDiD and TROP route to estimator-native sensitivity in
        ``estimator_native_diagnostics`` and emit a pointer here.
        """
        if "sensitivity" in self._precomputed:
            return self._format_precomputed_sensitivity(self._precomputed["sensitivity"])

        name = type(self._results).__name__
        if name == "SyntheticDiDResults":
            return {
                "status": "skipped",
                "reason": (
                    "SyntheticDiD uses native sensitivity analogues "
                    "(``in_time_placebo``, ``sensitivity_to_zeta_omega``) "
                    "rather than HonestDiD; see "
                    "``estimator_native_diagnostics``."
                ),
                "method": "estimator_native",
            }
        if name == "TROPResults":
            return {
                "status": "skipped",
                "reason": (
                    "TROP identification is factor-model-based; HonestDiD "
                    "bounds do not apply. Use the factor-model fit metrics "
                    "(effective rank, LOOCV score, selected lambdas) in "
                    "``estimator_native_diagnostics`` as the analogue."
                ),
                "method": "estimator_native",
            }

        # Varying-base CS gate: handled at ``_instance_skip_reason``, so
        # this code path is not reached for a varying-base CS fit unless
        # the user passed ``precomputed={'sensitivity': ...}`` (handled
        # above). Kept here as a comment anchor; see _instance_skip_reason.

        import warnings as _warnings

        try:
            from typing import cast

            from diff_diff.honest_did import HonestDiD

            # Capture any non-fatal UserWarnings HonestDiD emits (bootstrap
            # diag-covariance fallback on CS, library-extension note on
            # dCDH, dropped non-consecutive horizons, etc.) so BR/DR do not
            # silently narrate sensitivity as clean when the helper
            # flagged caveats. The try/except below still handles fatal
            # errors; captured warnings ride on the returned dict.
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                # The sensitivity_method string is validated at runtime by
                # HonestDiD; the Literal annotation is for static typing only.
                honest = HonestDiD(
                    method=cast(Any, self._sensitivity_method),
                    alpha=self._alpha,
                )
                sens = honest.sensitivity_analysis(
                    self._results,
                    M_grid=list(self._sensitivity_M_grid),
                )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "method": self._sensitivity_method,
                "reason": f"HonestDiD.sensitivity_analysis raised " f"{type(exc).__name__}: {exc}",
            }

        captured = [str(w.message) for w in caught if issubclass(w.category, Warning)]
        formatted = self._format_sensitivity_results(sens)
        if captured:
            formatted["warnings"] = captured
        return formatted

    def _format_sensitivity_results(self, sens: Any) -> Dict[str, Any]:
        grid = []
        raw_M = getattr(sens, "M_values", None)
        raw_cis = getattr(sens, "robust_cis", None)
        raw_bounds = getattr(sens, "bounds", None)
        M_values: List[Any] = list(raw_M) if raw_M is not None else []
        cis: List[Any] = list(raw_cis) if raw_cis is not None else []
        bounds: List[Any] = list(raw_bounds) if raw_bounds is not None else []
        for i, M in enumerate(M_values):
            ci = cis[i] if i < len(cis) else (None, None)
            bd = bounds[i] if i < len(bounds) else (None, None)
            lo = _to_python_float(ci[0])
            hi = _to_python_float(ci[1])
            robust_to_zero = lo is not None and hi is not None and (lo > 0 or hi < 0)
            grid.append(
                {
                    "M": _to_python_float(M),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "bound_lower": _to_python_float(bd[0]),
                    "bound_upper": _to_python_float(bd[1]),
                    "robust_to_zero": robust_to_zero,
                }
            )
        bkd = _to_python_float(getattr(sens, "breakdown_M", None))
        if bkd is None:
            conclusion = "robust_over_grid"
        elif bkd >= 1.0:
            conclusion = f"robust_to_M_{bkd:.2f}"
        else:
            conclusion = "fragile"
        return {
            "status": "ran",
            "method": getattr(sens, "method", self._sensitivity_method),
            "grid": grid,
            "breakdown_M": bkd,
            "original_estimate": _to_python_float(getattr(sens, "original_estimate", None)),
            "original_se": _to_python_float(getattr(sens, "original_se", None)),
            "conclusion": conclusion,
        }

    def _format_precomputed_sensitivity(self, obj: Any) -> Dict[str, Any]:
        """Accept either ``SensitivityResults`` (grid) or ``HonestDiDResults`` (single M).

        The single-M branch preserves ``original_estimate`` and
        ``original_se`` for parity with the grid branch — both
        ``SensitivityResults`` and ``HonestDiDResults`` carry these fields,
        and downstream tooling that reads the schema should see a
        consistent shape regardless of which object was passed. (The
        grid path surfaces them via ``_format_sensitivity_results``.)
        """
        if hasattr(obj, "M_values") and hasattr(obj, "breakdown_M"):
            formatted = self._format_sensitivity_results(obj)
            formatted["precomputed"] = True
            return formatted
        # Single-M HonestDiDResults: adapt with no breakdown_M.
        ci_lb = _to_python_float(getattr(obj, "ci_lb", None))
        ci_ub = _to_python_float(getattr(obj, "ci_ub", None))
        return {
            "status": "ran",
            "method": getattr(obj, "method", self._sensitivity_method),
            "grid": [
                {
                    "M": _to_python_float(getattr(obj, "M", None)),
                    "ci_lower": ci_lb,
                    "ci_upper": ci_ub,
                    "bound_lower": _to_python_float(getattr(obj, "lb", None)),
                    "bound_upper": _to_python_float(getattr(obj, "ub", None)),
                    "robust_to_zero": (
                        ci_lb is not None and ci_ub is not None and (ci_lb > 0 or ci_ub < 0)
                    ),
                }
            ],
            "breakdown_M": None,
            "original_estimate": _to_python_float(getattr(obj, "original_estimate", None)),
            "original_se": _to_python_float(getattr(obj, "original_se", None)),
            "conclusion": "single_M_precomputed",
            "precomputed": True,
        }

    def _check_bacon(self) -> Dict[str, Any]:
        """Surface Bacon decomposition: read-out when applicable, else skip.

        If ``results`` is itself a ``BaconDecompositionResults``, read fields.
        If ``data`` + ``first_treat`` are supplied, call ``bacon_decompose``.
        Otherwise, skip with a helpful reason.
        """
        if "bacon" in self._precomputed:
            return self._format_bacon(self._precomputed["bacon"])

        r = self._results
        name = type(r).__name__
        if name == "BaconDecompositionResults":
            return self._format_bacon(r)

        data = self._data
        outcome = self._outcome
        unit = self._unit
        time = self._time
        first_treat = self._first_treat
        if data is None or outcome is None or unit is None or time is None or first_treat is None:
            return {
                "status": "skipped",
                "reason": "Bacon decomposition requires data + outcome + unit + time "
                "+ first_treat on DiagnosticReport; not all supplied.",
            }
        # Round-40 P1 CI review on PR #318: defense-in-depth. The
        # instance-level applicability gate should have already returned
        # a skip when the result carries ``survey_metadata`` but no
        # ``survey_design`` is available to thread through. Guard at
        # the runner too in case a future caller bypasses the gate.
        if getattr(r, "survey_metadata", None) is not None and self._survey_design is None:
            return {
                "status": "skipped",
                "reason": (
                    "Original fit used a survey design; Goodman-Bacon "
                    "replay under defaults would produce an unweighted "
                    "decomposition for a different design than the "
                    "weighted estimate. Pass ``survey_design=<SurveyDesign>`` "
                    "on DiagnosticReport / BusinessReport, or supply "
                    "``precomputed={'bacon': ...}`` with a survey-aware "
                    "decomposition."
                ),
            }

        try:
            from diff_diff.bacon import bacon_decompose

            bacon = bacon_decompose(
                data,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                weights="exact",  # paper-faithful Eqs. 7-9 / 10e-g
                survey_design=self._survey_design,
            )
        except ValueError as exc:
            # PR #454 R1 P1: the exact-mode survey path enforces
            # within-unit constancy of survey columns via
            # ``_validate_unit_constant_survey``. Survey-backed reports on
            # panels with time-varying within-unit weights / strata / PSU
            # / FPC fail that check. The library still supports those
            # panels via explicit ``bacon_decompose(weights="approximate", ...)``,
            # so emit a structured skip pointing users at the
            # ``precomputed={'bacon': ...}`` escape hatch rather than an
            # opaque ``error`` block.
            if "varies within units" in str(exc):
                return {
                    "status": "skipped",
                    "reason": (
                        "Survey design has within-unit-varying columns "
                        "(weights / strata / PSU / FPC), which the "
                        'paper-faithful ``weights="exact"`` Bacon path '
                        "rejects. Run ``bacon_decompose(data, ..., "
                        'weights="approximate", survey_design=design)`` '
                        "yourself and pass via "
                        "``DiagnosticReport(..., precomputed={'bacon': result})`` "
                        f"to populate this section. Validator detail: {exc}"
                    ),
                }
            return {
                "status": "error",
                "reason": f"bacon_decompose raised {type(exc).__name__}: {exc}",
            }
        except NotImplementedError as exc:
            # PR #454 R4 P3: ``BaconDecomposition.fit()`` raises
            # ``NotImplementedError`` for replicate-weight survey designs
            # (bacon.py rejects them because Bacon is a diagnostic that
            # does not compute replicate-based variance). Match the
            # within-unit-varying skip pattern above: emit a structured
            # skip naming the ``precomputed={'bacon': ...}`` escape hatch
            # so survey-backed reports with replicate-weight designs
            # produce an actionable skip rather than an opaque
            # ``status="error"`` block.
            return {
                "status": "skipped",
                "reason": (
                    "Survey design uses replicate weights, which the "
                    "Bacon decomposition does not support (bacon is a "
                    "diagnostic and does not compute replicate-based "
                    "variance). To populate this section, run "
                    "``bacon_decompose(data, ..., "
                    "survey_design=SurveyDesign(weights=..., strata=..., "
                    "psu=..., fpc=...))`` with a TSL-based design and "
                    "pass via "
                    "``DiagnosticReport(..., precomputed={'bacon': result})``. "
                    f"Rejection detail: {exc}"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"bacon_decompose raised {type(exc).__name__}: {exc}",
            }
        return self._format_bacon(bacon)

    def _format_bacon(self, bacon: Any) -> Dict[str, Any]:
        treated_vs_never = _to_python_float(getattr(bacon, "total_weight_treated_vs_never", None))
        earlier_vs_later = _to_python_float(getattr(bacon, "total_weight_earlier_vs_later", None))
        later_vs_earlier = _to_python_float(getattr(bacon, "total_weight_later_vs_earlier", None))
        twfe = _to_python_float(getattr(bacon, "twfe_estimate", None))
        forbidden = later_vs_earlier if later_vs_earlier is not None else 0.0
        if forbidden > 0.10:
            verdict = "materially_contaminated"
        elif forbidden > 0.01:
            verdict = "minor_forbidden_weight"
        else:
            verdict = "clean"
        return {
            "status": "ran",
            "twfe_estimate": twfe,
            "weight_by_type": {
                "treated_vs_never": treated_vs_never,
                "earlier_vs_later": earlier_vs_later,
                "later_vs_earlier": later_vs_earlier,
            },
            "forbidden_weight": later_vs_earlier,
            "verdict": verdict,
            "n_timing_groups": _to_python_scalar(getattr(bacon, "n_timing_groups", None)),
        }

    def _check_design_effect(self) -> Dict[str, Any]:
        """Read survey design-effect from ``results.survey_metadata``.

        Emits a plain-English ``band_label`` alongside the numeric
        fields so downstream prose can classify the correction without
        re-deriving the threshold rule. REPORTING.md describes the
        band breakpoints (round-32 P2 CI review on PR #318 flagged
        that the docs advertised the label but the implementation was
        only emitting the numeric fields plus ``is_trivial``).

        Bands (per REPORTING.md):
          * ``deff < 0.95`` -> ``"improves_precision"`` (effective N
            is LARGER than nominal N — a precision-improving design;
            round-35 split this out from the old ``trivial`` bucket);
          * ``0.95 <= deff < 1.05`` -> ``"trivial"`` (effectively no
            effect on inference);
          * ``1.05 <= deff < 2`` -> ``"slightly_reduces"``;
          * ``2 <= deff < 5`` -> ``"materially_reduces"``;
          * ``deff >= 5`` -> ``"large_warning"``.
        ``None`` deff (or non-finite) -> ``band_label=None`` (no
        classification).
        """
        sm = getattr(self._results, "survey_metadata", None)
        if sm is None:
            return {
                "status": "skipped",
                "reason": "No survey_metadata attached to results.",
            }
        deff = _to_python_float(getattr(sm, "design_effect", None))
        eff_n = _to_python_float(getattr(sm, "effective_n", None))
        # Round-35 P2 CI review on PR #318: ``is_trivial`` used to be
        # ``0.95 <= deff <= 1.05`` while ``band_label`` treated
        # anything ``< 1.05`` as trivial. On a precision-improving
        # design (``deff < 0.95``) BR's summary keyed off
        # ``not is_trivial`` and narrated "Survey design reduces
        # effective sample size", which is directionally wrong — the
        # effective N is LARGER than the nominal N. Split the band
        # into a dedicated ``improves_precision`` label for
        # ``deff < 0.95`` and keep ``is_trivial`` restricted to the
        # tight "effectively no effect" window so the schema
        # carries the precision-improving signal explicitly.
        #
        # Round-43 P2 CI review on PR #318: the ``is_trivial`` upper
        # bound used ``<= 1.05`` (closed interval) but REPORTING.md
        # defines the ``trivial`` band as ``0.95 <= deff < 1.05``
        # (half-open) and ``slightly_reduces`` as ``1.05 <= deff < 2``.
        # At exactly ``deff == 1.05`` the schema emitted
        # ``band_label="slightly_reduces"`` while also setting
        # ``is_trivial=True``, suppressing the non-trivial prose that
        # the documented threshold says should fire. Align the
        # ``is_trivial`` bound with the band-label bound.
        is_trivial = deff is not None and 0.95 <= deff < 1.05
        if deff is None or not np.isfinite(deff):
            band_label: Optional[str] = None
        elif deff < 0.95:
            band_label = "improves_precision"
        elif deff < 1.05:
            band_label = "trivial"
        elif deff < 2.0:
            band_label = "slightly_reduces"
        elif deff < 5.0:
            band_label = "materially_reduces"
        else:
            band_label = "large_warning"
        return {
            "status": "ran",
            "deff": deff,
            "effective_n": eff_n,
            "weight_type": getattr(sm, "weight_type", None),
            "n_strata": _to_python_scalar(getattr(sm, "n_strata", None)),
            "n_psu": _to_python_scalar(getattr(sm, "n_psu", None)),
            "df_survey": _to_python_scalar(getattr(sm, "df_survey", None)),
            "replicate_method": getattr(sm, "replicate_method", None),
            "is_trivial": is_trivial,
            "band_label": band_label,
        }

    def _check_heterogeneity(self) -> Dict[str, Any]:
        """Compute effect-stability metrics (CV, range, sign consistency)."""
        effects = self._collect_effect_scalars()
        if not effects:
            return {
                "status": "skipped",
                "reason": "No group / event-study / period effects available.",
            }
        vals = np.array(effects, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {
                "status": "skipped",
                "reason": "All effect values are non-finite.",
            }
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        cv = sd / abs(mean) if abs(mean) > 0.1 * sd and abs(mean) > 0 else None
        sign_consistent = bool(np.all(finite >= 0) or np.all(finite <= 0))
        return {
            "status": "ran",
            "source": self._heterogeneity_source(),
            "n_effects": int(finite.size),
            "min": mn,
            "max": mx,
            "mean": mean,
            "sd": sd,
            "range": mx - mn,
            "cv": cv,
            "sign_consistent": sign_consistent,
        }

    def _check_epv(self) -> Dict[str, Any]:
        """Read EPV diagnostics from ``results.epv_diagnostics``.

        The diff-diff convention (see ``diff_diff/staggered.py`` around the
        low-EPV summary warning) is that ``epv_diagnostics`` is a dict keyed
        by cell identifier (e.g. ``(g, t)`` for staggered) whose values are
        per-cell dicts with ``is_low`` (bool) and ``epv`` (float). The
        threshold lives on ``results.epv_threshold`` (default 10) rather
        than being hardcoded.
        """
        r = self._results
        epv = getattr(r, "epv_diagnostics", None)
        if epv is None:
            return {
                "status": "skipped",
                "reason": "Estimator did not produce results.epv_diagnostics for this fit.",
            }
        threshold = _to_python_float(getattr(r, "epv_threshold", 10)) or 10.0

        if isinstance(epv, dict):
            low_cells = [k for k, v in epv.items() if isinstance(v, dict) and v.get("is_low")]
            epv_floats: List[float] = []
            for v in epv.values():
                if not isinstance(v, dict):
                    continue
                raw = v.get("epv")
                if raw is None:
                    continue
                converted = _to_python_float(raw)
                if converted is not None:
                    epv_floats.append(converted)
            min_epv: Optional[float] = min(epv_floats) if epv_floats else None
            return {
                "status": "ran",
                "threshold": threshold,
                "n_cells_low": len(low_cells),
                "n_cells_total": len(epv),
                "min_epv": min_epv,
                "affected_cohorts": [_to_python_scalar(c) for c in low_cells],
            }

        # Legacy object-shaped fallback (not currently emitted by the library
        # but kept so custom subclasses that mirror the old shape still work).
        low_cells_attr = getattr(epv, "low_epv_cells", None) or []
        return {
            "status": "ran",
            "threshold": threshold,
            "n_cells_low": int(len(low_cells_attr)),
            "n_cells_total": _to_python_scalar(getattr(epv, "n_cells_total", None)),
            "min_epv": _to_python_float(getattr(epv, "min_epv", None)),
            "affected_cohorts": [_to_python_scalar(c) for c in low_cells_attr],
        }

    def _check_estimator_native(self) -> Dict[str, Any]:
        """SDiD / TROP native validation surfaces.

        SDiD: ``pre_treatment_fit`` (weighted-PT analogue), weight
        concentration (``get_weight_concentration``), ``in_time_placebo``
        (placebo-timing sweep), and ``sensitivity_to_zeta_omega``
        (regularization sensitivity).

        TROP: factor-model fit metrics (``effective_rank``, ``loocv_score``,
        selected ``lambda_*``).

        SyntheticControl: pre-treatment fit (``pre_rmspe``), donor-weight
        concentration, and — each surfaced only when already computed — the
        in-space placebo permutation p-value (``in_space_placebo``), the ADH-2015
        leave-one-out donor robustness (``leave_one_out``), and the in-time
        backdating placebo (``in_time_placebo``).
        """
        r = self._results
        name = type(r).__name__
        if name == "SyntheticDiDResults":
            return self._sdid_native(r)
        if name == "TROPResults":
            return self._trop_native(r)
        if name == "SyntheticControlResults":
            return self._scm_native(r)
        return {
            "status": "not_applicable",
            "reason": f"{name} does not expose native validation methods.",
        }

    def _sdid_native(self, r: Any) -> Dict[str, Any]:
        """Populate SDiD-native diagnostics section."""
        out: Dict[str, Any] = {"status": "ran", "estimator": "SyntheticDiD"}
        out["pre_treatment_fit"] = _to_python_float(getattr(r, "pre_treatment_fit", None))
        # Weight concentration via the public method on SyntheticDiDResults.
        try:
            wc = r.get_weight_concentration(top_k=5)
            out["weight_concentration"] = {
                "effective_n": _to_python_float(wc.get("effective_n")),
                "herfindahl": _to_python_float(wc.get("herfindahl")),
                "top_k": _to_python_scalar(wc.get("top_k")),
                "top_k_share": _to_python_float(wc.get("top_k_share")),
            }
        except Exception as exc:  # noqa: BLE001
            out["weight_concentration"] = {
                "status": "error",
                "reason": f"get_weight_concentration raised " f"{type(exc).__name__}: {exc}",
            }
        # In-time placebo — runs only when the fit snapshot is available.
        try:
            placebo_df = r.in_time_placebo()
            out["in_time_placebo"] = {
                "n_placebos": int(len(placebo_df)),
                "max_abs_effect": _to_python_float(
                    placebo_df["att"].abs().max() if len(placebo_df) > 0 else None
                ),
                "mean_abs_effect": _to_python_float(
                    placebo_df["att"].abs().mean() if len(placebo_df) > 0 else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            out["in_time_placebo"] = {
                "status": "skipped",
                "reason": f"in_time_placebo unavailable: " f"{type(exc).__name__}: {exc}",
            }
        # Zeta-omega sensitivity.
        try:
            zeta_df = r.sensitivity_to_zeta_omega()
            atts = zeta_df["att"].astype(float).tolist() if len(zeta_df) > 0 else []
            out["zeta_sensitivity"] = {
                "grid": [
                    {
                        "multiplier": _to_python_float(row.get("multiplier")),
                        "att": _to_python_float(row.get("att")),
                        "pre_fit_rmse": _to_python_float(row.get("pre_fit_rmse")),
                        "effective_n": _to_python_float(row.get("effective_n")),
                    }
                    for row in zeta_df.to_dict(orient="records")
                ],
                "att_range": ([min(atts), max(atts)] if atts else None),
            }
        except Exception as exc:  # noqa: BLE001
            out["zeta_sensitivity"] = {
                "status": "skipped",
                "reason": f"sensitivity_to_zeta_omega unavailable: " f"{type(exc).__name__}: {exc}",
            }
        return out

    def _trop_native(self, r: Any) -> Dict[str, Any]:
        """Populate TROP-native factor-model diagnostics section."""
        return {
            "status": "ran",
            "estimator": "TROP",
            "factor_model": {
                "effective_rank": _to_python_float(getattr(r, "effective_rank", None)),
                "loocv_score": _to_python_float(getattr(r, "loocv_score", None)),
                "lambda_time": _to_python_float(getattr(r, "lambda_time", None)),
                "lambda_unit": _to_python_float(getattr(r, "lambda_unit", None)),
                "lambda_nn": _to_python_float(getattr(r, "lambda_nn", None)),
                "n_pre_periods": _to_python_scalar(getattr(r, "n_pre_periods", None)),
                "n_post_periods": _to_python_scalar(getattr(r, "n_post_periods", None)),
            },
        }

    def _scm_native(self, r: Any) -> Dict[str, Any]:
        """Populate classic-SCM-native diagnostics section.

        Always surfaces the pre-treatment fit (``pre_rmspe``) and donor-weight
        concentration. The in-space placebo permutation inference (ADH 2010 §2.4)
        is reported only when the user has ALREADY run ``in_space_placebo()`` —
        DR never triggers it implicitly, because it refits one synthetic control
        per donor (potentially many nested V searches) and the placebo layer is
        opt-in by design. (This differs from SDiD's cheaper in-time-placebo sweep,
        which ``_sdid_native`` runs inline.) The ADH-2015 §4 diagnostics —
        leave-one-out donor robustness (``leave_one_out()``) and the in-time
        (backdating) placebo (``in_time_placebo()``) — are surfaced the same way:
        opt-in, reported only once the user has run them (each refits the synthetic
        control one or more times), else a ``status="not_run"`` stub.
        """
        out: Dict[str, Any] = {"status": "ran", "estimator": "SyntheticControl"}
        out["pre_rmspe"] = _to_python_float(getattr(r, "pre_rmspe", None))
        out["n_donors"] = _to_python_scalar(getattr(r, "n_donors", None))
        out["v_method"] = getattr(r, "v_method", None)
        # Donor-weight concentration (Herfindahl + top weight): SCM places all mass
        # on the donor simplex, so concentration is the natural "how few donors
        # drive the synthetic" diagnostic.
        try:
            weights = [float(w) for w in r.donor_weights.values()]
            out["weight_concentration"] = {
                "herfindahl": _to_python_float(sum(w * w for w in weights)),
                "top_weight": _to_python_float(max(weights) if weights else None),
                "n_donors_with_weight": len(weights),
            }
        except Exception as exc:  # noqa: BLE001
            out["weight_concentration"] = {
                "status": "error",
                "reason": f"donor_weights unavailable: {type(exc).__name__}: {exc}",
            }
        # In-space placebo: surface only if already computed (opt-in; see docstring).
        if getattr(r, "_placebo_df", None) is not None:
            n_placebos = int(getattr(r, "n_placebos", 0) or 0)
            placebo_p = getattr(r, "placebo_p_value", np.nan)
            block = {
                "placebo_p_value": _to_python_float(placebo_p),
                "rmspe_ratio": _to_python_float(getattr(r, "rmspe_ratio", None)),
                "n_placebos": _to_python_scalar(n_placebos),
                "n_failed": _to_python_scalar(getattr(r, "n_failed", None)),
                "n_infeasible": _to_python_scalar(getattr(r, "n_infeasible", None)),
            }
            # Distinguish a valid run from an attempted-but-infeasible one so BR/DR
            # consumers see an explicit status/reason rather than a bare NaN p-value.
            # The SPECIFIC cause comes from the results' recorded ``_placebo_status``
            # (n_placebos/n_failed alone cannot tell a non-converged treated fit
            # apart from too-few-donors).
            if n_placebos > 0 and np.isfinite(placebo_p):
                block["status"] = "ran"
            else:
                placebo_status = getattr(r, "_placebo_status", None)
                _reasons = {
                    "treated_fit_nonconverged": (
                        "in_space_placebo() was run but the treated unit's own SCM "
                        "fit did not converge at fit time, so its RMSPE ratio is not "
                        "a valid optimum to rank against placebos; placebo_p_value "
                        "is NaN."
                    ),
                    "too_few_donors": (
                        "in_space_placebo() was run but fewer than 2 donors are "
                        "available (each placebo is fit against the other donors); "
                        "placebo_p_value is NaN."
                    ),
                    "all_placebos_failed": (
                        "in_space_placebo() was run but every donor refit failed to "
                        "converge, so no placebo entered the reference set; "
                        "placebo_p_value is NaN. Raise n_starts or loosen the "
                        "optimizer tolerances."
                    ),
                    "all_placebos_infeasible": (
                        "in_space_placebo() was run but every donor refit was "
                        "structurally infeasible under v_method='cv' (the "
                        "pseudo-treated donor pool is indistinguishable in a "
                        "re-aggregated CV window), so no placebo entered the reference "
                        "set; placebo_p_value is NaN. Adjust the predictors, v_cv_t0, "
                        "or the donor pool."
                    ),
                    "all_placebos_unusable": (
                        "in_space_placebo() was run but no donor refit was usable: "
                        "some failed to converge AND some were structurally infeasible "
                        "(see n_failed / n_infeasible); placebo_p_value is NaN."
                    ),
                }
                block["status"] = "infeasible"
                # Machine-readable code distinguishing a solver convergence failure
                # ("all_placebos_failed") from structural infeasibility
                # ("all_placebos_infeasible" / "too_few_donors") or a mix
                # ("all_placebos_unusable"), without parsing `reason`.
                block["reason_code"] = placebo_status
                block["reason"] = _reasons.get(
                    placebo_status,
                    "in_space_placebo() was run but produced no valid reference set "
                    "(fewer than 2 donors, a non-converged treated fit, or all donor "
                    "refits failed / were structurally infeasible); placebo_p_value is "
                    "NaN.",
                )
            out["in_space_placebo"] = block
        else:
            out["in_space_placebo"] = {
                "status": "not_run",
                "reason": (
                    "Call results.in_space_placebo() to run in-space placebo "
                    "permutation inference (opt-in; refits one synthetic control "
                    "per donor)."
                ),
            }

        # Leave-one-out donor robustness (ADH 2015 §4): opt-in, surfaced once run.
        if getattr(r, "_loo_df", None) is not None:
            loo_status = getattr(r, "_loo_status", None)
            if loo_status == "ran":
                att_range = getattr(r, "_loo_att_range", None)
                out["leave_one_out"] = {
                    "status": "ran",
                    # Headline single-donor-dependence metric: the largest baseline-
                    # relative swing (max |delta_att|). Preferred over att_range, which
                    # can look narrow even when every drop shifts the ATT far from the
                    # full-fit baseline in the same direction.
                    "max_abs_delta_att": _to_python_float(
                        getattr(r, "_loo_max_abs_delta_att", None)
                    ),
                    "att_range": (
                        [_to_python_float(att_range[0]), _to_python_float(att_range[1])]
                        if att_range is not None
                        else None
                    ),
                    "n_failed": _to_python_scalar(getattr(r, "_loo_n_failed", None)),
                    "n_infeasible": _to_python_scalar(getattr(r, "_loo_n_infeasible", None)),
                }
            else:
                _loo_reasons = {
                    "treated_fit_nonconverged": (
                        "leave_one_out() was run but the treated unit's own SCM fit "
                        "did not converge at fit time, so the baseline ATT is not a "
                        "valid reference for the leave-one-out deltas."
                    ),
                    "too_few_donors": (
                        "leave_one_out() was run but fewer than 2 donors are available "
                        "(dropping one must leave a non-empty pool)."
                    ),
                    "all_refits_failed": (
                        "leave_one_out() was run but every donor-drop refit failed to "
                        "converge, so no valid leave-one-out estimate was produced "
                        "(see the status='failed' rows); raise n_starts or loosen the "
                        "optimizer tolerances."
                    ),
                    "all_refits_infeasible": (
                        "leave_one_out() was run but every donor-drop refit was "
                        "structurally infeasible under v_method='cv' (the reduced donor "
                        "pool is indistinguishable in a re-aggregated CV window; see the "
                        "status='infeasible' rows); adjust the predictors, v_cv_t0, or "
                        "the donor pool."
                    ),
                    "all_refits_unusable": (
                        "leave_one_out() was run but no donor-drop refit was usable: "
                        "some failed to converge AND some were structurally infeasible "
                        "(see n_failed / n_infeasible)."
                    ),
                }
                out["leave_one_out"] = {
                    "status": "infeasible",
                    # Machine-readable code so consumers can distinguish a numerical
                    # convergence failure ("all_refits_failed") from structural
                    # infeasibility ("all_refits_infeasible" / "too_few_donors") or a
                    # mix ("all_refits_unusable") without parsing `reason`. The n_failed
                    # / n_infeasible counts give the exact breakdown.
                    "reason_code": loo_status,
                    "n_failed": _to_python_scalar(getattr(r, "_loo_n_failed", None)),
                    "n_infeasible": _to_python_scalar(getattr(r, "_loo_n_infeasible", None)),
                    "reason": _loo_reasons.get(
                        loo_status, "leave_one_out() produced no valid refits."
                    ),
                }
        else:
            out["leave_one_out"] = {
                "status": "not_run",
                "reason": (
                    "Call results.leave_one_out() to run leave-one-out donor "
                    "robustness (opt-in; refits once per reportably-weighted donor)."
                ),
            }

        # In-time (backdating) placebo (ADH 2015 §4): opt-in, surfaced once run.
        if getattr(r, "_in_time_df", None) is not None:
            in_time_status = getattr(r, "_in_time_status", None)
            if in_time_status == "ran":
                itp = r._in_time_df
                ran = itp[itp["status"] == "ran"] if "status" in itp else itp
                max_abs_att = float(ran["placebo_att"].abs().max()) if len(ran) else None
                out["in_time_placebo"] = {
                    "status": "ran",
                    # Full coverage breakdown so a partially-usable sweep is not
                    # overstated: n_dates is the requested grid; n_ran are the usable
                    # placebos; n_failed / n_infeasible are the dropped remainder.
                    "n_dates": _to_python_scalar(int(len(itp))),
                    "n_ran": _to_python_scalar(int(len(ran))),
                    "n_failed": _to_python_scalar(getattr(r, "_in_time_n_failed", None)),
                    "n_infeasible": _to_python_scalar(getattr(r, "_in_time_n_infeasible", None)),
                    "max_abs_placebo_att": _to_python_float(max_abs_att),
                }
            else:
                _it_reasons = {
                    "treated_fit_nonconverged": (
                        "in_time_placebo() was run but the treated unit's own SCM fit "
                        "did not converge at fit time."
                    ),
                    "too_few_pre_periods": (
                        "in_time_placebo() was run but there are too few pre-treatment "
                        "periods for any feasible placebo date (need >=3)."
                    ),
                    "all_dates_infeasible": (
                        "in_time_placebo() was run but every placebo date was "
                        "infeasible (no pre-fake period, all predictors dropped, or "
                        "the supplied custom_v had zero mass on the surviving "
                        "predictors after truncation)."
                    ),
                    "all_dates_failed": (
                        "in_time_placebo() was run but every placebo refit failed to "
                        "converge (none was dimensionally infeasible); raise n_starts "
                        "or loosen the optimizer tolerances."
                    ),
                    "all_dates_unusable": (
                        "in_time_placebo() was run but no placebo date produced a usable "
                        "result: some refits failed to converge AND some dates were "
                        "dimensionally infeasible (see n_failed / n_infeasible)."
                    ),
                }
                out["in_time_placebo"] = {
                    "status": "infeasible",
                    # Machine-readable code distinguishing a numerical convergence
                    # failure ("all_dates_failed") from structural infeasibility
                    # ("all_dates_infeasible" / "too_few_pre_periods") or a mix
                    # ("all_dates_unusable"), without parsing `reason`. The n_failed /
                    # n_infeasible counts give the exact breakdown.
                    "reason_code": in_time_status,
                    "n_failed": _to_python_scalar(getattr(r, "_in_time_n_failed", None)),
                    "n_infeasible": _to_python_scalar(getattr(r, "_in_time_n_infeasible", None)),
                    "reason": _it_reasons.get(
                        in_time_status, "in_time_placebo() produced no valid refits."
                    ),
                }
        else:
            out["in_time_placebo"] = {
                "status": "not_run",
                "reason": (
                    "Call results.in_time_placebo() to run the in-time (backdating) "
                    "placebo (opt-in; refits per backdated date)."
                ),
            }

        # Test-inversion confidence set (Firpo & Possebom 2018 §4): opt-in, surfaced once
        # the user has run results.confidence_set() (it reuses the in-space placebo
        # reference set — no refits). The analytical conf_int stays NaN; this is a SEPARATE
        # permutation set at level 1 - gamma, possibly unbounded or non-contiguous.
        ecs = getattr(r, "effect_confidence_set", None)
        if ecs is not None:
            ecs_status = ecs.get("status")
            _lo, _hi = ecs.get("lower"), ecs.get("upper")
            block = {
                "status": ecs_status,
                "family": ecs.get("family"),
                "parameter": ecs.get("parameter"),
                "gamma": _to_python_float(ecs.get("gamma")),
                # Emit each endpoint independently: a finite float, else None for a non-finite
                # side (NaN for an empty set, +/-inf for an unbounded tail) -- keeps the dict
                # JSON-safe while preserving the FINITE side of a one-sided unbounded set.
                "lower": float(_lo) if isinstance(_lo, (int, float)) and np.isfinite(_lo) else None,
                "upper": float(_hi) if isinstance(_hi, (int, float)) and np.isfinite(_hi) else None,
                "contiguous": bool(ecs.get("contiguous")),
                "n_placebos": _to_python_scalar(ecs.get("n_placebos")),
            }
            if ecs_status == "unbounded":
                block["reason"] = (
                    "confidence_set() ran but the set is unbounded (gamma below the "
                    "1/(J+1) permutation granularity, or the treated unit lacks the best "
                    "pre-treatment fit); endpoint(s) are +/-inf."
                )
            elif ecs_status == "empty":
                block["reason"] = (
                    "confidence_set() ran but the set is empty (every effect in the "
                    "family is rejected at gamma); endpoints are NaN."
                )
            out["confidence_set"] = block
        else:
            out["confidence_set"] = {
                "status": "not_run",
                "reason": (
                    "Call results.confidence_set() for a test-inversion confidence set of "
                    "the effect path (Firpo-Possebom 2018; opt-in, reuses the in-space "
                    "placebo reference set)."
                ),
            }

        # CWZ (2021) conformal inference: opt-in, surfaced once the user has run one of
        # conformal_test() / conformal_average_effect() / conformal_confidence_intervals().
        # A SEPARATE permutation object — it fits its OWN permutation-invariant proxy under
        # the null on all periods and permutes residuals over time (independent of the
        # in-space placebo). The analytical conf_int / p_value stay NaN.
        ci_summary = getattr(r, "conformal_inference", None)
        if ci_summary is not None:
            kind = ci_summary.get("kind")
            block = {
                "status": ci_summary.get("status"),
                "kind": kind,
                "scheme": ci_summary.get("scheme"),
                "n_perms": _to_python_scalar(ci_summary.get("n_perms")),
                "n_post": _to_python_scalar(ci_summary.get("n_post")),
            }
            if kind == "joint":
                block["q"] = ci_summary.get("q")
                block["p_value"] = _to_python_float(ci_summary.get("joint_p_value"))
                block["proxy_converged"] = bool(ci_summary.get("proxy_converged"))
            elif kind == "average":
                _lo, _hi = ci_summary.get("lower"), ci_summary.get("upper")
                block["alpha"] = _to_python_float(ci_summary.get("alpha"))
                block["lower"] = (
                    float(_lo) if isinstance(_lo, (int, float)) and np.isfinite(_lo) else None
                )
                block["upper"] = (
                    float(_hi) if isinstance(_hi, (int, float)) and np.isfinite(_hi) else None
                )
                block["point_estimate"] = _to_python_float(ci_summary.get("point_estimate"))
                block["contiguous"] = bool(ci_summary.get("contiguous"))
                block["n_blocks"] = _to_python_scalar(ci_summary.get("n_blocks"))
            elif kind == "pointwise":
                block["alpha"] = _to_python_float(ci_summary.get("alpha"))
                block["n_grid_limited"] = _to_python_scalar(ci_summary.get("n_grid_limited"))
                block["n_empty"] = _to_python_scalar(ci_summary.get("n_empty"))
                block["n_unbounded"] = _to_python_scalar(ci_summary.get("n_unbounded"))
            out["conformal_inference"] = block
        else:
            out["conformal_inference"] = {
                "status": "not_run",
                "reason": (
                    "Call results.conformal_test() / conformal_confidence_intervals() / "
                    "conformal_average_effect() for CWZ (2021) conformal inference (opt-in; "
                    "fits its own permutation-invariant proxy under the null, permutes "
                    "residuals over time)."
                ),
            }
        return out

    # -- Heterogeneity helpers --------------------------------------------

    def _collect_effect_scalars(self) -> List[float]:
        """Collect scalar **post-treatment** effect values across group / event-
        study / TROP sources.

        Pre-period coefficients (placebos and normalization constraints)
        and synthetic reference-marker rows are explicitly excluded —
        mixing them into the heterogeneity dispersion / sign-consistency
        summary silently redefines the estimand, which the round-6 CI
        review flagged on PR #318.

        Returns an empty list if no recognized effect container yields
        any post-treatment entries.
        """
        r = self._results
        # 1. group_effects: per-cohort post-treatment ATT(g) by construction.
        ge = getattr(r, "group_effects", None)
        if ge is not None:
            return self._scalars_from_mapping(ge)
        # 2. MultiPeriodDiDResults: use the ``post_period_effects`` property
        # (post-treatment only) instead of ``period_effects`` (which mixes
        # pre- and post-treatment coefficients).
        ppe = getattr(r, "post_period_effects", None)
        if ppe is not None:
            return self._scalars_from_mapping(ppe)
        # 3. event_study_effects: dict keyed by relative time -> dict with
        # 'effect'. Filter to **post-treatment** horizons (rel_time >= 0),
        # exclude reference markers (``n_groups == 0`` on CS/SA;
        # ``n_obs == 0`` on Stacked/TwoStage/Imputation/EfficientDiD), and
        # exclude entries with non-finite effect.
        es = getattr(r, "event_study_effects", None)
        if es is not None:
            # Anticipation-aware post-treatment cutoff: include horizons
            # from the anticipation window onward (where treatment-
            # affected effects can live) per REGISTRY.md §CallawaySantAnna
            # lines 355-395; round-15 CI review flagged the prior
            # ``rel >= 0`` rule as excluding anticipation-window effects
            # from the heterogeneity dispersion summary.
            post_cutoff = _pre_post_boundary(r)
            post_only: List[float] = []
            try:
                items = list(es.items())
            except Exception:  # noqa: BLE001
                items = []
            for key, entry in items:
                try:
                    rel = int(key)
                except (TypeError, ValueError):
                    # Non-integer keys — unknown shape; skip conservatively
                    # rather than mixing into the dispersion summary.
                    continue
                if rel < post_cutoff:
                    continue
                if isinstance(entry, dict):
                    if entry.get("n_groups") == 0 or entry.get("n_obs") == 0:
                        continue
                eff = _extract_scalar_effect(entry)
                if eff is None or not np.isfinite(eff):
                    continue
                post_only.append(eff)
            return post_only
        # 4. TROP: treatment_effects dict keyed by (unit, time) -> float.
        # TROP produces counterfactual deltas only at observed points for
        # treated units (the factor-model construction), so these are
        # post-treatment by design.
        te = getattr(r, "treatment_effects", None)
        if te is not None:
            return self._scalars_from_mapping(te)
        # 5. CS default aggregation: group_time_effects dict keyed by
        # (g, t) -> dict. Filter to t >= g (post-treatment cells); the
        # pre-treatment cells (t < g) are identification-deviation
        # placebos, not effect heterogeneity.
        gte = getattr(r, "group_time_effects", None)
        if gte is not None:
            post_cells: List[float] = []
            try:
                items = list(gte.items())
            except Exception:  # noqa: BLE001
                items = []
            for key, entry in items:
                g_t = None
                if isinstance(key, tuple) and len(key) == 2:
                    g_t = key
                else:
                    g_val = (
                        getattr(entry, "group", None)
                        if not isinstance(entry, dict)
                        else entry.get("group")
                    )
                    t_val = (
                        getattr(entry, "time", None)
                        if not isinstance(entry, dict)
                        else entry.get("time")
                    )
                    if g_val is not None and t_val is not None:
                        g_t = (g_val, t_val)
                if g_t is not None:
                    try:
                        g_num = float(g_t[0])
                        t_num = float(g_t[1])
                        # Estimator-specific post cutoff. CS /
                        # EfficientDiD / SA treat ``t >= g - anticipation``
                        # as treatment-affected (anticipation window is
                        # post-announcement). Wooldridge aggregation is
                        # documented as ``t >= g`` with the anticipation
                        # window rendered as placebos, not post-
                        # treatment effects (REGISTRY.md §Wooldridge
                        # lines 1351-1352). Round-16 CI review flagged
                        # the blanket anticipation shift as Wooldridge-
                        # unfaithful.
                        if type(r).__name__ == "WooldridgeDiDResults":
                            anticipation = 0
                        else:
                            anticipation = getattr(r, "anticipation", 0) or 0
                            try:
                                anticipation = int(anticipation)
                            except (TypeError, ValueError):
                                anticipation = 0
                        if t_num < g_num - anticipation:
                            continue
                    except (TypeError, ValueError):
                        pass
                eff = _extract_scalar_effect(entry)
                if eff is None or not np.isfinite(eff):
                    continue
                post_cells.append(eff)
            return post_cells
        return []

    @staticmethod
    def _scalars_from_mapping(mapping: Any) -> List[float]:
        """Extract scalar effect values from various result-mapping shapes."""
        out: List[float] = []
        values: List[Any]
        values_fn = getattr(mapping, "values", None)
        if callable(values_fn):
            try:
                values = list(values_fn())
            except Exception:  # noqa: BLE001
                return []
        else:
            try:
                values = list(mapping)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                return []
        for val in values:
            eff = _extract_scalar_effect(val)
            if eff is not None:
                out.append(eff)
        return out

    def _heterogeneity_source(self) -> str:
        """Name the attribute that produced the scalars (for the schema).

        Mirrors the dispatch order in ``_collect_effect_scalars`` and
        reports the actual post-treatment surface consumed (e.g.,
        ``post_period_effects`` rather than ``period_effects`` on
        ``MultiPeriodDiDResults``, and ``event_study_effects_post`` to
        make it clear pre-period / reference-marker rows were filtered).
        """
        r = self._results
        if getattr(r, "group_effects", None) is not None:
            return "group_effects"
        if getattr(r, "post_period_effects", None) is not None:
            return "post_period_effects"
        if getattr(r, "event_study_effects", None) is not None:
            return "event_study_effects_post"
        if getattr(r, "treatment_effects", None) is not None:
            return "treatment_effects"
        if getattr(r, "group_time_effects", None) is not None:
            return "group_time_effects_post"
        return "unknown"

    def _pt_hausman(self) -> Dict[str, Any]:
        """EfficientDiD native PT check via ``EfficientDiD.hausman_pretest``.

        This is the correct PT check for EfficientDiD (PT-All vs PT-Post); the
        generic event-study approach is inappropriate for this estimator per
        ``practitioner._parallel_trends_step`` guidance.
        """
        data = self._data
        outcome = self._outcome
        unit = self._unit
        time = self._time
        first_treat = self._first_treat
        missing = [
            name
            for name, val in (
                ("data", data),
                ("outcome", outcome),
                ("unit", unit),
                ("time", time),
                ("first_treat", first_treat),
            )
            if val is None
        ]
        if (
            missing
            or data is None
            or outcome is None
            or unit is None
            or time is None
            or first_treat is None
        ):
            return {
                "status": "skipped",
                "method": "hausman",
                "reason": (
                    "EfficientDiD.hausman_pretest requires data + outcome + unit + "
                    f"time + first_treat kwargs on DiagnosticReport; missing: "
                    f"{', '.join(missing)}."
                ),
            }

        # Fit-faithful guard. ``EfficientDiDResults`` exposes
        # ``control_group``, ``anticipation``, and ``estimation_path``
        # (``"nocov"`` or ``"dr"``) plus ``survey_metadata``, but not the
        # ``covariates`` list, ``cluster`` column, or nuisance kwargs
        # needed to replay a DR / clustered / survey-weighted fit. If
        # the original fit used any of those paths, rerunning the
        # pretest under defaults would diagnose a different design than
        # the estimate being summarized. Skip with an explicit reason
        # instead of silently fibbing.
        r = self._results
        estimation_path = getattr(r, "estimation_path", "nocov")
        has_survey = getattr(r, "survey_metadata", None) is not None
        if estimation_path != "nocov" or has_survey:
            reasons: List[str] = []
            if estimation_path == "dr":
                reasons.append(
                    "the original fit used the doubly-robust path with "
                    "covariates (``covariates`` list is not stored on "
                    "``EfficientDiDResults``)"
                )
            if has_survey:
                reasons.append(
                    "the original fit used a survey design (replay would "
                    "require the full ``SurveyDesign`` object)"
                )
            return {
                "status": "skipped",
                "method": "hausman",
                "reason": (
                    "Cannot faithfully replay the Hausman pretest: "
                    + "; ".join(reasons)
                    + ". Rerunning the pretest under defaults would "
                    "diagnose a different design than the estimate. "
                    "Rerun ``EfficientDiD.hausman_pretest(...)`` "
                    "manually with the original fit's kwargs or pass "
                    "``precomputed={'parallel_trends': ...}`` if you have "
                    "a pretest result."
                ),
            }

        # Propagate settings we can read off the result. On the
        # ``nocov`` / no-survey path we just gated to, the design
        # kwargs that matter for fit-faithful replay are
        # ``control_group``, ``anticipation``, and — when the fit was
        # clustered — ``cluster``. ``EfficientDiDResults`` persists the
        # cluster column so a clustered Hausman statistic is reported
        # for a clustered fit rather than a silently-unclustered one.
        hausman_kwargs: Dict[str, Any] = {}
        fit_control_group = getattr(r, "control_group", None)
        if isinstance(fit_control_group, str):
            hausman_kwargs["control_group"] = fit_control_group
        fit_anticipation = getattr(r, "anticipation", None)
        if isinstance(fit_anticipation, (int, float)) and np.isfinite(fit_anticipation):
            hausman_kwargs["anticipation"] = int(fit_anticipation)
        fit_cluster = getattr(r, "cluster_name", None)
        if isinstance(fit_cluster, str) and fit_cluster:
            hausman_kwargs["cluster"] = fit_cluster

        try:
            from diff_diff.efficient_did import EfficientDiD

            pt = EfficientDiD.hausman_pretest(
                data,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                alpha=self._alpha,
                **hausman_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "method": "hausman",
                "reason": f"hausman_pretest raised {type(exc).__name__}: {exc}",
            }

        p_value = _to_python_float(getattr(pt, "p_value", None))
        # ``HausmanPretestResult`` exposes ``statistic`` (not
        # ``test_statistic``); keep a fallback in case a precomputed
        # passthrough object uses the alternate name.
        test_stat = _to_python_float(getattr(pt, "statistic", getattr(pt, "test_statistic", None)))
        return {
            "status": "ran",
            "method": "hausman",
            "joint_p_value": p_value,
            "test_statistic": test_stat,
            "df": _to_python_scalar(getattr(pt, "df", None)),
            "verdict": _pt_verdict(p_value),
        }

    def _pt_synthetic_fit(self) -> Dict[str, Any]:
        """SDiD weighted pre-treatment-fit PT analogue.

        SDiD's design-enforced fit quality substitutes for a standard PT test:
        the synthetic control is explicitly constructed to match the treated
        group's pre-period trajectory, so small ``pre_treatment_fit`` RMSE
        means the weighted-PT analogue is satisfied.
        """
        r = self._results
        fit = _to_python_float(getattr(r, "pre_treatment_fit", None))
        if fit is None:
            return {
                "status": "skipped",
                "method": "synthetic_fit",
                "reason": "SyntheticDiDResults.pre_treatment_fit is not populated " "on this fit.",
            }
        # Proxy verdict: unlike a classical PT p-value, this is a fit-quality
        # metric. Classify conservatively — phrasing in BR will explain that
        # this is SDiD's design-enforced analogue, not a PT hypothesis test.
        return {
            "status": "ran",
            "method": "synthetic_fit",
            "pre_treatment_fit_rmse": fit,
            "verdict": "design_enforced_pt",
        }

    def _pt_scm_fit(self) -> Dict[str, Any]:
        """Classic SCM donor-weighted pre-treatment-fit PT analogue.

        Classic synthetic control (Abadie-Diamond-Hainmueller 2010) is not
        parallel-trends-based: the donor weights are chosen to match the treated
        unit's pre-period trajectory, so a small ``pre_rmspe`` is the
        design-enforced substitute for a PT test. Unlike SDiD this reads
        ``pre_rmspe`` (always populated on a successful fit). The in-space placebo
        permutation inference lives in ``estimator_native_diagnostics``.
        """
        r = self._results
        fit = _to_python_float(getattr(r, "pre_rmspe", None))
        if fit is None:
            return {
                "status": "skipped",
                "method": "scm_fit",
                "reason": "SyntheticControlResults.pre_rmspe is not populated on this fit.",
            }
        return {
            "status": "ran",
            "method": "scm_fit",
            "pre_treatment_fit_rmse": fit,
            "verdict": "design_enforced_pt",
        }

    def _pt_factor(self) -> Dict[str, Any]:
        """TROP has no PT concept — its identification is factor-model-based."""
        return {
            "status": "not_applicable",
            "reason": "TROP uses factor-model identification; parallel trends is "
            "not applicable. See estimator_native_diagnostics for the "
            "factor-model fit metrics.",
            "method": "factor",
        }

    def _format_precomputed_pt(self, obj: Any) -> Dict[str, Any]:
        """Adapt a pre-computed parallel-trends result to the schema shape.

        Accepted inputs (round-23 P1 CI review on PR #318):
          * A dict from ``utils.check_parallel_trends`` with ``p_value``
            (2x2 PT shape) — ``joint_p_value`` inherits from ``p_value``
            when only the 2x2 key is supplied.
          * A schema-shaped dict with ``joint_p_value`` and optional
            ``test_statistic`` / ``df`` / ``method`` (the same shape
            ``to_dict()["parallel_trends"]`` emits on the default path),
            so a PT block from one DR run can be replayed into another.
          * A native result object exposing ``p_value`` (or
            ``joint_p_value``) plus optional ``statistic`` /
            ``test_statistic`` and ``df`` — in particular, EfficientDiD's
            ``HausmanPretestResult``, which is what the ``_pt_hausman``
            skip message points users toward when replay fails on a
            non-nocov / survey fit.

        Previously the formatter rejected non-dict inputs outright and
        only read ``p_value``, so ``HausmanPretestResult`` could not be
        passed through at all and a schema-shaped dict silently lost its
        ``joint_p_value`` / ``test_statistic`` / ``df`` fields.
        """

        def _read(name: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        # Accept joint_p_value preferentially, but fall back to the 2x2
        # ``p_value`` key so ``utils.check_parallel_trends`` dicts still
        # work as before.
        raw_p = _read("joint_p_value")
        if raw_p is None:
            raw_p = _read("p_value")
        p_value = _to_python_float(raw_p)

        # ``HausmanPretestResult`` exposes ``statistic``; schema-shaped
        # dicts and the default DR path both use ``test_statistic``.
        raw_stat = _read("test_statistic")
        if raw_stat is None:
            raw_stat = _read("statistic")
        test_statistic = _to_python_float(raw_stat)

        df = _to_python_scalar(_read("df"))

        # Method inference (round-26 P2 CI review on PR #318). Downstream
        # BR / DR prose keys off ``method`` to pick the right subject and
        # statistic label (``"joint p"`` for event-study Wald /
        # Bonferroni, ``"p"`` for the 2x2 slope-difference and Hausman
        # single-statistic tests, no label for design-enforced paths).
        # Defaulting to ``"precomputed"`` made raw 2x2 dicts and native
        # Hausman objects render with the wrong subject ("Pre-treatment
        # data") and label ("joint p"). Infer from the distinguishing
        # fields when ``method`` is not explicit:
        #   * ``HausmanPretestResult`` / shape: has ``statistic``, plus
        #     at least one of ``att_all`` / ``att_post`` / ``recommendation``
        #     (disambiguates from the schema-shaped dict which may also
        #     carry ``test_statistic`` but does not carry the Hausman-
        #     specific companion fields).
        #   * ``utils.check_parallel_trends`` 2x2 dict: carries
        #     ``trend_difference`` / ``treated_trend`` / ``control_trend``
        #     as its distinguishing fields.
        method = _read("method")
        if method is None:
            hausman_markers = _read("statistic") is not None and any(
                _read(tag) is not None
                for tag in ("att_all", "att_post", "recommendation", "reject")
            )
            slope_markers = any(
                _read(tag) is not None
                for tag in ("trend_difference", "treated_trend", "control_trend")
            )
            if hausman_markers:
                method = "hausman"
            elif slope_markers:
                method = "slope_difference"
            else:
                method = "precomputed"

        # If no recognized p-value field was supplied at all, surface an
        # error rather than silently producing ``joint_p_value=None``.
        # Stay permissive about dict shapes — absence of ``test_statistic``
        # or ``df`` is fine (2x2 PT has neither), but a complete absence
        # of a p-value / joint-p-value means the input is not a PT result.
        if raw_p is None:
            return {
                "status": "error",
                "method": method,
                "reason": (
                    "precomputed['parallel_trends'] must expose either "
                    "``joint_p_value`` (schema shape / HausmanPretestResult) or "
                    "``p_value`` (check_parallel_trends 2x2 shape). Got an object "
                    "with neither: pass a dict with one of those keys, or a "
                    "native result object (e.g., HausmanPretestResult) exposing "
                    "``p_value``."
                ),
            }

        out: Dict[str, Any] = {
            "status": "ran",
            "method": method,
            "joint_p_value": p_value,
            "verdict": _pt_verdict(p_value),
            "precomputed": True,
        }
        if test_statistic is not None:
            out["test_statistic"] = test_statistic
        if df is not None:
            out["df"] = df
        # Preserve the survey-F denominator df when replaying a schema-
        # shaped PT block from the default path (round-28 P3 CI review
        # on PR #318). Without this, the finite-sample correction
        # recorded on the source block is silently dropped at replay.
        df_denom = _to_python_float(_read("df_denom"))
        if df_denom is not None:
            out["df_denom"] = df_denom
        return out

    # -- Headline metric extraction ----------------------------------------

    def _extract_headline_metric(self) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of the scalar headline metric from the result."""
        extracted = _extract_scalar_headline(self._results, fallback_alpha=self._alpha)
        if extracted is None:
            return None
        name, value, se, p, ci, alpha = extracted
        return {
            "name": name,
            "value": value,
            "se": se,
            "p_value": p,
            "conf_int": ci,
            "alpha": alpha,
        }


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------
def _extract_scalar_headline(
    results: Any,
    fallback_alpha: float = 0.05,
) -> Optional[
    Tuple[
        str,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[List[float]],
        Optional[float],
    ]
]:
    """Extract ``(name, value, se, p_value, conf_int, alpha)`` from a fitted result.

    Centralizes the scalar-headline mapping shared by both ``BusinessReport``
    and ``DiagnosticReport`` so schema drift (e.g. ``ContinuousDiDResults``
    using ``overall_att_se`` / ``overall_att_p_value`` /
    ``overall_att_conf_int`` instead of the ``overall_att`` stem) is handled
    in one place.

    Each row in the attribute-alias table below is tried in priority order.
    The first point-estimate attribute that resolves to a non-None value
    wins; the companion SE / p-value / CI attributes are then resolved from
    the same row, taking the first alias that exists on the result object.
    """
    # (name, [se aliases], [p-value aliases], [ci aliases])
    alias_table: List[Tuple[str, List[str], List[str], List[str]]] = [
        # Staggered / multi-period aggregations
        (
            "overall_att",
            ["overall_se", "overall_att_se"],
            ["overall_p_value", "overall_att_p_value"],
            ["overall_conf_int", "overall_att_conf_int"],
        ),
        # MultiPeriodDiDResults
        ("avg_att", ["avg_se"], ["avg_p_value"], ["avg_conf_int"]),
        # Simple DiDResults / SyntheticDiDResults / TROPResults / TripleDifferenceResults
        ("att", ["se"], ["p_value"], ["conf_int"]),
    ]
    for name, se_aliases, p_aliases, ci_aliases in alias_table:
        val = getattr(results, name, None)
        if val is None:
            continue
        se = next(
            (
                _to_python_float(getattr(results, a, None))
                for a in se_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        p = next(
            (
                _to_python_float(getattr(results, a, None))
                for a in p_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        ci = next(
            (
                _to_python_ci(getattr(results, a, None))
                for a in ci_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        alpha = _to_python_float(getattr(results, "alpha", fallback_alpha))
        return (name, _to_python_float(val), se, p, ci, alpha)
    return None


def _extract_scalar_effect(val: Any) -> Optional[float]:
    """Pull a scalar effect out of the many shapes results expose.

    Handles: ``PeriodEffect`` / ``GroupTimeEffect`` objects (``.effect``
    or ``.att`` attr), dicts with an ``"effect"`` or ``"att"`` key, and
    bare scalars. Wooldridge stores ``att`` in its ``group_time_effects``
    / ``group_effects`` / ``event_study_effects`` payloads rather than
    ``effect`` (round-16 CI review on PR #318).
    """
    if isinstance(val, dict):
        eff = val.get("effect")
        if eff is None:
            eff = val.get("att")
        if eff is None:
            return None
        try:
            return float(eff)
        except (TypeError, ValueError):
            return None
    eff_attr = getattr(val, "effect", None)
    if eff_attr is None:
        eff_attr = getattr(val, "att", None)
    if eff_attr is not None:
        try:
            return float(eff_attr)
        except (TypeError, ValueError):
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _power_tier(ratio: Optional[float]) -> str:
    """Map ``mdv / |att|`` to a phrasing tier used by ``BusinessReport``.

    Tiers per ``docs/methodology/REPORTING.md``:
      * ``well_powered``:         ratio < 0.25
      * ``moderately_powered``:  0.25 <= ratio < 1.0
      * ``underpowered``:         ratio >= 1.0
      * ``unknown``:              ratio is None or non-finite
    """
    if ratio is None or not np.isfinite(ratio):
        return "unknown"
    if ratio < 0.25:
        return "well_powered"
    if ratio < 1.0:
        return "moderately_powered"
    return "underpowered"


def _apply_diag_fallback_downgrade(tier: str, cov_source: str) -> str:
    """Conservatively downgrade ``well_powered`` to ``moderately_powered``
    when ``compute_pretrends_power`` used the diagonal-SE approximation
    while the full ``event_study_vcov`` was available on the source fit.

    REPORTING.md's conservative deviation: off-diagonal pre-period
    correlations are ignored under the diagonal fallback, so a
    ``well_powered`` verdict can overstate the real informativeness of
    the pre-test. The downgrade applies at every DR path
    (``_check_pretrends_power`` and ``_format_precomputed_pretrends_power``)
    so BR ``summary()`` / ``full_report()`` / ``to_dict()`` and DR
    ``summary()`` all read the same adjusted tier. Round-14 CI review
    flagged per-surface divergence; round-20 flagged that the precomputed
    adapter bypassed the downgrade entirely.

    PR-B (Roth 2022 audit) note: new fits set
    ``PreTrendsPowerResults.covariance_source`` directly at fit time
    based on the actual extraction path, so the report-layer adapters
    bypass ``_infer_cov_source`` whenever the persisted field is set.
    The "available but unused" sentinel is still produced for legacy
    ``PreTrendsPowerResults`` objects that lack the field — there we
    cannot distinguish a pre-PR-B fit (which DID drop to diag despite
    the populated source-fit matrix) from a post-PR-B fit, so the
    conservative downgrade still applies to legacy-ambiguous results.
    """
    if tier == "well_powered" and cov_source == "diag_fallback_available_full_vcov_unused":
        return "moderately_powered"
    return tier


def _pre_post_boundary(results: Any) -> int:
    """Return the relative-time cutoff that separates true pre-period
    horizons from treatment (and post-treatment) horizons.

    Horizons ``rel < _pre_post_boundary(results)`` are true pre-period
    coefficients suitable for PT tests and pre-trends power. Horizons
    ``rel >= _pre_post_boundary(results)`` include the anticipation
    window and post-treatment effects — these are the "affected by
    treatment (or anticipated treatment)" horizons, and are what
    heterogeneity dispersion should summarize.

    For anticipation-aware staggered estimators (CS, SA, EfficientDiD,
    etc., per REGISTRY.md §CallawaySantAnna lines 355-395), a fit with
    ``anticipation=k`` moves the identification boundary to
    ``e = -1 - k`` and treats ``e ∈ [-k, -1]`` as the anticipation
    window. True pre-periods are ``e < -k``. Returns ``-anticipation``
    (non-positive integer) in that case, falling back to ``0`` (the
    standard ``e < 0`` boundary) when no anticipation field is exposed.

    Round-15 CI review on PR #318 flagged the hard-coded ``rel < 0``
    rule as a methodology mismatch on anticipation fits.

    Estimator-specific override: Wooldridge aggregation keeps
    ``t >= g`` and treats anticipation-window cells as placebos, not
    post-treatment effects (REGISTRY.md §Wooldridge lines 1351-1352).
    The boundary for ``WooldridgeDiDResults`` is therefore ``0``
    regardless of the ``anticipation`` value stored on the result.
    """
    if type(results).__name__ == "WooldridgeDiDResults":
        return 0
    anticipation = getattr(results, "anticipation", 0)
    try:
        k = int(anticipation)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(k) or k < 0:
        return 0
    return -k


def _collect_pre_period_coefs(
    results: Any,
) -> Tuple[List[Tuple[Any, float, float, Optional[float]]], int]:
    """Return ``(sorted list of (key, effect, se, p_value), n_dropped_undefined)``
    for pre-period coefficients.

    Handles three shapes:
      * ``pre_period_effects``: dict-of-``PeriodEffect`` on ``MultiPeriodDiDResults``.
      * ``event_study_effects``: dict-of-dict (with ``effect`` / ``se`` / ``p_value`` keys)
        on the staggered estimators (CS / SA / ImputationDiD / Stacked / EDiD / etc.).
        Pre-period entries are those with negative relative-time keys.
      * ``placebo_event_study``: dict-of-dict on
        ``ChaisemartinDHaultfoeuilleResults`` — dCDH's dynamic placebos
        ``DID^{pl}_l`` are the estimator's pre-period analogue.

    Filtering rules (critical for methodology-safe PT tests):

    * Entries marked as reference markers (``n_groups == 0`` on CS / SA or
      ``n_obs == 0`` on Stacked / TwoStage / Imputation event-study shape)
      are excluded. These are synthetic ``effect=0, se=NaN`` rows injected
      for universal-base normalization and are NOT counted in
      ``n_dropped_undefined`` — they never represented a real pre-period.
    * Entries whose ``effect`` or ``se`` is non-finite (NaN / inf) or whose
      ``se <= 0`` are excluded as undefined inference (``safe_inference``
      contract, ``utils.py:175``). These ARE real pre-periods whose
      inference is undefined, so they contribute to
      ``n_dropped_undefined``. Round-33 P0 CI review on PR #318 flagged
      that the Bonferroni fallback silently shrank the test family when
      this happened, turning partially-undefined PT surfaces into clean
      stakeholder-facing verdicts. Callers (``_pt_event_study``) use
      ``n_dropped_undefined`` to force an inconclusive verdict rather
      than silently shrinking.

    Returns ``([], 0)`` when none of the three sources provides valid
    pre-period entries.
    """
    results_list: List[Tuple[Any, float, float, Optional[float]]] = []
    n_dropped_undefined = 0
    pre = getattr(results, "pre_period_effects", None)
    # dCDH exposes pre-period placebos via ``placebo_event_study``; the
    # round-6 CI review flagged that routing dCDH through the generic
    # ``event_study_effects`` path produced empty pre-coef lists and
    # silently skipped the PT check.
    dcdh_placebo = getattr(results, "placebo_event_study", None)
    if pre:
        for k, pe in pre.items():
            eff = getattr(pe, "effect", None)
            se = getattr(pe, "se", None)
            p = getattr(pe, "p_value", None)
            if eff is None or se is None:
                n_dropped_undefined += 1
                continue
            try:
                eff_f = float(eff)
                se_f = float(se)
            except (TypeError, ValueError):
                n_dropped_undefined += 1
                continue
            if not (np.isfinite(eff_f) and np.isfinite(se_f) and se_f > 0):
                n_dropped_undefined += 1
                continue
            results_list.append((k, eff_f, se_f, _to_python_float(p)))
    elif dcdh_placebo:
        # dCDH placebo horizons are the pre-period surface.
        for k, entry in dcdh_placebo.items():
            if not isinstance(entry, dict):
                continue
            eff = entry.get("effect")
            se = entry.get("se")
            p = entry.get("p_value")
            if eff is None or se is None:
                n_dropped_undefined += 1
                continue
            try:
                eff_f = float(eff)
                se_f = float(se)
            except (TypeError, ValueError):
                n_dropped_undefined += 1
                continue
            if not (np.isfinite(eff_f) and np.isfinite(se_f) and se_f > 0):
                n_dropped_undefined += 1
                continue
            results_list.append((k, eff_f, se_f, _to_python_float(p)))
    else:
        # Anticipation-aware cutoff: for CS/SA/EfficientDiD fits with
        # ``anticipation=k``, treat horizons ``e ∈ [-k, -1]`` as the
        # anticipation window (not true pre-periods) and only use
        # ``e < -k`` for PT tests.
        pre_cutoff = _pre_post_boundary(results)
        es = getattr(results, "event_study_effects", None) or {}
        for k, entry in es.items():
            # Pre-period relative-time keys are negative (convention: e=-1, -2, ...).
            try:
                rel = int(k)
            except (TypeError, ValueError):
                continue
            if rel >= pre_cutoff:
                continue
            if not isinstance(entry, dict):
                continue
            # Drop universal-base reference markers. These are synthetic,
            # not a real pre-period, so they do not count toward
            # ``n_dropped_undefined``.
            if entry.get("n_groups") == 0 or entry.get("n_obs") == 0:
                continue
            # Wooldridge stores ``att`` rather than ``effect`` in its
            # event-study payloads; accept either (round-16 CI review).
            eff = entry.get("effect")
            if eff is None:
                eff = entry.get("att")
            se = entry.get("se")
            p = entry.get("p_value")
            if eff is None or se is None:
                n_dropped_undefined += 1
                continue
            try:
                eff_f = float(eff)
                se_f = float(se)
            except (TypeError, ValueError):
                n_dropped_undefined += 1
                continue
            if not (np.isfinite(eff_f) and np.isfinite(se_f) and se_f > 0):
                n_dropped_undefined += 1
                continue
            results_list.append((k, eff_f, se_f, _to_python_float(p)))
    results_list.sort(key=lambda t: t[0] if isinstance(t[0], (int, float)) else str(t[0]))
    return results_list, n_dropped_undefined


def _smallest_failing_grid_m_dr(sens: Dict[str, Any]) -> Optional[float]:
    """Return the smallest evaluated M on the HonestDiD sensitivity
    grid if it already has the robust CI including zero, else ``None``.
    Matches ``business_report._smallest_failing_grid_m`` — both helpers
    must stay in sync for cross-surface parity. See PR #341 R1 review.

    ``breakdown_M`` is an interpolated threshold between grid points,
    so "the smallest grid point fails" is only a valid claim when the
    smallest actually-evaluated M has ``robust_to_zero == False``. On
    a grid that starts at M=0 where the smallest evaluated point is
    still robust, the breakdown value is information about what
    happens between grid points — not at the smallest grid point.
    """
    grid_points = sens.get("grid") or []
    sorted_grid = sorted(
        (p for p in grid_points if isinstance(p.get("M"), (int, float))),
        key=lambda p: p["M"],
    )
    if not sorted_grid:
        return None
    smallest = sorted_grid[0]
    if not smallest.get("robust_to_zero", True):
        return float(smallest["M"])
    return None


def _pt_verdict(p: Optional[float]) -> str:
    """Map a pre-trends joint p-value to the three-bin verdict enum.

    Verdicts per ``docs/methodology/REPORTING.md``:
      - p >= 0.30  -> ``no_detected_violation`` (phrasing hedges on power
        unless DR also reports that the test is well-powered via
        ``compute_pretrends_power``).
      - 0.05 <= p < 0.30  -> ``some_evidence_against``.
      - p < 0.05  -> ``clear_violation``.
    """
    if p is None or not np.isfinite(p):
        return "inconclusive"
    if p < 0.05:
        return "clear_violation"
    if p < 0.30:
        return "some_evidence_against"
    return "no_detected_violation"


def _to_python_float(value: Any) -> Optional[float]:
    """Convert numpy scalars to built-in ``float``; preserve None; return None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy scalars to built-in Python types where possible; pass through otherwise."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python_ci(ci: Any) -> Optional[List[float]]:
    """Convert a 2-tuple CI to ``[float, float]``; return None when malformed."""
    if ci is None:
        return None
    try:
        lo, hi = ci
    except (TypeError, ValueError):
        return None
    lo_f = _to_python_float(lo)
    hi_f = _to_python_float(hi)
    if lo_f is None or hi_f is None:
        return None
    return [lo_f, hi_f]


# ---------------------------------------------------------------------------
# Prose rendering helpers
# ---------------------------------------------------------------------------
def _check_headline(check: str, section: Dict[str, Any]) -> Optional[Any]:
    """Return the most descriptive scalar for the per-check row in to_dataframe()."""
    if section.get("status") != "ran":
        return None
    if check == "parallel_trends":
        return section.get("joint_p_value")
    if check == "pretrends_power":
        return section.get("mdv_share_of_att")
    if check == "sensitivity":
        return section.get("breakdown_M")
    if check == "bacon":
        return section.get("forbidden_weight")
    if check == "design_effect":
        return section.get("deff")
    if check == "heterogeneity":
        return section.get("cv")
    if check == "epv":
        return section.get("min_epv")
    if check == "estimator_native":
        # SDiD reports ``pre_treatment_fit``; classic SCM reports ``pre_rmspe`` (its
        # design-enforced pre-fit) — fall back so SCM's tabular headline is not None.
        fit = section.get("pre_treatment_fit")
        return fit if fit is not None else section.get("pre_rmspe")
    return None


def _pt_subject_phrase(method: Optional[str]) -> str:
    """Return a source-faithful subject for DR's PT verdict sentence.

    Round-8 CI review: the generic "pre-treatment event-study
    coefficients" wording mis-describes the 2x2 slope-difference check
    (``method="slope_difference"``) and EfficientDiD's Hausman PT-All
    vs PT-Post pretest (``method="hausman"``). See REGISTRY.md
    §EfficientDiD line 907 for the Hausman test's operating vector.
    """
    if method == "slope_difference":
        return "The pre-period slope-difference test"
    if method == "hausman":
        return "The Hausman PT-All vs PT-Post pretest"
    if method in {
        "joint_wald",
        "joint_wald_event_study",
        "joint_wald_no_vcov",
        "bonferroni",
        # Survey-aware event-study PT variants use an F(k, df_survey)
        # reference rather than chi-square(k); the subject is still the
        # pre-period event-study coefficient vector — only the
        # reference distribution changes (round-28 / round-29 CI
        # review on PR #318). Recognizing the ``_survey`` suffix here
        # lets DR prose match the BR prose and the REPORTING.md
        # contract.
        "joint_wald_survey",
        "joint_wald_event_study_survey",
    }:
        return "Pre-treatment event-study coefficients"
    if method == "synthetic_fit":
        return "The synthetic-control pre-treatment fit"
    if method == "scm_fit":
        return "The synthetic-control donor-weighted pre-treatment fit"
    if method == "factor":
        return "The factor-model pre-treatment fit"
    return "Pre-treatment data"


def _pt_stat_label(method: Optional[str]) -> Optional[str]:
    """Label for the joint-statistic p-value in the PT prose.

    Wald / Bonferroni paths take a joint p-value (``joint p``); the 2x2
    slope-difference and Hausman paths are single-statistic tests
    (``p``). Design-enforced paths return ``None`` so the sentence
    omits a statistic. Survey F-reference variants remain joint tests
    on the pre-period coefficient vector and keep the ``joint p``
    label — the correction is a different reference distribution, not
    a different test.
    """
    if method in {
        "joint_wald",
        "joint_wald_event_study",
        "joint_wald_no_vcov",
        "bonferroni",
        "joint_wald_survey",
        "joint_wald_event_study_survey",
    }:
        return "joint p"
    if method in {"slope_difference", "hausman"}:
        return "p"
    if method in {"synthetic_fit", "scm_fit", "factor"}:
        return None
    return "joint p"


def _render_overall_interpretation(schema: Dict[str, Any], labels: Dict[str, str]) -> str:
    """Synthesize a plain-English paragraph across DR checks.

    The paragraph names the headline effect, the dominant validity concern
    (typically parallel trends or sensitivity), secondary caveats
    (heterogeneity, design effect, Bacon), and one concrete next action.
    Never produces traffic-light verdicts — severity is conveyed by natural
    language per ``docs/methodology/REPORTING.md``.
    """
    sentences: List[str] = []
    headline = schema.get("headline_metric") or {}
    est = schema.get("estimator", "the estimator")
    outcome = labels.get("outcome_label", "the outcome")
    treatment = labels.get("treatment_label", "the treatment")

    # Sentence 1: headline.
    # Round-36 P0 CI review on PR #318: a non-finite headline value
    # (NaN ATT from a failed fit, e.g., rank-deficient design matrix or
    # zero effective sample) previously passed the ``val is not None``
    # guard because ``NaN is not None``. Since ``NaN > 0`` and
    # ``NaN < 0`` are both false, the directional branch fell through
    # to "did not change" and the sentence rendered as "did not change
    # ... by nan (p = nan, 95% CI: nan to nan)". BR's equivalent
    # headline renderer already gates on ``np.isfinite(value)`` and
    # emits an estimation-failure sentence; DR now mirrors that.
    val = headline.get("value") if isinstance(headline, dict) else None
    ci = headline.get("conf_int") if isinstance(headline, dict) else None
    p = headline.get("p_value") if isinstance(headline, dict) else None
    val_finite = isinstance(val, (int, float)) and np.isfinite(val)
    # PR #347 R4 P1: if the estimator intentionally produces no scalar
    # aggregate (dCDH ``trends_linear=True`` + ``L_max>=2``), route
    # through explicit no-scalar prose rather than the
    # estimation-failure branch below. The headline block carries
    # ``status="no_scalar_by_design"`` in that case.
    if isinstance(headline, dict) and headline.get("status") == "no_scalar_by_design":
        # PR #347 R13 P1: route through the headline-level ``reason``
        # field so the DR interpretation never drifts from the
        # schema-level populated-vs-empty branching.
        reason = headline.get("reason")
        base = (
            f"On {est}, {treatment} does not produce a scalar aggregate "
            f"effect on {outcome} under this configuration."
        )
        if isinstance(reason, str) and reason:
            sentences.append(f"{base} {reason}")
        else:
            sentences.append(base + " (by design)")
    elif val is not None and not val_finite:
        sentences.append(
            f"On {est}, {treatment}'s effect on {outcome} is non-finite "
            "(the estimation did not produce a usable point estimate). "
            "Inspect the fit for rank deficiency, zero effective sample, "
            "or a survey-design collapse before interpreting."
        )
    elif val_finite:
        direction = "increased" if val > 0 else "decreased" if val < 0 else "did not change"
        # Use the headline's own alpha rather than hardcoding 95 so prose
        # stays consistent with the rendered interval when alpha != 0.05.
        headline_alpha = headline.get("alpha") if isinstance(headline, dict) else None
        if isinstance(headline_alpha, (int, float)) and 0 < headline_alpha < 1:
            ci_level = int(round((1.0 - headline_alpha) * 100))
        else:
            ci_level = 95
        ci_finite = (
            isinstance(ci, (list, tuple))
            and len(ci) == 2
            and all(isinstance(v, (int, float)) and np.isfinite(v) for v in ci)
        )
        ci_str = f" ({ci_level}% CI: {ci[0]:.3g} to {ci[1]:.3g})" if ci_finite else ""
        p_str = f", p = {p:.3g}" if isinstance(p, (int, float)) and np.isfinite(p) else ""
        sentences.append(
            f"On {est}, {treatment} {direction} {outcome} by {val:.3g}{ci_str}{p_str}."
        )

    # Sentence 2: name the target parameter (BR/DR gap #6). Rendered
    # right after the headline so the reader sees what the scalar
    # represents before pre-trends / sensitivity context. Only the
    # terse ``name`` goes in the interpretation paragraph; the full
    # ``definition`` lives in DR's "## Target Parameter" markdown
    # section and in the structured ``schema["target_parameter"]``
    # dict for agents that want the long form.
    tp = schema.get("target_parameter") or {}
    tp_name = tp.get("name")
    if tp_name:
        sentences.append(f"Target parameter: {tp_name}.")

    # Sentence 3: parallel trends + power (method-aware prose per the
    # round-8 CI review on PR #318; PT method can be slope_difference
    # (2x2), joint_wald / bonferroni (event study), hausman (EfficientDiD
    # PT-All vs PT-Post), synthetic_fit (SDiD), or factor (TROP), and the
    # generic "event-study coefficients" wording is wrong for the
    # 2x2 and Hausman paths).
    pt = schema.get("parallel_trends") or {}
    pp = schema.get("pretrends_power") or {}
    # Only point to "the sensitivity analysis below" when a sensitivity
    # block actually ran. For estimators routing to native diagnostics
    # (SDiD / TROP) or fits where sensitivity was skipped / not
    # applicable, the clause would be misleading (round-12 CI review).
    sens_ran = (schema.get("sensitivity") or {}).get("status") == "ran"
    if pt.get("status") == "ran":
        verdict = pt.get("verdict")
        jp = pt.get("joint_p_value")
        method = pt.get("method")
        subject = _pt_subject_phrase(method)
        stat_label = _pt_stat_label(method)
        jp_str = (
            f" ({stat_label} = {jp:.3g})" if isinstance(jp, (int, float)) and stat_label else ""
        )
        sens_tail_pending = " pending sensitivity analysis" if sens_ran else ""
        sens_tail_alongside = (
            " Interpret the headline alongside the sensitivity analysis below." if sens_ran else ""
        )
        sens_tail_bounded = (
            " See the sensitivity analysis below for bounded-violation guarantees."
            if sens_ran
            else ""
        )
        sens_tail_reliable = (
            " See the HonestDiD sensitivity analysis below for a more reliable signal."
            if sens_ran
            else ""
        )
        if verdict == "clear_violation":
            sentences.append(
                f"{subject} clearly reject parallel trends{jp_str}. The "
                "headline estimate should be treated as tentative" + sens_tail_pending + "."
            )
        elif verdict == "some_evidence_against":
            sentences.append(
                f"{subject} show some evidence against parallel trends"
                f"{jp_str}." + sens_tail_alongside
            )
        elif verdict == "no_detected_violation":
            tier = pp.get("tier") if pp.get("status") == "ran" else "unknown"
            if tier == "well_powered":
                sentences.append(
                    f"{subject} are consistent with parallel trends"
                    f"{jp_str} and the test is well-powered (the max pre-period "
                    "level deviation at the MDV is a small share of the estimated "
                    "effect), so a material pre-trend would likely have been "
                    "detected."
                )
            elif tier == "moderately_powered":
                sentences.append(
                    f"{subject} do not reject parallel trends"
                    f"{jp_str}; the test is moderately informative." + sens_tail_bounded
                )
            else:
                sentences.append(
                    f"{subject} do not reject parallel trends"
                    f"{jp_str}, but the test has limited power — a non-rejection "
                    "does not prove the assumption." + sens_tail_reliable
                )
        elif verdict == "design_enforced_pt":
            rmse = pt.get("pre_treatment_fit_rmse")
            if pt.get("method") == "scm_fit":
                # Classic SCM: a single treated UNIT, donor-weighted level match;
                # significance is the in-space placebo, not SDiD's weighted PT.
                sentences.append(
                    f"The synthetic control reproduces the treated unit's "
                    f"pre-period trajectory with pre-RMSPE = {rmse:.3g} (classic "
                    f"SCM's donor-weighted, design-enforced analogue of parallel "
                    f"trends; significance is assessed via in-space placebo "
                    f"permutation, not a parallel-trends test)."
                    if isinstance(rmse, (int, float))
                    else "Classic SCM's donor-weighted synthetic control is designed "
                    "to match the treated unit's pre-period trajectory; significance "
                    "comes from in-space placebo permutation inference."
                )
            else:
                sentences.append(
                    f"The synthetic control matches the treated group's "
                    f"pre-period trajectory with RMSE = "
                    f"{rmse:.3g} (SDiD's design-enforced analogue of parallel "
                    f"trends)."
                    if isinstance(rmse, (int, float))
                    else "SDiD's synthetic control is designed to satisfy the "
                    "weighted parallel-trends analogue."
                )
        elif verdict == "inconclusive":
            # Round-35 P1 CI review on PR #318: DR summary / overall
            # interpretation must surface the inconclusive state
            # explicitly rather than omitting the PT sentence. A missing
            # sentence was indistinguishable from "PT did not run", and
            # stakeholders reading the summary could not tell that the
            # joint test had been attempted but yielded undefined
            # inference.
            n_dropped = pt.get("n_dropped_undefined")
            if isinstance(n_dropped, int) and n_dropped > 0:
                rows_word = "row" if n_dropped == 1 else "rows"
                sentences.append(
                    f"Pre-trends is inconclusive on this fit: "
                    f"{n_dropped} pre-period {rows_word} had undefined "
                    "inference (zero / negative SE or a non-finite "
                    "per-period p-value), so the joint test cannot be "
                    "formed. Treat parallel trends as unassessed."
                )
            else:
                sentences.append(
                    "Pre-trends is inconclusive on this fit: pre-period "
                    "inference was undefined, so the joint test cannot "
                    "be formed. Treat parallel trends as unassessed."
                )

    # Sentence 3: sensitivity. The "robust across the grid" phrasing is reserved
    # for genuine SensitivityResults grids; a precomputed single-M HonestDiDResults
    # is narrated as a point check ("at M=<value>") even though breakdown_M is None.
    sens = schema.get("sensitivity") or {}
    if sens.get("status") == "ran":
        bkd = sens.get("breakdown_M")
        conclusion = sens.get("conclusion")
        if conclusion == "single_M_precomputed":
            grid = sens.get("grid") or []
            point = grid[0] if grid else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                if robust:
                    sentences.append(
                        f"HonestDiD sensitivity (single point checked): "
                        f"at M = {m_val:.2g}, the robust CI excludes zero. "
                        f"This is a point check, not a grid — use "
                        f"HonestDiD.sensitivity() for a breakdown value."
                    )
                else:
                    sentences.append(
                        f"HonestDiD sensitivity (single point checked): "
                        f"at M = {m_val:.2g}, the robust CI includes zero. "
                        f"Run HonestDiD.sensitivity() across a grid to find "
                        f"the breakdown value."
                    )
        elif bkd is None:
            sentences.append(
                "The effect remains significant across the entire HonestDiD "
                "grid — robust to plausible parallel-trends violations."
            )
        elif isinstance(bkd, (int, float)) and bkd >= 1.0:
            sentences.append(
                f"HonestDiD sensitivity: the result remains significant under "
                f"parallel-trends violations up to {bkd:.2g}x the observed "
                f"pre-period variation."
            )
        else:
            # Round-1 BR/DR canonical-validation (2026-04-19) then
            # tightened per CI review on PR #341 R1: the "smallest
            # grid point" wording is only semantically correct when
            # the smallest M actually evaluated on the sensitivity
            # grid has ``robust_to_zero == False``. ``breakdown_M``
            # is the interpolated threshold between grid points, so
            # a small breakdown value on a grid starting at M=0
            # (where the smallest evaluated point is still robust)
            # would previously have been narrated as "smallest grid
            # point fails" — stronger than the evaluated grid
            # supports. Mirror BR's fix: check the grid directly.
            if isinstance(bkd, (int, float)):
                smallest_failed_m = _smallest_failing_grid_m_dr(sens)
                if smallest_failed_m is not None:
                    sentences.append(
                        "HonestDiD sensitivity: the result is fragile — "
                        "the confidence interval includes zero even at "
                        "the smallest M evaluated on the sensitivity "
                        f"grid (M = {smallest_failed_m:.2g})."
                    )
                else:
                    sentences.append(
                        f"HonestDiD sensitivity: the result is fragile — "
                        f"the confidence interval includes zero once "
                        f"violations reach {bkd:.2g}x the pre-period "
                        f"variation."
                    )

    # Sentence 4: one secondary caveat if present.
    bacon = schema.get("bacon") or {}
    if bacon.get("status") == "ran" and bacon.get("verdict") == "materially_contaminated":
        fw = bacon.get("forbidden_weight")
        if isinstance(fw, (int, float)):
            sentences.append(
                f"Goodman-Bacon decomposition flags {fw:.0%} of TWFE weight on "
                f"'forbidden' later-vs-earlier comparisons — consider a "
                f"heterogeneity-robust estimator (CS / SA / BJS / Gardner) if "
                f"not already in use."
            )
    deff = schema.get("design_effect") or {}
    if deff.get("status") == "ran" and not deff.get("is_trivial"):
        d = deff.get("deff")
        eff_n = deff.get("effective_n")
        if isinstance(d, (int, float)) and d >= 1.05:
            eff_str = f", effective n = {eff_n:.0f}" if isinstance(eff_n, (int, float)) else ""
            sentences.append(
                f"Survey design effect is {d:.2g} (variance inflation relative "
                f"to simple random sampling{eff_str})."
            )

    # Sentence 5: next step
    next_steps = schema.get("next_steps") or []
    if next_steps:
        top = next_steps[0]
        if top.get("label"):
            sentences.append(f"Next step: {top['label']}.")

    if not sentences:
        return ""
    return " ".join(s for s in sentences if s)


def _render_dr_full_report(results: "DiagnosticReportResults") -> str:
    """Render a markdown report from a populated ``DiagnosticReportResults``."""
    schema = results.schema
    lines: List[str] = []
    lines.append("# Diagnostic Report")
    lines.append("")
    lines.append(f"**Estimator**: `{schema.get('estimator')}`")
    headline = schema.get("headline_metric")
    if headline:
        # PR #347 R5 P2: the no-scalar-by-design branch
        # (``trends_linear=True`` + ``L_max>=2`` on dCDH) has
        # ``value`` / ``se`` / ``p_value`` all ``None``. Formatting
        # those straight into the ``**Headline**: ... = None`` line
        # would produce a malformed top headline even though the
        # "## Target Parameter" section below correctly explains
        # the design. Route to explicit no-scalar markdown.
        if headline.get("status") == "no_scalar_by_design":
            lines.append("**Headline**: no scalar aggregate by design.")
            reason = headline.get("reason")
            if reason:
                lines.append("")
                lines.append(reason)
        else:
            lines.append(
                f"**Headline**: {headline.get('name')} = "
                f"{headline.get('value')} "
                f"(SE {headline.get('se')}, p = {headline.get('p_value')})"
            )
    lines.append("")

    # BR/DR gap #6: target-parameter section between headline metadata
    # and the overall-interpretation paragraph.
    tp = schema.get("target_parameter") or {}
    if tp.get("name") or tp.get("definition"):
        lines.append("## Target Parameter")
        lines.append("")
        if tp.get("name"):
            lines.append(f"- **{tp['name']}**")
        if tp.get("definition"):
            lines.append(f"- {tp['definition']}")
        if tp.get("aggregation"):
            lines.append(f"- Aggregation tag: `{tp['aggregation']}`")
        if tp.get("headline_attribute"):
            lines.append(f"- Headline attribute: `{tp['headline_attribute']}`")
        if tp.get("reference"):
            lines.append(f"- Reference: {tp['reference']}")
        lines.append("")

    lines.append("## Overall Interpretation")
    lines.append("")
    lines.append(schema.get("overall_interpretation", "") or "_No synthesis available._")
    lines.append("")

    section_order = [
        ("Parallel trends", "parallel_trends"),
        ("Pre-trends power", "pretrends_power"),
        ("HonestDiD sensitivity", "sensitivity"),
        ("Goodman-Bacon decomposition", "bacon"),
        ("Effect-stability / heterogeneity", "heterogeneity"),
        ("Survey design effect", "design_effect"),
        ("Propensity-score EPV", "epv"),
        ("Estimator-native diagnostics", "estimator_native_diagnostics"),
        ("Placebo battery", "placebo"),
    ]
    for title, key in section_order:
        section = schema.get(key) or {}
        status = section.get("status", "not_run")
        lines.append(f"## {title}")
        lines.append(f"- status: `{status}`")
        if status == "skipped" or status == "not_applicable":
            reason = section.get("reason")
            if reason:
                lines.append(f"- reason: {reason}")
        else:
            for k, v in section.items():
                if k in ("status", "reason"):
                    continue
                if isinstance(v, (dict, list)):
                    continue
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    if schema.get("next_steps"):
        lines.append("## Next Steps")
        for s in schema["next_steps"]:
            if s.get("label"):
                lines.append(f"- {s['label']}")
                if s.get("why"):
                    lines.append(f"  - why: {s['why']}")
    return "\n".join(lines)
