"""
BusinessReport — plain-English stakeholder narrative from any diff-diff result.

Wraps any of the 16 fitted result types and produces:

- ``summary()``: a short paragraph block suitable for an email or Slack message.
- ``full_report()``: a multi-section markdown report with headline, assumptions,
  pre-trends, main result, robustness, sample, and an optional academic appendix.
- ``to_dict()``: a stable AI-legible structured schema (single source of truth —
  prose is rendered from this dict, not templated alongside it).

Design principles:

- Plain English, not academic jargon. The library ships this in addition to, not
  in place of, the estimator's existing ``results.summary()`` academic output.
- No estimator fitting. Every effect, SE, p-value, CI, and sensitivity bound
  is either read from ``results``, derived by the auto-constructed
  ``DiagnosticReport`` from the result's own post-fit
  ``aggregate('event_study')`` surface (a view or retained-kit recompute,
  used only when the raw ``event_study_effects`` field is absent; see the
  ``DiagnosticReport`` module docstring), or produced by an existing
  diff-diff utility. The report layer does compose a few cross-period
  summaries from per-period inputs already on the result (joint-Wald / Bonferroni
  pre-trends p-value, MDV-to-ATT ratio, heterogeneity dispersion over
  post-treatment effects); see ``docs/methodology/REPORTING.md`` for the full
  enumeration.
- Optional business context via keyword args (``outcome_label``, ``outcome_unit``,
  ``business_question``, ``treatment_label``). Without them, BusinessReport uses
  generic fallbacks — the zero-config path works.
- Diagnostic integration is implicit by default: ``BusinessReport(results)``
  auto-constructs a ``DiagnosticReport`` so the summary can mention pre-trends,
  robustness, and design-effect findings. Pass ``auto_diagnostics=False`` or an
  explicit ``diagnostics=`` object to override.

Methodology deviations (no traffic-light gates, pre-trends verdict thresholds,
power-aware phrasing, unit-translation policy, schema stability) are documented
in ``docs/methodology/REPORTING.md``. The ``to_dict()`` schema is marked
experimental in v3.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Union

import numpy as np

from diff_diff._reporting_helpers import describe_target_parameter
from diff_diff.diagnostic_report import DiagnosticReport, DiagnosticReportResults
from diff_diff.results_base import Diagnostic

BUSINESS_REPORT_SCHEMA_VERSION = "2.0"

__all__ = [
    "BusinessReport",
    "BusinessContext",
    "BUSINESS_REPORT_SCHEMA_VERSION",
]

# Recognized ``outcome_unit`` values mapped to a coarse "kind" used by the
# formatter. Unrecognized strings are accepted and rendered verbatim without
# arithmetic translation (``unit_kind = "unknown"``).
_UNIT_KINDS: Dict[str, str] = {
    "$": "currency",
    "usd": "currency",
    "%": "percent",
    "pp": "percentage_points",
    "percentage_points": "percentage_points",
    "percent": "percent",
    "log_points": "log_points",
    "log": "log_points",
    "count": "count",
    "users": "count",
}


@dataclass(frozen=True)
class BusinessContext:
    """Frozen bundle of business-framing metadata used when rendering prose.

    Populated from ``BusinessReport`` constructor kwargs. Falls back to
    neutral labels when fields are not supplied.
    """

    outcome_label: str
    outcome_unit: Optional[str]
    outcome_direction: Optional[str]
    business_question: Optional[str]
    treatment_label: str
    alpha: float


class BusinessReport:
    """Produce a stakeholder-ready narrative from any diff-diff results object.

    Parameters
    ----------
    results : Any
        A fitted diff-diff ESTIMATOR results object. Diagnostic results
        (anything subclassing ``diff_diff.Diagnostic``, e.g.
        ``BaconDecompositionResults``, ``HonestDiDResults``) are rejected
        by type — pass diagnostics via ``diagnostics=`` or
        ``DiagnosticReport(precomputed=...)`` instead.
    outcome_label : str, optional
        Stakeholder-friendly outcome name (e.g. ``"Revenue per user"``).
    outcome_unit : str, optional
        Unit label: ``"$"`` / ``"%"`` / ``"pp"`` / ``"log_points"`` / ``"count"``
        (recognized for formatting) or any free-form string (used verbatim
        without arithmetic translation).
    outcome_direction : str, optional
        ``"higher_is_better"`` or ``"lower_is_better"``. Drives whether the
        effect is described as "lift" / "drag" rather than just "increase" /
        "decrease".
    business_question : str, optional
        Question the analysis answers (prepended to the summary).
    treatment_label : str, optional
        Stakeholder-friendly treatment name (e.g. ``"the campaign"``).
    alpha : float, optional
        Significance level. Defaults to ``results.alpha`` when not supplied.
        Single knob: drives both CI level and significance phrasing.
    honest_did_results : HonestDiDResults or SensitivityResults, optional
        Pre-computed sensitivity result. When supplied, this is forwarded to
        the internal ``DiagnosticReport`` so sensitivity is not re-computed.
    auto_diagnostics : bool, default True
        When ``True`` and ``diagnostics`` is ``None``, auto-construct a
        ``DiagnosticReport``. Set ``False`` to skip diagnostics entirely.
    diagnostics : DiagnosticReport or DiagnosticReportResults, optional
        Explicit diagnostics object. Takes precedence over ``auto_diagnostics``.
    include_appendix : bool, default True
        Whether ``full_report()`` appends the estimator's academic
        ``results.summary()`` output under a "Technical Appendix" section.
    data, outcome, treatment, unit, time, first_treat : optional
        Raw panel + column names forwarded to the auto-constructed
        ``DiagnosticReport`` so data-dependent checks (2x2 PT on simple
        DiD, Bacon-from-scratch, EfficientDiD Hausman pretest) can run.
    survey_design : SurveyDesign, optional
        The ``SurveyDesign`` object used to fit a survey-weighted
        estimator. Forwarded to the auto-constructed ``DiagnosticReport``
        for fit-faithful Goodman-Bacon replay. When the fit carries
        ``survey_metadata`` but ``survey_design`` is not supplied, Bacon
        is skipped with an explicit reason rather than replaying an
        unweighted decomposition for a design that does not match the
        estimate. The simple 2x2 parallel-trends helper
        (``utils.check_parallel_trends``) has no survey-aware variant;
        on a survey-backed ``DiDResults`` it is skipped unconditionally
        regardless of ``survey_design``. Supply
        ``precomputed={'parallel_trends': ...}`` with a survey-aware
        pretest to opt in. See ``docs/methodology/REPORTING.md``.
    precomputed : dict, optional
        Pre-computed diagnostic objects forwarded to the auto-
        constructed ``DiagnosticReport`` (same keys as
        ``DiagnosticReport(precomputed=...)``): ``"parallel_trends"``,
        ``"sensitivity"``, ``"pretrends_power"``, ``"bacon"``. DR
        validates keys and rejects estimator-incompatible entries
        (e.g., HonestDiD bounds or generic PT on SDiD / TROP).
        ``honest_did_results`` remains a shorthand for ``sensitivity``;
        an explicit ``precomputed['sensitivity']`` wins on conflict.
    """

    def __init__(
        self,
        results: Any,
        *,
        outcome_label: Optional[str] = None,
        outcome_unit: Optional[str] = None,
        outcome_direction: Optional[str] = None,
        business_question: Optional[str] = None,
        treatment_label: Optional[str] = None,
        alpha: Optional[float] = None,
        honest_did_results: Optional[Any] = None,
        auto_diagnostics: bool = True,
        diagnostics: Optional[Union[DiagnosticReport, DiagnosticReportResults]] = None,
        include_appendix: bool = True,
        data: Optional[Any] = None,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        unit: Optional[str] = None,
        time: Optional[str] = None,
        first_treat: Optional[str] = None,
        survey_design: Optional[Any] = None,
        precomputed: Optional[Dict[str, Any]] = None,
    ):
        # The unified event-study container (row M-092) is rejected
        # explicitly: BusinessReport's headline schema is SCALAR
        # (effect/se/ci/p on one inference row), while EventStudyResults
        # is per-period by construction and the TWFE event-study mode
        # deliberately ships no overall ATT - so the report would render
        # an all-null headline under the generic fall-through
        # (no-silent-failures). BR/DR admission of EventStudyResults
        # surfaces is tracked in TODO.md.
        from diff_diff.results_base import EventStudyResults as _ESR

        if isinstance(results, _ESR):
            raise TypeError(
                "BusinessReport does not yet support EventStudyResults "
                "surfaces (the TWFE event-study mode and "
                "aggregate('event_study') containers): the narrative "
                "headline is scalar and per-period surfaces carry no "
                "overall ATT. Build the report from the fitted "
                "estimator's scalar results (e.g. a static "
                "TwoWayFixedEffects fit); use the event-study surface "
                "with HonestDiD, PreTrendsPower, or plot_event_study. "
                "EventStudyResults admission is tracked in TODO.md."
            )
        # Marked diagnostic results are rejected BY TYPE (spec section
        # 3.5, ledger row M-091): BusinessReport's primary input is a
        # fitted ESTIMATOR result carrying the canonical inference row.
        if isinstance(results, Diagnostic):
            if type(results).__name__ == "BaconDecompositionResults":
                raise TypeError(
                    "BaconDecompositionResults is a diagnostic, not an estimator; "
                    "wrap the underlying estimator with BusinessReport and pass the "
                    "Bacon object to DiagnosticReport(precomputed={'bacon': ...})."
                )
            raise TypeError(
                f"{type(results).__name__} is a diagnostic result, not an "
                "estimator result; BusinessReport takes the fitted "
                "estimator's results as its primary input. Pass diagnostic "
                "objects via the diagnostics= parameter (as a "
                "DiagnosticReport) or interpret them alongside the report."
            )

        if diagnostics is not None and not isinstance(
            diagnostics, (DiagnosticReport, DiagnosticReportResults)
        ):
            raise TypeError(
                "diagnostics= must be a DiagnosticReport or "
                "DiagnosticReportResults instance; "
                f"got {type(diagnostics).__name__}."
            )

        # Estimator-aware validation for ``honest_did_results``. SDiD /
        # TROP route robustness to ``estimator_native_diagnostics``
        # (SDiD: ``in_time_placebo``, ``sensitivity_to_zeta_omega``;
        # TROP: factor-model fit metrics) and do not accept HonestDiD
        # bounds because they are methodology-incompatible with the
        # documented native-routing contract in REPORTING.md. Reject
        # the passthrough here so it doesn't silently forward to the
        # auto-constructed ``DiagnosticReport`` (which now also
        # rejects it at construction time — round-21 P1 CI review on
        # PR #318).
        if honest_did_results is not None and type(results).__name__ in {
            "SyntheticDiDResults",
            "TROPResults",
            "SyntheticControlResults",
        }:
            raise ValueError(
                f"{type(results).__name__} routes robustness to "
                "``estimator_native_diagnostics`` — ``honest_did_results`` "
                "is not accepted on this estimator because HonestDiD "
                "bounds are methodology-incompatible with the native "
                "routing documented in REPORTING.md. Use the result "
                "object's native diagnostics "
                "(SDiD: ``in_time_placebo()``, ``sensitivity_to_zeta_omega()``, "
                "``pre_treatment_fit``; TROP: ``effective_rank``, "
                "``loocv_score``; SyntheticControl: ``in_space_placebo()``, "
                "``pre_rmspe``, ``get_placebo_df()``) — BusinessReport surfaces "
                "these automatically under ``estimator_native_diagnostics``."
            )

        # Round-44 P1 CI review on PR #318: mirror the SDiD/TROP
        # rejection pattern for ``CallawaySantAnna`` fits with
        # ``base_period != "universal"``. HonestDiD Rambachan-Roth
        # bounds are not valid for interpretation on the consecutive-
        # comparison pre-period surface produced by ``varying`` base,
        # so narrating precomputed sensitivity (whether passed as
        # ``honest_did_results`` or ``precomputed['sensitivity']``)
        # alongside a displayed varying-base fit mixes provenance the
        # bounds don't support. DR enforces the same guard at
        # construction; BR duplicates the check so the error fires
        # before the auto-DR is built, matching the existing
        # SDiD/TROP UX. REGISTRY.md §CallawaySantAnna line 410,
        # §HonestDiD line 2458.
        _cs_with_varying_base = type(results).__name__ in (
            "CallawaySantAnnaResults",
            "DMLDiDResults",
        ) and (getattr(results, "base_period", "universal") != "universal")
        if _cs_with_varying_base:
            _rejected_inputs: List[str] = []
            if honest_did_results is not None:
                _rejected_inputs.append("honest_did_results")
            if precomputed is not None and "sensitivity" in precomputed:
                _rejected_inputs.append("precomputed['sensitivity']")
            if _rejected_inputs:
                _base_period = getattr(results, "base_period", "universal")
                _est_name = type(results).__name__.removesuffix("Results")
                raise ValueError(
                    f"{type(results).__name__} with "
                    f"``base_period={_base_period!r}`` cannot be "
                    "summarized alongside a precomputed HonestDiD "
                    "sensitivity object. The Rambachan-Roth bounds are "
                    "not valid for interpretation on the consecutive-"
                    "comparison pre-period surface this base yields "
                    "(REGISTRY.md §CallawaySantAnna / §HonestDiD). "
                    "Rejected inputs: " + ", ".join(_rejected_inputs) + ". "
                    "Re-fit the main estimator with "
                    f"``{_est_name}(base_period='universal')`` "
                    "before passing precomputed sensitivity, or drop "
                    "the sensitivity passthrough to let BR skip the "
                    "section with a methodology-critical reason."
                )

        self._results = results
        self._honest_did_results = honest_did_results
        self._auto_diagnostics = auto_diagnostics
        self._diagnostics_arg = diagnostics
        self._include_appendix = include_appendix
        # Raw-data passthrough so the auto-constructed DR can run
        # data-dependent checks (2x2 PT on simple DiD, Bacon-from-
        # scratch on staggered estimators, EfficientDiD Hausman
        # pretest). Without these, the auto path silently skips those
        # checks (round-12 CI review on PR #318).
        self._dr_data = data
        self._dr_outcome = outcome
        self._dr_treatment = treatment
        self._dr_unit = unit
        self._dr_time = time
        self._dr_first_treat = first_treat
        # Round-40 P1 CI review on PR #318: survey-backed fits need
        # the ``SurveyDesign`` threaded through to the auto-constructed
        # DR so Bacon decomposition is fit-faithful and the 2x2 PT
        # skip path triggers for DiDResults with ``survey_metadata``.
        # Without this passthrough, the auto path silently replays an
        # unweighted decomposition / PT verdict for a weighted fit.
        self._dr_survey_design = survey_design
        # Round-43 P2 CI review on PR #318: BR docs and docstrings
        # advertised a ``precomputed={'parallel_trends': ...}`` opt-in
        # for survey-aware 2x2 PT and other escape hatches, but BR did
        # not actually accept a ``precomputed=`` kwarg — the auto path
        # only synthesized ``{"sensitivity": honest_did_results}``, so
        # callers following the BR docs hit a ``TypeError`` on
        # ``__init__``. Accept the passthrough here and forward every
        # key to the auto-constructed DR (which owns validation against
        # its implemented-key set and estimator-aware rejection rules).
        # ``honest_did_results`` still feeds into ``sensitivity`` as a
        # convenience; an explicit ``precomputed['sensitivity']`` wins
        # on conflict.
        self._dr_precomputed: Dict[str, Any] = dict(precomputed or {})
        # Round-43 P2 CI review on PR #318: mirror DR's eager key
        # validation so users get the "unsupported key" error at BR
        # construction rather than lazily when the DR is built inside
        # ``to_dict()``. Kept in sync with ``DiagnosticReport``'s
        # ``_supported_precomputed`` set; the cheapest way to avoid
        # drift would be to import the set, but DR currently scopes it
        # locally to ``__init__`` so mirror the literal here with a
        # pointer comment.
        _br_supported_precomputed = {
            "parallel_trends",
            "sensitivity",
            "pretrends_power",
            "bacon",
        }
        _br_unsupported = set(self._dr_precomputed) - _br_supported_precomputed
        if _br_unsupported:
            raise ValueError(
                "precomputed= contains keys that are not implemented: "
                f"{sorted(_br_unsupported)}. Supported keys: "
                f"{sorted(_br_supported_precomputed)}. ``design_effect``, "
                "``heterogeneity``, and ``epv`` are read from the fitted "
                "result (heterogeneity may also derive the post-fit "
                "aggregate('event_study') surface) and do not accept "
                "precomputed overrides."
            )

        resolved_alpha = alpha if alpha is not None else getattr(results, "alpha", 0.05)
        self._context = BusinessContext(
            outcome_label=outcome_label or "the outcome",
            outcome_unit=outcome_unit,
            outcome_direction=outcome_direction,
            business_question=business_question,
            treatment_label=treatment_label or "the treatment",
            alpha=float(resolved_alpha),
        )

        self._cached_schema: Optional[Dict[str, Any]] = None

    # -- Public API ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the AI-legible structured schema (single source of truth)."""
        if self._cached_schema is None:
            self._cached_schema = self._build_schema()
        return self._cached_schema

    def to_json(self, *, indent: int = 2) -> str:
        """Return ``to_dict()`` serialized as JSON."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Return a short plain-English paragraph block (6-10 sentences)."""
        return _render_summary(self.to_dict())

    def full_report(self) -> str:
        """Return a structured multi-section markdown report."""
        base = _render_full_report(self.to_dict())
        if self._include_appendix:
            appendix_text = None
            try:
                appendix = self._results.summary()
                if appendix:
                    appendix_text = str(appendix)
            except Exception as exc:  # noqa: BLE001
                appendix_error = type(exc).__name__ or "Exception"
                base = (
                    base
                    + "\n\n## Technical Appendix\n\n"
                    + "Technical appendix unavailable: estimator summary rendering failed "
                    + f"({appendix_error}).\n"
                )
            if appendix_text:
                base = base + "\n\n## Technical Appendix\n\n```\n" + appendix_text + "\n```\n"
        return base

    def export_markdown(self) -> str:
        """Alias for ``full_report()`` (discoverability)."""
        return self.full_report()

    def headline(self) -> str:
        """Return just the headline sentence."""
        return _render_headline_sentence(self.to_dict())

    def caveats(self) -> List[Dict[str, str]]:
        """Return the list of structured caveats (severity + topic + message)."""
        return list(self.to_dict().get("caveats", []))

    def __repr__(self) -> str:
        estimator = type(self._results).__name__
        headline = self.to_dict().get("headline") or {}
        val = headline.get("effect")
        if isinstance(val, (int, float)) and np.isfinite(val):
            return f"BusinessReport(results={estimator}, effect={val:.3g})"
        return f"BusinessReport(results={estimator})"

    def __str__(self) -> str:
        return self.summary()

    # -- Implementation detail ---------------------------------------------

    def _resolve_diagnostics(self) -> Optional[DiagnosticReportResults]:
        """Return the DiagnosticReportResults to embed, or ``None`` if skipped."""
        if self._diagnostics_arg is not None:
            if isinstance(self._diagnostics_arg, DiagnosticReportResults):
                return self._diagnostics_arg
            if isinstance(self._diagnostics_arg, DiagnosticReport):
                return self._diagnostics_arg.run_all()
            raise TypeError("diagnostics= must be a DiagnosticReport or DiagnosticReportResults")
        if not self._auto_diagnostics:
            return None
        # Round-43 P2 CI review on PR #318: forward the user's
        # ``precomputed`` dict through to DR. ``honest_did_results``
        # stays a convenience shortcut for ``sensitivity`` only; an
        # explicit ``precomputed['sensitivity']`` from the caller
        # wins. DR handles key validation (rejects unsupported keys
        # and estimator-incompatible sensitivities / parallel_trends
        # entries) so BR just merges and forwards.
        precomputed: Dict[str, Any] = dict(self._dr_precomputed)
        if self._honest_did_results is not None:
            precomputed.setdefault("sensitivity", self._honest_did_results)
        dr = DiagnosticReport(
            self._results,
            alpha=self._context.alpha,
            precomputed=precomputed or None,
            outcome_label=self._context.outcome_label,
            treatment_label=self._context.treatment_label,
            data=self._dr_data,
            outcome=self._dr_outcome,
            treatment=self._dr_treatment,
            unit=self._dr_unit,
            time=self._dr_time,
            first_treat=self._dr_first_treat,
            survey_design=self._dr_survey_design,
        )
        return dr.run_all()

    def _build_schema(self) -> Dict[str, Any]:
        """Assemble the structured schema.

        Pulls validation content (PT, sensitivity, Bacon, DEFF, EPV, ...) from
        the internal ``DiagnosticReport``; extracts the stakeholder-facing
        headline and sample metadata from the fitted result itself.
        """
        estimator_name = type(self._results).__name__
        diagnostics_results = self._resolve_diagnostics()
        dr_schema: Optional[Dict[str, Any]] = (
            diagnostics_results.schema if diagnostics_results is not None else None
        )

        # PR #347 R4 P1: compute target_parameter BEFORE extracting
        # the headline so the no-scalar-by-design case
        # (``aggregation == "no_scalar_headline"``, e.g., dCDH
        # ``trends_linear=True`` with ``L_max >= 2``) can route the
        # headline through a dedicated branch that names the intentional
        # NaN rather than an estimation-failure path.
        target_parameter = describe_target_parameter(self._results)
        if target_parameter.get("aggregation") == "no_scalar_headline":
            # PR #347 R12 P1: the no-scalar ``reason`` must distinguish
            # the populated-surface case (per-horizon table exists) from
            # the empty-surface subcase (``linear_trends_effects=None``
            # — no horizons survived estimation). Telling a user with
            # an empty surface to "see linear_trends_effects" is
            # dead-end guidance.
            _surface_empty = getattr(self._results, "linear_trends_effects", None) is None
            # PR #347 R14 P1: the empty-surface reason must use the
            # covariate-adjusted label when covariates are active.
            _has_controls = getattr(self._results, "covariate_residuals", None) is not None
            _empty_surface_label = "DID^{X,fd}_l" if _has_controls else "DID^{fd}_l"
            if _surface_empty:
                no_scalar_reason = (
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
                no_scalar_reason = (
                    "The fitted estimator intentionally does not produce a "
                    "scalar overall ATT on this configuration "
                    "(``trends_linear=True`` with ``L_max >= 2``). Per-horizon "
                    "cumulated level effects are on "
                    "``results.linear_trends_effects[l]``."
                )
            headline = {
                "status": "no_scalar_by_design",
                "effect": None,
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
                "alpha_was_honored": True,
                "alpha_override_caveat": None,
                "ci_level": int(round((1.0 - self._context.alpha) * 100)),
                "p_value": None,
                "is_significant": False,
                "near_significance_threshold": False,
                "unit": self._context.outcome_unit,
                "unit_kind": _UNIT_KINDS.get(
                    self._context.outcome_unit.lower() if self._context.outcome_unit else "",
                    "unknown",
                ),
                "sign": "none",
                "breakdown_M": None,
                "reason": no_scalar_reason,
            }
        else:
            headline = self._extract_headline(dr_schema)
        sample = self._extract_sample()
        heterogeneity = _lift_heterogeneity(dr_schema)
        pre_trends = _lift_pre_trends(dr_schema)
        sensitivity = _lift_sensitivity(dr_schema)
        robustness = _lift_robustness(dr_schema)
        assumption = _apply_anticipation_to_assumption(
            _describe_assumption(estimator_name, self._results),
            self._results,
        )
        next_steps = (dr_schema or {}).get("next_steps", [])
        caveats = _build_caveats(self._results, headline, sample, dr_schema)
        references = _references_for(estimator_name)

        if diagnostics_results is None:
            diagnostics_block: Dict[str, Any] = {
                "status": "skipped",
                "reason": "auto_diagnostics=False",
            }
        else:
            diagnostics_block = {
                "status": "ran",
                "schema": dr_schema,
                "overall_interpretation": (
                    dr_schema.get("overall_interpretation", "") if dr_schema is not None else ""
                ),
            }

        return {
            "schema_version": BUSINESS_REPORT_SCHEMA_VERSION,
            "estimator": {
                "class_name": estimator_name,
                "display_name": estimator_name,
            },
            "context": {
                "outcome_label": self._context.outcome_label,
                "outcome_unit": self._context.outcome_unit,
                "outcome_direction": self._context.outcome_direction,
                "business_question": self._context.business_question,
                "treatment_label": self._context.treatment_label,
                "alpha": self._context.alpha,
            },
            "headline": headline,
            "target_parameter": target_parameter,
            "assumption": assumption,
            "pre_trends": pre_trends,
            "sensitivity": sensitivity,
            "sample": sample,
            "heterogeneity": heterogeneity,
            "robustness": robustness,
            "diagnostics": diagnostics_block,
            "next_steps": next_steps,
            "caveats": caveats,
            "references": references,
        }

    def _extract_headline(self, dr_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the headline effect + CI + p-value from the result."""
        r = self._results
        # Delegate the attribute-alias lookup to the shared helper in the
        # diagnostic_report module so BR and DR agree on which fields a
        # result class exposes for its headline (including
        # ``ContinuousDiDResults`` which uses ``overall_att_se`` /
        # ``overall_att_p_value`` / ``overall_att_conf_int``).
        from diff_diff.diagnostic_report import _extract_scalar_headline

        extracted = _extract_scalar_headline(r, fallback_alpha=self._context.alpha)
        att: Optional[float] = None
        se: Optional[float] = None
        p: Optional[float] = None
        ci: Optional[List[float]] = None
        alpha = self._context.alpha
        result_alpha: Optional[float] = None
        if extracted is not None:
            _name, att, se, p, ci, result_alpha = extracted

        # On any alpha mismatch, preserve the fitted CI at its native
        # level. A faithful CI cannot be recomputed from point estimate
        # and SE alone without reproducing the fit's inference contract
        # (finite-df t-quantile, percentile bootstrap, wild cluster
        # bootstrap, survey replicate quantile, rank-deficient
        # undefined-df, etc.), and the 16 result classes do not expose
        # a uniform descriptor for that. Two separate alpha values:
        # ``display_alpha`` drives ``ci_level`` so the displayed CI
        # label matches the preserved bounds; the caller's requested
        # alpha drives the significance phrasing (``is_significant`` /
        # ``near_threshold``). A caveat records the override.
        display_alpha = alpha
        phrasing_alpha = alpha
        alpha_was_honored = True
        alpha_override_caveat: Optional[str] = None
        if (
            result_alpha is not None
            and not np.isclose(alpha, result_alpha)
            and att is not None
            and se is not None
        ):
            inference_method = getattr(r, "inference_method", "analytical")
            if inference_method == "wild_bootstrap":
                inference_label = "wild cluster bootstrap"
            elif (
                inference_method == "bootstrap" or getattr(r, "bootstrap_results", None) is not None
            ):
                inference_label = "bootstrap"
            elif getattr(r, "bootstrap_distribution", None) is not None:
                inference_label = "bootstrap"
            elif getattr(r, "variance_method", None) in {"bootstrap", "jackknife", "placebo"}:
                variance_method = getattr(r, "variance_method", None)
                inference_label = f"{variance_method} variance"
            else:
                df_survey = getattr(
                    r,
                    "df_survey",
                    getattr(getattr(r, "survey_metadata", None), "df_survey", None),
                )
                if isinstance(df_survey, (int, float)) and df_survey > 0:
                    inference_label = "finite-df survey"
                elif isinstance(df_survey, (int, float)) and df_survey == 0:
                    # Rank-deficient replicate design: the fit deliberately
                    # left inference undefined. Preserve (NaN bounds remain NaN).
                    inference_label = "undefined-df (replicate-weight)"
                else:
                    # Ordinary analytical fit with a finite but unexposed
                    # ``df`` (``DifferenceInDifferences`` / ``MultiPeriodDiD``
                    # / most staggered estimators / TROP). We cannot
                    # reproduce the t-quantile without the fit's ``df``.
                    inference_label = "analytical (native degrees of freedom)"

            display_alpha = float(result_alpha)
            alpha_was_honored = False
            alpha_override_caveat = (
                f"Requested alpha ({phrasing_alpha:.2f}) was not honored "
                f"for the confidence interval because this fit uses "
                f"{inference_label} inference; the displayed CI remains "
                f"at the fit's native level "
                f"({int(round((1.0 - result_alpha) * 100))}%). The "
                f"significance phrasing still uses the requested alpha."
            )

        unit = self._context.outcome_unit
        unit_kind = _UNIT_KINDS.get(unit.lower() if unit else "", "unknown")
        sign = (
            "positive"
            if (att is not None and att > 0)
            else (
                "negative"
                if (att is not None and att < 0)
                else ("null" if att == 0 else "undefined")
            )
        )
        if att is None or not np.isfinite(att):
            sign = "undefined"
        ci_level = int(round((1.0 - display_alpha) * 100))
        # bool(...) coerces away numpy bool_ — when ``p`` is a numpy NaN (e.g.
        # SyntheticControl, whose analytical p_value is always NaN), ``np.isfinite``
        # yields a numpy bool that is NOT JSON-serializable in the schema.
        is_significant = bool(
            p is not None and np.isfinite(p) and p < phrasing_alpha if p is not None else False
        )
        near_threshold = bool(
            p is not None
            and np.isfinite(p)
            and (phrasing_alpha - 0.01) < p < (phrasing_alpha + 0.001)
        )
        # Use DR-computed breakdown_M if available for quick reference.
        breakdown_M: Optional[float] = None
        if dr_schema:
            sens_section = dr_schema.get("sensitivity") or {}
            if sens_section.get("status") == "ran":
                breakdown_M = sens_section.get("breakdown_M")

        return {
            "effect": att,
            "se": se,
            "ci_lower": ci[0] if ci else None,
            "ci_upper": ci[1] if ci else None,
            "alpha_was_honored": alpha_was_honored,
            "alpha_override_caveat": alpha_override_caveat,
            "ci_level": ci_level,
            "p_value": p,
            "is_significant": is_significant,
            "near_significance_threshold": near_threshold,
            "unit": unit,
            "unit_kind": unit_kind,
            "sign": sign,
            "breakdown_M": breakdown_M,
        }

    def _extract_sample(self) -> Dict[str, Any]:
        """Extract sample metadata from the fitted result."""
        r = self._results
        survey = self._extract_survey_block()
        n_treated = _safe_int(getattr(r, "n_treated", getattr(r, "n_treated_units", None)))
        n_control_units = _safe_int(getattr(r, "n_control", getattr(r, "n_control_units", None)))

        # Control-group semantics. For estimators that expose a
        # ``control_group`` kwarg (CS, EfficientDiD, ContinuousDiD,
        # StaggeredTripleDiff, ...), the meaning of ``n_control_units``
        # depends on it. When the mode is "not-yet-treated" (dynamic
        # comparison set), the fixed tally stored on the result is only
        # the fully-untreated subset — the actual comparison set varies
        # by (g, t) cell. Label the exposed count accordingly so prose
        # surfaces the dynamic context instead of misreporting
        # "0 control" (round-13 / round-17 / round-18 CI review).
        #
        # Canonicalize both ``"not_yet_treated"`` (CS / EfficientDiD /
        # ContinuousDiD / Wooldridge) and ``"notyettreated"``
        # (StaggeredTripleDiff) as the same dynamic mode.
        #
        # Per-estimator fixed-subset field:
        #   * CS / SA / Imputation / TwoStage / EfficientDiD /
        #     dCDH / ContinuousDiD — ``n_control_units`` is the
        #     never-treated tally; surface as ``n_never_treated``.
        #   * StaggeredTripleDiff — ``n_control_units`` is a composite
        #     total; the fixed subset is ``n_never_enabled`` (stored
        #     separately on the result).
        #   * Wooldridge — ``n_control_units`` is total eligible
        #     comparisons (never-treated + future-treated) and does not
        #     map to a never-treated count. Keep on the fixed-count
        #     path even in dynamic mode.
        #   * Stacked — ``n_control_units`` is "distinct control units
        #     across the trimmed set" (stacked_did_results.py L59-62).
        #     Under ``clean_control="not_yet_treated"``, the trimmed
        #     set uses the rule ``A_s > a + kappa_post`` which admits
        #     future-treated controls; it is NOT a never-treated tally
        #     and cannot be relabeled as ``n_never_treated``. Keep
        #     Stacked on the fixed-count path (round-21 P1 CI review
        #     on PR #318 flagged the earlier relabeling as a
        #     semantic-contract violation).
        control_group = _control_group_choice(r)
        name = type(r).__name__
        n_never_treated: Optional[int] = None
        n_never_enabled: Optional[int] = None
        n_control: Optional[int] = n_control_units
        _never_treated_count_contract = name in {
            "CallawaySantAnnaResults",
            "DMLDiDResults",
            "SunAbrahamResults",
            "ImputationDiDResults",
            "TwoStageDiDResults",
            "EfficientDiDResults",
            "ChaisemartinDHaultfoeuilleResults",
            "ContinuousDiDResults",
        }
        _canonical_control = (
            control_group.replace("_", "").lower() if isinstance(control_group, str) else None
        )
        # Stacked has two dynamic (sub-experiment-specific) modes:
        # ``not_yet_treated`` (A_s > a + kappa_post) and ``strict``
        # (A_s > a + kappa_post + kappa_pre). Only ``never_treated``
        # (A_s = infinity) is a fixed never-treated pool. Round-22 P1
        # CI review on PR #318 flagged that ``strict`` was being
        # misrendered as a fixed control design.
        is_stacked_dynamic = name == "StackedDiDResults" and _canonical_control in {
            "notyettreated",
            "strict",
        }
        is_dynamic_control = _canonical_control == "notyettreated" or is_stacked_dynamic
        # StaggeredTripleDiff comparison-group contract:
        # ``n_control_units`` is a composite total that also includes
        # the eligibility-denied / larger-cohort cells. Regardless of
        # the ``control_group`` mode the valid fixed comparison is the
        # never-enabled cohort (``staggered_triple_diff.py:384``,
        # REGISTRY.md §StaggeredTripleDifference line 1730). Round-37
        # P1 CI review on PR #318: under ``control_group="never_treated"``
        # (i.e., ``_canonical_control == "nevertreated"``) the composite
        # total was being narrated as "control". Surface
        # ``n_never_enabled`` instead on both the ``nevertreated`` and
        # the dynamic ``notyettreated`` modes.
        if name == "StaggeredTripleDiffResults" and _canonical_control == "nevertreated":
            n_never_enabled = _safe_int(getattr(r, "n_never_enabled", None))
            n_control = None
        if is_dynamic_control:
            if name == "StaggeredTripleDiffResults":
                n_never_enabled = _safe_int(getattr(r, "n_never_enabled", None))
                n_control = None
            elif name == "StackedDiDResults":
                # ``n_control_units`` is "distinct control units across
                # the trimmed set" (stacked_did_results.py L59-62) which
                # includes future-treated controls by construction under
                # both dynamic modes. Do NOT relabel as
                # ``n_never_treated``; instead surface the count under
                # ``n_distinct_controls_trimmed`` (sub-experiment-
                # specific context) and clear ``n_control`` so the
                # report does not narrate a fixed control pool.
                n_control = None
            elif _never_treated_count_contract:
                n_never_treated = n_control_units
                n_control = None

        # Panel-vs-RCS count semantics. CallawaySantAnnaResults stores
        # treated/control counts as OBSERVATIONS (not units) when the
        # fit used ``panel=False`` — ``staggered_results.py L183-L184``
        # renders those counts as "obs:" rather than "units:". BR
        # previously labeled them as "units" / "present in the panel",
        # which misstates the sample composition for repeated cross-
        # section fits. Carry the flag into the schema so rendering can
        # branch. Round-28 P2 CI review on PR #318.
        count_unit = "observations" if getattr(r, "panel", True) is False else "units"

        sample_block: Dict[str, Any] = {
            "n_obs": _safe_int(getattr(r, "n_obs", None)),
            "n_treated": n_treated,
            "n_control": n_control,
            "n_never_treated": n_never_treated,
            "control_group": control_group if isinstance(control_group, str) else None,
            "dynamic_control": is_dynamic_control,
            "n_periods": _safe_int(getattr(r, "n_periods", None)),
            "pre_periods": _safe_list_len(getattr(r, "pre_periods", None)),
            "post_periods": _safe_list_len(getattr(r, "post_periods", None)),
            "count_unit": count_unit,
            "survey": survey,
        }
        if n_never_enabled is not None:
            sample_block["n_never_enabled"] = n_never_enabled
        # Stacked-specific: surface the distinct-control-units tally on a
        # dedicated key so agents see the sub-experiment-specific
        # comparison count without misreading it as a never-treated
        # subset (round-21 / round-22 CI review).
        if name == "StackedDiDResults":
            sample_block["n_distinct_controls_trimmed"] = n_control_units
        return sample_block

    def _extract_survey_block(self) -> Optional[Dict[str, Any]]:
        sm = getattr(self._results, "survey_metadata", None)
        if sm is None:
            return None
        deff = _safe_float(getattr(sm, "design_effect", None))
        return {
            "weight_type": getattr(sm, "weight_type", None),
            "effective_n": _safe_float(getattr(sm, "effective_n", None)),
            "design_effect": deff,
            # Round-43 P2 CI review on PR #318: the ``is_trivial``
            # upper bound matches DR's ``_check_design_effect`` and
            # REPORTING.md's ``trivial`` band definition
            # ``0.95 <= deff < 1.05`` (half-open). The prior closed
            # interval ``<= 1.05`` produced ``is_trivial=True`` at
            # exactly ``deff == 1.05`` while the DR schema emitted
            # ``band_label="slightly_reduces"`` for the same value,
            # suppressing BR's non-trivial prose at that boundary.
            "is_trivial": deff is not None and 0.95 <= deff < 1.05,
            "n_strata": _safe_int(getattr(sm, "n_strata", None)),
            "n_psu": _safe_int(getattr(sm, "n_psu", None)),
            "df_survey": _safe_int(getattr(sm, "df_survey", None)),
            "replicate_method": getattr(sm, "replicate_method", None),
        }


# ---------------------------------------------------------------------------
# Schema helpers (module-private)
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_ci(ci: Any) -> Optional[List[float]]:
    if ci is None:
        return None
    try:
        lo, hi = ci
    except (TypeError, ValueError):
        return None
    lo_f = _safe_float(lo)
    hi_f = _safe_float(hi)
    if lo_f is None or hi_f is None:
        return None
    return [lo_f, hi_f]


def _safe_list_len(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(len(val))
    except TypeError:
        return None


def _lift_pre_trends(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull pre-trends + power into a single BR-facing block."""
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    pt = dr.get("parallel_trends") or {}
    pp = dr.get("pretrends_power") or {}
    if pt.get("status") != "ran":
        return {
            "status": pt.get("status", "not_run"),
            "reason": pt.get("reason"),
            # DR attaches derived-surface provenance to gate-skipped
            # sections too (a successful-but-empty derivation) — the skip
            # path must not drop it (None on raw routes, same as below).
            "pre_period_source": pt.get("pre_period_source"),
        }
    return {
        "status": "computed",
        "method": pt.get("method"),
        "joint_p_value": pt.get("joint_p_value"),
        "verdict": pt.get("verdict"),
        "n_pre_periods": pt.get("n_pre_periods"),
        # Preserve DR's inconclusive-PT provenance on the BR schema so
        # downstream consumers (and BR's own summary renderer) see the
        # undefined-row count and DR's detailed reason without having
        # to re-consult the DR schema (round-39 P3 CI review on PR
        # #318). These fields are populated only when
        # ``verdict == "inconclusive"`` per ``_pt_event_study``'s
        # inconclusive branch (``diagnostic_report.py:999``).
        "n_dropped_undefined": pt.get("n_dropped_undefined"),
        # Provenance of the pre-period surface: "aggregate_event_study"
        # when DR derived it from the post-fit
        # ``results.aggregate('event_study')`` container. BR always emits
        # the key — ``None`` on raw-field routes (DR itself omits the key
        # there; ``dict.get`` maps that to None). Lifted explicitly for
        # the same reason as ``n_dropped_undefined`` — this function is a
        # field whitelist.
        "pre_period_source": pt.get("pre_period_source"),
        "reason": pt.get("reason"),
        # Carry the denominator df through when the survey F-reference
        # branch was used so BR consumers can flag the finite-sample
        # correction without re-consulting the DR schema (round-28 P3
        # CI review on PR #318).
        "df_denom": pt.get("df_denom"),
        "power_status": pp.get("status"),
        # Dedicated reason field so schema consumers see the fallback
        # explanation when ``compute_pretrends_power`` cannot run
        # (``status in {"skipped", "error", "not_applicable"}``).
        # REPORTING.md lines 118-125 promise this provenance; round-29
        # P3 CI review on PR #318 flagged that only the enum status was
        # being exposed and the reason was dropped at the lift boundary.
        # ``power_status`` stays the machine-readable enum; ``power_reason``
        # carries the plain-English explanation.
        "power_reason": pp.get("reason"),
        "power_tier": pp.get("tier"),
        "mdv": pp.get("mdv"),
        # Level-scale max pre-period violation under the MDV
        # (PR-B R12: `mdv * max(|violation_weights|)`). Carried alongside
        # the raw `mdv` so BR schema consumers and the full-report
        # renderer can show both quantities. Pre-R14 this was silently
        # dropped at the BR lift boundary so the new renderer line never
        # fired even though DR emitted the value.
        "max_abs_pre_violation": pp.get("max_abs_pre_violation"),
        "mdv_share_of_att": pp.get("mdv_share_of_att"),
        # Carry the covariance-source annotation through so BR can hedge the
        # power-tier phrasing when compute_pretrends_power silently used a
        # diagonal fallback despite event_study_vcov being available.
        "power_covariance_source": pp.get("covariance_source"),
    }


def _lift_sensitivity(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    sens = dr.get("sensitivity") or {}
    if sens.get("status") != "ran":
        # Preserve ``method`` through to the BR schema so downstream
        # consumers can distinguish a native-routed skip
        # (``method="estimator_native"`` for SDiD / TROP, where
        # robustness is covered by the native battery) from a
        # methodology-blocked skip (e.g., CS with
        # ``base_period='varying'``). Without it, agents reading the BR
        # schema alone cannot tell these cases apart and would have to
        # re-consult the DR schema to disambiguate.
        return {
            "status": sens.get("status", "not_run"),
            "reason": sens.get("reason"),
            "method": sens.get("method"),
        }
    return {
        "status": "computed",
        "method": sens.get("method"),
        "breakdown_M": sens.get("breakdown_M"),
        "conclusion": sens.get("conclusion"),
        "grid": sens.get("grid"),
    }


def _lift_heterogeneity(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the heterogeneity section of the BR schema.

    Round-31 P2 CI review on PR #318: the lift previously returned
    ``None`` on any non-``ran`` path, which broke the schema contract
    that every top-level BR key resolves to a dict with a ``status``
    field. Downstream consumers had to special-case this one section.
    Now returns a dict-shaped ``{"status": ..., "reason": ...}`` block
    mirroring DR's own status enum so ``schema["heterogeneity"]
    ["status"]`` is always readable.
    """
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    het = dr.get("heterogeneity") or {}
    status = het.get("status")
    if status != "ran":
        return {
            "status": status or "not_run",
            "reason": het.get("reason"),
        }
    return {
        "status": "ran",
        "source": het.get("source"),
        "n_effects": het.get("n_effects"),
        "min": het.get("min"),
        "max": het.get("max"),
        "cv": het.get("cv"),
        "sign_consistent": het.get("sign_consistent"),
    }


def _lift_robustness(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    bacon = dr.get("bacon") or {}
    native = dr.get("estimator_native_diagnostics") or {}
    native_block = {
        "status": native.get("status"),
        "estimator": native.get("estimator"),
        "pre_treatment_fit": native.get("pre_treatment_fit"),
    }
    # Classic SCM exposes pre_rmspe + donor-weight concentration + the (opt-in)
    # in-space placebo rather than SDiD's pre_treatment_fit; surface those so the
    # top-level robustness block is not empty for SyntheticControl.
    if native.get("estimator") == "SyntheticControl":
        native_block["pre_rmspe"] = native.get("pre_rmspe")
        native_block["weight_concentration"] = native.get("weight_concentration")
        native_block["in_space_placebo"] = native.get("in_space_placebo")
        # ADH-2015 robustness diagnostics (opt-in; "not_run" stub until run).
        native_block["leave_one_out"] = native.get("leave_one_out")
        native_block["in_time_placebo"] = native.get("in_time_placebo")
    return {
        "bacon": {
            "status": bacon.get("status"),
            "forbidden_weight": bacon.get("forbidden_weight"),
            "verdict": bacon.get("verdict"),
        },
        "estimator_native": native_block,
    }


def _anticipation_periods(results: Any) -> int:
    """Return the non-negative anticipation-period count from a result, or 0.

    Helper for ``_describe_assumption``. Anticipation-capable estimators
    (MultiPeriodDiD, CS, SA, ImputationDiD, TwoStageDiD, Stacked, EfficientDiD,
    StaggeredTripleDiff, ContinuousDiD, Wooldridge) expose ``anticipation``
    as an int defaulting to ``0``.
    """
    a = getattr(results, "anticipation", 0)
    try:
        k = int(a)
    except (TypeError, ValueError):
        return 0
    return k if k > 0 else 0


def _control_group_choice(results: Any) -> Optional[str]:
    """Return the control-group choice string for a fitted result, normalized
    across estimator-specific attribute names.

    Every anticipation-capable estimator now exposes the control-group
    choice as ``results.control_group`` (``StackedDiDResults`` joined via
    row M-095; its ``__setstate__`` migrates pre-rename pickles, so no
    ``clean_control`` fallback is needed - reading the deprecated alias
    here would fire its FutureWarning on every report build).
    """
    cg = getattr(results, "control_group", None)
    if isinstance(cg, str):
        return cg
    return None


_STRICT_NO_ANTICIPATION_PATTERNS = (
    # Ordered from most specific to least specific so the first match
    # wins on strings that could match multiple patterns. Matches are
    # case-sensitive because every occurrence in ``_describe_assumption``
    # is a fixed canonical phrase.
    ", plus no anticipation",
    "plus no anticipation",
    " Also assumes no anticipation (Assumption NA), overlap "
    "(Assumption O), and absorbing / irreversible treatment.",
    " Also assumes no anticipation.",
    "Also assumes no anticipation.",
    " and no anticipation",
)


def _strip_strict_no_anticipation(desc: str) -> str:
    """Remove any strict no-anticipation phrasing from ``desc``.

    Several base assumption descriptions in ``_describe_assumption``
    hard-code a strict "plus no anticipation" / "Also assumes no
    anticipation" clause (CS / SA / Imputation / TwoStage / Wooldridge
    generic, StackedDiD sub-experiment, EfficientDiD PT-Post, EfficientDiD
    PT-All, ContinuousDiD, TripleDifference, SyntheticDiD, TROP, dCDH,
    and the fallback unconditional branch). When a fit actually allows
    anticipation the helper must REPLACE that wording, not append a
    contradictory clause on top of it. Round-30 P1 CI review on PR #318.
    """
    if not desc:
        return desc
    out = desc
    for pattern in _STRICT_NO_ANTICIPATION_PATTERNS:
        out = out.replace(pattern, "")
    # Collapse any doubled whitespace or dangling punctuation left by
    # the removal (e.g., "cohorts,  with..." -> "cohorts, with...";
    # "cohorts .  " -> "cohorts.").
    out = re.sub(r"\s+\.", ".", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r" {2,}", " ", out)
    return out.strip()


def _apply_anticipation_to_assumption(block: Dict[str, Any], results: Any) -> Dict[str, Any]:
    """If the fit used ``anticipation > 0``, flip ``no_anticipation`` off,
    strip any strict no-anticipation wording from the base description,
    and append an anticipation-aware clause.

    Round-17 CI review flagged the strict "plus no anticipation" language
    on anticipation-enabled fits. Per REGISTRY.md §CallawaySantAnna lines
    355-395 and the matching sections for SA / MultiPeriod / Wooldridge /
    EfficientDiD, a fit with ``anticipation=k`` shifts the effective
    treatment boundary by ``k`` pre-periods; the identifying assumption
    becomes "no treatment effects earlier than ``k`` periods before the
    treatment start" rather than strict no-anticipation. Round-30 CI
    review caught that the previous implementation only appended — the
    resulting prose said both "strict no-anticipation holds" and
    "anticipation is allowed" in the same paragraph.
    """
    k = _anticipation_periods(results)
    if k <= 0:
        return block
    block = dict(block)  # don't mutate the caller's dict
    block["no_anticipation"] = False
    block["anticipation_periods"] = k
    period_word = "period" if k == 1 else "periods"
    clause = (
        f" Anticipation is allowed for the {k} {period_word} immediately "
        "before treatment: the identifying contract requires no treatment "
        f"effects earlier than {k} {period_word} before the treatment "
        "start (not strict no-anticipation)."
    )
    desc = block.get("description", "")
    if isinstance(desc, str):
        block["description"] = _strip_strict_no_anticipation(desc) + clause
    return block


def _describe_assumption(estimator_name: str, results: Any = None) -> Dict[str, Any]:
    """Return the identifying-assumption block for an estimator."""
    if estimator_name in {
        "SyntheticDiDResults",
    }:
        return {
            "parallel_trends_variant": "weighted_pt",
            "no_anticipation": True,
            "description": (
                "Synthetic-Difference-in-Differences identifies the ATT under a "
                "weighted parallel-trends analogue: the synthetic control is "
                "chosen to match the treated group's pre-period trajectory."
            ),
        }
    if estimator_name in {"TROPResults"}:
        return {
            "parallel_trends_variant": "factor_model",
            "no_anticipation": True,
            "description": (
                "TROP uses low-rank factor-model identification rather than a "
                "parallel-trends assumption; unobserved heterogeneity is "
                "captured through latent factor loadings."
            ),
        }
    if estimator_name in {"SyntheticControlResults"}:
        return {
            # Distinct from SDiD's "synthetic_fit" weighted-PT analogue: classic
            # SCM is a donor-weighted level match (matches the DR "scm_fit" method).
            "parallel_trends_variant": "scm_fit",
            "no_anticipation": True,
            "description": (
                "Classic synthetic control identifies the single treated unit's "
                "counterfactual via a donor-weighted match to its pre-treatment "
                "trajectory (a design-enforced fit, not a parallel-trends test); "
                "significance comes from in-space placebo permutation inference "
                "rather than an analytical standard error."
            ),
        }
    if estimator_name == "ContinuousDiDResults":
        # Callaway, Goodman-Bacon & Sant'Anna (2024), two-level PT:
        # REGISTRY.md §ContinuousDiD > Identification.
        return {
            "parallel_trends_variant": "dose_pt_or_strong_pt",
            "no_anticipation": True,
            "description": (
                "ContinuousDiD identifies dose-specific treatment effects "
                "under two possible parallel-trends conditions (Callaway, "
                "Goodman-Bacon & Sant'Anna 2024). Parallel Trends (PT) "
                "assumes untreated potential outcome paths are equal across "
                "all dose groups and the untreated group (conditional on "
                "dose), identifying ATT(d|d) and the binarized ATT^loc but "
                "NOT ATT(d), ACRT, or cross-dose comparisons. Strong "
                "Parallel Trends (SPT) additionally rules out selection "
                "into dose on the basis of treatment effects and is "
                "required to identify the dose-response curve ATT(d), "
                "marginal effect ACRT(d), and cross-dose contrasts."
            ),
        }
    if estimator_name in {"TripleDifferenceResults", "StaggeredTripleDiffResults"}:
        # Ortiz-Villavicencio & Sant'Anna (2025) — identification is the
        # triple-difference cancellation across the 2x2x2 cells, not
        # ordinary DiD parallel trends; see REGISTRY.md §TripleDifference
        # and §StaggeredTripleDifference.
        return {
            "parallel_trends_variant": "triple_difference_cancellation",
            "no_anticipation": True,
            "description": (
                "Triple-difference identification relies on the DDD "
                "decomposition (Ortiz-Villavicencio & Sant'Anna 2025): "
                "the ATT is recovered from `DDD = DiD_A + DiD_B - DiD_C` "
                "across the Group x Period x Eligibility (or Treatment) "
                "cells, which differences out group-specific and "
                "period-specific unobservables without requiring separate "
                "parallel trends to hold between each cell pair. The "
                "identifying restriction is therefore weaker than ordinary "
                "DiD parallel trends but assumes that the residual "
                "unobservable component is additively separable across the "
                "three dimensions; practical overlap and common-support "
                "conditions still apply on the propensity score when "
                "covariates are used."
            ),
        }
    if estimator_name == "ChaisemartinDHaultfoeuilleResults":
        # de Chaisemartin & D'Haultfoeuille (2020, 2024) — identification is
        # transition-based across (joiner, leaver, stable-control) cells
        # around each switching period, not a group-time ATT parallel-
        # trends restriction. Writing up dCDH as "parallel trends across
        # treatment cohorts" was flagged as a source-faithfulness bug in
        # PR #318 review; REGISTRY.md §ChaisemartinDHaultfoeuille is
        # explicit about the transition-set construction.
        #
        # Phase-3 features (``controls``, ``trends_linear``,
        # ``heterogeneity``) each modify the identifying contract and
        # change the estimand from ``DID_l`` to ``DID^X_l`` /
        # ``DID^{fd}_l`` / the heterogeneity-test variant. When active,
        # append an explicit clause so the description does not
        # misrepresent the identifying assumption (the reviewer has
        # flagged several parallel source-faithfulness gaps elsewhere
        # — explicitly surfacing Phase-3 config matches the per-estimator
        # walkthrough pattern).
        base_description = (
            "Identification is transition-based (de Chaisemartin & "
            "D'Haultfoeuille 2020; dynamic companion 2024). At each "
            "switching period, the estimator contrasts joiners "
            "(D:0->1), leavers (D:1->0), and stable-treated / "
            "stable-untreated control cells that share the same "
            "treatment state across adjacent periods, yielding the "
            "contemporaneous ``DID_M`` and per-horizon ``DID_l`` / "
            "``DID_{g,l}`` building blocks. The identifying "
            "restriction is parallel trends within each transition's "
            "stable-control cell (not a single group-time ATT PT "
            "condition across all cohorts) plus no anticipation; "
            "with non-binary treatment the stable-control match is "
            "additionally on exact baseline dose ``D_{g,1}``. "
            "Reversible treatment is natively supported, unlike the "
            "absorbing-treatment designs that rely on a fixed "
            "treatment-onset cohort."
        )
        has_controls = (
            results is not None and getattr(results, "covariate_residuals", None) is not None
        )
        # PR #347 R10 P1: read the persisted ``trends_linear`` flag
        # first — empty-horizon trends-linear fits set
        # ``linear_trends_effects=None`` but are still trends-linear
        # per the estimator contract. Legacy fit objects predating
        # the persisted field fall back to the presence inference.
        _trends_persisted = getattr(results, "trends_linear", None) if results is not None else None
        if isinstance(_trends_persisted, bool):
            has_trends = _trends_persisted
        else:
            has_trends = (
                results is not None and getattr(results, "linear_trends_effects", None) is not None
            )
        has_heterogeneity = (
            results is not None and getattr(results, "heterogeneity_effects", None) is not None
        )
        active_parts: List[str] = []
        if has_controls and has_trends:
            active_parts.append(
                "the estimand is ``DID^{X,fd}_l`` (covariate-residualized "
                "first-differences), and identification holds conditional on "
                "the covariates entering the first-stage regression and "
                "allowing group-specific linear trends"
            )
        elif has_controls:
            active_parts.append(
                "the estimand is ``DID^X_l``, and identification holds "
                "conditional on the covariates entering the first-stage "
                "residualization"
            )
        elif has_trends:
            active_parts.append(
                "the estimand is ``DID^{fd}_l`` (first-differenced) and the "
                "identifying restriction is relaxed to allow group-specific "
                "linear pre-trends"
            )
        if has_heterogeneity:
            active_parts.append("heterogeneity tests ``beta^{het}_l`` are reported per horizon")
        if active_parts:
            phase3_clause = " Phase-3 configuration: " + "; ".join(active_parts) + "."
            base_description = base_description + phase3_clause
        return {
            "parallel_trends_variant": "transition_based",
            "no_anticipation": True,
            "description": base_description,
        }
    if estimator_name == "EfficientDiDResults":
        # Chen, Sant'Anna & Xie (2025) — identification is parameterized
        # by ``pt_assumption`` ("all" vs "post"). PT-All is the stronger
        # regime (PT across all groups/periods, over-identified — paper
        # Lemma 2.1), PT-Post the weaker (PT only in post-treatment,
        # just-identified reduction to single-baseline DiD per Corollary
        # 3.2). Also read ``control_group`` when present (not_yet_treated
        # vs last_cohort) to be source-faithful to REGISTRY.md §EfficientDiD
        # lines 736-738 and 907.
        pt_assumption = getattr(results, "pt_assumption", "all")
        control_group = getattr(results, "control_group", None)
        # The estimator only accepts ``control_group`` values of
        # ``"never_treated"`` (the default) or ``"last_cohort"``. When
        # ``last_cohort`` is used, the latest treatment cohort is
        # reclassified as a pseudo-never-treated comparison and time
        # periods at/after its onset are dropped; describing such a fit
        # with generic never-treated language would misstate the
        # identifying setup (see REGISTRY.md §EfficientDiD line 908).
        is_last_cohort = control_group == "last_cohort"
        if pt_assumption == "post":
            variant = "pt_post"
            if is_last_cohort:
                control_clause = (
                    "the comparison group is the latest treated cohort "
                    "reclassified as pseudo-never-treated (periods "
                    "at/after that cohort's treatment start are "
                    "dropped)"
                )
            else:
                control_clause = "the comparison group is never-treated"
            description = (
                "Identification under PT-Post (Chen, Sant'Anna & Xie "
                "2025): parallel trends holds only in post-treatment "
                "periods, " + control_clause + ", and the baseline is period g-1 only. This is the "
                "weaker of the two regimes — just-identified and "
                "reducing to standard single-baseline DiD (Corollary "
                "3.2). Also assumes no anticipation (Assumption NA), "
                "overlap (Assumption O), and absorbing / irreversible "
                "treatment."
            )
        else:
            variant = "pt_all"
            if is_last_cohort:
                baseline_clause = (
                    "using the latest treated cohort as a pseudo-never-"
                    "treated comparison (periods at/after that cohort's "
                    "treatment start are dropped); any earlier cohort "
                    "and any pre-treatment period can serve as baseline"
                )
            else:
                baseline_clause = (
                    "using never-treated units as comparison; any "
                    "not-yet-treated cohort and any pre-treatment period "
                    "can serve as baseline"
                )
            description = (
                "Identification under PT-All (Chen, Sant'Anna & Xie "
                "2025): parallel trends holds for all groups and all "
                "periods, "
                + baseline_clause
                + ". The estimator is over-identified (Lemma 2.1), and "
                "the paper's optimal combination weights are applied. "
                "Also assumes no anticipation (Assumption NA), overlap "
                "(Assumption O), and absorbing / irreversible "
                "treatment. The Hausman PT-All vs PT-Post pretest "
                "(operating on the post-treatment event-study vector "
                "ES(e), Theorem A.1) checks whether the stronger "
                "PT-All regime is tenable."
            )
        block: Dict[str, Any] = {
            "parallel_trends_variant": variant,
            "no_anticipation": True,
            "description": description,
        }
        if isinstance(control_group, str):
            block["control_group"] = control_group
        return block
    if estimator_name == "StackedDiDResults":
        # Wing, Freedman & Hollingsworth (2024) — identification is
        # sub-experiment common trends plus the IC1 (event window fits
        # within the data range) and IC2 (clean controls exist for the
        # event) inclusion conditions, NOT the generic "group-time ATT
        # parallel trends" clause used for CS / SA / etc. (round-22 P1
        # CI review on PR #318). The active ``control_group`` rule
        # (pre-M-095 name: ``clean_control``) determines which units
        # qualify as valid controls for each adoption event. REGISTRY.md
        # §StackedDiD (identification / clean-control rules).
        clean_control = getattr(results, "control_group", None)
        if clean_control == "never_treated":
            control_clause = (
                "controls are restricted to units that are never treated "
                "over the panel (``A_s = infinity``)"
            )
        elif clean_control == "strict":
            control_clause = (
                "controls for event ``a`` are units satisfying the strict "
                "rule ``A_s > a + kappa_post + kappa_pre`` (strictly "
                "untreated across the full pre- and post-event window)"
            )
        else:
            # Default: "not_yet_treated" — A_s > a + kappa_post.
            control_clause = (
                "controls for event ``a`` are units satisfying ``A_s > a + "
                "kappa_post`` (not yet treated through the end of the "
                "event's post-window, so future-treated units can serve "
                "as controls for earlier events)"
            )
        block = {
            "parallel_trends_variant": "stacked_sub_experiment",
            "no_anticipation": True,
            "description": (
                "Identification under Stacked DiD (Wing, Freedman & "
                "Hollingsworth 2024): within each stacked sub-experiment "
                "parallel trends holds between the treated cohort and the "
                "corresponding clean-control set over the event window "
                "``[-kappa_pre, +kappa_post]``; "
                + control_clause
                + ". Sub-experiments are restricted by IC1 (the event "
                "window fits within the available time range) and IC2 "
                "(at least one clean control exists). The aggregate ATT is "
                "a weighted sum over sub-experiments, so the common-trends "
                "assumption is sub-experiment-specific, not a single "
                "panel-wide group-time ATT condition. Also assumes no "
                "anticipation."
            ),
        }
        if isinstance(clean_control, str):
            block["control_group"] = clean_control
            # Deprecated reporting key kept through the 3.9 shim window;
            # retired at 4.0 (row M-095, section 5 policy).
            block["clean_control"] = clean_control
        return block
    if estimator_name == "ImputationDiDResults":
        # Borusyak, Jaravel & Spiess (2024) — identification is through
        # an untreated-potential-outcome model: unit+time FE (optionally
        # plus covariates) fitted on untreated observations only
        # (``Omega_0``) deliver the counterfactual ``Y_it(0)``, and the
        # treatment effect ``tau_it`` is the residual on treated
        # observations. Writing this as generic "group-time ATT
        # parallel trends" misstates the identifying model — the
        # restriction is on the UNTREATED outcome's additive FE
        # structure, not on cohort-time ATT equality. REGISTRY.md
        # §ImputationDiD lines 1000-1013 and Assumption 1 (parallel
        # trends) + Assumption 2 (no anticipation on untreated
        # observations). Round-42 P1 CI review on PR #318 flagged this
        # source-faithfulness gap.
        return {
            "parallel_trends_variant": "untreated_outcome_fe_model",
            "no_anticipation": True,
            "description": (
                "Identification under Imputation DiD (Borusyak, Jaravel "
                "& Spiess 2024): the untreated potential outcome "
                "``Y_it(0)`` follows an additive unit+time fixed-effects "
                "model ``Y_it(0) = alpha_i + beta_t [+ X'_it * delta] + "
                "epsilon_it``. Step 1 estimates those FE on untreated "
                "observations only (``Omega_0`` = never-treated plus "
                "not-yet-treated cells); Step 2 imputes the "
                "counterfactual for treated observations from the "
                "fitted FE; Step 3 aggregates ``tau_hat_it = Y_it - "
                "Y_hat_it(0)`` with researcher-chosen weights. The "
                "identifying restriction is therefore parallel trends "
                "of the UNTREATED outcome model (Assumption 1) — "
                "``E[Y_it(0)] = alpha_i + beta_t``, holding across all "
                "observations — rather than equality of cohort-time "
                "ATTs. Also assumes no anticipation on untreated "
                "observations (Assumption 2) and absorbing treatment."
            ),
        }
    if estimator_name == "TwoStageDiDResults":
        # Gardner (2022) — identification is the same as BJS
        # ImputationDiD (point estimates are algebraically equivalent
        # per REGISTRY.md §TwoStageDiD line 1130): unit+time FE
        # estimated on untreated observations only deliver the
        # untreated potential-outcome trajectory; Stage 2 regresses
        # the resulting residuals on treatment indicators. Writing
        # this as generic "group-time ATT parallel trends" loses the
        # load-bearing detail that Stage 1 operates only on untreated
        # cells. REGISTRY.md §TwoStageDiD lines 1113-1128 and
        # Assumption (same as ImputationDiD). Round-42 P1 CI review on
        # PR #318 flagged this source-faithfulness gap.
        return {
            "parallel_trends_variant": "untreated_outcome_fe_model",
            "no_anticipation": True,
            "description": (
                "Identification under Two-Stage DiD (Gardner 2022): "
                "Stage 1 fits unit + time fixed effects on untreated "
                "observations only (``Omega_0``), residualizing the "
                "outcome as ``y_tilde_it = Y_it - alpha_hat_i - "
                "beta_hat_t``; Stage 2 regresses residualized outcomes "
                "on the treatment indicator across treated observations "
                "to recover the ATT. The point estimates are "
                "algebraically equivalent to Borusyak-Jaravel-Spiess "
                "imputation (both rely on the same untreated-outcome FE "
                "model to construct the counterfactual). The "
                "identifying restriction is therefore parallel trends "
                "of the UNTREATED outcome: ``E[Y_it(0)] = alpha_i + "
                "beta_t`` for all observations (not a group-time ATT "
                "equality across cohorts). Also assumes no anticipation "
                "(``Y_it = Y_it(0)`` for all untreated observations) "
                "and absorbing / irreversible treatment."
            ),
        }
    if estimator_name == "DMLDiDResults":
        is_rcs = getattr(results, "panel", True) is False
        if is_rcs:
            tail = (
                "cross-fitting (DML2) removing own-observation overfitting. "
                "The repeated-cross-section design (Chang Case 2) "
                "additionally assumes STATIONARY cross-sectional sampling "
                "(Assumption 2.3: each wave samples the same target "
                "population, so the composition of (D, X) is stable across "
                "waves while outcomes are the period-specific potential "
                "outcomes — not data-checkable), and valid "
                "normal inference requires the nuisance learners to satisfy "
                "Chang (2020)'s Case 2 rate conditions (Assumption 3.2(h))."
            )
        else:
            tail = (
                "cross-fitting (DML2) removing own-observation overfitting; "
                "valid normal inference additionally requires the nuisance "
                "learners to satisfy Chang (2020)'s rate conditions "
                "(Assumption 3.1(f))."
            )
        return {
            "parallel_trends_variant": "conditional_on_covariates",
            "no_anticipation": True,
            "description": (
                "Identification relies on CONDITIONAL parallel trends "
                "(Abadie 2005 / Chang 2020): untreated potential-outcome "
                "trends are parallel only after conditioning on the "
                "supplied covariates, per treatment cohort and period "
                "(group-time ATT), plus no anticipation. The "
                "Neyman-orthogonal score makes the estimate first-order "
                "insensitive to machine-learning regularization bias, with " + tail
            ),
        }
    if estimator_name in {
        "CallawaySantAnnaResults",
        "SunAbrahamResults",
        "WooldridgeDiDResults",
    }:
        return {
            "parallel_trends_variant": "conditional_or_group_time",
            "no_anticipation": True,
            "description": (
                "Identification relies on parallel trends across treatment "
                "cohorts and time periods (group-time ATT), plus no "
                "anticipation."
            ),
        }
    return {
        "parallel_trends_variant": "unconditional",
        "no_anticipation": True,
        "description": (
            "Identification relies on the standard DiD parallel-trends "
            "assumption plus no anticipation of treatment by either group."
        ),
    }


def _build_caveats(
    _results: Any,
    headline: Dict[str, Any],
    sample: Dict[str, Any],
    dr_schema: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assemble the plain-English caveats list for the headline schema."""
    caveats: List[Dict[str, Any]] = []

    # NaN ATT is the highest-severity caveat.
    if headline.get("sign") == "undefined":
        caveats.append(
            {
                "severity": "warning",
                "topic": "estimation_failure",
                "message": (
                    "Estimation produced a non-finite effect. Inspect data "
                    "preparation and model specification before interpreting."
                ),
            }
        )

    # Alpha override could not be honored (bootstrap / finite-df inference).
    alpha_override_msg = headline.get("alpha_override_caveat")
    if isinstance(alpha_override_msg, str) and alpha_override_msg:
        caveats.append(
            {
                "severity": "info",
                "topic": "alpha_override_preserved",
                "message": alpha_override_msg,
            }
        )

    # Near-threshold p-value.
    if headline.get("near_significance_threshold"):
        caveats.append(
            {
                "severity": "info",
                "topic": "near_significance",
                "message": (
                    "The p-value is close to the conventional significance "
                    "threshold; small changes to the sample or specification "
                    "could move it either way."
                ),
            }
        )

    # Few treated units.
    nt = sample.get("n_treated")
    if nt is not None and nt <= 3:
        caveats.append(
            {
                "severity": "warning",
                "topic": "few_treated",
                "message": (
                    f"Only {nt} treated units in this fit; standard errors "
                    "rely on large-cluster asymptotics and may be unreliable. "
                    "Consider SyntheticDiD or an exact-permutation inference "
                    "alternative."
                ),
            }
        )

    # Non-trivial design effect.
    survey = sample.get("survey")
    if survey and not survey.get("is_trivial"):
        deff = survey.get("design_effect")
        eff_n = survey.get("effective_n")
        if isinstance(deff, (int, float)) and deff >= 5.0:
            caveats.append(
                {
                    "severity": "warning",
                    "topic": "design_effect",
                    "message": (
                        f"Very large survey design effect (DEFF = {deff:.2g}). "
                        "Inspect the weight distribution and consider weight "
                        "trimming if driven by outlier weights."
                    ),
                }
            )
        elif isinstance(deff, (int, float)) and deff >= 1.5:
            if isinstance(eff_n, (int, float)):
                caveats.append(
                    {
                        "severity": "info",
                        "topic": "design_effect",
                        "message": (
                            f"Survey design reduces effective sample size: "
                            f"DEFF = {deff:.2g}; effective n = {eff_n:.0f}."
                        ),
                    }
                )

    # Bacon forbidden comparisons.
    # Round-45 P1 CI review on PR #318: Goodman-Bacon is a
    # decomposition of TWFE weights (see ``bacon.py`` header and
    # Goodman-Bacon 2021). On fits already produced by a
    # heterogeneity-robust estimator (CS / SA / BJS / TwoStageDiD /
    # Wooldridge / EfficientDiD / StackedDiD / dCDH / TripleDifference /
    # StaggeredTripleDiff / SDiD / TROP), a high forbidden-weight share
    # says "TWFE would have been materially biased on this rollout",
    # not "the displayed estimator needs to be replaced" — the
    # displayed estimator is already robust to the heterogeneity that
    # Bacon flags. DR partly preserves this with "if not already in
    # use" prose; BR must carry the same distinction through to the
    # caveat. The TWFE-style estimators whose results route through
    # Bacon and for which the "switch to a robust estimator"
    # recommendation is load-bearing are the DiDResults-type fits; all
    # other result classes are already robust.
    _TWFE_STYLE_RESULTS: FrozenSet[str] = frozenset(
        {"DiDResults", "MultiPeriodDiDResults", "TwoWayFixedEffectsResults"}
    )
    if dr_schema:
        bacon = dr_schema.get("bacon") or {}
        if bacon.get("status") == "ran":
            fw = bacon.get("forbidden_weight")
            if isinstance(fw, (int, float)) and fw > 0.10:
                _estimator_name = type(_results).__name__
                if _estimator_name in _TWFE_STYLE_RESULTS:
                    bacon_message = (
                        f"Goodman-Bacon decomposition places {fw:.0%} "
                        "of implicit TWFE weight on 'forbidden' "
                        "later-vs-earlier comparisons. TWFE may be "
                        "materially biased under heterogeneous effects. "
                        "Re-estimate with a heterogeneity-robust "
                        "estimator (CallawaySantAnna, SunAbraham, "
                        "ImputationDiD, or TwoStageDiD)."
                    )
                else:
                    bacon_message = (
                        f"Goodman-Bacon decomposition places {fw:.0%} "
                        "of TWFE weight on 'forbidden' later-vs-earlier "
                        "comparisons. A TWFE benchmark on this rollout "
                        "would be materially biased under heterogeneous "
                        "effects; the displayed estimator is already "
                        "heterogeneity-robust, so this is a statement "
                        "about the rollout design (avoid reporting TWFE "
                        "alongside this fit), not about the current "
                        "result's validity."
                    )
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "bacon_contamination",
                        "message": bacon_message,
                    }
                )

        # Fragile sensitivity.
        sens = dr_schema.get("sensitivity") or {}
        if sens.get("status") == "ran":
            bkd = sens.get("breakdown_M")
            if isinstance(bkd, (int, float)) and bkd < 0.5:
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "sensitivity_fragility",
                        "message": (
                            f"HonestDiD breakdown value is {bkd:.2g}: the "
                            "result's confidence interval includes zero "
                            "once parallel-trends violations reach less than "
                            "half the observed pre-period variation. Treat "
                            "the headline as tentative."
                        ),
                    }
                )

        # Sensitivity was skipped for methodology reasons (e.g., CS fit with
        # ``base_period='varying'`` — HonestDiD bounds are not interpretable
        # there). Surface the reason as a warning-severity caveat so readers
        # do not assume the headline is robust across the R-R grid.
        #
        # Exception (round-20 P2 CI review on PR #318): SDiD and TROP route
        # robustness to ``estimator_native_diagnostics`` and mark the HonestDiD
        # sensitivity block ``status="skipped", method="estimator_native"``.
        # Surfacing "sensitivity was not run" as a warning contradicts the
        # documented native-routing contract when the native battery actually
        # ran. Suppress the warning and point readers at the native block
        # instead.
        if sens.get("status") == "skipped":
            reason = sens.get("reason")
            method = sens.get("method")
            native = dr_schema.get("estimator_native_diagnostics") or {}
            native_ran = native.get("status") == "ran"
            if method == "estimator_native" and native_ran:
                caveats.append(
                    {
                        "severity": "info",
                        "topic": "sensitivity_native_routed",
                        "message": (
                            "HonestDiD was not run for this estimator. Robustness "
                            "is covered by the estimator-native sensitivity "
                            "diagnostics reported under "
                            "``estimator_native_diagnostics``."
                        ),
                    }
                )
            elif isinstance(reason, str) and reason:
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "sensitivity_skipped",
                        "message": ("HonestDiD sensitivity was not run on this fit. " + reason),
                    }
                )

        # Non-fatal warnings captured from delegated diagnostics
        # (e.g., HonestDiD's bootstrap diag-covariance fallback, dropped
        # non-consecutive horizons on dCDH). DR already records these in
        # ``schema["warnings"]``; mirror the methodology-critical ones
        # into BR's caveat list so summary/full-report prose can surface
        # them without readers having to inspect the DR schema.
        for msg in dr_schema.get("warnings", []) or []:
            if not isinstance(msg, str) or not msg:
                continue
            # Skip alpha-override and design-effect messages already
            # covered by dedicated caveats above.
            lower = msg.lower()
            if "sensitivity:" in lower or "pretrends_power:" in lower:
                caveats.append(
                    {
                        "severity": "info",
                        "topic": "diagnostic_warning",
                        "message": msg,
                    }
                )

    # Unit mismatch caveat (log_points + unit override).
    unit_kind = headline.get("unit_kind")
    if unit_kind == "log_points":
        caveats.append(
            {
                "severity": "info",
                "topic": "unit_policy",
                "message": (
                    "The effect is reported in log-points as estimated; "
                    "BusinessReport does not arithmetically translate log-points "
                    "to percent or level changes. For small effects, log-points "
                    "approximate percentage changes."
                ),
            }
        )
    return caveats


def _pt_method_subject(method: Optional[str]) -> str:
    """Return a source-faithful sentence subject for the PT verdict prose.

    The ``parallel_trends.method`` field distinguishes between the
    2x2 slope-difference check, the pre-period event-study Wald /
    Bonferroni variants, EfficientDiD's Hausman PT-All vs PT-Post
    pretest, SDiD's weighted pre-treatment fit, and TROP's factor-
    model identification. Generic "pre-treatment event-study" wording
    is wrong for the first and third cases. See round-8 CI review on
    PR #318 and REGISTRY.md §EfficientDiD (Hausman pretest).
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
        # Survey-aware event-study PT variants use an F reference
        # distribution with denominator df = ``survey_metadata.df_survey``
        # (round-27 P1 fix, documented in REPORTING.md). The subject
        # remains the pre-period event-study coefficients; prose elsewhere
        # flags the finite-sample correction via ``df_denom``.
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


def _pt_method_stat_label(method: Optional[str]) -> Optional[str]:
    """Return the joint-statistic label appropriate to the PT method.

    Returns ``"joint p"`` for Wald / Bonferroni paths (including the
    survey-aware F-reference variants, which remain joint tests on the
    pre-period coefficient vector — only the reference distribution
    changes), ``"p"`` for the 2x2 slope-difference and Hausman paths
    (single-statistic tests), and ``None`` for design-enforced paths
    that have no p-value.
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
        # Design-enforced fit-based paths have no p-value label (SCM's significance
        # is the in-space placebo, not a PT joint test).
        return None
    return "joint p"


def _references_for(estimator_name: str) -> List[Dict[str, str]]:
    """Map the estimator to the appropriate citation references."""
    base = [
        {
            "role": "sensitivity",
            "citation": (
                "Rambachan, A., & Roth, J. (2023). A More Credible Approach "
                "to Parallel Trends. Review of Economic Studies."
            ),
        },
        {
            "role": "workflow",
            "citation": (
                "Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., "
                "& Sant'Anna, P. H. C. (2025). Difference-in-Differences "
                "Designs: A Practitioner's Guide."
            ),
        },
    ]
    estimator_refs = {
        "CallawaySantAnnaResults": {
            "role": "estimator",
            "citation": (
                "Callaway, B., & Sant'Anna, P. H. C. (2021). "
                "Difference-in-Differences with multiple time periods. "
                "Journal of Econometrics."
            ),
        },
        "SyntheticDiDResults": {
            "role": "estimator",
            "citation": (
                "Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., "
                "& Wager, S. (2021). Synthetic Difference in Differences."
            ),
        },
        "SyntheticControlResults": {
            "role": "estimator",
            "citation": (
                "Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic "
                "Control Methods for Comparative Case Studies. JASA, 105(490)."
            ),
        },
        "SunAbrahamResults": {
            "role": "estimator",
            "citation": (
                "Sun, L., & Abraham, S. (2021). Estimating dynamic treatment "
                "effects in event studies. Journal of Econometrics."
            ),
        },
        "DMLDiDResults": {
            "role": "estimator",
            "citation": (
                "Chang, N.-C. (2020). Double/debiased machine learning for "
                "difference-in-differences models. The Econometrics "
                "Journal, 23(2), 177-191."
            ),
        },
        "ImputationDiDResults": {
            "role": "estimator",
            "citation": (
                "Borusyak, K., Jaravel, X., & Spiess, J. (2024). " "Revisiting event-study designs."
            ),
        },
        "EfficientDiDResults": {
            "role": "estimator",
            "citation": (
                "Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). "
                "Efficient Estimation of Treatment Effects in Staggered "
                "DiD Designs."
            ),
        },
        "ChaisemartinDHaultfoeuilleResults": {
            "role": "estimator",
            "citation": (
                "de Chaisemartin, C., & D'Haultfœuille, X. (2020). "
                "Two-way fixed effects estimators with heterogeneous "
                "treatment effects. American Economic Review."
            ),
        },
    }
    if estimator_name in estimator_refs:
        return [estimator_refs[estimator_name]] + base
    return base


# ---------------------------------------------------------------------------
# Prose rendering
# ---------------------------------------------------------------------------
def _format_value(value: Optional[float], unit: Optional[str], unit_kind: str) -> str:
    """Format a numeric effect with its unit. No arithmetic translation."""
    if value is None or not np.isfinite(value):
        return "undefined"
    if unit_kind == "currency":
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.2f}"
    if unit_kind == "percent":
        return f"{value:.2f}%"
    if unit_kind == "percentage_points":
        return f"{value:.2f} pp"
    if unit_kind == "log_points":
        return f"{value:.3g} log-points"
    if unit_kind == "count":
        return f"{value:,.0f}"
    # unknown / free-form
    if unit:
        return f"{value:.3g} {unit}"
    return f"{value:.3g}"


def _significance_phrase(p: Optional[float], alpha: float) -> str:
    """Return a plain-English significance phrase.

    Tiers per ``docs/methodology/REPORTING.md``:
      * p < 0.001: "strongly supported by the data"
      * 0.001 <= p < 0.01: "well-supported"
      * 0.01 <= p < alpha: "statistically significant at the X% level"
      * alpha <= p < 0.10: CI-includes-zero language
      * p >= 0.10: consistent-with-no-effect language
    """
    if p is None or not np.isfinite(p):
        return "statistical significance cannot be assessed (p-value unavailable)"
    ci_level = int(round((1.0 - alpha) * 100))
    if p < 0.001:
        return "the direction of the effect is strongly supported by the data"
    if p < 0.01:
        return "the direction of the effect is well-supported by the data"
    if p < alpha:
        return f"the effect is statistically significant at the {ci_level}% level"
    if p < 0.10:
        return (
            "the confidence interval includes zero; the direction is suggestive "
            "but not statistically significant"
        )
    return "the confidence interval includes zero; the data are consistent with no effect"


def _smallest_failing_grid_m(sens: Dict[str, Any]) -> Optional[float]:
    """If the smallest evaluated M on the HonestDiD sensitivity grid
    already has the robust CI including zero, return that M. Returns
    ``None`` when the grid is missing or when the smallest evaluated
    point is still robust — in the latter case ``breakdown_M`` is an
    interpolated threshold between grid points, not a statement about
    the smallest grid point itself.

    Matches the twin helper in ``diagnostic_report.py``; keep the two
    in sync for cross-surface parity.
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


def _sentence_first_upper(text: str) -> str:
    """Uppercase only the first character of ``text``, preserving all
    other casing. Unlike ``str.capitalize()``, which lowercases every
    character after the first, this keeps user-supplied abbreviations
    and proper nouns intact.

    Examples
    --------
    >>> _sentence_first_upper("the NJ minimum-wage increase")
    'The NJ minimum-wage increase'
    >>> _sentence_first_upper("Castle Doctrine law adoption")
    'Castle Doctrine law adoption'
    """
    if not text:
        return text
    return text[0].upper() + text[1:]


def _direction_verb(effect: float, outcome_direction: Optional[str]) -> str:
    """Return a direction-aware verb for the headline sentence.

    When ``outcome_direction`` is unset we use neutral change verbs
    (``increased`` / ``decreased``). When it is supplied, we additionally
    flavor the verb with a value-laden connotation so the stakeholder can
    read off whether the estimated effect points in the desired direction:

    - ``higher_is_better``: positive effect -> "lifted"; negative -> "reduced"
    - ``lower_is_better``:  positive effect -> "worsened"; negative -> "improved"
    - None:                 positive -> "increased"; negative -> "decreased"
    """
    if effect == 0:
        return "did not change"
    if outcome_direction == "higher_is_better":
        return "lifted" if effect > 0 else "reduced"
    if outcome_direction == "lower_is_better":
        return "worsened" if effect > 0 else "improved"
    return "increased" if effect > 0 else "decreased"


def _render_headline_sentence(schema: Dict[str, Any]) -> str:
    """Render the headline sentence from the schema.

    Uses the absolute value in the magnitude slot when the verb already
    conveys direction ("decreased ... by $0.14" rather than "decreased ...
    by -$0.14"). CI bounds are rendered at their natural signed values.
    When ``outcome_direction`` is supplied, the verb picks up a value-laden
    connotation ("lifted" / "reduced" vs neutral "increased" / "decreased").
    """
    ctx = schema.get("context", {})
    h = schema.get("headline", {})
    # PR #347 R4 P1: the dCDH ``trends_linear=True`` + ``L_max>=2``
    # configuration does not produce a scalar headline by design —
    # ``overall_att`` is intentionally NaN (per
    # ``chaisemartin_dhaultfoeuille.py:2828-2834``). Render explicit
    # "no scalar headline by design" prose instead of routing through
    # the non-finite / estimation-failure path.
    if h.get("status") == "no_scalar_by_design":
        # PR #347 R13 P1: the headline-level ``reason`` field is the
        # single source for the no-scalar prose and is already
        # branched on populated-vs-empty surface in ``_build_schema``.
        # Use it verbatim so the headline sentence never drifts from
        # the schema-level message on the empty-surface subcase.
        treatment = ctx.get("treatment_label", "the treatment")
        outcome_label = ctx.get("outcome_label", "the outcome")
        treatment_sentence = _sentence_first_upper(treatment)
        reason = h.get("reason")
        if isinstance(reason, str) and reason:
            return (
                f"{treatment_sentence} does not produce a scalar aggregate "
                f"effect on {outcome_label} under this configuration. " + reason
            )
        return (
            f"{treatment_sentence} does not produce a scalar aggregate effect "
            f"on {outcome_label} under this configuration (by design)."
        )
    effect = h.get("effect")
    outcome = ctx.get("outcome_label", "the outcome")
    treatment = ctx.get("treatment_label", "the treatment")
    outcome_direction = ctx.get("outcome_direction")
    unit = h.get("unit")
    unit_kind = h.get("unit_kind", "unknown")

    if effect is None or not np.isfinite(effect):
        return (
            f"We were unable to produce a finite estimate of {treatment}'s "
            f"effect on {outcome}. Inspect the data and model specification."
        )

    verb = _direction_verb(effect, outcome_direction)
    magnitude = _format_value(abs(effect), unit, unit_kind)
    lo = h.get("ci_lower")
    hi = h.get("ci_upper")
    # Round-37 P1 CI review on PR #318: on a finite point estimate
    # whose CI bounds are NaN (undefined inference — survey-df
    # collapse, zero effective clusters, etc.), the previous isinstance
    # check passed because ``NaN`` is a ``float`` and the sentence
    # rendered ``(... 95% CI: undefined to undefined)``. Gate on
    # ``np.isfinite`` like DR's own headline renderer already does;
    # add an explicit inference-unavailable trailer instead of the
    # broken CI clause.
    ci_str = ""
    ci_finite = (
        isinstance(lo, (int, float))
        and isinstance(hi, (int, float))
        and np.isfinite(lo)
        and np.isfinite(hi)
    )
    if ci_finite:
        lo_s = _format_value(lo, unit, unit_kind)
        hi_s = _format_value(hi, unit, unit_kind)
        ci_str = f" ({h.get('ci_level', 95)}% CI: {lo_s} to {hi_s})"
    elif isinstance(lo, (int, float)) or isinstance(hi, (int, float)):
        # At least one bound was supplied but not finite -> inference
        # undefined. Replace the CI clause with an explicit marker so
        # downstream prose does not claim a confidence interval that
        # is not actually available.
        ci_str = " (inference unavailable: confidence interval is undefined for this fit)"
    by_clause = f" by {magnitude}" if effect != 0 else ""
    # Round-1 BR/DR canonical-validation (2026-04-19): Python's
    # ``str.capitalize()`` lowercases everything except the first
    # character, so ``"the NJ minimum-wage increase".capitalize()``
    # returns ``"The nj minimum-wage increase"`` — flattening the
    # ``NJ`` abbreviation. Real canonical datasets (Card-Krueger,
    # Castle Doctrine) carry proper-noun / acronym tokens in the
    # user-supplied ``treatment_label``, so preserve user casing and
    # only ensure the first character is uppercase.
    treatment_sentence = _sentence_first_upper(treatment)
    return f"{treatment_sentence} {verb} {outcome}{by_clause}{ci_str}."


def _render_summary(schema: Dict[str, Any]) -> str:
    """Render the short-form stakeholder summary paragraph."""
    sentences: List[str] = []
    ctx = schema.get("context", {})
    question = ctx.get("business_question")
    if question:
        sentences.append(f"Question: {question}")

    # Headline sentence with significance phrase.
    sentences.append(_render_headline_sentence(schema))
    # BR/DR gap #6 (target-parameter clarity): name what the headline
    # scalar actually represents so the stakeholder can map the number
    # to a specific estimand. Rendered immediately after the headline
    # and before the significance phrase. The summary surfaces only
    # the short ``name`` so the paragraph stays within the
    # 6-10-sentence target; ``definition`` lives in the full report
    # and in the structured schema for agents that want the long form.
    tp = schema.get("target_parameter", {}) or {}
    tp_name = tp.get("name")
    if tp_name:
        sentences.append(f"Target parameter: {tp_name}.")
    h = schema.get("headline", {})
    p = h.get("p_value")
    alpha = ctx.get("alpha", 0.05)
    if p is not None and np.isfinite(p):
        sig = _significance_phrase(p, alpha)
        sentences.append(f"Statistically, {sig}.")
        if h.get("near_significance_threshold"):
            sentences.append(
                "The p-value is close to the conventional threshold; "
                "small changes to the sample could move it either way."
            )

    # Pre-trends + power-aware phrasing.
    pt = schema.get("pre_trends", {}) or {}
    if pt.get("status") == "computed":
        jp = pt.get("joint_p_value")
        verdict = pt.get("verdict")
        # ``tier`` already incorporates the diagonal-fallback downgrade —
        # ``DiagnosticReport._check_pretrends_power`` applies it centrally
        # so every report surface (BR summary, BR full_report, BR schema,
        # DR summary) reads the same adjusted value (round-14 CI review).
        tier = pt.get("power_tier")
        method = pt.get("method")
        subject = _pt_method_subject(method)
        stat_label = _pt_method_stat_label(method)
        jp_phrase = (
            f" ({stat_label} = {jp:.3g})" if isinstance(jp, (int, float)) and stat_label else ""
        )
        # Only point to "the sensitivity analysis below" when a
        # sensitivity block actually ran. For estimators that route to
        # native diagnostics (SDiD / TROP) or fits where sensitivity was
        # skipped / not applicable, the clause would mislead (round-12
        # CI review on PR #318).
        sens_ran = (schema.get("sensitivity", {}) or {}).get("status") == "computed"
        sens_tail_major = " pending the sensitivity analysis below" if sens_ran else ""
        sens_tail_alongside = " alongside the sensitivity analysis below" if sens_ran else ""
        sens_tail_see_bounded = (
            " See the sensitivity analysis below for bounded-violation guarantees."
            if sens_ran
            else ""
        )
        sens_tail_see_reliable = " See the sensitivity analysis below." if sens_ran else ""
        if verdict == "clear_violation":
            sentences.append(
                f"{subject} clearly reject parallel trends{jp_phrase}; the "
                "headline should be treated as tentative" + sens_tail_major + "."
            )
        elif verdict == "some_evidence_against":
            sentences.append(
                f"{subject} show some evidence against parallel trends"
                f"{jp_phrase}; interpret the headline"
                + (sens_tail_alongside if sens_ran else " with caution")
                + "."
            )
        elif verdict == "no_detected_violation":
            if tier == "well_powered":
                sentences.append(
                    f"{subject} are consistent with parallel trends, and "
                    "the test is well-powered (the max pre-period level "
                    "deviation at the MDV is small relative to the "
                    "estimated effect)."
                )
            elif tier == "moderately_powered":
                sentences.append(
                    f"{subject} do not reject parallel trends; the test is "
                    "moderately informative." + sens_tail_see_bounded
                )
            else:
                sentences.append(
                    f"{subject} do not reject parallel trends, but the test "
                    "has limited power — a non-rejection does not prove the "
                    "assumption." + sens_tail_see_reliable
                )
        elif verdict == "design_enforced_pt":
            if method == "scm_fit":
                sentences.append(
                    "The synthetic control is designed to reproduce the treated "
                    "unit's pre-period trajectory via donor weights (classic SCM's "
                    "design-enforced analogue of parallel trends); significance "
                    "comes from in-space placebo permutation inference, not a "
                    "parallel-trends test."
                )
            else:
                sentences.append(
                    "The synthetic control is designed to match the treated "
                    "group's pre-period trajectory (SDiD's weighted-parallel-"
                    "trends analogue)."
                )
        elif verdict == "inconclusive":
            # Round-35 P1 CI review on PR #318: a ``verdict=="inconclusive"``
            # state means one or more pre-period coefficients had
            # undefined inference (zero SE, NaN p-value) and the joint
            # test cannot be formed. BR previously omitted the sentence
            # entirely, so stakeholder prose silently skipped the
            # identifying-assumption diagnostic. Name the state
            # explicitly and quote the undefined-row count when
            # available.
            n_dropped = pt.get("n_dropped_undefined")
            if isinstance(n_dropped, int) and n_dropped > 0:
                rows_word = "row" if n_dropped == 1 else "rows"
                sentences.append(
                    f"The pre-trends test is inconclusive on this fit: "
                    f"{n_dropped} pre-period {rows_word} had undefined "
                    "inference (zero / negative SE or a non-finite "
                    "per-period p-value), so the joint test cannot be "
                    "formed. Treat parallel trends as unassessed rather "
                    "than supported."
                )
            else:
                sentences.append(
                    "The pre-trends test is inconclusive on this fit: "
                    "pre-period inference was undefined, so the joint "
                    "test cannot be formed. Treat parallel trends as "
                    "unassessed rather than supported."
                )

    # Sensitivity. A ``single_M_precomputed`` sensitivity block has
    # ``breakdown_M=None`` by construction because only one M was evaluated;
    # narrate it as a point check, NOT as grid-wide robustness.
    sens = schema.get("sensitivity", {}) or {}
    if sens.get("status") == "computed":
        bkd = sens.get("breakdown_M")
        conclusion = sens.get("conclusion")
        if conclusion == "single_M_precomputed":
            grid_points = sens.get("grid") or []
            point = grid_points[0] if grid_points else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                if robust:
                    sentences.append(
                        f"HonestDiD (single point checked): at M = {m_val:.2g}, "
                        f"the robust confidence interval excludes zero. This is "
                        f"a point check, not a breakdown analysis — run "
                        f"HonestDiD.sensitivity() across a grid of M values "
                        f"for a full robustness claim."
                    )
                else:
                    sentences.append(
                        f"HonestDiD (single point checked): at M = {m_val:.2g}, "
                        f"the robust confidence interval includes zero. Run "
                        f"HonestDiD.sensitivity() across a grid to find the "
                        f"breakdown value."
                    )
        elif bkd is None:
            sentences.append(
                "HonestDiD: the result remains significant across the "
                "full grid — robust to plausible parallel-trends violations."
            )
        elif isinstance(bkd, (int, float)) and bkd >= 1.0:
            sentences.append(
                f"HonestDiD: the result remains significant under "
                f"parallel-trends violations up to {bkd:.2g}x the observed "
                f"pre-period variation."
            )
        elif isinstance(bkd, (int, float)):
            # Round-1 BR/DR canonical-validation (2026-04-19) then
            # tightened per CI review on PR #341 R1:
            # ``breakdown_M`` is the smallest M at which the robust
            # CI includes zero (interpolated between grid points) —
            # not a claim about any specific grid point. Earlier fix
            # keyed off ``bkd <= 0.05`` which incorrectly asserted
            # "smallest grid point fails" even for grids that start
            # at M=0 where the smallest evaluated point is still
            # robust (e.g., grid=[0, 0.25, ...] with bkd=0.03). The
            # "smallest grid point" wording is only accurate when
            # the smallest evaluated M on the grid itself fails
            # (``robust_to_zero == False``); otherwise fall through
            # to the numeric multiplier.
            smallest_failed_m = _smallest_failing_grid_m(sens)
            if smallest_failed_m is not None:
                sentences.append(
                    "HonestDiD: the result is fragile — the confidence "
                    "interval includes zero even at the smallest M "
                    f"evaluated on the sensitivity grid (M = "
                    f"{smallest_failed_m:.2g})."
                )
            else:
                sentences.append(
                    f"HonestDiD: the result is fragile — the confidence "
                    f"interval includes zero once violations reach {bkd:.2g}x "
                    f"the pre-period variation."
                )

    # Sample sentence. For fits with a dynamic comparison set (CS /
    # ContinuousDiD / StaggeredTripleDiff / EfficientDiD /
    # StackedDiD under ``control_group in {"not_yet_treated",
    # "strict"}``) the fixed control count is suppressed because the
    # comparison group varies by cohort/sub-experiment; narrate the
    # mode explicitly rather than misreporting a fixed-subset tally as
    # "control" (rounds 13 / 17 / 18 / 22 CI review).
    sample = schema.get("sample", {}) or {}
    # ``schema["estimator"]`` is a dict with ``class_name``; unwrap it
    # for the per-estimator dynamic-control phrasing branch below.
    estimator_block = schema.get("estimator") or {}
    estimator = estimator_block.get("class_name") if isinstance(estimator_block, dict) else None
    n_obs = sample.get("n_obs")
    n_t = sample.get("n_treated")
    n_c = sample.get("n_control")
    n_nt = sample.get("n_never_treated")
    n_ne = sample.get("n_never_enabled")
    is_dynamic = sample.get("dynamic_control")
    cg = sample.get("control_group")
    # Panel-vs-RCS count-unit label. For repeated cross-section fits
    # (``panel=False`` on CallawaySantAnna), treated / never-treated
    # tallies are observation counts, not unit counts. Keep the
    # "N treated" phrasing (the N is still correct), but adjust the
    # never-treated clause so it does not claim "units present in
    # the panel" for an RCS sample.
    count_unit = sample.get("count_unit", "units")
    ne_unit_word = "observations" if count_unit == "observations" else "units"
    if isinstance(n_obs, int):
        if isinstance(n_t, int) and isinstance(n_c, int):
            sentences.append(f"Sample: {n_obs:,} observations ({n_t:,} treated, {n_c:,} control).")
        elif is_dynamic and isinstance(n_t, int):
            if isinstance(n_ne, int) and n_ne > 0:
                subset_clause = f"; {n_ne:,} never-enabled {ne_unit_word} are also present"
            elif isinstance(n_nt, int) and n_nt > 0:
                subset_clause = f"; {n_nt:,} never-treated {ne_unit_word} are also present"
            else:
                subset_clause = ""
            # Estimator-specific dynamic-comparison phrasing. StackedDiD
            # uses sub-experiment-specific clean controls (IC1/IC2
            # trimming) rather than a not-yet-treated rollout; the
            # generic phrasing misstates the identification setup.
            if estimator == "StackedDiDResults":
                cc_label = cg if isinstance(cg, str) else "clean_control"
                n_distinct = sample.get("n_distinct_controls_trimmed")
                distinct_clause = (
                    f" across {n_distinct:,} distinct control units in the trimmed stack"
                    if isinstance(n_distinct, int)
                    else ""
                )
                sentences.append(
                    f"Sample: {n_obs:,} observations ({n_t:,} treated) with a "
                    f"sub-experiment-specific clean-control comparison "
                    f"(``control_group='{cc_label}'``): each adoption event is "
                    f"compared against the units satisfying the rule relative "
                    f"to that event's window, not a single fixed control "
                    f"group{distinct_clause}{subset_clause}."
                )
            else:
                sentences.append(
                    f"Sample: {n_obs:,} observations ({n_t:,} treated) with a "
                    "dynamic not-yet-treated comparison group (the control set "
                    f"varies by cohort and period){subset_clause}."
                )
        elif (
            estimator == "StaggeredTripleDiffResults"
            and isinstance(n_t, int)
            and isinstance(n_ne, int)
            and n_ne > 0
        ):
            # Round-38 P2 CI review on PR #318: StaggeredTripleDiff
            # under fixed ``control_group="never_treated"`` had the
            # schema moved to ``n_never_enabled`` (round-37) but the
            # renderers fell through to the generic
            # ``Sample: N observations.`` sentence because the
            # ``is_dynamic_control`` branch didn't fire. REGISTRY.md
            # §StaggeredTripleDifference line 1730 names the
            # never-enabled cohort as the valid fixed comparison on
            # this path; the prose must say so.
            sentences.append(
                f"Sample: {n_obs:,} observations ({n_t:,} treated, " f"{n_ne:,} never-enabled)."
            )
        else:
            sentences.append(f"Sample: {n_obs:,} observations.")
        survey = sample.get("survey")
        if survey and not survey.get("is_trivial"):
            deff = survey.get("design_effect")
            eff_n = survey.get("effective_n")
            if isinstance(deff, (int, float)) and isinstance(eff_n, (int, float)):
                # Round-35 P2 CI review on PR #318: ``deff < 0.95`` is a
                # precision-improving design (effective N is LARGER than
                # nominal N). Narrating that as "reduces effective sample
                # size" is directionally wrong. Branch on the sign of
                # the departure from 1.
                if deff < 1.0:
                    sentences.append(
                        f"Survey design improves effective sample size to "
                        f"~{eff_n:,.0f} (DEFF = {deff:.2g})."
                    )
                else:
                    sentences.append(
                        f"Survey design reduces effective sample size to "
                        f"~{eff_n:,.0f} (DEFF = {deff:.2g})."
                    )

    # Highest-severity caveat (if any).
    caveats = schema.get("caveats", [])
    warning_caveats = [c for c in caveats if c.get("severity") == "warning"]
    if warning_caveats:
        top = warning_caveats[0]
        sentences.append(f"Caveat: {top.get('message')}")

    return " ".join(s for s in sentences if s)


def _render_full_report(schema: Dict[str, Any]) -> str:
    """Render the structured multi-section markdown report."""
    ctx = schema.get("context", {})
    h = schema.get("headline", {})
    sample = schema.get("sample", {})
    pt = schema.get("pre_trends", {}) or {}
    sens = schema.get("sensitivity", {}) or {}
    assumption = schema.get("assumption", {})
    het = schema.get("heterogeneity")
    caveats = schema.get("caveats", [])
    references = schema.get("references", [])
    next_steps = schema.get("next_steps", [])

    lines: List[str] = []
    lines.append(f"# Business Report: {ctx.get('outcome_label', 'Outcome')}")
    lines.append("")
    if ctx.get("business_question"):
        lines.append(f"**Question**: {ctx['business_question']}")
        lines.append("")
    lines.append(f"**Estimator**: `{schema.get('estimator', {}).get('class_name')}`")
    lines.append("")

    # Headline
    lines.append("## Headline")
    lines.append("")
    lines.append(_render_headline_sentence(schema))
    p = h.get("p_value")
    alpha = ctx.get("alpha", 0.05)
    if isinstance(p, (int, float)):
        lines.append("")
        lines.append(f"Statistically, {_significance_phrase(p, alpha)}.")
    lines.append("")

    # Target parameter (BR/DR gap #6): name what the headline scalar
    # represents so the stakeholder can map the number to a specific
    # estimand. Rendered between "Headline" and "Identifying Assumption"
    # because the target parameter is about what the scalar IS, whereas
    # identifying assumption is about what makes it valid.
    tp = schema.get("target_parameter", {}) or {}
    if tp.get("name") or tp.get("definition"):
        lines.append("## Target Parameter")
        lines.append("")
        if tp.get("name"):
            lines.append(f"- **{tp['name']}**")
        if tp.get("definition"):
            lines.append(f"- {tp['definition']}")
        lines.append("")

    # Identifying assumption
    lines.append("## Identifying Assumption")
    lines.append("")
    lines.append(assumption.get("description", "") or "Standard DiD parallel-trends assumption.")
    lines.append("")

    # Pre-trends
    lines.append("## Pre-Trends")
    lines.append("")
    if pt.get("status") == "computed":
        jp = pt.get("joint_p_value")
        verdict = pt.get("verdict")
        tier = pt.get("power_tier")
        # Use the method-aware statistic label the summary path already
        # uses: "joint p" for Wald / Bonferroni event-study, "p" for
        # slope-difference / Hausman single-statistic tests, and None
        # for design-enforced SDiD / TROP paths where there is no
        # p-value at all. Round-25 P2 CI review on PR #318 flagged the
        # hard-coded "joint p" wording as misdescribing 2x2 / Hausman
        # fits and inventing a nonexistent p-value for SDiD / TROP.
        method = pt.get("method")
        stat_label = _pt_method_stat_label(method)
        if stat_label and isinstance(jp, (int, float)):
            lines.append(f"- Verdict: `{verdict}` ({stat_label} = {jp:.3g})")
        elif stat_label:
            lines.append(f"- Verdict: `{verdict}` ({stat_label} unavailable)")
        else:
            lines.append(f"- Verdict: `{verdict}`")
        if tier:
            lines.append(f"- Power tier: `{tier}`")
        mdv = pt.get("mdv")
        max_abs_pre = pt.get("max_abs_pre_violation")
        ratio = pt.get("mdv_share_of_att")
        if isinstance(mdv, (int, float)):
            lines.append(f"- Minimum detectable violation (MDV): {mdv:.3g}")
        if isinstance(max_abs_pre, (int, float)):
            lines.append(f"- Max pre-period level deviation at MDV: {max_abs_pre:.3g}")
        if isinstance(ratio, (int, float)):
            # PR-B R12: ratio is now max_abs_pre_violation / |ATT|, the
            # level-scale comparable to ATT (not raw γ-unit mdv on linear
            # fits). Label updated to match the numerator definition in
            # REPORTING.md "Power-aware phrasing" Note.
            lines.append(f"- Max pre-period level deviation / |ATT|: {ratio:.2g}")
    else:
        lines.append(f"- Pre-trends not computed: {pt.get('reason', 'unavailable')}")
    lines.append("")

    # Sensitivity. A single-M HonestDiDResults passthrough has
    # breakdown_M=None by construction because only one M was evaluated;
    # the "robust across full grid" phrasing is reserved for genuine
    # grid-over-M SensitivityResults.
    lines.append("## Sensitivity (HonestDiD)")
    lines.append("")
    if sens.get("status") == "computed":
        bkd = sens.get("breakdown_M")
        concl = sens.get("conclusion")
        lines.append(f"- Method: `{sens.get('method')}`")
        if concl == "single_M_precomputed":
            grid_points = sens.get("grid") or []
            point = grid_points[0] if grid_points else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                lines.append(f"- Single point checked: M = {m_val:.3g}")
                lines.append(
                    f"- Robust CI at M = {m_val:.3g}: "
                    f"{'excludes zero' if robust else 'includes zero'}"
                )
                lines.append(
                    "- Run `HonestDiD.sensitivity()` across a grid of M "
                    "values to find the breakdown value."
                )
            else:
                lines.append("- Single-M passthrough (breakdown not available)")
        elif isinstance(bkd, (int, float)):
            lines.append(f"- Breakdown M: {bkd:.3g}")
        else:
            lines.append("- Breakdown M: robust across full grid (no breakdown)")
        lines.append(f"- Conclusion: `{concl}`")
    else:
        lines.append(f"- Sensitivity not computed: {sens.get('reason', 'unavailable')}")
    lines.append("")

    # Sample
    lines.append("## Sample")
    lines.append("")
    if isinstance(sample.get("n_obs"), int):
        lines.append(f"- Observations: {sample['n_obs']:,}")
    if isinstance(sample.get("n_treated"), int):
        lines.append(f"- Treated: {sample['n_treated']:,}")
    # ``n_control`` is only populated for estimators whose control set
    # is a fixed tally. For dynamic modes (CS / ContinuousDiD /
    # StaggeredTripleDiff / EfficientDiD / StackedDiD under
    # ``clean_control in {"not_yet_treated", "strict"}``) the comparison
    # group is dynamic per cohort/sub-experiment; report the estimator-
    # specific fixed subset (``n_never_enabled`` for triple-difference;
    # ``n_never_treated`` elsewhere; ``n_distinct_controls_trimmed`` for
    # Stacked) when available, then name the dynamic-comparison mode
    # explicitly.
    estimator_block = schema.get("estimator") or {}
    estimator_name = (
        estimator_block.get("class_name") if isinstance(estimator_block, dict) else None
    )
    cg = sample.get("control_group")
    # Panel-vs-RCS count-unit label for the full report. Mirrors the
    # summary path: CallawaySantAnna's ``panel=False`` mode stores
    # counts as observations, not units (round-28 P2).
    md_count_unit = sample.get("count_unit", "units")
    md_ne_unit_word = "observations" if md_count_unit == "observations" else "units"
    md_sample_location = (
        "in the repeated cross-section sample"
        if md_count_unit == "observations"
        else "in the panel"
    )
    if isinstance(sample.get("n_control"), int):
        lines.append(f"- Control: {sample['n_control']:,}")
    elif (
        estimator_name == "StaggeredTripleDiffResults"
        and isinstance(sample.get("n_never_enabled"), int)
        and sample["n_never_enabled"] > 0
        and not sample.get("dynamic_control")
    ):
        # Round-38 P2 CI review on PR #318: fixed
        # ``control_group="never_treated"`` on StaggeredTripleDiff
        # clears ``n_control`` (composite total) and populates
        # ``n_never_enabled`` (the valid fixed comparison cohort per
        # REGISTRY.md line 1730). The full report must render that
        # fixed count — the dynamic-control branch below would not
        # fire on this path.
        lines.append(
            f"- Never-enabled units (fixed comparison cohort): " f"{sample['n_never_enabled']:,}"
        )
    elif sample.get("dynamic_control"):
        if isinstance(sample.get("n_never_enabled"), int) and sample["n_never_enabled"] > 0:
            lines.append(
                f"- Never-enabled {md_ne_unit_word} present "
                f"{md_sample_location}: {sample['n_never_enabled']:,}"
            )
        elif isinstance(sample.get("n_never_treated"), int) and sample["n_never_treated"] > 0:
            lines.append(
                f"- Never-treated {md_ne_unit_word} present "
                f"{md_sample_location}: {sample['n_never_treated']:,}"
            )
        if estimator_name == "StackedDiDResults":
            n_distinct = sample.get("n_distinct_controls_trimmed")
            if isinstance(n_distinct, int):
                lines.append(f"- Distinct control units in trimmed stack: {n_distinct:,}")
            cc_label = cg if isinstance(cg, str) else "clean_control"
            lines.append(
                f"- Comparison group: sub-experiment-specific clean controls "
                f"(``control_group='{cc_label}'``; each adoption event is "
                "compared against units satisfying the rule relative to that "
                "event's window, not a single fixed control group)"
            )
        else:
            lines.append(
                "- Comparison group: dynamic not-yet-treated units "
                "(varies by cohort and period; no fixed control count)"
            )
    survey = sample.get("survey")
    if survey:
        if survey.get("is_trivial"):
            lines.append("- Survey design: trivial DEFF (~1.0)")
        else:
            deff = survey.get("design_effect")
            eff_n = survey.get("effective_n")
            if isinstance(deff, (int, float)):
                lines.append(f"- Survey DEFF: {deff:.2g}")
            if isinstance(eff_n, (int, float)):
                lines.append(f"- Effective N: {eff_n:,.0f}")
    lines.append("")

    # Heterogeneity — only render the populated section when the check
    # actually ran. Round-32 P2 CI review on PR #318: round-31 changed
    # ``_lift_heterogeneity`` to always return a dict (stable schema
    # contract), but the renderer's ``if het:`` truthiness guard then
    # entered the block on every fit and printed ``Source: None``,
    # ``N effects: None``, etc. Gate on the ``status`` enum instead.
    if isinstance(het, dict) and het.get("status") == "ran":
        lines.append("## Heterogeneity")
        lines.append("")
        lines.append(f"- Source: `{het.get('source')}`")
        lines.append(f"- N effects: {het.get('n_effects')}")
        mn = het.get("min")
        mx = het.get("max")
        if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
            lines.append(f"- Range: {mn:.3g} to {mx:.3g}")
        cv = het.get("cv")
        if isinstance(cv, (int, float)):
            lines.append(f"- CV: {cv:.3g}")
        lines.append(f"- Sign consistent: {het.get('sign_consistent')}")
        lines.append("")

    # Caveats
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            sev = c.get("severity", "info")
            lines.append(f"- **{sev.upper()}** — {c.get('message')}")
        lines.append("")

    # Next steps
    if next_steps:
        lines.append("## Next Steps")
        lines.append("")
        for s in next_steps:
            if s.get("label"):
                lines.append(f"- {s['label']}")
                if s.get("why"):
                    lines.append(f"  - _why_: {s['why']}")
        lines.append("")

    # References
    if references:
        lines.append("## References")
        lines.append("")
        for ref in references:
            lines.append(f"- {ref.get('citation')}")
        lines.append("")

    return "\n".join(lines)
