"""Shared helpers for BusinessReport and DiagnosticReport.

This module hosts per-estimator dispatch logic that both BR and DR
consume. It lives in its own module rather than in ``business_report``
or ``diagnostic_report`` to avoid a circular-import risk: BR already
imports from DR (for ``DiagnosticReportResults``), so placing shared
helpers here keeps the dependency graph tidy if DR ever needs to
reference a BR symbol.

Current contents:

- ``describe_target_parameter(results)`` — returns the
  ``target_parameter`` block documenting what scalar the headline
  represents. Introduced for BR/DR gap #6 (target-parameter
  clarity); see ``docs/methodology/REPORTING.md`` and per-estimator
  entries in ``docs/methodology/REGISTRY.md`` for the canonical
  wording.
"""

from __future__ import annotations

from typing import Any, Dict

from diff_diff.results_base import Diagnostic


def describe_target_parameter(results: Any) -> Dict[str, Any]:
    """Return the target-parameter block for a fitted result.

    The block names what scalar the headline number actually
    represents (overall ATT, DID_M, dose-response ATT(d|d), etc.) so
    BR/DR output is self-contained. Baker et al. (2025) Step 2 is
    "Define the target parameter"; this helper does that work for the
    user.

    Shape::

        {
            "name": str,                # stakeholder-facing name
            "definition": str,          # plain-English description
            "aggregation": str,         # machine-readable tag
            "headline_attribute": str,  # which raw attribute the scalar comes from
            "reference": str,           # citation pointer
        }

    Each per-estimator branch cites REGISTRY.md as the single source
    of truth for the canonical wording (per
    ``feedback_verify_claims.md``). All wording choices are
    deliberate:

    - ``ImputationDiD`` / ``TwoStageDiD``: the ``aggregate`` fit-time
      kwarg controls which horizon / group tables get populated but
      does NOT change ``overall_att``. The headline is always the
      sample-mean overall ATT (per BJS 2024 Step 3 with
      ``w_it = 1/N_1``); disambiguate via the event-study or group
      aggregate if you need the horizon / group target.
    - ``CallawaySantAnna``: ``overall_att`` is cohort-size-weighted
      across post-treatment ``ATT(g, t)`` cells regardless of the
      fit-time ``aggregate`` kwarg. The event-study / group
      aggregations live on dedicated fields
      (``event_study_effects`` / ``group_effects``).
    - ``ContinuousDiD``: the regime (PT vs. SPT) is a user-level
      assumption, not a library setting. The ``definition`` names
      both regime readings (``ATT^loc`` under PT,
      ``ATT^glob`` under SPT) so the user can pick the
      interpretation that matches their assumption.
    - ``WooldridgeDiD``: ``overall_att`` depends on the fit-time
      ``method`` — for OLS ETWFE (``method="ols"``) it is the
      observation-count-weighted average of ``ATT(g, t)``
      coefficients from the saturated regression; for nonlinear
      ETWFE (``method="logit"`` / ``"poisson"``) it is the
      average-structural-function (ASF) contrast across cohort x
      time cells. Both paths preserve ``overall_att`` across
      ``.aggregate("event_study")`` calls (which only populate additional
      event-study tables).
    """
    name = type(results).__name__

    if name == "DiDResults":
        # Covers both ``DifferenceInDifferences`` (2x2 DiD) and
        # ``TwoWayFixedEffects`` (TWFE with unit + time FE). Both
        # estimators return ``DiDResults``; there is no separate
        # ``TwoWayFixedEffectsResults`` class as of this PR
        # (confirmed in PR #347 R1 review). The description covers
        # both interpretations because the result carries no
        # estimator-provenance marker BR/DR can dispatch on. Adding a
        # dedicated TWFE result class (or persisting provenance on
        # DiDResults) is queued as follow-up so this branch can split
        # in a future PR.
        #
        # PR #347 R12 P2: the ``aggregation`` tag is
        # ``"did_or_twfe"`` (not ``"2x2"``) because real TWFE fits
        # can be weighted averages of 2x2 comparisons, potentially
        # with forbidden later-vs-earlier weights. Downstream agents
        # dispatching on ``aggregation`` must not treat this as a
        # clean 2x2 fit — the ambiguity is intrinsic until an
        # estimator-provenance marker exists.
        return {
            "name": "ATT (2x2 or TWFE within-transformed coefficient)",
            "definition": (
                "The average treatment effect on the treated. For "
                "``DifferenceInDifferences``, this is the 2x2 DiD "
                "contrast between treated-unit change and control-unit "
                "change across pre / post. For ``TwoWayFixedEffects``, "
                "this is the coefficient on the treatment-by-post "
                "interaction in a regression with unit and time fixed "
                "effects; under homogeneous treatment effects it is "
                "the ATT, and under heterogeneous effects with staggered "
                "adoption it is a weighted average of 2x2 comparisons "
                "that may include forbidden later-vs-earlier comparisons "
                "(see Goodman-Bacon)."
            ),
            "aggregation": "did_or_twfe",
            "headline_attribute": "att",
            "reference": ("REGISTRY.md Sec. DifferenceInDifferences / TwoWayFixedEffects"),
        }

    if name == "MultiPeriodDiDResults":
        return {
            "name": "average ATT over post-treatment periods (event-study average)",
            "definition": (
                "The equally-weighted average of the post-treatment event-study "
                "coefficients. Each coefficient is an ATT at a single event time "
                "relative to treatment onset; the headline averages across "
                "post-treatment horizons."
            ),
            "aggregation": "event_study",
            "headline_attribute": "avg_att",
            "reference": "REGISTRY.md Sec. MultiPeriodDiD",
        }

    if name == "CallawaySantAnnaResults":
        return {
            "name": "overall ATT (cohort-size-weighted average of ATT(g,t))",
            "definition": (
                "A cohort-size-weighted average of group-time ATTs "
                "``ATT(g, t)`` across post-treatment cells (``t >= g``). "
                "``overall_att`` is the simple-aggregation headline regardless "
                "of the fit-time ``aggregate`` kwarg; event-study and group "
                "aggregations populate ``event_study_effects`` / "
                "``group_effects`` fields when requested."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Callaway & Sant'Anna (2021); REGISTRY.md Sec. CallawaySantAnna",
        }

    if name == "SunAbrahamResults":
        return {
            "name": "overall ATT (interaction-weighted average of CATT(g, e))",
            "definition": (
                "An interaction-weighted (IW) average of cohort-specific "
                "ATT(g, e) effects estimated via saturated cohort-by-event-time "
                "interactions. Weights are the sample shares of each cohort at "
                "each event time; the headline is the resulting overall ATT."
            ),
            "aggregation": "iw",
            "headline_attribute": "overall_att",
            "reference": "Sun & Abraham (2021); REGISTRY.md Sec. SunAbraham",
        }

    if name == "ImputationDiDResults":
        return {
            "name": "overall ATT (sample-mean imputation average)",
            "definition": (
                "The sample-mean overall ATT ``tau_hat_w = (1/N_1) * sum "
                "tau_hat_it`` across treated observations, where "
                "``tau_hat_it = Y_it - Y_hat_it(0)`` and ``Y_hat_it(0)`` is "
                "imputed from a unit+time fixed-effects model fitted on "
                "untreated observations only (BJS 2024 Step 3). The fit-time "
                "``aggregate`` kwarg populates additional horizon / group "
                "tables but does NOT change ``overall_att`` — for the "
                "horizon or group estimand, consult the event-study or group "
                "aggregate directly."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Borusyak-Jaravel-Spiess (2024); REGISTRY.md Sec. ImputationDiD",
        }

    if name == "TwoStageDiDResults":
        return {
            "name": "overall ATT (two-stage residualization average)",
            "definition": (
                "The sample-mean overall ATT recovered from Stage 2 of "
                "Gardner's two-stage procedure: Stage 1 fits unit+time fixed "
                "effects on untreated observations only, and Stage 2 regresses "
                "the residualized outcome on the treatment indicator across "
                "treated observations. Point estimate is algebraically "
                "equivalent to Borusyak-Jaravel-Spiess imputation. As with "
                "ImputationDiD, the fit-time ``aggregate`` kwarg populates "
                "additional tables but does NOT change ``overall_att``."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Gardner (2022); REGISTRY.md Sec. TwoStageDiD",
        }

    if name == "StackedDiDResults":
        # M-095: the field is control_group; __setstate__ migration
        # guarantees it exists even on pre-rename pickles, so no
        # clean_control fallback (a getattr default would silently
        # misreport the comparison group after the 4.0 removal).
        clean_control = getattr(results, "control_group", None)
        if clean_control == "never_treated":
            control_clause = "Controls are the never-treated units (``A_s = infinity``)."
        elif clean_control == "strict":
            control_clause = (
                "Controls for each sub-experiment ``a`` are units strictly "
                "untreated across the full pre/post event window "
                "(``A_s > a + kappa_post + kappa_pre``)."
            )
        else:  # "not_yet_treated" or unknown -> default rule
            control_clause = (
                "Controls for each sub-experiment ``a`` are units not yet "
                "treated through the event's post-window "
                "(``A_s > a + kappa_post``)."
            )
        return {
            "name": "overall ATT (average of post-treatment event-study coefficients)",
            "definition": (
                "The average of post-treatment event-study coefficients "
                "``delta_h`` (h >= -anticipation), estimated from the stacked "
                "sub-experiment panel with delta-method SE. Each sub-"
                "experiment aligns a treated cohort with its clean-control "
                "set over the event window ``[-kappa_pre, +kappa_post]``; "
                "each per-horizon ``delta_h`` is the paper's "
                "``theta_kappa^e`` treated-share-weighted cross-event "
                "aggregate. The ``overall_att`` headline is the equally-"
                "weighted average of these per-horizon coefficients, not a "
                "separate cross-event weighted aggregate at the ATT level. " + control_clause
            ),
            "aggregation": "stacked",
            "headline_attribute": "overall_att",
            "reference": "Wing-Freedman-Hollingsworth (2024); REGISTRY.md Sec. StackedDiD",
        }

    if name == "WooldridgeDiDResults":
        # PR #347 R4 P2: Wooldridge ETWFE has two identification paths
        # (REGISTRY.md splits them at Sec. WooldridgeDiD): the OLS
        # path computes ``overall_att`` as an observation-count-
        # weighted aggregation of ``ATT(g, t)`` coefficients from the
        # saturated regression, while the nonlinear (logit / Poisson)
        # paths produce an ASF-based ATT from the average-structural-
        # function contrast. ``WooldridgeDiDResults.method`` persists
        # the choice; branch on it so OLS fits aren't mislabeled with
        # nonlinear-ASF wording.
        method = getattr(results, "method", "ols")
        if method == "ols":
            return {
                "name": (
                    "overall ATT (observation-count-weighted average of "
                    "ATT(g,t) from saturated OLS ETWFE)"
                ),
                "definition": (
                    "The overall ATT under OLS ETWFE (Wooldridge 2025): the "
                    "saturated regression fits cohort x time ATT(g, t) "
                    "coefficients, and ``overall_att`` is their "
                    "observation-count-weighted average across post-"
                    'treatment cells. Calling ``.aggregate("event_study")`` '
                    "populates additional event-study tables but does NOT "
                    "change the ``overall_att`` scalar."
                ),
                "aggregation": "simple",
                "headline_attribute": "overall_att",
                "reference": ("Wooldridge (2025); REGISTRY.md Sec. WooldridgeDiD (OLS path)"),
            }
        return {
            "name": (
                f"overall ATT (ASF-based average from Wooldridge ETWFE, " f"method={method!r})"
            ),
            "definition": (
                f"The overall ATT under Wooldridge ETWFE with a nonlinear "
                f"link function (``method={method!r}``, typically logit or "
                f"Poisson QMLE): the average-structural-function (ASF) "
                f"contrast between treated and counterfactual untreated "
                f"outcomes averaged across cohort x time cells with "
                f"observation-count weights. The ASF handles the "
                f"nonlinearity; OLS ETWFE uses the saturated-regression "
                f'coefficient path instead. Calling ``.aggregate("event_study")`` '
                f"populates additional event-study tables but does NOT "
                f"change the ``overall_att`` scalar."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": (
                "Wooldridge (2023, 2025); REGISTRY.md Sec. WooldridgeDiD " "(nonlinear / ASF path)"
            ),
        }

    if name == "EfficientDiDResults":
        # PR #347 R7 P1: the BR/DR headline ``overall_att`` is the
        # library's cohort-size-weighted average over post-treatment
        # ``(g, t)`` cells (see ``efficient_did.py`` around line 1274
        # and REGISTRY.md Sec. EfficientDiD). This is distinct from
        # the paper's ``ES_avg`` uniform event-time average.
        # Disambiguating this in the stakeholder-facing definition
        # keeps the user from mistaking one for the other — the
        # regime (PT-All vs PT-Post) describes identification, not
        # the aggregation choice for the headline scalar.
        pt_assumption = getattr(results, "pt_assumption", "all")
        library_aggregation_note = (
            " The BR/DR headline ``overall_att`` is the library's "
            "cohort-size-weighted average of ATT(g, t) over post-"
            "treatment cells, NOT the paper's ``ES_avg`` uniform event-"
            "time average (see REGISTRY.md Sec. EfficientDiD for the "
            "distinction)."
        )
        if pt_assumption == "post":
            return {
                "name": "overall ATT under PT-Post (single-baseline)",
                "definition": (
                    "The overall ATT identified under the weaker PT-Post "
                    "regime (parallel trends hold only in post-treatment "
                    "periods). The baseline is period ``g - 1`` only; the "
                    "estimator is just-identified and reduces to standard "
                    "single-baseline DiD (Corollary 3.2)." + library_aggregation_note
                ),
                "aggregation": "pt_post_single_baseline",
                "headline_attribute": "overall_att",
                "reference": "Chen-Sant'Anna-Xie (2025) Cor. 3.2; REGISTRY.md Sec. EfficientDiD",
            }
        return {
            "name": "overall ATT under PT-All (over-identified combined)",
            "definition": (
                "The overall ATT identified under the stronger PT-All regime "
                "(parallel trends hold for all groups and all periods). The "
                "estimator is over-identified (Lemma 2.1) and applies "
                "optimal-combination weights to achieve the semiparametric "
                "efficiency bound on the no-covariate path." + library_aggregation_note
            ),
            "aggregation": "pt_all_combined",
            "headline_attribute": "overall_att",
            "reference": "Chen-Sant'Anna-Xie (2025) Lemma 2.1; REGISTRY.md Sec. EfficientDiD",
        }

    if name == "ContinuousDiDResults":
        return {
            "name": "overall ATT (dose-aggregated)",
            "definition": (
                "The overall ATT aggregated across treated dose levels. Under "
                "Parallel Trends (PT) this identifies ``ATT^loc`` — the "
                "binarized ATT for treated-vs-untreated; under Strong Parallel "
                "Trends (SPT) it identifies ``ATT^glob`` — the population "
                "average ATT across dose levels. The regime is a user-level "
                "assumption, not a library setting; the dose-response curve "
                "``ATT(d)`` / marginal effect ``ACRT(d)`` / per-dose "
                "``ATT(d|d)`` are available on the result object for "
                "dose-specific inference."
            ),
            "aggregation": "dose_overall",
            "headline_attribute": "overall_att",
            "reference": "Callaway-Goodman-Bacon-Sant'Anna (2024); REGISTRY.md Sec. ContinuousDiD",
        }

    if name == "TripleDifferenceResults":
        return {
            "name": "ATT (triple-difference)",
            "definition": (
                "The ATT identified via the DDD cancellation "
                "``DDD = DiD_A + DiD_B - DiD_C`` across the Group x Period x "
                "Eligibility cells. Differences out group-specific and period-"
                "specific unobservables without requiring separate parallel-"
                "trends assumptions between each cell pair."
            ),
            "aggregation": "ddd",
            "headline_attribute": "att",
            "reference": "Ortiz-Villavicencio & Sant'Anna (2025); REGISTRY.md Sec. TripleDifference",
        }

    if name == "StaggeredTripleDiffResults":
        return {
            "name": "overall ATT (cohort-weighted staggered triple-difference)",
            "definition": (
                "A cohort-weighted aggregate of per-(g, g_c, t) triple-"
                "difference ATTs ``DDD(g, g_c, t)`` via the GMM optimal-"
                "combination weighting. Extends the DDD cancellation to "
                "staggered adoption timing with a third (eligibility) "
                "dimension."
            ),
            "aggregation": "staggered_ddd",
            "headline_attribute": "overall_att",
            "reference": "Ortiz-Villavicencio & Sant'Anna (2025); REGISTRY.md Sec. StaggeredTripleDifference",
        }

    if name == "ChaisemartinDHaultfoeuilleResults":
        l_max = getattr(results, "L_max", None)
        has_controls = getattr(results, "covariate_residuals", None) is not None
        # PR #347 R9 P1: read the persisted ``trends_linear`` flag
        # directly rather than inferring from
        # ``linear_trends_effects is not None``. The estimator can set
        # ``linear_trends_effects = None`` when the cumulated-horizon
        # dict is empty (no estimable horizons) while still
        # unconditionally NaN-ing ``overall_att`` under
        # ``trends_linear=True`` + ``L_max >= 2``
        # (``chaisemartin_dhaultfoeuille.py:2828-2834``). The previous
        # inference missed that edge case; the new explicit flag
        # (persisted on ``ChaisemartinDHaultfoeuilleResults``) closes
        # the gap. Older fits without the persisted flag fall back to
        # the legacy inference.
        _persisted = getattr(results, "trends_linear", None)
        if isinstance(_persisted, bool):
            has_trends = _persisted
        else:
            has_trends = getattr(results, "linear_trends_effects", None) is not None
        reference = (
            "de Chaisemartin & D'Haultfoeuille (2020, 2024); "
            "REGISTRY.md Sec. ChaisemartinDHaultfoeuille"
        )
        # PR #347 R2 P1 review: mirror the exact ``overall_att``
        # contract from ``chaisemartin_dhaultfoeuille.py`` lines
        # 2602-2634 (L_max>=2 overrides with cost-benefit ``delta``,
        # NaN if non-estimable) and lines 2828-2834 (``trends_linear``
        # + ``L_max>=2`` suppresses the scalar entirely with NaN).
        # The result class's own ``_estimand_label`` at
        # ``chaisemartin_dhaultfoeuille_results.py:454-490`` is the
        # source-of-truth; this branch tracks that logic.

        # Trends + L_max>=2: overall_att is NaN. No scalar aggregate;
        # per-horizon effects are on ``linear_trends_effects``.
        if has_trends and l_max is not None and l_max >= 2:
            # PR #347 R12 P1: distinguish the populated-surface case
            # (``linear_trends_effects`` has horizons) from the
            # empty-surface subcase (``linear_trends_effects is
            # None``: requested ``trends_linear=True`` but no
            # horizons survived). Pointing users to a nonexistent
            # dict in the latter is dead-end guidance.
            horizon_surface_empty = getattr(results, "linear_trends_effects", None) is None
            if has_controls:
                estimand_label = "DID^{X,fd}_l"
            else:
                estimand_label = "DID^{fd}_l"
            if horizon_surface_empty:
                # PR #347 R14 P1: name the estimand label with the
                # covariate-adjusted form (``DID^{X,fd}_l``) when
                # covariates are active rather than hardcoding
                # ``DID^{fd}_l``. The native result label already
                # handles this; propagate to the reporting-layer
                # prose.
                definition_text = (
                    "Under ``trends_linear=True`` with ``L_max >= 2``, the "
                    "estimator intentionally does NOT produce a scalar "
                    "aggregate in ``overall_att`` (it is NaN by design, "
                    "matching R's ``did_multiplegt_dyn`` with "
                    "``trends_lin=TRUE``). On this fit, no cumulated level "
                    f"effects ``{estimand_label}`` survived estimation "
                    "(the horizon surface is empty — either no eligible "
                    "switchers at any positive horizon or all horizons "
                    "were dropped). There is no scalar headline AND no "
                    "per-horizon table to fall back on; re-fit with a "
                    "larger ``L_max`` or with ``trends_linear=False`` if "
                    "you need a reportable estimand."
                )
            else:
                estimand_label = estimand_label + " (see linear_trends_effects)"
                definition_text = (
                    "Under ``trends_linear=True`` with ``L_max >= 2``, the "
                    "estimator intentionally does NOT produce a scalar "
                    "aggregate in ``overall_att`` (it is NaN by design, "
                    "matching R's ``did_multiplegt_dyn`` with "
                    "``trends_lin=TRUE``). Per-horizon cumulated level "
                    "effects ``DID^{fd}_l`` (or ``DID^{X,fd}_l`` when "
                    "covariates are active) live on "
                    "``results.linear_trends_effects[l]``. Consult those "
                    "rather than the headline."
                )
            return {
                "name": estimand_label,
                "definition": definition_text,
                "aggregation": "no_scalar_headline",
                "headline_attribute": None,
                "reference": reference,
            }

        if l_max is None:
            # DID_M — period-aggregated contemporaneous-switch ATT.
            base, agg_base, base_label = (
                "DID_M",
                "M",
                "period-aggregated contemporaneous-switch ATT",
            )
            definition_core = (
                "The contemporaneous-switch ATT averaged across switching "
                "periods. At each period the estimator contrasts joiners "
                "(``D: 0 -> 1``), leavers (``D: 1 -> 0``), and stable-"
                "control cells that share the same treatment state across "
                "adjacent periods; ``DID_M`` averages these per-period "
                "contrasts."
            )
        elif l_max == 1:
            # DID_1 — single-horizon per-group estimand.
            base, agg_base, base_label = (
                "DID_1",
                "DID_1",
                "single-horizon per-group dynamic ATT",
            )
            definition_core = (
                "The per-group dynamic ATT at event horizon ``l = 1`` "
                "post-switch (Equation 3 of the dCDH dynamic companion "
                "paper). The estimator contrasts joiners and stable "
                "controls conditioning on baseline treatment ``D_{g,1}``; "
                "the aggregate averages across cohorts with treated-share "
                "weights. This is the per-group ``DID_{g,1}`` building "
                "block averaged, NOT the per-period ``DID_M`` (the two can "
                "differ by O(1%) on mixed-direction panels — see the "
                "``Phase 2 DID_1 vs Phase 1 DID_M`` Note in REGISTRY.md)."
            )
        else:
            # L_max >= 2: cost-benefit delta aggregate.
            base, agg_base, base_label = (
                "delta",
                "delta",
                "cost-benefit cross-horizon aggregate",
            )
            definition_core = (
                "The cost-benefit aggregate "
                "``delta = sum_l w_l * DID_l`` (Lemma 4 of the dCDH "
                "dynamic companion paper), a weighted cross-horizon "
                "combination where ``w_l`` reflects the cumulative dose at "
                "each horizon. ``overall_att`` holds this delta when "
                "``L_max >= 2``; if delta is non-estimable (no eligible "
                "switchers at any horizon) it is NaN with the same "
                "``overall_se`` / CI surface."
            )

        # Suffix for controls / trends. ``trends_linear`` + ``L_max >= 2``
        # was handled above (no-scalar case); here ``has_trends`` can
        # only co-occur with ``L_max in {None, 1}`` (or ``L_max == None``
        # where the no-scalar rule does not apply).
        if has_controls and has_trends:
            suffix, agg_suffix, suffix_clause = (
                "^{X,fd}",
                "_x_fd",
                " Identification is conditional on the first-stage "
                "covariates and allows group-specific linear pre-trends.",
            )
        elif has_controls:
            suffix, agg_suffix, suffix_clause = (
                "^X",
                "_x",
                " Identification is conditional on the first-stage " "covariates.",
            )
        elif has_trends:
            suffix, agg_suffix, suffix_clause = (
                "^{fd}",
                "_fd",
                " The identifying restriction allows group-specific linear " "pre-trends.",
            )
        else:
            suffix, agg_suffix, suffix_clause = "", "", ""

        # Assemble the name label to match the result class's
        # ``_estimand_label()``: for ``delta``, suffix follows the base
        # (``delta^X``); for DID variants, suffix goes on ``DID`` with
        # the subscript preserved (``DID^X_1``, ``DID^{fd}_M``).
        if base == "delta":
            name_label = f"delta{suffix}" if suffix else "delta"
        elif suffix:
            did_part = base.split("_")[0]
            sub_part = base.split("_")[1] if "_" in base else ""
            name_label = f"{did_part}{suffix}_{sub_part}" if sub_part else f"{did_part}{suffix}"
        else:
            name_label = base

        return {
            "name": f"{name_label} ({base_label})",
            "definition": definition_core + suffix_clause,
            "aggregation": agg_base + agg_suffix,
            "headline_attribute": "overall_att",
            "reference": reference,
        }

    if name == "SyntheticDiDResults":
        return {
            "name": "synthetic-DiD ATT (time- and unit-weighted synthetic-control comparison)",
            "definition": (
                "The treatment-effect contrast ``tau_hat^{sdid} = sum_t "
                "lambda_t (Y_bar_{tr, t} - sum_j omega_j Y_{j, t})`` between "
                "the treated group and a synthetic-control composed of "
                "unit-weighted donors (``omega_j``) aggregated with time-"
                "weighted pre-period matching weights (``lambda_t``). "
                "Identification is through the weighted parallel-trends "
                "analogue rather than unconditional PT."
            ),
            "aggregation": "synthetic",
            "headline_attribute": "att",
            "reference": "Arkhangelsky et al. (2021); REGISTRY.md Sec. SyntheticDiD",
        }

    if name == "BaconDecompositionResults":
        # BaconDecompositionResults is a diagnostic, not an estimator:
        # BR refuses it (raises TypeError), but DR accepts it as a
        # read-out and the schema needs a target-parameter block.
        # The "target parameter" of a Bacon decomposition is the TWFE
        # coefficient it is decomposing — named here for schema
        # completeness.
        return {
            "name": "TWFE ATT being decomposed",
            "definition": (
                "Goodman-Bacon decomposition is a diagnostic of the TWFE "
                "coefficient ``twfe_estimate``: it partitions TWFE weights "
                "across treated-vs-never, earlier-vs-later, and "
                "later-vs-earlier (forbidden) 2x2 comparisons. The scalar "
                "``twfe_estimate`` is the weighted sum of these 2x2 DiDs."
            ),
            "aggregation": "twfe",
            "headline_attribute": "twfe_estimate",
            "reference": "Goodman-Bacon (2021); REGISTRY.md Sec. BaconDecomposition",
        }

    if name == "TROPResults":
        return {
            "name": "TROP ATT (factor-model-adjusted weighted average)",
            "definition": (
                "A weighted average of per-(unit, time) treatment effects "
                "``tau_hat_{it}(lambda_hat)`` estimated under low-rank factor-"
                "model identification. Unobserved heterogeneity is captured "
                "through latent factor loadings rather than a parallel-trends "
                "assumption; the aggregate averages across treated cells with "
                "weights ``W_{it}``."
            ),
            "aggregation": "factor_model",
            "headline_attribute": "att",
            "reference": "REGISTRY.md Sec. TROP",
        }

    if name == "SyntheticControlResults":
        return {
            "name": "SCM ATT (mean post-treatment gap for the single treated unit)",
            "definition": (
                "The average over the post-treatment periods of the gap "
                "``alpha_hat_{1t} = Y_{1t} - sum_j w_j Y_{jt}`` between the single "
                "treated unit and its donor-weighted synthetic control (Abadie, "
                "Diamond & Hainmueller 2010). There is no population-averaging or "
                "sampling estimand — it is the effect on the one treated unit; "
                "significance is assessed by in-space placebo permutation inference "
                "(no analytical standard error)."
            ),
            "aggregation": "single_unit_gap",
            "headline_attribute": "att",
            "reference": "REGISTRY.md Sec. SyntheticControl",
        }

    if name == "SpilloverDiDResults":
        return {
            "name": "total effect on the treated (Butts spillover-aware ATT)",
            "definition": (
                "The total effect on the treated ``tau_total`` from Butts (2021) "
                "ring-indicator spillover DiD, identified off FAR-AWAY control "
                "observations (``d_it > d_bar``, Assumption 5) rather than any "
                "not-yet-/never-treated pool. The estimator decomposes into the "
                "DIRECT effect on treated units plus per-ring spillover-on-control "
                "effects that relax SUTVA within the treated units' spatial "
                "neighborhood; ``att`` is the headline total effect, while the "
                "per-ring ``spillover_effects`` and (when ``event_study=True``) the "
                "per-event-time direct dynamics are available on the result object "
                "for disaggregated inference."
            ),
            "aggregation": "spillover",
            "headline_attribute": "att",
            "reference": "Butts (2021); REGISTRY.md Sec. SpilloverDiD",
        }

    # Marked diagnostic results (spec section 3.5): no causal-effect
    # inference row exists, so no ATT-style target parameter applies.
    # The named ``BaconDecompositionResults`` branch above returns first
    # (its block documents the TWFE coefficient being decomposed). This
    # branch is defensive depth for direct callers: BR / DR reject
    # marked non-Bacon diagnostics before ever calling this helper.
    if isinstance(results, Diagnostic):
        return {
            "name": f"Diagnostic result ({name})",
            "definition": (
                f"``{name}`` is a diagnostic result: it assesses a design, "
                "an identifying assumption, or robustness, and carries no "
                "causal-effect inference row. There is no target parameter; "
                "interpret it alongside the primary estimator's results."
            ),
            "aggregation": "diagnostic",
            "headline_attribute": "",
            "reference": "docs/v4-design.md Sec. 3.5",
        }

    # Default: unrecognized result class. Fall through with a neutral
    # block — agents / downstream consumers can still dispatch on
    # ``aggregation="unknown"`` and fall back to generic ATT narration.
    return {
        "name": "ATT (unrecognized result class)",
        "definition": (
            f"The target parameter for ``{name}`` is not specifically "
            "documented in BR/DR. Defaulting to generic ATT narration; see "
            "the estimator's own documentation for the canonical estimand."
        ),
        "aggregation": "unknown",
        "headline_attribute": "att",
        "reference": "REGISTRY.md",
    }
