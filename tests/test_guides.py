"""Tests for the bundled LLM guide accessor."""

import importlib.resources

import pytest

from diff_diff import get_llm_guide
from diff_diff._guides_api import _VARIANT_TO_FILE


@pytest.mark.parametrize("variant", ["concise", "full", "practitioner", "autonomous"])
def test_all_variants_load(variant):
    text = get_llm_guide(variant)
    assert isinstance(text, str)
    assert len(text) > 1000


def test_default_is_concise():
    assert get_llm_guide() == get_llm_guide("concise")


def test_full_is_largest():
    """`llms-full.txt` is the API-docs roll-up; it should remain larger
    than the short `concise` summary and the workflow-prose
    `practitioner` guide. The `autonomous` reference guide is
    deliberately excluded from this comparison: it serves a different
    audience (LLM agents reasoning about estimator choice) and has
    grown organically through Wave 1 + Wave 2 review rounds with
    estimator-matrix detail, worked examples, and contract citations
    that don't have a counterpart in `llms-full.txt`'s API roll-up.
    Either of the two can be larger without violating any user-facing
    invariant."""
    lengths = {v: len(get_llm_guide(v)) for v in ("concise", "full", "practitioner")}
    assert lengths["full"] > lengths["concise"]
    assert lengths["full"] > lengths["practitioner"]


def test_content_stability_practitioner_workflow():
    assert "8-step" in get_llm_guide("practitioner").lower()


def test_content_stability_self_reference_after_rewrite():
    assert "get_llm_guide" in get_llm_guide("concise")


def test_content_stability_autonomous_fingerprints():
    text = get_llm_guide("autonomous")
    assert "profile_panel" in text
    assert "estimator-support matrix" in text.lower()
    # Wave 2 additions: outcome / dose shape field references.
    assert "outcome_shape" in text
    assert "treatment_dose" in text
    assert "is_count_like" in text
    # has_never_treated is the authoritative ContinuousDiD gate;
    # treatment_dose fields are descriptive only.
    assert "has_never_treated" in text
    # The ContinuousDiD prerequisite summary must continue to mention
    # the duplicate-row hard stop alongside the field-based gates -
    # `_precompute_structures()` silently resolves duplicate cells via
    # last-row-wins, so a reader treating the summary as exhaustive
    # could route duplicate-containing panels into a silent-overwrite
    # path. Guard against that wording regression.
    assert "duplicate_unit_time_rows" in text, (
        "ContinuousDiD prerequisite summary must mention the "
        "`duplicate_unit_time_rows` alert: the precompute path resolves "
        "duplicate (unit, time) cells via last-row-wins, so duplicates "
        "must be removed before fitting."
    )
    # ContinuousDiD also requires strictly positive treated doses
    # (`continuous_did.py:287-294` raises on negative dose support).
    # The autonomous guide must list `dose_min > 0` so an agent reading
    # `treatment_dose.dose_min == -1.5` knows to route the panel away
    # from ContinuousDiD before paying for the failed fit.
    assert "dose_min > 0" in text, (
        "ContinuousDiD prerequisite summary must mention "
        "`dose_min > 0`: the estimator hard-rejects negative treated "
        "dose support at line 287-294 of continuous_did.py."
    )
    # The five profile-side screening checks are necessary but not
    # sufficient: `ContinuousDiD.fit()` takes a separate `first_treat`
    # column (which `profile_panel` does not see) and applies
    # additional validation. The autonomous guide must explicitly
    # mention the `first_treat` validation surface so an agent
    # passing the profile-side screen still knows to validate the
    # `first_treat` column they will supply to `fit()`.
    assert "first_treat" in text, (
        "ContinuousDiD documentation must mention the separate "
        "`first_treat` column that `ContinuousDiD.fit()` validates "
        "(NaN/inf/negative rejection, dose=0 unit drops, force-zero "
        "coercion). The five profile-side screening checks alone are "
        "necessary but not sufficient for fit-time success."
    )


def test_autonomous_contains_worked_examples_section():
    """The §5 worked-examples section walks an agent through three
    end-to-end PanelProfile -> reasoning -> validation flows. Each
    example carries a unique fingerprint phrase keyed off its
    PanelProfile -> estimator path; these regressions guard the
    examples from accidental deletion or scope drift."""
    text = get_llm_guide("autonomous")
    assert "## §5. Worked examples" in text
    # §5.1: binary staggered with never-treated -> CallawaySantAnna
    assert "§5.1 Binary staggered panel with never-treated controls" in text
    assert 'control_group="never_treated"' in text
    # §5.2: continuous dose -> ContinuousDiD prerequisites via treatment_dose
    assert "§5.2 Continuous-dose panel with zero-dose controls" in text
    assert "TreatmentDoseShape(" in text
    # §5.3: count-shaped outcome -> WooldridgeDiD QMLE
    assert "§5.3 Count-shaped outcome" in text
    assert 'WooldridgeDiD(method="poisson")' in text
    assert 'WooldridgeDiD(family="poisson")' not in text, (
        "WooldridgeDiD takes `method=` not `family=`; the wrong kwarg "
        "in the autonomous guide would produce runtime errors when an "
        "agent follows the worked example."
    )


def test_autonomous_count_outcome_uses_asf_outcome_scale_estimand():
    """§4.11 and §5.3 must describe `WooldridgeDiD(method="poisson")`'s
    `overall_att` as an ASF-based outcome-scale difference (matching the
    estimator at `wooldridge.py:1225` and the reporting helper at
    `_reporting_helpers.py:262-281`), NOT as a multiplicative /
    proportional / log-link effect. An agent following an example that
    described the headline as "multiplicative" would misreport the
    scalar - the library's reported `overall_att` is `E[exp(η_1)] -
    E[exp(η_0)]`, a difference on the natural outcome scale.

    Guards against regressing the wording back to "multiplicative
    effect" / "proportional change" framing. Multiplicative
    interpretations may appear in the guide as a clearly-marked
    derived post-hoc reading, but never as the description of the
    estimator's reported `overall_att`."""
    text = get_llm_guide("autonomous")
    # Locate §4.11 and §5.3 blocks; check that within them the Poisson
    # path is described with ASF / outcome-scale wording, NOT as the
    # estimator's reported scalar being multiplicative or proportional.
    sec_4_11_start = text.index("### §4.11 Outcome-shape considerations")
    sec_4_11_end = text.index("## §5. Worked examples")
    sec_4_11 = text[sec_4_11_start:sec_4_11_end]

    sec_5_3_start = text.index("### §5.3 Count-shaped outcome")
    sec_5_3_end = text.index("## §6. Post-fit validation utilities")
    sec_5_3 = text[sec_5_3_start:sec_5_3_end]

    forbidden_phrases = (
        "multiplicative effect under qmle",
        "estimates the multiplicative effect",
        "multiplicative (log-link) effect",
        "report the multiplicative effect",
        "report the multiplicative",
    )
    for section_name, body in (("§4.11", sec_4_11), ("§5.3", sec_5_3)):
        lowered = body.lower()
        for phrase in forbidden_phrases:
            assert phrase not in lowered, (
                f"{section_name} of the autonomous guide describes the "
                f"WooldridgeDiD Poisson `overall_att` with the phrase "
                f"{phrase!r}; the estimator returns an ASF-based "
                f"outcome-scale difference (`E[exp(η_1)] - E[exp(η_0)]`), "
                f"not a multiplicative ratio. See `wooldridge.py:1225` "
                f"and `_reporting_helpers.py:262-281`."
            )

    # Positive: each block must explicitly anchor the estimand to the
    # ASF / outcome-scale framing so future edits can't silently weaken
    # the description.
    assert "ASF" in sec_5_3, "§5.3 must reference the ASF interpretation"
    assert "outcome scale" in sec_5_3.lower(), (
        "§5.3 must label the WooldridgeDiD `overall_att` as an "
        "outcome-scale quantity to prevent multiplicative-ratio drift."
    )


def test_autonomous_negative_dose_path_does_not_route_to_had():
    """The §5.2 negative-dose counter-example must not present
    `HeterogeneousAdoptionDiD` as a direct routing alternative
    when `dose_min < 0`. HAD's contract requires non-negative
    dose support and raises on negative post-period dose
    (`had.py:1450-1459`, paper Section 2). Routing to HAD on a
    negative-dose panel without re-encoding would steer the agent
    into an unsupported estimator path. Guards against the wording
    regressing back to a too-broad "HAD as fallback" framing on
    this branch."""
    text = get_llm_guide("autonomous")
    # Locate counter-example #5 (negative-dose path) within §5.2.
    sec_5_2_start = text.index("### §5.2 Continuous-dose panel")
    sec_5_3_start = text.index("### §5.3 Count-shaped outcome")
    sec_5_2 = text[sec_5_2_start:sec_5_3_start]
    # The negative-dose paragraph must explicitly state HAD is NOT a
    # routing alternative on this branch. We assert the disqualifying
    # phrase is present; we do not forbid `HeterogeneousAdoptionDiD`
    # entirely because the section may legitimately mention it as a
    # candidate AFTER re-encoding.
    assert "HAD" in sec_5_2 or "HeterogeneousAdoptionDiD" in sec_5_2, (
        "§5.2 must mention HAD by name on the negative-dose branch "
        "so its non-applicability can be explicitly called out."
    )
    assert "had.py:1450-1459" in sec_5_2, (
        "§5.2 must cite `had.py:1450-1459` on the negative-dose "
        "branch to anchor HAD's non-negative-dose contract (HAD "
        "raises on negative post-period dose, paper Section 2). "
        "Without this citation, the agent could route a "
        "negative-dose panel directly to HAD and hit a fit-time "
        "error."
    )


def test_autonomous_worked_examples_avoid_recommender_language():
    """Worked examples must mirror the rest of the guide's discipline:
    no prescriptive language in the example reasoning. Multiple paths
    must remain explicit."""
    text = get_llm_guide("autonomous")
    # Locate the §5 block; check its body for forbidden phrasing.
    start = text.index("## §5. Worked examples")
    end = text.index("## §6. Post-fit validation utilities")
    section_5 = text[start:end].lower()
    forbidden = (
        "you should always",
        "always pick",
        "we recommend",
        "the best estimator is",
    )
    for phrase in forbidden:
        assert phrase not in section_5, (
            f"§5 worked examples contain prescriptive phrase {phrase!r}; "
            "the guide must keep multiple paths explicit."
        )


def test_autonomous_contains_intact_estimator_matrix():
    # Section 3 is a markdown table with 10 data columns + the estimator
    # name column -> rows have at least 11 pipe characters. This guards
    # against the matrix being accidentally deleted or truncated.
    text = get_llm_guide("autonomous")
    assert any(
        line.count("|") >= 11 for line in text.splitlines()
    ), "Section 3 estimator-support matrix appears to be missing or truncated."


def test_wheel_content_matches_package_resource():
    for variant, filename in _VARIANT_TO_FILE.items():
        on_disk = (
            importlib.resources.files("diff_diff.guides")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        assert get_llm_guide(variant) == on_disk


def test_utf8_encoding_preserved():
    # llms-full.txt contains the non-ASCII ligature '\u0153' (oe, from
    # "D'Haultfoeuille"); verify UTF-8 roundtrips through the packaged guide.
    text = get_llm_guide("full")
    assert "\u0153" in text


@pytest.mark.parametrize("bad", ["bogus", "", "CONCISE", None, 0, True, ["x"]])
def test_unknown_variant_raises(bad):
    with pytest.raises(ValueError, match="Unknown guide variant"):
        get_llm_guide(bad)


def test_exported_in_namespace():
    import diff_diff

    assert "get_llm_guide" in diff_diff.__all__
    assert callable(diff_diff.get_llm_guide)


def test_module_docstring_mentions_helper():
    import diff_diff

    assert "get_llm_guide" in diff_diff.__doc__


# ---------------------------------------------------------------------------
# llms-full.txt — HeterogeneousAdoptionDiD coverage (Phase 5)
# ---------------------------------------------------------------------------
class TestLLMsFullHADCoverage:
    """Lock the HAD section additions to llms-full.txt against deletion
    or framing drift. Phase 5 surfaces the agent-facing API contract for
    HeterogeneousAdoptionDiD on the bundled-in-wheel guide."""

    def test_llms_full_has_had_section(self):
        text = get_llm_guide("full")
        assert "### HeterogeneousAdoptionDiD" in text

    def test_llms_full_had_results_classes(self):
        text = get_llm_guide("full")
        assert "### HeterogeneousAdoptionDiDResults" in text
        assert "### HeterogeneousAdoptionDiDEventStudyResults" in text

    def test_llms_full_had_pretests_section(self):
        text = get_llm_guide("full")
        assert "## HAD Pretests" in text
        for fn in (
            "did_had_pretest_workflow",
            "qug_test",
            "stute_test",
            "yatchew_hr_test",
            "stute_joint_pretest",
            "joint_pretrends_test",
            "joint_homogeneity_test",
        ):
            assert fn in text, f"HAD Pretests section missing reference to {fn}"

    def test_llms_full_had_choosing_row(self):
        text = get_llm_guide("full")
        # The Choosing-an-Estimator table must list HAD with a row that
        # accurately reflects the contract: HAD targets WAS at the dose
        # support boundary and is compatible with universal-rollout
        # panels (and panels with a small never-treated share — paper
        # edge case at REGISTRY § HeterogeneousAdoptionDiD edge cases).
        idx = text.index("## Choosing an Estimator")
        choosing = text[idx:]
        assert "HeterogeneousAdoptionDiD" in choosing
        # Row must mention WAS as the estimand differentiator (not a
        # blanket "if untreated → not HAD" rule which would be wrong
        # per registry).
        assert "WAS" in choosing

    def test_llms_full_had_section_methodology_compatible_with_untreated(self):
        # Per docs/methodology/REGISTRY.md HeterogeneousAdoptionDiD edge
        # cases (line ~2403): "Authors do NOT require untreated units
        # to be dropped" and (line ~2408) the staggered event-study path
        # explicitly RETAINS never-treated units. The HAD section must
        # NOT carry framing that says HAD is incompatible with
        # never-treated / untreated units.
        text = get_llm_guide("full")
        had_start = text.index("### HeterogeneousAdoptionDiD")
        had_end = text.index("### StackedDiD", had_start)
        had_text = text[had_start:had_end].lower()
        # Negative assertions on framing that contradicts the registry.
        assert "no comparison group" not in had_text
        assert "missing comparison" not in had_text
        forbidden_phrases = (
            "no never-treated units",
            "requires no untreated",
            "drop untreated",
            "must not contain untreated",
            "not compatible with untreated",
        )
        for phrase in forbidden_phrases:
            assert phrase not in had_text, (
                f"HAD section must not carry the phrase {phrase!r}: "
                f"per REGISTRY § HeterogeneousAdoptionDiD edge cases, "
                f"HAD is compatible with a small share of never-treated "
                f"units and explicitly retains them on staggered "
                f"event-study panels (Appendix B.2)."
            )

    def test_llms_full_had_constructor_signature_matches_real_api(self):
        # Documented constructor parameter list must align with the
        # actual HeterogeneousAdoptionDiD.__init__ signature. Catches
        # the failure mode where the guide invents kwargs that don't
        # exist (h, b, rcond) or omits real ones (d_lower, kernel,
        # vcov_type, robust, cluster).
        import inspect

        from diff_diff import HeterogeneousAdoptionDiD

        sig_params = set(inspect.signature(HeterogeneousAdoptionDiD.__init__).parameters)
        sig_params.discard("self")
        text = get_llm_guide("full")
        had_start = text.index("### HeterogeneousAdoptionDiD")
        had_end = text.index("### StackedDiD", had_start)
        had_text = text[had_start:had_end]
        block_start = had_text.index("HeterogeneousAdoptionDiD(")
        # Multi-line signature ends with "\n)" — close-paren on its own
        # line. Searching for ")" alone would hit close-parens inside
        # parameter comments (e.g. "(default)").
        block_end = had_text.index("\n)", block_start)
        ctor_block = had_text[block_start:block_end]
        for param in sig_params:
            assert f"{param}:" in ctor_block or f"{param} " in ctor_block, (
                f"Constructor block in the HAD guide section is missing "
                f"the real public parameter {param!r}. The guide must "
                f"document the actual HeterogeneousAdoptionDiD.__init__ "
                f"signature."
            )

    def test_llms_full_had_fit_signature_matches_real_api(self):
        # Documented fit() parameter list must align with the actual
        # HeterogeneousAdoptionDiD.fit signature.
        import inspect

        from diff_diff import HeterogeneousAdoptionDiD

        sig_params = set(inspect.signature(HeterogeneousAdoptionDiD.fit).parameters)
        sig_params.discard("self")
        text = get_llm_guide("full")
        had_start = text.index("### HeterogeneousAdoptionDiD")
        had_end = text.index("### StackedDiD", had_start)
        had_text = text[had_start:had_end]
        block_start = had_text.index("had.fit(")
        block_end = had_text.index(") -> ", block_start)
        fit_block = had_text[block_start:block_end]
        for param in sig_params:
            assert f"{param}:" in fit_block or f"{param} " in fit_block, (
                f"fit() block in the HAD guide section is missing the "
                f"real public parameter {param!r}. The guide must "
                f"document the actual HeterogeneousAdoptionDiD.fit "
                f"signature."
            )
        # 3.7.0: survey= / weights= were removed from HAD.fit(); the guide
        # signature must not re-introduce them (they are no longer real
        # parameters, so an agent copying them would hit a TypeError). The
        # `survey:` / `weights:` forms do not collide with `survey_design:`.
        for removed in ("survey:", "weights:"):
            assert removed not in fit_block, (
                f"HAD fit() guide block must not document the removed "
                f"{removed!r} parameter (dropped in 3.7.0; survey_design= "
                f"is the sole weighting entry)."
            )

    def test_llms_full_paper_citation(self):
        # Lead-author "D'Haultfœuille" appears in the HAD section.
        # Naturally preserves the UTF-8 'œ' fingerprint asserted by
        # test_utf8_encoding_preserved without a synthetic mark.
        text = get_llm_guide("full")
        had_start = text.index("### HeterogeneousAdoptionDiD")
        had_end = text.index("### StackedDiD", had_start)
        had_text = text[had_start:had_end]
        assert "D'Haultfœuille" in had_text

    def test_llms_full_had_results_class_field_lists_match_real_dataclass(self):
        # Every public dataclass field on HeterogeneousAdoptionDiDResults
        # and HeterogeneousAdoptionDiDEventStudyResults must appear in the
        # documented field table. Catches the failure mode where new
        # result fields land but the guide isn't updated, so agents
        # treating llms-full.txt as the authoritative surface miss
        # available diagnostics / metadata.
        import dataclasses

        from diff_diff import (
            HeterogeneousAdoptionDiDEventStudyResults,
            HeterogeneousAdoptionDiDResults,
        )

        text = get_llm_guide("full")

        # Single-period result class
        sp_start = text.index("### HeterogeneousAdoptionDiDResults")
        sp_end = text.index("### HeterogeneousAdoptionDiDEventStudyResults", sp_start)
        sp_block = text[sp_start:sp_end]
        for field in dataclasses.fields(HeterogeneousAdoptionDiDResults):
            assert f"`{field.name}`" in sp_block, (
                f"HeterogeneousAdoptionDiDResults guide block is missing "
                f"the public dataclass field {field.name!r}. The table "
                f"must enumerate every field so agents see all available "
                f"diagnostics / metadata."
            )

        # Event-study result class
        es_start = text.index("### HeterogeneousAdoptionDiDEventStudyResults")
        es_end = text.index("### TROPResults", es_start)
        es_block = text[es_start:es_end]
        for field in dataclasses.fields(HeterogeneousAdoptionDiDEventStudyResults):
            assert f"`{field.name}`" in es_block, (
                f"HeterogeneousAdoptionDiDEventStudyResults guide block "
                f"is missing the public dataclass field {field.name!r}."
            )

    def test_llms_full_had_section_documents_mass_point_survey_vcov_requirement(self):
        # Per had.py:3495-3507 the mass-point design rejects the default
        # classical vcov family on the survey_design= path
        # (NotImplementedError). The HAD section must surface this
        # requirement so an agent reading llms-full.txt and writing a
        # weighted mass-point fit knows to pass vcov_type='hc1'
        # explicitly. Without this caveat the documented fit() example
        # can fail at fit time on a mass-point panel.
        text = get_llm_guide("full")
        had_start = text.index("### HeterogeneousAdoptionDiD")
        had_end = text.index("### StackedDiD", had_start)
        had_text = text[had_start:had_end]
        # Must mention the mass-point + survey vcov requirement.
        # Accept either explicit "vcov_type" mention near "mass" wording
        # or the explicit "hc1" / "robust=True" pairing with mass-point.
        lower = had_text.lower()
        assert "vcov_type" in lower and ("mass-point" in lower or "mass_point" in lower), (
            "HAD section must document the mass-point + survey vcov "
            "requirement: passing vcov_type='hc1' (or robust=True) is "
            "required on design='mass_point' under survey_design= "
            "(per had.py:3495-3507). Without this caveat the documented "
            "weighted fit example can raise NotImplementedError."
        )
        # 3.7.0: the mass-point guidance must not reference the removed
        # weights= shortcut (fit(weights=<array>) no longer exists).
        assert "weights=` shortcut" not in had_text, (
            "HAD section must not describe the removed `weights=` shortcut "
            "(dropped in 3.7.0); the sole weighting entry is survey_design=."
        )

    def test_llms_full_had_variance_formula_describes_all_designs(self):
        # After the 3.7.0 survey_design= consolidation, HAD.fit() emits only
        # the Binder-TSL labels: weighted continuous fits populate
        # "survey_binder_tsl" and weighted mass-point fits
        # "survey_binder_tsl_2sls" (the pweight / pweight_2sls labels were
        # removed with the weights= kwarg). The documented description must
        # cover BOTH survey labels so agents reading the guide on a weighted
        # mass-point fit do not misread the available inference metadata.
        text = get_llm_guide("full")
        sp_start = text.index("### HeterogeneousAdoptionDiDResults")
        sp_end = text.index("### HeterogeneousAdoptionDiDEventStudyResults", sp_start)
        sp_block = text[sp_start:sp_end]
        # Find the variance_formula row in the table.
        for line in sp_block.splitlines():
            if line.startswith("| `variance_formula`"):
                for label in (
                    "survey_binder_tsl",
                    "survey_binder_tsl_2sls",
                ):
                    assert label in line, (
                        f"variance_formula row must enumerate the {label!r} "
                        f"label - weighted mass-point fits populate "
                        f"survey_binder_tsl_2sls per had.py. Line: {line!r}"
                    )
                # The pweight / pweight_2sls labels were removed with the
                # weights= kwarg in 3.7.0 and must not reappear.
                assert "pweight" not in line, (
                    f"variance_formula row must not mention the removed "
                    f"pweight labels after the 3.7.0 consolidation. "
                    f"Line: {line!r}"
                )
                break
        else:
            pytest.fail("variance_formula row not found in HAD results table")
        # effective_dose_mean: must mention mass-point Wald-IV dose gap.
        for line in sp_block.splitlines():
            if line.startswith("| `effective_dose_mean`"):
                assert "mass_point" in line or "Wald-IV" in line or "mass-point" in line, (
                    f"effective_dose_mean row must mention mass-point "
                    f"semantics - weighted mass-point fits populate the "
                    f"weighted Wald-IV dose gap per had.py:3642-3660. "
                    f"Line: {line!r}"
                )
                break
        else:
            pytest.fail("effective_dose_mean row not found in HAD results table")

    def test_llms_full_had_event_study_mirrors_weighted_metadata_semantics(self):
        # R9 P1 (Documentation/Tests): the event-study results table at
        # ### HeterogeneousAdoptionDiDEventStudyResults must enumerate the
        # SAME variance_formula labels and the SAME mass-point / Wald-IV
        # semantics for effective_dose_mean as the single-period table; the
        # event-study fit path populates these fields with the same labels.
        # After the 3.7.0 survey_design= consolidation those are the two
        # Binder-TSL labels (the pweight / pweight_2sls labels were removed
        # with the weights= kwarg). Without parallel coverage, agents
        # reading the guide on an event-study fit cannot identify the
        # inference family.
        text = get_llm_guide("full")
        es_start = text.index("### HeterogeneousAdoptionDiDEventStudyResults")
        # End at the next top-level result section (### TROPResults is the
        # next entry after the HAD event-study block per llms-full.txt).
        es_end = text.index("### TROPResults", es_start)
        es_block = text[es_start:es_end]
        for line in es_block.splitlines():
            if line.startswith("| `variance_formula`"):
                for label in (
                    "survey_binder_tsl",
                    "survey_binder_tsl_2sls",
                ):
                    assert label in line, (
                        f"event-study variance_formula row must enumerate "
                        f"{label!r} (event-study path applies the same "
                        f"label uniformly across horizons). Line: {line!r}"
                    )
                # The pweight / pweight_2sls labels were removed with the
                # weights= kwarg in 3.7.0 and must not reappear.
                assert "pweight" not in line, (
                    f"event-study variance_formula row must not mention the "
                    f"removed pweight labels after the 3.7.0 consolidation. "
                    f"Line: {line!r}"
                )
                break
        else:
            pytest.fail(
                "event-study variance_formula row not found in HAD event-study results table"
            )
        for line in es_block.splitlines():
            if line.startswith("| `effective_dose_mean`"):
                assert "mass_point" in line or "Wald-IV" in line or "mass-point" in line, (
                    f"event-study effective_dose_mean row must mention "
                    f"mass-point Wald-IV semantics (event-study path "
                    f"populates the same denominator per had.py:721-734). "
                    f"Line: {line!r}"
                )
                break
        else:
            pytest.fail(
                "event-study effective_dose_mean row not found in HAD event-study results table"
            )

    def test_llms_practitioner_step_4_distinguishes_had_from_continuous(self):
        # The official practitioner workflow guide (returned by
        # get_llm_guide("practitioner")) routes continuous treatments. It
        # must distinguish ContinuousDiD (per-dose ATT(d), requires
        # never-treated controls) from HeterogeneousAdoptionDiD (WAS at
        # dose boundary, compatible with universal rollout). Pre-PR the
        # decision tree routed ALL continuous-intensity designs to
        # ContinuousDiD - which is wrong for no-untreated panels.
        text = get_llm_guide("practitioner")
        # Locate the Step 4 decision tree.
        s4_start = text.index("## Step 4: Choose Estimation Method")
        # Step 5 is the next section header; cap the slice there.
        s5_start = text.index("## Step ", s4_start + 1)
        s4_block = text[s4_start:s5_start]
        # Both HAD and ContinuousDiD must appear in the continuous branch.
        assert "HeterogeneousAdoptionDiD" in s4_block, (
            "practitioner guide Step 4 decision tree must mention "
            "HeterogeneousAdoptionDiD as the alternative to ContinuousDiD "
            "on no-untreated / universal-rollout panels."
        )
        assert "ContinuousDiD" in s4_block
        # Universal-rollout / no-untreated framing should be present so
        # readers know which branch routes where.
        assert "never-treated" in s4_block.lower() or "untreated" in s4_block.lower(), (
            "practitioner guide Step 4 must describe the never-treated / "
            "universal-rollout distinction that drives the HAD vs "
            "ContinuousDiD routing."
        )

    def test_llms_practitioner_step_4_continuous_routes_estimand_first(self):
        """The continuous-treatment branch must route by ESTIMAND first
        (WAS vs ATT(d)/ACRT(d)), NOT by untreated-unit presence alone.
        Per REGISTRY HeterogeneousAdoptionDiD edge cases, HAD remains
        valid with a small never-treated share; the actual differentiator
        is the target estimand. Pre-pilot-402 the decision tree routed
        any panel with never-treated units to ContinuousDiD, which
        misroutes WAS-target practitioners on panels that happen to
        include a never-treated share.
        """
        text = get_llm_guide("practitioner")
        s4_start = text.index("## Step 4: Choose Estimation Method")
        s5_start = text.index("## Step ", s4_start + 1)
        s4_block = text[s4_start:s5_start].lower()
        # The estimand-first framing must appear: both estimand labels
        # must be on the routing side (left of -> arrows), not just
        # buried inside a single estimator's description.
        assert "target estimand" in s4_block or "route by estimand" in s4_block, (
            "Step 4 continuous-treatment branch must use explicit "
            "estimand-first routing language ('target estimand', "
            "'route by estimand') so the WAS vs ATT(d) choice "
            "comes first, not 'never-treated present?'."
        )
        # And the WAS-with-small-never-treated-share compatibility
        # must be stated explicitly so HAD's edge case isn't masked.
        assert "small never-treated share" in s4_block, (
            "Step 4 must explicitly note that HAD remains valid "
            "with a small never-treated share; otherwise readers "
            "will route HAD-appropriate WAS panels to ContinuousDiD."
        )

    def test_llms_full_had_pretests_documents_earlier_pre_period_precondition(self):
        # Same precondition as the practitioner test: per
        # docs/methodology/REGISTRY.md HeterogeneousAdoptionDiD
        # § "Assumption 7 / step 2 closure" + had_pretests.py:4738-4756 +
        # 2769, aggregate="event_study" closes step 2 ONLY IF the
        # panel carries at least one earlier placebo pre-period beyond
        # the base F-1. The HAD Pretests section in llms-full.txt must
        # document this precondition so agents do not assume any
        # multi-period event-study fit closes step 2.
        text = get_llm_guide("full")
        pretests_start = text.index("## HAD Pretests")
        pretests_end = text.index("## Honest DiD", pretests_start)
        pretests_block = text[pretests_start:pretests_end]
        lower = pretests_block.lower()
        assert "earlier" in lower and ("pre-period" in lower or "placebo" in lower), (
            "HAD Pretests section must document the 'earlier pre-period' "
            "precondition for step-2 closure on the event-study path."
        )
        assert "skipped" in lower or "pretrends_joint=none" in lower, (
            "HAD Pretests section must surface the "
            "'joint pre-trends skipped' / pretrends_joint=None fallback "
            "when no earlier pre-period exists."
        )

    def test_llms_full_had_pretests_assumption_labels_correct(self):
        # Per docs/methodology/REGISTRY.md HeterogeneousAdoptionDiD
        # § "Assumptions / Theorems / Estimators":
        #   - Assumption 5 = Design 1 sign identification (NOT testable)
        #   - Assumption 6 = Design 1 WAS_d_lower identification (NOT testable)
        #   - Assumption 7 = pre-trends (paper Section 4.2 step 2)
        #   - Assumption 8 = linearity (paper Section 4.2 step 3)
        # The HAD Pretests section must NOT mislabel these:
        #   - qug_test is the support-infimum test (H0: d_lower = 0),
        #     NOT "Assumption 5" (which is non-testable per registry).
        #   - stute_test is Assumption 8 (linearity), NOT Assumption 7.
        text = get_llm_guide("full")
        pretests_start = text.index("## HAD Pretests")
        pretests_end = text.index("## Honest DiD", pretests_start)
        pretests_block = text[pretests_start:pretests_end]
        # qug_test bullet: must positively label QUG as a support-infimum
        # test, NOT as a positive "Assumption 5 support condition" claim
        # (a negative disclaimer "does NOT test Assumption 5" is fine).
        forbidden_qug_positive_claims = (
            "Assumption 5 support condition",
            "QUG (Assumption 5",
            "qug_test`) — Assumption 5",
            "qug_test(d)` — Assumption 5",
        )
        # stute_test bullet: must positively label as Assumption 8
        # linearity, NOT as Assumption 7 mean-independence.
        forbidden_stute_positive_claims = (
            "stute_test(d, dy)` — Assumption 7",
            "Stute (Assumption 7",
            "Assumption 7 mean-independence",
        )
        for line in pretests_block.splitlines():
            if line.startswith("- `qug_test"):
                # Positive claim of what QUG IS:
                assert (
                    "support-infimum" in line
                    or "support infimum" in line
                    or "Theorem 4" in line
                    or "H_0: d_lower" in line
                ), (
                    f"qug_test bullet must positively label QUG as the "
                    f"support-infimum / Theorem-4 test. Line: {line!r}"
                )
                for phrase in forbidden_qug_positive_claims:
                    assert phrase not in line, (
                        f"qug_test bullet must not positively claim QUG "
                        f"is an 'Assumption 5' test ({phrase!r}). QUG "
                        f"tests H_0: d_lower = 0; Assumption 5 is the "
                        f"Design 1 sign-identification condition (NOT "
                        f"testable per registry). A negative disclaimer "
                        f"that QUG does NOT test Assumption 5 is fine. "
                        f"Line: {line!r}"
                    )
            if line.startswith("- `stute_test"):
                # Positive claim of what Stute IS:
                assert "Assumption 8" in line or "linearity" in line.lower(), (
                    f"stute_test bullet must positively label as "
                    f"Assumption 8 / linearity test. Line: {line!r}"
                )
                for phrase in forbidden_stute_positive_claims:
                    assert phrase not in line, (
                        f"stute_test bullet must not positively claim "
                        f"Stute is an Assumption 7 mean-independence "
                        f"test ({phrase!r}). stute_test is Assumption 8 "
                        f"linearity (paper Section 4.2 step 3); "
                        f"Assumption 7 is pre-trends (step 2, only "
                        f"covered on the event-study path). Line: {line!r}"
                    )


class TestLLMsFullStackedDiDCoverage:
    """Pin the StackedDiD section of llms-full.txt to the real API.

    Adding a public parameter (here: balance= on __init__, covariates= on fit())
    requires updating diff_diff/guides/llms-full.txt — these tests catch drift.
    """

    def _stacked_section(self):
        text = get_llm_guide("full")
        start = text.index("### StackedDiD")
        nxt = text.index("\n### ", start + 1)
        return text[start:nxt]

    def test_llms_full_has_stacked_section(self):
        assert "### StackedDiD" in get_llm_guide("full")

    def test_llms_full_stacked_constructor_signature_matches_real_api(self):
        import inspect

        from diff_diff import StackedDiD

        sig_params = set(inspect.signature(StackedDiD.__init__).parameters)
        sig_params.discard("self")
        section = self._stacked_section()
        block_start = section.index("StackedDiD(")
        block_end = section.index("\n)", block_start)
        ctor_block = section[block_start:block_end]
        for param in sig_params:
            assert f"{param}:" in ctor_block or f"{param} " in ctor_block, (
                f"StackedDiD constructor block in llms-full.txt is missing the real "
                f"public parameter {param!r} (adding a public param requires updating "
                f"the guide)."
            )

    def test_llms_full_stacked_fit_documents_covariates(self):
        import inspect

        from diff_diff import StackedDiD

        assert "covariates" in inspect.signature(StackedDiD.fit).parameters
        section = self._stacked_section()
        fit_start = section.index("stacked.fit(")
        fit_block = section[fit_start : section.index("\n)", fit_start)]
        assert "covariates" in fit_block, (
            "StackedDiD.fit() exposes covariates= but the llms-full.txt fit() block "
            "does not document it."
        )

    def test_llms_full_stacked_fit_aggregate_line_documents_shim(self):
        # M-024: the documented fit signature must carry the sentinel
        # default + deprecation marker (the dCDH precedent's form), not
        # the stale `aggregate: str = None` - no other pin covers this
        # line, so it could silently go stale.
        section = self._stacked_section()
        fit_start = section.index("stacked.fit(")
        fit_block = section[fit_start : section.index("\n)", fit_start)]
        agg_line = next(
            line for line in fit_block.splitlines() if line.strip().startswith("aggregate")
        )
        assert "NOT_SUPPLIED" in agg_line
        assert "DEPRECATED (M-024)" in agg_line
        assert "results.aggregate()" in agg_line
        # balance= must be documented somewhere in the section (constructor param)
        assert "balance" in section


class TestLLMsFullLPDiDCoverage:
    """Pin the LPDiD section of llms-full.txt to the real API.

    Adding a public parameter to LPDiD.__init__ or LPDiD.fit() requires updating
    diff_diff/guides/llms-full.txt — these tests catch drift.
    """

    def _lpdid_section(self):
        text = get_llm_guide("full")
        start = text.index("### LPDiD")
        nxt = text.index("\n### ", start + 1)
        return text[start:nxt]

    def test_llms_full_has_lpdid_section(self):
        assert "### LPDiD" in get_llm_guide("full")

    def test_llms_full_lpdid_constructor_signature_matches_real_api(self):
        import inspect

        from diff_diff import LPDiD

        sig_params = set(inspect.signature(LPDiD.__init__).parameters)
        sig_params.discard("self")
        section = self._lpdid_section()
        block_start = section.index("LPDiD(")
        block_end = section.index("\n)", block_start)
        ctor_block = section[block_start:block_end]
        for param in sig_params:
            assert f"{param}:" in ctor_block or f"{param} " in ctor_block, (
                f"LPDiD constructor block in llms-full.txt is missing the real "
                f"public parameter {param!r} (adding a public param requires updating "
                f"the guide)."
            )

    def test_llms_full_lpdid_fit_signature_matches_real_api(self):
        import inspect

        from diff_diff import LPDiD

        sig_params = set(inspect.signature(LPDiD.fit).parameters)
        sig_params.discard("self")
        section = self._lpdid_section()
        fit_start = section.index("lpdid.fit(")
        fit_block = section[fit_start : section.index(") -> ", fit_start)]
        for param in sig_params:
            assert f"{param}:" in fit_block or f"{param} " in fit_block, (
                f"LPDiD.fit() block in llms-full.txt is missing the real public "
                f"parameter {param!r} (adding a public param requires updating the guide)."
            )


class TestLLMsFullEfficientDiDShimLine:
    """M-023/M-120: the documented EfficientDiD fit signature must carry the
    sentinel defaults + deprecation markers (the M-024 precedent) - no other
    pin covers these lines, so they could silently go stale."""

    def _efficient_section(self):
        text = get_llm_guide("full")
        start = text.index("### EfficientDiD")
        nxt = text.index("\n### ", start + 1)
        return text[start:nxt]

    def test_llms_full_efficient_fit_aggregate_line_documents_shim(self):
        section = self._efficient_section()
        fit_start = section.index("edid.fit(")
        fit_block = section[fit_start : section.index("\n)", fit_start)]
        agg_line = next(
            line for line in fit_block.splitlines() if line.strip().startswith("aggregate")
        )
        assert "NOT_SUPPLIED" in agg_line
        assert "DEPRECATED (M-023)" in agg_line
        assert "results.aggregate()" in agg_line
        bal_line = next(
            line for line in fit_block.splitlines() if line.strip().startswith("balance_e")
        )
        assert "NOT_SUPPLIED" in bal_line
        assert "DEPRECATED (M-120)" in bal_line

    def _section(self, header):
        text = get_llm_guide("full")
        start = text.index(header)
        nxt = text.index("\n### ", start + 1)
        return text[start:nxt]

    def _assert_shim_lines(self, section, fit_call, agg_row, bal_row):
        fit_start = section.index(fit_call)
        fit_block = section[fit_start : section.index("\n)", fit_start)]
        agg_line = next(
            line for line in fit_block.splitlines() if line.strip().startswith("aggregate")
        )
        assert "NOT_SUPPLIED" in agg_line
        assert f"DEPRECATED ({agg_row})" in agg_line
        assert "results.aggregate()" in agg_line
        bal_line = next(
            line for line in fit_block.splitlines() if line.strip().startswith("balance_e")
        )
        assert "NOT_SUPPLIED" in bal_line
        assert f"DEPRECATED ({bal_row})" in bal_line

    def test_llms_full_imputation_fit_aggregate_line_documents_shim(self):
        # The M-021/M-118 twin of the EfficientDiD pin above - no other
        # test covers the ImputationDiD signature block's shim comments.
        self._assert_shim_lines(self._section("### ImputationDiD"), "imp.fit(", "M-021", "M-118")

    def test_llms_full_two_stage_fit_aggregate_line_documents_shim(self):
        self._assert_shim_lines(self._section("### TwoStageDiD"), ".fit(", "M-022", "M-119")
