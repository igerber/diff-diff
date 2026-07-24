# Pre-Registered Decision Rule: Plan-Review Engine Evaluation

Committed BEFORE any campaign run (the PR that adds this file timestamps the
pre-registration). The verdict is computed MECHANICALLY (`verdict.py`) from the
final graded table against these rules; the rules may not be edited after the
campaign starts.

## The experiment

| Arm | Engine (variant) | Reviewer(s) (mode) | Claude model | Repeats | Role |
|-----|------------------|--------------------|--------------|---------|------|
| A | control (pinned-SHA review-plan.md) | single | claude-fable-5 | k=2 | control (production workflow) |
| B | candidate | single | claude-fable-5 | k=2 | primary: regression gate vs A |
| C | candidate | dual: + codex gpt-5.6-sol @ xhigh | claude-fable-5 | k=2 | primary: is dual worth it vs B |
| D | candidate | single | claude-sonnet-5 | k=2 | probe (non-gating; decision-grade for model default) |
| E | candidate | dual: + codex gpt-5.6-terra @ xhigh | claude-fable-5 | k=2 | probe (non-gating; decision-grade for codex default) |

Two production decisions gate on this campaign: A vs B (did the new engine
regress?) and C vs B (does the second, adversarial codex reviewer add reliable
catches?). D and E never gate; at k=2 their reads are decision-grade for the
cheaper-model defaults if the primaries pass. All arms run k=2, which also
closes the repeat-count partition leak the reviewer-eval campaign accepted.

For dual arms the GRADED ARTIFACT is the merged+verified report (what the
production engine emits); raw reviewer pairs are stored for diagnostics only.

## Severity vocabulary (neutral scale)

Ground truth, extraction output, grading, and the FP gate all use ONE neutral
scale so neither engine's native vocabulary (CRITICAL/MEDIUM/LOW vs P0–P3) can
unblind or skew grading:

| Neutral | Control-engine labels | Candidate-engine labels |
|---------|----------------------|-------------------------|
| `blocker` | CRITICAL | P0, P1 |
| `major` | MEDIUM | P2 |
| `minor` | LOW | P3 |

## Definitions

- **Reliably caught** (per arm, per ground-truth defect): the graded table
  marks it `caught` in EVERY OK repeat of that arm (INFRA_ERROR repeats are
  excluded from the denominator — infra noise is not a recall signal).
- **Unstably caught**: caught in some but not all OK repeats.
- **Missed**: caught in no repeat. `partial` (right file or class, wrong
  defect or location) counts as missed everywhere.
- **False positive (FP)**: on an `expect_no_blockers` (s3_negative) case, any
  extracted finding whose severity is above that case's `allow_severities`
  (default allows `major`/`minor`, i.e. any `blocker` is an FP). Known-FP
  topics listed by the case are never counted — topics carry stable `id`s in
  `case.json`, graders mark a matching FP row with `known_topic_id` (auditable
  rather than omitted), and `verdict.py` excludes those rows mechanically.
- **UNDETERMINED**: when either compared arm has FEWER OK repeats than its
  scheduled k=2 on any compared case (an infra-shortened arm has less FP
  exposure and would win the FP gate by failing more), the verdict is
  UNDETERMINED — never GO/NO-GO/PARITY. Re-run the failed repeats; unequal
  exposure is not comparable evidence.

## Primary gates (B vs A)

1. **Regression (NO-GO):** any `must_catch` defect that A reliably catches and
   B misses in both B-repeats. (A unstably-caught vs B missed → flagged for
   judgment in the report, not an automatic NO-GO.)
2. **FP gate (NO-GO):** B's total FP count across s3_negative cases exceeds A's.
3. **GO:** zero regressions AND the FP gate passes AND at least one strict
   improvement: B reliably catches a defect A misses in both A-repeats, OR B
   has strictly fewer FPs than A (with no catch regression).
4. **PARITY:** zero regressions but no strict improvement → the user decides
   (the mechanical hash-gate fixes land regardless; PARITY only means the new
   criteria showed no measurable review-quality gain).

## Dual read (C vs B) — same gate structure, separate decision

C vs B is evaluated with the same regression/FP/improvement structure. Dual
earns the "Recommended" slot in the production three-way ask only for plan
classes where it shows RELIABLE marginal catches over B (a defect C reliably
catches that B misses in both repeats). If C only matches B, single-reviewer
stays the default and dual remains a manual escalation.

## Probe reads (informational, never gating)

- **D vs B** (same engine, cheaper Claude): if D matches B's reliable catches
  with no added FPs, the cheaper model becomes the default reviewer.
- **E vs C** (same dual engine, cheaper codex): if E matches C's reliable
  catches with no added FPs, terra becomes the default codex side.

## Grading protocol (blinded, extraction-based, multi-grader)

1. After `run`, the `extract` stage reduces EVERY stored review to a uniform
   findings schema (defect claim, cited location, neutral severity, verbatim
   evidence quote) using the pinned extraction model — closing the
   report-structure leak: old vs new engine formats and dual-arm agreement tags
   would otherwise unblind the very contrast being judged.
2. `compare --blinded` bundles the EXTRACTIONS (never the raw reviews) with
   arm identities replaced by neutral `M*` labels, model references scrubbed,
   and latency/model/CLI metadata redacted; `blinding.json` (the label→arm
   mapping) is sealed — graders NEVER read it.
3. **2 INDEPENDENT graders** (separate subagent contexts) each read ONLY the
   blinded bundle plus this rubric and fill, per (case, ground-truth defect,
   arm label, repeat): `caught` / `partial` / `missed`, with the extraction's
   evidence quote; plus ONE negative-assessment cell per (negative case, arm
   label, repeat) whose `findings` list may be empty — an omitted cell fails
   validation, never reads as zero FPs. Each grader copies the bundle's
   embedded **Bundle ID** into the table; the id IS the hash of the exact
   grader-visible bundle bytes (id slot tokenized), so every rendered thing —
   per-run extraction text, header, labels — is bound by construction.
   `verdict` re-hashes the bundle file and rejects a table whose id does not
   match; extraction prompt/model identity is enforced separately by the
   per-run extraction metadata that `compare` requires to be homogeneous; and
   any verdict computed without a blinded bundle is labeled NON-GATING.
4. **Hallucination check:** a `caught` entry's evidence quote must actually
   name the defect; a finding claiming to have verified behavior the plan/repo
   cannot produce is graded as an FP-class note, never a catch.
5. **Adversarial reconciliation:** any cell where the graders disagree goes to
   a verifier agent that re-reads that case's blinded extractions (and, only
   on dispute, may consult the raw reviews) and rules with quoted evidence.
6. Only after the reconciled table is final are labels unblinded via
   `blinding.json` and the gates applied mechanically (`verdict.py`).

## Blinding caveats (accepted)

- Extraction is itself an LLM stage and could drop a catch; mitigations: the
  verbatim-quote requirement, the dress rehearsal (an extraction error that
  drops a fixture catch fails the rehearsal), and raw-on-dispute in
  reconciliation. Raw reviews are stored unmodified for post-verdict analysis.
- Sanitization is best-effort; graders are instructed to grade on content and
  never on guessed identity.

## Corpus floor

The campaign runs only with ≥ 8 verified cases including ≥ 3 s2_historical
(defect visible in the plan text — a later CI finding alone does NOT qualify;
the user spot-checks borderline s2 labels) and ≥ 2 s3_negative (the FP gate is
vacuous without negative controls). Below the floor: stop and surface rather
than produce a weak read. Enforced mechanically: `verdict` checks the
manifest's corpus composition and the k=2 design and labels any verdict
computed under violations **NON-GATING** (`gating: false` with the violations
listed) — a rehearsal or subset run can never be mistaken for a campaign-grade
decision.

## Campaign-readiness gate (from the approved step-1 plan)

The campaign does not start until (1) the harness + mechanics PRs are merged
with the full `/ai-review-local` → CI AI review cycle clean, with
DECISION_RULE.md and candidates/ explicitly named as methodology review
surfaces; and (2) the dress rehearsal passes: the full pipeline — `run` (k=2)
→ `extract` → `compare --blinded` → two-grader mini pass → `verdict` — end to
end on the committed fixture case, including the dual arm C (so the merge path
is exercised before campaign spend).

The rehearsal additionally includes a **format-parity extraction probe**: the
committed control-format sample review
(`corpus/fixture/fx-mini-plan/control_format_review.md`), whose ONLY mention
of the seeded `safe_inference_v2` defect sits under the control engine's
`## Questions for the Author` section, is fed through the extraction stage,
and the extraction must retain that defect as a finding (at least `major`).
An extraction that drops it fails the rehearsal — a format-specific rule that
discarded control-native question findings would bias every arm contrast
toward the candidate and manufacture a false GO.

## Campaign 1 outcome (2026-07-24) — EXPLORATORY / NON-GATING

**This campaign is NON-GATING under its own rules** — treat it as a strong
directional signal, not a passed pre-registration. Three deviations from the
protocol above:

1. **Corpus floor / FP gate unmet.** All three s3 negatives turned out
   base_sha-contaminated (real-defect-laden, not clean), so the severity-
   threshold FP gate (line 64) was invalid and NOT applied; precision was
   re-measured post hoc as a true-vs-false hallucination rate. With zero valid
   negatives the corpus floor (≥2, line 133) is unmet, which by line 135-139
   makes the verdict **NON-GATING**.
2. **Criteria regression (would be a NO-GO).** Primary gate 1 makes any
   must-catch defect A reliably catches and B misses an automatic NO-GO;
   `trop-silent-drop` is exactly that (A caught, B/C missed). The candidate
   criteria therefore did **not** pass the A-vs-B regression gate — the rewrite
   was at best a wash (one improvement, one regression), tracked for a criteria
   re-validation.
3. **Model deviation.** The arms above register `claude-fable-5`; the campaign
   was run on `claude-opus-4-8`.

What the campaign **does** show, robustly and independent of the unmet FP gate:
dual review (Claude @ Opus 4.8 + codex-`gpt-5.6-sol` @ xhigh) reliably caught
**7/9** must-catch defects vs **1/9** for every single-reviewer arm (control,
candidate-single, Sonnet), with **0 regressions** in the B→C contrast; codex-
**sol** ≫ codex-terra (7/9 vs 2/9); dual hallucination rate **3.4%** vs 0%
single, but the dual arm's extra findings were 31 true / 2 false — favorable.
That sensitivity signal is what selects **dual** as the default. Full numbers in
`verdicts/campaign-1.md`. The engine promoted to `.claude/skills/plan-review/`
is arm C **as run** (candidate criteria + dual), carrying the `trop-silent-drop`
regression as a tracked follow-up. A clean, pre-registered re-validation
(uncontaminated negatives, the production-adapted merge prompt + model, an
explicit criteria-regression rule) is the tracked path to a gating result.
