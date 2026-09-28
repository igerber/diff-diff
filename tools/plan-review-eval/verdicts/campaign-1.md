# Plan-review engine — Campaign 1 verdict (2026-07-24)

Committed aggregate result of the first plan-review measurement campaign. The
real-plan corpus stays local (gitignored); this file records numbers and
methodology lessons only, per the workflow-eval data-handling rule.

## Status — EXPLORATORY / NON-GATING

This campaign is a strong **directional signal**, not a passed pre-registration.
Under `DECISION_RULE.md`'s own rules it is **NON-GATING**: the s3 negatives were
base_sha-contaminated (so the FP gate was invalid and the ≥2-valid-negative
corpus floor unmet → NON-GATING per that doc's line 135-139); the candidate
criteria carried a must-catch regression (`trop-silent-drop`) that Primary gate 1
defines as a NO-GO; and the run used `claude-opus-4-8` where the arms register
`claude-fable-5`. What survives all of that is the **sensitivity** result (dual
7/9 vs single 1/9, 0 regressions B→C), which is what selects dual as the
default. The engine shipped to `.claude/skills/plan-review/` is arm C **as run**
(candidate criteria + dual); a clean re-validation is a tracked follow-up. See
the DECISION_RULE.md "Campaign 1 outcome" note.

## Design

- **Corpus**: 12 cases — 5 `s1_synthetic` (real donor plan + injected defects,
  tiered easy/medium/hard), 4 `s2_historical` (real plans whose implementations
  hit plan-visible trouble), 3 `s3_negative` (intended clean controls).
- **Arms** (single-field contrasts; `treatment_fields = variant, mode, model`):
  - A control — old criteria (pinned SHA), single Claude
  - B candidate — new criteria, single Claude
  - C candidate — new criteria, **dual** (Claude + codex `gpt-5.6-sol`)
  - D candidate — new criteria, single Claude @ `claude-sonnet-5` (probe)
  - E candidate — new criteria, dual (Claude + codex `gpt-5.6-terra`) (probe)
- **Reviewer model**: `claude-opus-4-8` (A/B/C/E Claude side); codex @ `xhigh`.
  Extraction stage `claude-sonnet-5`.
- **Protocol**: k=2, 120 blinded reviews, neutral-severity extraction, sealed
  arm→label mapping, 2 independent graders (99% cell agreement, conservative
  reconciliation). Pre-registered `DECISION_RULE.md`.

## Sensitivity — reliably-caught (both k=2 repeats)

| Arm | what it is | must-catch (of 9) | all defects (of 18) |
|-----|-----------|:---:|:---:|
| A | control, old criteria, single | 1/9 | 2/18 |
| B | candidate, new criteria, single | 1/9 | 4/18 |
| **C** | **candidate + dual (codex-sol)** | **7/9** | **12/18** |
| D | candidate, single, Sonnet | 1/9 | 4/18 |
| E | candidate + dual (codex-terra) | 2/9 | 7/18 |

Registered contrasts:
- **A vs B (criteria effect)**: wash — 1 improvement, 1 regression on must-catch
  (`trop-silent-drop`: old criteria caught it, new criteria missed — a real
  blind spot the rewrite introduced; tracked follow-up). The criteria rewrite
  alone does **not** improve catching.
- **B vs C (dual effect)**: 6 improvements, 0 regressions. **Dual review is the
  entire value** — the codex second reviewer, not the criteria.
- **codex-sol ≫ codex-terra**: C 7/9 vs E 2/9 (same criteria + Claude side).
- **Claude tier is not the lever**: Sonnet single (D) == Opus single (A/B) at
  1/9. Catching power comes from adding codex, not a bigger Claude.
- **Complementarity**: within C, codex caught the math/methodology defects;
  Claude (repo read access) uniquely caught a codebase-structure defect codex
  missed (`df-inheritance-trap`). Neither alone matches the union.
- **Ceiling**: 2 must-catch defects caught by no arm (deep numerical/parity
  claims needing execution, not reading).

## Precision — true-vs-false hallucination rate

Every negative-control finding (213) fact-checked against its base_sha and
classified **true** / **false** / **unverifiable** (a claim the fact-check could
neither confirm nor refute against the base — subjective/forward-looking wording
or a judgment call, counted against neither precision nor recall). Each row
reconciles as `findings = true + false + unverifiable`; false-rate = false /
findings:

| Arm | findings | true | false | unverif. | false-rate |
|-----|:---:|:---:|:---:|:---:|:---:|
| A | 52 | 36 | 0 | 16 | 0.0% |
| B | 25 | 24 | 0 | 1 | 0.0% |
| **C** | 58 | 44 | **2** | 12 | **3.4%** |
| D | 37 | 37 | 0 | 0 | 0.0% |
| E | 41 | 33 | 2 | 6 | 4.9% |
| **Σ** | **213** | **174** | **4** | **35** | **1.9%** |

Overall false rate **1.9%** (4/213). All 4 hallucinations came from the dual
arms; every single arm was 0%. But the dual arm's 33 findings beyond
single-candidate B were **31 true, 2 false** — the extra volume is
overwhelmingly real. 3 of the 4 false findings were one systematic
overstatement (mis-citing a CONTRIBUTING.md tutorial requirement as a mandate).

## Recommendation (exploratory — see Status)

On the sensitivity signal, ship **arm C as run: candidate criteria + dual review
(Claude @ Opus 4.8 + codex-sol @ xhigh) + merge/verify**. Dual is a ~7×
sensitivity gain over any single reviewer at a ~3.4% hallucination cost —
favorable, since a false finding costs a dismissal while a missed defect costs a
bad plan. This is a directional pick from a NON-GATING campaign, not a validated
verdict: the candidate criteria carry the `trop-silent-drop` regression and the
FP axis on a genuinely-clean plan is unmeasured (both tracked). A clean
re-validation is the path to a gating result.

## Methodology lesson — negative controls

The 3 `s3_negative` plans were selected because their *implementations* passed
CI code review — but a plan-reviewer reads the plan *text* against the repo,
where stale line-anchors and already-merged-PR references at the pinned
base_sha are **real** defects. Verified: one plan's cited `REGISTRY.md:3506`
is actually at `:3594` at its base; another's base commit *is* its own
"PR-1." So the negatives were not clean, and the severity-thresholded FP gate
(any blocker on a "clean" plan = false positive) mislabeled true findings as
FPs. Precision was therefore re-measured as **hallucination rate** (true vs
false), which is the safety-critical axis and cleared. What remains
**unmeasured**: trivia-flooding on a *genuinely* clean plan — the corpus has
none. Closing that needs a small future run on constructed-clean negatives
(tracked follow-up), not a repeat of the 120-review matrix.
