# GPT-5.6 Reviewer Evaluation - Verdict (2026-07-18)

Campaign: 66/66 runs ok, 11 cases, arms A=gpt-5.5@xhigh k2 (control), B=gpt-5.6-sol@xhigh
k2, C=gpt-5.6-terra@xhigh k1, D=gpt-5.6-sol@max k1. Blinded grading: 3 independent
graders on comparison.blinded.md (labels M1-M4; graders never saw blinding.json);
near-perfect inter-grader agreement (single gate-irrelevant cell disagreement:
s2-acrt/C/r0 graded missed-missed-partial; partial counts as missed). Unblinded only
after tables were final. Mapping: A=M4, B=M3, C=M2, D=M1.

## Catch table (10 ground-truth bugs; "reliable" = caught in ALL of that arm's repeats)

| Bug (stratum) | A 5.5@xh r0/r1 | B sol@xh r0/r1 | C terra r0 | D sol@max r0 |
|---|---|---|---|---|
| s1-coef-dict-collision (S1) | C / C | C / C | C | C |
| s1-cs-dr-plugin-se (S1) | C / C | C / C | C | C |
| s2-acrt-single-dose (S2, hist. caught) | M / C | C / C | M | C |
| s2-cic-practitioner-screen (S2) | M / M | C / C | C | C |
| s2-had-cluster-guard (S2) | P / M | C / C | M | C |
| s2-nan-dof-fail-closed (S2) | M / M | C / P | P | C |
| s4-cs-unbalanced-nan-outcome (S4) | M / M | C / C | M | C |
| s4-lpdid-survey-unreduced-design (S4) | M / C | C / C | P | P |
| s4-wcr-saturated-guard-order:b1 (S4) | M / M | M / M | M | M |
| s4-wcr-saturated-guard-order:b2 (S4) | M / M | M / M | M | M |
| **Reliable catches** | **2/10** (+2 unstable) | **7/10** (+1 unstable) | **3/10** | **7/10** |

C=caught, P=partial (counts as missed for gates), M=missed.

## Pre-registered gates (B vs A)

1. **Regression gate: PASS.** A reliably catches only the two S1 reverts; B catches both
   2/2. Zero must_catch bugs where A reliable + B missed-both. (A could not even
   reproduce its own historical S2 catches under the current prompt: s2-had/s2-nan-dof/
   s2-cic missed in both repeats, s2-acrt in one.)
2. **FP gate: MECHANICAL FAIL, INVALID CONTROL.** B posted 1 "FP" (P2, s3-wcr-pfloor
   r0) vs A's 0. Post-verdict code inspection confirms the flagged finding is FACTUALLY
   CORRECT: the fixture's reworded comment claims "an exact zero is never reported"
   while the floor>=alpha branch returns raw_p, which can be exactly 0 - a real
   inaccuracy introduced during case authoring. The case is invalid as a clean negative
   control (it contains a genuine defect above its allowed severity). Excluding the
   invalid control: B FPs = 0 = A FPs -> gate PASSES. On the remaining valid control
   (s3-changelog-prose) all four arms were clean. Both readings reported; the
   pre-registered rule text was not edited.
3. **Improvement gate: PASS (x3 strict).** B reliably catches, with A missing BOTH
   repeats: s2-cic, s2-had, and s4-cs-unbalanced (an S4 missed-bug probe - the
   highest-weighted evidence class). Further B advantages on unstable-A cases:
   s2-acrt, s4-lpdid (A caught only 1/2).

**Verdict: GO** (conditional only on accepting the invalid-control analysis in gate 2;
mechanical-literal reading is NO-GO on a control that demonstrably contains a real bug).

## Secondary reads (informational, non-gating)

- **Terra vs 5.5:** 3/10 vs 2/10 (+2 unstable) - "competitive with 5.5" roughly holds;
  no reason to prefer it for review quality.
- **Sol@max vs Sol@xhigh:** 7/10 vs 7/10; max caught s2-nan-dof where xhigh was
  unstable, but missed s4-lpdid (partial) which xhigh caught 2/2. No decisive gain at
  ~1.5-2x latency (campaign p95 ~1060s vs ~590s). Default ship stays Sol@xhigh.
- **Latency (unblinded artifacts):** A ~220-420s, B ~480-900s, C ~115-445s,
  D ~335-1060s. Sol@xhigh is ~1.5-2x slower than 5.5 per review.
- **Repeat stability:** B unstable on 1/8 catches; A unstable on 2/4 - B is more stable.
- **Shared blind spot:** both s4-wcr guard-order bugs missed by ALL arms (keep as
  future-corpus anchors). On that case B and D instead asserted an unverified
  "CI-inversion P0" with claimed R reproductions.
- **Hallucination notes:** A: 0; B: 1 (asserted R `did` execution, s4-cs r0); C: 1
  (fabricated hand-calc fixture numbers); D: 1 (claimed fwildclusterboot run).
  Sol arms make confident external-tool claims - same class as known codex behavior;
  grading discounted them (claims checked against ground truth).

## Grader notes (blinded, consensus)

M3 (=B) "most sensitive and most consistent, precise locations, quantified
reproductions"; M4 (=A) "least sensitive, repeat-unstable, cleanest on negative
controls"; M2 (=C) "uneven, drifts to adjacent blockers"; M1 (=D) "nearly as strong as
M3, verbose".

## Provenance (committed audit record)

This file is the sanitized, committed record of the campaign verdict (raw model
outputs stay local under the gitignored `runs/`). Local artifact integrity hashes
(sha256, first 16 hex) at verdict time:

- `runs/gpt56/comparison.md`: `7e036c4be42d21f7`
- `runs/gpt56/comparison.blinded.md`: `6295ad0c91887d11`
- `runs/gpt56/blinding.json`: `9b075197bd4f32ab`
- Manifest: 66 run_ids, configs A,B,C,D, k=2, k_per {C:1, D:1}

Post-verdict actions recorded here for completeness: the user accepted the GO
(including the invalid-control analysis of gate 2) on 2026-07-18; the swap shipped
with the control-arm flip in `config/configs.json`; the s3-wcr-pfloor fixture
comment overclaim that produced the contested "FP" was corrected in the same PR
(see the case.json notes), so the control is valid for future campaigns.
