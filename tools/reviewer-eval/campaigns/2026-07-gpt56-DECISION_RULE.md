# Pre-Registered Decision Rule: gpt-5.5 -> gpt-5.6 Reviewer Evaluation

Committed BEFORE any campaign run (the PR that adds this file timestamps the
pre-registration). The verdict is computed MECHANICALLY from the graded tables
against these rules; the rules may not be edited after the campaign starts.

## The experiment

| Arm | Model | Effort | Repeats | Role |
|-----|-------|--------|---------|------|
| A | gpt-5.5 | xhigh | k=2 | control (current production) |
| B | gpt-5.6-sol | xhigh | k=2 | primary candidate |
| C | gpt-5.6-terra | xhigh | k=1 | cost-fallback probe (informational) |
| D | gpt-5.6-sol | max | k=1 | effort probe (informational) |

The production decision is A vs B. C and D never gate the swap.

## Definitions

- **Reliably caught** (per arm, per ground-truth bug): named the same defect at
  the same location/symbol in EVERY repeat of that arm.
- **Unstably caught**: caught in some but not all repeats.
- **Missed**: caught in no repeat.
- **False positive (FP)**: on an `expect_no_blockers` case, any finding whose
  severity is ABOVE that case's `allow_severities` (default allows P2/P3, i.e.
  P0/P1 are FPs; the documented-deviation case allows only P3, so P0-P2 count).
  Known-FP topics listed by the case are never counted.
- **Calibration failure**: flagging a REGISTRY-documented deviation above P3.
  Counted as an FP.

## Primary gates (B vs A)

1. **Regression (NO-GO):** any `must_catch` bug that A reliably catches and B
   misses in both B-repeats. (A unstably-caught vs B missed -> flagged for
   judgment in the report, not an automatic NO-GO.)
   `must_catch` gates regressions ONLY.
2. **FP gate (NO-GO):** B's total FP count across S3 negative controls exceeds
   A's.
3. **GO:** zero regressions AND the FP gate passes AND at least one strict
   improvement:
   - B reliably catches ANY ground-truth bug (`must_catch` true OR false —
     aspirational S4 catches count, and S4 is the highest-value evidence) that A
     misses in both A-repeats, OR
   - B has strictly fewer FPs than A (with no catch regression).
4. **PARITY:** zero regressions but no strict improvement -> the user decides
   (vendor-cadence value vs staying on gpt-5.5).

## Secondary reads (informational only)

- **C vs A:** does Terra actually match gpt-5.5 on our corpus (the half-cost
  claim)? Relevant only if B fails its gates.
- **D vs B:** does `max` effort add reliable catches over xhigh on the same
  model? Informs an optional follow-up; the default ship remains Sol @ xhigh.
- Latency per arm (from the unblinded artifacts) is reported alongside, since a
  much slower CI review has real cost even at equal accuracy.

## Grading protocol (blinded, multi-grader)

1. `compare --blinded` produces `comparison.blinded.md` (arm identities replaced
   by neutral M* labels; model self-references scrubbed; latency redacted) and a
   sealed `blinding.json` that graders NEVER read.
2. 2-3 INDEPENDENT graders (separate subagent contexts) each read ONLY the
   blinded bundle plus this rubric, and fill, per (case, ground-truth bug, arm
   label): caught / partial / missed, with a verbatim evidence quote from the
   review; plus per-case FP lists with severities. "Partial" (right file or
   class, wrong defect or location) counts as MISSED for the gates.
3. **Hallucination check:** a catch's evidence quote must actually name the
   defect; a finding that claims to have reproduced behavior the diff cannot
   produce is graded as an FP-class note, never a catch.
4. **Adversarial reconciliation:** any cell where graders disagree goes to a
   verifier agent that re-reads that case's blinded reviews and rules with
   quoted evidence.
5. Only after the tables are final are labels unblinded via `blinding.json` and
   the gates above applied mechanically.

## Blinding caveats (accepted)

- Repeat counts partition the labels into {k=2}={A,B} and {k=1}={C,D}; the
  decisive A-vs-B contrast stays 50/50 blind within its group.
- Sanitization is best-effort; graders are instructed to grade on content and
  never on guessed identity.

## Corpus floor

The campaign runs only with >= 8 verified cases including >= 2 S4
(missed-by-gpt-5.5) cases; below that, stop and surface rather than produce a
weak read.
