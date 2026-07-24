# Plan-Review Engine Comparison Harness

Measures proposed changes to the plan-review workflow's ENGINE (criteria,
reviewer composition, models) against the current production workflow, BEFORE
they land — the same measure-first discipline `tools/reviewer-eval` applied to
the CI code reviewer (see its 2026-07 gpt-5.6 campaign). Second consumer of
the shared `tools/eval_core/` engine.

**Status:** Campaign 1 has run — **exploratory / NON-GATING** (its
pre-registered gates did not apply: the s3 negative controls were
base_sha-contaminated and the candidate criteria carried a must-catch
regression). It selected the dual engine on a strong sensitivity signal (7/9 vs
1/9); a clean re-validation is tracked. See `verdicts/campaign-1.md`. The
go/no-go rule is pre-registered in `DECISION_RULE.md` and was not edited once
the campaign started. Local-only, never wired to CI.

## Layout

```
tools/eval_core/           # shared engine: models, store, runner, compare (+ blinding)
tools/plan-review-eval/
├── run_eval.py            # CLI: verify-corpus · smoke · run · extract · compare · verdict
├── verdict.py             # mechanical gate computation (unit-tested; no LLM)
├── plan_adapters/         # plan bindings: corpus_loader, criteria_source, plan_reviewer, worktree
├── candidates/            # the engine UNDER TEST (criteria, prompts, merge+verify, extraction)
├── config/configs.json    # the five arms + control-criteria pin + extraction pin
├── DECISION_RULE.md       # pre-registered gates + grading protocol + corpus floor
└── corpus/
    ├── manifest.schema.json
    ├── fixture/           # committed fabricated case (CI tests, smoke, dress rehearsal)
    └── cases/             # REAL cases — gitignored; the user's plans are never committed
```

## The arms (all k=2)

A control (pinned-SHA `review-plan.md`, single Claude) · B candidate criteria,
single Claude · C candidate, dual Claude+codex-sol with merge+verify · D probe
(Sonnet) · E probe (codex-terra). Two gating contrasts: A-vs-B (regression)
and B-vs-C (is dual worth it). See `DECISION_RULE.md`.

The control arm's criteria come from `git show <pinned-sha>` (pin + rationale
in `config/configs.json`) — never a committed copy, so the control cannot
drift and survives the command file's eventual retirement. The candidate
engine lives in `candidates/` as lab artifacts; the program's step 3 promotes
the winning configuration into a live skill.

## Usage

```bash
# 1. Verify the corpus materializes (no reviewer calls; fast)
python tools/plan-review-eval/run_eval.py verify-corpus

# 2. Smoke: fixture case, one arm, k=1. Bare `smoke` runs the control arm
#    (proves the pinned git-show path + the claude -p invocation); smoke the
#    dual arm too before any campaign — it proves the codex + merge path.
python tools/plan-review-eval/run_eval.py smoke
python tools/plan-review-eval/run_eval.py smoke --configs C

# 3. Dress rehearsal (campaign-readiness gate; see DECISION_RULE.md):
python tools/plan-review-eval/run_eval.py run --subdir rehearsal --cases fx-mini-plan --k 2
python tools/plan-review-eval/run_eval.py extract --subdir rehearsal
python tools/plan-review-eval/run_eval.py compare --subdir rehearsal --blinded
#    ... two-grader mini pass over the blinded bundle -> grades.json ...
python tools/plan-review-eval/run_eval.py verdict --subdir rehearsal --grades grades.json --candidate B

# 4. The campaign (only after the readiness gate passes)
python tools/plan-review-eval/run_eval.py run --subdir campaign --k 2
python tools/plan-review-eval/run_eval.py extract --subdir campaign
python tools/plan-review-eval/run_eval.py compare --subdir campaign --blinded
python tools/plan-review-eval/run_eval.py verdict --subdir campaign --grades <reconciled>.json --candidate B
python tools/plan-review-eval/run_eval.py verdict --subdir campaign --grades <reconciled>.json --candidate C --control B
# (A-vs-B and B-vs-C are the ONLY registered gating contrasts; any other pair —
#  including the D/E probes — is labeled NON-GATING by verdict.)
```

## How scoring works

Runs store each arm's raw (or, for dual arms, merged) review verbatim. The
`extract` stage reduces every review to a uniform, format-neutral findings
schema (defect claim + location + neutral severity + verbatim evidence quote)
— this is what closes the report-structure blinding leak: the two engines'
native formats (CRITICAL/MEDIUM/LOW prose vs P0–P3 lists, dual-arm agreement
tags) would otherwise reveal exactly the contrast being judged. `compare
--blinded` bundles the extractions under neutral `M*` labels; two independent
subagent graders fill the caught/partial/missed + FP tables; disagreements go
to adversarial reconciliation (raw reviews consultable ONLY on dispute);
`verdict` applies the pre-registered gates mechanically.

## Data handling

"Never committed" is a GIT statement, not a privacy boundary — know where plan
content actually goes:

- Reviewer/merge/extraction stages TRANSMIT plan content to model providers
  (Anthropic via the Claude CLI; OpenAI via codex for dual arms).
- Raw reviews, merged reports, extractions, and grading bundles are stored
  under `runs/` (gitignored, local).
- The codex CLI's `--sandbox read-only` prevents writes but is NOT a read
  boundary confined to the worktree (see the tracked isolation item in
  `TODO.md`); the Claude reviewer runs under the default permission model,
  which denies out-of-workspace reads headlessly.
- Sanitize a case's plan text before adding it to the corpus if it contains
  anything you would not paste into a model prompt.

## Fidelity notes (documented divergences)

- The control arm inlines the pinned criteria into its spawn prompt (the
  production flow had the agent read the live file; the pinned file may not
  exist in the case worktree).
- Production dual-mode verification runs in the planning session with
  conversation context; the harness's merge+verify subprocess has none.
- Reviewer subprocesses run with read-only built-in tools (`Read,Grep,Glob`)
  in a detached worktree at the case's `base_sha` — the repo as the plan saw
  it, so codebase-correctness findings grade against the right tree.

## Provenance guarantees and limits (documented scope)

What the campaign provenance machinery **guarantees** (accident-grade, per the
threat model recorded in `DEFERRED.md` — the gate prevents accidents, not a
hostile local actor with write access):

- The recorded protocol identity covers the decision rule, configs, every
  candidate artifact, the control spawn prompt, every Python source in this
  tree and `eval_core/` (walked, never hand-listed), and the external codex
  wrapper (`.claude/scripts/openai_review.py`).
- A read→import→re-read stability bracket proves the code Python executes —
  including the dynamically loaded wrapper, which is handed to the reviewer
  from inside the bracket — is byte-identical to the hashed sources; any
  change across the bracket aborts the snapshot.
- Registration is write-once and precedes observation: a subdirectory IS a
  campaign, its manifest (protocol identity + full sample-plan fingerprint:
  arms, k, overrides, case ids, base SHAs, frozen plan bytes) reaches disk
  before any reviewer call, and an invocation differing in either refuses to
  touch it. Post-run stages gate at entry against the recorded identity from
  one in-memory snapshot, re-check with a fresh read at exit, and stamp the
  protocol sha into extraction metadata and `blinding.json`; the blind
  mapping is recomputed at verdict, and the bundle id is the hash of the
  exact grader-visible bytes.

Documented **limits** (out of scope by decision, not oversight):

- Byte-level TOCTOU below the bracket's two reads (e.g. an OS-level file swap
  landing between them) is detected-and-aborted when observable, but cannot
  be excluded without executing from an immutable copied tree; the recorded
  accident threat model does not require that.
- A local actor who can write to the repo can defeat any self-hosted
  provenance check (the `DEFERRED.md` decision record).
- Smoke runs re-register the fixture subdir freely: they are liveness checks,
  never graded, and the verdict stage independently refuses any manifest
  containing fixture data.
