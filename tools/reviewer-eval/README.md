# Codex Reviewer Comparison Harness

A small local tool for deciding whether to upgrade the Codex PR reviewer (e.g.
`gpt-5.5 → gpt-5.6-sol`) **before** it goes live. It runs the control and
candidate arms over a corpus of real diff-diff review cases, saves each arm's
raw review, and emits a side-by-side bundle you (or an LLM) read into a
caught / missed / false-positive table — optionally **blinded** (arm identities
stripped) so graders can't favor a model.

**Status:** harness supports N-arm matrices (current config: the 4-arm gpt-5.6
evaluation), per-arm repeat counts, and blinded grading. Local-only — not wired
to CI. The go/no-go decision rule is pre-registered in `DECISION_RULE.md`.

## Why this exists

An early reviewer update regressed by *missing real issues*, and it changed the
model **and** the prompt at once — so the cause couldn't be isolated. This harness
keeps upgrades empirical: vary **only** the declared treatment fields, reproduce
the CI invocation faithfully, and compare every arm on the same cases.

## Layout

```
tools/reviewer-eval/
├── run_eval.py          # CLI: verify-corpus · smoke · run · compare
├── engine/              # generic glue: models, store, runner, compare (+ blinding)
├── adapters/            # diff-diff bindings: ci_prompt, codex_reviewer, corpus_loader, worktree
├── config/configs.json  # the arms (one role=control) + declared treatment_fields
├── DECISION_RULE.md     # pre-registered GO/NO-GO rule + grading rubric
└── corpus/              # cases/{s1_synthetic, s2_historical, s3_negative, s4_missed} + schema + synonyms
```

## Usage

```bash
# 1. Verify the corpus materializes (no codex; fast)
python tools/reviewer-eval/run_eval.py verify-corpus

# 2. Smoke test (1 case, control arm, first real codex call); smoke each
#    candidate arm too — `smoke --configs D` live-proves the max effort level
python tools/reviewer-eval/run_eval.py smoke --configs A

# 3. Full matrix — k=2 repeats on the primary A/B, single-shot probe arms
python tools/reviewer-eval/run_eval.py run --subdir gpt56 --configs A,B,C,D --k 2 --k-per C=1,D=1

# 4. Emit the side-by-side bundle to grade (+ the identity-stripped one)
python tools/reviewer-eval/run_eval.py compare --subdir gpt56 --blinded
```

Run artifacts and the bundles land under `runs/` (gitignored). `--blinded` also
writes `comparison.blinded.md` plus a sealed `blinding.json` (the label→arm
mapping) — graders read ONLY the blinded bundle; unblind their finished tables
via `blinding.json`.

## How scoring works

Each arm produces a raw markdown review. `compare` collates, per case, the
ground-truth bugs followed by every arm's **raw** review (plus a grading
instruction pointing at `.github/codex/prompts/pr_review.md` for the severity
rubric all arms were given). An LLM — a subagent, or you in-conversation —
reads that bundle top-to-bottom and fills the caught / missed / FP table. No
regex parsing of review prose: free-form, model-specific output (models format
findings differently) is read directly. For the blind protocol (independent
graders, adversarial reconciliation, the pre-registered decision rule), see
`DECISION_RULE.md`.

## What faithful reproduction means

The candidate is measured the way CI will run it: `adapters/ci_prompt.py`
rebuilds the CI prompt (the **current** `pr_review.md` — the prompt under
validation, identical for all arms; deliberately NOT base-sourced, see
`adapters/ci_prompt.py` — + `git diff --name-status` +
`--unified=5` with the same pathspec exclusions; REGISTRY is **not** inlined —
Codex reads it from the worktree), `adapters/worktree.py` materializes each case
in a detached worktree, and `adapters/codex_reviewer.py` reuses the production
`openai_review.call_codex` with byte-identical flags. The runner asserts the
Codex CLI version is identical across arms and that the arms differ only in the
declared `treatment_fields` (model, and for the gpt-5.6 matrix also effort),
each arm in a clean single-field contrast.

## Corpus strata

- **S1 synthetic** — injected bugs (revert a real fix). Sanity floor.
- **S2 historical** — real PR states just before a confirmed bug was caught.
- **S3 negative** — clean PRs (precision controls; any P0/P1 = false positive).
- **S4 missed** — real bugs the AI reviewer *missed* (the failure-mode probe).

## Known limitations

- **Tutorial-notebook cases are supported with CI-equivalent context.** CI
  special-cases only `docs/tutorials/*.ipynb`: it excludes them from the diff *and*
  appends a sanitized `<notebook-prose>` block extracted via
  `tools/notebook_md_extract.py`. The harness reproduces both (same per-output /
  per-notebook / aggregate caps, fail-soft extraction, truncation marker,
  zero-extracted fallback, close-tag sanitization — see
  `ci_prompt.build_notebook_prose_block`). Documented divergence: the extractor is
  sourced from the current repo rather than each case's base SHA (same rationale as
  `pr_review.md` sourcing). Non-tutorial `.ipynb` ride the normal diff path, exactly
  as CI handles them.
- One `run` invocation = one experiment (run all arms together; `--k-per` covers
  per-arm repeat counts). `compare` reads the per-run manifest
  (`runs/<subdir>-manifest.json`), so rerunning into the same `--subdir` with a
  changed model **replaces** the comparison rather than mixing the old and new runs.
- Blinding is best-effort: repeat counts still partition arms into {k=2} vs {k=1}
  groups, and the sanitizer can't catch every conceivable self-reference. Graders
  are instructed to grade on content, never on guessed identity (see
  `DECISION_RULE.md`).
