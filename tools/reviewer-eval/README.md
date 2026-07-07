# Codex Reviewer A/B Comparison Harness

A small local tool for deciding whether to upgrade the Codex PR reviewer (e.g.
`gpt-5.4 → gpt-5.5`) **before** it goes live. It runs the current and candidate
models over a corpus of real diff-diff review cases, saves each model's raw
review, and emits a side-by-side bundle you (or an LLM) read into a
caught / missed / false-positive table.

**Status:** minimal harness; 2 seed cases. Local-only — not wired to CI. The
real go/no-go needs the corpus grown to ~10 cases first (see "Next").

## Why this exists

The last reviewer update regressed by *missing real issues*, and it changed the
model **and** the prompt at once — so the cause couldn't be isolated. This harness
keeps upgrades empirical: change **one** variable (the model), reproduce the CI
invocation faithfully, and compare both arms on the same cases.

## Layout

```
tools/reviewer-eval/
├── run_eval.py          # CLI: verify-corpus · smoke · run · compare
├── engine/              # generic glue: models, store, runner, compare
├── adapters/            # diff-diff bindings: ci_prompt, codex_reviewer, corpus_loader, worktree
├── config/configs.json  # the two arms (A = control, B = candidate)
└── corpus/              # cases/{s1_synthetic, s3_negative, ...} + schema + synonyms
```

## Usage

```bash
# 1. Verify the corpus materializes (no codex; fast)
python tools/reviewer-eval/run_eval.py verify-corpus

# 2. Smoke test (1 case, control arm, first real codex call)
python tools/reviewer-eval/run_eval.py smoke --configs A

# 3. Full A/B run (both arms, all cases) — saves each arm's raw review
python tools/reviewer-eval/run_eval.py run --configs A,B

# 4. Emit the side-by-side bundle to grade
python tools/reviewer-eval/run_eval.py compare --subdir full
```

Run artifacts and the bundle land under `runs/` (gitignored).

## How scoring works

Each model produces a raw markdown review. `compare` collates, per case, the
ground-truth bugs followed by both arms' **raw** reviews (plus a grading
instruction pointing at `.github/codex/prompts/pr_review.md` for the severity
rubric both models were given). An LLM — a subagent, or you in-conversation —
reads that bundle top-to-bottom and fills the caught / missed / FP table. No
regex parsing of review prose: free-form, model-specific output (gpt-5.4 vs
gpt-5.5 format differently) is read directly.

## What faithful reproduction means

The candidate is measured the way CI will run it: `adapters/ci_prompt.py`
rebuilds the CI prompt (the **current** `pr_review.md` — the prompt under
validation, identical for both arms; deliberately NOT base-sourced, see
`adapters/ci_prompt.py` — + `git diff --name-status` +
`--unified=5` with the same pathspec exclusions; REGISTRY is **not** inlined —
Codex reads it from the worktree), `adapters/worktree.py` materializes each case
in a detached worktree, and `adapters/codex_reviewer.py` reuses the production
`openai_review.call_codex` with byte-identical flags. The runner asserts the
Codex CLI version is identical across arms, so the model is the only variable.

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
- One `run` invocation = one experiment (run `--configs A,B` together). `compare`
  reads the per-run manifest (`runs/<subdir>-manifest.json`), so rerunning into the
  same `--subdir` with a changed model **replaces** the comparison rather than
  mixing the old and new runs.

## Next

Grow the corpus from 2 seed cases to ~10 real cases (mine bugs, pin SHAs, freeze
`inject.diff` patches), run both arms, read the bundle, and decide. That curation
plus the live A/B run is deliberately a separate step from this harness.
