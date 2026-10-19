### Internal
- **CI AI review fits oversized diffs into the codex input cap.** The compiled review
  prompt of PR #861 reached 1.72M characters against codex's hard 1,048,576-character
  limit (`input_too_large`), so the review never ran; the workflow capped notebook prose
  but never the unified diff. `.github/workflows/ai_pr_review.yml` now appends the diff
  last, under a whole-prompt byte budget (1,000,000) derived from everything else already
  in the prompt, through the new `tools/ai_review_diff_budget.py` (staged from `BASE_SHA`
  like the notebook extractor; untrimmed one-shot fallback when absent on base). An
  in-budget patch passes through byte-for-byte; an oversized one has whole per-file blocks
  replaced by one-line stubs - `tests/` / `benchmarks/` / `tools/` first, `diff_diff/` /
  `rust/` only when unavoidable, largest first - each naming the `git diff` command the
  reviewer can run for that file. Pinned by `tests/test_ai_review_diff_budget.py`; the
  staging/ordering contract joins `TestWorkflowDoesNotExecutePRHeadCode`.
