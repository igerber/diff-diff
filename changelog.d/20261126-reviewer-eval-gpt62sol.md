### Changed
- **AI review surfaces move to gpt-6.2-sol on codex-cli 0.172.0**: the CI PR
  reviewer (`.github/workflows/ai_pr_review.yml`), the `/ai-review-local` default
  model for both of its backends (`openai_review.DEFAULT_MODEL`) and the
  plan-review dual engine's codex half now use gpt-6.2-sol. The `xhigh` reasoning
  effort is pinned on the codex invocations (CI, the local codex backend,
  plan review); the local Responses API backend sets no effort and is not the
  validated path. This follows reviewer-eval campaign 3, which returned a mechanical
  GO against gpt-6.1-sol (6 vs 5 reliably caught bugs of 10, no regression, no
  false positive on the negative controls, one judgment flag;
  `tools/reviewer-eval/campaigns/2026-11-gpt62sol-VERDICT.md`). gpt-6.2-sol is not
  reachable on codex-cli 0.162.1, so the CI `codex-version` pin and the harness
  pin move to 0.172.0 together; a local codex CLI older than that cannot run the
  new default (the local script and the plan-review helper now say so when the
  model is rejected). The plan-review surface carries the model over without a
  plan-review campaign, as in 2026-10.
- **`/ai-review-local` no longer tells the codex reviewer it has no shell
  access**: the local prompt adaptation replaced the criteria's `grep` directive
  with "no shell access in this path" for BOTH backends, although only the api
  backend lacks a shell (`openai_review._API_ONLY_SUBSTITUTIONS` now scopes it).
  gpt-6.1-sol ignored the sentence; gpt-6.2-sol obeyed it, running no shell
  command in 3 of 3 off-corpus probes and reading the published repository
  instead, which cannot contain unpushed changes. With the fix it ran 24 to 34
  shell commands per review. A model rejection by an older codex CLI now ends
  with a hint naming the minimum CLI version (0.172.0) and `--model`.

### Internal
- **Codex reviewer campaign 3 (gpt-6.1-sol -> gpt-6.2-sol)**: `tools/reviewer-eval`
  registers arm `G` = gpt-6.2-sol @ xhigh and the contrast `(F, G)`
  (`reviewer_verdict.REGISTERED_CONTRASTS`) under the SAME pre-registered rule as
  campaigns 1 and 2 (`DECISION_RULE.md` re-issued; campaign 2's rule archived as
  `campaigns/2026-10-gpt61sol-DECISION_RULE.md`), with the gpt-6.1-sol control
  re-sampled on the new CLI and the campaign-2 grades reported next to it
  descriptively only. The rule anchors the freeze to the local campaign commit
  (tag plus ledger ref) rather than a registration PR. The harness's isolated
  codex home sets `[features] apps = false` for every arm: under the
  ChatGPT-account login gpt-6.2-sol called the account's GitHub connector in
  pre-registration probes, a channel the API-key-authenticated CI reviewer is
  taken not to have. After the swap `G` carries `role=control`, and the
  candidate-exposure gate also exempts any arm that was the control of a
  registered contrast, so the previous production arm stays runnable as a
  baseline. `reviewer_verdict.COMPLETED_CAMPAIGNS` records each finished
  contrast's first-dispatch revision in tracked code, and `verdict` reports a re-run
  whose local ledger starts anywhere else as non-mechanical (the ledger and the
  frozen ref exist only in the checkout that ran the campaign, so a fresh
  checkout could otherwise re-register a finished contrast). `run --configs` defaults to `F,G`; `openai_review.PRICING` gains
  `gpt-6.2-sol`; the session-parser fixtures gain a codex-cli 0.172.0 transcript.
