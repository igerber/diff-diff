### Internal
- **Codex reviewer swap gpt-5.6-sol -> gpt-6.1-sol (reviewer-eval 2026-10 campaigns, PR-2)**:
  two blinded, pre-registered 44-review campaigns on the frozen `tools/reviewer-eval`
  harness. Campaign 1 (B = gpt-5.6-sol vs E = gpt-6-astra, both @ xhigh, k=2): mechanical
  DEFINITIVE **NO-GO** on one regression (`s4-cs-unbalanced-nan-outcome`), accepted by the
  user; no astra swap (`campaigns/2026-10-gpt6-VERDICT.md`). Campaign 2 (B vs F =
  gpt-6.1-sol @ xhigh, k=2, fresh control sample, same rule re-issued for F):
  mechanical DEFINITIVE **PARITY** with zero rule-defined regressions, zero downgrades,
  0 vs 0 false positives, two favourable catch-status changes against one
  UNSTABLE-to-MISSED judgment flag, ~15% lower latency and half the input price; the
  user accepted PARITY as GO
  (`campaigns/2026-10-gpt61sol-VERDICT.md`). Both verdicts ship with their redacted
  grading records (`campaigns/2026-10-gpt6*-grading/`). Production pins move in lockstep
  to `gpt-6.1-sol`: the CI workflow `model`, `openai_review.DEFAULT_MODEL` and the
  plan-review `CODEX_MODEL` (carried over from the code-review campaign, not re-run
  through plan-review-eval; TODO.md row), with the plan-review `CODEX_TIMEOUT_S` raised
  2400 -> 3600 to equal the harness ceiling. Harness: arm `F` registered with the contrast
  `(B, F)`, campaign 1's rule archived as `campaigns/2026-10-gpt6-DECISION_RULE.md`,
  `run --configs` defaults to the last registered shape, `PRICING` gains `gpt-6.1-sol`,
  arm F takes `role=control` (B retained as previous production), the candidate-exposure
  gate exempts the control arm so bare `smoke` keeps working after a swap, and the
  DEFERRED `gpt-6-sol` cost-fallback row is retired (codex-cli 0.162.1's ChatGPT-account
  login accepts the gpt-6 Sol tier; campaign 2 was that evaluation).
