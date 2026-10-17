# GPT-6.1 Sol Reviewer Evaluation - Verdict (2026-10-17): PARITY, accepted as GO

Pre-registered rule: `DECISION_RULE.md` (the campaign-1 rule re-issued for arm F;
registered at local commit `e1674c0e` on branch `codex-sol61`, the commit this PR
squashes in; the local tag `campaign-gpt61sol` keeps that sha resolvable). Arms: B =
gpt-5.6-sol @ xhigh k=2 (control, re-sampled; production at the time), F =
gpt-6.1-sol @ xhigh k=2 (candidate; its first corpus run was this campaign). 44/44
runs OK in one attempt (37 minutes at `--max-parallel 4`), 11 cases, zero INFRA_ERROR.
Per the user's direction the campaign ran before any PR was opened; the ledger and
frozen ref record the exposure revision exactly as for a PR-frozen HEAD.

**Mechanical verdict (`run_eval.py verdict`, DEFINITIVE with `--audit-hits`): `PARITY (DEFINITIVE)`.**
Regressions: 0. FP: F 0 vs B 0. Downgrades: 0. Strict improvements: 0
(F is UNSTABLE, not RELIABLE, on `s2-nan-dof-fail-closed:b1` where B is MISSED; F is RELIABLE on
`s4-lpdid-survey-unreduced-design:b1` where B is UNSTABLE, not MISSED). Judgment flag:
`s2-acrt-single-dose:b1` B UNSTABLE / F MISSED. Contested gates: none. Precedence reaches step (7).

**User's final call (2026-10-17): accept PARITY as GO.** Swap production (CI reviewer,
local `/ai-review-local` default, plan-review dual engine, timeout T = 3600) to
gpt-6.1-sol. Reason recorded: no rule-defined regression against two independent
control samples; two favourable catch-status changes (`s4-lpdid-survey-unreduced-design:b1`
B UNSTABLE -> F RELIABLE, `s2-nan-dof-fail-closed:b1` B MISSED -> F UNSTABLE) against one
unfavourable one that the rule reports as a judgment flag (`s2-acrt-single-dose:b1` B
UNSTABLE -> F MISSED); 11 vs 10 caught repeat-cells; half the input price; lower latency;
and gpt-6.1-sol is the current-generation Sol while 5.6-sol is labelled older-generation
by the vendor. (The local pre-PR review of this PR caught an earlier wording of this
paragraph that overstated the catch comparison as "never worse on any bug".)

## Catch table (10 ground-truth bugs; C = caught, P = partial, M = missed; partial counts as missed for every gate)

| Bug (stratum) | B r0 / r1 | F r0 / r1 | B status | F status |
|---|---|---|---|---|
| s1-coef-dict-collision:b1 (S1) | C / C | C / C | reliable | reliable |
| s1-cs-dr-plugin-se:b1 (S1) | C / C | C / C | reliable | reliable |
| s2-acrt-single-dose:b1 (S2) | P / C | M / M | unstable | missed |
| s2-cic-practitioner-screen:b1 (S2) | P / P | P / P | missed | missed |
| s2-had-cluster-guard:b1 (S2) | C / C | C / C | reliable | reliable |
| s2-nan-dof-fail-closed:b1 (S2) | P / P | C / P | missed | unstable |
| s4-cs-unbalanced-nan-outcome:b1 (S4) | C / C | C / C | reliable | reliable |
| s4-lpdid-survey-unreduced-design:b1 (S4) | C / P | C / C | unstable | reliable |
| s4-wcr-saturated-guard-order:b1 (S4) | M / M | M / M | missed | missed |
| s4-wcr-saturated-guard-order:b2 (S4) | M / M | M / M | missed | missed |
| **Reliable catches** | | | **4/10** | **5/10** |
| **Caught repeat-cells (of 20)** | | | **10** | **11** |

Mapping (unblinded after the tables were final): B = M2, F = M1.

## Pre-registered gates (F vs B)

1. **Regression NO-GO: PASS.** No bug is RELIABLE for B and MISSED for F.
2. **FP NO-GO: PASS.** 0 vs 0 on both negative controls.
3. **GO: not met.** Zero regressions and zero downgrades, but no strict improvement under the rule's definitions (see above).
4. **PARITY: the outcome.** Inputs for the user's decision are in the informational reads below; the user's call is recorded at the top.

## Astra side by side (descriptive only; no gate compares E and F)

Same corpus, same control model, independent control samples. The control's
per-bug status is identical across the two campaigns, which is what makes the
E/F columns comparable. R = reliable, U = unstable, M = missed.

| Bug | B sol (campaign 1) | E astra | B sol (campaign 2) | F 6.1-sol |
|---|---|---|---|---|
| s1-coef-dict-collision:b1 | R | R | R | R |
| s1-cs-dr-plugin-se:b1 | R | R | R | R |
| s2-acrt-single-dose:b1 | U | M | U | M |
| s2-cic-practitioner-screen:b1 | M | M | M | M |
| s2-had-cluster-guard:b1 | R | R | R | R |
| s2-nan-dof-fail-closed:b1 | M | R | M | U |
| s4-cs-unbalanced-nan-outcome:b1 | R | M | R | R |
| s4-lpdid-survey-unreduced-design:b1 | U | R | U | R |
| s4-wcr-saturated-guard-order:b1 | M | M | M | M |
| s4-wcr-saturated-guard-order:b2 | M | M | M | M |
| Reliable catches | 4 | 5 | 4 | 5 |
| Caught repeat-cells (of 20) | 10 | 10 | 10 | 11 |

F keeps the `s4-cs-unbalanced-nan-outcome` catch that astra lost, and trades
astra's reliable `s2-nan-dof-fail-closed` catch for an unstable one.

## Informational reads (non-gating)

| | B (gpt-5.6-sol) | F (gpt-6.1-sol) |
|---|---|---|
| Latency mean / p95 / max (s) | 213 / 414 / 472 | 181 / 277 / 416 |
| Candidate p95 > 2x control | | False |
| Input price (USD per 1M, standard tier) | 4.00 | 2.00 |
| Initial-prompt cost estimate, whole arm (USD, +/-50%) | 1.26 | 0.63 |
| Reviews with a non-shell event (web search) | 19 of 22 | 21 of 22 |
| Mean shell commands per review | 15.1 | 22.2 |
| Hallucination / unverified external-tool-claim cells | 22 | 22 |
| Confirmed leaks / unconfirmed flags | 0 / 0 | 0 / 0 |

The hallucination counter is uninformative on this corpus: every review in both
arms cites a test run or a numeric reproduction the blinded graders could not
verify, so each arm scores 22 of 22 cells. The 2026-07 campaign recorded this
as a known Sol-family trait; it now holds for every arm measured.

Vendor lifecycle (prose, operator-supplied): codex-cli 0.162.1's model catalog
labels gpt-5.6-sol "older generation workhorse"; gpt-6-astra is "frontier
intelligence for the most demanding work" and gpt-6.1-sol "performs near the
Astra level on complex work at lower cost" (developers.openai.com/api/docs/models,
read 2026-10-17).

## Session-log audit (`run_eval.py audit`, post-unblinding, informational)

Coverage: covered=4, no_commands=0, unaudited=40, total=44; vacuous=False.
Nearly every review web-searched (codex default features, decision 8), which
classifies the cell `unaudited`; the shell log is still scanned and reported.

- Cells with at least one hit: 42 of 44; cells whose hits are all orientation-only: 5. Hits per arm: {'B': 52, 'F': 63}.
- Most frequent markers: `repo_root` x36, `-maxdepth` x30, `GIT_LOG_RE` x29, `tools/reviewer-eval/` x14, `BARE_SHOW_RE` x11, `worktrees/` x10, `../` x5, `DOTDOT_RE` x5. The bulk are codex's own AGENTS.md discovery walk up the parent directories (`find -maxdepth`, `../..` loops, absolute paths under the enclosing checkout) and pytest invocations through the enclosing checkout's `.venv`; none of these reads the answer key.
- Hits on ref-reaching markers (`main`/`master`, `origin/`, `refs/`, tags, `rev-list`, `reflog`, `for-each-ref`, `cat-file`, `fsck`, `worktree list`): none.
- S1 cells whose shell log shows `git log` / `git show HEAD` (the synthetic HEAD's commit subject names the injected bug): s1-cs-dr-plugin-se/F/r0, s1-cs-dr-plugin-se/F/r1. Both arms caught both S1 bugs reliably in every repeat regardless, the verifier found no review citing the commit message as its source (confirmation criterion 3), and no gate is touched; recorded for transparency.
- The committed `hits.json` copy replaces each hit's `line` with its sha256 (command lines can carry file contents); the full lines stay in the gitignored original whose sha256 is listed below.

## Grading record

Three independent blinded grader subagents (inputs: `comparison.blinded.md`,
`GRADING_RUBRIC.md`, `.github/codex/prompts/pr_review.md`; Bundle ID
`cd55711a27f7080f`; blind labels M2=B, M1=F, mapping unknown to
them). Isolation is by instruction plus check, not a sandbox: the grader prompt
listed the three absolute paths as the only permitted reads and forbade Bash
and every other path, and each returned table was grepped for paths, run ids,
config ids and model names before acceptance (all three clean). The scratch
copies of the rubric and prompt hashed to `blinding.json`'s `rubric_sha256` /
`prompt_sha256`, and the scratch bundle was byte-identical to the run-dir bundle
(`shasum -a 256` equal), before any grader was spawned.

Inter-grader agreement: 36 of 40 bug cells unanimous; all 8 negative-control
cells unanimous (every list empty); 0 `LEAK?` flags raised. Verifier pass 1 ruled
the disputed cells; verifier pass 2 examined all 44 reviews against the leak
confirmation criteria and confirmed none, with no grade corrections. Reconciled
S3 FP totals: B 0, F 0.

Committed grading files under `campaigns/2026-10-gpt61sol-grading/` (blind labels; machine paths
redacted to `<REPO>`, `<COMMON>`, `<MAIN>`, `<WT-OTHER-n>`, `<CODEX_HOME>`,
`<SCRATCH>`; sha256 campaign-time original -> committed copy):

- `grader-1.json`: `60bb846d12c49c23865cb89ca490d95fbbcf643a7f01be78cde1369be1100f90` -> `60bb846d12c49c23865cb89ca490d95fbbcf643a7f01be78cde1369be1100f90`
- `grader-2.json`: `15ef73acbf4b94c8ee658cbdabe6ec4946b29681a4ce5fa9f4e001e8124973e4` -> `15ef73acbf4b94c8ee658cbdabe6ec4946b29681a4ce5fa9f4e001e8124973e4`
- `grader-3.json`: `c5d5c64cc372e573b70231c5eca79ac85fc5b30fb6ed75ac25e23060912c92d1` -> `c5d5c64cc372e573b70231c5eca79ac85fc5b30fb6ed75ac25e23060912c92d1`
- `verifier.json`: `c9b70f54f35ba80c7e94ef032b54a30e3fa7098d49ef0b473f0de224eb9a8f3c` -> `c9b70f54f35ba80c7e94ef032b54a30e3fa7098d49ef0b473f0de224eb9a8f3c`
- `reconciled.json`: `91f9272cf4d06ec66378609e23a183707f8e1f661f2c3fac6c55216d6d82b423` -> `91f9272cf4d06ec66378609e23a183707f8e1f661f2c3fac6c55216d6d82b423`
- `hits.json`: `b29397e503bea8d2f76efd4cb5af8ed52658f5c743edd4cfd5b20fc44496e7aa` -> `e05a846eddb8f9d7b994a76b5b042f1c26e77bf191295b8be616bafb7cd3f461`
- `verdict-provisional.txt`: `a57d3c819b6a9eaa0f0bfc2114699d9e7166885e061da2ccb528fbd7e7615a38` -> `a57d3c819b6a9eaa0f0bfc2114699d9e7166885e061da2ccb528fbd7e7615a38`
- `verdict-definitive.txt`: `b931bd6c09cf859f8a4d083199b4090565db2e92fa8d799e478756b1f07d50e9` -> `d1984ac805538ff42d55c9f0a4e4ba7f66ec5e3c3e6346397d587f5c5f2c309c`

## Grader notes (blinded, consensus)

Three graders agreed on 44 of 48 cells; the four disputes were the s2-cic cells
(one grader counted the CiC-axis attack on the same screen as a catch; the
verifier ruled all four partial, matching campaign 1). M1 (= F) runs more shell
commands per review and reports pytest as unable to start in the sandbox in
several cells; M2 (= B) cites test counts. Shared blind spots unchanged: both
s4-wcr guard-order bugs and the QDiD framing in s2-cic.

## Deviations from the plan, recorded

- Campaign 2 ran on a local, unpushed commit (`e1674c0e`) at the user's direction
  ("go straight to the new test runs before opening another PR"); the registration
  edits and both verdicts ship together in this PR. The frozen sha is not on any
  remote; the local tag and the recorded tree sha are the audit anchors.
- The rehearsal block was re-run on `e1674c0e` concurrently with the campaign's
  first minutes (same harness bytes); it passed every expectation, including the
  `--control B --candidate F` verdict refusing only at protocol guard (f).

## Provenance (committed audit record)

- Frozen HEAD `e1674c0e2a138f63300696f2b967268920c7e07b`, tree `4329ca6dd7eeb2f84b032145a53d54d0214d00a0`, local tag `campaign-gpt61sol` (never pushed), branch `codex-sol61`, subdir `runs/gpt61sol`.
- Manifest: 44 OK run_ids, configs ['B', 'F'], k=2; `repo_dirty`=False, `run_head_shas`=['e1674c0e2a138f63300696f2b967268920c7e07b'], `run_head_missing`=0, `run_dirty_any`=False, `run_marker_mismatch`=0, `run_not_isolated`=0; `verdict` recomputed the aggregates from the sealed artifacts and found no difference.
- `artifacts_sha` (sealed at `run` exit): `37939d7ab22888d9c0d73ef9c97380e1df48b05f7b1df633b542a1fcc162ac61`.
- Exposure ledger: first dispatch of the contrast (frozen revision); mechanical=True; ledger entries for this contrast=1; subdirs=['gpt61sol'].
- Attempts: 1 dispatched attempt, 0 reruns, 0 INFRA_ERROR cells. Run-log records (verbatim):
  - `=== run attempt 2026-10-17T19:14:42Z HEAD=e1674c0e2a138f63300696f2b967268920c7e07b`
  - `RUN EXIT 0`
  - `=== sealed e1674c0e2a138f63300696f2b967268920c7e07b 37939d7ab22888d9c0d73ef9c97380e1df48b05f7b1df633b542a1fcc162ac61`
- Codex CLI 0.162.1 on every arm (runner-asserted); isolated `CODEX_HOME` (credential + `sandbox_mode = "read-only"` + `approval_policy = "never"`, codex default features otherwise; the default feature surface is the gitignored `runs/features-0.162.1.txt`, captured by `codex features list` under the isolated home on 2026-10-16).
- Hashes (sha256) at verdict time: `runs/gpt61sol/comparison.md` `9060951065b60b058f92447c7dd733e20740ea0cad45b37871c113d2a6fb0779`; `runs/gpt61sol/comparison.blinded.md` `2102d1301323b9f9f4480aaa7d454d9ed85f1256c24742aaafbdd3ee4c2fe323`; `runs/gpt61sol/blinding.json` `77465c3f6407ea60a01de6a895861cde874b4e04f7618c5637d05f8ca7489c4c` (bundle_id `cd55711a27f7080f`, run_ids_sha `b02ea16a5f05b574c52d51c9e3a98c410aa66655617212ed2cf9aa7b76cf08c7`, rubric_sha256 `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, prompt_sha256 `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`).
- Files at the tag: `DECISION_RULE.md` `1ae891b721b017c866f5df13ed14faa9d5dfdc0d96367ecd4b89085e5ca47f3f`, `GRADING_RUBRIC.md` `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, `.github/codex/prompts/pr_review.md` `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`.
- run_ids (44): `s1-coef-dict-collision.B.r0.20aa9cdfe720d4bf`, `s1-coef-dict-collision.B.r1.6052f259b02bb528`, `s1-coef-dict-collision.F.r0.9fc2c9b209283f2b`, `s1-coef-dict-collision.F.r1.85d53c850a62cfd1`, `s1-cs-dr-plugin-se.B.r0.05074433eec6818a`, `s1-cs-dr-plugin-se.B.r1.fc7146a2c93d3504`, `s1-cs-dr-plugin-se.F.r0.412d0ad970061fa3`, `s1-cs-dr-plugin-se.F.r1.fb16eb6b1e275c5f`, `s2-acrt-single-dose.B.r0.4b72df486f75dedb`, `s2-acrt-single-dose.B.r1.a740a6a97984c8b8`, `s2-acrt-single-dose.F.r0.326b80328a8d3a78`, `s2-acrt-single-dose.F.r1.2d2a1be27eaa1060`, `s2-cic-practitioner-screen.B.r0.6b3b0cc6400ca460`, `s2-cic-practitioner-screen.B.r1.ae2a0146b73da067`, `s2-cic-practitioner-screen.F.r0.7f782de8a8275a79`, `s2-cic-practitioner-screen.F.r1.1a700c6a6699c5c7`, `s2-had-cluster-guard.B.r0.84d356cf06268d78`, `s2-had-cluster-guard.B.r1.d93ccde8dcab37d8`, `s2-had-cluster-guard.F.r0.c60f07888d9870a1`, `s2-had-cluster-guard.F.r1.a8d6493c293553e3`, `s2-nan-dof-fail-closed.B.r0.b4a41e147c4f76d4`, `s2-nan-dof-fail-closed.B.r1.f7c068891af83ecb`, `s2-nan-dof-fail-closed.F.r0.a6507d13f73cd5c5`, `s2-nan-dof-fail-closed.F.r1.65b2cab77e0194bc`, `s3-changelog-prose.B.r0.e915ea4986624500`, `s3-changelog-prose.B.r1.6145465e121eb71c`, `s3-changelog-prose.F.r0.5b91fd9013564cf1`, `s3-changelog-prose.F.r1.07863d32d1d8428e`, `s3-wcr-pfloor-comment.B.r0.4701bae6768929fa`, `s3-wcr-pfloor-comment.B.r1.aedad7cc7c2b6cff`, `s3-wcr-pfloor-comment.F.r0.fbd4b310f9cbbb68`, `s3-wcr-pfloor-comment.F.r1.eb55ec4ff8bfaf17`, `s4-cs-unbalanced-nan-outcome.B.r0.210c316a4170c0de`, `s4-cs-unbalanced-nan-outcome.B.r1.e16b6e9f972fe784`, `s4-cs-unbalanced-nan-outcome.F.r0.00de911eed37c6d7`, `s4-cs-unbalanced-nan-outcome.F.r1.c88216ee083a35db`, `s4-lpdid-survey-unreduced-design.B.r0.efaaeba38d503b0d`, `s4-lpdid-survey-unreduced-design.B.r1.23f6433415d5d879`, `s4-lpdid-survey-unreduced-design.F.r0.448caf8e76238e6a`, `s4-lpdid-survey-unreduced-design.F.r1.cd778455b6a48164`, `s4-wcr-saturated-guard-order.B.r0.bde484b74d09a291`, `s4-wcr-saturated-guard-order.B.r1.650895b917c6144c`, `s4-wcr-saturated-guard-order.F.r0.58e5f9ed2f0b3e49`, `s4-wcr-saturated-guard-order.F.r1.a19d8ecb7eb4e035`.

