# GPT-6.2 Sol Reviewer Evaluation - Verdict (2026-11-26): GO, accepted

Pre-registered rule: `DECISION_RULE.md` (the campaign-2 rule re-issued for control F
and candidate G; registered at local commit `51e4a85e` on branch `sol-review`, the
commit this PR squashes in; the local tag `campaign-gpt62sol` keeps that sha
resolvable). Arms: F = gpt-6.1-sol @ xhigh k=2 (control, re-sampled; production at
the time), G = gpt-6.2-sol @ xhigh k=2 (candidate; its first corpus run was this
campaign). Both arms on codex-cli 0.172.0. 44/44 runs OK in one attempt at
`--max-parallel 4`, 11 cases, zero INFRA_ERROR. Per the user's standing direction
the campaign ran on a local commit before any PR was opened.

**Mechanical verdict (`run_eval.py verdict`, DEFINITIVE with `--audit-hits`): `GO (DEFINITIVE)`.**
Regressions: 0. FP: G 0 vs F 0. Downgrades: 0. Strict improvements: 1
(`s4-wcr-saturated-guard-order:b1`: F MISSED, G RELIABLE). Judgment flag:
`s4-cs-unbalanced-nan-outcome:b1` F UNSTABLE / G MISSED. Contested gates: none.
Precedence reaches step (6).

**User's final call (2026-11-26): accept the mechanical GO.** Swap production (CI
reviewer, local `/ai-review-local` default, plan-review dual engine) to gpt-6.2-sol,
conditional on fixing the local prompt first (see "Local prompt finding"). The
inputs the user had when deciding: the catch table below, the judgment flag and its
history (next paragraph), the CLI-bump read, latency, cost, the audit and the
local-prompt probes.

**The judgment flag, in context.** G missed `s4-cs-unbalanced-nan-outcome:b1` in
both repeats (partial both times). The fresh control caught it in one of two
repeats, so the rule reports a flag, not a regression. The committed campaign-2
sample of the same control model caught it in both repeats, and this is the bug on
which gpt-6-astra regressed in campaign 1. Pooled over the two control samples
gpt-6.1-sol caught it in 3 of 4 repeats; gpt-6.2-sol in 0 of 2. No gate pools
samples; this is recorded so the miss is not read as noise by default.

## Catch table (10 ground-truth bugs; C = caught, P = partial, M = missed; partial counts as missed for every gate)

| Bug (stratum) | F r0 / r1 | G r0 / r1 | F status | G status |
|---|---|---|---|---|
| s1-coef-dict-collision:b1 (S1) | C / C | C / C | reliable | reliable |
| s1-cs-dr-plugin-se:b1 (S1) | C / C | C / C | reliable | reliable |
| s2-acrt-single-dose:b1 (S2) | P / P | C / P | missed | unstable |
| s2-cic-practitioner-screen:b1 (S2) | P / P | P / P | missed | missed |
| s2-had-cluster-guard:b1 (S2) | C / C | C / C | reliable | reliable |
| s2-nan-dof-fail-closed:b1 (S2) | C / C | C / C | reliable | reliable |
| s4-cs-unbalanced-nan-outcome:b1 (S4) | C / P | P / P | unstable | missed |
| s4-lpdid-survey-unreduced-design:b1 (S4) | C / C | C / C | reliable | reliable |
| s4-wcr-saturated-guard-order:b1 (S4) | M / M | C / C | missed | reliable |
| s4-wcr-saturated-guard-order:b2 (S4) | M / M | M / M | missed | missed |
| **Reliable catches** | | | **5/10** | **6/10** |
| **Caught repeat-cells (of 20)** | | | **11** | **13** |

Mapping (unblinded after the tables were final): F = M2, G = M1.

## Pre-registered gates (G vs F)

1. **Regression NO-GO: PASS.** No bug is RELIABLE for F and MISSED for G.
2. **FP NO-GO: PASS.** 0 vs 0 on both negative controls.
3. **GO: MET.** Zero regressions, zero downgrades, one strict improvement:
   `s4-wcr-saturated-guard-order:b1` (the wild-bootstrap wrapper's storage vcov on a
   rank-deficient full-dummy design), which no arm of any earlier campaign
   (gpt-5.6-sol, gpt-6-astra, gpt-6.1-sol) caught in any repeat.
4. **PARITY: not reached.**

## CLI-bump read (pre-registered, descriptive; no gate uses it)

Same corpus, prompt and rubric bytes, same model id (gpt-6.1-sol). Three things
differ between the two F columns: the CLI (0.162.1 vs 0.172.0), the isolated home's
connector setting (campaign 2 ran codex's default, app connectors available;
campaign 3 ran `[features] apps = false`), and the sample itself (independent
repeats). R = reliable, U = unstable, M = missed.

| Bug | F on 0.162.1 (campaign 2) | F on 0.172.0 (campaign 3) | G on 0.172.0 |
|---|---|---|---|
| s1-coef-dict-collision:b1 | R | R | R |
| s1-cs-dr-plugin-se:b1 | R | R | R |
| s2-acrt-single-dose:b1 | M | M | U |
| s2-cic-practitioner-screen:b1 | M | M | M |
| s2-had-cluster-guard:b1 | R | R | R |
| s2-nan-dof-fail-closed:b1 | U | R | R |
| s4-cs-unbalanced-nan-outcome:b1 | R | U | M |
| s4-lpdid-survey-unreduced-design:b1 | R | R | R |
| s4-wcr-saturated-guard-order:b1 | M | M | R |
| s4-wcr-saturated-guard-order:b2 | M | M | M |
| Reliable catches | 5 | 5 | 6 |
| Caught repeat-cells (of 20) | 11 | 11 | 13 |

The control's totals are identical across the two campaigns; two bugs moved between
reliable and unstable, one repeat each way. A difference of that size cannot be
attributed to any one of the three differences above (the CLI, the connector
setting or repeat variation), and this comparison cannot separate them. The CI `codex-version` pin moves to 0.172.0 with the swap
(gpt-6.2-sol is not reachable on 0.162.1).

## Informational reads (non-gating)

| | F (gpt-6.1-sol) | G (gpt-6.2-sol) |
|---|---|---|
| Latency mean / p95 / max (s) | 217 / 311 / 353 | 231 / 352 / 392 |
| Candidate p95 > 2x control | | False |
| Input price (USD per 1M, standard tier, read 2026-11-26) | 2.00 | 2.00 |
| Initial-prompt cost estimate, whole arm (USD, +/-50%) | 0.63 | 0.63 |
| Reviews with a non-shell event | 21 of 22 | 22 of 22 |
| Mean non-shell events per review | 11.5 | 11.4 |
| Mean shell commands per review (min) | 25.1 (10) | 24.1 (7) |
| Cells with a non-empty grader note (the engine's `hallucinations` counter) | 22 | 22 |
| Cells where the graders recorded an unverified external-tool claim | 19 | 21 |
| Confirmed leaks / unconfirmed flags | 0 / 0 | 0 / 0 |

The engine's `hallucinations` counter (`reviewer_verdict.informational`) counts
every cell with a non-empty `notes` field, and the graders also used `notes` for
grading rationale, so it reads 22 of 22 for both arms and overstates the claim
count. Read from the three grader files instead, a cell counts when a grader
note records a test run or numeric reproduction the grader could not verify:
F 19 of 22, G 21 of 22, with all three graders agreeing on every one of the 44
cells. The four cells without such a claim are negative-control reviews that
say tests were not run or that the review was static (`s3-changelog-prose` F r1,
`s3-wcr-pfloor-comment` F r0, F r1 and G r0). No gate reads this figure. A
dedicated field for unverified claims needs a rubric change, which is bound by
hash to each campaign's bundle, so it is a `TODO.md` row for the next campaign's
registration rather than a change here.

Vendor lifecycle (prose, operator-supplied): codex-cli 0.172.0's model catalog
labels gpt-6.2-sol "Fast, go-to model for coding and everyday work" and gpt-6.1-sol
"Previous generation model for coding and everyday work" (catalog fetched under the
isolated home, 2026-11-26).

## Pre-registration probes and the two changes they caused

All probes are off-corpus single reviews of merged PR #841's diff (554 changed
lines) at xhigh on codex-cli 0.172.0. They are recorded in `config/configs.json`.

| Model, prompt, home | Wall-clock (s) | Shell commands | Web searches | Connector calls |
|---|---|---|---|---|
| gpt-6.1-sol, local prompt, isolated | 305 | 12 | 14 | 0 |
| gpt-6.2-sol, local prompt, isolated | 222 | 0 | 14 | 24 (`github.fetch_file`) |
| gpt-6.2-sol, local prompt, isolated, `apps = false` | 226 | 0 | 26 | 0 |
| gpt-6.2-sol, local prompt, isolated, `apps = false`, web search off | 154 | 0 | 0 | 0 |
| gpt-6.2-sol, CI prompt, isolated | 242 | 22 | 10 | 0 |
| gpt-6.2-sol, CI prompt + an explicit inspect-the-checkout block, isolated | 567 | 43 | 14 | 1 (`github.fetch_pr_comments`) |

"Local prompt" is `openai_review.compile_prompt` (the `/ai-review-local` path);
"CI prompt" is `ci_prompt.build_ci_prompt`, the one the harness and CI run.

1. **Connector off for every arm (before the freeze, user decision).** Under the
   ChatGPT-account login the candidate called the account's GitHub connector,
   which reads the published repository. CI authenticates with an API key and is
   taken to have no connector (an inference from the auth mode). The isolated home
   sets `[features] apps = false`; the campaign's raw codex stderr shows zero
   `mcp:` events across all 44 reviews.
2. **Local prompt finding (after the verdict, same PR).** Under the local prompt
   gpt-6.2-sol ran no shell command in 3 of 3 probes; under the CI prompt it reads
   the checkout unprompted (and did so in all 22 campaign reviews). The campaign
   does not exercise the local prompt, so the GO does not cover it.

   Root cause: `openai_review._SUBSTITUTIONS` replaced the criteria's `grep`
   directive with "no shell access in this path; do not claim to have run
   `grep`", a sentence written for the single-shot api backend but applied to the
   codex backend as well. gpt-6.1-sol ignored it; gpt-6.2-sol obeyed it and went
   to the web and the connector for the code. The fix makes that substitution
   api-only (`_API_ONLY_SUBSTITUTIONS`). An interim fix, an added "Repository
   Access" prompt section, also restored shell use (34, 31 and 27 shell commands
   in three probes) but was dropped once the cause was known, so the local codex
   prompt stays as close as possible to the measured CI prompt. Probes with the
   final fix (same diff, gpt-6.2-sol):

   | Prompt, home | Wall-clock (s) | Shell commands | Web searches | Connector calls |
   |---|---|---|---|---|
   | local prompt, substitution api-only, isolated | 336 | 34 | 14 | 0 |
   | local prompt, substitution api-only, real codex home | 320 | 26 | 14 | 0 |
   | local prompt, substitution api-only, real codex home | 406 | 24 | 26 | 0 |
   | plan-review reviewer prompt (unchanged), real codex home | 283 | 35 | 0 | 0 |

   These show that gpt-6.2-sol reads the working tree under the local and
   plan-review prompts. They do not measure review quality on those surfaces;
   that stays carried over from this campaign (TODO.md rows). The four
   pre-registration probes in the table above that used the local prompt all
   carried the "no shell access" sentence; the two CI-prompt probes did not.

## Session-log audit (`run_eval.py audit`, post-unblinding, informational)

Coverage: covered=1, no_commands=0, unaudited=43, total=44; vacuous=False. Nearly
every review web-searched, which classifies the cell `unaudited`; the shell log is
still scanned and reported.

- Cells with at least one hit: 42 of 44. Hits per arm: {'F': 82, 'G': 168}.
- Most frequent markers: `repo_root` x172, `GIT_LOG_RE` x39, `tools/reviewer-eval/`
  x26, `GIT_DIR_RE` x22, `../` x13, `worktrees/` x12, `DOTDOT_RE` x7, `--format` x7,
  `BARE_SHOW_RE` x6. The bulk are pytest invocations through the enclosing
  checkout's `.venv` (F 45 commands, G 84) and codex's AGENTS.md discovery walk up
  the parent directories.
- Hits on ref-reaching markers (`main`/`master`, `origin/`, `refs/`, tags,
  `rev-list`, `reflog`, `for-each-ref`, `cat-file`, `fsck`, `worktree list`): none.
- Reads of the enclosing checkout outside the reviewer's own worktree, classified
  by hand from every shell command that names it: no command read the corpus, a run
  artifact or another worktree, and none read source, tests or docs from the
  enclosing checkout. G read the enclosing checkout's `changelog.d/README.md` (11
  commands) and `DEFERRED.md` (6 commands), process files that do not exist at the
  historical case commits; F did not. For the improved bug
  (`s4-wcr-saturated-guard-order:b1`) G's two `DEFERRED.md` searches match one row
  of that file, about TWFE HC2 dummy deduplication, which does not describe the bug.
- Web search: 502 `web search:` lines across the 44 reviews. No line with a visible
  query names this project's repository; the repository URLs that appear are
  third-party R packages (DRDID, did, HonestDiD, nprobust, clubSandwich). Many
  lines carry no visible query, so this check is partial, and the grader `LEAK?`
  flag plus the verifier passes remain the leak check.
- The committed `hits.json` and `verdict-definitive.txt` copies replace each hit's
  `line` with its sha256 (command lines carry machine paths); the full lines stay
  in the archived originals whose sha256 is listed below.

## Grading record

Three independent blinded grader subagents (inputs: `comparison.blinded.md`,
`GRADING_RUBRIC.md`, `.github/codex/prompts/pr_review.md`; Bundle ID
`bef1af4c1e8a8c0d`; blind labels M2=F, M1=G, mapping unknown to them and not read
by the orchestrating session until the tables were final). Isolation is by
instruction plus check, not a sandbox: the grader prompt listed the three absolute
paths as the only permitted reads and forbade Bash and every other path, and each
table was grepped for paths, run ids, config ids and model names before acceptance
(all three clean). The scratch copies of the rubric and prompt hashed to
`blinding.json`'s `rubric_sha256` / `prompt_sha256`, and the scratch bundle was
byte-identical to the run-dir bundle, before any grader was spawned.

Inter-grader agreement: 39 of 40 bug cells unanimous; all 8 negative-control cells
unanimous (every list empty); 0 `LEAK?` flags raised. Verifier pass 1 ruled the one
disputed cell (`s4-cs-unbalanced-nan-outcome:b1`, M2 r1: two graders partial, one
missed; ruled partial; both count as missed). Verifier pass 2 examined all 44
reviews against the leak confirmation criteria and confirmed none, with no grade
corrections and no S3 corrections. Reconciled S3 FP totals: F 0, G 0.

Committed grading files under `campaigns/2026-11-gpt62sol-grading/` (blind labels;
sha256 campaign-time original -> committed copy):

- `grader-1.json`: `cdbc009c962b1af36018710b21da4be6043c57b04d0cbd1c51d29d4ad5d590a0` -> `cdbc009c962b1af36018710b21da4be6043c57b04d0cbd1c51d29d4ad5d590a0`
- `grader-2.json`: `acc7875810375c56402b77a81e492a4fd36d6c71dd07d9f745a15ea4943a4329` -> `acc7875810375c56402b77a81e492a4fd36d6c71dd07d9f745a15ea4943a4329`
- `grader-3.json`: `ab8b268529c075271e7ef68f20bef48101916a53cca3fcfe80bc867eb305276f` -> `ab8b268529c075271e7ef68f20bef48101916a53cca3fcfe80bc867eb305276f`
- `verifier.json`: `fc08d47f6269c0f0447f3c8963217c3c3e4bca2c821fdd257e1db8332212e438` -> `fc08d47f6269c0f0447f3c8963217c3c3e4bca2c821fdd257e1db8332212e438`
- `reconciled.json`: `ed6c59c5e752ccec6efa05acdeb9e36e2c28584c037d000b4120d6ff6f53ee11` -> `ed6c59c5e752ccec6efa05acdeb9e36e2c28584c037d000b4120d6ff6f53ee11`
- `hits.json`: `74adb7cfc5f5e2ea68d305e5a43d536da78f063fddd07f172082f91f51c7a458` -> `b3d918af4ad1208cce6f5ccaf1f49b2cf5133cd9fac9090189f85c3ad3f7a8a3`
- `verdict-provisional.txt`: `d63c06a4adaec27b4c9464dd909f1c76277fc706e58c7aa1ff600897751f9899` -> `d63c06a4adaec27b4c9464dd909f1c76277fc706e58c7aa1ff600897751f9899`
- `verdict-definitive.txt`: `1638cec78dae2c23cffc442affd58872d90cd2aeeb70af972af46c1884cfe115` -> `66c373a2f833b8d0f54882637d53a997e75b77cf4bd1087df8377b5b0a01035b`

## Deviations from the plan and the earlier campaigns, recorded

- The isolated home disables one default feature (`apps`); the earlier campaigns
  disabled none. Decided by the user before the freeze and written into the rule.
- Graders and the verifier wrote their JSON to a designated scratch file with the
  Write tool instead of returning the table in their reply (one permitted write
  path besides the three permitted reads).
- The verifier's prompt carried the graders' verdict grid (letters only, blind
  labels) and the one dispute, so that pass 2 could check grades as well as leaks.
- The campaign-1 "rehearsal block" was not re-run: its definition is not in the
  repository. The control smoke on the registration commit (32/32 exec blocks
  parsed, isolated home) and the harness unit tests stood in for it.
- The registration commit was amended once after the control smoke (rule sentence
  recording the smoke, the 0.172.0 stderr fixture and its test) before the tag was
  set; the smoke ran on the pre-amend commit `eb9e5605`, whose harness code is
  byte-identical to the frozen `51e4a85e`.
- After the swap F is no longer the role=control arm but stays exempt from the
  candidate-exposure gate as the control of a registered contrast
  (`run_eval._candidate_exposure_violations`); without that it could not be smoked
  or re-sampled as a baseline.
- Fixed in the same PR: with both F and G exempt, a fresh checkout (which has neither the local ledger nor the frozen ref)
  could re-dispatch F,G and have `verdict` read it as the first pre-registered
  exposure. `reviewer_verdict.COMPLETED_CAMPAIGNS` now records each finished
  contrast's first-dispatch revision in tracked code and such a re-run is
  reported non-mechanical. The same gap existed for (B, F) after the 2026-10 swap.
- Recorded as a known limit rather than changed: codex can print a `web search:` line while a shell command is still running, and
  `call_codex`'s session parser keeps it in that command's recorded text (22 of
  this campaign's 250 audit hits carry such a line in their hashed text).
  Human-mode stderr cannot tell such a line from a literal line of a multi-line
  command, so the parser over-includes instead of risking a hidden command line.
  Hit counts, coverage classes and every gate are unaffected; a structured event
  source is a TODO.md row.

## Provenance (committed audit record)

- Frozen HEAD `51e4a85efea6805fdd50517239520578a253fb25`, tree `202c46b188e966553f350c47feae2d29fe752f92`, local tag `campaign-gpt62sol` (never pushed), ledger ref `refs/campaign/F-G/frozen`, branch `sol-review`, subdir `runs/gpt62sol`.
- Manifest: 44 OK run_ids, configs ['F', 'G'], k=2; `repo_dirty`=False, `run_head_shas`=['51e4a85efea6805fdd50517239520578a253fb25'], `run_head_missing`=0, `run_dirty_any`=False, `run_marker_mismatch`=0, `run_not_isolated`=0; `verdict` recomputed the aggregates from the sealed artifacts and found no difference.
- `artifacts_sha` (sealed at `run` exit): `bc5dbec5dd2339a2303058e790670c4a604b64e7a9af9840a1a63c5befb0974c`.
- Exposure ledger: first dispatch of the contrast (frozen revision); mechanical=True; ledger entries for this contrast=1; subdirs=['gpt62sol'].
- Attempts: 1 dispatched attempt, 0 reruns, 0 INFRA_ERROR cells. Run-log records (verbatim):
  - `=== run attempt 2026-11-26T18:48:08Z HEAD=51e4a85efea6805fdd50517239520578a253fb25`
  - `RUN EXIT 0`
  - `=== sealed 51e4a85efea6805fdd50517239520578a253fb25 bc5dbec5dd2339a2303058e790670c4a604b64e7a9af9840a1a63c5befb0974c`
- Codex CLI 0.172.0 on every arm (runner-asserted); isolated `CODEX_HOME` (credential + `sandbox_mode = "read-only"` + `approval_policy = "never"` + `[features] apps = false`, codex default features otherwise; the default feature surfaces of 0.162.1 and 0.172.0 are the gitignored `runs/features-<version>.txt`, captured by `codex features list` under the isolated home on 2026-11-26).
- Raw run artifacts, both comparison bundles, the run logs, the ledger and the unredacted grading files are archived outside the repository (operator's machine, `gpt62sol.tar.gz`), so a later campaign can re-grade these reviews jointly; the campaign-2 raw reviews were not kept.
- Hashes (sha256) at verdict time: `runs/gpt62sol/comparison.md` `4f7140f0fa6929a8fcac4d96eb0de88d68a6502601531224c22dce00a575f14e`; `runs/gpt62sol/comparison.blinded.md` `638dfb03aa550ba5de19faf3100b24d8f0d4c0fb1d06717ae3612bb48adf650e`; `runs/gpt62sol/blinding.json` `8ba3516978971a41f97da2c372f517c37fdd81fb7149c98441cdb4689a21b73c` (bundle_id `bef1af4c1e8a8c0d`, run_ids_sha `3723da96ab8a3cadad87fbdcb72a88cca83f5b6882504a477238d8e558ac71ae`, rubric_sha256 `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, prompt_sha256 `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`).
- Files at the tag: `DECISION_RULE.md` `b9c3fe84b4187b1f9c71ff44a05b81994101e733724912eface326584e89825d`, `GRADING_RUBRIC.md` `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, `.github/codex/prompts/pr_review.md` `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`.
- run_ids (44): `s1-coef-dict-collision.F.r0.93992ca8baf2bad3`, `s1-coef-dict-collision.F.r1.930e81099a4eb20c`, `s1-coef-dict-collision.G.r0.b695bcb59ec3b801`, `s1-coef-dict-collision.G.r1.6d70dfaadbdc1d9f`, `s1-cs-dr-plugin-se.F.r0.b59f3181ab4da2ec`, `s1-cs-dr-plugin-se.F.r1.b0475a92b2cffe0a`, `s1-cs-dr-plugin-se.G.r0.78caab17f2ab2b05`, `s1-cs-dr-plugin-se.G.r1.ec8849257dbf5657`, `s2-acrt-single-dose.F.r0.cebe1343d32e2e7b`, `s2-acrt-single-dose.F.r1.e26c09fb22579bd7`, `s2-acrt-single-dose.G.r0.b231149e8534bb16`, `s2-acrt-single-dose.G.r1.0ced3dd16f2fc728`, `s2-cic-practitioner-screen.F.r0.c5c4ffe28b828620`, `s2-cic-practitioner-screen.F.r1.f122871e49be6bc0`, `s2-cic-practitioner-screen.G.r0.dbea1bec23546fb0`, `s2-cic-practitioner-screen.G.r1.0d38a8200e5cfff8`, `s2-had-cluster-guard.F.r0.6893c53b65b723f1`, `s2-had-cluster-guard.F.r1.10f176b8eb051405`, `s2-had-cluster-guard.G.r0.223c7bc620e82c1a`, `s2-had-cluster-guard.G.r1.a26451c0c9d588d9`, `s2-nan-dof-fail-closed.F.r0.0eea9e5041469ba9`, `s2-nan-dof-fail-closed.F.r1.61e6609a1745414a`, `s2-nan-dof-fail-closed.G.r0.64c7622df8f45a9b`, `s2-nan-dof-fail-closed.G.r1.e54cd701471e8cc6`, `s3-changelog-prose.F.r0.5e41bef1f30516b3`, `s3-changelog-prose.F.r1.d31f193a4c23200e`, `s3-changelog-prose.G.r0.2e90a59439abd6a6`, `s3-changelog-prose.G.r1.7b9380a87f4f9188`, `s3-wcr-pfloor-comment.F.r0.81d0fc3af85453c6`, `s3-wcr-pfloor-comment.F.r1.873641b0ec21aa3a`, `s3-wcr-pfloor-comment.G.r0.3536473b06ec6493`, `s3-wcr-pfloor-comment.G.r1.b87109ea10408ff2`, `s4-cs-unbalanced-nan-outcome.F.r0.33fc3b9649d22640`, `s4-cs-unbalanced-nan-outcome.F.r1.617e9e78b10bed24`, `s4-cs-unbalanced-nan-outcome.G.r0.3197ea31b2c25b04`, `s4-cs-unbalanced-nan-outcome.G.r1.6012956b89760905`, `s4-lpdid-survey-unreduced-design.F.r0.dfc0c66c58d7e899`, `s4-lpdid-survey-unreduced-design.F.r1.0c265154b391cc43`, `s4-lpdid-survey-unreduced-design.G.r0.e013350972835802`, `s4-lpdid-survey-unreduced-design.G.r1.13c3f3cf5599b7d2`, `s4-wcr-saturated-guard-order.F.r0.98a70cee320a2ebd`, `s4-wcr-saturated-guard-order.F.r1.ec670a1ab6c1d880`, `s4-wcr-saturated-guard-order.G.r0.74a9646834a88436`, `s4-wcr-saturated-guard-order.G.r1.b51520d5bac89132`.
