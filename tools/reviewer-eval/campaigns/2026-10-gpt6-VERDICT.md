# GPT-6 Astra Reviewer Evaluation - Verdict (2026-10-17): NO-GO

Pre-registered rule: `2026-10-gpt6-DECISION_RULE.md` (archived from `DECISION_RULE.md`
when campaign 2 re-issued it for arm F; pre-registered in PR #849, frozen at the
#849 squash merge `225a4afb` on `main`). Arms: B = gpt-5.6-sol @ xhigh k=2 (control,
production at the time), E = gpt-6-astra @ xhigh k=2 (candidate; its first corpus
run was this campaign). 44/44 runs OK in one attempt (41 minutes at
`--max-parallel 4`), 11 cases, zero INFRA_ERROR.

**Mechanical verdict (`run_eval.py verdict`, DEFINITIVE with `--audit-hits`): `NO-GO (DEFINITIVE)`.**
Gate (1) regression: `s4-cs-unbalanced-nan-outcome:b1` is RELIABLE for B and MISSED for E.
Gate (2) FP: E 0 vs B 0, passes. Downgrades: 0. Strict improvements: `s2-nan-dof-fail-closed:b1`
(E reliable, B missed), which precedence step (3) does not reach. Judgment flag:
`s2-acrt-single-dose:b1` B UNSTABLE / E MISSED (reported, not gating). Contested gates: none.

**User's final call (2026-10-17): accept mechanical (NO-GO).** No swap to gpt-6-astra.
Reason recorded: the regression is a methodology miss of the kind the reviewer
exists to catch (sol named the NaN-outcome routing defect in both repeats; astra
twice reported a neighbouring NaN-cohort issue in the same file instead), the
arms are otherwise close to even on this corpus, and astra costs 2.5x on CI.

## Catch table (10 ground-truth bugs; C = caught, P = partial, M = missed; partial counts as missed for every gate)

| Bug (stratum) | B r0 / r1 | E r0 / r1 | B status | E status |
|---|---|---|---|---|
| s1-coef-dict-collision:b1 (S1) | C / C | C / C | reliable | reliable |
| s1-cs-dr-plugin-se:b1 (S1) | C / C | C / C | reliable | reliable |
| s2-acrt-single-dose:b1 (S2) | P / C | P / M | unstable | missed |
| s2-cic-practitioner-screen:b1 (S2) | P / P | P / P | missed | missed |
| s2-had-cluster-guard:b1 (S2) | C / C | C / C | reliable | reliable |
| s2-nan-dof-fail-closed:b1 (S2) | P / P | C / C | missed | reliable |
| s4-cs-unbalanced-nan-outcome:b1 (S4) | C / C | P / P | reliable | missed |
| s4-lpdid-survey-unreduced-design:b1 (S4) | P / C | C / C | unstable | reliable |
| s4-wcr-saturated-guard-order:b1 (S4) | P / M | M / M | missed | missed |
| s4-wcr-saturated-guard-order:b2 (S4) | M / M | M / M | missed | missed |
| **Reliable catches** | | | **4/10** | **5/10** |
| **Caught repeat-cells (of 20)** | | | **10** | **10** |

Mapping (unblinded after the tables were final): B = M2, E = M1.

## Pre-registered gates (E vs B)

1. **Regression NO-GO: FAIL.** `s4-cs-unbalanced-nan-outcome:b1`: B caught 2/2, E partial 2/2 (three graders unanimous).
2. **FP NO-GO: PASS.** 0 vs 0 on both negative controls (every finding at or below the allowed severities; the s3-wcr P3 on the documented floor appears in every arm and is allowed).
3. **GO conditions:** not reached (precedence step 3 fires first). For the record: zero downgrades, one strict improvement (`s2-nan-dof-fail-closed:b1`, E caught 2/2 where B was partial 2/2), and E also reliable on `s4-lpdid-survey-unreduced-design:b1` where B was unstable (not a rule improvement because B is not MISSED).
4. **PARITY:** not reached.

Precedence (0)-(7) applied as written; outcome at step (3).

## Informational reads (non-gating)

| | B (gpt-5.6-sol) | E (gpt-6-astra) |
|---|---|---|
| Latency mean / p95 / max (s) | 209 / 343 / 432 | 214 / 331 / 392 |
| Candidate p95 > 2x control | | False |
| Input price (USD per 1M, standard tier) | 4.00 | 10.00 |
| Initial-prompt cost estimate, whole arm (USD, +/-50%) | 1.26 | 3.15 |
| Reviews with a non-shell event (web search) | 18 of 22 | 21 of 22 |
| Mean shell commands per review | 14.0 | 16.6 |
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

Coverage: covered=5, no_commands=0, unaudited=39, total=44; vacuous=False.
Nearly every review web-searched (codex default features, decision 8), which
classifies the cell `unaudited`; the shell log is still scanned and reported.

- Cells with at least one hit: 39 of 44; cells whose hits are all orientation-only: 5. Hits per arm: {'B': 44, 'E': 61}.
- Most frequent markers: `repo_root` x41, `-maxdepth` x33, `GIT_LOG_RE` x25, `tools/reviewer-eval/` x8, `worktrees/` x7, `--format` x6, `GIT_DIR_RE` x6, `BARE_SHOW_RE` x4. The bulk are codex's own AGENTS.md discovery walk up the parent directories (`find -maxdepth`, `../..` loops, absolute paths under the enclosing checkout) and pytest invocations through the enclosing checkout's `.venv`; none of these reads the answer key.
- Hits on ref-reaching markers (`main`/`master`, `origin/`, `refs/`, tags, `rev-list`, `reflog`, `for-each-ref`, `cat-file`, `fsck`, `worktree list`): none.
- S1 cells whose shell log shows `git log` / `git show HEAD` (the synthetic HEAD's commit subject names the injected bug): s1-coef-dict-collision/E/r0, s1-coef-dict-collision/E/r1, s1-cs-dr-plugin-se/B/r0. Both arms caught both S1 bugs reliably in every repeat regardless, the verifier found no review citing the commit message as its source (confirmation criterion 3), and no gate is touched; recorded for transparency.
- The committed `hits.json` copy replaces each hit's `line` with its sha256 (command lines can carry file contents); the full lines stay in the gitignored original whose sha256 is listed below.

## Grading record

Three independent blinded grader subagents (inputs: `comparison.blinded.md`,
`GRADING_RUBRIC.md`, `.github/codex/prompts/pr_review.md`; Bundle ID
`d27906ede74da295`; blind labels M2=B, M1=E, mapping unknown to
them). Isolation is by instruction plus check, not a sandbox: the grader prompt
listed the three absolute paths as the only permitted reads and forbade Bash
and every other path, and each returned table was grepped for paths, run ids,
config ids and model names before acceptance (all three clean). The scratch
copies of the rubric and prompt hashed to `blinding.json`'s `rubric_sha256` /
`prompt_sha256`, and the scratch bundle was byte-identical to the run-dir bundle
(`shasum -a 256` equal), before any grader was spawned.

Inter-grader agreement: 33 of 40 bug cells unanimous; all 8 negative-control
cells unanimous (every list empty); 0 `LEAK?` flags raised. Verifier pass 1 ruled
the disputed cells; verifier pass 2 examined all 44 reviews against the leak
confirmation criteria and confirmed none, with no grade corrections. Reconciled
S3 FP totals: B 0, E 0.

Committed grading files under `campaigns/2026-10-gpt6-grading/` (blind labels; machine paths
redacted to `<REPO>`, `<COMMON>`, `<MAIN>`, `<WT-OTHER-n>`, `<CODEX_HOME>`,
`<SCRATCH>`; sha256 campaign-time original -> committed copy):

- `grader-1.json`: `0ee28b8ae353d6662efe89e3a3726ba1351afb8c72dd8943194356d515da1012` -> `0ee28b8ae353d6662efe89e3a3726ba1351afb8c72dd8943194356d515da1012`
- `grader-2.json`: `8f807cf646a6d1d41c30049b67f1e3df02dafa8f02b5759a05c29caaee5be408` -> `8f807cf646a6d1d41c30049b67f1e3df02dafa8f02b5759a05c29caaee5be408`
- `grader-3.json`: `5e50f8622adf65d6cbaf0b161fdfbabf966773293cf117627259f3013ff89940` -> `5e50f8622adf65d6cbaf0b161fdfbabf966773293cf117627259f3013ff89940`
- `verifier.json`: `73ddba0b85647d7e6ff1e0d6ee61ffcb9722fa1612f6839795759c26f3fa1959` -> `73ddba0b85647d7e6ff1e0d6ee61ffcb9722fa1612f6839795759c26f3fa1959`
- `reconciled.json`: `8fc6b8b810e422737c729137c6c504145339c8bb65321c94b7eabbfef5fcf2c8` -> `8fc6b8b810e422737c729137c6c504145339c8bb65321c94b7eabbfef5fcf2c8`
- `hits.json`: `955b56350217c5f32ad82d3374953052cb9db8ccd54de2d9e4b881fb0227ad4b` -> `27c1b65712e224bd06adeec1b42645085c5c72d42cb9404d21c48ab0b9690085`
- `verdict-provisional.txt`: `af8fbcc6aad9f3fc77aaaba9812b40cc5f2f7f3521ad32e40f93d2e38591372f` -> `af8fbcc6aad9f3fc77aaaba9812b40cc5f2f7f3521ad32e40f93d2e38591372f`
- `verdict-definitive.txt`: `9e107f60f2f9239dc200631704347b6cf87d209aeab02538685674a08a8fbdfa` -> `48540e912a16efa8b442e363536ff8246c230597f30bbf7cfaa4cda10ff8f4a2`

## Grader notes (blinded, consensus)

The same qualitative picture from all three graders: M1 (= E) and M2 (= B) both
produce location-precise, reproduction-style findings; M1 reports pytest as
unable to start in the sandbox and falls back to "direct" Python reproductions,
M2 cites test counts. Shared blind spots: both s4-wcr guard-order bugs (every
arm, every repeat, as in 2026-07) and the QDiD framing in s2-cic (all four cells
attack the same screen on the CiC axis; verifier ruled partial). Neither arm
reproduces the 2026-07 observation that sol misses the S4 bugs: B caught
s4-cs 2/2 and s4-lpdid 1/2 in this run.

## Deviations from the plan, recorded

- The campaign ran on `main`'s tip after #849 merged, not on the open PR branch;
  the ledger's first dispatch record is that sha and the tag pins it locally.
- The pre-freeze rehearsal was not repeated on `225a4afb` because the registered
  surfaces were byte-identical to the rehearsed `ec6226ae` (only the 3.13.0
  version bump and CHANGELOG differed).

## Provenance (committed audit record)

- Frozen HEAD `225a4afb73106614c37d34b21100eafcdf34be43`, tree `5c05e9e2c5019c4f98067c28dbb63d9322c94bac`, local tag `campaign-gpt6` (never pushed), branch `codex-upgrade`, subdir `runs/gpt6`.
- Manifest: 44 OK run_ids, configs ['B', 'E'], k=2; `repo_dirty`=False, `run_head_shas`=['225a4afb73106614c37d34b21100eafcdf34be43'], `run_head_missing`=0, `run_dirty_any`=False, `run_marker_mismatch`=0, `run_not_isolated`=0; `verdict` recomputed the aggregates from the sealed artifacts and found no difference.
- `artifacts_sha` (sealed at `run` exit): `83337a6b7e139d7ecade26a6560e47b98b9a55659fd666e0fb7de0c1e47d6ee1`.
- Exposure ledger: first dispatch of the contrast (frozen revision); mechanical=True; ledger entries for this contrast=1; subdirs=['gpt6'].
- Attempts: 1 dispatched attempt, 0 reruns, 0 INFRA_ERROR cells. Run-log records (verbatim):
  - `=== run attempt 2026-10-17T17:53:35Z HEAD=225a4afb73106614c37d34b21100eafcdf34be43`
  - `RUN EXIT 0`
  - `=== sealed 225a4afb73106614c37d34b21100eafcdf34be43 83337a6b7e139d7ecade26a6560e47b98b9a55659fd666e0fb7de0c1e47d6ee1`
- Codex CLI 0.162.1 on every arm (runner-asserted); isolated `CODEX_HOME` (credential + `sandbox_mode = "read-only"` + `approval_policy = "never"`, codex default features otherwise; the default feature surface is the gitignored `runs/features-0.162.1.txt`, captured by `codex features list` under the isolated home on 2026-10-16).
- Hashes (sha256) at verdict time: `runs/gpt6/comparison.md` `69dab0d9e44688b572767a046e9c2b7b1b066c91365491bee587d52906158400`; `runs/gpt6/comparison.blinded.md` `0ff258435e1b4f7a87eb019c2715294422b661cd16ee44af1193cae27f77da4b`; `runs/gpt6/blinding.json` `9b61cae0d27a3720e678cd37a5b7679d864ad5e97ae4348bced0d0511c29fdbb` (bundle_id `d27906ede74da295`, run_ids_sha `413aae215614d4909ab25c85e66b750b233ec7a731179001c3feb926242e4264`, rubric_sha256 `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, prompt_sha256 `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`).
- Files at the tag: `DECISION_RULE.md` `adb0ac511f7cef179f343ab4cd425edd420eddcf6b52640b27c11c71c983cd1a`, `GRADING_RUBRIC.md` `0388e64b0b7defc1ec7a8d4af1950c7a15aa4ad43c62021f69668ba07f4cb6dd`, `.github/codex/prompts/pr_review.md` `098ecc7ee3bcfe10fedca4635f277966d717037def7f4162d145de494a217c80`.
- run_ids (44): `s1-coef-dict-collision.B.r0.20aa9cdfe720d4bf`, `s1-coef-dict-collision.B.r1.6052f259b02bb528`, `s1-coef-dict-collision.E.r0.68031b6fd69d91a8`, `s1-coef-dict-collision.E.r1.dbd7255ebee2c3b3`, `s1-cs-dr-plugin-se.B.r0.05074433eec6818a`, `s1-cs-dr-plugin-se.B.r1.fc7146a2c93d3504`, `s1-cs-dr-plugin-se.E.r0.f9425b6303772854`, `s1-cs-dr-plugin-se.E.r1.6f972c909ca3317a`, `s2-acrt-single-dose.B.r0.4b72df486f75dedb`, `s2-acrt-single-dose.B.r1.a740a6a97984c8b8`, `s2-acrt-single-dose.E.r0.9cf50a17bfb4b7b1`, `s2-acrt-single-dose.E.r1.60289d8963706496`, `s2-cic-practitioner-screen.B.r0.6b3b0cc6400ca460`, `s2-cic-practitioner-screen.B.r1.ae2a0146b73da067`, `s2-cic-practitioner-screen.E.r0.2f01ad7139538355`, `s2-cic-practitioner-screen.E.r1.568da36ad60df6de`, `s2-had-cluster-guard.B.r0.84d356cf06268d78`, `s2-had-cluster-guard.B.r1.d93ccde8dcab37d8`, `s2-had-cluster-guard.E.r0.83185d5fc44ea848`, `s2-had-cluster-guard.E.r1.5073afcaefe3d417`, `s2-nan-dof-fail-closed.B.r0.b4a41e147c4f76d4`, `s2-nan-dof-fail-closed.B.r1.f7c068891af83ecb`, `s2-nan-dof-fail-closed.E.r0.58171b295011ffef`, `s2-nan-dof-fail-closed.E.r1.476917d44198ed86`, `s3-changelog-prose.B.r0.e915ea4986624500`, `s3-changelog-prose.B.r1.6145465e121eb71c`, `s3-changelog-prose.E.r0.d094a4936c921514`, `s3-changelog-prose.E.r1.173ade27085e8433`, `s3-wcr-pfloor-comment.B.r0.4701bae6768929fa`, `s3-wcr-pfloor-comment.B.r1.aedad7cc7c2b6cff`, `s3-wcr-pfloor-comment.E.r0.0f52602211c3d990`, `s3-wcr-pfloor-comment.E.r1.77bc1cebbe9b298f`, `s4-cs-unbalanced-nan-outcome.B.r0.210c316a4170c0de`, `s4-cs-unbalanced-nan-outcome.B.r1.e16b6e9f972fe784`, `s4-cs-unbalanced-nan-outcome.E.r0.ca60b0c855ef3f89`, `s4-cs-unbalanced-nan-outcome.E.r1.8ce5e6f66e7db98b`, `s4-lpdid-survey-unreduced-design.B.r0.efaaeba38d503b0d`, `s4-lpdid-survey-unreduced-design.B.r1.23f6433415d5d879`, `s4-lpdid-survey-unreduced-design.E.r0.7dd4f402504a9d4e`, `s4-lpdid-survey-unreduced-design.E.r1.069b2a344443fca7`, `s4-wcr-saturated-guard-order.B.r0.bde484b74d09a291`, `s4-wcr-saturated-guard-order.B.r1.650895b917c6144c`, `s4-wcr-saturated-guard-order.E.r0.405e2ec712320ccc`, `s4-wcr-saturated-guard-order.E.r1.bdbe12f3335d000d`.

