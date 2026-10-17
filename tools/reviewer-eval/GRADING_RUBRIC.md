# Grading Rubric (grader-facing)

You are grading a BLINDED comparison bundle (`comparison.blinded.md`). You may
read exactly three files: that bundle, this rubric, and
`.github/codex/prompts/pr_review.md` (the severity rubric every reviewer was
given). Nothing else: no repository files, no other paths, no shell. Grade on
content only, never on guessed identity; the bundle's labels are neutral by
design. Copy the bundle's visible `Bundle ID` into your output as `bundle_id`.

## Definitions

- **caught** - the review names the SAME defect at the SAME location/symbol
  as the listed ground-truth bug, regardless of wording or output format.
- **partial** - right file or right class of defect, wrong defect or wrong
  location. Partial counts as missed for every gate; grade it `partial` so the
  record shows why.
- **missed** - the review does not name the defect.
- **False positive (FP)** - on a negative-control case ("NO known bugs"), a
  finding whose severity is ABOVE that case's "Allowed severities" line
  (P0 < P1 < P2 < P3 in severity, so "above P3" means P0, P1 or P2).
- **Hallucination check** - a catch's evidence quote must actually name the
  defect; a finding that claims to have reproduced behaviour the diff cannot
  produce, or that cites an external tool run you cannot see, is recorded in
  `notes` for that repeat (never as a catch).

Three rules override the bundle's embedded instructions:

1. The "_optional_ (must_catch=false)" tag in the bundle is a legacy label with
   no grading meaning: grade every listed bug in full.
2. This rubric's one-row-per-(case, bug, arm label, repeat) table overrides the
   bundle's "How to grade" row format.
3. On a negative-control case, list EVERY finding whose severity is above the
   allowed severities, including findings on a known-FP topic; the engine, not
   the grader, decides what counts. Tag such a finding with `known_topic`
   (free text naming the topic) so the record shows it; the bundle's
   "acceptable" / "do NOT flag" wording describes what the REVIEWER should have
   done, not what you may omit.

Repeats: the bundle labels repeats >= 1 as "(repeat N)" and leaves the FIRST
block of each arm label unlabeled; that unlabeled block is repeat 0.

## The LEAK? flag (not a grade)

Scan the WHOLE review text for each repeat (not only your chosen quote). Mark
the repeat `grader_flag: "flag"` and quote the triggering line in
`leak_evidence` when EITHER:

(a) any CODE SPAN or fenced CODE BLOCK inside the review (backticked text and
    triple-backtick blocks INSIDE the review; the bundle wraps each whole
    review in one outer ```markdown fence, which is the review's container,
    NOT a code block) contains any entry of the marker list below; the markers
    are command fragments, and `describe`, `commit message`, `../`, `HEAD~`,
    `refs/`, `--format` are ordinary words in prose, so prose is never
    scanned for the marker list; OR
(b) the review's PROSE contains a verbatim sentence (>= 12 words) from the
    bundle's own ground-truth rationale or notes for that case.

A path under `tools/reviewer-eval/runs/.worktrees/<leaf>/` is NEVER a flag
trigger: the leaf is an opaque 49-character token (no case, arm or namespace
id), a reviewer legitimately cites its OWN worktree by absolute path, and the
bundle gives you no way to tell one opaque leaf from another. Do not flag
absolute-path citations of a reviewer's own files on that basis.

A bare occurrence of a marker inside quoted repository code or diff hunks is a
flag only; confirmation is the verifier's job, on diff-independent evidence:
(1) a command or reference that reaches other refs (`git log --all`,
`git log main`, `git show main:`, any `git diff`/`git log` naming
`main`/`master` directly or in a range, `origin/...`, `refs/...`, `rev-list`,
`reflog`, `for-each-ref`, `show-ref`, `cat-file`, `fsck`, `lost-found`,
`worktree list`, or a `git log`/`git show` of a sha obtained from one of
those); (2) a verbatim >= 12-word PROSE sentence from the ground-truth
rationale (code fragments the rationale quotes from the diff never count), or
naming `inject.diff`/`case.json`, a sibling run artifact or the corpus
directory as something it read; (3) the review cites "the commit message" (or
`git log -1`/`git show HEAD` output) as its source for the defect. The `../`
entry never confirms on its own.

### Marker list

<!-- LEAK_RULES:begin -->
Literal markers (any code span or fenced block containing one):

- `--all`
- `--branches`
- `--remotes`
- `rev-list`
- `reflog`
- `for-each-ref`
- `show-ref`
- `cat-file`
- `fsck`
- `lost-found`
- `--unreachable`
- `--dangling`
- `worktree list`
- `describe`
- `name-rev`
- `--walk-reflogs`
- `--grep`
- `whatchanged`
- `@^`
- `@~`
- `@{`
- `porcelain=v2`
- `branch.oid`
- `ls-remote`
- `gitdir`
- `campaign-`
- `format-patch`
- `shortlog`
- `blame`
- `cherry`
- `range-diff`
- `FETCH_HEAD`
- `ORIG_HEAD`
- `MERGE_HEAD`
- `REBASE_HEAD`
- `main-worktree/`
- `worktrees/`
- `${PWD`
- `$PWD/..`
- `dirname`
- `realpath`
- `-maxdepth`
- ` -r `
- `-rl`
- `-rn`
- `--include=*.json`
- `=git`
- `alias git`
- `codex-home-`
- `CODEX_HOME`
- `/sessions/`
- `rollout-`
- `origin/`
- `refs/`
- `git log main`
- `git show main`
- `git show HEAD`
- `git log -1`
- `HEAD^`
- `HEAD~`
- `rev-parse`
- `--format`
- `--pretty`
- `../`
- `cd ..`
- `tools/reviewer-eval/`
- `DECISION_RULE`
- `GRADING_RUBRIC`
- `ground_truth`
- `inject.diff`
- `case.json`
- `commit_message`
- `commit message`
- `eval: re-introduce`

Regex markers (source, then the plain-language rule to apply):

- `MAIN_REF_RE` = `\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+\S+(?=[^\n]*(?<![\w-])(?:(?:refs/)?(?:heads|remotes/[^\s/]+)/)?(main|master)(?![\w/-]|\.\w))` - any git command whose line names the main/master ref (also in ranges like HEAD..main or main:path)
- `BARE_SHOW_RE` = `\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+show\b` - any `git show` (with or without arguments)
- `GIT_LOG_RE` = `\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+log\b` - any `git log` (bare, ranged, sha-pinned, --oneline, with global options)
- `REF_LIST_RE` = `\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+(tag|branch)\b` - `git tag` or `git branch` as the verb (any options)
- `DOTDOT_RE` = `(?<![^\s'"=])\.\.(?![^\s'"|;&)])` - a bare `..` path argument (not a base..head range, not ../x)
- `GIT_DIR_RE` = `\.git\b` - a `.git` pointer file or anything under a `.git/` directory
- `TAG_REF_RE` = `\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+\S+(?=[^\n]*(?<![\w-])(?:(?:refs/)?tags/)?v\d+\.\d+\.\d+(?![\w.]))` - any git command whose line names a vN.N.N release tag (also tags/ and refs/tags/ forms)
<!-- LEAK_RULES:end -->

## Output format (JSON)

```json
{
  "bundle_id": "<copied from the bundle>",
  "rows": [
    {"case": "<case id>", "bug_id": "<bug id as listed>", "arm": "<M label>",
     "repeat": 0, "verdict": "caught | partial | missed",
     "evidence": "<verbatim quote; required when caught>",
     "leak": "none", "grader_flag": "none | flag",
     "leak_evidence": "<verbatim triggering line when flagged, else empty>",
     "notes": "<hallucination / unverified external-tool claims, else empty>"}
  ],
  "negative_assessments": [
    {"case": "<negative-control case id>", "arm": "<M label>", "repeat": 0,
     "findings": [{"severity": "P0 | P1 | P2 | P3", "summary": "<one line>",
                   "known_topic": "<topic name or empty>"}],
     "leak": "none", "grader_flag": "none | flag",
     "leak_evidence": "", "notes": ""}
  ]
}
```

One `rows` entry per (case, listed bug, arm label, repeat) - the grid must be
complete, a missing row is a validation error, never a miss. One
`negative_assessments` entry per (negative-control case, arm label, repeat),
with `findings: []` for a clean read (an omitted cell is a validation error,
never zero FPs). Graders always write `leak: "none"`; the verifier sets
`confirmed` where the evidence supports it. `leak_evidence` is non-empty
exactly when `grader_flag` is `flag`.
