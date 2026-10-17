"""Post-unblinding session-log audit for the reviewer-eval harness.

INFORMATIONAL and OVER-MATCHING BY DESIGN (DECISION_RULE.md, user decisions 4
and 6): it never flips or contests a gate. For every manifest run artifact it
takes ``usage["session_commands"]`` (the shell command lines codex executed,
parsed by ``openai_review.call_codex`` from codex's stderr), strips the run's
OWN worktree path (which codex received as ``--cd`` and echoes everywhere),
and scans the remainder for the ONE authoritative LEAK MARKER LIST:

* ``LEAK_MARKERS`` - literal substrings (git verbs/flags that reach other refs
  or the shared object DB, corpus/answer-key file names, cwd-derivation and
  recursive-search idioms, codex-home paths);
* seven regexes - ``MAIN_REF_RE``, ``BARE_SHOW_RE``, ``GIT_LOG_RE``,
  ``REF_LIST_RE``, ``DOTDOT_RE``, ``GIT_DIR_RE``, ``TAG_REF_RE`` - each
  tolerating git's global options between ``git`` and the subcommand;
* computed, per-campaign markers (never part of ``LEAK_RULES``): the enclosing
  checkout (``repo_root``), the checkout's branch name, the git common dir and
  its parent checkout, every other worktree of the repository, every other
  recorded ref name (git-scoped), and the isolated codex home.

``LEAK_RULES`` (literals + the seven regex sources, in order) is what
``GRADING_RUBRIC.md`` renders for graders; the rubric-hygiene test asserts the
rendering equals this list exactly (one source, two renderings, no third copy).

Coverage per artifact is exactly one of THREE classes (the single definition,
shared with DECISION_RULE.md and every test): ``covered``, ``no_commands``,
``unaudited``. Both ``covered`` and ``no_commands`` require
``codex_home_isolated == True`` and ``session_other_events == 0``; the audit
is VACUOUS when no artifact is covered or no_commands.

Python 3.9 compatible (runs on the CI floor leg).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

# Common prefix: `git` followed by any run of global options (dash options with
# or without `=value`; the argument-taking globals -C/-c/--git-dir/--work-tree/
# --namespace/--attr-source/--config-env also skip a split-form argument), then
# whitespace. Generic on purpose: an enumerated list was a false-negative source
# (`git --no-replace-objects show HEAD` matched nothing).
G = (
    r"\bgit([ \t]+(-[Cc][ \t]*\S+|--(git-dir|work-tree|namespace|attr-source|config-env)"
    r"(=|[ \t]+)\S+|--?[A-Za-z][\w-]*(=\S*)?))*[ \t]+"
)

# Any git command whose SAME line names the main/master ref as a token (ranges
# and colon forms included: `HEAD..main`, `main...HEAD`, `main:path`); optional
# heads/, refs/heads/ and remotes/<r>/ prefixes; `main.py`, `src/main/`,
# `domain.py`, `remain` do not match.
MAIN_REF_RE = re.compile(
    G
    + r"\S+(?=[^\n]*(?<![\w-])(?:(?:refs/)?(?:heads|remotes/[^\s/]+)/)?(main|master)(?![\w/-]|\.\w))"
)
# Any `git show` (reveals the synthetic HEAD's commit message).
BARE_SHOW_RE = re.compile(G + r"show\b")
# ANY `git log` form: every one prints commit subjects, and the synthetic HEAD's
# subject names the bug.
GIT_LOG_RE = re.compile(G + r"log\b")
# Ref-listing verbs under global options (`git -C x branch -a`, `git --no-pager tag`).
REF_LIST_RE = re.compile(G + r"(tag|branch)\b")
# A BARE `..` path token (`ls ..`, `rg x ..`, `cd ..; ls`), NOT the `..` inside
# a `base..head` range or `../x` (the `../` literal's job).
DOTDOT_RE = re.compile(r"""(?<![^\s'"=])\.\.(?![^\s'"|;&)])""")
# The worktree's `.git` pointer file and anything under a `.git/` dir; `.github`,
# `.gitignore` and `.gitattributes` do not match (\b needs a non-word char).
GIT_DIR_RE = re.compile(r"\.git\b")
# Any git command whose SAME line names a semver release tag (`v3.12.0`,
# `tags/v3.12.0`, `refs/tags/v3.12.0`), including `HEAD..v3.12.0` ranges.
TAG_REF_RE = re.compile(G + r"\S+(?=[^\n]*(?<![\w-])(?:(?:refs/)?tags/)?v\d+\.\d+\.\d+(?![\w.]))")

REGEX_RULES = (
    ("MAIN_REF_RE", MAIN_REF_RE),
    ("BARE_SHOW_RE", BARE_SHOW_RE),
    ("GIT_LOG_RE", GIT_LOG_RE),
    ("REF_LIST_RE", REF_LIST_RE),
    ("DOTDOT_RE", DOTDOT_RE),
    ("GIT_DIR_RE", GIT_DIR_RE),
    ("TAG_REF_RE", TAG_REF_RE),
)

# Plain-language glosses rendered next to each regex source in the rubric.
REGEX_GLOSS = {
    "MAIN_REF_RE": "any git command whose line names the main/master ref (also in ranges like HEAD..main or main:path)",
    "BARE_SHOW_RE": "any `git show` (with or without arguments)",
    "GIT_LOG_RE": "any `git log` (bare, ranged, sha-pinned, --oneline, with global options)",
    "REF_LIST_RE": "`git tag` or `git branch` as the verb (any options)",
    "DOTDOT_RE": "a bare `..` path argument (not a base..head range, not ../x)",
    "GIT_DIR_RE": "a `.git` pointer file or anything under a `.git/` directory",
    "TAG_REF_RE": "any git command whose line names a vN.N.N release tag (also tags/ and refs/tags/ forms)",
}

# The literal markers, complete and in order (DECISION_RULE.md step-1 list).
LEAK_MARKERS: "tuple[str, ...]" = (
    "--all",
    "--branches",
    "--remotes",
    "rev-list",
    "reflog",
    "for-each-ref",
    "show-ref",
    "cat-file",
    "fsck",
    "lost-found",
    "--unreachable",
    "--dangling",
    "worktree list",
    "describe",
    "name-rev",
    "--walk-reflogs",
    "--grep",
    "whatchanged",
    "@^",
    "@~",
    "@{",
    "porcelain=v2",
    "branch.oid",
    "ls-remote",
    "gitdir",
    "campaign-",
    "format-patch",
    "shortlog",
    "blame",
    "cherry",
    "range-diff",
    "FETCH_HEAD",
    "ORIG_HEAD",
    "MERGE_HEAD",
    "REBASE_HEAD",
    "main-worktree/",
    "worktrees/",
    "${PWD",
    "$PWD/..",
    "dirname",
    "realpath",
    "-maxdepth",
    " -r ",
    "-rl",
    "-rn",
    "--include=*.json",
    "=git",
    "alias git",
    "codex-home-",
    "CODEX_HOME",
    "/sessions/",
    "rollout-",
    "origin/",
    "refs/",
    "git log main",
    "git show main",
    "git show HEAD",
    "git log -1",
    "HEAD^",
    "HEAD~",
    "rev-parse",
    "--format",
    "--pretty",
    "../",
    "cd ..",
    "tools/reviewer-eval/",
    "DECISION_RULE",
    "GRADING_RUBRIC",
    "ground_truth",
    "inject.diff",
    "case.json",
    "commit_message",
    "commit message",
    "eval: re-introduce",
)

# ONE source, two renderings: the rubric renders exactly this (literals verbatim,
# then each regex as (name, source)); the hygiene test compares against it.
LEAK_RULES: "tuple[tuple[str, str], ...]" = tuple(
    [("literal", m) for m in LEAK_MARKERS] + [(name, rx.pattern) for name, rx in REGEX_RULES]
)

# Rule NAMES whose hits are routine orientation (reporting-only; a hit is
# orientation-only iff ALL its markers are in this set).
ORIENTATION_MARKERS = frozenset(
    {
        "GIT_LOG_RE",
        "BARE_SHOW_RE",
        "git log -1",
        "git show HEAD",
        "HEAD^",
        "HEAD~",
        "@^",
        "@~",
        "@{",
        "rev-parse",
        "--format",
        "--pretty",
    }
)

# A `\` + newline continuation (with optional indentation) is one logical line.
_CONTINUATION_RE = re.compile(r"[ \t]*\\\n[ \t]*")
_SEMVER_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_PATH_BOUNDARY = r"(?=/|\s|['\"]|$)"
WT_PLACEHOLDER = "<WT>"


def artifacts_sha(subdir_path: str, run_ids: "list[str]") -> str:
    """sha256 over the sorted (run_id, sha256 of that artifact file's bytes)
    pairs, where the file for a run_id is ``<subdir>/<run_id>.json`` (the
    ``RunStore._path`` layout). A missing file is a refusal (FileNotFoundError),
    never a skip, so ``compare``, ``audit`` and ``verdict`` seal the same files.
    """
    pairs = []
    for run_id in sorted(set(run_ids)):
        path = os.path.join(subdir_path, f"{run_id}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"artifact missing for run_id {run_id!r}: {path}")
        with open(path, "rb") as fh:
            pairs.append((run_id, hashlib.sha256(fh.read()).hexdigest()))
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()


def _home_forms(path: str) -> "list[str]":
    """``~/`` and ``$HOME/`` spellings of ``path`` when it lies under the home dir."""
    home = os.path.expanduser("~")
    out = []
    if path and home and (path == home or path.startswith(home + os.sep)):
        rel = path[len(home) :]
        out.append("~" + rel)
        out.append("$HOME" + rel)
    return out


def strip_own_path(entry: str, worktree_dir: str, repo_root: str) -> str:
    """Replace every spelling of the run's OWN worktree path with ``<WT>``, in
    this FIXED order (longer forms first; the relative form is a suffix of the
    absolute one and the leaf a suffix of both): realpath, recorded absolute,
    repo-relative (only when under ``repo_root``), ``~/`` and ``$HOME/`` forms,
    then the bare leaf LAST. Joins backslash-newline continuations first.
    """
    text = _CONTINUATION_RE.sub(" ", entry)
    if not worktree_dir:
        return text
    forms: "list[str]" = []
    real = os.path.realpath(worktree_dir)
    forms.append(real)
    forms.append(worktree_dir)
    root_real = os.path.realpath(repo_root) if repo_root else ""
    for base in (root_real, repo_root):
        if base and real.startswith(base + os.sep):
            forms.append(real[len(base) + 1 :])
        if base and worktree_dir.startswith(base + os.sep):
            forms.append(worktree_dir[len(base) + 1 :])
    forms += _home_forms(worktree_dir) + _home_forms(real)
    leaf = os.path.basename(worktree_dir.rstrip(os.sep))
    for form in forms:
        if form:
            text = text.replace(form, WT_PLACEHOLDER)
    if leaf:
        text = text.replace(leaf, WT_PLACEHOLDER)
    return text


def _path_markers(
    repo_root: str,
    markers: dict,
) -> "list[tuple[str, str]]":
    """(name, spelling) pairs for every computed PATH marker, longest first."""
    out: "list[tuple[str, str]]" = []

    def add(name: str, path: Optional[str]) -> None:
        if not path:
            return
        spellings = {path, os.path.realpath(path)}
        for p in list(spellings):
            spellings.update(_home_forms(p))
        for sp in spellings:
            out.append((name, sp))

    add("repo_root", repo_root)
    common = markers.get("repo_common_dir") or ""
    add("repo_common_dir", common)
    if common:
        add("repo_main_checkout", os.path.dirname(common.rstrip(os.sep)))
    for wt in markers.get("repo_worktree_paths") or []:
        add("repo_worktree_path", wt)
    add("codex_home_path", markers.get("codex_home_path"))
    # Longest spelling first so a read under repo_root is attributed to
    # repo_root, not also to the (shorter) primary checkout it sits beside.
    out.sort(key=lambda t: len(t[1]), reverse=True)
    return out


def _ref_name_regex(ref_names: "list[str]", branch: str) -> "Optional[re.Pattern]":
    """Git-scoped regex over the recorded ref names minus main/master, the
    campaign branch, HEAD, bare remote names and semver tags (TAG_REF_RE's job)."""
    names = {n for n in ref_names if n and n != "HEAD"}
    remotes = {n.split("/", 1)[0] for n in names if "/" in n}
    names -= {"main", "master", branch}
    names = {n for n in names if not (n in remotes and "/" not in n)}
    names = {n for n in names if not _SEMVER_TAG_RE.match(n)}
    if not names:
        return None
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(G + r"\S+(?=[^\n]*(?<![\w/-])(" + alts + r")(?![\w/.-]|\.\w))")


def scan_entry(
    line: str,
    repo_root: str,
    markers: dict,
    path_markers: "Optional[list[tuple[str, str]]]" = None,
    ref_regex: "Optional[re.Pattern]" = None,
    literals: "tuple[str, ...]" = LEAK_MARKERS,
    regexes: "tuple[tuple[str, Any], ...]" = REGEX_RULES,
) -> "list[str]":
    """Marker names that hit one STRIPPED entry (deterministic order)."""
    hits: "list[str]" = []
    for lit in literals:
        if lit in line:
            hits.append(lit)
    for name, rx in regexes:
        if rx.search(line):
            hits.append(name)
    if path_markers is None:
        path_markers = _path_markers(repo_root, markers)
    taken: "list[tuple[int, int]]" = []
    for name, spelling in path_markers:
        for m in re.finditer(re.escape(spelling) + _PATH_BOUNDARY, line):
            span = (m.start(), m.end())
            if any(a < span[1] and span[0] < b for a, b in taken):
                continue  # a LONGER marker already owns this span
            taken.append(span)
            if name not in hits:
                hits.append(name)
    branch = markers.get("repo_branch") or ""
    if branch and re.search(
        G + r"\S+(?=[^\n]*(?<![\w/-])" + re.escape(branch) + r"(?![\w/.-]|\.\w))", line
    ):
        hits.append("repo_branch")
    if ref_regex is not None and ref_regex.search(line):
        hits.append("repo_ref_name")
    return hits


def coverage_class(usage: dict) -> str:
    """EXACTLY the three classes of DECISION_RULE.md; no defaults."""
    if not isinstance(usage, dict):
        return "unaudited"
    if usage.get("codex_home_isolated") is not True:
        return "unaudited"
    if usage.get("session_other_events") != 0:
        return "unaudited"
    cmds = usage.get("session_commands")
    start = usage.get("session_commands_start_lines")
    execs = usage.get("session_commands_exec_lines")
    truncated = usage.get("session_commands_truncated")
    if not isinstance(cmds, list) or not isinstance(start, int) or not isinstance(execs, int):
        return "unaudited"
    if truncated is not False:  # missing / None / 1 / True: truncation state unknown or set
        return "unaudited"
    if cmds:
        if any(c.endswith("[truncated]") for c in cmds):
            return "unaudited"
        if execs == start == len(cmds):
            return "covered"
        return "unaudited"
    if start == 0 and execs == 0:
        return "no_commands"
    return "unaudited"


def audit(
    runs: list,
    subdir_path: str,
    repo_root: str,
    manifest_markers: dict,
    run_ids_sha: str,
    artifacts_sha_value: str,
) -> dict:
    """Audit every (OK) run artifact; return the ``hits.json`` structure.

    ``manifest_markers`` = the manifest's ``{repo_branch, repo_common_dir,
    repo_worktree_paths, repo_ref_names, codex_home_path}``; each artifact's own
    ``usage`` values of the same keys are UNIONED in (a worktree that existed at
    ANY sample point is a marker). Refuses (ValueError) if the live artifacts no
    longer hash to ``artifacts_sha_value``.
    """
    run_ids = [getattr(r, "run_id", "") for r in runs]
    live = artifacts_sha(subdir_path, run_ids)
    if live != artifacts_sha_value:
        raise ValueError(
            f"artifacts changed since they were sealed: live artifacts_sha {live} != "
            f"recorded {artifacts_sha_value}"
        )
    branch = manifest_markers.get("repo_branch") or ""
    common = manifest_markers.get("repo_common_dir") or ""
    wts = set(manifest_markers.get("repo_worktree_paths") or [])
    refs = set(manifest_markers.get("repo_ref_names") or [])
    homes = set()
    if manifest_markers.get("codex_home_path"):
        homes.add(manifest_markers["codex_home_path"])
    for r in runs:
        u = getattr(r, "usage", None) or {}
        wts.update(u.get("repo_worktree_paths") or [])
        refs.update(u.get("repo_ref_names") or [])
        if u.get("codex_home_path"):
            homes.add(u["codex_home_path"])
        if not common and u.get("repo_common_dir"):
            common = u["repo_common_dir"]
    markers = {
        "repo_branch": branch,
        "repo_common_dir": common,
        "repo_worktree_paths": sorted(wts),
        "codex_home_path": None,
    }
    path_markers: "list[tuple[str, str]]" = []
    for home in sorted(homes):
        path_markers += _path_markers("", {"codex_home_path": home})
    path_markers += _path_markers(repo_root, markers)
    path_markers.sort(key=lambda t: len(t[1]), reverse=True)
    ref_regex = _ref_name_regex(sorted(refs), branch)

    hits: "list[dict]" = []
    unaudited: "list[dict]" = []
    no_commands: "list[dict]" = []
    counts = {"covered": 0, "no_commands": 0, "unaudited": 0, "total": 0}
    for r in sorted(runs, key=lambda x: (x.case_id, x.config_id, x.repeat_idx)):
        cell = {"case": r.case_id, "arm": r.config_id, "repeat": r.repeat_idx}
        u = getattr(r, "usage", None) or {}
        cls = coverage_class(u)
        counts[cls] += 1
        counts["total"] += 1
        if cls == "unaudited":
            unaudited.append(cell)
        elif cls == "no_commands":
            no_commands.append(cell)
        for entry in u.get("session_commands") or []:
            if not isinstance(entry, str):
                continue
            stripped = strip_own_path(entry, u.get("worktree_dir") or "", repo_root)
            found = scan_entry(stripped, repo_root, markers, path_markers, ref_regex)
            if found:
                hits.append(
                    {
                        **cell,
                        "markers": found,
                        "line": stripped,
                        "orientation_only": set(found) <= ORIENTATION_MARKERS,
                    }
                )
    vacuous = counts["covered"] == 0 and counts["no_commands"] == 0
    return {
        "run_ids_sha": run_ids_sha,
        "artifacts_sha": artifacts_sha_value,
        "branch": branch,
        "hits": hits,
        "unaudited": unaudited,
        "no_commands": no_commands,
        "coverage": counts,
        "vacuous": vacuous,
    }


def render_rules_for_rubric() -> str:
    """The rubric's LEAK? trigger list: literals verbatim, then each regex as its
    source plus a one-line gloss. ``GRADING_RUBRIC.md`` embeds exactly this text
    between its `<!-- LEAK_RULES:begin -->` / `<!-- LEAK_RULES:end -->` markers."""
    lines = ["Literal markers (any code span or fenced block containing one):", ""]
    for m in LEAK_MARKERS:
        lines.append(f"- `{m}`")
    lines += ["", "Regex markers (source, then the plain-language rule to apply):", ""]
    for name, rx in REGEX_RULES:
        lines.append(f"- `{name}` = `{rx.pattern}` - {REGEX_GLOSS[name]}")
    return "\n".join(lines) + "\n"


__all__ = [
    "G",
    "MAIN_REF_RE",
    "BARE_SHOW_RE",
    "GIT_LOG_RE",
    "REF_LIST_RE",
    "DOTDOT_RE",
    "GIT_DIR_RE",
    "TAG_REF_RE",
    "REGEX_RULES",
    "LEAK_MARKERS",
    "LEAK_RULES",
    "ORIENTATION_MARKERS",
    "WT_PLACEHOLDER",
    "artifacts_sha",
    "strip_own_path",
    "scan_entry",
    "coverage_class",
    "audit",
    "render_rules_for_rubric",
]
