#!/usr/bin/env python3
"""Prepare and validate every dynamic value ``/submit-pr`` feeds to the shell.

Why this exists
---------------
``/submit-pr`` builds a branch, syncs against a base, pushes, and opens a PR. The
title, branch name, and base branch are user- or generator-influenced text. When any
of them reaches a shell — even as ``VAR="…"`` — a backtick or ``$(...)`` executes.
git *accepts* both in a ref name (``git check-ref-format --branch 'x-`whoami`'``
passes), so an ordinary title like ``Fix `safe_inference` guard`` is a live payload.

The safe boundary is therefore **not** "sanitise before building the command" — it is
"never let the raw value touch a shell at all". So the untrusted values arrive here as
**files** (the caller writes them with the Write tool, which never invokes a shell)
and this script reads their contents with Python I/O. Nothing untrusted is ever an
argv element or a shell assignment.

This module:

- **never** evaluates input, interpolates it into a shell, or touches the network;
- reads the raw title / explicit branch / explicit base from files;
- reduces branch and base to a form containing no shell metacharacters *and* valid as
  a git ref — generated names by whitelist sanitisation + ref normalisation, explicit
  names by *rejection* if unsafe (never a silent rewrite);
- honours an already-checked-out feature branch instead of a title-derived name;
- resolves fork-vs-direct context from read-only ``git remote`` queries.

Outputs are written one value per file under ``--scratch``; the caller reads each with
``"$(cat …)"``. Every emitted branch/base value matches ``[A-Za-z0-9._/-]+`` and passes
``git check-ref-format``, so the caller may use them in git/gh commands safely, quoted.

When the title file is empty the script **generates a fallback title from base-free
git commands** (the last commit subject) rather than erroring — so the no-title
``/submit-pr`` invocation works, and the caller never needs to run a base-referencing
command like ``git log <base>..HEAD`` in prose, which would put a raw, unvalidated
base on a shell command line.

Exit non-zero with a message on any unsafe or malformed explicit input; the caller
must abort rather than proceed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# A ref name is shell-safe iff it contains only these characters. STRICTER than git,
# which permits backticks, ``$``, ``(``/``)`` and more — all shell-dangerous. Uppercase
# is allowed: it has no shell semantics, and real branches like ``Release/4.0`` or bases
# like ``V4`` must not be rejected. Generated names are still lowercased by
# :func:`sanitize_branch_portion`; this wider set only affects explicit/current refs.
_SAFE_REF = re.compile(r"[A-Za-z0-9._/-]+")

# For a *title-derived branch portion* (before the type prefix), '/' is also excluded
# — the only slash in the final name is the one we prepend.
_UNSAFE_PORTION_CHAR = re.compile(r"[^a-z0-9._-]")

_BRANCH_PREFIXES = ("feature/", "fix/", "refactor/", "docs/")
_MAX_PORTION = 50


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — the unit-test surface
# ---------------------------------------------------------------------------


def normalize_ref_portion(portion: str) -> str:
    """Fix git-invalid ref forms in an already char-whitelisted portion.

    ``git check-ref-format`` rejects ``..``, a component that starts or ends with
    ``.``, and a trailing ``.lock``. A title like ``v1.2..final`` or ``foo.lock``
    survives character-whitelisting but would make ``git checkout -b`` fail.
    """
    portion = re.sub(r"\.{2,}", ".", portion)  # no ".."
    portion = re.sub(r"\.lock$", "", portion)  # no trailing ".lock"
    portion = portion.strip(".-_")  # no leading/trailing . - _
    return portion


def sanitize_branch_portion(text: str) -> str:
    """Reduce free text (e.g. a PR title) to a safe branch portion, no prefix.

    Lowercase; spaces to hyphens; every character outside ``[a-z0-9._-]`` — including
    ``/`` — becomes a hyphen; collapse hyphen runs and underscore runs *separately* so
    a lone underscore survives (``safe_inference`` stays intact); normalise git-invalid
    dot forms; trim and truncate. The result always matches ``_SAFE_REF``.
    """
    s = text.lower().replace(" ", "-")
    s = _UNSAFE_PORTION_CHAR.sub("-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"_{2,}", "_", s)
    s = normalize_ref_portion(s)
    return s[:_MAX_PORTION].strip(".-_")


def is_shell_safe_ref(name: str) -> bool:
    """True iff ``name`` contains only whitelist characters (no metacharacters)."""
    return bool(name) and _SAFE_REF.fullmatch(name) is not None


def owner_repo(url: str) -> "str | None":
    """Extract ``owner/repo`` from any git remote URL, portably.

    Handles ``git@host:owner/repo.git``, ``https://host/owner/repo(.git)``, and
    ``ssh://git@host/owner/repo.git``. Returns ``None`` if it cannot be parsed.
    """
    # Strip trailing slash(es) FIRST, then the optional ``.git`` — a bare
    # ``…/owner/repo/`` (no ``.git``) must still parse.
    u = re.sub(r"\.git$", "", url.strip().rstrip("/"))
    m = re.search(r"[/:]([^/]+/[^/]+)$", u)
    return m.group(1) if m else None


def build_head_ref(is_fork: bool, fork_owner: "str | None", branch: str) -> str:
    """The ``--head`` value for ``gh pr create``.

    Direct: just the branch. Fork: ``owner:branch`` built by concatenation, never
    interpolation. Assumes ``branch`` is already validated safe.

    In fork mode a missing ``fork_owner`` is a hard error, not a fall back to the
    unqualified branch — ``gh pr create --repo <upstream> --head <branch>`` would then
    pick a same-named branch in the *base* repo instead of the fork's.
    """
    if is_fork:
        if not fork_owner:
            raise ValueError(
                "Fork workflow but could not parse the fork owner from origin's URL; "
                "refusing to open a PR with an unqualified head ref."
            )
        return f"{fork_owner}:{branch}"
    return branch


def resolve_base(explicit: "str | None", default: str = "main") -> str:
    """Return the base branch, or raise ``ValueError`` if an explicit base is unsafe.

    The default applies only when no base was supplied — an explicit-but-invalid base
    is an error, never a silent fall back to ``main``.
    """
    if explicit is None:
        return default
    if not is_shell_safe_ref(explicit):
        raise ValueError(f"Refusing unsafe base branch {explicit!r}: only [A-Za-z0-9._/-] allowed.")
    if not _git_ref_ok(explicit):
        raise ValueError(f"Base branch {explicit!r} is not a valid git ref.")
    return explicit


def resolve_branch(
    explicit: "str | None",
    current_branch: "str | None",
    base: str,
    title: str,
    change_type: str,
) -> str:
    """Decide the branch name to use, or raise ``ValueError``.

    Precedence:

    1. **Explicit ``--branch``** — honoured only if shell-safe *and* a valid git ref,
       else rejected (never silently rewritten). If it conflicts with a different
       already-checked-out feature branch, that is an error the user must resolve.
    2. **An existing feature branch** (``current_branch`` set and not the base) — used
       verbatim, so the command pushes and PRs the branch you are actually on rather
       than a title-derived name. Rejected if it somehow carries unsafe characters.
    3. **Generated** from the title: sanitised, ref-normalised, prefixed by change
       type, and required to pass ``git check-ref-format`` (falling back to a safe
       ``<prefix>change`` if the title reduces to nothing valid).
    """
    on_feature = bool(current_branch) and current_branch != base

    if explicit is not None:
        if not is_shell_safe_ref(explicit):
            raise ValueError(
                f"Refusing unsafe branch name {explicit!r}: only [A-Za-z0-9._/-] allowed."
            )
        if not _git_ref_ok(explicit):
            raise ValueError(f"Branch name {explicit!r} is not a valid git ref.")
        if on_feature and current_branch != explicit:
            raise ValueError(
                f"On feature branch {current_branch!r} but --branch {explicit!r} was "
                f"given. Switch branch or drop --branch; refusing to mix them."
            )
        return explicit

    if on_feature:
        assert current_branch is not None
        if not is_shell_safe_ref(current_branch):
            raise ValueError(f"Current branch {current_branch!r} has unsafe characters; rename it.")
        return current_branch

    prefix = change_type if change_type.endswith("/") else f"{change_type}/"
    if prefix not in _BRANCH_PREFIXES:
        prefix = "feature/"
    portion = sanitize_branch_portion(title) or "change"
    candidate = f"{prefix}{portion}"
    if not _git_ref_ok(candidate):
        candidate = f"{prefix}change"
    return candidate


# ---------------------------------------------------------------------------
# Thin git wiring (read-only) and file I/O
# ---------------------------------------------------------------------------


def _git(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _git_ref_ok(name: str) -> bool:
    return _git("check-ref-format", "--branch", name).returncode == 0


def _remote_url(remote: str) -> "str | None":
    r = _git("remote", "get-url", remote)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def generate_fallback_title() -> str:
    """A default PR title from **base-free** git state, used when none was supplied.

    Deliberately references no base branch: the alternative — the caller running
    ``git log <base>..HEAD`` in prose to build a title — would place a raw, unvalidated
    ``--base`` on a shell command line before validation, reopening the injection
    boundary.

    The last commit subject is only meaningful when the tree is **clean** (the commit
    *is* the change). In ``/submit-pr``'s normal flow the title is resolved *before*
    the commit, so the tree is dirty and ``git log -1`` would return the previous,
    unrelated commit — use a neutral title in that case.
    """
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    if dirty:
        return "Update working tree changes"
    r = _git("log", "-1", "--format=%s")
    subject = r.stdout.strip() if r.returncode == 0 else ""
    return subject or "Update"


def _read_file(path: "str | None") -> "str | None":
    """Read a value file's content, or None if the path is absent/empty/missing.

    Content is read as data — never parsed by a shell — so any payload is inert.
    """
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    return text or None


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Prepare safe PR-creation values.")
    # Untrusted values arrive as FILES (written by the caller's Write tool), never as
    # argv strings — so a backtick/$() in a title cannot execute reaching this process.
    p.add_argument(
        "--title-file",
        default=None,
        help="File holding the raw PR title; if absent/empty, a base-free title is generated",
    )
    p.add_argument("--branch-file", default=None, help="File holding an explicit branch")
    p.add_argument("--base-file", default=None, help="File holding an explicit base")
    p.add_argument(
        "--current-branch",
        default="",
        help="Output of `git branch --show-current` (trusted git state; may be empty)",
    )
    p.add_argument(
        "--change-type",
        default="feature",
        help="Prefix for a generated branch (feature|fix|refactor|docs)",
    )
    # --draft is a trusted boolean (flag presence), not untrusted text, so it is a
    # normal argv flag. Emitting it as a value file keeps the caller's re-read pattern
    # uniform, so a draft request cannot be silently dropped on the way to `gh`.
    p.add_argument("--draft", action="store_true", help="Mark the PR as a draft")
    p.add_argument("--scratch", required=True, help="Directory for output value files")
    args = p.parse_args(argv)

    os.makedirs(args.scratch, exist_ok=True)

    title = _read_file(args.title_file)
    if title is None:
        # No title supplied: generate one from base-free git state rather than error,
        # so `/submit-pr` with no title works and prose never runs a base-referencing
        # title command. (title-file is optional-in-effect; the value is still opaque.)
        title = generate_fallback_title()
    explicit_branch = _read_file(args.branch_file)
    explicit_base = _read_file(args.base_file)
    current_branch = args.current_branch.strip() or None

    try:
        base = resolve_base(explicit_base)
        branch = resolve_branch(explicit_branch, current_branch, base, title, args.change_type)

        upstream = _remote_url("upstream")
        is_fork = upstream is not None
        if is_fork:
            # Fail CLOSED: in fork mode both the upstream repo and the fork owner must
            # parse, or we would target the wrong repository / an unqualified head.
            target_repo = owner_repo(upstream)
            if not target_repo:
                raise ValueError(
                    f"Fork workflow but could not parse owner/repo from upstream URL "
                    f"{upstream!r}."
                )
            origin = _remote_url("origin")
            origin_or = owner_repo(origin) if origin else None
            fork_owner = origin_or.split("/")[0] if origin_or else None
            head_ref = build_head_ref(True, fork_owner, branch)  # raises if no fork_owner
        else:
            target_repo = ""
            head_ref = build_head_ref(False, None, branch)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # One value per file; caller reads each with "$(cat …)". Title is copied to a
    # canonical name so the caller passes --title "$(cat …/pr-title.txt)".
    _write(os.path.join(args.scratch, "pr-title.txt"), title)
    _write(os.path.join(args.scratch, "pr-base.txt"), base)
    _write(os.path.join(args.scratch, "pr-branch.txt"), branch)
    _write(os.path.join(args.scratch, "pr-headref.txt"), head_ref)
    _write(os.path.join(args.scratch, "pr-target-repo.txt"), target_repo or "")
    _write(os.path.join(args.scratch, "pr-is-fork.txt"), "true" if is_fork else "false")
    _write(os.path.join(args.scratch, "pr-draft.txt"), "true" if args.draft else "false")

    print(f"base={base}")
    print(f"branch={branch}")
    print(f"is_fork={'true' if is_fork else 'false'}")
    print(f"head_ref={head_ref}")
    print(f"target_repo={target_repo or ''}")
    print(f"draft={'true' if args.draft else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
