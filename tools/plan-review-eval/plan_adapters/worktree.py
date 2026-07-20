"""Materialize a plan case's repo state in a throwaway detached git worktree.

A plan case is reviewed against the repository AS IT WAS when the plan was
written (``fixture.base_sha``), not today's HEAD — otherwise the reviewer
produces spurious "path doesn't exist" findings against files that were added
or moved later. Unlike the reviewer-eval worktree adapter, there is nothing to
patch or revert: a plan is a text document describing future changes, so the
only fixture kind is a detached checkout at ``base_sha``.

Worktrees live under ``runs/.worktrees/`` (gitignored). A materialization
failure raises ``MaterializeError``; the runner turns that into an INFRA_ERROR
RunResult — never a missed catch.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass


class MaterializeError(RuntimeError):
    """A case could not be faithfully materialized."""


@dataclass
class Materialized:
    worktree_dir: str
    base_sha: str


def _git(repo: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


def _resolve(repo: str, rev: str) -> str:
    cp = _git(repo, ["rev-parse", "--verify", f"{rev}^{{commit}}"], check=False)
    if cp.returncode != 0:
        raise MaterializeError(f"cannot resolve revision {rev!r}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def _worktree_leaf(key: str) -> str:
    """Collision-resistant, path-safe leaf name (digest, not a lossy slug) so
    distinct keys never alias one checkout and reserved/traversal ids can never
    escape ``worktrees_root`` — the leaf is pure ``[A-Za-z0-9-]``."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-")[:32]
    return f"{slug}-{digest}" if slug else digest


def materialize(
    case_id: str,
    fixture: dict,
    repo_root: str,
    worktrees_root: str,
    worktree_key: str | None = None,
) -> Materialized:
    """Create a detached worktree at the case's ``base_sha``.

    ``worktree_key`` MUST be unique per concurrently-running invocation
    (parallel arms share a case_id but need distinct checkouts); it defaults to
    ``case_id`` for sequential callers (verify-corpus).
    """
    kind = fixture.get("kind", "plan_at_sha")
    if kind != "plan_at_sha":
        raise MaterializeError(
            f"{case_id}: unsupported fixture kind {kind!r} — the only kind this "
            f"adapter materializes is 'plan_at_sha' (a detached checkout at "
            f"base_sha); refusing to silently guess a repository state."
        )
    base = fixture.get("base_sha")
    if not base:
        raise MaterializeError(f"{case_id}: fixture missing base_sha")
    if _git(repo_root, ["cat-file", "-e", f"{base}^{{commit}}"], check=False).returncode != 0:
        raise MaterializeError(
            f"{case_id}: commit {base} not present in repo {repo_root}; fetch it first."
        )
    base = _resolve(repo_root, base)

    os.makedirs(worktrees_root, exist_ok=True)
    wt = os.path.join(worktrees_root, _worktree_leaf(worktree_key or case_id))
    if os.path.exists(wt):
        cleanup(wt, repo_root, worktrees_root)

    cp = _git(repo_root, ["worktree", "add", "--detach", wt, base], check=False)
    if cp.returncode != 0:
        raise MaterializeError(f"{case_id}: worktree add failed: {cp.stderr.strip()}")
    return Materialized(worktree_dir=wt, base_sha=base)


def cleanup(worktree_dir: str, repo_root: str, worktrees_root: str | None = None) -> None:
    """Remove a worktree (best-effort; prune dangling admin files).

    If the path survived (orphaned dir from an interrupted run), force-remove
    it — with two containment guards, BOTH applied BEFORE any git command
    (git worktree remove itself follows a symlinked leaf):

    * a leaf that is itself a SYMLINK is unlinked without following it (a leaf
      pointing at an external ``.worktrees/victim`` must never be recursed into
      — ``realpath`` alone would resolve to the external target, whose parent
      is conveniently named ``.worktrees``, and delete it);
    * the recursive delete requires the canonical target to be a STRICT child
      of the canonical TRUSTED root (``worktrees_root`` when provided; the
      leaf's parent must at minimum be named ``.worktrees`` otherwise).
    """
    # Containment BEFORE any git command: `git worktree remove --force` itself
    # resolves a symlinked leaf and would remove a REGISTERED external worktree
    # — so the symlink/containment guards must run first, not after.
    try:
        if os.path.islink(worktree_dir):
            os.unlink(worktree_dir)  # never follow a symlinked leaf
            _git(repo_root, ["worktree", "prune"], check=False)
            return
    except OSError:
        return
    real = os.path.realpath(worktree_dir)
    parent = os.path.dirname(real)
    if worktrees_root is not None:
        root = os.path.realpath(worktrees_root)
        try:
            contained = os.path.commonpath([root, real]) == root and real != root
        except ValueError:  # different drives / mixed abs-rel
            contained = False
        if not contained:
            return
    elif os.path.basename(parent) != ".worktrees" or real == parent:
        return
    _git(repo_root, ["worktree", "remove", "--force", worktree_dir], check=False)
    _git(repo_root, ["worktree", "prune"], check=False)
    if os.path.isdir(real):
        shutil.rmtree(real, ignore_errors=True)


__all__ = ["materialize", "cleanup", "Materialized", "MaterializeError"]
