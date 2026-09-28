"""Materialize a case's diff state in a throwaway detached git worktree.

Never touches the user's primary checkout or current branch. Worktrees live
under ``runs/.worktrees/`` (gitignored). Supports three fixture kinds:

  * ``git_range``     — checkout pinned ``head_sha``; diff is ``base_sha..head_sha``
                        (S2/S3/S4: real historical PR states).
  * ``stored_patch``  — checkout ``base_sha``; ``git apply`` a frozen ``inject.diff``;
                        commit locally; diff is ``base_sha..HEAD`` (S1, preferred —
                        survives HEAD drift, unlike a live revert).
  * ``git_revert``    — checkout ``base_sha``; ``git revert --no-commit`` a fix
                        commit; commit locally (S1 alternative; brittle on drift).

A materialization failure raises ``MaterializeError``; the runner turns that
into an INFRA_ERROR RunResult — NEVER a missed bug (plan: infra noise must not
trip a recall floor).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


class MaterializeError(RuntimeError):
    """A case could not be faithfully materialized."""


@dataclass
class Materialized:
    worktree_dir: str
    base_sha: str
    head_sha: str


def _git(repo: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


def _resolve(repo: str, rev: str) -> str:
    cp = _git(repo, ["rev-parse", "--verify", f"{rev}^{{commit}}"], check=False)
    if cp.returncode != 0:
        raise MaterializeError(f"cannot resolve revision {rev!r}: {cp.stderr.strip()}")
    return cp.stdout.strip()


def _worktree_leaf(key: str) -> str:
    """Collision-resistant, path-safe leaf directory name for a worktree key.

    A digest (NOT a lossy slug) so distinct keys like ``a/b`` and ``a_b`` never alias
    the same checkout (which would let one parallel job clobber another's), and
    reserved/traversal ids (``.`` / ``..`` / anything with separators) can never escape
    ``worktrees_root`` — the leaf is pure ``[A-Za-z0-9-]``. A short readable prefix
    aids debugging; the digest guarantees uniqueness even when prefixes collide.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-")[:32]
    return f"{slug}-{digest}" if slug else digest


def _ensure_present(repo: str, sha: str) -> None:
    if _git(repo, ["cat-file", "-e", f"{sha}^{{commit}}"], check=False).returncode != 0:
        raise MaterializeError(
            f"commit {sha} not present in repo {repo}; fetch it before materializing."
        )


def _resolve_patch_path(case_id: str, case_dir: Optional[str], patch_rel: Optional[str]) -> str:
    """Resolve ``fixture.patch`` within ``case_dir`` and reject anything that escapes
    it (an absolute path or a ``..`` traversal), so a case cannot make the eval read
    files outside its own directory."""
    if not patch_rel:
        raise MaterializeError(f"{case_id}: stored_patch missing 'patch'")
    if os.path.isabs(patch_rel):
        raise MaterializeError(
            f"{case_id}: patch path {patch_rel!r} must be relative to the case directory"
        )
    base = os.path.realpath(case_dir or ".")
    full = os.path.realpath(os.path.join(base, patch_rel))
    if full != base and not full.startswith(base + os.sep):
        raise MaterializeError(
            f"{case_id}: patch path {patch_rel!r} escapes its case directory {case_dir!r}"
        )
    return os.path.join(case_dir or "", patch_rel)


def materialize(
    case_id: str,
    fixture: dict,
    repo_root: str,
    worktrees_root: str,
    case_dir: Optional[str] = None,
    worktree_key: Optional[str] = None,
) -> Materialized:
    """Create the worktree for ``case_id`` and return its location + SHAs.

    ``case_dir`` is the directory holding the case's ``inject.diff`` (for
    stored_patch). ``worktrees_root`` is created if absent.

    ``worktree_key`` names the worktree directory; it MUST be unique per
    concurrently-running invocation. Parallel A/B runs of the same case share a
    ``case_id`` but need distinct worktrees (else one arm's setup removes the
    other's checkout mid-review). Callers pass e.g. ``"<case>.<config>.r<rep>"``;
    it defaults to ``case_id`` for sequential callers (verify-corpus).
    """
    kind = fixture.get("kind")
    base = fixture.get("base_sha")
    if not base:
        raise MaterializeError(f"{case_id}: fixture missing base_sha")
    # Resolve + contain the stored_patch path BEFORE any worktree work, so an
    # absolute/escaping path fails fast (no leaked worktree) and never reads outside
    # the case directory.
    patch_path = (
        _resolve_patch_path(case_id, case_dir, fixture.get("patch"))
        if kind == "stored_patch"
        else None
    )
    _ensure_present(repo_root, base)
    base = _resolve(repo_root, base)

    os.makedirs(worktrees_root, exist_ok=True)
    # Leaf is a path-safe digest of the key — never the raw (possibly traversal-laden
    # or colliding) case id, so a reserved id like ".." can't target worktrees_root's
    # parent and distinct keys never share a checkout.
    wt = os.path.join(worktrees_root, _worktree_leaf(worktree_key or case_id))
    # Clean any stale worktree from a previous crashed run.
    if os.path.exists(wt):
        cleanup(wt, repo_root, worktrees_root)

    if kind == "git_range":
        head = fixture.get("head_sha")
        if not head:
            raise MaterializeError(f"{case_id}: git_range fixture missing head_sha")
        _ensure_present(repo_root, head)
        head = _resolve(repo_root, head)
        cp = _git(repo_root, ["worktree", "add", "--detach", wt, head], check=False)
        if cp.returncode != 0:
            raise MaterializeError(f"{case_id}: worktree add failed: {cp.stderr.strip()}")
        return Materialized(worktree_dir=wt, base_sha=base, head_sha=head)

    if kind in ("stored_patch", "git_revert"):
        cp = _git(repo_root, ["worktree", "add", "--detach", wt, base], check=False)
        if cp.returncode != 0:
            raise MaterializeError(f"{case_id}: worktree add failed: {cp.stderr.strip()}")
        try:
            if kind == "stored_patch":
                # patch_path was resolved + contained above (fail-fast, pre-worktree);
                # it is non-None whenever kind == "stored_patch".
                assert patch_path is not None
                if not os.path.exists(patch_path):
                    raise MaterializeError(f"{case_id}: patch not found at {patch_path}")
                ap = _git(wt, ["apply", "--whitespace=nowarn", patch_path], check=False)
                if ap.returncode != 0:
                    raise MaterializeError(
                        f"{case_id}: git apply failed (HEAD likely drifted from the "
                        f"frozen patch): {ap.stderr.strip()}"
                    )
            else:  # git_revert
                rc = fixture.get("revert_commit")
                if not rc:
                    raise MaterializeError(f"{case_id}: git_revert missing revert_commit")
                _ensure_present(repo_root, rc)
                rv = _git(wt, ["revert", "--no-commit", rc], check=False)
                if rv.returncode != 0:
                    raise MaterializeError(
                        f"{case_id}: git revert failed (conflict on drift): " f"{rv.stderr.strip()}"
                    )
            add = _git(wt, ["add", "-A"], check=False)
            if add.returncode != 0:
                raise MaterializeError(f"{case_id}: git add failed: {add.stderr.strip()}")
            msg = fixture.get("commit_message", f"eval: inject bug for {case_id}")
            # Identity is provided inline so the eval never depends on global git config.
            commit = _git(
                wt,
                [
                    "-c",
                    "user.name=codex-eval",
                    "-c",
                    "user.email=codex-eval@local",
                    "commit",
                    "--no-verify",
                    "-m",
                    msg,
                ],
                check=False,
            )
            if commit.returncode != 0:
                raise MaterializeError(
                    f"{case_id}: commit failed (empty patch?): {commit.stderr.strip()}"
                )
            head = _resolve(wt, "HEAD")
            return Materialized(worktree_dir=wt, base_sha=base, head_sha=head)
        except MaterializeError:
            cleanup(wt, repo_root, worktrees_root)
            raise
        except Exception as exc:  # noqa: BLE001 - clean up + rewrap as a case-scoped error
            # Any other post-`worktree add` failure (e.g. a CalledProcessError from a
            # check=True git call) must NOT leak the detached worktree or crash
            # verify-corpus/run with a raw traceback — clean up and surface it as an
            # infra-level MaterializeError (-> INFRA_ERROR RunResult, never a missed bug).
            cleanup(wt, repo_root, worktrees_root)
            raise MaterializeError(f"{case_id}: materialization failed: {exc}") from exc

    raise MaterializeError(f"{case_id}: unknown fixture kind {kind!r}")


def cleanup(worktree_dir: str, repo_root: str, worktrees_root: str | None = None) -> None:
    """Remove a worktree (best-effort; prune dangling admin files).

    If the path SURVIVED (an interrupted/partial run can leave an orphaned dir that
    is no longer a registered worktree — `git worktree remove` then fails, and the
    next `git worktree add` to the same key would fail too), force-remove it — with
    two containment guards, BOTH applied BEFORE any git command (git worktree
    remove itself follows a symlinked leaf): a leaf that is itself a SYMLINK is unlinked without
    following it (a leaf pointing at an external ``.worktrees/victim`` must never be
    recursed into — realpath alone would resolve to the external target, whose
    parent is conveniently named ``.worktrees``, and delete it); and the recursive
    delete requires the canonical target to be a STRICT child of the canonical
    TRUSTED root (``worktrees_root`` when provided; the leaf's parent must at
    minimum be named ``.worktrees`` otherwise).
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
