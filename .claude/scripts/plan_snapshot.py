#!/usr/bin/env python3
"""Argv-safe snapshot/persist protocol for the plan-review hash gate.

The ExitPlanMode hook approves a plan only when its review file certifies the
SHA-256 of the plan's current bytes. For that certification to MEAN anything,
the review must have examined exactly those bytes — a guarantee prose
workflows repeatedly failed to provide (live-file reads race with concurrent
edits; basename-keyed snapshots collide across invocations; `~`/symlink path
forms drift between writers and the hook). This helper owns the whole
protocol in tested code; the command prose only invokes it.

Subcommands (all untrusted values arrive via FILES written with the Write
tool — never argv literals, never shell interpolation):

  snapshot --plan-path-file F
      Normalize + validate the plan path (leading `~` expanded as data;
      canonical realpath; charset whitelist), read the plan bytes ONCE, write
      them to an INVOCATION-UNIQUE snapshot under
      `$HOME/.claude/plans/.snapshots/` (atomic tmp+rename, never
      overwritten) together with a STATE file recording the canonical plan
      path, snapshot path, and the reviewed sha, and print JSON:
        {"state_path", "snapshot_path", "meta_path", "body_path",
         "plan_path", "plan_sha256", "review_path"}
      The review is conducted AGAINST THE SNAPSHOT. The single
      invocation-unique STATE token keys the whole rest of the protocol:
      the caller writes its meta/body to exactly the printed
      `meta_path`/`body_path` (derived from the state path), so two
      concurrent reviews can never cross-wire inputs. The caller must also
      confirm the printed `plan_path` is the plan it intended (a concurrent
      overwrite of the ingress file is thereby detected, not silently used).
      `review_path` lives in `$HOME/.claude/plans/` — exactly where the
      ExitPlanMode hook looks, keyed by the CANONICAL (realpath) basename.

  persist --state-file F
      Load the state; require the recorded snapshot to be a regular,
      non-symlink file under the owned snapshots dir whose bytes STILL hash
      to the RECORDED sha (the digest captured at snapshot time is consumed —
      a later rewrite of the snapshot can never be certified). Then re-read
      the live plan; if its bytes do not hash to that recorded sha, exit 3
      ("plan changed during review") and clean up — persisting either hash
      would certify content that was never reviewed-as-current. (Bytes equal
      to the snapshot mean certification is exact even if the file changed
      and changed back meanwhile.) Otherwise atomically write the review to
      the recorded `review_path` with frontmatter built from the meta JSON at
      `meta_path` (plan path + plan_sha256 come from the STATE, never the
      caller), clean up the snapshot/state/meta/body files, and print the
      review path.

  abort --state-file F
      Clean up an invocation that will not persist (review failed, user
      cancelled): delete the snapshot/state/meta/body files. Same containment
      as persist; never touches the plan or any review.

  check --plan-path-file F
      Read-only staleness probe: hash the live plan and compare against the
      plan_sha256 recorded in its review file. Prints JSON:
        {"plan_path", "plan_sha256", "review_path", "review_exists",
         "review_plan_sha256", "fresh"}
      (`fresh` is true iff the review exists and its recorded sha matches the
      live plan bytes.) Replaces any prose-side `shasum`/`grep` interpolation.

Exit codes: 0 ok · 2 invalid input (bad path/meta) · 3 plan changed during
review · 4 environment failure (unreadable/unwritable). Frontmatter is
emitted as plain single-line scalars per the hook's parsing contract; meta
values are sanitized to a single line so a hostile assessment string cannot
break the frontmatter block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from typing import NoReturn

_META_KEYS = ("reviewed_at", "assessment", "critical_count", "medium_count", "low_count", "flags")


def _fail(code: int, msg: str) -> NoReturn:
    print(f"plan_snapshot: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _read_value_file(path: str, what: str) -> str:
    """Read a single-value ingress file: exactly the content minus ONE trailing
    newline (the Write-tool convention). No blanket strip — silently rewriting
    a path that legitimately ends in whitespace would target the wrong file."""
    try:
        with open(path, encoding="utf-8") as fh:
            value = fh.read()
    except OSError as exc:
        _fail(2, f"cannot read {what} file {path}: {exc}")
    return value[:-1] if value.endswith("\n") else value


def _normalize_plan_path(raw: str) -> str:
    """Leading `~` expanded AS DATA, then canonical realpath.

    Any absolute path is accepted — spaces, Unicode, whatever the filesystem
    allows. This helper NEVER passes the path through a shell (file ingress in,
    Python I/O throughout), so a shell-safety whitelist here would only reject
    legitimate plans. The paths this helper GENERATES (snapshot/state/meta/
    body) have digest+nonce leaves and are substituted into later commands as
    quoted literals.
    """
    if not raw:
        _fail(2, "empty plan path")
    # The frontmatter and the hook are line-based: a path containing CR/LF can
    # never round-trip through them, and leading/trailing whitespace is far
    # more likely an ingress accident than a real filename — reject both with
    # a clear error instead of silently normalizing to a different file.
    if "\n" in raw or "\r" in raw:
        _fail(
            2,
            f"plan path contains a newline — not representable in the line-based review frontmatter: {raw!r}",
        )
    if raw != raw.strip():
        _fail(2, f"plan path has leading/trailing whitespace (ingress accident?): {raw!r}")
    path = os.path.expanduser(raw) if raw.startswith("~") else raw
    if not os.path.isabs(path):
        _fail(2, f"plan path must be absolute (got {raw!r})")
    return os.path.realpath(path)


def _plans_home() -> str:
    """Reviews and snapshots ALWAYS live in $HOME/.claude/plans — exactly the
    directory the ExitPlanMode hook reads, regardless of where the plan is."""
    return os.path.join(os.path.expanduser("~"), ".claude", "plans")


def _review_path(plan_path: str) -> str:
    """Collision-free review key: canonical basename + canonical-path digest.
    Two plans named plan.md in different repos must never share (and silently
    overwrite) one review. The hook derives the identical key. The review path
    is data everywhere — it is never substituted into shell source."""
    base = os.path.basename(plan_path)
    stem = base[:-3] if base.endswith(".md") else base
    digest = hashlib.sha256(plan_path.encode()).hexdigest()[:12]
    return os.path.join(_plans_home(), f"{stem}.{digest}.review.md")


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        _fail(4, f"cannot write {path}: {exc}")


def cmd_snapshot(args: argparse.Namespace) -> int:
    plan_path = _normalize_plan_path(_read_value_file(args.plan_path_file, "plan-path"))
    try:
        with open(plan_path, "rb") as fh:
            plan_bytes = fh.read()
    except OSError as exc:
        _fail(4, f"cannot read plan {plan_path}: {exc}")
    sha = hashlib.sha256(plan_bytes).hexdigest()

    snap_dir = os.path.join(_plans_home(), ".snapshots")
    try:
        os.makedirs(snap_dir, exist_ok=True)
    except OSError as exc:
        _fail(4, f"cannot create snapshot dir {snap_dir}: {exc}")
    # Invocation-unique leaf: canonical-path digest + nonce, STRICT [a-f0-9.]
    # charset by construction — the raw plan stem NEVER appears in a generated
    # path, because these paths are later substituted into shell commands as
    # quoted literals and a stem like `weird$(cmd).md` would execute there.
    # Arbitrary plan names stay fully supported as DATA (recorded in the state
    # file, not the filename).
    leaf = f"{hashlib.sha256(plan_path.encode()).hexdigest()[:12]}.{uuid.uuid4().hex[:8]}"
    snapshot_path = os.path.join(snap_dir, f"{leaf}.md")
    _atomic_write(snapshot_path, plan_bytes)
    state_path = os.path.join(snap_dir, f"{leaf}.state.json")
    state = {
        "plan_path": plan_path,
        "snapshot_path": snapshot_path,
        "plan_sha256": sha,
        "review_path": _review_path(plan_path),
    }
    _atomic_write(state_path, json.dumps(state).encode("utf-8"))

    print(
        json.dumps(
            {
                "state_path": state_path,
                "snapshot_path": snapshot_path,
                "meta_path": os.path.join(snap_dir, f"{leaf}.meta.json"),
                "body_path": os.path.join(snap_dir, f"{leaf}.body.md"),
                **state,
            }
        )
    )
    return 0


def _one_line(value: object) -> str:
    """Meta values become single-line plain scalars (frontmatter can't break)."""
    return " ".join(str(value).split())


def _cleanup_invocation(state_path: str, state: dict) -> None:
    stem = state_path[: -len(".state.json")]
    for path in (
        state.get("snapshot_path", ""),
        state_path,
        f"{stem}.meta.json",
        f"{stem}.body.md",
    ):
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def cmd_persist(args: argparse.Namespace) -> int:
    state_path = os.path.realpath(args.state_file)
    snap_dir = os.path.realpath(os.path.join(_plans_home(), ".snapshots"))
    if not state_path.endswith(".state.json") or os.path.dirname(state_path) != snap_dir:
        _fail(2, f"state file must be a .state.json inside {snap_dir} (got {state_path})")
    try:
        state = json.loads(_read_value_file(state_path, "state"))
    except ValueError as exc:
        _fail(2, f"state file is not valid JSON: {exc}")
    plan_path = state.get("plan_path", "")
    snapshot_path = state.get("snapshot_path", "")
    recorded_sha = state.get("plan_sha256", "")
    review_path = state.get("review_path", "")
    if not (plan_path and snapshot_path and recorded_sha and review_path):
        _fail(2, "state file is missing required fields")

    # The snapshot must be the regular, non-symlink file the state recorded,
    # inside the owned snapshots dir, whose bytes STILL hash to the RECORDED
    # sha — the digest captured at snapshot time is what gets certified; a
    # later rewrite of the snapshot can never be.
    snap_real = os.path.realpath(snapshot_path)
    if os.path.dirname(snap_real) != snap_dir or os.path.islink(snapshot_path):
        _fail(2, f"snapshot {snapshot_path} is not a regular file in {snap_dir}")
    try:
        with open(snap_real, "rb") as fh:
            snap_bytes = fh.read()
    except OSError as exc:
        _fail(2, f"cannot read snapshot {snapshot_path}: {exc}")
    if hashlib.sha256(snap_bytes).hexdigest() != recorded_sha:
        _cleanup_invocation(state_path, state)
        _fail(
            2,
            f"snapshot {snapshot_path} no longer hashes to the recorded sha — the "
            f"snapshot was altered after capture; nothing was persisted.",
        )

    try:
        with open(plan_path, "rb") as fh:
            live_sha = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        _fail(4, f"cannot re-read live plan {plan_path}: {exc}")
    if live_sha != recorded_sha:
        _cleanup_invocation(state_path, state)
        _fail(
            3,
            f"{plan_path} was modified during the review (live {live_sha[:12]}… != "
            f"reviewed snapshot {recorded_sha[:12]}…). The review was NOT persisted; "
            f"re-run it against the current plan.",
        )

    stem = state_path[: -len(".state.json")]
    try:
        meta = json.loads(_read_value_file(f"{stem}.meta.json", "meta"))
    except ValueError as exc:
        _fail(2, f"meta file is not valid JSON: {exc}")
    if not isinstance(meta, dict):
        _fail(2, "meta must be a JSON object")
    try:
        with open(f"{stem}.body.md", encoding="utf-8") as fh:
            body = fh.read()
    except OSError as exc:
        _fail(2, f"cannot read body file {stem}.body.md: {exc}")

    lines = ["---", f"plan: {plan_path}", f"plan_sha256: {recorded_sha}"]
    for key in _META_KEYS:
        if key in meta:
            value = meta[key]
            if key == "flags" and isinstance(value, list):
                rendered = "[" + ", ".join(json.dumps(_one_line(v)) for v in value) + "]"
                lines.append(f"flags: {rendered}")
            else:
                lines.append(f"{key}: {json.dumps(_one_line(value))}")
    lines.append("---")
    _atomic_write(review_path, ("\n".join(lines) + "\n\n" + body).encode("utf-8"))
    _cleanup_invocation(state_path, state)
    print(review_path)
    return 0


def cmd_abort(args: argparse.Namespace) -> int:
    state_path = os.path.realpath(args.state_file)
    snap_dir = os.path.realpath(os.path.join(_plans_home(), ".snapshots"))
    if not state_path.endswith(".state.json") or os.path.dirname(state_path) != snap_dir:
        _fail(2, f"state file must be a .state.json inside {snap_dir} (got {state_path})")
    try:
        state = json.loads(_read_value_file(state_path, "state"))
    except ValueError:
        state = {}
    _cleanup_invocation(state_path, state if isinstance(state, dict) else {})
    print("aborted")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    plan_path = _normalize_plan_path(_read_value_file(args.plan_path_file, "plan-path"))
    try:
        with open(plan_path, "rb") as fh:
            live_sha = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        _fail(4, f"cannot read plan {plan_path}: {exc}")
    review_path = _review_path(plan_path)
    recorded = ""
    exists = os.path.exists(review_path)
    if exists:
        try:
            with open(review_path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            lines = []
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if line.startswith("plan_sha256:"):
                    recorded = line.split(":", 1)[1].strip().strip("\"'")
                    break
    print(
        json.dumps(
            {
                "plan_path": plan_path,
                "plan_sha256": live_sha,
                "review_path": review_path,
                "review_exists": exists,
                "review_plan_sha256": recorded,
                "fresh": bool(exists and recorded and recorded == live_sha),
            }
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("snapshot")
    ps.add_argument("--plan-path-file", required=True)
    ps.set_defaults(func=cmd_snapshot)
    pp = sub.add_parser("persist")
    pp.add_argument("--state-file", required=True)
    pp.set_defaults(func=cmd_persist)
    pc = sub.add_parser("check")
    pc.add_argument("--plan-path-file", required=True)
    pc.set_defaults(func=cmd_check)
    pa = sub.add_parser("abort")
    pa.add_argument("--state-file", required=True)
    pa.set_defaults(func=cmd_abort)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
