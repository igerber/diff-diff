"""Fit a unified diff into the AI reviewer's input budget.

Used by `.github/workflows/ai_pr_review.yml`. Codex rejects a prompt longer
than 1,048,576 characters outright (`input_too_large`), and a large PR's
unified diff alone can exceed that (PR #861: 1.63 MB of diff after the
notebook exclusion). This filter reads the patch `git diff --unified=5`
produced, and when it is longer than the remaining budget replaces whole
per-file blocks with one-line stubs - largest files first, lowest-value
paths first - until the output fits. Every omitted file stays listed in the
prompt's `Changed files:` section, and each stub carries the exact
`git diff` command the (agentic, repo-checked-out) reviewer can run to see
that file's hunks on demand.

Under budget the patch passes through byte-for-byte, so the prompt of every
PR that fitted before this filter existed is unchanged.

Splitting is on `diff --git ` at column 0: in unified-diff output every body
line carries a one-character prefix (` `, `+`, `-`, `\\`), so a column-0
`diff --git ` line is always a file header, whatever the file contains.

Drop order (deterministic, greedy):
  tier 0 - tests/, benchmarks/, tools/, .claude/   (largest first)
  tier 1 - everything else                         (largest first)
  tier 2 - diff_diff/, rust/                       (largest first; the
           reviewer's primary subject, dropped only when unavoidable)

All sizes are BYTES: the cap counts characters, and for UTF-8 text the byte
count is an upper bound on both code points and UTF-16 units, so fitting in
bytes always fits in characters. stdlib only - the workflow has no
`pip install` step - and no shell is ever invoked (the patch is read from a
file; SHAs are validated hex and appear only as text in the stubs).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

HEADER = b"diff --git "
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")

TIER_FIRST = ("tests/", "benchmarks/", "tools/", ".claude/")
TIER_LAST = ("diff_diff/", "rust/")


class FileBlock(NamedTuple):
    path: str  # display path (b-side, else a-side), decoded for the stub
    data: bytes  # the block's bytes, header line included


def split_blocks(patch: bytes) -> tuple[bytes, List[FileBlock]]:
    """Split a unified diff into its preamble and per-file blocks.

    The preamble is anything before the first `diff --git ` header (normally
    empty); it is kept verbatim.
    """
    if not patch:
        return b"", []
    starts: List[int] = []
    if patch.startswith(HEADER):
        starts.append(0)
    pos = 0
    needle = b"\n" + HEADER
    while True:
        i = patch.find(needle, pos)
        if i < 0:
            break
        starts.append(i + 1)
        pos = i + 1
    if not starts:
        return patch, []
    preamble = patch[: starts[0]]
    blocks: List[FileBlock] = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(patch)
        data = patch[s:e]
        blocks.append(FileBlock(path=_block_path(data), data=data))
    return preamble, blocks


_ESCAPES = {
    b"a": b"\a",
    b"b": b"\b",
    b"t": b"\t",
    b"n": b"\n",
    b"v": b"\v",
    b"f": b"\f",
    b"r": b"\r",
    b"\\": b"\\",
    b'"': b'"',
}


def _c_unquote(raw: bytes) -> bytes:
    """Undo git's C-style path quoting (`"a/b\303\251.py"`)."""
    if not (len(raw) >= 2 and raw[:1] == b'"' and raw[-1:] == b'"'):
        return raw
    body = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i : i + 1]
        if c != b"\\" or i + 1 >= len(body):
            out += c
            i += 1
            continue
        nxt = body[i + 1 : i + 2]
        if nxt in _ESCAPES:
            out += _ESCAPES[nxt]
            i += 2
        elif nxt.isdigit():
            digits = body[i + 1 : i + 4]
            n = len(digits)
            while n > 0 and not all(48 <= d <= 55 for d in digits[:n]):
                n -= 1
            if n == 0:
                out += nxt
                i += 2
            else:
                out.append(int(digits[:n], 8) & 0xFF)
                i += 1 + n
        else:
            out += nxt
            i += 2
    return bytes(out)


def _quoted_token_end(raw: bytes) -> Optional[int]:
    """Index just past the closing quote of the C-quoted token at `raw[0]`."""
    if raw[:1] != b'"':
        return None
    i = 1
    while i < len(raw):
        c = raw[i : i + 1]
        if c == b"\\":
            i += 2
            continue
        if c == b'"':
            return i + 1
        i += 1
    return None


def _strip_side(raw: bytes, prefix: bytes) -> Optional[bytes]:
    """`b/path` (possibly C-quoted as `"b/path"`) -> `path`, else None."""
    val = _c_unquote(raw.rstrip(b"\t"))
    if val.startswith(prefix):
        return val[len(prefix) :]
    return None


def _block_path(data: bytes) -> str:
    """The block's canonical repo path, as git's own headers give it.

    Priority: the extended `rename to` / `copy to` header (renames and
    copies, even without hunks), then `+++ b/` (else `--- a/` for a
    deletion), then the `diff --git a/P b/P` line itself (binary, mode-only
    and other hunk-less blocks). C-quoted paths are unquoted. Only ever
    rendered as text in a stub and used for tiering.
    """
    lines = data.split(b"\n", 60)[:60]
    for line in lines[1:]:
        for key in (b"rename to ", b"copy to "):
            if line.startswith(key):
                return _decode(_c_unquote(line[len(key) :]))
    for line in lines[1:]:
        if line.startswith(b"+++ "):
            side = _strip_side(line[4:], b"b/")
            if side is not None:
                return _decode(side)
    for line in lines[1:]:
        if line.startswith(b"--- "):
            side = _strip_side(line[4:], b"a/")
            if side is not None:
                return _decode(side)
    rem = lines[0][len(HEADER) :]
    if rem.startswith(b'"'):
        # `"a/P" "b/P"`: scan the a-side token honouring backslash escapes
        # (a path may itself contain `" `), then unquote the b-side.
        end = _quoted_token_end(rem)
        if end is not None and rem[end : end + 1] == b" ":
            side = _strip_side(rem[end + 1 :], b"b/")
            if side is not None:
                return _decode(side)
    elif rem.startswith(b"a/") and len(rem) >= 5 and (len(rem) - 5) % 2 == 0:
        # Unquoted `a/P b/P`: the same path on both sides, so P is determined
        # by the remainder's length even when P itself contains " b/".
        n = (len(rem) - 5) // 2
        cand = rem[2 : 2 + n]
        if rem == b"a/" + cand + b" b/" + cand:
            return _decode(cand)
    return _decode(rem)


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def tier(path: str) -> int:
    if path.startswith(TIER_LAST):
        return 2
    if path.startswith(TIER_FIRST):
        return 0
    return 1


def _shell_quote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


def stub(block: FileBlock, base_sha: Optional[str], head_sha: Optional[str]) -> bytes:
    text = f"[diff omitted for prompt budget] {block.path} " f"({len(block.data):,} bytes)."
    if base_sha and head_sha:
        text += (
            " View it: git --no-pager diff --unified=5 "
            f"{base_sha} {head_sha} -- {_shell_quote(':(literal)' + block.path)}"
        )
    return text.encode("utf-8") + b"\n"


def _banner(n_omitted: int, n_total: int, budget: int, full_size: int) -> bytes:
    text = (
        f"[prompt budget] {n_omitted} of {n_total} file diffs are omitted below so the "
        f"prompt fits the reviewer input limit (budget {budget:,} bytes, full diff "
        f'{full_size:,} bytes). Omitted files remain listed under "Changed files:" '
        f"above; each stub gives the git command that shows its hunks.\n\n"
    )
    return text.encode("utf-8")


def _bounded(text: str, budget: int) -> bytes:
    """A diagnostic line that never exceeds the budget (empty if it cannot fit)."""
    raw = text.encode("utf-8")
    if len(raw) > budget:
        print(f"ai_review_diff_budget: {text.rstrip()}", file=sys.stderr)
        return raw[:budget]
    return raw


def fit_to_budget(
    patch: bytes,
    budget: int,
    base_sha: Optional[str] = None,
    head_sha: Optional[str] = None,
) -> bytes:
    """Return `patch` unchanged if `len(patch) <= budget`, else a trimmed patch.

    Greedy: replace blocks by their stubs in (tier, -size) order until the
    output fits, skipping blocks whose stub would not be smaller than the
    block. The result is ALWAYS at most `budget` bytes: if even every
    shrinking replacement cannot get there, the output is a single summary
    line, itself truncated to the budget (empty at budget 0).
    """
    if len(patch) <= budget:
        return patch
    preamble, blocks = split_blocks(patch)
    if not blocks:
        # No file headers - nothing to trim structurally; hard-truncate.
        marker = b"\n[prompt budget] diff truncated here.\n"
        if len(marker) > budget:
            return _bounded("[prompt budget] diff truncated here.\n", budget)
        return patch[: budget - len(marker)] + marker

    stubs = [stub(b, base_sha, head_sha) for b in blocks]
    shrinking = [i for i in range(len(blocks)) if len(stubs[i]) < len(blocks[i].data)]
    order = sorted(shrinking, key=lambda i: (tier(blocks[i].path), -len(blocks[i].data), i))
    omitted = [False] * len(blocks)

    def total(n_omitted: int) -> int:
        body = sum(len(stubs[i]) if omitted[i] else len(blocks[i].data) for i in range(len(blocks)))
        return len(preamble) + len(_banner(n_omitted, len(blocks), budget, len(patch))) + body

    n_omitted = 0
    for i in order:
        if total(n_omitted) <= budget:
            break
        omitted[i] = True
        n_omitted += 1

    if total(n_omitted) > budget:
        return _bounded(
            f"[prompt budget] the unified diff ({len(patch):,} bytes across {len(blocks)} "
            f"files) does not fit the remaining budget of {budget:,} bytes even as stubs; "
            f'see "Changed files:" above and run git diff per file.\n',
            budget,
        )

    out = [preamble, _banner(n_omitted, len(blocks), budget, len(patch))]
    for i, b in enumerate(blocks):
        out.append(stubs[i] if omitted[i] else b.data)
    return b"".join(out)


def _sha(value: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return None
    if not _SHA_RE.match(value):
        raise argparse.ArgumentTypeError(f"not a hex object id: {value!r}")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--diff", required=True, type=Path, help="unified diff file to fit")
    ap.add_argument("--budget", required=True, type=int, help="max output size in bytes")
    ap.add_argument("--base-sha", type=_sha, default=None, help="for the per-stub git command")
    ap.add_argument("--head-sha", type=_sha, default=None, help="for the per-stub git command")
    ap.add_argument("-o", "--output", type=Path, default=None, help="default: stdout")
    args = ap.parse_args(argv)
    if args.budget < 0:
        ap.error("--budget must be >= 0")
    patch = args.diff.read_bytes()
    out = fit_to_budget(patch, args.budget, args.base_sha, args.head_sha)
    if args.output is None:
        sys.stdout.buffer.write(out)
        sys.stdout.buffer.flush()
    else:
        args.output.write_bytes(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
