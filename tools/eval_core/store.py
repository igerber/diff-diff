"""Persistence for run artifacts + content-hash resume keys.

GENERIC — stdlib-only (``json``, ``hashlib``, ``os``). Stores one JSON file per
RunResult keyed by a content hash of ``(case_id, config_id, repeat,
experiment_tag, case_tag)`` (see ``run_key``) so a crashed full run resumes by
skipping already-completed runs — and a changed model/prompt/backend
(``experiment_tag``) or edited case (``case_tag``) yields a distinct key rather
than resuming a stale artifact. The store never imports a VCS or reviewer backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from eval_core.models import RunResult, run_result_from_dict, to_jsonable


def run_key(
    case_id: str,
    config_id: str,
    repeat_idx: int,
    experiment_tag: str = "",
    case_tag: str = "",
) -> str:
    """Stable content-addressed key for one run (used as the filename stem).

    ``experiment_tag`` is an opaque identity (from ``Reviewer.experiment_tag``)
    folding in everything that defines *which experiment* this is beyond
    case/config/repeat — the model, effort, prompt, and reviewer CLI version. It
    is part of the digest so that changing the model (or prompt) under the same
    config id ("A"/"B") yields a DISTINCT key, preventing a rerun from silently
    resuming a stale review for a different experiment.

    ``case_tag`` is the parallel identity for the CASE content (from
    ``Reviewer.case_tag``) — base SHA, patch/diff bytes, PR context. Folding it in
    means editing a corpus case under the same id also yields a DISTINCT key, so a
    rerun never resumes a stale review against the OLD case content.
    """
    raw = "|".join([case_id, config_id, str(repeat_idx), experiment_tag or "", case_tag or ""])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe = f"{case_id}.{config_id}.r{repeat_idx}".replace("/", "_")
    return f"{safe}.{digest}"


class RunStore:
    """A directory of run-result JSON files."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    def has(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def load(self, key: str) -> Optional[RunResult]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return run_result_from_dict(json.load(fh))
        except (OSError, ValueError, KeyError):
            return None

    def save(self, key: str, result: RunResult) -> None:
        """Atomic write (tmp + replace) so a crash can't leave a partial file."""
        path = self._path(key)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(result), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def load_all(self) -> list[RunResult]:
        out: list[RunResult] = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json") or name.endswith(".tmp"):
                continue
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as fh:
                    out.append(run_result_from_dict(json.load(fh)))
            except (OSError, ValueError, KeyError):
                continue
        return out


def write_json(path: str, obj: object) -> None:
    """Write any to_jsonable-able object to ``path`` (parents created)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["run_key", "RunStore", "write_json", "read_json"]
