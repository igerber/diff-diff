"""Load the JSON corpus into ``eval_core.models.Case`` objects.

Resolves each ground-truth bug's ``bug_class`` to keyword lists via
``corpus/bug_class_synonyms.json`` so the rendered comparison bundle carries the
right vocabulary. Threads each case's on-disk directory into
``fixture["_case_dir"]`` so the worktree adapter can find ``inject.diff``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from eval_core.models import Case, GroundTruthBug

from adapters import worktree


def _load_synonyms(corpus_dir: str) -> dict[str, list[str]]:
    path = os.path.join(corpus_dir, "bug_class_synonyms.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _post_diff_paths(name_status: str) -> set:
    """POST-diff paths from ``git diff --name-status`` (for expected-file validation).

    ``ground_truth.file`` is defined on the post-diff file, so:
      ``M/A/T/... <path>``      -> ``path``
      ``R<score> <old> <new>``  -> ``new`` (destination only)
      ``C<score> <old> <new>``  -> ``new`` (destination only)
      ``D <path>``              -> none (deleted; no post-diff file)
    For R/C the destination is the LAST column; for M/A/T there's a single path.
    (Notebook detection keeps its own all-columns parse in ``ci_prompt``.)
    """
    out = set()
    for line in name_status.splitlines():
        cols = line.split("\t")
        if len(cols) < 2 or cols[0].strip()[:1] == "D":
            continue
        dest = cols[-1].strip()
        if dest:
            out.add(dest)
    return out


def _validate_touched_files(case, touched: set) -> Optional[str]:
    """Validate a case's POST-diff touched paths against its file contract. Pure (no
    git) so it's unit-testable; ``verify()`` supplies ``touched``. Returns an error
    string, or None if OK.

    - ``expect_no_blockers`` (negative control): MUST declare ``expected_files`` and
      the diff must touch EXACTLY those — otherwise a mis-pinned clean case could
      become a code-changing diff the grader still treats as blocker-free.
    - ``expected_files`` on any case: exact-match contract.
    - otherwise: every ``ground_truth.file`` must be touched (diff may touch more).
    """
    declared = set(case.expected_files or [])
    if case.expect_no_blockers and not declared:
        return "negative-control case (expect_no_blockers) must declare expected_files"
    if declared:
        if touched != declared:
            return (
                f"diff touched {sorted(touched)} but case declares "
                f"expected_files {sorted(declared)}"
            )
        return None
    expected = {b.file for b in case.ground_truth}
    missing = _missing_expected(touched, expected)
    if expected and missing:
        return f"diff does not touch expected file(s) {sorted(missing)}; touched {sorted(touched)}"
    return None


def _missing_expected(touched: set, expected: set) -> set:
    """Expected repo-relative files NOT present in the touched set, by EXACT equality.

    ``git diff --name-status`` and ``ground_truth.file`` are both repo-relative, so
    require exact path equality — suffix matching (``a.endswith(b)``) false-matches
    duplicate basenames (this repo has many: ``runner.py``, ``compare.py``,
    ``README.md``), letting a malformed case verify against the wrong file.
    """
    return {f for f in expected if f not in touched}


def _bug_from_dict(d: dict, synonyms: dict[str, list[str]]) -> GroundTruthBug:
    lw = d.get("line_window", [0, 0])
    bug_class = d.get("bug_class", "")
    keywords = list(d.get("class_keywords", [])) or synonyms.get(bug_class, [])
    return GroundTruthBug(
        id=d["id"],
        file=d["file"],
        line_window=(int(lw[0]), int(lw[1])),
        bug_class=bug_class,
        expected_severity=d.get("expected_severity", "P1"),
        must_catch=d.get("must_catch", True),
        anchor_symbol=d.get("anchor_symbol", ""),
        class_keywords=keywords,
        rationale=d.get("rationale", ""),
        provenance=d.get("provenance", {}),
    )


def _case_from_dict(d: dict, case_dir: str, synonyms: dict[str, list[str]]) -> Case:
    fixture = dict(d.get("fixture", {}))
    fixture["_case_dir"] = case_dir
    # Carry pr_context through the fixture so the reviewer can read it.
    if "pr_context" in d:
        fixture["pr_context"] = d["pr_context"]
    return Case(
        id=d["id"],
        stratum=d["stratum"],
        title=d.get("title", ""),
        fixture=fixture,
        ground_truth=[_bug_from_dict(b, synonyms) for b in d.get("ground_truth", [])],
        expect_no_blockers=d.get("expect_no_blockers", False),
        allow_severities=d.get("allow_severities", ["P2", "P3"]),
        known_fp_topics=d.get("known_fp_topics", []),
        expected_files=d.get("expected_files", []),
        weight=float(d.get("weight", 1.0)),
        notes=d.get("notes", ""),
    )


class CorpusLoader:
    def __init__(self, corpus_dir: str, repo_root: str):
        self.corpus_dir = corpus_dir
        self.repo_root = repo_root
        self.cases_dir = os.path.join(corpus_dir, "cases")
        self._synonyms = _load_synonyms(corpus_dir)

    def load_cases(self, strata: Optional[list[str]] = None) -> list[Case]:
        cases: list[Case] = []
        if not os.path.isdir(self.cases_dir):
            return cases
        for stratum in sorted(os.listdir(self.cases_dir)):
            # Distinguish None (no --strata -> all strata) from [] (a bare --strata with
            # no values -> an explicit empty selection that must match NOTHING, so a typo
            # / empty shell expansion fails closed instead of silently running everything).
            if strata is not None and stratum not in strata:
                continue
            stratum_dir = os.path.join(self.cases_dir, stratum)
            if not os.path.isdir(stratum_dir):
                continue
            for case_id in sorted(os.listdir(stratum_dir)):
                case_dir = os.path.join(stratum_dir, case_id)
                case_json = os.path.join(case_dir, "case.json")
                if not os.path.exists(case_json):
                    continue
                with open(case_json, encoding="utf-8") as fh:
                    d = json.load(fh)
                # Selection filters on the DIRECTORY stratum but the Case carries the
                # JSON-declared one; a mismatch would let `--strata X` run a case
                # reported under stratum Y, silently confounding the stratified A/B.
                if d.get("stratum") != stratum:
                    raise ValueError(
                        f"stratum mismatch in {case_json}: declared {d.get('stratum')!r} "
                        f"but filed under cases/{stratum}/ — they must match."
                    )
                cases.append(_case_from_dict(d, case_dir, self._synonyms))
        # case.id is the primary key across caching, artifact naming, and bundle
        # grouping. Fail closed on duplicates before any run: two cases sharing an
        # id would alias caches, overwrite the same artifact, and collapse into one
        # comparison section with no warning.
        seen: dict[str, str] = {}
        for case in cases:
            # case.id is metadata, never a filesystem path component, but reject
            # reserved dot segments outright so they can't masquerade as ids.
            if case.id in (".", "..") or not case.id.strip():
                raise ValueError(
                    f"reserved/empty case id {case.id!r} in "
                    f"{case.fixture.get('_case_dir', '?')}: ids must be real identifiers."
                )
            prior = seen.get(case.id)
            if prior is not None:
                raise ValueError(
                    f"duplicate case id {case.id!r}: defined by both {prior} and "
                    f"{case.fixture.get('_case_dir', '?')}; case ids must be unique."
                )
            seen[case.id] = case.fixture.get("_case_dir", "?")
        return cases

    def verify(self, case: Case) -> Optional[str]:
        """Materialize the case and assert the diff touches expected files.

        Returns an error string, or None if OK. Worktree is always cleaned up.
        """
        runs_root = os.path.join(self.corpus_dir, "..", "runs")
        worktrees_root = os.path.join(os.path.abspath(runs_root), ".worktrees")
        fixture = dict(case.fixture)
        case_dir = fixture.get("_case_dir", "")
        try:
            mat = worktree.materialize(
                case.id, fixture, self.repo_root, worktrees_root, case_dir=case_dir
            )
        except worktree.MaterializeError as exc:
            return f"materialize failed: {exc}"
        try:
            from adapters.ci_prompt import git_name_status

            name_status = git_name_status(mat.worktree_dir, mat.base_sha, mat.head_sha)
            if not name_status.strip():
                return "empty diff (base==head or patch was a no-op)"
            # Tutorial-notebook cases are supported: build_ci_prompt appends the
            # CI-equivalent <notebook-prose> block (see adapters/ci_prompt.py).
            # ground_truth.file is defined on the POST-diff file, so validate against
            # post-diff paths only (rename/copy -> destination; delete -> none). Using
            # every name-status column would let a case that records the pre-rename
            # path "verify" against the wrong file.
            return _validate_touched_files(case, _post_diff_paths(name_status))
        finally:
            worktree.cleanup(mat.worktree_dir, self.repo_root)


__all__ = ["CorpusLoader"]
