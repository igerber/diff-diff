### Internal
- **Repo-process tooling** (infra sweep, part 2): a weekly `LWDiD Data
  Canary` CI lane that fails loudly when the SHA-pinned Prop 99 / Walmart
  loaders fall back to synthetic data (previously a visible-but-green test
  skip), then runs the replication tests it de-gates; a
  `tests/test_tracking_files.py` contract guard for TODO.md/DEFERRED.md
  (no deferred-work pointers in TODO rows, no ledger-lifecycle
  restatements beside `M-xxx` cross-links, documented table shapes with
  per-row column counts — which also surfaced and fixed two DEFERRED.md
  rows whose unescaped pipes broke the rendered tables and one row
  restating M-010's version target); and the `/push-pr-update`
  committed-range methodology scan restored via `premerge_scan.py --range`
  with the comparison ref passed as quoted data.
