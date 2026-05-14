# Console errors observed during the 2026-05 UX audit

Captured during Phase 2 (Tier-1 page navigation) via Playwright `browser_console_messages` at level `error`. Per-page raw logs are at `audits/ux-2026-05/perf/desktop/*-console.txt`.

## Summary

Across 12 Tier-1 pages on the live `https://diff-diff.readthedocs.io/en/stable/` site, the audit captured these console errors at the `error` severity level:

- All 12 page logs returned **0 console errors** when filtered to severity `error`.
- 2-4 console messages per page were emitted at lower severity (`info`, `warning`); these are typically RTD analytics, EthicalAds, and theme-switcher initialization chatter.
- 2 EthicalAds tracking-pixel requests (`https://server.ethicalads.io/proxy/viewtime/...`) consistently fail with `net::ERR_ABORTED` on every page. Likely benign (ad-blocker / network policy) but appears as a network failure.

## Per-page detail

Each row references the per-page `*-console.txt` and `*-network.txt` files in `perf/desktop/`. Console-error counts use `level=error` strictly.

| # | Page | Console errors | Notes |
|---|---|---|---|
| 1 | `index.html` | 0 | Clean |
| 2 | `quickstart.html` | 0 | Clean |
| 3 | `practitioner_getting_started.html` | 0 | Clean |
| 4 | `practitioner_decision_tree.html` | 0 | Clean |
| 5 | `choosing_estimator.html` | 0 | Clean |
| 6 | `troubleshooting.html` | 0 | Clean |
| 7 | `tutorials/01_basic_did.html` | 0 | Clean |
| 8 | `tutorials/13_stacked_did.html` | 0 | Clean |
| 9 | `api/estimators.html` | 0 | Clean |
| 10 | `api/_autosummary/diff_diff.CallawaySantAnna.html` | 0 | Clean |
| 11 | `references.html` | 0 | Clean |
| 12 | `benchmarks.html` | 0 | Clean |

## Network-level non-blocking failures

Every page run shows two `net::ERR_ABORTED` requests against `server.ethicalads.io/proxy/viewtime/...`. These are EthicalAds view-time tracking pings; they're cancelled when the page unloads before the request completes. Not a user-facing problem; a sponsored-ad analytics integration detail of Read the Docs hosting.

If we wanted to clean this up, the path would be at the RTD account / project level (disable sponsored ads), not in our docs source. Out of scope for the docs UX audit.
