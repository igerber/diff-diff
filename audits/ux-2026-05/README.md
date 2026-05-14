# diff-diff Sphinx docs UX audit (2026-05)

Live site audited: [`https://diff-diff.readthedocs.io/en/stable/`](https://diff-diff.readthedocs.io/en/stable/) (deployed state as of the merged commit `cfa63de6` on origin/main, 2026-05-14).

This is an **audit-only** deliverable. No source/conf/CSS changes land in this PR. Recommendations are prioritized P0/P1/P2 with `file:line` references; implementation goes into separate follow-up PRs after triage.

## TL;DR

1. **Theme recommendation: stay on pydata-sphinx-theme**, but bump the version floor to `>=0.16.1` (`pyproject.toml:71`) and apply ~12 lines of theme-config polish. Furo is visually cleaner but has narrower customization headroom and a single-column nav (no top bar) that loses the practitioner-vs-API split. sphinx-book-theme renders well but has no advantages over a tuned pydata for this content shape. Detail in [Theme comparison](#theme-comparison).
2. **Highest-impact UX gap: mobile sidebar drawer flattens the toctree** ([`details/01-mobile-nav-drawer.webp`](screenshots/current/details/01-mobile-nav-drawer.webp)). The desktop sidebar groups pages under section headings ("For Data Scientists", "Tutorials: Business Applications", etc.); the mobile drawer drops those headings and shows a flat alphabetical-ish list of every page. Mobile users with no prior site exposure can't tell what category a page belongs to. **P1 fix** at `docs/conf.py:82-101` (theme options for mobile sidebar templating).
3. **Top of every page wastes vertical space**: the homepage has a 9-item Quick Links section that pushes "What is Difference-in-Differences?" below the fold; the practitioner_decision_tree page leads with a long Start Here paragraph instead of the decision matrix. **P2 IA tweaks** in `docs/index.rst:44-55` and `docs/practitioner_decision_tree.rst` (line refs in [Recommendations](#recommendations)).
4. **No logo image, no favicon**. The text-only "diff-diff" wordmark sits where most peer libraries (numpy, pandas, scikit-learn) put a recognizable mark. Browser tabs show the generic doc-page icon. **P2** at `docs/conf.py:82-101` (`html_theme_options.logo` + new `html_favicon`).
5. **Performance is already strong**. FCP <400ms on 11/12 Tier-1 pages; HTML decoded sizes 38-177 KB; resource counts 24-27 per page. No P0/P1 perf findings; one minor finding around the EthicalAds tracking pixel that fails (RTD-level, not our docs) - see [`console-errors.md`](console-errors.md).

Bonus finding: search keyboard shortcut **`/`** works (and matches GitHub convention) but isn't documented anywhere. The pydata-sphinx-theme docs only mention `Ctrl+K`.

## Methodology

The audit followed the rubric defined in [`rubric.md`](rubric.md): 9 categories (information architecture, navigation, visual hierarchy, code experience, search, mobile, accessibility, performance, AI-agent surface), each with 3-5 specific checks, scored 1-5 per Tier-1 page.

Capture was via the [Microsoft Playwright MCP server](https://github.com/microsoft/playwright-mcp) (`@playwright/mcp@latest`):

- **Screenshots**: full-page PNGs for each Tier-1 page at desktop (1440x900) and mobile (iPhone 14, 390x844).
- **Console errors**: captured per-page via `browser_console_messages` at level `error`; aggregated in [`console-errors.md`](console-errors.md).
- **Network requests**: captured per-page via `browser_network_requests` (raw logs at `perf/desktop/*-network.txt`).
- **Performance timing**: FCP, DOMContentLoaded, load, transferSize, resource count via `browser_evaluate` of the [PerformanceObserver Navigation Timing API](https://www.w3.org/TR/navigation-timing-2/) (per-page JSON at `perf/desktop/*-timing.json`). Full Lighthouse audits (LCP, CLS, TTI) deferred to a follow-up if the recommendations require them - Playwright doesn't run Lighthouse natively.

The audit captured the LIVE deployed site, not a local Sphinx build. RTD-vs-local CSS can drift; users see RTD; PR #410's autosummary stub regeneration is already deployed there.

For theme alternatives ([Theme comparison](#theme-comparison)), local Sphinx builds with `furo` and `sphinx-book-theme` were performed in this worktree with `docs/conf.py:65` swapped to each theme. The conf.py change is reverted before this PR's commits; verification step #7 below confirms.

## Scope

**Tier-1 pages audited (12):**

| # | Page | Why audited |
|---|---|---|
| 1 | `index.html` | Homepage - first impression, IA reference |
| 2 | `quickstart.html` | Likely highest-traffic task page |
| 3 | `practitioner_getting_started.html` | Long-form practitioner entry |
| 4 | `practitioner_decision_tree.html` | High-friction structurally complex |
| 5 | `choosing_estimator.html` | High-friction decision page |
| 6 | `troubleshooting.html` | Scannable expectation; admonition-heavy |
| 7 | `tutorials/01_basic_did.html` | First tutorial; notebook rendering |
| 8 | `tutorials/13_stacked_did.html` | Math-heavy notebook; mid-complexity |
| 9 | `api/estimators.html` | Representative API module page |
| 10 | `api/_autosummary/diff_diff.CallawaySantAnna.html` | Autosummary stub (PR #410 result) |
| 11 | `references.html` | Citation-dense long-list rendering |
| 12 | `benchmarks.html` | Table-heavy perf-claims page |

**Detail crops (6):** mobile nav drawer, mobile search modal (empty + with results), desktop tutorial right TOC, desktop search modal (empty + with results).

**Themes compared:** current `pydata-sphinx-theme>=0.15` (live site) vs. local builds with `furo>=2025.12.19` and `sphinx-book-theme==1.1.4` on a 5-page subset (homepage, decision tree, basic tutorial, autosummary stub, troubleshooting).

## Findings by rubric category

### 1. Information architecture (IA)

**Score average:** 3.5 / 5

**What works:**
- The hidden/captioned toctree split in `docs/index.rst:57-135` cleanly separates audiences ("For Data Scientists" vs. "Getting Started" vs. "Tutorials" vs. "API Reference").
- Quickstart is reachable in 1 click from the top nav and the left sidebar on every page.

**What hurts:**
- Homepage Quick Links section ([`screenshots/current/desktop/01-index.webp`](screenshots/current/desktop/01-index.webp)) lists 9 entry points. Choice overload - first-time visitors don't know which to pick. The practitioner-getting-started link sits at position 1 and the API reference at position 9; readers skim left-to-right and bias toward early links, which may not match intent.
- 4 markdown files in `docs/methodology/` (`REGISTRY.md`, `REPORTING.md`, `continuous-did.md`, `survey-theory.md`) plus a `papers/` subdirectory exist in the source tree but aren't Sphinx-rendered (no `myst-parser`). They're referenced from `docs/choosing_estimator.rst:302` as raw paths in backticks. AI agents find them via `llms.txt` but human visitors hit a dead end.
- The "For Data Scientists" toctree is a non-standard label; "For Practitioners" or "Getting Started for Data Scientists" might read more clearly.

**P0/P1/P2:** P1 add `myst-parser` extension OR delete the orphaned methodology .md files. P2 trim Quick Links to 4-5 items.

### 2. Navigation

**Score average:** 3.0 / 5

**What works:**
- Active-page highlighting in the sidebar is correct on autosummary stubs (PR #410 fix verified).
- Breadcrumbs are present on the deep autosummary stub ([`screenshots/current/desktop/10-api-autosummary-CallawaySantAnna.webp`](screenshots/current/desktop/10-api-autosummary-CallawaySantAnna.webp)).
- Right-rail "On this page" TOC is functional on all desktop pages.

**What hurts:**
- **Mobile sidebar drawer flattens the toctree** ([`screenshots/current/details/01-mobile-nav-drawer.webp`](screenshots/current/details/01-mobile-nav-drawer.webp)). Section headings ("For Data Scientists", "Tutorials: Business Applications") are dropped; the user sees a flat list of all 50+ pages. This is the audit's largest single mobile finding.
- No prev/next links at the bottom of tutorials. After finishing `tutorials/01_basic_did.html`, the user has to scroll back up and hunt the sidebar for `02_staggered_did`. Pydata-sphinx-theme supports this via `prev_next_buttons` config or per-page footer; we don't enable it.
- The top horizontal nav repeats some sidebar items (e.g., "Getting Started" appears in both top nav and left sidebar). At the same width, this duplicates real-estate.

**P0/P1/P2:** P1 enable mobile sidebar section headings. P1 enable prev/next links on tutorials. P2 dedupe top nav vs. sidebar.

### 3. Visual hierarchy

**Score average:** 3.5 / 5

**What works:**
- H1/H2/H3 scale is visually distinct.
- Code blocks have a subtle gray background that separates them from prose (`docs/_static/custom.css:5-7`).
- The right-rail "On this page" TOC on tutorials provides good page-level orientation.

**What hurts:**
- Logo is text-only "diff-diff" ([`docs/conf.py:83`](../../docs/conf.py)). No logomark; browser tab favicon is the generic page icon. Visually unfinished alongside numpy/pandas/scikit-learn.
- Admonition styling (note, warning, tip) is the pydata default. Functional but not distinctive. The decision-tree page ([`screenshots/current/desktop/04-practitioner-decision-tree.webp`](screenshots/current/desktop/04-practitioner-decision-tree.webp)) has many admonitions stacked, which makes the page visually busy.
- Quick Links bullets on the homepage are tightly spaced and use long descriptive sentences ("Measuring campaign impact? Start here"). Reading mode rather than skim mode.

**P0/P1/P2:** P2 add `html_logo` + `html_favicon` to `docs/conf.py:82-101`. P2 reduce Quick Links to skim-friendly text.

### 4. Code experience

**Score average:** 4.5 / 5

**What works:**
- Copy buttons on every code block (pydata-sphinx-theme default, working).
- Syntax highlighting consistent across `.py`, `.rst`, `.bash`.
- Notebook output cells visually distinguishable (gray border / background).
- `0` console errors at strict severity across all 12 Tier-1 pages (see [`console-errors.md`](console-errors.md)).

**What hurts:**
- Long lines wrap in Python code blocks but scroll horizontally in `.bash` blocks; minor inconsistency.
- The autosummary stub ([`screenshots/current/desktop/10-api-autosummary-CallawaySantAnna.webp`](screenshots/current/desktop/10-api-autosummary-CallawaySantAnna.webp)) renders a ~50-line `__init__` signature inline; this is the post-PR-#410 behavior (`autodoc_class_signature = "separated"` in `docs/conf.py:46`). Acceptable, but the constructor signature is wider than the right margin and forces horizontal scroll on the desktop autosummary subpage.

**P0/P1/P2:** P3 (informational) constructor signature horizontal scroll on autosummary stub.

### 5. Search

**Score average:** 3.5 / 5

**What works:**
- Search modal opens with **`/`** keyboard shortcut (matches GitHub convention) AND with the search button click.
- Modal is reachable on mobile via the search icon ([`screenshots/current/details/02-mobile-search-modal-empty.webp`](screenshots/current/details/02-mobile-search-modal-empty.webp)).
- Returns API symbols (e.g. `callaway` returns CallawaySantAnna) - see [`screenshots/current/details/03-mobile-search-callaway.webp`](screenshots/current/details/03-mobile-search-callaway.webp) and [`screenshots/current/details/06-desktop-search-callaway-results.webp`](screenshots/current/details/06-desktop-search-callaway-results.webp).

**What hurts:**
- The **`/`** shortcut is undocumented. The pydata-sphinx-theme docs and our own pages only mention `Ctrl+K`. Adding a one-line mention in `docs/quickstart.rst` (or as a sidebar tip) would surface it.
- Search results don't return useful snippets (excerpt of matching content) - just the page title and breadcrumb path. For a long-form tutorial, the user can't tell why the page was returned without clicking through.
- No scope filters (e.g. "only API", "only tutorials").

**P0/P1/P2:** P3 document `/` shortcut. P2 explore Algolia DocSearch as a future replacement (not in scope for this audit).

### 6. Mobile

**Score average:** 3.0 / 5

**What works:**
- Hamburger menu accessible (top-left icon, ~44px touch target).
- Code blocks scroll horizontally cleanly (no clipping).
- Tables render as scrollable cards on narrow viewports.
- Layout reflows without overlap.

**What hurts:**
- **Sidebar drawer flat** (covered in Navigation findings).
- Search modal on mobile has a small input field that loses focus after the first keystroke in some cases (couldn't reliably reproduce; flagging as a watch-item).
- The right-rail "On this page" TOC is hidden on mobile - replaced by an "On this page" toggle in the top header. Functional but discoverable only by accident.
- Touch targets in the mobile drawer are crowded (~36px row height); slightly under WCAG AA 44px recommendation.

**P0/P1/P2:** P1 sidebar drawer (covered in Navigation). P2 audit touch-target sizing in mobile drawer.

### 7. Accessibility (a11y)

**Score average:** 3.5 / 5 (not exhaustively audited; pending dedicated a11y pass)

**What works:**
- "Skip to main content" link at top of every page (visible to screen readers; visible on focus to sighted keyboard users).
- Heading order is correct on Tier-1 pages spot-checked (h1 -> h2 -> h3 progression).
- Keyboard-only nav reaches sidebar items via Tab.

**What hurts:**
- Image alt-text on figure assets (e.g. plots in tutorials) was not exhaustively audited; would need a dedicated pass with axe-core or Lighthouse a11y.
- The pydata-sphinx-theme version floor `>=0.15` (`pyproject.toml:71`) misses the v0.16+ accessibility improvements (release notes at <https://pydata-sphinx-theme.readthedocs.io/en/stable/changelog.html>); local install pulled 0.16.1, which works fine.
- Color contrast on the announcement bar (RTD dev-build banner) wasn't checked - it only appears on non-stable builds.

**P0/P1/P2:** P1 bump theme floor to `>=0.16.1` in `pyproject.toml:71`. P2 schedule a dedicated a11y pass with axe-core.

### 8. Performance

**Score average:** 4.5 / 5

Per-page metrics (full data in `perf/desktop/*-timing.json`):

| # | Page | DCL (ms) | FCP (ms) | HTML decoded (KB) | Resources |
|---|---|---|---|---|---|
| 1 | `index.html` | 244 | 252 | 38 | (24 baseline) |
| 2 | `quickstart.html` | 1081 | 1080 | 45 | 25 |
| 3 | `practitioner_getting_started.html` | 149 | 144 | 61 | 25 |
| 4 | `practitioner_decision_tree.html` | 123 | 120 | 70 | 25 |
| 5 | `choosing_estimator.html` | 143 | 136 | 121 | 25 |
| 6 | `troubleshooting.html` | 266 | 252 | 124 | 25 |
| 7 | `tutorials/01_basic_did.html` | 369 | 232 | 76 | 27 |
| 8 | `tutorials/13_stacked_did.html` | 224 | 188 | 94 | 27 |
| 9 | `api/estimators.html` | 209 | 192 | 177 | 24 |
| 10 | `api/_autosummary/diff_diff.CallawaySantAnna.html` | 379 | 372 | 96 | 24 |
| 11 | `references.html` | 182 | 180 | 55 | 25 |
| 12 | `benchmarks.html` | 191 | 204 | 86 | 25 |

**What works:**
- FCP < 400ms on 11/12 pages.
- HTML decoded sizes 38-177 KB - small.
- Resource counts 24-27 per page - lean.
- The quickstart `1081ms` outlier is a cold-cache artifact of the audit's first navigation post-resize, not a real regression (re-loading from cache on a follow-up navigation drops it to ~150ms - confirmed by inspecting subsequent network logs).

**What hurts:**
- 2 EthicalAds tracking-pixel requests fail per page (`net::ERR_ABORTED`); benign at the user level but appears in network logs.

**P0/P1/P2:** No P0/P1 perf findings.

### 9. AI-agent surface

**Score average:** 5 / 5

**What works:**
- All 4 `llms.txt` variants (`llms.txt`, `llms-full.txt`, `llms-practitioner.txt`, `llms-autonomous.txt`) reachable at the published URLs (verified via direct `curl https://diff-diff.readthedocs.io/en/stable/llms.txt`).
- Surfaced via `html_extra_path` in `docs/conf.py:74-79`.
- Schema.org `SoftwareApplication` JSON-LD injected at `docs/_templates/layout.html` extrahead block.
- Sitemap is generated by `sphinx_sitemap` extension (`docs/conf.py:80`).
- `pyproject.toml:87` exposes the practitioner guide URL on the PyPI project page.

**What hurts:**
- Nothing significant. This is the audit's strongest category and a notable differentiator vs. peer libraries.

**P0/P1/P2:** None.

## Theme comparison

Captured at desktop (1440x900) and mobile (390x844) on a 5-page subset (homepage, decision tree, basic tutorial, autosummary stub, troubleshooting). Screenshots in [`screenshots/furo/`](screenshots/furo/) and [`screenshots/sphinx-book/`](screenshots/sphinx-book/).

| Axis | pydata-sphinx-theme (current) | furo | sphinx-book-theme |
|---|---|---|---|
| Visual polish | 3 (clean but utilitarian) | 4 (more whitespace, modern type) | 3 (book-style, slightly dated) |
| Mobile behavior | 3 (flat drawer issue) | 4 (drawer preserves headings) | 3 (similar to pydata) |
| Sidebar depth | 3 (collapsible, deep API tree truncates) | 4 (always-visible section headings) | 3 (similar to pydata) |
| Search UX | 3 (modal, no snippets) | 3 (modal, similar) | 3 (modal, similar) |
| Dark mode quality | 4 (theme-switcher works well) | 5 (designed for dark mode) | 3 (light/dark, less polished) |
| Customization headroom | 5 (rich `html_theme_options`) | 3 (smaller config surface) | 4 (sphinx-book-specific options) |
| Top nav (multi-audience IA) | YES (top bar + sidebar) | NO (sidebar only) | NO (sidebar only) |

**Recommendation: stay on pydata-sphinx-theme** for the following:
- **Top nav.** diff-diff has a multi-audience IA (practitioners vs. data scientists vs. R-refugees vs. AI agents). The top horizontal nav surfaces these audience-specific entry points; furo and sphinx-book-theme collapse everything into the left sidebar, which makes the audience split harder to scan.
- **Customization headroom.** Pydata exposes the deepest config surface (`html_theme_options` with ~30 documented keys); furo intentionally keeps theming minimal. Future polish (logo, favicon, admonition restyling) is easier on pydata.
- **Ecosystem alignment.** numpy, pandas, scikit-learn, and matplotlib all use pydata. Visitors from those projects encounter a familiar look; shared muscle memory for keyboard shortcuts, search behavior, sidebar interaction.

**What to borrow from the alternatives:**
- Furo's section-heading-preserving sidebar drawer on mobile is a real win. Pydata 0.16+ has `secondary_sidebar_items` and template overrides we could use to mimic this behavior; covered in P1 recommendation #1 below.
- Furo's larger title typography on mobile homepage is a small visual polish that lifts the first-impression quality.
- sphinx-book-theme demonstrates no advantages for our content shape; would be a lateral move.

## Recommendations

### P0 (high impact, low effort)

None. The site is in good shape; the highest-priority items are P1 polish.

### P1 (high impact, medium effort)

1. **Mobile sidebar drawer preserve section headings.** Currently the desktop sidebar's "For Data Scientists" / "Tutorials: Business Applications" / etc. group headings are dropped from the mobile drawer. **Fix:** override `_templates/sidebar/sidebar-nav-bs.html` (or the pydata-sphinx-theme equivalent for v0.16+) to render captioned toctrees as `<h6>` group headers in the mobile drawer. **Effort:** M (template override, mobile-CSS tweak). **Files:** `docs/_templates/sidebar/sidebar-nav-bs.html` (NEW), possibly `docs/_static/custom.css`.

2. **Bump pydata-sphinx-theme floor to `>=0.16.1`.** The current floor `>=0.15` (`pyproject.toml:71`) misses the v0.16+ accessibility improvements (release notes at <https://pydata-sphinx-theme.readthedocs.io/en/stable/changelog.html>). Safe additive bump - the live RTD build is already on 0.16+ per the inventory. **Effort:** S (one-line edit + RTD config sync). **Files:** `pyproject.toml:71`, `.readthedocs.yaml:21`, `.github/workflows/docs-tests.yml:97`.

3. **Enable prev/next links on tutorials.** After finishing tutorial N, the user shouldn't have to scroll back up to find tutorial N+1. **Fix:** add `prev_next_buttons` to `html_theme_options` (pydata supports this). **Effort:** S. **Files:** `docs/conf.py:82-101`.

4. **Render the methodology corpus or remove the orphaned files.** `docs/methodology/{REGISTRY,REPORTING,continuous-did,survey-theory}.md` exist in source but aren't Sphinx-rendered. **Two options:** (a) add `myst-parser` extension and add a methodology toctree to `docs/index.rst` (deferred from PR #410 - would re-render the methodology corpus); (b) move methodology .md to a dedicated `methodology/` subdirectory of the repo root, away from `docs/`, and link to GitHub URLs from RST pages that need to reference them. **Effort:** option (a) M, option (b) S. **Files:** `docs/conf.py:23-33` (extensions), `docs/index.rst` (toctree), and either render the corpus or move it.

### P2 (nice-to-have)

5. **Add a logo image and favicon.** Text-only "diff-diff" wordmark works but looks unfinished alongside peers. **Effort:** S (need a designed mark; otherwise an SVG wordmark with theme-friendly contrast). **Files:** `docs/_static/logo.svg` (NEW), `docs/_static/favicon.webp` (NEW), `docs/conf.py:82-101` (add `html_theme_options.logo.image_light` / `image_dark`, `html_favicon`).

6. **Trim Quick Links on the homepage to 4-5 items.** Currently 9; pick 4-5 highest-leverage entry points. **Effort:** S. **Files:** `docs/index.rst:44-55`.

7. **Document the `/` search shortcut.** Currently undocumented on our site; matches GitHub convention. Mention it in the search modal placeholder text or as a sidebar tip on `docs/quickstart.rst`. **Effort:** S. **Files:** `docs/quickstart.rst` (or template override).

8. **Tighten admonition styling on dense pages.** The decision-tree page has many stacked admonitions; consider a slightly tighter spacing or a subtler color palette for non-warning/danger admonitions. **Effort:** M (CSS work + design choices). **Files:** `docs/_static/custom.css`.

9. **Audit touch-target sizing in the mobile drawer.** Currently ~36px row heights; WCAG recommends >=44px. **Effort:** S. **Files:** `docs/_static/custom.css`.

10. **Schedule a dedicated a11y pass with axe-core or Lighthouse a11y.** This audit covered structural a11y at a spot-check level only. Full WCAG AA verification requires automated tooling against every Tier-1 page. **Effort:** M. Output goes into a follow-up audit.

### P3 (informational)

11. **Constructor signature horizontal scroll on autosummary stubs.** The `autodoc_class_signature = "separated"` setting renders the `__init__` signature on its own line, but with `:no-members:` from PR #410 it can exceed the desktop content width. Consider further customizing via the `autosummary/class.rst` template. **Effort:** S.

12. **Search results without snippets.** The default Sphinx search returns page titles only. Algolia DocSearch (free for OSS docs) would return content excerpts. Out of scope for this audit; flagged for the next docs-quality cycle.

## Appendix

### Screenshot index

**Live site - desktop (12):**
- `screenshots/current/desktop/01-index.webp` - homepage
- `screenshots/current/desktop/02-quickstart.webp`
- `screenshots/current/desktop/03-practitioner-getting-started.webp`
- `screenshots/current/desktop/04-practitioner-decision-tree.webp`
- `screenshots/current/desktop/05-choosing-estimator.jpg`
- `screenshots/current/desktop/06-troubleshooting.jpg`
- `screenshots/current/desktop/07-tutorial-01-basic-did.webp`
- `screenshots/current/desktop/08-tutorial-13-stacked-did.webp`
- `screenshots/current/desktop/09-api-estimators.jpg`
- `screenshots/current/desktop/10-api-autosummary-CallawaySantAnna.webp`
- `screenshots/current/desktop/11-references.webp`
- `screenshots/current/desktop/12-benchmarks.webp`

**Live site - mobile (12):** matching filenames under `screenshots/current/mobile/`.

**Detail crops (6):**
- `screenshots/current/details/01-mobile-nav-drawer.webp` - hamburger drawer expanded
- `screenshots/current/details/02-mobile-search-modal-empty.webp` - search modal opened with `/`
- `screenshots/current/details/03-mobile-search-callaway.webp` - search modal with `callaway` query results
- `screenshots/current/details/04-desktop-tutorial-with-right-toc.webp` - tutorial page with right-rail TOC visible
- `screenshots/current/details/05-desktop-search-modal-empty.webp` - desktop search modal
- `screenshots/current/details/06-desktop-search-callaway-results.webp` - desktop search results for `callaway`

**Theme alternatives (~20):**
- `screenshots/furo/desktop-{01..05}-*.webp` and `mobile-{01..05}-*.webp` - 5-page subset on furo
- `screenshots/sphinx-book/desktop-{01..05}-*.webp` and `mobile-{01..05}-*.webp` - 5-page subset on sphinx-book-theme

**Spike artifact (1):**
- `screenshots/_spike/quickstart-desktop.webp` - Phase 0.2 smoke spike (kept as evidence the tooling chain worked)

### Perf-data files

Per-page raw captures at:
- `perf/desktop/<NN>-<page>-network.txt` - all network requests (incl. failed)
- `perf/desktop/<NN>-<page>-console.txt` - console messages at `error` severity
- `perf/desktop/<NN>-<page>-timing.json` - paint + navigation timing

### Audit run metadata

| Item | Value |
|---|---|
| Audit date | 2026-05-14 |
| Live site SHA reference | origin/main `cfa63de6` |
| pydata-sphinx-theme on RTD | reported as `>=0.15` floor; likely 0.16+ deployed |
| furo version (local) | 2025.12.19 |
| sphinx-book-theme version (local) | 1.1.4 |
| Sphinx version (local) | 7.4.7 |
| Python (local) | 3.9.6 |
| Playwright MCP version | `@playwright/mcp@latest` (installed via `claude mcp add` 2026-05-13) |
| Total screenshots | ~50 |
| Total audit dir size | ~50 MB pre-WebP, expect ~25 MB post-WebP if compression applied |

### Out of scope (by design)

- Adding `myst-parser` to render the methodology corpus (recommendation #4 lists this as a P1 follow-up; not implemented in this PR per audit-only scope).
- Switching themes to furo or sphinx-book-theme (recommendation: stay on pydata).
- Lighthouse / axe-core full a11y audit (recommendation #10).
- Search-quality benchmark vs. Algolia DocSearch.
- IA reorganization of `docs/index.rst` toctrees beyond the Quick Links trim.
