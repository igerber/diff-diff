# diff-diff Sphinx docs UX audit rubric (2026-05)

## Purpose

Define the rubric used to evaluate the diff-diff documentation site at `https://diff-diff.readthedocs.io/en/stable/`. Scores and findings here feed into the audit deliverable at `README.md` in this directory.

This is an audit-only artifact: no implementation lands in the same PR. Recommendations are prioritized P0/P1/P2 in the main audit doc.

## Scoring scale (1-5)

| Score | Meaning |
|---|---|
| 5 | Exemplary - matches best-in-class documentation sites (scikit-learn, pandas, numpy quality bar) |
| 4 | Good - works well; minor refinements possible |
| 3 | Acceptable - works but introduces friction users notice |
| 2 | Friction-heavy - frustrates users; obvious blocker |
| 1 | Broken or missing entirely |

Per-cell judgement uses the rubric checks as a guide, not a hard checklist - if a page genuinely satisfies a category through a different mechanism than the listed check, that still scores well.

## Tier-1 pages audited

1. `index.html` (homepage)
2. `quickstart.html`
3. `practitioner_getting_started.html`
4. `practitioner_decision_tree.html`
5. `choosing_estimator.html`
6. `troubleshooting.html`
7. `tutorials/01_basic_did.html`
8. `tutorials/13_stacked_did.html`
9. `api/estimators.html`
10. `api/_autosummary/diff_diff.CallawaySantAnna.html`
11. `references.html`
12. `benchmarks.html`

Each page is captured at desktop (1440x900) and mobile (iPhone 14, 390x844) viewports.

## Categories

### 1. Information architecture (IA)

**Definition:** How easily can a new visitor reach the content they need? Is the site structure visible and predictable?

**Checks:**
- Quickstart reachable in 1 click from any page (sidebar or header)
- Tutorial sections discoverable without scrolling on the index
- Practitioner vs. API split visible from the index
- Orphan pages exist (the 4 unbuilt methodology `.md` files in `docs/methodology/`)

**Findings:**
- (filled during Phase 2)

### 2. Navigation

**Definition:** Quality of the navigation chrome - sidebar, breadcrumbs, active-page highlighting, prev/next links.

**Checks:**
- Sidebar depth (`navigation_depth=3` per `docs/conf.py:90`) truncates the API tree usefully or unhelpfully
- Breadcrumbs present on deep API stubs
- Active-page highlighting in sidebar correct on autosummary stubs
- Prev/next links present and accurate at the bottom of tutorials

**Findings:**
- (filled during Phase 2)

### 3. Visual hierarchy

**Definition:** Typography, spacing, contrast, and visual separation. Is the page readable at a glance? Does the eye know where to look?

**Checks:**
- H1/H2/H3 scale distinguishable
- Code blocks visually separated from prose
- Admonitions (note, warning, tip) consistently styled
- Logo - text-only "diff-diff" wordmark reads as brand or unfinished

**Findings:**
- (filled during Phase 2)

### 4. Code experience

**Definition:** Quality of code-block rendering, including copy buttons, syntax highlighting, line wrapping, notebook input/output distinction.

**Checks:**
- Copy-button present and working on every code block
- Long lines either wrap cleanly or scroll horizontally
- Syntax highlighting consistent across `.py`, `.rst`, `.bash`
- Notebook output cells visually distinguishable from input cells
- Per-page console errors (captured in `console-errors.md` during navigation)

**Findings:**
- (filled during Phase 2)

### 5. Search

**Definition:** Quality of the search affordance - how it's invoked, what it returns, how useful results are.

**Checks:**
- Indexes API symbols (not just page titles) - try `callaway`, `synthetic`, `honest`
- Useful snippets returned (excerpt of matching content)
- Reachable on mobile (no hidden behind small/missing affordance)
- Keyboard shortcut documented (e.g. `/` or `Ctrl+K`)

**Findings:**
- (filled during Phase 2)

### 6. Mobile

**Definition:** How the site performs at narrow viewports. iPhone 14 viewport (390x844) used as canonical reference.

**Checks:**
- Hamburger sidebar accessible
- Tables scrollable, not clipped
- Code blocks readable without horizontal scroll (or scroll works smoothly)
- MathJax renders without overflow
- Touch targets >= 44px (sidebar links, nav buttons)

**Findings:**
- (filled during Phase 2)

### 7. Accessibility (a11y)

**Definition:** WCAG conformance and keyboard / screen-reader usability. Per the v0.16+ accessibility improvements in pydata-sphinx-theme, a base level should be achievable today.

**Checks:**
- WCAG AA color contrast on body text + code blocks
- Keyboard-only sidebar nav reaches all items
- Heading order correct (no h2 -> h4 jumps)
- Image alt-text on inline figure assets

**Findings:**
- (filled during Phase 2)

### 8. Performance (perf)

**Definition:** Page weight, request count, and paint timings. Captured via Playwright's native APIs (full Lighthouse audit deferred to a follow-up if recommendations require it - Playwright doesn't run Lighthouse natively).

**Checks:**
- FCP and DOMContentLoaded via `performance.getEntriesByType('paint'|'navigation')` for top 3 pages
- Total page weight + request count via Playwright `browser_network_requests` for all 12 Tier-1 pages
- MathJax lazy-load (large script - shouldn't block initial render on non-math pages)
- Theme JS bundle size

**Findings:**
- (filled during Phase 2; perf-numbers table goes in `README.md` appendix)

### 9. AI-agent surface

**Definition:** Quality of the structured signals available to AI agents and crawlers. diff-diff is unusual in shipping `llms.txt` variants - audit how well this surface holds together.

**Checks:**
- `llms.txt` files reachable + valid (the 4 surfaced via `docs/conf.py:74-79`: `llms.txt`, `llms-full.txt`, `llms-practitioner.txt`, `llms-autonomous.txt`)
- JSON-LD in `docs/_templates/layout.html` validates per Schema.org
- Sitemap (`sphinx_sitemap` extension) covers autosummary stubs
- Canonical URLs correct on RTD vs local

**Findings:**
- (filled during Phase 2)

## Per-page score summary

Filled during Phase 2. Empty cells = not yet scored. Each cell `1-5` per the scoring scale above.

| # | Page | IA | Nav | VH | Code | Search | Mobile | A11y | Perf | AI-agent |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `index.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 2 | `quickstart.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 3 | `practitioner_getting_started.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 4 | `practitioner_decision_tree.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 5 | `choosing_estimator.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 6 | `troubleshooting.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 7 | `tutorials/01_basic_did.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 8 | `tutorials/13_stacked_did.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 9 | `api/estimators.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 10 | `api/_autosummary/diff_diff.CallawaySantAnna.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 11 | `references.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| 12 | `benchmarks.html` | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| - | **Average** | _ | _ | _ | _ | _ | _ | _ | _ | _ |

## Aggregate by category

| Category | Avg | Min page | Max page | Top 1-2 issues |
|---|---|---|---|---|
| IA | _ | _ | _ | _ |
| Nav | _ | _ | _ | _ |
| VH | _ | _ | _ | _ |
| Code | _ | _ | _ | _ |
| Search | _ | _ | _ | _ |
| Mobile | _ | _ | _ | _ |
| A11y | _ | _ | _ | _ |
| Perf | _ | _ | _ | _ |
| AI-agent | _ | _ | _ | _ |
