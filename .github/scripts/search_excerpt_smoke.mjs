/* Behavioral smoke test for the search-excerpt anchor fix.
 *
 * Loads the BUILT Sphinx search machinery (doctools.js + searchtools.js) and
 * the shipped shims (searchtools-css-escape.js, legacy-fragment-redirect.js)
 * into a jsdom window, then drives the exact code path search results use -
 * Search.htmlToText(pageHtml, anchor) - against every section id in the
 * built tutorial HTML that is not a plain CSS identifier, and against a
 * synthetic page with one id per CSS-special class (apostrophe,
 * parentheses+comma, colon, legacy leading digit). Without the CSS.escape
 * wrapper each of these throws SyntaxError and the excerpt is dropped; the
 * test asserts a non-empty, section-scoped excerpt instead. Also asserts the
 * legacy-fragment redirect rewrites a pre-rename numbered hash.
 *
 * The real anchors are DISCOVERED from the built pages, never hard-coded:
 * tutorial headings get reworded, and a pinned heading would fail this test
 * for a reason that has nothing to do with search.
 *
 * Run from the repo root after a docs build (jsdom is pinned by the
 * committed package.json + package-lock.json next to this script):
 *   npm ci --prefix .github/scripts --ignore-scripts
 *   node .github/scripts/search_excerpt_smoke.mjs
 */
import { readFileSync, readdirSync } from "node:fs";
import { JSDOM } from "jsdom";

const ROOT = "docs/_build/html";
let failures = 0;

const check = (label, ok, detail) => {
  console.log(`${ok ? "PASS" : "FAIL"}: ${label}${detail ? ` - ${detail}` : ""}`);
  if (!ok) failures += 1;
};

// jsdom quirks vs browsers: window.CSS is absent (Part 1's env stub inlines
// a minimal spec-subset escaper so the wrapper's guard passes and its logic
// runs against jsdom's real selector engine; "\3<digit> " is the correct
// CSS hex escape for ASCII digits since '0'-'9' are 0x30-0x39; "&" is
// also emitted as its hex escape "\26 ", because jsdom's selector engine
// does not match the equally valid "\&" that browsers accept), and
// document.readyState stays "loading" until jsdom finishes its async load
// (so shims register DOMContentLoaded listeners - await before asserting).
const domLoaded = (window) =>
  new Promise((resolve) => {
    if (window.document.readyState !== "loading") resolve();
    else window.document.addEventListener("DOMContentLoaded", () => resolve());
  });

// --- Anchor discovery ----------------------------------------------------
// Section ids in the built tutorial pages that a bare `#id` selector cannot
// address (anything outside [A-Za-z0-9_-], or a leading digit), each with
// the text of its own heading. Sorted by file so the run is deterministic.
const TUTORIALS_DIR = `${ROOT}/tutorials`;
const NEEDS_ESCAPE = /[^A-Za-z0-9_-]|^\d/;

const headingText = (section) => {
  const heading = section.querySelector("h1, h2, h3, h4, h5, h6");
  if (!heading) return null;
  const clone = heading.cloneNode(true);
  clone.querySelectorAll(".headerlink").forEach((el) => el.remove());
  return clone.textContent.trim();
};

const discoverSpecialAnchors = () => {
  const found = [];
  const files = readdirSync(TUTORIALS_DIR).filter((f) => f.endsWith(".html")).sort();
  for (const file of files) {
    const html = readFileSync(`${TUTORIALS_DIR}/${file}`, "utf8");
    const { document } = new JSDOM(html).window;
    for (const section of document.querySelectorAll('[role="main"] section[id]')) {
      const heading = headingText(section);
      if (heading && NEEDS_ESCAPE.test(section.id)) {
        found.push({ file, html, id: section.id, heading });
      }
    }
  }
  return found;
};

const specialAnchors = discoverSpecialAnchors();
// A zero here means discovery broke (theme markup changed) or the tutorials
// no longer have a single CSS-special heading. Either way the real-page
// checks below would pass without testing anything, so fail and look.
check(
  "built tutorial pages contain CSS-special section ids to test",
  specialAnchors.length > 0,
  `${specialAnchors.length} found`,
);

// --- Part 0: production emit order on the built search page --------------
// The shim relies on being emitted BEFORE searchtools.js (it wraps at
// DOMContentLoaded, and listener order follows script order). Pin that
// order here so a theme/template change that flips it fails CI instead of
// silently racing the wrapper against the search runner.
{
  const searchPage = readFileSync(`${ROOT}/search.html`, "utf8");
  const shimAt = searchPage.indexOf("searchtools-css-escape.js");
  const toolsAt = searchPage.indexOf('src="_static/searchtools.js"');
  check(
    "search.html emits the shim before searchtools.js",
    shimAt !== -1 && toolsAt !== -1 && shimAt < toolsAt,
    `shim@${shimAt} searchtools@${toolsAt}`,
  );
}

// --- Part 1: Search.htmlToText excerpt rendering (production order) ------
// Reproduces the real page lifecycle instead of a settled-DOM eval: the
// scripts run as inline classic <script> tags in the PRODUCTION order -
// doctools, shim (Search undefined at its parse time), then searchtools -
// while document.readyState is "loading", so the shim and searchtools race
// through DOMContentLoaded exactly as on the live search page. If the
// wrapper loses that race, the special-anchor cases below throw.
// (The full query pipeline is not driven: performSearch defers without a
// loaded searchindex and fetches result pages; Search.htmlToText is the
// excerpt entry point either way.)
{
  const inline = (relPath) => {
    const src = readFileSync(`${ROOT}/${relPath}`, "utf8");
    if (src.includes("</script")) {
      throw new Error(`${relPath} contains "</script" - cannot inline safely`);
    }
    return `<script>${src}</script>`;
  };
  const envStub =
    "<script>window.DOCUMENTATION_OPTIONS = {}; window.CSS = { escape: " +
    "(value) => String(value).replace(/[\\0-,./:-@[-^`{-~]|^\\d/g, " +
    "(ch) => (ch >= '0' && ch <= '9' ? `\\\\3${ch} ` : " +
    "ch === '&' ? '\\\\26 ' : `\\\\${ch}`)) };" +
    "</script>";
  const harnessHtml = [
    "<!doctype html><html><head>",
    envStub,
    inline("_static/doctools.js"),
    inline("_static/searchtools-css-escape.js"),
    inline("_static/searchtools.js"),
    "</head><body></body></html>",
  ].join("\n");
  const dom = new JSDOM(harnessHtml, {
    url: "http://localhost/search.html",
    runScripts: "dangerously",
  });
  const { window } = dom;
  await domLoaded(window);
  const Search = window.eval("Search");

  const excerptCheck = (label, page, anchor, expectedPrefix) => {
    try {
      const text = Search.htmlToText(page, anchor);
      check(
        label,
        typeof text === "string" && text.trim().startsWith(expectedPrefix),
        text ? `got ${JSON.stringify(text.trim().slice(0, 60))}` : "empty result",
      );
    } catch (err) {
      check(label, false, `threw ${err}`);
    }
  };

  // Real pages: every discovered anchor. The expected prefix is the
  // section's own heading text (rendered prose, so a typographic apostrophe
  // where the id has a straight one): startsWith proves the excerpt is
  // section-scoped, not the whole-page fallback searchtools uses when the
  // anchor lookup finds nothing.
  for (const { file, html, id, heading } of specialAnchors) {
    excerptCheck(`section-scoped excerpt for ${file}#${id}`, html, `#${id}`, heading);
  }

  // Synthetic page: one id per CSS-special class, so each class stays
  // covered whatever the tutorials' headings currently are. The digit case
  // is a legacy anchor: built pages no longer have them (headings were
  // renumbered) but they still arrive via old inbound search/deep links,
  // and the wrapper must keep them from throwing.
  const syntheticCases = [
    ["Callaway-Sant'Anna-Estimator", "apostrophe"],
    ["Group-Time-Effects-ATT(g,t)", "parentheses and comma"],
    ["Understanding-Why-TWFE-Fails:-Goodman-Bacon-Decomposition", "colon"],
    ["3.-Legacy-Digit-Heading", "legacy leading digit"],
  ];
  const syntheticPage =
    '<div role="main">' +
    syntheticCases
      .map(
        ([id, kind]) =>
          `<section id="${id}"><h2>${kind} heading</h2><p>${kind} section body</p></section>`,
      )
      .join("") +
    "</div>";
  for (const [id, kind] of syntheticCases) {
    excerptCheck(
      `excerpt for synthetic ${kind} anchor`,
      syntheticPage,
      `#${id}`,
      `${kind} heading`,
    );
  }
}

// --- Part 2: legacy fragment redirect ------------------------------------
// Runs the shipped shim against a real built page: a numbered form of an id
// that exists must be rewritten to that id, and the id itself left alone.
// The target is the first discovered anchor, so it is also CSS-special.
if (specialAnchors.length) {
  const { file, html, id } = specialAnchors[0];
  const shim = readFileSync(`${ROOT}/_static/legacy-fragment-redirect.js`, "utf8");
  const hashAfterShim = async (fragment) => {
    const dom = new JSDOM(html, {
      url: `http://localhost/tutorials/${file}#${encodeURIComponent(fragment)}`,
      runScripts: "outside-only",
    });
    dom.window.eval(shim);
    await domLoaded(dom.window);
    return decodeURIComponent(dom.window.location.hash);
  };

  const redirected = await hashAfterShim(`3.-${id}`);
  check(
    `legacy numbered fragment redirects to renamed target (${file})`,
    redirected === `#${id}`,
    `hash is ${JSON.stringify(redirected)}`,
  );

  // A hash that still resolves (or matches nothing) must be left alone.
  const untouched = await hashAfterShim(id);
  check(
    "valid fragment left untouched",
    untouched === `#${id}`,
    `hash is ${JSON.stringify(untouched)}`,
  );
}

// --- Part 3: redirect targets are unambiguous ----------------------------
// The legacy redirect resolves a stripped fragment with getElementById, so
// its correctness rests on ids being unique per page. nbsphinx guarantees
// this today (repeated heading titles - "Background", "Summary" - get an id
// only on their FIRST occurrence, in the pre-rename state too, so no legacy
// numbered fragment ever pointed at a later duplicate). Pin the uniqueness
// so a toolchain change that starts emitting duplicate ids fails here
// instead of silently making redirects ambiguous.
{
  let dupPages = 0;
  for (const file of readdirSync(TUTORIALS_DIR).filter((f) => f.endsWith(".html"))) {
    const ids = [...readFileSync(`${TUTORIALS_DIR}/${file}`, "utf8").matchAll(/\sid="([^"]+)"/g)]
      .map((m) => m[1]);
    const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
    if (dupes.length) {
      dupPages += 1;
      console.log(`FAIL: duplicate ids in ${file}: ${[...new Set(dupes)].join(", ")}`);
    }
  }
  check("built tutorial pages have unique ids (unambiguous redirect targets)", dupPages === 0);
}

if (failures) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall search-excerpt smoke checks passed");
