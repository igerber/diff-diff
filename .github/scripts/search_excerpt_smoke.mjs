/* Behavioral smoke test for the search-excerpt anchor fix.
 *
 * Loads the BUILT Sphinx search machinery (doctools.js + searchtools.js) and
 * the shipped shims (searchtools-css-escape.js, legacy-fragment-redirect.js)
 * into a jsdom window, then drives the exact code path search results use -
 * Search.htmlToText(pageHtml, anchor) - against built tutorial HTML for every
 * CSS-special anchor class (apostrophe, parentheses+comma, colon, and a
 * synthetic legacy digit anchor). Without the CSS.escape wrapper each of
 * these throws SyntaxError and the excerpt is dropped; the test asserts a
 * non-empty, section-scoped excerpt instead. Also asserts the
 * legacy-fragment redirect rewrites a pre-rename numbered hash.
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
// CSS hex escape for ASCII digits since '0'-'9' are 0x30-0x39), and
// document.readyState stays "loading" until jsdom finishes its async load
// (so shims register DOMContentLoaded listeners - await before asserting).
const domLoaded = (window) =>
  new Promise((resolve) => {
    if (window.document.readyState !== "loading") resolve();
    else window.document.addEventListener("DOMContentLoaded", () => resolve());
  });

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
    "(ch) => (ch >= '0' && ch <= '9' ? `\\\\3${ch} ` : `\\\\${ch}`)) };" +
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

  const page = readFileSync(`${ROOT}/tutorials/02_staggered_did.html`, "utf8");
  // Expected prefixes are the section heading TEXT (note the typographic
  // apostrophe in rendered prose vs the straight one in the id): asserting
  // startsWith proves the excerpt is section-scoped, not the whole-page
  // fallback searchtools uses when the anchor lookup finds nothing.
  const cases = [
    ["#Callaway-Sant'Anna-Estimator", "Callaway-Sant’Anna Estimator"],
    ["#Group-Time-Effects-ATT(g,t)", "Group-Time Effects ATT(g,t)"],
    [
      "#Understanding-Why-TWFE-Fails:-Goodman-Bacon-Decomposition",
      "Understanding Why TWFE Fails: Goodman-Bacon Decomposition",
    ],
    ["#Aggregating-Effects", "Aggregating Effects"],
  ];
  for (const [anchor, expectedPrefix] of cases) {
    try {
      const text = Search.htmlToText(page, anchor);
      check(
        `section-scoped excerpt for ${anchor}`,
        typeof text === "string" && text.trim().startsWith(expectedPrefix),
        text ? `got ${JSON.stringify(text.trim().slice(0, 60))}` : "empty result",
      );
    } catch (err) {
      check(`section-scoped excerpt for ${anchor}`, false, `threw ${err}`);
    }
  }

  // Legacy digit anchors no longer exist in built pages (headings were
  // renumbered) but still arrive via old inbound search/deep links; the
  // wrapper must keep them from throwing. Synthetic page keeps this case
  // covered independently of the notebook content.
  const legacyPage =
    '<div role="main"><section id="3.-Legacy-Digit-Heading"><h2>Legacy Digit Heading</h2>' +
    "<p>legacy digit section body</p></section></div>";
  try {
    const text = Search.htmlToText(legacyPage, "#3.-Legacy-Digit-Heading");
    check(
      "excerpt for synthetic legacy digit anchor",
      typeof text === "string" && text.includes("legacy digit section body"),
      text ? undefined : "empty result",
    );
  } catch (err) {
    check("excerpt for synthetic legacy digit anchor", false, `threw ${err}`);
  }
}

// --- Part 2: legacy fragment redirect ------------------------------------
{
  const page = readFileSync(`${ROOT}/tutorials/02_staggered_did.html`, "utf8");
  const legacyHash = "#3.-Callaway-Sant'Anna-Estimator";
  const dom = new JSDOM(page, {
    url: `http://localhost/tutorials/02_staggered_did.html${legacyHash}`,
    runScripts: "outside-only",
  });
  const { window } = dom;
  window.eval(readFileSync(`${ROOT}/_static/legacy-fragment-redirect.js`, "utf8"));
  await domLoaded(window);
  check(
    "legacy numbered fragment redirects to renamed target",
    window.location.hash === "#Callaway-Sant'Anna-Estimator",
    `hash is ${JSON.stringify(window.location.hash)}`,
  );

  // A hash that still resolves (or matches nothing) must be left alone.
  const dom2 = new JSDOM(page, {
    url: "http://localhost/tutorials/02_staggered_did.html#Aggregating-Effects",
    runScripts: "outside-only",
  });
  dom2.window.eval(
    readFileSync(`${ROOT}/_static/legacy-fragment-redirect.js`, "utf8"),
  );
  await domLoaded(dom2.window);
  check(
    "valid fragment left untouched",
    dom2.window.location.hash === "#Aggregating-Effects",
    `hash is ${JSON.stringify(dom2.window.location.hash)}`,
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
  const tutorialsDir = `${ROOT}/tutorials`;
  let dupPages = 0;
  for (const file of readdirSync(tutorialsDir).filter((f) => f.endsWith(".html"))) {
    const ids = [...readFileSync(`${tutorialsDir}/${file}`, "utf8").matchAll(/\sid="([^"]+)"/g)]
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
