/* Make search-result excerpts survive CSS-special anchor ids.
 *
 * Sphinx's searchtools.js renders each search-result excerpt by fetching the
 * target page and running document.querySelector('[role="main"] ' + anchor)
 * with the raw section anchor (htmlToText). nbsphinx keeps raw heading text
 * in notebook anchors, so ids like "#Callaway-Sant'Anna-Estimator",
 * "#Group-Time-Effects-ATT(g,t)" or "#Understanding-Why-TWFE-Fails:-..."
 * are not valid CSS selectors: querySelector throws SyntaxError and the
 * excerpt is silently dropped from the results list.
 *
 * Wrapping Search.htmlToText and passing the id through CSS.escape() makes
 * every anchor a valid selector. Already-valid ids escape to themselves, so
 * this is a no-op for normal pages.
 *
 * Load-order note: search.html emits this file BEFORE searchtools.js (custom
 * js files render with the early script group; searchtools.js comes from the
 * search page template), so Search does not exist yet at parse time. The
 * wrap is deferred to DOMContentLoaded, which runs registered-order-first:
 * this listener is registered before searchtools.js registers its own
 * DOMContentLoaded query runner, so the wrap is in place before any search
 * executes.
 */
(function () {
  function wrapHtmlToText() {
    if (
      typeof Search === "undefined" ||
      !Search.htmlToText ||
      !window.CSS ||
      !CSS.escape
    ) {
      return;
    }
    var origHtmlToText = Search.htmlToText.bind(Search);
    Search.htmlToText = function (htmlString, anchor) {
      if (anchor && anchor.charAt(0) === "#") {
        anchor = "#" + CSS.escape(anchor.slice(1));
      }
      return origHtmlToText(htmlString, anchor);
    };
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wrapHtmlToText);
  } else {
    wrapHtmlToText();
  }
})();
