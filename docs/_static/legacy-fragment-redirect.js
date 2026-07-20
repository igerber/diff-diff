/* Redirect pre-2026-07 numbered section fragments to their renamed targets.
 *
 * Tutorial headings used to carry numeric prefixes ("## 3. Fit Event Study"),
 * giving nbsphinx anchors like "#3.-Fit-Event-Study". The prefixes were
 * stripped (the anchors crashed search-excerpt rendering and the numbering
 * duplicated the grouped tutorials index), which renamed those fragments to
 * "#Fit-Event-Study". Page paths are unchanged; this shim keeps old deep
 * links working: when the requested fragment does not exist but the same id
 * without a leading "N.-" / "N.M.-" prefix does, it replaces the hash (no
 * extra history entry) so the browser scrolls to the renamed section.
 *
 * Repeated heading titles ("Background", "Summary") cannot make this
 * ambiguous: those duplicates were already unnumbered BEFORE the rename
 * (no "#N.-Background" fragment ever existed to redirect), and nbsphinx
 * emits an id only for the first occurrence, so getElementById has exactly
 * one possible target. The search smoke test pins per-page id uniqueness.
 */
(function () {
  function migrateLegacyFragment() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) return;
    var id;
    try {
      id = decodeURIComponent(hash.slice(1));
    } catch (err) {
      return;
    }
    if (document.getElementById(id)) return; // fragment still valid
    var m = id.match(/^\d+(?:\.\d+)*\.-(.+)$/);
    if (!m) return;
    var target = document.getElementById(m[1]);
    if (!target) return;
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", "#" + m[1]);
      if (typeof target.scrollIntoView === "function") target.scrollIntoView();
    } else {
      window.location.replace("#" + m[1]);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", migrateLegacyFragment);
  } else {
    migrateLegacyFragment();
  }
})();
