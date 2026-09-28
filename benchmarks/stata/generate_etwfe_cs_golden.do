*! Golden generator: WooldridgeDiD (ETWFE) and CallawaySantAnna ATT(g,t) vs the
*! canonical Stata implementations `jwdid` and `csdid`.
*!
*! Purpose
*!   Anchor BOTH staggered estimators against their reference implementations on
*!   the genuine Callaway-Sant'Anna `mpdta` panel, and MEASURE the ETWFE-vs-CS
*!   relationship instead of asserting it.
*!
*!   This arm exists because `tests/test_wooldridge.py` previously asserted that
*!   ETWFE ATT(g,t) EQUALS CallawaySantAnna ATT(g,t) within 5e-3. That claim is
*!   false on real data: at (g=2007, t=2007) the two estimators differ by 0.0171
*!   (-0.0431 vs -0.0261). The assertion only ever passed because the loader it
*!   read from was silently returning a synthetic, effect-homogeneous DGP (see
*!   issue #722 / PR #723) on which the two estimators do coincide.
*!
*!   Stata `jwdid` and `csdid` reproduce that SAME disagreement, which is what
*!   establishes the divergence as a property of the estimators rather than a bug
*!   in either implementation. This generator records both reference vectors so
*!   the Python side can pin each estimator to its own anchor.
*!
*! Consuming test
*!   tests/test_etwfe_cs_stata_parity.py
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/etwfe_cs_stata_golden.json
*!
*! Usage (run from the repo root)
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!       benchmarks/stata/generate_etwfe_cs_golden.do
*!   Then confirm the log is clean:  grep -E '^r\([0-9]+\);' generate_etwfe_cs_golden.log
*!   (Stata batch mode ALWAYS exits 0, even on error - never trust the exit code.)
*!
*! SSC dependencies (install via benchmarks/stata/requirements.do)
*!   drdid, csdid  - Callaway-Sant'Anna (Rios-Avila / Sant'Anna)
*!   jwdid, hdfe   - Wooldridge ETWFE (Rios-Avila); hdfe is a jwdid dependency
*!
*! Notes
*!   - Reads (does NOT regenerate) benchmarks/data/mpdta_stata_panel.csv, the
*!     SHA-256-verified upstream `mpdta.csv`
*!     (2283bea1221a152420f98dfa20f633c5d054ea51d881115c8cd702a97bcd3167). The
*!     panel is committed rather than fetched so this arm never depends on
*!     network availability - which is the exact failure mode that produced the
*!     false assertion in the first place.
*!   - Both commands are run WITHOUT covariates. `csdid`'s method is therefore
*!     immaterial (reg/ipw/dr coincide absent covariates); `method(reg)` is
*!     passed only to make the no-covariate path explicit.
*!   - SEs are emitted alongside the point estimates. The historical uniform SE
*!     gap vs `jwdid` (1.0280 at G=20 shrinking to 1.0010 at G=500) was defect
*!     D2 of the 3.9 variance program: the library's clustered CR1 factor used
*!     only the visible treatment-cell count, omitting the absorbed FE not
*!     nested in the unit cluster. Under the K_reference convergence
*!     (docs/methodology/variance-conventions.md) the SEs match reghdfe's at
*!     machine precision, and the `ladder` block below pins that agreement at
*!     every cluster count alongside reghdfe's own df accounting
*!     (df_a / rank / df_r), which the Python side cross-checks against
*!     absorbed_fe_cr1_k_increment.
*!   - The `never` arm is the external anchor for issue #724: `jwdid ... never`
*!     omits the `g-1` reference cell per cohort (W2025 Eq. 6.1/6.4), which the
*!     library previously left to QR rank detection.

version 19
clear all
set more off
set type double

* Format a scalar as a JSON number at round-trip-exact precision. Stata's %21.17g
* renders |x|<1 as ".455"/"-.119" (leading dot); JSON requires a leading 0, so we
* patch ".x" -> "0.x" and "-.x" -> "-0.x". Returns r(s).
capture program drop _jnum
program define _jnum, rclass
    args x fmt
    if "`fmt'" == "" local fmt "%21.17g"
    local s = strtrim(string(`x', "`fmt'"))
    if substr("`s'", 1, 1) == "."       local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

capture program drop _adover
program define _adover, rclass
    args pkg
    capture findfile `pkg'.ado
    if _rc {
        return local v "MISSING"
        exit
    }
    local fn = r(fn)
    tempname vh
    local ver ""
    local first ""
    local n = 0
    file open `vh' using "`fn'", read text
    file read `vh' line
    while r(eof) == 0 & `n' < 15 {
        if substr(`"`macval(line)'"', 1, 2) == "*!" {
            local body = strtrim(substr(`"`macval(line)'"', 3, .))
            if `"`first'"' == "" local first `"`body'"'
            if strpos(lower(`"`body'"'), "version") > 0 {
                local ver `"`body'"'
                continue, break
            }
        }
        local ++n
        file read `vh' line
    }
    file close `vh'
    if `"`ver'"' == "" local ver `"`first'"'
    * No parseable "*!" version header (drdid is one such package). Fall back to
    * a checksum of the .ado itself: an opaque identifier is still a usable DRIFT
    * signal, whereas "unknown" silently disables drift detection for that
    * dependency -- the SSC packages are unpinned, so this metadata is the only
    * thing standing between a silent upstream change and an unexplained golden
    * diff (codex R6 DT-2).
    if `"`ver'"' == "" {
        capture checksum "`fn'"
        if _rc == 0 {
            local ver "checksum:`r(checksum)' len:`r(filelen)'"
        }
        else {
            local ver "unknown"
        }
    }
    * JSON-sanitize: drop double-quotes/backslashes/tabs, truncate.
    local ver = subinstr(`"`ver'"', `"""', "'", .)
    local ver = subinstr(`"`ver'"', "\", "/", .)
    local ver = subinstr(`"`ver'"', char(9), " ", .)
    if length(`"`ver'"') > 100 local ver = substr(`"`ver'"', 1, 100)
    return local v `"`ver'"'
end


* ------------------------------------------------------------------------------
* Fail closed if an SSC dependency is missing: a silent skip here would emit a
* truncated golden, which is the same class of failure this arm exists to kill.
* ------------------------------------------------------------------------------
foreach p in drdid csdid jwdid hdfe {
    capture which `p'
    if _rc {
        display as error "Missing SSC package `p'. Run benchmarks/stata/requirements.do first."
        exit 111
    }
}

* Record installed SSC versions: SSC has no version history, so this is the only
* drift signal if a regenerated golden ever moves (README "Known constraints").
foreach p in drdid csdid jwdid hdfe {
    _adover `p'
    local v_`p' = r(v)
}

* ------------------------------------------------------------------------------
* Load the committed panel and confirm the expected schema.
* ------------------------------------------------------------------------------
import delimited using "benchmarks/data/mpdta_stata_panel.csv", clear varnames(1)
rename firsttreat first_treat
confirm numeric variable year countyreal lpop lemp first_treat treat
quietly count
assert r(N) == 2500
quietly levelsof countyreal, local(_units)
assert `: word count `_units'' == 500

xtset countyreal year

* ------------------------------------------------------------------------------
* CSDID: Callaway-Sant'Anna, not-yet-treated control, no covariates.
* Cell coefficients land in e(b) named  g<cohort>:t_<base>_<period>.
* ------------------------------------------------------------------------------
csdid lemp, ivar(countyreal) time(year) gvar(first_treat) notyet method(reg)
matrix cs_b = e(b)
matrix cs_V = e(V)
local cs_names : colnames cs_b
local cs_eqs   : coleq cs_b

* ------------------------------------------------------------------------------
* JWDID: Wooldridge ETWFE. Cell coefficients are named `<cohort>_<period>` in
* e(b) after the ATT(g,t) table is formed.
* ------------------------------------------------------------------------------
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)
matrix jw_b = e(b)
matrix jw_V = e(V)
local jw_names : colnames jw_b

* ------------------------------------------------------------------------------
* JWDID with `never`: never-treated controls only. This is the external anchor
* for issue #724 -- jwdid omits the g-1 reference cell per cohort (W2025
* Eq. 6.1/6.4), which is exactly what the library failed to do.
* ------------------------------------------------------------------------------
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) never
matrix jn_b = e(b)
matrix jn_V = e(V)
local jn_names : colnames jn_b

* ------------------------------------------------------------------------------
* JWDID on an ALL-EVENTUALLY-TREATED panel: the same pinned data with the
* never-treated counties dropped. External anchor for W2025 Section 5.4 -- the
* last cohort (2007) becomes the reference, so jwdid estimates only the earlier
* cohorts and reports a SMALLER N without comment. The library computes the same
* cell set, and warns.
*
* Plain `jwdid`, NOT `jwdid ... never`: there is no never-treated group left for
* the latter to use.
*
* preserve/restore so the subset cannot reach the schema asserts above or the
* three arms already estimated, which share this in-memory dataset. e(N) is
* captured because the row-count claim is the point of this arm and the golden
* otherwise records only coefficients and SEs.
* ------------------------------------------------------------------------------
preserve
drop if first_treat == 0
quietly levelsof countyreal, local(_at_units)
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)
matrix ja_b = e(b)
matrix ja_V = e(V)
local ja_names : colnames ja_b
local ja_N = e(N)
local ja_nunits : word count `_at_units'
restore

* ------------------------------------------------------------------------------
* Subsample LADDER: jwdid on nested rosters spanning G ~ 20..500, the artifact
* required by the K_reference convergence work (TODO row "Stata subsample
* ladder"). Roster rule (deterministic, reproduced verbatim by the Python
* side): the first N units per first_treat cohort by ASCENDING countyreal,
* N in {5, 10, 20, 40, 80, 200, 500}; cohort sizes are 309/20/40/131, so the
* rungs land at G = 20/40/80/140/220/391/500. Each rung records reghdfe's own
* df accounting (df_a and its initial/nested/redundant decomposition, rank,
* df_r) so the golden pins not just the SEs but the K-accounting that
* produces them - the Python test cross-checks df_a against
* absorbed_fe_cr1_k_increment per rung.
* ------------------------------------------------------------------------------
local ladder_rungs "5 10 20 40 80 200 500"
foreach N of local ladder_rungs {
    preserve
    keep countyreal first_treat
    duplicates drop
    bysort first_treat (countyreal): gen _idx = _n
    quietly keep if _idx <= `N'
    keep countyreal
    tempfile roster`N'
    quietly save `roster`N''
    restore, preserve
    quietly merge m:1 countyreal using `roster`N'', keep(match) nogenerate
    jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)
    matrix lad_b_`N' = e(b)
    matrix lad_V_`N' = e(V)
    local lad_names_`N' : colnames lad_b_`N'
    local lad_n_`N' = e(N)
    local lad_G_`N' = e(N_clust)
    local lad_dfa_`N' = e(df_a)
    local lad_dfai_`N' = e(df_a_initial)
    local lad_dfan_`N' = e(df_a_nested)
    local lad_dfared_`N' = e(df_a_redundant)
    local lad_rank_`N' = e(rank)
    local lad_dfr_`N' = e(df_r)
    restore
}
* Pin the roster rule itself: a changed rule or panel would move these.
assert `lad_G_5' == 20
assert `lad_G_10' == 40
assert `lad_G_20' == 80
assert `lad_G_40' == 140
assert `lad_G_80' == 220
assert `lad_G_200' == 391
assert `lad_G_500' == 500
assert `lad_n_500' == 2500

* ------------------------------------------------------------------------------
* Emit the golden.
* ------------------------------------------------------------------------------
local sver = c(stata_version)
local sedition = c(edition_real)

tempname fh
file open `fh' using "benchmarks/data/etwfe_cs_stata_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_etwfe_cs_golden.do","' _n
file write `fh' `"    "source_panel": "benchmarks/data/mpdta_stata_panel.csv","' _n
file write `fh' `"    "source_sha256": "2283bea1221a152420f98dfa20f633c5d054ea51d881115c8cd702a97bcd3167","' _n
file write `fh' `"    "ssc_versions": {"' _n
file write `fh' `"      "drdid": "`v_drdid'","' _n
file write `fh' `"      "csdid": "`v_csdid'","' _n
file write `fh' `"      "jwdid": "`v_jwdid'","' _n
file write `fh' `"      "hdfe": "`v_hdfe'""' _n
file write `fh' `"    },"' _n
file write `fh' `"    "stata_edition": "`sedition'","' _n
file write `fh' `"    "csdid_cmd": "csdid lemp, ivar(countyreal) time(year) gvar(first_treat) notyet method(reg)","' _n
file write `fh' `"    "jwdid_cmd": "jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)","' _n
file write `fh' `"    "jwdid_never_cmd": "jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) never","' _n
file write `fh' `"    "jwdid_alltreated_cmd": "drop if first_treat == 0; jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)","' _n
file write `fh' `"    "note": "No covariates, so csdid method is immaterial. ETWFE and CS genuinely DISAGREE at (2007,2007).","' _n
file write `fh' `"    "stata_version": `sver'"' _n
file write `fh' "  }," _n

* --- csdid cells -------------------------------------------------------------
file write `fh' `"  "csdid": {"'
local sep ""
local k = colsof(cs_b)
forvalues i = 1/`k' {
    local nm : word `i' of `cs_names'
    local eq : word `i' of `cs_eqs'
    local nm "`eq':`nm'"
    scalar bval = cs_b[1, `i']
    scalar seval = sqrt(cs_V[`i', `i'])
    if !missing(bval) {
        _jnum bval
        local a = r(s)
        _jnum seval
        local s = r(s)
        file write `fh' "`sep'" _n `"    "`nm'": {"att": `a', "se": `s'}"'
        local sep ","
    }
}
file write `fh' _n "  }," _n

* --- jwdid cells -------------------------------------------------------------
file write `fh' `"  "jwdid": {"'
local sep ""
local k = colsof(jw_b)
forvalues i = 1/`k' {
    local nm : word `i' of `jw_names'
    scalar bval = jw_b[1, `i']
    scalar seval = sqrt(jw_V[`i', `i'])
    if !missing(bval) {
        _jnum bval
        local a = r(s)
        _jnum seval
        local s = r(s)
        file write `fh' "`sep'" _n `"    "`nm'": {"att": `a', "se": `s'}"'
        local sep ","
    }
}
file write `fh' _n "  }," _n

* --- jwdid never-treated cells (issue #724 anchor) ---------------------------
file write `fh' `"  "jwdid_never": {"'
local sep ""
local k = colsof(jn_b)
forvalues i = 1/`k' {
    local nm : word `i' of `jn_names'
    scalar bval = jn_b[1, `i']
    scalar seval = sqrt(jn_V[`i', `i'])
    if !missing(bval) {
        _jnum bval
        local a = r(s)
        _jnum seval
        local s2 = r(s)
        file write `fh' "`sep'" _n `"    "`nm'": {"att": `a', "se": `s2'}"'
        local sep ","
    }
}
file write `fh' _n "  }," _n

* --- jwdid on the all-eventually-treated subset (W2025 Sec 5.4 anchor) --------
* Carries "n" because the row-count reduction IS the finding: jwdid drops the
* fully-treated periods silently and only reports a smaller N.
file write `fh' `"  "jwdid_alltreated": {"' _n
file write `fh' `"    "n": `ja_N',"' _n
file write `fh' `"    "n_units": `ja_nunits',"' _n
file write `fh' `"    "cells": {"'
local sep ""
local k = colsof(ja_b)
forvalues i = 1/`k' {
    local nm : word `i' of `ja_names'
    scalar bval = ja_b[1, `i']
    scalar seval = sqrt(ja_V[`i', `i'])
    if !missing(bval) {
        _jnum bval
        local a = r(s)
        _jnum seval
        local s3 = r(s)
        file write `fh' "`sep'" _n `"      "`nm'": {"att": `a', "se": `s3'}"'
        local sep ","
    }
}
file write `fh' _n "    }" _n
file write `fh' "  }," _n

* --- subsample ladder (K_reference gate at every cluster count) ---------------
file write `fh' `"  "ladder": {"' _n
file write `fh' `"    "roster_rule": "first N units per first_treat cohort by ascending countyreal","' _n
file write `fh' `"    "jwdid_cmd": "jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat)","' _n
file write `fh' `"    "rungs": {"'
local rsep ""
foreach N of local ladder_rungs {
    file write `fh' "`rsep'" _n `"      "`N'": {"' _n
    file write `fh' `"        "n_per_cohort": `N',"' _n
    file write `fh' `"        "G": `lad_G_`N'',"' _n
    file write `fh' `"        "n": `lad_n_`N'',"' _n
    file write `fh' `"        "df_a": `lad_dfa_`N'',"' _n
    file write `fh' `"        "df_a_initial": `lad_dfai_`N'',"' _n
    file write `fh' `"        "df_a_nested": `lad_dfan_`N'',"' _n
    file write `fh' `"        "df_a_redundant": `lad_dfared_`N'',"' _n
    file write `fh' `"        "rank": `lad_rank_`N'',"' _n
    file write `fh' `"        "df_r": `lad_dfr_`N'',"' _n
    file write `fh' `"        "cells": {"'
    local csep ""
    local k = colsof(lad_b_`N')
    forvalues i = 1/`k' {
        local nm : word `i' of `lad_names_`N''
        scalar bval = lad_b_`N'[1, `i']
        scalar seval = sqrt(lad_V_`N'[`i', `i'])
        if !missing(bval) {
            _jnum bval
            local a = r(s)
            _jnum seval
            local s4 = r(s)
            file write `fh' "`csep'" _n `"          "`nm'": {"att": `a', "se": `s4'}"'
            local csep ","
        }
    }
    file write `fh' _n "        }" _n
    file write `fh' "      }"
    local rsep ","
}
file write `fh' _n "    }" _n
file write `fh' "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/etwfe_cs_stata_golden.json"
