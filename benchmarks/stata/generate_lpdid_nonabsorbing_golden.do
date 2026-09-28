*! Golden generator: LPDiD NON-ABSORBING SEs vs the authors' Stata `lpdid` package.
*!
*! Purpose
*!   Anchor the library's non-absorbing LP-DiD modes (Dube, Girardi, Jorda &
*!   Taylor 2025, JAE Eq. 12 / Eq. 13) against the authors' reference
*!   implementation, SSC `lpdid` (Busch & Girardi, in collaboration with Dube,
*!   Jorda and Taylor), END-TO-END - the package builds its own clean samples,
*!   unlike the sibling `teffects` arm which reconstructs them by hand. This is
*!   the first external anchor for the non-absorbing REWEIGHTED SE (previously
*!   pinned-only via RW_SE_PIN with a ~5e-5 feols weighted-cluster convention
*!   gap), for the non-absorbing POOLED windows (points and SEs), and for the
*!   Eq. 12 reweighted point.
*!
*! Mapping (pinned in meta and gated by the consuming test)
*!   lpdid, nonabsorbing(, firsttreat notyet)  ==  LPDiD(non_absorbing="first_entry")
*!   lpdid, nonabsorbing(L)                    ==  LPDiD(non_absorbing="effect_stabilization",
*!                                                       stabilization_window=L)
*!
*! Arms
*!   A: Eq. 12 (firsttreat notyet), FULL committed panel, vw + rw:
*!      all ES horizons (incl. placebos) + pooled Pre/Post. Full parity surface.
*!   B: Eq. 13 (nonabsorbing(3)), convention-neutral 47-unit SUBSAMPLE, vw + rw:
*!      post horizons + pooled Post are parity surfaces; pre rows + pooled Pre
*!      are recorded (att + obs only) as measured DIVERGENCE documentation --
*!      the package builds placebo samples by recursive lagged intersection of
*!      CCS_0 while the library uses the backward window [t-max(L,-h), t-1]
*!      (a paper-silent surface; see REGISTRY ## LPDiD Deviation 4).
*!   C: Eq. 13, FULL panel, vw only, event study only (att + obs): measured
*!      divergence documentation for the two sample-admission convention
*!      differences (pre-panel boundary handling; exact-L re-entry spells).
*!
*! Subsample drop rule (computable identically here and in pandas; gate 5 of the
*! consuming test asserts both sub-classes):
*!   drop units with min(treat)==1                      (always-treated: 31..40)
*!   drop units with any(dD==1 & L3.dD==-1), dD=D.treat (exact-L respell: 24,25,27)
*!   -> 47 units / 658 rows.
*!
*! Consuming test
*!   tests/test_lpdid_nonabsorbing_stata_parity.py
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/lpdid_nonabsorbing_stata_golden.json
*!
*! Usage (run from the repo root; install SSC deps ONCE via
*! benchmarks/stata/requirements.do first - this generator fails closed)
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!       benchmarks/stata/generate_lpdid_nonabsorbing_golden.do
*!   Then confirm the log is clean:
*!       grep -E '^r\([0-9]+\);' generate_lpdid_nonabsorbing_golden.log
*!   (Stata batch mode ALWAYS exits 0, even on error - never trust the exit code.)
*!
*! Notes
*!   - Reads (does NOT regenerate) benchmarks/data/lpdid_nonabsorbing_panel.csv,
*!     whose sole owner is benchmarks/R/generate_lpdid_golden.R (Section 5).
*!     60 units x 14 periods, balanced, gap-free.
*!   - SSC packages are unpinned; meta.ssc_versions records the installed
*!     versions of every fail-closed dependency as the only drift signal.
*!   - In-.do point gates are informational; the Python test is authoritative.
*!     Arm A vw tau0 is gated against the committed R golden's first_entry["0"]
*!     att (R-anchored); Arm B vw tau0 against the library-computed point
*!     (library-anchored - no R value exists for the subsample).

version 19
clear all
set more off
set type double

* Format a scalar as a JSON number at round-trip-exact precision. Stata's %21.17g
* renders |x|<1 as ".455"/"-.119" (leading dot); JSON requires a leading 0, so we
* patch ".x" -> "0.x" and "-.x" -> "-0.x". Every value this arm records is
* expected FINITE, so a missing scalar fails the generator loudly (string(.) is
* "." which the leading-dot patch would turn into invalid JSON "0.") instead of
* silently corrupting the golden. Returns r(s).
capture program drop _jnum
program define _jnum, rclass
    args x fmt
    if missing(`x') {
        display as error "_jnum: missing value where a finite scalar was expected"
        exit 9
    }
    if "`fmt'" == "" local fmt "%21.17g"
    local s = strtrim(string(`x', "`fmt'"))
    if substr("`s'", 1, 1) == "."       local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

* Installed-version capture for an unpinned SSC dependency. Widened from the
* ETWFE arm's _adover (generate_etwfe_cs_golden.do): this arm's own packages
* defeat a "*!"-only parse - lpdid.ado's header is a PLAIN-`*` comment
* ("* lpdid program, version 1.0.3"; no "*!" anywhere) and _gfilter.ado's "*!"
* line carries no "version" token ("*! 1.1.1 NJC 19 March 2006"). Preference
* order: a "*!" line containing "version"; else the FIRST "*!" line (boottest's
* "*! boottest 4.5.3 ..." must beat the GPL boilerplate "version 3 of the
* License" plain-comment line below it); else the first PLAIN comment line
* containing "version" (lpdid has no "*!" at all - its header is
* "* lpdid program, version 1.0.3"); else the checksum alone. The file's
* checksum + length are ALWAYS appended to whatever version text was found
* (SSC has no immutable archive, so a same-version-string upstream edit would
* otherwise be invisible in the drift metadata). Takes the FILE basename to
* probe (may differ from the package name, e.g. egenmore -> _gfilter).
capture program drop _adover
program define _adover, rclass
    args probefile
    capture findfile `probefile'.ado
    if _rc {
        return local v "MISSING"
        exit
    }
    local fn = r(fn)
    tempname vh
    local ver ""
    local cver ""
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
        else if substr(strtrim(`"`macval(line)'"'), 1, 1) == "*" {
            local body = strtrim(substr(strtrim(`"`macval(line)'"'), 2, .))
            if `"`cver'"' == "" & strpos(lower(`"`body'"'), "version") > 0 {
                local cver `"`body'"'
            }
        }
        local ++n
        file read `vh' line
    }
    file close `vh'
    if `"`ver'"' == "" local ver `"`first'"'
    if `"`ver'"' == "" local ver `"`cver'"'
    * ALWAYS append the file checksum + length: SSC has no immutable archive, so a
    * same-version-string replacement of an ado would otherwise be invisible in
    * the golden's drift metadata (local review R2).
    capture checksum "`fn'"
    if _rc {
        display as error "_adover: checksum failed for `fn' - the drift signal cannot be recorded"
        exit 9
    }
    local csum "checksum:`r(checksum)' len:`r(filelen)'"
    if `"`ver'"' == "" {
        local ver `"`csum'"'
    }
    else {
        local ver `"`ver' [`csum']"'
    }
    * JSON-sanitize: drop double-quotes/backslashes/tabs, truncate (cap covers the
    * version text plus the always-appended [checksum:... len:...] suffix).
    local ver = subinstr(`"`ver'"', `"""', "'", .)
    local ver = subinstr(`"`ver'"', "\", "/", .)
    local ver = subinstr(`"`ver'"', char(9), " ", .)
    if length(`"`ver'"') > 160 local ver = substr(`"`ver'"', 1, 160)
    return local v `"`ver'"'
end

* ------------------------------------------------------------------------------
* Fail closed if ANY run-time dependency is missing: a silent skip would emit a
* truncated golden, and a missing egenmore helper would otherwise surface as an
* opaque mid-run error inside lpdid. (_gfilter is executed by every lpdid pooled
* spec; _gclsst is lpdid's own startup which-check; both ship in SSC egenmore.)
* ------------------------------------------------------------------------------
foreach p in lpdid reghdfe ftools require boottest listreg _gfilter _gclsst {
    capture which `p'
    if _rc {
        display as error "Missing SSC dependency `p'. Run benchmarks/stata/requirements.do first."
        exit 111
    }
}

* Record installed SSC versions for every fail-closed dependency (guard set ==
* version set, matching the ETWFE-arm contract). Both guarded egenmore files are
* recorded: `egenmore` via _gfilter.ado (executed by the pooled spec) and
* `egenmore_gclsst` via _gclsst.ado (lpdid's startup which-check) - a change to
* either file must move the drift metadata.
foreach p in lpdid reghdfe ftools require boottest listreg {
    _adover `p'
    local v_`p' = r(v)
}
_adover _gfilter
local v_egenmore = r(v)
_adover _gclsst
local v_egenmore_gclsst = r(v)

* ------------------------------------------------------------------------------
* Informational in-.do point pins (fatal asserts at reldif < 1e-6, loose enough
* to survive SSC solver drift; the Python test is the authoritative gate).
*   rpt_fe : committed R golden first_entry["0"] att (R-anchored;
*            benchmarks/data/lpdid_nonabsorbing_golden.json, digits=12)
*   rpt_es : library-computed Eq. 13 subsample vw tau0 (library-anchored,
*            informational - no R value exists for the subsample)
* ------------------------------------------------------------------------------
scalar rpt_fe = 1.846815337457
scalar rpt_es = 1.8115854492

* ------------------------------------------------------------------------------
* Load the committed panel and confirm the expected schema/shape (balanced,
* gap-free; the library's non-absorbing modes require it, so the golden must be
* generated from the same grid).
* ------------------------------------------------------------------------------
import delimited using "benchmarks/data/lpdid_nonabsorbing_panel.csv", varnames(1) clear
confirm numeric variable unit time treat y
quietly count
assert r(N) == 840          // 60 units x 14 periods
bysort unit: gen _n_obs = _N
assert _n_obs == 14         // balanced: every unit observed at every period
drop _n_obs

* ------------------------------------------------------------------------------
* Arm A: Eq. 12 (first_entry), FULL panel, vw + rw.
* e(results) rows: pre3 pre2 pre1 tau0..tau4 (pre1 = zero reference, skipped).
* Columns: coefficient se t p ci_low ci_high obs -> capture 1, 2, 7.
* ------------------------------------------------------------------------------
foreach mode in vw rw {
    local rwopt = cond("`mode'" == "rw", "rw", "")
    preserve
        lpdid y, unit(unit) time(time) treat(treat) pre_window(3) post_window(4) ///
            nonabsorbing(, firsttreat notyet) `rwopt' nograph
        matrix J = e(results)
        matrix P = e(pooled_results)
        foreach h of numlist 3 2 {
            scalar fe_`mode'_att_m`h' = J[4 - `h', 1]
            scalar fe_`mode'_se_m`h'  = J[4 - `h', 2]
            scalar fe_`mode'_N_m`h'   = J[4 - `h', 7]
        }
        foreach h of numlist 0 1 2 3 4 {
            scalar fe_`mode'_att_`h' = J[4 + `h', 1]
            scalar fe_`mode'_se_`h'  = J[4 + `h', 2]
            scalar fe_`mode'_N_`h'   = J[4 + `h', 7]
        }
        foreach w in Pre Post {
            local r = cond("`w'" == "Pre", 1, 2)
            local wl = lower("`w'")
            scalar fe_`mode'_att_p`wl' = P[`r', 1]
            scalar fe_`mode'_se_p`wl'  = P[`r', 2]
            scalar fe_`mode'_N_p`wl'   = P[`r', 7]
        }
    restore
}
* in-.do smoke gate: Arm A vw tau0 vs the committed R golden value.
assert reldif(fe_vw_att_0, rpt_fe) < 1e-6

* ------------------------------------------------------------------------------
* Arm B: Eq. 13 (effect stabilization, L=3), convention-neutral SUBSAMPLE,
* vw + rw. Drop rule (see header); the variable is named dD, never bare D, so it
* cannot shadow the difference-operator prefix.
* ------------------------------------------------------------------------------
preserve
    tsset unit time
    gen dD = D.treat
    bysort unit: egen alw = min(treat)
    gen respell = (dD == 1 & L3.dD == -1)
    bysort unit: egen dropu = max(respell)
    drop if alw == 1 | dropu == 1
    quietly count
    assert r(N) == 658          // 47 units x 14 periods
    drop dD alw respell dropu
    foreach mode in vw rw {
        local rwopt = cond("`mode'" == "rw", "rw", "")
        lpdid y, unit(unit) time(time) treat(treat) pre_window(3) post_window(4) ///
            nonabsorbing(3) `rwopt' nograph
        matrix J = e(results)
        matrix P = e(pooled_results)
        * divergent placebo rows: att + obs only (documentation, gate 6b)
        foreach h of numlist 3 2 {
            scalar es_`mode'_att_m`h' = J[4 - `h', 1]
            scalar es_`mode'_N_m`h'   = J[4 - `h', 7]
        }
        * parity-gated post rows: att + se + obs
        foreach h of numlist 0 1 2 3 4 {
            scalar es_`mode'_att_`h' = J[4 + `h', 1]
            scalar es_`mode'_se_`h'  = J[4 + `h', 2]
            scalar es_`mode'_N_`h'   = J[4 + `h', 7]
        }
        scalar es_`mode'_att_ppre  = P[1, 1]
        scalar es_`mode'_N_ppre    = P[1, 7]
        scalar es_`mode'_att_ppost = P[2, 1]
        scalar es_`mode'_se_ppost  = P[2, 2]
        scalar es_`mode'_N_ppost   = P[2, 7]
    }
    * in-.do smoke gate: Arm B vw tau0 vs the library-computed point
    * (library-anchored, informational).
    assert reldif(es_vw_att_0, rpt_es) < 1e-6
restore

* ------------------------------------------------------------------------------
* Arm C: Eq. 13, FULL panel, vw only, event study only. Divergence documentation
* (att + obs; gate 6a asserts both the att divergence and the obs mismatch that
* locks the boundary/respell sample-admission conventions).
* ------------------------------------------------------------------------------
quietly count
assert r(N) == 840          // guard: Arm B's restore really returned the full panel
lpdid y, unit(unit) time(time) treat(treat) pre_window(3) post_window(4) ///
    nonabsorbing(3) nograph only_event
matrix J = e(results)
foreach h of numlist 0 1 2 3 4 {
    scalar ef_att_`h' = J[4 + `h', 1]
    scalar ef_N_`h'   = J[4 + `h', 7]
}

* ------------------------------------------------------------------------------
* Emit JSON by hand (Stata has no jsonlite) at %21.17g (round-trip-exact double).
* No timestamp (byte-identical regeneration).
* ------------------------------------------------------------------------------
* Stata compound-quote note: the `"' delimiter swallows a trailing double-quote,
* so every STRING field is written with a trailing comma (`...","' -> `...",`);
* the meta block therefore ENDS with numeric fields, which have no trailing quote.
local sver = strtrim(string(c(stata_version), "%4.1f"))
* c(flavor) misreports "IC" on SE and c(edition) is unreliable; c(MP)/c(SE) are
* the authoritative 0/1 flags (c(BE) is UNDEFINED, and cond() evaluates all
* branches eagerly, so BE is resolved by elimination, not referenced).
local sedition = cond(c(MP)==1, "MP", cond(c(SE)==1, "SE", "BE"))

* Helper: write one ES entry with att/se/N. Assembles trimmed locals first.
tempname fh
file open `fh' using "benchmarks/data/lpdid_nonabsorbing_stata_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "estimator": "LPDiD non-absorbing SEs - authors' Stata lpdid package, end-to-end","' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_lpdid_nonabsorbing_golden.do","' _n
file write `fh' `"    "source_panel": "benchmarks/data/lpdid_nonabsorbing_panel.csv","' _n
file write `fh' `"    "source_sha256": "71af98c96dec08abc281002b7b88df74033dac7d0ff7707eff9f3ee0c30a3964","' _n
file write `fh' `"    "cmd": "lpdid y, unit(unit) time(time) treat(treat) pre_window(3) post_window(4) nograph + nonabsorbing(, firsttreat notyet) [rw] on the full panel; nonabsorbing(3) [rw] on the subsample and (only_event, vw) on the full panel","' _n
file write `fh' `"    "mapping": "nonabsorbing(, firsttreat notyet) == first_entry; nonabsorbing(L) == effect_stabilization","' _n
file write `fh' `"    "drop_rule": "min(treat)==1 | any(dD==1 & L3.dD==-1)","' _n
file write `fh' `"    "point_anchor": "first_entry vw tau0 gated in-.do vs the committed R golden first_entry[0] att (R-anchored); effect_stab_sub vw tau0 vs the library point (library-anchored, informational); the Python test is authoritative for every surface","' _n
file write `fh' `"    "se_convention": "package reghdfe vce(cluster unit) CR1; the library's cluster SE matches at ~1e-16 (vw) / ~1e-9 (rw) on identical samples","' _n
file write `fh' `"    "convention_notes": "effect_stab full-panel and placebo/pooled-pre rows are DIVERGENCE DOCUMENTATION, not parity: the package admits always-treated units at early t via missing-lag semantics, requires L+1 untreated periods before re-entry (stricter than JAE Eq. 13's levels condition), and builds placebo samples by recursive lagged intersection; see REGISTRY ## LPDiD Deviation 4","' _n
file write `fh' `"    "ssc_versions": {"' _n
file write `fh' `"      "lpdid": "`v_lpdid'","' _n
file write `fh' `"      "reghdfe": "`v_reghdfe'","' _n
file write `fh' `"      "ftools": "`v_ftools'","' _n
file write `fh' `"      "require": "`v_require'","' _n
file write `fh' `"      "boottest": "`v_boottest'","' _n
file write `fh' `"      "listreg": "`v_listreg'","' _n
file write `fh' `"      "egenmore": "`v_egenmore'","' _n
file write `fh' `"      "egenmore_gclsst": "`v_egenmore_gclsst'""' _n
file write `fh' "    }," _n
file write `fh' `"    "dropped_units": [24, 25, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],"' _n
file write `fh' `"    "stata_edition": "`sedition'","' _n
file write `fh' `"    "stata_version": `sver',"' _n
file write `fh' `"    "pre_window": 3,"' _n
file write `fh' `"    "post_window": 4,"' _n
file write `fh' `"    "stabilization_window": 3,"' _n
file write `fh' `"    "n_rows_full": 840,"' _n
file write `fh' `"    "n_rows_sub": 658"' _n
file write `fh' "  }," _n

* --- first_entry block: vw + rw, full ES + pooled, all fields ---
file write `fh' `"  "first_entry": {"' _n
local firstmode 1
foreach mode in vw rw {
    if !`firstmode' file write `fh' "," _n
    local firstmode 0
    file write `fh' `"    "`mode'": {"' _n
    file write `fh' `"      "es": {"'
    local sep ""
    foreach h of numlist 3 2 {
        _jnum fe_`mode'_att_m`h'
        local a = r(s)
        _jnum fe_`mode'_se_m`h'
        local s = r(s)
        _jnum fe_`mode'_N_m`h' "%12.0f"
        local nn = r(s)
        file write `fh' "`sep'" _n `"        "-`h'": {"att": `a', "se": `s', "N": `nn'}"'
        local sep ","
    }
    foreach h of numlist 0 1 2 3 4 {
        _jnum fe_`mode'_att_`h'
        local a = r(s)
        _jnum fe_`mode'_se_`h'
        local s = r(s)
        _jnum fe_`mode'_N_`h' "%12.0f"
        local nn = r(s)
        file write `fh' "`sep'" _n `"        "`h'": {"att": `a', "se": `s', "N": `nn'}"'
        local sep ","
    }
    file write `fh' _n "      }," _n
    file write `fh' `"      "pooled": {"' _n
    foreach w in pre post {
        _jnum fe_`mode'_att_p`w'
        local a = r(s)
        _jnum fe_`mode'_se_p`w'
        local s = r(s)
        _jnum fe_`mode'_N_p`w' "%12.0f"
        local nn = r(s)
        local wsep = cond("`w'" == "pre", ",", "")
        file write `fh' `"        "`w'": {"att": `a', "se": `s', "N": `nn'}`wsep'"' _n
    }
    file write `fh' "      }" _n
    file write `fh' "    }"
}
file write `fh' _n "  }," _n

* --- effect_stab_sub block: vw + rw; post rows full fields, pre rows att+N ---
file write `fh' `"  "effect_stab_sub": {"' _n
file write `fh' `"    "n_rows": 658,"' _n
local firstmode 1
foreach mode in vw rw {
    if !`firstmode' file write `fh' "," _n
    local firstmode 0
    file write `fh' `"    "`mode'": {"' _n
    file write `fh' `"      "es": {"'
    local sep ""
    foreach h of numlist 3 2 {
        _jnum es_`mode'_att_m`h'
        local a = r(s)
        _jnum es_`mode'_N_m`h' "%12.0f"
        local nn = r(s)
        file write `fh' "`sep'" _n `"        "-`h'": {"att": `a', "N": `nn'}"'
        local sep ","
    }
    foreach h of numlist 0 1 2 3 4 {
        _jnum es_`mode'_att_`h'
        local a = r(s)
        _jnum es_`mode'_se_`h'
        local s = r(s)
        _jnum es_`mode'_N_`h' "%12.0f"
        local nn = r(s)
        file write `fh' "`sep'" _n `"        "`h'": {"att": `a', "se": `s', "N": `nn'}"'
        local sep ","
    }
    file write `fh' _n "      }," _n
    file write `fh' `"      "pooled": {"' _n
    _jnum es_`mode'_att_ppre
    local a = r(s)
    _jnum es_`mode'_N_ppre "%12.0f"
    local nn = r(s)
    file write `fh' `"        "pre": {"att": `a', "N": `nn'},"' _n
    _jnum es_`mode'_att_ppost
    local a = r(s)
    _jnum es_`mode'_se_ppost
    local s = r(s)
    _jnum es_`mode'_N_ppost "%12.0f"
    local nn = r(s)
    file write `fh' `"        "post": {"att": `a', "se": `s', "N": `nn'}"' _n
    file write `fh' "      }" _n
    file write `fh' "    }"
}
file write `fh' _n "  }," _n

* --- effect_stab_full_vw block: att + N only ---
file write `fh' `"  "effect_stab_full_vw": {"' _n
file write `fh' `"    "es": {"'
local sep ""
foreach h of numlist 0 1 2 3 4 {
    _jnum ef_att_`h'
    local a = r(s)
    _jnum ef_N_`h' "%12.0f"
    local nn = r(s)
    file write `fh' "`sep'" _n `"      "`h'": {"att": `a', "N": `nn'}"'
    local sep ","
}
file write `fh' _n "    }" _n
file write `fh' "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/lpdid_nonabsorbing_stata_golden.json (arms A/B/C)"
