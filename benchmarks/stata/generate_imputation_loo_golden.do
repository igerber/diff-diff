*! Golden generator: ImputationDiD leave-one-out (A.9) SE vs Stata `did_imputation, leaveout`.
*!
*! Purpose
*!   Produce an INDEPENDENT external anchor for the ImputationDiD leave-one-out (LOO)
*!   standard error - the Borusyak-Jaravel-Spiess (2024) Supplementary Appendix A.9
*!   finite-sample variance refinement (opt-in `leave_one_out=True`). No R package
*!   computes it (R `didimputation` omits LOO), so the library LOO SE was validated
*!   only by an internal psi-identity + hand-calc + MC coverage. The authors' own
*!   Stata `did_imputation` ships the same option (`leaveout`); this generator turns
*!   that into a measured cross-implementation anchor.
*!
*!   SECOND Stata arm, and the FIRST SSC-dependent one. `did_imputation` is NOT
*!   pinned by `version 19` (SSC has no version history), so this generator records
*!   the installed package versions in meta.ssc_versions for drift detection. It does
*!   NOT install anything - run benchmarks/stata/requirements.do once first.
*!
*! Consuming test
*!   tests/test_imputation_loo_stata_parity.py
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/didimputation_loo_stata_golden.json
*!
*! Usage (run from the repo root, AFTER benchmarks/stata/requirements.do)
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!       benchmarks/stata/generate_imputation_loo_golden.do
*!   Then confirm the log is clean:  grep -E '^r\([0-9]+\);' generate_imputation_loo_golden.log
*!   (Stata batch mode ALWAYS exits 0, even on error - never trust the exit code.)
*!
*! Notes
*!   - Reads (does NOT regenerate) benchmarks/data/didimputation_test_panel.csv, whose
*!     sole owner is benchmarks/R/generate_didimputation_golden.R (180 units, cohorts
*!     {3,5} + never-treated, t=1..8). If that panel is ever regenerated, the R golden
*!     AND this Stata golden must both be regenerated.
*!   - No clean-sample reconstruction: did_imputation consumes the raw panel; the only
*!     mapping is Ei = first_treat (missing for never-treated).
*!   - avgeffectsby(Ei t) is passed EXPLICITLY (== library aux_partition="cohort_horizon").
*!     did_imputation currently defaults to avgeffectsby(Ei t) too; pinning it explicitly
*!     makes the validation estimand self-describing and robust to a future default change.
*!   - The in-.do point gate (1e-6, informational) checks the Stata coef against the
*!     R-anchored points from didimputation_golden.json and aborts early on a gross bug.
*!     The AUTHORITATIVE parity gate is tests/test_imputation_loo_stata_parity.py, which
*!     compares the recomputed library output against this committed Stata golden.

version 19
clear all
set more off
set type double

* Format a scalar as a JSON number at round-trip-exact precision. Stata's %21.17g
* renders |x|<1 as ".021"/"-.5" (leading dot); JSON requires a leading 0, so we
* patch ".x" -> "0.x" and "-.x" -> "-0.x". Returns r(s).
capture program drop _jnum
program define _jnum, rclass
    args x fmt
    if "`fmt'" == "" local fmt "%21.17g"
    local s = strtrim(string(`x', "`fmt'"))
    if substr("`s'", 1, 1) == "."      local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

* Extract a JSON-safe version string for an installed ado. Scans the leading `*!`
* header block and PREFERS a line containing "version" (SSC packages put the real
* version/date there - e.g. did_imputation's is on line 2, "Version: ...", after the
* line-1 description); falls back to the first `*!` line. Sanitized (quotes/backslashes
* /tabs stripped, truncated). Returns r(v)="MISSING" if the ado is absent.
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
    if `"`ver'"' == "" local ver "unknown"
    * JSON-sanitize: drop double-quotes/backslashes/tabs, truncate.
    local ver = subinstr(`"`ver'"', `"""', "'", .)
    local ver = subinstr(`"`ver'"', "\", "/", .)
    local ver = subinstr(`"`ver'"', char(9), " ", .)
    if length(`"`ver'"') > 100 local ver = substr(`"`ver'"', 1, 100)
    return local v `"`ver'"'
end

* ------------------------------------------------------------------------------
* Dependency check + version capture (do NOT install; see requirements.do).
* ------------------------------------------------------------------------------
foreach p in ftools require reghdfe did_imputation {
    capture which `p'
    if _rc {
        di as error "`p' is not installed. Run: stata -b do benchmarks/stata/requirements.do"
        error 111
    }
}
_adover ftools
local v_ftools = r(v)
_adover require
local v_require = r(v)
_adover reghdfe
local v_reghdfe = r(v)
_adover did_imputation
local v_didimp = r(v)
* Each captured identifier must be a real version/date line (not the bare description),
* else drift would go undetected (review: did_imputation's version is on the 2nd *! line).
assert strpos(lower(`"`v_didimp'"'), "version") > 0
assert strpos(lower(`"`v_reghdfe'"'), "version") > 0
assert strpos(lower(`"`v_ftools'"'),  "version") > 0
assert strpos(lower(`"`v_require'"'), "version") > 0

* ------------------------------------------------------------------------------
* R-anchored POINT estimates (benchmarks/data/didimputation_golden.json, digits=12).
* Hard-coded for the in-.do smoke gate only; the Python test reads the JSON and is
* authoritative. Points are identical for LOO and non-LOO (LOO only changes the SE).
* ------------------------------------------------------------------------------
scalar rpt_all =  2.045668026901
scalar rpt_0   =  0.9929366801146
scalar rpt_1   =  1.512964957441
scalar rpt_2   =  2.006499440277
scalar rpt_3   =  2.499854710124
scalar rpt_4   =  2.964817090473
scalar rpt_5   =  3.46735160262

* ------------------------------------------------------------------------------
* Load the committed panel; map first_treat -> Ei (missing for never-treated).
* ------------------------------------------------------------------------------
import delimited using "benchmarks/data/didimputation_test_panel.csv", clear varnames(1)
confirm numeric variable unit time first_treat y
quietly count
assert r(N) == 1440          // 180 units x 8 periods
gen Ei = first_treat
replace Ei = . if first_treat == 0

* ------------------------------------------------------------------------------
* Overall ATT: LOO + non-LOO. Coefficient is named `tau`.
* ------------------------------------------------------------------------------
did_imputation y unit time Ei, leaveout avgeffectsby(Ei t) cluster(unit)
scalar att_all = _b[tau]
scalar se_all  = _se[tau]
scalar N_all   = e(N)
assert reldif(att_all, rpt_all) < 1e-6

did_imputation y unit time Ei, avgeffectsby(Ei t) cluster(unit)
scalar se_all_nl = _se[tau]

* ------------------------------------------------------------------------------
* Event study horizons 0..5: LOO + non-LOO. Coefficients tau0..tau5.
* ------------------------------------------------------------------------------
did_imputation y unit time Ei, horizons(0/5) leaveout avgeffectsby(Ei t) cluster(unit)
forvalues h = 0/5 {
    scalar att_`h' = _b[tau`h']
    scalar se_`h'  = _se[tau`h']
    assert reldif(att_`h', rpt_`h') < 1e-6
}
did_imputation y unit time Ei, horizons(0/5) avgeffectsby(Ei t) cluster(unit)
forvalues h = 0/5 {
    scalar se_nl_`h' = _se[tau`h']
}

* ------------------------------------------------------------------------------
* Emit JSON by hand at %21.17g. Reuses the LPDiD conventions: every STRING field
* carries a trailing comma (the compound-quote `"' delimiter swallows a trailing
* double-quote), so the meta block ENDS with a numeric field.
* ------------------------------------------------------------------------------
local sver = strtrim(string(c(stata_version), "%4.1f"))
local sedition = cond(c(MP)==1, "MP", cond(c(SE)==1, "SE", "BE"))
tempname fh
file open `fh' using "benchmarks/data/didimputation_loo_stata_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "estimator": "ImputationDiD leave-one-out (BJS 2024 App. A.9) SE - Stata did_imputation leaveout","' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_imputation_loo_golden.do","' _n
file write `fh' `"    "source_panel": "benchmarks/data/didimputation_test_panel.csv","' _n
file write `fh' `"    "point_anchor": "benchmarks/data/didimputation_golden.json (overall.att, event_study.att)","' _n
file write `fh' `"    "cmd": "did_imputation y unit time Ei, [horizons(0/5)] leaveout avgeffectsby(Ei t) cluster(unit)","' _n
file write `fh' `"    "avgeffectsby": "Ei t (== library aux_partition=cohort_horizon; pinned explicitly)","' _n
file write `fh' `"    "se_convention": "A.9 leave-one-out finite-sample variance; unit-clustered; se_nonloo is the non-leaveout cluster SE (== R didimputation).","' _n
file write `fh' `"    "ssc_versions": {"' _n
file write `fh' `"      "did_imputation": "`v_didimp'","' _n
file write `fh' `"      "reghdfe": "`v_reghdfe'","' _n
file write `fh' `"      "ftools": "`v_ftools'","' _n
file write `fh' `"      "require": "`v_require'""' _n
file write `fh' `"    },"' _n
file write `fh' `"    "stata_edition": "`sedition'","' _n
file write `fh' `"    "stata_version": `sver'"' _n
file write `fh' "  }," _n

* overall block
_jnum att_all
local a = r(s)
_jnum se_all
local s = r(s)
_jnum se_all_nl
local snl = r(s)
_jnum N_all "%12.0f"
local nn = r(s)
file write `fh' `"  "overall": {"att": `a', "se": `s', "se_nonloo": `snl', "N": `nn'},"' _n

* event_study block (comma-prefixed; no trailing comma, no blank first line)
file write `fh' `"  "event_study": {"'
local sep ""
forvalues h = 0/5 {
    _jnum att_`h'
    local a = r(s)
    _jnum se_`h'
    local s = r(s)
    _jnum se_nl_`h'
    local snl = r(s)
    file write `fh' "`sep'" _n `"    "`h'": {"att": `a', "se": `s', "se_nonloo": `snl'}"'
    local sep ","
}
file write `fh' _n "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/didimputation_loo_stata_golden.json (overall + 6 horizons)"
