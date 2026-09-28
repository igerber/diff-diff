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
*!     (default avgeffectsby(Ei t) arm: overall + horizons 0..5, LOO + non-LOO;
*!      "variants" block: balanced-panel avgeffectsby(Ei) [== aux_partition="cohort",
*!      overall] and avgeffectsby(K) [== "horizon", overall + horizons 0..5];
*!      "unbalanced" block: deterministic subsample, all three partitions, overall)
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
*!   - No clean-sample reconstruction: did_imputation consumes the raw panel; the
*!     mappings are Ei = first_treat (missing for never-treated) and K = time - Ei
*!     (the relative time did_imputation groups on under avgeffectsby(K)).
*!   - Every avgeffectsby() partition is passed EXPLICITLY: Ei t (== library
*!     aux_partition="cohort_horizon", the default arm), Ei (== "cohort"), K (==
*!     "horizon"). did_imputation currently defaults to avgeffectsby(Ei t); pinning each
*!     explicitly keeps the validation estimand self-describing and robust to a future
*!     default change.
*!   - The unbalanced block re-runs all three partitions on a deterministic subsample
*!     (drop if mod(unit,4)==0 & time>=6; N=1305) because on the BALANCED panel the
*!     cohort partition is an arithmetic identity with the default (only v!=0 rows
*!     contribute per group), so only an unbalanced sample exercises it distinctly.
*!   - The in-.do point gates (1e-6, informational) check the Stata coef against the
*!     R-anchored points from didimputation_golden.json for the balanced arms, and
*!     against a library-computed point for the unbalanced block (no R anchor exists for
*!     the subsample) - both abort early on a gross bug. The AUTHORITATIVE parity gate is
*!     tests/test_imputation_loo_stata_parity.py, which compares the recomputed library
*!     output against this committed Stata golden.

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
gen K = time - Ei

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
* Balanced `horizon` variant: avgeffectsby(K) == library aux_partition="horizon".
* Overall + horizons 0..5, LOO + non-LOO. Points are partition-invariant, so the
* R-anchored rpt_* pins apply unchanged.
* ------------------------------------------------------------------------------
did_imputation y unit time Ei, leaveout avgeffectsby(K) cluster(unit)
scalar att_hz_all = _b[tau]
scalar se_hz_all  = _se[tau]
assert reldif(att_hz_all, rpt_all) < 1e-6

did_imputation y unit time Ei, avgeffectsby(K) cluster(unit)
scalar se_hz_all_nl = _se[tau]

did_imputation y unit time Ei, horizons(0/5) leaveout avgeffectsby(K) cluster(unit)
forvalues h = 0/5 {
    scalar att_hz_`h' = _b[tau`h']
    scalar se_hz_`h'  = _se[tau`h']
    assert reldif(att_hz_`h', rpt_`h') < 1e-6
}
did_imputation y unit time Ei, horizons(0/5) avgeffectsby(K) cluster(unit)
forvalues h = 0/5 {
    scalar se_hz_nl_`h' = _se[tau`h']
}

* ------------------------------------------------------------------------------
* Balanced `cohort` variant: avgeffectsby(Ei) == library aux_partition="cohort".
* Overall only - on this BALANCED panel the cohort partition is an arithmetic
* identity with the default (only v!=0 rows contribute per group, and the
* uniform-weight overall makes the cohort mean equal the mean of cell means), so
* the assert below pins the degeneracy (observed ~1e-12; standard 1e-6 bound for
* robustness to SSC solver drift). The distinct-cohort measurement lives in the
* unbalanced block.
* ------------------------------------------------------------------------------
did_imputation y unit time Ei, leaveout avgeffectsby(Ei) cluster(unit)
scalar att_co_all = _b[tau]
scalar se_co_all  = _se[tau]
assert reldif(att_co_all, rpt_all) < 1e-6
assert reldif(se_co_all, se_all) < 1e-6

did_imputation y unit time Ei, avgeffectsby(Ei) cluster(unit)
scalar se_co_all_nl = _se[tau]

* ------------------------------------------------------------------------------
* Unbalanced subsample: drop the tail periods of every 4th unit, then re-run all
* three partitions (overall only). This is where `cohort` genuinely diverges
* from the default (~23% larger SE). The point pin is LIBRARY-anchored (no R
* anchor exists for the subsample); informational - the Python test is
* authoritative.
* ------------------------------------------------------------------------------
scalar rpt_ub = 1.9647746414090286
preserve
drop if mod(unit, 4) == 0 & time >= 6
quietly count
assert r(N) == 1305

did_imputation y unit time Ei, leaveout avgeffectsby(Ei t) cluster(unit)
scalar att_ub_ch = _b[tau]
scalar se_ub_ch  = _se[tau]
assert reldif(att_ub_ch, rpt_ub) < 1e-6
did_imputation y unit time Ei, avgeffectsby(Ei t) cluster(unit)
scalar se_ub_ch_nl = _se[tau]

did_imputation y unit time Ei, leaveout avgeffectsby(Ei) cluster(unit)
scalar att_ub_co = _b[tau]
scalar se_ub_co  = _se[tau]
assert reldif(att_ub_co, rpt_ub) < 1e-6
did_imputation y unit time Ei, avgeffectsby(Ei) cluster(unit)
scalar se_ub_co_nl = _se[tau]

did_imputation y unit time Ei, leaveout avgeffectsby(K) cluster(unit)
scalar att_ub_hz = _b[tau]
scalar se_ub_hz  = _se[tau]
assert reldif(att_ub_hz, rpt_ub) < 1e-6
did_imputation y unit time Ei, avgeffectsby(K) cluster(unit)
scalar se_ub_hz_nl = _se[tau]
restore

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
file write `fh' `"    "point_anchor": "benchmarks/data/didimputation_golden.json (overall.att, event_study.att) for the balanced arms; library-computed (informational) for the unbalanced block","' _n
file write `fh' `"    "cmd": "did_imputation y unit time Ei, [horizons(0/5)] [leaveout] avgeffectsby(Ei t | Ei | K) cluster(unit); unbalanced block re-runs all three partitions on the subsample","' _n
file write `fh' `"    "avgeffectsby": "default arm: Ei t (== library aux_partition=cohort_horizon; pinned explicitly)","' _n
file write `fh' `"    "avgeffectsby_variants": "Ei == cohort, K == horizon (K = t - Ei); each pinned explicitly","' _n
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

* event_study block (comma-prefixed entries, no blank first line; the block close
* carries a trailing comma because the variants/unbalanced blocks follow)
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
file write `fh' _n "  }," _n

* variants block: balanced-panel avgeffectsby(Ei) / avgeffectsby(K)
file write `fh' `"  "variants": {"' _n
_jnum att_co_all
local a = r(s)
_jnum se_co_all
local s = r(s)
_jnum se_co_all_nl
local snl = r(s)
file write `fh' `"    "cohort": {"overall": {"att": `a', "se": `s', "se_nonloo": `snl'}},"' _n
_jnum att_hz_all
local a = r(s)
_jnum se_hz_all
local s = r(s)
_jnum se_hz_all_nl
local snl = r(s)
file write `fh' `"    "horizon": {"' _n
file write `fh' `"      "overall": {"att": `a', "se": `s', "se_nonloo": `snl'},"' _n
file write `fh' `"      "event_study": {"'
local sep ""
forvalues h = 0/5 {
    _jnum att_hz_`h'
    local a = r(s)
    _jnum se_hz_`h'
    local s = r(s)
    _jnum se_hz_nl_`h'
    local snl = r(s)
    file write `fh' "`sep'" _n `"        "`h'": {"att": `a', "se": `s', "se_nonloo": `snl'}"'
    local sep ","
}
file write `fh' _n "      }" _n
file write `fh' "    }" _n
file write `fh' "  }," _n

* unbalanced block: deterministic subsample, all three partitions, overall only
file write `fh' `"  "unbalanced": {"' _n
file write `fh' `"    "drop_rule": "mod(unit,4)==0 & time>=6","' _n
file write `fh' `"    "n_rows": 1305,"' _n
_jnum att_ub_ch
local a = r(s)
_jnum se_ub_ch
local s = r(s)
_jnum se_ub_ch_nl
local snl = r(s)
file write `fh' `"    "cohort_horizon": {"att": `a', "se": `s', "se_nonloo": `snl'},"' _n
_jnum att_ub_co
local a = r(s)
_jnum se_ub_co
local s = r(s)
_jnum se_ub_co_nl
local snl = r(s)
file write `fh' `"    "cohort": {"att": `a', "se": `s', "se_nonloo": `snl'},"' _n
_jnum att_ub_hz
local a = r(s)
_jnum se_ub_hz
local s = r(s)
_jnum se_ub_hz_nl
local snl = r(s)
file write `fh' `"    "horizon": {"att": `a', "se": `s', "se_nonloo": `snl'}"' _n
file write `fh' "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/didimputation_loo_stata_golden.json (default arm + variants + unbalanced)"
