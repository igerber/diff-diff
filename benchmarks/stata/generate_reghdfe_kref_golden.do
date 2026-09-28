*! Golden generator: reghdfe K_reference anchor on a DISCONNECTED two-way panel.
*!
*! Purpose
*!   External anchor for the K_reference clustered-CR1 accounting on an
*!   IRREGULAR (disconnected) design, in two arms:
*!
*!   cross_cluster -- the PARITY anchor for the exact non-nested RANK term:
*!   with nothing nested in the cluster, reghdfe's pairwise dof method
*!   computes the exact span rank (df_a = 28 = U + T - C, denominator
*!   N - 29), agreeing with fixest ssc(K.fixef="full", K.exact=TRUE) and
*!   with the library at machine precision.
*!
*!   unit_cluster -- a DOCUMENTATION arm for the nested COMPOSITION: with
*!   unit nested-dropped, reghdfe counts the time REMAINDER at the
*!   per-dim approximation T - 1 = 9 (its pairwise correction only covers
*!   pairs where both dims survive), implying K = rank + df_a + constant
*!   = 11. The library's K_reference counts the remainder at its exact
*!   rank GIVEN the nested span (28 - 20 = 8, so K = 10) -- the consistent
*!   extension of the exact-rank principle both references apply when
*!   nothing is nested. NO external reference implements that composition
*!   (fixest's K.exact composes incoherently with its nested drop, see
*!   benchmarks/R/generate_fixest_cr1_nonnested_golden.R), so this arm pins
*!   the DEVIATION exactly: se_reghdfe / se_library ==
*!   sqrt((N - 10) / (N - 11)), one df, and the df_a decomposition records
*!   why. On connected designs (exact == approx remainder) the two agree at
*!   machine precision -- the jwdid subsample ladder
*!   (etwfe_cs_stata_golden.json) pins that at every cluster count.
*!
*! Design (deterministic, NO RNG -- the Python side rebuilds the frame from
*! the same integer formulas, so no data needs to be embedded or shipped)
*!   100 rows: units 0-9 observed in periods 0-4, units 10-19 in periods 5-9
*!   (two bipartite components, C = 2; span rank of [unit FE, time FE]
*!   including constants = 20 + 10 - 2 = 28).
*!     unit = floor((_n-1)/5)            time = mod(_n-1,5) + 5*(unit>=10)
*!     x    = (mod(_n*7, 13) - 6)/13     z    = (mod(_n*11, 17) - 8)/17
*!     out  = 0.5*x + 0.2*mod(unit,3) + 0.1*time + z
*!   Every operation is exact integer arithmetic followed by one IEEE
*!   division / multiply-add chain evaluated left-to-right in both
*!   languages, so the frames agree bit-for-bit.
*!
*! Arms
*!   unit_cluster : vce(cluster unit). Unit FE nested in the cluster ->
*!                  dropped (nested = 20); reghdfe counts the time
*!                  remainder approximately (df_a = 9 = T - 1, redundant =
*!                  nested + the global constant only), denominator
*!                  N - 11. Library: exact remainder 8 -> K_reference =
*!                  10 (the documented one-df deviation above).
*!   cross_cluster: vce(cluster c5), c5 = mod(unit+time, 5) crossing both
*!                  dims -> nothing nested, df_a = exact span = 28,
*!                  denominator N - 29 (agrees with the library AND fixest
*!                  ssc(K.fixef="full", K.exact=TRUE), df.K = 29).
*!
*! Consuming test
*!   tests/test_variance_conventions.py::TestReghdfeKReferenceParity
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/reghdfe_kref_golden.json
*!
*! Usage (run from the repo root)
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!       benchmarks/stata/generate_reghdfe_kref_golden.do
*!   Then confirm the log is clean:  grep -E '^r\([0-9]+\);' generate_reghdfe_kref_golden.log
*!   (Stata batch mode ALWAYS exits 0, even on error - never trust the exit code.)
*!
*! SSC dependencies
*!   reghdfe (with ftools)

version 19
clear all
set more off
set type double

* JSON number formatter (same as generate_etwfe_cs_golden.do).
capture program drop _jnum
program define _jnum, rclass
    args x fmt
    if "`fmt'" == "" local fmt "%21.17g"
    local s = strtrim(string(`x', "`fmt'"))
    if substr("`s'", 1, 1) == "."       local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

capture which reghdfe
if _rc {
    display as error "Missing SSC package reghdfe. Run benchmarks/stata/requirements.do first."
    exit 111
}

* ------------------------------------------------------------------------------
* Deterministic disconnected panel (see header).
* ------------------------------------------------------------------------------
set obs 100
gen unit = floor((_n - 1) / 5)
gen time = mod(_n - 1, 5) + 5 * (unit >= 10)
gen x = (mod(_n * 7, 13) - 6) / 13
gen z = (mod(_n * 11, 17) - 8) / 17
gen out = 0.5 * x + 0.2 * mod(unit, 3) + 0.1 * time + z
gen c5 = mod(unit + time, 5)

* ------------------------------------------------------------------------------
* Arm 1: cluster = unit (nested drop + exact remainder rank).
* ------------------------------------------------------------------------------
reghdfe out x, absorb(unit time) vce(cluster unit)
assert e(df_a) == 9
assert e(df_a_nested) == 20
assert e(rank) == 1
local u_coef = _b[x]
local u_se = _se[x]
local u_dfa = e(df_a)
local u_dfai = e(df_a_initial)
local u_dfan = e(df_a_nested)
local u_dfared = e(df_a_redundant)
local u_rank = e(rank)
local u_dfr = e(df_r)
local u_G = e(N_clust)
local u_N = e(N)

* ------------------------------------------------------------------------------
* Arm 2: cluster = c5 (nothing nested; exact span rank).
* ------------------------------------------------------------------------------
reghdfe out x, absorb(unit time) vce(cluster c5)
assert e(df_a) == 28
assert e(df_a_nested) == 0
assert e(df_a_redundant) == 2
local x_coef = _b[x]
local x_se = _se[x]
local x_dfa = e(df_a)
local x_dfai = e(df_a_initial)
local x_dfan = e(df_a_nested)
local x_dfared = e(df_a_redundant)
local x_rank = e(rank)
local x_dfr = e(df_r)
local x_G = e(N_clust)
local x_N = e(N)

* ------------------------------------------------------------------------------
* reghdfe version (drift signal; SSC has no version history).
* ------------------------------------------------------------------------------
local rv "unknown"
capture findfile reghdfe.ado
if _rc == 0 {
    tempname vh
    file open `vh' using "`r(fn)'", read text
    file read `vh' line
    local n = 0
    while r(eof) == 0 & `n' < 15 {
        if substr(`"`macval(line)'"', 1, 2) == "*!" {
            local body = strtrim(substr(`"`macval(line)'"', 3, .))
            if strpos(lower(`"`body'"'), "version") > 0 {
                local rv `"`body'"'
                continue, break
            }
        }
        local ++n
        file read `vh' line
    }
    file close `vh'
    local rv = subinstr(`"`rv'"', `"""', "'", .)
    local rv = subinstr(`"`rv'"', "\", "/", .)
}

* ------------------------------------------------------------------------------
* Emit the golden.
* ------------------------------------------------------------------------------
local sver = c(stata_version)

tempname fh
file open `fh' using "benchmarks/data/reghdfe_kref_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_reghdfe_kref_golden.do","' _n
file write `fh' `"    "reghdfe_version": "`rv'","' _n
file write `fh' `"    "stata_version": `sver',"' _n
file write `fh' `"    "dgp": "deterministic disconnected two-way panel; see generator header","' _n
file write `fh' `"    "note": "cross_cluster = exact-span parity anchor (K=29, matches library + fixest full/K.exact). unit_cluster = documentation arm: reghdfe counts the nested-remainder approximately (implied K=11) where the library uses the exact remainder (K=10); the test pins se ratio == sqrt((N-10)/(N-11)). No external reference implements nested-drop + exact-remainder.""' _n
file write `fh' "  }," _n
foreach arm in u x {
    if "`arm'" == "u" {
        file write `fh' `"  "unit_cluster": {"' _n
        file write `fh' `"    "cmd": "reghdfe out x, absorb(unit time) vce(cluster unit)","' _n
    }
    else {
        file write `fh' `"  "cross_cluster": {"' _n
        file write `fh' `"    "cmd": "reghdfe out x, absorb(unit time) vce(cluster c5)","' _n
    }
    file write `fh' `"    "n": ``arm'_N',"' _n
    file write `fh' `"    "G": ``arm'_G',"' _n
    file write `fh' `"    "df_a": ``arm'_dfa',"' _n
    file write `fh' `"    "df_a_initial": ``arm'_dfai',"' _n
    file write `fh' `"    "df_a_nested": ``arm'_dfan',"' _n
    file write `fh' `"    "df_a_redundant": ``arm'_dfared',"' _n
    file write `fh' `"    "rank": ``arm'_rank',"' _n
    file write `fh' `"    "df_r": ``arm'_dfr',"' _n
    _jnum ``arm'_coef'
    local a = r(s)
    _jnum ``arm'_se'
    local s = r(s)
    file write `fh' `"    "coef": `a',"' _n
    file write `fh' `"    "se": `s'"' _n
    if "`arm'" == "u" file write `fh' "  }," _n
    else              file write `fh' "  }" _n
}
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/reghdfe_kref_golden.json"
