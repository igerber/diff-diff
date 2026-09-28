*! Golden generator: LPDiD regression-adjustment (RA) SE vs Stata `teffects ra`.
*!
*! Purpose
*!   Produce an INDEPENDENT external anchor for the LP-DiD regression-adjustment
*!   standard error (Dube, Girardi, Jorda & Taylor 2025). The canonical RA SE is
*!   Stata `teffects ra ... atet vce(cluster)` only - no R package computes it
*!   (`alexCardazzi/lpdid` does direct covariate inclusion, not RA). Until now the
*!   library RA IF-cluster SE was pinned against itself (`RA_SE_PIN` in
*!   tests/test_methodology_lpdid.py) and calibration-validated only by a
*!   Monte-Carlo coverage study. This generator converts that self-pin into a
*!   measured cross-implementation anchor.
*!
*!   This is the FIRST Stata arm in the repo. `teffects` is NATIVE to Stata (no
*!   SSC dependency), so `version 19` fully pins the numerical behavior.
*!
*! Consuming test
*!   tests/test_lpdid_ra_stata_parity.py
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/lpdid_ra_stata_golden.json
*!
*! Usage (run from the repo root)
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!       benchmarks/stata/generate_lpdid_ra_golden.do
*!   Then confirm the log is clean:  grep -E '^r\([0-9]+\);' generate_lpdid_ra_golden.log
*!   (Stata batch mode ALWAYS exits 0, even on error - never trust the exit code.)
*!
*! Notes
*!   - Reads (does NOT regenerate) benchmarks/data/lpdid_test_panel.csv, whose sole
*!     owner is benchmarks/R/generate_lpdid_golden.R. 60 units x 12 periods, one
*!     deliberate interior gap at (unit==60, time==7) on a never-treated unit.
*!   - Independently reconstructs the per-horizon clean sample by porting the R
*!     `prep` + `clean_h` recipe (generate_lpdid_golden.R:97-123). `tsfill` is
*!     load-bearing: it mirrors R `fill_gaps()` so the absorbing `treat` recompute
*!     and the F./L. long differences see the completed calendar grid (a lead
*!     across the unit-60 gap must yield a real value / a missing outcome, not a
*!     spurious neighbor).
*!   - The in-.do point gate (1e-8, informational) aborts early on a gross port
*!     bug; the AUTHORITATIVE point gate is the Python test asserting Stata ATET
*!     vs benchmarks/data/lpdid_golden.json `ra_cov[h][0]` at 1e-10.

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
    if substr("`s'", 1, 1) == "."      local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

* ------------------------------------------------------------------------------
* R-anchored RA POINT estimates (benchmarks/data/lpdid_golden.json `ra_cov[h][0]`,
* R generator digits=12). Hard-coded here only for the in-.do smoke gate; the
* Python test reads the JSON directly and is the authoritative check.
* ------------------------------------------------------------------------------
scalar rpt_0  =  2.888871838355
scalar rpt_1  =  3.22526891068
scalar rpt_2  =  3.577950221628
scalar rpt_3  =  3.973278128237
scalar rpt_4  =  4.265103781445
scalar rpt_m2 =  0.1204444745696
scalar rpt_m3 = -0.1197046094519

* ------------------------------------------------------------------------------
* Load the committed panel and confirm the expected schema/types.
* ------------------------------------------------------------------------------
import delimited using "benchmarks/data/lpdid_test_panel.csv", clear varnames(1)
confirm numeric variable unit time treat y x
quietly count
assert r(N) == 719          // 60*12 - 1 interior gap

* ------------------------------------------------------------------------------
* prep: port of generate_lpdid_golden.R:97-110.
*   - tsfill materializes the interior-gap row (mirrors R fill_gaps()).
*   - treat is recomputed as an ABSORBING fill on the completed grid.
* ------------------------------------------------------------------------------
tsset unit time
tsfill                                                   // <-- mirrors fill_gaps()

* treat_date = first period the imported treat==1, per unit (never-treated -> .).
* tsfill rows carry treat==. so cond(treat==1,.)-> . and egen min ignores them.
bysort unit (time): egen treat_date = min(cond(treat==1, time, .))

replace treat = (!missing(treat_date) & time >= treat_date)   // absorbing on grid
gen double Ly = L.y                                            // base for long diff
gen tdiff = treat - L.treat
replace tdiff = 0 if missing(tdiff) | tdiff < 0
gen byte obs = !missing(y)

* ------------------------------------------------------------------------------
* Per-horizon: rebuild the clean sample (clean_h) inside preserve/restore, run
* teffects ra ... atet vce(cluster unit), capture ATET/SE/N/G into scalars.
* Post h in {0,1,2,3,4}; pre h in {2,3} (key = -h; h=-1 is the omitted reference).
* ------------------------------------------------------------------------------
tempname fh
foreach h of numlist 0 1 2 3 4 {
    preserve
        * clean_h(post): Dy = lead(y,h) - Ly ; Fh = lead(treat,h)
        gen double Dy = F`h'.y - Ly
        gen Fh = F`h'.treat
        keep if obs & !missing(Dy) & !missing(tdiff) & !missing(Fh) & (tdiff==1 | Fh==0)
        teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)
        scalar att_`h' = _b[r1vs0.tdiff]
        scalar se_`h'  = _se[r1vs0.tdiff]
        scalar N_`h'   = e(N)
        scalar G_`h'   = e(N_clust)
        * in-.do smoke gate (informational; Python test is authoritative)
        assert reldif(att_`h', rpt_`h') < 1e-8
    restore
}
foreach h of numlist 2 3 {
    preserve
        * clean_h(pre): Dy = lag(y,h) - Ly ; filter on treat, not Fh
        gen double Dy = L`h'.y - Ly
        keep if obs & !missing(Dy) & !missing(tdiff) & !missing(treat) & (tdiff==1 | treat==0)
        teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)
        scalar att_m`h' = _b[r1vs0.tdiff]
        scalar se_m`h'  = _se[r1vs0.tdiff]
        scalar N_m`h'   = e(N)
        scalar G_m`h'   = e(N_clust)
        assert reldif(att_m`h', rpt_m`h') < 1e-8
    restore
}

* ------------------------------------------------------------------------------
* Emit JSON by hand (Stata has no jsonlite) at %21.17g (round-trip-exact double).
* Per horizon: {att, se, N, G}. No timestamp (byte-identical regeneration).
* ------------------------------------------------------------------------------
* Stata compound-quote note: the `"' delimiter swallows a trailing double-quote,
* so every STRING field is written with a trailing comma (`...","' -> `...",`);
* the meta block therefore ENDS with numeric fields, which have no trailing quote.
local sver = strtrim(string(c(stata_version), "%4.1f"))
* Derive the running edition so provenance is correct if regenerated under a
* different edition. c(flavor) misreports "IC" on SE and c(edition) is unreliable;
* c(MP)/c(SE) are the authoritative 0/1 flags (c(BE) is UNDEFINED, and cond()
* evaluates all branches eagerly, so BE is resolved by elimination, not referenced).
local sedition = cond(c(MP)==1, "MP", cond(c(SE)==1, "SE", "BE"))
file open `fh' using "benchmarks/data/lpdid_ra_stata_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "estimator": "LPDiD regression-adjustment (RA) SE - Stata teffects ra atet","' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_lpdid_ra_golden.do","' _n
file write `fh' `"    "source_panel": "benchmarks/data/lpdid_test_panel.csv","' _n
file write `fh' `"    "point_anchor": "benchmarks/data/lpdid_golden.json ra_cov[h][0]","' _n
file write `fh' `"    "stata_edition": "`sedition'","' _n
file write `fh' `"    "cmd": "teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)","' _n
file write `fh' `"    "se_convention": "cluster-robust IF at unit, NO finite-sample factor (teffects convention); t(G-1) reference on the library side","' _n
file write `fh' `"    "note": "Independent Stata reconstruction of the per-horizon clean sample (ports generate_lpdid_golden.R prep/clean_h). N and G are emitted to power the Python row-count sample-integrity gate.","' _n
file write `fh' `"    "stata_version": `sver',"' _n
file write `fh' `"    "pre_window": 3,"' _n
file write `fh' `"    "post_window": 4"' _n
file write `fh' "  }," _n
file write `fh' `"  "ra_se": {"'

* post entries (0..4) then pre (-2,-3); comma-PREFIXED so there is no trailing
* comma and no blank first line. Each entry assembled into trimmed locals first.
local sep ""
foreach h of numlist 0 1 2 3 4 {
    _jnum att_`h'
    local a = r(s)
    _jnum se_`h'
    local s = r(s)
    _jnum N_`h' "%12.0f"
    local nn = r(s)
    _jnum G_`h' "%12.0f"
    local gg = r(s)
    file write `fh' "`sep'" _n `"    "`h'": {"att": `a', "se": `s', "N": `nn', "G": `gg'}"'
    local sep ","
}
foreach h of numlist 2 3 {
    _jnum att_m`h'
    local a = r(s)
    _jnum se_m`h'
    local s = r(s)
    _jnum N_m`h' "%12.0f"
    local nn = r(s)
    _jnum G_m`h' "%12.0f"
    local gg = r(s)
    file write `fh' "`sep'" _n `"    "-`h'": {"att": `a', "se": `s', "N": `nn', "G": `gg'}"'
    local sep ","
}
file write `fh' _n "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/lpdid_ra_stata_golden.json (7 horizons)"
