*! Golden generator: LWDiD parity vs the authors' Stata `lwdid` package (SSC).
*!
*! Purpose
*!   Produce the INDEPENDENT external anchor for the LWDiD estimator
*!   (Lee & Wooldridge 2025, 2026) ahead of PR #588's merge: full-precision
*!   small-N results on Prop 99 (incl. randomization inference at high reps)
*!   and Castle Doctrine (tau_omega), plus the large-N Walmart event-study
*!   WATT points and multiplier-bootstrap SEs for the six suite configs.
*!   Printed-table goldens only reach three decimals; this arm anchors the
*!   unprinted paths (RI convention, bootstrap SEs) at full precision.
*!
*! Consuming tests
*!   tests/test_methodology_lwdid.py (gated on diff_diff.lwdid; activates
*!   when PR #588 lands) and tests/test_lwdid_stata_golden_schema.py
*!   (ungated schema/cardinality check that runs on main).
*!
*! Outputs (checked into the repo)
*!   benchmarks/data/lwdid_stata_golden.json
*!
*! Usage (run from the repo root)
*!   1. FAIL-CLOSED warm-up - verifies the loader cache on disk (loader
*!      success alone does not guarantee the cached bytes; cache writes are
*!      best-effort):
*!        python -c "from diff_diff import load_prop99, load_walmart; \
*!          import hashlib, pathlib; \
*!          dfs = {'prop99': load_prop99(), 'walmart': load_walmart()}; \
*!          assert all(df.attrs.get('source') != 'synthetic_fallback' for df in dfs.values()); \
*!          pins = {'prop99': '16c3ac1da351788817433fc890ec2f502a8bdfcb46cbc8d693653330e71d5a65', \
*!                  'walmart': '410885572143dceb9daa643a8097768f1bc3493f9437451a9e4d1d5dc1e18d14'}; \
*!          cache = pathlib.Path.home() / '.cache' / 'diff_diff' / 'datasets'; \
*!          [1/0 for n, w in pins.items() if hashlib.sha256((cache / (n + '.dta')).read_bytes()).hexdigest() != w]"
*!   2. /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
*!        benchmarks/stata/generate_lwdid_golden.do
*!   3. Confirm the log is clean: grep -E '^r\([0-9]+\);' generate_lwdid_golden.log
*!      (Stata batch mode ALWAYS exits 0, even on error - never trust the
*!      exit code.)
*!
*! Notes
*!   - `lwdid` is an SSC package (no version pinning possible); meta records
*!     the verbatim `which lwdid` version line under `ssc_versions` so drift
*!     is detectable, and the capture FAILS CLOSED if no version line is found.
*!   - Small-N mode has NO control-group option: its composite regression is
*!     never-treated-based by construction - empirically equivalent to the
*!     Python `control_group='never_treated'` fits the parity tests use.
*!   - Large-N default pool = never-treated + not-yet-treated (no `never`
*!     option passed), matching the Python default
*!     `control_group='not_yet_treated'`.
*!   - In-.do smoke gates are informational early-abort checks against the
*!     2026-08-15/16 measured values; the Python tests are authoritative.

version 19
clear all
set more off
set type double

* ------------------------------------------------------------------------------
* Fail-closed dependency guard (SSC package; cannot be pinned by `version`).
* ------------------------------------------------------------------------------
capture which lwdid
if _rc {
    display as error "lwdid not installed - run benchmarks/stata/requirements.do"
    exit 111
}

* ------------------------------------------------------------------------------
* _jnum: format a scalar as a JSON number at round-trip-exact precision.
* Stata's %21.17g renders |x|<1 as ".455"/"-.455" (leading dot); JSON requires
* a leading 0, so patch ".x" -> "0.x" and "-.x" -> "-0.x". Stata MISSING is
* emitted as JSON null (the parity tables can carry missing cells). Returns r(s).
* ------------------------------------------------------------------------------
capture program drop _jnum
program define _jnum, rclass
    args x fmt
    if "`fmt'" == "" local fmt "%21.17g"
    if missing(`x') {
        return local s "null"
        exit
    }
    local s = strtrim(string(`x', "`fmt'"))
    if substr("`s'", 1, 1) == "."      local s = "0" + "`s'"
    else if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end

* ------------------------------------------------------------------------------
* _adover: capture an ado-file's version header for meta.ssc_versions.
* FAILS CLOSED: aborts if the captured line does not look like a version line
* (an "unknown" capture would silently disable drift detection).
* ------------------------------------------------------------------------------
capture program drop _adover
program define _adover, rclass
    args pkg
    quietly findfile `pkg'.ado
    local path "`r(fn)'"
    tempname fh
    file open `fh' using "`path'", read text
    file read `fh' line
    local ver ""
    local tries 0
    while r(eof) == 0 & `tries' < 10 {
        if strpos("`line'", "*!") == 1 & strpos(lower("`line'"), "version") > 0 {
            local ver = strtrim(substr("`line'", 3, .))
            continue, break
        }
        file read `fh' line
        local tries = `tries' + 1
    }
    file close `fh'
    if strpos(lower("`ver'"), "version") == 0 {
        display as error "no version line found in `pkg'.ado - refusing to emit undetectable provenance"
        exit 498
    }
    return local s "`ver'"
end

_adover lwdid
local lwdid_ver "`r(s)'"
display "lwdid version line: `lwdid_ver'"

local RIREPS 100000
local RISEED 20260815
local BREPS 9999
local BSEED 20260815
local home : env HOME
local cache "`home'/.cache/diff_diff/datasets"

* ==============================================================================
* Block 1: Prop 99 (small-N), demean + detrend, RI at high reps.
* ==============================================================================
use "`cache'/prop99.dta", clear
confirm numeric variable state year first_year lcigsale
quietly count
assert r(N) == 1209

foreach r in demean detrend {
    lwdid lcigsale, small ivar(state) tvar(year) gvar(first_year) ///
        rolling(`r') ri rireps(`RIREPS') riseed(`RISEED')
    scalar p99_att_`r'  = e(att)
    scalar p99_se_`r'   = e(se_att)
    scalar p99_pri_`r'  = e(p_ri)
}
* informational smoke gates (measured 2026-08-15)
assert reldif(p99_att_demean,  -0.4221746150201265) < 1e-10
assert reldif(p99_att_detrend, -0.2269886995561676) < 1e-10
assert abs(p99_pri_detrend - 0.0508) < 0.005

* ==============================================================================
* Block 2: Castle Doctrine (small-N staggered), demean + detrend.
* ==============================================================================
import delimited using "benchmarks/data/real/castle_lw_subset.csv", clear case(preserve)
replace effyear = 0 if missing(effyear)
egen sid = group(state)
confirm numeric variable sid year effyear lhomicide
quietly count
assert r(N) == 550

foreach r in demean detrend {
    lwdid lhomicide, small ivar(sid) tvar(year) gvar(effyear) rolling(`r')
    scalar cas_att_`r' = e(att)
    scalar cas_se_`r'  = e(se_att)
}
* informational smoke gates (measured 2026-08-16 under `set type double`:
* `import delimited` reads the CSV as doubles here, unlike the exploratory
* smoke whose float import gave 0.09174538052... - an 8th-decimal float
* artifact; the double values below match the Python double-precision read)
assert reldif(cas_att_demean, 0.091745387139613596) < 1e-10
assert reldif(cas_se_demean,  0.0571027) < 1e-6
assert reldif(cas_att_detrend, 0.066550335128826035) < 1e-10
assert reldif(cas_se_detrend,  0.0560124) < 1e-6

* ==============================================================================
* Block 3: Walmart (large-N), 3 configs x 2 outcomes, multiplier bootstrap.
* save() results go to tempfiles (auto-deleted; nothing written under the repo).
* Emitter runs AFTER all six fits so the panel-replacement hazard of reading a
* result .dta never touches a pending fit.
* ==============================================================================
use "`cache'/walmart.dta", clear
confirm numeric variable cid year first_year log_retail_emp log_wholesale_emp x1 x2 x3
quietly count
assert r(N) == 29371

* config list: key|rolling|method|covars|outcome
local c1 "detrend_ra__log_retail_emp|detrend|ra||log_retail_emp"
local c2 "detrend_ipwra__log_retail_emp|detrend|ipwra|x1 x2 x3|log_retail_emp"
local c3 "demean_ipwra__log_retail_emp|demean|ipwra|x1 x2 x3|log_retail_emp"
local c4 "detrend_ra__log_wholesale_emp|detrend|ra||log_wholesale_emp"
local c5 "detrend_ipwra__log_wholesale_emp|detrend|ipwra|x1 x2 x3|log_wholesale_emp"
local c6 "demean_ipwra__log_wholesale_emp|demean|ipwra|x1 x2 x3|log_wholesale_emp"

forvalues i = 1/6 {
    * gettoken with parse("|") returns delimiter tokens; an EMPTY covars
    * field surfaces as a second "|" token, handled below.
    gettoken key  rest : c`i', parse("|")
    gettoken bar  rest : rest, parse("|")
    gettoken roll rest : rest, parse("|")
    gettoken bar  rest : rest, parse("|")
    gettoken method rest : rest, parse("|")
    gettoken bar  rest : rest, parse("|")
    gettoken covs rest : rest, parse("|")
    if "`covs'" == "|" {
        local covs ""
        gettoken outc rest : rest, parse("|")
    }
    else {
        gettoken bar  rest : rest, parse("|")
        gettoken outc rest : rest, parse("|")
    }
    display as txt "config `i': key=`key' rolling=`roll' method=`method' covs=[`covs'] outcome=`outc'"

    tempfile res`i'
    set seed `BSEED'
    preserve
    lwdid `outc' `covs', ivar(cid) tvar(year) gvar(first_year) ///
        rolling(`roll') method(`method') reps(`BREPS') save(`res`i'')
    restore
    local key`i' "`key'"
}

* ------------------------------------------------------------------------------
* Emit JSON.
* Compound-quote note: the `"' delimiter swallows a trailing double-quote, so
* every STRING field is written with a trailing comma; numeric meta fields have
* no trailing quote hazard. No timestamps (byte-identical regeneration).
* ------------------------------------------------------------------------------
local sver = strtrim(string(c(stata_version), "%4.1f"))
local sedition = cond(c(MP)==1, "MP", cond(c(SE)==1, "SE", "BE"))

tempname fh
file open `fh' using "benchmarks/data/lwdid_stata_golden.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "meta": {"' _n
file write `fh' `"    "estimator": "LWDiD (Lee & Wooldridge 2025, 2026) - authors' Stata lwdid package parity","' _n
file write `fh' `"    "generator": "benchmarks/stata/generate_lwdid_golden.do","' _n
file write `fh' `"    "ssc_versions": {"lwdid": "`lwdid_ver'"},"' _n
file write `fh' `"    "stata_edition": "`sedition'","' _n
file write `fh' `"    "bootstrap_scheme": "lwdid large-N multiplier bootstrap (package default; reps() draws with set seed); compared against diff-diff's unit-level Rademacher multiplier bootstrap at B=999","' _n
file write `fh' `"    "control_pool": {"small_N": "composite regression, never-treated-based by construction (no control option exists); equivalent to Python control_group='never_treated'", "large_N": "never-treated + not-yet-treated default (the 'never' option is NOT passed); matches Python default control_group='not_yet_treated'"},"' _n
file write `fh' `"    "datasets": {"prop99": {"url": "http://fmwww.bc.edu/repec/bocode/l/lw_smoking.dta", "sha256": "16c3ac1da351788817433fc890ec2f502a8bdfcb46cbc8d693653330e71d5a65"}, "walmart": {"url": "http://fmwww.bc.edu/repec/bocode/l/lw_walmart.dta", "sha256": "410885572143dceb9daa643a8097768f1bc3493f9437451a9e4d1d5dc1e18d14"}, "castle": {"path": "benchmarks/data/real/castle_lw_subset.csv"}},"' _n
file write `fh' `"    "stata_version": `sver',"' _n
file write `fh' `"    "rireps": `RIREPS',"' _n
file write `fh' `"    "riseed": `RISEED',"' _n
file write `fh' `"    "bootstrap_reps": `BREPS',"' _n
file write `fh' `"    "bootstrap_seed": `BSEED'"' _n
file write `fh' "  }," _n

* --- prop99 block: {"demean": {"att":..,"se":..,"p_ri":..}, "detrend": {...}}
file write `fh' `"  "prop99": {"'
local sep ""
foreach r in demean detrend {
    _jnum p99_att_`r'
    local a = r(s)
    _jnum p99_se_`r'
    local s = r(s)
    _jnum p99_pri_`r'
    local p = r(s)
    file write `fh' "`sep'" _n `"    "`r'": {"att": `a', "se": `s', "p_ri": `p'}"'
    local sep ","
}
file write `fh' _n "  }," _n

* --- castle block: {"demean": {"att":..,"se":..}, "detrend": {...}}
file write `fh' `"  "castle": {"'
local sep ""
foreach r in demean detrend {
    _jnum cas_att_`r'
    local a = r(s)
    _jnum cas_se_`r'
    local s = r(s)
    file write `fh' "`sep'" _n `"    "`r'": {"att": `a', "se": `s'}"'
    local sep ","
}
file write `fh' _n "  }," _n

* --- walmart block: one key per (config, outcome); wrapper {"watt": {...}, "overall": {...}}
file write `fh' `"  "walmart": {"'
local csep ""
forvalues i = 1/6 {
    preserve
    use "`res`i''", clear
    * schema gate (from the save() spike): effect ryear watt se + aggregates
    confirm string variable effect
    confirm numeric variable ryear watt se
    quietly count
    local nrows = r(N)
    quietly count if !missing(ryear)
    local nwatt = r(N)
    assert `nrows' == `nwatt' + 2          // exactly Pre_avg + Post_avg extra
    * fail closed on incomplete regeneration: every emitted cell nonmissing,
    * exactly one row per event time and per aggregate label
    assert !missing(watt) & !missing(se)
    quietly duplicates report ryear if !missing(ryear)
    assert r(unique_value) == r(N)
    file write `fh' "`csep'" _n `"    "`key`i''": {"' _n
    file write `fh' `"      "watt": {"'
    local rsep ""
    quietly levelsof ryear if !missing(ryear), local(rs)
    foreach rv of local rs {
        quietly summarize watt if ryear == `rv', meanonly
        scalar w_pt = r(mean)
        quietly summarize se if ryear == `rv', meanonly
        scalar w_se = r(mean)
        _jnum w_pt
        local a = r(s)
        _jnum w_se
        local s = r(s)
        local rint = int(`rv')
        file write `fh' "`rsep'" _n `"        "`rint'": [`a', `s']"'
        local rsep ","
    }
    file write `fh' _n "      }," _n
    file write `fh' `"      "overall": {"'
    local osep ""
    foreach agg in Pre_avg Post_avg {
        quietly summarize watt if effect == "`agg'", meanonly
        scalar o_pt = r(mean)
        quietly summarize se if effect == "`agg'", meanonly
        scalar o_se = r(mean)
        _jnum o_pt
        local a = r(s)
        _jnum o_se
        local s = r(s)
        file write `fh' "`osep'" _n `"        "`agg'": [`a', `s']"'
        local osep ","
    }
    file write `fh' _n "      }" _n
    file write `fh' "    }"
    local csep ","
    restore
}
file write `fh' _n "  }" _n
file write `fh' "}" _n
file close `fh'

display "Wrote benchmarks/data/lwdid_stata_golden.json"
