*! Stata SSC requirements for diff-diff parity arms.
*!
*! Run this ONCE before regenerating any SSC-dependent golden. The generators do
*! NOT auto-install: `ssc install` always fetches the LATEST version, which would
*! (a) break byte-identical regeneration across dates, (b) require network at
*! generation time, and (c) defeat the drift-detection that each golden records in
*! its meta.ssc_versions. Installing is therefore a deliberate, separate step.
*!
*! Usage (from the repo root):
*!   /Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do benchmarks/stata/requirements.do
*!
*! SSC has no version history, so there is no way to pin a specific release here;
*! the recorded meta.ssc_versions in each golden is the only drift signal.
*!
*! Packages (dependency order matters - ftools/require before reghdfe):
*!   ftools, require   - reghdfe infrastructure
*!   reghdfe           - fast FE regression (did_imputation backend)
*!   did_imputation    - Borusyak-Jaravel-Spiess (2024) imputation DiD; the only
*!                       implementation of the App. A.9 leave-one-out variance
*!                       (consumed by generate_imputation_loo_golden.do)
*!   drdid, csdid      - Callaway-Sant'Anna reference implementation
*!   hdfe, jwdid       - Wooldridge ETWFE reference implementation (hdfe is a
*!                       jwdid dependency; jwdid errors with "You need to install
*!                       hdfe from SSC" without it)
*!                       (both consumed by generate_etwfe_cs_golden.do)
*!   lwdid             - Lee & Wooldridge rolling DiD, the authors' reference
*!                       implementation (consumed by generate_lwdid_golden.do)
*!   lpdid             - Dube-Girardi-Jorda-Taylor LP-DiD, the authors' reference
*!                       implementation (consumed by
*!                       generate_lpdid_nonabsorbing_golden.do)
*!   boottest, egenmore, listreg
*!                     - lpdid startup dependencies (lpdid `which`-checks all three
*!                       and exits without them; egenmore's filter() runs in every
*!                       lpdid pooled spec). egenmore installs no `egenmore.ado` -
*!                       only `_g*.ado` helpers - so the loop probes BOTH
*!                       `_gfilter` (executed by the pooled spec) and `_gclsst`
*!                       (lpdid's own startup which-check) and reinstalls the
*!                       `egenmore` package if either is missing; requiring both
*!                       means an incomplete egenmore install is repaired by
*!                       rerunning this script.

version 19
foreach p in ftools require reghdfe did_imputation drdid csdid hdfe jwdid lwdid boottest egenmore listreg lpdid {
    local missing 0
    if "`p'" == "egenmore" {
        foreach probe in _gfilter _gclsst {
            capture which `probe'
            if _rc local missing 1
        }
    }
    else {
        capture which `p'
        if _rc local missing 1
    }
    if `missing' {
        di as txt "Installing `p' from SSC ..."
        ssc install `p', replace
    }
    else {
        di as txt "`p' already installed."
    }
}
di as result "Stata SSC requirements satisfied."
