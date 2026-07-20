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

version 19
foreach p in ftools require reghdfe did_imputation {
    capture which `p'
    if _rc {
        di as txt "Installing `p' from SSC ..."
        ssc install `p', replace
    }
    else {
        di as txt "`p' already installed."
    }
}
di as result "Stata SSC requirements satisfied."
