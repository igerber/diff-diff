### Fixed
- **`hc2` / `hc2_bm` no longer report an understated finite variance at a
  leverage-one observation**: the shared one-way leverage meat floored
  `1 - h_ii` at 1e-10, which did not inflate the perfectly-fitted row's
  term but silently dropped its outcome noise (probe `[1, D]` with a single
  treated unit: HC2 SE 0.19 against the exact classical 0.47), and
  `hc2_bm` paired that finite SE with a NaN Satterthwaite DOF. Both now
  fail closed exactly like `hc3` (see Behavioral Changes). The Rust HC2
  kernel mirrors the change (no floor; it signals and the Python dispatcher
  re-dispatches to the NumPy branch, which emits the single warning). The
  sibling floor in the one-way Bell-McCaffrey DOF helper is removed: at
  leverage one every contrast's DOF is NaN because the HC2 variance it
  belongs to is undefined; below leverage one nothing changes.
- **Zero-count `fweight` rows no longer trip the HC3 leverage guard or the
  HC1 fallback**: under frequency weights the leverage of a zero-count row
  is the unweighted quadratic form against the weighted bread and is
  unbounded, so an inert row could NaN the `hc3` vcov or push `hc2` into an
  HC1 fallback that broke expansion parity. Zero-count rows are excluded
  from the guard and from the meat, so compressed HC2/HC3 equal the literal
  `np.repeat` expansion at any leverage.
- `LWDiD` no longer emits a second, differently worded "HC2 variance is
  undefined" warning per cell on leverage-one designs; the shared kernel's
  warning is the only one (behavior otherwise unchanged: LWDiD already
  failed closed there).

### Behavioral Changes
- **Leverage-one designs under `vcov_type="hc2"` (every weight type) and
  under unweighted, unclustered `vcov_type="hc2_bm"` now return an all-NaN
  covariance and DOF vector with a `UserWarning`** ("HC2 variance is
  undefined: N observation(s) have hat-matrix leverage ~1 ...") whenever a
  positive-weight row has `h_ii >= 1 - 1e-8`, matching the released `hc3`
  contract and R `sandwich::vcovHC` (NaN at hat values ~1). Point
  estimates are unchanged; `se`, `t_stat`, `p_value` and confidence
  intervals are NaN. The former warn-and-fall-back-to-HC1 branch for
  over-one leverage is retired (that case is inside the new guard).
  Affected surfaces: `DifferenceInDifferences`, `MultiPeriodDiD` and
  `LinearRegression` on both families; `TwoWayFixedEffects` on explicit
  `hc2` and on `hc2_bm` in event-study `spec="pooled"` without `unit=`;
  `SunAbraham` and `WooldridgeDiD` (OLS) full-dummy fits under one-way
  `hc2` whenever the design has a singleton cohort x period cell. Weighted
  (pweight) `hc2_bm` — including a no-op `weights=np.ones(n)` — keeps the
  clubSandwich singleton-CR2 generalized-inverse result and stays finite,
  as does clustered CR2; this asymmetry is a documented deviation from
  `clubSandwich` for the unweighted case (maintainer decision, 2026-09).
  Remedy: `vcov_type="classical"` (exact inference under homoskedastic
  normal errors; needs positive residual df) or add observations to the
  perfectly-fitted cell. `hc1` is deliberately not offered as a remedy: it
  also omits the zero-residual row and would return the understated number.
- The shared leverage-one warning (also used by `hc3`) now names the
  offending row indices and recommends classical exact inference or more
  observations in the cell instead of "add treated units".
