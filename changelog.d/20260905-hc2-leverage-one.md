### Behavioral Changes
- **HC2 and unweighted, unclustered HC2-BM now fail closed at leverage one:**
  an effective observation with hat-matrix leverage at least `1 - 1e-8`
  produces a warning and entirely NaN covariance (and requested degrees of
  freedom), preserving point estimates while suppressing undefined inference.
  Python and Rust agree; over-one leverage no longer substitutes HC1.
  Older Rust extensions without the fail-closed HC2 capability use NumPy
  for HC2 while retaining their other accelerations.
  Weighted and clustered HC2-BM retain their separate CR2 conventions,
  including all-ones probability weights.

### Fixed
- **Zero-weight observations do not invalidate HC2/HC3 inference:** excluded
  rows contribute zero to the covariance and cannot trigger the leverage
  guard. Zero-frequency rows now agree with dropping those rows or expanding
  the frequency counts literally.
- **LWDiD uses the shared HC2 covariance guard:** leverage-one regressions
  retain their point estimate and unavailable influence contribution while
  emitting one covariance warning per regression, without a duplicate local
  warning.
