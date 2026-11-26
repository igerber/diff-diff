### Documentation
- **Fundamentals tutorials repaired and brought to the tutorial conventions**
  (`01_basic_did`, `02_staggered_did`, `03_synthetic_did`, `08_triple_diff`,
  `09_real_world_examples`). Each now states its estimand and identifying
  assumptions before the first fit, checks estimates against a stated truth
  where the data is simulated, clusters panel fits at the unit, and is
  committed with executed outputs and a drift test. Corrections of substance:
  - `01`: the clustering example printed "Cluster-robust SE is 0.26x larger"
    for a cluster-robust standard error that is smaller; the section now
    explains why it is smaller in that design and when the opposite holds.
  - `02`: the true effect path is stated (it was mis-described and never
    compared with an estimate); `aggregate("event")`, which raises in 4.0, is
    gone; pre-treatment estimates are read with simultaneous bands instead of
    pointwise pass/fail; the Callaway-Sant'Anna vs Sun-Abraham "robustness
    check" is replaced by an explanation of why the two coincide there.
  - `03`: time weights are described correctly (fit on control units); the
    count-based placebo p-value and its floor are explained; shipped
    diagnostics replace hand-rolled ones.
  - `08`: both plain two-way DiDs are shown to miss the truth where the triple
    difference recovers it; the staggered example states its truth and
    explains the two overall summaries.
  - `09`: the Card-Krueger conclusion no longer rests on unclustered standard
    errors; the Castle doctrine section explains the TWFE vs
    Callaway-Sant'Anna gap by the baseline period and reads the pre-treatment
    estimates as a warning rather than passing over them.
