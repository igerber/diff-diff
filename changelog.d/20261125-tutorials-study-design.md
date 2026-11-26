### Documentation
- **Study-design tutorials repaired and brought to the tutorial conventions**
  (`04_parallel_trends`, `05_honest_did`, `06_power_analysis`,
  `07_pretrends_power`, `24_staggered_vs_collapsed_power`). Each now states
  what it estimates and assumes before the first fit, checks results against a
  stated truth, and is committed with executed outputs and a drift test.
  Corrections of substance:
  - `04`: pre-treatment diagnostics are presented as evidence, not as a
    pass/fail gate; the group placebo passes `treatment=` so that treated
    units are dropped (without it they sat in the placebo control group); the
    permutation example no longer labels the placebo mean as the estimate.
  - `05`: the relative-magnitudes restriction is stated on first differences,
    as implemented (it was written on levels, which cannot reproduce the
    output); the breakdown value is searched on a grid that reaches it; the
    conventional interval is shown to miss the simulated truth where both
    honest intervals cover it; relative magnitudes and smoothness are no
    longer compared at "the same M"; `l_vec` and the staggered route are
    shown.
  - `06`: closed-form and simulated sample sizes are compared for the same
    design (32 against 35 units, where the old comparison set 63 against 42
    and blamed simulation noise); `sigma` and `rho` are defined once, with the
    double-counting trap worked through; panel simulations cluster by unit.
  - `07`: the linear minimum detectable violation is treated as a slope and
    translated into the bias it implies, and is no longer passed to Honest DiD
    as a smoothness bound; power grids sit on the scale of the MDV instead of
    saturating at 100%; the form of the pre-test is named; a new section shows
    by simulation that, in the notebook's design, bias is larger and coverage
    lower among the samples that pass the pre-test.
  - `24`: the removed `overall_att` name is gone from the prose; the event
    study is drawn with `plot_event_study`; blanket warning filters are
    removed (the fits are warning-free); the claim that the `simulate_*`
    functions draw independently per estimator is corrected.
