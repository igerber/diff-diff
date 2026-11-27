### Documentation
- **Staggered-estimator tutorials repaired and brought to the tutorial
  conventions** (`10_trop`, `11_imputation_did`, `12_two_stage_did`,
  `13_stacked_did`, `14_continuous_did`, `15_efficient_did`,
  `16_wooldridge_etwfe`). Each now states what it estimates and assumes before
  the first fit, checks estimates against a stated truth, and is committed
  with executed outputs and a drift test. Corrections of substance:
  - `10`: every fit used the default iteration cap and did not converge on the
    tutorial's data; the fits now converge and the cap is explained. The claim
    that TROP recovers the effect where Synthetic DiD is biased, which the
    notebook's own numbers contradicted, is replaced by a comparison across
    simulated panels with TROP cross-validated on each: DiD is unreliable
    under factor confounding, and TROP and Synthetic DiD have similar observed
    errors in that design. The five-draw Monte Carlo, the duplicated Summary and the
    stale plot annotation are gone.
  - `11`, `12`, `13`: the simulated data used the generator's default of
    effects that grow over time while every table was labelled "True effect =
    2.0" (the true average was about 2.42); the generator is now called with a
    constant effect, so the stated truth is the truth.
  - `11`: Imputation DiD is no longer described as efficient "under
    homogeneous effects". It is unbiased under arbitrary effect heterogeneity,
    and efficient under spherical errors (equal variance, uncorrelated across
    units and periods).
  - `12`: the pre-treatment coefficients the prose described are now produced
    (`pretrends=True`); the comparison with Imputation DiD separates what is
    identical (the overall ATT and the post-treatment point estimates) from
    what differs (standard errors, and the pre-treatment coefficients).
  - `13`: the overall estimate is stated to be an average over the event
    window; the clustering section no longer claims `unit_subexp` gives
    narrower standard errors (it gives wider ones in the example); clean
    control definitions are no longer ranked by "strength of identification".
  - `14`: the removed `overall_att` name is gone; strong parallel trends is
    described as an assumption about outcomes under treatment (no selection
    into dose on gains), which the old text mis-stated as the standard
    assumption.
  - `15`: the comparison set under PT-All is described correctly; the
    hard-coded "condition number > 100" flag, which marked most of the
    tutorial's own cells, is gone; `EfficientDiD.hausman_pretest` is shown.
  - `16` (Wooldridge): ETWFE and Callaway-Sant'Anna are compared on the same
    control group and the same pre-treatment reference period, where they
    coincide at every event time; the Poisson and logit examples gain a
    truth check.
  - Event studies in `14`, `15` and `16` are drawn with `plot_event_study`.
- **Imputation DiD efficiency condition corrected across the documentation**
  (README, the three `llms` guides, the estimator-choice page, references and
  the API pages): efficient under spherical errors (equal variance,
  uncorrelated across units and periods) and robust to effect heterogeneity, not "efficient under homogeneous
  effects". The Two-Stage API page's comparison table no longer says the
  two-stage and imputation estimators are consistent only under homogeneity.
- The `did2s` reference (Butts & Gardner 2022) is cited as *The R Journal*
  14(3), not 14(1).
