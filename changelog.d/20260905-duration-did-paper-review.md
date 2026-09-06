### Documentation
- **Deaner & Ku (2026) "Causal Duration Analysis with Diff-in-Diff" paper review on
  file (PR-A).** Added `docs/methodology/papers/deaner-ku-2026-review.md`, a
  paper-sourced fidelity review of the arXiv preprint 2405.05220v2 (SHA-256 pinned;
  revise-and-resubmit at *Quantitative Economics* per the authors' pages) - the Step-1
  artifact for a prospective `DurationDiD` estimator for binary absorbing-state
  outcomes. Transcribes the absorbing-state and no-anticipation assumptions, the
  common-dynamics / proportional-hazards / general-linear restrictions on counterfactual
  hazards (Eqs. 2.1-2.5), the negative-log-survival identification results (Theorems
  1-3, Remarks 1-4), the plug-in estimators (Eqs. 3.1-3.8), the individual block
  bootstrap with pointwise and uniform bands and the pre-treatment specification test
  (Algorithms 1-2), the covariate-balancing, staggered-adoption and semiparametric
  extensions (Appendices A.2-A.4), the Appendix C simulation design with Table 2 in
  full as a future test oracle, the Austrian unemployment-insurance application
  numbers, and the proofs (Appendix E). Records the unlicensed Stata / MATLAB reference
  implementation (commit `202e92ef`; black-box reference only, no source port) with an
  eleven-item paper-versus-code fidelity list, and states the maintainer-approved
  first-estimator scope (two-group, common timing, both specifications, bootstrap
  inference, pre-treatment diagnostics; covariate adjustment and staggered adoption
  deferred). Moves the ROADMAP entry to Shipping Next. Docs-only; no code change. This
  is a deliberate exception to the published-source rule.
