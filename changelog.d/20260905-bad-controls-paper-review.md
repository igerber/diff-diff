### Documentation
- **Caetano, Callaway, Payne & Sant'Anna (2026) "bad controls" paper review on file
  (PR-A).** Added `docs/methodology/papers/caetano-2026-review.md`, a paper-sourced
  fidelity review of the arXiv preprint 2608.03881v2 (plus its Supplementary Appendix
  v1, both SHA-256 pinned) - the Step-1 artifact for a prospective bad-controls
  extension of the DiD-with-covariates family. Transcribes the formal definition of a
  bad control (Conditions 1-2), the identification failure and the bias of the
  include / discard conventions (Section 3), the two new identification approaches
  (pre-treatment conditioning, Theorem 1 / Proposition 1; covariate unconfoundedness,
  Theorem 2), the staggered results (Theorem 3, Propositions 2-3, pre-tests incl.
  `ATT_X(g,t)`), the imputation estimator (Eqs. 5-7, influence function S8) and the
  Neyman-orthogonal DR / DML estimator (Eqs. 8-11, Algorithm 1, Propositions 5-7,
  Assumptions 9 / S2), the SC linearity alternative, all five Monte Carlo DGPs with
  Tables S1-S6 in full, and the NLSY job-displacement application. Records the
  GPL-3 status of the authors' `badcontrols` R package (black-box oracle only, no
  source port) and a `Relation to Existing diff-diff Estimators` section mapping
  Approach 1 onto `CallawaySantAnna` base-period covariates and the DR estimator onto
  the `DMLDiD` cross-fitting stack. Docs-only; no code change. This is a deliberate
  exception to the published-source rule, made for the authors' standing.
