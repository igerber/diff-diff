### Added
- **Chang (2020) §4.2.2 RCS simulation-DGP replication** (`tests/test_methodology_dml_did.py`,
  DML PR-B2): the paper's own kernel-design repeated-cross-section DGP as
  maintainer validation fixtures for `DMLDiD(panel=False)` — a DGP-shape pin
  (distributions, all three innovation scales, the design's built-in confounded
  contrast → θ₀+1, and both correct-specification facts), seed-pinned recovery at
  both paper sample sizes with a discriminating comparison against the unadjusted
  contrast, and a slow Monte Carlo coverage lane. The §4.2 parameterizations are
  extracted into the paper review; the §4.2.1 ML design is documented as not
  replicable with the bundled unpenalized learners (narrowed TODO row) and the
  REGISTRY carries the replication-scope Note.
