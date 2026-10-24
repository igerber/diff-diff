### Internal
- **The TWFE event-study surface is built directly from the shared core's payload**
  (4.0 cut, PR 4c-i; ledger rows `M-010` / `M-011`): `DifferenceInDifferences._fit_event_study_core`
  returns a private `_EventStudyCorePayload`, `TwoWayFixedEffects.fit(..., event_study=True)`
  builds its `EventStudyResults` from it via `results_base._from_event_study_payload`, and only
  `MultiPeriodDiD.fit` still wraps the payload into `MultiPeriodDiDResults`. `_from_mpd` and the
  payload route share one calendar-surface builder, so they cannot drift; a committed oracle
  captured on the pre-refactor tree (`tests/fixtures/v4_twfe_event_study_oracles.json`, capture
  script `tests/_capture_v4_merge_mpd_oracles.py`) pins every field of the TWFE surface and of
  its MultiPeriodDiD twin across eleven variance lanes. No user-visible behavior changes.
- Spec item 4c is split into 4c-i (this PR) and 4c-ii (the removals); the 4c-ii bullet records
  the `spec="within"` benchmark-arm relabel and the `M-016` view's record type (user decisions
  2026-10-24). Ledger `M-010` / `M-011` notes updated; no status flips.

### Testing
- The MultiPeriodDiD-fixture suites (honest_did, pretrends, methodology pretrends/honest,
  visualization, Conley, survey, replicate weights, vcov_type, df_convention, the estimators
  event-study behaviour pins, the event-study surface roster, the DiagnosticReport /
  BusinessReport generic fixtures) now run on `TwoWayFixedEffects().fit(..., event_study=True)`
  (or a static TWFE fit for the report classes); MultiPeriodDiD's own tests, the MPD-only
  designs (multi-`absorb`, `fixed_effects=`, wild bootstrap, container-field assertions, the
  MPD-branch consumer tests) and the MPD-vs-TWFE parity oracles are unchanged until PR 4c-ii.
  New `tests/test_twfe_event_study_decoupling.py`, shared helpers
  `tests/helpers/twfe_es_oracle_lanes.py`, `event_study_average.py` and
  `event_study_surface_builders.py`; the variance-conventions inventory gains the two TWFE
  event-study rows; a TODO row tracks the one-way HC2-BM absorbed-FE R-parity lane.
