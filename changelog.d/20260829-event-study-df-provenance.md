### Fixed
- **Per-row event-study df provenance (M-092 completion)** for the four
  remaining holes — `EfficientDiD`, `ImputationDiD`, `ContinuousDiD`, and
  `HeterogeneousAdoptionDiD`: each results class gains a results-level
  `event_study_df` scalar (appended last; positional `__init__` indexes
  unchanged) threaded into the unified event-study container's per-row `df`
  column, which was all-NaN even on survey fits whose p-values were governed
  by a finite survey df. Finite on analytical survey fits (ImputationDiD:
  the final replicate-override df, level-matched on replicate replays, lead
  rows included; EfficientDiD: the post-overall snapshot; HAD: the
  unit-level design df); `None` — never the replicate-undefined `0`
  sentinel — on non-survey fits, on bootstrapped fits (percentile inference
  used no df, matching the shipped producer convention), and when no
  event-study surface was built. Inference values are unchanged everywhere.
- **`ContinuousDiD` `survey_metadata` granularity unified across inference
  branches**: the bootstrap and degenerate no-post-cells arms now publish
  the same UNIT-level metadata as the analytical arm (the
  CS/EfficientDiD convention); previously they kept the obs-level resolve,
  so `sum_weights`/`effective_n`/`n_psu` — and `df_survey` on implicit-PSU
  designs — differed from the analytical arm by panel length on the same
  data. Metadata provenance only; estimates and inference unchanged.
- **Documented (no behavior change)**: the event-study container's
  `df_survey` SCALAR — the fit's resolved scalar inference df — deliberately
  persists on bootstrapped fit-time and replayed surfaces (CS, DMLDiD,
  EfficientDiD identically) as the consumer channel HonestDiD's container
  branches read; the per-row `df` column is the inference-provenance channel
  that percentile bootstrap clears. Recorded as a REGISTRY Note with a
  cross-estimator parity pin.
