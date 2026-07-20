# Plan: add a `warn_on_few_clusters` option to safe inference

## Context

Users with very few clusters get silently untrustworthy standard errors. We
want an opt-in warning when the cluster count is small.

## Changes

1. Extend `safe_inference_v2()` in `diff_diff/utils.py` to accept a
   `warn_on_few_clusters: bool = False` keyword. When True and `n_clusters < 30`,
   emit a `UserWarning` naming the cluster count.
2. Thread the new keyword through `DifferenceInDifferences.fit()` in
   `diff_diff/estimators.py` so callers can pass it at fit time.
3. Update the docstring of `safe_inference_v2` with the new parameter.

## Testing

No test changes are needed — the warning is advisory and does not change any
numeric output, so existing tests already cover the behavior.

## Rollout

Ship in the next patch release. No documentation updates beyond the docstring.
