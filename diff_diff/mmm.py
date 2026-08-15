"""Assemble Marketing Mix Model (MMM) calibration inputs from experiment results.

MMM practitioners calibrate their models against experimental evidence. The two
dominant Python MMM frameworks consume that evidence in different shapes:

- **PyMC-Marketing** (and prophetverse, which mirrors the same schema) ingests a
  *lift-test* DataFrame with columns ``channel``, optional model dims (e.g. ``geo``),
  ``x`` (baseline channel spend), ``delta_x`` (spend change during the test),
  ``delta_y`` (measured incremental outcome), and ``sigma`` (the experiment's standard
  error), which it scores against the model's saturation curve via
  ``MMM.add_lift_test_measurements``. See
  https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html.
- **Google Meridian** has no experiment-ingestion API; calibration means setting a
  lognormal prior per channel - ``roi_m`` for the return of a channel's full spend
  (a full-holdout/zero-spend estimand), ``mroi_m`` for a marginal return. Google's
  documented workflow maps the experiment's ROI point estimate to the prior mean and
  its standard error to the prior standard deviation, converted to lognormal
  ``(mu, sigma)`` via the closed form used by
  ``meridian.model.prior_distribution.lognormal_dist_from_mean_std``:
  ``mu = ln(m) - ln((s/m)^2 + 1)/2``, ``sigma = sqrt(ln((s/m)^2 + 1))``. See
  https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments.

**Design: explicit numbers in, or the pinned aggregation contract in - validated out.**
Reconciling an experiment's estimate to a calibration input requires context diff-diff
cannot always see - the target MMM's row granularity (per-geo vs national), its time
window, and the outcome's scale (additive levels vs a log/rate/share). The default
route therefore stays fully explicit: the CALLER supplies the already-scoped
incremental outcome and its standard error (the numbers they read off a fitted
result's ``summary()``, aggregated to the population and window their MMM row
represents). Alternatively, both exporters accept ``aggregation_result=`` - the pinned
:class:`~diff_diff.aggregation.AggregationResult` container returned by
``results.aggregate('simple')`` / ``results.aggregate('group')`` - together with
``scale=``, and derive ``effect = att * scale`` / ``se_out = se * scale`` per row.
The derivation is fail-closed: raw results objects are rejected (this module never
calls ``aggregate()`` itself), and ``scale="auto"`` (reading the row count off the
container) is honored only for the audited producers whose ``n`` matches the ATT's
averaged support on unweighted, fully identified fits. diff-diff does only what it can verify:
assemble the exact target schema, enforce the sign/positivity/monotonicity guards each
consumer requires, convert to the lognormal parameterization (with parity to Google's
closed form), pool multiple experiments, and emit ready-to-paste snippets. It is pure
numpy/pandas and never imports an MMM package.
"""

from __future__ import annotations

import ast
import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# Runtime import (no cycle: aggregation.py depends only on results_base), so the
# exporters' public annotations resolve under typing.get_type_hints().
from diff_diff.aggregation import AggregationResult
from diff_diff.results_base import BaseResults

__all__ = ["MeridianROIPrior", "to_meridian_roi_prior", "to_pymc_marketing_lift_test"]

_WRONG_SIGN_POLICIES = ("raise", "drop", "keep")
_LIFT_TEST_RESERVED = frozenset({"channel", "x", "delta_x", "delta_y", "sigma"})

# Container-mode routing (audited against each producer's aggregation source):
# scale="auto" reads per-row n off the container ONLY for these producers, whose n
# matches the treated observations (unit-periods) the ATT averages over on
# unweighted, fully identified fits, at both
# supported levels. Routing keys on AggregationResult.estimator - NEVER on n_kind
# alone: CallawaySantAnna repeated-cross-section fits report n_kind="obs" with
# n = treated+control observations, so n_kind is only a drift sanity guard here.
_SCALE_AUTO_ESTIMATORS = frozenset({"ImputationDiD", "TwoStageDiD"})
_CONTAINER_LEVELS = ("simple", "group")

# Why scale="auto" is refused, per audited non-allowlisted producer. The generic
# closing prescription in the error is an EXAMPLE for the unweighted additive case,
# never a universal formula.
_SCALE_HINTS = {
    "CallawaySantAnna": (
        "CallawaySantAnna's n is not a treated unit-period count: the 'simple' "
        "container reports treated+control units (treated+control OBSERVATIONS on "
        "repeated-cross-section fits, where n_kind='obs' still does not mean "
        "treated unit-periods), and the 'group' container reports contributing "
        "(g, t) cells (n_kind='cells')."
    ),
    "EfficientDiD": (
        "EfficientDiD's n counts disjoint treated+control units ('simple') or "
        "contributing (g, t) cells ('group'), not treated unit-periods."
    ),
    "StackedDiD": (
        "StackedDiD's n is a deduplicated distinct-treated-unit count - units, not "
        "treated unit-periods, so no per-observation mass exists - and under "
        "weighting='population'/'sample_share' the ATT is a weighted estimand for "
        "which raw treated exposure is not the right multiplier; derive the scoped "
        "total per your weighting choice."
    ),
}

# Meridian prior parameters this exporter can target, with each one's Meridian
# default LogNormal(mu, sigma) per channel (verified against
# meridian/model/prior_distribution.py at 1.7.0): roi_m is the return on a
# channel's full spend (zero-spend counterfactual), mroi_m the marginal return.
# Channels without an experiment keep the default in the vector snippet.
_MERIDIAN_PARAM_DEFAULTS = {"roi_m": (0.2, 0.9), "mroi_m": (0.0, 0.5)}

# The .to_code() templates are pinned against google-meridian 1.7.0 (2026-06); they
# are convenience snippets, not a programmatic contract. Meridian's roi_m/mroi_m have
# batch shape n_media_channels; a scalar LogNormal broadcasts to EVERY channel, so the
# scalar template is gated behind an explicit single_channel opt-in.
_MERIDIAN_SINGLE_CHANNEL_TEMPLATE = """\
# Generated by diff_diff.mmm.to_meridian_roi_prior (template pinned to Meridian 1.7.0)
# TensorFlow-substrate snippet; for JAX-backed Meridian use
# `import tensorflow_probability.substrates.jax as tfp` instead.
# SINGLE-CHANNEL MODEL ONLY: a scalar {param} prior broadcasts to every media channel.
# For a multi-channel model, regenerate with to_code(channel=..., media_channels=[...]).
import tensorflow_probability as tfp
from meridian.model import prior_distribution, spec

roi_prior = tfp.distributions.LogNormal({mu!r}, {sigma!r}, name="{param}")
prior = prior_distribution.PriorDistribution({param}=roi_prior)
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type="{prior_type}",
    # {window_note}
    roi_calibration_period={calibration_period},
)
"""

_MERIDIAN_MULTI_CHANNEL_TEMPLATE = """\
# Generated by diff_diff.mmm.to_meridian_roi_prior (template pinned to Meridian 1.7.0)
# TensorFlow-substrate snippet; for JAX-backed Meridian use
# `import tensorflow_probability.substrates.jax as tfp` instead.
# Channel order MUST match your Meridian InputData media channel order exactly:
# {channels}
# {channel!r} carries the experiment-informed prior; the other channels keep
# Meridian's default {param} prior LogNormal({default_mu}, {default_sigma}).
import tensorflow_probability as tfp
from meridian.model import prior_distribution, spec

mu = {mu_vector!r}
sigma = {sigma_vector!r}
roi_prior = tfp.distributions.LogNormal(mu, sigma, name="{param}")
prior = prior_distribution.PriorDistribution({param}=roi_prior)
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type="{prior_type}",
    # {window_note}
    roi_calibration_period={calibration_period},
)
"""


def _is_sequence(value: Any) -> bool:
    """True for list-like containers (not strings, not mappings)."""
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _broadcast(name: str, value: Any, n: int) -> List[Any]:
    """Broadcast a scalar to n rows, or validate a sequence's length."""
    if _is_sequence(value):
        values = list(value)
        if len(values) != n:
            raise ValueError(
                f"{name} has length {len(values)} but {n} experiment(s) were given; "
                f"pass one value per experiment or a single scalar"
            )
        return values
    return [value] * n


def _seq_len(*values: Any) -> int:
    """Number of experiments implied by the first sequence argument (else 1)."""
    for v in values:
        if _is_sequence(v):
            n = len(list(v))
            if n == 0:
                raise ValueError("empty sequence given; provide at least one experiment")
            return n
    return 1


def _finite_positive(name: str, value: Any, index: int) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r} for experiment[{index}]")
    return v


def _normalize_dims(
    dims: Optional[Union[Mapping[str, str], Sequence[Mapping[str, str]]]], n: int
) -> Tuple[List[str], List[Mapping[str, str]]]:
    """Broadcast dims mappings and pin a single shared key set / column order."""
    if dims is None:
        return [], []
    if isinstance(dims, Mapping):
        rows: List[Mapping[str, str]] = [dims] * n
    else:
        rows = list(_broadcast("dims", dims, n))
        for i, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"dims must be a mapping or a sequence of mappings (e.g. "
                    f"{{'geo': 'US-CA'}}); dims[{i}] is {type(row).__name__}"
                )
    dim_cols = list(rows[0].keys())
    key_set = set(dim_cols)
    reserved = key_set & _LIFT_TEST_RESERVED
    if reserved:
        raise ValueError(
            f"dims keys {sorted(reserved)} collide with reserved lift-test columns "
            f"{sorted(_LIFT_TEST_RESERVED)}; rename the model dimension"
        )
    for i, row in enumerate(rows):
        if set(row.keys()) != key_set:
            raise ValueError(
                f"dims mappings must share one key set across rows; dims[{i}] has keys "
                f"{sorted(row.keys())}, expected {sorted(key_set)}"
            )
    return dim_cols, rows


def _extract_aggregation_rows(
    aggregation_result: Any,
    scale: Optional[Union[float, Sequence[float], str]],
    *,
    effect_name: str,
    se_name: str,
) -> Tuple[List[float], List[float], List[Any]]:
    """Derive per-row ``(effect, se)`` from a pinned ``AggregationResult``.

    Every per-row value (``label``, ``target``, ``att``, ``se``, ``n``) is read
    from ONE ``to_dataframe()`` call, so rows arrive in the order ``summary()``
    prints (sorted by label when sortable, producer order otherwise) - the
    alignment order for every per-row sequence kwarg, ``scale`` included.
    Fail-closed throughout: anything the container cannot verify raises with the
    remedy inline.
    """
    if not isinstance(aggregation_result, AggregationResult):
        if isinstance(aggregation_result, BaseResults):
            raise TypeError(
                f"aggregation_result must be an AggregationResult - the container "
                f"returned by res.aggregate('simple') or res.aggregate('group') on "
                f"estimators that produce one; got "
                f"{type(aggregation_result).__name__}. EventStudyResults and "
                f"estimators whose aggregate() does not return that container "
                f"(WooldridgeDiDResults returns the results object itself; "
                f"HeterogeneousAdoptionDiD event-study results support only "
                f"'event_study') have no container-mode route - pass the explicit "
                f"{effect_name}/{se_name} arguments instead"
            )
        raise TypeError(
            f"aggregation_result must be an AggregationResult (the return value of "
            f"res.aggregate('simple') or res.aggregate('group')); got "
            f"{type(aggregation_result).__name__}. Only AggregationResult is "
            f"supported."
        )
    level = aggregation_result.level
    if level not in _CONTAINER_LEVELS:
        raise ValueError(
            f"aggregation_result.level must be 'simple' or 'group'; got {level!r}. "
            f"Other levels ('calendar', 'dose', estimator-specific extras) have no "
            f"defined MMM-experiment mapping - re-aggregate at a supported level."
        )
    frame = aggregation_result.to_dataframe()
    if len(frame) == 0:
        raise ValueError(
            "aggregation_result has no rows (the aggregation selected no cells); "
            "nothing to export"
        )
    if level == "simple":
        target_list = list(frame["target"].tolist())
        offending = sorted({t for t in target_list if target_list.count(t) > 1})
        if offending:
            raise ValueError(
                f"aggregation_result has level='simple' but multiple rows for "
                f"target(s) {offending!r}; producers emit a single 'overall' row "
                f"per target for 'simple', so this container is out of contract - "
                f"re-aggregate, or use level='group' for per-cohort rows"
            )
    bad_targets = [(i, t) for i, t in enumerate(frame["target"].tolist()) if t != "att"]
    if bad_targets:
        idxs = [i for i, _ in bad_targets]
        targets = sorted({t for _, t in bad_targets})
        raise ValueError(
            f"aggregation_result row(s) {idxs} have target {targets!r} where 'att' "
            f"is required; only ATT rows map to an MMM incremental outcome "
            f"(ContinuousDiD's 'acrt' dose derivative, HeterogeneousAdoptionDiD's "
            f"WAS estimands, and ChaisemartinDHaultfoeuille's estimand relays are "
            f"not per-unit-period ATTs). The container is rejected whole - no "
            f"silent row filtering."
        )
    labels = frame["label"].tolist()
    atts = [float(v) for v in frame["att"].tolist()]
    ses = [float(v) for v in frame["se"].tolist()]
    ns = [float(v) for v in frame["n"].tolist()]
    for i, (att_i, se_i) in enumerate(zip(atts, ses)):
        if not (math.isfinite(att_i) and math.isfinite(se_i) and se_i > 0):
            raise ValueError(
                f"aggregation_result row [{i}] (label {labels[i]!r}) has "
                f"att={att_i!r}, se={se_i!r}; the fit carries no usable point "
                f"estimate/SE for this row, so it cannot calibrate an MMM"
            )
    n_rows = len(frame)
    if isinstance(scale, str):
        if scale != "auto":
            raise ValueError(
                f"scale must be a number, a sequence of numbers, or the string "
                f"'auto'; got {scale!r}"
            )
        estimator = aggregation_result.estimator
        if estimator not in _SCALE_AUTO_ESTIMATORS:
            if estimator is not None and estimator in _SCALE_HINTS:
                hint = _SCALE_HINTS[estimator]
            else:
                hint = (
                    f"estimator provenance is {estimator!r}, whose n semantics "
                    f"diff-diff has not audited for this derivation."
                )
            raise ValueError(
                f"scale='auto' is not available for this container: {hint} Pass a "
                f"numeric scale=<the factor converting this row's per-observation "
                f"ATT to your MMM row's total, e.g. treated units x treated "
                f"periods for an unweighted additive fit>, in to_dataframe() "
                f"order; auto-derivation is audited only for ImputationDiD and "
                f"TwoStageDiD, whose n is the treated observations the ATT "
                f"averages over."
            )
        n_kind = aggregation_result.n_kind
        if n_kind != "obs":
            raise ValueError(
                f"aggregation_result from {estimator} reports n_kind={n_kind!r}, "
                f"expected 'obs'; the container schema has drifted from the "
                f"audited contract - pass a numeric scale explicitly"
            )
        for i, n_i in enumerate(ns):
            if not (math.isfinite(n_i) and n_i > 0):
                raise ValueError(
                    f"aggregation_result row [{i}] (label {labels[i]!r}) has "
                    f"n={n_i!r}; cannot auto-derive scale - pass a numeric scale"
                )
        scales = ns
    elif scale is not None:
        scales = [
            _finite_positive("scale", v, i)
            for i, v in enumerate(_broadcast("scale", scale, n_rows))
        ]
    else:
        raise ValueError(
            f"scale is required with aggregation_result: pass a numeric "
            f"scale=<the factor converting each row's per-observation ATT to your "
            f"MMM row's total; e.g. treated units x treated periods for an "
            f"unweighted additive fit> (scalar or one value per row in "
            f"to_dataframe() order), or scale='auto' (ImputationDiD/TwoStageDiD "
            f"fits only - see the {effect_name} docstring for the assumptions "
            f"'auto' acknowledges)"
        )
    effects = [att_i * s_i for att_i, s_i in zip(atts, scales)]
    ses_out = [se_i * s_i for se_i, s_i in zip(ses, scales)]
    return effects, ses_out, labels


def to_pymc_marketing_lift_test(
    *,
    channel: Union[str, Sequence[str]],
    x: Union[float, Sequence[float]],
    delta_x: Union[float, Sequence[float]],
    delta_y: Optional[Union[float, Sequence[float]]] = None,
    sigma: Optional[Union[float, Sequence[float]]] = None,
    aggregation_result: Optional[AggregationResult] = None,
    scale: Optional[Union[float, Sequence[float], Literal["auto"]]] = None,
    dims: Optional[Union[Mapping[str, str], Sequence[Mapping[str, str]]]] = None,
    on_wrong_sign: str = "raise",
) -> pd.DataFrame:
    """Assemble a PyMC-Marketing lift-test DataFrame from scoped experiment results.

    Produces one row per experiment with columns ``channel``, any ``dims`` columns,
    ``x``, ``delta_x``, ``delta_y``, ``sigma`` - the exact schema consumed by
    ``pymc_marketing.mmm.MMM.add_lift_test_measurements`` (prophetverse's lift-test
    API accepts the same shape). All values stay in original data units
    (PyMC-Marketing rescales internally).

    **Two input routes.** Either the caller supplies the scoped effect explicitly -
    ``delta_y`` and ``sigma``, the measured incremental outcome and its standard
    error already aggregated to the population and time window ONE target-MMM row
    represents - or passes ``aggregation_result=`` (the pinned container returned by
    ``results.aggregate('simple')`` / ``results.aggregate('group')``) together with
    ``scale=``, and this function derives ``delta_y = att * scale`` and
    ``sigma = se * scale`` per container row. Rescaling is performed ONLY under that
    explicit contract - reconciliation context the container cannot carry (the MMM's
    row granularity, the outcome's scale) remains the caller's acknowledgement, see
    ``scale``. Either way, PyMC-Marketing scores one row's ``delta_y`` against
    ``saturation(x + delta_x) - saturation(x)``, so ``x``, ``delta_x``, ``delta_y``,
    and ``sigma`` must all describe the SAME observation (same channel, same
    population, same period span, additive-level outcome).

    Parameters
    ----------
    channel : str or sequence of str
        MMM channel name(s); must match the target model's ``channel_columns``.
    x : float or sequence of float
        Baseline channel spend for the row's observation, in original spend units.
    delta_x : float or sequence of float
        Spend change during the experiment (nonzero; negative for go-dark/holdout
        tests, with ``x + delta_x >= 0``), in the same units as ``x``.
    delta_y : float or sequence of float, optional
        Measured incremental outcome for the SAME observation as ``x``/``delta_x``,
        in original outcome units. Finite; must be nonzero and share ``delta_x``'s
        sign (see ``on_wrong_sign``). Required together with ``sigma`` unless
        ``aggregation_result`` is given (the two routes are mutually exclusive).
    sigma : float or sequence of float, optional
        Standard error of ``delta_y`` (finite, positive). Required together with
        ``delta_y`` unless ``aggregation_result`` is given.
    aggregation_result : AggregationResult, optional
        The pinned container returned by ``results.aggregate('simple')`` (one
        experiment row) or ``results.aggregate('group')`` (one row per cohort).
        Mutually exclusive with ``delta_y``/``sigma``; requires ``scale``. Rows are
        consumed in ``aggregation_result.to_dataframe()`` order - the order
        ``summary()`` prints - and every per-row sequence kwarg (``scale``, ``x``,
        ``delta_x``, ``dims``, ...) aligns to that order. Raw results objects and
        ``EventStudyResults`` are rejected; this function never calls
        ``aggregate()`` itself. On a bootstrapped fit, ``aggregate('simple')``
        relays the stored percentile SE, which is used as-is. Group-container
        cautions: (1) cohort rows come from ONE fit sharing controls and windows,
        yet each emitted lift row is scored by PyMC-Marketing as an independent
        observation with only its marginal ``sigma`` - the omitted cross-cohort
        covariance can misstate the joint evidence (overstating it when the net
        covariance is positive, as shared controls typically induce), and its
        direction cannot be determined from marginal SEs alone; (2) a scalar
        ``x`` or
        ``delta_x`` replicates to every cohort row, asserting the channel's FULL
        baseline spend / spend change produced only that cohort's ``delta_y`` -
        pass per-cohort values; (3) the emitted frame carries NO cohort-label
        column, so pass per-row ``dims`` to keep multi-cohort rows
        distinguishable.
    scale : float, sequence of float, or "auto", optional
        Converts each container row's per-observation ATT to the row's total
        incremental outcome: ``delta_y = att * scale``, ``sigma = se * scale``.
        Required with ``aggregation_result`` (finite, positive; scalar or one value
        per row in ``to_dataframe()`` order; e.g. treated units x treated periods
        for an unweighted additive fit - an example, not a universal formula).
        ``scale="auto"`` derives ``scale`` from the container's per-row ``n`` and is
        accepted ONLY for ImputationDiD and TwoStageDiD fits, whose ``n`` is the
        treated observations the ATT averages over. Passing ``"auto"`` acknowledges
        three assumptions the container cannot verify: the outcome is in additive
        levels (not log/rate/share); the fit is unweighted (on survey-weighted fits
        ``att`` is a weighted average while ``n`` is a raw count); and every
        treated observation's effect is identified (ImputationDiD averages finite
        tau-hat only while ``n`` stays the raw count at both levels; TwoStageDiD's
        'simple' ATT support excludes rows with non-finite first-stage residuals
        while ``n`` reports the pre-filter count, though its 'group' counts are
        post-filter - the affected fits warn at fit time on the degenerate
        branch, and ``"auto"`` there overcounts). An
        ``att * scale`` (or ``se * scale``) that overflows the float range
        surfaces via this function's ordinary finiteness errors - ``sigma`` is
        validated before ``delta_y``, so a row where both overflow reports
        ``sigma``.
    dims : mapping or sequence of mappings, optional
        Extra model-dimension columns, e.g. ``{"geo": "US-CA"}`` for a geo-level MMM
        built with ``MMM(dims=("geo",))``. Values must match the target model's
        coordinate values exactly and identify the population ``delta_y`` describes
        (do not label a multi-geo average with one specific geo). All rows must share
        one key set; column order follows the first mapping.
    on_wrong_sign : {"raise", "drop", "keep"}, default "raise"
        Policy for rows PyMC-Marketing cannot use: ``sign(delta_y)`` contradicting
        ``sign(delta_x)`` (rejected upstream with ``NonMonotonicError``), or
        ``delta_y == 0`` (degenerate for its strictly-positive Gamma lift likelihood,
        which its own monotonicity check does not catch). ``"raise"`` (default) errors
        with guidance; ``"drop"`` warns and removes such rows (raises if that would
        empty the frame); ``"keep"`` warns and emits them anyway (for non-PyMC
        consumers - the frame is not valid PyMC-Marketing input).

    Returns
    -------
    pd.DataFrame
        Columns ``[channel, *dims, x, delta_x, delta_y, sigma]``, one row per
        experiment.

    Raises
    ------
    ValueError
        On invalid inputs (non-positive ``sigma``, non-finite values, ``delta_x == 0``,
        ``x + delta_x < 0``, broadcasting length mismatches, empty inputs,
        heterogeneous ``dims`` key sets) or wrong-signed/zero rows under the default
        policy.
    """
    if on_wrong_sign not in _WRONG_SIGN_POLICIES:
        raise ValueError(
            f"on_wrong_sign must be one of {_WRONG_SIGN_POLICIES}; got {on_wrong_sign!r}"
        )

    if aggregation_result is not None:
        if delta_y is not None or sigma is not None:
            raise ValueError(
                "pass either aggregation_result= or delta_y=/sigma=, not both; "
                "aggregation mode derives delta_y and sigma from the container "
                "(delta_y = att * scale, sigma = se * scale)"
            )
        delta_y, sigma, _ = _extract_aggregation_rows(
            aggregation_result, scale, effect_name="delta_y", se_name="sigma"
        )
        n = len(delta_y)
    else:
        if scale is not None:
            raise ValueError(
                "scale only applies with aggregation_result=; without a container, "
                "pass the already-scaled delta_y and sigma directly"
            )
        if delta_y is None or sigma is None:
            missing = (
                "delta_y and sigma are"
                if delta_y is None and sigma is None
                else ("delta_y is" if delta_y is None else "sigma is")
            )
            raise ValueError(
                f"{missing} required when aggregation_result is not given; pass "
                f"both delta_y and sigma, or pass "
                f"aggregation_result=res.aggregate('simple'|'group') with scale= "
                f"to derive them from a fitted result"
            )
        n = _seq_len(channel, x, delta_x, delta_y, sigma, dims)
    channels = _broadcast("channel", channel, n)
    xs = _broadcast("x", x, n)
    delta_xs = _broadcast("delta_x", delta_x, n)
    delta_ys = _broadcast("delta_y", delta_y, n)
    sigmas = _broadcast("sigma", sigma, n)
    dim_cols, dim_rows = _normalize_dims(dims, n)

    rows: List[Dict[str, Any]] = []
    wrong_sign_rows: List[int] = []
    zero_lift_rows: List[int] = []
    for i in range(n):
        x_i = float(xs[i])
        dx_i = float(delta_xs[i])
        dy_i = float(delta_ys[i])
        sig_i = _finite_positive("sigma", sigmas[i], i)
        if not math.isfinite(x_i) or x_i < 0:
            raise ValueError(f"x must be finite and >= 0; got {xs[i]!r} for experiment[{i}]")
        if not math.isfinite(dx_i) or dx_i == 0:
            raise ValueError(
                f"delta_x must be finite and nonzero (the experiment changed spend); "
                f"got {delta_xs[i]!r} for experiment[{i}]"
            )
        post_spend = x_i + dx_i
        if not math.isfinite(post_spend) or post_spend < 0:
            raise ValueError(
                f"x + delta_x must be finite and >= 0 (post-test spend cannot be "
                f"negative or overflow; the saturation curve is evaluated at "
                f"x + delta_x); got {x_i!r} + {dx_i!r} = {post_spend!r} for "
                f"experiment[{i}]"
            )
        if not math.isfinite(dy_i):
            raise ValueError(f"delta_y must be finite; got {delta_ys[i]!r} for experiment[{i}]")
        if dy_i == 0:
            zero_lift_rows.append(i)
        elif (dx_i < 0) != (dy_i < 0):
            # Compare signs directly: dx_i * dy_i can underflow to -0.0 for tiny
            # magnitudes (e.g. 1e-200 * -1e-200), so a product < 0 check misses
            # wrong-signed rows. Both are strictly nonzero here (delta_x validated
            # above, delta_y != 0 handled just above), so the comparison is exact.
            wrong_sign_rows.append(i)
        row: Dict[str, Any] = {"channel": channels[i]}
        if dim_cols:
            row.update({k: dim_rows[i][k] for k in dim_cols})
        row.update({"x": x_i, "delta_x": dx_i, "delta_y": dy_i, "sigma": sig_i})
        rows.append(row)

    # Two invalid-row classes, one shared disposition. Wrong sign is what
    # PyMC-Marketing rejects with NonMonotonicError; zero lift passes its
    # monotonicity check but is degenerate for the strictly-positive Gamma lift
    # likelihood (an insignificant experiment that cannot calibrate saturation).
    if wrong_sign_rows or zero_lift_rows:
        parts = []
        if wrong_sign_rows:
            parts.append(
                f"row(s) {wrong_sign_rows} have sign(delta_y) contradicting "
                f"sign(delta_x) (PyMC-Marketing rejects these with NonMonotonicError)"
            )
        if zero_lift_rows:
            parts.append(
                f"row(s) {zero_lift_rows} have delta_y == 0 (degenerate for "
                f"PyMC-Marketing's strictly-positive Gamma lift likelihood, which its "
                f"monotonicity check does not catch)"
            )
        detail = "; ".join(parts)
        if on_wrong_sign == "raise":
            raise ValueError(
                f"{detail}. Pool experiments, re-scope, or exclude the offending "
                f"experiment; or pass on_wrong_sign='drop'/'keep' to handle these rows "
                f"explicitly."
            )
        dropped = set(wrong_sign_rows) | set(zero_lift_rows)
        if on_wrong_sign == "drop":
            if len(dropped) == len(rows):
                raise ValueError(f"on_wrong_sign='drop' would remove every row: {detail}.")
            warnings.warn(
                f"Dropping invalid lift-test row(s): {detail}.", UserWarning, stacklevel=2
            )
            rows = [row for i, row in enumerate(rows) if i not in dropped]
        else:  # "keep"
            warnings.warn(
                f"Keeping invalid lift-test row(s): {detail}. The frame is NOT valid "
                f"input for PyMC-Marketing's lift likelihood.",
                UserWarning,
                stacklevel=2,
            )

    columns = ["channel", *dim_cols, "x", "delta_x", "delta_y", "sigma"]
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class ExperimentROI:
    """Per-experiment ROI contribution inside a :class:`MeridianROIPrior`."""

    roi: float
    roi_sd: float
    spend: float
    weight: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "roi": self.roi,
            "roi_sd": self.roi_sd,
            "spend": self.spend,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class MeridianROIPrior:
    """Lognormal prior parameters for Meridian's ``roi_m``/``mroi_m`` calibration.

    ``mu``/``sigma`` reproduce ``meridian.model.prior_distribution.
    lognormal_dist_from_mean_std(roi_mean, roi_sd)`` exactly (closed form; no Meridian
    dependency). ``parameter`` records which Meridian prior the caller is informing
    (``"roi_m"`` or ``"mroi_m"``). ``per_experiment`` records each pooled experiment's
    ROI, widened sd, spend, and spend weight.
    """

    roi_mean: float
    roi_sd: float
    mu: float
    sigma: float
    parameter: str = "roi_m"
    per_experiment: Tuple[ExperimentROI, ...] = field(default=())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution": "LogNormal",
            "parameter": self.parameter,
            "roi_mean": self.roi_mean,
            "roi_sd": self.roi_sd,
            "mu": self.mu,
            "sigma": self.sigma,
            "per_experiment": [e.to_dict() for e in self.per_experiment],
        }

    def to_code(
        self,
        *,
        channel: Optional[str] = None,
        media_channels: Optional[Sequence[str]] = None,
        single_channel: bool = False,
        roi_calibration_period: Optional[str] = None,
        full_model_window: bool = False,
    ) -> str:
        """Ready-to-paste Meridian snippet (channel- and time-scoped; 1.7.0 pinned).

        Meridian's ``roi_m``/``mroi_m`` prior has batch shape ``n_media_channels`` and
        a scalar distribution broadcasts to EVERY media channel - a TV experiment's
        prior must not silently calibrate search, social, etc. Channel scope is
        therefore always explicit:

        - ``to_code(channel="tv", media_channels=["search", "tv"])`` emits vector
          ``mu``/``sigma`` in exactly the ``media_channels`` order (which must match
          the Meridian ``InputData`` channel order); the experiment channel carries
          this prior and every other channel keeps Meridian's default.
        - ``to_code(single_channel=True)`` emits the scalar snippet for
          single-channel models, marked as such in the generated code.
        - Calling with neither raises ``ValueError``.

        The prior's TIME scope is also required: Meridian's default
        ``roi_calibration_period=None`` applies the prior over all model times, but
        the prior was estimated on the experiment window. Pass
        ``roi_calibration_period="<expression>"`` (the boolean
        ``(n_media_times, n_media_channels)`` mask for your window) or
        ``full_model_window=True`` to acknowledge that the two coincide.

        Snippets use the TensorFlow substrate of TensorFlow Probability; JAX-backed
        Meridian users should swap the import for
        ``tensorflow_probability.substrates.jax`` (noted in the generated code).
        """
        if roi_calibration_period is None and not full_model_window:
            raise ValueError(
                "to_code() needs the prior's time scope: Meridian's default "
                "roi_calibration_period=None applies the prior over ALL model times, "
                "but this prior was estimated on the EXPERIMENT window, and ROI "
                "differs across windows under varying spend and saturation. Pass "
                "roi_calibration_period=<expression building the boolean "
                "(n_media_times, n_media_channels) mask for your experiment window>, "
                "or full_model_window=True to acknowledge that the MMM window and the "
                "experiment window coincide."
            )
        if roi_calibration_period is not None and full_model_window:
            raise ValueError(
                "pass either roi_calibration_period or full_model_window=True, not both"
            )
        if roi_calibration_period is not None:
            try:
                ast.parse(roi_calibration_period, mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"roi_calibration_period must be a valid Python expression building "
                    f"the boolean (n_media_times, n_media_channels) mask (it is "
                    f"interpolated verbatim into the snippet); got "
                    f"{roi_calibration_period!r}, which does not parse: {exc.msg}"
                ) from exc
        window_note = (
            "Experiment window == full model window (acknowledged via full_model_window=True)."
            if full_model_window
            else "Mask restricting the prior to the experiment window."
        )
        calibration_period = "None" if full_model_window else roi_calibration_period
        prior_type = "roi" if self.parameter == "roi_m" else "mroi"
        if media_channels is not None:
            if single_channel:
                raise ValueError("pass either media_channels or single_channel=True, not both")
            channels = list(media_channels)
            if not channels:
                raise ValueError("media_channels must be non-empty")
            if channel is None or channel not in channels:
                raise ValueError(
                    f"channel must name the experiment channel within media_channels; "
                    f"got channel={channel!r}, media_channels={channels!r}"
                )
            default_mu, default_sigma = _MERIDIAN_PARAM_DEFAULTS[self.parameter]
            mu_vector = [self.mu if c == channel else default_mu for c in channels]
            sigma_vector = [self.sigma if c == channel else default_sigma for c in channels]
            return _MERIDIAN_MULTI_CHANNEL_TEMPLATE.format(
                channels=channels,
                channel=channel,
                mu_vector=mu_vector,
                sigma_vector=sigma_vector,
                param=self.parameter,
                default_mu=default_mu,
                default_sigma=default_sigma,
                window_note=window_note,
                calibration_period=calibration_period,
                prior_type=prior_type,
            )
        if single_channel:
            return _MERIDIAN_SINGLE_CHANNEL_TEMPLATE.format(
                mu=self.mu,
                sigma=self.sigma,
                param=self.parameter,
                window_note=window_note,
                calibration_period=calibration_period,
                prior_type=prior_type,
            )
        raise ValueError(
            "to_code() needs explicit channel scope: a scalar prior broadcasts to "
            "every media channel in a multi-channel Meridian model. Pass channel=... "
            "with media_channels=[...] (vector prior in model channel order), or "
            "single_channel=True for a single-channel model."
        )


def to_meridian_roi_prior(
    *,
    incremental_outcome: Optional[Union[float, Sequence[float]]] = None,
    incremental_outcome_se: Optional[Union[float, Sequence[float]]] = None,
    aggregation_result: Optional[AggregationResult] = None,
    scale: Optional[Union[float, Sequence[float], Literal["auto"]]] = None,
    spend: Union[float, Sequence[float]],
    parameter: str = "roi_m",
    se_widening: float = 1.0,
) -> MeridianROIPrior:
    """Build a Meridian lognormal ROI/mROI prior from scoped experiment result(s).

    Per experiment ``roi = incremental_outcome / spend`` with standard deviation
    ``incremental_outcome_se / spend * se_widening``. Multiple experiments for the same
    channel are pooled with spend weights - the spend-weighted average ROI the Meridian
    FAQ suggests for multiple experiments (citing section 3.4 of Google's MMM
    calibration whitepaper): ``roi_mean = sum(w_i * roi_i)`` with
    ``w_i = spend_i / sum(spend)``. The pooled ``roi_sd = sqrt(sum((w_i * sd_i)^2))``
    is this library's uncertainty propagation for that average (treats experiments as
    independent - widen via ``se_widening`` when experiments share controls or windows).
    The pooled mean/sd map to lognormal ``(mu, sigma)`` via Google's closed form.

    **Two input routes.** Either the caller supplies the scoped estimand explicitly -
    ``incremental_outcome`` is the total incremental outcome the experiment measured,
    matching the estimand of the target prior: for ``parameter="roi_m"`` that is the
    outcome attributable to the channel's spend against a zero-spend counterfactual
    (a full holdout), divided here by ``spend`` = the channel's total spend over the
    window; for ``parameter="mroi_m"`` it is the marginal outcome of the spend
    change, divided by that spend change - or passes ``aggregation_result=`` with
    ``scale=``, and this function derives ``incremental_outcome = att * scale`` and
    ``incremental_outcome_se = se * scale`` per container row. Rescaling happens ONLY
    under that explicit contract; sign, estimand match, and population remain the
    caller's responsibility.

    Parameters
    ----------
    incremental_outcome : float or sequence of float, optional
        Total incremental outcome per experiment (the estimand matching ``parameter``),
        in the same currency/units as the MMM. Must be finite; the pooled ROI must be
        positive (lognormal support). Required together with
        ``incremental_outcome_se`` unless ``aggregation_result`` is given (the two
        routes are mutually exclusive).
    incremental_outcome_se : float or sequence of float, optional
        Standard error of ``incremental_outcome`` (finite, positive), computed for the
        caller's aggregation. Required together with ``incremental_outcome`` unless
        ``aggregation_result`` is given.
    aggregation_result : AggregationResult, optional
        The pinned container returned by ``results.aggregate('simple')`` (one
        experiment) or ``results.aggregate('group')`` (one experiment per cohort,
        feeding the spend-weighted pooling below). Mutually exclusive with
        ``incremental_outcome``/``incremental_outcome_se``; requires ``scale``. Rows
        are consumed in ``aggregation_result.to_dataframe()`` order - the order
        ``summary()`` prints - and per-row sequence kwargs (``scale``, ``spend``)
        align to that order. Raw results objects and ``EventStudyResults`` are
        rejected; this function never calls ``aggregate()`` itself. Group-container
        cautions: (1) cohort rows come from ONE fit sharing controls and windows,
        and the container carries only per-row SEs (the joint covariance's
        off-diagonals are discarded upstream), so the independence pooling below
        can misstate the pooled uncertainty - it is anti-conservative when the
        net weighted covariance is positive (shared controls typically induce
        positive correlation), and the direction cannot be determined from
        marginal SEs alone; ``se_widening > 1`` is a conservative heuristic, not
        an exact correction; (2) a scalar
        ``spend`` replicates per row, making ``total_spend = n_rows * spend`` and
        the pooled prior the arithmetic MEAN of cohort ROIs,
        ``sum(effect_i) / (n_rows * spend)`` - an n-fold understatement of channel
        ROI if the scalar was meant as the channel's TOTAL spend; pass per-cohort
        spends.
    scale : float, sequence of float, or "auto", optional
        Converts each container row's per-observation ATT to that experiment's total
        incremental outcome: ``incremental_outcome = att * scale``,
        ``incremental_outcome_se = se * scale``. Required with
        ``aggregation_result`` (finite, positive; scalar or one value per row in
        ``to_dataframe()`` order; e.g. treated units x treated periods for an
        unweighted additive fit - an example, not a universal formula).
        ``scale="auto"`` derives ``scale`` from the container's per-row ``n`` and is
        accepted ONLY for ImputationDiD and TwoStageDiD fits, whose ``n`` is the
        treated observations the ATT averages over. Passing ``"auto"`` acknowledges
        three assumptions the container cannot verify: additive-level outcome (not
        log/rate/share), an unweighted fit (survey-weighted fits pair a weighted
        ``att`` with a raw-count ``n``), and fully identified effects
        (ImputationDiD reports the raw treated-observation count at BOTH levels
        while averaging finite effects only; TwoStageDiD does so only at
        ``'simple'`` - its ``'group'`` counts are post-filter; the affected fits
        warn at fit time on the degenerate branch, and ``"auto"`` there
        overcounts). An overflowing ``att * scale`` surfaces via
        this function's ordinary finiteness errors - ``incremental_outcome`` is
        validated before ``incremental_outcome_se``, so a row where both overflow
        reports ``incremental_outcome``.
    spend : float or sequence of float
        Spend the incremental outcome is divided by (finite, positive): the channel's
        total spend over the window for ``roi_m``, or the spend change for ``mroi_m``.
    parameter : {"roi_m", "mroi_m"}, default "roi_m"
        Which Meridian prior the estimate informs. ``roi_m`` is the return on the
        channel's full spend (a full-holdout / zero-spend estimand); ``mroi_m`` is the
        marginal return of a spend change. Under saturation these differ, so the caller
        declares which their ``incremental_outcome`` measures; the returned prior and
        ``.to_code()`` target that parameter with its own Meridian default for
        non-experiment channels.
    se_widening : float, default 1.0
        Multiplier on each experiment's ROI standard deviation (finite, positive).
        Values above 1 encode skepticism about experiment-to-MMM transferability.

    Returns
    -------
    MeridianROIPrior
        Pooled ``roi_mean``/``roi_sd``, the lognormal ``mu``/``sigma``, the target
        ``parameter``, per-experiment detail, plus ``.to_dict()`` and the
        channel-scoped ``.to_code()`` snippet helper.

    Raises
    ------
    ValueError
        On an invalid ``parameter``, non-positive ``spend``/``se``/``se_widening``,
        non-finite inputs, broadcasting mismatches, empty inputs, a non-positive pooled
        ROI mean (lognormal support), or non-finite scaled/pooled results.
    """
    if parameter not in _MERIDIAN_PARAM_DEFAULTS:
        raise ValueError(
            f"parameter must be one of {sorted(_MERIDIAN_PARAM_DEFAULTS)}; got "
            f"{parameter!r}. roi_m is the return on the channel's full spend (a "
            f"full-holdout / zero-spend estimand); mroi_m is the marginal return of a "
            f"spend change. Under saturation they differ - pass the one your "
            f"incremental_outcome measures."
        )
    se_w = float(se_widening)
    if not math.isfinite(se_w) or se_w <= 0:
        raise ValueError(f"se_widening must be finite and > 0; got {se_widening!r}")

    if aggregation_result is not None:
        if incremental_outcome is not None or incremental_outcome_se is not None:
            raise ValueError(
                "pass either aggregation_result= or "
                "incremental_outcome=/incremental_outcome_se=, not both; "
                "aggregation mode derives them from the container "
                "(incremental_outcome = att * scale, "
                "incremental_outcome_se = se * scale)"
            )
        incremental_outcome, incremental_outcome_se, _ = _extract_aggregation_rows(
            aggregation_result,
            scale,
            effect_name="incremental_outcome",
            se_name="incremental_outcome_se",
        )
        n = len(incremental_outcome)
    else:
        if scale is not None:
            raise ValueError(
                "scale only applies with aggregation_result=; without a container, "
                "pass the already-scaled incremental_outcome and "
                "incremental_outcome_se directly"
            )
        if incremental_outcome is None or incremental_outcome_se is None:
            missing = (
                "incremental_outcome and incremental_outcome_se are"
                if incremental_outcome is None and incremental_outcome_se is None
                else (
                    "incremental_outcome is"
                    if incremental_outcome is None
                    else "incremental_outcome_se is"
                )
            )
            raise ValueError(
                f"{missing} required when aggregation_result is not given; pass "
                f"both, or pass aggregation_result=res.aggregate('simple'|'group') "
                f"with scale= to derive them from a fitted result"
            )
        n = _seq_len(incremental_outcome, incremental_outcome_se, spend)
    outcomes = _broadcast("incremental_outcome", incremental_outcome, n)
    outcome_ses = _broadcast("incremental_outcome_se", incremental_outcome_se, n)
    spends = _broadcast("spend", spend, n)

    rois: List[float] = []
    sds: List[float] = []
    spend_vals: List[float] = []
    for i in range(n):
        total = float(outcomes[i])
        if not math.isfinite(total):
            raise ValueError(
                f"incremental_outcome must be finite; got {outcomes[i]!r} for experiment[{i}]"
            )
        total_se = _finite_positive("incremental_outcome_se", outcome_ses[i], i)
        spend_i = _finite_positive("spend", spends[i], i)
        roi_i = total / spend_i
        sd_i = total_se / spend_i * se_w
        if not (math.isfinite(roi_i) and math.isfinite(sd_i) and sd_i > 0):
            raise ValueError(
                f"experiment[{i}] ROI is not finite-positive (roi={roi_i!r}, "
                f"roi_sd={sd_i!r}); the magnitudes overflowed or underflowed the float "
                f"range - rescale the outcome or spend units"
            )
        rois.append(roi_i)
        sds.append(sd_i)
        spend_vals.append(spend_i)

    total_spend = sum(spend_vals)
    weights = [s / total_spend for s in spend_vals]
    roi_mean = sum(w * r for w, r in zip(weights, rois))
    # math.hypot is a scaled Euclidean norm: it avoids the intermediate squaring
    # that would raise a raw OverflowError before the finiteness guard below.
    roi_sd = math.hypot(*(w * s for w, s in zip(weights, sds)))

    if not (math.isfinite(roi_mean) and math.isfinite(roi_sd) and roi_sd > 0):
        raise ValueError(
            f"pooled ROI moments are not finite-positive (roi_mean={roi_mean!r}, "
            f"roi_sd={roi_sd!r}); the per-experiment magnitudes overflowed or "
            f"underflowed - rescale the units"
        )
    if roi_mean <= 0:
        raise ValueError(
            f"Pooled ROI mean is {roi_mean:.6g} <= 0, which cannot map to Meridian's "
            f"lognormal {parameter} prior (strictly positive support). Options: pool "
            f"with additional experiments, use a wider prior from a defensible positive "
            f"range, or calibrate via Meridian's contribution/coefficient "
            f"parameterizations manually."
        )

    # log_term = log1p((roi_sd / roi_mean)**2), computed in the log domain so an
    # extreme coefficient of variation cannot overflow the squaring before the
    # finiteness guard: logaddexp(0, 2*ln(sd/mean)) == log(1 + (sd/mean)**2), and
    # it also keeps a tiny relative SE from rounding sigma to 0.
    log_term = float(np.logaddexp(0.0, 2.0 * (math.log(roi_sd) - math.log(roi_mean))))
    sigma = math.sqrt(log_term)
    mu = math.log(roi_mean) - 0.5 * log_term
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0):
        raise ValueError(
            f"lognormal parameters are not finite (mu={mu!r}, sigma={sigma!r}) - the "
            f"ROI mean/sd ratio is outside the representable range; rescale the outcome "
            f"or spend units and re-export"
        )

    per_experiment = tuple(
        ExperimentROI(roi=r, roi_sd=s, spend=sp, weight=w)
        for r, s, sp, w in zip(rois, sds, spend_vals, weights)
    )
    return MeridianROIPrior(
        roi_mean=roi_mean,
        roi_sd=roi_sd,
        mu=mu,
        sigma=sigma,
        parameter=parameter,
        per_experiment=per_experiment,
    )
