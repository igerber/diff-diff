"""
de Chaisemartin-D'Haultfoeuille (dCDH) estimator for reversible-treatment DiD.

The dCDH estimator is the most general DiD estimator in the diff-diff library
for **non-absorbing (reversible) treatments** — treatment can switch on AND off
over time, switcher vs non-switcher comparisons are its primitive object, and it
allows dynamic (carryover) effects with explicit joiner/leaver (``DID_+`` /
``DID_-``) decomposition. ``LPDiD`` (``non_absorbing="first_entry"`` /
``"effect_stabilization"``) and ``TROP`` (``non_absorbing=True``, under a
no-dynamic-effects assumption) also accept non-absorbing treatment under stronger
assumptions. The remaining staggered estimators in the library
(``CallawaySantAnna``, ``SunAbraham``, ``ImputationDiD``, ``TwoStageDiD``,
``EfficientDiD``, ``WooldridgeDiD``) assume treatment is absorbing.

Phase 1 ships the contemporaneous-switch case ``DID_M`` (= ``DID_1`` at
horizon ``l = 1`` of the dynamic companion paper). Phases 2 and 3 add
dynamic horizons and covariates respectively, on the *same* class — see
``ROADMAP.md`` for the full progression. The forward-compatibility
parameters in :meth:`ChaisemartinDHaultfoeuille.fit` raise
``NotImplementedError`` with phase pointers until later phases land.

References
----------
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2022, revised July 2023).
  Difference-in-Differences Estimators of Intertemporal Treatment Effects.
  NBER Working Paper 29873. Web Appendix Section 3.7.3 contains the
  cohort-recentered plug-in variance formula implemented here.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import NOT_SUPPLIED, require_arg, resolve_renamed_kwarg
from diff_diff.chaisemartin_dhaultfoeuille_bootstrap import (
    ChaisemartinDHaultfoeuilleBootstrapMixin,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import (
    ChaisemartinDHaultfoeuilleResults,
    DCDHBootstrapResults,
)
from diff_diff.linalg import solve_ols
from diff_diff.utils import safe_inference

__all__ = [
    "ChaisemartinDHaultfoeuille",
    "chaisemartin_dhaultfoeuille",
    "twowayfeweights",
    "TWFEWeightsResult",
]


# =============================================================================
# Public dataclass for the standalone TWFE diagnostic helper
# =============================================================================


class TWFEWeightsResult:
    """
    Lightweight container for the standalone ``twowayfeweights`` helper.

    Returned by :func:`twowayfeweights`. Mirrors the per-cell decomposition
    information that the dCDH estimator stores on its results object when
    ``twfe_diagnostic=True``, but available as a standalone function for
    users who only want the diagnostic without fitting the full estimator.
    """

    __slots__ = ("weights", "fraction_negative", "sigma_fe", "beta_fe")

    def __init__(
        self,
        weights: pd.DataFrame,
        fraction_negative: float,
        sigma_fe: float,
        beta_fe: float,
    ) -> None:
        self.weights = weights
        self.fraction_negative = fraction_negative
        self.sigma_fe = sigma_fe
        self.beta_fe = beta_fe

    def __repr__(self) -> str:
        return (
            f"TWFEWeightsResult(beta_fe={self.beta_fe:.4f}, "
            f"fraction_negative={self.fraction_negative:.4f}, "
            f"sigma_fe={self.sigma_fe:.4f}, n_cells={len(self.weights)})"
        )


# =============================================================================
# Shared validation + cell aggregation helper
# =============================================================================


def _validate_and_aggregate_to_cells(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    time: str,
    treatment: str,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Validate input data and aggregate to ``(g, t)`` cells per the dCDH contract.

    Used by both :meth:`ChaisemartinDHaultfoeuille.fit` and
    :func:`twowayfeweights` so the validation rules and aggregation
    behavior are identical across the two public entry points.

    The contract (matching ``REGISTRY.md`` ``## ChaisemartinDHaultfoeuille``):

    1. **Required columns** ``outcome``, ``group``, ``time``, ``treatment``
       must all be present in ``data`` (raises ``ValueError`` listing
       any missing).
    2. **Treatment** must coerce to numeric and contain no ``NaN``
       (raises ``ValueError`` — silent dropping would change cell counts
       without informing the user).
    3. **Outcome** must coerce to numeric and contain no ``NaN`` (same
       reasoning).
    4. **Treatment** must be numeric. Both binary ``{0, 1}`` and
       non-binary (ordinal or continuous) treatment are supported.
       Non-binary treatment requires ``L_max >= 1`` in ``fit()`` because
       the per-period DID path uses binary joiner/leaver categorization.
    5. **Cell aggregation** via ``groupby([group, time]).agg(...)``
       producing ``y_gt`` (cell mean of ``outcome``), ``d_gt`` (cell
       mean of ``treatment``), and ``n_gt`` (count of original
       observations in the cell).
    6. **Within-cell-varying treatment** (any cell where ``d_min !=
       d_max``) raises ``ValueError``. Treatment must be constant
       within each ``(group, time)`` cell; fuzzy DiD is deferred to a
       separate dCdH 2018 paper. Pre-aggregate your data to constant
       cell-level treatment before calling ``fit()`` or
       ``twowayfeweights()``.

    When ``weights`` is provided (survey pweights), cell means use weighted
    averages: ``y_gt = sum(w_i * y_i) / sum(w_i)``.  An additional column
    ``w_gt`` (total weight per cell) is included in the output for downstream
    IF expansion.

    Returns the aggregated cell DataFrame with columns
    ``[group, time, y_gt, d_gt, n_gt]`` (plus ``w_gt`` when weighted),
    sorted by ``[group, time]`` with a fresh index.

    Raises
    ------
    ValueError
        On missing columns; NaN values in any of the ``group``, ``time``,
        ``treatment``, or ``outcome`` columns (``group`` and ``time`` are
        rejected pre-``groupby`` because ``groupby`` silently drops NaN
        keys, which would change the estimation sample without warning);
        non-numeric treatment / outcome that cannot be coerced via
        ``pd.to_numeric``; or within-cell-varying treatment (any
        ``(group, time)`` cell where ``d_min != d_max``, since fuzzy DiD
        is out of scope and deferred to a separate dCdH 2018 paper).
        Integer-coded non-binary treatment (the ``by_path`` /
        ``paths_of_interest`` requirement) is enforced separately at
        ``fit()`` time, not here at aggregation time — this helper
        accepts continuous ``d_gt`` cell means and lets ``fit()`` decide
        whether the integer-only contract applies.

        Under the survey-weighted path (``weights`` is not ``None``),
        zero-weight rows are pre-filtered before any NaN / coercion /
        within-cell validation per the ``SurveyDesign.subpopulation()``
        out-of-sample contract — invalid values in zero-weight rows
        therefore do NOT raise. NaN / coercion / within-cell checks
        still apply to all positive-weight rows.
    """
    # 1. Required columns
    missing = [c for c in (outcome, group, time, treatment) if c not in data.columns]
    if missing:
        raise ValueError(
            f"ChaisemartinDHaultfoeuille / twowayfeweights: column(s) {missing!r} "
            f"not found in data. Required columns: outcome, group, time, treatment."
        )

    df = data.copy()

    # 1a. SurveyDesign.subpopulation() contract: zero-weight rows are
    # out-of-sample. Pre-filter them *before* any NaN/coercion validation
    # so that invalid values in excluded rows do not abort the fit.
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        pos_mask = weights_arr > 0
        if not pos_mask.all():
            df = df.loc[pos_mask].reset_index(drop=True)
            weights = weights_arr[pos_mask]

    # 1b. Group and time NaN checks (before groupby, which silently drops NaN keys)
    n_nan_group = int(df[group].isna().sum())
    if n_nan_group > 0:
        raise ValueError(
            f"Group column {group!r} contains {n_nan_group} NaN value(s). "
            "groupby silently drops NaN keys, which would change the "
            "estimation sample without warning. Drop or impute NaN group "
            "values before calling fit() or twowayfeweights()."
        )
    n_nan_time = int(df[time].isna().sum())
    if n_nan_time > 0:
        raise ValueError(
            f"Time column {time!r} contains {n_nan_time} NaN value(s). "
            "groupby silently drops NaN keys, which would change the "
            "estimation sample without warning. Drop or impute NaN time "
            "values before calling fit() or twowayfeweights()."
        )

    # 2. Treatment numeric coercion + NaN check
    try:
        df[treatment] = pd.to_numeric(df[treatment])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not coerce treatment column {treatment!r} to numeric: {exc}"
        ) from exc
    n_nan_treat = int(df[treatment].isna().sum())
    if n_nan_treat > 0:
        raise ValueError(
            f"Treatment column {treatment!r} contains {n_nan_treat} NaN value(s). "
            "ChaisemartinDHaultfoeuille requires non-missing treatment indicators "
            "on every observation; impute or drop NaN treatment rows before fitting "
            "so the dropped count is explicit."
        )

    # 3. Outcome numeric coercion + NaN check
    try:
        df[outcome] = pd.to_numeric(df[outcome])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not coerce outcome column {outcome!r} to numeric: {exc}") from exc
    n_nan_outcome = int(df[outcome].isna().sum())
    if n_nan_outcome > 0:
        raise ValueError(
            f"Outcome column {outcome!r} contains {n_nan_outcome} NaN value(s). "
            "Drop or impute missing outcomes before calling fit() so the "
            "exclusion is explicit (silently averaging over present values "
            "would distort per-cell means)."
        )

    # 4. Treatment must be numeric (binary or non-binary both accepted)
    # No longer enforces {0, 1} - non-binary and continuous treatment supported.

    # 5. Cell aggregation (compute min/max for within-cell check)
    if weights is not None:
        # Survey-weighted cell aggregation (zero-weight rows already
        # filtered upstream at step 1a).
        # y_gt = sum(w_i * y_i) / sum(w_i) within each (g, t) cell.
        # Treatment is constant within cells (checked below), so weighted
        # and unweighted means are identical for d_gt.
        df["_w_"] = weights
        df["_wy_"] = weights * df[outcome].values
        g_obj = df.groupby([group, time], as_index=False)
        cell = g_obj.agg(
            _wy_sum=("_wy_", "sum"),
            w_gt=("_w_", "sum"),
            d_gt=(treatment, "mean"),
            d_min=(treatment, "min"),
            d_max=(treatment, "max"),
            n_gt=(treatment, "count"),
        )
        cell["y_gt"] = cell["_wy_sum"] / cell["w_gt"]
        cell = cell.drop(columns=["_wy_sum"])
        # Zero-weight cells: drop entirely so downstream validators
        # (ragged-panel, baseline requirement) don't see them.
        zero_w_mask = cell["w_gt"] <= 0
        if zero_w_mask.any():
            cell = cell[~zero_w_mask].reset_index(drop=True)
        df.drop(columns=["_w_", "_wy_"], inplace=True)
    else:
        cell = df.groupby([group, time], as_index=False).agg(
            y_gt=(outcome, "mean"),
            d_gt=(treatment, "mean"),
            d_min=(treatment, "min"),
            d_max=(treatment, "max"),
            n_gt=(treatment, "count"),
        )

    # 6. Within-cell-varying treatment rejection.
    # All observations in a cell must have the same treatment value
    # (for both binary and non-binary treatment). Detect by checking
    # that cell min equals cell max.
    non_constant_mask = cell["d_min"] != cell["d_max"]
    if non_constant_mask.any():
        n_non_constant = int(non_constant_mask.sum())
        example_cells = cell.loc[non_constant_mask, [group, time, "d_gt", "d_min", "d_max"]].head(5)
        raise ValueError(
            f"Within-cell-varying treatment detected in {n_non_constant} "
            f"(group, time) cell(s). dCDH requires treatment to be "
            f"constant within each (group, time) cell. Cells where "
            f"d_min != d_max indicate that some units have different "
            f"treatment values. Pre-aggregate your data to constant "
            f"cell-level treatment before calling fit() or "
            f"twowayfeweights(). Fuzzy DiD is deferred to a separate "
            f"dCDH paper (see ROADMAP.md out-of-scope). Affected cells "
            f"(first 5):\n{example_cells}"
        )
    # Drop the min/max columns; keep d_gt as float (no int cast - supports
    # ordinal and continuous treatment). w_gt retained when weighted.
    drop_cols = ["d_min", "d_max"]
    cell = cell.drop(columns=drop_cols)

    # Sort to ensure deterministic order in downstream operations
    cell = cell.sort_values([group, time]).reset_index(drop=True)
    return cell


def _validate_paths_of_interest(
    paths_of_interest: Any,
) -> List[Tuple[int, ...]]:
    """Validate and canonicalize ``paths_of_interest`` to ``List[Tuple[int, ...]]``.

    Rejects non-sequence inputs, empty lists, non-tuple/list path entries,
    empty path entries, non-int elements (including ``bool`` and
    ``np.bool_``), and entries with mixed lengths. Numpy integer types
    (``np.integer``) are accepted and canonicalized to Python ``int``
    so the resulting tuples are usable as dict keys interchangeably with
    paths emitted by ``_enumerate_treatment_paths`` (which casts via
    ``int(round(float(v)))``).
    """
    if not isinstance(paths_of_interest, (list, tuple)):
        raise ValueError(
            f"paths_of_interest must be a list/tuple of int tuples, "
            f"got {type(paths_of_interest).__name__}."
        )
    if len(paths_of_interest) == 0:
        raise ValueError("paths_of_interest must be non-empty.")
    canonical: List[Tuple[int, ...]] = []
    for i, p in enumerate(paths_of_interest):
        if not isinstance(p, (list, tuple)):
            raise ValueError(
                f"paths_of_interest[{i}] must be a tuple/list of ints, " f"got {type(p).__name__}."
            )
        if len(p) == 0:
            raise ValueError(f"paths_of_interest[{i}] must be non-empty.")
        canonical_path: List[int] = []
        for j, v in enumerate(p):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise ValueError(
                    f"paths_of_interest[{i}][{j}] must be an int, got "
                    f"{v!r} of type {type(v).__name__}."
                )
            canonical_path.append(int(v))
        canonical.append(tuple(canonical_path))
    lens = {len(p) for p in canonical}
    if len(lens) > 1:
        raise ValueError(
            f"paths_of_interest entries must all have the same length "
            f"(L_max+1); got mixed lengths {sorted(lens)}."
        )
    return canonical


# =============================================================================
# Main estimator class
# =============================================================================


class ChaisemartinDHaultfoeuille(ChaisemartinDHaultfoeuilleBootstrapMixin, BaseEstimator):
    """
    de Chaisemartin-D'Haultfoeuille (dCDH) estimator.

    The most general library estimator for **reversible (non-absorbing)
    treatments** - treatment may switch on AND off over time, with explicit
    joiner/leaver (``DID_+`` / ``DID_-``) decomposition (``LPDiD`` and ``TROP``
    also support non-absorbing treatment under stronger assumptions; see their
    ``non_absorbing`` parameters). Computes the contemporaneous-switch DiD ``DID_M`` from the
    AER 2020 paper (equivalently ``DID_1`` at horizon ``l = 1`` of the
    dynamic companion paper, NBER WP 29873) plus the full multi-horizon
    event study ``DID_l`` for ``l = 1..L_max`` via the ``L_max`` parameter
    on :meth:`fit`.

    Supported:

    - Headline ``DID_M`` plus multi-horizon ``DID_l`` event study
    - Joiners-only ``DID_+`` and leavers-only ``DID_-`` decompositions
    - Single-lag placebo ``DID_M^pl`` and dynamic placebos ``DID^{pl}_l``
      (computed automatically by default; gate via ``placebo=False``)
    - Analytical SE via the cohort-recentered plug-in formula from Web
      Appendix Section 3.7.3; multiplier bootstrap clustered at the group
      level by default via ``n_bootstrap``; under ``survey_design`` with
      strictly-coarser PSUs the bootstrap automatically upgrades to
      PSU-level Hall-Mammen wild clustering (see REGISTRY.md
      ``ChaisemartinDHaultfoeuille`` Note on survey + bootstrap)
    - Normalized estimator ``DID^n_l``, cost-benefit aggregate ``delta``,
      and sup-t simultaneous confidence bands
    - Residualization-style covariate adjustment (``DID^X``) via
      ``covariates=``, group-specific linear trends (``DID^{fd}``) via
      ``trends_linear=True``, state-set-specific trends via
      ``trends_nonparam=``, heterogeneity testing, non-binary treatment,
      HonestDiD sensitivity integration on placebos via ``honest_did=True``
    - Per-path event-study disaggregation via ``by_path=k`` (top-k most
      common observed treatment paths within the window
      ``[F_g-1, F_g-1+L_max]``; requires ``drop_larger_lower=False``;
      supports binary or integer-coded discrete treatment) or via
      ``paths_of_interest=[(...), ...]`` for an explicit user-specified
      path subset (Python-only API; mutex with ``by_path=k``)
    - Survey support via ``survey_design=``: pweight with strata/PSU/FPC
      via Taylor Series Linearization (analytical) or replicate-weight
      variance (BRR/Fay/JK1/JKn/SDR)
    - TWFE decomposition diagnostic from Theorem 1 of AER 2020

    Only ``aggregate`` on :meth:`fit` still raises ``NotImplementedError``.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional, default=None
        Must be ``None`` (the default). User-specified clustering via
        this kwarg is not supported — passing any non-``None`` value
        raises ``NotImplementedError`` at construction time (and the
        same gate fires from ``set_params``). The effective clustering
        depends on how you call ``fit()``:

        - **Default (no survey_design)**: clustered at the group level
          via the cohort-recentered influence-function plug-in
          (analytical SEs) and the multiplier bootstrap.
        - **Under ``survey_design`` with auto-inject or explicit
          ``psu=group``**: PSU coincides with the group and the
          group-level and PSU-level paths are bit-identical.
        - **Under ``survey_design`` with strictly-coarser PSUs**: the
          multiplier bootstrap automatically upgrades to PSU-level
          Hall-Mammen wild clustering.

        So dCDH does NOT always cluster at the group level — see
        REGISTRY.md ``ChaisemartinDHaultfoeuille`` Notes on cluster
        contract and survey + bootstrap for the full matrix. Custom
        user-specified clustering at a coarser or finer level than the
        group is a planned extension.
    n_bootstrap : int, default=0
        Number of multiplier-bootstrap iterations. ``0`` (default) uses
        only the analytical SE. Set to ``999`` or higher for stable
        bootstrap inference.
    bootstrap_weights : str, default="rademacher"
        Type of multiplier-bootstrap weights: ``"rademacher"``,
        ``"mammen"``, or ``"webb"``. Ignored unless ``n_bootstrap > 0``.
    seed : int, optional
        Random seed for the multiplier bootstrap.
    placebo : bool, default=True
        If ``True`` (default), automatically compute the single-lag
        placebo ``DID_M^pl`` (AER 2020 placebo specification) on the same data.
        Set to ``False`` to skip the placebo computation for speed; the
        results object will still expose ``placebo_*`` fields, but with
        NaN values and ``placebo_available=False``.
    twfe_diagnostic : bool, default=True
        If ``True`` (default), compute the TWFE decomposition diagnostic
        from Theorem 1 of AER 2020: per-``(g, t)`` weights, fraction of
        treated cells with negative weights, and ``sigma_fe`` (the
        smallest cell-effect standard deviation that could flip the sign
        of the plain TWFE coefficient). The diagnostic answers "what
        would the plain TWFE estimator say on the data you passed in?",
        so it runs on the **FULL pre-filter cell sample** (the same
        input as the standalone :func:`twowayfeweights` function), NOT
        on the post-filter estimation sample used by ``DID_M``. When
        the ragged-panel filter or ``drop_larger_lower`` drops groups,
        the fitted ``results.twfe_*`` values describe a LARGER sample
        (pre-filter) than ``results.overall_att`` and a ``UserWarning``
        is emitted to make the divergence explicit. See REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` ``Note (TWFE diagnostic sample
        contract)`` for the full rationale.
    drop_larger_lower : bool, default=True
        If ``True`` (default, matches R ``DIDmultiplegtDYN``), drops
        groups whose treatment switches more than once (multi-switch
        groups) before estimation. This is required for the analytical
        variance formula to be consistent with the AER 2020 Theorem 3
        point estimate — both formulas operate on the same post-drop
        dataset. Setting to ``False`` is supported for diagnostic
        comparison but produces an inconsistent estimator-variance
        pairing for multi-switch groups; a warning is emitted.
    by_path : int, optional, default=None
        If set to a positive integer ``k``, disaggregate the per-horizon
        event study by the observed treatment trajectory in the window
        ``[F_g - 1, F_g, ..., F_g - 1 + L_max]``, reporting ATT + SE +
        inference for the ``k`` most common observed paths (ties broken
        lexicographically on the path tuple). If ``k`` exceeds the number
        of observed paths, all paths are returned and a ``UserWarning``
        is emitted. ``None`` (the default) disables the disaggregation.

        Requires ``drop_larger_lower=False`` (multi-switch groups are
        the object of interest) and ``L_max >= 1`` (the path window
        depends on ``L_max``). Compatible with non-binary integer-coded
        treatment (D in Z); path tuples become integer-state tuples
        like ``(0, 2, 2, 2)``. D values must be integer-valued
        (``D == round(D)``); a ``ValueError`` is raised at fit-time on
        continuous D. Compatible with ``survey_design`` for analytical
        Binder TSL SE and replicate-weight bootstrap; per-path SE
        routes through the cell-period allocator, with non-path
        switcher-side contributions skipped (control contributions
        remain unchanged, matching the joiners/leavers IF convention).
        ``n_bootstrap > 0`` (multiplier bootstrap) under
        ``survey_design`` is not yet supported and raises
        ``NotImplementedError``. Top-k path ranking under
        ``survey_design`` remains group-cardinality-based (unweighted),
        not population-weight-based — survey weights do not affect
        which paths are selected as "top-k".

        Compatible with ``heterogeneity="<col>"`` — per-path
        heterogeneity coefficient is computed by re-running the
        Lemma 7 regression on each path-restricted switcher
        subsample. Cohort dummies absorb baseline (no R-divergence
        warning needed). Surfaces on
        ``results.path_heterogeneity_effects`` keyed
        ``{path: {l: {beta, se, t_stat, p_value, conf_int, n_obs}}}``
        and on ``to_dataframe(level="by_path")`` via ``het_*``
        columns. Mirrors R ``did_multiplegt_dyn(..., by_path,
        predict_het)`` per-by_level. Composes with ``survey_design``
        (analytical Binder TSL + replicate-weight) via the existing
        cell-period IF allocator path. Incompatible with
        ``design2`` and ``honest_did`` (each combination raises
        ``NotImplementedError`` in the current release).

        Mutually exclusive with ``paths_of_interest`` — use
        ``by_path=k`` for top-k automatic ranking by frequency, or
        ``paths_of_interest=[(...), ...]`` for an explicit user-
        specified path list. Setting both raises ``ValueError``.

        Compatible with ``controls`` (DID^X residualization) -- the
        per-baseline OLS residualization runs once on first-differenced
        ``Y`` BEFORE path enumeration, so per-path point estimates,
        bootstrap SE, per-path placebos, and per-path sup-t bands all
        consume the residualized ``Y_mat`` automatically (Frisch-
        Waugh-Lovell). Per-period effects remain unadjusted, consistent
        with the existing ``controls`` + per-period DID contract.

        **Deviation from R on multi-baseline switcher panels:** R
        ``did_multiplegt_dyn(..., by_path, controls)`` re-runs the
        per-baseline residualization on each path's restricted
        subsample (path's switchers + same-baseline not-yet-treated
        controls), so its residualization coefficients vary per path
        when switchers have different baseline values. Our global-
        residualization architecture coincides with R on single-
        baseline panels (every switcher shares the same ``D_{g,1}``)
        and per-path point estimates match exactly on the one-
        observation-per-``(g, t)`` regime; on multi-observation-per-
        cell panels the existing DID^X cell-weighting deviation from
        R applies (see ``docs/methodology/REGISTRY.md`` "Note (Phase
        3 DID^X covariate adjustment)"; independent of the by_path
        lift). On multi-baseline switcher panels, point estimates can
        diverge — a ``UserWarning`` is emitted at fit-time when this
        configuration is detected. SE inherits the cross-path cohort-
        sharing deviation from R documented for ``path_effects``.

        Compatible with ``trends_linear`` (DID^{fd} group-specific
        linear trends) -- first-differencing replaces ``Y`` with
        ``Z = Y_t - Y_{t-1}`` once globally before path enumeration,
        so per-path raw second-differences DID^{fd}_{path, l} surface
        on ``path_effects[path]["horizons"][l]`` automatically. Per-path
        cumulated level effects ``delta_{path, l} = sum_{l'=1..l}
        DID^{fd}_{path, l'}`` are surfaced on the new
        ``results.path_cumulated_event_study[path][l]`` field
        (mirroring the global ``linear_trends_effects`` cumulation;
        inner dict keyed by horizon directly, no ``"horizons"`` wrapper).
        SE on the cumulated layer is the conservative upper bound
        (sum of per-horizon component SEs, NaN-consistent), matching
        the global ``linear_trends_effects`` SE convention. Path
        enumeration runs on the post-first-differenced ``N_mat_fd``:
        switchers with ``F_g==2`` fail the window-eligibility check
        and are dropped from path enumeration entirely, so a path
        whose switchers all have ``F_g < 3`` is silently absent from
        ``path_effects`` (the existing global ``F_g < 3`` warning
        still fires). Per-path R parity matches R
        ``did_multiplegt_dyn(..., by_path, trends_lin)`` on per-path
        cumulated point estimates under single-baseline panels with
        sufficient pre-window depth (``F_g >= 4`` for every selected-
        path switcher). R re-runs the per-path full pipeline on each
        path's restricted subsample; same multi-baseline divergence
        pattern as ``controls`` (a ``UserWarning`` fires when switcher
        baselines take multiple values). **F_g=3 boundary-case
        divergence:** `F_g=3` switchers have only 1 valid pre-window
        Z value after first-differencing and the ``time==1`` filter,
        which causes Python's global-then-disaggregate architecture
        to diverge from R's per-path full-pipeline call (30%+ on
        point estimates observed empirically). A separate
        ``UserWarning`` fires at fit-time when the panel includes any
        `F_g=3` switchers and `by_path + trends_linear` is set, so
        practitioners hitting this boundary regime see the divergence
        flag explicitly. **Placebo under trends_linear returns RAW
        per-horizon values, not cumulated** -- there is no per-path
        placebo cumulation surface (verified empirically against R
        via the existing ``joiners_only_trends_lin`` parity scenario).

        Compatible with ``trends_nonparam`` (state-set trends) -- the
        set membership column is validated and stored once globally
        (time-invariance, NaN rejection, partition coarseness checks
        unchanged); per-path analytical SE, bootstrap SE, per-path
        placebos, and per-path sup-t bands all inherit the
        set-restricted control pool automatically through the
        ``set_ids`` parameter threaded through the per-path IF
        helpers. Per-path R parity matches R
        ``did_multiplegt_dyn(..., by_path, trends_nonparam)`` on
        per-path point estimates under single-baseline panels.

        Compatible with ``n_bootstrap > 0`` -- the top-k paths are
        enumerated once on the observed data (paths held fixed across
        bootstrap draws, matching R ``did_multiplegt_dyn(..., by_path,
        bootstrap=B)``) and bootstrap SE / percentile CI / percentile
        p-value are written to ``path_effects[path]["horizons"][l]``
        in place of the analytical fields. See REGISTRY.md for the
        full bootstrap contract.

        Compatible with ``placebo=True`` -- when both are active,
        per-path backward-horizon placebos ``DID^{pl}_{path, l}`` for
        ``l = 1..L_max`` are surfaced on
        ``results.path_placebo_event_study[path][-l]`` (negative-int
        keys mirroring ``placebo_event_study``). The same per-path SE
        convention is applied backward (joiners/leavers IF precedent;
        cohort-recentered plug-in with path-specific divisor); the
        cross-path cohort-sharing deviation from R is inherited from
        the analytical event-study path.

        With ``n_bootstrap > 0``, per-path joint sup-t simultaneous
        confidence bands are also computed across horizons
        ``1..L_max`` within each path. A path-specific critical value
        ``c_p`` (constructed from a fresh shared-weights multiplier-
        bootstrap draw per path) is surfaced at top level as
        ``results.path_sup_t_bands[path] = {"crit_value", "alpha",
        "n_bootstrap", "method", "n_valid_horizons"}``, applied
        per-horizon as ``cband_conf_int`` on
        ``path_effects[path]["horizons"][l]``, and rendered as
        ``cband_lower`` / ``cband_upper`` columns on
        ``results.to_dataframe(level="by_path")`` (mirroring the
        OVERALL ``level="event_study"`` schema). Bands cover joint
        inference WITHIN a single path across horizons; they do NOT
        provide simultaneous coverage across paths. Python-only
        library extension; R ``did_multiplegt_dyn`` provides no joint
        bands at any surface. See REGISTRY.md ``Note (Phase 3 by_path
        per-path joint sup-t bands)``.

        SE convention: per-path IF parallels the joiners / leavers
        construction — the switcher-side contribution is zeroed for
        groups not in the selected path, and the cohort structure and
        control pool are unchanged. Plug-in SE uses the path-specific
        divisor ``N_l_path`` (count of path switchers eligible at horizon
        ``l``), matching how ``joiners_se`` / ``leavers_se`` use their
        respective counts as divisors. See REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` ``Note`` on ``by_path`` for the
        full contract.

        Results are exposed on ``results.path_effects`` as a dict keyed
        by the path tuple, with nested ``"horizons"`` dicts per
        horizon ``l``. Also available via
        ``results.to_dataframe(level="by_path")``.
    paths_of_interest : list of tuple of int, optional, default=None
        Explicit user-specified treatment paths to disaggregate by, as
        an alternative to ``by_path=k``'s top-k automatic ranking.
        Each path tuple must have length ``L_max + 1`` and represents
        the treatment trajectory in the window
        ``[F_g - 1, F_g, ..., F_g - 1 + L_max]``, e.g.
        ``[(0, 1, 1, 1), (0, 1, 0, 0)]`` for two paths under
        ``L_max=3``. Mutually exclusive with ``by_path``; setting both
        raises ``ValueError``.

        Validation:

        - Each path element must be an ``int`` (``bool`` and
          ``np.bool_`` rejected; ``np.integer`` accepted and
          canonicalized to Python ``int``).
        - All paths must have the same length (uniformity validated
          at ``__init__``; length match against ``L_max + 1``
          validated at fit-time).
        - Empty list raises ``ValueError``.
        - Duplicate paths are deduplicated with a ``UserWarning``.
        - A path with zero observed groups in the panel emits a
          ``UserWarning`` and is omitted from ``path_effects``.

        Compatible with non-binary integer treatment (paths can
        contain integer states like ``(0, 2, 2)``).

        Compatible with all downstream surfaces inherited by
        ``by_path``: bootstrap, per-path placebos, per-path joint
        sup-t bands, ``controls``, ``trends_linear``,
        ``trends_nonparam``, ``survey_design`` (analytical Binder
        TSL + replicate-weight; multiplier bootstrap under survey
        remains gated, same as ``by_path=k``), and ``heterogeneity``
        (per-path heterogeneity coefficient surfaces on
        ``results.path_heterogeneity_effects``). Mechanical
        extension to path enumeration; no methodology change.

        **Order semantics**: paths appear in
        ``results.path_effects`` in the user-specified order, modulo
        deduplication and unobserved-path filtering.

        **Python-only API extension; no R equivalent.** R's
        ``did_multiplegt_dyn(..., by_path=k)`` only accepts a positive
        int (top-k) or ``-1`` (all paths); there is no list-based
        path selection in R.

        Results expose the same surfaces as ``by_path``:
        ``results.path_effects`` (dict keyed by path tuple),
        ``results.path_placebo_event_study``,
        ``results.path_sup_t_bands``,
        ``results.path_cumulated_event_study`` (under
        ``trends_linear``), and the ``level="by_path"`` DataFrame.
    rank_deficient_action : str, default="warn"
        Action when the TWFE decomposition diagnostic OLS encounters a
        rank-deficient design matrix: ``"warn"``, ``"error"``, or
        ``"silent"``. Only used when ``twfe_diagnostic=True``.

    Attributes
    ----------
    results_ : ChaisemartinDHaultfoeuilleResults
        Estimation results after calling :meth:`fit`.
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    The analytical CI is **conservative** under Assumption 8 (independent
    groups) of the dynamic companion paper, and exact only under iid
    sampling. This is documented as a deliberate deviation from "default
    nominal coverage" in ``REGISTRY.md``.

    Examples
    --------
    Basic single-switch panel:

    >>> from diff_diff import ChaisemartinDHaultfoeuille
    >>> from diff_diff.prep_dgp import generate_reversible_did_data
    >>> data = generate_reversible_did_data(n_groups=80, n_periods=6, seed=42)
    >>> est = ChaisemartinDHaultfoeuille()
    >>> results = est.fit(
    ...     data, outcome="outcome", unit="group",
    ...     time="period", treatment="treatment",
    ... )
    >>> abs(results.overall_att - 2.0) < 1.0  # close to the true effect
    True
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        placebo: bool = True,
        twfe_diagnostic: bool = True,
        drop_larger_lower: bool = True,
        by_path: Optional[int] = None,
        paths_of_interest: Optional[Sequence[Sequence[int]]] = None,
        rank_deficient_action: str = "warn",
    ) -> None:
        # Parameter validation
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be non-negative, got {n_bootstrap}")
        if by_path is not None:
            if isinstance(by_path, bool) or not isinstance(by_path, int):
                raise ValueError(
                    f"by_path must be None or a positive int, got "
                    f"{by_path!r} of type {type(by_path).__name__}."
                )
            if by_path <= 0:
                raise ValueError(
                    f"by_path must be a positive int (top-k most common paths), "
                    f"got {by_path}. Use by_path=None to disable, or "
                    f"paths_of_interest for explicit path selection."
                )
        if paths_of_interest is not None:
            paths_of_interest = _validate_paths_of_interest(paths_of_interest)
        if by_path is not None and paths_of_interest is not None:
            raise ValueError(
                "by_path and paths_of_interest are mutually exclusive. "
                "Use by_path=k for top-k automatic ranking, OR "
                "paths_of_interest=[(...), ...] for explicit user-"
                "specified paths. Set one and leave the other as None."
            )
        if cluster is not None:
            raise NotImplementedError(
                f"cluster={cluster!r}: user-specified clustering is not "
                f"supported in ChaisemartinDHaultfoeuille. dCDH clusters at "
                f"the group level by default via the cohort-recentered "
                f"influence-function plug-in (analytical SEs) and the "
                f"multiplier bootstrap. Under survey_design with strictly-"
                f"coarser PSUs, bootstrap clustering automatically upgrades "
                f"to PSU-level Hall-Mammen wild. To use the default path, "
                f"pass cluster=None (the "
                f"default). Custom clustering is reserved for a future "
                f"phase. See REGISTRY.md ChaisemartinDHaultfoeuille section "
                f"for the full contract."
            )

        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.placebo = placebo
        self.twfe_diagnostic = twfe_diagnostic
        self.drop_larger_lower = drop_larger_lower
        self.by_path = by_path
        self.paths_of_interest = paths_of_interest
        self.rank_deficient_action = rank_deficient_action

        self.is_fitted_ = False
        self.results_: Optional[ChaisemartinDHaultfoeuilleResults] = None

    # ------------------------------------------------------------------
    # sklearn-style parameter introspection
    # ------------------------------------------------------------------

    # get_params/set_params come from BaseEstimator.

    def _validate_invariants(self) -> None:
        """Run the post-mutation validation rules. Mirrors `__init__`."""
        # Re-run __init__ validation rules so the post-set state is valid.
        if self.rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{self.rank_deficient_action}'"
            )
        if self.bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{self.bootstrap_weights}'"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be non-negative, got {self.n_bootstrap}")
        if self.by_path is not None:
            if isinstance(self.by_path, bool) or not isinstance(self.by_path, int):
                raise ValueError(
                    f"by_path must be None or a positive int, got "
                    f"{self.by_path!r} of type {type(self.by_path).__name__}."
                )
            if self.by_path <= 0:
                raise ValueError(
                    f"by_path must be a positive int (top-k most common paths), "
                    f"got {self.by_path}. Use by_path=None to disable, or "
                    f"paths_of_interest for explicit path selection."
                )
        if self.paths_of_interest is not None:
            self.paths_of_interest = _validate_paths_of_interest(self.paths_of_interest)
        if self.by_path is not None and self.paths_of_interest is not None:
            raise ValueError(
                "by_path and paths_of_interest are mutually exclusive. "
                "Use by_path=k for top-k automatic ranking, OR "
                "paths_of_interest=[(...), ...] for explicit user-"
                "specified paths. Set one and leave the other as None."
            )
        if self.cluster is not None:
            raise NotImplementedError(
                f"cluster={self.cluster!r}: user-specified clustering is "
                f"not supported in ChaisemartinDHaultfoeuille. dCDH clusters "
                f"at the group level by default; under survey_design with "
                f"strictly-coarser PSUs the bootstrap automatically upgrades "
                f"to PSU-level Hall-Mammen wild clustering. Pass cluster=None "
                f"(the default) to use this path. User-specified custom "
                f"clustering is reserved for a future phase. See REGISTRY.md "
                f"ChaisemartinDHaultfoeuille section for the full contract."
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: Any = NOT_SUPPLIED,
        time: Any = NOT_SUPPLIED,
        treatment: Any = NOT_SUPPLIED,
        # ---------- forward-compat parameters ----------
        aggregate: Optional[str] = None,
        L_max: Optional[int] = None,
        covariates: Any = NOT_SUPPLIED,
        trends_linear: Optional[bool] = None,
        trends_nonparam: Optional[Any] = None,
        honest_did: bool = False,
        # ---------- Phase 3 extensions ----------
        heterogeneity: Optional[str] = None,
        design2: bool = False,
        # ---------- deferred (separate effort) ----------
        survey_design: Any = None,
        # ---------- deprecated aliases (removed in 4.0) ----------
        group: Any = NOT_SUPPLIED,
        controls: Any = NOT_SUPPLIED,
    ) -> ChaisemartinDHaultfoeuilleResults:
        """
        Fit the dCDH estimator on individual-level panel data.

        Parameters
        ----------
        data : pd.DataFrame
            Individual-level panel. Must contain columns for ``outcome``,
            ``unit``, ``time``, and ``treatment``. The estimator
            internally aggregates to ``(unit, time)`` cells.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name (the R ``DIDmultiplegt`` "group").
            Treatment must be constant within each ``(unit, time)`` cell
            after aggregation; ``ValueError`` is raised if any cell has
            fractional treatment after grouping (within-cell-varying
            treatment indicates a fuzzy design not supported in Phase 1).
        time : str
            Time period column name. Must be sortable.
        treatment : str
            Per-observation treatment column. Must be numeric and constant
            within each ``(unit, time)`` cell. Both binary ``{0, 1}`` and
            non-binary (ordinal or continuous) treatment are supported.
            Non-binary treatment requires ``L_max >= 1``.
        aggregate : str, optional
            **Reserved for Phase 3.** Must be ``None``; any other value
            raises ``NotImplementedError``.
        L_max : int, optional
            Maximum event-study horizon. When set, computes ``DID_l``
            for ``l = 1, ..., L_max`` using the per-group building block
            from Equation 3 of the dynamic companion paper. When
            ``None`` (default), only the ``l = 1`` contemporaneous-
            switch estimator ``DID_M`` is computed (Phase 1 behavior).
            Must be a positive integer not exceeding the number of
            post-baseline periods in the panel.
        covariates : list of str, optional
            Column names for covariate adjustment via residualization-style
            ``DID^X`` (Web Appendix Section 1.2). Requires ``L_max >= 1``.
            One ``theta_hat`` per baseline treatment value, estimated by
            OLS on not-yet-treated observations. NOT doubly-robust.
        trends_linear : bool, optional
            If ``True``, estimate group-specific linear trends via
            ``DID^{fd}`` (Web Appendix Section 1.3, Lemma 6). Requires
            ``L_max >= 1`` and at least 3 time periods.
        trends_nonparam : str, optional
            Column name for state-set membership. Restricts the control
            pool to groups in the same set (Web Appendix Section 1.4).
            Requires ``L_max >= 1`` and time-invariant values per group.
        honest_did : bool, default=False
            Run HonestDiD sensitivity analysis (Rambachan & Roth 2023) on
            the placebo + event study surface. Requires ``L_max >= 1``.
            Default: relative magnitudes (DeltaRM, Mbar=1.0), targeting
            the equal-weight average over all post-treatment horizons
            (``l_vec=None``). Results stored on
            ``results.honest_did_results``; ``None`` with a warning if
            the solver fails. For custom parameters (e.g., targeting
            the on-impact effect only via ``l_vec``), call
            ``compute_honest_did(results, ...)`` post-hoc instead.
        heterogeneity : str, optional
            Column name for a time-invariant covariate to test for
            heterogeneous effects (Web Appendix Section 1.5, Lemma 7).
            Per-horizon OLS regressions are computed for forward
            horizons (1..L_max), and ALSO for backward (placebo)
            horizons (-1..-L_max) when ``placebo=True`` is set
            (post-2026-05-15: per-path placebo predict_het R-parity
            against ``did_multiplegt_dyn(by_path, predict_het, placebo)``).
            Joint Wald F-test across rows is NOT computed
            (per-horizon inference only). Cannot be combined with
            ``covariates``, ``trends_linear``, or ``trends_nonparam``.
            Requires ``L_max >= 1``. Under ``by_path`` /
            ``paths_of_interest``, per-path heterogeneity coefficients
            also surface on ``results.path_heterogeneity_effects`` and
            on ``to_dataframe(level="by_path")`` via ``het_*`` columns
            (positive AND negative-horizon rows populated when
            ``placebo=True``). Under ``survey_design``, backward-
            horizon (placebo) heterogeneity is NOT computed (the pre-
            period Binder TSL cell allocator is deferred to a follow-
            up methodology PR); a ``UserWarning`` fires at fit-time
            and forward-horizon heterogeneity continues to compute
            normally.
        design2 : bool, default=False
            If ``True``, identify and report switch-in/switch-out
            (Design-2) groups. Convenience wrapper (descriptive summary,
            not full paper re-estimation). Requires
            ``drop_larger_lower=False`` to retain 2-switch groups.
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference.
            Supports ``weight_type='pweight'`` with two variance paths:
            (1) Taylor Series Linearization using strata / PSU / FPC
            (analytical) via the **cell-period IF allocator** that
            attributes per-``(g, t)``-cell mass and aggregates through
            Binder (1983), and (2) replicate-weight variance using
            BRR / Fay / JK1 / JKn / SDR methods (analytical, closed-
            form). Survey weights produce weighted cell means for the
            point estimate. Under a survey design without an explicit
            ``psu``, ``fit()`` auto-injects ``psu=<group_col>`` as a
            safe default (the group is the effective sampling unit).
            **Strata and PSU may vary across cells of a group** but
            must be constant within each ``(g, t)`` cell (trivially
            true in one-obs-per-cell panels; enforced otherwise with
            ``ValueError``). Three supported combinations under the
            auto-injected ``psu=<group_col>``:
            (1) strata constant within group (any ``nest`` flag works);
            (2) strata vary within group **and** ``nest=True`` — the
            resolver re-labels the synthesized ``psu`` uniquely within
            strata; (3) strata vary within group **and** ``nest=False``
            — rejected up front with a targeted ``ValueError``; pass
            ``SurveyDesign(..., nest=True)`` or an explicit
            ``psu=<col>`` with globally-unique labels instead. When ``n_bootstrap > 0`` and a survey
            design is supplied, the multiplier bootstrap operates at
            the PSU level (Hall-Mammen wild PSU bootstrap) — under the
            default auto-inject this collapses to a group-level
            clustered bootstrap. Under within-group-varying PSU the
            bootstrap uses a cell-level wild PSU allocator — a group
            contributing cells to multiple PSUs receives independent
            multiplier draws per PSU (see the Survey + bootstrap
            contract Note in REGISTRY.md). **Scope note (terminal
            missingness under any cell-period-allocator path):** on
            panels where a terminally-missing group is in a cohort
            whose other groups still contribute at the missing
            period, every survey variance path that uses the cell-
            period allocator raises a targeted ``ValueError``:
            Binder TSL with within-group-varying PSU, Rao-Wu
            replicate-weight ATT (which always uses the cell
            allocator), and the cell-level wild PSU bootstrap.
            Cohort-recentering leaks centered IF mass onto cells
            with no positive-weight obs, which the cell-period
            allocator cannot allocate to any observation or PSU.
            Pre-process the panel (drop late-exit groups or trim to
            a balanced sub-panel), or — for Binder TSL only — use
            an explicit ``psu=<group_col>`` so the analytical path
            routes through the legacy group-level allocator.
            Replicate ATT and within-group-varying-PSU bootstrap
            have no such allocator fallback. **Replicate weights with ``n_bootstrap > 0``
            raises ``NotImplementedError``** (replicate variance is
            closed-form; bootstrap would double-count variance). See
            REGISTRY.md ``ChaisemartinDHaultfoeuille`` Notes for the
            full contract.
        group : str, optional
            Deprecated alias for ``unit`` (row M-033); warns with
            ``FutureWarning`` and will be removed in 4.0.
        controls : list of str, optional
            Deprecated alias for ``covariates`` (row M-034); warns with
            ``FutureWarning`` and will be removed in 4.0.

        Returns
        -------
        ChaisemartinDHaultfoeuilleResults

        Raises
        ------
        ValueError
            If required columns are missing, treatment is not binary, or
            the panel has too few groups / periods.
        NotImplementedError
            If any forward-compat parameter is set to a non-default
            value, with a clear pointer to the relevant ROADMAP phase.
        """
        unit = resolve_renamed_kwarg(
            "ChaisemartinDHaultfoeuille.fit",
            "group",
            group,
            "unit",
            unit,
            default=NOT_SUPPLIED,
        )
        require_arg("ChaisemartinDHaultfoeuille.fit", "unit", unit)
        require_arg("ChaisemartinDHaultfoeuille.fit", "time", time)
        require_arg("ChaisemartinDHaultfoeuille.fit", "treatment", treatment)
        covariates = resolve_renamed_kwarg(
            "ChaisemartinDHaultfoeuille.fit",
            "controls",
            controls,
            "covariates",
            covariates,
            default=None,
        )
        # Body-local names; the public parameters are unit/covariates
        # (M-033 / M-034).
        group = unit
        controls = covariates
        # ------------------------------------------------------------------
        # Step 1: Column validation
        # ------------------------------------------------------------------
        required_cols = [outcome, group, time, treatment]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # ------------------------------------------------------------------
        # Step 2: Forward-compat gates
        # ------------------------------------------------------------------
        _check_forward_compat_gates(
            aggregate=aggregate,
            L_max=L_max,
            controls=controls,
            trends_linear=trends_linear,
            trends_nonparam=trends_nonparam,
            honest_did=honest_did,
        )

        # ------------------------------------------------------------------
        # Step 3: Survey resolution
        # ------------------------------------------------------------------
        from diff_diff.survey import _resolve_survey_for_fit

        resolved_survey, survey_weights, _, survey_metadata = _resolve_survey_for_fit(
            survey_design, data, "analytical"
        )

        # dCDH contract: the group is the effective sampling unit for
        # the TSL IF expansion psi_i = U[g] * (w_i / W_g). When the user
        # passed a SurveyDesign without an explicit PSU,
        # compute_survey_if_variance() would fall back to per-observation
        # PSUs — which contradicts the per-group structure the IF
        # expansion assumes and inflates df_survey. Auto-inject
        # `psu=<group>` and re-resolve so downstream variance, df_survey,
        # and HonestDiD critical values match the documented contract.
        # Strata / FPC / weight_type / nest are preserved.
        # Skipped for replicate-weight designs — they're rejected below.
        if (
            resolved_survey is not None
            and resolved_survey.psu is None
            and (
                resolved_survey.replicate_weights is None
                or resolved_survey.replicate_weights.shape[1] == 0
            )
        ):
            # Pre-auto-inject contract check: the auto-inject path
            # synthesizes ``psu=<group>`` and preserves the user's
            # ``nest`` flag. Under ``nest=False`` (the default), the
            # survey resolver requires globally-unique PSU labels when
            # strata are present; if strata varies within group, the
            # synthesized PSU column reuses group labels across strata
            # and trips the cross-stratum PSU uniqueness check at
            # resolution time. Under ``nest=True`` the resolver
            # re-labels ``(stratum, psu)`` uniquely within strata
            # (``diff_diff/survey.py:299-302``), so varying strata is
            # fine — let the auto-inject proceed. Only the
            # ``nest=False`` + varying-strata + omitted-psu triple
            # warrants an up-front targeted error.
            if resolved_survey.strata is not None and not getattr(survey_design, "nest", False):
                _strata_varies_pre, _ = _strata_psu_vary_within_group(
                    resolved_survey,
                    data,
                    group,
                    survey_weights,
                )
                if _strata_varies_pre:
                    raise ValueError(
                        "ChaisemartinDHaultfoeuille survey support: "
                        "strata that vary across cells of the same "
                        "group require either an explicit "
                        "`psu=<col>` (any column whose labels are "
                        "globally unique within strata) or the "
                        "original `SurveyDesign(..., nest=True)` "
                        "flag so the auto-injected `psu=<group>` is "
                        "re-labeled uniquely within strata by the "
                        "resolver. The default `nest=False` auto-"
                        "inject path reuses group labels across "
                        "strata and trips the cross-stratum PSU "
                        "uniqueness check in survey resolution. "
                        "Either (a) set strata constant within each "
                        "group, (b) pass `SurveyDesign(..., "
                        "nest=True)`, or (c) pass an explicit "
                        "`psu=<col>` with globally-unique labels."
                    )

            from diff_diff.survey import SurveyDesign as _SurveyDesign

            # Build a synthesized PSU column on a private copy of data
            # so the caller's DataFrame is untouched. Valid group values
            # flow through as their own PSU label; NaN/invalid group
            # values on zero-weight rows (SurveyDesign.subpopulation()
            # excluded rows) are replaced with a single shared dummy
            # label so the PSU resolver accepts them. Zero-weight rows
            # contribute psi_i = 0 to the variance; keeping them in the
            # resolved design preserves the full-design df_survey
            # contract (n_psu / n_strata reflect the full sample, not
            # the positive-weight subset).
            psu_col_name = "__dcdh_eff_psu__"
            synth_data = data.copy()
            synth_psu = synth_data[group].copy()
            try:
                invalid_mask = synth_psu.isna().to_numpy()
            except (AttributeError, TypeError):
                invalid_mask = np.zeros(len(synth_psu), dtype=bool)
            if invalid_mask.any():
                synth_psu = synth_psu.astype(object)
                synth_psu.loc[invalid_mask] = "__dcdh_excluded_null_psu__"
            synth_data[psu_col_name] = synth_psu

            eff_design = _SurveyDesign(
                weights=survey_design.weights,
                strata=survey_design.strata,
                psu=psu_col_name,
                fpc=getattr(survey_design, "fpc", None),
                weight_type=getattr(survey_design, "weight_type", "pweight"),
                nest=getattr(survey_design, "nest", False),
                lonely_psu=getattr(survey_design, "lonely_psu", "remove"),
            )
            resolved_survey, survey_weights, _, survey_metadata = _resolve_survey_for_fit(
                eff_design, synth_data, "analytical"
            )

        if resolved_survey is not None:
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"ChaisemartinDHaultfoeuille survey support requires "
                    f"weight_type='pweight', got '{resolved_survey.weight_type}'. "
                    f"The survey IF variance math assumes probability weights."
                )
            # Replicate-weight designs (BRR/Fay/JK1/JKn/SDR) are supported
            # for analytical variance via compute_replicate_if_variance()
            # at each IF site (see _survey_se_from_group_if). The combination
            # of replicate weights and n_bootstrap > 0 is rejected inside
            # the bootstrap entry block (replicate variance is closed-form;
            # bootstrap would double-count variance). Matches library
            # precedent: efficient_did.py:989, staggered.py:1869,
            # two_stage.py:251-253.

            # Cell-period IF allocator contract: strata and PSU must be
            # constant within each (g, t) cell, a strict relaxation of
            # the previous within-group constancy rule. Both the
            # analytical TSL path and the PSU-level wild bootstrap now
            # honor within-group-varying PSU via the cell-period
            # allocator (the bootstrap dispatcher routes PSU-within-
            # group-constant regimes through the legacy group-level
            # path for bit-identity with prior releases).
            _validate_cell_constant_strata_psu(
                resolved_survey,
                data,
                group,
                time,
                survey_weights,
            )

        # Design-2 precondition: requires drop_larger_lower=False
        if design2 and self.drop_larger_lower:
            raise ValueError(
                "design2=True requires drop_larger_lower=False because "
                "Design-2 groups have exactly 2 treatment changes (join "
                "then leave), which are dropped by the default "
                "drop_larger_lower=True filter. Construct the estimator "
                "with ChaisemartinDHaultfoeuille(drop_larger_lower=False)."
            )

        # ------------------------------------------------------------------
        # by_path / paths_of_interest preconditions and Phase 3
        # compatibility gates
        # ------------------------------------------------------------------
        if self.by_path is not None or self.paths_of_interest is not None:
            if self.drop_larger_lower:
                raise ValueError(
                    "by_path / paths_of_interest requires "
                    "drop_larger_lower=False because multi-switch groups "
                    "are the object of interest for per-path "
                    "disaggregation, but the default "
                    "drop_larger_lower=True filter removes them. "
                    "Construct the estimator with "
                    "ChaisemartinDHaultfoeuille(drop_larger_lower=False, "
                    "by_path=k) or "
                    "ChaisemartinDHaultfoeuille("
                    "drop_larger_lower=False, "
                    "paths_of_interest=[(...), ...])."
                )
            if L_max is None:
                raise ValueError(
                    "by_path / paths_of_interest requires L_max >= 1. "
                    "The path window spans [F_g - 1, F_g - 1 + L_max] "
                    "and therefore depends on the event-study horizon. "
                    "Set L_max when calling fit()."
                )
            if self.paths_of_interest is not None:
                expected_len = L_max + 1
                for p in self.paths_of_interest:
                    if len(p) != expected_len:
                        raise ValueError(
                            f"paths_of_interest entries must have "
                            f"length L_max+1={expected_len} (window "
                            f"[F_g-1, ..., F_g-1+L_max]); got path "
                            f"{p!r} of length {len(p)}."
                        )
            if design2:
                raise NotImplementedError(
                    "by_path / paths_of_interest combined with design2 "
                    "is deferred to a future release."
                )
            if honest_did:
                raise NotImplementedError(
                    "by_path / paths_of_interest combined with honest_did "
                    "(HonestDiD sensitivity analysis) is deferred to a "
                    "future release."
                )
            if survey_design is not None and self.n_bootstrap > 0:
                raise NotImplementedError(
                    "by_path / paths_of_interest combined with both "
                    "survey_design and n_bootstrap>0 (multiplier "
                    "bootstrap) is not yet supported (the survey-aware "
                    "perturbation pivot for path-restricted IFs has "
                    "not been derived). Use n_bootstrap=0 for "
                    "analytical Binder TSL SE under survey_design, or "
                    "use replicate weights "
                    "(SurveyDesign(..., replicate_weights=...)) for "
                    "design-based bootstrap variance."
                )

        # ------------------------------------------------------------------
        # Step 4-5: Validate input + aggregate to (g, t) cells via the
        # shared helper used by both fit() and twowayfeweights(). The
        # helper enforces NaN/binary/within-cell-rounding rules from
        # REGISTRY.md and returns a sorted cell DataFrame with columns
        # [group, time, y_gt, d_gt, n_gt].
        # ------------------------------------------------------------------
        cell = _validate_and_aggregate_to_cells(
            data=data,
            outcome=outcome,
            group=group,
            time=time,
            treatment=treatment,
            weights=survey_weights,
        )

        # Retain observation-level survey info for IF expansion (Step 3
        # of survey integration: group-level IF → observation-level psi).
        # `time_ids` is per-row; `periods` (the sorted column index used
        # by the pivoted U_per_period matrices) is populated below once
        # it's known. Together they enable cell-level IF expansion
        # psi_i = U[g_i, t_i] * (w_i / W_{g_i, t_i}) under the cell-
        # period allocator (REGISTRY.md survey IF expansion contract).
        _obs_survey_info = None
        if resolved_survey is not None:
            _obs_survey_info = {
                "group_ids": data[group].values,
                "time_ids": data[time].values,
                "weights": survey_weights,
                "resolved": resolved_survey,
                "periods": None,
            }

        # Replicate-weight variance tracker: each _compute_se call under
        # a replicate design returns n_valid (number of replicate columns
        # that produced a finite estimate). The effective df_survey is
        # min(n_valid) - 1 across all IF sites — matches the precedent in
        # `diff_diff/efficient_did.py:1133-1135` and
        # `diff_diff/triple_diff.py:676-686`. Under TSL (analytical),
        # _compute_se returns None for n_valid and df_survey falls through
        # to resolved_survey.df_survey (= n_psu - n_strata).
        _replicate_n_valid_list: List[int] = []

        # ------------------------------------------------------------------
        # Step 4b: Covariate aggregation (DID^X, Web Appendix Section 1.2)
        # ------------------------------------------------------------------
        if controls is not None:
            if not controls:
                raise ValueError(
                    "covariates must be a non-empty list of column names, "
                    "got an empty list. Pass covariates=None to disable "
                    "covariate adjustment."
                )
            if L_max is None:
                raise ValueError(
                    "Covariate adjustment (DID^X) requires L_max >= 1. The "
                    "per-period DID path does not support covariate "
                    "residualization. Set L_max to use the per-group "
                    "DID_{g,l} path with covariate adjustment."
                )
            missing_controls = [c for c in controls if c not in data.columns]
            if missing_controls:
                raise ValueError(
                    f"Covariate column(s) {missing_controls!r} not found in "
                    f"data. Available columns: {list(data.columns)}"
                )
            # SurveyDesign.subpopulation() contract: zero-weight rows are
            # out-of-sample. Scope BOTH validation and aggregation to the
            # positive-weight subset so excluded rows with missing/invalid
            # covariates do not abort the fit and cell aggregation aligns
            # with the effective sample used by _validate_and_aggregate_to_cells.
            if survey_weights is not None:
                pos_mask_ctrl = np.asarray(survey_weights) > 0
                data_eff = data.loc[pos_mask_ctrl]
                survey_weights_eff = np.asarray(survey_weights)[pos_mask_ctrl]
            else:
                data_eff = data
                survey_weights_eff = None
            data_controls = data_eff[controls].copy()
            for c in controls:
                try:
                    data_controls[c] = pd.to_numeric(data_controls[c])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Could not coerce control column {c!r} to numeric: {exc}"
                    ) from exc
                n_nan = int(data_controls[c].isna().sum())
                if n_nan > 0:
                    raise ValueError(
                        f"Control column {c!r} contains {n_nan} NaN value(s). "
                        "Drop or impute missing covariates before fitting."
                    )
                n_inf = int(np.isinf(data_controls[c].to_numpy()).sum())
                if n_inf > 0:
                    raise ValueError(
                        f"Control column {c!r} contains {n_inf} Inf value(s). "
                        "Remove or replace non-finite covariates before fitting."
                    )
            # Aggregate covariates to cell means (same groupby as treatment/outcome).
            # Build x_agg_input from the same effective-sample frame so rows
            # align with data_controls.
            x_agg_input = data_eff[[group, time]].copy()
            x_agg_input[controls] = data_controls[controls].values
            if survey_weights_eff is not None:
                # Survey-weighted covariate cell means: sum(w*x)/sum(w)
                x_agg_input["_w_"] = survey_weights_eff
                for c in controls:
                    x_agg_input[f"_wx_{c}"] = survey_weights_eff * x_agg_input[c].values
                wx_cols = [f"_wx_{c}" for c in controls]
                g_agg = x_agg_input.groupby([group, time], as_index=False).agg(
                    {**{wc: "sum" for wc in wx_cols}, "_w_": "sum"}
                )
                for c in controls:
                    w_safe = g_agg["_w_"].replace(0, 1)
                    g_agg[c] = g_agg[f"_wx_{c}"] / w_safe
                x_cell_agg = g_agg[[group, time] + controls]
            else:
                x_cell_agg = x_agg_input.groupby([group, time], as_index=False)[controls].mean()
            cell = cell.merge(x_cell_agg, on=[group, time], how="left")

        # ------------------------------------------------------------------
        # Step 5a: Compute the TWFE diagnostic on the FULL pre-filter cell
        #          dataset, so the diagnostic reflects the data the user
        #          actually passed in. This MUST run BEFORE Step 5b (the
        #          ragged-panel filter) so that the fitted diagnostic and
        #          the standalone twowayfeweights() function produce
        #          identical results on ragged panels — both operate on
        #          the same _validate_and_aggregate_to_cells() output.
        # ------------------------------------------------------------------
        twfe_diagnostic_payload = None
        # TWFE diagnostic assumes binary treatment (d_arr == 1 for
        # treated mask). Skip for non-binary data with a warning.
        is_binary_pre = set(cell["d_gt"].unique()).issubset({0.0, 1.0, 0, 1})
        if self.twfe_diagnostic and not is_binary_pre:
            warnings.warn(
                "TWFE diagnostic (twfe_diagnostic=True) is not supported for "
                "non-binary treatment. The diagnostic assumes binary {0, 1} "
                "treatment. Skipping TWFE diagnostic for this fit.",
                UserWarning,
                stacklevel=2,
            )
        elif self.twfe_diagnostic:
            try:
                twfe_diagnostic_payload = _compute_twfe_diagnostic(
                    cell=cell,
                    group_col=group,
                    time_col=time,
                    rank_deficient_action=self.rank_deficient_action,
                )
            except Exception as exc:  # noqa: BLE001
                # Honor rank_deficient_action="error": if the user
                # explicitly requested strict failure on rank-deficient
                # designs, re-raise instead of downgrading to a warning.
                # Only genuinely non-fatal failures (e.g., numerical
                # issues unrelated to rank deficiency) should be
                # swallowed as warnings.
                if self.rank_deficient_action == "error" and isinstance(exc, ValueError):
                    raise
                warnings.warn(
                    f"TWFE decomposition diagnostic failed: {exc}. "
                    "Skipping diagnostic; main estimation continues.",
                    UserWarning,
                    stacklevel=2,
                )
                twfe_diagnostic_payload = None

        # ------------------------------------------------------------------
        # Step 5b: Ragged panel validation
        #
        # The cohort/variance path treats D_{g,1} as the canonical
        # baseline and walks adjacent observed periods to detect first
        # switches. Ragged panels with missing baseline rows or interior
        # gaps would either crash the cohort enumeration (NaN -> int
        # cast) or silently misclassify cohorts. Two-tier handling:
        #
        # (a) Reject groups missing the FIRST GLOBAL period (the
        #     baseline) with a clear ValueError listing offenders.
        # (b) Drop groups with INTERIOR GAPS (missing intermediate
        #     periods between their first and last observed period)
        #     with an explicit UserWarning.
        # ------------------------------------------------------------------
        all_periods_pre_drop = sorted(cell[time].unique().tolist())
        if len(all_periods_pre_drop) < 2:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires at least 2 distinct time "
                f"periods in the panel, got {len(all_periods_pre_drop)}."
            )
        first_global_period = all_periods_pre_drop[0]

        # (a) Reject groups missing the first global period
        groups_with_baseline = set(cell.loc[cell[time] == first_global_period, group].tolist())
        all_groups_pre_validation = set(cell[group].unique().tolist())
        groups_missing_baseline = sorted(all_groups_pre_validation - groups_with_baseline)
        if groups_missing_baseline:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires every group to have an "
                f"observation at the first global period "
                f"(period={first_global_period!r}). "
                f"{len(groups_missing_baseline)} group(s) are missing this baseline. "
                f"Examples: {groups_missing_baseline[:5]}"
                + (
                    f" (and {len(groups_missing_baseline) - 5} more)"
                    if len(groups_missing_baseline) > 5
                    else ""
                )
                + ". Drop these groups or back-fill the baseline before fitting "
                "so the exclusion is explicit."
            )

        # (b) Drop groups with interior gaps
        period_index = {p: i for i, p in enumerate(all_periods_pre_drop)}
        groups_with_interior_gaps: List[Any] = []
        for g_id, sub in cell.groupby(group):
            g_periods = sub[time].tolist()
            g_min_idx = period_index[min(g_periods)]
            g_max_idx = period_index[max(g_periods)]
            expected_count = g_max_idx - g_min_idx + 1
            if len(g_periods) != expected_count:
                groups_with_interior_gaps.append(g_id)
        n_groups_dropped_interior_gap = len(groups_with_interior_gaps)
        if groups_with_interior_gaps:
            warnings.warn(
                f"Dropping {len(groups_with_interior_gaps)} group(s) with interior "
                f"period gaps (missing observations between their first and last "
                f"observed period). Examples: {groups_with_interior_gaps[:5]}"
                + (
                    f" (and {len(groups_with_interior_gaps) - 5} more)"
                    if len(groups_with_interior_gaps) > 5
                    else ""
                )
                + ". dCDH requires consecutive observed periods for the "
                "cohort/variance path; back-fill or interpolate the missing "
                "periods if you want these groups in the estimation.",
                UserWarning,
                stacklevel=2,
            )
            cell = cell[~cell[group].isin(groups_with_interior_gaps)].reset_index(drop=True)
            if cell.empty:
                raise ValueError(
                    "After dropping groups with interior period gaps, no groups "
                    "remain. Provide a balanced panel or back-fill missing periods."
                )

        all_periods_pre_drop = sorted(cell[time].unique().tolist())
        if len(all_periods_pre_drop) < 2:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires at least 2 periods, "
                f"got {len(all_periods_pre_drop)}"
            )

        # ------------------------------------------------------------------
        # Step 6: Drop A5-violating (multi-switch) cells per drop_larger_lower
        # ------------------------------------------------------------------
        n_groups_dropped_crossers = 0
        if self.drop_larger_lower:
            cell, n_groups_dropped_crossers = _drop_crossing_cells(
                cell=cell, group_col=group, d_col="d_gt"
            )
        else:
            warnings.warn(
                "drop_larger_lower=False: the analytical variance formula will "
                "be inconsistent with the point estimate for any multi-switch "
                "groups present in the data, producing a biased SE. Use only "
                "for diagnostic comparison against R or when you are confident "
                "no multi-switch groups exist.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 6b: TWFE diagnostic sample-contract notice
        #
        # The fitted twfe_* values (if the diagnostic succeeded in
        # Step 5a) were computed on the FULL pre-filter cell sample,
        # matching the standalone twowayfeweights() output. Steps 5b
        # and 6 may have dropped groups since then. When they did, the
        # fitted diagnostic and the dCDH point estimate describe
        # DIFFERENT samples, so we surface that divergence as a
        # UserWarning per the REGISTRY contract Note. Users see the
        # warning at fit time and can decide whether to pre-process
        # their data before re-fitting (or accept the documented
        # divergence).
        #
        # The warning fires whenever the user requested the diagnostic
        # AND filters dropped groups, even if _compute_twfe_diagnostic
        # itself failed (rank-deficient fallback) and
        # twfe_diagnostic_payload is None. The warning text uses "(if
        # the diagnostic succeeded)" to remain accurate in both cases.
        # ------------------------------------------------------------------
        if self.twfe_diagnostic and (n_groups_dropped_interior_gap + n_groups_dropped_crossers) > 0:
            warnings.warn(
                f"TWFE diagnostic sample-contract notice: the dCDH point "
                f"estimate, results.units, and inference fields use a "
                f"POST-FILTER sample after Step 5b dropped "
                f"{n_groups_dropped_interior_gap} interior-gap group(s) "
                f"and Step 6 dropped {n_groups_dropped_crossers} multi-"
                f"switch group(s). The fitted results.twfe_* values (if "
                f"the diagnostic succeeded) were computed on the FULL "
                f"pre-filter cell sample, so they describe a LARGER "
                f"sample (pre-filter) than overall_att. The standalone "
                f"twowayfeweights() function also uses the pre-filter "
                f"sample. This is the documented Phase 1 contract — see "
                f"REGISTRY.md ChaisemartinDHaultfoeuille `Note (TWFE "
                f"diagnostic sample contract)` for the rationale. To "
                f"reproduce the dCDH estimation sample for an external "
                f"TWFE comparison, pre-process your data to drop the "
                f"{n_groups_dropped_interior_gap + n_groups_dropped_crossers} "
                f"flagged groups before re-fitting.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 7: Singleton-baseline identification (footnote 15 of dynamic paper)
        # ------------------------------------------------------------------
        # The singleton-baseline filter identifies groups whose baseline
        # treatment value D_{g,1} is unique in the panel. Per footnote 15
        # of the dynamic paper, these have no baseline-matched cohort peer
        # and contribute zero variance under the cohort framework.
        #
        # IMPORTANT: under Python's documented period-based stable-control
        # interpretation, a singleton-baseline group can STILL be a valid
        # stable_0 / stable_1 control for the point estimate, even though
        # it has no cohort peer. The filter is therefore applied at the
        # variance stage only — the cell DataFrame retains these groups
        # so they can serve as stable controls.
        # Use the validated first global period as the canonical baseline.
        # Step 5b guarantees every group has an observation at this period,
        # so we can read it directly without a groupby.first() that could
        # otherwise return a later observed period for late-entry groups.
        baselines_per_group = cell.loc[cell[time] == first_global_period, [group, "d_gt"]].rename(
            columns={"d_gt": "_baseline"}
        )
        baseline_counts = baselines_per_group["_baseline"].value_counts()
        singleton_baseline_values = baseline_counts[baseline_counts < 2].index.tolist()
        singleton_baseline_groups: List[Any] = (
            baselines_per_group.loc[
                baselines_per_group["_baseline"].isin(singleton_baseline_values), group
            ].tolist()
            if singleton_baseline_values
            else []
        )
        n_groups_dropped_singleton_baseline = len(singleton_baseline_groups)
        if n_groups_dropped_singleton_baseline > 0:
            warnings.warn(
                f"Singleton-baseline filter (footnote 15 of dynamic paper): "
                f"{n_groups_dropped_singleton_baseline} group(s) excluded from "
                f"the cohort-recentered VARIANCE computation only — they remain "
                f"in the point-estimate sample as period-based stable controls. "
                f"Examples: {singleton_baseline_groups[:5]}"
                + (
                    f" (and {n_groups_dropped_singleton_baseline - 5} more)"
                    if n_groups_dropped_singleton_baseline > 5
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )

        if cell.empty or cell[group].nunique() == 0:
            raise ValueError(
                "After dropping multi-switch cells (drop_larger_lower=True), no "
                "groups remain. The dataset cannot support dCDH estimation. "
                "Check the input panel for diversity in treatment patterns."
            )

        # Determine the post-filter group set, period set, and per-group state
        all_groups = sorted(cell[group].unique().tolist())
        all_periods = sorted(cell[time].unique().tolist())
        n_obs_post = int(cell["n_gt"].sum())

        # ------------------------------------------------------------------
        # L_max validation (Phase 2): must be a positive integer not
        # exceeding the number of post-baseline periods. Validated here
        # (after period detection) rather than in _check_forward_compat_gates
        # (which runs before data is processed).
        # ------------------------------------------------------------------
        if L_max is not None:
            if not isinstance(L_max, int) or L_max < 1:
                raise ValueError(f"L_max must be a positive integer or None, got {L_max!r}.")
            n_post_baseline = len(all_periods) - 1
            if L_max > n_post_baseline:
                raise ValueError(
                    f"L_max={L_max} exceeds available post-baseline periods "
                    f"({n_post_baseline}). Maximum L_max for this panel "
                    f"is {n_post_baseline}."
                )

        if honest_did and L_max is None:
            raise ValueError(
                "honest_did=True requires L_max >= 1 for multi-horizon placebos. "
                "Set L_max to compute DID^{pl}_l placebos that HonestDiD uses as "
                "pre-period coefficients."
            )
        if honest_did and not self.placebo:
            raise ValueError(
                "honest_did=True requires placebo computation. The estimator was "
                "constructed with placebo=False. Use "
                "ChaisemartinDHaultfoeuille(placebo=True) (the default)."
            )

        # Pivot to (group x time) matrices for vectorized computations
        d_pivot = cell.pivot(index=group, columns=time, values="d_gt").reindex(
            index=all_groups, columns=all_periods
        )
        y_pivot = cell.pivot(index=group, columns=time, values="y_gt").reindex(
            index=all_groups, columns=all_periods
        )
        n_pivot = (
            cell.pivot(index=group, columns=time, values="n_gt")
            .reindex(index=all_groups, columns=all_periods)
            .fillna(0)
            .astype(int)
        )
        D_mat = d_pivot.to_numpy()
        Y_mat = y_pivot.to_numpy()
        N_mat = n_pivot.to_numpy()

        # Finalize survey obs-info with the pivot's column index so that
        # cell-level IF expansion can map per-row time values to matrix
        # column indices in _survey_se_from_group_if.
        if _obs_survey_info is not None:
            _obs_survey_info["periods"] = np.asarray(all_periods)

        # ------------------------------------------------------------------
        # Step 7b: Covariate residualization (DID^X)
        #
        # When controls are specified, residualize Y_mat by partialling
        # out covariate effects per baseline treatment group. This
        # transforms Y_mat so the per-group multi-horizon DID path
        # (event_study_effects, overall_att, joiners/leavers, by_path
        # surfaces, placebos, sup-t bands) automatically produces
        # covariate-adjusted estimates. The per-period DID path
        # (per_period_effects) intentionally remains on raw outcomes —
        # it uses binary joiner/leaver categorization and is not part
        # of the DID^X contract per REGISTRY.md "Note (Phase 3 DID^X
        # covariate adjustment)". See Web Appendix Section 1.2.
        # ------------------------------------------------------------------
        covariate_diagnostics: Optional[Dict[str, Any]] = None
        _switch_metadata_computed = False

        if controls is not None:
            # Pivot covariates to (n_groups, n_periods, n_covariates)
            X_pivots = []
            for c in controls:
                x_piv = cell.pivot(index=group, columns=time, values=c).reindex(
                    index=all_groups, columns=all_periods
                )
                X_pivots.append(x_piv.to_numpy())
            X_cell = np.stack(X_pivots, axis=2)

            # Need switch metadata for residualization (baselines, F_g)
            baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                _compute_group_switch_metadata(D_mat, N_mat)
            )
            _switch_metadata_computed = True

            # by_path + controls residualization-sample deviation from R.
            # R's `did_multiplegt_dyn(..., by_path, controls)` calls
            # `did_multiplegt_main()` once per path with `df_main` filtered
            # to: rows of the path's switchers OR rows where
            # `yet_to_switch=1 AND baseline matches the path's baseline`
            # (R/R/did_multiplegt_dyn.R lines 401-405). Inside the per-path
            # `did_multiplegt_main()` call, the per-baseline first-stage
            # residualization regression uses `(g, t)` cells where g's
            # treatment hasn't changed yet at t. Critically, R's path-
            # restricted subset INCLUDES the pre-switch rows of OTHER-path
            # switchers via the `yet_to_switch=1 AND baseline matches`
            # clause, so the first-stage SAMPLE that R uses for path B
            # equals: pre-switch rows of all switchers with matching
            # baseline + all rows of never-switchers with matching
            # baseline. This is BIT-IDENTICAL to the first-stage sample
            # we use under our global residualization — first-stage
            # coefficients (and therefore residualized outcomes) coincide,
            # and per-path point estimates match R exactly **under single-
            # baseline switcher panels** (every switcher has the same
            # `D_{g,1}`, regardless of how `F_g` varies across paths or
            # within a path). Empirical confirmation: the
            # `multi_path_reversible_by_path_controls` R-parity scenario
            # has 4 paths with switcher `F_g` values spanning [0..6] under
            # `D_{g,1}=0` for every switcher, and Python matches R to
            # rtol ~1e-11 across all `(path, horizon)` cells.
            #
            # On MULTI-baseline switcher panels the per-baseline regression
            # coefficients diverge per path under R (R's per-path subset
            # for path B drops switchers whose baseline differs from B's
            # baseline), so point estimates can diverge between Python and
            # R — warn the user explicitly. The check filters to switcher
            # groups only (never-switchers do not contribute to "switcher
            # baseline" multiplicity even if they appear at multiple
            # `D_{g,1}` values across the never-treated / always-treated
            # control mix). SE inheritance (cross-path cohort-sharing) is
            # documented separately in REGISTRY.md.
            if self.by_path is not None or self.paths_of_interest is not None:
                _switcher_mask = first_switch_idx_arr >= 0
                if _switcher_mask.any():
                    _switcher_baselines = baselines[_switcher_mask]
                    if np.unique(_switcher_baselines).size > 1:
                        warnings.warn(
                            "by_path / paths_of_interest + controls: "
                            "switcher baselines D_{g,1} take multiple values "
                            "in this panel. Python residualizes once on the "
                            "full panel before path enumeration; R "
                            "`did_multiplegt_dyn(..., by_path, controls)` "
                            "re-runs residualization per path on the "
                            "path-restricted subsample, so per-path point "
                            "estimates can diverge between Python and R on "
                            "this panel. See `docs/methodology/REGISTRY.md` "
                            "(`Note (Phase 3 by_path ...)` -> Per-path "
                            "covariate residualization) for the full "
                            "deviation contract.",
                            UserWarning,
                            stacklevel=2,
                        )

            Y_mat_residualized, covariate_diagnostics, _failed_baselines = (
                _compute_covariate_residualization(
                    Y_mat=Y_mat,
                    X_cell=X_cell,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    rank_deficient_action=self.rank_deficient_action,
                )
            )
            # Zero out N_mat for failed-stratum groups so the downstream
            # eligibility checks (N_mat[g, idx] > 0) naturally exclude
            # them from all DID/IF/placebo computation.
            if _failed_baselines:
                for g_idx in range(len(baselines)):
                    if float(baselines[g_idx]) in _failed_baselines:
                        N_mat[g_idx, :] = 0
            # Keep raw Y_mat for the per-period DID path (which does not
            # support covariate residualization - it uses binary joiner/leaver
            # categorization). The residualized matrix is used only by the
            # per-group multi-horizon path (L_max >= 1).
            Y_mat_raw = Y_mat
            Y_mat = Y_mat_residualized

        # ------------------------------------------------------------------
        # Step 7c: First-differencing for linear trends (DID^{fd})
        #
        # When trends_linear=True, replace Y_mat with Z_mat (first-
        # differenced outcomes) so that DID_{g,l}(Z) = DID^{fd}_{g,l}.
        # N_mat is also adjusted: N_mat_fd marks which Z values are valid.
        # IMPORTANT: _compute_group_switch_metadata uses the ORIGINAL
        # N_mat (treatment path metadata), not N_mat_fd.
        # ------------------------------------------------------------------
        _is_trends_linear = trends_linear is True
        linear_trends_effects: Optional[Dict[int, Dict[str, Any]]] = None
        # N_mat_orig preserves observation counts for switch-metadata and
        # cohort-identification code that must NOT see the first-differenced
        # N_mat_fd. When trends_linear=False, N_mat_orig == N_mat.
        N_mat_orig = N_mat

        if _is_trends_linear:
            if L_max is None:
                raise ValueError(
                    "Group-specific linear trends (DID^{fd}) requires "
                    "L_max >= 1. Set L_max to use the per-group "
                    "DID_{g,l} path with trend adjustment."
                )
            if len(all_periods) < 3:
                raise ValueError(
                    "Group-specific linear trends (DID^{fd}) requires "
                    "at least 3 time periods (F_g >= 3 in the paper). "
                    f"Got {len(all_periods)} period(s)."
                )
            # Compute switch metadata on original N_mat if not done yet
            if not _switch_metadata_computed:
                baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                    _compute_group_switch_metadata(D_mat, N_mat)
                )
                _switch_metadata_computed = True
            # Count and warn about excluded groups (F_g < 3 -> f_g < 2)
            n_excluded_fd = int(((first_switch_idx_arr >= 0) & (first_switch_idx_arr < 2)).sum())
            if n_excluded_fd > 0:
                warnings.warn(
                    f"DID^{{fd}} (trends_linear=True): {n_excluded_fd} "
                    f"switching group(s) have F_g < 3 (fewer than 2 "
                    f"pre-switch periods) and are excluded from the "
                    f"trend-adjusted estimation.",
                    UserWarning,
                    stacklevel=2,
                )
            # Multi-baseline switcher panel detection (`by_path + trends_linear`):
            # mirror the analogous warning fired by `by_path + controls` at
            # `:1565-1584`. Python first-differences once globally before
            # path enumeration; R `did_multiplegt_dyn(..., by_path, trends_lin)`
            # re-runs the full pipeline (including first-differencing) per
            # path's restricted subsample. On single-baseline switcher
            # panels the two architectures coincide; on multi-baseline
            # switcher panels (some switchers have `D_{g,1}=0`, others
            # `D_{g,1}=1`) per-path point estimates can diverge — warn the
            # user explicitly so they don't silently consume estimates
            # that disagree with R. The check filters to switcher groups
            # only (never-switchers / always-treated controls don't
            # contribute to switcher baseline multiplicity).
            if self.by_path is not None or self.paths_of_interest is not None:
                _switcher_mask_tl = first_switch_idx_arr >= 0
                if _switcher_mask_tl.any():
                    _switcher_baselines_tl = baselines[_switcher_mask_tl]
                    if np.unique(_switcher_baselines_tl).size > 1:
                        warnings.warn(
                            "by_path / paths_of_interest + trends_linear: "
                            "switcher baselines D_{g,1} take multiple values "
                            "in this panel. Python first-differences once on "
                            "the full panel before path enumeration; R "
                            "`did_multiplegt_dyn(..., by_path, trends_lin)` "
                            "re-runs the full pipeline (including "
                            "first-differencing) on each path's restricted "
                            "subsample, so per-path point estimates can "
                            "diverge between Python and R on this panel. "
                            "See `docs/methodology/REGISTRY.md` "
                            "(`Note (Phase 3 by_path ...)` -> Per-path "
                            "linear-trends DID^{fd}) for the full "
                            "deviation contract.",
                            UserWarning,
                            stacklevel=2,
                        )
                # F_g=3 boundary-case divergence (`by_path + trends_linear`).
                # `F_g=3` switchers have exactly 2 pre-switch periods,
                # which after trends_linear's first-difference and
                # `time != 1` filter leaves only 1 valid pre-window Z
                # value. R re-runs the full pipeline on each path's
                # restricted subsample (path's switchers + same-baseline
                # yet-to-treat controls), and this single-pre-period
                # regime triggers different control-eligibility
                # treatment in R's per-path call than Python's global-
                # then-disaggregate architecture. Empirically observed
                # 30-165% rel diff on path 1 of the parity fixture's
                # earlier `F_g=3` variant; the shipped parity scenario
                # uses `F_g >= 4` exclusively. Fire a targeted warning
                # whenever the panel contains any `F_g=3` switchers AND
                # `by_path` is requested, so practitioners hitting this
                # boundary regime see the divergence flag explicitly.
                _f_g_three_count = int((first_switch_idx_arr == 2).sum())
                if _f_g_three_count > 0:
                    warnings.warn(
                        f"by_path / paths_of_interest + trends_linear: "
                        f"{_f_g_three_count} switching group(s) have "
                        f"F_g=3 (exactly 2 "
                        f"pre-switch periods). After first-differencing "
                        f"and the time==1 filter, these groups have "
                        f"only 1 valid pre-window Z value, which "
                        f"triggers a documented boundary-case "
                        f"divergence between Python's global-then-"
                        f"disaggregate architecture and R's per-path "
                        f"full-pipeline call. Per-path point estimates "
                        f"on paths whose switchers include F_g=3 can "
                        f"diverge from `did_multiplegt_dyn(..., "
                        f"by_path, trends_lin)` by 30%+ on point "
                        f"estimates. See `docs/methodology/REGISTRY.md` "
                        f"(`Note (Phase 3 by_path ...)` -> Per-path "
                        f"linear-trends DID^{{fd}}) for the full "
                        f"deviation contract.",
                        UserWarning,
                        stacklevel=2,
                    )
            N_mat_orig = N_mat.copy()
            Y_mat, N_mat = _compute_first_differenced_matrix(Y_mat, N_mat)

        # ------------------------------------------------------------------
        # Step 7d: State-set trends validation (trends_nonparam)
        #
        # When trends_nonparam is set (a column name), restrict the
        # control pool for each switcher to groups in the same set.
        # ------------------------------------------------------------------
        set_ids_arr: Optional[np.ndarray] = None

        if trends_nonparam is not None:
            if L_max is None:
                raise ValueError(
                    "State-set-specific trends (trends_nonparam) requires "
                    "L_max >= 1. Set L_max to use the per-group "
                    "DID_{g,l} path with state-set trends."
                )
            set_col = str(trends_nonparam)
            if set_col not in data.columns:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} not found in "
                    f"data. Available columns: {list(data.columns)}"
                )
            # SurveyDesign.subpopulation() contract: scope NaN and
            # time-invariance validation to positive-weight rows so
            # excluded obs with missing set IDs do not abort the fit.
            if survey_weights is not None:
                pos_mask_tnp = np.asarray(survey_weights) > 0
                data_tnp = data.loc[pos_mask_tnp]
            else:
                data_tnp = data
            # Reject NaN/missing set assignments (effective sample only)
            n_na_set = int(data_tnp[set_col].isna().sum())
            if n_na_set > 0:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} contains "
                    f"{n_na_set} NaN/missing value(s). All groups must "
                    f"have a valid set assignment."
                )
            # Aggregate set membership per group (must be time-invariant)
            set_per_group = data_tnp.groupby(group)[set_col].nunique()
            time_varying = set_per_group[set_per_group > 1]
            if len(time_varying) > 0:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} must be "
                    f"time-invariant within each group. "
                    f"{len(time_varying)} group(s) have varying values. "
                    f"Examples: {time_varying.index.tolist()[:5]}"
                )
            # Set partition must be coarser than group (multiple groups
            # per set). A group-level partition creates singleton sets
            # with no within-set controls available.
            set_map_check = data_tnp.groupby(group)[set_col].first()
            n_sets = set_map_check.nunique()
            n_groups_total = len(set_map_check)
            if n_sets >= n_groups_total:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} defines "
                    f"{n_sets} distinct sets for {n_groups_total} "
                    f"groups. The set partition must be coarser than "
                    f"group (multiple groups per set) to provide "
                    f"within-set controls."
                )
            # Extract set membership per group aligned with all_groups
            set_map = data_tnp.groupby(group)[set_col].first()
            set_ids_arr = np.array([set_map.loc[g] for g in all_groups], dtype=object)

        # ------------------------------------------------------------------
        # Step 8-9: Switching-cell counts and per-period DIDs (Theorem 3)
        #          with explicit A11 zero-retention pseudocode
        # ------------------------------------------------------------------
        (
            per_period_effects,
            a11_warnings,
            did_plus_t_arr,
            did_minus_t_arr,
            n_10_t_arr,
            n_01_t_arr,
            n_00_t_arr,
            n_11_t_arr,
            a11_plus_zeroed_arr,
            a11_minus_zeroed_arr,
        ) = _compute_per_period_dids(
            D_mat=D_mat,
            # Use raw (unadjusted) outcomes for per-period DID. Covariate
            # residualization applies only to the per-group multi-horizon
            # path (L_max >= 1). The per-period path uses binary
            # joiner/leaver categorization and is not part of the DID^X
            # contract (Web Appendix Section 1.2).
            # Use raw outcomes for per-period DID when controls or
            # trends_linear is active (both transform Y_mat).
            Y_mat=(
                Y_mat_raw
                if controls is not None
                else (y_pivot.to_numpy() if _is_trends_linear else Y_mat)
            ),
            N_mat=N_mat_orig,
            periods=all_periods,
        )
        if a11_warnings:
            warnings.warn(
                f"Assumption 11 (existence of stable controls) violated in "
                f"{len(a11_warnings)} period(s); the affected DID_+/DID_- values "
                f"are zeroed but their switcher counts are retained in the N_S "
                f"denominator (matching paper convention). Affected: "
                f"{', '.join(a11_warnings[:3])}"
                + (f" (and {len(a11_warnings) - 3} more)" if len(a11_warnings) > 3 else ""),
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 10: Aggregate DID_M = sum_t (n_10_t * did_plus_t + n_01_t * did_minus_t) / N_S
        # ------------------------------------------------------------------
        N_S = int(n_10_t_arr.sum() + n_01_t_arr.sum())
        # For non-binary treatment, the per-period DID path may find N_S=0
        # because it uses binary joiner/leaver categorization. When L_max
        # is set, the multi-horizon path (which handles non-binary correctly
        # via per-group DID_{g,l}) will compute the effects. Only raise if
        # L_max is also None (i.e., no fallback path).
        is_binary = set(np.unique(D_mat[~np.isnan(D_mat)])).issubset({0.0, 1.0})
        if not is_binary and L_max is None:
            raise ValueError(
                "Non-binary treatment requires L_max >= 1. The per-period DID "
                "path uses binary joiner/leaver categorization; set L_max to "
                "use the per-group DID_{g,l} building block which handles "
                "non-binary treatment."
            )
        if (self.by_path is not None or self.paths_of_interest is not None) and not is_binary:
            finite_D = D_mat[~np.isnan(D_mat)]
            if finite_D.size > 0 and not np.all(finite_D == np.round(finite_D)):
                bad_examples = np.unique(finite_D[finite_D != np.round(finite_D)])[:3]
                raise ValueError(
                    f"by_path / paths_of_interest with non-binary "
                    f"treatment requires integer-coded treatment values "
                    f"(D in Z). Found non-integer values: "
                    f"{bad_examples.tolist()!r}. Round/discretize D "
                    f"before fitting, or set by_path=None and "
                    f"paths_of_interest=None with continuous treatment."
                )
        if N_S == 0 and (L_max is None or is_binary):
            raise ValueError(
                "No switching cells found in the data after filtering: every "
                "group has constant treatment for the entire panel. dCDH "
                "requires at least one (g, t) cell where the group's treatment "
                "differs from the previous period."
            )
        if N_S > 0:
            overall_att = float((n_10_t_arr @ did_plus_t_arr + n_01_t_arr @ did_minus_t_arr) / N_S)
        else:
            # Non-binary treatment with L_max: per-period DID is not
            # applicable. The multi-horizon path will provide overall_att
            # via the cost-benefit delta.
            overall_att = float("nan")

        # ------------------------------------------------------------------
        # Step 11: Joiners and leavers views
        # ------------------------------------------------------------------
        joiner_total = int(n_10_t_arr.sum())
        leaver_total = int(n_01_t_arr.sum())
        joiners_available = joiner_total > 0
        leavers_available = leaver_total > 0
        if joiners_available:
            joiners_att = float((n_10_t_arr @ did_plus_t_arr) / joiner_total)
        else:
            joiners_att = float("nan")
        if leavers_available:
            leavers_att = float((n_01_t_arr @ did_minus_t_arr) / leaver_total)
        else:
            leavers_att = float("nan")

        # Joiner / leaver sample-size metadata.
        # n_*_cells: total switching cells across all periods (sum of per-period
        #            cell counts; each (g, t) joiner/leaver cell counted once).
        # n_*_obs:   actual observation count (sum of n_gt over the same cells),
        #            which differs from cells when individual-level inputs have
        #            multiple original observations per (g, t).
        n_joiner_cells = int(n_10_t_arr.sum())
        n_leaver_cells = int(n_01_t_arr.sum())
        n_joiner_obs = 0
        n_leaver_obs = 0
        for t_idx in range(1, len(all_periods)):
            d_curr = D_mat[:, t_idx]
            d_prev = D_mat[:, t_idx - 1]
            n_curr = N_mat[:, t_idx]
            n_prev = N_mat[:, t_idx - 1]
            present = (n_curr > 0) & (n_prev > 0)
            joiner_mask_t = (d_prev == 0) & (d_curr == 1) & present
            leaver_mask_t = (d_prev == 1) & (d_curr == 0) & present
            n_joiner_obs += int(n_curr[joiner_mask_t].sum())
            n_leaver_obs += int(n_curr[leaver_mask_t].sum())

        # ------------------------------------------------------------------
        # Step 12: Placebo (DID_M^pl)
        # ------------------------------------------------------------------
        placebo_available = False
        placebo_effect = float("nan")
        if self.placebo:
            if len(all_periods) < 3:
                warnings.warn(
                    f"Placebo DID_M^pl requires at least 3 time "
                    f"periods; the post-filter panel has only {len(all_periods)}. "
                    "Skipping the placebo computation. Pass placebo=False to "
                    "suppress this warning, or use a panel with T >= 3.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                placebo_payload = _compute_placebo(
                    D_mat=D_mat, Y_mat=Y_mat, N_mat=N_mat, periods=all_periods
                )
                if placebo_payload is None:
                    warnings.warn(
                        "Placebo DID_M^pl could not be computed: no qualifying "
                        "switching cells with the required 3-period stable "
                        "history exist after filtering. The placebo fields on "
                        "the results object are NaN with placebo_available=False.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    placebo_effect, placebo_available, placebo_a11_warnings = placebo_payload
                    # Surface placebo A11 violations via a consolidated warning
                    # mirroring the main DID path's contract. The affected
                    # per-period placebo contributions are zeroed in the
                    # numerator with their switcher counts retained in the
                    # placebo N_S^pl denominator (placebo zero-retention).
                    if placebo_a11_warnings:
                        warnings.warn(
                            f"Placebo (DID_M^pl) Assumption 11 violations in "
                            f"{len(placebo_a11_warnings)} period(s); the affected "
                            f"placebo contributions are zeroed but their switcher "
                            f"counts are retained in the placebo N_S denominator "
                            f"(matching placebo paper convention). Affected: "
                            + ", ".join(placebo_a11_warnings[:3])
                            + (
                                f" (and {len(placebo_a11_warnings) - 3} more)"
                                if len(placebo_a11_warnings) > 3
                                else ""
                            ),
                            UserWarning,
                            stacklevel=2,
                        )

        # ------------------------------------------------------------------
        # Step 12b: Per-group switch metadata (shared by Phase 1 IF and
        #           Phase 2 multi-horizon). May already be computed by
        #           Step 7b (covariate residualization).
        # ------------------------------------------------------------------
        if not _switch_metadata_computed:
            baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                _compute_group_switch_metadata(D_mat, N_mat_orig)
            )

        # ------------------------------------------------------------------
        # Step 12c: Multi-horizon per-group computation (L_max >= 1)
        # ------------------------------------------------------------------
        multi_horizon_dids: Optional[Dict[int, Dict[str, Any]]] = None
        multi_horizon_if: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        multi_horizon_se: Optional[Dict[int, float]] = None
        multi_horizon_inference: Optional[Dict[int, Dict[str, Any]]] = None

        if L_max is not None and L_max >= 1:
            multi_horizon_dids = _compute_multi_horizon_dids(
                D_mat=D_mat,
                Y_mat=Y_mat,
                N_mat=N_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                L_max=L_max,
                set_ids=set_ids_arr,
            )
            # Surface A11 warnings from multi-horizon computation
            mh_a11 = multi_horizon_dids.pop("_a11_warnings", None)  # type: ignore[call-overload]  # sentinel str key in int-keyed dict
            if mh_a11:
                warnings.warn(
                    f"Multi-horizon control-availability violations in "
                    f"{len(mh_a11)} (group, horizon) pair(s): affected "
                    f"groups are excluded from N_l (no observed baseline-"
                    f"matched controls at the outcome period). Examples: "
                    + ", ".join(mh_a11[:3])
                    + (f" (and {len(mh_a11) - 3} more)" if len(mh_a11) > 3 else ""),
                    UserWarning,
                    stacklevel=2,
                )

            # Guard: if no eligible switchers at horizon 1 (e.g., all
            # groups have constant treatment), raise ValueError.
            if 1 in multi_horizon_dids and multi_horizon_dids[1]["N_l"] == 0:
                raise ValueError(
                    "No switching groups found at horizon 1 after filtering. "
                    "dCDH requires at least one group whose treatment changes "
                    "from the baseline period."
                )

            multi_horizon_if = _compute_per_group_if_multi_horizon(
                D_mat=D_mat,
                Y_mat=Y_mat,
                N_mat=N_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                L_max=L_max,
                set_ids=set_ids_arr,
                compute_per_period=_obs_survey_info is not None,
            )

            # Per-horizon analytical SE via cohort recentering.
            # Reuse the singleton-baseline exclusion from Step 7 and
            # build cohort IDs per horizon.
            singleton_baseline_set = set(singleton_baseline_groups)
            eligible_mask_var = np.array(
                [g not in singleton_baseline_set for g in all_groups], dtype=bool
            )
            # Lift eligible_groups_var once for downstream by_path /
            # paths_of_interest call sites; mirrors the inline
            # `_elig_groups_l` construction at the global per-horizon
            # path so both surfaces share the same variance-eligibility
            # ordering. Used to align per-group IF entries with the
            # cell-period allocator's `eligible_groups` argument.
            eligible_groups_var: List[Any] = [
                all_groups[g] for g in range(len(all_groups)) if eligible_mask_var[g]
            ]

            multi_horizon_se = {}
            multi_horizon_inference = {}
            # Cache the cohort-recentered per-cell IF tensors built
            # inside this loop so the bootstrap block below can reuse
            # them without recomputing cohort-recentering. Keyed by
            # horizon; value shape (n_eligible_var, n_periods). Populated
            # only when a survey design is active (else the per-period
            # tensor is None and the bootstrap has no cell-level path
            # to take).
            _mh_pp_cache: Dict[int, np.ndarray] = {}
            # multi_horizon_if is computed unconditionally in this branch.
            assert multi_horizon_if is not None
            # Compute inference for ALL horizons 1..L_max (including l=1)
            # so the event_study_effects dict uses a consistent estimand
            # (per-group DID_{g,l}) across all horizons.
            for l_h in range(1, L_max + 1):
                U_l, U_pp_l = multi_horizon_if[l_h]
                # Cohort IDs for this horizon: (D_{g,1}, F_g, S_g) triples
                # are the same as Phase 1 (cohort identity depends on first
                # switch, not on the horizon). Filter to eligible.
                cohort_keys_l = [
                    (
                        float(baselines[g]),
                        int(first_switch_idx_arr[g]),
                        int(switch_direction_arr[g]),
                    )
                    for g in range(len(all_groups))
                ]
                unique_c: Dict[Tuple[float, int, int], int] = {}
                cid_l = np.zeros(len(all_groups), dtype=int)
                for g in range(len(all_groups)):
                    if not eligible_mask_var[g]:
                        cid_l[g] = -1
                        continue
                    key = cohort_keys_l[g]
                    if key not in unique_c:
                        unique_c[key] = len(unique_c)
                    cid_l[g] = unique_c[key]

                # Use the full variance-eligible group set (singleton-
                # baseline exclusion only). Do NOT intersect with
                # did_eligible — never-switchers and later-switching
                # controls can have non-zero IF mass via their control
                # roles, and dropping them understates the SE.
                U_l_elig = U_l[eligible_mask_var]
                cid_elig = cid_l[eligible_mask_var]
                U_centered_l = _cohort_recenter(U_l_elig, cid_elig)
                # Only build the cell-level attribution when the IF
                # helper actually produced a per-period tensor (i.e.,
                # a survey design is present). Otherwise the plug-in
                # path consumes U_centered_l only.
                if U_pp_l is not None:
                    U_centered_pp_l: Optional[np.ndarray] = _cohort_recenter_per_period(
                        U_pp_l[eligible_mask_var], cid_elig
                    )
                    _mh_pp_cache[l_h] = U_centered_pp_l
                else:
                    U_centered_pp_l = None
                N_l_h = multi_horizon_dids[l_h]["N_l"]
                _elig_groups_l = [
                    all_groups[g] for g in range(len(all_groups)) if eligible_mask_var[g]
                ]
                se_l, n_valid_l = _compute_se(
                    U_centered=U_centered_l,
                    divisor=N_l_h,
                    obs_survey_info=_obs_survey_info,
                    eligible_groups=_elig_groups_l,
                    U_centered_per_period=U_centered_pp_l,
                )
                if n_valid_l is not None:
                    _replicate_n_valid_list.append(n_valid_l)
                multi_horizon_se[l_h] = se_l

                did_l_val = multi_horizon_dids[l_h]["did_l"]
                _df_s = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
                t_l, p_l, ci_l = safe_inference(
                    did_l_val, se_l, alpha=self.alpha, df=_inference_df(_df_s, resolved_survey)
                )
                multi_horizon_inference[l_h] = {
                    "effect": did_l_val,
                    "se": se_l,
                    "t_stat": t_l,
                    "p_value": p_l,
                    "conf_int": ci_l,
                    "n_obs": N_l_h,
                }

            # Emit <50% switcher warning for far horizons
            if multi_horizon_dids.get(1, {}).get("N_l", 0) > 0:
                N_1_ref = multi_horizon_dids[1]["N_l"]
                thin_horizons = [
                    l_h
                    for l_h in range(2, L_max + 1)
                    if multi_horizon_dids[l_h]["N_l"] < 0.5 * N_1_ref
                    and multi_horizon_dids[l_h]["N_l"] > 0
                ]
                if thin_horizons:
                    warnings.warn(
                        f"Fewer than 50% of l=1 switchers contribute at "
                        f"horizon(s) {thin_horizons}. Far-horizon estimates "
                        f"may be noisy. The paper recommends not reporting "
                        f"horizons where fewer than ~50% of switchers "
                        f"contribute (Favara-Imbs application, footnote 14).",
                        UserWarning,
                        stacklevel=2,
                    )

        # by_path disaggregation by observed treatment trajectory
        path_effects: Optional[Dict[Tuple[int, ...], Dict[str, Any]]] = None
        path_placebos: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = None
        path_cumulated_event_study: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = (
            None
        )
        if (
            (self.by_path is not None or self.paths_of_interest is not None)
            and L_max is not None
            and L_max >= 1
            and multi_horizon_dids is not None
        ):
            _df_s_bp = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
            path_effects = _compute_path_effects(
                D_mat=D_mat,
                Y_mat=Y_mat,
                N_mat=N_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                L_max=L_max,
                by_path=self.by_path,
                paths_of_interest=self.paths_of_interest,
                eligible_mask_var=eligible_mask_var,
                multi_horizon_dids=multi_horizon_dids,
                all_groups=all_groups,
                alpha=self.alpha,
                df_inference=_inference_df(_df_s_bp, resolved_survey),
                set_ids=set_ids_arr,
                obs_survey_info=_obs_survey_info,
                eligible_groups=eligible_groups_var,
                replicate_n_valid_list=_replicate_n_valid_list,
            )
            # NOTE: per-path cumulated layer is computed AFTER the
            # bootstrap propagation block below (search for
            # `path_cumulated_event_study =`) so it reads the final
            # post-bootstrap per-horizon SEs rather than the analytical
            # ones that path_effects was just populated with. This
            # mirrors the global `linear_trends_effects` cumulation
            # which also runs after the event_study bootstrap propagation.

        # Phase 2: placebos, normalized effects, cost-benefit delta
        multi_horizon_placebos: Optional[Dict[int, Dict[str, Any]]] = None
        placebo_horizon_if: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        placebo_horizon_se: Optional[Dict[int, float]] = None
        placebo_horizon_inference: Optional[Dict[int, Dict[str, Any]]] = None
        normalized_effects_dict: Optional[Dict[int, Dict[str, Any]]] = None
        cost_benefit_result: Optional[Dict[str, Any]] = None

        if L_max is not None and L_max >= 1 and multi_horizon_dids is not None:
            # Dynamic placebos DID^{pl}_l
            if self.placebo:
                multi_horizon_placebos = _compute_multi_horizon_placebos(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    set_ids=set_ids_arr,
                )
                # Surface placebo A11 warnings
                pl_a11 = multi_horizon_placebos.pop("_a11_warnings", None)  # type: ignore[call-overload]  # sentinel str key in int-keyed dict
                if pl_a11:
                    warnings.warn(
                        f"Multi-horizon placebo control-availability "
                        f"violations in {len(pl_a11)} (group, lag) pair(s): "
                        f"affected groups are excluded from N^{{pl}}_l "
                        f"(no observed controls). Examples: "
                        + ", ".join(pl_a11[:3])
                        + (f" (and {len(pl_a11) - 3} more)" if len(pl_a11) > 3 else ""),
                        UserWarning,
                        stacklevel=2,
                    )

            # Placebo IF computation + analytical SE
            if multi_horizon_placebos is not None:
                placebo_horizon_if = _compute_per_group_if_placebo_horizon(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    compute_per_period=_obs_survey_info is not None,
                    set_ids=set_ids_arr,
                )
                # Per-placebo-horizon analytical SE via cohort recentering
                # (same pattern as positive-horizon SE at Step 12c).
                placebo_horizon_se = {}
                placebo_horizon_inference = {}
                # placebo_horizon_if is computed unconditionally in this branch.
                assert placebo_horizon_if is not None
                singleton_baseline_set_pl = set(singleton_baseline_groups)
                eligible_mask_pl = np.array(
                    [g not in singleton_baseline_set_pl for g in all_groups],
                    dtype=bool,
                )
                # Cache per-placebo-horizon cohort-recentered per-cell
                # IF tensors for the bootstrap block below (same
                # pattern as _mh_pp_cache for positive horizons).
                _pl_pp_cache: Dict[int, np.ndarray] = {}
                for lag_l in range(1, L_max + 1):
                    pl_data = multi_horizon_placebos.get(lag_l)
                    if pl_data is None or pl_data["N_pl_l"] == 0:
                        placebo_horizon_se[lag_l] = float("nan")
                        continue
                    U_pl, U_pp_pl = placebo_horizon_if[lag_l]
                    # Cohort IDs (same as positive horizons)
                    cohort_keys_pl = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cpl: Dict[Tuple[float, int, int], int] = {}
                    cid_pl = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_pl[g]:
                            cid_pl[g] = -1
                            continue
                        key = cohort_keys_pl[g]
                        if key not in unique_cpl:
                            unique_cpl[key] = len(unique_cpl)
                        cid_pl[g] = unique_cpl[key]
                    U_pl_elig = U_pl[eligible_mask_pl]
                    cid_elig_pl = cid_pl[eligible_mask_pl]
                    U_centered_pl_l = _cohort_recenter(U_pl_elig, cid_elig_pl)
                    if U_pp_pl is not None:
                        U_centered_pp_pl_l: Optional[np.ndarray] = _cohort_recenter_per_period(
                            U_pp_pl[eligible_mask_pl], cid_elig_pl
                        )
                        _pl_pp_cache[lag_l] = U_centered_pp_pl_l
                    else:
                        U_centered_pp_pl_l = None
                    _elig_groups_pl = [
                        all_groups[g] for g in range(len(all_groups)) if eligible_mask_pl[g]
                    ]
                    se_pl_l, n_valid_pl_l = _compute_se(
                        U_centered=U_centered_pl_l,
                        divisor=pl_data["N_pl_l"],
                        obs_survey_info=_obs_survey_info,
                        eligible_groups=_elig_groups_pl,
                        U_centered_per_period=U_centered_pp_pl_l,
                    )
                    if n_valid_pl_l is not None:
                        _replicate_n_valid_list.append(n_valid_pl_l)
                    placebo_horizon_se[lag_l] = se_pl_l
                    pl_val = pl_data["placebo_l"]
                    _df_s = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
                    t_pl_l, p_pl_l, ci_pl_l = safe_inference(
                        pl_val,
                        se_pl_l,
                        alpha=self.alpha,
                        df=_inference_df(_df_s, resolved_survey),
                    )
                    placebo_horizon_inference[lag_l] = {
                        "effect": pl_val,
                        "se": se_pl_l,
                        "t_stat": t_pl_l,
                        "p_value": p_pl_l,
                        "conf_int": ci_pl_l,
                        "n_obs": pl_data["N_pl_l"],
                    }

            # Per-path backward-horizon placebos under by_path. Sibling
            # of the per-path event-study computation above; keyed by
            # path tuple -> negative-int lag (-l for lag l) to match
            # `placebo_event_study`'s convention. Inherits the cross-
            # path cohort-sharing SE deviation from R documented for
            # `path_effects` (full-panel cohort-centered plug-in vs
            # R's per-path re-run).
            if (
                (self.by_path is not None or self.paths_of_interest is not None)
                and self.placebo
                and multi_horizon_placebos is not None
            ):
                _df_s_bp_pl = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
                path_placebos = _compute_path_placebos(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    by_path=self.by_path,
                    paths_of_interest=self.paths_of_interest,
                    eligible_mask_var=eligible_mask_var,
                    multi_horizon_placebos=multi_horizon_placebos,
                    alpha=self.alpha,
                    df_inference=_inference_df(_df_s_bp_pl, resolved_survey),
                    set_ids=set_ids_arr,
                    obs_survey_info=_obs_survey_info,
                    eligible_groups=eligible_groups_var,
                    replicate_n_valid_list=_replicate_n_valid_list,
                )

            # Per-path inference for replicate-weight designs is
            # refreshed in the final R2 P1b block below (alongside
            # global event-study / placebo / heterogeneity surfaces),
            # so it reflects the FINAL `_replicate_n_valid_list` after
            # heterogeneity / overall / joiners / leavers IF sites
            # have appended their own `n_valid` values. Computing it
            # here would only see per-path appends and miss any later
            # df shrinkage from those subsequent IF sites.

            # Normalized effects DID^n_l (suppressed under trends_linear
            # because event_study_effects holds second-differences DID^{fd}_l,
            # not level effects - normalizing second-differences is wrong)
            if not _is_trends_linear:
                normalized_effects_dict = _compute_normalized_effects(
                    multi_horizon_dids=multi_horizon_dids,
                    D_mat=D_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    L_max=L_max,
                )

            # Cost-benefit delta (only meaningful when L_max >= 2)
            if L_max >= 2:
                cost_benefit_result = _compute_cost_benefit_delta(
                    multi_horizon_dids=multi_horizon_dids,
                    D_mat=D_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    L_max=L_max,
                )
                if cost_benefit_result.get("has_leavers", False):
                    warnings.warn(
                        "Assumption 7 (D_{g,t} >= D_{g,1}) is violated: leavers "
                        "present. The cost-benefit delta is computed on the full "
                        "sample (both joiners and leavers); delta_joiners and "
                        "delta_leavers are available separately on "
                        "results.cost_benefit_delta.",
                        UserWarning,
                        stacklevel=2,
                    )

        # ------------------------------------------------------------------
        # Step 13-16: Cohort identification, influence-function vectors,
        #             cohort-recentered plug-in variance
        # ------------------------------------------------------------------
        (
            U_centered_overall,
            n_groups_for_overall_var,
            n_cohorts,
            n_groups_dropped_never_switching,
            U_centered_joiners,
            U_centered_leavers,
            _eligible_group_ids,
            U_centered_pp_overall,
            U_centered_pp_joiners,
            U_centered_pp_leavers,
        ) = _compute_cohort_recentered_inputs(
            D_mat=D_mat,
            # Phase 1 IF uses per-period structure: use raw outcomes
            # when controls or trends_linear transform Y_mat.
            Y_mat=(
                Y_mat_raw
                if controls is not None
                else (y_pivot.to_numpy() if _is_trends_linear else Y_mat)
            ),
            N_mat=N_mat_orig,
            n_10_t_arr=n_10_t_arr,
            n_00_t_arr=n_00_t_arr,
            n_01_t_arr=n_01_t_arr,
            n_11_t_arr=n_11_t_arr,
            a11_plus_zeroed_arr=a11_plus_zeroed_arr,
            a11_minus_zeroed_arr=a11_minus_zeroed_arr,
            all_groups=all_groups,
            singleton_baseline_groups=singleton_baseline_groups,
            compute_per_period=_obs_survey_info is not None,
        )

        # Analytical SE for DID_M (survey-aware when survey_design provided)
        overall_se, n_valid_overall = _compute_se(
            U_centered=U_centered_overall,
            divisor=N_S,
            obs_survey_info=_obs_survey_info,
            eligible_groups=_eligible_group_ids,
            U_centered_per_period=U_centered_pp_overall,
        )
        if n_valid_overall is not None:
            _replicate_n_valid_list.append(n_valid_overall)
        # Detect the degenerate-cohort case: every variance-eligible group
        # forms its own (D_{g,1}, F_g, S_g) cohort, so the centered
        # influence function is identically zero and `_plugin_se` returns
        # NaN. Surface this as a UserWarning so users see the variance is
        # unidentified rather than silently mistaking NaN for "missing
        # data" or 0.0 for infinite precision. The bootstrap path inherits
        # the same degeneracy on this panel because it multiplies the
        # same all-zero centered IF by random weights.
        if np.isnan(overall_se) and n_groups_for_overall_var > 0 and N_S > 0:
            warnings.warn(
                f"Cohort-recentered analytical variance is unidentified: "
                f"every variance-eligible group forms its own "
                f"(D_{{g,1}}, F_g, S_g) cohort "
                f"({n_groups_for_overall_var} groups across {n_cohorts} "
                f"cohorts), so the centered influence function vector is "
                f"identically zero. The DID_M point estimate is still "
                f"valid; SE / t_stat / p_value / conf_int are NaN-"
                f"consistent. To get a non-degenerate analytical SE, "
                f"include more groups so cohorts have peers (real-world "
                f"panels typically have G >> K). The bootstrap path "
                f"inherits the same degeneracy on this data.",
                UserWarning,
                stacklevel=2,
            )
        _df_survey = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att,
            overall_se,
            alpha=self.alpha,
            df=_inference_df(_df_survey, resolved_survey),
        )

        # Joiners SE (uses joiner-only centered IF; conservative bound)
        if joiners_available:
            joiners_se, n_valid_joiners = _compute_se(
                U_centered=U_centered_joiners,
                divisor=joiner_total,
                obs_survey_info=_obs_survey_info,
                eligible_groups=_eligible_group_ids,
                U_centered_per_period=U_centered_pp_joiners,
            )
            if n_valid_joiners is not None:
                _replicate_n_valid_list.append(n_valid_joiners)
            _df_survey = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
            joiners_t, joiners_p, joiners_ci = safe_inference(
                joiners_att,
                joiners_se,
                alpha=self.alpha,
                df=_inference_df(_df_survey, resolved_survey),
            )
        else:
            joiners_se, joiners_t, joiners_p, joiners_ci = (
                float("nan"),
                float("nan"),
                float("nan"),
                (float("nan"), float("nan")),
            )

        # Leavers SE
        if leavers_available:
            leavers_se, n_valid_leavers = _compute_se(
                U_centered=U_centered_leavers,
                divisor=leaver_total,
                obs_survey_info=_obs_survey_info,
                eligible_groups=_eligible_group_ids,
                U_centered_per_period=U_centered_pp_leavers,
            )
            if n_valid_leavers is not None:
                _replicate_n_valid_list.append(n_valid_leavers)
            _df_survey = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
            leavers_t, leavers_p, leavers_ci = safe_inference(
                leavers_att,
                leavers_se,
                alpha=self.alpha,
                df=_inference_df(_df_survey, resolved_survey),
            )
        else:
            leavers_se, leavers_t, leavers_p, leavers_ci = (
                float("nan"),
                float("nan"),
                float("nan"),
                (float("nan"), float("nan")),
            )

        # Phase 1 per-period placebo (L_max=None): SE is NaN because the
        # per-period DID_M^pl aggregation path does not have an IF
        # derivation. Multi-horizon placebos (L_max >= 1) use the per-group
        # placebo IF computed above and have valid SE.
        placebo_se = float("nan")
        placebo_t = float("nan")
        placebo_p = float("nan")
        placebo_ci: Tuple[float, float] = (float("nan"), float("nan"))
        if placebo_available and L_max is None:
            warnings.warn(
                "Single-period placebo SE (L_max=None) is NaN. The "
                "per-period DID_M^pl aggregation path does not have an "
                "influence-function derivation. Use L_max >= 1 for "
                "multi-horizon placebos with valid SE. The placebo "
                "point estimate (results.placebo_effect) is still "
                "meaningful.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 18: Build per-period decomposition with explicit n_*_t fields
        # ------------------------------------------------------------------
        n_treated_obs_post = int(N_mat[D_mat == 1].sum())

        # ------------------------------------------------------------------
        # Step 19: Bootstrap if requested
        # ------------------------------------------------------------------
        bootstrap_results: Optional[DCDHBootstrapResults] = None
        if self.n_bootstrap > 0:
            if resolved_survey is not None:
                # Replicate-weight designs have their own closed-form
                # variance (Rao-Wu / BRR / Fay / JK1 / JKn / SDR via
                # compute_replicate_if_variance). Combining replicate
                # variance with a multiplier bootstrap would double-count
                # variance. Match library precedent:
                # efficient_did.py:989, staggered.py:1869,
                # two_stage.py:251-253.
                if (
                    resolved_survey.replicate_weights is not None
                    and resolved_survey.replicate_weights.shape[1] > 0
                ):
                    raise NotImplementedError(
                        "dCDH survey support rejects the combination of "
                        "replicate weights and n_bootstrap > 0. Replicate-"
                        "weight variance is closed-form "
                        "(compute_replicate_if_variance); use n_bootstrap=0. "
                        "For a bootstrap-based SE, use strata/PSU/FPC design "
                        "instead of replicate weights."
                    )

                # Warning fires only when PSU is **strictly coarser
                # than group** on an otherwise within-group-constant
                # design (multiple eligible groups share a PSU label
                # but no group spans more than one PSU). Two regimes
                # are explicitly excluded:
                # - PSU=group (auto-inject default): identity-map
                #   fast path — no warning needed.
                # - Within-group-varying PSU: the cell-level
                #   allocator honors the per-cell PSU structure;
                #   "n_psu < n_groups" is expected whenever cells
                #   of a group share a PSU with cells of another
                #   group, which does not indicate coarser-than-group
                #   clustering in the Hall-Mammen sense.
                # Count unique PSUs across ALL positive-weight obs of
                # eligible groups AND detect within-group-varying
                # PSU; suppress the warning in that regime.
                psu_arr_warn = getattr(resolved_survey, "psu", None)
                if psu_arr_warn is None or _obs_survey_info is None:
                    # No PSU info — can't compare to group count.
                    n_psu_eff_warn, n_groups_eff_warn = -1, -1
                    psu_varies_within_warn = False
                else:
                    obs_gids_warn = np.asarray(_obs_survey_info["group_ids"])
                    obs_ws_warn = np.asarray(_obs_survey_info["weights"], dtype=np.float64)
                    pos_mask_warn = obs_ws_warn > 0
                    psu_codes_warn = np.asarray(psu_arr_warn)
                    eligible_gid_set = set(_eligible_group_ids)
                    elig_obs_mask_warn = pos_mask_warn & np.array(
                        [g in eligible_gid_set for g in obs_gids_warn],
                        dtype=bool,
                    )
                    if elig_obs_mask_warn.any():
                        elig_psu_labels_arr = psu_codes_warn[elig_obs_mask_warn]
                        n_psu_eff_warn = int(len(np.unique(elig_psu_labels_arr)))
                        n_groups_eff_warn = len(_eligible_group_ids)
                        # Detect within-group-varying PSU on the
                        # eligible subset so we can suppress the
                        # "strictly coarser PSU" warning there.
                        psu_varies_within_warn = bool(
                            pd.DataFrame(
                                {
                                    "g": obs_gids_warn[elig_obs_mask_warn],
                                    "p": elig_psu_labels_arr,
                                }
                            )
                            .groupby("g")["p"]
                            .nunique()
                            .gt(1)
                            .any()
                        )
                    else:
                        n_psu_eff_warn, n_groups_eff_warn = -1, -1
                        psu_varies_within_warn = False
                if 0 <= n_psu_eff_warn < n_groups_eff_warn and not psu_varies_within_warn:
                    warnings.warn(
                        f"Bootstrap with survey_design uses Hall-Mammen "
                        f"wild multiplier weights at the PSU level "
                        f"(n_psu={n_psu_eff_warn} PSUs across "
                        f"n_groups={n_groups_eff_warn} groups). For "
                        f"designs with substantially unequal PSU sizes, "
                        f"the wild bootstrap may under-cover relative to "
                        f"analytical TSL inference; consider "
                        f"n_bootstrap=0 for the TSL variance. If n_psu "
                        f"is close to n_groups, lonely-PSU removal "
                        f"(lonely_psu='remove') may be collapsing "
                        f"singletons.",
                        UserWarning,
                        stacklevel=2,
                    )
            joiners_inputs = (
                (U_centered_joiners, joiner_total, joiners_att, U_centered_pp_joiners)
                if joiners_available
                else None
            )
            leavers_inputs = (
                (U_centered_leavers, leaver_total, leavers_att, U_centered_pp_leavers)
                if leavers_available
                else None
            )
            # Phase 1 placebo bootstrap: the Phase 1 per-period placebo
            # DID_M^pl still uses NaN SE (no IF derivation for the
            # per-period aggregation). The multi-horizon placebo bootstrap
            # below handles Phase 2+ placebos when placebo_horizon_if is
            # available.
            placebo_inputs = None

            # Phase 2: build placebo-horizon bootstrap inputs from the
            # cohort-centered placebo IF vectors.
            pl_boot_inputs = None
            if (
                placebo_horizon_if is not None
                and multi_horizon_placebos is not None
                and L_max is not None
                and L_max >= 1
            ):
                singleton_baseline_set_pl_b = set(singleton_baseline_groups)
                eligible_mask_pl_b = np.array(
                    [g not in singleton_baseline_set_pl_b for g in all_groups],
                    dtype=bool,
                )
                pl_boot_inputs = {}
                for lag_l in range(1, L_max + 1):
                    pl_data = multi_horizon_placebos.get(lag_l)
                    if pl_data is None or pl_data["N_pl_l"] == 0:
                        continue
                    U_pl_full, _ = placebo_horizon_if[lag_l]
                    U_pl_elig_b = U_pl_full[eligible_mask_pl_b]
                    cohort_keys_pl_b = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cpl_b: Dict[Tuple[float, int, int], int] = {}
                    cid_pl_b = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_pl_b[g]:
                            cid_pl_b[g] = -1
                            continue
                        key = cohort_keys_pl_b[g]
                        if key not in unique_cpl_b:
                            unique_cpl_b[key] = len(unique_cpl_b)
                        cid_pl_b[g] = unique_cpl_b[key]
                    cid_elig_pl_b = cid_pl_b[eligible_mask_pl_b]
                    U_centered_pl_b = _cohort_recenter(U_pl_elig_b, cid_elig_pl_b)
                    pl_boot_inputs[lag_l] = (
                        U_centered_pl_b,
                        pl_data["N_pl_l"],
                        pl_data["placebo_l"],
                        _pl_pp_cache.get(lag_l),
                    )

            # Phase 2: build multi-horizon bootstrap inputs from the
            # cohort-centered IF vectors computed in Step 12c.
            mh_boot_inputs = None
            if (
                multi_horizon_if is not None
                and multi_horizon_dids is not None
                and multi_horizon_se is not None
                and L_max is not None
                and L_max >= 1
            ):
                singleton_baseline_set_b = set(singleton_baseline_groups)
                eligible_mask_b = np.array(
                    [g not in singleton_baseline_set_b for g in all_groups], dtype=bool
                )
                mh_boot_inputs = {}
                # Include ALL horizons 1..L_max so the sup-t critical
                # value is calibrated over the same set that receives
                # cband_conf_int. For l=1, use the per-group IF (not
                # the Phase 1 per-period IF) so the bootstrap matches
                # the event_study_effects[1] estimand.
                for l_h in range(1, L_max + 1):
                    h_data = multi_horizon_dids.get(l_h)
                    if h_data is None or h_data["N_l"] == 0:
                        continue
                    U_l_full, _ = multi_horizon_if[l_h]
                    # Full variance-eligible group set (matching
                    # analytical SE path: singleton-baseline only)
                    U_l_elig = U_l_full[eligible_mask_b]
                    # Use the same cohort IDs as the analytical SE path
                    cohort_keys_b = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cb: Dict[Tuple[float, int, int], int] = {}
                    cid_b = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_b[g]:
                            cid_b[g] = -1
                            continue
                        key = cohort_keys_b[g]
                        if key not in unique_cb:
                            unique_cb[key] = len(unique_cb)
                        cid_b[g] = unique_cb[key]
                    cid_elig = cid_b[eligible_mask_b]
                    U_centered_h = _cohort_recenter(U_l_elig, cid_elig)
                    mh_boot_inputs[l_h] = (
                        U_centered_h,
                        h_data["N_l"],
                        h_data["did_l"],
                        _mh_pp_cache.get(l_h),
                    )

            # Under a survey design with PSU information, build both
            # (a) a group-level `group_id_to_psu_code` dict (one PSU
            # code per eligible group) and (b) a per-cell PSU tensor
            # `psu_codes_per_cell` of shape (n_eligible, n_periods).
            # The bootstrap mixin's dispatcher inspects (b) to decide
            # whether PSU is within-group-constant: when constant, it
            # runs the legacy group-level bootstrap via (a) for
            # bit-identity with pre-release behavior; when varying,
            # it switches to the cell-level wild PSU bootstrap that
            # draws one multiplier per (g, t)'s PSU. Under auto-inject
            # `psu=group` each group has a unique PSU code and every
            # cell of a group shares it — the dispatcher routes to
            # the legacy path and the identity-map fast path
            # reproduces the pre-PSU behavior bit-for-bit. See
            # REGISTRY.md ChaisemartinDHaultfoeuille Survey +
            # bootstrap contract Note.
            group_id_to_psu_code_bootstrap: Optional[Dict[Any, int]] = None
            eligible_group_ids_bootstrap: Optional[np.ndarray] = None
            psu_codes_per_cell_bootstrap: Optional[np.ndarray] = None
            if (
                resolved_survey is not None
                and getattr(resolved_survey, "psu", None) is not None
                and _obs_survey_info is not None
            ):
                obs_psu_codes = np.asarray(resolved_survey.psu)
                obs_gids_boot = np.asarray(_obs_survey_info["group_ids"])
                obs_tids_boot = np.asarray(_obs_survey_info["time_ids"])
                obs_weights_boot = np.asarray(_obs_survey_info["weights"], dtype=np.float64)
                pos_mask_boot = obs_weights_boot > 0
                gid_to_idx = {gid: i for i, gid in enumerate(_eligible_group_ids)}
                tid_to_idx = {t: i for i, t in enumerate(all_periods)}
                n_elig_boot = len(_eligible_group_ids)
                n_per_boot = len(all_periods)
                g_idx_arr = np.array(
                    [gid_to_idx.get(g, -1) for g in obs_gids_boot],
                    dtype=np.int64,
                )
                t_idx_arr = np.array(
                    [tid_to_idx.get(t, -1) for t in obs_tids_boot],
                    dtype=np.int64,
                )
                # Factor PSU labels to dense int codes over the
                # **eligible-subset** positive-weight observations only
                # (not the full positive-weight population). Restricting
                # to eligible obs ensures the resulting dense codes
                # range ONLY over PSUs actually used by variance-
                # eligible groups, so downstream n_psu = max(code) + 1
                # is exact: no gaps from singleton-baseline-excluded
                # groups that would silently trigger the identity
                # fast path in `_generate_psu_or_group_weights`.
                elig_obs_mask = pos_mask_boot & (g_idx_arr >= 0) & (t_idx_arr >= 0)
                elig_psu_labels = obs_psu_codes[elig_obs_mask]
                dense_per_row: Optional[np.ndarray] = None
                if elig_psu_labels.size > 0:
                    _, elig_dense_codes = np.unique(
                        elig_psu_labels,
                        return_inverse=True,
                    )
                    elig_dense_codes = np.asarray(elig_dense_codes, dtype=np.int64)
                    dense_per_row = np.full(
                        len(obs_psu_codes),
                        -1,
                        dtype=np.int64,
                    )
                    dense_per_row[elig_obs_mask] = elig_dense_codes

                # Per-cell PSU tensor: (n_eligible, n_periods), -1 sentinel
                # for ineligible / zero-weight cells. Populated
                # unconditionally when `dense_per_row` exists — a row
                # that ends up all-sentinel (eligible group with no
                # positive-weight obs) is masked out at unroll time,
                # not by discarding the entire tensor. See also the
                # dispatcher's `_psu_varies_within_group` helper which
                # ignores sentinel entries row-wise.
                if dense_per_row is not None:
                    psu_codes_per_cell = np.full(
                        (n_elig_boot, n_per_boot),
                        -1,
                        dtype=np.int64,
                    )
                    psu_codes_per_cell[
                        g_idx_arr[elig_obs_mask],
                        t_idx_arr[elig_obs_mask],
                    ] = dense_per_row[elig_obs_mask]
                    psu_codes_per_cell_bootstrap = psu_codes_per_cell

                    # Group-level dict: one PSU code per eligible
                    # group. For rows that are all-sentinel (eligible
                    # group has no positive-weight obs), assign code
                    # `0` as a harmless placeholder — the group's IF
                    # mass is zero, so the bootstrap multiplier it
                    # receives is irrelevant on either the legacy or
                    # the cell-level path. Always populate the dict
                    # so the legacy group-level path keeps clustering
                    # correctly when psu_varies=False even if some
                    # eligible groups happen to have no positive-
                    # weight obs.
                    group_psu_labels: List[int] = []
                    for i in range(n_elig_boot):
                        row = psu_codes_per_cell[i]
                        valid = row[row >= 0]
                        if valid.size == 0:
                            group_psu_labels.append(0)
                        else:
                            group_psu_labels.append(int(valid[0]))
                    group_id_to_psu_code_bootstrap = {
                        gid: code for gid, code in zip(_eligible_group_ids, group_psu_labels)
                    }
                    eligible_group_ids_bootstrap = np.asarray(_eligible_group_ids)

            # Collect per-(path, horizon) bootstrap inputs when by_path is
            # active. Uses the sibling helper to walk the same enumeration
            # / per-path IF / cohort-recentering pipeline that
            # `_compute_path_effects` uses (kept separate per the review
            # architectural preference — see `_collect_path_bootstrap_inputs`).
            path_bootstrap_inputs = None
            if (
                (self.by_path is not None or self.paths_of_interest is not None)
                and L_max is not None
                and L_max >= 1
                and multi_horizon_dids is not None
                and path_effects is not None
                and len(path_effects) > 0
            ):
                path_bootstrap_inputs = _collect_path_bootstrap_inputs(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    by_path=self.by_path,
                    paths_of_interest=self.paths_of_interest,
                    eligible_mask_var=eligible_mask_var,
                    multi_horizon_dids=multi_horizon_dids,
                    path_effects=path_effects,
                    set_ids=set_ids_arr,
                )

            # Sibling collector for per-path backward placebos. Mirrors
            # the path_bootstrap_inputs gating: only invoke when by_path
            # + placebo are both active, multi_horizon_placebos is
            # populated, and analytical path_placebos returned a non-
            # empty dict.
            path_placebo_bootstrap_inputs = None
            if (
                (self.by_path is not None or self.paths_of_interest is not None)
                and self.placebo
                and L_max is not None
                and L_max >= 1
                and multi_horizon_placebos is not None
                and path_placebos is not None
                and len(path_placebos) > 0
            ):
                path_placebo_bootstrap_inputs = _collect_path_placebo_bootstrap_inputs(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    by_path=self.by_path,
                    paths_of_interest=self.paths_of_interest,
                    eligible_mask_var=eligible_mask_var,
                    multi_horizon_placebos=multi_horizon_placebos,
                    path_placebos=path_placebos,
                    set_ids=set_ids_arr,
                )

            br = self._compute_dcdh_bootstrap(
                n_groups_for_overall=n_groups_for_overall_var,
                u_centered_overall=U_centered_overall,
                divisor_overall=N_S,
                original_overall=overall_att,
                joiners_inputs=joiners_inputs,
                leavers_inputs=leavers_inputs,
                placebo_inputs=placebo_inputs,
                multi_horizon_inputs=mh_boot_inputs,
                placebo_horizon_inputs=pl_boot_inputs,
                path_bootstrap_inputs=path_bootstrap_inputs,
                path_placebo_bootstrap_inputs=path_placebo_bootstrap_inputs,
                group_id_to_psu_code=group_id_to_psu_code_bootstrap,
                eligible_group_ids=eligible_group_ids_bootstrap,
                u_per_period_overall=U_centered_pp_overall,
                psu_codes_per_cell=psu_codes_per_cell_bootstrap,
            )
            bootstrap_results = br

            # Replace the analytical SE with the bootstrap SE for the
            # targets that have valid bootstrap output, AND propagate
            # the bootstrap percentile p-value and CI directly to the
            # top-level fields. The t-stat is computed from the SE via
            # safe_inference()[0] so the project anti-pattern rule
            # (never compute t_stat = effect / se inline) stays
            # satisfied — bootstrap does not define an alternative
            # t-stat semantic for percentile bootstrap, so the
            # SE-based t-stat is the natural choice.
            #
            # Library precedent: imputation.py:790-805,
            # two_stage.py:778-787, and efficient_did.py:1009-1013 all
            # propagate bootstrap p/CI to the public surface while
            # keeping a SE-derived t-stat. Round 10 brings dCDH in line
            # with that pattern (the prior code silently recomputed
            # normal-theory p/CI from the bootstrap SE, which made the
            # public inference surface a hybrid).
            #
            # See REGISTRY.md ChaisemartinDHaultfoeuille `Note
            # (bootstrap inference surface)` and the regression test
            # ``test_bootstrap_p_value_and_ci_propagated_to_top_level``.
            # Bootstrap contract: once the caller opts into n_bootstrap > 0,
            # bootstrap SE / percentile CI / percentile p-value replace the
            # analytical values. When the bootstrap SE comes back non-finite
            # (e.g., n_bootstrap too small, degenerate bootstrap distribution,
            # zero-IF target), the full inference tuple goes to NaN rather
            # than silently falling back to analytical — mixing bootstrap-
            # contract and analytical-contract semantics within one result
            # object would be a public-surface inconsistency. Same treatment
            # applies to the event_study_effects propagation below and the
            # path_effects propagation further down.
            if np.isfinite(br.overall_se):
                overall_se = br.overall_se
                overall_p = br.overall_p_value if br.overall_p_value is not None else np.nan
                overall_ci = br.overall_ci if br.overall_ci is not None else (np.nan, np.nan)
                overall_t = safe_inference(
                    overall_att,
                    overall_se,
                    alpha=self.alpha,
                    df=_inference_df(_df_survey, resolved_survey),
                )[0]
            else:
                overall_se = np.nan
                overall_p = np.nan
                overall_ci = (np.nan, np.nan)
                overall_t = np.nan
            if joiners_available:
                if br.joiners_se is not None and np.isfinite(br.joiners_se):
                    joiners_se = br.joiners_se
                    joiners_p = br.joiners_p_value if br.joiners_p_value is not None else np.nan
                    joiners_ci = br.joiners_ci if br.joiners_ci is not None else (np.nan, np.nan)
                    joiners_t = safe_inference(
                        joiners_att,
                        joiners_se,
                        alpha=self.alpha,
                        df=_inference_df(_df_survey, resolved_survey),
                    )[0]
                else:
                    joiners_se = np.nan
                    joiners_p = np.nan
                    joiners_ci = (np.nan, np.nan)
                    joiners_t = np.nan
            if leavers_available:
                if br.leavers_se is not None and np.isfinite(br.leavers_se):
                    leavers_se = br.leavers_se
                    leavers_p = br.leavers_p_value if br.leavers_p_value is not None else np.nan
                    leavers_ci = br.leavers_ci if br.leavers_ci is not None else (np.nan, np.nan)
                    leavers_t = safe_inference(
                        leavers_att,
                        leavers_se,
                        alpha=self.alpha,
                        df=_inference_df(_df_survey, resolved_survey),
                    )[0]
                else:
                    leavers_se = np.nan
                    leavers_p = np.nan
                    leavers_ci = (np.nan, np.nan)
                    leavers_t = np.nan

        # ------------------------------------------------------------------
        # Step 20: Build the results dataclass
        # ------------------------------------------------------------------
        # event_study_effects: when L_max is None, l=1 mirrors Phase 1
        # DID_M (per-period path). When L_max >= 1, ALL horizons including
        # l=1 use the per-group DID_{g,l} path for a consistent estimand.
        if multi_horizon_inference is not None and 1 in multi_horizon_inference:
            # Per-group mode: use per-group path for all horizons.
            # When L_max >= 1, the per-group DID_{g,1} is the correct
            # estimand for overall_att (not the binary-only per-period
            # DID_M). This handles both pure non-binary (N_S=0) and
            # mixed binary/non-binary panels (N_S > 0 but incomplete).
            l1_inf = multi_horizon_inference[1]
            overall_att = l1_inf["effect"]
            overall_se = l1_inf["se"]
            overall_t = l1_inf["t_stat"]
            overall_p = l1_inf["p_value"]
            overall_ci = l1_inf["conf_int"]
            event_study_effects: Dict[int, Dict[str, Any]] = dict(multi_horizon_inference)
        else:
            # Phase 1 mode (L_max=None): l=1 from per-period path
            event_study_effects = {
                1: {
                    "effect": overall_att,
                    "se": overall_se,
                    "t_stat": overall_t,
                    "p_value": overall_p,
                    "conf_int": overall_ci,
                    "n_obs": N_S,
                }
            }

        # Phase 2: propagate bootstrap results to event_study_effects.
        # Same bootstrap-contract rule as the overall/joiners/leavers block
        # above and the path_effects block below: non-finite bootstrap SE
        # writes NaN to the full inference tuple rather than falling back
        # to analytical.
        if bootstrap_results is not None and bootstrap_results.event_study_ses:
            for l_h in bootstrap_results.event_study_ses:
                if l_h in event_study_effects:
                    bs_se = bootstrap_results.event_study_ses.get(l_h)
                    bs_ci = (
                        bootstrap_results.event_study_cis.get(l_h)
                        if bootstrap_results.event_study_cis
                        else None
                    )
                    bs_p = (
                        bootstrap_results.event_study_p_values.get(l_h)
                        if bootstrap_results.event_study_p_values
                        else None
                    )
                    eff = event_study_effects[l_h]["effect"]
                    if bs_se is not None and np.isfinite(bs_se):
                        event_study_effects[l_h]["se"] = bs_se
                        event_study_effects[l_h]["p_value"] = bs_p if bs_p is not None else np.nan
                        event_study_effects[l_h]["conf_int"] = (
                            bs_ci if bs_ci is not None else (np.nan, np.nan)
                        )
                        event_study_effects[l_h]["t_stat"] = safe_inference(
                            eff, bs_se, alpha=self.alpha, df=None
                        )[0]
                    else:
                        event_study_effects[l_h]["se"] = np.nan
                        event_study_effects[l_h]["p_value"] = np.nan
                        event_study_effects[l_h]["conf_int"] = (np.nan, np.nan)
                        event_study_effects[l_h]["t_stat"] = np.nan

            # Add sup-t bands to event_study_effects entries
            if bootstrap_results.cband_crit_value is not None:
                crit = bootstrap_results.cband_crit_value
                for l_h in event_study_effects:
                    se = event_study_effects[l_h]["se"]
                    eff = event_study_effects[l_h]["effect"]
                    if np.isfinite(se) and se > 0:
                        event_study_effects[l_h]["cband_conf_int"] = (
                            eff - crit * se,
                            eff + crit * se,
                        )

        # Phase 3: propagate bootstrap results to path_effects (by_path).
        # Mirrors the event_study propagation above: replace the analytical
        # SE / p-value / CI with the bootstrap percentile statistics
        # (Round-10 library convention — `br.path_p_values` and
        # `br.path_cis` are already percentile-based via
        # `compute_effect_bootstrap_stats`), and re-derive the t-stat
        # from the bootstrap SE via `safe_inference` per the anti-pattern
        # rule. Point estimates (`effect`, `n_obs`, `n_groups`,
        # `frequency_rank`) are unchanged from the analytical path.
        if (
            bootstrap_results is not None
            and bootstrap_results.path_ses
            and path_effects is not None
        ):
            for path_key, horizon_ses in bootstrap_results.path_ses.items():
                if path_key not in path_effects:
                    continue
                for l_h, bs_se in horizon_ses.items():
                    if l_h not in path_effects[path_key]["horizons"]:
                        continue
                    bs_ci = (
                        bootstrap_results.path_cis.get(path_key, {}).get(l_h)
                        if bootstrap_results.path_cis
                        else None
                    )
                    bs_p = (
                        bootstrap_results.path_p_values.get(path_key, {}).get(l_h)
                        if bootstrap_results.path_p_values
                        else None
                    )
                    # Bootstrap replaces analytical inference for
                    # this (path, horizon) regardless of outcome. If
                    # the bootstrap SE is non-finite (e.g., n_bootstrap
                    # too small, degenerate bootstrap distribution, or
                    # zero-IF path inherited from the analytical
                    # degenerate-cohort branch), the full inference
                    # tuple goes to NaN — we must NOT fall back to
                    # analytical inference here, since the caller
                    # explicitly chose the bootstrap path by setting
                    # n_bootstrap > 0. Falling back would silently mix
                    # bootstrap-contract semantics with analytical-
                    # contract semantics within the same result
                    # object.
                    eff_p = path_effects[path_key]["horizons"][l_h]["effect"]
                    if bs_se is not None and np.isfinite(bs_se):
                        path_effects[path_key]["horizons"][l_h]["se"] = bs_se
                        path_effects[path_key]["horizons"][l_h]["p_value"] = (
                            bs_p if bs_p is not None else np.nan
                        )
                        path_effects[path_key]["horizons"][l_h]["conf_int"] = (
                            bs_ci if bs_ci is not None else (np.nan, np.nan)
                        )
                        path_effects[path_key]["horizons"][l_h]["t_stat"] = safe_inference(
                            eff_p, bs_se, alpha=self.alpha, df=None
                        )[0]
                    else:
                        path_effects[path_key]["horizons"][l_h]["se"] = np.nan
                        path_effects[path_key]["horizons"][l_h]["p_value"] = np.nan
                        path_effects[path_key]["horizons"][l_h]["conf_int"] = (
                            np.nan,
                            np.nan,
                        )
                        path_effects[path_key]["horizons"][l_h]["t_stat"] = np.nan

        # Per-path cumulated layer (under trends_linear). Computed AFTER
        # the bootstrap propagation block above so the cumulated SE / t /
        # p / CI are derived from the FINAL post-bootstrap per-horizon
        # path SEs rather than the analytical ones path_effects was
        # initially populated with at fit-time. Mirrors the global
        # `linear_trends_effects` placement at `:3405-3454` which also
        # runs after the event_study bootstrap propagation. Honors the
        # library-wide NaN-on-invalid bootstrap contract: any non-finite
        # component SE in the running-sum upper bound yields a NaN
        # cumulated SE / t / p / CI, regardless of whether the source
        # was an analytical singularity or a non-finite bootstrap draw.
        if (
            (self.by_path is not None or self.paths_of_interest is not None)
            and _is_trends_linear
            and L_max is not None
            and L_max >= 1
            and multi_horizon_dids is not None
            and path_effects is not None
            and len(path_effects) > 0
        ):
            _df_s_bp_cum = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
            path_cumulated_event_study = _compute_path_cumulated_event_study(
                D_mat=D_mat,
                N_mat=N_mat,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                L_max=L_max,
                by_path=self.by_path,
                paths_of_interest=self.paths_of_interest,
                multi_horizon_dids=multi_horizon_dids,
                path_effects=path_effects,
                alpha=self.alpha,
                df_inference=_inference_df(_df_s_bp_cum, resolved_survey),
            )

        # Phase 3: propagate bootstrap results to per-path placebos
        # (by_path + placebo). Sibling of the path_effects propagation
        # block above. Library-wide NaN-on-invalid bootstrap contract:
        # non-finite bootstrap SE writes NaN to the full inference
        # tuple rather than falling back to the analytical SE -- the
        # caller opted into bootstrap by setting n_bootstrap > 0, and
        # mixing analytical + bootstrap semantics inside one result
        # object is a public-surface inconsistency.
        if (
            bootstrap_results is not None
            and bootstrap_results.path_placebo_ses
            and path_placebos is not None
        ):
            for path_key, lag_ses in bootstrap_results.path_placebo_ses.items():
                if path_key not in path_placebos:
                    continue
                for lag_l, bs_se_pl in lag_ses.items():
                    neg_key = -lag_l
                    if neg_key not in path_placebos[path_key]:
                        continue
                    bs_ci_pl = (
                        bootstrap_results.path_placebo_cis.get(path_key, {}).get(lag_l)
                        if bootstrap_results.path_placebo_cis
                        else None
                    )
                    bs_p_pl = (
                        bootstrap_results.path_placebo_p_values.get(path_key, {}).get(lag_l)
                        if bootstrap_results.path_placebo_p_values
                        else None
                    )
                    eff_pl = path_placebos[path_key][neg_key]["effect"]
                    if bs_se_pl is not None and np.isfinite(bs_se_pl):
                        path_placebos[path_key][neg_key]["se"] = bs_se_pl
                        path_placebos[path_key][neg_key]["p_value"] = (
                            bs_p_pl if bs_p_pl is not None else np.nan
                        )
                        path_placebos[path_key][neg_key]["conf_int"] = (
                            bs_ci_pl if bs_ci_pl is not None else (np.nan, np.nan)
                        )
                        path_placebos[path_key][neg_key]["t_stat"] = safe_inference(
                            eff_pl, bs_se_pl, alpha=self.alpha, df=None
                        )[0]
                    else:
                        path_placebos[path_key][neg_key]["se"] = np.nan
                        path_placebos[path_key][neg_key]["p_value"] = np.nan
                        path_placebos[path_key][neg_key]["conf_int"] = (np.nan, np.nan)
                        path_placebos[path_key][neg_key]["t_stat"] = np.nan

        # Phase 3: propagate per-path sup-t critical values to per-
        # horizon `cband_conf_int` entries on path_effects (by_path +
        # n_bootstrap > 0). Sibling of the OVERALL event-study cband
        # propagation at `:2865-2875`. For each path with a finite
        # crit, write `cband_conf_int = (eff - c_p*se, eff + c_p*se)`
        # into each horizon's dict whose bootstrap-replaced SE is
        # finite > 0. Mirror the OVERALL absent-key pattern: non-finite
        # SE horizons simply don't get the `cband_conf_int` key.
        if (
            bootstrap_results is not None
            and bootstrap_results.path_cband_crit_values is not None
            and path_effects is not None
        ):
            for path_key, crit in bootstrap_results.path_cband_crit_values.items():
                if path_key not in path_effects:
                    continue
                if not np.isfinite(crit):
                    continue
                for l_h, h_dict in path_effects[path_key]["horizons"].items():
                    se = h_dict.get("se", np.nan)
                    eff = h_dict.get("effect", np.nan)
                    if np.isfinite(se) and se > 0:
                        h_dict["cband_conf_int"] = (
                            eff - crit * se,
                            eff + crit * se,
                        )

        # When L_max >= 1 and the per-group path is active, sync
        # overall_* from event_study_effects[1] AFTER bootstrap propagation
        # so that bootstrap SE/p/CI flow to the top-level surface.
        if L_max is not None and L_max >= 1 and 1 in event_study_effects:
            es1 = event_study_effects[1]
            overall_att = es1["effect"]
            overall_se = es1["se"]
            overall_t = es1["t_stat"]
            overall_p = es1["p_value"]
            overall_ci = es1["conf_int"]
            # Sync nested bootstrap_results.overall_* to DID_1 only when
            # L_max == 1. When L_max >= 2, the cost-benefit delta overrides
            # overall_* later, so bootstrap_results.overall_* should stay
            # on the scalar DID_M bootstrap (or be overridden by delta logic).
            if (
                L_max == 1
                and bootstrap_results is not None
                and bootstrap_results.event_study_ses
                and 1 in bootstrap_results.event_study_ses
            ):
                bootstrap_results.overall_se = bootstrap_results.event_study_ses[1]
                bootstrap_results.overall_ci = (
                    bootstrap_results.event_study_cis[1]
                    if bootstrap_results.event_study_cis and 1 in bootstrap_results.event_study_cis
                    else (np.nan, np.nan)
                )
                # Clear the DID_M distribution - it doesn't match the
                # DID_1 summary statistics. The per-horizon bootstrap
                # stats are accessible via event_study_ses/cis/p_values.
                bootstrap_results.bootstrap_distribution = None
                bootstrap_results.overall_p_value = (
                    bootstrap_results.event_study_p_values[1]
                    if bootstrap_results.event_study_p_values
                    and 1 in bootstrap_results.event_study_p_values
                    else np.nan
                )

        # Phase 2: override overall_att with cost-benefit delta when L_max > 1
        effective_overall_att = overall_att
        effective_overall_se = overall_se
        effective_overall_t = overall_t
        effective_overall_p = overall_p
        effective_overall_ci = overall_ci
        if cost_benefit_result is not None and L_max is not None and L_max >= 2:
            delta_val = cost_benefit_result["delta"]
            if not np.isfinite(delta_val):
                # Delta is non-estimable (e.g., no eligible switchers at
                # any horizon). Set all overall_* to NaN rather than
                # silently falling back to the Phase 1 DID_M values,
                # since the results surface labels them as delta.
                effective_overall_att = float("nan")
                effective_overall_se = float("nan")
                effective_overall_t = float("nan")
                effective_overall_p = float("nan")
                effective_overall_ci = (float("nan"), float("nan"))
            else:
                effective_overall_att = delta_val
                # Cost-benefit delta SE: compute from per-horizon bootstrap
                # distributions if available (delta = sum w_l * DID_l, so
                # delta_b = sum w_l * DID_l_b for each bootstrap rep).
                # Delta-method SE: Var(delta) = sum w_l^2 * Var(DID_l)
                # (treating horizons as independent, conservative under
                # Assumption 8). Works on both analytical and bootstrap
                # SEs since event_study_effects[l]["se"] holds whichever
                # was propagated.
                # Require ALL positively-weighted horizons to have finite
                # SE. If any has NaN, delta SE is NaN (NaN-consistent
                # inference contract: no partial aggregation).
                weights = cost_benefit_result.get("weights", {})
                var_delta = 0.0
                all_finite = True
                for l_w, w_l in weights.items():
                    if w_l <= 0:
                        continue
                    se_l = event_study_effects.get(l_w, {}).get("se", float("nan"))
                    if not np.isfinite(se_l):
                        all_finite = False
                        break
                    var_delta += (w_l * se_l) ** 2
                delta_se = (
                    float(np.sqrt(var_delta)) if all_finite and var_delta > 0 else float("nan")
                )

                if np.isfinite(delta_se):
                    effective_overall_se = delta_se
                    effective_overall_t, effective_overall_p, effective_overall_ci = safe_inference(
                        delta_val,
                        delta_se,
                        alpha=self.alpha,
                        df=_inference_df(_df_survey, resolved_survey),
                    )
                else:
                    effective_overall_se = float("nan")
                    effective_overall_t = float("nan")
                    effective_overall_p = float("nan")
                    effective_overall_ci = (float("nan"), float("nan"))

        # Phase 2: build placebo_event_study with negative keys.
        # Use analytical SE from placebo IF (computed above), with
        # bootstrap override when available.
        placebo_event_study_dict: Optional[Dict[int, Dict[str, Any]]] = None
        if multi_horizon_placebos is not None:
            placebo_event_study_dict = {}
            for lag_l, pl_data in multi_horizon_placebos.items():
                if pl_data["N_pl_l"] > 0:
                    # Pull analytical SE from placebo IF computation
                    if placebo_horizon_inference is not None and lag_l in placebo_horizon_inference:
                        inf = placebo_horizon_inference[lag_l]
                        placebo_event_study_dict[-lag_l] = {
                            "effect": inf["effect"],
                            "se": inf["se"],
                            "t_stat": inf["t_stat"],
                            "p_value": inf["p_value"],
                            "conf_int": inf["conf_int"],
                            "n_obs": inf["n_obs"],
                        }
                    else:
                        # Fallback: NaN SE (Phase 1 path or missing IF)
                        pl_se = float("nan")
                        pl_t, pl_p, pl_ci = safe_inference(
                            pl_data["placebo_l"],
                            pl_se,
                            alpha=self.alpha,
                            df=_inference_df(_df_survey, resolved_survey),
                        )
                        placebo_event_study_dict[-lag_l] = {
                            "effect": pl_data["placebo_l"],
                            "se": pl_se,
                            "t_stat": pl_t,
                            "p_value": pl_p,
                            "conf_int": pl_ci,
                            "n_obs": pl_data["N_pl_l"],
                        }
                else:
                    placebo_event_study_dict[-lag_l] = {
                        "effect": float("nan"),
                        "se": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "conf_int": (float("nan"), float("nan")),
                        "n_obs": 0,
                    }

        # Propagate bootstrap results to placebo_event_study (must run
        # after placebo_event_study_dict is assembled above).
        if (
            bootstrap_results is not None
            and bootstrap_results.placebo_horizon_ses
            and placebo_event_study_dict is not None
        ):
            for lag_l in bootstrap_results.placebo_horizon_ses:
                neg_key = -lag_l
                if neg_key in placebo_event_study_dict:
                    bs_se = bootstrap_results.placebo_horizon_ses.get(lag_l)
                    bs_ci = (
                        bootstrap_results.placebo_horizon_cis.get(lag_l)
                        if bootstrap_results.placebo_horizon_cis
                        else None
                    )
                    bs_p = (
                        bootstrap_results.placebo_horizon_p_values.get(lag_l)
                        if bootstrap_results.placebo_horizon_p_values
                        else None
                    )
                    # Same bootstrap-contract rule as overall / joiners /
                    # leavers / event_study_effects / path_effects above:
                    # once the caller opts into n_bootstrap > 0, the
                    # bootstrap output replaces analytical inference on
                    # this surface regardless of outcome. Non-finite
                    # bootstrap SE writes NaN to the full inference tuple
                    # rather than silently leaving analytical values in
                    # place — that would mix bootstrap-contract and
                    # analytical-contract semantics in the same rendered
                    # output (dynamic placebo rows appear in
                    # `results.to_dataframe(level="event_study")` alongside
                    # positive-horizon entries).
                    eff = placebo_event_study_dict[neg_key]["effect"]
                    if bs_se is not None and np.isfinite(bs_se):
                        placebo_event_study_dict[neg_key]["se"] = bs_se
                        placebo_event_study_dict[neg_key]["p_value"] = (
                            bs_p if bs_p is not None else np.nan
                        )
                        placebo_event_study_dict[neg_key]["conf_int"] = (
                            bs_ci if bs_ci is not None else (np.nan, np.nan)
                        )
                        placebo_event_study_dict[neg_key]["t_stat"] = safe_inference(
                            eff,
                            bs_se,
                            alpha=self.alpha,
                            df=_inference_df(_df_survey, resolved_survey),
                        )[0]
                    else:
                        placebo_event_study_dict[neg_key]["se"] = np.nan
                        placebo_event_study_dict[neg_key]["p_value"] = np.nan
                        placebo_event_study_dict[neg_key]["conf_int"] = (
                            np.nan,
                            np.nan,
                        )
                        placebo_event_study_dict[neg_key]["t_stat"] = np.nan

        # Phase 2: build normalized_effects with SE
        normalized_effects_out: Optional[Dict[int, Dict[str, Any]]] = None
        if normalized_effects_dict is not None and multi_horizon_se is not None:
            normalized_effects_out = {}
            for l_h, n_data in normalized_effects_dict.items():
                denom = n_data["denominator"]
                eff = n_data["effect"]
                # SE via delta method: SE(DID^n_l) = SE(DID_l) / delta^D_l
                se_did_l = multi_horizon_se.get(l_h, float("nan"))
                se_norm = se_did_l / denom if np.isfinite(denom) and denom > 0 else float("nan")
                t_n, p_n, ci_n = safe_inference(
                    eff, se_norm, alpha=self.alpha, df=_inference_df(_df_survey, resolved_survey)
                )
                normalized_effects_out[l_h] = {
                    "effect": eff,
                    "se": se_norm,
                    "t_stat": t_n,
                    "p_value": p_n,
                    "conf_int": ci_n,
                    "denominator": denom,
                }

        # ------------------------------------------------------------------
        # DID^{fd} cumulation: recover level effects from second-differences
        #
        # DID^{fd}_l identifies delta_{g,l} - delta_{g,l-1} (Lemma 6).
        # Cumulate per-group: for each group eligible at horizon l,
        # sum DID^{fd}_{g,l'} for l'=1..l, then average over that
        # eligible set. This matches R's did_multiplegt_dyn which
        # cumulates per-group then aggregates (NOT sum-of-aggregates,
        # which mixes different eligible populations).
        # ------------------------------------------------------------------
        if _is_trends_linear and multi_horizon_dids is not None:
            cumulated = {}
            n_groups_total = D_mat.shape[0]
            # Accumulate per-group running sum of DID^{fd}_{g,l'}
            running_per_group = np.zeros(n_groups_total)
            for l_h in range(1, (L_max or 0) + 1):
                if l_h not in multi_horizon_dids:
                    continue
                mh = multi_horizon_dids[l_h]
                did_g_l = mh["did_g_l"]  # (n_groups,) per-group DID
                eligible = mh["eligible_mask"]  # (n_groups,) bool
                N_l = mh["N_l"]
                if N_l == 0:
                    continue
                # Add this horizon's per-group DID to running sum
                # (NaN for ineligible groups; use 0 for accumulation)
                increment = np.where(np.isfinite(did_g_l), did_g_l, 0.0)
                running_per_group += increment
                # Average the cumulated sum over groups eligible at THIS horizon
                # Weight by S_g (switch direction) and divide by N_l
                S_arr = switch_direction_arr.astype(float)
                cum_effect = float(np.sum(S_arr[eligible] * running_per_group[eligible]) / N_l)
                # SE: conservative upper bound (sum of per-horizon SEs).
                # NaN-consistency: if ANY component SE up to horizon l is
                # non-finite, the cumulated SE is NaN (not 0.0).
                if event_study_effects is not None:
                    component_ses = [
                        event_study_effects.get(ll, {}).get("se", np.nan)
                        for ll in range(1, l_h + 1)
                    ]
                    if all(np.isfinite(s) for s in component_ses):
                        running_se_ub = sum(component_ses)
                    else:
                        running_se_ub = float("nan")
                else:
                    running_se_ub = float("nan")
                cum_t, cum_p, cum_ci = safe_inference(
                    cum_effect,
                    running_se_ub,
                    alpha=self.alpha,
                    df=_inference_df(_df_survey, resolved_survey),
                )
                cumulated[l_h] = {
                    "effect": cum_effect,
                    "se": running_se_ub,
                    "t_stat": cum_t,
                    "p_value": cum_p,
                    "conf_int": cum_ci,
                }
            linear_trends_effects = cumulated if cumulated else None

        # When trends_linear=True and L_max>=2, suppress cost_benefit_delta
        # and NaN out the overall_* surface. R's did_multiplegt_dyn with
        # trends_lin=TRUE does not compute an aggregate "average total
        # effect" - users should access cumulated level effects via
        # results.linear_trends_effects[l] instead.
        if _is_trends_linear and L_max is not None and L_max >= 2:
            cost_benefit_result = None
            effective_overall_att = float("nan")
            effective_overall_se = float("nan")
            effective_overall_t = float("nan")
            effective_overall_p = float("nan")
            effective_overall_ci = (float("nan"), float("nan"))

        # ------------------------------------------------------------------
        # Heterogeneity testing (Web Appendix Section 1.5, Lemma 7)
        # ------------------------------------------------------------------
        heterogeneity_effects: Optional[Dict[int, Dict[str, Any]]] = None
        if heterogeneity is not None:
            if L_max is None:
                raise ValueError(
                    "heterogeneity testing requires L_max >= 1. Set L_max "
                    "to use the per-group DID_{g,l} path."
                )
            het_col = str(heterogeneity)
            if het_col not in data.columns:
                raise ValueError(f"heterogeneity column {het_col!r} not found in data.")
            # R's predict_het disallows controls; our partial implementation
            # follows this restriction to avoid inconsistent behavior.
            if controls is not None:
                raise ValueError(
                    "heterogeneity cannot be combined with covariates. "
                    "R's did_multiplegt_dyn disallows predict_het with "
                    "controls; remove one of the two options."
                )
            if _is_trends_linear:
                raise ValueError(
                    "heterogeneity cannot be combined with trends_linear. "
                    "The heterogeneity test operates on level outcome "
                    "changes but trends_linear uses second-differenced "
                    "outcomes; the results would be inconsistent."
                )
            if trends_nonparam is not None:
                raise ValueError(
                    "heterogeneity cannot be combined with trends_nonparam. "
                    "The heterogeneity test does not thread state-set "
                    "control-pool restrictions; the results would be "
                    "inconsistent with the fitted estimator."
                )
            # Extract per-group covariate (must be time-invariant).
            # SurveyDesign.subpopulation() contract: scope time-invariance
            # check to positive-weight rows so excluded obs with NaN/varying
            # het values do not abort the fit.
            if survey_weights is not None:
                pos_mask_het = np.asarray(survey_weights) > 0
                data_het = data.loc[pos_mask_het]
            else:
                data_het = data
            het_per_group = data_het.groupby(group)[het_col].nunique()
            het_varying = het_per_group[het_per_group > 1]
            if len(het_varying) > 0:
                raise ValueError(
                    f"heterogeneity column {het_col!r} must be "
                    f"time-invariant within each group. "
                    f"{len(het_varying)} group(s) have varying values."
                )
            het_map = data_het.groupby(group)[het_col].first()
            X_het = np.array([float(het_map.loc[g]) for g in all_groups])
            # Use original Y_mat (not first-differenced) for heterogeneity
            # test, since it operates on level differences Y[out] - Y[ref].
            # When trends_linear, the DID^{fd} second-differences are in
            # event_study_effects but the het test uses level outcomes.
            Y_het = Y_mat if not _is_trends_linear else y_pivot.to_numpy()
            N_het = N_mat_orig
            # Survey + placebo + heterogeneity: backward-horizon
            # (placebo) predict_het is deferred under survey designs
            # because the Binder TSL cell-period allocator's REGISTRY
            # justification is tied to post-period attribution
            # (pre-period cell allocator derivation gap). Compute
            # forward-horizon heterogeneity only and warn so the user
            # knows backward-horizon results are NOT silently produced.
            _het_placebo = L_max if self.placebo else 0
            if _het_placebo > 0 and _obs_survey_info is not None:
                warnings.warn(
                    "survey_design + heterogeneity + placebo=True: "
                    "backward-horizon (placebo) predict_het is not yet "
                    "supported under survey designs (pre-period cell "
                    "allocator deferred — see REGISTRY.md "
                    "ChaisemartinDHaultfoeuille placebo predict_het "
                    "Note). Forward-horizon predict_het is computed "
                    "normally. Drop placebo, drop heterogeneity, or "
                    "drop survey_design to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
                _het_placebo = 0
            heterogeneity_effects = _compute_heterogeneity_test(
                Y_mat=Y_het,
                N_mat=N_het,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                X_het=X_het,
                L_max=L_max,
                placebo=_het_placebo,
                alpha=self.alpha,
                rank_deficient_action=self.rank_deficient_action,
                group_ids_order=np.array(all_groups),
                obs_survey_info=_obs_survey_info,
                replicate_n_valid_list=_replicate_n_valid_list,
            )

        # Per-path heterogeneity (mirrors R `did_multiplegt_dyn(...,
        # by_path, predict_het)` per-by_level dispatch). Empty-state
        # contract: None when not requested (no `heterogeneity` kwarg
        # or no `by_path`/`paths_of_interest` selector); `{}` when
        # requested but no path is observed (mirrors `path_effects`).
        path_heterogeneity_effects: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = (
            None
        )
        if heterogeneity is not None and (
            self.by_path is not None or self.paths_of_interest is not None
        ):
            # Per-path placebo predict_het under survey: same gating
            # rationale as the global call above. The global call's
            # warning already fired (heterogeneity is computed at both
            # surfaces in the same fit() flow), so per-path skips
            # silently — no duplicate warning. Forward-horizon per-path
            # predict_het + survey continues to work via the existing
            # _compute_path_heterogeneity_test → _compute_heterogeneity_test
            # forward path.
            _path_het_placebo = L_max if (self.placebo and L_max is not None) else 0
            if _path_het_placebo > 0 and _obs_survey_info is not None:
                _path_het_placebo = 0
            path_heterogeneity_effects = _compute_path_heterogeneity_test(
                Y_mat=Y_het,
                N_mat=N_het,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                X_het=X_het,
                L_max=L_max,
                by_path=self.by_path,
                paths_of_interest=self.paths_of_interest,
                D_mat=D_mat,
                placebo=_path_het_placebo,
                alpha=self.alpha,
                rank_deficient_action=self.rank_deficient_action,
                group_ids_order=np.array(all_groups),
                obs_survey_info=_obs_survey_info,
                replicate_n_valid_list=_replicate_n_valid_list,
            )

        twfe_weights_df = None
        twfe_fraction_negative = None
        twfe_sigma_fe = None
        twfe_beta_fe = None
        if twfe_diagnostic_payload is not None:
            twfe_weights_df = twfe_diagnostic_payload.weights
            twfe_fraction_negative = twfe_diagnostic_payload.fraction_negative
            twfe_sigma_fe = twfe_diagnostic_payload.sigma_fe
            twfe_beta_fe = twfe_diagnostic_payload.beta_fe

        # When L_max >= 1, the overall estimand is per-group DID_1
        # (not per-period DID_M). The joiner/leaver decomposition is a
        # per-period DID_M concept and can differ from DID_1 on mixed
        # panels, so it's suppressed for all L_max >= 1 cases. N_S and
        # n_treated_obs are updated from the per-group path.
        effective_N_S = N_S
        effective_n_treated = n_treated_obs_post
        effective_joiners_available = joiners_available
        effective_leavers_available = leavers_available
        if (
            L_max is not None
            and L_max >= 1
            and multi_horizon_dids is not None
            and 1 in multi_horizon_dids
        ):
            # Use horizon-1 eligible switcher count as the effective N_S
            effective_N_S = multi_horizon_dids[1]["N_l"]
            if not is_binary:
                # For non-binary: count all observations where treatment
                # differs from baseline
                effective_n_treated = (
                    int(N_mat[D_mat != D_mat[:, 0:1]].sum()) if D_mat.shape[1] > 1 else 0
                )
            if not is_binary:
                # Suppress binary-only Phase 1 artifacts on non-binary
                # panels: per_period_effects and single-period placebo
                # are DID_M concepts that don't apply to non-binary data.
                per_period_effects = {}
                placebo_effect = float("nan")
                placebo_se = float("nan")
                placebo_t = float("nan")
                placebo_p = float("nan")
                placebo_ci = (float("nan"), float("nan"))
                placebo_available = False
            # Suppress joiner/leaver decomposition for all L_max >= 1
            # (the decomposition is a per-period DID_M concept, not
            # applicable to the per-group DID_1 estimand)
            effective_joiners_available = False
            effective_leavers_available = False

        # R2 P1b: Finalize replicate-df propagation across public
        # surfaces. When heterogeneity is active, its n_valid is
        # appended to `_replicate_n_valid_list` AFTER the main
        # surfaces (overall / joiners / leavers / event study /
        # placebo horizon) have already computed their t/p/CI fields
        # with an intermediate `_df_survey`. If heterogeneity reports
        # the smallest n_valid, the main-surface inference would be
        # anti-conservative relative to `survey_metadata.df_survey`
        # and HonestDiD. Re-run safe_inference with the FINAL
        # effective df so every surface agrees.
        _final_eff_df = _effective_df_survey(resolved_survey, _replicate_n_valid_list)
        # ONE canonical final inference df, shared by the refresh below and
        # by the `event_study_df` provenance recorded at construction, so the
        # two can never drift apart.
        _final_inf_df = _inference_df(_final_eff_df, resolved_survey)
        if _replicate_n_valid_list:
            # Recompute `effective_overall_*` directly — that's what
            # ships in results at line ~2776+. `effective_overall_att/se`
            # may differ from the raw `overall_att/se` under the delta
            # (cost-benefit) path at L_max >= 2; recomputing from
            # `effective_*` ensures both paths get the final df.
            if np.isfinite(effective_overall_se):
                effective_overall_t, effective_overall_p, effective_overall_ci = safe_inference(
                    effective_overall_att,
                    effective_overall_se,
                    alpha=self.alpha,
                    df=_final_inf_df,
                )
            # Keep `overall_*` in sync for any downstream code that
            # reads them directly (e.g., placebo_event_study_dict
            # construction flows from overall_*).
            overall_t, overall_p, overall_ci = safe_inference(
                overall_att,
                overall_se,
                alpha=self.alpha,
                df=_final_inf_df,
            )
            if joiners_available:
                joiners_t, joiners_p, joiners_ci = safe_inference(
                    joiners_att,
                    joiners_se,
                    alpha=self.alpha,
                    df=_final_inf_df,
                )
            if leavers_available:
                leavers_t, leavers_p, leavers_ci = safe_inference(
                    leavers_att,
                    leavers_se,
                    alpha=self.alpha,
                    df=_final_inf_df,
                )
            if multi_horizon_inference is not None:
                for _lag_r2, _info_r2 in list(multi_horizon_inference.items()):
                    _t_r2, _p_r2, _ci_r2 = safe_inference(
                        _info_r2["effect"],
                        _info_r2["se"],
                        alpha=self.alpha,
                        df=_final_inf_df,
                    )
                    _info_r2["t_stat"] = _t_r2
                    _info_r2["p_value"] = _p_r2
                    _info_r2["conf_int"] = _ci_r2
            else:
                # Phase 1 (L_max=None): `event_study_effects[1]` is a VALUE
                # COPY of the overall inference, built above from the
                # PRE-refresh `overall_*`; the loop just above covers only
                # the Phase 2 `multi_horizon_inference` surface. Without
                # this mirror the stored l=1 row keeps the INTERMEDIATE
                # df's t/p/CI while `overall_*` and
                # `survey_metadata.df_survey` report the final (tightened)
                # df - the same value-copy hazard the placebo mirror below
                # documents. Re-sync from the refreshed `overall_*` so the
                # single event-study row agrees with the overall estimate it
                # is a copy of.
                if event_study_effects and 1 in event_study_effects:
                    event_study_effects[1]["t_stat"] = overall_t
                    event_study_effects[1]["p_value"] = overall_p
                    event_study_effects[1]["conf_int"] = overall_ci
            if placebo_horizon_inference is not None:
                for _lag_r2, _info_r2 in list(placebo_horizon_inference.items()):
                    _t_r2, _p_r2, _ci_r2 = safe_inference(
                        _info_r2["effect"],
                        _info_r2["se"],
                        alpha=self.alpha,
                        df=_final_inf_df,
                    )
                    _info_r2["t_stat"] = _t_r2
                    _info_r2["p_value"] = _p_r2
                    _info_r2["conf_int"] = _ci_r2
                    # `placebo_event_study_dict` holds VALUE copies
                    # (not shared references) of the inner dicts, so
                    # mutating `placebo_horizon_inference[lag]` above
                    # does NOT propagate to the public surface. Update
                    # the negative-key mirror explicitly so the
                    # recomputed t/p/CI ship in results.
                    if (
                        placebo_event_study_dict is not None
                        and -_lag_r2 in placebo_event_study_dict
                    ):
                        placebo_event_study_dict[-_lag_r2]["t_stat"] = _t_r2
                        placebo_event_study_dict[-_lag_r2]["p_value"] = _p_r2
                        placebo_event_study_dict[-_lag_r2]["conf_int"] = _ci_r2
            if heterogeneity_effects:
                for _lag_r2, _info_r2 in list(heterogeneity_effects.items()):
                    if np.isfinite(_info_r2["se"]):
                        _t_r2, _p_r2, _ci_r2 = safe_inference(
                            _info_r2["beta"],
                            _info_r2["se"],
                            alpha=self.alpha,
                            df=_final_inf_df,
                        )
                        _info_r2["t_stat"] = _t_r2
                        _info_r2["p_value"] = _p_r2
                        _info_r2["conf_int"] = _ci_r2
            # Per-path heterogeneity (Wave 5 #11): per-(path, l) entries
            # snapshot df_inference at compute-time. Refresh with final df
            # so t/p/CI match `survey_metadata.df_survey`. Schema differs
            # from per-path event-study (`{path: {l: ...}}` vs
            # `{path: {"horizons": {l: ...}}}`), so inline loop here
            # rather than reusing `_refresh_path_inference`.
            if path_heterogeneity_effects:
                for _path_r2, _horizons_r2 in list(path_heterogeneity_effects.items()):
                    for _l_r2, _info_r2 in list(_horizons_r2.items()):
                        if np.isfinite(_info_r2["se"]):
                            _t_r2, _p_r2, _ci_r2 = safe_inference(
                                _info_r2["beta"],
                                _info_r2["se"],
                                alpha=self.alpha,
                                df=_final_inf_df,
                            )
                            _info_r2["t_stat"] = _t_r2
                            _info_r2["p_value"] = _p_r2
                            _info_r2["conf_int"] = _ci_r2
            # Normalized effects: another public surface built with the
            # pre-heterogeneity `_df_survey`. Recompute inference with
            # the final df so t/p/CI match the other surfaces (and the
            # NaN contract when the final df becomes undefined).
            if normalized_effects_out is not None:
                for _lag_r2, _info_r2 in list(normalized_effects_out.items()):
                    _t_r2, _p_r2, _ci_r2 = safe_inference(
                        _info_r2["effect"],
                        _info_r2["se"],
                        alpha=self.alpha,
                        df=_final_inf_df,
                    )
                    _info_r2["t_stat"] = _t_r2
                    _info_r2["p_value"] = _p_r2
                    _info_r2["conf_int"] = _ci_r2
            # Per-path event-study and placebo surfaces: their helpers
            # snapshotted df_inference BEFORE appending their own n_valid
            # contributions, and the global event-study / placebo /
            # heterogeneity / overall / joiners / leavers IF sites
            # appended their n_valid AFTER per-path runs. Refresh per-path
            # inference with the final df so it agrees with the global
            # surfaces and `survey_metadata.df_survey`.
            _refresh_path_inference(
                path_effects=path_effects,
                path_placebos=path_placebos,
                alpha=self.alpha,
                df_final=_final_inf_df,
            )

        # Persist the final effective df_survey into survey_metadata so
        # downstream consumers — HonestDiD bounds (honest_did.py:973
        # reads results.survey_metadata.df_survey), exported metadata,
        # and users — all see the same df that the recomputed
        # inference above used. SurveyMetadata is a mutable @dataclass
        # (diff_diff/survey.py:681), so direct attribute assignment is
        # safe.
        if survey_metadata is not None:
            survey_metadata.df_survey = _final_eff_df

        # df PROVENANCE (spec section 5, row M-092): the scalar df every
        # stored event-study / placebo row's safe_inference received.
        # Reuses `_final_inf_df` - the SAME value the refresh above applied
        # to every stored row - so provenance and inference cannot drift.
        # Recorded iff finite and > 0.
        # Cleared under bootstrap, whose percentile p/CIs never used a df:
        # required, not defensive, because TSL survey + bootstrap is a live
        # mode (only REPLICATE + bootstrap is rejected) where this
        # expression evaluates finite.
        _es_df_provenance: Optional[float] = None
        if (
            bootstrap_results is None
            and _final_inf_df is not None
            and np.isfinite(_final_inf_df)
            and _final_inf_df > 0
        ):
            _es_df_provenance = float(_final_inf_df)

        results = ChaisemartinDHaultfoeuilleResults(
            overall_att=effective_overall_att,
            overall_se=effective_overall_se,
            overall_t_stat=effective_overall_t,
            overall_p_value=effective_overall_p,
            overall_conf_int=effective_overall_ci,
            joiners_att=joiners_att if effective_joiners_available else float("nan"),
            joiners_se=joiners_se if effective_joiners_available else float("nan"),
            joiners_t_stat=joiners_t if effective_joiners_available else float("nan"),
            joiners_p_value=joiners_p if effective_joiners_available else float("nan"),
            joiners_conf_int=(
                joiners_ci if effective_joiners_available else (float("nan"), float("nan"))
            ),
            n_joiner_cells=n_joiner_cells if effective_joiners_available else 0,
            n_joiner_obs=n_joiner_obs if effective_joiners_available else 0,
            joiners_available=effective_joiners_available,
            leavers_att=leavers_att if effective_leavers_available else float("nan"),
            leavers_se=leavers_se if effective_leavers_available else float("nan"),
            leavers_t_stat=leavers_t if effective_leavers_available else float("nan"),
            leavers_p_value=leavers_p if effective_leavers_available else float("nan"),
            leavers_conf_int=(
                leavers_ci if effective_leavers_available else (float("nan"), float("nan"))
            ),
            n_leaver_cells=n_leaver_cells if effective_leavers_available else 0,
            n_leaver_obs=n_leaver_obs if effective_leavers_available else 0,
            leavers_available=effective_leavers_available,
            placebo_effect=placebo_effect,
            placebo_se=placebo_se,
            placebo_t_stat=placebo_t,
            placebo_p_value=placebo_p,
            placebo_conf_int=placebo_ci,
            placebo_available=placebo_available,
            per_period_effects=per_period_effects,
            units=all_groups,
            time_periods=all_periods,
            n_obs=n_obs_post,
            n_treated_obs=effective_n_treated,
            n_switcher_cells=effective_N_S,
            n_cohorts=n_cohorts,
            n_groups_dropped_crossers=n_groups_dropped_crossers,
            n_groups_dropped_singleton_baseline=n_groups_dropped_singleton_baseline,
            n_groups_dropped_never_switching=n_groups_dropped_never_switching,
            event_study_effects=event_study_effects,
            L_max=L_max,
            placebo_event_study=placebo_event_study_dict,
            event_study_df=_es_df_provenance,
            twfe_weights=twfe_weights_df,
            twfe_fraction_negative=twfe_fraction_negative,
            twfe_sigma_fe=twfe_sigma_fe,
            twfe_beta_fe=twfe_beta_fe,
            alpha=self.alpha,
            normalized_effects=normalized_effects_out,
            cost_benefit_delta=cost_benefit_result,
            sup_t_bands=(
                {
                    "crit_value": bootstrap_results.cband_crit_value,
                    "alpha": self.alpha,
                    "n_bootstrap": self.n_bootstrap,
                    "method": "multiplier_bootstrap",
                }
                if bootstrap_results is not None and bootstrap_results.cband_crit_value is not None
                else None
            ),
            bootstrap_results=bootstrap_results,
            covariate_residuals=(
                _build_covariate_diagnostics_df(covariate_diagnostics, controls)
                if covariate_diagnostics is not None
                else None
            ),
            linear_trends_effects=linear_trends_effects,
            trends_linear=_is_trends_linear,
            heterogeneity_effects=heterogeneity_effects,
            path_heterogeneity_effects=path_heterogeneity_effects,
            design2_effects=(
                _compute_design2_effects(
                    D_mat=D_mat,
                    # Design-2 always uses raw level outcomes (not residualized,
                    # not first-differenced). Use y_pivot as the canonical raw source.
                    Y_mat=y_pivot.to_numpy(),
                    N_mat=N_mat_orig,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max if L_max is not None else 1,
                )
                if design2
                else None
            ),
            path_effects=path_effects,
            path_placebo_event_study=path_placebos,
            path_cumulated_event_study=path_cumulated_event_study,
            path_sup_t_bands=(
                # When by_path + n_bootstrap > 0 is active, surface a
                # dict (possibly empty) — preserving the documented
                # `None` (not requested) vs `{}` (requested but empty)
                # contract that mirrors `path_effects` / `path_placebo_
                # event_study` empty-state behavior. The empty case
                # arises in two ways:
                #   1. `path_effects == {}` — no observed path has a
                #      complete window; the per-path bootstrap collector
                #      is skipped upstream and `path_cband_crit_values`
                #      stays `None`. We materialize `{}` here.
                #   2. Bootstrap ran but no path passed both gates
                #      (>=2 valid horizons AND a strict majority — more
                #      than 50% — of finite sup-t draws);
                #      `path_cband_crit_values == {}` — passes through.
                {
                    path_key: {
                        "crit_value": crit,
                        "alpha": self.alpha,
                        "n_bootstrap": self.n_bootstrap,
                        "method": "multiplier_bootstrap",
                        "n_valid_horizons": (
                            bootstrap_results.path_cband_n_valid_horizons.get(path_key, 0)
                            if bootstrap_results is not None
                            and bootstrap_results.path_cband_n_valid_horizons is not None
                            else 0
                        ),
                    }
                    for path_key, crit in (
                        bootstrap_results.path_cband_crit_values
                        if bootstrap_results is not None
                        and bootstrap_results.path_cband_crit_values is not None
                        else {}
                    ).items()
                    if np.isfinite(crit)
                }
                if (
                    (self.by_path is not None or self.paths_of_interest is not None)
                    and self.n_bootstrap > 0
                )
                else None
            ),
            survey_metadata=survey_metadata,
            _estimator_ref=self,
        )

        # ------------------------------------------------------------------
        # HonestDiD integration (when honest_did=True)
        # ------------------------------------------------------------------
        if honest_did and results.placebo_event_study:
            try:
                from diff_diff.honest_did import compute_honest_did

                results.honest_did_results = compute_honest_did(
                    results,
                    method="relative_magnitude",
                    M=1.0,
                    alpha=self.alpha,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"HonestDiD computation failed ({type(exc).__name__}): "
                    f"{exc}. results.honest_did_results will be None. "
                    f"You can retry with compute_honest_did(results, ...) "
                    f"using different parameters.",
                    UserWarning,
                    stacklevel=2,
                )
                results.honest_did_results = None

        self.results_ = results
        self.is_fitted_ = True
        return results


# =============================================================================
# Module-level helpers
# =============================================================================


def _check_forward_compat_gates(
    aggregate: Optional[str],
    L_max: Optional[int],
    controls: Optional[List[str]],
    trends_linear: Optional[bool],
    trends_nonparam: Any,
    honest_did: bool,
) -> None:
    """Raise ``NotImplementedError`` for any non-default Phase 3 parameter.

    Phase 2 parameters (``L_max``) are validated inline in ``fit()``
    after period detection. The ``aggregate`` parameter is still
    reserved for Phase 3.
    """
    if aggregate is not None:
        raise NotImplementedError(
            f"aggregate={aggregate!r} is reserved for Phase 3 of dCDH. "
            "Multi-horizon event study effects are computed automatically "
            "when L_max is set. See ROADMAP.md Phase 3."
        )
    # L_max is validated inline in fit() after period detection (needs
    # the period count). Not gated here.
    # controls gate lifted — DID^X covariate residualization implemented.
    # Validation (L_max >= 1 required) is in fit() after L_max detection.
    # trends_linear gate lifted - DID^{fd} linear trends implemented.
    # Validation (L_max >= 1, n_periods >= 3 required) is in fit().
    # trends_nonparam gate lifted - state-set trends implemented.
    # Validation (L_max >= 1, column exists, time-invariant) is in fit().
    # honest_did gate lifted - integration implemented.
    # Validation (L_max >= 1 required) is in fit() after L_max detection.


def _drop_crossing_cells(
    cell: pd.DataFrame, group_col: str, d_col: str
) -> Tuple[pd.DataFrame, int]:
    """
    Drop groups with more than one treatment-change period.

    The dCDH estimator uses the **first treatment change** (``F_g``) as
    the cohort marker for both the per-group building block ``DID_{g,l}``
    and the variance computation. Groups with a second treatment change
    at a later period would confound the multi-horizon estimates because
    ``DID_{g,l}`` attributes the full outcome change from ``F_g-1`` to
    ``F_g-1+l`` to the first switch, while the second switch also
    contributes to that outcome change.

    For binary treatment, >1 change means a reversal (e.g., 0->1->0).
    For non-binary, >1 change includes both reversals (0->2->1) and
    monotone multi-step paths (0->1->2). Both are dropped because the
    dCDH framework requires a single treatment-change event per group.
    A single jump of any magnitude (e.g., 0->3->3->3) has exactly
    1 change period and is kept.

    Parameters
    ----------
    cell : pd.DataFrame
        Cell-level dataset with columns for ``group_col`` and ``d_col``.
        Must be sorted by group and time.
    group_col : str
    d_col : str
        Treatment column name.

    Returns
    -------
    filtered : pd.DataFrame
        Subset of ``cell`` with all multi-switch groups removed.
    n_dropped : int
        Number of groups dropped.
    """
    # Count the number of periods with non-zero treatment changes per
    # group. A group with > 1 such period has changed treatment more
    # than once (multi-switch). This generalizes correctly to non-binary
    # treatment: a single jump 0->3 has 1 non-zero diff, while 0->1->0
    # has 2 non-zero diffs.
    diffs = cell.groupby(group_col)[d_col].diff().fillna(0)
    n_changes = (diffs != 0).groupby(cell[group_col]).sum()
    multi_switch_groups = n_changes[n_changes > 1].index.tolist()
    n_dropped = len(multi_switch_groups)
    if n_dropped > 0:
        warnings.warn(
            f"drop_larger_lower=True dropped {n_dropped} multi-switch group(s) "
            f"matching R DIDmultiplegtDYN behavior. Examples: "
            f"{multi_switch_groups[:5]}"
            + (f" (and {n_dropped - 5} more)" if n_dropped > 5 else ""),
            UserWarning,
            stacklevel=3,
        )
        cell = cell[~cell[group_col].isin(multi_switch_groups)].reset_index(drop=True)
    return cell, n_dropped


def _compute_per_period_dids(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Tuple[
    Dict[Any, Dict[str, Any]],
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute per-period DID_+,t and DID_-,t with explicit A11 zero-retention.

    Returns
    -------
    per_period_effects : dict
        Keyed by period; values are full per-period dicts including the
        ``did_*_t_a11_zeroed`` flags.
    a11_warnings : list of str
        One string per period that triggered an A11 violation.
    did_plus_t_arr : np.ndarray
        DID_+,t values aligned to ``periods[1:]``.
    did_minus_t_arr : np.ndarray
        DID_-,t values aligned to ``periods[1:]``.
    n_10_t_arr : np.ndarray
        Joiner cell counts aligned to ``periods[1:]``.
    n_01_t_arr : np.ndarray
        Leaver cell counts aligned to ``periods[1:]``.
    n_00_t_arr : np.ndarray
        Stable-untreated cell counts aligned to ``periods[1:]``.
    n_11_t_arr : np.ndarray
        Stable-treated cell counts aligned to ``periods[1:]``.
    a11_plus_zeroed_arr : np.ndarray
        Boolean flags marking periods where DID_+,t was zeroed by the
        A11 convention (joiners present but no stable_0 controls).
    a11_minus_zeroed_arr : np.ndarray
        Mirror for DID_-,t.
    """
    n_periods = len(periods)
    per_period_effects: Dict[Any, Dict[str, Any]] = {}
    a11_warnings: List[str] = []
    did_plus_t_list: List[float] = []
    did_minus_t_list: List[float] = []
    n_10_t_list: List[int] = []
    n_01_t_list: List[int] = []
    n_00_t_list: List[int] = []
    n_11_t_list: List[int] = []
    a11_plus_zeroed_list: List[bool] = []
    a11_minus_zeroed_list: List[bool] = []

    for t_idx in range(1, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        y_curr = Y_mat[:, t_idx]
        y_prev = Y_mat[:, t_idx - 1]
        n_curr = N_mat[:, t_idx]

        # Cell-presence guard: a (g, t) cell only counts if BOTH t and t-1
        # were observed for that group (n_gt > 0 and n_{g,t-1} > 0).
        n_prev = N_mat[:, t_idx - 1]
        present = (n_curr > 0) & (n_prev > 0)

        joiner_mask = (d_prev == 0) & (d_curr == 1) & present
        stable0_mask = (d_prev == 0) & (d_curr == 0) & present
        leaver_mask = (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_prev == 1) & (d_curr == 1) & present

        # Library equal-cell weighting: N_{a,b,t} here counts (g, t)
        # cells in each transition state. Each cell contributes once
        # regardless of how many original observations fed into the
        # y_gt cell mean. This is a documented library deviation from
        # AER 2020 Equation 3 (which defines N_{d,d',t} = sum N_{g,t},
        # i.e., observation-sum weighting) and from R DIDmultiplegtDYN
        # (individual-row weighting on unbalanced inputs). See
        # REGISTRY.md ## ChaisemartinDHaultfoeuille L517 "Note (deviation
        # from R DIDmultiplegtDYN ...)" for the full text.
        n_10 = int(joiner_mask.sum())
        n_00 = int(stable0_mask.sum())
        n_01 = int(leaver_mask.sum())
        n_11 = int(stable1_mask.sum())

        # --- DID_+,t (joiners side) ---
        did_plus_t_a11_zeroed = False
        if n_10 == 0:
            did_plus_t = 0.0
        elif n_00 == 0:
            # A11 violation: joiners exist but no stable_0 controls
            did_plus_t = 0.0
            did_plus_t_a11_zeroed = True
            a11_warnings.append(f"period {periods[t_idx]}: joiners present, no stable_0")
        else:
            # Unweighted means over cells (each cell contributes equally)
            joiner_avg = float((y_curr[joiner_mask] - y_prev[joiner_mask]).mean())
            stable0_avg = float((y_curr[stable0_mask] - y_prev[stable0_mask]).mean())
            did_plus_t = joiner_avg - stable0_avg

        # --- DID_-,t (leavers side) ---
        did_minus_t_a11_zeroed = False
        if n_01 == 0:
            did_minus_t = 0.0
        elif n_11 == 0:
            did_minus_t = 0.0
            did_minus_t_a11_zeroed = True
            a11_warnings.append(f"period {periods[t_idx]}: leavers present, no stable_1")
        else:
            stable1_avg = float((y_curr[stable1_mask] - y_prev[stable1_mask]).mean())
            leaver_avg = float((y_curr[leaver_mask] - y_prev[leaver_mask]).mean())
            did_minus_t = stable1_avg - leaver_avg

        per_period_effects[periods[t_idx]] = {
            "did_plus_t": did_plus_t,
            "did_minus_t": did_minus_t,
            "n_10_t": n_10,
            "n_01_t": n_01,
            "n_00_t": n_00,
            "n_11_t": n_11,
            "did_plus_t_a11_zeroed": did_plus_t_a11_zeroed,
            "did_minus_t_a11_zeroed": did_minus_t_a11_zeroed,
        }
        did_plus_t_list.append(did_plus_t)
        did_minus_t_list.append(did_minus_t)
        n_10_t_list.append(n_10)
        n_01_t_list.append(n_01)
        n_00_t_list.append(n_00)
        n_11_t_list.append(n_11)
        a11_plus_zeroed_list.append(did_plus_t_a11_zeroed)
        a11_minus_zeroed_list.append(did_minus_t_a11_zeroed)

    return (
        per_period_effects,
        a11_warnings,
        np.array(did_plus_t_list, dtype=float),
        np.array(did_minus_t_list, dtype=float),
        np.array(n_10_t_list, dtype=int),
        np.array(n_01_t_list, dtype=int),
        np.array(n_00_t_list, dtype=int),
        np.array(n_11_t_list, dtype=int),
        np.array(a11_plus_zeroed_list, dtype=bool),
        np.array(a11_minus_zeroed_list, dtype=bool),
    )


def _compute_placebo(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Optional[Tuple[float, bool, List[str]]]:
    """
    Compute the single-lag placebo DID_M^pl from AER 2020 placebo specification.

    Same logic as DID_M but evaluated on the pre-event difference
    ``Y_{g, t-1} - Y_{g, t-2}`` for cells with three-period histories.
    Requires ``T >= 3``.

    Mirrors the main path's A11 zero-retention machinery: when placebo
    joiners exist but no 3-period stable_0 controls do (or symmetric
    for leavers/stable_1), the affected per-period contribution is set
    to zero AND a warning string is appended to ``placebo_a11_warnings``.
    The caller is responsible for surfacing the consolidated warning.
    The zero-retention preserves the period's switcher count in the
    placebo ``N_S^pl`` denominator, biasing the placebo toward zero in
    the offending direction (matching placebo paper convention).

    Returns
    -------
    None if ``T < 3`` or no qualifying cells. Otherwise a tuple
    ``(placebo_effect, True, placebo_a11_warnings)`` where
    ``placebo_a11_warnings`` is a list of one string per period that
    triggered an A11 violation in the placebo numerator.
    """
    n_periods = len(periods)
    if n_periods < 3:
        return None

    placebo_plus_per_t: List[float] = []
    placebo_minus_per_t: List[float] = []
    n_10_per_t: List[int] = []
    n_01_per_t: List[int] = []
    placebo_a11_warnings: List[str] = []

    for t_idx in range(2, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        d_pre_prev = D_mat[:, t_idx - 2]
        y_prev = Y_mat[:, t_idx - 1]
        y_pre_prev = Y_mat[:, t_idx - 2]

        # Cell-presence guard: a (g, t) cell only counts if all three
        # consecutive periods (t-2, t-1, t) were observed for the group.
        present = (N_mat[:, t_idx] > 0) & (N_mat[:, t_idx - 1] > 0) & (N_mat[:, t_idx - 2] > 0)

        # Joiners that have a 3-period history with stable D=0 in t-2 and t-1
        joiner_mask = (d_pre_prev == 0) & (d_prev == 0) & (d_curr == 1) & present
        # Stable_0 controls with stable D=0 in t-2 and t-1
        stable0_mask = (d_pre_prev == 0) & (d_prev == 0) & (d_curr == 0) & present
        # Mirror for leavers/stable_1 (3-period stable treatment then leave)
        leaver_mask = (d_pre_prev == 1) & (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_pre_prev == 1) & (d_prev == 1) & (d_curr == 1) & present

        # Placebo weights: library equal-cell counts (same documented
        # deviation as the main DID_M path; see REGISTRY.md L517).
        n_10 = int(joiner_mask.sum())
        n_00 = int(stable0_mask.sum())
        n_01 = int(leaver_mask.sum())
        n_11 = int(stable1_mask.sum())

        # Joiners side: distinguish "no joiners" (natural zero) from
        # "joiners but no stable_0" (A11 violation, flagged + warned)
        if n_10 == 0:
            placebo_plus_t = 0.0
        elif n_00 == 0:
            placebo_plus_t = 0.0
            placebo_a11_warnings.append(
                f"period {periods[t_idx]}: placebo joiners present, no stable_0"
            )
        else:
            joiner_avg = float((y_prev[joiner_mask] - y_pre_prev[joiner_mask]).mean())
            stable0_avg = float((y_prev[stable0_mask] - y_pre_prev[stable0_mask]).mean())
            # Backward-difference convention (pre-period minus reference,
            # Y_{t-2} - Y_{t-1}), matching the multi-horizon placebo path
            # `_compute_multi_horizon_placebos` (`switcher_change - ctrl_avg`
            # with `Y_bwd - Y_ref`, times switch_direction=+1 for joiners) and
            # R `did_multiplegt_dyn`. Equivalently `stable0_avg - joiner_avg`.
            placebo_plus_t = stable0_avg - joiner_avg

        # Leavers side: symmetric A11 distinction
        if n_01 == 0:
            placebo_minus_t = 0.0
        elif n_11 == 0:
            placebo_minus_t = 0.0
            placebo_a11_warnings.append(
                f"period {periods[t_idx]}: placebo leavers present, no stable_1"
            )
        else:
            stable1_avg = float((y_prev[stable1_mask] - y_pre_prev[stable1_mask]).mean())
            leaver_avg = float((y_prev[leaver_mask] - y_pre_prev[leaver_mask]).mean())
            # Backward-difference convention times switch_direction=-1 for
            # leavers (see the joiners-side note above): the multi-horizon path
            # contributes `leaver_avg - stable1_avg` here.
            placebo_minus_t = leaver_avg - stable1_avg

        placebo_plus_per_t.append(placebo_plus_t)
        placebo_minus_per_t.append(placebo_minus_t)
        n_10_per_t.append(n_10)
        n_01_per_t.append(n_01)

    n_10_arr = np.array(n_10_per_t, dtype=int)
    n_01_arr = np.array(n_01_per_t, dtype=int)
    N_S_pl = int(n_10_arr.sum() + n_01_arr.sum())
    if N_S_pl == 0:
        return None
    placebo_effect = float(
        (n_10_arr @ np.array(placebo_plus_per_t) + n_01_arr @ np.array(placebo_minus_per_t))
        / N_S_pl
    )
    return placebo_effect, True, placebo_a11_warnings


# ======================================================================
# Phase 3: Covariate residualization helpers
# ======================================================================


def _compute_covariate_residualization(
    Y_mat: np.ndarray,
    X_cell: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    rank_deficient_action: str = "warn",
) -> Tuple[np.ndarray, Dict[str, Any], set]:
    """Residualize outcomes by partialling out covariates per baseline treatment.

    Implements ``DID^X`` from Web Appendix Section 1.2 of de Chaisemartin &
    D'Haultfoeuille (2024). For each baseline treatment value *d*, estimates
    ``theta_hat_d`` via OLS of first-differenced outcomes on first-differenced
    covariates with time FEs, restricted to not-yet-treated observations.
    Then residualizes at levels: ``Y_tilde[g,t] = Y[g,t] - X[g,t] @ theta_hat_d``.

    The level-residualization is equivalent to difference-residualization by
    the Frisch-Waugh-Lovell theorem, so all downstream DID computations
    (which use ``Y[g, out] - Y[g, ref]``) automatically produce the correct
    covariate-adjusted estimates.

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
        Cell-level outcome means.
    X_cell : np.ndarray, shape (n_groups, n_periods, n_covariates)
        Cell-level covariate means.
    N_mat : np.ndarray, shape (n_groups, n_periods)
        Observation counts per cell (>0 if observed).
    baselines : np.ndarray, shape (n_groups,)
        ``D_{g,1}`` baseline treatment values (float).
    first_switch_idx : np.ndarray, shape (n_groups,)
        Column index of first treatment change (-1 if never-switching).

    Returns
    -------
    Y_residualized : np.ndarray, shape (n_groups, n_periods)
        Outcome matrix with covariate effects removed.
    diagnostics : dict
        Keyed by baseline value (float). Each entry has ``theta_hat``
        (covariate coefficients), ``n_obs`` (OLS sample size), and
        ``r_squared`` (first-stage R-squared).
    """
    from diff_diff.linalg import solve_ols

    n_groups, n_periods = Y_mat.shape
    n_covariates = X_cell.shape[2]
    Y_resid = Y_mat.copy()
    diagnostics: Dict[float, Dict[str, Any]] = {}
    failed_baselines: set = set()

    # Pre-compute observation validity masks for first-differencing.
    # both_observed[g, t] = True iff N_mat[g, t] > 0 AND N_mat[g, t-1] > 0
    both_observed = np.zeros((n_groups, n_periods), dtype=bool)
    both_observed[:, 1:] = (N_mat[:, 1:] > 0) & (N_mat[:, :-1] > 0)

    # not_yet_switched[g, t] = True iff group g has not switched by period t
    # (first_switch_idx[g] == -1 means never-switcher -> always True)
    t_indices = np.arange(n_periods)[np.newaxis, :]  # (1, n_periods)
    f_g_col = first_switch_idx[:, np.newaxis]  # (n_groups, 1)
    not_yet_switched = (f_g_col == -1) | (f_g_col > t_indices)

    for d_val in np.unique(baselines):
        d_mask = baselines == d_val  # (n_groups,)

        # Valid OLS observations: baseline matches, not-yet-treated, both
        # periods observed, t >= 1 (first-differencing needs t and t-1).
        valid = d_mask[:, np.newaxis] & not_yet_switched & both_observed
        valid_g, valid_t = np.where(valid)

        n_obs = len(valid_g)
        if n_obs == 0:
            diagnostics[float(d_val)] = {
                "theta_hat": np.full(n_covariates, np.nan),
                "n_obs": 0,
                "r_squared": np.nan,
            }
            # NaN out outcomes for failed strata so they're excluded
            # from downstream DID computation (don't mix raw + adjusted).
            group_indices = np.where(d_mask)[0]
            Y_resid[group_indices, :] = np.nan
            failed_baselines.add(float(d_val))
            warnings.warn(
                f"No not-yet-treated observations for baseline treatment "
                f"d={d_val}. Cannot estimate covariate slope theta_hat. "
                f"Groups with this baseline are excluded from the "
                f"covariate-adjusted estimation.",
                UserWarning,
                stacklevel=3,
            )
            continue

        # First-differenced outcomes and covariates
        dY = Y_mat[valid_g, valid_t] - Y_mat[valid_g, valid_t - 1]  # (n_obs,)
        dX = X_cell[valid_g, valid_t] - X_cell[valid_g, valid_t - 1]  # (n_obs, K)

        # Check for non-finite values (NaN from missing covariates/outcomes)
        finite_mask = np.isfinite(dY) & np.all(np.isfinite(dX), axis=1)
        if not finite_mask.all():
            dY = dY[finite_mask]
            dX = dX[finite_mask]
            n_obs = len(dY)
            if n_obs == 0:
                diagnostics[float(d_val)] = {
                    "theta_hat": np.full(n_covariates, np.nan),
                    "n_obs": 0,
                    "r_squared": np.nan,
                }
                continue
            valid_t_finite = valid_t[finite_mask]
        else:
            valid_t_finite = valid_t

        # Build design: [intercept, dX, time_dummies (reference dropped)]
        # The intercept is required when dropping one time dummy as
        # reference category; without it the omitted period's FE is
        # forced to zero, biasing theta_hat.
        intercept = np.ones((n_obs, 1))
        unique_t = np.unique(valid_t_finite)
        n_time_fe = len(unique_t) - 1
        if n_time_fe > 0:
            time_dummies = np.zeros((n_obs, n_time_fe))
            for i, t_val in enumerate(unique_t[1:]):
                time_dummies[:, i] = (valid_t_finite == t_val).astype(float)
            design = np.hstack([intercept, dX, time_dummies])
        else:
            design = np.hstack([intercept, dX])

        # Small-sample guard: skip if fewer obs than parameters
        n_params = design.shape[1]
        if n_obs < n_params:
            diagnostics[float(d_val)] = {
                "theta_hat": np.full(n_covariates, np.nan),
                "n_obs": n_obs,
                "r_squared": np.nan,
            }
            # NaN out outcomes for failed strata (don't mix raw + adjusted)
            group_indices_fail = np.where(d_mask)[0]
            Y_resid[group_indices_fail, :] = np.nan
            failed_baselines.add(float(d_val))
            warnings.warn(
                f"DID^X: baseline d={d_val} has {n_obs} not-yet-treated "
                f"observations but {n_params} regressors. Groups with "
                f"this baseline are excluded from covariate-adjusted "
                f"estimation.",
                UserWarning,
                stacklevel=3,
            )
            continue

        # OLS: dY = [dX, time_FE] @ beta + epsilon
        coefs, residuals, _vcov = solve_ols(
            design,
            dY,
            return_vcov=True,
            rank_deficient_action=rank_deficient_action,
        )

        # Extract covariate coefficients (indices 1..n_covariates;
        # index 0 is the intercept)
        theta_hat = coefs[1 : 1 + n_covariates]

        # R-squared of first-stage regression
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((dY - dY.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        diagnostics[float(d_val)] = {
            "theta_hat": theta_hat.copy(),
            "n_obs": n_obs,
            "r_squared": r_squared,
        }

        # Guard: if some control coefficients are NaN (rank-deficient
        # OLS dropped collinear controls), residualize with only the
        # finite subset. Replace NaN coefficients with 0 so einsum
        # only uses the identified controls.
        nan_mask = ~np.isfinite(theta_hat)
        if nan_mask.any():
            n_dropped = int(nan_mask.sum())
            warnings.warn(
                f"DID^X: rank-deficient first-stage OLS for baseline "
                f"d={d_val} dropped {n_dropped} collinear control(s). "
                f"Residualization uses the {n_covariates - n_dropped} "
                f"identified control(s).",
                UserWarning,
                stacklevel=3,
            )
            theta_hat = np.where(np.isfinite(theta_hat), theta_hat, 0.0)

        # Residualize Y at levels for all groups with this baseline.
        # Vectorized level residualization: Y_tilde[g, t] = Y[g, t] - X[g, t] @ theta_hat
        group_indices = np.where(d_mask)[0]
        if len(group_indices) > 0:
            # X_sub: (n_d_groups, n_periods, n_covariates), theta: (n_covariates,)
            X_sub = X_cell[group_indices]  # (n_d, T, K)
            adjustment = np.einsum("gtk,k->gt", X_sub, theta_hat)  # (n_d, T)
            # Mask: only adjust cells that are observed and have finite covariates
            valid = (N_mat[group_indices] > 0) & np.all(np.isfinite(X_sub), axis=2)
            Y_resid[group_indices] = np.where(
                valid, Y_mat[group_indices] - adjustment, Y_mat[group_indices]
            )

    return Y_resid, diagnostics, failed_baselines


def _compute_first_differenced_matrix(
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-difference the outcome matrix for ``DID^{fd}`` estimation.

    Transforms ``Y_mat`` into first-differences for the group-specific
    linear trends estimator (Web Appendix Section 1.3, Lemma 6). When
    passed to ``_compute_multi_horizon_dids()`` and the IF function,
    the standard ``DID_{g,l}`` formula on ``Z_mat`` produces
    ``DID^{fd}_{g,l}`` exactly.

    The ``F_g >= 3`` constraint (paper, 1-indexed) maps to
    ``first_switch_idx >= 2`` (0-indexed). This is enforced
    automatically: ``N_mat_fd[:, 0] = 0`` causes groups with
    ``first_switch_idx = 1`` to fail the ``N_mat > 0`` eligibility
    check at their reference period.

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
        Cell-level outcome means (possibly already residualized).
    N_mat : np.ndarray, shape (n_groups, n_periods)
        Observation counts per cell.

    Returns
    -------
    Z_mat : np.ndarray, shape (n_groups, n_periods)
        First-differenced outcomes. ``Z[:, 0] = NaN``,
        ``Z[:, t] = Y[:, t] - Y[:, t-1]`` for ``t >= 1``.
    N_mat_fd : np.ndarray, shape (n_groups, n_periods)
        Adjusted observation counts. ``N_fd[:, 0] = 0``,
        ``N_fd[:, t] = min(N[:, t], N[:, t-1])`` for ``t >= 1``.
    """
    n_groups, n_periods = Y_mat.shape
    Z_mat = np.full((n_groups, n_periods), np.nan)
    Z_mat[:, 1:] = Y_mat[:, 1:] - Y_mat[:, :-1]

    N_mat_fd = np.zeros_like(N_mat)
    N_mat_fd[:, 1:] = np.minimum(N_mat[:, 1:], N_mat[:, :-1])

    return Z_mat, N_mat_fd


def _compute_heterogeneity_test(
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    X_het: np.ndarray,
    L_max: int,
    placebo: int = 0,
    alpha: float = 0.05,
    rank_deficient_action: str = "warn",
    group_ids_order: Optional[np.ndarray] = None,
    obs_survey_info: Optional[Dict[str, Any]] = None,
    replicate_n_valid_list: Optional[List[int]] = None,
    path_groups: Optional[Set[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Test for heterogeneous treatment effects (Web Appendix Section 1.5).

    Regresses ``S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})`` on ``X_g`` plus
    cohort indicator dummies ``(D_{g,1}, F_g, S_g)``. Under Assumption 15
    (Lemma 7), the coefficient on ``X_g`` is an unbiased estimator of the
    variance-weighted average of effect differences. Standard OLS inference
    is valid - no need to account for DID estimation error.

    Forward horizons ``l in 1..L_max`` use ``out_idx = F_g - 1 + l`` (post-
    period). Backward placebo horizons ``l in -1..-placebo`` use the same
    construction with negated ``l``, so ``out_idx = F_g - 1 + l < F_g - 1``
    (pre-period). Lemma 7's OLS inference is symmetric — same regression
    structure with negated time index — and mirrors R's
    ``did_multiplegt_dyn(..., predict_het=list("X", c(-1)), placebo=N)``
    per-horizon dispatcher (``DIDmultiplegtDYN:::did_multiplegt_main``
    placebo block at the ``effect = matrix(-i, ...)`` rbind site).

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
    N_mat : np.ndarray, shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
    X_het : np.ndarray, shape (n_groups,)
        Time-invariant covariate to test for heterogeneity.
    L_max : int
    alpha : float
    group_ids_order : np.ndarray, optional
        Canonical post-filter group id list aligned to Y_mat row order.
        Required when ``obs_survey_info`` is supplied.
    obs_survey_info : dict, optional
        Observation-level survey info with keys ``group_ids`` (raw per-row
        group labels), ``time_ids`` (raw per-row period labels),
        ``weights`` (per-row survey weights), ``resolved``
        (ResolvedSurveyDesign), and ``periods`` (sorted canonical period
        array matching ``Y_mat``'s column order). When provided, the
        regression uses WLS with per-group weights
        ``W_g = sum of obs survey weights in group g``. The group-level
        WLS coefficient IF is
        ``ψ_g = inv(X'WX)[1,:] @ x_g * W_g * r_g``. Two observation-level
        expansions of ``ψ_g`` coexist on this path, split by variance
        helper so each path uses the allocator that preserves
        byte-identity for its aggregation rule:

        * **Binder TSL** (``compute_survey_if_variance``): the
          cell-period single-cell allocator —
          ``ψ_i = ψ_g * (w_i / W_{g, out_idx})`` for obs in
          ``(g, out_idx)``, zero elsewhere. Under PSU=group per-obs
          distribution differs from the legacy
          ``ψ_i = ψ_g * (w_i / W_g)`` but PSU-level aggregates
          telescope to the same ``ψ_g``, so Binder variance is
          byte-identical to the pre-cell-period release. Under
          within-group-varying PSU mass lands in the post-period PSU
          of the transition (DID_l post-period convention).
        * **Rao-Wu replicate** (``compute_replicate_if_variance``):
          the legacy group-level allocator ``ψ_i = ψ_g * (w_i / W_g)``.
          Replicate variance computes ``θ_r = sum_i ratio_ir * ψ_i``
          at observation level, so moving ψ_g mass onto the
          post-period cell would silently change the replicate SE
          whenever a replicate column's ratios vary within a group
          (which the library allows — e.g., per-row BRR/Fay/SDR
          matrices). Keeping the legacy allocator on this branch
          preserves byte-identity of replicate SE across every
          previously-supported fit. Replicate + within-group-varying
          PSU is unreachable by construction (``SurveyDesign``
          rejects ``replicate_weights`` combined with explicit
          ``strata/psu/fpc``).

        The effective df for t-critical values follows the site-level
        ``min(df_s, n_valid_het - 1)`` rule and the helper mutates
        ``replicate_n_valid_list`` so the final
        ``_effective_df_survey(...)`` sees this site's n_valid.
    replicate_n_valid_list : list[int], optional
        Shared accumulator for replicate-weight ``n_valid`` counts across
        IF sites. When provided and a replicate design is in use, this
        function appends its own ``n_valid_het`` before computing the
        local effective df — so both the local inference fields and the
        final ``survey_metadata.df_survey`` (set by ``fit()``) reflect
        this site's contribution.

    Returns
    -------
    dict
        ``{l: {beta, se, t_stat, p_value, conf_int, n_obs}}`` per horizon.
    """
    from diff_diff.linalg import solve_ols
    from diff_diff.utils import safe_inference

    n_groups, n_periods = Y_mat.shape
    results: Dict[int, Dict[str, Any]] = {}

    # Survey setup (once, before horizon loop). When inactive, df_s=None and
    # the existing plain-OLS path runs unchanged. Forward-horizon survey
    # heterogeneity is supported; the per-iteration backward-horizon gate
    # below catches the unsupported case (survey + l_h < 0) defensively.
    # Under normal `fit()` flow the global / per-path call sites pass
    # `placebo=0` when survey is active (with a UserWarning), so the
    # backward iterations are never reached and the gate is a defensive
    # backstop only — direct callers passing `placebo > 0 + survey` get
    # the explicit raise.
    use_survey = obs_survey_info is not None and group_ids_order is not None
    if use_survey:
        # The flag definition above guarantees these (mypy can't track it).
        assert obs_survey_info is not None and group_ids_order is not None
        from diff_diff.survey import (
            compute_replicate_if_variance,
            compute_survey_if_variance,
        )

        obs_gids_raw = np.asarray(obs_survey_info["group_ids"])
        obs_w_raw = np.asarray(obs_survey_info["weights"], dtype=np.float64)
        resolved = obs_survey_info["resolved"]
        # df_s starts from the shared effective df (base-capped by
        # design df). Under replicate designs the helper handles the
        # min(resolved.df_survey, min(n_valid) - 1) reduction and
        # preserves None when the base df is undefined (QR-rank ≤ 1).
        # Using the helper here — rather than re-deriving locally —
        # keeps heterogeneity's df consistent with the main dCDH
        # surfaces (R2 P1a). `list(... or [])` avoids accidental
        # mutation of the caller's shared tracker at this site; the
        # explicit append happens inside the horizon loop below.
        df_s = _effective_df_survey(resolved, list(replicate_n_valid_list or []))
        # Contract: only obs whose group is in the canonical post-filter
        # list contribute. Groups dropped upstream (Step 5b interior gaps,
        # Step 6 multi-switch) appear in obs_gids_raw but must be
        # zero-weighted in the IF expansion.
        gid_list = (
            group_ids_order.tolist()
            if hasattr(group_ids_order, "tolist")
            else list(group_ids_order)
        )
        gid_set = set(gid_list)
        valid = np.array([g in gid_set for g in obs_gids_raw])
        # Per-group total weight aligned to Y_mat row order
        W_g_all = np.zeros(n_groups, dtype=np.float64)
        for i, gid in enumerate(gid_list):
            mask_g = (obs_gids_raw == gid) & valid
            W_g_all[i] = obs_w_raw[mask_g].sum()
    else:
        df_s = None

    # Iterate forward (1..L_max) and backward placebo (-1..-placebo)
    # horizons in a single loop. Backward horizons emit "placebo
    # predict_het" rows matching R's did_multiplegt_dyn(..., predict_het,
    # placebo) per-by_level dispatcher (R's placebo block at the
    # `effect = matrix(-i, ...)` rbind site).
    horizons_to_compute = list(range(1, L_max + 1)) + list(range(-1, -placebo - 1, -1))
    for l_h in horizons_to_compute:
        # Per-iteration survey gate (defensive backstop). Forward
        # iterations under survey work; backward iterations raise
        # because the Binder TSL cell-period allocator's REGISTRY
        # justification is tied to post-period attribution. fit() gates
        # this upstream by passing `placebo=0` when survey is active
        # (with a UserWarning when the user requested placebo het), so
        # the raise is reachable only via direct callers.
        if use_survey and l_h < 0:
            raise NotImplementedError(
                "survey_design with backward-horizon (placebo) "
                "predict_het is not yet supported. The Binder TSL "
                "cell-period allocator is derived for post-period "
                "attribution (REGISTRY.md ChaisemartinDHaultfoeuille "
                "survey IF expansion Note); pre-period (l < 0) "
                "attribution requires a separate methodology "
                "derivation. Forward-horizon predict_het + "
                "survey_design is supported."
            )
        # Eligible switchers at this horizon (same logic as multi-horizon DID)
        eligible = []
        dep_var = []
        x_vals = []
        cohort_keys = []

        for g in range(n_groups):
            if path_groups is not None and g not in path_groups:
                continue
            f_g = first_switch_idx[g]
            if f_g < 0:
                continue  # never-switcher
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l_h
            if out_idx >= n_periods:
                continue
            if ref_idx < 0:
                continue
            if out_idx < 0:
                # Backward horizons can push out_idx below 0 when
                # F_g - 1 + l_h < 0 (e.g., F_g=2, l_h=-2). numpy
                # negative indexing on N_mat[g, out_idx] would silently
                # wrap to a tail period rather than skip — guard
                # explicitly so the eligibility filter is correct for
                # any l_h sign.
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue
            S_g = float(switch_direction[g])
            y_diff = Y_mat[g, out_idx] - Y_mat[g, ref_idx]
            eligible.append(g)
            dep_var.append(S_g * y_diff)
            x_vals.append(X_het[g])
            cohort_keys.append((float(baselines[g]), int(f_g), int(switch_direction[g])))

        n_obs = len(eligible)
        if n_obs < 3:
            results[l_h] = {
                "beta": float("nan"),
                "se": float("nan"),
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
                "n_obs": n_obs,
            }
            continue

        dep_arr = np.array(dep_var)
        x_arr = np.array(x_vals).reshape(-1, 1)

        # Design: [intercept, X_g, cohort_dummies (reference dropped)]
        # The intercept is required when dropping one cohort dummy as
        # reference; without it the omitted cohort's mean is forced to
        # zero, which biases beta^{het}_l.
        intercept = np.ones((n_obs, 1))
        unique_cohorts = sorted(set(cohort_keys))
        n_cohort_dummies = len(unique_cohorts) - 1
        if n_cohort_dummies > 0:
            cohort_map = {c: i for i, c in enumerate(unique_cohorts)}
            cohort_idx = np.array([cohort_map[c] for c in cohort_keys])
            cohort_dummies = np.zeros((n_obs, len(unique_cohorts)))
            cohort_dummies[np.arange(n_obs), cohort_idx] = 1.0
            # Drop first cohort as reference
            cohort_dummies = cohort_dummies[:, 1:]
            design = np.hstack([intercept, x_arr, cohort_dummies])
        else:
            design = np.hstack([intercept, x_arr])

        # Compute post-drop numerical rank for both the small-sample
        # guard and the t-distribution df. `_detect_rank_deficiency` is
        # the same helper `solve_ols` calls internally; calling it
        # explicitly here lets the guard use post-drop rank (matching
        # R's `df.residual = n_obs - rank(design)` convention from
        # `DIDmultiplegtDYN:::did_multiplegt_main` `qt(0.975,
        # df.residual(model))`) instead of pre-drop column count,
        # which would incorrectly short-circuit cases where solve_ols's
        # R-style alias drop leaves `n_obs > rank > 0` (e.g., cohort-
        # dummy collinearity at high horizons). For full-rank designs
        # `rank == n_params` and behavior is bit-identical to the
        # pre-PR `n_obs - n_params` path. The extra O(nk^2) cost is
        # negligible at heterogeneity scale (k = intercept + X_het +
        # cohort dummies, typically 5-30 columns; n = path-switcher
        # count, typically 30-300 groups).
        from diff_diff.linalg import _detect_rank_deficiency

        rank, _dropped, _pivot = _detect_rank_deficiency(design)

        # Guard: need MORE observations than rank for a well-defined
        # residual df. When `n_obs <= rank`, the regression has zero
        # residual df (perfect fit or under-identified) and inference
        # is undefined. This is the rank-based replacement for the
        # pre-PR `n_obs <= n_params` short-circuit.
        if n_obs <= rank:
            results[l_h] = {
                "beta": float("nan"),
                "se": float("nan"),
                "t_stat": float("nan"),
                "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
                "n_obs": n_obs,
            }
            continue

        df_ols = int(n_obs) - int(rank)

        if not use_survey:
            # Plain OLS path: standard inference per Lemma 7.
            coefs, _residuals, vcov = solve_ols(
                design,
                dep_arr,
                return_vcov=True,
                rank_deficient_action=rank_deficient_action,
            )
            # Under rank-deficient designs solve_ols R-style-drops one
            # or more columns and NaN-fills their coefs. If `X_het`
            # (column index 1) is the dropped column, the heterogeneity
            # coefficient is unidentified — NaN-fill the inference
            # tuple (matches R's lm() returning NA for aliased
            # coefficients). Other columns being dropped is fine: the
            # X_het coefficient and its (1,1) vcov entry remain
            # identified, df_ols already reflects the post-drop rank.
            if not np.isfinite(coefs[1]):
                results[l_h] = {
                    "beta": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": n_obs,
                }
                continue
            beta_het = float(coefs[1])
            se_het = float("nan")
            if vcov is not None and np.isfinite(vcov[1, 1]) and vcov[1, 1] > 0:
                se_het = float(np.sqrt(vcov[1, 1]))
            t_stat, p_val, ci = safe_inference(beta_het, se_het, alpha=alpha, df=df_ols)
        else:
            # Survey-aware path: WLS with per-group weights + TSL IF variance.
            W_elig = W_g_all[eligible]
            # solve_ols handles sqrt-weight scaling natively when
            # weight_type='pweight' (linalg.py). Skip vcov — we compute
            # design-based variance ourselves below.
            coefs, _residuals, _vcov_ignored = solve_ols(
                design,
                dep_arr,
                weights=W_elig,
                weight_type="pweight",
                return_vcov=False,
                rank_deficient_action=rank_deficient_action,
            )
            # Rank-deficiency short-circuit: if any coef is NaN, return NaN
            # inference. Mixing solve_ols's R-style drop with a pinv-derived
            # IF would describe different estimands.
            if not np.all(np.isfinite(coefs)):
                results[l_h] = {
                    "beta": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": n_obs,
                }
                continue

            beta_het = float(coefs[1])
            # Original-scale residuals (solve_ols applies sqrt-weight scaling
            # internally and back-transforms residuals, but we need them for
            # our IF computation below).
            r_g = dep_arr - design @ coefs

            # Group-level IF for β_X: ψ_g[X] = inv(X'WX)[1,:] @ x_g * W_g * r_g.
            # Under full rank (gated above), pinv == inv. Wrap matmuls in
            # errstate: macOS Accelerate BLAS can emit spurious divide/overflow
            # warnings on sparse-cohort designs even though the result is finite.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                XtWX = design.T @ (W_elig[:, None] * design)
                XtWX_inv = np.linalg.pinv(XtWX)
                psi_g = (XtWX_inv[1, :] @ design.T) * W_elig * r_g  # (n_eligible,)

            # Allocator dispatch. Two observation-level expansions of
            # ψ_g coexist on this path, split by variance helper:
            #
            #   * Binder TSL (compute_survey_if_variance): cell-period
            #     single-cell allocator —
            #       ψ_i = ψ_g * (w_i / W_{g, out_idx})
            #     for obs in (g, out_idx), zero elsewhere. Under
            #     PSU=group, per-obs distribution differs from the
            #     legacy ψ_i = ψ_g * (w_i / W_g) but PSU-level
            #     aggregates telescope to ψ_g, so Binder variance is
            #     byte-identical. Under within-group-varying PSU, mass
            #     lands in the post-period PSU of the transition, which
            #     is what Binder needs. DID_l single-cell convention —
            #     see REGISTRY.md ChaisemartinDHaultfoeuille survey IF
            #     expansion Note.
            #
            #   * Rao-Wu replicate (compute_replicate_if_variance):
            #     legacy group-level allocator —
            #       ψ_i = ψ_g * (w_i / W_g)
            #     for obs in group g. Replicate variance computes
            #     θ_r = sum_i ratio_ir * ψ_i at observation level, so
            #     moving ψ_g onto the post-period cell only would
            #     silently change the replicate SE whenever a
            #     replicate column's ratios vary within group (e.g.,
            #     the per-row replicate matrices this library
            #     accepts). The group-level allocator preserves
            #     byte-identity for all replicate usages under
            #     PSU=group. The replicate + within-group-varying
            #     PSU case is not reachable (SurveyDesign rejects
            #     replicate_weights combined with explicit psu).
            if getattr(resolved, "uses_replicate_variance", False):
                psi_obs = np.zeros(len(obs_w_raw), dtype=np.float64)
                for e_idx, g_idx in enumerate(eligible):
                    gid = gid_list[g_idx]
                    mask_g = (obs_gids_raw == gid) & valid
                    w_sum_g = obs_w_raw[mask_g].sum()
                    if w_sum_g > 0:
                        psi_obs[mask_g] = psi_g[e_idx] * (obs_w_raw[mask_g] / w_sum_g)
                var_s, n_valid_het = compute_replicate_if_variance(psi_obs, resolved)
                if replicate_n_valid_list is not None:
                    replicate_n_valid_list.append(n_valid_het)
                # Reduce df_s to reflect this horizon's n_valid.
                # R2 P1a: when the shared base-capped df_s is None
                # (undefined base df, e.g., QR-rank ≤ 1), the
                # heterogeneity df MUST stay None — per-site n_valid
                # cannot rescue a rank-deficient design. The
                # _inference_df wrapper at the safe_inference call
                # below coerces None to 0 under replicate, forcing
                # NaN inference.
                if df_s is None:
                    df_s_local = None
                else:
                    df_s_local = min(int(df_s), int(n_valid_het) - 1)
            else:
                # Survey heterogeneity path: obs-level info is present.
                assert obs_survey_info is not None
                obs_tids = np.asarray(obs_survey_info["time_ids"])
                periods_arr = np.asarray(obs_survey_info["periods"])
                psi_obs = np.zeros(len(obs_w_raw), dtype=np.float64)
                for e_idx, g_idx in enumerate(eligible):
                    gid = gid_list[g_idx]
                    out_idx = first_switch_idx[g_idx] - 1 + l_h
                    t_val_out = periods_arr[out_idx]
                    mask_cell = (obs_gids_raw == gid) & (obs_tids == t_val_out) & valid
                    w_cell = obs_w_raw[mask_cell].sum()
                    if w_cell > 0:
                        psi_obs[mask_cell] = psi_g[e_idx] * (obs_w_raw[mask_cell] / w_cell)
                var_s = compute_survey_if_variance(psi_obs, resolved)
                df_s_local = df_s
            se_het = float(np.sqrt(var_s)) if np.isfinite(var_s) and var_s > 0 else float("nan")
            t_stat, p_val, ci = safe_inference(
                beta_het,
                se_het,
                alpha=alpha,
                df=_inference_df(df_s_local, resolved),
            )

        results[l_h] = {
            "beta": beta_het,
            "se": se_het,
            "t_stat": t_stat,
            "p_value": p_val,
            "conf_int": ci,
            "n_obs": n_obs,
        }

    return results


def _compute_design2_effects(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
) -> Optional[Dict[str, Any]]:
    """Compute Design-2 switch-in/switch-out effects (Web Appendix Section 1.6).

    Identifies groups with exactly 2 treatment changes (join then leave),
    computes the exit period E_g, and provides delta^+ (post-join) and
    delta^- (post-leave) summaries.

    This is a convenience wrapper that reports descriptive statistics about
    the switch-in and switch-out subpopulations rather than a full
    re-estimation (which would require specialized control pools as
    described in the paper). See REGISTRY.md for documentation.

    Returns None if no join-then-leave groups exist.
    """
    n_groups, n_periods = D_mat.shape

    # Identify join-then-leave groups: exactly 2 treatment changes where
    # the first is a join (D increases) and the second is a leave (D decreases)
    design2_groups = []
    exit_periods = []

    for g in range(n_groups):
        changes = []
        for t in range(1, n_periods):
            if N_mat[g, t] <= 0 or N_mat[g, t - 1] <= 0:
                continue
            if D_mat[g, t] != D_mat[g, t - 1]:
                direction = 1 if D_mat[g, t] > D_mat[g, t - 1] else -1
                changes.append((t, direction))
        if len(changes) == 2 and changes[0][1] == 1 and changes[1][1] == -1:
            design2_groups.append(g)
            exit_periods.append(changes[1][0])

    if len(design2_groups) == 0:
        return None

    # Compute summary statistics for the switch-in/switch-out subpopulation
    switch_in_effects = []
    switch_out_effects = []

    for i, g in enumerate(design2_groups):
        f_g = first_switch_idx[g]
        e_g = exit_periods[i]
        ref_idx = f_g - 1

        # Switch-in: Y[g, f_g] - Y[g, f_g-1] (effect of joining)
        if ref_idx >= 0 and N_mat[g, f_g] > 0 and N_mat[g, ref_idx] > 0:
            switch_in = float(Y_mat[g, f_g] - Y_mat[g, ref_idx])
            switch_in_effects.append(switch_in)

        # Switch-out: Y[g, e_g] - Y[g, e_g-1] (effect of leaving)
        if e_g - 1 >= 0 and N_mat[g, e_g] > 0 and N_mat[g, e_g - 1] > 0:
            switch_out = float(Y_mat[g, e_g] - Y_mat[g, e_g - 1])
            switch_out_effects.append(switch_out)

    result: Dict[str, Any] = {
        "n_design2_groups": len(design2_groups),
        "switch_in": {
            "n_groups": len(switch_in_effects),
            "mean_effect": float(np.mean(switch_in_effects)) if switch_in_effects else np.nan,
        },
        "switch_out": {
            "n_groups": len(switch_out_effects),
            "mean_effect": float(np.mean(switch_out_effects)) if switch_out_effects else np.nan,
        },
    }
    return result


def _build_covariate_diagnostics_df(
    diagnostics: Dict[str, Any],
    control_names: List[str],
) -> pd.DataFrame:
    """Build a tidy DataFrame from the per-baseline residualization diagnostics."""
    rows = []
    for d_val, diag in sorted(diagnostics.items()):
        theta = diag["theta_hat"]
        for k, name in enumerate(control_names):
            rows.append(
                {
                    "baseline_treatment": d_val,
                    "covariate": name,
                    "theta_hat": float(theta[k]) if np.isfinite(theta[k]) else np.nan,
                    "n_obs": diag["n_obs"],
                    "r_squared": diag["r_squared"],
                }
            )
    return pd.DataFrame(rows)


# ======================================================================
# Phase 2: Multi-horizon helpers
# ======================================================================


def _compute_group_switch_metadata(
    D_mat: np.ndarray,
    N_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-group switch metadata from the pivoted panel matrices.

    For each group g, identifies the baseline treatment ``D_{g,1}``, the
    first-switch period index ``F_g`` (or -1 if never-switching), and the
    switch direction ``S_g`` (+1 joiner, -1 leaver, 0 never-switching).
    Also computes ``T_g`` - the last period index at which there is still
    a baseline-matched control that hasn't switched (needed for horizon
    eligibility).

    This helper is shared by Phase 1 (cohort-recentered IF in
    ``_compute_cohort_recentered_inputs``) and Phase 2 (multi-horizon
    ``DID_{g,l}`` computation).

    Parameters
    ----------
    D_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted treatment matrix (cell-level, binary in Phase 1).
    N_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted observation-count matrix. Zero means group g is missing
        at period t.

    Returns
    -------
    baselines : np.ndarray of shape (n_groups,), dtype int
        ``D_{g,1}`` for each group (treatment at the first global period).
    first_switch_idx : np.ndarray of shape (n_groups,), dtype int
        Period index of g's first treatment change (-1 if never-switching).
        This is ``F_g`` in the paper's notation, expressed as a column
        index into D_mat (0-based).
    switch_direction : np.ndarray of shape (n_groups,), dtype int
        ``S_g``: +1 if treatment increases at first switch (joiner),
        -1 if decreases (leaver), 0 if never-switching.
    T_g : np.ndarray of shape (n_groups,), dtype int
        For each group, the last period index at which a baseline-matched
        not-yet-switched control still exists. Groups whose baseline
        value has no other group that switches later get ``T_g = -1``
        (they have no valid control at any horizon). This is used for
        horizon eligibility: ``DID_{g,l}`` is computable iff
        ``first_switch_idx[g] - 1 + l <= T_g[g]``.

    Raises
    ------
    ValueError
        If any group is missing the first global period in N_mat (this
        should have been caught by fit() Step 5b validation).
    """
    n_groups, n_periods = D_mat.shape

    # Defensive: fit() Step 5b rejects groups missing the baseline.
    if N_mat.size > 0 and (N_mat[:, 0] <= 0).any():
        raise ValueError(
            "_compute_group_switch_metadata: at least one group is missing "
            "the first global period in N_mat. fit() Step 5b should have "
            "rejected this."
        )

    baselines = D_mat[:, 0].astype(float)
    first_switch_idx = np.full(n_groups, -1, dtype=int)
    switch_direction = np.zeros(n_groups, dtype=int)

    for g in range(n_groups):
        for t in range(1, n_periods):
            if N_mat[g, t] <= 0 or N_mat[g, t - 1] <= 0:
                continue
            if D_mat[g, t] != D_mat[g, t - 1]:
                first_switch_idx[g] = t
                switch_direction[g] = 1 if D_mat[g, t] > D_mat[g, t - 1] else -1
                break

    # T_g: for each group g, the last period at which there is still a
    # baseline-matched group whose treatment has NOT changed. This is
    # max_{g': D_{g',1} = D_{g,1}} (F_{g'} - 1), i.e., the period just
    # before the latest-switching control in g's baseline cohort.
    # Never-switching groups (F = -1) have F-1 = T (last period), so
    # they extend T_g to the panel end for their baseline cohort.
    unique_baselines = np.unique(baselines)
    max_control_period = {}  # baseline -> max period index with a valid control
    for d in unique_baselines:
        baseline_mask = baselines == d
        # For each group with this baseline, the last period at which it
        # can still serve as a not-yet-switched control is F_g - 1
        # (or n_periods - 1 if never-switching).
        f_vals = first_switch_idx[baseline_mask]
        control_last = np.where(f_vals == -1, n_periods - 1, f_vals - 1)
        max_control_period[float(d)] = int(control_last.max()) if control_last.size > 0 else -1

    T_g = np.array(
        [max_control_period.get(float(baselines[g]), -1) for g in range(n_groups)],
        dtype=int,
    )

    return baselines, first_switch_idx, switch_direction, T_g


def _compute_multi_horizon_dids(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute the per-group building block ``DID_{g,l}`` and its aggregate
    ``DID_l`` for horizons ``l = 1, ..., L_max``.

    Implements Equation 3 and Equation 5 of the dynamic companion paper
    (NBER WP 29873). For each switching group g eligible at horizon l::

        DID_{g,l} = Y_{g, F_g-1+l} - Y_{g, F_g-1}
                    - mean_{g' in controls} (Y_{g', F_g-1+l} - Y_{g', F_g-1})

    where the control set is ``{g': D_{g',1} = D_{g,1}, F_{g'} > F_g-1+l}``.

    The aggregate is ``DID_l = (1/N_l) * sum S_g * DID_{g,l}`` over
    eligible groups.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
        From ``_compute_group_switch_metadata()``.
    L_max : int
        Maximum horizon to compute.

    Returns
    -------
    dict mapping horizon l -> {
        "did_l": float,           # aggregate DID_l (NaN if N_l=0)
        "N_l": int,               # count of eligible switching groups
        "did_g_l": np.ndarray,    # per-group DID_{g,l} (NaN for non-eligible)
        "eligible_mask": np.ndarray,  # boolean shape (n_groups,)
        "switcher_fraction": float,   # N_l / N_1 (NaN if N_1=0)
    }
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    # Pre-compute per-baseline lookup of (group_indices, first_switch_indices)
    # for efficient control-pool identification.
    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Dict[str, Any]] = {}
    a11_multi_warnings: List[str] = []
    N_1 = 0  # will be set at l=1 for switcher_fraction

    for l in range(1, L_max + 1):  # noqa: E741
        did_g_l = np.full(n_groups, np.nan)

        # Eligibility: switching group with F_g - 1 + l_h observable.
        # F_g is stored as a column index (0-based), so the outcome
        # period is first_switch_idx[g] - 1 + l. This must be a valid
        # column AND the group must be observed there (N_mat > 0).
        # Also, T_g[g] must be >= first_switch_idx[g] - 1 + l (controls
        # available at the outcome period).
        eligible = np.zeros(n_groups, dtype=bool)
        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1  # period just before first switch
            out_idx = f_g - 1 + l  # outcome period for horizon l
            if ref_idx < 0 or out_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue  # no baseline-matched control available
            eligible[g] = True

        N_l = int(eligible.sum())
        if l == 1:
            N_1 = N_l

        if N_l == 0:
            results[l] = {
                "did_l": float("nan"),
                "N_l": 0,
                "did_g_l": did_g_l,
                "eligible_mask": eligible,
                "switcher_fraction": float("nan"),
            }
            continue

        # Compute DID_{g,l} for each eligible group.
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l
            d_base = float(baselines[g])

            # Switcher's outcome change
            switcher_change = Y_mat[g, out_idx] - Y_mat[g, ref_idx]

            # Control pool: same baseline, not yet switched by out_idx.
            # F_{g'} > out_idx (hasn't switched yet) OR F_{g'} = -1
            # (never switches). Both must be observed at ref_idx and
            # out_idx.
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > out_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, out_idx] > 0)
            )
            # State-set trends: restrict controls to same set as switcher
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]

            if ctrl_pool.size == 0:
                # No observed controls at this horizon (may be terminal
                # missingness, not a true A11 violation). Exclude the
                # group from N_l rather than zero-retaining, so the
                # missing-data case doesn't bias DID_l toward zero.
                eligible[g] = False
                a11_multi_warnings.append(
                    f"horizon {l}, group_idx {g}: "
                    f"no baseline-matched controls at outcome period"
                )
                continue

            ctrl_changes = Y_mat[ctrl_pool, out_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_avg = float(ctrl_changes.mean())
            did_g_l[g] = switcher_change - ctrl_avg

        # Recompute N_l after control-pool exclusions
        N_l = int(eligible.sum())
        if l == 1:
            N_1 = N_l
        if N_l == 0:
            results[l] = {
                "did_l": float("nan"),
                "N_l": 0,
                "did_g_l": did_g_l,
                "eligible_mask": eligible,
                "switcher_fraction": float("nan"),
            }
            continue

        # Aggregate: DID_l = (1/N_l) * sum S_g * DID_{g,l}
        S_eligible = switch_direction[eligible].astype(float)
        did_g_eligible = did_g_l[eligible]
        did_l = float((S_eligible * did_g_eligible).sum() / N_l)

        results[l] = {
            "did_l": did_l,
            "N_l": N_l,
            "did_g_l": did_g_l,
            "eligible_mask": eligible,
            "switcher_fraction": N_l / N_1 if N_1 > 0 else float("nan"),
        }

    # Attach A11 warnings to the results for the caller to surface
    if a11_multi_warnings:
        results["_a11_warnings"] = a11_multi_warnings  # type: ignore[index]  # sentinel str key in int-keyed dict

    return results


def _compute_per_group_if_multi_horizon(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
    compute_per_period: bool = True,
    switcher_subset_mask: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Compute per-group influence function ``U^G_{g,l}`` for ``l = 1..L_max``.

    Each group g contributes to ``DID_l`` in two capacities:

    1. **As a switcher** (if g is eligible at horizon l): contributes
       ``S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})`` to the numerator.
    2. **As a control** (if g serves as a not-yet-switched control for
       some other switcher g'): contributes
       ``-S_{g'} * (1/N^{g'}_{out}) * (Y_{g, out} - Y_{g, ref})``
       where ref/out are g's reference/outcome periods.

    The result satisfies ``sum(U_l) == N_l * DID_l``, which is verified
    as a sanity check.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
        From ``_compute_group_switch_metadata()``.
    L_max : int
    switcher_subset_mask : np.ndarray of bool, shape (n_groups,), optional
        When supplied, restricts the switcher iteration to groups where
        the mask is ``True``. Groups outside the subset contribute as
        controls only (their switcher-side contribution is skipped). The
        control pool is unchanged — this mirrors how the joiners-only /
        leavers-only IF is constructed (see ``_compute_cohort_recentered_inputs``).
        Used by ``by_path`` to zero out switcher contributions for groups
        not in the selected path. Default ``None`` preserves the legacy
        behavior of iterating over all switchers.

    Returns
    -------
    dict mapping horizon l -> (U_g_l, U_per_period_l) tuple
        - ``U_g_l``: np.ndarray of shape (n_groups,). NOT cohort-centered;
          the caller applies ``_cohort_recenter()`` before computing SE.
        - ``U_per_period_l``: np.ndarray of shape (n_groups, n_periods).
          Per-``(g, t)``-cell contributions attributed to the outcome
          period ``t = F_g - 1 + l`` (same column for switcher and its
          controls, since they share the outcome period). Satisfies
          ``U_per_period_l.sum(axis=1) == U_g_l``. Consumed by the cell-
          period IF allocator for within-group-varying PSU designs.
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    # Pre-compute per-baseline group indices for control-pool lookup.
    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        U_l = np.zeros(n_groups, dtype=float)
        U_per_period_l: Optional[np.ndarray] = (
            np.zeros((n_groups, n_periods), dtype=float) if compute_per_period else None
        )

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            if switcher_subset_mask is not None and not switcher_subset_mask[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l
            if ref_idx < 0 or out_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue

            d_base = float(baselines[g])
            S_g = float(switch_direction[g])

            # Control pool for this switcher at this horizon
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > out_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, out_idx] > 0)
            )
            # State-set trends: restrict controls to same set as switcher
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]
            n_ctrl = ctrl_pool.size

            if n_ctrl == 0:
                # No controls: A11-like, DID_{g,l} = 0. The switcher's
                # contribution to U_l is zero, but its count is in N_l.
                continue

            # Switcher contribution: +S_g * (Y_{g, out} - Y_{g, ref}).
            # Per-cell attribution convention: assign the whole contrast
            # to the outcome cell (g, out_idx). See REGISTRY.md's Note
            # on survey IF expansion for the rationale behind this
            # convention (library choice, not a derived result).
            switcher_change = Y_mat[g, out_idx] - Y_mat[g, ref_idx]
            U_l[g] += S_g * switcher_change
            if U_per_period_l is not None:
                U_per_period_l[g, out_idx] += S_g * switcher_change

            # Control contributions: each control g' in the pool gets
            # -S_g * (1/n_ctrl) * (Y_{g', out} - Y_{g', ref}). Same
            # post-period attribution as the switcher side.
            ctrl_changes = Y_mat[ctrl_pool, out_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_contrib = (S_g / n_ctrl) * ctrl_changes
            U_l[ctrl_pool] -= ctrl_contrib
            if U_per_period_l is not None:
                U_per_period_l[ctrl_pool, out_idx] -= ctrl_contrib

        results[l] = (U_l, U_per_period_l)

    return results


def _enumerate_treatment_paths(
    D_mat: np.ndarray,
    first_switch_idx: np.ndarray,
    N_mat: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[
    List[Tuple[int, ...]],
    Dict[Tuple[int, ...], np.ndarray],
    Dict[Tuple[int, ...], int],
]:
    """
    Enumerate observed treatment paths and select either the top-``by_path``
    most common or the user-specified ``paths_of_interest`` subset.

    For each switcher group ``g``, the path is the treatment tuple
    ``(D_{g, F_g-1}, D_{g, F_g}, ..., D_{g, F_g-1+L_max})`` — length
    ``L_max + 1``, matching R's ``did_multiplegt_dyn`` window convention.
    Groups whose window falls outside the panel or whose window contains
    unobserved cells are skipped (they contribute to no path bucket).

    Paths are ranked by group frequency; ties are broken lexicographically
    on the path tuple for deterministic ordering. If ``by_path`` exceeds
    the number of observed paths, all observed paths are returned with
    a ``UserWarning``.

    When ``paths_of_interest`` is provided, the user-specified subset is
    used instead of the top-k ranking. Duplicate paths emit a
    ``UserWarning`` and are deduplicated; paths not observed in the
    panel emit a ``UserWarning`` and are omitted from the result.

    Parameters
    ----------
    D_mat : np.ndarray of shape (n_groups, n_periods)
        Treatment matrix. Binary (0/1) or integer-coded discrete
        treatment (D in Z); upstream validation enforces D == round(D)
        when ``not is_binary``.
    first_switch_idx : np.ndarray of shape (n_groups,)
        Index of first switch per group; ``-1`` for never-switching groups.
    N_mat : np.ndarray of shape (n_groups, n_periods)
        Cell-count matrix (zero where the cell is unobserved).
    L_max : int
        Event-study horizon; window length is ``L_max + 1``.
    by_path : int or None
        Number of most-common paths to select. Mutually exclusive with
        ``paths_of_interest``; exactly one of the two is non-None when
        the per-path branch fires.
    paths_of_interest : list of tuple[int, ...] or None, default None
        User-specified path subset. When provided, ``by_path`` is
        ignored.

    Returns
    -------
    selected_paths : list of tuple[int, ...]
        Selected path tuples. Under ``by_path``, ordered by descending
        frequency. Under ``paths_of_interest``, in user-specified order
        modulo deduplication and unobserved-path filtering.
    path_to_group_mask : dict[tuple[int, ...], np.ndarray of bool]
        Per-path boolean mask over all ``n_groups`` identifying switchers
        that follow that path.
    path_to_count : dict[tuple[int, ...], int]
        Count of switcher groups per selected path.
    """
    n_groups, n_periods = D_mat.shape
    path_of_group: Dict[int, Tuple[int, ...]] = {}
    for g in range(n_groups):
        f_g = int(first_switch_idx[g])
        if f_g < 0:
            continue
        start = f_g - 1
        stop = start + L_max + 1
        if start < 0 or stop > n_periods:
            continue
        window_counts = N_mat[g, start:stop]
        if np.any(window_counts <= 0):
            continue
        window = D_mat[g, start:stop]
        if np.any(np.isnan(window)):
            continue
        path_of_group[g] = tuple(int(round(float(v))) for v in window)

    path_counts: Dict[Tuple[int, ...], int] = {}
    for path in path_of_group.values():
        path_counts[path] = path_counts.get(path, 0) + 1

    if paths_of_interest is not None:
        # Canonicalization (Tuple[int, ...] with Python int) happens in
        # __init__ / set_params via _validate_paths_of_interest, so
        # duplicates such as `[(np.int64(0), 1, 1, 1), (0, 1, 1, 1)]`
        # collapse to the same tuple here and the seen-set check fires.
        observed_paths_set = set(path_counts.keys())
        seen: set = set()
        selected_paths: List[Tuple[int, ...]] = []
        for p in paths_of_interest:
            if p in seen:
                warnings.warn(
                    f"paths_of_interest contains duplicate path {p!r}; " f"deduplicating.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            seen.add(p)
            if p not in observed_paths_set:
                warnings.warn(
                    f"paths_of_interest path {p!r} has zero observed "
                    f"groups in the panel; this path will be omitted "
                    f"from path_effects.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            selected_paths.append(p)
    else:
        observed_paths = sorted(path_counts.keys(), key=lambda p: (-path_counts[p], p))
        n_observed = len(observed_paths)
        if by_path is None:
            # Defensive: caller always passes one of the two selectors.
            selected_paths = []
        elif by_path >= n_observed:
            if by_path > n_observed and n_observed > 0:
                warnings.warn(
                    f"by_path={by_path} exceeds the number of observed "
                    f"paths ({n_observed}). Returning all observed paths.",
                    UserWarning,
                    stacklevel=2,
                )
            selected_paths = observed_paths
        else:
            selected_paths = observed_paths[:by_path]

    path_to_group_mask: Dict[Tuple[int, ...], np.ndarray] = {}
    for path in selected_paths:
        mask = np.zeros(n_groups, dtype=bool)
        for g_idx, g_path in path_of_group.items():
            if g_path == path:
                mask[g_idx] = True
        path_to_group_mask[path] = mask

    path_to_count = {p: path_counts[p] for p in selected_paths}
    return selected_paths, path_to_group_mask, path_to_count


def _compute_path_effects(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    eligible_mask_var: np.ndarray,
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    all_groups: List[Any],
    alpha: float,
    df_inference: Optional[int] = None,
    set_ids: Optional[np.ndarray] = None,
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
    obs_survey_info: Optional[Dict[str, Any]] = None,
    eligible_groups: Optional[List[Any]] = None,
    replicate_n_valid_list: Optional[List[int]] = None,
) -> Optional[Dict[Tuple[int, ...], Dict[str, Any]]]:
    """
    Compute per-path event-study effects using the joiners/leavers IF pattern.

    For each of the top-``by_path`` observed treatment paths, compute:

    - ``DID_{path, l}`` for ``l = 1..L_max`` as the within-path mean of
      ``DID_{g, l}`` — equals ``sum(U_l_path) / N_l_path`` where
      ``U_l_path`` is the per-group IF with switcher contributions zeroed
      for groups not in the path (control contributions and cohort
      structure unchanged).
    - SE depends on ``obs_survey_info``:

      * Non-survey (``obs_survey_info is None``): plug-in SE via
        ``_plugin_se(U_centered_path, divisor=N_l_path)`` after cohort-
        recentering with the ORIGINAL cohort structure. This mirrors how
        joiners_se / leavers_se use their respective counts as the
        divisor and preserve the full cohort structure.
      * Survey (``obs_survey_info is not None``): the path-restricted
        per-period IF is built and routed through
        ``_survey_se_from_group_if`` (analytical Binder TSL with the
        cell-period allocator; replicate-weight designs use the same
        cell allocator unconditionally). Under replicate weights every
        per-(path, l) fit appends ``n_valid`` to
        ``replicate_n_valid_list`` so the final ``df_survey`` reflects
        all per-path fits; the post-call ``_refresh_path_inference``
        re-runs ``safe_inference`` on every populated entry so the
        stored ``t_stat`` / ``p_value`` / ``conf_int`` use the final
        ``df_survey`` rather than the compute-time snapshot.

    Returns an empty dict ``{}`` when ``by_path`` was requested but no
    switcher group has a complete ``[F_g - 1, F_g - 1 + L_max]`` window
    (all switchers too late in the panel or with unobserved cells). The
    empty dict is distinguished from ``None`` at the result layer: ``None``
    means the feature was not requested, ``{}`` means requested but empty.
    A ``UserWarning`` is emitted so the caller sees that ``by_path`` was
    a no-op on this panel.
    """
    from diff_diff.utils import safe_inference

    selected_paths, path_to_group_mask, path_to_count = _enumerate_treatment_paths(
        D_mat=D_mat,
        first_switch_idx=first_switch_idx,
        N_mat=N_mat,
        L_max=L_max,
        by_path=by_path,
        paths_of_interest=paths_of_interest,
    )

    if not selected_paths:
        if paths_of_interest is not None:
            # Every requested path was unobserved (each emitted its own
            # per-path "zero observed groups" warning inside the
            # enumerator). Distinguish from the by_path=k case where
            # the panel itself has no complete-window path.
            warnings.warn(
                "paths_of_interest was requested but every "
                "user-specified path was either unobserved in the "
                "panel or had a window outside the L_max+1 "
                "convention (per-path 'zero observed groups' "
                "UserWarnings already issued). results.path_effects "
                "is populated as an empty dict to signal 'requested "
                "but empty'.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"by_path={by_path} was requested but no observed "
                f"treatment path has a complete window [F_g-1, "
                f"F_g-1+L_max={L_max}] within the panel. "
                f"results.path_effects is populated as an empty dict "
                f"to signal 'requested but empty'. Extend the panel "
                f"so switchers have L_max+1 consecutive observed "
                f"cells starting at F_g-1, or reduce L_max.",
                UserWarning,
                stacklevel=2,
            )
        return {}

    # Cohort ids for the variance-eligible set (same construction as the
    # per-horizon SE path at the primary fit() site: (D_{g,1}, F_g, S_g)).
    n_groups = D_mat.shape[0]
    cohort_keys = [
        (
            float(baselines[g]),
            int(first_switch_idx[g]),
            int(switch_direction[g]),
        )
        for g in range(n_groups)
    ]
    unique_c: Dict[Tuple[float, int, int], int] = {}
    cid = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask_var[g]:
            cid[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_c:
            unique_c[key] = len(unique_c)
        cid[g] = unique_c[key]
    cohort_id_eligible = cid[eligible_mask_var]

    path_effects: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    # `frequency_rank` is the within-selected-paths rank by descending
    # group count (lex tiebreak on the path tuple). Decoupled from the
    # iteration order over `selected_paths` so that under
    # `paths_of_interest` (user-specified order) the rank still
    # reflects true frequency. Under `by_path=k`, `selected_paths` is
    # already sorted by descending frequency so ranks coincide with
    # iteration order.
    rank_sorted_paths = sorted(
        selected_paths,
        key=lambda p: (-path_to_count[p], p),
    )
    path_to_freq_rank = {p: i + 1 for i, p in enumerate(rank_sorted_paths)}

    for path in selected_paths:
        switcher_mask = path_to_group_mask[path]
        n_path_groups = int(switcher_mask.sum())

        per_path_if = _compute_per_group_if_multi_horizon(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            baselines=baselines,
            first_switch_idx=first_switch_idx,
            switch_direction=switch_direction,
            T_g=T_g,
            L_max=L_max,
            set_ids=set_ids,
            compute_per_period=(obs_survey_info is not None),
            switcher_subset_mask=switcher_mask,
        )

        horizons: Dict[int, Dict[str, Any]] = {}
        for l_h in range(1, L_max + 1):
            U_l_path, U_pp_l_path = per_path_if[l_h]

            # N_l_path: path-restricted count of eligible switchers at
            # horizon l. Mirror _compute_multi_horizon_dids' eligibility
            # (the did_g_l array is NaN for non-eligible switchers).
            did_g_l = multi_horizon_dids[l_h].get("did_g_l")
            if did_g_l is None:
                n_l_path = 0
            else:
                n_l_path = int(np.sum(switcher_mask & ~np.isnan(did_g_l)))

            if n_l_path == 0:
                horizons[l_h] = {
                    "effect": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": 0,
                }
                continue

            U_l_path_elig = U_l_path[eligible_mask_var]
            # Point estimate: within-path mean DID
            effect_path = float(U_l_path.sum() / n_l_path)

            # SE: cohort-recenter with ORIGINAL cohort structure. Under
            # survey, route through _survey_se_from_group_if (cell-period
            # allocator). Otherwise plug-in with path-specific divisor
            # (joiners/leavers pattern).
            U_centered_path = _cohort_recenter(U_l_path_elig, cohort_id_eligible)
            if obs_survey_info is None:
                se_path = _plugin_se(U_centered=U_centered_path, divisor=n_l_path)
            else:
                assert U_pp_l_path is not None
                assert eligible_groups is not None
                U_pp_l_path_elig = U_pp_l_path[eligible_mask_var]
                U_centered_pp_path = _cohort_recenter_per_period(
                    U_pp_l_path_elig, cohort_id_eligible
                )
                U_scaled = U_centered_path / n_l_path
                U_pp_scaled = U_centered_pp_path / n_l_path
                se_path, n_valid_replicates = _survey_se_from_group_if(
                    U_centered=U_scaled,
                    eligible_groups=eligible_groups,
                    obs_survey_info=obs_survey_info,
                    U_centered_per_period=U_pp_scaled,
                )
                if n_valid_replicates is not None and replicate_n_valid_list is not None:
                    replicate_n_valid_list.append(n_valid_replicates)

            # Path-scoped degenerate-cohort warning. Mirrors the overall-
            # path surface (`Cohort-recentered analytical variance is
            # unidentified...` at the main fit() site): when the centered
            # path IF is identically zero, the variance is unidentified
            # for this (path, horizon) despite n_l_path > 0. Common when
            # a low-frequency path's switchers all fall in singleton
            # (D_{g,1}, F_g, S_g) cohorts after the path subset.
            if np.isnan(se_path) and U_centered_path.size > 0 and n_l_path > 0:
                warnings.warn(
                    f"Cohort-recentered analytical variance is "
                    f"unidentified for path={path} at horizon l={l_h}: "
                    f"the path-subset centered influence function is "
                    f"identically zero (every variance-eligible path "
                    f"switcher forms its own (D_{{g,1}}, F_g, S_g) "
                    f"cohort, or the path has a single contributing "
                    f"group). DID_{{path,l}} point estimate is still "
                    f"valid; SE / t_stat / p_value / conf_int are "
                    f"NaN-consistent. Rare paths with few contributing "
                    f"groups routinely hit this case — include more "
                    f"groups following this trajectory for a non-"
                    f"degenerate analytical SE.",
                    UserWarning,
                    stacklevel=2,
                )

            t_p, p_p, ci_p = safe_inference(effect_path, se_path, alpha=alpha, df=df_inference)

            horizons[l_h] = {
                "effect": effect_path,
                "se": se_path,
                "t_stat": t_p,
                "p_value": p_p,
                "conf_int": ci_p,
                "n_obs": n_l_path,
            }

        path_effects[path] = {
            "n_groups": n_path_groups,
            "frequency_rank": path_to_freq_rank[path],
            "horizons": horizons,
        }

    return path_effects


def _compute_path_cumulated_event_study(
    D_mat: np.ndarray,
    N_mat: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    path_effects: Dict[Tuple[int, ...], Dict[str, Any]],
    alpha: float,
    df_inference: Optional[int] = None,
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
) -> Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]:
    """
    Per-path cumulated level effects under ``trends_linear=True``.

    Mirrors the global ``linear_trends_effects`` cumulation
    (``chaisemartin_dhaultfoeuille.py:3340-3398``): for each enumerated
    path, accumulate per-group running sums of ``DID^{fd}_{g, l'}`` over
    ``l' = 1..l``, then average over the path's switchers eligible at
    horizon ``l``. SE is the conservative upper bound (sum of per-horizon
    component SEs from ``path_effects[path]["horizons"][l']["se"]``,
    NaN-consistent: if any component SE is non-finite the cumulated SE is
    NaN). Inference (t-stat, p-value, CI) via ``safe_inference``.

    Returns ``{path: {horizon: {effect, se, t_stat, p_value, conf_int,
    n_obs}}}`` directly (no ``horizons`` wrapper), aligned with the
    ``path_placebo_event_study`` and global ``linear_trends_effects``
    shapes. The outer keys match ``path_effects.keys()``; horizons that
    have NaN cumulated values still appear (for to_dataframe alignment).
    R parity: matches R ``did_multiplegt_dyn(..., by_path, trends_lin)``
    which returns cumulated ``Effect_l`` per path under single-baseline
    panels (validated against ``joiners_only_trends_lin`` empirically).
    """
    from diff_diff.utils import safe_inference

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected_paths, path_to_group_mask, _ = _enumerate_treatment_paths(
            D_mat=D_mat,
            first_switch_idx=first_switch_idx,
            N_mat=N_mat,
            L_max=L_max,
            by_path=by_path,
            paths_of_interest=paths_of_interest,
        )

    n_groups_total = D_mat.shape[0]
    S_arr = switch_direction.astype(float)

    path_cumulated: Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]] = {}

    for path in selected_paths:
        if path not in path_effects:
            continue
        switcher_mask = path_to_group_mask[path]
        # Per-group running sum of DID^{fd}_{g, l'} for path switchers
        running_per_group = np.zeros(n_groups_total)
        path_horizons_anal = path_effects[path].get("horizons", {})
        cumulated_path: Dict[int, Dict[str, Any]] = {}

        for l_h in range(1, L_max + 1):
            if l_h not in multi_horizon_dids:
                continue
            mh = multi_horizon_dids[l_h]
            did_g_l = mh.get("did_g_l")
            eligible_global = mh.get("eligible_mask")
            if did_g_l is None or eligible_global is None:
                cumulated_path[l_h] = {
                    "effect": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": 0,
                }
                continue
            # Add this horizon's per-group DID to running sum (NaN -> 0
            # for accumulation, matching the global cumulation pattern)
            increment = np.where(np.isfinite(did_g_l), did_g_l, 0.0)
            running_per_group += increment
            # Path-restricted eligible mask at horizon l
            eligible_path = switcher_mask & eligible_global
            n_l_path = int(eligible_path.sum())
            if n_l_path == 0:
                cumulated_path[l_h] = {
                    "effect": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": 0,
                }
                continue
            cum_effect = float(
                np.sum(S_arr[eligible_path] * running_per_group[eligible_path]) / n_l_path
            )
            # Conservative SE upper bound: sum of per-horizon component
            # SEs from path_effects (matches global formula at :3402-3413).
            # NaN-consistency: any non-finite component SE -> cumulated NaN.
            component_ses = [
                path_horizons_anal.get(ll, {}).get("se", float("nan")) for ll in range(1, l_h + 1)
            ]
            if all(np.isfinite(s) for s in component_ses):
                cum_se = float(sum(component_ses))
            else:
                cum_se = float("nan")
            cum_t, cum_p, cum_ci = safe_inference(
                cum_effect,
                cum_se,
                alpha=alpha,
                df=df_inference,
            )
            cumulated_path[l_h] = {
                "effect": cum_effect,
                "se": cum_se,
                "t_stat": cum_t,
                "p_value": cum_p,
                "conf_int": cum_ci,
                "n_obs": n_l_path,
            }

        path_cumulated[path] = cumulated_path

    return path_cumulated


def _compute_path_placebos(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    eligible_mask_var: np.ndarray,
    multi_horizon_placebos: Dict[int, Dict[str, Any]],
    alpha: float,
    df_inference: Optional[int] = None,
    set_ids: Optional[np.ndarray] = None,
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
    obs_survey_info: Optional[Dict[str, Any]] = None,
    eligible_groups: Optional[List[Any]] = None,
    replicate_n_valid_list: Optional[List[int]] = None,
) -> Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]]:
    """
    Compute per-path backward-horizon placebos ``DID^{pl}_{path, l}``.

    Sibling of ``_compute_path_effects``: walks the same path enumeration
    and cohort-id pipeline but loops over backward horizons (lag
    ``l = 1..L_max``) using ``_compute_per_group_if_placebo_horizon``
    with the new ``switcher_subset_mask`` parameter to zero out switcher
    contributions for groups not in the selected path. SE depends on
    ``obs_survey_info`` exactly like ``_compute_path_effects``:

    * Non-survey: cohort-recentered plug-in with path-specific divisor
      ``N^{pl}_{l, path}`` (joiners/leavers IF precedent applied
      backward).
    * Survey: the path-restricted per-period IF is routed through
      ``_survey_se_from_group_if`` (analytical Binder TSL cell-period
      allocator; replicate-weight designs use the cell allocator
      unconditionally). Under replicate weights, every per-(path, lag)
      fit appends ``n_valid`` to ``replicate_n_valid_list``, and the
      shared post-call ``_refresh_path_inference`` re-runs
      ``safe_inference`` on every populated entry so the stored
      inference fields use the final ``df_survey``.

    Inner-dict keys are **negative** ints (-l for lag l) to match the
    overall ``placebo_event_study`` convention, so a unified
    ``{**path_effects[p]["horizons"], **path_placebo_event_study[p]}``
    view is well-formed.

    Returns ``{path: {-l: {effect, se, t_stat, p_value, conf_int,
    n_obs}}}`` directly (no ``n_groups`` / ``frequency_rank`` wrapper —
    those are already on ``path_effects[path]``; the rendering layer
    sorts by that rank). Returns ``{}`` when ``by_path`` was requested
    but no path has a complete window (mirrors ``_compute_path_effects``);
    the empty dict is the "requested but empty" sentinel distinct from
    ``None``.

    Inherits the cross-path cohort-sharing SE deviation from R that
    PR #360 documented for ``_compute_path_effects`` (full-panel
    cohort-centered plug-in vs R's per-path re-run): tracks R within
    numerical tolerance on single-path cohort panels; diverges on
    cohort-mixed panels. See ``Note (Phase 3 by_path ...)`` in
    ``docs/methodology/REGISTRY.md``.

    The ``_enumerate_treatment_paths`` call here is wrapped in
    ``warnings.catch_warnings`` to suppress the overflow ``UserWarning``
    duplicate — the analytical event-study pass
    (``_compute_path_effects``) has already surfaced that warning to
    the caller.
    """
    from diff_diff.utils import safe_inference

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected_paths, path_to_group_mask, _ = _enumerate_treatment_paths(
            D_mat=D_mat,
            first_switch_idx=first_switch_idx,
            N_mat=N_mat,
            L_max=L_max,
            by_path=by_path,
            paths_of_interest=paths_of_interest,
        )

    if not selected_paths:
        return {}

    n_groups = D_mat.shape[0]
    cohort_keys = [
        (
            float(baselines[g]),
            int(first_switch_idx[g]),
            int(switch_direction[g]),
        )
        for g in range(n_groups)
    ]
    unique_c: Dict[Tuple[float, int, int], int] = {}
    cid = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask_var[g]:
            cid[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_c:
            unique_c[key] = len(unique_c)
        cid[g] = unique_c[key]
    cohort_id_eligible = cid[eligible_mask_var]

    path_placebos: Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]] = {}

    for path in selected_paths:
        switcher_mask = path_to_group_mask[path]

        per_path_pl_if = _compute_per_group_if_placebo_horizon(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            baselines=baselines,
            first_switch_idx=first_switch_idx,
            switch_direction=switch_direction,
            T_g=T_g,
            L_max=L_max,
            set_ids=set_ids,
            compute_per_period=(obs_survey_info is not None),
            switcher_subset_mask=switcher_mask,
        )

        horizons: Dict[int, Dict[str, Any]] = {}
        for lag_l in range(1, L_max + 1):
            U_pl_l_path, U_pp_pl_l_path = per_path_pl_if[lag_l]

            pl_data = multi_horizon_placebos.get(lag_l)
            if pl_data is None:
                n_pl_l_path = 0
            else:
                eligible_mask_pl = pl_data.get("eligible_mask")
                if eligible_mask_pl is None:
                    n_pl_l_path = 0
                else:
                    n_pl_l_path = int(np.sum(switcher_mask & eligible_mask_pl))

            if n_pl_l_path == 0:
                horizons[-lag_l] = {
                    "effect": float("nan"),
                    "se": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                    "n_obs": 0,
                }
                continue

            U_pl_l_path_elig = U_pl_l_path[eligible_mask_var]
            effect_pl_path = float(U_pl_l_path.sum() / n_pl_l_path)

            U_centered_pl_path = _cohort_recenter(U_pl_l_path_elig, cohort_id_eligible)
            if obs_survey_info is None:
                se_pl_path = _plugin_se(U_centered=U_centered_pl_path, divisor=n_pl_l_path)
            else:
                assert U_pp_pl_l_path is not None
                assert eligible_groups is not None
                U_pp_pl_l_path_elig = U_pp_pl_l_path[eligible_mask_var]
                U_centered_pp_pl_path = _cohort_recenter_per_period(
                    U_pp_pl_l_path_elig, cohort_id_eligible
                )
                U_pl_scaled = U_centered_pl_path / n_pl_l_path
                U_pp_pl_scaled = U_centered_pp_pl_path / n_pl_l_path
                se_pl_path, n_valid_pl_replicates = _survey_se_from_group_if(
                    U_centered=U_pl_scaled,
                    eligible_groups=eligible_groups,
                    obs_survey_info=obs_survey_info,
                    U_centered_per_period=U_pp_pl_scaled,
                )
                if n_valid_pl_replicates is not None and replicate_n_valid_list is not None:
                    replicate_n_valid_list.append(n_valid_pl_replicates)

            if np.isnan(se_pl_path) and U_centered_pl_path.size > 0 and n_pl_l_path > 0:
                warnings.warn(
                    f"Cohort-recentered analytical variance is "
                    f"unidentified for path={path} at placebo lag "
                    f"l={lag_l}: the path-subset centered placebo "
                    f"influence function is identically zero (every "
                    f"variance-eligible path switcher forms its own "
                    f"(D_{{g,1}}, F_g, S_g) cohort, or the path has a "
                    f"single contributing group). DID^{{pl}}_{{path,l}} "
                    f"point estimate is still valid; SE / t_stat / "
                    f"p_value / conf_int are NaN-consistent. Rare paths "
                    f"with few contributing groups routinely hit this "
                    f"case at placebo horizons.",
                    UserWarning,
                    stacklevel=2,
                )

            t_pl, p_pl, ci_pl = safe_inference(
                effect_pl_path, se_pl_path, alpha=alpha, df=df_inference
            )

            horizons[-lag_l] = {
                "effect": effect_pl_path,
                "se": se_pl_path,
                "t_stat": t_pl,
                "p_value": p_pl,
                "conf_int": ci_pl,
                "n_obs": n_pl_l_path,
            }

        path_placebos[path] = horizons

    return path_placebos


def _compute_path_heterogeneity_test(
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    X_het: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    paths_of_interest: Optional[List[Tuple[int, ...]]],
    D_mat: np.ndarray,
    placebo: int = 0,
    alpha: float = 0.05,
    rank_deficient_action: str = "warn",
    group_ids_order: Optional[np.ndarray] = None,
    obs_survey_info: Optional[Dict[str, Any]] = None,
    replicate_n_valid_list: Optional[List[int]] = None,
) -> Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]]:
    """Per-path heterogeneity test (Web Appendix Section 1.5, Lemma 7).

    For each selected path ``p``, runs ``_compute_heterogeneity_test`` on
    the path-restricted switcher subsample. Cohort dummies absorb baseline
    by construction, so the path-restricted regression is methodologically
    well-posed even when path switchers span multiple baselines.

    Mirrors R ``did_multiplegt_dyn(..., by_path, predict_het)`` semantics:
    the R per-path dispatcher re-runs ``did_multiplegt_main(...,
    predict_het=...)`` on each path-restricted subsample, which is exactly
    what this helper does in Python.

    When ``placebo > 0``, ``_compute_heterogeneity_test`` also emits
    per-path heterogeneity on backward (placebo) horizons. Inner-dict keys
    are negative ints ``-1..-placebo`` to match the global
    ``heterogeneity_effects`` convention and the existing per-path
    placebo allocator at ``_compute_path_placebos`` (negative keys for
    unified ``{**positive, **negative}`` dict merging downstream).

    The ``_enumerate_treatment_paths`` call here re-derives the path
    enumeration (already computed elsewhere in fit() for ``path_effects``).
    The call is wrapped in ``warnings.catch_warnings()`` to suppress
    duplicate unobserved-path / by_path-exceeds-observed warnings; the
    upstream ``_compute_path_effects`` call already surfaced them.

    Returns
    -------
    Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]
        ``{path: {l: {beta, se, t_stat, p_value, conf_int, n_obs}}}``.
        Returns ``{}`` if ``selected_paths`` is empty. Return type is
        ``Optional[...]`` for caller-contract symmetry; the helper itself
        never produces ``None`` (the empty-state distinction
        ``None`` not requested vs ``{}`` requested but empty lives at
        the caller site in ``fit()``).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected_paths, path_to_group_mask, _ = _enumerate_treatment_paths(
            D_mat=D_mat,
            first_switch_idx=first_switch_idx,
            N_mat=N_mat,
            L_max=L_max,
            by_path=by_path,
            paths_of_interest=paths_of_interest,
        )
    out: Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]] = {}
    for path in selected_paths:
        mask = path_to_group_mask[path]
        path_groups: Set[int] = {int(g) for g in np.flatnonzero(mask)}
        out[path] = _compute_heterogeneity_test(
            Y_mat=Y_mat,
            N_mat=N_mat,
            baselines=baselines,
            first_switch_idx=first_switch_idx,
            switch_direction=switch_direction,
            T_g=T_g,
            X_het=X_het,
            L_max=L_max,
            placebo=placebo,
            alpha=alpha,
            rank_deficient_action=rank_deficient_action,
            group_ids_order=group_ids_order,
            obs_survey_info=obs_survey_info,
            replicate_n_valid_list=replicate_n_valid_list,
            path_groups=path_groups,
        )
    return out


def _collect_path_bootstrap_inputs(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    eligible_mask_var: np.ndarray,
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    path_effects: Dict[Tuple[int, ...], Dict[str, Any]],
    set_ids: Optional[np.ndarray] = None,
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
) -> Dict[Tuple[int, ...], Dict[int, Tuple[np.ndarray, int, float, None]]]:
    """
    Collect per-(path, horizon) inputs for the bootstrap mixin.

    Walks the same path enumeration / per-path IF / cohort-recentering
    pipeline that ``_compute_path_effects`` uses, but returns the
    intermediate ``(U_centered_path, n_l_path, effect_path)`` triples
    needed by ``_compute_dcdh_bootstrap``. Lives as a sibling of
    ``_compute_path_effects`` (not extending it) to keep the
    already-large analytical helper focused and avoid a polymorphic
    return shape. The bootstrap-only recomputation is O(n_paths *
    n_groups * L_max), which is small compared to the bootstrap draw
    loop itself.

    The point estimate per ``(path, horizon)`` is read from
    ``path_effects`` to stay bit-identical with the analytical pass;
    the bootstrap distribution gets centered on this value by
    ``_bootstrap_one_target`` downstream.

    Returns a nested dict ``{path: {horizon: (U_centered, n, effect, None)}}``;
    the 4th slot is always ``None`` because the multiplier-bootstrap
    path under ``survey_design + by_path`` is gated out at fit-time
    (``n_bootstrap > 0`` + ``survey_design`` + per-path selectors raises
    ``NotImplementedError``); analytical and replicate-weight survey
    SE under per-path selectors flows through ``_compute_path_effects``
    directly and does not reach this helper.

    ``_enumerate_treatment_paths`` is called again here (the analytical
    pass already called it inside ``_compute_path_effects``). The
    enumeration result is deterministic given identical arguments, so
    the selected paths and group masks will match bit-for-bit. The
    surrounding ``warnings.catch_warnings`` suppresses the overflow
    ``UserWarning`` from the re-enumeration — the analytical pass has
    already surfaced that warning to the caller, and re-emitting it
    from the bootstrap helper would be a spurious duplicate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected_paths, path_to_group_mask, _ = _enumerate_treatment_paths(
            D_mat=D_mat,
            first_switch_idx=first_switch_idx,
            N_mat=N_mat,
            L_max=L_max,
            by_path=by_path,
            paths_of_interest=paths_of_interest,
        )

    n_groups = D_mat.shape[0]
    cohort_keys = [
        (
            float(baselines[g]),
            int(first_switch_idx[g]),
            int(switch_direction[g]),
        )
        for g in range(n_groups)
    ]
    unique_c: Dict[Tuple[float, int, int], int] = {}
    cid = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask_var[g]:
            cid[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_c:
            unique_c[key] = len(unique_c)
        cid[g] = unique_c[key]
    cohort_id_eligible = cid[eligible_mask_var]

    path_bootstrap_inputs: Dict[Tuple[int, ...], Dict[int, Tuple[np.ndarray, int, float, None]]] = (
        {}
    )

    for path in selected_paths:
        switcher_mask = path_to_group_mask[path]

        per_path_if = _compute_per_group_if_multi_horizon(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            baselines=baselines,
            first_switch_idx=first_switch_idx,
            switch_direction=switch_direction,
            T_g=T_g,
            L_max=L_max,
            set_ids=set_ids,
            compute_per_period=False,
            switcher_subset_mask=switcher_mask,
        )

        horizon_inputs: Dict[int, Tuple[np.ndarray, int, float, None]] = {}
        path_analytical = path_effects.get(path)
        if path_analytical is None:
            continue

        for l_h in range(1, L_max + 1):
            U_l_path, _ = per_path_if[l_h]
            did_g_l = multi_horizon_dids[l_h].get("did_g_l")
            if did_g_l is None:
                continue
            n_l_path = int(np.sum(switcher_mask & ~np.isnan(did_g_l)))
            if n_l_path == 0:
                continue

            U_l_path_elig = U_l_path[eligible_mask_var]
            U_centered_path = _cohort_recenter(U_l_path_elig, cohort_id_eligible)

            effect_path = float(path_analytical["horizons"][l_h]["effect"])
            horizon_inputs[l_h] = (U_centered_path, n_l_path, effect_path, None)

        if horizon_inputs:
            path_bootstrap_inputs[path] = horizon_inputs

    return path_bootstrap_inputs


def _collect_path_placebo_bootstrap_inputs(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    by_path: Optional[int],
    eligible_mask_var: np.ndarray,
    multi_horizon_placebos: Dict[int, Dict[str, Any]],
    path_placebos: Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]],
    set_ids: Optional[np.ndarray] = None,
    paths_of_interest: Optional[List[Tuple[int, ...]]] = None,
) -> Dict[Tuple[int, ...], Dict[int, Tuple[np.ndarray, int, float, None]]]:
    """
    Collect per-(path, lag) inputs for the placebo bootstrap mixin
    dispatch.

    Sibling of ``_collect_path_bootstrap_inputs``. Walks the same path
    enumeration / per-path placebo IF / cohort-recentering pipeline
    that ``_compute_path_placebos`` uses, but returns the
    ``(U_centered_path, n_pl_l_path, effect_pl_path)`` triples needed
    by ``_compute_dcdh_bootstrap``'s per-`(path, lag_l)` placebo
    dispatch block.

    Returned dict keys lag by **positive** int (l = 1..L_max), matching
    the inner-key convention of ``placebo_horizon_inputs`` already
    consumed by the bootstrap mixin. The propagation block in
    ``fit()`` translates back to negative-keyed
    ``path_placebo_event_study[path][-lag_l]`` post-bootstrap.

    The point estimate per ``(path, lag_l)`` is read from
    ``path_placebos[path][-lag_l]["effect"]`` (note: no ``["horizons"]``
    wrapper -- ``_compute_path_placebos`` returns the negative-keyed
    inner dict directly, unlike ``_compute_path_effects`` which wraps
    its horizons under a ``["horizons"]`` key) to stay bit-identical
    with the analytical pass; the bootstrap distribution gets centered
    on this value by ``_bootstrap_one_target`` downstream.

    The ``warnings.catch_warnings`` block suppresses the
    re-enumeration overflow ``UserWarning``; the analytical
    event-study pass (``_compute_path_effects``) already surfaced that
    warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected_paths, path_to_group_mask, _ = _enumerate_treatment_paths(
            D_mat=D_mat,
            first_switch_idx=first_switch_idx,
            N_mat=N_mat,
            L_max=L_max,
            by_path=by_path,
            paths_of_interest=paths_of_interest,
        )

    n_groups = D_mat.shape[0]
    cohort_keys = [
        (
            float(baselines[g]),
            int(first_switch_idx[g]),
            int(switch_direction[g]),
        )
        for g in range(n_groups)
    ]
    unique_c: Dict[Tuple[float, int, int], int] = {}
    cid = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask_var[g]:
            cid[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_c:
            unique_c[key] = len(unique_c)
        cid[g] = unique_c[key]
    cohort_id_eligible = cid[eligible_mask_var]

    path_placebo_bootstrap_inputs: Dict[
        Tuple[int, ...], Dict[int, Tuple[np.ndarray, int, float, None]]
    ] = {}

    for path in selected_paths:
        switcher_mask = path_to_group_mask[path]

        per_path_pl_if = _compute_per_group_if_placebo_horizon(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            baselines=baselines,
            first_switch_idx=first_switch_idx,
            switch_direction=switch_direction,
            T_g=T_g,
            L_max=L_max,
            set_ids=set_ids,
            compute_per_period=False,
            switcher_subset_mask=switcher_mask,
        )

        horizon_inputs: Dict[int, Tuple[np.ndarray, int, float, None]] = {}
        path_analytical = path_placebos.get(path)
        if path_analytical is None:
            continue

        for lag_l in range(1, L_max + 1):
            U_pl_l_path, _ = per_path_pl_if[lag_l]
            pl_data = multi_horizon_placebos.get(lag_l)
            if pl_data is None:
                continue
            eligible_mask_pl = pl_data.get("eligible_mask")
            if eligible_mask_pl is None:
                continue
            n_pl_l_path = int(np.sum(switcher_mask & eligible_mask_pl))
            if n_pl_l_path == 0:
                continue

            U_pl_l_path_elig = U_pl_l_path[eligible_mask_var]
            U_centered_pl_path = _cohort_recenter(U_pl_l_path_elig, cohort_id_eligible)

            effect_pl_path = float(path_analytical[-lag_l]["effect"])
            horizon_inputs[lag_l] = (
                U_centered_pl_path,
                n_pl_l_path,
                effect_pl_path,
                None,
            )

        if horizon_inputs:
            path_placebo_bootstrap_inputs[path] = horizon_inputs

    return path_placebo_bootstrap_inputs


def _compute_per_group_if_placebo_horizon(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
    compute_per_period: bool = True,
    switcher_subset_mask: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Compute per-group influence function for placebo horizons.

    Mirrors ``_compute_per_group_if_multi_horizon`` but for backward
    horizons, matching ``_compute_multi_horizon_placebos`` eligibility
    and control-pool logic exactly.

    For placebo lag ``l``, switcher ``g``'s contribution uses the
    backward outcome change ``Y_{g, F_g-1-l} - Y_{g, F_g-1}`` (paper
    convention: pre-period minus reference). Controls are identified
    by the **positive**-horizon cutoff ``F_{g'} > F_g - 1 + l`` AND
    observation at ``ref_idx``, ``backward_idx``, AND ``forward_idx``
    (the terminal-missingness guard from Phase 2 Round 9).

    Parameters
    ----------
    switcher_subset_mask : np.ndarray of bool, shape (n_groups,), optional
        When supplied, restricts the switcher iteration to groups where
        the mask is ``True``. Groups outside the subset contribute as
        controls only (their switcher-side contribution is skipped). The
        control pool is unchanged. Mirrors the same parameter on
        ``_compute_per_group_if_multi_horizon`` and is used by
        ``by_path`` placebos to zero out switcher contributions for
        groups not in the selected path. Default ``None`` preserves the
        legacy behavior of iterating over all switchers.

    Returns
    -------
    dict mapping lag l (positive int) -> (U_pl_l, U_per_period_pl_l) tuple
        - ``U_pl_l``: np.ndarray of shape (n_groups,). NOT cohort-
          centered; the caller applies ``_cohort_recenter()``.
        - ``U_per_period_pl_l``: np.ndarray of shape (n_groups, n_periods).
          Per-``(g, t)`` contributions attributed to ``t = backward_idx``
          (the pre-period observation that supports the contrast).
          Satisfies ``U_per_period_pl_l.sum(axis=1) == U_pl_l``.
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        U_pl = np.zeros(n_groups, dtype=float)
        U_per_period_pl: Optional[np.ndarray] = (
            np.zeros((n_groups, n_periods), dtype=float) if compute_per_period else None
        )

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            if switcher_subset_mask is not None and not switcher_subset_mask[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l
            forward_idx = ref_idx + l

            # Dual eligibility (matches _compute_multi_horizon_placebos)
            if backward_idx < 0 or forward_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, backward_idx] <= 0:
                continue
            if T_g[g] < forward_idx:
                continue

            d_base = float(baselines[g])
            S_g = float(switch_direction[g])

            # Control pool: same baseline, not switched by forward_idx,
            # observed at ref, backward, AND forward (terminal-missingness
            # guard). Matches _compute_multi_horizon_placebos exactly.
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > forward_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, backward_idx] > 0)
                & (N_mat[ctrl_indices, forward_idx] > 0)
            )
            # State-set trends: restrict controls to same set
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]
            n_ctrl = ctrl_pool.size

            if n_ctrl == 0:
                continue

            # Switcher contribution: paper convention backward - ref.
            # Attribute the whole contrast to the backward cell
            # (mirrors the multi-horizon / DID_M post-period
            # attribution convention).
            switcher_change = Y_mat[g, backward_idx] - Y_mat[g, ref_idx]
            U_pl[g] += S_g * switcher_change
            if U_per_period_pl is not None:
                U_per_period_pl[g, backward_idx] += S_g * switcher_change

            # Control contributions
            ctrl_changes = Y_mat[ctrl_pool, backward_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_contrib = (S_g / n_ctrl) * ctrl_changes
            U_pl[ctrl_pool] -= ctrl_contrib
            if U_per_period_pl is not None:
                U_per_period_pl[ctrl_pool, backward_idx] -= ctrl_contrib

        results[l] = (U_pl, U_per_period_pl)

    return results


def _compute_multi_horizon_placebos(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute dynamic placebo estimators ``DID^{pl}_l`` for ``l = 1..L_pl_max``.

    Mirrors ``_compute_multi_horizon_dids`` but looks BACKWARD from
    each group's reference period (Web Appendix Section 1.1, Lemma 5).

    **Dual eligibility condition:** a group g is eligible for placebo
    lag l iff:
    - ``F_g - 1 - l >= 0`` (enough pre-treatment history), AND
    - ``F_g - 1 + l <= T_g`` (positive-horizon control pool exists)

    The control set uses the *positive*-horizon cutoff:
    ``{g': D_{g',1} = D_{g,1}, F_{g'} > F_g - 1 + l}``.

    Returns
    -------
    dict mapping lag l (positive int) -> {
        "placebo_l": float,
        "N_pl_l": int,
        "eligible_mask": np.ndarray,
    }
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Dict[str, Any]] = {}
    a11_placebo_warnings: List[str] = []

    for l in range(1, L_max + 1):  # noqa: E741
        eligible = np.zeros(n_groups, dtype=bool)
        pl_g_l = np.full(n_groups, np.nan)

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l  # the pre-treatment outcome period
            forward_idx = ref_idx + l  # for control-pool eligibility

            # Dual eligibility: backward must be in range, forward must
            # have controls available
            if backward_idx < 0 or forward_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, backward_idx] <= 0:
                continue
            if T_g[g] < forward_idx:
                continue
            eligible[g] = True

        N_pl_l = int(eligible.sum())
        if N_pl_l == 0:
            results[l] = {
                "placebo_l": float("nan"),
                "N_pl_l": 0,
                "eligible_mask": eligible,
            }
            continue

        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l
            forward_idx = ref_idx + l
            d_base = float(baselines[g])

            # Switcher's backward outcome change: pre-period minus reference
            # (paper convention: Y_{F_g-1-l} - Y_{F_g-1})
            switcher_change = Y_mat[g, backward_idx] - Y_mat[g, ref_idx]

            # Control pool: same baseline, not switched by forward_idx,
            # AND observed at all three relevant periods (ref, backward,
            # AND forward - the last ensures terminally missing controls
            # don't leak into the placebo computation).
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > forward_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, backward_idx] > 0)
                & (N_mat[ctrl_indices, forward_idx] > 0)
            )
            # State-set trends: restrict controls to same set
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]

            if ctrl_pool.size == 0:
                eligible[g] = False
                a11_placebo_warnings.append(f"placebo lag {l}, group_idx {g}: no controls")
                continue

            ctrl_changes = Y_mat[ctrl_pool, backward_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_avg = float(ctrl_changes.mean())
            pl_g_l[g] = switcher_change - ctrl_avg

        # Recompute N_pl_l after control-pool exclusions
        N_pl_l = int(eligible.sum())
        if N_pl_l == 0:
            results[l] = {
                "placebo_l": float("nan"),
                "N_pl_l": 0,
                "eligible_mask": eligible,
            }
            continue

        S_eligible = switch_direction[eligible].astype(float)
        pl_g_eligible = pl_g_l[eligible]
        placebo_l = float((S_eligible * pl_g_eligible).sum() / N_pl_l)

        results[l] = {
            "placebo_l": placebo_l,
            "N_pl_l": N_pl_l,
            "eligible_mask": eligible,
        }

    if a11_placebo_warnings:
        results["_a11_warnings"] = a11_placebo_warnings  # type: ignore[index]  # sentinel str key in int-keyed dict

    return results


def _compute_normalized_effects(
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    D_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    L_max: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute normalized event-study effects ``DID^n_l = DID_l / delta^D_l``.

    Uses the general formula (Eq 15) that works for both binary and
    non-binary treatment (future-proofing for Phase 3).

    For binary treatment: ``delta^D_{g,l} = l`` (joiners) or ``-l``
    (leavers), so ``|delta^D_{g,l}| = l`` and ``DID^n_l = DID_l / l``.

    Returns
    -------
    dict mapping l -> {effect, denominator}
    """
    n_groups = D_mat.shape[0]
    results: Dict[int, Dict[str, Any]] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            results[l] = {"effect": float("nan"), "denominator": float("nan")}
            continue

        eligible = h["eligible_mask"]
        N_l = h["N_l"]
        did_l = h["did_l"]

        # Per-group incremental dose: delta^D_{g,l} = sum_{k=0}^{l-1} (D_{g,F_g+k} - D_{g,1})
        # General formula, works for non-binary treatment.
        delta_D_g = np.zeros(n_groups)
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            d_base = baselines[g]
            dose_sum = 0.0
            for k in range(l):
                col = f_g + k
                if col < D_mat.shape[1]:
                    dose_sum += D_mat[g, col] - d_base
            delta_D_g[g] = dose_sum

        # Aggregate dose denominator
        delta_D_l = float(np.abs(delta_D_g[eligible]).sum() / N_l)

        if delta_D_l <= 0:
            results[l] = {"effect": float("nan"), "denominator": 0.0}
            continue

        results[l] = {
            "effect": did_l / delta_D_l,
            "denominator": delta_D_l,
        }

    return results


def _compute_cost_benefit_delta(
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    D_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    L_max: int,
) -> Dict[str, Any]:
    """
    Compute the cost-benefit aggregate ``delta`` from Section 3.3, Lemma 4.

    ``delta = sum_l w_l * DID_l`` where
    ``w_l = N_l / sum_{g,l'} |D_{g,F_g-1+l'} - D_{g,1}|``.

    When leavers are present (Assumption 7 violated), also computes
    ``delta_joiners`` and ``delta_leavers`` separately.

    Returns
    -------
    dict with keys: delta, weights, has_leavers, delta_joiners, delta_leavers
    """

    # Per-horizon dose via Lemma 4: w_l uses the PER-PERIOD dose
    # D_{g,F_g-1+l} - D_{g,1} (NOT the cumulative delta^D_{g,l}).
    # For binary joiners this is 1 per (g,l) pair, so w_l = N_l / sum N_l'.
    total_dose = 0.0
    per_horizon_dose: Dict[int, float] = {}
    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            per_horizon_dose[l] = 0.0
            continue
        eligible = h["eligible_mask"]
        dose_l = 0.0
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            col = f_g - 1 + l
            if col < D_mat.shape[1]:
                dose_l += abs(float(D_mat[g, col] - baselines[g]))
        per_horizon_dose[l] = dose_l
        total_dose += dose_l

    if total_dose <= 0:
        return {
            "delta": float("nan"),
            "weights": {},
            "has_leavers": False,
            "delta_joiners": float("nan"),
            "delta_leavers": float("nan"),
        }

    # Horizon weights: w_l = N_l / total_dose (but using dose, not N_l)
    # Per Lemma 4: w_l = N_l * E[|delta^D_{g,l}|] / total_dose
    # which simplifies to per_horizon_dose[l] / total_dose
    weights: Dict[int, float] = {}
    delta = 0.0
    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            weights[l] = 0.0
            continue
        w_l = per_horizon_dose[l] / total_dose
        weights[l] = w_l
        delta += w_l * h["did_l"]

    # Check for leavers (Assumption 7 violation)
    has_leavers = bool(np.any(switch_direction < 0))

    delta_joiners = float("nan")
    delta_leavers = float("nan")
    if has_leavers:
        # Compute delta separately for joiners and leavers
        for direction, attr_name in [(1, "joiners"), (-1, "leavers")]:
            dir_dose = 0.0
            dir_horizon_dose: Dict[int, float] = {}
            for l in range(1, L_max + 1):  # noqa: E741
                h = multi_horizon_dids.get(l)
                if h is None or h["N_l"] == 0:
                    dir_horizon_dose[l] = 0.0
                    continue
                eligible = h["eligible_mask"]
                dose_l = 0.0
                for g in np.where(eligible)[0]:
                    if switch_direction[g] != direction:
                        continue
                    f_g = first_switch_idx[g]
                    col = f_g - 1 + l
                    if col < D_mat.shape[1]:
                        dose_l += abs(float(D_mat[g, col] - baselines[g]))
                dir_horizon_dose[l] = dose_l
                dir_dose += dose_l

            if dir_dose > 0:
                dir_delta = 0.0
                for l in range(1, L_max + 1):  # noqa: E741
                    h = multi_horizon_dids.get(l)
                    if h is None or h["N_l"] == 0:
                        continue
                    eligible = h["eligible_mask"]
                    # Per-direction DID_l
                    dir_eligible = eligible & (switch_direction == direction)
                    n_dir = int(dir_eligible.sum())
                    if n_dir == 0:
                        continue
                    did_g_l = h["did_g_l"]
                    S = switch_direction[dir_eligible].astype(float)
                    did_l_dir = float((S * did_g_l[dir_eligible]).sum() / n_dir)
                    w_dir = dir_horizon_dose[l] / dir_dose
                    dir_delta += w_dir * did_l_dir
                if attr_name == "joiners":
                    delta_joiners = dir_delta
                else:
                    delta_leavers = dir_delta

    return {
        "delta": delta,
        "weights": weights,
        "has_leavers": has_leavers,
        "delta_joiners": delta_joiners,
        "delta_leavers": delta_leavers,
    }


def _compute_full_per_group_contributions(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    n_10_t_arr: np.ndarray,
    n_00_t_arr: np.ndarray,
    n_01_t_arr: np.ndarray,
    n_11_t_arr: np.ndarray,
    a11_plus_zeroed_arr: np.ndarray,
    a11_minus_zeroed_arr: np.ndarray,
    side: str = "overall",
    compute_per_period: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the per-group influence function ``U^G_g`` for ``DID_M``,
    ``DID_+``, or ``DID_-`` by summing role-weighted outcome differences
    across all periods (full ``Lambda^G_{g,l=1}`` from Section 3.7.2 of
    the dynamic companion paper, evaluated at horizon ``l = 1``).

    Decomposition (for ``side='overall'``)::

        N_S * DID_M = sum_t [
              sum_{g in joiners(t)}  (Y_{g,t} - Y_{g,t-1})
            - (n_10_t / n_00_t) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})
            + (n_01_t / n_11_t) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
            - sum_{g in leavers(t)}  (Y_{g,t} - Y_{g,t-1})
        ]

    Each ``(g, t)`` cell contributes to ``U^G_g`` once per period, with
    the role weight determined by its ``(D_{g,t-1}, D_{g,t})`` transition.
    A switching group typically contributes from MULTIPLE periods (its
    own switch period + every period where it serves as a stable
    control); a never-switching group contributes only via its stable-
    control roles (which can be non-zero when it serves as a control
    for other cohorts' switches).

    Periods where ``DID_+,t`` or ``DID_-,t`` were zeroed under the A11
    convention contribute zero on the affected side, matching the
    point estimate.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted treatment, outcome, and observation-count matrices.
    n_10_t_arr, n_00_t_arr, n_01_t_arr, n_11_t_arr : np.ndarray
        Per-period CELL counts aligned to ``periods[1:]``.
    a11_plus_zeroed_arr, a11_minus_zeroed_arr : np.ndarray of bool
        Per-period A11-zeroing flags aligned to ``periods[1:]``.
    side : {"overall", "joiners", "leavers"}
        Which contribution to compute:

        - ``"overall"``: returns ``U^G_g`` such that ``U.sum() == N_S * DID_M``
        - ``"joiners"``: returns ``U^G_g`` such that ``U.sum() == joiner_total * DID_+``
          (only the joiners + stable_0 terms)
        - ``"leavers"``: returns ``U^G_g`` such that ``U.sum() == leaver_total * DID_-``
          (only the leavers + stable_1 terms, with the leavers side's sign convention)

    Returns
    -------
    U : np.ndarray of shape (n_groups,)
        Per-group contributions. NOT cohort-centered; the caller is
        responsible for centering before computing the SE.
    U_per_period : np.ndarray of shape (n_groups, n_periods) or ``None``
        Per-``(g, t)``-cell contributions, attributed to the post-
        period ``t`` of each transition pair. Satisfies
        ``U_per_period.sum(axis=1) == U`` exactly. NOT cohort-centered.
        Used by the cell-period IF allocator in
        ``_survey_se_from_group_if`` so that PSU/strata that vary
        across the cells of g can be honored by design-based variance.
        ``None`` when ``compute_per_period=False`` (the caller has no
        survey design, so the allocator is not needed and building the
        dense O(n_groups * n_periods) tensor would be wasted work).
    """
    if side not in ("overall", "joiners", "leavers"):
        raise ValueError(f"side must be one of overall/joiners/leavers, got {side!r}")

    n_groups, n_periods = D_mat.shape
    U = np.zeros(n_groups, dtype=float)
    U_per_period: Optional[np.ndarray] = (
        np.zeros((n_groups, n_periods), dtype=float) if compute_per_period else None
    )

    include_joiners_side = side in ("overall", "joiners")
    include_leavers_side = side in ("overall", "leavers")

    # Per-cell attribution convention (not a derivation from the
    # observation-level survey linearization — see REGISTRY.md
    # ``ChaisemartinDHaultfoeuille`` Note on survey IF expansion):
    # attribute each (Y_curr - Y_prev) transition as a single
    # difference to its post-period cell (g, t_idx). Preserves the
    # row-sum identity U_per_period.sum(axis=1) == U and therefore
    # the group-sum invariance that makes the cell expansion
    # byte-identical to the pre-allocator convention under PSU=group.
    for t_idx in range(1, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        y_diff = Y_mat[:, t_idx] - Y_mat[:, t_idx - 1]
        n_curr = N_mat[:, t_idx]
        n_prev = N_mat[:, t_idx - 1]
        present = (n_curr > 0) & (n_prev > 0)

        joiner_mask = (d_prev == 0) & (d_curr == 1) & present
        stable0_mask = (d_prev == 0) & (d_curr == 0) & present
        leaver_mask = (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_prev == 1) & (d_curr == 1) & present

        n_10_t = int(n_10_t_arr[t_idx - 1])
        n_00_t = int(n_00_t_arr[t_idx - 1])
        n_01_t = int(n_01_t_arr[t_idx - 1])
        n_11_t = int(n_11_t_arr[t_idx - 1])

        # Joiners side (+y_diff for joiners; -(n_10/n_00)*y_diff for stable_0)
        if (
            include_joiners_side
            and not bool(a11_plus_zeroed_arr[t_idx - 1])
            and n_10_t > 0
            and n_00_t > 0
        ):
            U[joiner_mask] += y_diff[joiner_mask]
            U[stable0_mask] -= (n_10_t / n_00_t) * y_diff[stable0_mask]
            if U_per_period is not None:
                U_per_period[joiner_mask, t_idx] += y_diff[joiner_mask]
                U_per_period[stable0_mask, t_idx] -= (n_10_t / n_00_t) * y_diff[stable0_mask]

        # Leavers side (-y_diff for leavers; +(n_01/n_11)*y_diff for stable_1)
        if (
            include_leavers_side
            and not bool(a11_minus_zeroed_arr[t_idx - 1])
            and n_01_t > 0
            and n_11_t > 0
        ):
            U[leaver_mask] -= y_diff[leaver_mask]
            U[stable1_mask] += (n_01_t / n_11_t) * y_diff[stable1_mask]
            if U_per_period is not None:
                U_per_period[leaver_mask, t_idx] -= y_diff[leaver_mask]
                U_per_period[stable1_mask, t_idx] += (n_01_t / n_11_t) * y_diff[stable1_mask]

    return U, U_per_period


def _cohort_recenter(
    U: np.ndarray,
    cohort_ids: np.ndarray,
) -> np.ndarray:
    """
    Subtract cohort-conditional means from U.

    For each cohort id, computes ``U_bar_k = mean(U[cohort==k])`` and
    returns ``U - U_bar_{cohort(g)}``. This is the per-group cohort-
    recentering step from Web Appendix Section 3.7.3 of the dynamic
    companion paper. Critical: subtracts the cohort mean, NOT a single
    grand mean — using a grand mean silently produces a smaller,
    incorrect variance.
    """
    U_centered = U.astype(float).copy()
    if U.size == 0:
        return U_centered
    unique_cohorts = np.unique(cohort_ids)
    for k in unique_cohorts:
        in_cohort = cohort_ids == k
        if in_cohort.any():
            U_centered[in_cohort] = U[in_cohort] - U[in_cohort].mean()
    return U_centered


def _cohort_recenter_per_period(
    U_per_period: np.ndarray,
    cohort_ids: np.ndarray,
) -> np.ndarray:
    """
    Column-wise cohort recentering of a per-(g, t) IF attribution.

    For each period column t, subtract the cohort-conditional mean of
    that column: ``U_centered[g, t] = U[g, t] - mean_{g' in cohort(g)}
    U[g', t]``. The row sum identity is preserved:

        sum_t U_centered[g, t]
            = sum_t U[g, t] - sum_t cohort_mean[cohort(g), t]
            = U[g] - cohort_mean_of_U[cohort(g)]
            = U_centered[g]

    Hence the cell-level Binder TSL aggregation telescopes to the same
    group-level sum produced by ``_cohort_recenter`` on ``U``, giving
    byte-identical variance under within-group-constant PSU (old
    validator's accepted input set) and a per-cell attribution under
    within-group-varying PSU that follows the library's post-period
    convention (see the REGISTRY.md Note on survey IF expansion — a
    formal derivation from the observation-level survey linearization
    is still open, tracked in ``DEFERRED.md``).
    """
    U_centered = U_per_period.astype(float).copy()
    if U_per_period.size == 0:
        return U_centered
    unique_cohorts = np.unique(cohort_ids)
    for k in unique_cohorts:
        in_cohort = cohort_ids == k
        if in_cohort.any():
            # Per-period mean across groups in this cohort
            col_means = U_per_period[in_cohort].mean(axis=0)
            U_centered[in_cohort] = U_per_period[in_cohort] - col_means[np.newaxis, :]
    return U_centered


def _compute_cohort_recentered_inputs(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    n_10_t_arr: np.ndarray,
    n_00_t_arr: np.ndarray,
    n_01_t_arr: np.ndarray,
    n_11_t_arr: np.ndarray,
    a11_plus_zeroed_arr: np.ndarray,
    a11_minus_zeroed_arr: np.ndarray,
    all_groups: List[Any],
    singleton_baseline_groups: List[Any],
    compute_per_period: bool = True,
) -> Tuple[
    np.ndarray,  # U_centered_overall (n_eligible,)
    int,  # n_groups_for_overall
    int,  # n_cohorts
    int,  # n_groups_dropped_never_switching
    np.ndarray,  # U_centered_joiners
    np.ndarray,  # U_centered_leavers
    List[Any],  # eligible_group_ids
    Optional[np.ndarray],  # U_centered_per_period_overall (n_eligible, n_periods) or None
    Optional[np.ndarray],  # U_centered_per_period_joiners or None
    Optional[np.ndarray],  # U_centered_per_period_leavers or None
]:
    """
    Compute the cohort-centered influence-function vectors for variance.

    Implements the full ``Lambda^G_{g,l=1}`` weight vector from
    Section 3.7.2 of the dynamic companion paper (NBER WP 29873) at
    horizon ``l = 1``: each group's per-period role weights (joiner,
    stable_0, leaver, stable_1) sum to a per-group ``U^G_g`` value
    that, summed across groups, recovers ``N_S * DID_M``.

    Cohorts are defined by the triple ``(D_{g,1}, F_g, S_g)`` where
    ``F_g`` is the first switch period and ``S_g`` is the switch
    direction (+1 joiner, -1 leaver, 0 never-switching). Never-
    switching groups form their own cohorts indexed by baseline only.

    Per footnote 15 of the dynamic paper (passed in via
    ``singleton_baseline_groups``), groups whose baseline ``D_{g,1}``
    value is unique in the post-drop panel have no cohort peer and are
    excluded from the variance computation only. They remain in the
    point-estimate sample as period-based stable controls (this
    matches Python's documented period-vs-cohort stable-control
    interpretation; the cell DataFrame entering ``_compute_per_period_dids``
    retains them).

    Returns
    -------
    U_centered_overall : np.ndarray
        Cohort-centered IF vector for ``DID_M`` over the variance-
        eligible groups (post-singleton-filter).
    n_groups_for_overall : int
        ``U_centered_overall.size`` for sanity-checking by the caller.
    n_cohorts : int
        Distinct cohorts in the variance-eligible group set.
    n_groups_dropped_never_switching : int
        Count of never-switching groups for results metadata. (They
        ARE included in the variance computation under the full IF
        formula because they can have non-zero contributions when
        serving as stable controls; this count is reported for
        backwards compatibility with the existing results dataclass
        field but no longer represents an actual exclusion.)
    U_centered_joiners : np.ndarray
        Cohort-centered IF vector for ``DID_+`` (joiners-only side).
    U_centered_leavers : np.ndarray
        Cohort-centered IF vector for ``DID_-`` (leavers-only side).
    """
    n_groups, n_periods = D_mat.shape

    if n_groups == 0:
        empty_pp: Optional[np.ndarray] = (
            np.zeros((0, 0), dtype=float) if compute_per_period else None
        )
        return (
            np.array([], dtype=float),
            0,
            0,
            0,
            np.array([], dtype=float),
            np.array([], dtype=float),
            [],
            empty_pp,
            empty_pp,
            empty_pp,
        )

    # Per-group switch metadata via the shared helper (factored out in
    # Phase 2 so both the cohort-recentered IF path and the multi-
    # horizon DID_{g,l} path share the same computation).
    baselines, first_switch_idx, switch_direction, _T_g = _compute_group_switch_metadata(
        D_mat, N_mat
    )

    n_groups_dropped_never_switching = int((switch_direction == 0).sum())

    # Variance-eligibility mask: include all groups EXCEPT singleton-
    # baseline groups (footnote 15) which have no cohort peer.
    singleton_baseline_set = set(singleton_baseline_groups)
    eligible_mask = np.array([g not in singleton_baseline_set for g in all_groups], dtype=bool)

    # Cohort identification: (D_{g,1}, F_g, S_g) triples for the
    # variance-eligible group set. Never-switching groups (S_g = 0)
    # have F_g = -1 and form cohorts indexed by baseline alone.
    cohort_keys = [
        (float(baselines[g]), int(first_switch_idx[g]), int(switch_direction[g]))
        for g in range(n_groups)
    ]
    unique_cohorts: Dict[Tuple[float, int, int], int] = {}
    cohort_id = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask[g]:
            cohort_id[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_cohorts:
            unique_cohorts[key] = len(unique_cohorts)
        cohort_id[g] = unique_cohorts[key]
    n_cohorts = len(unique_cohorts)

    # Compute the full IF vectors + per-period attributions via the
    # helper. Skip the O(n_groups * n_periods) per-period tensor
    # allocation when the caller won't use it (no survey design).
    U_overall_full, U_per_period_overall_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="overall",
        compute_per_period=compute_per_period,
    )
    U_joiners_full, U_per_period_joiners_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="joiners",
        compute_per_period=compute_per_period,
    )
    U_leavers_full, U_per_period_leavers_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="leavers",
        compute_per_period=compute_per_period,
    )

    # Restrict to variance-eligible groups (drop singleton-baseline groups)
    U_overall = U_overall_full[eligible_mask]
    U_joiners = U_joiners_full[eligible_mask]
    U_leavers = U_leavers_full[eligible_mask]
    cohort_id_eligible = cohort_id[eligible_mask]

    # Cohort-recenter each IF vector (group level for plug-in path)
    U_centered_overall = _cohort_recenter(U_overall, cohort_id_eligible)
    U_centered_joiners = _cohort_recenter(U_joiners, cohort_id_eligible)
    U_centered_leavers = _cohort_recenter(U_leavers, cohort_id_eligible)

    # Per-period cohort recentering for the cell-period survey allocator.
    # Column-wise centering preserves sum_t U_centered_pp[g, t] =
    # U_centered[g], so Binder TSL PSU-level aggregation telescopes to
    # the same group-level sum under within-group-constant PSU (byte-
    # identical to the old allocator). Under within-group-varying PSU
    # the per-cell attribution follows the library's post-period
    # convention — a formal derivation from the observation-level
    # survey linearization is an open question tracked in DEFERRED.md.
    # Skip entirely when the 2D tensor was not computed.
    if U_per_period_overall_full is not None:
        U_centered_pp_overall: Optional[np.ndarray] = _cohort_recenter_per_period(
            U_per_period_overall_full[eligible_mask], cohort_id_eligible
        )
    else:
        U_centered_pp_overall = None
    if U_per_period_joiners_full is not None:
        U_centered_pp_joiners: Optional[np.ndarray] = _cohort_recenter_per_period(
            U_per_period_joiners_full[eligible_mask], cohort_id_eligible
        )
    else:
        U_centered_pp_joiners = None
    if U_per_period_leavers_full is not None:
        U_centered_pp_leavers: Optional[np.ndarray] = _cohort_recenter_per_period(
            U_per_period_leavers_full[eligible_mask], cohort_id_eligible
        )
    else:
        U_centered_pp_leavers = None

    # Eligible group IDs for survey IF expansion
    eligible_group_ids = [all_groups[g] for g in range(n_groups) if eligible_mask[g]]

    return (
        U_centered_overall,
        U_centered_overall.size,
        n_cohorts,
        n_groups_dropped_never_switching,
        U_centered_joiners,
        U_centered_leavers,
        eligible_group_ids,
        U_centered_pp_overall,
        U_centered_pp_joiners,
        U_centered_pp_leavers,
    )


def _plugin_se(U_centered: np.ndarray, divisor: int) -> float:
    """
    Compute the cohort-recentered plug-in standard error.

    Implements ``SE = sqrt(sum_g U_centered[g]^2 / N_l) / sqrt(N_l)``,
    which is the simplified form of Section 3.7.3's plug-in formula
    after the cohort recentering has been applied to ``U_centered``.

    The plain ``(1/N_l) * sum_g U_centered^2 / N_l`` form gives the
    variance; we take its square root for the SE.

    Returns ``NaN`` in three degenerate cases:

    1. ``U_centered`` is empty (no variance-eligible groups).
    2. ``divisor <= 0`` (no switching cells in N_S).
    3. ``sum(U_centered**2) <= 0`` — every cohort is a singleton, so
       cohort recentering produces an identically-zero centered IF
       vector and the variance is unidentified. The caller should
       detect this case (NaN return + non-empty input) and emit a
       user-facing warning explaining the degenerate-cohort condition.
       Returning ``NaN`` rather than ``0.0`` prevents the silently
       implies-infinite-precision failure mode.
    """
    n = U_centered.size
    if n == 0 or divisor <= 0:
        return float("nan")
    sum_sq = float((U_centered**2).sum())
    if sum_sq <= 0:
        # Degenerate-cohort case: every cohort is a singleton, so
        # cohort recentering produces all zeros. The variance is
        # unidentified — return NaN rather than 0.0 so downstream
        # inference is NaN-consistent and the caller surfaces a
        # warning. See the **Note** in REGISTRY.md
        # ChaisemartinDHaultfoeuille.
        return float("nan")
    sigma_hat_sq = sum_sq / divisor
    if not np.isfinite(sigma_hat_sq) or sigma_hat_sq < 0:
        return float("nan")
    return float(np.sqrt(sigma_hat_sq) / np.sqrt(divisor))


def _strata_psu_vary_within_group(
    resolved: Any,
    data: pd.DataFrame,
    group_col: str,
    survey_weights: Optional[np.ndarray],
) -> Tuple[bool, bool]:
    """Return (strata_varies_within_group, psu_varies_within_group).

    Diagnostic helper. No longer used to gate any out-of-scope
    combination in ``fit()`` — the analytical TSL path, the
    heterogeneity WLS path, and the PSU-level wild multiplier
    bootstrap all support within-group-varying PSU/strata via the
    cell-period allocator. The helper is retained for callers that
    need to branch on the regime (e.g., documentation, diagnostic
    warnings). Zero-weight rows are excluded (subpopulation
    contract).
    """
    if resolved is None:
        return False, False
    pos_mask = np.asarray(survey_weights) > 0
    if not pos_mask.any():
        return False, False
    g_eff = np.asarray(data[group_col].values)[pos_mask]
    strata_varies = False
    psu_varies = False
    if resolved.strata is not None:
        s_eff = np.asarray(resolved.strata)[pos_mask]
        strata_varies = bool(
            pd.DataFrame({"g": g_eff, "s": s_eff}).groupby("g")["s"].nunique().gt(1).any()
        )
    if resolved.psu is not None:
        p_eff = np.asarray(resolved.psu)[pos_mask]
        psu_varies = bool(
            pd.DataFrame({"g": g_eff, "p": p_eff}).groupby("g")["p"].nunique().gt(1).any()
        )
    return strata_varies, psu_varies


def _validate_cell_constant_strata_psu(
    resolved: Any,
    data: pd.DataFrame,
    group_col: str,
    time_col: str,
    survey_weights: Optional[np.ndarray],
) -> None:
    """Reject survey designs where strata or PSU vary within a (g, t) cell.

    Under the cell-period IF allocator, ``psi_i`` attributes each
    observation's mass to its ``(g, t)`` cell scaled by proportional
    weight share:
    ``psi_i = U_centered_per_period[g, t] * (w_i / W_{g, t})``. Binder
    TSL aggregates at PSU level, so strata / PSU labels within a cell
    must be unambiguous. In one-obs-per-cell panels (the canonical dCDH
    structure) this check is trivially satisfied; in multi-obs-per-cell
    panels, a cell split across strata or PSUs is rejected. This is a
    strict relaxation of the old within-group constancy rule shipped
    before the cell-period allocator (REGISTRY.md
    ``ChaisemartinDHaultfoeuille`` Note on survey IF expansion).

    Zero-weight rows are excluded (subpopulation contract).
    """
    if resolved is None:
        return
    pos_mask = np.asarray(survey_weights) > 0
    if not pos_mask.any():
        return
    g_eff = np.asarray(data[group_col].values)[pos_mask]
    t_eff = np.asarray(data[time_col].values)[pos_mask]
    if resolved.strata is not None:
        s_eff = np.asarray(resolved.strata)[pos_mask]
        df_cell = pd.DataFrame({"g": g_eff, "t": t_eff, "s": s_eff})
        varying = df_cell.groupby(["g", "t"])["s"].nunique()
        bad = varying[varying > 1]
        if len(bad) > 0:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille survey support requires "
                f"strata to be constant within each (group, time) cell "
                f"under the cell-period IF allocator, but {len(bad)} "
                f"cell(s) have multiple strata (examples: "
                f"{bad.index.tolist()[:5]}). The cell allocator treats "
                f"the (g, t) cell as the effective sampling unit for "
                f"stratified Binder variance; within-cell stratum "
                f"variation is ambiguous. The allocator DOES support "
                f"strata that vary across cells of the same group "
                f"(relaxation over the previous within-group constancy "
                f"rule shipped in earlier releases)."
            )
    if resolved.psu is not None:
        p_eff = np.asarray(resolved.psu)[pos_mask]
        df_cell = pd.DataFrame({"g": g_eff, "t": t_eff, "p": p_eff})
        varying = df_cell.groupby(["g", "t"])["p"].nunique()
        bad = varying[varying > 1]
        if len(bad) > 0:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille survey support requires "
                f"PSU to be constant within each (group, time) cell "
                f"under the cell-period IF allocator, but {len(bad)} "
                f"cell(s) have multiple PSUs (examples: "
                f"{bad.index.tolist()[:5]}). The cell allocator treats "
                f"the (g, t) cell as the effective sampling unit; "
                f"within-cell PSU variation is ambiguous. The allocator "
                f"DOES support PSU that varies across cells of the same "
                f"group (relaxation over the previous within-group "
                f"constancy rule shipped in earlier releases)."
            )


def _refresh_path_inference(
    path_effects: Optional[Dict[Tuple[int, ...], Dict[str, Any]]],
    path_placebos: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]],
    alpha: float,
    df_final: Optional[int],
) -> None:
    """Refresh per-path inference fields (t_stat, p_value, conf_int) with
    the final ``df_final`` so they agree with the global surfaces and
    ``results.survey_metadata.df_survey`` after all replicate-weight
    ``n_valid`` appends complete.

    Per-path event-study and placebo helpers compute inference using a
    snapshot of ``_replicate_n_valid_list`` taken at fit-time BEFORE
    they append their own ``n_valid`` contributions. The final R2 P1b
    block in ``fit()`` already refreshes the global surfaces (overall /
    joiners / leavers / multi_horizon_inference / placebo_horizon_inference
    / heterogeneity / normalized) with the final df; this helper is its
    per-path counterpart, called from the same final block.

    No-op under TSL (analytical) or non-survey fits — they skip
    replicate-n_valid bookkeeping entirely. Mutates dicts in place.
    """
    from diff_diff.utils import safe_inference

    def _refresh_entry(entry: Dict[str, Any]) -> None:
        eff = entry.get("effect")
        se = entry.get("se")
        if eff is None or se is None:
            return
        if not np.isfinite(se):
            return
        t_new, p_new, ci_new = safe_inference(eff, se, alpha=alpha, df=df_final)
        entry["t_stat"] = t_new
        entry["p_value"] = p_new
        entry["conf_int"] = ci_new

    if path_effects is not None:
        for path_data in path_effects.values():
            horizons = path_data.get("horizons", {})
            for entry in horizons.values():
                _refresh_entry(entry)
    if path_placebos is not None:
        for path_horizons in path_placebos.values():
            for entry in path_horizons.values():
                _refresh_entry(entry)


def _inference_df(
    effective_df: Optional[int],
    resolved_survey: Any,
) -> Optional[int]:
    """Coerce an effective-df value for use in ``safe_inference(df=...)``.

    ``_effective_df_survey()`` returns ``None`` when the replicate
    design's base df is undefined (QR-rank ≤ 1) or when the reduced
    df would be < 1. ``safe_inference`` treats ``df=None`` as the
    standard normal (z-inference) and only returns NaN for
    ``df <= 0``. Under a replicate design, "undefined effective df"
    MUST map to NaN inference per REGISTRY.md — NOT to z-inference.

    Returns:
    - ``effective_df`` when defined (t-inference with that df).
    - ``0`` when ``effective_df is None`` AND ``resolved_survey`` uses
      replicate variance — forces ``safe_inference`` to return NaN
      t/p/CI via its ``df <= 0`` branch.
    - ``None`` otherwise (no survey, or TSL design where the resolver
      always returns an int df — z-inference is never reachable here).
    """
    if effective_df is not None:
        return effective_df
    if resolved_survey is None:
        return None
    if getattr(resolved_survey, "uses_replicate_variance", False):
        return 0
    return None


def _effective_df_survey(
    resolved_survey: Any,
    replicate_n_valid_list: List[int],
) -> Optional[int]:
    """Compute the effective ``df_survey`` for t-critical values.

    Under replicate-weight designs, each IF site returns an ``n_valid``
    count (number of replicate columns that produced a finite estimate).
    This helper **reduces** — never overwrites — the design-level
    ``resolved_survey.df_survey`` (which for replicate designs is
    ``QR-rank - 1`` per R's ``survey::degf()`` convention; see
    ``diff_diff/survey.py:590``). Returns
    ``min(resolved_survey.df_survey, min(n_valid) - 1)`` when both are
    defined.

    Matches the precedent in ``diff_diff/efficient_did.py:1133-1135``
    and ``diff_diff/triple_diff.py:676-686`` (both reduce, not
    replace). Under TSL (analytical) or non-survey fits,
    ``replicate_n_valid_list`` is empty and this falls through to the
    design-level df. Returns ``None`` when either the base df is
    undefined (rank ≤ 1) or the reduced df would be < 1 — preserving
    the NaN-inference contract from ``safe_inference``.
    """
    if resolved_survey is None:
        return None
    base_df = resolved_survey.df_survey
    if not replicate_n_valid_list:
        return base_df
    reduced_df = int(min(replicate_n_valid_list)) - 1
    if base_df is None or reduced_df < 1:
        return None
    return min(int(base_df), reduced_df)


def _compute_se(
    U_centered: np.ndarray,
    divisor: int,
    obs_survey_info: Optional[dict],
    eligible_groups: Optional[list] = None,
    U_centered_per_period: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[int]]:
    """Dispatch to plug-in SE or survey-design-aware SE.

    When ``obs_survey_info`` is ``None``, falls back to the simple
    plug-in formula.  Otherwise, expands group-level IFs and delegates
    to TSL variance (or replicate-weight variance when the resolved
    design carries replicate weights).

    ``U_centered_per_period`` is the cell-period attribution of
    ``U_centered``. Each row ``g`` satisfies
    ``U_centered_per_period[g, :].sum() == U_centered[g]``. When
    supplied, the survey-aware path expands influence to observations
    using the cell-level allocator
    ``psi_i = U_centered_per_period[g_i, t_i] * (w_i / W_{g_i, t_i})``;
    when ``None`` the path falls back to the group-level allocator
    ``psi_i = U_centered[g_i] * (w_i / W_{g_i})``. Byte-identical
    under within-group-constant PSU since both allocators telescope to
    the same PSU-level sum (REGISTRY.md survey IF expansion Note).

    Returns ``(se, n_valid_replicates)``. ``n_valid_replicates`` is
    ``None`` under the plug-in / TSL paths, and the number of valid
    replicate columns under the replicate path.
    """
    if obs_survey_info is None:
        return _plugin_se(U_centered=U_centered, divisor=divisor), None
    if eligible_groups is None:
        return _plugin_se(U_centered=U_centered, divisor=divisor), None
    if divisor <= 0:
        return float("nan"), None
    # dCDH IFs are numerator-scale (U.sum() == N_S * DID_M).
    # compute_survey_if_variance() expects estimator-scale psi.
    # Scale by 1/divisor to normalize before survey expansion.
    U_scaled = U_centered / divisor
    U_pp_scaled = U_centered_per_period / divisor if U_centered_per_period is not None else None
    return _survey_se_from_group_if(
        U_centered=U_scaled,
        eligible_groups=eligible_groups,
        obs_survey_info=obs_survey_info,
        U_centered_per_period=U_pp_scaled,
    )


def _survey_se_from_group_if(
    U_centered: np.ndarray,
    eligible_groups: list,
    obs_survey_info: dict,
    U_centered_per_period: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[int]]:
    """Compute survey-design-aware SE from per-group / per-cell centered IFs.

    Expands influence-function mass to observation level using the
    **cell-period allocator**
    ``psi_i = U_centered_per_period[g_i, t_i] * (w_i / W_{g_i, t_i})``
    when ``U_centered_per_period`` is supplied, or the legacy group-
    level allocator ``psi_i = U_centered[g_i] * (w_i / W_{g_i})``
    otherwise. Both allocators are equivalent at the PSU-level sum
    under within-group-constant PSU (per-cell sums telescope to
    ``U_centered[g]``), so Binder TSL variance is byte-identical on
    inputs the old within-group-constant validator accepted. The cell-
    period allocator additionally supports PSU/strata that vary across
    cells of g — the lift the allocator buys for Stage 2. Dispatches
    to ``compute_survey_if_variance()`` (TSL) or
    ``compute_replicate_if_variance()`` (BRR/Fay/JK1/JKn/SDR) based on
    the resolved design. See REGISTRY.md ``ChaisemartinDHaultfoeuille``
    Note on survey IF expansion for the contract.

    Parameters
    ----------
    U_centered : np.ndarray
        Cohort-recentered per-group IF values (one per eligible group).
        Used for degenerate-cohort detection and fallback expansion.
    eligible_groups : list
        Group IDs corresponding to entries of ``U_centered`` /
        ``U_centered_per_period`` (variance-eligible groups after
        singleton-baseline exclusion).
    obs_survey_info : dict
        Observation-level survey data retained from ``fit()`` setup:
        ``{"group_ids", "time_ids", "weights", "resolved", "periods"}``.
        ``time_ids`` and ``periods`` are required for the cell-level
        allocator; when either is absent, the group-level allocator is
        used.
    U_centered_per_period : np.ndarray of shape (n_eligible, n_periods), optional
        Cohort-recentered per-``(g, t)``-cell IF attributions, with
        ``U_centered_per_period.sum(axis=1) == U_centered``. When
        supplied, the cell allocator is used.

    Returns
    -------
    Tuple[float, Optional[int]]
        ``(se, n_valid_replicates)``. ``se`` is the survey-design-aware
        SE, or NaN if degenerate. ``n_valid_replicates`` is the number
        of valid replicate columns used for variance under the replicate
        path, or ``None`` under the TSL (analytical) path.
    """
    from diff_diff.survey import (
        compute_replicate_if_variance,
        compute_survey_if_variance,
    )

    # Degenerate-cohort contract (mirror _plugin_se): when the centered
    # IF is empty or every cohort is a singleton (→ recentered IF is
    # identically zero), the variance is unidentified. Return NaN
    # rather than sqrt(0)=0 so the fit-time warning fires and
    # inference stays NaN-consistent across every surface routed
    # through _compute_se (overall, joiners/leavers, multi-horizon
    # ATT, placebos, normalized/cumulated, heterogeneity).
    if U_centered.size == 0:
        return float("nan"), None
    if float((U_centered**2).sum()) <= 0:
        return float("nan"), None

    group_ids = obs_survey_info["group_ids"]
    weights = obs_survey_info["weights"]
    resolved = obs_survey_info["resolved"]
    time_ids = obs_survey_info.get("time_ids")
    periods = obs_survey_info.get("periods")

    # Zero-weight rows are out-of-sample (SurveyDesign.subpopulation()).
    # Skip them before the group-ID factorization so NaN / non-comparable
    # group IDs on excluded rows cannot crash np.unique. psi stays full-
    # length with zeros in excluded positions so the alignment with
    # resolved.strata / resolved.psu inside compute_survey_if_variance
    # is preserved.
    weights_arr = np.asarray(weights, dtype=np.float64)
    pos_mask = weights_arr > 0
    n_obs = len(weights_arr)
    psi = np.zeros(n_obs, dtype=np.float64)

    if not pos_mask.any():
        return float("nan"), None

    gids_eff = np.asarray(group_ids)[pos_mask]
    w_eff = weights_arr[pos_mask]

    use_cell_allocator = (
        U_centered_per_period is not None
        and time_ids is not None
        and periods is not None
        and U_centered_per_period.size > 0
    )

    # Binder TSL fallback: when the cell allocator would apply but
    # PSU is within-group-constant (PSU=group, strictly-coarser PSU
    # within-group-constant, or the auto-inject default), the cell
    # and group allocators are equivalent at PSU-level aggregation
    # via the row-sum identity `sum_{c in g} u_cell[c] == u_centered[g]`.
    # Prefer the legacy group-level path in that regime: it sidesteps
    # the sentinel-mass guard (below) that would otherwise fire
    # spuriously on terminally-missing panels whose PSU structure
    # does not require cell-level resolution. Matches the bootstrap
    # dispatcher's routing rule (`_compute_dcdh_bootstrap` +
    # `_psu_varies_within_group`).
    #
    # **Replicate-weight carve-out:** replicate designs do not carry
    # `resolved.psu`, but the Class A replicate contract shipped in
    # PR #323 applies `compute_replicate_if_variance` to the cell-
    # allocator `psi_obs` — per-row-varying replicate ratios are
    # allocator-sensitive, so forcing replicate fits onto the legacy
    # group-level allocator would silently change their SE. The
    # fallback therefore skips replicate fits; the sentinel-mass
    # guard still fires on mass leakage when it applies. The Class
    # A replicate allocator contract is asserted by
    # `tests/test_survey_dcdh_replicate_psu.py::TestReplicateClassA::
    # test_att_cell_allocator_with_varying_replicate_ratios`.
    if use_cell_allocator and not getattr(resolved, "uses_replicate_variance", False):
        psu_arr = getattr(resolved, "psu", None)
        if psu_arr is None:
            use_cell_allocator = False
        else:
            psu_eff = np.asarray(psu_arr)[pos_mask]
            eligible_set = set(eligible_groups)
            elig_row_mask = np.array([g in eligible_set for g in gids_eff], dtype=bool)
            if elig_row_mask.any():
                psu_varies_within = bool(
                    pd.DataFrame(
                        {
                            "g": gids_eff[elig_row_mask],
                            "p": psu_eff[elig_row_mask],
                        }
                    )
                    .groupby("g")["p"]
                    .nunique()
                    .gt(1)
                    .any()
                )
                if not psu_varies_within:
                    use_cell_allocator = False
            else:
                use_cell_allocator = False

    if use_cell_allocator:
        # The flag definition above guarantees these (mypy can't track it).
        assert U_centered_per_period is not None and time_ids is not None and periods is not None
        tids_eff = np.asarray(time_ids)[pos_mask]
        # Map row's group to an index in eligible_groups (−1 when the
        # group is ineligible — singleton-baseline exclusion drops it).
        eligible_idx_by_gid = {gid: i for i, gid in enumerate(eligible_groups)}
        elig_idx_eff = np.array(
            [eligible_idx_by_gid.get(gid, -1) for gid in gids_eff],
            dtype=np.int64,
        )
        # Map row's time to a column index in U_centered_per_period.
        # get_indexer returns −1 for values absent from `periods`
        # (should be rare in practice — positive-weight rows almost
        # always survive cell aggregation; if one slips through we
        # treat it like an ineligible row rather than crash).
        col_idx_eff = np.asarray(pd.Index(periods).get_indexer(tids_eff), dtype=np.int64)
        valid_cell = (elig_idx_eff >= 0) & (col_idx_eff >= 0)
        n_eligible = len(eligible_groups)
        n_periods_pp = U_centered_per_period.shape[1]
        # Per-(g, t) weight totals W_{g,t} over the effective sample.
        W_cell = np.zeros((n_eligible, n_periods_pp), dtype=np.float64)
        np.add.at(
            W_cell,
            (elig_idx_eff[valid_cell], col_idx_eff[valid_cell]),
            w_eff[valid_cell],
        )
        # Sentinel-mass guard (mirror of `_unroll_target_to_cells` on
        # the bootstrap path). Under terminal missingness,
        # `_cohort_recenter_per_period` subtracts cohort column means
        # across the full period grid, so a group with no observation
        # at period t can acquire non-zero centered mass at that cell.
        # The cell-level expansion `psi_i = U[g,t] * (w_i / W_{g,t})`
        # has no observation to attach that mass to (W_{g,t} = 0), so
        # silently dropping it would understate the SE. Raise a
        # targeted ValueError instead (consistent with the cell-level
        # bootstrap's `_unroll_target_to_cells` guard).
        missing_cell_mask = W_cell == 0
        if missing_cell_mask.any():
            leaked = U_centered_per_period[missing_cell_mask]
            if leaked.size > 0 and bool(np.any(np.abs(leaked) > 1e-12)):
                # Branch the wording on replicate-vs-TSL so replicate
                # ATT users debugging this error are not misdirected
                # to "within-group-varying PSU" (replicate designs
                # have no PSU structure). Core mechanics and the
                # pre-processing workaround are shared.
                is_replicate = bool(getattr(resolved, "uses_replicate_variance", False))
                path_name = (
                    "Rao-Wu replicate-weight ATT variance"
                    if is_replicate
                    else "Analytical Binder TSL survey SE"
                )
                trigger_detail = (
                    "terminal missingness on a replicate-weight "
                    "design (replicate ATT unconditionally uses the "
                    "cell-period allocator per the Class A contract; "
                    "PSU structure is not involved)"
                    if is_replicate
                    else "terminal missingness combined with within-"
                    "group-varying PSU (the Binder TSL cell-period "
                    "allocator cannot allocate leaked mass to any "
                    "observation)"
                )
                raise ValueError(
                    f"{path_name} cannot be computed on this panel: "
                    "cohort-recentered IF mass landed on (g, t) cells "
                    "with no positive-weight observations "
                    f"(W_{{g, t}} = 0). This typically occurs with "
                    f"{trigger_detail}: _cohort_recenter_per_period "
                    "subtracts cohort column means across the full "
                    "period grid, so a group with no observation at "
                    "period t acquires non-zero centered mass there, "
                    "which the cell-period allocator cannot allocate "
                    "to any observation. Pre-process the panel to "
                    "remove terminal missingness (drop late-exit "
                    "groups or trim to a balanced sub-panel) before "
                    "fitting."
                )
        # Lookup U_centered_per_period and W_cell per row.
        u_obs_cell = np.zeros(w_eff.shape[0], dtype=np.float64)
        u_obs_cell[valid_cell] = U_centered_per_period[
            elig_idx_eff[valid_cell], col_idx_eff[valid_cell]
        ]
        w_cell_at_row = np.zeros(w_eff.shape[0], dtype=np.float64)
        w_cell_at_row[valid_cell] = W_cell[elig_idx_eff[valid_cell], col_idx_eff[valid_cell]]
        safe_w = np.where(w_cell_at_row > 0, w_cell_at_row, 1.0)
        psi[pos_mask] = u_obs_cell * (w_eff / safe_w)
    else:
        # Legacy group-level allocator (no per-period attribution
        # provided, or time/period info unavailable). Preserved for
        # defensive fallback and for unit tests that exercise the
        # legacy allocator. No current caller in fit() uses this
        # branch — ATT / joiners / leavers / placebos all thread
        # U_centered_per_period, and heterogeneity (as of PR 3)
        # constructs its own cell-period psi_obs and calls
        # compute_survey_if_variance directly.
        group_to_u = {gid: U_centered[idx] for idx, gid in enumerate(eligible_groups)}
        u_obs_eff = np.array([group_to_u.get(gid, 0.0) for gid in gids_eff])
        unique_gids, inverse = np.unique(gids_eff, return_inverse=True)
        w_totals_per_group = np.bincount(inverse, weights=w_eff)
        w_obs_total_eff = w_totals_per_group[inverse]
        safe_w = np.where(w_obs_total_eff > 0, w_obs_total_eff, 1.0)
        psi[pos_mask] = u_obs_eff * (w_eff / safe_w)

    # Dispatch: replicate variance (BRR/Fay/JK1/JKn/SDR) vs TSL.
    # Mirrors the inline branch in TripleDifference:1206-1238 and
    # EfficientDiD:1119-1142. Under replicate, returns (variance, n_valid);
    # the caller propagates min(n_valid) to df_survey.
    if getattr(resolved, "uses_replicate_variance", False):
        variance, n_valid = compute_replicate_if_variance(psi, resolved)
        if not np.isfinite(variance) or variance < 0:
            return float("nan"), n_valid
        return float(np.sqrt(variance)), n_valid

    variance = compute_survey_if_variance(psi, resolved)
    if not np.isfinite(variance) or variance < 0:
        return float("nan"), None
    return float(np.sqrt(variance)), None


def _build_group_time_design(
    cell: pd.DataFrame,
    group_col: str,
    time_col: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dense (intercept + group dummies + time dummies) design matrix.

    Used by the TWFE decomposition diagnostic. The first group and first
    period are dropped as the reference categories. Returns the matrix
    and a list of column names.
    """
    if cell.empty:
        raise ValueError(
            "Cannot compute TWFE diagnostic on an empty cell DataFrame. "
            "Provide a panel with at least 2 groups and 2 time periods."
        )
    groups = sorted(cell[group_col].unique().tolist())
    times = sorted(cell[time_col].unique().tolist())
    n = len(cell)
    n_groups = len(groups)
    n_times = len(times)
    if n_groups < 2 or n_times < 2:
        raise ValueError(
            f"TWFE diagnostic requires at least 2 groups and 2 time periods, "
            f"got {n_groups} group(s) and {n_times} period(s)."
        )

    # Columns: [intercept, group_1, ..., group_{G-1}, time_1, ..., time_{T-1}]
    n_cols = 1 + (n_groups - 1) + (n_times - 1)
    X = np.zeros((n, n_cols), dtype=float)
    X[:, 0] = 1.0  # intercept

    group_to_col = {g: 1 + i for i, g in enumerate(groups[1:])}
    time_to_col = {t: 1 + (n_groups - 1) + i for i, t in enumerate(times[1:])}

    group_arr = cell[group_col].to_numpy()
    time_arr = cell[time_col].to_numpy()
    for i in range(n):
        g = group_arr[i]
        t = time_arr[i]
        if g in group_to_col:
            X[i, group_to_col[g]] = 1.0
        if t in time_to_col:
            X[i, time_to_col[t]] = 1.0

    column_names = (
        ["intercept"] + [f"group[{g}]" for g in groups[1:]] + [f"time[{t}]" for t in times[1:]]
    )
    return X, column_names


def _compute_twfe_diagnostic(
    cell: pd.DataFrame,
    group_col: str,
    time_col: str,
    rank_deficient_action: str,
) -> TWFEWeightsResult:
    """
    Compute the per-cell TWFE decomposition diagnostic from Theorem 1 of AER 2020.

    Steps:

    1. Regress ``d_gt`` on group + time fixed effects via :func:`solve_ols`.
    2. Compute residuals ``eps_{g, t}`` from the regression.
    3. Compute per-cell **contribution weights** (the Theorem 1
       decomposition object):
       ``cw_{g,t} = N_{g,t} * eps_{g,t} / sum_{treated} N * eps``
       These are exported in the ``weights`` column of the returned
       ``TWFEWeightsResult``.
    4. Count negative contribution weights among treated cells.
    5. Compute the plain TWFE coefficient as a separate regression of
       ``y_gt`` on the same FE plus the treatment indicator.
    6. Compute ``sigma_fe`` from the Corollary 1 **paper weights**
       (a distinct object from the contribution weights):
       ``w_paper = eps / sum_treated(s * eps)`` where
       ``s = N_{g,t} / N_1`` are observation shares. The paper weight
       is centered at 1 under observation-share weighting. Then:
       ``sigma_fe = |beta_fe| / sqrt(sum_treated(s * (w_paper - 1)^2))``
       which is the smallest standard deviation of cell-level treatment
       effects that could flip the sign of the plain TWFE estimator.
    """
    X, _ = _build_group_time_design(cell, group_col, time_col)
    d_arr = cell["d_gt"].to_numpy().astype(float)
    # Cell weight for Theorem 1: under survey_design, survey-weighted
    # cell totals (w_gt) replace raw cell counts (n_gt) so the FE
    # regressions, normalization denominator, and Corollary 1 shares
    # match the observation-level pweighted TWFE estimand. Without
    # survey_design (w_gt column absent), fall back to n_gt — the
    # non-survey path is unchanged.
    if "w_gt" in cell.columns:
        cell_weight = cell["w_gt"].to_numpy().astype(float)
    else:
        cell_weight = cell["n_gt"].to_numpy().astype(float)
    y_arr = cell["y_gt"].to_numpy().astype(float)

    # Step 1-2: regress d on FE
    coef_d, residuals_d, _ = solve_ols(
        X,
        d_arr,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=cell_weight,
    )
    eps = residuals_d

    # Step 3: per-cell weights — normalize by sum over treated cells
    treated_mask = d_arr == 1
    denom = float((cell_weight[treated_mask] * eps[treated_mask]).sum())
    if denom == 0:
        # Cannot normalize: the design has zero treated mass after FE absorption.
        # Warn so the user knows the diagnostic returned NaN values rather than
        # silently substituting them.
        warnings.warn(
            "TWFE decomposition diagnostic could not normalize per-cell "
            "weights: the sum of N_{g,t} * residual over treated cells is "
            "zero. This typically means the design matrix has perfect "
            "collinearity between treatment and the group/period fixed "
            "effects. Returning NaN for fraction_negative, sigma_fe, and "
            "beta_fe.",
            UserWarning,
            stacklevel=3,
        )
        weights_df = cell[[group_col, time_col]].copy()
        weights_df["weight"] = 0.0
        return TWFEWeightsResult(
            weights=weights_df,
            fraction_negative=float("nan"),
            sigma_fe=float("nan"),
            beta_fe=float("nan"),
        )
    contribution_weights = (cell_weight * eps) / denom

    weights_df = cell[[group_col, time_col]].copy()
    weights_df["weight"] = contribution_weights

    fraction_negative = float((contribution_weights[treated_mask] < 0).sum() / treated_mask.sum())

    # Step 5: plain TWFE regression of y on (FE + d_gt)
    X_with_d = np.column_stack([X, d_arr.reshape(-1, 1)])
    coef_fe, _, _ = solve_ols(
        X_with_d,
        y_arr,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=cell_weight,
    )
    beta_fe = float(coef_fe[-1])

    # Step 6: sigma_fe per Corollary 1 of AER 2020
    #
    # The paper defines w_{g,t} = eps_{g,t} / E_treated[eps], which
    # is DIFFERENT from the contribution weights w_gt exported in the
    # weights DataFrame (contribution_weight = s * w_paper). The paper
    # weight has the property that sum(s * w_paper) = 1 (centered at
    # 1 under observation-share weighting). sigma_fe uses the paper
    # weight:
    #
    #   w_paper = eps / sum_treated(s * eps)
    #   sigma(w) = sqrt(sum_treated(s * (w_paper - 1)^2))
    #   sigma_fe = |beta_fe| / sigma(w)
    #
    # where s_{g,t} = N_{g,t} / N_1 are observation shares (under
    # survey_design, cell_weight is w_gt so shares are effective-
    # weight shares; non-survey path is byte-identical).
    eps_treated = eps[treated_mask]
    w_treated_arr = cell_weight[treated_mask]
    w1 = float(w_treated_arr.sum())  # total treated weight (N_1 or W_1)
    if w1 > 0:
        shares = w_treated_arr / w1  # s_{g,t} = w_{g,t} / w_1
        denom_paper = float((shares * eps_treated).sum())
        if abs(denom_paper) > 0:
            w_paper = eps_treated / denom_paper  # paper's w_{g,t}
            # Weighted variance around 1 (the weighted mean of w_paper is 1 by construction)
            var_w = float((shares * (w_paper - 1.0) ** 2).sum())
        else:
            var_w = 0.0
    else:
        var_w = 0.0
    if var_w > 0 and np.isfinite(beta_fe):
        sigma_fe = float(abs(beta_fe) / np.sqrt(var_w))
    else:
        sigma_fe = float("nan")

    return TWFEWeightsResult(
        weights=weights_df,
        fraction_negative=fraction_negative,
        sigma_fe=sigma_fe,
        beta_fe=beta_fe,
    )


# =============================================================================
# Convenience functions
# =============================================================================


def chaisemartin_dhaultfoeuille(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    time: str,
    treatment: str,
    **kwargs: Any,
) -> ChaisemartinDHaultfoeuilleResults:
    """
    One-shot convenience wrapper around
    :class:`ChaisemartinDHaultfoeuille`.

    Equivalent to::

        ChaisemartinDHaultfoeuille(**init_kwargs).fit(
            data, outcome=..., group=..., time=..., treatment=...,
            **fit_kwargs,
        )

    All keyword arguments are split between ``__init__`` and ``fit`` based
    on which signature accepts them. Useful for one-line use in scripts.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, group, time, treatment : str
    **kwargs : Any
        Forwarded to ``ChaisemartinDHaultfoeuille.__init__`` or
        ``.fit()`` based on parameter name.

    Returns
    -------
    ChaisemartinDHaultfoeuilleResults
    """
    import inspect

    init_keys = {
        name
        for name, p in inspect.signature(ChaisemartinDHaultfoeuille.__init__).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and name != "self"
    }
    init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in init_keys}
    est = ChaisemartinDHaultfoeuille(**init_kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=group,
        time=time,
        treatment=treatment,
        **fit_kwargs,
    )


def twowayfeweights(
    data: pd.DataFrame,
    outcome: str,
    unit: Any = NOT_SUPPLIED,
    time: Any = NOT_SUPPLIED,
    treatment: Any = NOT_SUPPLIED,
    rank_deficient_action: str = "warn",
    survey_design: Any = None,
    group: Any = NOT_SUPPLIED,
) -> TWFEWeightsResult:
    """
    Standalone TWFE decomposition diagnostic.

    Computes the per-cell weights, fraction negative, and ``sigma_fe``
    from Theorem 1 of de Chaisemartin & D'Haultfoeuille (2020), without
    fitting the full dCDH estimator. Mirrors the standalone Stata
    ``twowayfeweights`` package.

    Parameters
    ----------
    data : pd.DataFrame
        Individual-level panel.
    outcome : str
    unit : str
        Unit identifier column (the R/Stata ``twowayfeweights`` "group").
    time : str
    treatment : str
    rank_deficient_action : str, default="warn"
        Action when the FE design matrix is rank-deficient.
    survey_design : SurveyDesign, optional
        If provided, cell aggregation uses survey-weighted cell means
        (matching ``fit(..., survey_design=sd).twfe_*``). Required to preserve
        fit-vs-helper parity under survey-backed inputs. Only
        ``weight_type='pweight'`` is supported; other types raise ValueError.
        Replicate-weight designs (BRR/Fay/JK1/JKn/SDR) are accepted —
        the TWFE diagnostic has no SE field on ``TWFEWeightsResult``,
        so replicate weights only affect the cell aggregation path
        (aggregated numbers are identical to
        ``fit(..., survey_design=sd).twfe_*`` under the same input).

    group : str, optional
        Deprecated alias for ``unit`` (row M-097); warns with
        ``FutureWarning`` and will be removed in 4.0.

    Returns
    -------
    TWFEWeightsResult
        Object with attributes ``weights`` (DataFrame), ``fraction_negative``
        (float), ``sigma_fe`` (float), and ``beta_fe`` (float).
    """
    unit = resolve_renamed_kwarg(
        "twowayfeweights", "group", group, "unit", unit, default=NOT_SUPPLIED
    )
    require_arg("twowayfeweights", "unit", unit)
    require_arg("twowayfeweights", "time", time)
    require_arg("twowayfeweights", "treatment", treatment)
    # Body-local name; the public parameter is unit (M-097).
    group = unit
    # Survey resolution (optional): mirrors the fit() path so that the
    # standalone helper produces identical numbers to fit(..., survey_design=sd).
    survey_weights = None
    if survey_design is not None:
        from diff_diff.survey import _resolve_survey_for_fit

        resolved, survey_weights, _, _ = _resolve_survey_for_fit(survey_design, data, "analytical")
        if resolved is not None and resolved.weight_type != "pweight":
            raise ValueError(
                f"twowayfeweights() survey support requires "
                f"weight_type='pweight', got '{resolved.weight_type}'. "
                f"The TWFE diagnostic under survey uses survey-weighted cell "
                f"means; other weight types are not supported."
            )
        # Replicate-weight designs are accepted. TWFE diagnostic has no
        # SE field on TWFEWeightsResult — the replicate weights only
        # participate through the full-sample weight (resolved.weights),
        # which drives survey-weighted cell aggregation in
        # _validate_and_aggregate_to_cells. Diagnostic numbers
        # (beta_fe, sigma_fe, fraction_negative) match
        # fit(..., survey_design=sd).twfe_* under replicate input.

    # Validation + cell aggregation via the same helper used by
    # ChaisemartinDHaultfoeuille.fit() — enforces NaN/binary/within-cell
    # rules from REGISTRY.md so the standalone diagnostic does not
    # silently mishandle malformed input.
    cell = _validate_and_aggregate_to_cells(
        data=data,
        outcome=outcome,
        group=group,
        time=time,
        treatment=treatment,
        weights=survey_weights,
    )
    # TWFE diagnostic assumes binary treatment (d_arr == 1 for treated mask).
    if not set(cell["d_gt"].unique()).issubset({0.0, 1.0, 0, 1}):
        raise ValueError(
            "twowayfeweights() requires binary treatment {0, 1}. "
            "Non-binary treatment is supported by fit() with L_max >= 1 "
            "but the TWFE diagnostic (Theorem 1 of AER 2020) assumes "
            "binary treatment."
        )
    return _compute_twfe_diagnostic(
        cell=cell,
        group_col=group,
        time_col=time,
        rank_deficient_action=rank_deficient_action,
    )
