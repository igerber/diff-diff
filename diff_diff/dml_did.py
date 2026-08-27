"""DMLDiD: Double/Debiased Machine Learning DiD (Chang 2020), staggered.

Implements Chang (2020, The Econometrics Journal 23(2), 177-191) as a
staggered ATT(g,t) estimator: each Callaway-Sant'Anna style (g, t) cell is
a 2-period Chang problem. ``panel=True`` (default) runs Case 1 (repeated
outcomes) — cross-fitted nuisances (propensity ``g_hat`` and control
outcome-change regression ``m_hat``), the Neyman-orthogonal score
``psi_1`` (``chang_panel_score``), and the augmented-score plug-in
variance (``chang_panel_score_augmented``;
``SE = sqrt(mean(psi_bar**2)/n)``). ``panel=False`` runs Case 2 (declared
repeated cross sections) — level outcomes, the single control-only
``(T - lam_hat) * Y`` outcome nuisance (``chang_rcs_score``), and the
λ-corrected Theorem 2 variance (``chang_rcs_score_augmented``). The
classic 2-period design is the degenerate single-cell case. Covariates
are REQUIRED (Chang's estimator exists for the high-dimensional-X
setting; use CallawaySantAnna without covariates).

``DMLDiD`` writes the CallawaySantAnna per-(g,t) ``influence_func_info``
payload (per-sampling-unit entries ``psi_bar_i / n_cell`` — per unit on
panel fits, per observation on RCS fits — so ``sqrt(sum(if**2))`` IS the
cell SE) and inherits the CS aggregation + multiplier-bootstrap mixins:
event study with sup-t bands, group/simple aggregation (plus total on
panel fits only; RCS fits fail ``total`` closed), and post-fit
``results.aggregate()`` with bootstrap replay.

See docs/methodology/REGISTRY.md "DMLDiD" for equations, implementation
Notes (global p-hat, D-stratified folds, pooled fold weighting, trimming,
skip-reason vocabulary) and the DoubleML parity anchors.
"""

import decimal
import secrets
import warnings
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._crossfit import DegenerateFoldError, assign_folds, cross_fit_predict
from diff_diff._dr_scores import (
    _chang_rcs_score_augmented_with_slope,
    chang_panel_score,
    chang_panel_score_augmented,
    chang_rcs_score,
)
from diff_diff._learners import (
    _CLASSIFIER_NAMES,
    _REGRESSOR_NAMES,
    make_learner,
    validate_learner,
)
from diff_diff.dml_did_results import DMLDiDResults
from diff_diff.linalg import _check_propensity_diagnostics
from diff_diff.staggered import (
    _build_aggregation_kit,
    _nan_gt_entry,
)
from diff_diff.staggered import (
    select_base_period as _select_base_period_impl,
)
from diff_diff.staggered import (
    valid_periods_for_group as _valid_periods_for_group_impl,
)
from diff_diff.staggered_aggregation import CallawaySantAnnaAggregationMixin
from diff_diff.staggered_bootstrap import CallawaySantAnnaBootstrapMixin
from diff_diff.utils import (
    safe_inference,
    validate_anticipation,
    validate_covariate_names,
    validate_n_bootstrap,
    validate_pscore_trim,
)

__all__ = ["DMLDiD"]

# Magnitude bound for time/first_treat labels (with anticipation headroom):
# above 2**62, a uint64 label survives an int64 round-trip check by wrapping
# TWICE (mod-2**64 bijection) while the working values are corrupted
# negatives, and int64 cohort arithmetic (max(t, base) + anticipation) can
# overflow. Realistic labels — years, YYYYMMDD, unix s/ms, ns timestamps
# (~1.8e18) — sit far below 2**62 ~ 4.6e18.
_LABEL_MAGNITUDE_BOUND = 2**62


def _validate_n_folds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"n_folds must be an integer >= 2, got {value!r}")
    if value < 2:
        raise ValueError(f"n_folds must be an integer >= 2, got {value}")
    return int(value)


def _validate_learner_spec(spec: Any, *, kind: str, param_name: str) -> None:
    """Eager learner-spec validation naming the ACTUAL constructor param.

    ``make_learner`` hard-codes ``param_name="learner"`` for objects, which
    would mislabel the offending param; strings use the same name check.
    """
    if isinstance(spec, str):
        valid = _REGRESSOR_NAMES if kind == "regressor" else _CLASSIFIER_NAMES
        if spec not in valid:
            raise ValueError(
                f"Unknown {kind} learner {spec!r} for {param_name}; valid names "
                f"for kind={kind!r} are {sorted(valid)}"
            )
        return
    validate_learner(spec, kind=kind, param_name=param_name)


def _raw_label_is_infinite(value: Any) -> bool:
    """True iff the RAW label element is itself infinite (either sign).

    Distinguishes a genuine ``inf`` sentinel (recodable / rejectable as
    non-finite) from a FINITE oversized value — ``Decimal("1e400")`` or the
    string ``"1e400"`` — that ``pd.to_numeric``'s float64 conversion
    overflowed to ``inf``. Exact semantics: Python cross-type comparison for
    numerics (Decimal == inf is exact), exact ``int``/``Decimal`` parsing
    for strings; non-numeric non-string elements cannot be infinite.
    """
    import numbers

    if isinstance(value, numbers.Number) and not isinstance(value, complex):
        try:
            return bool(value == float("inf")) or bool(value == float("-inf"))
        except (TypeError, decimal.InvalidOperation):  # pragma: no cover - defensive
            return False
    if isinstance(value, str):
        try:
            int(value)
            return False
        except ValueError:
            try:
                return bool(decimal.Decimal(value).is_infinite())
            except decimal.InvalidOperation:
                return False
    return False


def _raise_if_conversion_created_inf(col_label: str, raw: np.ndarray, inf_mask: np.ndarray) -> None:
    """Reject float-conversion overflow of FINITE oversized raw labels.

    ``inf_mask`` flags the positions whose float64 conversion is infinite;
    any such position whose RAW element is finite carries a label far above
    the 2**62 magnitude bound and must fail loudly (the +inf recode and the
    non-finite rejection would otherwise misclassify it).
    """
    for pos in np.flatnonzero(inf_mask):
        value = raw[int(pos)]
        if not _raw_label_is_infinite(value):
            raise ValueError(
                f"column {col_label!r} carries label {value!r} whose numeric "
                "conversion overflows float64 to infinity; such labels are far "
                "above the 2**62 magnitude bound for cohort arithmetic — "
                "rescale them (e.g. to period indices)"
            )


_PRIMITIVE_CONFIG_TYPES = (bool, int, float, str, type(None))


def _config_value_is_primitive(value: Any) -> bool:
    """EXACT primitive type check for a native learner's config value.

    A subclass of ``float``/``str``/``int`` carries user code — its
    ``__repr__`` fires inside the learner's configuration repr and its
    value is interpolated into library error messages — so trust only
    exact builtins (plus numpy scalars, whose reprs are library-controlled).
    """
    if type(value) in _PRIMITIVE_CONFIG_TYPES:
        return True
    return isinstance(value, (np.integer, np.floating)) and type(value).__module__ == np.__name__


def _is_native_learner_spec(spec: Any) -> bool:
    # EXACT type check, not isinstance: a user-defined SUBCLASS of a native
    # learner carries arbitrary user code (overridden __repr__, raised
    # exception text) and must stay on the untrusted/foreign path — the
    # sanitization contract (_learner_spec_label, _sanitize_learner_error)
    # keys off this predicate. The CONFIG VALUES must be exact primitives
    # too: a float/str subclass stored as e.g. ridge ``alpha`` would fire
    # its own __repr__ inside the native configuration repr and can seep
    # into library error text, re-opening the nested leak the outer type
    # check closes.
    from diff_diff._learners import LinearLearner, LogitLearner, RidgeLearner, SieveLearner

    if isinstance(spec, str):
        return True
    if type(spec) not in (LinearLearner, LogitLearner, RidgeLearner, SieveLearner):
        return False
    return all(
        _config_value_is_primitive(getattr(spec, attr, None)) for attr in type(spec)._CONFIG_ATTRS
    )


def _learner_spec_label(spec: Any) -> str:
    """Results-facing label for a learner spec (repr-of-spec contract).

    String names verbatim; EXACT library-native learner objects keep their
    controlled configuration repr; everything else — subclasses of the
    natives included — publishes only the qualified class name, since an
    arbitrary user-defined ``__repr__`` could embed credentials, private
    paths, or large payloads into ``summary()`` / ``to_dict()`` exports.
    """
    if isinstance(spec, str):
        return spec
    if _is_native_learner_spec(spec):
        return repr(spec)
    cls = type(spec)
    return f"<{cls.__module__}.{cls.__qualname__}>"


def _validate_seed(seed: Any) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be None or a non-negative integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be None or a non-negative integer, got {seed}")
    return int(seed)


class DMLDiD(CallawaySantAnnaBootstrapMixin, CallawaySantAnnaAggregationMixin, BaseEstimator):
    """Chang (2020) DML DiD: staggered ATT(g,t) with cross-fitted ML nuisances.

    ``panel=True`` (default) estimates Case 1 on panel data; ``panel=False``
    estimates Case 2 on declared repeated cross sections (level outcomes,
    λ-corrected variance).

    Per (g, t) cell: K-fold cross-fitting of the propensity
    (``propensity_learner``, out-of-fold ``predict_proba``) and the
    control-only outcome regression (``outcome_learner``, trained on the
    cell's controls, Chang's ``I_kz^c``), then the pooled orthogonal score
    mean and the augmented-score plug-in SE. On panel fits the folds are
    D-stratified and the outcome nuisance is the outcome-change regression
    ``E[dY | X, D=0]``; on RCS fits the folds are D x T stratified and the
    nuisance is the level regression ``E[(T - lam)Y | X, D=0]`` with the
    lambda-corrected variance. Aggregation (event study, group, simple;
    plus total on panel fits — RCS fits fail ``total`` closed) is POST-FIT
    via ``results.aggregate()``.

    Parameters
    ----------
    propensity_learner : str or object, default "logit"
        ``"logit"``, or ANY object with ``fit``/``predict_proba`` (the
        classifier protocol) — e.g. a user-constructed or sklearn estimator.
        String names select library defaults only.
    outcome_learner : str or object, default "linear"
        ``"linear"``, ``"ridge"``, ``"sieve"``, or ANY object with
        ``fit``/``predict`` (the regressor protocol).
    n_folds : int, default 5
        Cross-fitting folds K (DML2; per-cell assignment).
    control_group : str, default "never_treated"
        ``"never_treated"`` or ``"not_yet_treated"`` (CS semantics).
    anticipation : int, default 0
        Anticipation periods (shifts the base period and the
        not-yet-treated threshold; CS semantics).
    alpha : float, default 0.05
        Significance level.
    n_bootstrap : int, default 0
        Multiplier-bootstrap iterations (0 = analytical inference).
    bootstrap_weights : str, optional
        ``"rademacher"`` (default), ``"mammen"``, or ``"webb"``.
    seed : int, optional
        Root seed for the per-cell fold draws and the bootstrap. With
        ``seed=None``, POINT ESTIMATES vary across fits — cross-fitting
        draws random folds; set ``seed`` for reproducible results with the
        library's deterministic built-in learners. A user-supplied
        STOCHASTIC learner object must additionally be seeded by the user
        (e.g. sklearn ``random_state``) — ``cross_fit_predict`` deep-copies
        the learner template where copyable but never seeds its internal
        RNG.
    base_period : str, default "varying"
        ``"varying"`` or ``"universal"`` (CS semantics; universal
        materializes per-cohort zero reference cells).
    cband : bool, default True
        Uniform (sup-t) bands on the post-fit event-study aggregation's
        bootstrap replay.
    pscore_trim : float, default 0.01
        Propensity clip bound: fitted propensities are clipped to
        ``[pscore_trim, 1 - pscore_trim]`` after the extremeness warning
        (clip, never drop; Chang's paper gives no trimming rule). Must be in
        ``(0, 0.5)`` and large enough that ``1 - pscore_trim < 1`` in
        float64 (a sub-ulp trim would disable the upper clip).
    panel : bool, default True
        ``True`` estimates Chang's Case 1 (repeated outcomes) on panel data
        (one row per unit-period, cell score on outcome CHANGES). ``False``
        estimates Case 2 (repeated cross sections) on DECLARED
        cross-sectional data — one observation per row with a row-unique
        ``unit`` ID, cell score on outcome LEVELS with the post-period
        sampling share λ̂ and the λ-corrected Theorem 2 variance. Case 2
        additionally assumes stationary cross-sectional sampling (Chang
        Assumption 2.3: each wave samples the SAME target population —
        the composition of ``(D, X)`` is stable across waves, while
        outcomes are the period-specific potential outcomes, so trends
        and treatment effects are expected, not violations), which is not
        data-checkable — ``fit()`` warns. ``aggregate('total')`` is unavailable on RCS fits (fails
        closed, the library-wide RC convention).
    """

    _BOOTSTRAP_LABEL: ClassVar[str] = "DMLDiD"

    def __init__(
        self,
        propensity_learner: Union[str, object] = "logit",
        outcome_learner: Union[str, object] = "linear",
        n_folds: int = 5,
        control_group: str = "never_treated",
        anticipation: int = 0,
        alpha: float = 0.05,
        n_bootstrap: int = 0,
        bootstrap_weights: Optional[str] = None,
        seed: Optional[int] = None,
        base_period: str = "varying",
        cband: bool = True,
        pscore_trim: float = 0.01,
        panel: bool = True,
    ) -> None:
        # Raw assignment, then ONE shared validator (also re-run at the top
        # of fit() as the direct-mutation defense) validates and normalizes
        # every estimate/inference-moving parameter.
        self.propensity_learner = propensity_learner
        self.outcome_learner = outcome_learner
        self.n_folds = n_folds
        self.control_group = control_group
        self.anticipation = anticipation
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.base_period = base_period
        self.cband = cband
        self.pscore_trim = pscore_trim
        self.panel = panel
        self._revalidate_config()

        # Fitted-state lifecycle (house convention; not inherited under this MRO).
        self.results_: Optional[DMLDiDResults] = None
        self.is_fitted_ = False

    def _revalidate_config(self) -> None:
        """Validate + normalize EVERY config param from current attributes.

        Called at ``__init__`` and again at the start of ``fit()`` (the
        direct-mutation defense): a mutated ``control_group``/``base_period``
        would otherwise fall through an ``else`` branch and silently select a
        valid-but-unintended methodology while reporting the invalid label.
        ``anticipation`` runs FIRST (assignment form, M-144 — the
        anticipation-policy suite's ordering contract) and normalization is
        idempotent, so a second run is a no-op on valid state.
        """
        self.anticipation = validate_anticipation(self.anticipation)
        _validate_learner_spec(
            self.propensity_learner, kind="classifier", param_name="propensity_learner"
        )
        _validate_learner_spec(self.outcome_learner, kind="regressor", param_name="outcome_learner")
        # Specs stored VERBATIM (a passed learner object is the same object
        # in get_params()); fit-time make_learner does the resolution.
        self.n_folds = _validate_n_folds(self.n_folds)
        if self.control_group not in ("never_treated", "not_yet_treated"):
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{self.control_group}'"
            )
        alpha = self.alpha
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError(f"alpha must be a float in (0, 1), got {alpha!r}")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        validate_n_bootstrap(self.n_bootstrap)
        self.n_bootstrap = int(self.n_bootstrap)
        if self.bootstrap_weights is None:
            self.bootstrap_weights = "rademacher"  # mixin declares str, not Optional
        if self.bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{self.bootstrap_weights}'"
            )
        self.seed = _validate_seed(self.seed)
        if self.base_period not in ("varying", "universal"):
            raise ValueError(
                f"base_period must be 'varying' or 'universal', got '{self.base_period}'"
            )
        if not isinstance(self.cband, (bool, np.bool_)):
            raise ValueError(
                f"cband must be a bool, got {self.cband!r} (type "
                f"{type(self.cband).__name__}) — truthy strings like 'False' "
                "would silently enable bands"
            )
        self.cband = bool(self.cband)
        self.pscore_trim = validate_pscore_trim(self.pscore_trim)
        if not isinstance(self.panel, (bool, np.bool_)):
            raise ValueError(
                f"panel must be a bool, got {self.panel!r} (type "
                f"{type(self.panel).__name__}) — truthy strings like 'False' "
                "would silently select the panel lane"
            )
        self.panel = bool(self.panel)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_and_prepare(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[Iterable[str]],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate inputs; return the numeric working frame + covariate list."""
        # FULL config re-validation FIRST (mutation defense; anticipation
        # leads inside _revalidate_config — the ordering is load-bearing for
        # the anticipation-policy suite, which fits a bare DataFrame and
        # requires the config error to precede column checks).
        self._revalidate_config()

        # covariates are REQUIRED (Chang's estimator exists for the
        # high-dimensional-X setting).
        if isinstance(covariates, (str, bytes)):
            raise ValueError(
                "covariates must be a sequence of column names, not a bare "
                f"string ({covariates!r}); wrap it in a list: covariates="
                f"[{covariates!r}]"
            )
        # Materialize EXACTLY ONCE, before any check that iterates: a
        # one-shot iterable (generator) would otherwise be consumed by the
        # emptiness check and every later list() would see an empty
        # sequence — an intercept-only fit silently dropping the requested
        # covariate adjustment.
        if covariates is not None:
            covariates = list(covariates)
        if covariates is None or len(covariates) == 0:
            raise ValueError(
                "DMLDiD requires covariates: Chang (2020)'s estimator exists "
                "for the conditional-on-covariates setting. For an "
                "unconditional staggered DiD, use CallawaySantAnna (with or "
                "without covariates)."
            )
        validate_covariate_names(
            covariates,
            reserved_names=(outcome, unit, time, first_treat),
            estimator="DMLDiD",
        )
        # (Already a list — materialized once above; a tuple would satisfy
        # the sequence contract but pandas treats a tuple as ONE column key
        # in df[covariates].)

        # Role columns distinct (a collision estimates the wrong quantity
        # silently — verified on CS, which fits outcome=time and returns 0).
        roles = {"outcome": outcome, "unit": unit, "time": time, "first_treat": first_treat}
        seen: Dict[str, str] = {}
        for role, col in roles.items():
            if col in seen:
                raise ValueError(
                    f"outcome/unit/time/first_treat must name distinct columns; "
                    f"{role}={col!r} collides with {seen[col]}={col!r}"
                )
            seen[col] = role

        required_cols = [outcome, unit, time, first_treat, *covariates]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = data.loc[:, required_cols].copy()

        # Identifier checks BEFORE any numeric coercion.
        if df[unit].isna().any():
            raise ValueError(
                f"unit column {unit!r} contains missing values; every row "
                "needs a unit identifier (no silent groupby drop)"
            )
        if df[time].isna().any():
            raise ValueError(f"time column {time!r} contains missing values")
        if df[first_treat].isna().any():
            raise ValueError(
                f"first_treat column {first_treat!r} contains missing values; "
                "use 0 (or +inf) for never-treated units"
            )

        # Numeric coercion of the label columns (targeted error on failure).
        raw_time = df[time].to_numpy(dtype=object)
        raw_ft = df[first_treat].to_numpy(dtype=object)
        try:
            time_numeric = pd.to_numeric(df[time])
            ft_numeric = pd.to_numeric(df[first_treat])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"time/first_treat columns must be numeric-castable: {exc}") from exc
        for col_label, series in ((time, time_numeric), (first_treat, ft_numeric)):
            if pd.api.types.is_complex_dtype(series):
                raise ValueError(f"column {col_label!r} is complex-valued; labels must be real")

        # ±inf time is rejected (it would corrupt base-period arithmetic).
        # Conversion-CREATED infinity (a finite raw label like
        # Decimal("1e400") or "1e400" that pd.to_numeric overflowed to inf)
        # gets its own targeted magnitude error — "non-finite values" would
        # misdescribe a finite input.
        time_vals = time_numeric.to_numpy()
        time_float = time_vals.astype(np.float64)
        time_inf = ~np.isfinite(time_float)
        if time_inf.any():
            _raise_if_conversion_created_inf(time, raw_time, time_inf)
            raise ValueError(f"time column {time!r} contains non-finite values")

        # first_treat: +inf -> 0 recode (CS parity), negative -> hard error.
        # The recode applies ONLY to genuinely infinite RAW values: a finite
        # oversized label whose conversion overflowed to +inf must NOT be
        # silently reclassified as never-treated (the raw-verification guard
        # masks recoded positions, so the recode is the one door around the
        # exact-label certificate — gate it on the raw values).
        ft_vals = ft_numeric.to_numpy()
        ft_float = ft_vals.astype(np.float64)
        ft_any_inf = np.isinf(ft_float)
        if ft_any_inf.any():
            _raise_if_conversion_created_inf(first_treat, raw_ft, ft_any_inf)
        inf_mask = np.isposinf(ft_float)
        if inf_mask.any():
            n_inf_units = df.loc[inf_mask, unit].nunique()
            warnings.warn(
                f"{n_inf_units} unit(s) have first_treat=inf; recoding to 0 "
                f"(never-treated). Use first_treat=0 to suppress this warning.",
                UserWarning,
                stacklevel=3,
            )
            ft_vals = ft_vals.copy()
            ft_vals[inf_mask] = 0
            ft_float[inf_mask] = 0.0
        if np.isneginf(ft_float).any():
            raise ValueError(
                f"first_treat column {first_treat!r} contains -inf; "
                "never-treated units are coded 0 (or +inf)"
            )
        if np.any(ft_float < 0):
            raise ValueError(
                f"first_treat column {first_treat!r} contains negative values; "
                "cohorts are first-treatment periods (never-treated = 0)"
            )

        # Label guards (ALL run after the recode — +inf would otherwise trip
        # the magnitude bound before the recode could fire).
        time_norm, ft_norm = self._normalize_label_columns(
            time_vals, ft_vals, raw_time, raw_ft, inf_mask, time, first_treat
        )
        # assign() replaces the columns wholesale — a setitem into an int64
        # column with float labels would raise pandas' LossySetitemError.
        df = df.assign(**{time: time_norm, first_treat: ft_norm})

        # Outcome and covariates: numeric-castable with a targeted error
        # naming the column, complex rejected, then CONVERTED to float64
        # (load-bearing: pd.to_numeric preserves int64/bool, where a bool
        # outcome TypeErrors in the subtraction and int64 overflow WRAPS to a
        # finite wrong value invisible to errstate/isfinite).
        for col in [outcome, *covariates]:
            try:
                converted = pd.to_numeric(df[col])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"column {col!r} must be numeric-castable for DMLDiD: {exc}"
                ) from exc
            if pd.api.types.is_complex_dtype(converted):
                raise ValueError(
                    f"column {col!r} is complex-valued; casting would silently "
                    "discard imaginary parts — supply real-valued data"
                )
            df[col] = converted.astype(np.float64)

        # Structural design guards. Panel: one row per (unit, time) and a
        # time-invariant cohort label. Declared RCS: one row per unit — with
        # row-unique IDs the two panel guards below are subsumed (duplicate
        # (unit, time) is impossible and per-row first_treat is trivially
        # unit-constant).
        if not self.panel:
            if df[unit].duplicated().any():
                raise ValueError(
                    "panel=False requires unique unit IDs (one observation per "
                    "unit). Found duplicate unit IDs. If your data is a panel, "
                    "use panel=True."
                )
            self._check_control_group_availability(df, unit, first_treat)
            # Emit only AFTER the declared-RCS structure has fully
            # validated — a failing call raises without the misleading
            # suggestion that estimation began.
            warnings.warn(
                "panel=False uses Chang (2020) Case 2 repeated-cross-section "
                "scores, which assume stationary cross-sectional sampling "
                "(Assumption 2.3): each wave samples the SAME target "
                "population — conditional on the period, rows are i.i.d. "
                "draws from the distribution of (Y(0), D, X) (pre) or "
                "(Y(1), D, X) (post), so the composition of (D, X) is stable "
                "across waves while outcomes are the period-specific "
                "potential outcomes (trends and treatment effects are "
                "expected, not violations). This assumption is not "
                "data-checkable.",
                UserWarning,
                stacklevel=3,
            )
            return df, covariates

        # Duplicate (unit, time) rows.
        dup_mask = df.duplicated(subset=[unit, time], keep=False)
        if dup_mask.any():
            example = df.loc[dup_mask, [unit, time]].iloc[0]
            raise ValueError(
                f"duplicate (unit, time) rows: e.g. unit={example[unit]!r} at "
                f"time={example[time]!r}; a panel has one row per unit-period"
            )

        # first_treat time-varying within unit.
        ft_per_unit = df.groupby(unit)[first_treat].nunique()
        if (ft_per_unit > 1).any():
            bad = ft_per_unit[ft_per_unit > 1].index[0]
            raise ValueError(
                f"first_treat varies within unit {bad!r}; the cohort label "
                "must be constant per unit"
            )

        self._check_control_group_availability(df, unit, first_treat)

        return df, covariates

    def _check_control_group_availability(
        self, df: pd.DataFrame, unit: str, first_treat: str
    ) -> None:
        """Control-group availability (CS parity, incl. the NESTING).

        Shared by both design lanes: on declared RCS the groupby-first
        degenerates correctly to per-row under the unique-row-ID guard.
        """
        unit_cohorts_series = df.groupby(unit)[first_treat].first()
        n_never = int((unit_cohorts_series == 0).sum())
        n_cohorts = int(unit_cohorts_series[unit_cohorts_series > 0].nunique())
        if n_never == 0 and self.control_group == "never_treated":
            raise ValueError(
                "No never-treated units found. Check 'first_treat' column. "
                "Use control_group='not_yet_treated' if all units are eventually treated."
            )
        if n_never == 0 and self.control_group == "not_yet_treated":
            if n_cohorts < 2:
                raise ValueError(
                    "not_yet_treated control group requires at least 2 treatment "
                    "cohorts when there are no never-treated units."
                )

    def _normalize_label_columns(
        self,
        time_vals: np.ndarray,
        ft_vals: np.ndarray,
        raw_time: np.ndarray,
        raw_ft: np.ndarray,
        ft_inf_mask: np.ndarray,
        time: str,
        first_treat: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Three label guards + joint canonical-dtype normalization.

        (1) elementwise exact raw-vs-normalized comparison (Python
        cross-type equality; strings parsed exactly; nunique fallback for
        exotic non-str non-numeric elements), (2) the 2**62 magnitude bound
        with anticipation headroom, (3) int64-first / float64-fallback
        normalization with round-trip certificates. Guards (2)-(3) certify
        the post-``pd.to_numeric`` cast; guard (1) covers the coercion step
        itself — the losslessness certificate is the conjunction.
        """
        # (2) magnitude bound (pre-cast; also closes the uint64 double-wrap
        # and the int64 cohort-arithmetic overflow). Compared EXACTLY in the
        # source dtype — a float64 cast of an int just above 2**62 rounds
        # back DOWN to the bound and would slip through a float comparison.
        for col_label, vals in ((time, time_vals), (first_treat, ft_vals)):
            arr = np.asarray(vals)
            if arr.size == 0:
                continue
            if np.issubdtype(arr.dtype, np.integer):
                max_abs: Any = max(abs(int(arr.max())), abs(int(arr.min())))
            else:
                max_abs = float(np.max(np.abs(arr.astype(np.float64))))
            if max_abs > _LABEL_MAGNITUDE_BOUND:
                raise ValueError(
                    f"column {col_label!r} carries labels above 2**62 in "
                    f"magnitude (max |value| = {max_abs:.6g}); such labels are "
                    "not exactly representable for cohort arithmetic — rescale "
                    "them (e.g. to period indices)"
                )
            if max_abs + self.anticipation > _LABEL_MAGNITUDE_BOUND:
                raise ValueError(
                    f"max |{col_label}| + anticipation exceeds 2**62 "
                    f"({max_abs:.6g} + {self.anticipation}); the not-yet-treated "
                    "threshold arithmetic would overflow — rescale the labels"
                )

        # (3) joint canonical dtype: int64 when BOTH columns round-trip
        # losslessly (family-consistent integer rendering), else float64
        # when both round-trip, else targeted error.
        def _lossless(cast: np.ndarray, orig: np.ndarray) -> bool:
            return bool(np.array_equal(cast.astype(orig.dtype), orig))

        candidates: List[np.ndarray] = []
        int_ok = True
        for vals in (time_vals, ft_vals):
            arr = np.asarray(vals)
            cast_i = arr.astype(np.int64)
            if not _lossless(cast_i, arr):
                int_ok = False
                break
            candidates.append(cast_i)
        if not int_ok:
            candidates = []
            for col_label, vals in ((time, time_vals), (first_treat, ft_vals)):
                arr = np.asarray(vals)
                cast_f = arr.astype(np.float64)
                if not _lossless(cast_f, arr):
                    raise ValueError(
                        f"column {col_label!r} labels are not exactly "
                        "representable as int64 or float64; rescale them "
                        "(e.g. to period indices)"
                    )
                candidates.append(cast_f)
        time_norm, ft_norm = candidates

        # (1) elementwise exact raw comparison (catches precision
        # pd.to_numeric ITSELF destroyed: merges AND pure shifts). Runs
        # BEFORE the float-lane arithmetic certificates: a coerced object
        # column with genuinely lost precision should surface the
        # precision diagnosis, not a downstream-arithmetic one.
        for col_label, raw, norm, recoded in (
            (time, raw_time, time_norm, None),
            (first_treat, raw_ft, ft_norm, ft_inf_mask),
        ):
            self._verify_raw_labels(col_label, raw, norm, recoded)

        # Float-lane anticipation-arithmetic exactness certificate: the
        # magnitude bound alone does not make float64 label +/- anticipation
        # EXACT — e.g. labels that are multiples of 512 near 2**62 pass every
        # round-trip, yet `max(t, base) + anticipation` can round ONTO a
        # later cohort and silently flip its not-yet-treated eligibility
        # (and `g - anticipation` base-period arithmetic rounds the same
        # way). Verify, per unique label, that v +/- anticipation is exact
        # in float64 (Fraction comparison is exact); reject otherwise.
        if not int_ok:
            from fractions import Fraction

        if not int_ok and self.anticipation > 0:
            a = int(self.anticipation)
            for col_label, norm in ((time, time_norm), (first_treat, ft_norm)):
                for v in np.unique(norm):
                    fv = float(v)
                    exact = Fraction(fv)
                    if Fraction(fv + a) != exact + a or Fraction(fv - a) != exact - a:
                        raise ValueError(
                            f"column {col_label!r}: label {fv!r} does not support "
                            f"exact anticipation arithmetic in float64 "
                            f"(+/- {a} rounds), so cohort-threshold and "
                            "base-period comparisons would silently shift — "
                            "rescale the labels (e.g. to period indices)"
                        )

        # Float-lane event-time subtraction certificate (ungated on
        # anticipation): every event time e = t - g (base - g included; the
        # base period is itself an observed period) must be EXACT in
        # float64. Mixed-magnitude label sets — e.g. periods 0.5/1.0/1.5
        # with a cohort at 2**62 - 2048 — pass every per-label certificate,
        # yet t - g rounds all of them to the SAME event-time key, silently
        # merging distinct horizons in the analytical and bootstrap
        # aggregations. Exact subtraction is also injective, so distinct
        # exact differences keep distinct keys.
        if not int_ok:
            period_fracs = [(float(v), Fraction(float(v))) for v in np.unique(time_norm)]
            for gval in np.unique(ft_norm):
                if not gval > 0:
                    continue
                fg = float(gval)
                exact_g = Fraction(fg)
                for ft_v, exact_t in period_fracs:
                    if Fraction(ft_v - fg) != exact_t - exact_g:
                        raise ValueError(
                            f"event-time arithmetic is not exact in float64 for "
                            f"period {ft_v!r} and cohort {fg!r} (t - g rounds), "
                            "so distinct event-study horizons would silently "
                            "merge — rescale the labels (e.g. to period indices)"
                        )

        return time_norm, ft_norm

    @staticmethod
    def _verify_raw_labels(
        col_label: str,
        raw: np.ndarray,
        normalized: np.ndarray,
        recoded_mask: Optional[np.ndarray],
    ) -> None:
        import numbers

        expected: List[Any] = []
        parseable = True
        for value in raw:
            if isinstance(value, numbers.Number) and not isinstance(value, complex):
                # Covers int/float/bool, numpy scalars, AND registered
                # exotic numerics (decimal.Decimal, fractions.Fraction) —
                # Python cross-type equality against the normalized
                # int/float is exact, so a Decimal label that
                # pd.to_numeric shifted fails the check like any other.
                expected.append(value)
            elif isinstance(value, str):
                try:
                    expected.append(int(value))
                except ValueError:
                    try:
                        # decimal.Decimal parses the string EXACTLY — a
                        # float() parse would carry the same precision loss
                        # as pd.to_numeric and let a shifted numeric string
                        # (e.g. "9007199254740992.5" -> ...992.0) slip
                        # through the equality check.
                        expected.append(decimal.Decimal(value))
                    except decimal.InvalidOperation:
                        parseable = False
                        break
            else:
                # Non-str non-Number element types (e.g. pandas Timestamps,
                # whose to_numeric ns conversion is exact by construction):
                # cardinality fallback below.
                parseable = False
                break
        if not parseable:
            # Exotic non-str non-numeric element types: fall back to the
            # per-column cardinality comparison.
            n_raw = len(pd.unique(raw))
            n_norm = len(np.unique(normalized))
            if n_raw != n_norm:
                raise ValueError(
                    f"column {col_label!r}: numeric coercion merged distinct "
                    f"labels ({n_raw} unique raw values -> {n_norm} after "
                    "coercion); labels are not exactly representable — "
                    "rescale them (e.g. to period indices)"
                )
            return
        expected_arr = np.asarray(expected, dtype=object)
        if recoded_mask is not None and recoded_mask.any():
            expected_arr = expected_arr.copy()
            expected_arr[recoded_mask] = 0
        # Compare via Python scalars (.item()): numpy-scalar == Decimal is
        # not guaranteed exact cross-type, Python int/float == Decimal is.
        mismatch = np.array(
            [nv.item() != ev for nv, ev in zip(normalized, expected_arr)], dtype=bool
        )
        if np.any(mismatch):
            idx = int(np.flatnonzero(mismatch)[0])
            raise ValueError(
                f"column {col_label!r}: numeric coercion changed label "
                f"{raw[idx]!r} to {normalized[idx]!r} (precision loss); labels "
                "are not exactly representable — rescale them (e.g. to period "
                "indices)"
            )

    # ------------------------------------------------------------------
    # Precompute (host-written, like every non-CS payload producer)
    # ------------------------------------------------------------------

    def _precompute(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: List[str],
    ) -> Dict[str, Any]:
        # CS groupby form — NOT sorted(unique): sorted() raises a bare
        # TypeError on mixed-type unit labels that CS accepts.
        unit_info = df.groupby(unit)[first_treat].first()
        all_units = unit_info.index.values
        unit_cohorts = unit_info.values
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        outcome_wide = df.pivot(index=unit, columns=time, values=outcome).reindex(all_units)
        outcome_matrix = outcome_wide.values
        period_to_col = {t: i for i, t in enumerate(outcome_wide.columns)}

        treatment_groups = sorted(g for g in df[first_treat].unique() if g > 0)
        time_periods = sorted(df[time].unique())

        cohort_masks = {g: unit_cohorts == g for g in treatment_groups}
        never_treated_mask = unit_cohorts == 0

        # One pivot per covariate (not one full-frame scan per period).
        cov_wide = {
            c: df.pivot(index=unit, columns=time, values=c).reindex(all_units) for c in covariates
        }
        covariate_by_period: Dict[Any, np.ndarray] = {
            t: np.column_stack([cov_wide[c][t].values for c in covariates]) for t in time_periods
        }

        n_units = len(all_units)
        # obs_per_unit OMITTED deliberately: only the RC path sets it, and a
        # non-None value divides the aggregation WIF (would silently shrink
        # SEs on a panel). Survey keys likewise omitted (mixin reads are
        # .get with panel fallbacks).
        return {
            "all_units": all_units,
            "unit_to_idx": unit_to_idx,
            "unit_cohorts": unit_cohorts,
            "outcome_matrix": outcome_matrix,
            "period_to_col": period_to_col,
            "observed_sorted": sorted(period_to_col),
            "cohort_masks": cohort_masks,
            "never_treated_mask": never_treated_mask,
            "covariate_by_period": covariate_by_period,
            "time_periods": time_periods,
            "treatment_groups": treatment_groups,
            "is_panel": True,
            "canonical_size": n_units,
            "n_units": n_units,
        }

    def _precompute_rcs(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: List[str],
    ) -> Dict[str, Any]:
        """Declared-RCS bookkeeping: rows ARE the sampling units.

        Mirrors the minimal CS RCS precompute contract: ``all_units`` =
        observation positions, ``unit_to_idx`` = None, per-ROW
        ``unit_cohorts``, ``canonical_size`` = n_obs — the aggregation and
        bootstrap mixins branch off exactly these (per-row multiplier
        weights; per-obs cohort bincount pg basis; ``aggregate('total')``
        fails closed on ``is_panel: False``).

        ``agg_cohort_masses`` is deliberately NOT set: under unique row IDs
        the aggregation cache's bincount fallback yields exactly the fixed
        per-cohort row masses (never-treated included, denominator = n_obs),
        so the WIF pg basis is already the fixed-cohort-mass one and the
        per-cell ``agg_weight`` (from ``rcs_cohort_masses``, int-keyed by the
        canonical labels) aligns the point-estimate weights with it.
        Supplying the key would be a numeric no-op that routes lookups
        through float()-keyed dict access, where distinct int64 cohorts
        above 2**53 (admissible through 2**62 by the label pipeline)
        collide. ``obs_per_unit`` and survey keys likewise omitted (true RCS
        is one row per unit — a non-None obs_per_unit would divide the WIF).
        """
        n_obs = len(df)
        unit_cohorts = df[first_treat].to_numpy()
        treatment_groups = sorted(g for g in df[first_treat].unique() if g > 0)
        time_periods = sorted(df[time].unique())
        period_to_col = {t: i for i, t in enumerate(time_periods)}
        return {
            "all_units": np.arange(n_obs),
            "unit_to_idx": None,
            "unit_cohorts": unit_cohorts,
            "obs_time": df[time].to_numpy(),
            "obs_outcome": df[outcome].to_numpy(),
            "obs_covariates": df[covariates].to_numpy(dtype=np.float64),
            "cohort_masks": {g: unit_cohorts == g for g in treatment_groups},
            # +inf cohorts were recoded to 0 upstream, so == 0 is complete.
            "never_treated_mask": unit_cohorts == 0,
            "period_to_col": period_to_col,
            "observed_sorted": sorted(period_to_col),
            "time_periods": time_periods,
            "treatment_groups": treatment_groups,
            "is_panel": False,
            "canonical_size": n_obs,
            "n_units": n_obs,
            "rcs_cohort_masses": {
                g: int(np.count_nonzero(unit_cohorts == g)) for g in treatment_groups
            },
        }

    def _sanitize_learner_error(self, exc: BaseException) -> str:
        """Persisted error text for cross_fit_diagnostics / to_dict exports.

        Library-native learner specs produce library-controlled messages —
        kept verbatim. With a FOREIGN learner object in play, the exception
        text may embed credentials, private paths, or data excerpts from the
        user's own code, so ONLY the exception type is recorded anywhere
        (the cell fails soft to a NaN skip; nothing re-raises the full
        message).
        """
        if _is_native_learner_spec(self.propensity_learner) and _is_native_learner_spec(
            self.outcome_learner
        ):
            return str(exc)
        return (
            f"{type(exc).__name__} (message withheld: a foreign learner "
            "object is in play and its error text may embed sensitive "
            "content — reproduce locally by calling the learner's fit "
            "directly on the failing cell's data to see the full message)"
        )

    # ------------------------------------------------------------------
    # Per-cell DML computation
    # ------------------------------------------------------------------

    def _compute_dml_gt(
        self,
        precomputed: Dict[str, Any],
        g: Any,
        t: Any,
        g_idx: int,
        t_idx: int,
        root_entropy: int,
        dropped_units_out: Optional[set] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """One (g, t) cell. Returns (gt_entry, if_entry|None, diagnostics|None)."""
        observed_sorted = precomputed["observed_sorted"]
        period_to_col = precomputed["period_to_col"]
        base = _select_base_period_impl(self.base_period, self.anticipation, g, t, observed_sorted)
        if base is None or base not in period_to_col or t not in period_to_col:
            return _nan_gt_entry(skip_reason="missing_period"), None, None

        unit_cohorts = precomputed["unit_cohorts"]
        outcome_matrix = precomputed["outcome_matrix"]
        treated_mask = precomputed["cohort_masks"][g]
        if self.control_group == "never_treated":
            control_mask = precomputed["never_treated_mask"]
        else:  # not_yet_treated (CS semantics: untreated at max(t, base) + k)
            nyt_threshold = max(t, base) + self.anticipation
            control_mask = precomputed["never_treated_mask"] | (
                (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
            )

        X_by_period = precomputed["covariate_by_period"]
        base_col = period_to_col[base]
        t_col = period_to_col[t]
        with np.errstate(over="ignore", invalid="ignore"):
            y_base = outcome_matrix[:, base_col]
            y_post = outcome_matrix[:, t_col]
            dY = y_post - y_base
            X_base = X_by_period[base]
            cov_finite = np.all(np.isfinite(X_base), axis=1)
            valid = np.isfinite(y_base) & np.isfinite(y_post) & np.isfinite(dY) & cov_finite

        treated_valid = treated_mask & valid
        control_valid = control_mask & valid
        if dropped_units_out is not None:
            dropped_units_out.update(
                np.flatnonzero((treated_mask | control_mask) & ~valid).tolist()
            )
        n_treated = int(np.sum(treated_valid))
        n_control = int(np.sum(control_valid))
        if n_treated == 0 or n_control == 0:
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="zero_treated_control",
                ),
                None,
                None,
            )

        cell_mask = treated_valid | control_valid
        cell_idx = np.flatnonzero(cell_mask)
        n_cell = cell_idx.shape[0]
        D_cell = treated_valid[cell_idx].astype(np.float64)
        dY_cell = dY[cell_idx]
        X_cell = X_base[cell_idx]
        p_hat = n_treated / n_cell
        # Empirical treated-share overlap diagnostic: every Chang bound
        # carries powers of 1/p_0 (paper review, few-treated edge case), and
        # the fitted-propensity check below cannot see a sparse EMPIRICAL
        # share when a learner returns non-extreme predictions.
        if min(p_hat, 1.0 - p_hat) < self.pscore_trim:
            warnings.warn(
                f"DMLDiD cell (g={g}, t={t}): empirical treated share "
                f"p_hat={p_hat:.4f} ({n_treated} treated / {n_control} "
                f"control) is extreme (min(p, 1-p) < pscore_trim="
                f"{self.pscore_trim}); the Chang score and variance scale "
                "with powers of 1/p_hat, so this cell's estimate may be "
                "unstable.",
                UserWarning,
                stacklevel=3,
            )

        # Per-cell fold draw: seeded from (root, (g_idx, t_idx)) — positions
        # in the sorted cohort/period rosters, invariant to other cells'
        # estimability. D-stratified (DoubleML parity; documented REGISTRY
        # deviation from Chang's plain random partition).
        seed_seq = np.random.SeedSequence(entropy=root_entropy, spawn_key=(g_idx, t_idx))
        rng = np.random.default_rng(seed_seq)
        diagnostics: Dict[str, Any] = {
            "propensity": None,
            "outcome": None,
            "p_hat": float(p_hat),
            "n_clipped_ps": None,
            "fold_seed": {"entropy": int(root_entropy), "spawn_key": [int(g_idx), int(t_idx)]},
        }

        try:
            folds = assign_folds(n_cell, self.n_folds, rng=rng, stratify=D_cell)
        except ValueError as exc:
            # Cell smaller than n_folds, or a singleton D-stratum (one
            # treated/control unit cannot be cross-fitted).
            diagnostics["skip_reason"] = "cross_fit_degenerate"
            diagnostics["error"] = str(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="cross_fit_degenerate",
                ),
                None,
                diagnostics,
            )

        context = f"DMLDiD (g={g}, t={t})"
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                ps_res = cross_fit_predict(
                    make_learner(self.propensity_learner, kind="classifier"),
                    X_cell,
                    D_cell,
                    folds,
                    predict_method="predict_proba",
                    context_label=f"{context} propensity",
                )
                or_res = cross_fit_predict(
                    make_learner(self.outcome_learner, kind="regressor"),
                    X_cell,
                    dY_cell,
                    folds,
                    predict_method="predict",
                    fit_mask=(D_cell == 0.0),
                    context_label=f"{context} outcome",
                )
        except DegenerateFoldError as exc:
            diagnostics["skip_reason"] = "cross_fit_degenerate"
            diagnostics["error"] = self._sanitize_learner_error(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="cross_fit_degenerate",
                ),
                None,
                diagnostics,
            )

        ps_raw = ps_res.oof_predictions
        _check_propensity_diagnostics(ps_raw, self.pscore_trim)
        ps = np.clip(ps_raw, self.pscore_trim, 1.0 - self.pscore_trim)
        n_clipped = int(np.sum((ps_raw < self.pscore_trim) | (ps_raw > 1.0 - self.pscore_trim)))
        m_hat = or_res.oof_predictions

        diagnostics["propensity"] = {
            "fold_losses": [float(v) for v in ps_res.fold_losses],
            "n_fit_per_fold": [int(v) for v in ps_res.n_fit_per_fold],
        }
        diagnostics["outcome"] = {
            "fold_losses": [float(v) for v in or_res.fold_losses],
            "n_fit_per_fold": [int(v) for v in or_res.n_fit_per_fold],
        }
        diagnostics["n_clipped_ps"] = n_clipped

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                summand = chang_panel_score(dY_cell, D_cell, m_hat, ps, p_hat)
                theta = float(np.mean(summand))
                psi_bar = chang_panel_score_augmented(summand, D_cell, theta, p_hat)
                se = float(np.sqrt(np.mean(psi_bar**2) / n_cell))
        except ValueError as exc:
            diagnostics["skip_reason"] = "non_finite_score"
            diagnostics["error"] = self._sanitize_learner_error(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="non_finite_score",
                ),
                None,
                diagnostics,
            )
        if not (np.isfinite(theta) and np.isfinite(se) and np.all(np.isfinite(psi_bar))):
            diagnostics["skip_reason"] = "non_finite_score"
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="non_finite_score",
                ),
                None,
                diagnostics,
            )

        t_stat, p_value, conf_int = safe_inference(theta, se, alpha=self.alpha)
        gt_entry = {
            "effect": theta,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_int": conf_int,
            "n_treated": n_treated,
            "n_control": n_control,
            "skip_reason": None,
        }

        # Payload: per-unit entries psi_bar_i / n_cell, so sqrt(sum(if^2))
        # IS the cell SE (the CS influence_func_info contract).
        n_units = precomputed["n_units"]
        inf_full = np.zeros(n_units)
        inf_full[cell_idx] = psi_bar / n_cell
        treated_idx = np.flatnonzero(treated_valid).astype(np.int64)
        control_idx = np.flatnonzero((inf_full != 0.0) & ~treated_valid).astype(np.int64)
        if_entry = {
            "treated_idx": treated_idx,
            "control_idx": control_idx,
            "treated_inf": inf_full[treated_idx],
            "control_inf": inf_full[control_idx],
        }
        return gt_entry, if_entry, diagnostics

    def _compute_dml_rcs_gt(
        self,
        precomputed: Dict[str, Any],
        g: Any,
        t: Any,
        g_idx: int,
        t_idx: int,
        root_entropy: int,
        dropped_units_out: Optional[set] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """One RCS (g, t) cell — Chang Case 2. Same return contract as the
        panel cell: (gt_entry, if_entry|None, diagnostics|None).

        Pooled two-period cell on LEVEL outcomes: four disjoint row groups
        (treated/control x {t, base}), one cross-fit propensity on the pooled
        rows, and the SINGLE Case 2 outcome nuisance — a control-only (both
        periods, Chang's I_kz^c) regression of (T - lam_hat) * y on X.
        """
        observed_sorted = precomputed["observed_sorted"]
        period_to_col = precomputed["period_to_col"]
        base = _select_base_period_impl(self.base_period, self.anticipation, g, t, observed_sorted)
        if base is None or base not in period_to_col or t not in period_to_col:
            return _nan_gt_entry(skip_reason="missing_period"), None, None

        unit_cohorts = precomputed["unit_cohorts"]
        treated_mask = precomputed["cohort_masks"][g]
        if self.control_group == "never_treated":
            control_mask = precomputed["never_treated_mask"]
        else:  # not_yet_treated (CS semantics: untreated at max(t, base) + k)
            nyt_threshold = max(t, base) + self.anticipation
            control_mask = precomputed["never_treated_mask"] | (
                (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
            )

        obs_time = precomputed["obs_time"]
        y_obs = precomputed["obs_outcome"]
        X_obs = precomputed["obs_covariates"]
        at_t = obs_time == t
        at_base = obs_time == base
        # Per-ROW complete cases (RCS covariates live on the row itself; no
        # base-period X exists).
        valid = np.isfinite(y_obs) & np.all(np.isfinite(X_obs), axis=1)

        treated_t = treated_mask & at_t & valid
        treated_b = treated_mask & at_base & valid
        control_t = control_mask & at_t & valid
        control_b = control_mask & at_base & valid
        if dropped_units_out is not None:
            dropped_units_out.update(
                np.flatnonzero((treated_mask | control_mask) & (at_t | at_base) & ~valid).tolist()
            )
        n_treated = int(np.sum(treated_t | treated_b))
        n_control = int(np.sum(control_t | control_b))
        # FOUR-group guard (CS RCS precedent): the Case 2 estimand needs
        # treated AND control rows in BOTH periods; skip vocabulary reuses
        # zero_treated_control for any empty group.
        if (
            min(
                int(treated_t.sum()),
                int(treated_b.sum()),
                int(control_t.sum()),
                int(control_b.sum()),
            )
            == 0
        ):
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="zero_treated_control",
                ),
                None,
                None,
            )

        cell_mask = treated_t | treated_b | control_t | control_b
        cell_idx = np.flatnonzero(cell_mask)
        n_cell = cell_idx.shape[0]
        D_cell = treated_mask[cell_idx].astype(np.float64)
        T_cell = at_t[cell_idx].astype(np.float64)
        y_cell = y_obs[cell_idx]
        X_cell = X_obs[cell_idx]
        # Global-within-cell shares (REGISTRY convention: mirrors the Case 1
        # global p_hat and DoubleML's d.mean()/t.mean()); strictly interior
        # BY the four-group guard.
        p_hat = float(D_cell.mean())
        lam_hat = float(T_cell.mean())
        if min(p_hat, 1.0 - p_hat) < self.pscore_trim:
            warnings.warn(
                f"DMLDiD cell (g={g}, t={t}): empirical treated share "
                f"p_hat={p_hat:.4f} over the pooled two-period rows "
                f"({n_treated} treated / {n_control} control) is extreme "
                f"(min(p, 1-p) < pscore_trim={self.pscore_trim}); the Chang "
                "score and variance scale with powers of 1/p_hat, so this "
                "cell's estimate may be unstable.",
                UserWarning,
                stacklevel=3,
            )
        if min(lam_hat, 1.0 - lam_hat) < self.pscore_trim:
            warnings.warn(
                f"DMLDiD cell (g={g}, t={t}): post-period sampling share "
                f"lam_hat={lam_hat:.4f} is extreme (min(lam, 1-lam) < "
                f"pscore_trim={self.pscore_trim}); Chang's Case 2 bounds "
                "carry powers of 1/(lam*(1-lam)) up to cubes, so this cell's "
                "estimate may be unstable.",
                UserWarning,
                stacklevel=3,
            )

        # Per-cell fold draw seeded exactly like the panel lane; strata are
        # the FOUR D x T classes (DoubleML's d + 2t encoding — REGISTRY
        # deviation Note): every training complement then carries control
        # rows in both periods by construction, Chang's fold-composition
        # requirement for fitting l_2 on I_kz^c.
        seed_seq = np.random.SeedSequence(entropy=root_entropy, spawn_key=(g_idx, t_idx))
        rng = np.random.default_rng(seed_seq)
        diagnostics: Dict[str, Any] = {
            "propensity": None,
            "outcome": None,
            "p_hat": float(p_hat),
            "lam_hat": float(lam_hat),
            "n_clipped_ps": None,
            "fold_seed": {"entropy": int(root_entropy), "spawn_key": [int(g_idx), int(t_idx)]},
        }

        try:
            folds = assign_folds(n_cell, self.n_folds, rng=rng, stratify=D_cell + 2.0 * T_cell)
        except ValueError as exc:
            # Cell smaller than n_folds, or a singleton D x T stratum (e.g.
            # ONE treated row in the base period cannot be cross-fitted).
            diagnostics["skip_reason"] = "cross_fit_degenerate"
            diagnostics["error"] = str(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="cross_fit_degenerate",
                ),
                None,
                diagnostics,
            )

        context = f"DMLDiD (g={g}, t={t})"
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                ps_res = cross_fit_predict(
                    make_learner(self.propensity_learner, kind="classifier"),
                    X_cell,
                    D_cell,
                    folds,
                    predict_method="predict_proba",
                    context_label=f"{context} propensity",
                )
                r_cell = (T_cell - lam_hat) * y_cell
                or_res = cross_fit_predict(
                    make_learner(self.outcome_learner, kind="regressor"),
                    X_cell,
                    r_cell,
                    folds,
                    predict_method="predict",
                    fit_mask=(D_cell == 0.0),
                    context_label=f"{context} outcome",
                )
        except DegenerateFoldError as exc:
            diagnostics["skip_reason"] = "cross_fit_degenerate"
            diagnostics["error"] = self._sanitize_learner_error(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="cross_fit_degenerate",
                ),
                None,
                diagnostics,
            )

        ps_raw = ps_res.oof_predictions
        _check_propensity_diagnostics(ps_raw, self.pscore_trim)
        ps = np.clip(ps_raw, self.pscore_trim, 1.0 - self.pscore_trim)
        n_clipped = int(np.sum((ps_raw < self.pscore_trim) | (ps_raw > 1.0 - self.pscore_trim)))
        m2_hat = or_res.oof_predictions

        diagnostics["propensity"] = {
            "fold_losses": [float(v) for v in ps_res.fold_losses],
            "n_fit_per_fold": [int(v) for v in ps_res.n_fit_per_fold],
        }
        diagnostics["outcome"] = {
            "fold_losses": [float(v) for v in or_res.fold_losses],
            "n_fit_per_fold": [int(v) for v in or_res.n_fit_per_fold],
        }
        diagnostics["n_clipped_ps"] = n_clipped

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                summand = chang_rcs_score(y_cell, D_cell, T_cell, m2_hat, ps, p_hat, lam_hat)
                theta = float(np.mean(summand))
                psi_bar, g2_lambda = _chang_rcs_score_augmented_with_slope(
                    summand, D_cell, T_cell, y_cell, m2_hat, ps, theta, p_hat, lam_hat
                )
                se = float(np.sqrt(np.mean(psi_bar**2) / n_cell))
        except ValueError as exc:
            diagnostics["skip_reason"] = "non_finite_score"
            diagnostics["error"] = self._sanitize_learner_error(exc)
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="non_finite_score",
                ),
                None,
                diagnostics,
            )
        if not (
            np.isfinite(theta)
            and np.isfinite(se)
            and np.isfinite(g2_lambda)
            and np.all(np.isfinite(psi_bar))
        ):
            diagnostics["skip_reason"] = "non_finite_score"
            return (
                _nan_gt_entry(
                    n_treated=n_treated,
                    n_control=n_control,
                    skip_reason="non_finite_score",
                ),
                None,
                diagnostics,
            )
        diagnostics["g2_lambda"] = float(g2_lambda)

        t_stat, p_value, conf_int = safe_inference(theta, se, alpha=self.alpha)
        gt_entry = {
            "effect": theta,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_int": conf_int,
            # DISPLAY counts: pooled two-period valid rows. Aggregation
            # weights come from agg_weight below (fixed cohort row mass, the
            # CS-RCS convention) — never from these counts.
            "n_treated": n_treated,
            "n_control": n_control,
            "skip_reason": None,
            "agg_weight": precomputed["rcs_cohort_masses"][g],
        }

        # Payload: per-OBS entries psi_bar_i / n_cell over BOTH periods'
        # rows, so sqrt(sum(if^2)) IS the cell SE. cell_idx is a flatnonzero
        # of one mask => strictly increasing and duplicate-free, and the
        # D-partition keeps treated_idx/control_idx disjoint (the fancy-+=
        # scatter contract in the aggregation layer).
        n_units = precomputed["n_units"]
        inf_full = np.zeros(n_units)
        inf_full[cell_idx] = psi_bar / n_cell
        treated_idx = cell_idx[D_cell == 1.0].astype(np.int64)
        control_idx = cell_idx[D_cell == 0.0].astype(np.int64)
        if_entry = {
            "treated_idx": treated_idx,
            "control_idx": control_idx,
            "treated_inf": inf_full[treated_idx],
            "control_inf": inf_full[control_idx],
        }
        return gt_entry, if_entry, diagnostics

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[Iterable[str]] = None,
    ) -> DMLDiDResults:
        """Estimate staggered ATT(g,t) via per-cell cross-fitted Chang scores."""
        df, covariates = self._validate_and_prepare(
            data, outcome, unit, time, first_treat, covariates
        )
        if self.panel:
            precomputed = self._precompute(df, outcome, unit, time, first_treat, covariates)
            cell_fn = self._compute_dml_gt
        else:
            precomputed = self._precompute_rcs(df, outcome, unit, time, first_treat, covariates)
            cell_fn = self._compute_dml_rcs_gt
        treatment_groups = precomputed["treatment_groups"]
        time_periods = precomputed["time_periods"]
        observed_sorted = precomputed["observed_sorted"]
        unit_cohorts = precomputed["unit_cohorts"]

        root_entropy = self.seed if self.seed is not None else secrets.randbits(63)

        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        cross_fit_diagnostics: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        skipped: Dict[str, List[Tuple[Any, Any]]] = {}
        skip_errors: List[str] = []
        dropped_units: set = set()

        for g_idx, g in enumerate(treatment_groups):
            for t in _valid_periods_for_group_impl(
                self.base_period, self.anticipation, g, time_periods, observed_sorted
            ):
                t_idx = time_periods.index(t)
                gt_entry, if_entry, diagnostics = cell_fn(
                    precomputed, g, t, g_idx, t_idx, root_entropy, dropped_units
                )
                group_time_effects[(g, t)] = gt_entry
                if if_entry is not None:
                    influence_func_info[(g, t)] = if_entry
                if diagnostics is not None:
                    cross_fit_diagnostics[(g, t)] = diagnostics
                reason = gt_entry.get("skip_reason")
                if reason is not None:
                    skipped.setdefault(reason, []).append((g, t))
                    if diagnostics is not None and "error" in diagnostics:
                        skip_errors.append(f"(g={g}, t={t}): {diagnostics['error']}")

        # All-degenerate check BEFORE reference cells (CS ordering): a
        # universal-base all-NaN fit must raise, not return reference-only
        # results.
        if not group_time_effects or not any(
            np.isfinite(v["effect"]) for v in group_time_effects.values()
        ):
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Consolidated skip warning (fires only when >= 1 cell survives).
        n_skipped = sum(len(v) for v in skipped.values())
        if n_skipped > 0:
            parts = [
                f"{len(cells)} {reason} {sorted(cells)}"
                for reason, cells in sorted(skipped.items())
            ]
            detail = ""
            if skip_errors:
                detail = " Learner errors: " + " | ".join(skip_errors)
            warnings.warn(
                f"{n_skipped} (group, time) cell(s) could not be estimated: "
                f"{'; '.join(parts)}.{detail}",
                UserWarning,
                stacklevel=2,
            )

        # Unbalanced-input warning (no-silent-failures): units dropped from a
        # cell they would otherwise join — missing/non-finite outcome,
        # non-finite covariate, or a dY overflow from two finite outcomes
        # (errstate suppresses the only runtime signal, so this warning is
        # the sole user-visible trace). Accumulated per cell during the
        # estimation loop — no second O(n_units x n_cells) sweep.
        if dropped_units:
            if self.panel:
                warnings.warn(
                    f"{len(dropped_units)} unit(s) were excluded from at least one "
                    "(group, time) cell they would otherwise join, due to a "
                    "missing or NON-FINITE outcome, a non-finite covariate at the "
                    "cell's base period, or an outcome difference overflowing to "
                    "non-finite. DMLDiD estimates each cell on its complete cases "
                    "(point weights use per-cell valid counts; aggregation cohort "
                    "masses use full cohorts — see REGISTRY.md).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"{len(dropped_units)} observation(s) were excluded from a "
                    "(group, time) cell they would otherwise join, due to a "
                    "missing or NON-FINITE outcome or a non-finite covariate "
                    "on the row. DMLDiD estimates each cell on its complete "
                    "cases (aggregation weights use fixed cohort row masses — "
                    "see REGISTRY.md).",
                    UserWarning,
                    stacklevel=2,
                )

        # Universal base period: per-cohort zero reference cells (full
        # nine-key CS dict; the kit hard-reads effect AND n_treated, the
        # aggregators read n_treated as the pg mass, renderers read se).
        reference_event_times: Optional[Tuple[Any, ...]] = None
        if self.base_period == "universal":
            for g in treatment_groups:
                base = _select_base_period_impl(
                    self.base_period, self.anticipation, g, g, observed_sorted
                )
                if base is None or (g, base) in group_time_effects:
                    continue
                cohort_mass = float(np.count_nonzero(unit_cohorts == g))
                if cohort_mass <= 0:
                    continue
                ref_entry: Dict[str, Any] = {
                    "effect": 0.0,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_treated": int(round(cohort_mass)),
                    "n_control": 0,
                    "skip_reason": None,
                    "is_reference": True,
                }
                if not self.panel:
                    # Keep the RCS pg basis uniform: reference cells carry the
                    # same fixed cohort row mass as estimated cells.
                    ref_entry["agg_weight"] = precomputed["rcs_cohort_masses"].get(
                        g, int(round(cohort_mass))
                    )
                group_time_effects[(g, base)] = ref_entry
                influence_func_info[(g, base)] = {
                    "treated_idx": np.array([], dtype=np.int64),
                    "control_idx": np.array([], dtype=np.int64),
                    "treated_inf": np.array([]),
                    "control_inf": np.array([]),
                }
            ref_event_times = set()
            for g in treatment_groups:
                b = _select_base_period_impl(
                    self.base_period, self.anticipation, g, g, observed_sorted
                )
                if b is not None:
                    ref_event_times.add(b - g)
            if ref_event_times:
                reference_event_times = tuple(sorted(ref_event_times))

        # Overall ATT (simple aggregation over post-treatment finite cells).
        overall_att, overall_se, _ = self._aggregate_simple(
            group_time_effects, influence_func_info, df, unit, precomputed
        )
        overall_t_stat, overall_p_value, overall_conf_int = safe_inference(
            overall_att, overall_se, alpha=self.alpha
        )

        # Optional multiplier bootstrap (keyword form; aggregate=None is the
        # supported skip-the-event-study-prep input — post-fit aggregate()
        # supplies its own level on replay). Adapted engine override block:
        # the fit-time event-study sub-block and the dead cband lines are
        # dropped; every df=df_survey becomes df=None (plain normal theory).
        bootstrap_results = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects,
                influence_func_info,
                aggregate=None,
                balance_e=None,
                treatment_groups=treatment_groups,
                time_periods=time_periods,
                df=df,
                unit=unit,
                precomputed=precomputed,
                cband=self.cband,
            )
            if bootstrap_results is not None:
                overall_se = bootstrap_results.overall_att_se
                overall_t_stat, overall_p_value, overall_conf_int = safe_inference(
                    overall_att, overall_se, alpha=self.alpha, df=None
                )
                overall_conf_int = bootstrap_results.overall_att_ci
                overall_p_value = bootstrap_results.overall_att_p_value
                if bootstrap_results.group_time_ses:
                    for gt_key in group_time_effects:
                        if gt_key in bootstrap_results.group_time_ses:
                            group_time_effects[gt_key]["se"] = bootstrap_results.group_time_ses[
                                gt_key
                            ]
                            group_time_effects[gt_key]["conf_int"] = (
                                bootstrap_results.group_time_cis[gt_key]
                            )
                            group_time_effects[gt_key]["p_value"] = (
                                bootstrap_results.group_time_p_values[gt_key]
                            )
                            t_val, _, _ = safe_inference(
                                group_time_effects[gt_key]["effect"],
                                bootstrap_results.group_time_ses[gt_key],
                                alpha=self.alpha,
                                df=None,
                            )
                            group_time_effects[gt_key]["t_stat"] = t_val

        n_treated_units = int(np.sum(unit_cohorts > 0))
        n_control_units = int(np.sum(unit_cohorts == 0))

        results = DMLDiDResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t_stat,
            overall_p_value=overall_p_value,
            overall_conf_int=overall_conf_int,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=int(len(df)),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            base_period=self.base_period,
            anticipation=self.anticipation,
            pscore_trim=self.pscore_trim,
            bootstrap_results=bootstrap_results,
            cband_crit_value=None,  # dead at fit under aggregate=None
            reference_event_times=reference_event_times,
            # Repr-of-spec contract: the RESULTS object stores a string —
            # never the learner object itself (result pickles must not
            # retain arbitrary user objects/buffers). Library-native
            # learners publish their controlled configuration repr; FOREIGN
            # objects publish only the qualified class name (an arbitrary
            # __repr__ could embed credentials/paths into summaries and
            # to_dict exports).
            propensity_learner=_learner_spec_label(self.propensity_learner),
            outcome_learner=_learner_spec_label(self.outcome_learner),
            n_folds=self.n_folds,
            cross_fit_diagnostics=cross_fit_diagnostics,
            seed=self.seed,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            cband=self.cband,
            panel=self.panel,
        )
        results._aggregation_kit = _build_aggregation_kit(
            cast(Any, self),  # duck-typed host contract (alpha/anticipation/cband)
            precomputed,
            influence_func_info,
            group_time_effects,
            bootstrap_results=bootstrap_results,
        )
        self.results_ = results
        self.is_fitted_ = True
        return results
