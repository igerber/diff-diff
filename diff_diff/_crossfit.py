"""Unit-level K-fold cross-fitting for DML nuisance estimation (private infra).

Fold assignment is deterministic and REPLAYABLE: ``assign_folds`` captures the
generator's bit-generator state BY VALUE before any draw (the
``aggregation.BootstrapReplaySpec`` discipline), so ``FoldAssignment.replay()``
reproduces the exact assignment from the stored state alone. Assignment
members are UNITS when ``cluster_ids is None`` and CLUSTERS otherwise (all
units of a cluster share a fold); balance and non-emptiness guarantees hold at
the member level.

``cross_fit_predict`` produces out-of-fold nuisance predictions for EVERY
unit: for each fold k the learner is fit on ``train_mask(k) & fit_mask`` and
predicts all units in fold k. Each fold fits a DEEP COPY of the user's
(never-fit) learner template, so no state — nested estimators and container
parameters included — can carry across folds; an un-deep-copyable learner is
reused with a loud warning under the fit-reset contract (see
``diff_diff._learners``).

Exception semantics (determinate):

- ``assign_folds`` and ``cross_fit_predict``'s OWN argument validation raise
  plain ``ValueError`` — except a wrong-typed ``rng`` (not an
  ``np.random.Generator``), which raises ``TypeError`` per Python convention.
- Fold-time degeneracy raises ``DegenerateFoldError`` (a ``ValueError``
  subclass): cheap universal pre-checks (empty fit set, zero positive weight,
  single-class labels for classifiers) raise it directly; any ``ValueError``
  from the learner's ``fit``/``predict``/``predict_proba`` or from prediction
  validation is re-raised as ``DegenerateFoldError`` chained via ``raise ...
  from exc`` with the fold index and the underlying message quoted verbatim
  (no solver-specific threshold enumeration, no masking — the original
  traceback is preserved). Non-``ValueError`` exceptions propagate untouched.
"""

import copy
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, cast, overload

import numpy as np
import pandas as pd

from diff_diff._learners import (
    ClassifierLearner,
    RegressorLearner,
    _validate_predictions,
    validate_learner,
)

__all__ = [
    "DegenerateFoldError",
    "FoldAssignment",
    "CrossFitResult",
    "assign_folds",
    "cross_fit_predict",
]

_LOG_LOSS_CLIP = 1e-15


def _fresh_learner(learner: Any) -> Any:
    """Per-fold learner isolation: a deep copy of the (never-fit) template.

    ``copy.deepcopy`` of the user's template gives every fold a fully
    independent learner — nested estimators, estimators inside lists/dicts,
    accumulators, and warm-start state included — so no state (and therefore
    no data from a previous complement, which includes the current evaluation
    fold) can carry across folds. This is strictly stronger than
    get_params-based reconstruction (which shares any estimator stored inside
    a container parameter). The template itself is never fit. A copy FAILURE
    is never silent: the instance is reused with a loud ``UserWarning`` naming
    the learner and the fit-reset assumption now being relied on
    (no-silent-failures rule).
    """
    try:
        return copy.deepcopy(learner)
    except Exception as exc:  # noqa: BLE001 - loud fallback, never silent
        # Exception CLASS only, never the message: a foreign learner's
        # __deepcopy__ error text can embed credentials/paths/data excerpts,
        # and this warning lands in notebook/CI logs (the same boundary as
        # DMLDiD's persisted-diagnostics sanitization).
        warnings.warn(
            f"cross_fit_predict: could not deep-copy the "
            f"{type(learner).__name__} template for this fold "
            f"({type(exc).__name__}); "
            "REUSING the same instance and relying on its fit-reset behavior. "
            "A warm-start/stateful learner in this situation can leak data "
            "across folds.",
            UserWarning,
            stacklevel=3,
        )
        return learner


def _unique_or_raise(arr: np.ndarray, name: str, **kwargs: Any) -> Any:
    """np.unique with mixed-type object labels surfaced as a targeted ValueError."""
    try:
        return np.unique(arr, **kwargs)
    except TypeError as exc:
        raise ValueError(
            f"{name} contains non-comparable mixed-type labels ({exc}); "
            "use one consistent label type"
        ) from exc


class DegenerateFoldError(ValueError):
    """A fold's fit subset is unusable (empty, degenerate, or the learner
    rejected it). The message names the fold index, the counts that made it
    degenerate, and a remedy."""


@dataclass(frozen=True, eq=False)
class FoldAssignment:
    """Replayable unit-level fold assignment.

    ``eq=False``: ndarray fields make the auto-generated ``__eq__``/``__hash__``
    unusable; compare field-wise in tests. Holds only ndarrays/ints/str/state
    dict — picklable, and never holds learner objects.
    """

    n_folds: int
    n_units: int
    fold_ids: np.ndarray  # (n_units,) int64 in [0, n_folds)
    bitgen_state: Dict[str, Any]  # deep-copied BY VALUE before any draw
    bitgen_name: str  # bit-generator class name, for replay reconstruction
    stratify_labels: Optional[np.ndarray] = None
    cluster_ids: Optional[np.ndarray] = None
    # Construction-time snapshot of bitgen_state; replay() reads this so later
    # in-place mutation of the dict cannot change replayed fold ids.
    _state_blob: bytes = field(init=False, repr=False, default=b"")

    def __post_init__(self) -> None:
        # Enforce the invariants every consumer relies on (cross_fit_predict
        # writes oof slots per test fold, so an out-of-range/missing fold id
        # would leave np.empty memory exposed) and freeze the arrays so a
        # hand-mutated assignment cannot silently violate the replay contract.
        fold_ids = np.asarray(self.fold_ids)
        if fold_ids.ndim != 1 or fold_ids.shape[0] != self.n_units:
            raise ValueError(
                f"fold_ids must be 1-dimensional with n_units={self.n_units} "
                f"entries, got shape {fold_ids.shape}"
            )
        if not np.issubdtype(fold_ids.dtype, np.integer):
            raise ValueError(f"fold_ids must be an integer array, got dtype {fold_ids.dtype}")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if np.any(fold_ids < 0) or np.any(fold_ids >= self.n_folds):
            raise ValueError(
                f"fold_ids values must lie in [0, {self.n_folds}), got range "
                f"[{fold_ids.min()}, {fold_ids.max()}]"
            )
        counts = np.bincount(fold_ids, minlength=self.n_folds)
        if np.any(counts == 0):
            raise ValueError(
                f"fold(s) {np.flatnonzero(counts == 0).tolist()} own no units; "
                "every fold must be non-empty"
            )
        frozen = fold_ids.astype(np.int64, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "fold_ids", frozen)
        # Snapshot the bit-generator state at construction: replay() reads this
        # serialized copy, so mutating the (necessarily mutable) bitgen_state
        # dict afterwards cannot silently change replayed fold ids. The public
        # field itself is also detached from the caller's reference.
        object.__setattr__(self, "bitgen_state", copy.deepcopy(self.bitgen_state))
        object.__setattr__(self, "_state_blob", pickle.dumps(self.bitgen_state))
        for attr in ("stratify_labels", "cluster_ids"):
            val = getattr(self, attr)
            if val is not None:
                val = np.asarray(val).copy()
                if val.ndim != 1 or val.shape[0] != self.n_units:
                    raise ValueError(
                        f"{attr} must be 1-dimensional with n_units="
                        f"{self.n_units} entries, got shape {val.shape}"
                    )
                if np.any(pd.isna(val)):
                    raise ValueError(f"{attr} contains missing values (None/NaN/NA)")
                val.setflags(write=False)
                object.__setattr__(self, attr, val)
        if self.cluster_ids is not None:
            # Cluster cohesion: every cluster's units share one fold (the
            # invariant cluster-level cross-fitting relies on — a split
            # cluster leaks information between train and test).
            _, first_idx, inv = np.unique(self.cluster_ids, return_index=True, return_inverse=True)
            cluster_fold = frozen[first_idx]
            mismatch = frozen != cluster_fold[inv]
            if np.any(mismatch):
                bad_cluster = self.cluster_ids[np.flatnonzero(mismatch)[0]]
                raise ValueError(
                    f"cluster {bad_cluster!r} spans multiple folds; all units "
                    "of a cluster must share one fold"
                )
            if self.stratify_labels is not None:
                strat_first = self.stratify_labels[first_idx]
                if not np.all(self.stratify_labels == strat_first[inv]):
                    raise ValueError(
                        "stratify_labels vary within a cluster; pass cluster-level "
                        "stratum labels"
                    )

    def test_mask(self, k: int) -> np.ndarray:
        return self.fold_ids == k

    def train_mask(self, k: int) -> np.ndarray:
        return self.fold_ids != k

    def iter_folds(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(k, train_indices, test_indices)`` as int64 index arrays."""
        for k in range(self.n_folds):
            test = np.flatnonzero(self.fold_ids == k)
            train = np.flatnonzero(self.fold_ids != k)
            yield k, train, test

    def counts(self) -> np.ndarray:
        """MEMBER counts per fold: units unclustered, clusters clustered."""
        if self.cluster_ids is None:
            return np.bincount(self.fold_ids, minlength=self.n_folds)
        counts = np.zeros(self.n_folds, dtype=np.int64)
        _, first_idx = np.unique(self.cluster_ids, return_index=True)
        for i in first_idx:
            counts[self.fold_ids[i]] += 1
        return counts

    def replay(self) -> "FoldAssignment":
        """Rebuild the rng from the stored state and re-run the assignment.

        The result must be identical to this assignment (by-value replay
        discipline). Raises a targeted ``ValueError`` for an unknown
        bit-generator name rather than mis-seeding.
        """
        bitgen_cls = getattr(np.random, self.bitgen_name, None)
        if bitgen_cls is None:
            raise ValueError(
                f"FoldAssignment.replay: unknown bit generator {self.bitgen_name!r}; "
                "cannot reconstruct the generator from the stored state"
            )
        rng = np.random.Generator(bitgen_cls())
        rng.bit_generator.state = pickle.loads(self._state_blob)
        return assign_folds(
            self.n_units,
            self.n_folds,
            rng=rng,
            stratify=self.stratify_labels,
            cluster_ids=self.cluster_ids,
        )


@dataclass(frozen=True, eq=False)
class CrossFitResult:
    """Out-of-fold predictions + per-fold diagnostics (picklable only).

    ``fold_losses[k]`` is the out-of-fold loss on fold k — MSE for
    ``predict``, log-loss for ``predict_proba`` — computed as a
    test-fold-``sample_weight``-weighted mean when weights were supplied
    (unweighted otherwise). A test fold with zero total weight gets
    ``fold_losses[k] = NaN`` (documented sentinel — a diagnostic, never an
    error). Log-loss clips probabilities to ``[1e-15, 1 - 1e-15]`` for the
    LOSS ONLY; ``oof_predictions`` are the learner's outputs, unclipped.
    """

    oof_predictions: np.ndarray  # (n_units,)
    fold_losses: np.ndarray  # (n_folds,)
    n_fit_per_fold: np.ndarray  # (n_folds,) int64
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _validate_per_unit_array(
    arr: Optional[np.ndarray],
    n_units: int,
    name: str,
    *,
    allow_none_values: bool = False,
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim != 1:
        raise ValueError(
            f"{name} must be 1-dimensional with one entry per unit, got ndim={out.ndim} "
            "(a column vector like (n, 1) is not accepted)"
        )
    if out.shape[0] != n_units:
        raise ValueError(f"{name} has length {out.shape[0]}, expected n_units={n_units}")
    if not allow_none_values:
        # Dtype-independent missing-value detection (None, np.nan inside an
        # object array, pd.NA, NaT, ...) — a missing label reaching np.unique
        # can silently split one "missing" cluster into several members.
        missing = pd.isna(out)
        if np.any(missing):
            raise ValueError(
                f"{name} contains missing values (None/NaN/NA) at "
                f"{int(np.sum(missing))} position(s); labels must be complete"
            )
    return out


def assign_folds(
    n_units: int,
    n_folds: int,
    *,
    rng: np.random.Generator,
    stratify: Optional[np.ndarray] = None,
    cluster_ids: Optional[np.ndarray] = None,
) -> FoldAssignment:
    """Deterministic, replayable member-level K-fold assignment.

    Members are units (``cluster_ids is None``) or clusters (all units of a
    cluster share a fold; ``stratify`` must then be constant within each
    cluster). Within each stratum (sorted-unique order) members are permuted
    and dealt with a single global fold cursor that carries across strata, so
    member counts per fold differ by <= 1 globally and per stratum, and no
    fold is member-empty whenever ``n_folds <= n_members``. Under clustering,
    unit-level fold sizes are NOT balanced (clusters differ in size).
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__} "
            "(legacy RandomState is not supported — its state is not replayable "
            "through this module's by-value discipline)"
        )
    for name, val in (("n_units", n_units), ("n_folds", n_folds)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {val!r}")
    if n_units < 1:
        raise ValueError(f"n_units must be >= 1, got {n_units}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    stratify = _validate_per_unit_array(stratify, n_units, "stratify")
    cluster_ids = _validate_per_unit_array(cluster_ids, n_units, "cluster_ids")

    # Capture the state BY VALUE before any draw (replay discipline).
    bitgen_state = copy.deepcopy(rng.bit_generator.state)
    bitgen_name = type(rng.bit_generator).__name__

    if cluster_ids is None:
        member_of_unit = np.arange(n_units)
        n_members = n_units
        member_strata = stratify
    else:
        unique_clusters, first_idx, member_of_unit = _unique_or_raise(
            cluster_ids, "cluster_ids", return_index=True, return_inverse=True
        )
        n_members = unique_clusters.shape[0]
        if stratify is not None:
            # One label per cluster (its first unit), then one vectorized
            # consistency check over all units — O(n_units), not
            # O(n_clusters * n_units).
            member_strata = stratify[first_idx]
            consistent = stratify == member_strata[member_of_unit]
            if not np.all(consistent):
                bad = member_of_unit[np.flatnonzero(~consistent)[0]]
                raise ValueError(
                    f"stratify varies within cluster {unique_clusters[bad]!r}; "
                    "pass cluster-level stratum labels (one label per cluster, "
                    "broadcast to its units)"
                )
        else:
            member_strata = None

    if n_folds > n_members:
        kind = "units" if cluster_ids is None else "clusters"
        raise ValueError(
            f"n_folds={n_folds} exceeds the number of assignment members "
            f"({n_members} {kind}); reduce n_folds"
        )

    if member_strata is None:
        strata_values = [None]
        members_by_stratum = [np.arange(n_members)]
    else:
        strata_values = list(_unique_or_raise(member_strata, "stratify"))
        members_by_stratum = [np.flatnonzero(member_strata == s) for s in strata_values]
        for s, members in zip(strata_values, members_by_stratum):
            if members.shape[0] < 2:
                raise ValueError(
                    f"stratum {s!r} has only {members.shape[0]} member(s); every "
                    "stratum needs >= 2 members — a singleton stratum vanishes from "
                    "the complement of its own fold (merge strata or drop stratify)"
                )

    member_fold = np.empty(n_members, dtype=np.int64)
    cursor = 0  # single global fold cursor carrying across strata
    for members in members_by_stratum:
        permuted = members[rng.permutation(members.shape[0])]
        for m in permuted:
            member_fold[m] = cursor % n_folds
            cursor += 1

    fold_ids = member_fold[member_of_unit]

    # Defensive post-assignment check: every fold must own >= 1 member.
    member_counts = np.bincount(member_fold, minlength=n_folds)
    assert np.all(member_counts > 0), "internal error: empty fold after assignment"

    return FoldAssignment(
        n_folds=n_folds,
        n_units=n_units,
        fold_ids=fold_ids,
        bitgen_state=bitgen_state,
        bitgen_name=bitgen_name,
        stratify_labels=None if stratify is None else stratify.copy(),
        cluster_ids=None if cluster_ids is None else cluster_ids.copy(),
    )


@overload
def cross_fit_predict(
    learner: RegressorLearner,
    X: np.ndarray,
    y: np.ndarray,
    folds: FoldAssignment,
    *,
    fit_mask: Optional[np.ndarray] = ...,
    predict_method: Literal["predict"] = ...,
    sample_weight: Optional[np.ndarray] = ...,
    context_label: str = ...,
) -> CrossFitResult: ...


@overload
def cross_fit_predict(
    learner: ClassifierLearner,
    X: np.ndarray,
    y: np.ndarray,
    folds: FoldAssignment,
    *,
    fit_mask: Optional[np.ndarray] = ...,
    predict_method: Literal["predict_proba"],
    sample_weight: Optional[np.ndarray] = ...,
    context_label: str = ...,
) -> CrossFitResult: ...


def cross_fit_predict(
    learner: object,
    X: np.ndarray,
    y: np.ndarray,
    folds: FoldAssignment,
    *,
    fit_mask: Optional[np.ndarray] = None,
    predict_method: str = "predict",
    sample_weight: Optional[np.ndarray] = None,
    context_label: str = "",
) -> CrossFitResult:
    """Out-of-fold predictions for every unit.

    Per fold k: fit the learner on ``train_mask(k) & fit_mask`` (e.g. the
    untreated units), predict ALL units in fold k. ``context_label``
    (e.g. the nuisance name or (g,t) cell) is prefixed into every
    ``DegenerateFoldError`` message to identify WHICH cross-fit failed.
    """
    n_units = folds.n_units
    label = f"{context_label}: " if context_label else ""

    if predict_method not in ("predict", "predict_proba"):
        raise ValueError(
            f"predict_method must be 'predict' or 'predict_proba', got {predict_method!r}"
        )
    kind = "regressor" if predict_method == "predict" else "classifier"
    validate_learner(learner, kind=kind, param_name="learner")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != n_units:
        raise ValueError(
            f"X must be 2-dimensional with folds.n_units={n_units} rows, " f"got shape {X.shape}"
        )
    if y.ndim != 1 or y.shape[0] != n_units:
        raise ValueError(
            f"y must be 1-dimensional with folds.n_units={n_units} entries, " f"got shape {y.shape}"
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf values")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf values")

    if fit_mask is None:
        fit_mask_arr = np.ones(n_units, dtype=bool)
    else:
        raw_mask = np.asarray(fit_mask)
        if raw_mask.ndim != 1 or raw_mask.shape[0] != n_units:
            raise ValueError(
                f"fit_mask must be 1-dimensional with {n_units} entries, "
                f"got shape {raw_mask.shape}"
            )
        if raw_mask.dtype != np.bool_:
            raise ValueError(
                f"fit_mask must be a boolean array, got dtype {raw_mask.dtype} "
                "(an int/float mask would silently select the wrong rows)"
            )
        fit_mask_arr = raw_mask

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1:
            raise ValueError(
                f"sample_weight must be 1-dimensional, got ndim={sample_weight.ndim} "
                "(a column vector like (n, 1) is not accepted)"
            )
        if sample_weight.shape[0] != n_units:
            raise ValueError(
                f"sample_weight has length {sample_weight.shape[0]}, expected {n_units}"
            )
        if not np.isfinite(sample_weight).all():
            raise ValueError("sample_weight contains NaN or Inf values")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")

    if predict_method == "predict_proba" and not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError(
            "y must be strictly binary 0/1 for predict_method='predict_proba' "
            "(the logit solver silently saturates on other encodings)"
        )

    oof = np.empty(n_units, dtype=np.float64)
    fold_losses = np.empty(folds.n_folds, dtype=np.float64)
    n_fit_per_fold = np.empty(folds.n_folds, dtype=np.int64)

    for k, train_idx, test_idx in folds.iter_folds():
        fit_idx = train_idx[fit_mask_arr[train_idx]]
        n_fit = fit_idx.shape[0]
        n_fit_per_fold[k] = n_fit
        w_fit = None if sample_weight is None else sample_weight[fit_idx]

        # (a) Universal cheap pre-checks -> DegenerateFoldError directly.
        if n_fit == 0:
            raise DegenerateFoldError(
                f"{label}fold {k}: the fit subset is empty (train size "
                f"{train_idx.shape[0]}, fit_mask keeps 0). Reduce n_folds, widen "
                "fit_mask, or check the stratify labels."
            )
        if w_fit is not None and not np.any(w_fit > 0):
            raise DegenerateFoldError(
                f"{label}fold {k}: all {n_fit} fit rows have zero sample_weight. "
                "Reduce n_folds or check the weights."
            )
        if predict_method == "predict_proba":
            labels = y[fit_idx] if w_fit is None else y[fit_idx][w_fit > 0]
            if np.unique(labels).shape[0] < 2:
                raise DegenerateFoldError(
                    f"{label}fold {k}: the fit subset has a single "
                    f"{'positive-weight ' if w_fit is not None else ''}class "
                    f"(n_fit={n_fit}). A classifier needs both classes in every "
                    "fold's complement; reduce n_folds or stratify by the label."
                )

        # (b) Learner errors during the fold -> DegenerateFoldError, chained.
        try:
            fold_learner = _fresh_learner(learner)
            # Unweighted path calls fit(X, y) WITHOUT the keyword: the
            # advertised duck-typed contract is fit/predict(_proba), so a
            # learner whose fit signature is only (X, y) must work when no
            # weights are in play. sample_weight= is passed only on
            # genuinely weighted paths, where an unsupported signature
            # raises TypeError — a caller protocol violation that PROPAGATES
            # (the DegenerateFoldError wrapper below catches ValueError
            # only; fold-data degeneracy, not signature bugs).
            fit_kwargs = {} if w_fit is None else {"sample_weight": w_fit}
            if kind == "regressor":
                reg = cast(RegressorLearner, fold_learner)
                reg.fit(X[fit_idx], y[fit_idx], **fit_kwargs)
                raw_pred = reg.predict(X[test_idx])
            else:
                clf = cast(ClassifierLearner, fold_learner)
                clf.fit(X[fit_idx], y[fit_idx], **fit_kwargs)
                raw_pred = clf.predict_proba(X[test_idx])
            pred = _validate_predictions(
                raw_pred,
                test_idx.shape[0],
                kind=kind,
                context=f"{label}fold {k}",
                classes=(getattr(fold_learner, "classes_", None) if kind == "classifier" else None),
            )
        except DegenerateFoldError:
            raise
        except ValueError as exc:
            raise DegenerateFoldError(
                f"{label}learner error in fold {k}: {exc}; the fold's fit subset "
                f"has n={n_fit}"
                + (f", n_pos_weight={int(np.sum(w_fit > 0))}" if w_fit is not None else "")
                + ". Reduce n_folds, widen fit_mask, or check the fold's data."
            ) from exc

        oof[test_idx] = pred

        # Out-of-fold loss (diagnostic; NaN sentinel for zero-weight folds).
        w_test = None if sample_weight is None else sample_weight[test_idx]
        if predict_method == "predict":
            errs = (y[test_idx] - pred) ** 2
        else:
            p_clip = np.clip(pred, _LOG_LOSS_CLIP, 1.0 - _LOG_LOSS_CLIP)
            errs = -(y[test_idx] * np.log(p_clip) + (1.0 - y[test_idx]) * np.log(1.0 - p_clip))
        if w_test is None:
            fold_losses[k] = float(np.mean(errs))
        elif np.sum(w_test) > 0:
            fold_losses[k] = float(np.sum(w_test * errs) / np.sum(w_test))
        else:
            fold_losses[k] = np.nan

    return CrossFitResult(
        oof_predictions=oof,
        fold_losses=fold_losses,
        n_fit_per_fold=n_fit_per_fold,
        diagnostics={"predict_method": predict_method, "context_label": context_label},
    )
