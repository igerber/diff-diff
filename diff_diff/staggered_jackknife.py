"""Cluster jackknife (CV3) inference for CallawaySantAnna.

Implements the delete-one-cluster jackknife of Karim, Nielsen, MacKinnon &
Webb (2026), "Improved Inference for CSDID Using the Cluster Jackknife"
(arXiv:2602.12043), equation (20)::

    CV3 = ((H - 1) / H) * sum_h (theta^(h) - theta)^2

with ``H`` the number of clusters, ``theta`` the full-sample estimate and
``theta^(h)`` the estimate when cluster ``h`` is omitted. Every delete-one
estimate comes from a full re-estimation; no influence function is used.

The paper states the formula for the overall ATT. Applying it element-wise
to each ATT(g,t), event-study and group estimate, and its outer-product form
to the event-study covariance, is a library extension documented in
``docs/methodology/REGISTRY.md`` (CallawaySantAnna, "Cluster jackknife").
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import JackknifeReplicates
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
    fixed_cohort_agg_weights,
)
from diff_diff.utils import safe_inference, safe_inference_batch

if TYPE_CHECKING:
    from diff_diff.staggered import CallawaySantAnna

__all__: List[str] = []

#: The per-cell keys point aggregation reads. Copied key-for-key: a key the
#: fit did not set (``agg_weight`` on panel cells) must stay absent, because
#: the aggregators read ``data.get("agg_weight", data["n_treated"])``.
_CELL_KEYS: Tuple[str, ...] = ("effect", "n_treated", "agg_weight", "is_reference")

#: Refit warnings that concern only the refit's own analytical variance,
#: which the jackknife discards.
_VARIANCE_ONLY_WARNINGS: Tuple[str, ...] = ("PSU(s) found", "analytical SE paths")

#: Deletion-unit count above which ``fit()`` warns about the refit cost.
_MANY_REPLICATES = 500

#: A reported quantity resting on fewer treated clusters than this draws a
#: warning. Library choice, not from the paper: in the size simulation
#: (benchmarks/python/cs_jackknife_size.py) a quantity identified by two
#: treated clusters rejects 6% to 18% at a nominal 5%, against 3% to 10%
#: with four.
_MIN_TREATED_CLUSTERS = 3

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class DeletionUnits:
    """The clusters a jackknife deletes, resolved from the estimation sample.

    Attributes
    ----------
    column : str
        Column of the caller's frame the deletion is made on.
    values : tuple
        Values of ``column`` to delete, one per replicate.
    labels : tuple
        Row labels retained on the results (raw labels or position codes).
    unit_name : str
        Name shown for the deletion unit.
    labels_are_codes : bool
        Whether ``labels`` are position codes.
    roles : tuple of str
        ``"treated"`` / ``"control"`` per deletion unit.
    cell_treated : dict
        ``{(g, t): positions}``: the deletion units (by position) holding
        treated units of that cell.
    spare_values, spare_labels, spare_roles : tuple
        The clusters of the estimation index that the structural rule leaves
        out (it expects their omission to move nothing). They are refitted
        too, and any whose omission does move an estimate becomes a deletion
        unit, so the rule is checked on every fit instead of trusted.
    """

    column: str
    values: Tuple[Any, ...]
    labels: Tuple[Any, ...]
    unit_name: str
    labels_are_codes: bool
    roles: Tuple[str, ...]
    cell_treated: Dict[Any, Tuple[int, ...]]
    spare_values: Tuple[Any, ...] = ()
    spare_labels: Tuple[Any, ...] = ()
    spare_roles: Tuple[str, ...] = ()


def jackknife_se(full: float, replicates: Any) -> float:
    """Jackknife standard error, the square root of equation (20).

    Centered on the full-sample estimate with factor ``(H - 1) / H``. NaN
    when fewer than two replicates are given or any value is non-finite.
    """
    reps = np.asarray(replicates, dtype=float)
    h = reps.size
    if h < 2 or not np.isfinite(full) or not np.all(np.isfinite(reps)):
        return float(np.nan)
    return float(np.sqrt((h - 1) / h * np.sum((reps - full) ** 2)))


def jackknife_vcov(full: np.ndarray, replicates: np.ndarray) -> np.ndarray:
    """Outer-product form of equation (20) for a vector of estimates.

    ``replicates`` is ``(H, k)``; the result is ``(k, k)`` with the
    element-wise jackknife variances on its diagonal.
    """
    deviations = np.asarray(replicates, dtype=float) - np.asarray(full, dtype=float)
    h = deviations.shape[0]
    return (h - 1) / h * (deviations.T @ deviations)


def resolve_deletion_units(
    data: pd.DataFrame,
    unit: str,
    cluster: Optional[str],
    precomputed: Dict[str, Any],
    influence_func_info: Dict[Any, Any],
    group_time_effects: Dict[Any, Dict[str, Any]],
) -> DeletionUnits:
    """Resolve the deletion units from the estimation sample.

    A deletion unit is a cluster whose omission can move an estimate. It
    holds at least one unit (observation on the repeated-cross-section
    route) that either

    - enters some ATT(g,t) cell as treated or as comparison, or
    - belongs to a cohort whose WHOLE-cohort mass is an aggregation weight:
      with a cluster column (or any route that weights by fixed cohort
      masses) that is every cohort with a finite cell, and under a universal
      base period it is every cohort with a materialized reference cell,
      whose weight is the cohort mass. Such a unit can have no usable
      outcome at all and still shift the weights when it is deleted.

    Every other cluster label of the caller's frame - an always-treated
    cluster, a cluster made only of rows with a missing unit ID - is returned
    as a spare: ``run_replicates`` refits without it and keeps it out only if
    that changes nothing. So ``H`` counts exactly the clusters whose omission
    moves an estimate, and no label of the frame escapes the check.

    ``data`` is the caller's frame, row-aligned with the fit's working copy.
    Cluster labels are read from it and never from the working copy: the fit
    recodes the first-treatment and time columns there (``inf`` to 0, numeric
    coercion), and a cluster column may be one of them, so a label taken from
    the working copy could match no row of the frame the deletion is made on.
    """
    index_units = np.asarray(precomputed["all_units"])
    n_index = len(index_units)
    is_panel = bool(precomputed.get("is_panel", True))
    # Only entries of the index that have rows in the caller's frame can be
    # deleted. On the panel route the index comes from a groupby over the
    # unit column, which for a categorical column lists unused categories
    # too (the pandas default before 3.0): such an entry has no rows, enters
    # nothing, and is neither a deletion unit nor a candidate for the audit.
    if is_panel:
        observed = pd.Index(pd.unique(data[unit].dropna()))
        present = np.asarray(pd.Index(index_units).isin(observed), dtype=bool)
    else:
        present = np.ones(n_index, dtype=bool)

    contributing = np.zeros(n_index, dtype=bool)
    for info in influence_func_info.values():
        if not info:
            continue
        for key in ("treated_idx", "control_idx"):
            idx = np.asarray(info.get(key, ()), dtype=np.intp)
            if idx.size:
                contributing[idx] = True
    cohorts = np.asarray(precomputed["unit_cohorts"])
    with np.errstate(invalid="ignore"):
        ever_treated = np.asarray(cohorts > 0, dtype=bool)
    uses_cohort_masses = fixed_cohort_agg_weights(precomputed) is not None
    mass_cohorts = {
        gt[0]
        for gt, cell in group_time_effects.items()
        if np.isfinite(cell["effect"]) and (uses_cohort_masses or cell.get("is_reference"))
    }
    if mass_cohorts:
        contributing |= np.isin(cohorts, list(mass_cohorts))
    contributing &= present

    def treated_by_cell(position_of_index: np.ndarray) -> Dict[Any, Tuple[int, ...]]:
        """Deletion-unit positions of each cell's treated units."""
        out: Dict[Any, Tuple[int, ...]] = {}
        for gt, info in influence_func_info.items():
            idx = np.asarray((info or {}).get("treated_idx", ()), dtype=np.intp)
            positions = np.unique(position_of_index[idx])
            out[gt] = tuple(int(p) for p in positions if p >= 0)
        return out

    if cluster is None:
        positions = np.flatnonzero(contributing)
        spare = np.flatnonzero(~contributing & present)
        position_of_index = np.full(n_index, -1, dtype=np.intp)
        position_of_index[positions] = np.arange(positions.size)
        # Labels are taken as pandas Index elements, never ``ndarray.tolist()``:
        # that turns datetime64 unit IDs into integers, which match no row.
        unit_index = pd.Index(index_units)
        return DeletionUnits(
            column=unit,
            values=tuple(unit_index[positions]),
            labels=tuple(int(p) for p in positions),
            unit_name="unit",
            labels_are_codes=True,
            roles=tuple("treated" if ever_treated[p] else "control" for p in positions),
            cell_treated=treated_by_cell(position_of_index),
            spare_values=tuple(unit_index[spare]),
            spare_labels=tuple(int(p) for p in spare),
            spare_roles=tuple("treated" if ever_treated[p] else "control" for p in spare),
        )

    if is_panel:
        per_index = (
            data.groupby(unit, sort=True, observed=True)[cluster]
            .first()
            .reindex(index_units)
            .to_numpy()
        )
    else:
        # Repeated cross-sections: the index space is the observation row.
        per_index = data[cluster].to_numpy()
    present_idx = np.flatnonzero(present)
    present_codes, uniques = pd.factorize(per_index[present_idx], sort=False)
    # ``codes`` is -1 for an index entry with no rows; every use below goes
    # through ``present``.
    codes = np.full(n_index, -1, dtype=np.intp)
    codes[present_idx] = present_codes
    n_clusters = len(uniques)
    treated_count = np.bincount(
        present_codes, weights=ever_treated[present_idx].astype(float), minlength=n_clusters
    )
    order = pd.unique(codes[contributing])
    # A cluster column that is one-to-one with the units present IS the unit
    # (or a relabelling of it), so it gets position codes: no raw unit
    # identifier is retained on the results object.
    one_to_one = n_clusters == present_idx.size
    first_member = np.full(n_clusters, -1, dtype=np.intp)
    if one_to_one:
        first_member[present_codes] = present_idx
    position_of_code = np.full(n_clusters, -1, dtype=np.intp)
    position_of_code[order] = np.arange(len(order))
    position_of_index = np.full(n_index, -1, dtype=np.intp)
    position_of_index[present_idx] = position_of_code[present_codes]
    spare = [c for c in range(n_clusters) if position_of_code[c] < 0]
    # Labels of the caller's frame that hold no unit of the estimation index
    # (clusters made only of rows with a missing unit ID). The fit does not
    # estimate on those rows, but they are not always inert - a row can add a
    # period to the calendar grid and so move a base period - so they are
    # audited like every other excluded cluster, never assumed away.
    # Membership and labels go through pandas Index objects, so typed labels
    # (datetime, period, categorical) compare and round-trip as themselves.
    label_index = pd.Index(uniques)
    raw_labels = pd.Index(pd.unique(data[cluster]))
    outside = list(raw_labels[~raw_labels.isin(label_index)])

    def labels_of(cluster_codes: Sequence[int]) -> Tuple[Any, ...]:
        if one_to_one:
            return tuple(int(first_member[c]) for c in cluster_codes)
        return tuple(label_index[c] for c in cluster_codes)

    def roles_of(cluster_codes: Sequence[int]) -> Tuple[str, ...]:
        return tuple("treated" if treated_count[c] > 0 else "control" for c in cluster_codes)

    return DeletionUnits(
        column=cluster,
        values=tuple(label_index[c] for c in order),
        labels=labels_of(order),
        unit_name="unit" if one_to_one else cluster,
        labels_are_codes=one_to_one,
        roles=roles_of(order),
        cell_treated=treated_by_cell(position_of_index),
        spare_values=tuple(label_index[c] for c in spare) + tuple(outside),
        # In code mode a cluster outside the index has no unit position; it
        # gets a code past the index (n_index, n_index + 1, ...).
        spare_labels=labels_of(spare)
        + (tuple(n_index + k for k in range(len(outside))) if one_to_one else tuple(outside)),
        spare_roles=roles_of(spare) + ("control",) * len(outside),
    )


def _finite_cells(cells: Dict[Any, Dict[str, Any]]) -> frozenset:
    """Keys of the cells with a finite effect (a cell can keep its key with NaN)."""
    return frozenset(gt for gt, cell in cells.items() if np.isfinite(cell["effect"]))


def _normalize(message: str) -> str:
    """A warning message with its numbers masked, for comparing across refits."""
    return _NUMBER.sub("#", message)


def _close(a: float, b: float) -> bool:
    """Equal up to rounding, NaN matching NaN."""
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return bool(np.isclose(a, b, rtol=1e-10, atol=1e-12))


def _moves_an_estimate(
    full_att: float,
    full_cells: Dict[Any, Dict[str, Any]],
    full_masses: Optional[Dict[Any, float]],
    rep_att: float,
    rep_cells: Dict[Any, Dict[str, Any]],
    rep_masses: Optional[Dict[Any, float]],
) -> bool:
    """Whether a delete-one refit differs from the full sample in any point input.

    Compares the overall ATT, the estimate and aggregation weight of every
    cell with a finite estimate, and the cohort masses of the cohorts that
    have such a cell. Cells without a finite estimate and masses no cell
    uses (an always-treated cohort has both) enter no aggregate, so their
    disappearance moves nothing.
    """
    full_finite = _finite_cells(full_cells)
    if not _close(full_att, rep_att) or full_finite != _finite_cells(rep_cells):
        return True
    for gt in full_finite:
        cell = full_cells[gt]
        other = rep_cells[gt]
        for key in ("effect", "n_treated", "agg_weight"):
            if (key in cell) != (key in other):
                return True
            if key in cell and not _close(float(cell[key]), float(other[key])):
                return True
    live = {gt[0] for gt in full_finite}
    full_live = {g: m for g, m in (full_masses or {}).items() if g in live}
    rep_live = {g: m for g, m in (rep_masses or {}).items() if g in live}
    if full_live.keys() != rep_live.keys():
        return True
    return any(not _close(float(full_live[g]), float(rep_live[g])) for g in full_live)


def run_replicates(
    estimator: "CallawaySantAnna",
    data: pd.DataFrame,
    fit_kwargs: Dict[str, Any],
    units: DeletionUnits,
    full_att: float,
    full_cells: Dict[Any, Dict[str, Any]],
    full_masses: Optional[Dict[Any, float]],
) -> Tuple[JackknifeReplicates, List[str], int]:
    """Re-estimate on every delete-one sample.

    Each refit is a fresh estimator with the fitted estimator's parameters
    and ``vcov_type="hc1"``; every other parameter, ``cluster`` included, is
    unchanged, so a refit takes the same routing and the same aggregation
    weights as the full-sample fit. Returns the replicate payload, the
    refit warnings that the full-sample configuration does not raise itself,
    and the number of delete-one refits actually run (which exceeds ``H``
    when spare clusters were audited and dropped).

    The clusters the structural rule left out (``units.spare_*``) are
    refitted as well. One whose omission moves any point input is a deletion
    unit after all and joins the replicates; one whose omission moves
    nothing is dropped, as the rule intended.
    """
    params = {**estimator.get_params(), "vcov_type": "hc1"}
    column = data[units.column]
    full_finite = _finite_cells(full_cells)
    full_point_cells = {
        gt: {k: cell[k] for k in _CELL_KEYS if k in cell} for gt, cell in full_cells.items()
    }

    # Warnings the same configuration raises on the full sample are not news.
    with warnings.catch_warnings(record=True) as baseline_caught:
        warnings.simplefilter("always")
        type(estimator)(**params).fit(data, **fit_kwargs)
    baseline = {_normalize(str(w.message)) for w in baseline_caught}

    labels: List[Any] = []
    roles: List[str] = []
    atts: List[float] = []
    cells: List[Dict[Any, Dict[str, Any]]] = []
    masses: List[Optional[Dict[Any, float]]] = []
    n_cells: List[int] = []
    changed: List[bool] = []
    undefined: List[Optional[str]] = []
    extra: Dict[str, Tuple[str, int]] = {}

    candidates = [
        (value, label, role, False)
        for value, label, role in zip(units.values, units.labels, units.roles)
    ] + [
        (value, label, role, True)
        for value, label, role in zip(units.spare_values, units.spare_labels, units.spare_roles)
    ]
    for value, label, role, is_spare in candidates:
        subset = data.loc[~column.isin([value])]
        if len(subset) == len(data):
            # Every candidate was resolved from rows of this frame, so a
            # deletion that removes none means the two went out of step.
            # Failing here is the only alternative to a silent no-op replicate.
            raise RuntimeError(
                f"Cluster jackknife: deleting {units.unit_name} {value!r} removed no "
                "rows. This is a bug in diff-diff; please report it."
            )
        refit = type(estimator)(**params)
        reason: Optional[str] = None
        rep_results = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                rep_results = refit.fit(subset, **fit_kwargs)
            except (ValueError, np.linalg.LinAlgError) as exc:
                reason = str(exc)

        rep_cells: Dict[Any, Dict[str, Any]] = {}
        rep_masses: Optional[Dict[Any, float]] = None
        rep_att = float(np.nan)
        if rep_results is not None:
            rep_att = float(rep_results.att)
            rep_cells = {
                gt: {k: cell[k] for k in _CELL_KEYS if k in cell}
                for gt, cell in rep_results.group_time_effects.items()
            }
            kit = getattr(rep_results, "_aggregation_kit", None)
            if kit is not None:
                rep_masses = fixed_cohort_agg_weights(kit.bookkeeping)
            if not np.isfinite(rep_att):
                reason = "the overall ATT is not estimable"
        if is_spare and not _moves_an_estimate(
            full_att, full_point_cells, full_masses, rep_att, rep_cells, rep_masses
        ):
            continue

        for w in caught:
            message = str(w.message)
            if any(marker in message for marker in _VARIANCE_ONLY_WARNINGS):
                continue
            key = _normalize(message)
            if key in baseline:
                continue
            extra[key] = (message, extra.get(key, (message, 0))[1] + 1)
        rep_finite = _finite_cells(rep_cells)
        labels.append(label)
        roles.append(role)
        atts.append(rep_att)
        cells.append(rep_cells)
        masses.append(rep_masses)
        n_cells.append(len(rep_finite))
        changed.append(rep_finite != full_finite)
        undefined.append(reason)

    payload = JackknifeReplicates(
        labels=tuple(labels),
        unit_name=units.unit_name,
        labels_are_codes=units.labels_are_codes,
        roles=tuple(roles),
        att=tuple(atts),
        cells=tuple(cells),
        cohort_masses=tuple(masses),
        n_cells=tuple(n_cells),
        changed=tuple(changed),
        undefined=tuple(undefined),
        cell_treated=dict(units.cell_treated),
    )
    extra_messages = [
        f"{message} ({count} refit{'s' if count != 1 else ''})" for message, count in extra.values()
    ]
    return payload, extra_messages, len(candidates)


def _any_undefined(payload: JackknifeReplicates) -> bool:
    return any(reason is not None for reason in payload.undefined)


def _describe(labels: Sequence[Any], limit: int = 5) -> str:
    shown = ", ".join(
        repr(label.item() if isinstance(label, np.generic) else label) for label in labels[:limit]
    )
    if len(labels) > limit:
        shown += f" and {len(labels) - limit} more"
    return shown


def changed_cells_message(payload: JackknifeReplicates, what: str) -> Optional[str]:
    """The paper's second caveat, when some deletion changes the cell set."""
    # An undefined replicate has its own warning (and makes every SE NaN).
    labels = [
        label
        for label, flag, reason in zip(payload.labels, payload.changed, payload.undefined)
        if flag and reason is None
    ]
    if not labels:
        return None
    coded = " (position codes)" if payload.labels_are_codes else ""
    return (
        f"Cluster jackknife: deleting {len(labels)} of {payload.n_replicates} "
        f"{payload.unit_name} deletion unit(s){coded} ({_describe(labels)}) changes "
        f"which ATT(g,t) cells are estimable, so those delete-one estimates of "
        f"{what} average different cohorts than the full sample (Karim, Nielsen, "
        "MacKinnon & Webb 2026, Section 4). A cell or aggregate that is not "
        "estimable in every delete-one sample has a NaN standard error. Inspect "
        "results.get_loo_effects_df() before relying on this inference."
    )


def n_treated_clusters(payload: JackknifeReplicates, cells: Sequence[Any]) -> int:
    """Number of deletion units holding treated units of any of ``cells``."""
    behind: set = set()
    for gt in cells:
        behind.update(payload.cell_treated.get(gt, ()))
    return len(behind)


def few_treated_message(what: str) -> str:
    """The warning for quantities resting on too few treated clusters."""
    return (
        f"Cluster jackknife: {what} on fewer than {_MIN_TREATED_CLUSTERS} treated "
        "clusters. The jackknife cannot fully correct for so few: in the library's "
        "size simulation a quantity identified by two treated clusters rejected a "
        "true null 6% to 18% of the time at a nominal 5% (3% to 10% with four). "
        "Treat this inference as indicative; see docs/methodology/REGISTRY.md, "
        "CallawaySantAnna, 'Cluster jackknife'."
    )


def _few_treated_fit_message(
    payload: JackknifeReplicates,
    group_time_effects: Dict[Any, Dict[str, Any]],
    anticipation: int,
) -> Optional[str]:
    """Fit-time form: the overall ATT and the ATT(g,t) cells."""
    live = [
        gt
        for gt, cell in group_time_effects.items()
        if not cell.get("is_reference") and np.isfinite(cell["effect"])
    ]
    post = [gt for gt in live if gt[1] >= gt[0] - anticipation]
    parts: List[str] = []
    n_overall = n_treated_clusters(payload, post)
    if post and n_overall < _MIN_TREATED_CLUSTERS:
        parts.append(f"the overall ATT rests on {n_overall}")
    few = [gt for gt in live if n_treated_clusters(payload, [gt]) < _MIN_TREATED_CLUSTERS]
    if few:
        parts.append(f"{len(few)} of {len(live)} ATT(g,t) cells rest")
    if not parts:
        return None
    return few_treated_message(" and ".join(parts))


def fit_messages(
    payload: JackknifeReplicates,
    extra: Sequence[str],
    group_time_effects: Dict[Any, Dict[str, Any]],
    anticipation: int,
    n_refits: int,
) -> List[str]:
    """The consolidated warnings a jackknife fit emits."""
    messages: List[str] = []
    bad = [
        (label, reason)
        for label, reason in zip(payload.labels, payload.undefined)
        if reason is not None
    ]
    if bad:
        coded = " (position codes)" if payload.labels_are_codes else ""
        messages.append(
            f"Cluster jackknife: deleting {payload.unit_name} deletion unit(s){coded} "
            f"{_describe([label for label, _ in bad])} leaves no estimate "
            f"({bad[0][1]}). The jackknife variance is undefined for this design: "
            "every jackknife standard error is NaN."
        )
    changed = changed_cells_message(
        payload, "the overall ATT and of the event-study and group aggregates"
    )
    if changed is not None:
        messages.append(changed)
    if extra:
        messages.append(
            "Cluster jackknife: delete-one refits raised warnings the full-sample "
            "configuration does not: "
            + "; ".join(extra[:3])
            + (f"; and {len(extra) - 3} more" if len(extra) > 3 else "")
        )
    few_treated = _few_treated_fit_message(payload, group_time_effects, anticipation)
    if few_treated is not None:
        messages.append(few_treated)
    if n_refits > _MANY_REPLICATES:
        # Cost is counted in refits run (audited spare clusters included, plus
        # the baseline estimation), not in the H the variance is built on.
        messages.append(
            f"Cluster jackknife: {payload.n_replicates} deletion units, and fit() ran "
            f"{n_refits + 1} extra estimations. The jackknife is meant for few "
            "clusters; with this many the default inference is usually adequate."
        )
    return messages


def apply_to_fit(
    payload: JackknifeReplicates,
    overall_att: float,
    group_time_effects: Dict[Any, Dict[str, Any]],
    alpha: float,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Replace the fit's analytical inference with jackknife inference.

    Writes the cell-level ``se`` / ``t_stat`` / ``p_value`` / ``conf_int``
    in place and returns the overall quartet. Reference cells are left as
    the fit produced them.
    """
    df = payload.n_replicates - 1
    undefined = _any_undefined(payload)
    overall_se = float(np.nan) if undefined else jackknife_se(overall_att, payload.att)
    overall_t, overall_p, overall_ci = safe_inference(overall_att, overall_se, alpha=alpha, df=df)

    keys = [
        gt
        for gt, cell in group_time_effects.items()
        if not cell.get("is_reference") and np.isfinite(cell["effect"])
    ]
    if keys:
        effects = np.array([float(group_time_effects[gt]["effect"]) for gt in keys])
        ses = np.full(len(keys), np.nan)
        if not undefined:
            for i, gt in enumerate(keys):
                reps = [
                    float(cells[gt]["effect"]) if gt in cells else np.nan for cells in payload.cells
                ]
                ses[i] = jackknife_se(effects[i], reps)
        t_stats, p_values, ci_lo, ci_hi = safe_inference_batch(effects, ses, alpha=alpha, df=df)
        for i, gt in enumerate(keys):
            cell = group_time_effects[gt]
            cell["se"] = float(ses[i])
            cell["t_stat"] = float(t_stats[i])
            cell["p_value"] = float(p_values[i])
            cell["conf_int"] = (float(ci_lo[i]), float(ci_hi[i]))
    return overall_se, overall_t, overall_p, overall_ci


class _PointOnlyAggregator(CallawaySantAnnaAggregationMixin):
    """The aggregation mixin with its variance work switched off.

    Replicate aggregates are needed as point estimates only, and replicates
    retain no influence functions, so both variance entry points return NaN
    without reading any.
    """

    def __init__(self, alpha: float, anticipation: int) -> None:
        self.alpha = alpha
        self.anticipation = anticipation

    def _compute_aggregated_se_with_wif(
        self,
        gt_pairs: Any,
        weights: Any,
        effects: Any,
        groups_for_gt: Any,
        influence_func_info: Any,
        df: Any,
        unit: Any,
        precomputed: Any = None,
        return_psi: bool = False,
    ) -> Any:
        if return_psi:
            return np.nan, np.empty(0), None
        return np.nan, None

    def _se_from_psi(self, psi_total: Any, precomputed: Any = None) -> Any:
        return np.nan, None


def _replicate_precomputed(masses: Optional[Dict[Any, float]]) -> Dict[str, Any]:
    """The bookkeeping a replicate's point aggregation reads: its own masses."""
    return {"agg_cohort_masses": masses} if masses is not None else {}


def _override_rows(
    rows: Dict[Any, Dict[str, Any]],
    keys: List[Any],
    replicate_values: np.ndarray,
    alpha: float,
    df: int,
) -> np.ndarray:
    """Write jackknife inference onto ``rows[key]`` for ``keys``; return the SEs."""
    effects = np.array([float(rows[k]["effect"]) for k in keys])
    ses = np.array(
        [jackknife_se(effects[j], replicate_values[:, j]) for j in range(len(keys))], dtype=float
    )
    t_stats, p_values, ci_lo, ci_hi = safe_inference_batch(effects, ses, alpha=alpha, df=df)
    for j, key in enumerate(keys):
        row = rows[key]
        row["se"] = float(ses[j])
        row["t_stat"] = float(t_stats[j])
        row["p_value"] = float(p_values[j])
        row["conf_int"] = (float(ci_lo[j]), float(ci_hi[j]))
        row.pop("cband_conf_int", None)
    return ses


def few_treated_event_study_message(
    payload: JackknifeReplicates,
    full_cells: Dict[Any, Dict[str, Any]],
    effects: Dict[Any, Dict[str, Any]],
    balance_e: Optional[int],
) -> Optional[str]:
    """Event-study horizons resting on too few treated clusters, if any."""
    retained = None
    if balance_e is not None:
        # The cohorts a balanced aggregation keeps (same rule as the aggregator).
        retained = {
            g
            for (g, t), cell in full_cells.items()
            if t - g == balance_e and np.isfinite(cell["effect"])
        }
    few = []
    for e, row in effects.items():
        if row.get("is_reference") or not np.isfinite(row["effect"]):
            continue
        cells = [
            gt
            for gt, cell in full_cells.items()
            if gt[1] - gt[0] == e
            and not cell.get("is_reference")
            and np.isfinite(cell["effect"])
            and (retained is None or gt[0] in retained)
        ]
        if n_treated_clusters(payload, cells) < _MIN_TREATED_CLUSTERS:
            few.append(e)
    if not few:
        return None
    return few_treated_message(
        f"the event-study estimates at event time(s) {_describe(sorted(few), limit=12)} rest"
    )


def few_treated_group_message(
    payload: JackknifeReplicates,
    full_cells: Dict[Any, Dict[str, Any]],
    effects: Dict[Any, Dict[str, Any]],
    anticipation: int,
) -> Optional[str]:
    """Cohorts whose group estimate rests on too few treated clusters, if any."""
    few = []
    for g, row in effects.items():
        if not np.isfinite(row["effect"]):
            continue
        cells = [
            gt
            for gt, cell in full_cells.items()
            if gt[0] == g and gt[1] >= g - anticipation and np.isfinite(cell["effect"])
        ]
        if n_treated_clusters(payload, cells) < _MIN_TREATED_CLUSTERS:
            few.append(g)
    if not few:
        return None
    return few_treated_message(
        f"the group estimates of cohort(s) {_describe(sorted(few), limit=12)} rest"
    )


def apply_to_event_study(
    payload: JackknifeReplicates,
    effects: Dict[Any, Dict[str, Any]],
    *,
    alpha: float,
    anticipation: int,
    groups: List[Any],
    time_periods: List[Any],
    balance_e: Optional[int],
) -> Tuple[Optional[np.ndarray], Optional[List[Any]]]:
    """Jackknife inference for an event-study aggregation.

    Overrides the non-reference rows of ``effects`` in place and returns the
    jackknife covariance with its index: the horizons that are finite in
    every delete-one sample, in row order.
    """
    df = payload.n_replicates - 1
    keys = [
        e
        for e, row in effects.items()
        if not row.get("is_reference") and np.isfinite(row["effect"])
    ]
    if not keys:
        return None, None
    values = np.full((payload.n_replicates, len(keys)), np.nan)
    if not _any_undefined(payload):
        agg = _PointOnlyAggregator(alpha, anticipation)
        for h in range(payload.n_replicates):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = agg._aggregate_event_study(
                    payload.cells[h],
                    {},
                    groups,
                    time_periods,
                    balance_e,
                    None,
                    None,
                    _replicate_precomputed(payload.cohort_masses[h]),
                ).effects
            for j, e in enumerate(keys):
                # ``keys`` are horizons estimated in the full sample. A
                # delete-one sample can leave such a horizon with reference
                # cells only (a universal base on a gapped grid); its
                # aggregate is then the finite reference value 0, which is
                # that sample's estimate and enters like any other.
                row = rep.get(e)
                if row is not None:
                    values[h, j] = float(row["effect"])
    ses = _override_rows(effects, keys, values, alpha, df)
    ok = np.isfinite(ses)
    if not ok.any():
        return None, None
    full = np.array([float(effects[e]["effect"]) for e in keys])
    vcov = jackknife_vcov(full[ok], values[:, ok])
    return vcov, [e for e, keep in zip(keys, ok) if keep]


def apply_to_group(
    payload: JackknifeReplicates,
    effects: Dict[Any, Dict[str, Any]],
    *,
    alpha: float,
    anticipation: int,
    groups: List[Any],
) -> None:
    """Jackknife inference for a group aggregation, written in place."""
    df = payload.n_replicates - 1
    keys = [g for g, row in effects.items() if np.isfinite(row["effect"])]
    if keys:
        values = np.full((payload.n_replicates, len(keys)), np.nan)
        if not _any_undefined(payload):
            agg = _PointOnlyAggregator(alpha, anticipation)
            for h in range(payload.n_replicates):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rep = agg._aggregate_by_group(
                        payload.cells[h],
                        {},
                        groups,
                        precomputed=_replicate_precomputed(payload.cohort_masses[h]),
                    )
                for j, g in enumerate(keys):
                    if g in rep:
                        values[h, j] = float(rep[g]["effect"])
        _override_rows(effects, keys, values, alpha, df)
    # The df this table's inference used, for every row (the analytical
    # aggregator records only a survey df here).
    for row in effects.values():
        row["df_used"] = float(df)


def loo_effects_frame(payload: JackknifeReplicates, full_att: float) -> pd.DataFrame:
    """The delete-one overall ATTs as a table, most influential first."""
    att_loo = np.asarray(payload.att, dtype=float)
    frame = pd.DataFrame(
        {
            "cluster": list(payload.labels),
            "role": list(payload.roles),
            "att_loo": att_loo,
            "delta_from_full": att_loo - full_att,
            "n_cells": list(payload.n_cells),
        }
    )
    # Same ordering as SyntheticDiDResults.get_loo_effects_df(): |delta|
    # descending, undefined replicates last.
    frame["_abs_delta"] = frame["delta_from_full"].abs()
    frame = frame.sort_values(by="_abs_delta", ascending=False, na_position="last").drop(
        columns="_abs_delta"
    )
    return frame.reset_index(drop=True)
