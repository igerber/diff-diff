"""Changes-in-Changes (CiC) and Quantile Difference-in-Differences (QDiD) estimators.

Implements the nonlinear difference-in-differences estimators of Athey & Imbens
(2006), "Identification and Inference in Nonlinear Difference-in-Differences
Models", Econometrica 74(2), 431-497, for the canonical 2x2 design (two groups,
two periods) with continuous outcomes:

- :class:`ChangesInChanges` (alias ``CiC``): the changes-in-changes estimator.
  The counterfactual second-period outcome distribution of the treated group is
  ``F_{Y^N,11}(y) = F_10(F_00^{-1}(F_01(y)))`` (Theorem 3.1, eq. 9); the ATT is
  the plug-in eq. (36) and quantile treatment effects follow eqs. (17)-(18).
- :class:`QDiD`: the quantile DiD comparison estimator (Section 3.3). The
  authors recommend CiC over QDiD (p. 447): QDiD's justifying model is not
  invariant to monotone rescaling of the outcome and places testable
  restrictions on the data.

Numerical conventions match the R ``qte`` package (v1.3.1, Callaway), the
project's parity target, exactly:

- CiC uses R type-1 quantiles (the paper's eq. (35)/(A.1) inf-based
  ceiling-order-statistic inverse) throughout.
- QDiD uses the additive quantile-DiD form with R type-7 (linear-interpolation)
  quantiles, matching ``qte::QDiD()``. This is population-equivalent to the
  paper's ``k^QDID`` transformation but a different finite-sample estimator
  (see the labeled Note in docs/methodology/REGISTRY.md).
- Inference is bootstrap-only (the paper's analytical influence-function
  variance is deferred): panel mode resamples units (both periods travel
  together); repeated cross-section mode draws a single pooled row resample.
  SEs are SDs over replicates with symmetric normal-approximation intervals.

Scope (v1, per docs/methodology/papers/athey-imbens-2006-review.md): 2x2
design, continuous outcomes, no covariates. Deferred and documented in the
methodology registry: covariates (Melly-Santangelo 2015), discrete-outcome
bounds (Section 4), analytical standard errors (Theorems 5.1-5.7), multiple
groups/periods (Section 6), and treatment-on-the-controls (Theorem 3.2).
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.bootstrap_utils import warn_bootstrap_failure_rate
from diff_diff.changes_in_changes_results import ChangesInChangesResults
from diff_diff.utils import safe_inference, safe_inference_batch, validate_binary

# Default quantile grid: qte's ``probs = seq(0.05, 0.95, 0.05)`` (19 points), pinned to
# R's EXACT seq() doubles. ``np.arange(0.05, 0.96, 0.05)`` differs from R at 5 of the 19
# indices by one ulp, and type-1 order-statistic selection is sensitive to those ulps on
# n*p integer boundaries - so the default is hardcoded rather than computed. Locked
# against the golden fixture's stored probs by tests/test_changes_in_changes_parity.py.
_DEFAULT_QUANTILES = np.array(
    [
        0.05,
        0.1,
        0.15000000000000002,
        0.2,
        0.25,
        0.3,
        0.35000000000000003,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6000000000000001,
        0.6500000000000001,
        0.7000000000000001,
        0.7500000000000001,
        0.8,
        0.8500000000000001,
        0.9000000000000001,
        0.95,
    ]
)

# Duplicate-share threshold above which the discrete-outcome warning fires. Library
# choice: the paper's continuous machinery (Assumption 5.1(iii)) has no finite-sample
# ties rule, and applying it to discrete data silently returns one endpoint of the
# Section 4 bounds rather than a point estimate.
_TIE_SHARE_WARN = 0.10

# Minimum share of finite bootstrap replicate rows required to report SEs
# (bootstrap_utils convention).
_MIN_VALID_REPLICATE_SHARE = 0.5

_CELL_LABELS = {
    "y00": "control pre-period (treatment=0, time=0)",
    "y01": "control post-period (treatment=0, time=1)",
    "y10": "treated pre-period (treatment=1, time=0)",
    "y11": "treated post-period (treatment=1, time=1)",
}


# =============================================================================
# Numeric core
# =============================================================================


def _build_cells(y: np.ndarray, g: np.ndarray, t: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Split outcomes into the four sorted (group, period) cells.

    Returns ``None`` if any cell is empty (bootstrap replicates use this to
    signal a failed draw; ``fit`` raises instead via :func:`_split_cells`).
    """
    cells = {}
    for key, (gv, tv) in {"y00": (0, 0), "y01": (0, 1), "y10": (1, 0), "y11": (1, 1)}.items():
        cell = y[(g == gv) & (t == tv)]
        if cell.size == 0:
            return None
        cells[key] = np.sort(cell)
    return cells


def _split_cells(y: np.ndarray, g: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Like :func:`_build_cells` but raises on empty cells (Assumption 5.1(ii))."""
    for key, (gv, tv) in {"y00": (0, 0), "y01": (0, 1), "y10": (1, 0), "y11": (1, 1)}.items():
        if not np.any((g == gv) & (t == tv)):
            raise ValueError(
                f"Empty (group, period) cell: no observations in the {_CELL_LABELS[key]} cell. "
                "All four 2x2 cells must be non-empty (Athey-Imbens Assumption 5.1(ii))."
            )
    cells = _build_cells(y, g, t)
    assert cells is not None
    return cells


def _ecdf(sorted_sample: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical CDF of ``sorted_sample`` evaluated at ``x`` (eq. 34, ``<=`` semantics).

    Values below the sample minimum map to 0.0; at or above the maximum to 1.0.
    """
    n = sorted_sample.shape[0]
    return np.searchsorted(sorted_sample, x, side="right") / n


def _quantile_type1(sorted_sample: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``quantile(x, probs, type=1)`` - exact port including R's fuzz arithmetic.

    This is the paper's eq. (35)/(A.1) inf-based inverse: the ceiling order
    statistic ``x_(ceil(n*p))`` with ``F^{-1}(0)`` = sample minimum. R computes
    ``j = floor(n*p * (1 + fuzz))`` with ``fuzz = 4 * .Machine$double.eps`` and
    ``h = (n*p > j)``, returning the 1-based order statistic ``x[j + h]``. The
    fuzz matters because ECDF-composed probabilities like ``k/n00`` can land one
    ulp above an integer after multiplication by ``n01``; a naive ceil picks a
    different order statistic than R there. Do not replace this with
    ``np.quantile(method="inverted_cdf")`` - its arithmetic differs from R's.
    """
    n = sorted_sample.shape[0]
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    fuzz = 4.0 * np.finfo(float).eps
    nppm = n * p
    j = np.floor(nppm * (1.0 + fuzz)).astype(np.int64)
    h = (nppm > j).astype(np.int64)
    idx = np.clip(j + h, 1, n)
    return sorted_sample[idx - 1]


def _quantile_type7(sorted_sample: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``quantile(x, probs)`` default type-7 == numpy's default ``linear`` method."""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    return np.quantile(sorted_sample, p, method="linear")


def _cic_point(
    cells: Dict[str, np.ndarray], quantiles: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """CiC ATT (eq. 36) and quantile effects (eq. 18), qte::CiC() arithmetic.

    Counterfactual draws: each treated pre-period outcome is ranked in the
    control pre-period distribution and pushed through the type-1 quantile of
    the control post-period distribution (``F_01^{-1}(F_00(y))``, eq. 15/36).
    """
    ranks = _ecdf(cells["y00"], cells["y10"])
    cf = _quantile_type1(cells["y01"], ranks)
    att = float(np.mean(cells["y11"]) - np.mean(cf))
    cf_sorted = np.sort(cf)
    qte = _quantile_type1(cells["y11"], quantiles) - _quantile_type1(cf_sorted, quantiles)
    return att, qte, cf_sorted


def _qdid_point(cells: Dict[str, np.ndarray], quantiles: np.ndarray) -> Tuple[float, np.ndarray]:
    """QDiD ATT and quantile effects, matching qte::QDiD() exactly.

    ``qte(tau) = Q7(y11,tau) - [Q7(y10,tau) + Q7(y01,tau) - Q7(y00,tau)]``. The
    ATT evaluates the control-group quantile functions at the treated
    pre-period's own-sample ECDF ranks with type-7 quantiles - qte 1.3.1's
    formula, which deviates in finite samples from the paper's k^QDID
    transformation mean (population-equivalent; see the REGISTRY Note).
    """
    q1 = _quantile_type7(cells["y11"], quantiles)
    q0 = (
        _quantile_type7(cells["y10"], quantiles)
        + _quantile_type7(cells["y01"], quantiles)
        - _quantile_type7(cells["y00"], quantiles)
    )
    ranks = _ecdf(cells["y10"], cells["y10"])
    att = float(
        np.mean(cells["y11"])
        - (
            np.mean(cells["y10"])
            + np.mean(_quantile_type7(cells["y01"], ranks))
            - np.mean(_quantile_type7(cells["y00"], ranks))
        )
    )
    return att, q1 - q0


def _interior_range(cells: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Eq. (17) plug-in interior range for CiC quantile effects.

    ``q_lower = F_10(min y00)``, ``q_upper = F_10(max y00)``: quantile effects
    are point-identified only inside ``(q_lower, q_upper)`` without the full
    support condition (Corollary 3.1 / Theorem 5.3).
    """
    y10 = cells["y10"]
    q_lower = float(_ecdf(y10, np.array([cells["y00"][0]]))[0])
    q_upper = float(_ecdf(y10, np.array([cells["y00"][-1]]))[0])
    return q_lower, q_upper


def _parse_2x2_formula(formula: str, data: pd.DataFrame) -> Tuple[str, str, str]:
    """Parse ``"outcome ~ treatment * time"`` style 2x2 formulas.

    Mirrors the DifferenceInDifferences formula grammar for the interaction
    forms; covariate terms are rejected (covariates are deferred from v1).
    """
    if "~" not in formula:
        raise ValueError("Formula must contain '~' to separate outcome from predictors")
    lhs, rhs = formula.split("~", 1)
    outcome = lhs.strip()
    rhs = rhs.strip()

    if "*" in rhs:
        parts = [p.strip() for p in rhs.split("*")]
        if len(parts) != 2:
            raise ValueError("Currently only supports single interaction (treatment * time)")
        treatment, time = parts
        if "+" in time:
            raise ValueError(
                "Covariates are not supported by ChangesInChanges/QDiD (deferred from v1; "
                "see the Melly-Santangelo covariate route in the methodology registry)."
            )
    elif ":" in rhs:
        terms = [t.strip() for t in rhs.split("+")]
        interaction = None
        mains: List[str] = []
        for term in terms:
            if ":" in term:
                if interaction is not None:
                    raise ValueError("Formula must contain exactly one interaction term")
                interaction = term
            else:
                mains.append(term)
        if interaction is None:
            raise ValueError(
                "Formula must include an interaction term (treatment * time or treatment:time)"
            )
        pair = [p.strip() for p in interaction.split(":")]
        if len(pair) != 2:
            raise ValueError("Interaction term must involve exactly two variables")
        if sorted(mains) != sorted(pair):
            raise ValueError(
                "Covariates are not supported by ChangesInChanges/QDiD (deferred from v1; "
                "the formula must be 'outcome ~ treatment + time + treatment:time')."
            )
        # Roles come from the MAIN-EFFECT order, not the interaction-term order:
        # CiC/QDiD are not symmetric in (treatment, time), and 'treated:post' vs
        # 'post:treated' must not silently swap semantics.
        treatment, time = mains[0], mains[1]
    else:
        raise ValueError(
            "Formula must include an interaction term (treatment * time or treatment:time)"
        )

    for name in (outcome, treatment, time):
        if name not in data.columns:
            raise ValueError(f"Column '{name}' from formula not found in data")
    return outcome, treatment, time


# =============================================================================
# Diagnostics
# =============================================================================


def _check_support(cells: Dict[str, np.ndarray]) -> None:
    """Warn on treated pre-period support outside the control pre-period range (CiC)."""
    if cells["y10"][0] < cells["y00"][0] or cells["y10"][-1] > cells["y00"][-1]:
        warnings.warn(
            "Treated pre-period outcomes fall outside the control pre-period support "
            "(Athey-Imbens Assumption 3.4 violated). The counterfactual distribution is "
            "only partially identified (Corollary 3.1): quantile effects are reliable "
            "only inside the reported (q_lower, q_upper) interior range, and the ATT "
            "involves extrapolation at the support edges.",
            UserWarning,
            stacklevel=2,
        )


def _check_ties(cells: Dict[str, np.ndarray]) -> None:
    """Warn on heavy ties (discrete-looking outcomes) in any cell."""
    max_share = 0.0
    for cell in cells.values():
        share = 1.0 - np.unique(cell).size / cell.size
        max_share = max(max_share, share)
    if max_share > _TIE_SHARE_WARN:
        warnings.warn(
            f"Outcome has heavy ties (up to {max_share:.0%} duplicate values within a "
            "(group, period) cell), suggesting a discrete or mixed distribution. The "
            "continuous CiC/QDiD machinery assumes continuously distributed outcomes "
            "(Athey-Imbens Assumption 5.1(iii)); with discrete outcomes only bounds are "
            "point-identified (Section 4, deferred) and the continuous formulas silently "
            "deliver one endpoint of those bounds.",
            UserWarning,
            stacklevel=2,
        )


def _check_qdid_monotonicity(cells: Dict[str, np.ndarray], quantiles: np.ndarray) -> None:
    """Warn when QDiD's counterfactual quantile curve is non-monotone (footnote 21)."""
    cq = (
        _quantile_type7(cells["y10"], quantiles)
        + _quantile_type7(cells["y01"], quantiles)
        - _quantile_type7(cells["y00"], quantiles)
    )
    if np.any(np.diff(cq) < -1e-12):
        warnings.warn(
            "QDiD's implied counterfactual quantile function is non-monotone on the "
            "requested grid (Athey-Imbens footnote 21: the QDiD model places testable "
            "restrictions on the data, and they appear violated here). Interpret the "
            "quantile effects with caution; ChangesInChanges does not impose this "
            "restriction and is the recommended estimator (p. 447).",
            UserWarning,
            stacklevel=2,
        )


# =============================================================================
# Bootstrap
# =============================================================================


def _bootstrap_replicates(
    point_fn: Callable[[Dict[str, np.ndarray], np.ndarray], Tuple[float, np.ndarray]],
    y: np.ndarray,
    g: np.ndarray,
    t: np.ndarray,
    unit_ids: Optional[np.ndarray],
    panel: bool,
    n_bootstrap: int,
    quantiles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap replicate matrix, shape ``(n_bootstrap, 1 + K)`` (col 0 = ATT).

    Resampling matches qte 1.3.1: panel mode samples unit ids with replacement
    (each unit's two periods travel together); repeated cross-section mode
    draws one pooled row resample of the stacked two-period data (unstratified,
    so cell sizes vary across draws). Replicates with an empty cell produce a
    NaN row rather than an exception.
    """
    n_cols = 1 + quantiles.shape[0]
    out = np.full((n_bootstrap, n_cols), np.nan)

    if panel:
        assert unit_ids is not None
        # Pre-pivot to unit-level arrays: one (y_pre, y_post, group) triple per unit.
        order = np.argsort(unit_ids, kind="stable")
        uid, y_o, g_o, t_o = unit_ids[order], y[order], g[order], t[order]
        pre_mask = t_o == 0
        # Balanced panel (enforced in fit): each unit has exactly one pre and one post row.
        y_pre = y_o[pre_mask]
        y_post = y_o[~pre_mask]
        g_unit = g_o[pre_mask]
        # uid is sorted, so pre/post slices align unit-by-unit.
        assert np.array_equal(uid[pre_mask], uid[~pre_mask])
        n_units = y_pre.shape[0]
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_units, n_units)
            gb = g_unit[idx]
            yb = np.concatenate([y_pre[idx], y_post[idx]])
            tb = np.concatenate([np.zeros(n_units), np.ones(n_units)])
            cells = _build_cells(yb, np.concatenate([gb, gb]), tb)
            if cells is None:
                continue
            att_b, qte_b = point_fn(cells, quantiles)[:2]
            out[b, 0] = att_b
            out[b, 1:] = qte_b
    else:
        n_rows = y.shape[0]
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_rows, n_rows)
            cells = _build_cells(y[idx], g[idx], t[idx])
            if cells is None:
                continue
            att_b, qte_b = point_fn(cells, quantiles)[:2]
            out[b, 0] = att_b
            out[b, 1:] = qte_b
    return out


def _bootstrap_inference(
    replicates: np.ndarray,
    qte_hat: np.ndarray,
    n_bootstrap: int,
    context: str,
) -> Tuple[float, np.ndarray, float, int]:
    """SEs and the sup-t critical value from the replicate matrix (qte conventions).

    Returns ``(att_se, qte_ses, sup_t_crit, n_valid)``. SEs are SDs over finite
    replicate rows (R ``sd``, ddof=1); if fewer than half the rows are finite,
    all SEs and the critical value are NaN (bootstrap_utils gate). The sup-t
    critical value ports qte's computeSE: an IQR-based scale ``sigmahalf`` per
    quantile column (type-1 quantiles; SD floored at 1e-9 as fallback when any
    column IQR is zero) and the hard-coded 0.95 type-1 quantile of the sup
    statistics - independent of ``alpha`` by construction (qte parity).
    """
    finite_rows = np.all(np.isfinite(replicates), axis=1)
    n_valid = int(np.count_nonzero(finite_rows))
    warn_bootstrap_failure_rate(n_valid, n_bootstrap, context)
    if n_valid < max(2, _MIN_VALID_REPLICATE_SHARE * n_bootstrap):
        k = replicates.shape[1] - 1
        return np.nan, np.full(k, np.nan), np.nan, n_valid

    good = replicates[finite_rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        ses = np.std(good, axis=0, ddof=1)
        att_se = float(ses[0])
        qte_ses = ses[1:]

        qte_cols = good[:, 1:]
        z_iqr = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
        q75 = np.array([_quantile_type1(np.sort(col), np.array([0.75]))[0] for col in qte_cols.T])
        q25 = np.array([_quantile_type1(np.sort(col), np.array([0.25]))[0] for col in qte_cols.T])
        sigmahalf = (q75 - q25) / z_iqr
        if np.any(sigmahalf == 0):
            sigmahalf = np.maximum(qte_ses, 1e-9)
        sup_stats = np.max(np.abs(qte_cols - qte_hat[None, :]) / sigmahalf[None, :], axis=1)
        sup_t_crit = float(_quantile_type1(np.sort(sup_stats), np.array([0.95]))[0])

    return att_se, qte_ses, sup_t_crit, n_valid


# =============================================================================
# Shared fit pipeline
# =============================================================================


def _fit_distributional(
    est: Any,
    data: pd.DataFrame,
    outcome: Optional[str],
    treatment: Optional[str],
    time: Optional[str],
    formula: Optional[str],
    unit: Optional[str],
    kind: str,
) -> ChangesInChangesResults:
    """Shared fit pipeline for ChangesInChanges and QDiD (``kind`` in {"cic", "qdid"})."""
    # Re-validate hyperparameters (set_params may have mutated them since __init__).
    _validate_all_params(est.get_params())
    quantiles = np.sort(
        np.asarray(_DEFAULT_QUANTILES if est.quantiles is None else est.quantiles, dtype=float)
    )

    # ---- column resolution -------------------------------------------------
    if formula is not None:
        outcome, treatment, time = _parse_2x2_formula(formula, data)
    elif outcome is None or treatment is None or time is None:
        raise ValueError(
            "Must provide either 'formula' or all of 'outcome', 'treatment', and 'time'"
        )
    used_cols = [outcome, treatment, time]
    if est.panel:
        if unit is None:
            raise ValueError("'unit' is required when panel=True (unit identifier column)")
        used_cols.append(unit)
    for col in used_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # ---- NA handling -------------------------------------------------------
    frame = data[used_cols].copy()
    n_before = len(frame)
    frame = frame.dropna()
    n_dropped = n_before - len(frame)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} row(s) with missing values in " f"{used_cols} before estimation.",
            UserWarning,
            stacklevel=2,
        )
    if len(frame) == 0:
        raise ValueError("No observations remain after dropping missing values")

    y_check = frame[outcome].to_numpy(dtype=float)
    if not np.all(np.isfinite(y_check)):
        n_nonfinite = int(np.count_nonzero(~np.isfinite(y_check)))
        raise ValueError(
            f"Outcome column '{outcome}' contains {n_nonfinite} non-finite value(s) "
            "(inf/-inf). Clean or drop these observations before fitting - they would "
            "silently corrupt the empirical CDFs, quantiles, and bootstrap."
        )

    validate_binary(frame[treatment].to_numpy(dtype=float), "treatment")
    validate_binary(frame[time].to_numpy(dtype=float), "time")

    # ---- panel hygiene -----------------------------------------------------
    unit_ids: Optional[np.ndarray] = None
    if est.panel:
        if frame.duplicated(subset=[unit, time]).any():
            raise ValueError(
                "panel=True requires at most one row per (unit, period); found duplicate "
                f"('{unit}', '{time}') combinations."
            )
        g_nunique = frame.groupby(unit)[treatment].nunique()
        if (g_nunique > 1).any():
            bad = g_nunique[g_nunique > 1].index.tolist()[:5]
            raise ValueError(
                "The treatment-group indicator must be constant within unit in the 2x2 "
                f"design (it marks group membership, not treatment receipt); units with "
                f"varying values include {bad}."
            )
        counts = frame.groupby(unit)[time].count()
        incomplete = counts[counts < 2].index
        if len(incomplete) > 0:
            warnings.warn(
                f"Dropped {len(incomplete)} unit(s) not observed in both periods "
                "(balanced-panel requirement, matching qte's makeBalancedPanel).",
                UserWarning,
                stacklevel=2,
            )
            frame = frame[~frame[unit].isin(incomplete)]
            if len(frame) == 0:
                raise ValueError("No balanced units remain after panel balancing")
        unit_ids = frame[unit].to_numpy()

    y = frame[outcome].to_numpy(dtype=float)
    g = frame[treatment].to_numpy(dtype=float).astype(np.int64)
    t = frame[time].to_numpy(dtype=float).astype(np.int64)

    # ---- cells + diagnostics -----------------------------------------------
    cells = _split_cells(y, g, t)
    _check_ties(cells)
    if kind == "cic":
        _check_support(cells)
        q_lower, q_upper = _interior_range(cells)
    else:
        _check_qdid_monotonicity(cells, quantiles)
        q_lower, q_upper = np.nan, np.nan

    # ---- point estimation ---------------------------------------------------
    if kind == "cic":
        att, qte, _ = _cic_point(cells, quantiles)
        point_fn: Callable[..., Any] = _cic_point
        context = "ChangesInChanges bootstrap"
    else:
        att, qte = _qdid_point(cells, quantiles)
        point_fn = _qdid_point
        context = "QDiD bootstrap"

    # ---- bootstrap ----------------------------------------------------------
    n_valid = 0
    if est.n_bootstrap > 0:
        rng = np.random.default_rng(est.seed)
        replicates = _bootstrap_replicates(
            point_fn, y, g, t, unit_ids, est.panel, est.n_bootstrap, quantiles, rng
        )
        # sup_t_crit is computed over ALL grid columns BEFORE the interior-range
        # NaN overwrite below (qte has no interior guard; excluding guarded
        # columns from the sup statistic would silently change c).
        att_se, qte_ses, sup_t_crit, n_valid = _bootstrap_inference(
            replicates, qte, est.n_bootstrap, context
        )
    else:
        att_se = np.nan
        qte_ses = np.full(quantiles.shape[0], np.nan)
        sup_t_crit = np.nan

    # ---- inference assembly ---------------------------------------------------
    # Joint-NaN contract: an invalid SE (non-finite or <= 0, e.g. a degenerate
    # bootstrap column with zero spread) must NaN the STORED se too, not only the
    # t/p/CI that safe_inference_batch masks - otherwise uniform_bands() would
    # build finite zero-width bands from a 0.0 se while pointwise inference is NaN.
    att_se = att_se if (np.isfinite(att_se) and att_se > 0) else np.nan
    qte_ses = np.where(np.isfinite(qte_ses) & (qte_ses > 0), qte_ses, np.nan)
    t_stat, p_value, conf_int = safe_inference(att, att_se, est.alpha)
    t_stats, p_values, ci_lo, ci_hi = safe_inference_batch(qte, qte_ses, est.alpha)

    if kind == "cic":
        exterior = ~((quantiles > q_lower) & (quantiles < q_upper))
        if np.any(exterior):
            warnings.warn(
                "Quantile effects at "
                f"{[round(float(q), 4) for q in quantiles[exterior]]} lie outside the "
                f"point-identified interior range ({q_lower:.4f}, {q_upper:.4f}) "
                "(Athey-Imbens eq. 17 / Theorem 5.3). Point estimates are reported for "
                "qte parity, but their inference fields are set to NaN.",
                UserWarning,
                stacklevel=2,
            )
            qte_ses = qte_ses.copy()
            qte_ses[exterior] = np.nan
            t_stats[exterior] = np.nan
            p_values[exterior] = np.nan
            ci_lo[exterior] = np.nan
            ci_hi[exterior] = np.nan

    quantile_effects = pd.DataFrame(
        {
            "quantile": quantiles,
            "qte": qte,
            "se": qte_ses,
            "t_stat": t_stats,
            "p_value": p_values,
            "conf_low": ci_lo,
            "conf_high": ci_hi,
        }
    )

    results = ChangesInChangesResults(
        att=att,
        se=att_se,
        t_stat=t_stat,
        p_value=p_value,
        conf_int=conf_int,
        quantile_effects=quantile_effects,
        q_lower=q_lower,
        q_upper=q_upper,
        sup_t_crit=sup_t_crit,
        n_obs=len(frame),
        cell_sizes={
            "control_pre": int(cells["y00"].size),
            "control_post": int(cells["y01"].size),
            "treated_pre": int(cells["y10"].size),
            "treated_post": int(cells["y11"].size),
        },
        n_bootstrap=est.n_bootstrap,
        n_bootstrap_valid=n_valid,
        panel=est.panel,
        estimator=kind,
        quantiles=quantiles,
        alpha=est.alpha,
    )
    est.results_ = results
    est.is_fitted_ = True
    return results


# =============================================================================
# Estimators
# =============================================================================


def _validate_quantiles(quantiles: Optional[Any]) -> None:
    if quantiles is None:
        return
    arr = np.asarray(quantiles, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"quantiles must be a non-empty 1-d array-like, got '{quantiles}'")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError(f"quantiles must be finite and strictly inside (0, 1), got '{quantiles}'")


def _validate_n_bootstrap(n_bootstrap: Any) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError(f"n_bootstrap must be a non-negative integer, got '{n_bootstrap}'")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be a non-negative integer, got '{n_bootstrap}'")


def _validate_alpha(alpha: Any) -> None:
    if not isinstance(alpha, (int, float, np.floating)) or isinstance(alpha, bool):
        raise ValueError(f"alpha must be a float strictly between 0 and 1, got '{alpha}'")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be a float strictly between 0 and 1, got '{alpha}'")


def _validate_panel(panel: Any) -> None:
    if not isinstance(panel, (bool, np.bool_)):
        raise ValueError(f"panel must be a boolean, got '{panel}'")


def _validate_seed(seed: Any) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be None or a non-negative integer, got '{seed}'")


def _validate_all_params(params: Dict[str, Any]) -> None:
    """Validate the full hyperparameter dict (used by __init__, set_params, and fit)."""
    _validate_quantiles(params["quantiles"])
    _validate_n_bootstrap(params["n_bootstrap"])
    _validate_alpha(params["alpha"])
    _validate_panel(params["panel"])
    _validate_seed(params["seed"])


class ChangesInChanges:
    """Changes-in-Changes estimator (Athey & Imbens 2006) for the 2x2 design.

    Estimates the ATT (eq. 36) and quantile treatment effects on the treated
    (eq. 18) by building the treated group's counterfactual untreated outcome
    distribution ``F_10(F_00^{-1}(F_01(y)))`` (Theorem 3.1). Point estimates
    match ``qte::CiC()`` (R, v1.3.1) exactly; the empirical inverse CDF is the
    paper's eq. (35)/(A.1) ceiling-order-statistic convention (R type-1).

    Inference is bootstrap-only (``n_bootstrap=0`` disables it and reports NaN
    inference fields): panel mode resamples units with both periods together;
    repeated cross-section mode draws a pooled row resample. Standard errors
    are replicate SDs with symmetric normal-approximation intervals;
    ``results_.sup_t_crit`` carries qte's sup-t critical value for uniform
    bands at a fixed 95% level (independent of ``alpha``, matching qte).

    Parameters
    ----------
    quantiles : array-like, optional
        Quantile grid strictly inside (0, 1). Default: 0.05 to 0.95 in steps
        of 0.05 (matches qte's ``probs``).
    n_bootstrap : int, default=200
        Bootstrap replicates. 0 disables inference (NaN se/t/p/CI).
    alpha : float, default=0.05
        Significance level for pointwise confidence intervals.
    panel : bool, default=False
        True when the same units are observed in both periods (requires
        ``unit=`` at fit time; enforces a balanced panel). Affects only the
        bootstrap resampling scheme - the point estimator uses the four
        marginal cell distributions either way.
    seed : int, optional
        Seed for the bootstrap RNG (``numpy.random.default_rng``).

    Notes
    -----
    Quantile effects are point-identified only on the eq. (17) interior range
    ``(q_lower, q_upper)``; effects outside it keep their point estimates (qte
    parity) but report NaN inference with a warning.

    Additive random group-time shocks (random effects at the group x period
    level) BIAS the CiC estimator - unlike linear DiD, where they only
    complicate inference - and are not detectable in a 2x2 design (Athey &
    Imbens 2006, p. 476). With more than two groups/periods they are testable
    (Theorem 6.4), but that extension is deferred.

    Covariates, discrete-outcome bounds, and analytical standard errors are
    deferred from v1 and documented in docs/methodology/REGISTRY.md.
    """

    def __init__(
        self,
        quantiles: Optional[Any] = None,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        panel: bool = False,
        seed: Optional[int] = None,
    ):
        # Stored verbatim (sklearn-clone contract): quantiles=None resolves to the
        # default grid at fit time, the raw None round-trips get_params().
        _validate_all_params(
            {
                "quantiles": quantiles,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "panel": panel,
                "seed": seed,
            }
        )
        self.quantiles = quantiles
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.panel = panel
        self.seed = seed
        self.is_fitted_ = False
        self.results_: Optional[ChangesInChangesResults] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return constructor hyperparameters (raw values, round-trips ``__init__``).

        ``deep`` is accepted for sklearn compatibility (``sklearn.base.clone``
        calls ``get_params(deep=False)``) and is ignored - there are no nested
        estimators.
        """
        return {
            "quantiles": self.quantiles,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "panel": self.panel,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> "ChangesInChanges":
        """Set hyperparameters transactionally (a failing call mutates nothing)."""
        valid = set(self.get_params())
        for key in params:
            if key not in valid:
                raise ValueError(f"Unknown parameter: {key}")
        _validate_all_params({**self.get_params(), **params})
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(
        self,
        data: pd.DataFrame,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        time: Optional[str] = None,
        formula: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> ChangesInChangesResults:
        """Fit the CiC estimator on a 2x2 dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format data with one row per observation.
        outcome, treatment, time : str, optional
            Column names: continuous outcome, binary group indicator (1 =
            treated group in BOTH periods), binary post-period indicator.
            Required unless ``formula`` is given.
        formula : str, optional
            R-style 2x2 formula, e.g. ``"y ~ treated * post"``. Covariate
            terms raise (deferred from v1).
        unit : str, optional
            Unit identifier column. Required when ``panel=True``; ignored
            (documented) when ``panel=False``, matching qte's ``idname``.
        """
        return _fit_distributional(self, data, outcome, treatment, time, formula, unit, "cic")


class QDiD:
    """Quantile Difference-in-Differences comparison estimator (2x2 design).

    Applies DiD quantile-by-quantile: ``qte(tau) = Q(y11, tau) - [Q(y10, tau)
    + Q(y01, tau) - Q(y00, tau)]`` with R type-7 (linear-interpolation)
    quantiles, matching ``qte::QDiD()`` (v1.3.1) exactly - including its ATT
    formula, which evaluates the control-group quantile functions at the
    treated pre-period's own-sample ranks. This finite-sample form deviates
    from the Athey-Imbens k^QDID transformation mean (they are
    population-equivalent; see the labeled Note in the methodology registry).

    Athey & Imbens recommend :class:`ChangesInChanges` over QDiD (2006,
    p. 447): QDiD's justifying model is not invariant to monotone
    transformations of the outcome, forces the unobservable distribution to be
    identical across all four cells, and places testable restrictions on the
    data (a warning fires when the implied counterfactual quantile function is
    non-monotone). QDiD's mean effect matches standard DiD's ATT in
    population; the paper provides no asymptotic theory for QDiD, so inference
    is a bootstrap convention shared with the qte package.

    Constructor parameters, fit signature, bootstrap behavior, and the results
    container are identical to :class:`ChangesInChanges` (no interior-range
    guard: eq. 17 has no QDiD analogue).
    """

    def __init__(
        self,
        quantiles: Optional[Any] = None,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        panel: bool = False,
        seed: Optional[int] = None,
    ):
        # Stored verbatim (sklearn-clone contract): quantiles=None resolves to the
        # default grid at fit time, the raw None round-trips get_params().
        _validate_all_params(
            {
                "quantiles": quantiles,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "panel": panel,
                "seed": seed,
            }
        )
        self.quantiles = quantiles
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.panel = panel
        self.seed = seed
        self.is_fitted_ = False
        self.results_: Optional[ChangesInChangesResults] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return constructor hyperparameters (raw values, round-trips ``__init__``).

        ``deep`` is accepted for sklearn compatibility (``sklearn.base.clone``
        calls ``get_params(deep=False)``) and is ignored - there are no nested
        estimators.
        """
        return {
            "quantiles": self.quantiles,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "panel": self.panel,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> "QDiD":
        """Set hyperparameters transactionally (a failing call mutates nothing)."""
        valid = set(self.get_params())
        for key in params:
            if key not in valid:
                raise ValueError(f"Unknown parameter: {key}")
        _validate_all_params({**self.get_params(), **params})
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(
        self,
        data: pd.DataFrame,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        time: Optional[str] = None,
        formula: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> ChangesInChangesResults:
        """Fit the QDiD estimator on a 2x2 dataset (see ChangesInChanges.fit)."""
        return _fit_distributional(self, data, outcome, treatment, time, formula, unit, "qdid")
