"""Result containers for the TWFE implicit-weight diagnostics.

See :mod:`diff_diff.twfe_weights` for the entry points that build these, and
for the upstream MIT attribution.

Both containers subclass :class:`diff_diff.Diagnostic`: they assess a design
(what a regression implicitly weights) rather than estimating a causal effect,
so neither carries the estimator quintet ``att``/``se``/``t_stat``/``p_value``/
``conf_int``. The headline scalars are named ``implied_att`` and ``estimate``
precisely so they do not read as inference-bearing point estimates - the
decomposition is an algebraic identity, exactly like
:class:`~diff_diff.BaconDecompositionResults`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results_base import Diagnostic

__all__ = ["ATTGTWeightsResult", "TWFEDecompositionResult"]

_AGGREGATION_LABELS = {
    "twfe": "TWFE regression",
    "overall": "ATT^O (Callaway & Sant'Anna overall)",
    "simple": "ATT^simple (Callaway & Sant'Anna simple)",
}

# Per-cell balance columns, in report order. The three ``_`` -prefixed groups
# mirror R's ``cov_bal_df`` under diff-diff naming; see the mapping table in
# the REGISTRY entry.
_BALANCE_STATS = (
    "unweighted_treated",
    "unweighted_control",
    "unweighted_diff",
    "weighted_treated",
    "weighted_control",
    "weighted_diff",
    "sd",
    "unweighted_log_ratio_sd",
    "weighted_log_ratio_sd",
    "unweighted_frac_extreme",
    "weighted_frac_extreme",
)


def _fmt(value: float, width: int = 12, digits: int = 4) -> str:
    """Right-aligned float that renders NaN without blowing up the layout."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{digits}f}"


@dataclass
class ATTGTWeightsResult(Diagnostic):
    """Weights that an estimand places on each group-time effect ATT(g, t).

    Returned by :func:`diff_diff.attgt_weights`. One row per ``(g, t)`` cell.

    Attributes
    ----------
    weights : pd.DataFrame
        Columns ``group``, ``time``, ``post``, ``weight``, ``att``. ``post``
        is ``1`` when ``time >= group`` (the cells the estimand targets),
        ``0`` for pre-treatment cells. ``att`` is the ATT(g, t) the weight
        multiplies, carried through from the source so that
        ``(weight * att).sum()`` reproduces ``implied_att``.
    aggregation : str
        Which estimand's weights these are: ``"twfe"``, ``"overall"``
        (ATT^O), or ``"simple"`` (ATT^simple).
    implied_att : float
        ``sum(weight * att)`` - what the estimand delivers given these
        ATT(g, t). For ``aggregation="twfe"`` this is the TWFE coefficient.
    n_negative : int
        Number of cells - PRE and post - receiving a negative weight. Under
        ``aggregation="twfe"`` the weights over the full ``g != 0`` grid sum
        to zero (post to +1, pre to -1), so this is non-zero in every
        staggered design; read ``n_negative_post`` for the pathology.
    negative_weight_share : float
        ``sum(|w| : w < 0) / sum(|w|)`` over ALL cells - how much of the total
        weight mass points the wrong way. Near 0.5 is normal under
        ``"twfe"`` for the same reason. ``0.0`` when no weight is negative.
    n_negative_post : int
        Number of POST-treatment cells receiving a negative weight. This is
        the classic staggered-adoption pathology: the regression subtracts
        treatment effects it should be adding. Zero for ``"overall"`` and
        ``"simple"`` by construction.
    negative_post_weight_share : float
        ``sum(|w| : w < 0, post) / sum(|w| : post)`` - the share of
        post-treatment weight mass that is negative. No R counterpart; see
        the methodology registry.
    n_cells : int
        Number of ``(g, t)`` cells contributing.
    source : str or None
        ``"CallawaySantAnnaResults"`` when built from a fitted result,
        ``"DataFrame"`` on the fallback path.
    control_group, base_period : str or None
        Design metadata carried from the source fit, when available.
    n_dropped_cells : int
        Cells excluded because their ATT(g, t) was non-estimable (NaN).
    """

    weights: pd.DataFrame
    aggregation: str
    implied_att: float
    n_negative: int
    negative_weight_share: float
    n_negative_post: int
    negative_post_weight_share: float
    n_cells: int
    source: Optional[str] = None
    control_group: Optional[str] = None
    base_period: Optional[str] = None
    n_dropped_cells: int = 0

    def __repr__(self) -> str:
        return (
            f"ATTGTWeightsResult(aggregation={self.aggregation!r}, "
            f"implied_att={self.implied_att:.4f}, "
            f"n_cells={self.n_cells}, n_negative={self.n_negative})"
        )

    def summary(self) -> str:
        """Formatted per-cell weight table with the negative-weight roll-up."""
        width = 72
        label = _AGGREGATION_LABELS.get(self.aggregation, self.aggregation)
        lines = [
            "=" * width,
            "Implicit Weights on ATT(g, t)".center(width),
            "=" * width,
            "",
            f"{'Estimand:':<28} {label}",
            f"{'Group-time cells:':<28} {self.n_cells:>10}",
        ]
        if self.n_dropped_cells:
            lines.append(f"{'Non-estimable cells dropped:':<28} {self.n_dropped_cells:>10}")
        if self.source is not None:
            lines.append(f"{'Source:':<28} {self.source}")
        if self.control_group is not None:
            lines.append(f"{'Control group:':<28} {self.control_group}")
        if self.base_period is not None:
            lines.append(f"{'Base period:':<28} {self.base_period}")
        lines += [
            "",
            "-" * width,
            f"{'Group':>8} {'Time':>8} {'Post':>6} {'Weight':>14} {'ATT(g,t)':>14}",
            "-" * width,
        ]
        for row in self.weights.itertuples(index=False):
            lines.append(
                f"{row.group:>8} {row.time:>8} {int(row.post):>6} "
                f"{_fmt(row.weight, 14, 6)} {_fmt(row.att, 14, 6)}"
            )
        lines += [
            "-" * width,
            "",
            f"{'Implied estimate:':<28} {_fmt(self.implied_att)}",
            f"{'Negative POST-period cells:':<28} {self.n_negative_post:>12}",
            f"{'Negative POST-weight share:':<28} {_fmt(self.negative_post_weight_share)}",
            f"{'Negative cells (all):':<28} {self.n_negative:>12}",
            f"{'Negative share (all):':<28} {_fmt(self.negative_weight_share)}",
            "",
        ]
        if self.n_negative_post:
            lines += [
                "Note: negative POST-period weights mean this estimand subtracts some",
                "      treatment-period ATT(g, t). Under heterogeneous effects the",
                "      estimate need not lie in the convex hull of those effects.",
                "",
            ]
        elif self.n_negative:
            lines += [
                "Note: the negative weights fall on PRE-treatment cells only, which is",
                "      how the TWFE weights sum to zero over the full grid; no",
                "      treatment-period effect is being subtracted.",
                "",
            ]
        lines.append("=" * width)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print :meth:`summary` to stdout."""
        print(self.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """Per-cell weight table (a copy)."""
        return self.weights.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serializable view of the result."""
        return {
            "aggregation": self.aggregation,
            "implied_att": self.implied_att,
            "n_cells": self.n_cells,
            "n_negative": self.n_negative,
            "negative_weight_share": self.negative_weight_share,
            "n_negative_post": self.n_negative_post,
            "negative_post_weight_share": self.negative_post_weight_share,
            "n_dropped_cells": self.n_dropped_cells,
            "source": self.source,
            "control_group": self.control_group,
            "base_period": self.base_period,
            "weights": self.weights.to_dict(orient="list"),
        }


@dataclass
class TWFEDecompositionResult(Diagnostic):
    """Decomposition of a TWFE estimate into weighted ATT(g, t).

    Returned by :func:`diff_diff.decompose_twfe_weights`.

    Attributes
    ----------
    cells : pd.DataFrame
        Columns ``group``, ``time``, ``post``, ``att``, ``weight``, ``ess``
        and ``remainder``. ``weight`` is the implicit weight the regression
        places on that cell's ATT(g, t) - R's ``alpha_weight``.
    method : str
        ``"fwl"`` - Frisch-Waugh-Lovell residual weights from the TWFE
        regression. (The only method currently implemented; upstream's AIPW
        decomposition is a documented follow-up.)
    estimate : float
        The estimate being decomposed - ``decomposition + remainder``.
    decomposition : float
        ``sum(weight * att)`` over all cells, pre and post.
    remainder : float
        Part of ``estimate`` not attributable to any ATT(g, t) cell.
        Identically ``0.0`` except under ``base_period="gmin1"``.
    pretrend_bias : float
        ``sum(weight * att)`` over PRE-treatment cells only. Under parallel
        trends every pre-treatment ATT(g, t) is zero and this vanishes; a
        non-zero value is the contribution of parallel-trends violations to
        ``estimate``.
    post_only : float
        ``sum(weight * att)`` over post-treatment cells only.
    base_period : str or None
        ``"first_period"`` or ``"gmin1"``.
    covariates : tuple of str
        Covariates the regression adjusted for. Empty tuple when none.
    effective_sample_size : float
        Weight-concentration roll-up. Small values relative to ``n_units``
        mean the estimate leans on few observations.
    n_units, n_periods : int
        Panel dimensions.
    balance : pd.DataFrame or None
        Per-cell implicit covariate balance, populated when
        ``balance_covariates=`` was requested. Read it via
        :meth:`covariate_balance`.
    """

    cells: pd.DataFrame
    method: str
    estimate: float
    decomposition: float
    remainder: float
    pretrend_bias: float
    post_only: float
    base_period: Optional[str]
    covariates: Tuple[str, ...]
    effective_sample_size: float
    n_units: int
    n_periods: int
    balance: Optional[pd.DataFrame] = field(default=None)

    def __repr__(self) -> str:
        return (
            f"TWFEDecompositionResult(method={self.method!r}, "
            f"estimate={self.estimate:.4f}, "
            f"pretrend_bias={self.pretrend_bias:.4f}, "
            f"n_cells={len(self.cells)})"
        )

    def summary(self) -> str:
        """Formatted decomposition table with the pre-trend contribution."""
        width = 78
        method_label = {
            "fwl": "TWFE regression (Frisch-Waugh-Lovell implicit weights)",
        }.get(self.method, self.method)
        covs = ", ".join(self.covariates) if self.covariates else "(none)"
        lines = [
            "=" * width,
            "Decomposition into Group-Time Effects".center(width),
            "=" * width,
            "",
            f"{'Method:':<30} {method_label}",
            f"{'Covariates:':<30} {covs}",
        ]
        if self.base_period is not None:
            lines.append(f"{'Base period:':<30} {self.base_period}")
        lines += [
            f"{'Units / periods:':<30} {self.n_units} / {self.n_periods}",
            f"{'Group-time cells:':<30} {len(self.cells)}",
            "",
            "-" * width,
            f"{'Group':>8} {'Time':>8} {'Post':>6} {'Weight':>14} "
            f"{'ATT(g,t)':>14} {'Contribution':>14}",
            "-" * width,
        ]
        for row in self.cells.itertuples(index=False):
            lines.append(
                f"{row.group:>8} {row.time:>8} {int(row.post):>6} "
                f"{_fmt(row.weight, 14, 6)} {_fmt(row.att, 14, 6)} "
                f"{_fmt(row.weight * row.att, 14, 6)}"
            )
        lines += [
            "-" * width,
            "",
            f"{'Estimate:':<30} {_fmt(self.estimate)}",
            f"{'  from ATT(g,t) cells:':<30} {_fmt(self.decomposition)}",
            f"{'  post-treatment only:':<30} {_fmt(self.post_only)}",
            f"{'  pre-trend violations:':<30} {_fmt(self.pretrend_bias)}",
            f"{'  remainder:':<30} {_fmt(self.remainder)}",
            "",
            f"{'Effective sample size:':<30} {_fmt(self.effective_sample_size)}",
            "",
        ]
        if abs(self.pretrend_bias) > 1e-10:
            lines += [
                "Note: a non-zero pre-trend contribution means pre-treatment",
                "      ATT(g, t) are not zero, so part of the estimate reflects",
                "      parallel-trends violations rather than treatment effects.",
                "",
            ]
        if self.balance is not None:
            lines += [
                "Covariate balance available via .covariate_balance().",
                "",
            ]
        lines.append("=" * width)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print :meth:`summary` to stdout."""
        print(self.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """Per-cell decomposition table (a copy)."""
        return self.cells.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serializable view of the result."""
        out: Dict[str, Any] = {
            "method": self.method,
            "estimate": self.estimate,
            "decomposition": self.decomposition,
            "remainder": self.remainder,
            "pretrend_bias": self.pretrend_bias,
            "post_only": self.post_only,
            "base_period": self.base_period,
            "covariates": list(self.covariates),
            "effective_sample_size": self.effective_sample_size,
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "cells": self.cells.to_dict(orient="list"),
        }
        if self.balance is not None:
            out["balance"] = self.balance.to_dict(orient="list")
        return out

    def covariate_balance(
        self,
        *,
        level: str = "summary",
        standardize: bool = True,
        post_only: bool = True,
    ) -> pd.DataFrame:
        """Implicit-weight covariate balance.

        Asks whether the weights the regression implicitly applies actually
        balance the covariates across the treated and comparison groups. If
        ``weighted_diff`` is no closer to zero than ``unweighted_diff``, the
        covariate adjustment is not buying what it appears to.

        Parameters
        ----------
        level : {"summary", "cell"}, default "summary"
            ``"summary"`` aggregates across ``(g, t)`` cells to one row per
            covariate, weighting each cell by its implicit weight.
            ``"cell"`` returns the unaggregated per-``(g, t)`` rows.
        standardize : bool, default True
            Append ``unweighted_std_diff`` / ``weighted_std_diff``, the
            differences divided by the pooled standard deviation. These are
            a diff-diff addition; R reports the raw differences only.
        post_only : bool, default True
            Restrict the summary roll-up to post-treatment cells, matching
            R's ``mp_covariate_bal_summary_helper``. Ignored when
            ``level="cell"``.

        Returns
        -------
        pd.DataFrame
            One row per covariate (``level="summary"``) or per
            ``(group, time, covariate)`` (``level="cell"``).

        Raises
        ------
        ValueError
            If balance was not requested at compute time, or ``level`` is
            not one of the two accepted values.
        """
        if self.balance is None:
            raise ValueError(
                "Covariate balance was not computed for this decomposition. "
                "Re-run with balance_covariates=, e.g.\n"
                "    decompose_twfe_weights(..., balance_covariates=['x1', 'x2'])"
            )
        if level not in ("summary", "cell"):
            raise ValueError(f"level must be 'summary' or 'cell', got {level!r}")

        table = self.balance.copy()
        if level == "cell":
            if standardize:
                table = _append_standardized(table)
            return table

        weights = self.cells.set_index(["group", "time"])["weight"]
        keys = pd.MultiIndex.from_arrays([table["group"], table["time"]])
        table["_w"] = weights.reindex(keys).to_numpy()
        if post_only:
            # Mask on the `post` COLUMN, not on a zero roll-up weight: R's
            # post-only helper never touches the pre cells, but a post cell
            # whose implicit weight happens to be exactly zero still
            # contributes - and still propagates its NA.
            table = table[table["post"].to_numpy().astype(bool)]

        # R propagates NA: if ANY contributing cell of a (covariate, statistic)
        # is NA, the summary is NA. pandas' sum() skips NaN, which would turn
        # the documented NA return of frac_treated_extreme (fewer than three
        # distinct values) into a spurious 0.0.
        stats = table[list(_BALANCE_STATS)].mul(table["_w"], axis=0)
        groups = table["covariate"].to_numpy()
        rolled = stats.groupby(groups, sort=False).sum(min_count=1)
        any_nan = table[list(_BALANCE_STATS)].isna().groupby(groups, sort=False).any()
        rolled = rolled.mask(any_nan)
        rolled.index.name = "covariate"
        out = rolled.reset_index()
        if standardize:
            out = _append_standardized(out)
        return out


def _append_standardized(table: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_std_diff`` columns (difference / pooled SD).

    A diff-diff addition on top of R's columns - additive, so parity is
    asserted on the R columns only. ``sd == 0`` yields NaN rather than an
    infinity, so a degenerate covariate does not poison a summary table.
    """
    out = table.copy()
    sd = out["sd"].to_numpy(dtype=float)
    safe = np.where(sd == 0, np.nan, sd)
    out["unweighted_std_diff"] = out["unweighted_diff"].to_numpy(dtype=float) / safe
    out["weighted_std_diff"] = out["weighted_diff"].to_numpy(dtype=float) / safe
    return out
