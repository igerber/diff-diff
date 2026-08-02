"""Optimal data-driven regression discontinuity plots (Calonico, Cattaneo &
Titiunik 2015, JASA) with end-to-end parity against R ``rdrobust`` 4.0.0's
``rdplot()``.

An RD plot has two ingredients (CCT 2015, Section 2.1): global order-``p``
polynomial fits of the outcome on the running variable, estimated separately
on each side of the cutoff, and binned local sample means of the outcome over
a disjoint partition of each side's support. :class:`RDPlot` selects the
number of bins with the paper's data-driven selectors (IMSE-optimal or
mimicking-variance, evenly-spaced or quantile-spaced partitions, spacings or
polynomial-regression variance estimators - the eight ``binselect`` values)
and returns the full set of plot quantities; rendering is a thin optional
matplotlib layer on the result.

Parity target: ``rdplot()`` in the CRAN rdrobust 4.0.0 sources (rdplot.R; the
line references in comments below point there). All numerical outputs -
``J``/``J_IMSE``/``J_MV``, the implied scale and WIMSE weights, bin means,
per-bin CIs, and the global-fit coefficients/curves - are golden-tested
against R output on the vendored Senate dataset and simulated designs.

Notes on scope (documented seams, mirroring :class:`RegressionDiscontinuity`):
sampling weights and R's ``subset=`` are not parameters (filter the DataFrame
instead); ``covs_eval`` is fixed at R's default ``"mean"``. R prints an
unconditional ``message()`` whenever covariates are used with the global
polynomial fit ("may deliver results that are not visually compatible with
the local binned means"); that caveat lives here in the docstring rather than
as an unavoidable runtime warning.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ._base import BaseEstimator
from ._deprecation import NOT_SUPPLIED, require_arg, resolve_renamed_kwarg
from ._rdrobust_port import (
    _covs_gamma,
    _normalize_kernel,
    _var0,
    covs_drop_fun,
    qrXXinv,
    rdrobust_kweight,
)
from .results_base import Diagnostic
from .utils import validate_covariate_names

__all__ = ["RDPlot", "RDPlotResult"]

# binselect value -> (partition meth, variance family, target, description).
# Description strings are transcribed exactly from rdplot.R:395-446.
_BINSELECT_INFO: Dict[str, Tuple[str, str, str, str]] = {
    "es": ("es", "hat", "imse", "IMSE-optimal evenly-spaced method using spacings estimators"),
    "espr": ("es", "chk", "imse", "IMSE-optimal evenly-spaced method using polynomial regression"),
    "esmv": (
        "es",
        "hat",
        "mv",
        "mimicking variance evenly-spaced method using spacings estimators",
    ),
    "esmvpr": (
        "es",
        "chk",
        "mv",
        "mimicking variance evenly-spaced method using polynomial regression",
    ),
    "qs": ("qs", "hat", "imse", "IMSE-optimal quantile-spaced method using spacings estimators"),
    "qspr": (
        "qs",
        "chk",
        "imse",
        "IMSE-optimal quantile-spaced method using polynomial regression",
    ),
    "qsmv": (
        "qs",
        "hat",
        "mv",
        "mimicking variance quantile-spaced method using spacings estimators",
    ),
    "qsmvpr": (
        "qs",
        "chk",
        "mv",
        "mimicking variance quantile-spaced method using polynomial regression",
    ),
}

# masspoints="adjust" remap under heavy ties (rdplot.R:130-135): the spacings
# variance estimators degenerate on tied data, so R switches each spacings
# selection to its polynomial-regression sibling. Explicit *pr picks are
# untouched.
_MASSPOINTS_REMAP = {"es": "espr", "esmv": "esmvpr", "qs": "qspr", "qsmv": "qsmvpr"}

_KERNEL_TYPE_LABEL = {
    "uniform": "Uniform",
    "epanechnikov": "Epanechnikov",
    "triangular": "Triangular",
}

_PairF = Tuple[float, float]
_PairI = Tuple[int, int]


def _r_seq_by(from_: float, to: float, by: float) -> np.ndarray:
    """R ``seq(from, to, by)`` for ``by > 0`` (seq.default): the point count
    is ``as.integer((to - from)/by + 1e-10)`` (truncation with R's fuzz) and
    the grid is clamped into ``to`` via ``pmin``."""
    m = int((to - from_) / by + 1e-10)
    return np.minimum(from_ + np.arange(m + 1) * by, to)


def _find_interval(x: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """R ``findInterval(x, vec, rightmost.closed=TRUE)`` (1-based interval
    ids). ``searchsorted(side='right')`` counts elements ``<= x``, which is
    findInterval's answer everywhere except ``x == vec[last]``, where
    rightmost.closed folds the point into the last interval. Values beyond
    ``vec[last]`` keep id ``len(vec)`` exactly like R - reachable when the
    quantile prob grid's last point lands one ulp short of 1 (the ``j/J``
    float products decide whether R's ``pmin`` clamp engages; the identical
    IEEE arithmetic here reproduces whichever way it falls)."""
    idx = np.searchsorted(vec, x, side="right")
    idx[x == vec[-1]] = len(vec) - 1
    return idx


def _pow_outer(u: np.ndarray, degree: int) -> np.ndarray:
    """R ``outer(u, 0:degree, "^")`` - elementwise pow, NOT the successive
    products of ``rdrobust_vander`` (rdplot.R builds every design this way)."""
    return np.power.outer(u, np.arange(degree + 1))


def _resolve_pair(
    value: Union[None, float, int, Tuple[Any, Any], List[Any]], default: float
) -> Tuple[float, float]:
    """R's scalar-or-length-2 argument convention (rdplot.R:77-104)."""
    if value is None:
        return float(default), float(default)
    if isinstance(value, (tuple, list)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


@dataclass
class RDPlotResult(Diagnostic):
    """Quantities of a data-driven RD plot (field names follow R's rdplot
    return list; ``vars_bins`` columns carry R's ``rdplot_*`` names).

    ``J`` is always integral: a fractional ``scale * J`` product takes the
    ceiling per CCT 2015 Equation 2 (see the Deviation note in
    :class:`RDPlot` - R 4.0.0 crashes on that surface instead).
    ``vars_bins`` contains only NON-EMPTY bins (R's ``tapply`` semantics)
    while ``bin_avg``/``bin_med`` summarize ALL bin lengths of the partition.
    CI columns are always computed (at ``ci_level``); ``ci_requested`` records
    whether ``ci=`` was passed and is the default for drawing error bars in
    :meth:`plot`.
    """

    coef: pd.DataFrame
    vars_bins: pd.DataFrame
    vars_poly: pd.DataFrame
    J: _PairF
    J_IMSE: _PairF
    J_MV: _PairF
    scale: _PairF
    rscale: _PairF
    bin_avg: _PairF
    bin_med: _PairF
    p: int
    cutoff: float
    h: _PairF
    N: _PairI
    N_h: _PairI
    binselect: str
    kernel_type: str
    ci_level: float
    ci_requested: bool
    covariate_coefficients: Optional[Dict[str, float]] = None
    covariates_dropped: Optional[List[str]] = None

    @property
    def wimse_variance_weight(self) -> _PairF:
        """Implied WIMSE variance weights ``1/(1 + rscale^3)`` per side
        (CCT 2015 Supplement S.1; printed by R's ``summary.rdplot``)."""
        return (1 / (1 + self.rscale[0] ** 3), 1 / (1 + self.rscale[1] ** 3))

    @property
    def wimse_bias_weight(self) -> _PairF:
        """Implied WIMSE bias weights ``rscale^3/(1 + rscale^3)`` per side."""
        return (
            self.rscale[0] ** 3 / (1 + self.rscale[0] ** 3),
            self.rscale[1] ** 3 / (1 + self.rscale[1] ** 3),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """The non-empty-bin table (``vars_bins``), copied."""
        return self.vars_bins.copy()

    def summary(self) -> str:
        """Formatted summary mirroring R's ``summary.rdplot`` layout
        (rdplot.R:610-654), including the implied scale and WIMSE weights."""

        def _num(v: float) -> str:
            return f"{v:10.3f}"

        def _int(v: float) -> str:
            return f"{v:10g}"

        lines = [
            "RD Plot (CCT 2015 / rdrobust rdplot parity)",
            "",
            f"Number of Obs.           {self.N[0] + self.N[1]:10d}",
            f"Kernel                   {self.kernel_type:>10}",
            f"Bin Method               {self.binselect}",
            "",
            f"{'':25}{'Left':>10}   {'Right':>10}",
            f"Number of Obs.           {self.N[0]:10d}   {self.N[1]:10d}",
            f"Eff. Number of Obs.      {self.N_h[0]:10d}   {self.N_h[1]:10d}",
            f"Order poly. fit (p)      {self.p:10d}   {self.p:10d}",
            f"BW poly. fit (h)         {_num(self.h[0])}   {_num(self.h[1])}",
            f"Number of bins scale     {_num(self.scale[0])}   {_num(self.scale[1])}",
            "",
            f"Bins Selected            {_int(self.J[0])}   {_int(self.J[1])}",
            f"Average Bin Length       {_num(self.bin_avg[0])}   {_num(self.bin_avg[1])}",
            f"Median Bin Length        {_num(self.bin_med[0])}   {_num(self.bin_med[1])}",
            "",
            f"IMSE-optimal bins        {_int(self.J_IMSE[0])}   {_int(self.J_IMSE[1])}",
            f"Mimicking Variance bins  {_int(self.J_MV[0])}   {_int(self.J_MV[1])}",
            "",
            "Relative to IMSE-optimal:",
            f"Implied scale            {_num(self.rscale[0])}   {_num(self.rscale[1])}",
            f"WIMSE variance weight    {_num(self.wimse_variance_weight[0])}"
            f"   {_num(self.wimse_variance_weight[1])}",
            f"WIMSE bias weight        {_num(self.wimse_bias_weight[0])}"
            f"   {_num(self.wimse_bias_weight[1])}",
        ]
        if self.covariate_coefficients is not None:
            lines += ["", "Covariate adjustment (nuisance coefficients, not causal effects):"]
            lines += [f"  {k:20} {v: .6g}" for k, v in self.covariate_coefficients.items()]
        return "\n".join(lines)

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        dots_color: Optional[str] = None,
        line_color: Optional[str] = None,
        show_ci: Optional[bool] = None,
    ) -> Any:
        """Render the RD plot (binned means, global polynomial curves, cutoff
        line; error bars when CIs were requested at fit time or forced via
        ``show_ci=True``). Requires matplotlib (an optional dependency).

        Returns the matplotlib axes. Rendering is cosmetic - the parity
        surface of this port is the NUMBERS on the result object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        if show_ci is None:
            show_ci = self.ci_requested

        vb = self.vars_bins
        if show_ci:
            ax.errorbar(
                vb["rdplot_mean_bin"],
                vb["rdplot_mean_y"],
                yerr=np.vstack(
                    [
                        vb["rdplot_mean_y"] - vb["rdplot_ci_l"],
                        vb["rdplot_ci_r"] - vb["rdplot_mean_y"],
                    ]
                ),
                fmt="none",
                ecolor="gray",
                elinewidth=1,
                alpha=0.8,
                zorder=1,
            )
        ax.scatter(
            vb["rdplot_mean_bin"],
            vb["rdplot_mean_y"],
            color=dots_color or "darkblue",
            s=18,
            zorder=2,
        )
        # Split by construction (500 points per side): both sides carry an
        # x == cutoff endpoint with DIFFERENT fitted values (the jump), so an
        # x-based mask would misassign the right curve's cutoff point.
        vp = self.vars_poly
        n_side = len(vp) // 2
        for segment in (vp.iloc[:n_side], vp.iloc[n_side:]):
            ax.plot(
                segment["rdplot_x"],
                segment["rdplot_y"],
                color=line_color or "red",
                zorder=3,
            )
        ax.axvline(self.cutoff, color="black", linewidth=0.8)
        ax.set_title(title if title is not None else "RD Plot")
        ax.set_xlabel(xlabel if xlabel is not None else "Running variable")
        ax.set_ylabel(ylabel if ylabel is not None else "Outcome")
        return ax


class RDPlot(BaseEstimator):
    """Data-driven RD plot builder (CCT 2015; rdrobust 4.0.0 ``rdplot()``
    parity).

    Parameters mirror R's ``rdplot()`` and keep its defaults - including
    ``kernel="uniform"`` and ``p=4``, which deliberately differ from
    :class:`RegressionDiscontinuity`'s estimation defaults (triangular, p=1):
    the global fit is a descriptive overlay, not the local RD estimator.

    Parameters
    ----------
    cutoff : float, default 0.0
        Threshold in the running variable.
    p : int, default 4
        Order of the global polynomial fits (R accepts any integer >= 0).
    nbins : int or (int, int), optional
        Manual number of bins per side; overrides the data-driven selector.
        Any positive count is accepted (R permissiveness); note that very
        large bin counts allocate proportional jump grids and, with
        covariates, a dense per-side bin-dummy matrix (documented
        performance seam).
    binselect : str, default "esmv"
        One of ``es, espr, esmv, esmvpr, qs, qspr, qsmv, qsmvpr`` - partition
        scheme (evenly/quantile spaced) x target (IMSE-optimal / mimicking
        variance) x variance estimator (spacings / polynomial regression).
        The spacings variants assume a continuously distributed OUTCOME
        (CCT 2015 Theorems 3-4); heavy outcome ties trigger a warning
        recommending the ``*pr`` sibling. QS cutpoints follow R: ``quantile``
        type-7 interpolation, not the paper's inf-form empirical inverse
        (documented deviation-from-paper inherited from rdrobust).
    scale : float or (float, float), optional
        Multiplies the selected number of bins (undersmoothing knob). A
        fractional product is resolved with the ceiling of CCT 2015
        Equation 2. Deviation from R: rdrobust 4.0.0 crashes on fractional
        products (vector-indexing accident) and rejects a length-2 scale
        outright on R >= 4.2 (vectorized-if error); both surfaces work here.
    kernel : str, default "uniform"
        Weighting for the global polynomial fit within ``h``.
    h : float or (float, float), optional
        Window for the global fit; default is each side's full range.
    support : (float, float), optional
        Extended support for bin construction; only WIDENS the observed
        range (R semantics).
    masspoints : str, default "adjust"
        ``"check"`` warns when either side has >= 20% duplicate running
        values; ``"adjust"`` additionally switches a spacings ``binselect``
        to its polynomial-regression sibling; ``"off"`` disables detection.
    ci : float, optional
        Confidence level (percent) for the per-bin intervals drawn by
        :meth:`RDPlotResult.plot`. CI columns are ALWAYS computed (at 95 when
        ``ci`` is None, matching R); ``ci`` only sets the level and turns the
        error bars on.
    covs_drop : bool, default True
        Drop redundant covariates (R's pipeline: stable name-length sort +
        LINPACK-style QR check) with a warning naming the dropped columns;
        ``False`` raises a deterministic error on collinear covariates.
    """

    def __init__(
        self,
        cutoff: float = 0.0,
        p: int = 4,
        nbins: Union[None, int, Tuple[int, int]] = None,
        binselect: str = "esmv",
        scale: Union[None, float, Tuple[float, float]] = None,
        kernel: str = "uniform",
        h: Union[None, float, Tuple[float, float]] = None,
        support: Optional[Tuple[float, float]] = None,
        masspoints: str = "adjust",
        ci: Optional[float] = None,
        covs_drop: bool = True,
    ):
        self.cutoff = cutoff
        self.p = p
        self.nbins = nbins
        self.binselect = binselect
        self.scale = scale
        self.kernel = kernel
        self.h = h
        self.support = support
        self.masspoints = masspoints
        self.ci = ci
        self.covs_drop = covs_drop
        self._validate_constructor_args()

    # ------------------------------------------------------------------
    # Configuration plumbing (sklearn-like)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_real_scalar(val: Any) -> bool:
        return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
            val, (bool, np.bool_)
        )

    @staticmethod
    def _is_int_scalar(val: Any) -> bool:
        return isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_))

    @classmethod
    def _check_pair(
        cls,
        name: str,
        value: Any,
        *,
        integer: bool = False,
        positive: bool = True,
    ) -> None:
        """Validate R's scalar-or-length-2 numeric arguments fail-closed."""
        items = list(value) if isinstance(value, (tuple, list)) else [value]
        if isinstance(value, (tuple, list)) and len(items) != 2:
            raise ValueError(f"{name} must be a scalar or a (left, right) pair; got {value!r}.")
        for item in items:
            ok = cls._is_int_scalar(item) if integer else cls._is_real_scalar(item)
            ok = ok and np.isfinite(item) and (item > 0 or not positive)
            if not ok:
                kind = "an integer" if integer else "a finite number"
                raise ValueError(f"{name} entries must be {kind} > 0; got {item!r}.")

    def _validate_constructor_args(self) -> None:
        if not (self._is_real_scalar(self.cutoff) and np.isfinite(self.cutoff)):
            raise ValueError(f"cutoff must be finite; got {self.cutoff!r}.")
        # R validates p >= 0 and integrality (rdplot.R:173-188); it imposes
        # no upper bound (extreme orders fail later in the linear algebra).
        if not (self._is_int_scalar(self.p) and self.p >= 0):
            raise ValueError(f"p must be an integer >= 0; got {self.p!r}.")
        if self.nbins is not None:
            self._check_pair("nbins", self.nbins, integer=True)
        if self.binselect not in _BINSELECT_INFO:
            raise ValueError(
                f"binselect must be one of {sorted(_BINSELECT_INFO)}; got {self.binselect!r}."
            )
        if self.scale is not None:
            self._check_pair("scale", self.scale)
        _normalize_kernel(self.kernel)  # raises on unknown kernel
        if self.h is not None:
            self._check_pair("h", self.h)
        if self.support is not None:
            if not (
                isinstance(self.support, (tuple, list))
                and len(self.support) == 2
                and all(self._is_real_scalar(v) and np.isfinite(v) for v in self.support)
                and float(self.support[0]) < float(self.support[1])
            ):
                raise ValueError(
                    "support must be a (lower, upper) pair of finite numbers "
                    f"with lower < upper; got {self.support!r}."
                )
        if self.masspoints not in ("check", "adjust", "off"):
            raise ValueError(
                f"masspoints must be 'check', 'adjust', or 'off'; got {self.masspoints!r}."
            )
        if self.ci is not None and not (self._is_real_scalar(self.ci) and 0 < self.ci < 100):
            raise ValueError(f"ci must be a confidence level in (0, 100); got {self.ci!r}.")
        if not isinstance(self.covs_drop, (bool, np.bool_)):
            raise ValueError(f"covs_drop must be a bool; got {self.covs_drop!r}.")

    # get_params/set_params come from BaseEstimator.

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        outcome: Any = NOT_SUPPLIED,
        running: Any = NOT_SUPPLIED,
        covariates: Optional[List[str]] = None,
        outcome_col: Any = NOT_SUPPLIED,
        running_col: Any = NOT_SUPPLIED,
    ) -> RDPlotResult:
        """Build the RD plot quantities.

        Parameters
        ----------
        data : pd.DataFrame
            Cross-sectional data.
        outcome, running : str
            Column names of the outcome and the running variable.
        covariates : list of str, optional
            Covariate-adjusted plot per rdplot's ``covs=`` (global fit
            partials the covariates out with a pooled-across-sides gamma;
            bin means use the fitted values of per-side regressions of the
            outcome on covariates and bin indicators, R's
            ``lm(y ~ z + factor(bin))`` with ``covs_eval="mean"``). Note
            R's caveat: covariate adjustment is meant for plotting alongside
            covariate-adjusted rdrobust ESTIMATES; the adjusted global fit
            may not be visually compatible with unadjusted binned means.
            ``covariate_coefficients`` is keyed in R's processing order:
            name-length-sorted under ``covs_drop=True`` (the sort decides
            which of a collinear set survives), the user-given order under
            ``covs_drop=False`` (R sorts only inside its drop pipeline;
            same contract as ``RegressionDiscontinuity``). Per-name values
            are identical either way on full-rank covariates.

        outcome_col, running_col : str, optional
            Deprecated aliases for ``outcome`` / ``running`` (rows
            M-088/M-089); each warns with ``FutureWarning`` and will be
            removed in 4.0.

        Returns
        -------
        RDPlotResult
        """
        qualname = "RDPlot.fit"
        outcome = resolve_renamed_kwarg(
            qualname,
            "outcome_col",
            outcome_col,
            "outcome",
            outcome,
            default=NOT_SUPPLIED,
        )
        require_arg(qualname, "outcome", outcome)
        running = resolve_renamed_kwarg(
            qualname,
            "running_col",
            running_col,
            "running",
            running,
            default=NOT_SUPPLIED,
        )
        require_arg(qualname, "running", running)
        # Body-local names; the public parameters are outcome/running
        # (M-088/M-089).
        outcome_col = outcome
        running_col = running
        if covariates is not None:
            if isinstance(covariates, str):
                # A bare string would iterate characters; fail closed.
                raise ValueError(f"covariates must be a list of column names; got {covariates!r}.")
            # Materialize before validating (a generator would be consumed
            # by the checks below and silently collapse to empty).
            covariates = list(covariates)
            if not covariates:
                raise ValueError("covariates must be a non-empty list of column names or None.")
            if not all(isinstance(name, str) for name in covariates):
                raise ValueError(f"covariates must be a list of column names; got {covariates!r}.")
            # Duplicate names corrupt the name-keyed coefficient dict and
            # the index-based column lookups (same contract as the
            # estimator's validate_covariate_names).
            validate_covariate_names(
                covariates,
                [outcome_col, running_col],
                estimator="RDPlot",
            )
        for col in [outcome_col, running_col] + list(covariates or []):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        c = float(self.cutoff)
        y_all = np.asarray(pd.to_numeric(data[outcome_col], errors="coerce"), dtype=np.float64)
        x_all = np.asarray(pd.to_numeric(data[running_col], errors="coerce"), dtype=np.float64)
        ok = np.isfinite(x_all) & np.isfinite(y_all)
        z_all: Optional[np.ndarray] = None
        if covariates is not None:
            z_all = np.column_stack(
                [
                    np.asarray(pd.to_numeric(data[col], errors="coerce"), dtype=np.float64)
                    for col in covariates
                ]
            )
            ok &= np.all(np.isfinite(z_all), axis=1)
        n_dropped = int(x_all.shape[0] - np.sum(ok))
        if n_dropped > 0:
            # Deviation from R (which drops silently via complete.cases);
            # same warning contract as RegressionDiscontinuity.fit().
            dropped_cols = f"{outcome_col!r}/{running_col!r}"
            if covariates is not None:
                dropped_cols += "/covariates"
            warnings.warn(
                f"Dropping {n_dropped} row(s) with missing or non-numeric "
                f"values in {dropped_cols}.",
                UserWarning,
                stacklevel=2,
            )
        x, y = x_all[ok], y_all[ok]
        z = z_all[ok] if z_all is not None else None
        n = x.size
        if n == 0:
            raise ValueError("No complete observations after dropping missing values.")

        # --- Covariate redundancy pipeline (rdplot.R:144-159; identical to
        # the estimator's: stable name-length sort decides which of a
        # collinear set survives, QR check on x-sorted rows) ---
        model_covariates: Optional[List[str]] = None
        covariates_dropped: Optional[List[str]] = None
        if covariates is not None and z is not None and self.covs_drop:
            model_covariates = sorted(list(covariates), key=len)
            z = np.column_stack([z[:, covariates.index(name)] for name in model_covariates])
            keep_idx, rank = covs_drop_fun(z[np.argsort(x, kind="stable")])
            if rank == 0:
                raise ValueError(
                    "All covariates are numerically zero (rank-0 covariate "
                    "matrix); remove the covariates instead."
                )
            covariates_dropped = []
            if rank < len(model_covariates):
                covariates_dropped = [
                    name
                    for i, name in enumerate(model_covariates)
                    if i not in set(keep_idx.tolist())
                ]
                warnings.warn(
                    "Multicollinearity detected in covariates: dropped "
                    f"redundant column(s) {covariates_dropped} "
                    "(covs_drop=True; set covs_drop=False for a strict "
                    "error instead).",
                    UserWarning,
                    stacklevel=2,
                )
                model_covariates = [
                    name for i, name in enumerate(model_covariates) if i in set(keep_idx.tolist())
                ]
                z = z[:, keep_idx]
        elif covariates is not None:
            model_covariates = list(covariates)

        x_min, x_max = float(np.min(x)), float(np.max(x))
        if self.support is not None:  # support only WIDENS (rdplot.R:62-67)
            if float(self.support[0]) < x_min:
                x_min = float(self.support[0])
            if float(self.support[1]) > x_max:
                x_max = float(self.support[1])
        if c <= x_min or c >= x_max:
            raise ValueError(
                f"cutoff must lie strictly inside the range of '{running_col}' "
                f"([{x_min}, {x_max}]); got {c}."
            )
        if n < 20:
            # R stops outright (rdplot.R:190-193) - no small-sample fallback.
            raise ValueError(f"Not enough observations to perform bin calculations (n={n} < 20).")

        ind_l, ind_r = x < c, x >= c
        x_l, y_l, x_r, y_r = x[ind_l], y[ind_l], x[ind_r], y[ind_r]
        n_l, n_r = int(x_l.size), int(x_r.size)
        # Defensive guard beyond R: with n >= 20 one side can still be
        # near-empty, where R degenerates opaquely (NA variance poisoning an
        # `if`). Fail with the side named instead.
        if n_l < 2 or n_r < 2:
            side = "below" if n_l < 2 else "above"
            raise ValueError(
                f"Need at least 2 observations on each side of the cutoff; "
                f"found {min(n_l, n_r)} {side} cutoff={c}."
            )
        # A side whose running variable has a SINGLE support point cannot be
        # partitioned: every spacing is zero, the variance components
        # collapse to 0, J becomes Inf, and the jump grid divides by zero -
        # R crashes opaquely inside seq() there. Defensive clear error.
        if np.unique(x_l).size < 2 or np.unique(x_r).size < 2:
            side = "below" if np.unique(x_l).size < 2 else "above"
            raise ValueError(
                f"The running variable has a single distinct value {side} "
                f"cutoff={c}; at least 2 distinct values per side are "
                "required to construct bins."
            )
        range_l, range_r = c - x_min, x_max - c

        scale_l, scale_r = _resolve_pair(self.scale, 1.0)
        h_l, h_r = _resolve_pair(self.h, 0.0)
        if self.h is None:
            h_l, h_r = range_l, range_r

        ci_requested = self.ci is not None
        ci_level = float(self.ci) if self.ci is not None else 95.0

        binselect = self.binselect
        # --- Mass points (rdplot.R:116-137) ---
        if self.masspoints in ("check", "adjust"):
            m_l = np.unique(x_l).size
            m_r = np.unique(x_r).size
            if (1 - m_l / n_l) >= 0.2 or (1 - m_r / n_r) >= 0.2:
                hint = (
                    " Try masspoints='adjust'."
                    if self.masspoints == "check"
                    else f" binselect switched to '{_MASSPOINTS_REMAP.get(binselect, binselect)}'"
                    " (polynomial-regression variance estimators; spacings"
                    " degenerate under ties)."
                )
                warnings.warn(
                    "Mass points detected in the running variable." + hint,
                    UserWarning,
                    stacklevel=2,
                )
                if self.masspoints == "adjust":
                    binselect = _MASSPOINTS_REMAP.get(binselect, binselect)

        # Defensive warning beyond R (which runs spacings selectors on any
        # outcome): the spacings/concomitant selectors assume a CONTINUOUSLY
        # distributed outcome (CCT 2015 Theorems 3-4; Remarks 1-2 route
        # noncontinuous outcomes to the series estimators). Heavy outcome
        # ties - the masspoints criterion applied to y - invalidate them;
        # warn and recommend the *pr sibling. No auto-remap: R does not
        # remap, and the numbers must stay golden-comparable.
        if _BINSELECT_INFO[binselect][1] == "hat":
            ties_l = 1 - np.unique(y_l).size / n_l
            ties_r = 1 - np.unique(y_r).size / n_r
            if ties_l >= 0.2 or ties_r >= 0.2:
                warnings.warn(
                    "Heavy ties detected in the outcome variable (possibly "
                    "discrete/binary); the spacings-based bin selectors "
                    "assume a continuously distributed outcome (CCT 2015 "
                    "Theorems 3-4). Consider binselect="
                    f"'{_MASSPOINTS_REMAP.get(binselect, binselect)}' "
                    "(polynomial-regression variance estimators, valid for "
                    "noncontinuous outcomes).",
                    UserWarning,
                    stacklevel=2,
                )

        # ------------------------------------------------------------------
        # Global polynomial curve of order p (rdplot.R:206-270)
        # ------------------------------------------------------------------
        p = int(self.p)
        W_h_l = rdrobust_kweight(x_l, c, h_l, self.kernel)
        W_h_r = rdrobust_kweight(x_r, c, h_r, self.kernel)
        n_h_l = int(np.sum(W_h_l > 0))
        n_h_r = int(np.sum(W_h_r > 0))
        if n_h_l == 0 or n_h_r == 0:
            # Defensive guard beyond R: a manual h smaller than the nearest
            # observation's distance zeroes every kernel weight, and R
            # silently fits a zero-weight design (ginv of a zero Gram),
            # returning an all-zero "curve" that used no data.
            side = "below" if n_h_l == 0 else "above"
            raise ValueError(
                f"h={self.h!r} leaves zero effective observations {side} the "
                f"cutoff for the global polynomial fit; increase h (default: "
                "each side's full range)."
            )

        R_p_l = _pow_outer(x_l - c, p)
        R_p_r = _pow_outer(x_r - c, p)
        invG_p_l = qrXXinv(np.sqrt(W_h_l)[:, None] * R_p_l)
        invG_p_r = qrXXinv(np.sqrt(W_h_r)[:, None] * R_p_r)

        gamma_p: Optional[np.ndarray] = None
        covariate_coefficients: Optional[Dict[str, float]] = None
        if z is None:
            gamma_p1_l = invG_p_l @ (R_p_l * W_h_l[:, None]).T @ y_l
            gamma_p1_r = invG_p_r @ (R_p_r * W_h_r[:, None]).T @ y_r
        else:
            z_l, z_r = z[ind_l], z[ind_r]
            D_l = np.column_stack([y_l, z_l])
            D_r = np.column_stack([y_r, z_r])
            U_p_l = (R_p_l * W_h_l[:, None]).T @ D_l
            U_p_r = (R_p_r * W_h_r[:, None]).T @ D_r
            beta_p_l = invG_p_l @ U_p_l
            beta_p_r = invG_p_r @ U_p_r
            ZWD_p_l = (z_l * W_h_l[:, None]).T @ D_l
            ZWD_p_r = (z_r * W_h_r[:, None]).T @ D_r
            colsZ = slice(1, D_l.shape[1])
            UiGU_l = U_p_l[:, colsZ].T @ (invG_p_l @ U_p_l)
            UiGU_r = U_p_r[:, colsZ].T @ (invG_p_r @ U_p_r)
            ZWZ_p = (ZWD_p_l[:, colsZ] - UiGU_l[:, colsZ]) + (ZWD_p_r[:, colsZ] - UiGU_r[:, colsZ])
            ZWY_p = (ZWD_p_l[:, :1] - UiGU_l[:, :1]) + (ZWD_p_r[:, :1] - UiGU_r[:, :1])
            diag_pre = np.diag(ZWD_p_l[:, colsZ]) + np.diag(ZWD_p_r[:, colsZ])
            gamma_p, covs_excluded, covs_set_degenerate = _covs_gamma(
                ZWZ_p, ZWY_p, diag_pre, bool(self.covs_drop)
            )
            assert model_covariates is not None
            if covs_excluded.any() or covs_set_degenerate:
                # Same guarded Deviation-from-R as the estimator: R's
                # ginv(tol=1e-20) inverts noise singular values on exactly-
                # degenerate partialled systems, making gamma platform-noise.
                parts = []
                if covs_excluded.any():
                    excluded_names = [model_covariates[i] for i in np.flatnonzero(covs_excluded)]
                    parts.append(
                        f"covariate(s) {excluded_names} are numerically "
                        "collinear with the polynomial design (e.g. constant) "
                        "and were excluded from the adjustment"
                    )
                if covs_set_degenerate:
                    parts.append(
                        "the covariate set is numerically rank-deficient "
                        "after partialling; a stabilized pseudo-inverse cut "
                        "was used - consider dropping a reference category"
                    )
                warnings.warn(
                    "Degenerate covariate adjustment: " + "; ".join(parts) + ".",
                    UserWarning,
                    stacklevel=2,
                )
            s_Y = np.concatenate([[1.0], -gamma_p[:, 0]])
            gamma_p1_l = beta_p_l @ s_Y
            gamma_p1_r = beta_p_r @ s_Y
            covariate_coefficients = {
                name: float(gamma_p[i, 0]) for i, name in enumerate(model_covariates)
            }

        nplot = 500
        x_plot_l = np.linspace(c - h_l, c, nplot)
        x_plot_r = np.linspace(c, c + h_r, nplot)
        y_hat_l = _pow_outer(x_plot_l - c, p) @ gamma_p1_l
        y_hat_r = _pow_outer(x_plot_r - c, p) @ gamma_p1_r
        if z is not None and gamma_p is not None:
            # covs_eval="mean" (rdplot.R:266-270): a single scalar shift.
            gammaZ = float(np.mean(z, axis=0) @ gamma_p[:, 0])
            y_hat_l = y_hat_l + gammaZ
            y_hat_r = y_hat_r + gammaZ

        # ------------------------------------------------------------------
        # Bin-count selectors (order-k global fits; rdplot.R:276-388)
        # ------------------------------------------------------------------
        # k=4 with a 4 -> 3 -> 2 fallback ladder that fires ONLY when
        # qrXXinv itself raises: R's qrXXinv absorbs finite singular Grams
        # via an internal ginv fallback at UNCHANGED k, so the try-error
        # ladder triggers on non-finite Grams (raw x**8 overflow) or SVD
        # failure. scipy raises ValueError on non-finite input, numpy
        # LinAlgError on SVD failure - catch both.
        def _selector_grams() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            for k_try in (4, 3, 2):
                rk_l_try = _pow_outer(x_l, k_try)
                rk_r_try = _pow_outer(x_r, k_try)
                try:
                    return (
                        k_try,
                        rk_l_try,
                        rk_r_try,
                        qrXXinv(rk_l_try),
                        qrXXinv(rk_r_try),
                    )
                except (ValueError, np.linalg.LinAlgError):
                    if k_try == 2:
                        raise
            raise AssertionError("unreachable")  # pragma: no cover

        k, rk_l, rk_r, invG_k_l, invG_k_r = _selector_grams()
        gamma_k1_l = invG_k_l @ rk_l.T @ y_l
        gamma_k2_l = invG_k_l @ rk_l.T @ y_l**2
        gamma_k1_r = invG_k_r @ rk_r.T @ y_r
        gamma_k2_r = invG_k_r @ rk_r.T @ y_r**2

        ord_l = np.argsort(x_l, kind="stable")
        ord_r = np.argsort(x_r, kind="stable")
        x_i_l, y_i_l = x_l[ord_l], y_l[ord_l]
        x_i_r, y_i_r = x_r[ord_r], y_r[ord_r]
        dxi_l, dyi_l = np.diff(x_i_l), np.diff(y_i_l)
        dxi_r, dyi_r = np.diff(x_i_r), np.diff(y_i_r)
        x_bar_i_l = (x_i_l[1:] + x_i_l[:-1]) / 2
        x_bar_i_r = (x_i_r[1:] + x_i_r[:-1]) / 2

        rk_i_l = _pow_outer(x_bar_i_l, k)
        rk_i_r = _pow_outer(x_bar_i_r, k)
        j_arange = np.arange(1, k + 1)
        drk_l = _pow_outer(x_l, k - 1) * j_arange
        drk_r = _pow_outer(x_r, k - 1) * j_arange
        drk_i_l = _pow_outer(x_bar_i_l, k - 1) * j_arange
        drk_i_r = _pow_outer(x_bar_i_r, k - 1) * j_arange

        # R's two-pass var() is EXACTLY zero on a constant vector; numpy's
        # single-pass mean leaves ~1e-32 roundoff for constants like 0.7,
        # which would skip the var==0 rescue below and crash the bin grid on
        # a zero jump (J_MV = Inf). Route through the port's exact-constancy
        # helper (_var0, the #686-ported R semantics) so constant sides give
        # var_y = 0.0 exactly, reproducing R's rescue and its 0/0 -> NaN
        # selector echoes.
        var_y_l = 0.0 if _var0(y_l) else float(np.var(y_l, ddof=1))
        var_y_r = 0.0 if _var0(y_r) else float(np.var(y_r, ddof=1))

        mu0_i_l = rk_i_l @ gamma_k1_l
        mu0_i_r = rk_i_r @ gamma_k1_r
        mu2_i_l = rk_i_l @ gamma_k2_l
        mu2_i_r = rk_i_r @ gamma_k2_r
        mu0_l = rk_l @ gamma_k1_l
        mu0_r = rk_r @ gamma_k1_r
        mu2_l = rk_l @ gamma_k2_l
        mu2_r = rk_r @ gamma_k2_r
        mu1_l = drk_l @ gamma_k1_l[1:]
        mu1_r = drk_r @ gamma_k1_r[1:]
        mu1_i_l = drk_i_l @ gamma_k1_l[1:]
        mu1_i_r = drk_i_r @ gamma_k1_r[1:]

        # Negative variance-function values are floored to the side's sample
        # variance (rdplot.R:356-371; an R-source fact absent from the paper).
        s2bar_l = mu2_i_l - mu0_i_l**2
        s2bar_r = mu2_i_r - mu0_i_r**2
        s2bar_l[s2bar_l < 0] = var_y_l
        s2bar_r[s2bar_r < 0] = var_y_r
        s2_l = mu2_l - mu0_l**2
        s2_r = mu2_r - mu0_r**2
        s2_l[s2_l < 0] = var_y_l
        s2_r[s2_r < 0] = var_y_r

        B_es = (
            (range_l**2 / (12 * n)) * float(np.sum(mu1_l**2)),
            (range_r**2 / (12 * n)) * float(np.sum(mu1_r**2)),
        )
        V_es_hat = (
            (0.5 / range_l) * float(np.sum(dxi_l * dyi_l**2)),
            (0.5 / range_r) * float(np.sum(dxi_r * dyi_r**2)),
        )
        V_es_chk = (
            (1 / range_l) * float(np.sum(dxi_l * s2bar_l)),
            (1 / range_r) * float(np.sum(dxi_r * s2bar_r)),
        )
        B_qs = (
            (n_l**2 / (24 * n)) * float(np.sum(dxi_l**2 * mu1_i_l**2)),
            (n_r**2 / (24 * n)) * float(np.sum(dxi_r**2 * mu1_i_r**2)),
        )
        V_qs_hat = (
            (1 / (2 * n_l)) * float(np.sum(dyi_l**2)),
            (1 / (2 * n_r)) * float(np.sum(dyi_r**2)),
        )
        V_qs_chk = (
            (1 / n_l) * float(np.sum(s2_l)),
            (1 / n_r) * float(np.sum(s2_r)),
        )

        # numpy scalar arithmetic so a zero-variance side propagates Inf/NaN
        # through the selectors exactly like R (0/0 -> NaN, x/0 -> Inf, no
        # exception); the var==0 rescue below then pins J to 1 as R does.
        def _J_imse(B: _PairF, V: _PairF) -> _PairF:
            # J.fun (functions.R:588)
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    float(np.ceil((((2 * np.float64(B[0])) / V[0]) * n) ** (1 / 3))),
                    float(np.ceil((((2 * np.float64(B[1])) / V[1]) * n) ** (1 / 3))),
                )

        def _J_mv(V: _PairF) -> _PairF:
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    float(np.ceil((np.float64(var_y_l) / V[0]) * n / math.log(n) ** 2)),
                    float(np.ceil((np.float64(var_y_r) / V[1]) * n / math.log(n) ** 2)),
                )

        meth, family, target, binselect_type = _BINSELECT_INFO[binselect]
        V_pair = {
            ("es", "hat"): V_es_hat,
            ("es", "chk"): V_es_chk,
            ("qs", "hat"): V_qs_hat,
            ("qs", "chk"): V_qs_chk,
        }[(meth, family)]
        B_pair = B_es if meth == "es" else B_qs
        J_IMSE = _J_imse(B_pair, V_pair)
        J_MV = _J_mv(V_pair)
        J_star_orig = J_IMSE if target == "imse" else J_MV

        # --- J resolution (rdplot.R:449-469) ---
        # Deviation from R: for a fractional scale*J product, R 4.0.0
        # propagates the unrounded value into 1:J / vector-indexing
        # arithmetic and CRASHES assembling vars_bins (verified live:
        # "arguments imply differing number of rows"); a length-2 scale dies
        # even earlier on R >= 4.2 (vectorized-if error at rdplot.R:178).
        # CCT 2015 Equation 2 places the ceiling over the RESCALED product
        # (J = ceil(omega * (2B/V n)^(1/3))), so the paper-faithful integer
        # partition is used here instead of replicating an accident.
        # np.ceil keeps a NaN/Inf selector value (zero-variance side, rescued
        # just below) flowing instead of raising like math.ceil would. The
        # max(1, .) clamp keeps an ultra-small positive scale (product below
        # the 1e-12 integer-product epsilon) from ceiling to zero bins - the
        # Eq 2 partition is at least one bin (NaN passes through max()
        # untouched for the variance rescue).
        with np.errstate(invalid="ignore"):
            J_star_l: float = float(np.maximum(1.0, np.ceil(scale_l * J_star_orig[0] - 1e-12)))
            J_star_r: float = float(np.maximum(1.0, np.ceil(scale_r * J_star_orig[1] - 1e-12)))
        if self.nbins is not None:
            nb_l, nb_r = _resolve_pair(self.nbins, 0)
            J_star_l, J_star_r = float(nb_l), float(nb_r)
            # R keeps this label even when binselect selected quantile
            # spacing - the BINS below still follow `meth` (label-only quirk).
            binselect_type = "manually evenly spaced"
        if var_y_l == 0:
            J_star_l = 1.0
            warnings.warn(
                "Not enough variability in the outcome variable below the threshold",
                UserWarning,
                stacklevel=2,
            )
        if var_y_r == 0:
            J_star_r = 1.0
            warnings.warn(
                "Not enough variability in the outcome variable above the threshold",
                UserWarning,
                stacklevel=2,
            )
        rscale = (J_star_l / J_IMSE[0], J_star_r / J_IMSE[1])

        # --- Bin construction (rdplot.R:471-508) ---
        jump_l = range_l / J_star_l
        jump_r = range_r / J_star_r
        if meth == "es":
            jumps_l = _r_seq_by(x_min, c, jump_l)
            jumps_r = _r_seq_by(c, x_max, jump_r)
        else:
            probs_l = _r_seq_by(0.0, 1.0, 1 / J_star_l)
            probs_r = _r_seq_by(0.0, 1.0, 1 / J_star_r)
            # R type-7 quantile == numpy's default linear interpolation (the
            # default has been type-7 in every numpy release; the explicit
            # method="linear" kwarg only exists from numpy 1.22 and the
            # library floor is 1.20, so the default is relied on and the
            # type-7 convention is locked by a hand-computed test).
            jumps_l = np.quantile(x_l, probs_l)
            jumps_r = np.quantile(x_r, probs_r)

        bin_x_l = _find_interval(x_l, jumps_l) - J_star_l - 1
        bin_x_r = _find_interval(x_r, jumps_r).astype(float)

        # Number of full bins actually delimited by the jump grid (1:J with
        # R's fractional truncation).
        t_l = int(J_star_l)
        t_r = int(J_star_r)
        bin_length_l = jumps_l[1 : t_l + 1] - jumps_l[:t_l]
        bin_length_r = jumps_r[1 : t_r + 1] - jumps_r[:t_r]
        bin_avg = (float(np.mean(bin_length_l)), float(np.mean(bin_length_r)))
        bin_med = (float(np.median(bin_length_l)), float(np.median(bin_length_r)))

        # Covariate-adjusted bin means (rdplot.R:492-499): per-side fitted
        # values of y ~ 1 + z + bin dummies, then bin means of the fits.
        # Fitted values are the projection onto the column space, so lstsq's
        # min-norm solution reproduces R's pivoted-QR lm() fits exactly even
        # under rank deficiency (this is a projection for display, not an
        # inference path - solve_ols's rank guards are not wanted here).
        y_for_bins_l, y_for_bins_r = y_l, y_r
        if z is not None:
            for side_bins, side_z, side_y, setter in (
                (bin_x_l, z[ind_l], y_l, "l"),
                (bin_x_r, z[ind_r], y_r, "r"),
            ):
                levels, codes = np.unique(side_bins, return_inverse=True)
                dummies = np.zeros((side_y.size, levels.size - 1))
                nonref = codes >= 1
                dummies[np.arange(side_y.size)[nonref], codes[nonref] - 1] = 1.0
                X = np.column_stack([np.ones(side_y.size), side_z, dummies])
                coefs, *_ = np.linalg.lstsq(X, side_y, rcond=None)
                fitted = X @ coefs
                if setter == "l":
                    y_for_bins_l = fitted
                else:
                    y_for_bins_r = fitted

        def _side_rows(
            bins: np.ndarray,
            xs: np.ndarray,
            ys_mean: np.ndarray,
            ys_sd: np.ndarray,
            jumps: np.ndarray,
            t_full: int,
            offset: float,
            left: bool,
        ) -> List[Dict[str, float]]:
            present = np.sort(np.unique(bins))
            # R-quirk (rdplot.R:503-508 vs 551-552): mean_bin is indexed by
            # the TRUE grid slot (b + J + 1 on the left, b on the right),
            # but min_bin/max_bin on the LEFT side are indexed by
            # `rev(-bins)` - the ASCENDING-sorted |bin| ids, i.e. the
            # occupied slots REFLECTED around the side's center. Identical
            # when no left bin is empty; with empty left bins R's own
            # min/max edges disagree with mean_bin (verified live: 10 of 20
            # senate/esmvpr left rows have mean_bin != (min+max)/2).
            # Replicated exactly; documented in REGISTRY.
            edge_slots = np.sort(np.abs(present)) if left else present
            rows = []
            for i, b in enumerate(present):
                m = bins == b
                N_bin = int(np.sum(m))
                mean_y = float(np.mean(ys_mean[m]))
                sd = float(np.std(ys_sd[m], ddof=1)) if N_bin > 1 else 0.0
                se = sd / math.sqrt(N_bin)
                quant = -float(stats.t.ppf((1 - ci_level / 100) / 2, max(N_bin - 1, 1)))
                # 1-based slots; out-of-range (grid undershoot overflow bin)
                # gives NA in R and NaN here. int() mirrors R's truncating
                # numeric indexing.
                true_slot = int(b + offset)
                edge_slot = int(edge_slots[i])
                if 1 <= true_slot <= t_full:
                    mean_bin = float(jumps[true_slot - 1] + jumps[true_slot]) / 2
                else:
                    mean_bin = float("nan")
                if 1 <= edge_slot <= t_full:
                    min_bin = float(jumps[edge_slot - 1])
                    max_bin = float(jumps[edge_slot])
                else:
                    min_bin = float("nan")
                    max_bin = float("nan")
                rows.append(
                    {
                        "rdplot_mean_bin": mean_bin,
                        "rdplot_mean_x": float(np.mean(xs[m])),
                        "rdplot_mean_y": mean_y,
                        "rdplot_min_bin": min_bin,
                        "rdplot_max_bin": max_bin,
                        "rdplot_se_y": se,
                        "rdplot_N": N_bin,
                        "rdplot_ci_l": mean_y - quant * se,
                        "rdplot_ci_r": mean_y + quant * se,
                    }
                )
            return rows

        rows = _side_rows(bin_x_l, x_l, y_for_bins_l, y_l, jumps_l, t_l, J_star_l + 1, True)
        rows += _side_rows(bin_x_r, x_r, y_for_bins_r, y_r, jumps_r, t_r, 0.0, False)
        vars_bins = pd.DataFrame(rows)
        vars_poly = pd.DataFrame(
            {
                "rdplot_x": np.concatenate([x_plot_l, x_plot_r]),
                "rdplot_y": np.concatenate([y_hat_l, y_hat_r]),
            }
        )
        coef = pd.DataFrame(
            {"Left": np.asarray(gamma_p1_l).ravel(), "Right": np.asarray(gamma_p1_r).ravel()},
            index=pd.Index(range(p + 1), name="power"),
        )

        return RDPlotResult(
            coef=coef,
            vars_bins=vars_bins,
            vars_poly=vars_poly,
            J=(J_star_l, J_star_r),
            J_IMSE=J_IMSE,
            J_MV=J_MV,
            scale=(scale_l, scale_r),
            rscale=rscale,
            bin_avg=bin_avg,
            bin_med=bin_med,
            p=p,
            cutoff=c,
            h=(h_l, h_r),
            N=(n_l, n_r),
            N_h=(n_h_l, n_h_r),
            binselect=binselect_type,
            kernel_type=_KERNEL_TYPE_LABEL[_normalize_kernel(self.kernel)],
            ci_level=ci_level,
            ci_requested=ci_requested,
            covariate_coefficients=covariate_coefficients,
            covariates_dropped=covariates_dropped,
        )
