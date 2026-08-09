"""Results container for the ChangesInChanges (CiC) and QDiD estimators."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._deprecation import deprecated_field_property
from diff_diff.results_base import BaseResults

_ESTIMATOR_TITLES = {
    "cic": "Changes-in-Changes (Athey & Imbens 2006) Results",
    "qdid": "Quantile Difference-in-Differences (QDiD) Results",
}


@dataclass
class ChangesInChangesResults(BaseResults):
    """Results for :class:`~diff_diff.changes_in_changes.ChangesInChanges` and
    :class:`~diff_diff.changes_in_changes.QDiD`.

    The headline ``att``/``se``/``t_stat``/``p_value``/``conf_int`` fields carry
    the mean effect; ``quantile_effects`` is a DataFrame with one row per
    requested quantile (columns ``quantile``, ``qte``, ``se``, ``t_stat``,
    ``p_value``, ``conf_low``, ``conf_high``). All inference derives from the
    bootstrap (replicate-SD standard errors, symmetric normal-approximation
    intervals at level ``alpha``); with ``n_bootstrap=0`` every inference field
    is NaN.

    ``q_lower``/``q_upper`` bound the point-identified interior quantile range
    for unconditional CiC fits (Athey-Imbens eq. 17; NaN for QDiD and for
    covariate fits, where the unconditional bounds are not the relevant
    objects). ``sup_t_crit`` is the qte package's sup-t critical value for
    uniform bands - computed at a FIXED 95% level regardless of ``alpha`` (qte
    parity); see :meth:`uniform_bands`. ``covariates`` records the covariate
    columns used by the conditional (quantile-regression) fit, or ``None``
    for unconditional fits.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    quantile_effects: pd.DataFrame
    q_lower: float
    q_upper: float
    sup_t_crit: float
    n_obs: int
    cell_sizes: Dict[str, int]
    n_bootstrap: int
    n_bootstrap_valid: int
    panel: bool
    method: str
    quantiles: np.ndarray = field(repr=False)
    alpha: float = 0.05
    covariates: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------
    @property
    def is_significant(self) -> bool:
        """Whether the headline ATT is significant at level ``alpha`` (False on NaN)."""
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for the headline ATT p-value ('' when NaN)."""
        from diff_diff.results import _get_significance_stars

        return "" if not np.isfinite(self.p_value) else _get_significance_stars(self.p_value)

    # ------------------------------------------------------------------
    # uniform bands
    # ------------------------------------------------------------------
    def uniform_bands(self) -> pd.DataFrame:
        """Simultaneous (sup-t) confidence bands over the quantile grid.

        ``qte +/- sup_t_crit * se`` per quantile, using the qte package's
        IQR-scaled sup-t critical value at its hard-coded 95% level - the band
        level does NOT follow ``alpha`` (qte parity). Rows whose ``se`` is NaN
        (no bootstrap, failed replicate gate, or outside the interior range in
        an unconditional CiC fit) get NaN bands.
        """
        qe = self.quantile_effects
        bands = pd.DataFrame(
            {
                "quantile": qe["quantile"],
                "qte": qe["qte"],
                "band_low": qe["qte"] - self.sup_t_crit * qe["se"],
                "band_high": qe["qte"] + self.sup_t_crit * qe["se"],
            }
        )
        return bands

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Flat headline dictionary (ATT inference, ranges, sizes, bootstrap metadata)."""
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "q_lower": self.q_lower,
            "q_upper": self.q_upper,
            "sup_t_crit": self.sup_t_crit,
            "n_obs": self.n_obs,
            "cell_sizes": dict(self.cell_sizes),
            "n_bootstrap": self.n_bootstrap,
            "n_bootstrap_valid": self.n_bootstrap_valid,
            "panel": self.panel,
            "method": self.method,
            # Dual-key 3.9 window (row M-143, the M-094/rdd.py twin): the old key
            # stays until 4.0 so serialized consumers keep working. Both read the
            # SAME attribute - they can never disagree.
            "estimator": self.method,
            "alpha": self.alpha,
            "covariates": list(self.covariates) if self.covariates else None,
            "inference_method": "bootstrap" if self.n_bootstrap > 0 else "none",
        }

    def to_dataframe(self, level: str = "quantiles") -> pd.DataFrame:
        """Return the quantile-effects table or the one-row ATT summary.

        Parameters
        ----------
        level : {"quantiles", "att"}
            ``"quantiles"`` (default) returns a copy of ``quantile_effects``;
            ``"att"`` returns a single-row frame with the headline fields.
        """
        if level == "quantiles":
            return self.quantile_effects.copy()
        if level == "att":
            return pd.DataFrame(
                [
                    {
                        "att": self.att,
                        "se": self.se,
                        "t_stat": self.t_stat,
                        "p_value": self.p_value,
                        "conf_low": self.conf_int[0],
                        "conf_high": self.conf_int[1],
                        "n_obs": self.n_obs,
                    }
                ]
            )
        raise ValueError(f"level must be 'quantiles' or 'att', got '{level}'")

    # ------------------------------------------------------------------
    # text summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Fixed-width text summary: headline ATT block plus the quantile-effects table."""
        from diff_diff.results import _get_significance_stars

        ci_pct = int(round((1 - self.alpha) * 100))
        width = 88
        bar = "=" * width
        dash = "-" * width

        def _fmt(x: Any, nd: int = 4) -> str:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return ""
            return "" if np.isnan(xf) else f"{xf:.{nd}f}"

        mode = "panel (unit block bootstrap)" if self.panel else "repeated cross-section"
        cs = self.cell_sizes
        lines = [
            bar,
            _ESTIMATOR_TITLES.get(self.method, "Distributional DiD Results").center(width),
            bar,
            f"Observations: {self.n_obs}    Mode: {mode}",
            (
                f"Cells: control pre={cs.get('control_pre')}, "
                f"control post={cs.get('control_post')}, "
                f"treated pre={cs.get('treated_pre')}, "
                f"treated post={cs.get('treated_post')}"
            ),
        ]
        if self.covariates:
            lines.append(
                f"Covariates: {', '.join(self.covariates)} (conditional ranks via "
                "per-cell linear quantile regression, 99-tau grid; qte xformla parity)"
            )
        if self.n_bootstrap > 0:
            lines.append(
                f"Inference: bootstrap ({self.n_bootstrap_valid}/{self.n_bootstrap} "
                "valid replicates), replicate-SD SEs, normal-approximation intervals"
            )
        else:
            lines.append("Inference: disabled (n_bootstrap=0); all inference fields are NaN")
        if self.method == "cic" and np.isfinite(self.q_lower) and np.isfinite(self.q_upper):
            lines.append(
                f"Point-identified interior quantile range (eq. 17): "
                f"({self.q_lower:.4f}, {self.q_upper:.4f})"
            )

        stars = "" if np.isnan(self.p_value) else _get_significance_stars(self.p_value)
        lines.extend(
            [
                dash,
                f"{'':>10}  {'Estimate':>10}  {'Std.Err':>10}  {'t':>8}  {'P>|t|':>8}"
                f"  [{ci_pct}% Conf. Int.]",
                dash,
                f"{'ATT':>10}  {_fmt(self.att):>10}  {_fmt(self.se):>10}"
                f"  {_fmt(self.t_stat, 2):>8}  {_fmt(self.p_value, 3):>8}"
                f"  [{_fmt(self.conf_int[0]):>9}, {_fmt(self.conf_int[1]):>9}] {stars}",
                "",
                "Quantile treatment effects:",
                dash,
            ]
        )
        for _, r in self.quantile_effects.iterrows():
            p = r["p_value"]
            row_stars = "" if pd.isna(p) else _get_significance_stars(float(p))
            lines.append(
                f"{r['quantile']:>10.2f}  {_fmt(r['qte']):>10}  {_fmt(r['se']):>10}"
                f"  {_fmt(r['t_stat'], 2):>8}  {_fmt(r['p_value'], 3):>8}"
                f"  [{_fmt(r['conf_low']):>9}, {_fmt(r['conf_high']):>9}] {row_stars}"
            )
        lines.append(bar)
        lines.append("Signif. codes: *** p<0.001, ** p<0.01, * p<0.05")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print :meth:`summary` to stdout."""
        print(self.summary())

    # Row M-143: read-only alias for the pre-3.9 ``estimator`` field name. No
    # annotation, so it stays a descriptor and never becomes a
    # __dataclass_fields__ entry. Being setter-less is deliberate and load-bearing:
    # ``setattr(res, "estimator", ...)`` must fail rather than silently write a
    # shadow attribute the renamed field would not see.
    estimator = deprecated_field_property("ChangesInChangesResults", "estimator", "method")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Migrate pickles created before the ``estimator`` -> ``method`` rename.

        Results pickled before row M-143 stored the tag under ``estimator``;
        rewriting the key on load keeps both ``method`` and the deprecated
        ``estimator`` alias working on old pickles. Dataclass unpickling routes
        through here for CURRENT objects too, so this must stay a pass-through
        in the common case.
        """
        if "estimator" in state and "method" not in state:
            state = dict(state)
            state["method"] = state.pop("estimator")
        self.__dict__.update(state)

    def __repr__(self) -> str:
        att_s = "nan" if np.isnan(self.att) else f"{self.att:.4f}"
        se_s = "nan" if np.isnan(self.se) else f"{self.se:.4f}"
        return (
            "ChangesInChangesResults("
            f"method={self.method!r}, ATT={att_s}, SE={se_s}, "
            f"n_quantiles={len(self.quantile_effects)}, "
            f"panel={self.panel}, n_bootstrap={self.n_bootstrap})"
        )


# QDiD shares the container; the ``method`` field distinguishes the two.
QDiDResults = ChangesInChangesResults
