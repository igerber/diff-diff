from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff._deprecation import warn_deprecated_kwarg
from diff_diff.results_base import BaseResults, _coverage_pct


@dataclass
class LPDiDResults(BaseResults):
    """Results container for the :class:`~diff_diff.lpdid.LPDiD` estimator.

    Holds the per-horizon ``event_study`` table and the ``pooled`` pre/post
    table (each a :class:`pandas.DataFrame` with ``coefficient``, ``se``,
    ``t_stat``, ``p_value``, ``conf_low``, ``conf_high``, ``n_obs``,
    ``n_clusters`` columns). The headline ATT is the pooled ``post`` row.

    ``n_control_units`` counts **never-treated** units only (the library-wide
    field convention, surfaced as "Never-treated units" in ``summary()``); under
    ``control_group="clean"`` the realized control pool at each horizon also
    includes not-yet-treated cohorts, whose per-horizon counts live in the
    ``n_obs`` / ``n_clusters`` columns of the tables.

    ``event_study_df`` and ``pooled_df`` carry inference degrees-of-freedom
    PROVENANCE - the df each row's ``safe_inference`` actually received -
    keyed by horizon and by ``"pre"``/``"post"`` window respectively. Both
    are NaN per entry where that row used normal theory or an undefined df.
    They are metadata only: the native tables above are unchanged, and
    ``event_study_df`` is what the unified event-study surface reads.
    """

    event_study: Optional[pd.DataFrame]
    pooled: Optional[pd.DataFrame]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    pre_window: int
    post_window: int
    control_group: str
    reweight: bool
    no_composition: bool
    pmd: Optional[Union[str, int]]
    alpha: float = 0.05
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    vcov_type: str = "hc1"
    rank_deficient_action: str = "warn"
    covariates: Optional[List[str]] = None
    absorb: Optional[List[str]] = None
    ylags: int = 0
    dylags: int = 0
    non_absorbing: Optional[str] = None
    stabilization_window: Optional[int] = None
    # Survey-design (Taylor-linearization) metadata, set only when fit() received a
    # ``survey_design``. ``survey_metadata`` is a ``SurveyMetadata`` (weight type, Kish
    # effective N, design effect, panel-level n_strata/n_psu, survey d.f.); ``n_strata``
    # / ``n_psu`` echo the panel-level effective design dimensions. ``vcov_type`` is then
    # ``"survey_tsl"`` and ``n_clusters`` is the realized headline (pooled-post) PSU count.
    survey_metadata: Optional[Any] = None
    n_strata: Optional[int] = None
    n_psu: Optional[int] = None
    # event_study_df (spec section 5, row M-092): per-horizon df PROVENANCE -
    # maps each estimated horizon to the df actually passed to its
    # safe_inference (realized ``n_clusters - 1`` on the cluster/RA paths,
    # the per-horizon-sample survey df ``n_PSU - n_strata`` under a survey
    # design; NaN when the row used normal theory or an undefined df). Not
    # derivable from the frames: the survey rule needs the per-sample
    # n_strata and the cluster rule is gated on a successful vcov, neither
    # of which the native tables record. The synthetic ``horizon == -1``
    # base row has no entry. None when no event study was fit.
    event_study_df: Optional[Dict[int, float]] = None
    # pooled_df: the same df PROVENANCE for the POOLED windows, keyed
    # ``"pre"`` / ``"post"`` to match the ``window`` column. The pooled-post
    # row is the headline ATT, which lives outside the unified event-study
    # surface, so this is standalone headline provenance (the
    # ``DiDResults.inference_df`` analogue) rather than a surface input. NaN
    # per window when that window's inference used normal theory or an
    # undefined df; None when no pooled windows were fit (``only_event``).
    pooled_df: Optional[Dict[str, float]] = None
    # The estimator's df_convention configuration echoed onto the results
    # ("residual" | "cluster" | "normal"; LPDiD's default is "cluster" -
    # the Stata lpdid t(G-1) convention; added 3.9). Appended LAST (the
    # generated __init__ positional indexes are public API).
    df_convention: Optional[str] = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @property
    def estimand(self) -> str:
        return "equally-weighted ATT" if self.reweight else "variance-weighted ATT"

    def _base_period_label(self) -> str:
        if self.pmd == "max":
            return "premean (all available pretreatment periods)"
        if isinstance(self.pmd, int) and not isinstance(self.pmd, bool):
            return f"premean (last {self.pmd} pretreatment periods)"
        return "first-lag (t-1)"

    def _pooled_row(self, window: str) -> Optional[pd.Series]:
        if self.pooled is None:
            return None
        match = self.pooled.loc[self.pooled["window"] == window]
        if match.empty:
            return None
        return match.iloc[0]

    # ------------------------------------------------------------------
    # headline inference aliases (over the pooled `post` row)
    # ------------------------------------------------------------------
    @property
    def att(self) -> float:
        row = self._pooled_row("post")
        return float(row["coefficient"]) if row is not None else float("nan")

    @property
    def se(self) -> float:
        row = self._pooled_row("post")
        return float(row["se"]) if row is not None else float("nan")

    @property
    def t_stat(self) -> float:
        row = self._pooled_row("post")
        return float(row["t_stat"]) if row is not None else float("nan")

    @property
    def p_value(self) -> float:
        row = self._pooled_row("post")
        return float(row["p_value"]) if row is not None else float("nan")

    @property
    def conf_int(self) -> Tuple[float, float]:
        row = self._pooled_row("post")
        if row is None:
            return (float("nan"), float("nan"))
        return (float(row["conf_low"]), float(row["conf_high"]))

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        # "event_study" is the canonical level spelling library-wide; the
        # drifted "event" warns and maps through the 3.9 shim window
        # (missed-rename row, section 8 rule 10 - same unification as
        # WooldridgeDiDResults.aggregate's M-086).
        if level == "event":
            warn_deprecated_kwarg(
                "LPDiDResults.to_dataframe",
                "level='event'",
                "use level='event_study' instead",
            )
            level = "event_study"
        if level == "event_study":
            if self.event_study is None:
                raise ValueError("event_study dataframe was not computed")
            return self.event_study.copy()
        if level == "pooled":
            if self.pooled is None:
                raise ValueError("pooled dataframe was not computed")
            return self.pooled.copy()
        raise ValueError("level must be 'event_study' or 'pooled'")

    def to_dict(self) -> Dict[str, Any]:
        pre = self._pooled_row("pre")
        ci = self.conf_int
        result: Dict[str, Any] = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": ci[0],
            "conf_int_upper": ci[1],
            "pre_att": float(pre["coefficient"]) if pre is not None else float("nan"),
            "pre_se": float(pre["se"]) if pre is not None else float("nan"),
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "pre_window": self.pre_window,
            "post_window": self.post_window,
            "control_group": self.control_group,
            "reweight": self.reweight,
            "no_composition": self.no_composition,
            "pmd": self.pmd,
            "estimand": self.estimand,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
            "rank_deficient_action": self.rank_deficient_action,
            "ylags": self.ylags,
            "dylags": self.dylags,
            "covariates": self.covariates,
            "absorb": self.absorb,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.pooled_df is not None:
            result["pooled_df"] = self.pooled_df
        if self.non_absorbing is not None:
            result["non_absorbing"] = self.non_absorbing
            result["stabilization_window"] = self.stabilization_window
        if self.survey_metadata is not None:
            result["n_strata"] = self.n_strata
            result["n_psu"] = self.n_psu
            result["weight_type"] = getattr(self.survey_metadata, "weight_type", None)
            result["df_survey"] = getattr(self.survey_metadata, "df_survey", None)
        if self.df_convention is not None:
            result["df_convention"] = self.df_convention
        result["inference_method"] = (
            "survey_tsl" if self.vcov_type == "survey_tsl" else "cluster_robust"
        )
        return result

    # ------------------------------------------------------------------
    # text summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        from diff_diff.results import (
            _format_survey_block,
            _format_vcov_label,
            _get_significance_stars,
        )

        # Confidence intervals in the event_study / pooled tables are computed at
        # fit time using ``self.alpha``; the displayed level must match them, so
        # summary() does not accept an alpha override (it would relabel without
        # recomputing the intervals).
        ci_pct = _coverage_pct(self.alpha)
        width = 88
        bar = "=" * width
        dash = "-" * width

        def _fmt(x: Any, nd: int = 4) -> str:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return ""
            return "" if np.isnan(xf) else f"{xf:.{nd}f}"

        lines: List[str] = [
            bar,
            "Local Projections DiD (Dube, Girardi, Jorda & Taylor 2025) Results".center(width),
            bar,
            f"Observations: {self.n_obs}    Treated units: {self.n_treated_units}"
            f"    Never-treated units: {self.n_control_units}",
            f"Estimand: {self.estimand}    Control group: {self.control_group}",
            f"Base period: {self._base_period_label()}    No composition: {self.no_composition}",
        ]
        if self.non_absorbing is not None:
            mode = (
                f"effect-stabilization (L={self.stabilization_window})"
                if self.non_absorbing == "effect_stabilization"
                else "first-entry"
            )
            lines.append(f"Treatment path: non-absorbing, {mode} (Dube et al. 2025 Sec. 4.2)")
        if self.covariates or self.absorb or self.ylags or self.dylags:
            cov_path = "regression-adjustment" if self.reweight else "direct inclusion"
            lag_bits = []
            if self.ylags:
                lag_bits.append(f"ylags={self.ylags}")
            if self.dylags:
                lag_bits.append(f"dylags={self.dylags}")
            lag_str = ("    " + ", ".join(lag_bits)) if lag_bits else ""
            lines.append(
                f"Covariates: {self.covariates or []}    Absorb: {self.absorb or []}"
                f"{lag_str}    ({cov_path})"
            )
        if self.vcov_type == "if_cluster":
            # Regression-adjustment path: influence-function cluster variance
            # (ImputationDiD/BJS family), not an OLS CR1 sandwich.
            g = f", G={self.n_clusters}" if self.n_clusters else ""
            vcov_label = f"Influence-function cluster-robust at {self.cluster_name}{g}"
        elif self.vcov_type == "survey_tsl":
            # Complex-survey path: stratified-PSU Taylor-linearization (Binder TSL)
            # sandwich. _format_vcov_label does not know "survey_tsl", so build the
            # label here (G = realized pooled-post PSU count). The design block below
            # carries the full survey metadata.
            g = f", G={self.n_clusters}" if self.n_clusters else ""
            psu = self.cluster_name or "PSU"
            vcov_label = f"Survey Taylor-linearization (stratified PSU) at {psu}{g}"
        else:
            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
        if vcov_label:
            lines.append(f"Std. errors: {vcov_label}")
        if self.survey_metadata is not None:
            lines.extend(_format_survey_block(self.survey_metadata, width))

        header = (
            f"{'':>8}  {'Estimate':>10}  {'Std.Err':>10}  {'t':>8}  {'P>|t|':>8}"
            f"  [{ci_pct}% Conf. Int.]"
        )

        def _table(df: pd.DataFrame, key: str) -> List[str]:
            rows: List[str] = [dash, header, dash]
            for _, r in df.iterrows():
                label = r[key]
                if key == "horizon" and int(r[key]) == -1:
                    rows.append(f"{int(label):>8}  {'0.0000':>10}  {'(reference)':>10}")
                    continue
                p = r["p_value"]
                stars = "" if pd.isna(p) else _get_significance_stars(float(p))
                label_str = f"{int(label):>8}" if key == "horizon" else f"{str(label):>8}"
                rows.append(
                    f"{label_str}  {_fmt(r['coefficient']):>10}  {_fmt(r['se']):>10}"
                    f"  {_fmt(r['t_stat'], 2):>8}  {_fmt(r['p_value'], 3):>8}"
                    f"  [{_fmt(r['conf_low']):>9}, {_fmt(r['conf_high']):>9}] {stars}"
                )
            return rows

        if self.event_study is not None:
            lines.append("")
            lines.append("Event study (relative horizon):")
            lines.extend(_table(self.event_study, "horizon"))
        if self.pooled is not None:
            lines.append("")
            lines.append("Pooled (pre = placebo, post = ATT):")
            lines.extend(_table(self.pooled, "window"))

        lines.append(bar)
        lines.append("Signif. codes: *** p<0.001, ** p<0.01, * p<0.05")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:
        cluster = f", cluster={self.cluster_name}, G={self.n_clusters}" if self.cluster_name else ""
        att = self.att
        se = self.se
        att_s = "nan" if np.isnan(att) else f"{att:.4f}"
        se_s = "nan" if np.isnan(se) else f"{se:.4f}"
        return (
            "LPDiDResults("
            f"estimand={'reweight' if self.reweight else 'variance-weighted'}, "
            f"post_ATT={att_s}, SE={se_s}, "
            f"pre_window={self.pre_window}, post_window={self.post_window}, "
            f"control_group={self.control_group!r}{cluster})"
        )
