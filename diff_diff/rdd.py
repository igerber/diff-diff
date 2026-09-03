"""
Regression discontinuity design (RDD) estimation - sharp and fuzzy, with
optional covariate adjustment - and robust bias-corrected inference,
parity-targeting R ``rdrobust`` 4.0.0.

Implements the local-polynomial RD estimators of Calonico, Cattaneo &
Titiunik (2014). SHARP (default): treatment is assigned by
``running >= cutoff``; the effect is the jump in the conditional
expectation of the outcome at the cutoff. FUZZY (pass
``fit(..., takeup=...)`` with the OBSERVED take-up column):
crossing the cutoff shifts take-up rather than determining it, and the
estimand is the local Wald ratio - the outcome jump divided by the
take-up jump - which for BINARY take-up under monotonicity is the LATE
for compliers at the cutoff (non-binary take-up keeps the ratio-of-jumps
reading; the ``estimand`` results field says which applies). Both designs
use kernel-weighted polynomial regressions on each side with data-driven
MSE/CER-optimal bandwidths and robust
bias-corrected (RBC) inference; the fuzzy bias correction is the
linearization of the ratio (not per-component), matching CCT 2014
Section 3.2 and rdrobust exactly.

Covariate adjustment (``fit(..., covariates=[...])``; Calonico, Cattaneo,
Farrell & Titiunik 2019, R's ``covs=``): covariates enter ADDITIVELY with
a common coefficient pooled across sides (CCFT 2019 Equation 2 - the only
specification with a clean guarantee; treatment-interacted and demeaned
variants are documented as inconsistent-or-inferior there). UNLIKE the
library's DiD estimators, where ``covariates`` switches identification to
conditional parallel trends, RD covariates DO NOT change the estimand -
the ``att`` still measures the same cutoff jump/ratio and the
``estimand`` label is unchanged; adjustment buys precision (shorter CIs)
when covariates predict the outcome near the cutoff. The operative
requirement is covariate BALANCE at the cutoff (zero RD effect on each
covariate); imbalanced covariates make the adjusted estimator
inconsistent, and adjusting "for" imbalance cannot restore
identification. Balance is testable with the estimator itself::

    balance = RegressionDiscontinuity().fit(df, outcome="z1",
                                            running="x")
    balance.p_value  # small p = imbalance; do not adjust for z1

Bandwidths are covariate-AWARE (covariates propagate into selection, not
just estimation, as in R). Collinear covariates are dropped with a
warning under ``covs_drop=True`` (R's default; the warning names the
dropped columns).

Canonical inference binding
---------------------------
``RegressionDiscontinuityResults`` binds the library-canonical fields to ONE
internally coherent inference row - the ROBUST row of rdrobust's output:
``att`` is the bias-corrected point estimate ``tau_bc`` (the linearized
bias-corrected RATIO on fuzzy fits), ``se`` its robust standard error, and
``t_stat``/``p_value``/``conf_int`` are computed from that same pair, so
the library-wide identities hold (``t_stat == att/se``, ``conf_int``
centered on ``att``). The ``estimand`` results field names what ``att``
measures for the fit at hand. This deliberately differs from rdrobust's
PRINTED headline, which reports the conventional estimate ``tau_cl`` in the
coefficient column while taking inference from the robust row; ``tau_cl`` is
first-class here as ``att_conventional`` (with its own full inference row),
and ``summary()`` prints the familiar three-row rdrobust table. Fuzzy fits
additionally expose the first stage (take-up jump) as a full three-row
mirror (``first_stage*`` fields) and print it above the treatment effects,
as R does.

rdrobust equivalents
--------------------
=======================  ==========================================
diff-diff                R rdrobust
=======================  ==========================================
``cutoff``               ``c``
``vcov_type``            ``vce``
``alpha``                ``1 - level/100``
``h``, ``b``, ``rho``    ``h``, ``b``, ``rho`` (same semantics)
``p``, ``q``             ``p``, ``q``
``bwselect``             ``bwselect`` (same 10-option menu)
``kernel``               ``kernel`` (accepts "tri"/"epa"/"uni" too)
``masspoints``           ``masspoints`` ("adjust"/"check"/"off")
``nnmatch``              ``nnmatch``
``takeup`` (fit)         ``fuzzy`` (observed take-up variable)
``sharpbw``              ``sharpbw`` (same default and semantics)
``covariates`` (fit)     ``covs`` (column names instead of a matrix)
``covs_drop``            ``covs_drop`` (same default and semantics)
=======================  ==========================================

Not in v1 (documented seams, see REGISTRY.md): cluster-robust variance,
weights, ``deriv``/kink estimands, ``scalepar``, ``stdvars``, hc0-hc3
variance modes, weak-IV-robust fuzzy inference (Feir-Lemieux-Marmer),
and a packaged covariate-balance helper (the recipe above covers it).

References
----------
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust Nonparametric
  Confidence Intervals for Regression-Discontinuity Designs. *Econometrica*,
  82(6), 2295-2326.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017).
  rdrobust: Software for regression-discontinuity designs. *Stata Journal*,
  17(2), 372-404.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the Effect of
  Bias Estimation on Coverage Accuracy in Nonparametric Inference. *JASA*,
  113(522), 767-779.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019).
  Regression Discontinuity Designs Using Covariates. *Review of Economics
  and Statistics*, 101(3), 442-451.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import (
    NOT_SUPPLIED,
    deprecated_field_property,
    require_arg,
    resolve_renamed_kwarg,
)
from diff_diff._rdrobust_port import (
    BWSELECT_OPTIONS,
    _fuzzy_identification_stop,
    _normalize_kernel,
    covs_drop_fun,
    rdbwselect,
    rdrobust_fit,
)
from diff_diff.results_base import BaseResults, _coverage_pct
from diff_diff.utils import safe_inference, validate_covariate_names

__all__ = [
    "RegressionDiscontinuity",
    "RegressionDiscontinuityResults",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class RegressionDiscontinuityResults(BaseResults):
    """Results of a regression discontinuity fit (sharp or fuzzy; the
    ``estimand`` field names which one, and ``first_stage*`` fields are
    populated on fuzzy fits only).

    Canonical inference fields (``att``, ``se``, ``t_stat``, ``p_value``,
    ``conf_int``) all describe the ROBUST bias-corrected row: ``att`` is the
    bias-corrected point estimate and ``conf_int`` is centered on it (see
    module docstring for the binding rationale and the deviation from
    rdrobust's printed headline). The conventional row is exposed as
    explicit ``*_conventional`` fields. The (rarely used) middle
    "Bias-Corrected" row shares its coefficient with ``att`` (both are
    ``tau_bc``) and its standard error with ``se_conventional`` - only its
    inference triple carries the ``*_bias_corrected`` suffix
    (``t_stat_bias_corrected``, ``p_value_bias_corrected``,
    ``conf_int_bias_corrected``); there are deliberately no redundant
    ``att_bias_corrected`` / ``se_bias_corrected`` fields. Together the
    three rows mirror rdrobust's output exactly.
    """

    # Canonical (robust row; internally coherent)
    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    alpha: float

    # Conventional row (rdrobust's printed headline coefficient)
    att_conventional: float
    se_conventional: float
    t_stat_conventional: float
    p_value_conventional: float
    conf_int_conventional: Tuple[float, float]

    # Bias-corrected middle row (tau_bc with the CONVENTIONAL SE; exposed
    # for rdrobust parity - prefer the robust row for inference)
    t_stat_bias_corrected: float
    p_value_bias_corrected: float
    conf_int_bias_corrected: Tuple[float, float]

    # Explicit duplicates for clarity
    se_robust: float

    # Bandwidths (rdrobust bws layout)
    h_left: float
    h_right: float
    b_left: float
    b_right: float

    # Sample composition
    n_obs: int
    n_left: int
    n_right: int
    n_h_left: int
    n_h_right: int
    n_b_left: int
    n_b_right: int
    n_unique_left: int
    n_unique_right: int
    n_dropped: int

    # Config echoes. ``bwselect`` is the RESOLVED selector label ("Manual"
    # when bandwidths were user-supplied or N<20 forced the full-range
    # fallback, matching rdrobust's printed "BW type"); ``h_input`` /
    # ``b_input`` / ``rho_input`` echo the constructor arguments as supplied
    # (None when data-driven; a warned-and-ignored ``b``-without-``h`` still
    # echoes here) - together with the other echoes they reconstruct the
    # full fit configuration from a saved result. Resolved per-side
    # bandwidths live in ``h_left``/``h_right``/``b_left``/``b_right``.
    cutoff: float
    p: int
    q: int
    kernel: str
    bwselect: str
    vcov_type: str
    nnmatch: int
    masspoints: str
    bwcheck: Optional[int]
    bwrestrict: bool
    scaleregul: float
    h_input: Optional[float]
    b_input: Optional[float]
    rho_input: Optional[float]
    # Design echoes: ``estimand`` names what ``att`` measures for THIS fit
    # - "sharp (ATE at the cutoff)"; "fuzzy (LATE for compliers at the
    # cutoff)" for BINARY take-up; or "fuzzy (local Wald ratio at the
    # cutoff; non-binary take-up)" when the take-up column is not {0, 1}
    # (the complier-LATE reading does not apply to dose take-up).
    # ``takeup`` is the fit-time take-up column name
    # (None on sharp fits; no ``_input`` suffix - that convention is
    # reserved for constructor arguments); ``sharpbw`` and ``covs_drop``
    # echo the constructor flags. The estimand label deliberately does NOT
    # change under covariate adjustment: CCFT 2019 covariates target the
    # SAME estimand (precision only) - see ``covariates`` below.
    # (The deprecated read-only alias ``treatment_col`` warns and returns
    # ``takeup``; removed in 4.0 - row M-094.)
    estimand: str
    sharpbw: bool
    takeup: Optional[str]
    covs_drop: bool

    # First-stage (take-up jump) three-row mirror - fuzzy fits only, all
    # None on sharp fits. Same binding rule as the main estimate: the
    # unsuffixed quintet is the coherent ROBUST row (first_stage = the
    # bias-corrected first-stage estimate tau_T_bc, first_stage_se = its
    # robust SE); the conventional row and the bias-corrected middle-row
    # inference triple mirror the main fields' suffix scheme.
    first_stage: Optional[float] = None
    first_stage_se: Optional[float] = None
    first_stage_t_stat: Optional[float] = None
    first_stage_p_value: Optional[float] = None
    first_stage_conf_int: Optional[Tuple[float, float]] = None
    first_stage_conventional: Optional[float] = None
    first_stage_se_conventional: Optional[float] = None
    first_stage_t_stat_conventional: Optional[float] = None
    first_stage_p_value_conventional: Optional[float] = None
    first_stage_conf_int_conventional: Optional[Tuple[float, float]] = None
    first_stage_t_stat_bias_corrected: Optional[float] = None
    first_stage_p_value_bias_corrected: Optional[float] = None
    first_stage_conf_int_bias_corrected: Optional[Tuple[float, float]] = None

    # Covariate adjustment (CCFT 2019) - all None on unadjusted fits.
    # ``covariates`` echoes the fit-time column names AS PASSED;
    # ``covariates_dropped`` lists columns removed as collinear by
    # covs_drop ([] when nothing was dropped); ``covariate_coefficients``
    # maps each RETAINED covariate name to its common (pooled across
    # sides) outcome-equation projection coefficient gamma - these are
    # nuisance coefficients for the adjustment, NOT causal effects of the
    # covariates. Fuzzy fits add ``first_stage_covariate_coefficients``
    # (the take-up-equation gamma). Name-keyed dicts make R's internal
    # name-length column sort invisible to users.
    covariates: Optional[List[str]] = None
    covariates_dropped: Optional[List[str]] = None
    covariate_coefficients: Optional[Dict[str, float]] = None
    first_stage_covariate_coefficients: Optional[Dict[str, float]] = None

    # Per-side order-p coefficient vectors (rdplot seam); the outcome pair
    # is always populated by fit(), so typed non-Optional despite the
    # dataclass default; the take-up pair is fuzzy-only. On
    # covariate-adjusted fits these are the ADJUSTED vectors (gamma
    # combination applied), matching R's beta_Y_p_* / beta_T_p_*.
    beta_p_left: np.ndarray = field(repr=False, default=None)
    beta_p_right: np.ndarray = field(repr=False, default=None)
    beta_t_p_left: Optional[np.ndarray] = field(repr=False, default=None)
    beta_t_p_right: Optional[np.ndarray] = field(repr=False, default=None)

    # Deprecated read-only alias for ``takeup`` (row M-094; removed in 4.0).
    # No annotation, so it stays a descriptor and never becomes a
    # __dataclass_fields__ entry.
    treatment_col = deprecated_field_property(
        "RegressionDiscontinuityResults", "treatment_col", "takeup"
    )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Migrate pickles created before the ``treatment_col`` -> ``takeup``
        rename (row M-094): rewrite the key on load so both the new field
        and the deprecated alias work on old pickles."""
        if "treatment_col" in state and "takeup" not in state:
            state = dict(state)
            state["takeup"] = state.pop("treatment_col")
        self.__dict__.update(state)

    def summary(self) -> str:
        """Human-readable summary with the three-row rdrobust table."""
        width = 72
        conf_level = _coverage_pct(self.alpha)
        lines = []
        lines.append("=" * width)
        design = "Fuzzy" if self.first_stage is not None else "Sharp"
        if self.covariates:
            # Mirrors R's rdmodel string ("Covariate-adjusted ... RD
            # estimates"); the estimand line below is deliberately
            # UNCHANGED - covariates buy precision, not a new estimand.
            design = f"Covariate-adjusted {design}"
        lines.append(f"{design} Regression Discontinuity (rdrobust parity)".center(width))
        lines.append("=" * width)
        lines.append(f"Cutoff:               {self.cutoff:g}")
        lines.append(f"Estimand:             {self.estimand}")
        if self.covariates:
            cov_line = f"Covariates ({len(self.covariates)}): " + ", ".join(self.covariates)
            if self.covariates_dropped:
                cov_line += "  [dropped: " + ", ".join(self.covariates_dropped) + "]"
            lines.append(cov_line)
        lines.append(f"Kernel: {self.kernel:<14} Bandwidth selector: {self.bwselect}")
        lines.append(
            f"Order (p, q): ({self.p}, {self.q})       VCE: {self.vcov_type} "
            f"(nnmatch={self.nnmatch})   Masspoints: {self.masspoints}"
        )
        lines.append(
            f"N = {self.n_obs} ({self.n_left} left / {self.n_right} right); "
            f"effective N_h = {self.n_h_left}/{self.n_h_right}, "
            f"N_b = {self.n_b_left}/{self.n_b_right}"
        )
        lines.append(
            f"h = [{self.h_left:.4f}, {self.h_right:.4f}]   "
            f"b = [{self.b_left:.4f}, {self.b_right:.4f}]"
        )
        lines.append("-" * width)
        header = (
            f"{'Method':<16}{'Coef.':>11}{'Std. Err.':>11}{'z':>9}"
            f"{'P>|z|':>9}{'[' + f'{conf_level}% Conf. Int.]':>16}"
        )
        if self.first_stage is not None:
            # Fuzzy: R prints a first-stage block above the treatment
            # effects (print.summary.rdrobust); same three-row structure.
            lines.append("First-stage estimates (treatment take-up jump)".center(width))
            lines.append(header)
            lines.append("-" * width)
            fs_rows = [
                (
                    "Conventional",
                    self.first_stage_conventional,
                    self.first_stage_se_conventional,
                    self.first_stage_t_stat_conventional,
                    self.first_stage_p_value_conventional,
                    self.first_stage_conf_int_conventional,
                ),
                (
                    "Bias-Corrected",
                    self.first_stage,
                    self.first_stage_se_conventional,
                    self.first_stage_t_stat_bias_corrected,
                    self.first_stage_p_value_bias_corrected,
                    self.first_stage_conf_int_bias_corrected,
                ),
                (
                    "Robust",
                    self.first_stage,
                    self.first_stage_se,
                    self.first_stage_t_stat,
                    self.first_stage_p_value,
                    self.first_stage_conf_int,
                ),
            ]
            for name, coef, se, z, pv, ci in fs_rows:
                assert coef is not None and se is not None and ci is not None
                assert z is not None and pv is not None
                lines.append(
                    f"{name:<16}{coef:>11.4f}{se:>11.4f}{z:>9.3f}{pv:>9.3f}"
                    f"  [{ci[0]:>7.4f}, {ci[1]:>7.4f}]"
                )
            lines.append("-" * width)
            lines.append("Treatment effect estimates".center(width))
        lines.append(header)
        lines.append("-" * width)
        rows = [
            (
                "Conventional",
                self.att_conventional,
                self.se_conventional,
                self.t_stat_conventional,
                self.p_value_conventional,
                self.conf_int_conventional,
            ),
            (
                "Bias-Corrected",
                self.att,
                self.se_conventional,
                self.t_stat_bias_corrected,
                self.p_value_bias_corrected,
                self.conf_int_bias_corrected,
            ),
            (
                "Robust",
                self.att,
                self.se_robust,
                self.t_stat,
                self.p_value,
                self.conf_int,
            ),
        ]
        for name, coef, se, z, pv, ci in rows:
            lines.append(
                f"{name:<16}{coef:>11.4f}{se:>11.4f}{z:>9.3f}{pv:>9.3f}"
                f"  [{ci[0]:>7.4f}, {ci[1]:>7.4f}]"
            )
        lines.append("-" * width)
        lines.append("Note: canonical att/se/t_stat/p_value/conf_int are the ROBUST row")
        lines.append("(att = bias-corrected estimate; rdrobust prints the conventional")
        lines.append("estimate as its headline coefficient - see att_conventional).")
        lines.append("=" * width)
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Flat scalar dict; confidence intervals split into lower/upper."""
        out: Dict[str, Any] = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "alpha": self.alpha,
            "att_conventional": self.att_conventional,
            "se_conventional": self.se_conventional,
            "t_stat_conventional": self.t_stat_conventional,
            "p_value_conventional": self.p_value_conventional,
            "conf_int_conventional_lower": self.conf_int_conventional[0],
            "conf_int_conventional_upper": self.conf_int_conventional[1],
            "t_stat_bias_corrected": self.t_stat_bias_corrected,
            "p_value_bias_corrected": self.p_value_bias_corrected,
            "conf_int_bias_corrected_lower": self.conf_int_bias_corrected[0],
            "conf_int_bias_corrected_upper": self.conf_int_bias_corrected[1],
            "se_robust": self.se_robust,
            "h_left": self.h_left,
            "h_right": self.h_right,
            "b_left": self.b_left,
            "b_right": self.b_right,
            "n_obs": self.n_obs,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_h_left": self.n_h_left,
            "n_h_right": self.n_h_right,
            "n_b_left": self.n_b_left,
            "n_b_right": self.n_b_right,
            "n_unique_left": self.n_unique_left,
            "n_unique_right": self.n_unique_right,
            "n_dropped": self.n_dropped,
            "cutoff": self.cutoff,
            "p": self.p,
            "q": self.q,
            "kernel": self.kernel,
            "bwselect": self.bwselect,
            "vcov_type": self.vcov_type,
            "nnmatch": self.nnmatch,
            "masspoints": self.masspoints,
            "bwcheck": self.bwcheck,
            "bwrestrict": self.bwrestrict,
            "scaleregul": self.scaleregul,
            "h_input": self.h_input,
            "b_input": self.b_input,
            "rho_input": self.rho_input,
            "estimand": self.estimand,
            "sharpbw": self.sharpbw,
            "takeup": self.takeup,
            # Deprecated key mirroring ``takeup`` through the 3.9 shim
            # window; dropped in 4.0 (row M-094, section 5 policy).
            "treatment_col": self.takeup,
            "covs_drop": self.covs_drop,
            # List/dict-valued covariate echoes (None on unadjusted fits;
            # the lpdid/continuous_did echo convention).
            "covariates": self.covariates,
            "covariates_dropped": self.covariates_dropped,
            "covariate_coefficients": self.covariate_coefficients,
            "first_stage_covariate_coefficients": self.first_stage_covariate_coefficients,
            "first_stage": self.first_stage,
            "first_stage_se": self.first_stage_se,
            "first_stage_t_stat": self.first_stage_t_stat,
            "first_stage_p_value": self.first_stage_p_value,
            "first_stage_conventional": self.first_stage_conventional,
            "first_stage_se_conventional": self.first_stage_se_conventional,
            "first_stage_t_stat_conventional": self.first_stage_t_stat_conventional,
            "first_stage_p_value_conventional": self.first_stage_p_value_conventional,
            "first_stage_t_stat_bias_corrected": self.first_stage_t_stat_bias_corrected,
            "first_stage_p_value_bias_corrected": self.first_stage_p_value_bias_corrected,
        }
        # First-stage CIs are None on sharp fits - guard the tuple splits.
        for key, ci in (
            ("first_stage_conf_int", self.first_stage_conf_int),
            ("first_stage_conf_int_conventional", self.first_stage_conf_int_conventional),
            ("first_stage_conf_int_bias_corrected", self.first_stage_conf_int_bias_corrected),
        ):
            out[f"{key}_lower"] = None if ci is None else ci[0]
            out[f"{key}_upper"] = None if ci is None else ci[1]
        return {k: _json_safe(v) for k, v in out.items()}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class RegressionDiscontinuity(BaseEstimator):
    """Regression discontinuity estimator - sharp and fuzzy, with
    optional covariate adjustment (rdrobust 4.0.0 parity).

    SHARP (default): treatment is defined by the running variable crossing
    a known cutoff (``running >= cutoff`` treated, matching rdrobust:
    units exactly at the cutoff are treated). FUZZY: pass the observed
    take-up column via ``fit(..., takeup=...)`` - the estimand
    becomes the local Wald ratio (complier LATE at the cutoff for binary
    take-up under monotonicity; the ``estimand`` results field says which
    reading applies) and the results gain a first-stage block.
    COVARIATE ADJUSTMENT: pass ``fit(..., covariates=[...])`` (R's
    ``covs=``) for the CCFT 2019 additive common-coefficient adjustment -
    the estimand is UNCHANGED (precision only; requires covariate balance
    at the cutoff, see the module docstring), bandwidths become
    covariate-aware, and collinear columns are dropped with a warning
    under ``covs_drop=True``. Point
    estimation uses kernel-weighted local polynomials of order ``p`` on
    each side; inference is robust bias-corrected per Calonico, Cattaneo &
    Titiunik (2014). Defaults reproduce ``rdrobust(y, x)`` /
    ``rdrobust(y, x, fuzzy=t)`` / ``rdrobust(y, x, covs=Z)``: ``p=1``,
    ``q=2``, triangular kernel, ``bwselect="mserd"``, nearest-neighbor
    variance with 3 matches, ``masspoints="adjust"``, ``covs_drop=True``.

    Parameters
    ----------
    cutoff : float, default 0.0
        The known threshold ``c`` of the running variable.
    p : int, default 1
        Local-polynomial order for point estimation; integer in 0..20
        (mirroring rdrobust's accepted surface; ``p=0`` is the
        local-constant fit).
    q : int or None, default None
        Order for the bias regression; an explicit ``q`` must satisfy
        ``p < q <= 20``. ``None`` resolves to ``p + 1`` WITHOUT
        re-validation, exactly as R does (rdrobust.R:53-57) - so ``p=20``
        with the default ``q`` yields ``q=21`` in both implementations
        while an explicit ``q=21`` is rejected.
    kernel : str, default "triangular"
        "triangular", "epanechnikov", or "uniform" (R spellings
        "tri"/"epa"/"uni" accepted).
    bwselect : str, default "mserd"
        Data-driven bandwidth selector; one of the 10 rdrobust options
        (mserd, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum,
        cercomb1, cercomb2). Ignored when ``h`` is supplied.
    h, b : float or None
        Manual main / bias bandwidths (both sides). ``h`` alone implies
        ``b = h``; ``h`` with ``rho`` implies ``b = h/rho`` (overriding a
        supplied ``b``, as in R); ``b`` without ``h`` is ignored with a
        warning (R ignores it silently - documented deviation).
    rho : float or None
        Bandwidth ratio ``h/b``. Without ``h``, applies to the SELECTED
        bandwidths (``b = h_selected/rho``), mirroring rdrobust.
    vcov_type : str, default "nn"
        Variance estimator. Only "nn" (same-side nearest-neighbor,
        rdrobust's default) is implemented in this release; "hc0"-"hc3"
        and cluster modes raise ``NotImplementedError``.
    nnmatch : int, default 3
        Minimum number of nearest neighbors for the NN variance.
    masspoints : str, default "adjust"
        Mass-point handling: "adjust" (rdrobust default), "check", "off".
    bwcheck : int or None, default None
        Minimum unique support points forced inside the bandwidth window.
    bwrestrict : bool, default True
        Clamp bandwidths to the running variable's observed range.
    scaleregul : float, default 1.0
        Scale of the IK-style regularization in bandwidth selection
        (0 removes it).
    sharpbw : bool, default False
        Fuzzy fits only (``fit(..., takeup=...)``): when True,
        bandwidths are selected for the SHARP reduced-form estimator on
        the outcome (rdrobust's "approach 1") instead of the default
        fuzzy-ratio objective. Automatically in effect - regardless of
        this flag - under one-sided perfect compliance (zero take-up
        variance on either side), exactly as in R. On sharp fits the flag
        has no effect and a warning is emitted (R ignores it silently -
        documented deviation). Never drops covariates from selection -
        with ``covariates`` it selects on the covariate-adjusted sharp
        objective, as in R.
    covs_drop : bool, default True
        Covariate-adjusted fits only (``fit(..., covariates=[...])``):
        when True (R's default), redundant (collinear) covariate columns
        are dropped with a warning naming them before fitting, and the
        covariate projection uses a pseudo-inverse; when False the solve
        is strict and collinear covariates raise a clear error. Without
        ``covariates`` the flag has no effect and setting it to False
        emits a warning (same pattern as ``sharpbw`` on sharp fits).
    alpha : float, default 0.05
        Significance level (rdrobust ``level = 100*(1-alpha)``).

    Examples
    --------
    >>> rd = RegressionDiscontinuity(cutoff=0.0)
    >>> results = rd.fit(df, outcome="y", running="x")
    >>> results.att, results.conf_int  # robust bias-corrected inference
    >>> fuzzy = rd.fit(df, "y", "x", takeup="takeup")  # fuzzy RD
    >>> fuzzy.att, fuzzy.first_stage  # local Wald ratio + take-up jump
    """

    def __init__(
        self,
        cutoff: float = 0.0,
        p: int = 1,
        q: Optional[int] = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        h: Optional[float] = None,
        b: Optional[float] = None,
        rho: Optional[float] = None,
        vcov_type: str = "nn",
        nnmatch: int = 3,
        masspoints: str = "adjust",
        bwcheck: Optional[int] = None,
        bwrestrict: bool = True,
        scaleregul: float = 1.0,
        sharpbw: bool = False,
        covs_drop: bool = True,
        alpha: float = 0.05,
    ):
        self.cutoff = cutoff
        self.p = p
        self.q = q
        self.kernel = kernel
        self.bwselect = bwselect
        self.h = h
        self.b = b
        self.rho = rho
        self.vcov_type = vcov_type
        self.nnmatch = nnmatch
        self.masspoints = masspoints
        self.bwcheck = bwcheck
        self.bwrestrict = bwrestrict
        self.scaleregul = scaleregul
        self.sharpbw = sharpbw
        self.covs_drop = covs_drop
        self.alpha = alpha
        self._validate_constructor_args()

    # ------------------------------------------------------------------
    # Configuration plumbing (sklearn-like)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_real_scalar(val: Any) -> bool:
        # Reject non-numeric types up front so every scalar knob fails with
        # the estimator's ValueError, not NumPy's TypeError (bool excluded:
        # True is not a bandwidth).
        return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
            val, (bool, np.bool_)
        )

    @staticmethod
    def _is_int_scalar(val: Any) -> bool:
        # bool is an int subclass; p=True must not silently become p=1.
        return isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_))

    def _validate_constructor_args(self) -> None:
        if not (self._is_real_scalar(self.cutoff) and np.isfinite(self.cutoff)):
            raise ValueError(f"cutoff must be finite; got {self.cutoff!r}.")
        # p/q bounds mirror rdrobust.R:47-57 exactly: integers in 0:20 with
        # q > p (p=0 is R's local-constant fit; q caps at 20 like p).
        if not (self._is_int_scalar(self.p) and 0 <= self.p <= 20):
            raise ValueError(f"p must be an integer in 0..20; got {self.p!r}.")
        if self.q is not None and not (self._is_int_scalar(self.q) and self.p < self.q <= 20):
            raise ValueError(
                f"q must be None (-> p+1) or an integer > p and <= 20; got {self.q!r}."
            )
        _normalize_kernel(self.kernel)  # raises on unknown kernel
        if self.bwselect not in BWSELECT_OPTIONS:
            raise ValueError(f"bwselect must be one of {BWSELECT_OPTIONS}; got {self.bwselect!r}.")
        for name, val in (("h", self.h), ("b", self.b), ("rho", self.rho)):
            if val is not None and not (self._is_real_scalar(val) and np.isfinite(val) and val > 0):
                raise ValueError(f"{name} must be None or finite and > 0; got {val!r}.")
        if self.vcov_type != "nn":
            raise NotImplementedError(
                "Only vcov_type='nn' (rdrobust's default nearest-neighbor "
                "variance) is implemented in this release; 'hc0'-'hc3' and "
                "cluster-robust modes are a documented seam."
            )
        if not (self._is_int_scalar(self.nnmatch) and self.nnmatch >= 1):
            raise ValueError(f"nnmatch must be an integer >= 1; got {self.nnmatch!r}.")
        if self.masspoints not in ("adjust", "check", "off"):
            raise ValueError(
                f"masspoints must be 'adjust', 'check', or 'off'; got {self.masspoints!r}."
            )
        if self.bwcheck is not None and not (
            self._is_int_scalar(self.bwcheck) and self.bwcheck >= 1
        ):
            raise ValueError(f"bwcheck must be None or an integer >= 1; got {self.bwcheck!r}.")
        if not isinstance(self.bwrestrict, (bool, np.bool_)):
            # No silent truthiness: a string like "False" must not coerce
            # to bandwidth-restriction ON.
            raise ValueError(f"bwrestrict must be a bool; got {self.bwrestrict!r}.")
        if not isinstance(self.sharpbw, (bool, np.bool_)):
            raise ValueError(f"sharpbw must be a bool; got {self.sharpbw!r}.")
        if not isinstance(self.covs_drop, (bool, np.bool_)):
            raise ValueError(f"covs_drop must be a bool; got {self.covs_drop!r}.")
        if not (
            self._is_real_scalar(self.scaleregul)
            and np.isfinite(self.scaleregul)
            and self.scaleregul >= 0
        ):
            raise ValueError(f"scaleregul must be finite and >= 0; got {self.scaleregul!r}.")
        if not (self._is_real_scalar(self.alpha) and 0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}.")

    # get_params/set_params come from BaseEstimator.

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        outcome: Any = NOT_SUPPLIED,
        running: Any = NOT_SUPPLIED,
        takeup: Any = NOT_SUPPLIED,
        covariates: Optional[List[str]] = None,
        outcome_col: Any = NOT_SUPPLIED,
        running_col: Any = NOT_SUPPLIED,
        treatment_col: Any = NOT_SUPPLIED,
    ) -> RegressionDiscontinuityResults:
        """Estimate the RD effect at the cutoff (sharp or fuzzy, optionally
        covariate-adjusted).

        Parameters
        ----------
        data : pd.DataFrame
            Cross-sectional data.
        outcome, running : str
            Column names of the outcome and the running variable.
        takeup : str or None, default None
            ``None`` (sharp design): treatment is derived as
            ``running >= cutoff``; no treatment column is needed. A column
            name activates the FUZZY design: the column holds the OBSERVED
            treatment take-up (typically binary, any numeric accepted,
            matching R's ``fuzzy=``), the estimand becomes the local Wald
            ratio, and the results gain the ``first_stage*`` block. The
            ``estimand`` label is data-dependent: for BINARY take-up it
            reads "fuzzy (LATE for compliers at the cutoff)" (the
            monotonicity-based complier reading); for non-binary (dose)
            take-up it reads "fuzzy (local Wald ratio at the cutoff;
            non-binary take-up)" - the complier-LATE interpretation does
            not apply there. A take-up column that is deterministic in
            the running variable reproduces the sharp fit exactly
            (first stage == 1).
        covariates : list of str or None, default None
            Column names of pre-determined covariates for the additive
            common-coefficient adjustment of CCFT (2019) (R's ``covs=``).
            The estimand is UNCHANGED - unlike the DiD estimators'
            conditional-parallel-trends role, RD covariates buy precision
            only, and require covariate BALANCE at the cutoff (zero RD
            effect on each covariate; testable by fitting each covariate
            as the outcome - imbalanced covariates make the adjusted
            estimator inconsistent). Continuous, discrete, or mixed
            columns are accepted; covariates propagate into bandwidth
            selection (covariate-aware, as in R). Collinear columns are
            dropped with a warning under ``covs_drop=True``; see the
            ``covariates*`` results fields for the echo and the fitted
            projection coefficients.
        outcome_col, running_col, treatment_col : str, optional
            Deprecated aliases for ``outcome`` / ``running`` / ``takeup``
            (rows M-040..M-042); each warns with ``FutureWarning`` and
            will be removed in 4.0.
        """
        qualname = "RegressionDiscontinuity.fit"
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
        takeup = resolve_renamed_kwarg(
            qualname,
            "treatment_col",
            treatment_col,
            "takeup",
            takeup,
            default=None,
        )
        # Body-local names; the public parameters are outcome/running/takeup
        # (M-040..M-042).
        outcome_col = outcome
        running_col = running
        treatment_col = takeup
        cols = [outcome_col, running_col]
        if treatment_col is not None:
            cols.append(treatment_col)
        if covariates is not None:
            if isinstance(covariates, str):
                # A bare string would iterate characters; fail closed.
                raise ValueError(f"covariates must be a list of column names; got {covariates!r}.")
            # Materialize BEFORE validating: a generator would be consumed
            # by the all() check and then silently collapse to an empty
            # list (disabling adjustment without a whisper).
            covariates = list(covariates)
            if not all(isinstance(name, str) for name in covariates):
                raise ValueError(f"covariates must be a list of column names; got {covariates!r}.")
            if not covariates:
                covariates = None  # empty list == no adjustment
        if covariates is not None:
            # Duplicate names and collisions with the fit's structural
            # columns corrupt the name-keyed coefficient dict.
            validate_covariate_names(
                covariates,
                cols,
                estimator="RegressionDiscontinuity",
            )
            cols.extend(covariates)
        for col in cols:
            if col not in data.columns:
                raise ValueError(f"Column {col!r} not found in data.")
        fuzzy_fit = treatment_col is not None
        if self.sharpbw and not fuzzy_fit:
            # Deviation from R, which silently ignores sharpbw on sharp
            # fits (no-silent-failures policy; same pattern as b-without-h).
            warnings.warn(
                "sharpbw has no effect without takeup (sharp design) " "and is ignored.",
                UserWarning,
                stacklevel=2,
            )
        if not self.covs_drop and covariates is None:
            # Same pattern as sharpbw-on-sharp: a non-default knob that
            # cannot apply must not pass silently.
            warnings.warn(
                "covs_drop=False has no effect without covariates and is ignored.",
                UserWarning,
                stacklevel=2,
            )
        y_raw = np.asarray(pd.to_numeric(data[outcome_col], errors="coerce"), dtype=np.float64)
        x_raw = np.asarray(pd.to_numeric(data[running_col], errors="coerce"), dtype=np.float64)
        ok = np.isfinite(y_raw) & np.isfinite(x_raw)
        t_raw: Optional[np.ndarray] = None
        if fuzzy_fit:
            # R's complete.cases filter includes the fuzzy column
            # (rdrobust.R:86-89) - the joint drop must too.
            t_raw = np.asarray(
                pd.to_numeric(data[treatment_col], errors="coerce"), dtype=np.float64
            )
            ok = ok & np.isfinite(t_raw)
        z_raw: Optional[np.ndarray] = None
        if covariates is not None:
            # R's complete.cases filter includes the covariate columns
            # (rdrobust.R:80-84) - the joint drop must too. Column order
            # here is AS PASSED; the R name-length sort applies below.
            z_raw = np.column_stack(
                [
                    np.asarray(pd.to_numeric(data[name], errors="coerce"), dtype=np.float64)
                    for name in covariates
                ]
            )
            ok = ok & np.all(np.isfinite(z_raw), axis=1)
        n_dropped = int(y_raw.shape[0] - np.sum(ok))
        if n_dropped > 0:
            # Deviation from R (which drops silently via complete.cases):
            dropped_cols = f"{outcome_col!r}/{running_col!r}"
            if fuzzy_fit:
                dropped_cols += f"/{treatment_col!r}"
            if covariates is not None:
                dropped_cols += "/covariates"
            warnings.warn(
                f"Dropping {n_dropped} row(s) with missing or non-numeric "
                f"values in {dropped_cols}.",
                UserWarning,
                stacklevel=2,
            )
        y = y_raw[ok]
        x = x_raw[ok]
        t = t_raw[ok] if t_raw is not None else None
        z = z_raw[ok] if z_raw is not None else None
        N = int(y.shape[0])
        if N == 0:
            raise ValueError("No complete-case observations to fit on.")
        c = float(self.cutoff)
        if not (np.min(x) <= c <= np.max(x)):
            raise ValueError(
                f"cutoff={c:g} lies outside the observed running-variable "
                f"range [{np.min(x):g}, {np.max(x):g}]."
            )
        p = int(self.p)
        q = int(self.q) if self.q is not None else p + 1
        kernel = _normalize_kernel(self.kernel)

        # --- Covariate column sort + redundant-column drop (hoisted from
        # rdrobust.R:121-140, like the fuzzy identification hoist below;
        # R's order: NaN drop -> covs_drop -> fuzzy stop -> mass points).
        # Under covs_drop=True R first sorts columns by NAME LENGTH
        # (order(nchar), stable - rdrobust.R:131); the sort decides which
        # of a collinear set survives, and all user-facing surfaces are
        # name-keyed so the internal order never leaks. The QR runs on
        # x-SORTED rows - the row order R (and the port entry points) use
        # - so near-threshold rank decisions cannot diverge from the
        # downstream calls. Passing the already-reduced matrix down means
        # the port's own entry-point drop finds full rank and stays
        # silent (no double warning).
        model_covariates: Optional[List[str]] = None
        covariates_dropped: Optional[List[str]] = None
        if covariates is not None:
            assert z is not None
            model_covariates = list(covariates)
            covariates_dropped = []
            if self.covs_drop:
                model_covariates = sorted(model_covariates, key=len)
                z = np.column_stack([z[:, covariates.index(name)] for name in model_covariates])
                keep_idx, rank = covs_drop_fun(z[np.argsort(x, kind="stable")])
                if rank == 0:
                    raise ValueError(
                        "All covariates are numerically zero (rank-0 "
                        "covariate matrix); remove the covariates instead."
                    )
                if rank < len(model_covariates):
                    covariates_dropped = [
                        name
                        for i, name in enumerate(model_covariates)
                        if i not in set(keep_idx.tolist())
                    ]
                    # R's warning is a generic "Multicollinearity issue
                    # detected in covs." - naming the dropped columns is a
                    # documented enhancement.
                    warnings.warn(
                        "Multicollinearity detected in covariates: "
                        f"dropped redundant column(s) {covariates_dropped} "
                        "(covs_drop=True; set covs_drop=False for a strict "
                        "error instead).",
                        UserWarning,
                        stacklevel=2,
                    )
                    model_covariates = [
                        name
                        for i, name in enumerate(model_covariates)
                        if i in set(keep_idx.tolist())
                    ]
                    z = z[:, keep_idx]

        # --- Fuzzy identification check (rdrobust.R:164-185) ---
        # Hoisted to run immediately after the NaN drop and BEFORE
        # mass-point detection, matching R's rdrobust ordering exactly
        # (live-verified: R raises this with NO mass-point warning on
        # degenerate fuzzy + tied data). The port re-checks defensively.
        if t is not None:
            _fuzzy_identification_stop(t[x < c], t[x >= c])

        # --- Mass points (rdrobust.R:365-380) ---
        # R's rdrobust() runs this detection ITSELF, before the manual-vs-
        # data-driven bandwidth branch, so the warning fires on manual-h
        # fits too (verified against installed 4.0.0). The port's
        # rdbwselect-level copy is silenced below (warn_masspoints=False)
        # to mirror R's single warning from the estimation call.
        n_left_pre = int(np.sum(x < c))
        n_right_pre = int(np.sum(x >= c))
        n_unique_left = int(np.unique(x[x < c]).shape[0])
        n_unique_right = int(np.unique(x[x >= c]).shape[0])
        if self.masspoints in ("check", "adjust") and n_left_pre > 0 and n_right_pre > 0:
            mass_l = 1.0 - n_unique_left / n_left_pre
            mass_r = 1.0 - n_unique_right / n_right_pre
            if mass_l >= 0.2 or mass_r >= 0.2:
                warnings.warn(
                    "Mass points detected in the running variable.",
                    UserWarning,
                    stacklevel=2,
                )
                if self.masspoints == "check":
                    warnings.warn(
                        "Try using option masspoints='adjust'.",
                        UserWarning,
                        stacklevel=2,
                    )

        # --- Bandwidth resolution (rdrobust.R:295-307, 501-504) ---
        h_user, b_user, rho = self.h, self.b, self.rho
        bwselect_label = self.bwselect
        if b_user is not None and h_user is None:
            # R silently ignores b without h; we warn (documented deviation).
            warnings.warn(
                "b= was supplied without h= and is ignored (matching "
                "rdrobust's behavior); supply h= to use a manual bias "
                "bandwidth.",
                UserWarning,
                stacklevel=2,
            )
            b_user = None
        if N < 20:
            # rdrobust.R:303-307: unconditional override, INCLUDING a
            # user-supplied h (the block runs after manual-h resolution).
            warnings.warn(
                "Not enough observations to perform bandwidth calculations. "
                "Estimates computed using entire sample.",
                UserWarning,
                stacklevel=2,
            )
            x_min, x_max = float(np.min(x)), float(np.max(x))
            full = max(abs(c - x_min), abs(c - x_max))
            h_l = h_r = b_l = b_r = full
            bwselect_label = "Manual"
        elif h_user is not None:
            bwselect_label = "Manual"
            if rho is None and b_user is None:
                b_resolved = h_user  # rho = 1 (rdrobust.R:296-299)
            elif rho is not None:
                if b_user is not None:
                    warnings.warn(
                        "Both b= and rho= supplied with h=; rho takes "
                        "precedence (b = h/rho), matching rdrobust.",
                        UserWarning,
                        stacklevel=2,
                    )
                b_resolved = h_user / rho  # rdrobust.R:300
            else:
                # rho is None and b_user is not None (first branch handled
                # the both-None case).
                assert b_user is not None
                b_resolved = b_user
            h_l = h_r = float(h_user)
            b_l = b_r = float(b_resolved)
        else:
            bw = rdbwselect(
                y,
                x,
                c=c,
                p=p,
                q=q,
                kernel=kernel,
                vce=self.vcov_type,
                nnmatch=int(self.nnmatch),
                masspoints=self.masspoints,
                bwcheck=None if self.bwcheck is None else int(self.bwcheck),
                bwrestrict=bool(self.bwrestrict),
                scaleregul=float(self.scaleregul),
                warn_masspoints=False,  # fit() already warned (rdrobust.R:365-380)
                fuzzy=t,
                sharpbw=bool(self.sharpbw),
                covs=z,
                covs_drop=bool(self.covs_drop),
            )
            h_l, h_r, b_l, b_r = bw.bws[self.bwselect]
            n_unique_left = bw.M_l if self.masspoints != "off" else n_unique_left
            n_unique_right = bw.M_r if self.masspoints != "off" else n_unique_right
            if rho is not None:
                # rdrobust.R:501-504: rho applies to the SELECTED bandwidths.
                b_l = h_l / rho
                b_r = h_r / rho

        # --- Estimation (port validates h/b finite and positive) ---
        fit = rdrobust_fit(
            y,
            x,
            c,
            h_l,
            h_r,
            b_l,
            b_r,
            p=p,
            q=q,
            kernel=kernel,
            vce=self.vcov_type,
            nnmatch=int(self.nnmatch),
            t=t,
            covs=z,
            covs_drop=bool(self.covs_drop),
            # The estimator owns the degeneracy warning (with column
            # names) - same plumbing pattern as warn_masspoints.
            warn_covs_degenerate=False,
        )

        # --- Degenerate covariate adjustment warning (estimator-level,
        # with column names; deviation from R, which silently inverts a
        # noise singular value on these systems - see the port's
        # _covs_gamma for the guard) ---
        if model_covariates is not None and fit.covs_excluded is not None:
            excluded_names = [
                name for name, flag in zip(model_covariates, fit.covs_excluded) if bool(flag)
            ]
            parts = []
            if excluded_names:
                parts.append(
                    f"covariate(s) {excluded_names} are numerically "
                    "collinear with the local polynomial design (e.g. "
                    "constant near the cutoff) and were excluded from "
                    "the adjustment"
                )
            if fit.covs_set_degenerate:
                parts.append(
                    "the covariate set is numerically rank-deficient "
                    "after partialling (e.g. a full dummy set); a "
                    "stabilized pseudo-inverse cut was used - consider "
                    "dropping a reference category"
                )
            if parts:
                warnings.warn(
                    "Degenerate covariate adjustment: " + "; ".join(parts) + ".",
                    UserWarning,
                    stacklevel=2,
                )

        # Estimand label: the complier-LATE reading requires BINARY
        # take-up (plus monotonicity); non-binary (dose) take-up - accepted,
        # matching R's fuzzy= - is the ratio-of-jumps estimand and must not
        # be labeled a LATE (the label is what `att` claims to measure).
        if not fuzzy_fit:
            estimand = "sharp (ATE at the cutoff)"
        elif t is not None and bool(np.all(np.isin(t, (0.0, 1.0)))):
            estimand = "fuzzy (LATE for compliers at the cutoff)"
        else:
            estimand = "fuzzy (local Wald ratio at the cutoff; non-binary take-up)"

        alpha = float(self.alpha)
        # Three inference rows (rdrobust.R:854-863), each through the
        # library's joint-NaN gate:
        t_rb, p_rb, ci_rb = safe_inference(fit.tau_bc, fit.se_rb, alpha=alpha)
        t_cl, p_cl, ci_cl = safe_inference(fit.tau_cl, fit.se_cl, alpha=alpha)
        t_bcm, p_bcm, ci_bcm = safe_inference(fit.tau_bc, fit.se_cl, alpha=alpha)

        # --- First-stage rows + weak-identification warning (fuzzy) ---
        fs: Dict[str, Any] = {}
        if fuzzy_fit:
            assert fit.tau_T_bc is not None and fit.tau_T_cl is not None
            assert fit.se_T_rb is not None and fit.se_T_cl is not None
            fs_t_rb, fs_p_rb, fs_ci_rb = safe_inference(fit.tau_T_bc, fit.se_T_rb, alpha=alpha)
            fs_t_cl, fs_p_cl, fs_ci_cl = safe_inference(fit.tau_T_cl, fit.se_T_cl, alpha=alpha)
            fs_t_bcm, fs_p_bcm, fs_ci_bcm = safe_inference(fit.tau_T_bc, fit.se_T_cl, alpha=alpha)
            fs = dict(
                first_stage=fit.tau_T_bc,
                first_stage_se=fit.se_T_rb,
                first_stage_t_stat=fs_t_rb,
                first_stage_p_value=fs_p_rb,
                first_stage_conf_int=fs_ci_rb,
                first_stage_conventional=fit.tau_T_cl,
                first_stage_se_conventional=fit.se_T_cl,
                first_stage_t_stat_conventional=fs_t_cl,
                first_stage_p_value_conventional=fs_p_cl,
                first_stage_conf_int_conventional=fs_ci_cl,
                first_stage_t_stat_bias_corrected=fs_t_bcm,
                first_stage_p_value_bias_corrected=fs_p_bcm,
                first_stage_conf_int_bias_corrected=fs_ci_bcm,
                beta_t_p_left=fit.beta_t_p_l,
                beta_t_p_right=fit.beta_t_p_r,
            )
            # Deviation from R (verified silent): warn when the take-up
            # jump is not distinguishable from zero at the fit's own alpha
            # - the ratio is then unreliable (CCT 2014 Theorem 3 requires
            # tau_T != 0; weak-IV-robust inference per Feir-Lemieux-Marmer
            # is a documented seam). Gate: FINITE robust CI containing 0,
            # so NaN-gated first stages (e.g. perfect compliance's se=0)
            # correctly do not fire.
            if (
                np.isfinite(fs_ci_rb[0])
                and np.isfinite(fs_ci_rb[1])
                and fs_ci_rb[0] <= 0.0 <= fs_ci_rb[1]
            ):
                warnings.warn(
                    "Weak first stage: the take-up discontinuity "
                    f"({fit.tau_T_bc:.4g}, robust CI [{fs_ci_rb[0]:.4g}, "
                    f"{fs_ci_rb[1]:.4g}]) is not distinguishable from zero "
                    f"at alpha={alpha:g}; the fuzzy (ratio) estimates are "
                    "unreliable under weak identification.",
                    UserWarning,
                    stacklevel=2,
                )

        return RegressionDiscontinuityResults(
            att=fit.tau_bc,
            se=fit.se_rb,
            t_stat=t_rb,
            p_value=p_rb,
            conf_int=ci_rb,
            alpha=alpha,
            att_conventional=fit.tau_cl,
            se_conventional=fit.se_cl,
            t_stat_conventional=t_cl,
            p_value_conventional=p_cl,
            conf_int_conventional=ci_cl,
            t_stat_bias_corrected=t_bcm,
            p_value_bias_corrected=p_bcm,
            conf_int_bias_corrected=ci_bcm,
            se_robust=fit.se_rb,
            h_left=float(h_l),
            h_right=float(h_r),
            b_left=float(b_l),
            b_right=float(b_r),
            n_obs=N,
            n_left=int(np.sum(x < c)),
            n_right=int(np.sum(x >= c)),
            n_h_left=fit.N_h_l,
            n_h_right=fit.N_h_r,
            n_b_left=fit.N_b_l,
            n_b_right=fit.N_b_r,
            n_unique_left=n_unique_left,
            n_unique_right=n_unique_right,
            n_dropped=n_dropped,
            cutoff=c,
            p=p,
            q=q,
            kernel=kernel,
            bwselect=bwselect_label,
            vcov_type=self.vcov_type,
            nnmatch=int(self.nnmatch),
            masspoints=self.masspoints,
            bwcheck=None if self.bwcheck is None else int(self.bwcheck),
            bwrestrict=bool(self.bwrestrict),
            scaleregul=float(self.scaleregul),
            h_input=None if self.h is None else float(self.h),
            b_input=None if self.b is None else float(self.b),
            rho_input=None if self.rho is None else float(self.rho),
            estimand=estimand,
            sharpbw=bool(self.sharpbw),
            takeup=treatment_col,
            covs_drop=bool(self.covs_drop),
            covariates=None if covariates is None else list(covariates),
            covariates_dropped=covariates_dropped,
            covariate_coefficients=(
                None
                if model_covariates is None or fit.gamma_p is None
                else {name: float(fit.gamma_p[i, 0]) for i, name in enumerate(model_covariates)}
            ),
            first_stage_covariate_coefficients=(
                None
                if not fuzzy_fit or model_covariates is None or fit.gamma_p is None
                else {name: float(fit.gamma_p[i, 1]) for i, name in enumerate(model_covariates)}
            ),
            beta_p_left=fit.beta_p_l,
            beta_p_right=fit.beta_p_r,
            **fs,
        )
