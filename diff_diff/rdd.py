"""
Sharp regression discontinuity design (RDD) estimation with robust
bias-corrected inference, parity-targeting R ``rdrobust`` 4.0.0.

Implements the local-polynomial sharp-RD estimator of Calonico, Cattaneo &
Titiunik (2014): treatment is assigned by ``running >= cutoff``; the effect
is the jump in the conditional expectation of the outcome at the cutoff,
estimated by kernel-weighted polynomial regressions on each side with
data-driven MSE/CER-optimal bandwidths, and reported with robust
bias-corrected (RBC) inference.

Canonical inference binding
---------------------------
``RegressionDiscontinuityResults`` binds the library-canonical fields to ONE
internally coherent inference row - the ROBUST row of rdrobust's output:
``att`` is the bias-corrected point estimate ``tau_bc``, ``se`` its robust
standard error, and ``t_stat``/``p_value``/``conf_int`` are computed from
that same pair, so the library-wide identities hold (``t_stat == att/se``,
``conf_int`` centered on ``att``). This deliberately differs from rdrobust's
PRINTED headline, which reports the conventional estimate ``tau_cl`` in the
coefficient column while taking inference from the robust row; ``tau_cl`` is
first-class here as ``att_conventional`` (with its own full inference row),
and ``summary()`` prints the familiar three-row rdrobust table.

rdrobust equivalents
--------------------
=====================  ==========================================
diff-diff              R rdrobust
=====================  ==========================================
``cutoff``             ``c``
``vcov_type``          ``vce``
``alpha``              ``1 - level/100``
``h``, ``b``, ``rho``  ``h``, ``b``, ``rho`` (same semantics)
``p``, ``q``           ``p``, ``q``
``bwselect``           ``bwselect`` (same 10-option menu)
``kernel``             ``kernel`` (accepts "tri"/"epa"/"uni" too)
``masspoints``         ``masspoints`` ("adjust"/"check"/"off")
``nnmatch``            ``nnmatch``
=====================  ==========================================

Not in v1 (documented seams, see REGISTRY.md): fuzzy designs, covariate
adjustment, cluster-robust variance, weights, ``deriv``/kink estimands,
``scalepar``, ``stdvars``, hc0-hc3 variance modes.

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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._rdrobust_port import (
    BWSELECT_OPTIONS,
    _normalize_kernel,
    rdbwselect_sharp,
    rdrobust_fit_sharp,
)
from diff_diff.utils import safe_inference

__all__ = [
    "RegressionDiscontinuity",
    "RegressionDiscontinuityResults",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class RegressionDiscontinuityResults:
    """Results of a sharp regression discontinuity fit.

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

    # Per-side order-p coefficient vectors (rdplot seam); always populated
    # by fit(), so typed non-Optional despite the dataclass default.
    beta_p_left: np.ndarray = field(repr=False, default=None)
    beta_p_right: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        """Human-readable summary with the three-row rdrobust table."""
        width = 72
        conf_level = 100 * (1 - self.alpha)
        lines = []
        lines.append("=" * width)
        lines.append("Sharp Regression Discontinuity (rdrobust parity)".center(width))
        lines.append("=" * width)
        lines.append(f"Cutoff:               {self.cutoff:g}")
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
            f"{'P>|z|':>9}{'[' + f'{conf_level:g}% Conf. Int.]':>16}"
        )
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
        }
        return {k: _json_safe(v) for k, v in out.items()}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class RegressionDiscontinuity:
    """Sharp regression discontinuity estimator (rdrobust 4.0.0 parity).

    Treatment is defined by the running variable crossing a known cutoff
    (``running >= cutoff`` treated, matching rdrobust: units exactly at the
    cutoff are treated). Point estimation uses kernel-weighted local
    polynomials of order ``p`` on each side; inference is robust
    bias-corrected per Calonico, Cattaneo & Titiunik (2014). Defaults
    reproduce ``rdrobust(y, x)``: ``p=1``, ``q=2``, triangular kernel,
    ``bwselect="mserd"``, nearest-neighbor variance with 3 matches,
    ``masspoints="adjust"``.

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
    alpha : float, default 0.05
        Significance level (rdrobust ``level = 100*(1-alpha)``).

    Examples
    --------
    >>> rd = RegressionDiscontinuity(cutoff=0.0)
    >>> results = rd.fit(df, outcome_col="y", running_col="x")
    >>> results.att, results.conf_int  # robust bias-corrected inference
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
        if not (
            self._is_real_scalar(self.scaleregul)
            and np.isfinite(self.scaleregul)
            and self.scaleregul >= 0
        ):
            raise ValueError(f"scaleregul must be finite and >= 0; got {self.scaleregul!r}.")
        if not (self._is_real_scalar(self.alpha) and 0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return raw constructor parameters (sklearn-compatible)."""
        del deep
        return {
            "cutoff": self.cutoff,
            "p": self.p,
            "q": self.q,
            "kernel": self.kernel,
            "bwselect": self.bwselect,
            "h": self.h,
            "b": self.b,
            "rho": self.rho,
            "vcov_type": self.vcov_type,
            "nnmatch": self.nnmatch,
            "masspoints": self.masspoints,
            "bwcheck": self.bwcheck,
            "bwrestrict": self.bwrestrict,
            "scaleregul": self.scaleregul,
            "alpha": self.alpha,
        }

    def set_params(self, **params: Any) -> "RegressionDiscontinuity":
        """Transactionally update parameters (validate before mutating)."""
        valid = set(self.get_params().keys())
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}. Valid: {sorted(valid)}.")
        merged = self.get_params()
        merged.update(params)
        type(self)(**merged)  # dry-run: raises before any mutation
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        running_col: str,
    ) -> RegressionDiscontinuityResults:
        """Estimate the sharp RD effect at the cutoff.

        Parameters
        ----------
        data : pd.DataFrame
            Cross-sectional data. Treatment is derived as
            ``running >= cutoff`` (no treatment column - sharp design).
        outcome_col, running_col : str
            Column names of the outcome and the running variable.
        """
        for col in (outcome_col, running_col):
            if col not in data.columns:
                raise ValueError(f"Column {col!r} not found in data.")
        y_raw = np.asarray(pd.to_numeric(data[outcome_col], errors="coerce"), dtype=np.float64)
        x_raw = np.asarray(pd.to_numeric(data[running_col], errors="coerce"), dtype=np.float64)
        ok = np.isfinite(y_raw) & np.isfinite(x_raw)
        n_dropped = int(y_raw.shape[0] - np.sum(ok))
        if n_dropped > 0:
            # Deviation from R (which drops silently via complete.cases):
            warnings.warn(
                f"Dropping {n_dropped} row(s) with missing or non-numeric "
                f"values in {outcome_col!r}/{running_col!r}.",
                UserWarning,
                stacklevel=2,
            )
        y = y_raw[ok]
        x = x_raw[ok]
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
            bw = rdbwselect_sharp(
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
            )
            h_l, h_r, b_l, b_r = bw.bws[self.bwselect]
            n_unique_left = bw.M_l if self.masspoints != "off" else n_unique_left
            n_unique_right = bw.M_r if self.masspoints != "off" else n_unique_right
            if rho is not None:
                # rdrobust.R:501-504: rho applies to the SELECTED bandwidths.
                b_l = h_l / rho
                b_r = h_r / rho

        # --- Estimation (port validates h/b finite and positive) ---
        fit = rdrobust_fit_sharp(
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
        )

        alpha = float(self.alpha)
        # Three inference rows (rdrobust.R:854-863), each through the
        # library's joint-NaN gate:
        t_rb, p_rb, ci_rb = safe_inference(fit.tau_bc, fit.se_rb, alpha=alpha)
        t_cl, p_cl, ci_cl = safe_inference(fit.tau_cl, fit.se_cl, alpha=alpha)
        t_bcm, p_bcm, ci_bcm = safe_inference(fit.tau_bc, fit.se_cl, alpha=alpha)

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
            beta_p_left=fit.beta_p_l,
            beta_p_right=fit.beta_p_r,
        )
