"""Pre-test diagnostics for the HeterogeneousAdoptionDiD estimator.

Paper Section 4 (de Chaisemartin, Ciccia, D'Haultfoeuille, Knau 2026,
arXiv:2405.04465v6) prescribes a four-step pre-testing workflow for TWFE
validity in HADs. This module ships the tests and the composite workflow:

Single-horizon tests:

1. :func:`qug_test` - order-statistic ratio test of the support infimum
   ``H_0: d_lower = 0`` (paper Theorem 4). Closed-form, tuning-free.
2. :func:`stute_test` - Cramer-von Mises cusum test of linearity of
   ``E[ΔY | D_2]`` with Mammen (1993) wild bootstrap p-value (paper
   Appendix D).
3. :func:`yatchew_hr_test` - heteroskedasticity-robust variance-ratio
   specification test (paper Theorem 7 / Equation 29). Feasible at
   ``G >= 100k``. Two nulls via the keyword-only ``null=`` argument:
   ``"linearity"`` (default; paper Theorem 7, fits ``Y ~ 1 + D``) and
   ``"mean_independence"`` (R-parity extension mirroring R
   ``YatchewTest::yatchew_test(order=0)``; fits ``Y ~ 1``). The
   downstream variance-ratio machinery is shared between the two
   modes — only the residual definition differs.

Joint / multi-period tests (Phase 3 follow-up):

4. :func:`stute_joint_pretest` - residuals-in core that generalizes the
   single-horizon Stute CvM to K horizons with shared-η wild bootstrap
   and sum-of-CvMs aggregation (Delgado 1993; Escanciano 2006).
5. :func:`joint_pretrends_test` - data-in wrapper for the mean-
   independence null (paper step 2 pre-trends across pre-period
   placebos, Section 4.2 footnote 6 + Section 4.3 paragraph 1).
6. :func:`joint_homogeneity_test` - data-in wrapper for the linearity
   null across post-periods (paper Section 4.3 joint extension,
   page 32).

Composite workflow:

:func:`did_had_pretest_workflow` has two dispatch modes:

- ``aggregate="overall"`` (default, two-period panel): runs steps 1 + 3
  via :func:`qug_test` + :func:`stute_test` + :func:`yatchew_hr_test`.
  Paper step 2 is NOT run on this path (a two-period panel has no pre-
  period placebo); the verdict explicitly flags the Assumption 7 gap
  via the ``"paper step 2 deferred"`` caveat so callers do not get an
  unconditional "TWFE safe" signal.
- ``aggregate="event_study"`` (multi-period panel, >= 3 periods): runs
  QUG at ``F`` + joint pre-trends Stute across earlier pre-periods +
  joint homogeneity-linearity Stute across post-periods. Closes the
  paper step-2 gap and does NOT emit the step-2-deferred caveat in the
  verdict when at least one earlier pre-period is available. The
  step-3 alternative (Yatchew-HR linearity) is subsumed by joint Stute
  on this path; the paper does not derive a joint Yatchew variant, so
  users who need Yatchew robustness under multi-period data can call
  :func:`yatchew_hr_test` on each ``(base, post)`` pair manually.
  (Step 4 in the paper's workflow is the decision itself - "use TWFE
  if none of the tests rejects" - not a separate test.)

Eq. 17 / Eq. 18 linear-trend detrending (paper Section 5.2 Pierce-Schott
application) shipped in PR #389 (Phase 4 R-parity) as the
``trends_lin: bool = False`` keyword-only kwarg on
:func:`joint_pretrends_test`, :func:`joint_homogeneity_test`, AND
:meth:`HeterogeneousAdoptionDiD.fit` (event-study path). Mirrors R
``DIDHAD::did_had(..., trends_lin=TRUE)``. Survey-weighted variant is
not yet derived from the paper and raises ``NotImplementedError``;
tracked in ``DEFERRED.md`` if user demand emerges. See
``docs/methodology/REGISTRY.md`` for the full algorithm narrative,
invariants, and deviation notes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff._deprecation import NOT_SUPPLIED, require_arg, resolve_renamed_kwarg
from diff_diff.bootstrap_utils import (
    apply_stratum_centering,
    generate_survey_multiplier_weights_batch,
)
from diff_diff.had import (
    _aggregate_first_difference,
    _aggregate_unit_resolved_survey,
    _aggregate_unit_weights,
    _json_safe_scalar,
    _validate_had_panel,
    _validate_had_panel_event_study,
)
from diff_diff.results_base import Diagnostic
from diff_diff.survey import (
    SurveyDesign,
    make_pweight_design,
)
from diff_diff.utils import _generate_mammen_weights

__all__ = [
    "QUGTestResults",
    "StuteTestResults",
    "YatchewTestResults",
    "StuteJointResult",
    "HADPretestReport",
    "qug_test",
    "stute_test",
    "yatchew_hr_test",
    "stute_joint_pretest",
    "joint_pretrends_test",
    "joint_homogeneity_test",
    "did_had_pretest_workflow",
]


_MIN_G_QUG = 2
_MIN_G_STUTE = 10
_MIN_G_YATCHEW = 3
_MIN_N_BOOTSTRAP = 99
_STUTE_LARGE_G_THRESHOLD = 100_000


# Scale-invariant tolerance for detecting a numerically exact linear OLS fit.
# The ratio SSR / TSS = sum(eps^2) / sum((dy - dybar)^2) equals 1 - R^2
# and is BOTH TRANSLATION-INVARIANT (centering absorbs additive shifts)
# and SCALE-INVARIANT (the ratio is dimensionless under multiplicative
# rescaling of dy). Under exact Assumption 8, residuals are mathematically
# zero; in practice FP round-off leaves eps on the order of machine-epsilon
# (~1e-16). Squared that is ~1e-32. The threshold ~1e-24 leaves ~10^8
# accumulated FP operations of margin so genuinely-noisy data is never
# mis-classified.
#
# IMPORTANT: the comparison is purely ``eps^2 <= tol * dy_centered^2`` with
# NO additive floor (e.g. ``max(dy_centered^2, 1.0)`` would break scale
# invariance - scaling dy by 1e-12 would make dy_centered^2 ~ 1e-24 but
# the floor would hold the threshold at 1.0, firing the shortcut on
# noisy data that should not trigger it). The ``dy_centered^2 == 0``
# edge case (constant dy) is handled by a separate branch above the
# relative comparison, so the relative form is only applied when the
# denominator is genuinely positive.
_EXACT_LINEAR_RELATIVE_TOL = 1e-24


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class QUGTestResults(Diagnostic):
    """Result of :func:`qug_test` (paper Theorem 4).

    The QUG test rejects ``H_0: d_lower = 0`` when the order-statistic
    ratio ``T = D_{(1)} / (D_{(2)} - D_{(1)})`` exceeds ``1/alpha - 1``.
    Under the null, the asymptotic limit law of ``T`` is the ratio of two
    independent Exp(1) random variables, with CDF ``F(t) = t / (1 + t)``,
    so ``p_value = 1 / (1 + T)``.

    Attributes
    ----------
    t_stat : float
        ``D_{(1)} / (D_{(2)} - D_{(1)})``. NaN when fewer than 2 non-zero
        observations remain or when the two smallest doses tie.
    p_value : float
        ``1 / (1 + t_stat)`` under the null. NaN when ``t_stat`` is NaN.
    reject : bool
        ``True`` iff ``t_stat > critical_value``. ``False`` on NaN statistic.
    alpha : float
        Significance level used.
    critical_value : float
        ``1 / alpha - 1``. Populated even when the statistic is NaN so
        downstream readers can inspect the decision threshold.
    n_obs : int
        Number of observations after filtering to ``d > 0``.
    n_excluded_zero : int
        Number of zero-dose observations excluded from the sample.
    d_order_1 : float
        Smallest positive dose ``D_{(1)}``. NaN when ``n_obs < 2``.
    d_order_2 : float
        Second-smallest positive dose ``D_{(2)}``. NaN when ``n_obs < 2``.
    """

    t_stat: float
    p_value: float
    reject: bool
    alpha: float
    critical_value: float
    n_obs: int
    n_excluded_zero: int
    d_order_1: float
    d_order_2: float

    def __repr__(self) -> str:
        return (
            f"QUGTestResults(t_stat={self.t_stat:.4f}, p_value={self.p_value:.4f}, "
            f"reject={self.reject}, alpha={self.alpha}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        lines = [
            "=" * width,
            "QUG null test (H_0: d_lower = 0)".center(width),
            "=" * width,
            f"{'Statistic T:':<30} {self.t_stat:>20.4f}",
            f"{'p-value:':<30} {self.p_value:>20.4f}",
            f"{'Critical value (1/alpha-1):':<30} {self.critical_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Excluded (d == 0):':<30} {self.n_excluded_zero:>20}",
            f"{'D_(1):':<30} {self.d_order_1:>20.4f}",
            f"{'D_(2):':<30} {self.d_order_2:>20.4f}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "qug",
            "t_stat": _json_safe_scalar(self.t_stat),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "critical_value": _json_safe_scalar(self.critical_value),
            "n_obs": int(self.n_obs),
            "n_excluded_zero": int(self.n_excluded_zero),
            "d_order_1": _json_safe_scalar(self.d_order_1),
            "d_order_2": _json_safe_scalar(self.d_order_2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class StuteTestResults(Diagnostic):
    """Result of :func:`stute_test` (paper Appendix D).

    The Stute test rejects the null that ``E[ΔY | D_2]`` is linear in
    ``D_2`` (paper Assumption 8) when the sorted-residual CvM statistic
    ``S = (1/G^2) Σ (Σ_{h=1}^g eps_{(h)})^2`` exceeds the Mammen wild
    bootstrap ``1 - alpha`` quantile.

    Attributes
    ----------
    cvm_stat : float
        CvM statistic. NaN when ``G < 10`` (below the threshold the
        statistic is not well-calibrated).
    p_value : float
        Bootstrap p-value ``(1 + sum(S_b >= S)) / (B + 1)``. NaN when
        the statistic is NaN.
    reject : bool
        ``True`` iff ``p_value <= alpha``. ``False`` on NaN.
    alpha : float
        Significance level used.
    n_bootstrap : int
        Number of Mammen wild bootstrap replications.
    n_obs : int
        Number of observations.
    seed : int or None
        Seed passed to ``np.random.default_rng``. ``None`` when unseeded.
    """

    cvm_stat: float
    p_value: float
    reject: bool
    alpha: float
    n_bootstrap: int
    n_obs: int
    seed: Optional[int]

    def __repr__(self) -> str:
        return (
            f"StuteTestResults(cvm_stat={self.cvm_stat:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"alpha={self.alpha}, n_bootstrap={self.n_bootstrap}, "
            f"n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        lines = [
            "=" * width,
            "Stute CvM linearity test (H_0: linear E[dY|D])".center(width),
            "=" * width,
            f"{'CvM statistic:':<30} {self.cvm_stat:>20.4f}",
            f"{'Bootstrap p-value:':<30} {self.p_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Bootstrap replications:':<30} {self.n_bootstrap:>20}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Seed:':<30} {str(self.seed):>20}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "stute",
            "cvm_stat": _json_safe_scalar(self.cvm_stat),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "n_bootstrap": int(self.n_bootstrap),
            "n_obs": int(self.n_obs),
            "seed": None if self.seed is None else int(self.seed),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class YatchewTestResults(Diagnostic):
    """Result of :func:`yatchew_hr_test` (paper Theorem 7 / Equation 29).

    Heteroskedasticity-robust specification test using Yatchew's
    difference-based variance estimator. Two nulls are supported via
    the ``null=`` argument on :func:`yatchew_hr_test` and reflected on
    the ``null_form`` attribute below: ``"linearity"`` (default; paper
    Theorem 7, the same null as :func:`stute_test`, residuals from OLS
    ``dy ~ 1 + d``) and ``"mean_independence"`` (R-parity extension
    mirroring R ``YatchewTest::yatchew_test(order=0)``, residuals from
    intercept-only OLS ``dy ~ 1``). The test statistic
    ``T_hr = sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W`` is
    asymptotically N(0, 1) under H_0 in both modes; rejection uses the
    one-sided standard-normal critical value. Only the residual
    definition (and therefore ``sigma2_lin``) differs between modes —
    the ``sigma2_diff`` / ``sigma2_W`` / sort-by-``d`` machinery is
    shared.

    Attributes
    ----------
    t_stat_hr : float
        Test statistic ``T_hr`` from paper Equation 29. NaN when
        ``G < 3``.
    p_value : float
        ``1 - Phi(T_hr)``. NaN when the statistic is NaN.
    reject : bool
        ``True`` iff ``T_hr >= critical_value``. ``False`` on NaN.
    alpha : float
        Significance level used.
    critical_value : float
        One-sided standard-normal critical value ``z_{1 - alpha}``.
    sigma2_lin : float
        Residual variance under the chosen null. Under
        ``null_form="linearity"``: residual variance from OLS of ``dy``
        on ``d``. Under ``null_form="mean_independence"``: ``(1/G) *
        sum((dy - mean(dy))^2)``, the population variance of ``dy``.
    sigma2_diff : float
        Yatchew differencing variance
        ``(1 / (2G)) * sum((dy_{(g)} - dy_{(g-1)})^2)`` - divisor is ``2G``
        (paper-literal), NOT ``2(G-1)``.
    sigma2_W : float
        Heteroskedasticity-robust scale
        ``sqrt((1 / (G-1)) * sum(eps_{(g)}^2 * eps_{(g-1)}^2))``.
    n_obs : int
        Number of observations.
    null_form : str
        ``"linearity"`` (default; H_0: ``E[dY|D]`` is linear in ``D``,
        residuals from OLS ``dy ~ 1 + d``) or ``"mean_independence"``
        (H_0: ``E[dY|D] = E[dY]``, residuals from intercept-only OLS
        ``dy ~ 1``). Mirrors R ``YatchewTest::yatchew_test``'s
        ``order`` argument (``order=1`` ↔ ``"linearity"``; ``order=0``
        ↔ ``"mean_independence"``).
    """

    t_stat_hr: float
    p_value: float
    reject: bool
    alpha: float
    critical_value: float
    sigma2_lin: float
    sigma2_diff: float
    sigma2_W: float
    n_obs: int
    null_form: str = "linearity"

    def __repr__(self) -> str:
        return (
            f"YatchewTestResults(t_stat_hr={self.t_stat_hr:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"alpha={self.alpha}, null_form={self.null_form!r}, "
            f"n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        title = {
            "linearity": "Yatchew-HR linearity test (H_0: linear E[dY|D])",
            "mean_independence": ("Yatchew-HR mean-independence test (H_0: E[dY|D] = E[dY])"),
        }.get(self.null_form, f"Yatchew-HR test (null_form={self.null_form!r})")
        lines = [
            "=" * width,
            title.center(width),
            "=" * width,
            f"{'T_hr statistic:':<30} {self.t_stat_hr:>20.4f}",
            f"{'p-value:':<30} {self.p_value:>20.4f}",
            f"{'Critical value (1-sided z):':<30} {self.critical_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'sigma^2_lin (OLS):':<30} {self.sigma2_lin:>20.4f}",
            f"{'sigma^2_diff (Yatchew):':<30} {self.sigma2_diff:>20.4f}",
            f"{'sigma^2_W (HR scale):':<30} {self.sigma2_W:>20.4f}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "yatchew_hr",
            "t_stat_hr": _json_safe_scalar(self.t_stat_hr),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "critical_value": _json_safe_scalar(self.critical_value),
            "sigma2_lin": _json_safe_scalar(self.sigma2_lin),
            "sigma2_diff": _json_safe_scalar(self.sigma2_diff),
            "sigma2_W": _json_safe_scalar(self.sigma2_W),
            "n_obs": int(self.n_obs),
            "null_form": str(self.null_form),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class StuteJointResult(Diagnostic):
    """Result of :func:`stute_joint_pretest` (joint Cramer-von Mises across horizons).

    Aggregates the per-horizon Stute (1997) CvM statistic into a joint
    specification test: ``S_joint = sum_k S_k``, where ``S_k`` is the
    single-horizon CvM on residuals ``eps_{g,k}``. Inference is via
    Mammen (1993) wild bootstrap with a **shared** multiplier ``eta_g``
    across horizons per unit (Delgado-Manteiga 2001; Hlavka-Huskova 2020)
    to preserve the unit-level dependence structure of the vector-valued
    empirical process.

    Two nulls are supported via the thin wrappers
    :func:`joint_pretrends_test` (mean-independence: ``E[Y_t - Y_base | D]
    = mu_t``, design matrix ``[1]``) and :func:`joint_homogeneity_test`
    (linearity: ``E[Y_t - Y_base | D_t] = beta_{0,t} + beta_{fe,t} * D``,
    design matrix ``[1, D]``). Both wrappers accept a ``trends_lin:
    bool = False`` keyword-only flag (PR #392): when ``True``, applies
    paper Eq 17 / Eq 18 linear-trend detrending before the joint CvM
    using per-group slope ``Y[g, F-1] - Y[g, F-2]``.

    Attributes
    ----------
    cvm_stat_joint : float
        Joint statistic ``S_joint = sum_k S_k``. NaN on NaN-propagation.
    p_value : float
        Bootstrap p-value ``(1 + sum(S*_b >= S_joint)) / (B + 1)``. NaN
        when the statistic is NaN. ``1.0`` when the per-horizon exact-
        linear short-circuit fires (all horizons machine-exact linear).
    reject : bool
        ``True`` iff ``p_value <= alpha``. Always ``False`` on NaN.
    alpha : float
        Significance level.
    horizon_labels : list of str
        Horizon identifiers as ``str(t)`` for each period. **String
        identity only** - NOT a chronological ordering key. Callers who
        need chronological order should preserve the original period
        values alongside (a downstream plotter sorting labels
        lexicographically will misorder e.g.
        ``["2003-Q10", "2003-Q2", ...]``).
    per_horizon_stats : dict[str, float]
        ``{label: S_k}`` diagnostic. Per-horizon p-values are NOT
        exposed (decomposing the joint bootstrap into K independent
        loops is a K-fold memory/time cost; deferred). Callers who need
        per-horizon p-values can call :func:`stute_test` separately on
        each (period, residual) pair.

        On NaN-propagation (any horizon has NaN input), this dict is
        preserved with ``{label: np.nan for label in horizon_labels}``,
        NOT an empty dict, NOT a partial dict: the keys carry diagnostic
        value (which horizons were attempted), the NaN values signal
        non-propagation.
    n_bootstrap : int
    n_obs : int
        Number of units ``G``.
    n_horizons : int
    seed : int or None
    null_form : str
        ``"mean_independence"`` (from :func:`joint_pretrends_test`) or
        ``"linearity"`` (from :func:`joint_homogeneity_test`).
        ``"custom"`` when called directly via :func:`stute_joint_pretest`
        without a wrapper.
    exact_linear_short_circuited : bool
        ``True`` when every horizon's residual SSR to centered TSS ratio
        is below :data:`_EXACT_LINEAR_RELATIVE_TOL`; bootstrap is
        skipped and ``p_value = 1.0``. The per-horizon check ensures a
        single degenerate horizon does not collapse the joint test when
        other horizons have nontrivial residuals.
    """

    cvm_stat_joint: float
    p_value: float
    reject: bool
    alpha: float
    horizon_labels: list
    per_horizon_stats: Dict[str, float]
    n_bootstrap: int
    n_obs: int
    n_horizons: int
    seed: Optional[int]
    null_form: str
    exact_linear_short_circuited: bool

    def __repr__(self) -> str:
        return (
            f"StuteJointResult(cvm_stat_joint={self.cvm_stat_joint:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"n_horizons={self.n_horizons}, null_form={self.null_form!r}, "
            f"n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        per_horizon_lines = [
            f"  {label:<20} {stat:>20.4f}" for label, stat in self.per_horizon_stats.items()
        ]
        null_label = {
            "mean_independence": "mean-independence (pre-trends)",
            "linearity": "linearity (post-homogeneity)",
        }.get(self.null_form, self.null_form)
        lines = [
            "=" * width,
            f"Joint Stute CvM test ({null_label})".center(width),
            "=" * width,
            f"{'Joint CvM statistic:':<30} {self.cvm_stat_joint:>20.4f}",
            f"{'Bootstrap p-value:':<30} {self.p_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Bootstrap replications:':<30} {self.n_bootstrap:>20}",
            f"{'Horizons:':<30} {self.n_horizons:>20}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Seed:':<30} {str(self.seed):>20}",
            f"{'Exact-linear short-circuit:':<30} " f"{str(self.exact_linear_short_circuited):>20}",
            "-" * width,
            "Per-horizon statistics:",
            *per_horizon_lines,
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "stute_joint",
            "cvm_stat_joint": _json_safe_scalar(self.cvm_stat_joint),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "horizon_labels": [str(label) for label in self.horizon_labels],
            "per_horizon_stats": {
                str(k): _json_safe_scalar(v) for k, v in self.per_horizon_stats.items()
            },
            "n_bootstrap": int(self.n_bootstrap),
            "n_obs": int(self.n_obs),
            "n_horizons": int(self.n_horizons),
            "seed": None if self.seed is None else int(self.seed),
            "null_form": str(self.null_form),
            "exact_linear_short_circuited": bool(self.exact_linear_short_circuited),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the top-level result fields."""
        return pd.DataFrame(
            [
                {
                    "test": "stute_joint",
                    "cvm_stat_joint": _json_safe_scalar(self.cvm_stat_joint),
                    "p_value": _json_safe_scalar(self.p_value),
                    "reject": bool(self.reject),
                    "alpha": float(self.alpha),
                    "n_bootstrap": int(self.n_bootstrap),
                    "n_obs": int(self.n_obs),
                    "n_horizons": int(self.n_horizons),
                    "null_form": str(self.null_form),
                }
            ]
        )


@dataclass
class HADPretestReport(Diagnostic):
    """Composite output of :func:`did_had_pretest_workflow`.

    Two dispatch shapes, distinguished by :attr:`aggregate`:

    ``aggregate="overall"`` (default, two-period panel): bundles paper
    steps 1 (QUG) and 3 (linearity via Stute + Yatchew-HR) on a
    two-period first-differenced sample. Step 2 (Assumption 7 pre-trends)
    is NOT implemented on this path and is explicitly flagged in the
    verdict; callers must run pre-trends separately.

    ``aggregate="event_study"`` (multi-period panel, >= 3 periods):
    bundles QUG + joint pre-trends Stute + joint homogeneity-linearity
    Stute. The joint Stute variants close the paper step-2 gap; the
    event-study verdict does NOT emit the "paper step 2 deferred"
    caveat. Step 3 adjudication uses joint Stute only - no joint Yatchew
    variant exists because the paper does not derive one; users who need
    Yatchew robustness under multi-period data can run
    :func:`yatchew_hr_test` on each (base, post) pair manually.

    Attributes
    ----------
    qug : QUGTestResults or None
        Populated by default; ``None`` only when the workflow runs under
        ``survey_design=`` (Phase 4.5 C path), where the QUG step
        is permanently skipped per Phase 4.5 C0 (extreme-value theory under
        complex sampling not a settled toolkit; see :func:`qug_test`).
    stute : StuteTestResults or None
        Populated when ``aggregate == "overall"``; ``None`` when
        ``aggregate == "event_study"``.
    yatchew : YatchewTestResults or None
        Populated when ``aggregate == "overall"``; ``None`` when
        ``aggregate == "event_study"``.
    pretrends_joint : StuteJointResult or None
        Populated when ``aggregate == "event_study"`` and at least one
        earlier pre-period exists; ``None`` on the overall path or when
        only the immediate base pre-period is available.
    homogeneity_joint : StuteJointResult or None
        Populated when ``aggregate == "event_study"``; ``None`` on the
        overall path.
    all_pass : bool
        On the **unweighted overall path**: same Phase 3 semantics - True
        iff QUG is conclusive AND at least one of Stute/Yatchew is
        conclusive AND no conclusive test rejects. On the **unweighted
        event-study path**: True iff ``np.isfinite(qug.p_value)``,
        ``pretrends_joint is not None and
        np.isfinite(pretrends_joint.p_value)``,
        ``np.isfinite(homogeneity_joint.p_value)``, AND none of the
        three rejects. On the **survey_design= path** (Phase 4.5 C) the
        QUG-conclusiveness gate is dropped (``qug=None`` per C0
        deferral); the linearity-conditional rule splits by aggregate:

        - ``aggregate="overall"`` survey: True iff at least one of
          Stute/Yatchew is conclusive AND no conclusive test rejects.
        - ``aggregate="event_study"`` survey: True iff
          ``pretrends_joint`` is non-None and conclusive,
          ``homogeneity_joint`` is conclusive, AND neither rejects.
          (Both joint variants must be conclusive on the event-study
          path - same step-2 + step-3 closure as the unweighted
          aggregate, just without the QUG step.)

        Mirrors Phase 3's ``bool(np.isfinite(p_value))`` convention - no
        ``.conclusive()`` helper on any result dataclass.
    verdict : str
        Human-readable classification. Paper rule applies symmetrically:
        TWFE is admissible only if NONE of the implemented tests
        rejects. Conclusive rejections are the primary verdict;
        unresolved steps append as ``"; additional steps unresolved:
        ..."`` rather than replacing the rejection.
    alpha : float
    n_obs : int
        Unit count. For overall: units after two-period first-difference
        aggregation. For event_study: units after balanced-panel
        validation and (if applicable) last-cohort auto-filter.
    aggregate : str
        ``"overall"`` or ``"event_study"``. Determines which component
        fields are populated and which branch of serialization methods
        to render.
    """

    qug: Optional[QUGTestResults]
    stute: Optional[StuteTestResults]
    yatchew: Optional[YatchewTestResults]
    all_pass: bool
    verdict: str
    alpha: float
    n_obs: int
    pretrends_joint: Optional[StuteJointResult] = None
    homogeneity_joint: Optional[StuteJointResult] = None
    aggregate: str = "overall"

    def __repr__(self) -> str:
        # Preserve Phase 3 repr bit-exactly on the overall path. The
        # aggregate kwarg is only surfaced on the event-study path so
        # downstream consumers comparing repr strings on two-period
        # reports see identical output.
        if self.aggregate == "event_study":
            return (
                f"HADPretestReport(aggregate={self.aggregate!r}, "
                f"all_pass={self.all_pass}, "
                f"verdict={self.verdict!r}, n_obs={self.n_obs})"
            )
        return (
            f"HADPretestReport(all_pass={self.all_pass}, "
            f"verdict={self.verdict!r}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary of all tests and the verdict."""
        width = 72
        # Preserve Phase 3 summary bit-exactly on the overall path. The
        # `aggregate: ...` header line is only rendered on the event-
        # study path; two-period reports produce the Phase 3 layout.
        # QUG block: rendered when self.qug is populated, else a skip note
        # (Phase 4.5 C survey_design= path leaves qug=None; see C0 deferral).
        qug_block = (
            self.qug.summary()
            if self.qug is not None
            else "(QUG step skipped - permanently deferred under survey designs per Phase 4.5 C0)"
        )
        if self.aggregate == "event_study":
            header = [
                "=" * width,
                "HAD pre-test workflow".center(width),
                f"aggregate: {self.aggregate}".center(width),
                "=" * width,
                qug_block,
                "",
            ]
            if self.pretrends_joint is not None:
                body = [self.pretrends_joint.summary(), ""]
            else:
                body = [
                    "(joint pre-trends skipped - no earlier pre-period)",
                    "",
                ]
            if self.homogeneity_joint is not None:
                body += [self.homogeneity_joint.summary(), ""]
        else:
            # aggregate == "overall" - Phase 3 layout preserved when qug is
            # not None (unweighted path); QUG-skip block on the survey path.
            header = [
                "=" * width,
                "HAD pre-test workflow".center(width),
                "=" * width,
                qug_block,
                "",
            ]
            body = []
            if self.stute is not None:
                body += [self.stute.summary(), ""]
            if self.yatchew is not None:
                body += [self.yatchew.summary(), ""]
        footer = [
            "=" * width,
            f"{'All pass:':<30} {str(self.all_pass):>40}",
            f"Verdict: {self.verdict}",
            "=" * width,
        ]
        return "\n".join(header + body + footer)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe nested dict of the full report.

        On ``aggregate="overall"``, the output schema is bit-exact with
        Phase 3 (``{qug, stute, yatchew, all_pass, verdict, alpha,
        n_obs}``) - no new keys, no aggregate field. On
        ``aggregate="event_study"``, the output carries ``aggregate``,
        ``pretrends_joint``, ``homogeneity_joint`` and omits the
        ``None``-valued ``stute`` / ``yatchew`` keys entirely.
        """
        # qug serializes as None on the survey_design= path (Phase 4.5 C
        # QUG-skip per C0 deferral); rendered as the existing dict on the
        # default unweighted path.
        qug_dict = None if self.qug is None else self.qug.to_dict()
        if self.aggregate == "event_study":
            return {
                "aggregate": str(self.aggregate),
                "qug": qug_dict,
                "pretrends_joint": (
                    None if self.pretrends_joint is None else self.pretrends_joint.to_dict()
                ),
                "homogeneity_joint": (
                    None if self.homogeneity_joint is None else self.homogeneity_joint.to_dict()
                ),
                "all_pass": bool(self.all_pass),
                "verdict": str(self.verdict),
                "alpha": float(self.alpha),
                "n_obs": int(self.n_obs),
            }
        # aggregate == "overall" - Phase 3 schema preserved bit-exactly on
        # the unweighted path (qug populated); the qug=None survey path
        # surfaces qug: null.
        return {
            "qug": qug_dict,
            "stute": None if self.stute is None else self.stute.to_dict(),
            "yatchew": None if self.yatchew is None else self.yatchew.to_dict(),
            "all_pass": bool(self.all_pass),
            "verdict": str(self.verdict),
            "alpha": float(self.alpha),
            "n_obs": int(self.n_obs),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy 3-row DataFrame (one row per implemented test).

        Columns (stable across aggregates):
        ``[test, statistic_name, statistic_value, p_value, reject, alpha,
        n_obs]``. Row identifiers vary by aggregate:

        - ``aggregate="overall"``: rows are ``qug``, ``stute``,
          ``yatchew_hr`` (Phase 3 schema, unchanged).
        - ``aggregate="event_study"``: rows are ``qug``,
          ``pretrends_joint``, ``homogeneity_joint``.

        Rows for ``None``-valued components (e.g. ``pretrends_joint`` when
        no earlier pre-period exists) are emitted with NaN statistic
        values and ``reject=False`` to preserve the 3-row shape.
        """
        # qug row: NaN-skip when self.qug is None (Phase 4.5 C survey_design=
        # path leaves qug=None per C0 deferral). Mirrors the joint NaN-row
        # shape from `_joint_row_or_nan` so the 3-row contract is preserved.
        if self.qug is None:
            qug_row = {
                "test": "qug",
                "statistic_name": "t_stat",
                "statistic_value": float("nan"),
                "p_value": float("nan"),
                "reject": False,
                "alpha": float(self.alpha),
                "n_obs": int(self.n_obs),
            }
        else:
            qug_row = {
                "test": "qug",
                "statistic_name": "t_stat",
                "statistic_value": _json_safe_scalar(self.qug.t_stat),
                "p_value": _json_safe_scalar(self.qug.p_value),
                "reject": bool(self.qug.reject),
                "alpha": float(self.qug.alpha),
                "n_obs": int(self.qug.n_obs),
            }
        if self.aggregate == "event_study":
            pre_row = self._joint_row_or_nan("pretrends_joint", self.pretrends_joint)
            hom_row = self._joint_row_or_nan("homogeneity_joint", self.homogeneity_joint)
            rows = [qug_row, pre_row, hom_row]
        else:
            stute_row = (
                {
                    "test": "stute",
                    "statistic_name": "cvm_stat",
                    "statistic_value": _json_safe_scalar(self.stute.cvm_stat),
                    "p_value": _json_safe_scalar(self.stute.p_value),
                    "reject": bool(self.stute.reject),
                    "alpha": float(self.stute.alpha),
                    "n_obs": int(self.stute.n_obs),
                }
                if self.stute is not None
                else {
                    "test": "stute",
                    "statistic_name": "cvm_stat",
                    "statistic_value": float("nan"),
                    "p_value": float("nan"),
                    "reject": False,
                    "alpha": float(self.alpha),
                    "n_obs": int(self.n_obs),
                }
            )
            yatchew_row = (
                {
                    "test": "yatchew_hr",
                    "statistic_name": "t_stat_hr",
                    "statistic_value": _json_safe_scalar(self.yatchew.t_stat_hr),
                    "p_value": _json_safe_scalar(self.yatchew.p_value),
                    "reject": bool(self.yatchew.reject),
                    "alpha": float(self.yatchew.alpha),
                    "n_obs": int(self.yatchew.n_obs),
                }
                if self.yatchew is not None
                else {
                    "test": "yatchew_hr",
                    "statistic_name": "t_stat_hr",
                    "statistic_value": float("nan"),
                    "p_value": float("nan"),
                    "reject": False,
                    "alpha": float(self.alpha),
                    "n_obs": int(self.n_obs),
                }
            )
            rows = [qug_row, stute_row, yatchew_row]
        cols = [
            "test",
            "statistic_name",
            "statistic_value",
            "p_value",
            "reject",
            "alpha",
            "n_obs",
        ]
        return pd.DataFrame(rows).reindex(columns=cols)

    def _joint_row_or_nan(
        self, test_label: str, joint: Optional[StuteJointResult]
    ) -> Dict[str, Any]:
        """Build a to_dataframe row for a joint-Stute component.

        When ``joint`` is ``None`` (e.g. pretrends_joint skipped because
        no earlier pre-period), emit a NaN row preserving the 3-row
        shape for downstream plotting.
        """
        if joint is None:
            return {
                "test": test_label,
                "statistic_name": "cvm_stat_joint",
                "statistic_value": float("nan"),
                "p_value": float("nan"),
                "reject": False,
                "alpha": float(self.alpha),
                "n_obs": int(self.n_obs),
            }
        return {
            "test": test_label,
            "statistic_name": "cvm_stat_joint",
            "statistic_value": _json_safe_scalar(joint.cvm_stat_joint),
            "p_value": _json_safe_scalar(joint.p_value),
            "reject": bool(joint.reject),
            "alpha": float(joint.alpha),
            "n_obs": int(joint.n_obs),
        }


# =============================================================================
# Private helpers
# =============================================================================


def _validate_1d_numeric(arr: np.ndarray, name: str) -> np.ndarray:
    """Return ``arr`` as a 1D float ndarray or raise ``ValueError``."""
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {a.shape}.")
    a = a.astype(np.float64, copy=False)
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN values.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values (inf).")
    return a


def _fit_ols_intercept_slope(
    d: np.ndarray,
    dy: np.ndarray,
    d_moments: "Optional[tuple[float, np.ndarray, float]]" = None,
) -> "tuple[float, float, np.ndarray]":
    """Fit ``dy = a + b*d + eps`` via closed-form OLS.

    Returns ``(a_hat, b_hat, residuals)`` where ``residuals`` has the
    same length as ``d`` in the ORIGINAL input order (not sorted).

    ``d_moments`` optionally supplies precomputed ``(d_mean, d_dev,
    var_d)`` for callers that refit against the SAME ``d`` many times
    (the Stute bootstrap loops). The values are deterministic functions
    of ``d`` computed with the identical expressions below, so passing
    them is bit-identical to recomputing per call.
    """
    if d_moments is not None:
        d_mean, d_dev, var_d = d_moments
    else:
        d_mean = d.mean()
        d_dev = d - d_mean
        var_d = np.dot(d_dev, d_dev)
    dy_mean = dy.mean()
    if var_d <= 0.0:
        # Degenerate case: all dose values equal. Slope undefined.
        # Caller is responsible for gating before we reach here; if we
        # do reach here, return (mean(dy), 0, dy - mean(dy)).
        return float(dy_mean), 0.0, dy - dy_mean
    b_hat = float(np.dot(d_dev, dy - dy_mean) / var_d)
    a_hat = float(dy_mean - b_hat * d_mean)
    residuals = dy - a_hat - b_hat * d
    return a_hat, b_hat, residuals


def _fit_weighted_ols_intercept_slope(
    d: np.ndarray, dy: np.ndarray, w: np.ndarray
) -> "tuple[float, float, np.ndarray]":
    """Weighted OLS analog of :func:`_fit_ols_intercept_slope`.

    Solves the weighted normal equations for ``dy = a + b*d + eps`` where
    each observation has weight ``w_g``. Returns ``(a_hat, b_hat,
    residuals)`` with ``residuals`` in the ORIGINAL input order (not
    sorted) and on the un-weighted scale (``residuals = dy - a_hat - b_hat * d``,
    NOT ``sqrt(w) * (dy - ...)``).

    At ``w = ones(G)`` reduces bit-exactly to ``_fit_ols_intercept_slope``
    (Phase 4.5 C stability invariant #1; locked at ``atol=1e-14`` by the
    survey-path tests).

    Closed form:
        b_hat = sum(w * (d - d_w_mean) * (dy - dy_w_mean)) / sum(w * (d - d_w_mean)^2)
        a_hat = dy_w_mean - b_hat * d_w_mean

    where ``d_w_mean = sum(w * d) / sum(w)`` (and similarly for ``dy``).
    """
    sw = float(np.sum(w))
    if sw <= 0.0:
        raise ValueError(
            f"_fit_weighted_ols_intercept_slope: sum(w) = {sw} <= 0; "
            "weighted OLS requires positive total mass."
        )
    d_wmean = float(np.sum(w * d) / sw)
    dy_wmean = float(np.sum(w * dy) / sw)
    d_dev = d - d_wmean
    var_d_w = float(np.sum(w * d_dev * d_dev))
    if var_d_w <= 0.0:
        # Degenerate case: all dose values equal (under weights).
        return float(dy_wmean), 0.0, dy - dy_wmean
    b_hat = float(np.sum(w * d_dev * (dy - dy_wmean)) / var_d_w)
    a_hat = float(dy_wmean - b_hat * d_wmean)
    residuals = dy - a_hat - b_hat * d
    return a_hat, b_hat, residuals


def _fit_ols_intercept_only(dy: np.ndarray) -> "tuple[float, float, np.ndarray]":
    """Fit ``dy = a + eps`` (intercept-only OLS, mean-independence null).

    Returns ``(a_hat, 0.0, residuals)`` where ``a_hat = mean(dy)`` and
    ``residuals = dy - a_hat`` in the ORIGINAL input order. Slope
    ``b_hat = 0.0`` is returned for tuple-symmetry with
    :func:`_fit_ols_intercept_slope`; the downstream Yatchew variance code
    consumes only ``residuals``.

    Mirrors R ``YatchewTest::yatchew_test(order=0)``.
    """
    a_hat = float(np.mean(dy))
    residuals = dy - a_hat
    return a_hat, 0.0, residuals


def _fit_weighted_ols_intercept_only(
    dy: np.ndarray, w: np.ndarray
) -> "tuple[float, float, np.ndarray]":
    """Weighted intercept-only OLS analog of :func:`_fit_ols_intercept_only`.

    Returns ``(a_hat, 0.0, residuals)`` where ``a_hat = sum(w * dy) / sum(w)``
    (the weighted mean) and ``residuals = dy - a_hat`` in the ORIGINAL input
    order on the un-weighted scale. At ``w = ones(G)`` reduces bit-exactly
    to :func:`_fit_ols_intercept_only`.
    """
    sw = float(np.sum(w))
    if sw <= 0.0:
        raise ValueError(
            f"_fit_weighted_ols_intercept_only: sum(w) = {sw} <= 0; "
            "weighted OLS requires positive total mass."
        )
    a_hat = float(np.sum(w * dy) / sw)
    residuals = dy - a_hat
    return a_hat, 0.0, residuals


def _iter_mammen_rows(
    n_bootstrap: int,
    G: int,
    rng: np.random.Generator,
    max_batch_bytes: int = 64 * 1024 * 1024,
):
    """Yield per-replicate Mammen weight rows drawn in memory-bounded batches.

    One ``rng.choice`` call per batch: numpy fills the output in C order,
    so consecutive batched draws consume the generator's variate stream
    identically to ``n_bootstrap`` sequential size-``G`` draws — the
    bootstrap law is unchanged draw-for-draw — while peak memory stays
    bounded (a single full ``(B, G)`` batch would be ``B*G*8`` bytes,
    ~800 MB at G=100k, B=999).
    """
    rows_per_batch = max(1, int(max_batch_bytes // (8 * max(G, 1))))
    drawn = 0
    while drawn < n_bootstrap:
        take = min(rows_per_batch, n_bootstrap - drawn)
        batch = _generate_mammen_weights((take, G), rng)
        for r in range(take):
            yield batch[r]
        drawn += take


def _cvm_statistic(
    eps_sorted: np.ndarray,
    d_sorted: np.ndarray,
    tie_blocks: "Optional[tuple[np.ndarray, np.ndarray]]" = None,
) -> float:
    """Compute the tie-safe Cramer-von Mises cusum statistic.

    Paper definition (Appendix D):

        c_G(d) := G^{-1/2} * sum_g 1{D_g <= d} * eps_g
        S := (1/G) * sum_g c_G^2(D_g) = (1/G^2) * sum_g (C_g)^2

    where ``C_g = sum_{h : D_h <= D_g} eps_h`` is the cumulative residual
    sum up to and including ALL observations with dose <= D_g. This
    definition is tie-safe: at a tied dose value ``D_g == D_{g+1}``, both
    c_G(D_g) and c_G(D_{g+1}) include all tie-block members, so the
    cumulative sum used at each tied observation is the cumulative sum
    through the END of the tie block.

    A naive per-observation ``cumsum`` on sorted residuals violates this
    at tie blocks (each tied observation sees a partial within-block
    cumulative sum). This implementation collapses each tie block to the
    post-tie cumulative sum before squaring, matching the paper definition.

    Parameters
    ----------
    eps_sorted : np.ndarray, shape (G,)
        Residuals sorted by ``d_sorted``.
    d_sorted : np.ndarray, shape (G,)
        Regressor values sorted ascending. Must be sorted consistently
        with ``eps_sorted``.
    tie_blocks : tuple of (np.ndarray, np.ndarray), optional
        Precomputed ``(tie_end_idx, counts)`` for callers that evaluate
        the statistic against the SAME ``d_sorted`` many times (the Stute
        bootstrap loops). Deterministic functions of ``d_sorted`` computed
        with the identical expressions below, so passing them is
        bit-identical to recomputing per call.

    Returns
    -------
    float
        ``S = (1 / G^2) * sum_g C_g^2``.
    """
    G = eps_sorted.shape[0]
    cumsum = np.cumsum(eps_sorted)
    # Tie-safe correction: replace within-tie-block values with the
    # cumulative sum at the END of each tie block. np.unique on the
    # already-sorted regressor gives per-unique-value counts; the last
    # index of each tie block is `cumsum(counts) - 1`, and np.repeat
    # expands that back to per-observation.
    if tie_blocks is not None:
        tie_end_idx, counts = tie_blocks
    else:
        _, counts = np.unique(d_sorted, return_counts=True)
        tie_end_idx = np.cumsum(counts) - 1
    cumsum_tie_safe = np.repeat(cumsum[tie_end_idx], counts)
    return float(np.sum(cumsum_tie_safe * cumsum_tie_safe) / (G * G))


def _has_lonely_psu_adjust_singletons(resolved: Any) -> bool:
    """Detect singleton strata under ``lonely_psu='adjust'``.

    Returns ``True`` iff (a) the resolved design uses
    ``lonely_psu='adjust'`` AND (b) at least one stratum has fewer than
    2 PSUs. Under those conditions, the bootstrap multiplier helper
    pools singletons into a pseudo-stratum with NONZERO multipliers
    while the analytical variance target requires a pseudo-stratum
    centering transform that is not derived for the Stute CvM
    (Phase 4.5 C R5 P1; mirrors the explicit lonely-PSU reject on
    HeterogeneousAdoptionDiD's sup-t bootstrap at ``had.py:2081-2118``).
    """
    if getattr(resolved, "lonely_psu", "remove") != "adjust":
        return False
    strata_arr = resolved.strata
    if strata_arr is None:
        return False
    psu_arr = resolved.psu
    for h in np.unique(strata_arr):
        mask_h = np.asarray(strata_arr) == h
        if psu_arr is not None:
            n_psu_h = int(np.unique(np.asarray(psu_arr)[mask_h]).shape[0])
        else:
            n_psu_h = int(mask_h.sum())
        if n_psu_h < 2:
            return True
    return False


def _cvm_statistic_weighted(
    eps_sorted: np.ndarray, d_sorted: np.ndarray, w_sorted: np.ndarray
) -> float:
    """Weighted analog of :func:`_cvm_statistic` (survey-weighted plug-in).

    The unweighted Stute CvM `S = (1/G) * sum_g c_G(D_g)^2` integrates
    the squared cusum process against the empirical CDF
    ``F_hat = (1/G) sum_i delta_{D_i}``. The weighted plug-in replaces
    ``F_hat`` by the survey-weighted EDF
    ``F_hat_w = (1/W) sum_i w_i delta_{D_i}``, which weights BOTH the
    inner cusum AND the outer integration measure:

        C_g = sum_{h : D_h <= D_g} w_h * eps_h     (inner cusum, weighted)
        S_w = (1 / W^2) * sum_g w_g * (C_g)^2      (outer measure, weighted)
        W   = sum(w)

    The outer ``w_g`` factor on each squared cusum (R7 P0 fix) is what
    distinguishes this from a count-weighted-cusum form
    ``(1/W^2) * sum_g C_g^2`` (no outer ``w_g``), which silently
    misreports survey-weighted Stute statistics for non-uniform weights.
    At ``w = ones(G)`` both forms reduce to ``(1/G^2) sum_g C_g^2``
    (unweighted) -- only non-uniform weights distinguish them.

    Tie-block collapse uses the same ``np.unique(d_sorted)`` count
    machinery as the unweighted form -- positions are determined by
    ``d_sorted`` ties (independent of weights), so the collapse pattern
    is weight-invariant. The outer ``w_sorted`` factor applies to the
    tie-collapsed cusum at each observation.

    Parameters
    ----------
    eps_sorted, d_sorted, w_sorted : np.ndarray, shape (G,)
        Residuals, regressor values, and weights sorted CONSISTENTLY by
        ``d``. Caller is responsible for the sort alignment.

    Returns
    -------
    float
        Weighted CvM statistic.
    """
    weighted_eps = w_sorted * eps_sorted
    cumsum = np.cumsum(weighted_eps)
    _, counts = np.unique(d_sorted, return_counts=True)
    tie_end_idx = np.cumsum(counts) - 1
    cumsum_tie_safe = np.repeat(cumsum[tie_end_idx], counts)
    W = float(np.sum(w_sorted))
    # R7 P0: integrate outer measure against F_hat_w via the w_sorted
    # factor on each squared cusum (NOT against uniform 1/G measure).
    return float(np.sum(w_sorted * cumsum_tie_safe * cumsum_tie_safe) / (W * W))


def _compose_verdict(
    qug: QUGTestResults, stute: StuteTestResults, yatchew: YatchewTestResults
) -> str:
    """Build the :class:`HADPretestReport` verdict string.

    Paper Section 4.2-4.3 specifies a four-step workflow; Phase 3 ships
    step 1 (QUG) and step 3 (linearity, via ``stute_test`` OR
    ``yatchew_hr_test``). The linearity step accepts either test, so a
    conclusive Stute result alone suffices even when Yatchew is NaN
    (e.g. tied doses, which Yatchew rejects by contract).

    Paper logic: TWFE is admissible only if NONE of the implemented
    tests rejects. A conclusive rejection must therefore never be hidden
    by a purely-inconclusive verdict just because another step is NaN -
    it is reported as the primary outcome and any unresolved steps are
    appended as a suffix.

    Priority:

    1. Collect all rejection reasons from CONCLUSIVE tests. If any
       conclusive test rejected, that is the primary verdict. Unresolved
       steps (QUG NaN, or BOTH linearity tests NaN) are appended as
       ``"; additional steps unresolved: ..."`` rather than replacing
       the rejection.
    2. If no conclusive test rejected but a required step is unresolved,
       return a pure ``"inconclusive - ..."`` verdict naming the
       unresolved step(s).
    3. Otherwise (all required steps conclusive and none reject),
       return the partial-workflow fail-to-reject verdict flagging the
       Assumption 7 gap, with a ``" (Yatchew NaN - skipped)"`` suffix
       when ONE linearity test was NaN and the other was conclusive.
    """
    qug_ok = bool(np.isfinite(qug.p_value))
    stute_ok = bool(np.isfinite(stute.p_value))
    yatchew_ok = bool(np.isfinite(yatchew.p_value))

    # Rejections from conclusive tests only. NaN-p tests have reject=False
    # by convention, so the ``ok and reject`` guard is defensive.
    qug_rej = qug_ok and qug.reject
    stute_rej = stute_ok and stute.reject
    yatchew_rej = yatchew_ok and yatchew.reject

    reasons = []
    if qug_rej:
        reasons.append("support infimum rejected - continuous_at_zero design invalid (QUG)")
    if stute_rej or yatchew_rej:
        which = ",".join(
            name for name, rejected in (("Stute", stute_rej), ("Yatchew", yatchew_rej)) if rejected
        )
        reasons.append(f"linearity rejected - heterogeneity bias ({which})")

    # Unresolved steps: QUG is required; step 3 requires at least one
    # conclusive linearity test.
    unresolved = []
    if not qug_ok:
        unresolved.append("QUG NaN")
    if not stute_ok and not yatchew_ok:
        unresolved.append("both Stute and Yatchew linearity tests NaN")

    if reasons:
        # A conclusive rejection is the primary outcome. Append any
        # unresolved-step note rather than replacing the rejection.
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict

    if unresolved:
        return "inconclusive - " + "; ".join(unresolved)

    # All required steps conclusive, none reject. Note any single skipped
    # linearity test (the OTHER linearity test was conclusive and
    # fail-to-reject, so step 3 IS resolved).
    skipped = []
    if not stute_ok:
        skipped.append("Stute NaN")
    if not yatchew_ok:
        skipped.append("Yatchew NaN")
    skip_note = f" ({'; '.join(skipped)} - skipped)" if skipped else ""
    return (
        "QUG and linearity diagnostics fail-to-reject"
        f"{skip_note}; Assumption 7 pre-trends test NOT run "
        "(paper step 2 deferred to Phase 3 follow-up)"
    )


# =============================================================================
# Public test functions
# =============================================================================


def qug_test(
    d: np.ndarray,
    alpha: float = 0.05,
    *,
    survey_design: Any = None,
) -> QUGTestResults:
    """Run the QUG null test for the support infimum (paper Theorem 4).

    Tests ``H_0: d_lower = 0`` using the order-statistic ratio
    ``T = D_{(1)} / (D_{(2)} - D_{(1)})``, rejecting when ``T > 1/alpha - 1``.
    Under the null, the asymptotic limit law of ``T`` is the ratio of two
    independent Exp(1) variables with CDF ``F(t) = t / (1 + t)``, so the
    one-sided p-value is ``1 / (1 + T)``.

    Zero-dose observations are filtered out (the test targets the infimum
    of the treated support). A ``UserWarning`` is emitted naming the
    exclusion count. When fewer than two positive doses remain, the test
    returns all-NaN inference with ``reject=False``.

    Parameters
    ----------
    d : np.ndarray, shape (G,)
        Post-period dose vector. Must be 1D numeric and contain no NaN.
    alpha : float, default 0.05
        One-sided significance level. Must satisfy ``0 < alpha < 1``.
    survey_design : ResolvedSurveyDesign or None, keyword-only, default None
        Permanently rejected with ``NotImplementedError`` (Phase 4.5 C0
        decision gate). Surface-symmetric kwarg with the rest of the HAD
        family — accepted in the signature so all 8 HAD entry points
        share the canonical kwarg name, but ``qug_test`` has no
        survey-aware migration target. See *Notes -- Survey/weighted
        data*.

    Returns
    -------
    QUGTestResults
        Result dataclass with ``t_stat``, ``p_value``, ``reject``, and
        sample metadata.

    Raises
    ------
    ValueError
        If ``d`` is not 1D numeric or contains NaN, or if ``alpha`` is
        not in ``(0, 1)``.
    NotImplementedError
        If ``survey_design`` is non-None. See *Notes -- Survey/weighted
        data*.

    Notes
    -----
    **Scope (what this test does NOT cover).** ``qug_test`` tests the
    Theorem 4 / Design 1' support-infimum null ``H_0: d_lower = 0``. It
    does not validate the full Assumption 4 (Assumption 4 also requires
    positive boundary density, twice-differentiable conditional-mean,
    bounded continuous conditional-variance, and bandwidth regularity —
    QUG is adjacent evidence on the ``d_lower = 0`` clause only). It
    does NOT and CANNOT test Assumptions 5 and 6 from the same paper
    (Section 3.1.2), which are required for sign identification (A5) and
    point identification (A6) of ``WAS_{d_lower}`` on the Design 1 family
    (``d_lower > 0``). Assumptions 5 and 6 are statements about
    conditional expectations near the support boundary and about
    counterfactual-mean alignment respectively; they are non-testable via
    pre-trends. See :class:`HeterogeneousAdoptionDiD` class docstring
    Notes for the full statement and T21 (HAD pretest workflow tutorial)
    for the verdict-language convention that surfaces this gap.

    Tie-break: when ``D_{(1)} == D_{(2)}`` the statistic is undefined.
    The test returns ``t_stat=NaN, p_value=NaN, reject=False`` with a
    ``UserWarning`` rather than raising.

    Survey/weighted data: QUG is permanently deferred under survey-weighted
    or pweight inputs (Phase 4.5 C0 decision gate, 2026-04). The test
    statistic uses extreme order statistics ``(D_{(1)}, D_{(2)})``, which
    are NOT smooth functionals of the empirical CDF -- standard survey
    machinery (Binder TSL linearization, multiplier bootstrap, Rao-Wu
    rescaled bootstrap) does not yield a calibrated test, and under
    cluster sampling the ``Exp(1)/Exp(1)`` limit law's independence
    assumption breaks. The extreme-value-theory-under-unequal-probability-
    sampling literature (Quintos et al. 2001, Beirlant et al.) addresses
    tail-index estimation, not boundary tests; no off-the-shelf
    survey-aware QUG exists. Phase 4.5 C ships survey-aware Stute via
    :func:`did_had_pretest_workflow` (which skips the QUG step under
    ``survey_design=`` and runs the linearity family with a PSU-level Mammen
    multiplier bootstrap for Stute and weighted OLS + pweight-sandwich
    variance components for Yatchew). See ``docs/methodology/REGISTRY.md``
    § "QUG Null Test" for the full methodology note.

    References
    ----------
    de Chaisemartin, Ciccia, D'Haultfoeuille, Knau (2026, arXiv:2405.04465v6),
    Theorem 4 and Section 4.2.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")

    # Phase 4.5 C0 decision gate: QUG-under-survey is permanently deferred.
    # Extreme-order-statistic functionals are not smooth in the empirical
    # CDF, so standard survey machinery (Binder TSL linearization, Rao-Wu
    # rescaled bootstrap) does not provide a calibrated test. See
    # REGISTRY.md § "QUG Null Test" for the full methodology note.
    if survey_design is not None:
        raise NotImplementedError(
            "qug_test does not support the survey_design= kwarg.\n"
            "\n"
            "QUG (de Chaisemartin et al. 2026, Theorem 4) tests "
            "H_0: d_lower = 0 via the ratio of the two smallest order "
            "statistics, T = D_(1) / (D_(2) - D_(1)). "
            "Extreme-order-statistic functionals are not smooth in the "
            "empirical CDF, so standard survey machinery (Binder "
            "linearization, multiplier bootstrap, Rao-Wu rescaled "
            "bootstrap) does not provide a calibrated test. Under "
            "cluster sampling the Exp(1)/Exp(1) limit law's independence "
            "assumption breaks. The literature on extreme-value theory "
            "under unequal-probability sampling (Quintos et al. 2001, "
            "Beirlant et al.) addresses tail-index estimation, not "
            "boundary tests; no off-the-shelf survey-aware QUG exists.\n"
            "\n"
            "For survey-aware HAD pretesting, use the joint Stute family "
            "via did_had_pretest_workflow(..., survey_design=..., "
            "aggregate=...) -- shipped in Phase 4.5 C. The workflow "
            "skips the QUG step under survey_design= with a UserWarning "
            "and runs the linearity family with a PSU-level Mammen "
            "multiplier bootstrap (Stute) + weighted OLS + pweight-"
            "sandwich variance components (Yatchew). See "
            "docs/methodology/REGISTRY.md § 'QUG Null Test' for the "
            "full methodology note."
        )

    d_arr = _validate_1d_numeric(d, "d")
    critical_value = 1.0 / alpha - 1.0

    # HAD support restriction: doses must be non-negative (paper Section 2).
    # Reject negative doses at the front door rather than silently filtering
    # them into the zero-exclusion counter.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"qug_test: d contains {n_neg} negative value(s); HAD doses "
            f"must be non-negative (paper Section 2). Check your dose "
            f"column or pre-process before calling qug_test."
        )

    mask = d_arr > 0
    d_nz = d_arr[mask]
    n_excluded = int(d_arr.shape[0] - d_nz.shape[0])
    if n_excluded > 0:
        warnings.warn(
            f"qug_test: excluded {n_excluded} observation(s) with d == 0 "
            f"(the QUG null test targets the infimum of the treated-dose "
            f"support; zero-dose observations are not in scope).",
            UserWarning,
            stacklevel=2,
        )

    n_obs = int(d_nz.shape[0])
    if n_obs < _MIN_G_QUG:
        warnings.warn(
            f"qug_test: only {n_obs} positive-dose observation(s); need "
            f"at least {_MIN_G_QUG}. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return QUGTestResults(
            t_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            n_obs=n_obs,
            n_excluded_zero=n_excluded,
            d_order_1=float("nan"),
            d_order_2=float("nan"),
        )

    # Use np.partition for O(G) extraction of the two smallest positive
    # doses (faster than full O(G log G) sort). For k=1, np.partition
    # guarantees partitioned[0] <= partitioned[1] = D_{(2)} (the 2nd-smallest),
    # which implies partitioned[0] = D_{(1)} (the minimum).
    partitioned = np.partition(d_nz, 1)
    D1 = float(partitioned[0])
    D2 = float(partitioned[1])

    if D2 == D1:
        warnings.warn(
            "qug_test: D_(1) == D_(2); the test statistic is undefined "
            "(ties at the minimum positive dose). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return QUGTestResults(
            t_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            n_obs=n_obs,
            n_excluded_zero=n_excluded,
            d_order_1=D1,
            d_order_2=D2,
        )

    t_stat = D1 / (D2 - D1)
    p_value = 1.0 / (1.0 + t_stat)
    reject = t_stat > critical_value

    return QUGTestResults(
        t_stat=float(t_stat),
        p_value=float(p_value),
        reject=bool(reject),
        alpha=alpha,
        critical_value=critical_value,
        n_obs=n_obs,
        n_excluded_zero=n_excluded,
        d_order_1=D1,
        d_order_2=D2,
    )


def stute_test(
    d: np.ndarray,
    dy: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    *,
    survey_design: Any = None,
) -> StuteTestResults:
    """Run the Stute Cramer-von Mises linearity test (paper Appendix D).

    Tests ``H_0: E[ΔY | D_2]`` is linear in ``D_2`` (paper Assumption 8).
    The test statistic is the sorted-residual cusum CvM

        S = (1 / G^2) * sum_{g=1}^G (sum_{h=1}^g eps_(h))^2

    where ``eps_(h)`` is the ``h``-th OLS residual after sorting by ``d``.
    The p-value is the bootstrap tail probability
    ``(1 + sum(S_b >= S)) / (B + 1)`` under the Mammen (1993) two-point
    wild bootstrap; each bootstrap iteration refits OLS on
    ``dy_b = a_hat + b_hat * d + eps * eta`` with multiplier weights ``eta``.

    Parameters
    ----------
    d, dy : np.ndarray, shape (G,)
        Dose and first-difference outcome vectors.
    alpha : float, default 0.05
        Significance level. Must satisfy ``0 < alpha < 1``.
    n_bootstrap : int, default 999
        Number of Mammen wild bootstrap replications. Must be ``>= 99``
        (below which the discretised p-value grid is too coarse).
    seed : int or None, default None
        Seed for ``np.random.default_rng``. Pass an integer for
        reproducible results.
    survey_design : ResolvedSurveyDesign or None, keyword-only, default None
        Already-resolved survey design (per-unit). Array-in helpers
        accept ``ResolvedSurveyDesign`` ONLY; passing a ``SurveyDesign``
        raises ``TypeError`` with migration guidance. For the pweight-only
        shortcut, use ``survey_design=make_pweight_design(arr)``. Triggers
        the survey-aware Stute calibration: PSU-level Mammen multipliers
        via
        :func:`diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch`,
        broadcast to per-unit residual perturbation, with weighted CvM
        recompute. Replicate-weight designs raise ``NotImplementedError``.

    Returns
    -------
    StuteTestResults

    Raises
    ------
    ValueError
        If ``d`` / ``dy`` are not 1D numeric, contain NaN, have unequal
        lengths, if any ``d`` value is negative (paper Section 2 HAD
        support restriction), if ``alpha`` is outside ``(0, 1)``, or if
        ``n_bootstrap < 99``.
    TypeError
        If ``survey_design=SurveyDesign(...)`` is passed; array-in helpers
        accept ``ResolvedSurveyDesign`` only. Use
        ``survey_design=make_pweight_design(arr)`` for pweight-only or
        pre-resolve via ``SurveyDesign(...).resolve(data)``.
    NotImplementedError
        If ``survey.replicate_weights is not None``. Replicate-weight
        pretests are a parallel follow-up after Phase 4.5 C; the
        per-replicate weight-ratio rescaling for the OLS-on-residuals
        refit step is not covered by the multiplier-bootstrap composition
        used here.

    Notes
    -----
    **Scope (what this test does NOT cover).** ``stute_test`` targets
    paper Assumption 8 (linearity of ``E[ΔY | D_2]`` in ``D_2``) — the
    raw helper always fits ``dy ~ 1 + d`` and tests the linearity null;
    it does NOT target Assumption 7 mean-independence pre-trends on its
    own. For Assumption 7 mean-independence (residuals from intercept-
    only ``dy ~ 1``), use :func:`joint_pretrends_test` (which routes
    ``null_form="mean_independence"`` into the joint CvM core). It does
    NOT and CANNOT test Assumptions 5 and 6 from de Chaisemartin et al.
    (2026) Section 3.1.2, which are required for sign / point
    identification of ``WAS_{d_lower}`` on the Design 1 family
    (``d_lower > 0``). Assumptions 5/6 are non-testable via pre-trends
    (boundary-conditional expectations and counterfactual-mean alignment
    statements); they are surfaced by the Design 1 fit-time
    ``UserWarning`` and by T21 tutorial prose, NOT by the workflow
    verdict string. See :class:`HeterogeneousAdoptionDiD` class
    docstring Notes for the full statement.

    Sample-size gate: below ``G = 10`` the CvM statistic is not
    well-calibrated. In that case the function emits ``UserWarning`` and
    returns all-NaN inference rather than raising.

    Large-G warning: at ``G > 100_000`` the per-iteration refit dominates
    runtime; the function emits a ``UserWarning`` pointing users to
    :func:`yatchew_hr_test`. Memory usage remains ``O(G)`` regardless
    (no G x G matrix).

    Survey/weighted data (Phase 4.5 C): when ``survey_design`` is
    supplied, the OLS baseline becomes weighted OLS
    (:func:`_fit_weighted_ols_intercept_slope`), the bootstrap multipliers
    become PSU-level Mammen draws (broadcast to per-obs perturbation), and
    the test statistic uses :func:`_cvm_statistic_weighted`. Per-unit
    constant-within-unit invariant on weights/strata/psu/fpc is the
    CALLER's responsibility; the workflow
    (:func:`did_had_pretest_workflow`) enforces it via
    :func:`_aggregate_unit_weights` /
    :func:`_aggregate_unit_resolved_survey` from ``had.py``. At ``w =
    ones(G)``, weighted helpers reduce bit-exactly to the unweighted
    versions but bootstrap p-values diverge by Monte-Carlo noise (different
    RNG consumption between batched ``generate_survey_multiplier_weights_batch``
    and the unweighted path's ``_generate_mammen_weights`` stream — the
    latter is drawn in memory-bounded batches that consume the stream
    identically to per-iteration draws); use the distribution-equivalence
    reduction test (large B) for trivial-pweight parity, NOT numerical
    equivalence.

    References
    ----------
    Stute, W. (1997). Nonparametric model checks for regression. Annals
    of Statistics 25, 613-641.
    Mammen, E. (1993). Bootstrap and wild bootstrap for high-dimensional
    linear models. Annals of Statistics 21, 255-285.
    de Chaisemartin et al. (2026), Appendix D.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")
    if n_bootstrap < _MIN_N_BOOTSTRAP:
        raise ValueError(
            f"n_bootstrap must be >= {_MIN_N_BOOTSTRAP} (below this the "
            f"discretised p-value grid is too coarse to be meaningful). "
            f"Got n_bootstrap={n_bootstrap}."
        )

    # Type guard: array-in helpers reject SurveyDesign (cannot resolve
    # column names without `data`).
    if survey_design is not None and isinstance(survey_design, SurveyDesign):
        raise TypeError(
            "stute_test: `survey_design=` accepts a pre-resolved "
            "ResolvedSurveyDesign only (array-in helpers have no `data` to "
            "resolve column names against). For pweight-only, use "
            "`survey_design=make_pweight_design(arr)`. For full PSU/strata/"
            "FPC, pre-resolve via `SurveyDesign(...).resolve(data)` and pass "
            "the result."
        )

    # Internal variable naming: downstream code reads `survey` (Phase 4.5 C
    # convention). Bind the canonical survey_design to the internal name so
    # the unchanged downstream logic consumes it transparently.
    survey = survey_design

    # Replicate-weight rejection: the per-replicate weight-ratio rescaling for
    # the OLS-on-residuals refit step is not covered by the multiplier-bootstrap
    # composition. Parallel follow-up after Phase 4.5 C.
    if survey is not None and getattr(survey, "replicate_weights", None) is not None:
        raise NotImplementedError(
            "stute_test: replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) "
            "are not yet supported on HAD pretests. The per-replicate weight-"
            "ratio rescaling for the OLS-on-residuals refit step is not covered "
            "by the multiplier-bootstrap composition. Replicate-weight pretests "
            "are a parallel follow-up after Phase 4.5 C."
        )
    # R1 P1: pweight-only guard on the direct-helper survey entry (mirrors
    # _resolve_pretest_unit_weights for the workflow path).
    if survey is not None and getattr(survey, "weight_type", "pweight") != "pweight":
        raise ValueError(
            f"stute_test: HAD pretests require weight_type='pweight'. Got "
            f"weight_type={survey.weight_type!r}. aweight / fweight have "
            "different sandwich-variance semantics that are not derived "
            "for the Stute CvM bootstrap calibration."
        )

    d_arr = _validate_1d_numeric(d, "d")
    dy_arr = _validate_1d_numeric(dy, "dy")
    if d_arr.shape[0] != dy_arr.shape[0]:
        raise ValueError(
            f"d and dy must have the same length; got d.shape={d_arr.shape}, "
            f"dy.shape={dy_arr.shape}."
        )
    # HAD support restriction (paper Section 2): doses must be non-negative.
    # Mirror the front-door guard from qug_test / _validate_had_panel.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"stute_test: d contains {n_neg} negative value(s); HAD doses "
            f"must be non-negative (paper Section 2). Check your dose "
            f"column or pre-process before calling stute_test."
        )

    G = int(d_arr.shape[0])
    if G < _MIN_G_STUTE:
        warnings.warn(
            f"stute_test: G = {G} is below the minimum {_MIN_G_STUTE} for "
            f"the CvM statistic to be well-calibrated. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteTestResults(
            cvm_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )
    if G > _STUTE_LARGE_G_THRESHOLD:
        warnings.warn(
            f"stute_test: G = {G} exceeds {_STUTE_LARGE_G_THRESHOLD}; the "
            f"per-replicate refit is O(G), so the {n_bootstrap}-replication "
            f"bootstrap cost grows linearly in G (measured ~1.5s at G=1e5, "
            f"B=999; ~10x that at G=1e6). Consider yatchew_hr_test() "
            f"instead (paper Theorem 7 recommends Yatchew-HR at large G).",
            UserWarning,
            stacklevel=2,
        )

    # Phase 4.5 C: resolve effective per-unit weights (None on the
    # unweighted path, preserves bit-exact regression). w is taken from
    # the resolved survey_design (rebound to `survey` above).
    # R4 P1: validate 1D explicitly so column-vector inputs (e.g.
    # df[["w"]].to_numpy()) raise instead of silently broadcasting.
    if survey is not None:
        w_arr = _validate_1d_numeric(np.asarray(survey.weights), "stute_test: survey.weights")
        if w_arr.shape[0] != G:
            raise ValueError(
                f"stute_test: survey.weights length {w_arr.shape[0]} does not "
                f"match d/dy length {G}."
            )
        if (w_arr <= 0).any():
            raise ValueError(
                "stute_test: survey weights must be strictly positive. "
                "Zero / negative weights would leave units in the "
                "variance / CvM computation while contributing zero "
                "population mass; pre-filter the panel to the positive-"
                "weight subpopulation before calling stute_test."
            )
    else:
        w_arr = None

    # R4 P0: normalize pweights to mean=1 (matches SurveyDesign.resolve()
    # convention). Makes the test statistic scale-invariant under uniform
    # rescaling of weights across every survey_design= entry form. Stute is internally scale-invariant in functional form,
    # but the survey-aware bootstrap helper consumes weight values
    # directly under non-trivial PSU/strata, so normalization is required
    # for cross-path agreement.
    if w_arr is not None:
        w_arr = w_arr * (float(w_arr.shape[0]) / float(np.sum(w_arr)))

    if w_arr is None:
        a_hat, b_hat, eps = _fit_ols_intercept_slope(d_arr, dy_arr)
    else:
        a_hat, b_hat, eps = _fit_weighted_ols_intercept_slope(d_arr, dy_arr, w_arr)

    # Genuine degeneracy: zero dose variation. The CvM cusum is defined
    # against the regressor, and constant d carries no signal to test
    # linearity against - emit NaN.
    if np.var(d_arr) <= 0.0:
        warnings.warn(
            "stute_test: constant d (zero dose variation); the Stute "
            "linearity test requires regressor variation. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteTestResults(
            cvm_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )

    # Numerically exact linear fit: Assumption 8 holds to IEEE precision,
    # so the Stute CvM statistic is formally 0 and every bootstrap draw is
    # also 0. Short-circuit to p=1 to avoid FP-noise-driven bootstrap
    # comparisons where cvm_stat and S_b are both at machine-epsilon scale.
    # Comparison is purely relative against CENTERED TSS: both translation-
    # invariant (centering absorbs additive shifts) and scale-invariant
    # (ratio is dimensionless under multiplicative dy rescaling).
    eps_norm_sq = float(np.sum(eps * eps))
    dy_centered_sq = float(np.sum((dy_arr - dy_arr.mean()) ** 2))
    if dy_centered_sq <= 0.0:
        # Constant dy (zero centered TSS): trivially linear in d.
        # Return p = 1 without running the bootstrap.
        return StuteTestResults(
            cvm_stat=0.0,
            p_value=1.0,
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )
    if eps_norm_sq <= _EXACT_LINEAR_RELATIVE_TOL * dy_centered_sq:
        return StuteTestResults(
            cvm_stat=0.0,
            p_value=1.0,
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )

    idx = np.argsort(d_arr, kind="stable")
    d_sorted = d_arr[idx]
    if w_arr is None:
        S = _cvm_statistic(eps[idx], d_sorted)
    else:
        S = _cvm_statistic_weighted(eps[idx], d_sorted, w_arr[idx])

    rng = np.random.default_rng(seed)
    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)
    fitted = a_hat + b_hat * d_arr  # baseline fitted values under H_0

    if w_arr is None:
        # Unweighted path: the Appendix-D per-replicate OLS refit with
        # hoisted loop invariants. The eta draws are batched through
        # memory-bounded rng.choice calls (identical variate stream as
        # per-iteration size-G draws — see _iter_mammen_rows), and the
        # d-moments and CvM tie-block indices are precomputed once (they
        # are deterministic functions of d, recomputed identically inside
        # the helpers). Each replicate applies the same 1-D refit/CvM
        # operations on the same values as the literal per-iteration
        # form, so bootstrap_S is bit-identical to pre-hoist code
        # (verified on a seed x DGP grid; ~2.3x at G in [1e3, 5e4]).
        d_mean_h = d_arr.mean()
        d_dev_h = d_arr - d_mean_h
        d_moments = (d_mean_h, d_dev_h, float(np.dot(d_dev_h, d_dev_h)))
        _, tie_counts = np.unique(d_sorted, return_counts=True)
        tie_blocks = (np.cumsum(tie_counts) - 1, tie_counts)
        for b, eta in enumerate(_iter_mammen_rows(n_bootstrap, G, rng)):
            dy_b = fitted + eps * eta
            _, _, eps_b = _fit_ols_intercept_slope(d_arr, dy_b, d_moments=d_moments)
            bootstrap_S[b] = _cvm_statistic(eps_b[idx], d_sorted, tie_blocks=tie_blocks)
    else:
        # Phase 4.5 C survey-aware path: PSU-level Mammen multipliers
        # (broadcast to per-obs perturbation), weighted OLS refit, weighted
        # CvM recompute. make_pweight_design routes the pweight-only entry
        # through the same synthetic-ResolvedSurveyDesign kernel.
        resolved_for_boot = survey if survey is not None else make_pweight_design(w_arr)
        # Stratified designs are supported via the standard stratified
        # clustered wild-bootstrap correction on the PSU multipliers
        # (within-stratum demean + sqrt(n_h/(n_h-1)) Bessel rescale),
        # applied uniformly before the per-obs broadcast eta_obs =
        # psu_mults[b, psu_col_idx] below. See REGISTRY
        # § "Note (Stute stratified survey-bootstrap calibration)" and
        # ``apply_stratum_centering`` (bootstrap_utils.py) for the
        # derivation; the same helper backs the HAD sup-t event-study
        # bootstrap at had.py:2151+.
        # R5 P1: reject lonely_psu='adjust' singleton-strata designs.
        # This pseudo-stratum centering transform has not been derived
        # for the Stute CvM (same gap as the HAD sup-t deviation at
        # REGISTRY § 'Note (HAD sup-t lonely_psu="adjust") deviation').
        if _has_lonely_psu_adjust_singletons(resolved_for_boot):
            raise NotImplementedError(
                "stute_test: SurveyDesign(lonely_psu='adjust') with "
                "singleton strata is not yet supported on the multiplier "
                "bootstrap. The bootstrap helper pools singletons with "
                "nonzero multipliers but the matching analytical "
                "variance target requires a pseudo-stratum centering "
                "transform that has not been derived for the Stute CvM. "
                "Use lonely_psu='remove' (drops singleton contributions) "
                "or 'certainty' (zero-variance singletons), or pre-"
                "process the panel to remove singleton strata."
            )
        # R3 P0: variance-unidentified survey-design guard. When
        # n_psu - n_strata <= 0 (e.g. unstratified single-PSU, or one PSU
        # per stratum under lonely_psu='remove'/'certainty'),
        # generate_survey_multiplier_weights_batch returns an all-zero
        # multiplier matrix. Without this guard, the code below would
        # treat zero perturbations as a valid bootstrap law and emit
        # p_value = 1/(B+1) for any positive observed CvM (spurious
        # rejection). Mirrors compute_survey_vcov's df_survey-driven
        # NaN treatment elsewhere in the package. The lonely_psu='adjust'
        # singleton case (which has nonzero multipliers but a separate
        # methodology gap) is already rejected above, so this branch
        # only catches genuinely degenerate designs.
        df_survey = resolved_for_boot.df_survey
        if df_survey is None or df_survey <= 0:
            warnings.warn(
                f"stute_test: survey design is variance-unidentified "
                f"(df_survey={df_survey}); the multiplier bootstrap "
                "cannot calibrate the test (single-PSU unstratified or "
                "one-PSU-per-stratum design). Returning NaN result.",
                UserWarning,
                stacklevel=2,
            )
            return StuteTestResults(
                cvm_stat=float(S),
                p_value=float("nan"),
                reject=False,
                alpha=alpha,
                n_bootstrap=int(n_bootstrap),
                n_obs=G,
                seed=seed,
            )
        psu_mults, psu_ids = generate_survey_multiplier_weights_batch(
            n_bootstrap, resolved_for_boot, weight_type="mammen", rng=rng
        )
        # Stratum centering + Bessel rescale on the PSU multipliers
        # before broadcast. Same algebra as the HAD sup-t bootstrap at
        # had.py:2151+ (applied to the influence tensor there), but
        # applied here to ``psu_mults`` because the Stute bootstrap is a
        # wild-residual / refit-in-loop bootstrap (no precomputed
        # influence tensor exists). See REGISTRY § "Note (Stute
        # stratified survey-bootstrap calibration)" for the derivation
        # and the non-strata calibration shift it introduces.
        apply_stratum_centering(psu_mults, resolved_for_boot, psu_ids, psu_axis=1)
        # Build per-obs PSU-column index. When psu is None (trivial path),
        # each obs is its own PSU and psu_ids = arange(G) - so psu_col_idx
        # is just arange(G).
        if resolved_for_boot.psu is None:
            psu_col_idx = np.arange(G)
        else:
            psu_to_col = {int(p): c for c, p in enumerate(psu_ids)}
            psu_arr = np.asarray(resolved_for_boot.psu)
            psu_col_idx = np.array([psu_to_col[int(psu_arr[g])] for g in range(G)])

        for b in range(n_bootstrap):
            eta_obs = psu_mults[b, psu_col_idx]  # (G,)
            dy_b = fitted + eps * eta_obs
            _, _, eps_b = _fit_weighted_ols_intercept_slope(d_arr, dy_b, w_arr)
            bootstrap_S[b] = _cvm_statistic_weighted(eps_b[idx], d_sorted, w_arr[idx])

    p_value = float((1.0 + float(np.sum(bootstrap_S >= S))) / (n_bootstrap + 1.0))
    reject = p_value <= alpha

    return StuteTestResults(
        cvm_stat=float(S),
        p_value=p_value,
        reject=bool(reject),
        alpha=alpha,
        n_bootstrap=int(n_bootstrap),
        n_obs=G,
        seed=seed,
    )


def yatchew_hr_test(
    d: np.ndarray,
    dy: np.ndarray,
    alpha: float = 0.05,
    *,
    null: Literal["linearity", "mean_independence"] = "linearity",
    survey_design: Any = None,
) -> YatchewTestResults:
    """Run the Yatchew heteroskedasticity-robust specification test.

    Tests one of two nulls (selected via ``null=``) using the
    variance-ratio statistic

        T_hr = sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W

    where

        sigma2_lin   = (1/G) * sum(eps^2)                        # residuals under chosen null
        sigma2_diff  = (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2) # Yatchew differencing
        sigma2_W     = sqrt((1/(G-1)) * sum(eps_{(g)}^2 * eps_{(g-1)}^2))

    and ``_{(g)}`` denotes sort by ``d``. Under ``null="linearity"``
    (default, paper Assumption 8 / Theorem 7) ``eps`` are residuals from
    OLS ``dy = a + b*d + eps``. Under ``null="mean_independence"`` ``eps
    = dy - mean(dy)`` (intercept-only OLS), mirroring R
    ``YatchewTest::yatchew_test(order=0)``. The ``sigma2_diff`` and
    ``sigma2_W`` formulas are identical between the two modes -
    the only delta is the residual definition. Rejection uses the
    one-sided standard-normal critical value ``z_{1-alpha}``.

    Parameters
    ----------
    d, dy : np.ndarray, shape (G,)
        Dose and first-difference outcome vectors.
    alpha : float, default 0.05
        One-sided significance level.
    null : {"linearity", "mean_independence"}, keyword-only, default "linearity"
        Which null hypothesis the test targets:

        - ``"linearity"`` (default): H_0 ``E[dY | D]`` is linear in ``D``
          (paper Assumption 8, Theorem 7). Residuals come from OLS
          ``dy = a + b*d + eps``. Bit-exact backcompat with pre-PR calls.
        - ``"mean_independence"``: H_0 ``E[dY | D] = E[dY]`` (mean
          independence of ``dY`` from ``D``). Residuals come from
          intercept-only OLS ``dy = a + eps``, so
          ``eps = dy - mean(dy)``. Mirrors R
          ``YatchewTest::yatchew_test(order=0)``. Used by the
          R-parity test on placebo Yatchew rows
          (``Credible-Answers/did_had`` runs ``order=0`` on placebos
          to test pre-trends as a non-parametric mean-independence
          assertion).

        ``d`` is required under both modes (the sort-by-``d``
        differencing step is null-agnostic).
    survey_design : ResolvedSurveyDesign or None, keyword-only, default None
        Already-resolved survey design (per-unit). Array-in helpers accept
        ``ResolvedSurveyDesign`` ONLY; passing a ``SurveyDesign`` raises
        ``TypeError``. For pweight-only, use
        ``survey_design=make_pweight_design(arr)``. When supplied, the OLS
        baseline becomes weighted OLS and all three variance components
        become their pweight-sandwich analogs. PSU clustering is NOT
        propagated through the variance-ratio statistic (would require
        deriving a survey-aware variance-of-variance estimator; out of
        scope per Phase 4.5 C). Replicate-weight designs raise
        ``NotImplementedError``.

    Returns
    -------
    YatchewTestResults

    Raises
    ------
    ValueError
        If ``d`` / ``dy`` are not 1D numeric, contain NaN, have unequal
        lengths, if any ``d`` value is negative (paper Section 2 HAD
        support restriction), or if ``alpha`` is outside ``(0, 1)``.
        Also raised if any weight is non-positive.
    TypeError
        If ``survey_design=SurveyDesign(...)`` is passed; array-in helpers
        accept ``ResolvedSurveyDesign`` only. Use
        ``survey_design=make_pweight_design(arr)`` for pweight-only or
        pre-resolve via ``SurveyDesign(...).resolve(data)``.
    NotImplementedError
        If ``survey.replicate_weights is not None`` (deferred follow-up).

    Notes
    -----
    **Scope (what this test does NOT cover).** ``yatchew_hr_test`` targets
    paper Assumption 8 (linearity of ``E[ΔY | D_2]`` in ``D_2``) under
    ``null="linearity"`` (default); ``null="mean_independence"`` swaps
    the residual definition to intercept-only ``dy ~ 1`` for R parity
    with ``YatchewTest::yatchew_test(order=0)`` on pre-trend placebos.
    It does NOT and CANNOT test Assumptions 5 and 6 from de
    Chaisemartin et al. (2026) Section 3.1.2, which are required for
    sign / point identification of ``WAS_{d_lower}`` on the Design 1
    family (``d_lower > 0``). Assumptions 5/6 are non-testable via
    pre-trends; they are surfaced by the Design 1 fit-time
    ``UserWarning`` and by T21 tutorial prose, NOT by the workflow
    verdict string. See :class:`HeterogeneousAdoptionDiD` class
    docstring Notes for the full statement.

    Sample-size gate: below ``G = 3`` the difference-variance estimator
    is undefined; the function emits ``UserWarning`` and returns NaN
    rather than raising.

    Dose ties: REJECTED with ``UserWarning`` + all-NaN result. The
    difference-based variance estimator ``sigma2_diff`` and the
    heteroskedasticity-robust scale ``sigma4_W`` both use adjacent
    differences of quantities sorted by ``d``; under tied doses the
    within-tie row ordering is arbitrary (stable sort falls back to input
    order) so the statistic becomes order-dependent rather than
    data-dependent. Callers with tied doses (mass-point designs,
    discretised dose registers) should use :func:`stute_test` instead -
    its tie-safe Cramer-von Mises statistic collapses tie blocks to the
    post-tie cumulative sum and is provably order-invariant under
    within-tie permutations.

    Exact-linear short-circuit: when the OLS residual sum-of-squares is
    below IEEE precision relative to the centered total sum of squares
    (``sum(eps^2) <= 1e-24 * sum((dy - dybar)^2)``, i.e. essentially
    ``1 - R^2 == 0``), the test short-circuits to ``t_stat_hr=-inf,
    p_value=1.0, reject=False`` - Assumption 8 holds exactly, the formal
    statistic is ``-inf`` under the one-sided critical value, and the
    correct decision is fail-to-reject. This shortcut is translation-
    invariant because the comparison is against centered TSS (not raw
    ``sum(dy^2)``).

    Degenerate ``sigma4_W = 0`` with non-zero residuals: when the
    adjacent-residual-product sum vanishes AFTER the exact-linear
    shortcut is bypassed (e.g. residuals alternate zero/non-zero after
    sorting), the formal statistic is ``+inf`` or ``-inf`` depending on
    the sign of the numerator ``sigma2_lin - sigma2_diff``. The function
    returns the sign-aware limit (``p=0, reject=True`` for positive
    numerator; ``p=1, reject=False`` for negative; ``NaN`` for zero)
    with a ``UserWarning``, rather than unconditionally mapping this to
    ``p=1`` (which would flip a legitimate rejection).

    Survey/weighted data (Phase 4.5 C): when ``survey_design`` is
    supplied, all three variance components use their pweight-sandwich
    analogs:

    - ``sigma2_lin = sum(w * eps^2) / sum(w)`` (weighted OLS residual variance).
    - ``sigma2_diff = sum(w_avg * (dy_g - dy_{g-1})^2) / (2 * sum(w))``
      where ``w_avg_g = (w_g + w_{g-1}) / 2`` and the divisor uses
      ``sum(w)`` (not ``sum(w_avg)``) so the formula reduces bit-exactly
      to the unweighted ``(1/(2G))`` divisor at ``w = ones(G)``.
    - ``sigma4_W = sum(w_avg * eps_g^2 * eps_{g-1}^2) / sum(w_avg)`` with
      arithmetic-mean pair weights; reduces to the unweighted ``(1/(G-1))``
      divisor at ``w = ones(G)``.
    - ``T_hr = sqrt(sum(w)) * (sigma2_lin - sigma2_diff) / sigma2_W``.

    The pair-weight convention follows Krieger-Pfeffermann (1997, §3) for
    design-consistent inference on smooth functionals; PSU clustering is
    NOT propagated through the variance-ratio statistic. Strictly positive
    weights are required (the adjacent-difference formula has
    ``sum(w_avg)`` in the denominator). Per-unit constant-within-unit
    invariant on weights/strata/psu/fpc is the CALLER's responsibility.

    References
    ----------
    Yatchew, A. (1997). An elementary estimator of the partial linear
    model. Economics Letters 57, 135-143.
    de Chaisemartin et al. (2026), Theorem 7 / Equation 29.
    Krieger, A., Pfeffermann, D. (1997). Testing of distribution functions
    from complex sample surveys. Journal of Official Statistics 13(2),
    123-142.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")

    if null not in ("linearity", "mean_independence"):
        raise ValueError(
            f"yatchew_hr_test: null must be one of "
            f"('linearity', 'mean_independence'), got {null!r}."
        )

    # Type guard: array-in helpers reject SurveyDesign (cannot resolve
    # column names without `data`).
    if survey_design is not None and isinstance(survey_design, SurveyDesign):
        raise TypeError(
            "yatchew_hr_test: `survey_design=` accepts a pre-resolved "
            "ResolvedSurveyDesign only (array-in helpers have no `data` to "
            "resolve column names against). For pweight-only, use "
            "`survey_design=make_pweight_design(arr)`. For full PSU/strata/"
            "FPC, pre-resolve via `SurveyDesign(...).resolve(data)`."
        )

    # Internal variable naming: downstream code reads `survey`.
    survey = survey_design

    # Replicate-weight rejection.
    if survey is not None and getattr(survey, "replicate_weights", None) is not None:
        raise NotImplementedError(
            "yatchew_hr_test: replicate-weight survey designs (BRR/Fay/JK1/JKn/"
            "SDR) are not yet supported on HAD pretests. Replicate-weight "
            "pretests are a parallel follow-up after Phase 4.5 C."
        )
    # R1 P1: pweight-only guard (aweight/fweight have different sandwich-
    # variance semantics not derived for the variance-ratio statistic).
    if survey is not None and getattr(survey, "weight_type", "pweight") != "pweight":
        raise ValueError(
            f"yatchew_hr_test: HAD pretests require weight_type='pweight'. "
            f"Got weight_type={survey.weight_type!r}."
        )

    d_arr = _validate_1d_numeric(d, "d")
    dy_arr = _validate_1d_numeric(dy, "dy")
    if d_arr.shape[0] != dy_arr.shape[0]:
        raise ValueError(
            f"d and dy must have the same length; got d.shape={d_arr.shape}, "
            f"dy.shape={dy_arr.shape}."
        )
    # HAD support restriction (paper Section 2): doses must be non-negative.
    # Mirror the front-door guard from qug_test / _validate_had_panel.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"yatchew_hr_test: d contains {n_neg} negative value(s); HAD "
            f"doses must be non-negative (paper Section 2). Check your "
            f"dose column or pre-process before calling yatchew_hr_test."
        )

    G = int(d_arr.shape[0])
    critical_value = float(stats.norm.ppf(1.0 - alpha))

    # Phase 4.5 C: resolve effective per-unit weights. Strictly positive
    # required (the adjacent-difference formula divides by sum(w_avg) which
    # collapses to zero in any contiguous-zero block).
    # R4 P1: validate 1D explicitly so column-vector inputs raise.
    if survey is not None:
        w_arr = _validate_1d_numeric(np.asarray(survey.weights), "yatchew_hr_test: survey.weights")
        if w_arr.shape[0] != G:
            raise ValueError(
                f"yatchew_hr_test: survey.weights length {w_arr.shape[0]} "
                f"does not match d/dy length {G}."
            )
        if (w_arr <= 0).any():
            raise ValueError(
                "yatchew_hr_test: survey.weights must be strictly positive "
                "(adjacent-difference variance is undefined under contiguous "
                "zero-weight blocks)."
            )
    else:
        w_arr = None

    # R4 P0: normalize pweights to mean=1 (matches SurveyDesign.resolve()
    # convention). Yatchew uses sqrt(sum(w)) as the effective sample size,
    # which without normalization would scale as sqrt(c) under uniform
    # rescaling weights -> w * c, producing different p-values for w vs
    # 100*w. Normalization makes the statistic scale-invariant across every
    # survey_design= entry form (resolve() also normalizes to mean=1).
    if w_arr is not None:
        w_arr = w_arr * (float(w_arr.shape[0]) / float(np.sum(w_arr)))

    if G < _MIN_G_YATCHEW:
        warnings.warn(
            f"yatchew_hr_test: G = {G} is below the minimum {_MIN_G_YATCHEW} "
            f"(need at least 2 sorted differences). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return YatchewTestResults(
            t_stat_hr=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=float("nan"),
            sigma2_diff=float("nan"),
            sigma2_W=float("nan"),
            n_obs=G,
            null_form=null,
        )

    # Tie / constant-dose front-door guard. Yatchew's difference-based
    # variance estimator uses adjacent differences of dy sorted by d;
    # under tied doses the within-tie ordering is arbitrary (stable sort
    # falls back to input row order), so the statistic becomes
    # non-methodological and order-dependent. Reject at the front door
    # with a UserWarning + NaN result rather than silently permuting.
    # Mass-point designs and other tied-dose panels should use
    # `stute_test` instead (its tie-safe CvM handles ties correctly).
    n_unique_d = int(np.unique(d_arr).shape[0])
    if n_unique_d < G:
        n_dups = G - n_unique_d
        warnings.warn(
            f"yatchew_hr_test: d contains {n_dups} duplicate value(s) "
            f"(only {n_unique_d} distinct dose values out of G={G}); "
            f"the difference-based variance estimator is not well-defined "
            f"under ties because adjacent-difference statistics depend on "
            f"arbitrary within-tie row ordering. Use stute_test() instead "
            f"(its tie-safe CvM handles ties correctly). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return YatchewTestResults(
            t_stat_hr=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=float("nan"),
            sigma2_diff=float("nan"),
            sigma2_W=float("nan"),
            n_obs=G,
            null_form=null,
        )

    # 4-arm dispatch on (null, weighted). Unweighted-linearity path is
    # bit-exact pre-PR (stability invariant #1). Mean-independence mode
    # mirrors R YatchewTest::yatchew_test(order=0) and uses intercept-only
    # OLS residuals; the downstream sigma2_diff / sigma2_W path is
    # identical across nulls.
    if null == "linearity":
        if w_arr is None:
            _, _, eps = _fit_ols_intercept_slope(d_arr, dy_arr)
        else:
            _, _, eps = _fit_weighted_ols_intercept_slope(d_arr, dy_arr, w_arr)
    else:  # null == "mean_independence"
        if w_arr is None:
            _, _, eps = _fit_ols_intercept_only(dy_arr)
        else:
            _, _, eps = _fit_weighted_ols_intercept_only(dy_arr, w_arr)
    if w_arr is None:
        sigma2_lin = float(np.mean(eps * eps))
        sum_w = float(G)  # uniform-weights effective sample size = G
    else:
        sum_w = float(np.sum(w_arr))
        sigma2_lin = float(np.sum(w_arr * eps * eps) / sum_w)

    # Numerically exact linear fit: same short-circuit as `stute_test`.
    # Assumption 8 holds to IEEE precision; the Yatchew statistic is
    # formally -inf (finite-negative numerator over zero denominator),
    # which maps to p = 1 under the one-sided standard-normal critical
    # value. Short-circuit so FP noise in ``sigma4_W`` cannot produce a
    # spuriously large finite ``T_hr``. Comparison is purely relative
    # against CENTERED TSS - translation- AND scale-invariant.
    eps_norm_sq = float(np.sum(eps * eps))
    dy_centered_sq = float(np.sum((dy_arr - dy_arr.mean()) ** 2))
    if dy_centered_sq <= 0.0 or eps_norm_sq <= _EXACT_LINEAR_RELATIVE_TOL * dy_centered_sq:
        # Exact-linear branch. Covers two cases:
        # - dy_centered_sq == 0: dy is constant (trivially linear).
        # - relative SSR below IEEE precision: near-exact OLS fit.
        # For reporting, compute sigma2_diff on the sorted dy (finite,
        # well-defined even in the exact-linear case). Use the weighted
        # divisor (2 * sum(w)) when weights are supplied so the reported
        # sigma2_diff matches the same convention as the active branch.
        idx_early = np.argsort(d_arr, kind="stable")
        if w_arr is None:
            sigma2_diff_exact = float(np.sum(np.diff(dy_arr[idx_early]) ** 2) / (2.0 * G))
        else:
            w_s_e = w_arr[idx_early]
            w_avg_e = 0.5 * (w_s_e[1:] + w_s_e[:-1])
            sigma2_diff_exact = float(
                np.sum(w_avg_e * np.diff(dy_arr[idx_early]) ** 2) / (2.0 * sum_w)
            )
        return YatchewTestResults(
            t_stat_hr=float("-inf"),
            p_value=1.0,
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=sigma2_lin,
            sigma2_diff=sigma2_diff_exact,
            sigma2_W=0.0,
            n_obs=G,
            null_form=null,
        )

    idx = np.argsort(d_arr, kind="stable")
    dy_s = dy_arr[idx]
    eps_s = eps[idx]

    diff_dy = np.diff(dy_s)  # length G - 1
    if w_arr is None:
        # Paper-literal divisor: 2G (NOT 2(G-1)). This matches paper review
        # line 168: sigma2_diff := (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2).
        sigma2_diff = float(np.sum(diff_dy * diff_dy) / (2.0 * G))
        # sigma4_W = (1/(G-1)) * sum(eps_(g)^2 * eps_(g-1)^2) using np.mean
        # which divides by the length of the input (G-1 here). Matches paper
        # review line 171.
        sigma4_W = float(np.mean(eps_s[1:] ** 2 * eps_s[:-1] ** 2))
    else:
        # Phase 4.5 C: pweight-sandwich weighted variance components.
        # Pair-weights (Krieger-Pfeffermann 1997 §3): w_avg_g = (w_g + w_{g-1})/2.
        # Reduction at w=ones(G): w_avg = ones(G-1), sum(w) = G, so
        #   sigma2_diff = sum(diff^2) / (2*G)        (matches existing 2G divisor)
        #   sigma4_W   = sum(prod) / (G-1)            (matches existing G-1 divisor)
        w_s = w_arr[idx]
        w_avg = 0.5 * (w_s[1:] + w_s[:-1])
        sigma2_diff = float(np.sum(w_avg * diff_dy * diff_dy) / (2.0 * sum_w))
        sigma4_W = float(np.sum(w_avg * eps_s[1:] ** 2 * eps_s[:-1] ** 2) / np.sum(w_avg))
    if sigma4_W <= 0.0:
        # sigma4_W = 0 AFTER the exact-linear short-circuit means OLS
        # residuals are NOT zero (the shortcut already caught that case)
        # but every adjacent pair of sorted squared residuals contains a
        # zero (e.g. residuals alternate zero / nonzero after sort).
        # The formal test statistic is ±inf depending on the sign of the
        # numerator ``sigma2_lin - sigma2_diff``; mapping every such case
        # to p=1 (as an earlier revision did) can flip a legitimate
        # rejection into a fail-to-reject.
        warnings.warn(
            f"yatchew_hr_test: sigma4_W = 0 with non-zero residuals "
            f"(sigma2_lin = {sigma2_lin:.6g}, sigma2_diff = {sigma2_diff:.6g}); "
            f"the formal test statistic is infinite. Returning the "
            f"sign-aware limit decision.",
            UserWarning,
            stacklevel=2,
        )
        numerator = sigma2_lin - sigma2_diff
        if numerator > 0.0:
            # T_hr -> +inf: reject (far into right tail).
            t_stat_hr_val = float("inf")
            p_value_val = 0.0
            reject_val = True
        elif numerator < 0.0:
            # T_hr -> -inf: fail-to-reject.
            t_stat_hr_val = float("-inf")
            p_value_val = 1.0
            reject_val = False
        else:
            # 0/0: genuinely indeterminate.
            t_stat_hr_val = float("nan")
            p_value_val = float("nan")
            reject_val = False
        return YatchewTestResults(
            t_stat_hr=t_stat_hr_val,
            p_value=p_value_val,
            reject=reject_val,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=sigma2_lin,
            sigma2_diff=sigma2_diff,
            sigma2_W=0.0,
            n_obs=G,
            null_form=null,
        )
    sigma2_W = float(np.sqrt(sigma4_W))

    # Phase 4.5 C: effective sample size = sum(w) (=G under uniform weights,
    # so unweighted path is bit-exact).
    t_stat_hr = float(np.sqrt(sum_w) * (sigma2_lin - sigma2_diff) / sigma2_W)
    p_value = float(1.0 - stats.norm.cdf(t_stat_hr))
    reject = t_stat_hr >= critical_value

    return YatchewTestResults(
        t_stat_hr=t_stat_hr,
        p_value=p_value,
        reject=bool(reject),
        alpha=alpha,
        critical_value=critical_value,
        sigma2_lin=sigma2_lin,
        sigma2_diff=sigma2_diff,
        sigma2_W=sigma2_W,
        n_obs=G,
        null_form=null,
    )


def _validate_multi_period_panel(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> "tuple[Any, list, list, pd.DataFrame, Optional[Dict[str, Any]]]":
    """Validate a multi-period HAD panel for joint pre-test dispatch.

    Thin wrapper over :func:`_validate_had_panel_event_study` (had.py) that
    inherits the full contract:

    - ``first_treat=None`` combined with a staggered panel → raises
      ``ValueError`` (the had.py helper does NOT silently accept; it
      requires an explicit first-treatment column to identify cohorts).
    - ``first_treat`` provided but identifies only one cohort → no
      auto-filter, proceeds.
    - ``first_treat`` provided with multiple cohorts → auto-filters
      to last-cohort + never-treated, emits ``UserWarning`` with
      ``filter_info`` summary.
    - Requires ≥ 3 time periods, balanced panel, ordered time dtype, and
      the pre-period D=0 invariant across all pre-periods.

    Additional guards on top of had.py:
    - ``len(t_pre_list) >= 1`` (need ≥ 1 pre-period for joint pre-trends
      infrastructure; had.py already enforces this).
    - ``len(t_post_list) >= 1`` (need ≥ 1 post-period for joint
      homogeneity; had.py already enforces this).

    Returns the same 5-tuple as the had.py helper:
    ``(F, t_pre_list, t_post_list, data_filtered, filter_info)``.
    """
    return _validate_had_panel_event_study(
        data,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        first_treat_col=first_treat_col,
    )


def _build_period_rank(data: pd.DataFrame, time_col: str) -> Dict[Any, int]:
    """Build a ``{period_label: chronological_rank}`` map.

    For ordered categorical time columns, uses the declared category
    order so that e.g. ``["q1", "q2", "q10"]`` ranks chronologically
    even though it sorts lexically in the opposite order. For numeric
    or datetime time columns, uses natural Python `sorted` order on
    the unique period labels. Object dtypes would fall back to
    lexicographic order - callers relying on chronology with object-
    dtype labels should convert to an ordered categorical first
    (this mirrors the contract in ``_validate_had_panel_event_study``).

    The rank map lets the joint-pretest wrappers compare period labels
    chronologically via ``rank[t1] < rank[t2]`` instead of raw Python
    ``t1 < t2``, which would silently misorder ordered-categorical
    panels (paper Appendix B.2 support contract).
    """
    time_dtype = data[time_col].dtype
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        return {c: i for i, c in enumerate(time_dtype.categories)}
    periods = sorted(data[time_col].unique())
    return {p: i for i, p in enumerate(periods)}


def _aggregate_for_joint_test(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    horizons: list,
    base_period: Any,
) -> "tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]":
    """Aggregate a multi-period panel for a joint-Stute test.

    Builds per-horizon first differences ``dy_t = Y_{g,t} - Y_{g,base}``
    and the unit-level dose ``D_g`` for the joint-Stute test. All units
    must appear in every (horizon + base_period) period, matching the
    balanced-panel invariant of the single-period :func:`stute_test`.

    Dose extraction: ``D_g = max_t D_{g,t}`` under the HAD contract
    "once treated, stay treated with same dose". For pre-periods
    ``D_{g,t} = 0`` and for post-periods ``D_{g,t}`` is time-invariant
    per unit, so ``max`` recovers the realized post-period dose.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    horizons : list
        Non-empty list of period labels to build ``dy_t`` for.
        ``base_period`` must not be in ``horizons``. All ``horizons``
        and ``base_period`` must exist in the time column.
    base_period : period label
        The reference period for the first difference.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
    dy_by_horizon : dict[str, np.ndarray]
        Keys are ``str(t)`` per horizon, values are ``dy_t`` arrays of
        shape ``(G,)``. Insertion order follows ``horizons``.
    unit_ids : np.ndarray, shape (G,)
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")
    if len(horizons) == 0:
        raise ValueError("horizons must be a non-empty list of period labels.")
    data_periods = set(data[time_col].unique())
    needed_periods = list(horizons) + [base_period]
    missing_periods = [t for t in needed_periods if t not in data_periods]
    if missing_periods:
        raise ValueError(
            f"Period(s) {missing_periods} not found in the time column "
            f"{time_col!r}. Available periods: "
            f"{sorted(data_periods, key=lambda x: (x is None, x))}."
        )
    if base_period in horizons:
        raise ValueError(
            f"base_period={base_period!r} must not appear in horizons " f"{list(horizons)!r}."
        )

    mask = data[time_col].isin(needed_periods)
    subset = data.loc[mask].copy()

    for col in [outcome_col, dose_col, unit_col]:
        col_series = subset[col]
        if bool(pd.isna(col_series).any()):
            n_nan = int(pd.isna(col_series).sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r} across "
                f"periods {needed_periods}. Joint pre-test does not "
                f"silently drop rows; drop or impute before calling."
            )

    # Row-level non-negative-dose guard (paper Section 2 HAD support
    # restriction `D_{g,t} >= 0`). Must run BEFORE the groupby/max()
    # collapse below, otherwise a negative post dose would silently
    # become 0 in the per-unit dose vector (since `max(0, -d) = 0` for
    # positive d), letting the wrappers run on invalid data and
    # potentially return finite results. This is the direct-wrapper
    # equivalent of the row-level check inside
    # `_validate_had_panel_event_study`, centralized so both
    # `joint_pretrends_test` and `joint_homogeneity_test` inherit it on
    # the `n_periods < 3` fallback path that skips the validator.
    negative_dose_mask = subset[dose_col] < 0
    if bool(negative_dose_mask.any()):
        n_neg = int(negative_dose_mask.sum())
        raise ValueError(
            f"{n_neg} negative dose value(s) found in column "
            f"{dose_col!r} across periods {needed_periods}. HAD support "
            f"restriction (paper Section 2) requires D_{{g,t}} >= 0 "
            f"for every (unit, period)."
        )

    counts = subset.groupby(unit_col).size()
    n_needed = len(needed_periods)
    if (counts != n_needed).any():
        n_bad = int((counts != n_needed).sum())
        raise ValueError(
            f"Panel unbalanced across needed periods {needed_periods}: "
            f"{n_bad} unit(s) do not appear in all {n_needed} period(s). "
            f"Joint pre-test requires a balanced sub-panel."
        )

    wide_y = subset.pivot(index=unit_col, columns=time_col, values=outcome_col)
    wide_y = wide_y.sort_index()
    unit_ids = np.asarray(wide_y.index)

    base_y = wide_y[base_period].to_numpy(dtype=np.float64)
    dy_by_horizon: Dict[str, np.ndarray] = {}
    for t in horizons:
        y_t = wide_y[t].to_numpy(dtype=np.float64)
        dy_by_horizon[str(t)] = y_t - base_y

    # Dose per unit is the HAD time-invariant post-period dose:
    # D_g = max_t D_{g,t}. Critically, compute this over the FULL data,
    # not just the subset of needed_periods - for joint pre-trends,
    # needed_periods contains only pre-periods (all D=0), so taking max
    # over the subset would yield D_g = 0 for every unit and collapse
    # the CvM sort to arbitrary ties. Paper HAD convention: dose is
    # fixed per unit once treated; pre-period zero-dose is enforced by
    # the upstream validator.
    d_per_unit = data.groupby(unit_col)[dose_col].max().sort_index()
    # Align dose with the subset's unit ordering (pivot sort_index uses
    # natural unit_col order; groupby/sort_index on the full data gives
    # the same order).
    d_per_unit = d_per_unit.loc[unit_ids]
    d_arr = d_per_unit.to_numpy(dtype=np.float64)

    return d_arr, dy_by_horizon, unit_ids


def _compose_verdict_event_study(
    qug: QUGTestResults,
    pretrends_joint: Optional[StuteJointResult],
    homogeneity_joint: Optional[StuteJointResult],
) -> str:
    """Build the event-study :class:`HADPretestReport` verdict.

    Mirrors :func:`_compose_verdict` (two-period path) idiom verbatim:
    hyphen-separated ``"<concern> - <detail> (<source>)"`` reason
    strings, ``"; "`` join, ``"; additional steps unresolved: ..."``
    suffix for conclusive rejections that coexist with unresolved
    steps, lowercase concerns.

    Coverage:
    - Step 1 (QUG): always runs on the event-study path.
    - Step 2 (Assumption 7 pre-trends): runs via ``pretrends_joint``
      when at least one earlier pre-period is available. When skipped
      (only the immediate base pre-period), the verdict flags the skip
      but does NOT emit the Phase-3 "paper step 2 deferred to Phase 3
      follow-up" caveat - this PR closes that gap.
    - Step 3 (Assumption 8 linearity/homogeneity): runs via
      ``homogeneity_joint`` (joint Stute only; no joint Yatchew variant
      exists in the paper). The step-3 alternative Yatchew-HR test is
      subsumed by joint Stute on this path. (Paper step 4 is the
      decision itself - "use TWFE if none of the tests rejects" - not
      a separate diagnostic, so it has no code path here.)

    Priority:
    1. Any conclusive test rejecting → primary verdict bundles each
       rejection reason. Unresolved / skipped steps append as a suffix.
    2. No conclusive rejection but a required step unresolved →
       ``"inconclusive - ..."``.
    3. All required steps conclusive and none reject → admissible
       fail-to-reject string (Section 4 coverage).
    """
    qug_ok = bool(np.isfinite(qug.p_value))
    pretrends_ok = pretrends_joint is not None and bool(np.isfinite(pretrends_joint.p_value))
    homogeneity_ok = homogeneity_joint is not None and bool(np.isfinite(homogeneity_joint.p_value))

    qug_rej = qug_ok and qug.reject
    pretrends_rej = pretrends_joint is not None and pretrends_ok and bool(pretrends_joint.reject)
    homogeneity_rej = (
        homogeneity_joint is not None and homogeneity_ok and bool(homogeneity_joint.reject)
    )

    reasons = []
    if qug_rej:
        reasons.append("support infimum rejected - continuous_at_zero design invalid (QUG)")
    if pretrends_rej:
        reasons.append("joint pre-trends rejected - assumption 7 violated (joint Stute)")
    if homogeneity_rej:
        reasons.append("joint linearity rejected - heterogeneity bias (joint Stute)")

    unresolved = []
    if not qug_ok:
        unresolved.append("QUG NaN")
    if pretrends_joint is None:
        unresolved.append("joint pre-trends skipped (no earlier pre-period)")
    elif not pretrends_ok:
        unresolved.append("joint pre-trends NaN")
    if homogeneity_joint is None:
        unresolved.append("joint linearity skipped")
    elif not homogeneity_ok:
        unresolved.append("joint linearity NaN")

    if reasons:
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict

    if unresolved:
        return "inconclusive - " + "; ".join(unresolved)

    return (
        "QUG, joint pre-trends, and joint linearity diagnostics "
        "fail-to-reject (TWFE admissible under Section 4 assumptions)"
    )


def stute_joint_pretest(
    residuals_by_horizon: Dict[Any, np.ndarray],
    fitted_by_horizon: Dict[Any, np.ndarray],
    doses: np.ndarray,
    design_matrix: np.ndarray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    null_form: str = "custom",
    survey_design: Any = None,
) -> StuteJointResult:
    """Joint Cramer-von Mises pretest across multiple horizons.

    Generalizes :func:`stute_test` to K horizons with the joint
    statistic ``S_joint = sum_k S_k``, where ``S_k`` is the single-
    horizon CvM on residuals ``eps_{g,k}``. Inference is via Mammen wild
    bootstrap with a **shared** multiplier ``eta_g`` across horizons per
    unit to preserve the vector-valued empirical process's unit-level
    dependence.

    **Note:** sum-of-CvMs aggregation follows the standard joint
    specification-test construction (Delgado 1993; Escanciano 2006). The
    paper does not prescribe an aggregation; sum-of-CvMs balances power
    across diffuse vs concentrated alternatives and bootstraps cleanly
    with the shared-eta structure.

    Bootstrap uses the literal per-iteration OLS refit form (paper
    Appendix D) for consistency with Phase 3's :func:`stute_test`.
    ``XtX_inv_Xt`` is precomputed once (same design matrix each
    iteration), so the refit cost is O(Gp) per bootstrap draw and the
    overall loop is dominated by :func:`_cvm_statistic` across K
    horizons.

    Parameters
    ----------
    residuals_by_horizon : dict[str, np.ndarray]
        ``{label: eps_g}`` per horizon. All values must have identical
        length ``G`` and be unit-ordered consistently with ``doses``.
    fitted_by_horizon : dict[str, np.ndarray]
        ``{label: fitted_g}`` per horizon. Required to reconstruct
        bootstrap outcomes ``dy*_{g,k} = fitted_{g,k} + eps_{g,k} *
        eta_g`` under the null.
    doses : np.ndarray, shape (G,)
        Dose per unit. Shared across horizons (HAD contract: dose is
        time-invariant per unit). Must be finite and non-negative.
    design_matrix : np.ndarray, shape (G, p)
        Regression design used in the per-horizon bootstrap refit.
        Mean-independence: ``[1]`` (intercept only). Linearity:
        ``[1, doses]``. The matrix is identical across horizons.
    alpha, n_bootstrap, seed : see :func:`stute_test`.
    null_form : str
        Diagnostic label recorded on the result
        (``"mean_independence"`` | ``"linearity"`` | ``"custom"``).
        The wrappers :func:`joint_pretrends_test` and
        :func:`joint_homogeneity_test` set this automatically.
    survey_design : ResolvedSurveyDesign or None, keyword-only, default None
        Already-resolved per-unit survey design (Phase 4.5 C). Array-in
        helpers accept ``ResolvedSurveyDesign`` ONLY; passing a
        ``SurveyDesign`` raises ``TypeError``. For pweight-only, use
        ``survey_design=make_pweight_design(arr)``. When supplied, the
        bootstrap is a PSU-level Mammen multiplier bootstrap with the
        multiplier matrix shared across horizons within each replicate
        (preserves both vector-valued empirical-process unit-level
        dependence + PSU clustering). Replicate-weight designs raise
        ``NotImplementedError``; non-pweight weight types are rejected.
        Variance-unidentified designs (``df_survey <= 0``) return NaN
        with a ``UserWarning`` instead of calibrating against an
        all-zero multiplier matrix.

    Returns
    -------
    StuteJointResult
        On the common path, a populated result with bootstrap-based
        ``p_value`` and ``cvm_stat_joint``. On the small-sample branch
        (``G < _MIN_G_STUTE``), constant-dose branch
        (``np.ptp(doses) <= 0``), or any-NaN branch in the input
        residuals / fitted arrays, returns an all-NaN result (with
        ``reject=False`` and the full ``per_horizon_stats`` dict keyed
        by the validated horizon labels) and emits a ``UserWarning``
        for the first two branches. Mirrors the single-horizon
        :func:`stute_test` contract so event-study workflows on small
        or staggered-filtered panels surface an inconclusive report
        rather than crashing.

    Raises
    ------
    ValueError
        On empty input, key-mismatch, stringified-label collisions
        between distinct raw keys, shape-mismatch, ``doses`` containing
        negative values, ``n_bootstrap < _MIN_N_BOOTSTRAP``, or invalid
        ``alpha``. ``G < _MIN_G_STUTE`` does NOT raise; see Returns.
    """
    # Type guard: array-in helpers reject SurveyDesign (cannot resolve
    # column names without `data`).
    if survey_design is not None and isinstance(survey_design, SurveyDesign):
        raise TypeError(
            "stute_joint_pretest: `survey_design=` accepts a pre-resolved "
            "ResolvedSurveyDesign only (array-in helpers have no `data` to "
            "resolve column names against). For pweight-only, use "
            "`survey_design=make_pweight_design(arr)`. For full PSU/strata/"
            "FPC, pre-resolve via `SurveyDesign(...).resolve(data)`."
        )

    # Internal variable naming: downstream code reads `survey`.
    survey = survey_design

    # Replicate-weight rejection.
    if survey is not None and getattr(survey, "replicate_weights", None) is not None:
        raise NotImplementedError(
            "stute_joint_pretest: replicate-weight survey designs (BRR/Fay/JK1/"
            "JKn/SDR) are not yet supported on HAD pretests. Replicate-weight "
            "pretests are a parallel follow-up after Phase 4.5 C."
        )
    # R1 P1: pweight-only guard.
    if survey is not None and getattr(survey, "weight_type", "pweight") != "pweight":
        raise ValueError(
            f"stute_joint_pretest: HAD pretests require weight_type='pweight'. "
            f"Got weight_type={survey.weight_type!r}."
        )

    if not isinstance(residuals_by_horizon, dict) or not isinstance(fitted_by_horizon, dict):
        raise ValueError(
            "residuals_by_horizon and fitted_by_horizon must be dicts " "keyed by horizon label."
        )
    if len(residuals_by_horizon) == 0:
        raise ValueError("residuals_by_horizon must contain at least one horizon.")
    if set(residuals_by_horizon.keys()) != set(fitted_by_horizon.keys()):
        raise ValueError(
            "residuals_by_horizon and fitted_by_horizon must have "
            "identical keys. Got "
            f"residuals keys: {sorted(residuals_by_horizon.keys())!r}, "
            f"fitted keys: {sorted(fitted_by_horizon.keys())!r}."
        )

    doses_arr = _validate_1d_numeric(np.asarray(doses), "doses")
    G = doses_arr.shape[0]
    if np.any(doses_arr < 0):
        raise ValueError(
            "doses must be non-negative (HAD contract - paper Section 2). "
            f"Found {int(np.sum(doses_arr < 0))} negative value(s)."
        )

    # G < _MIN_G_STUTE (CvM statistic not well-calibrated): mirror the
    # single-horizon `stute_test` contract - warn + return NaN result
    # rather than raise, so callers (including the event-study workflow
    # on a staggered panel whose last-cohort filter leaves fewer than
    # 10 units) get an inconclusive diagnostic instead of a crash. The
    # NaN return still satisfies the workflow's `np.isfinite(p_value)`
    # gating, so `all_pass` becomes False downstream.
    # Note: the actual `warn + return` happens below after horizon
    # labels are validated and collision-checked, so the NaN result
    # carries full per-horizon diagnostic keys.
    if n_bootstrap < _MIN_N_BOOTSTRAP:
        raise ValueError(f"n_bootstrap must be >= {_MIN_N_BOOTSTRAP}; got " f"{n_bootstrap}.")
    if not isinstance(alpha, (int, float)) or not (0 < float(alpha) < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    X = np.asarray(design_matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != G:
        raise ValueError(f"design_matrix must have shape (G, p) with G={G}; got " f"{X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("design_matrix contains non-finite values (NaN/inf).")

    raw_horizon_labels = list(residuals_by_horizon.keys())
    K = len(raw_horizon_labels)

    # Stringified-label collision guard: distinct raw keys whose str()
    # representations collide (e.g. {1: ..., "1": ..., 1.0: ...}) would
    # overwrite each other in residuals_arrays / fitted_arrays, letting
    # the surviving horizon be double-counted in S_joint = sum of S_k
    # and leaving `n_horizons` inconsistent with the number of distinct
    # diagnostic statistics. Reject explicitly rather than silently
    # collapsing the test.
    str_labels = [str(k) for k in raw_horizon_labels]
    if len(set(str_labels)) != len(str_labels):
        from collections import Counter

        dup_strs = [s for s, c in Counter(str_labels).items() if c > 1]
        collisions = {s: [k for k in raw_horizon_labels if str(k) == s] for s in dup_strs}
        raise ValueError(
            f"Horizon label collision after str() stringification: "
            f"{collisions!r}. The joint Stute helpers index residuals "
            f"and fitted values by str(label); distinct raw keys whose "
            f"stringified form collides would silently overwrite each "
            f"other and double-count the surviving horizon in S_joint. "
            f"Use string-distinct horizon labels (e.g. 1997 and 1998 "
            f'as int, or "1997" and "1998" as str; not both).'
        )

    any_nan = False
    residuals_arrays: Dict[str, np.ndarray] = {}
    fitted_arrays: Dict[str, np.ndarray] = {}
    for k in raw_horizon_labels:
        eps_k = np.asarray(residuals_by_horizon[k], dtype=np.float64)
        fit_k = np.asarray(fitted_by_horizon[k], dtype=np.float64)
        if eps_k.shape != (G,) or fit_k.shape != (G,):
            raise ValueError(
                f"Horizon {k!r}: residuals shape {eps_k.shape} and "
                f"fitted shape {fit_k.shape} must both be ({G},) to "
                f"align with doses."
            )
        if not (np.all(np.isfinite(eps_k)) and np.all(np.isfinite(fit_k))):
            any_nan = True
        residuals_arrays[str(k)] = eps_k
        fitted_arrays[str(k)] = fit_k

    # Re-key to str labels consistently (wrappers already pass str; direct
    # callers may pass int/object). String identity per the documented
    # horizon_labels contract. The collision guard above ensures this
    # stringification is injective on the provided keys.
    horizon_labels = str_labels

    # Small-G NaN result (paired with the comment near the top of this
    # function): mirror the single-horizon stute_test contract so the
    # event-study workflow on a small or staggered-filtered panel gets
    # an inconclusive diagnostic rather than an exception. Positioned
    # AFTER the label-collision / shape-alignment guards so the NaN
    # result carries a consistent per-horizon diagnostic surface.
    if G < _MIN_G_STUTE:
        warnings.warn(
            f"stute_joint_pretest: G = {G} is below the minimum "
            f"{_MIN_G_STUTE} for the CvM statistic to be well-calibrated. "
            f"Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    if any_nan:
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    # Zero-variation-in-D degeneracy guard: mirrors stute_test's intent
    # (had_pretests.py:~1233). The CvM cusum is defined against the
    # dose regressor; constant d has no cross-sectional variation for
    # the test to detect nonlinearity. Under the mean-independence null
    # this yields a mechanically-zero statistic (bogus fail-to-reject);
    # under the linearity null a singular [1, d] design matrix crashes
    # the refit. Emit warning + NaN result instead.
    #
    # Uses ``ptp`` (peak-to-peak = max - min) rather than ``np.var`` for
    # the degeneracy check: ``np.var`` of a truly constant array returns
    # a small non-zero value (~1e-32) due to E[X^2] - E[X]^2 rounding
    # noise, so a ``<= 0`` comparison misses the degeneracy. ``ptp`` is
    # bit-exact for identical inputs.
    if float(np.ptp(doses_arr)) <= 0.0:
        warnings.warn(
            "stute_joint_pretest: constant doses (zero cross-sectional "
            "variation); the joint Stute CvM requires dose variation. "
            "Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    # Phase 4.5 C: resolve effective per-unit weights (None → bit-exact
    # unweighted path).
    # R4 P1: validate 1D explicitly so column-vector inputs raise.
    if survey is not None:
        w_arr = _validate_1d_numeric(
            np.asarray(survey.weights), "stute_joint_pretest: survey.weights"
        )
        if w_arr.shape[0] != G:
            raise ValueError(
                f"stute_joint_pretest: survey.weights length {w_arr.shape[0]} "
                f"does not match doses length {G}."
            )
        # R1 P0: strictly-positive guard (mirrors stute_test single-horizon).
        if (w_arr <= 0).any():
            raise ValueError(
                "stute_joint_pretest: survey weights must be strictly "
                "positive. Zero / negative weights would leave units in "
                "the variance / CvM computation while contributing zero "
                "population mass."
            )
    else:
        w_arr = None

    # R4 P0: normalize pweights to mean=1 (matches SurveyDesign.resolve()
    # convention; same fix as stute_test / yatchew_hr_test).
    if w_arr is not None:
        w_arr = w_arr * (float(w_arr.shape[0]) / float(np.sum(w_arr)))

    idx = np.argsort(doses_arr, kind="stable")
    d_sorted = doses_arr[idx]

    per_horizon_stats: Dict[str, float] = {}
    if w_arr is None:
        for k in horizon_labels:
            per_horizon_stats[k] = _cvm_statistic(residuals_arrays[k][idx], d_sorted)
    else:
        w_sorted = w_arr[idx]
        for k in horizon_labels:
            per_horizon_stats[k] = _cvm_statistic_weighted(
                residuals_arrays[k][idx], d_sorted, w_sorted
            )
    S_joint = float(sum(per_horizon_stats.values()))

    # Per-horizon exact-linear short-circuit (scale- and translation-
    # invariant, matches Phase 3 invariant). A single degenerate horizon
    # does NOT collapse the joint test if other horizons have nontrivial
    # residuals.
    short_circuit = True
    for k in horizon_labels:
        eps_k = residuals_arrays[k]
        fit_k = fitted_arrays[k]
        dy_k = fit_k + eps_k
        tss_centered = float(np.sum((dy_k - dy_k.mean()) ** 2))
        if tss_centered == 0.0:
            # Outcome identically constant: treat as trivially linear for
            # this horizon (ratio = 0). Does not force short-circuit
            # because other horizons may still be nontrivial.
            ratio = 0.0
        else:
            ratio = float(np.sum(eps_k**2) / tss_centered)
        if ratio >= _EXACT_LINEAR_RELATIVE_TOL:
            short_circuit = False
            break

    if short_circuit:
        return StuteJointResult(
            cvm_stat_joint=S_joint,
            p_value=1.0,
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats=per_horizon_stats,
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=True,
        )

    # Precompute OLS projection matrix once: same X per bootstrap draw,
    # so (X'X)^-1 X' is constant across iterations. Keeps refit O(Gp)
    # per draw without changing semantics from the literal paper form.
    # Weighted variant uses (X' W X)^-1 X' W; same precompute idiom.
    # Catch rank-deficient designs explicitly rather than surfacing a
    # raw ``np.linalg.LinAlgError`` to direct callers of the public
    # residuals-in core; matches the front-door validation style of
    # the other guards in this function.
    try:
        if w_arr is None:
            XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
        else:
            # Weighted OLS projection: (X' W X)^-1 X' W
            XtWX = X.T @ (w_arr[:, np.newaxis] * X)
            XtW = X.T * w_arr  # broadcasts (p, G)
            XtX_inv_Xt = np.linalg.solve(XtWX, XtW)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"design_matrix is rank-deficient (singular X^T X); cannot "
            f"compute the OLS projection (X^T X)^-1 X^T for the "
            f"bootstrap refit. Check for duplicate or linearly-"
            f"dependent columns. shape={X.shape}."
        ) from exc

    rng = np.random.default_rng(seed)
    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)

    if w_arr is None:
        # Unweighted bit-exact path (stability invariant #1). Same hoisted
        # invariants as stute_test's unweighted loop: the eta draws come
        # from memory-bounded batched rng.choice calls (identical variate
        # stream as per-iteration draws — see _iter_mammen_rows) and the
        # CvM tie-block indices are precomputed once — bootstrap_S is
        # bit-identical to the per-iteration form.
        _, tie_counts = np.unique(d_sorted, return_counts=True)
        tie_blocks = (np.cumsum(tie_counts) - 1, tie_counts)
        # SHARED eta across horizons - preserves unit-level dependence
        # in the vector-valued empirical process. Independent-per-horizon
        # draws would overstate precision.
        for b, eta in enumerate(_iter_mammen_rows(n_bootstrap, G, rng)):
            S_b = 0.0
            for k in horizon_labels:
                dy_b = fitted_arrays[k] + residuals_arrays[k] * eta
                beta_b = XtX_inv_Xt @ dy_b
                eps_b = dy_b - X @ beta_b
                S_b += _cvm_statistic(eps_b[idx], d_sorted, tie_blocks=tie_blocks)
            bootstrap_S[b] = S_b
    else:
        # Phase 4.5 C survey-aware path: PSU-level Mammen multipliers
        # SHARED across horizons within each replicate. The (B, n_psu)
        # matrix is drawn ONCE per replicate; the per-horizon loop
        # broadcasts the SAME multipliers, preserving both the
        # vector-valued empirical-process unit-level dependence (paper
        # convention) AND PSU clustering (Krieger-Pfeffermann 1997).
        resolved_for_boot = survey if survey is not None else make_pweight_design(w_arr)
        # Stratified designs are supported via the standard stratified
        # clustered wild-bootstrap correction on the PSU multipliers
        # (within-stratum demean + sqrt(n_h/(n_h-1)) Bessel rescale),
        # applied uniformly before the per-obs broadcast eta_obs =
        # psu_mults[b, psu_col_idx] below. The joint variant shares the
        # SAME multiplier row across horizons within each replicate, so
        # the stratum correction applies once and inherits across
        # horizons (preserving cross-horizon empirical-process
        # dependence per Hlávka & Huškova 2020 § 3). See REGISTRY
        # § "Note (Stute stratified survey-bootstrap calibration)".
        # R5 P1: reject lonely_psu='adjust' singleton-strata designs.
        # Same pseudo-stratum centering gap as stute_test / HAD sup-t.
        if _has_lonely_psu_adjust_singletons(resolved_for_boot):
            raise NotImplementedError(
                "stute_joint_pretest: SurveyDesign(lonely_psu='adjust') "
                "with singleton strata is not yet supported on the "
                "multiplier bootstrap. The bootstrap helper pools "
                "singletons with nonzero multipliers but the matching "
                "analytical variance target requires a pseudo-stratum "
                "centering transform that has not been derived for the "
                "Stute CvM. Use lonely_psu='remove' (drops singleton "
                "contributions) or 'certainty' (zero-variance "
                "singletons), or pre-process the panel to remove "
                "singleton strata."
            )
        # R3 P0: variance-unidentified survey-design guard (mirrors
        # stute_test single-horizon).
        df_survey = resolved_for_boot.df_survey
        if df_survey is None or df_survey <= 0:
            warnings.warn(
                f"stute_joint_pretest: survey design is variance-"
                f"unidentified (df_survey={df_survey}); the multiplier "
                "bootstrap cannot calibrate the joint test. Returning "
                "NaN result.",
                UserWarning,
                stacklevel=2,
            )
            return StuteJointResult(
                cvm_stat_joint=S_joint,
                p_value=float("nan"),
                reject=False,
                alpha=float(alpha),
                horizon_labels=horizon_labels,
                per_horizon_stats=per_horizon_stats,
                n_bootstrap=int(n_bootstrap),
                n_obs=int(G),
                n_horizons=int(K),
                seed=None if seed is None else int(seed),
                null_form=str(null_form),
                exact_linear_short_circuited=False,
            )
        psu_mults, psu_ids = generate_survey_multiplier_weights_batch(
            n_bootstrap, resolved_for_boot, weight_type="mammen", rng=rng
        )
        # Stratum centering + Bessel rescale on the PSU multipliers
        # before broadcast. Single application here (shared with the
        # per-horizon loop below) propagates the same centered
        # multipliers across all horizons in each replicate, preserving
        # the joint Stute's cross-horizon empirical-process dependence.
        # See REGISTRY § "Note (Stute stratified survey-bootstrap
        # calibration)".
        apply_stratum_centering(psu_mults, resolved_for_boot, psu_ids, psu_axis=1)
        if resolved_for_boot.psu is None:
            psu_col_idx = np.arange(G)
        else:
            psu_to_col = {int(p): c for c, p in enumerate(psu_ids)}
            psu_arr = np.asarray(resolved_for_boot.psu)
            psu_col_idx = np.array([psu_to_col[int(psu_arr[g])] for g in range(G)])
        w_sorted = w_arr[idx]

        for b in range(n_bootstrap):
            eta_obs = psu_mults[b, psu_col_idx]  # (G,) - shared across horizons
            S_b = 0.0
            for k in horizon_labels:
                dy_b = fitted_arrays[k] + residuals_arrays[k] * eta_obs
                beta_b = XtX_inv_Xt @ dy_b
                eps_b = dy_b - X @ beta_b
                S_b += _cvm_statistic_weighted(eps_b[idx], d_sorted, w_sorted)
            bootstrap_S[b] = S_b

    p_value = float((1.0 + np.sum(bootstrap_S >= S_joint)) / (n_bootstrap + 1))
    reject = bool(p_value <= alpha)

    return StuteJointResult(
        cvm_stat_joint=S_joint,
        p_value=p_value,
        reject=reject,
        alpha=float(alpha),
        horizon_labels=horizon_labels,
        per_horizon_stats=per_horizon_stats,
        n_bootstrap=int(n_bootstrap),
        n_obs=int(G),
        n_horizons=int(K),
        seed=None if seed is None else int(seed),
        null_form=str(null_form),
        exact_linear_short_circuited=False,
    )


def _resolve_pretest_unit_weights(
    data: pd.DataFrame,
    unit_col: str,
    weights: Optional[np.ndarray],
    survey: Any,
    caller_name: str,
) -> "tuple[Optional[np.ndarray], Optional[Any]]":
    """Resolve the per-row ``survey_design`` input to per-unit (G,) form.

    Used by ``joint_pretrends_test``, ``joint_homogeneity_test``, and
    ``did_had_pretest_workflow`` (data-in entry points). Reuses the HAD
    helpers ``_aggregate_unit_weights`` and ``_aggregate_unit_resolved_survey``
    which enforce constant-within-unit invariant on weights and on every
    survey design column (strata, psu, fpc).

    ``weights`` is INTERNAL LEGACY plumbing: the public ``weights=`` alias
    was removed in 3.7.x and every current caller passes ``None``; the
    parameter and its per-unit aggregation branch are retained only so
    internal callers could thread a raw per-row array without a column
    (none do today). Replicate-weight survey designs raise
    ``NotImplementedError`` (deferred to a parallel follow-up after
    Phase 4.5 C).

    Returns
    -------
    (weights_unit, resolved_unit) : Tuple[Optional[np.ndarray], Optional[ResolvedSurveyDesign]]
        - If neither input supplied: ``(None, None)`` (unweighted path).
        - If ``weights`` supplied (internal only): ``(weights_unit, None)``.
        - If ``survey`` supplied: ``(None, resolved_unit)`` where
          ``resolved_unit.weights`` is the per-unit weight vector and
          strata/psu/fpc are also per-unit.
    """
    if weights is None and survey is None:
        return None, None
    if weights is not None and survey is not None:
        raise ValueError(f"{caller_name}: internal error - weights and survey both set.")
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        # R4 P1: validate 1D explicitly (column-vector inputs would otherwise
        # broadcast through downstream computations and silently corrupt
        # results).
        if weights_arr.ndim != 1:
            raise ValueError(
                f"{caller_name}: weights must be 1-dimensional, got shape "
                f"{weights_arr.shape}. (A common mistake is passing "
                "df[['w']].to_numpy() which produces (N, 1); use "
                "df['w'].to_numpy() for (N,).)"
            )
        weights_unit = _aggregate_unit_weights(data, weights_arr, unit_col)
        # R1 P0: strictly-positive weights required (matches the array-in
        # helper entry behavior; the CvM cusum + adjacent-difference
        # variance assume all rows contribute).
        if (weights_unit <= 0).any():
            raise ValueError(
                f"{caller_name}: weights must be strictly positive at the "
                "per-unit level. Zero / negative weights would leave units "
                "in the variance/CvM computation while contributing zero "
                "mass; use survey_design= with explicit lonely-PSU handling "
                "for principled subpopulation analysis."
            )
        # R4 P0: normalize per-unit weights to mean=1 (matches
        # SurveyDesign.resolve() convention so every entry path produces
        # identical statistic values; ensures Yatchew is scale-invariant
        # under uniform rescaling).
        weights_unit = weights_unit * (float(weights_unit.shape[0]) / float(np.sum(weights_unit)))
        return weights_unit, None
    # survey is not None
    if not hasattr(survey, "resolve"):
        # PR #376 R9 P3: error message names the canonical kwarg
        # `survey_design=` and points pre-resolved-design users to the
        # array-in pretest helpers where ResolvedSurveyDesign /
        # make_pweight_design(arr) belong.
        raise TypeError(
            f"{caller_name}: `survey_design=` accepts a SurveyDesign "
            f"instance (column-referencing, "
            f"gets `.resolve(data)`'d at fit time) on data-in surfaces; "
            f"got {type(survey).__name__} (no `.resolve()` method). "
            "If you have a pre-resolved ResolvedSurveyDesign or used "
            "`make_pweight_design(arr)`, that pattern is for the array-in "
            "pretest helpers (`stute_test`, `yatchew_hr_test`, "
            "`stute_joint_pretest`). On data-in surfaces, add the weights "
            "as a column on `data` and pass "
            "`survey_design=SurveyDesign(weights='col_name', ...)`."
        )
    resolved_full = survey.resolve(data)
    if getattr(resolved_full, "replicate_weights", None) is not None:
        raise NotImplementedError(
            f"{caller_name}: replicate-weight survey designs (BRR/Fay/JK1/JKn/"
            "SDR) are not yet supported on HAD pretests. Replicate-weight "
            "pretests are a parallel follow-up after Phase 4.5 C."
        )
    # R1 P1: pweight-only guard. aweight/fweight slip through pweight-only
    # formulas silently otherwise (mirrors HeterogeneousAdoptionDiD.fit() at
    # had.py:2976+ and survey._resolve_pweight_only at survey.py:914).
    if getattr(resolved_full, "weight_type", "pweight") != "pweight":
        raise ValueError(
            f"{caller_name}: HAD pretests require weight_type='pweight'. "
            f"Got weight_type={resolved_full.weight_type!r}. aweight / "
            "fweight have different sandwich-variance semantics that are "
            "not derived for the pretest variance components."
        )
    resolved_unit = _aggregate_unit_resolved_survey(data, resolved_full, unit_col)
    # R1 P0: strictly-positive weights at the per-unit level (mirrors the
    # weights= shortcut). Zero per-unit weights leave units in the dose-
    # variation check / CvM sum while contributing zero population mass,
    # which can produce silently-wrong pretest decisions.
    if (np.asarray(resolved_unit.weights) <= 0).any():
        raise ValueError(
            f"{caller_name}: survey weights must be strictly positive at "
            "the per-unit level. Zero / negative weights would leave units "
            "in the variance/CvM computation while contributing zero "
            "mass; this would produce silent wrong pretest decisions on "
            "subpopulation-restricted designs. Pre-filter the panel to "
            "the positive-weight subpopulation before calling the workflow."
        )
    return None, resolved_unit


def joint_pretrends_test(
    data: pd.DataFrame,
    outcome: Any = NOT_SUPPLIED,
    dose: Any = NOT_SUPPLIED,
    time: Any = NOT_SUPPLIED,
    unit: Any = NOT_SUPPLIED,
    pre_periods: Any = NOT_SUPPLIED,
    base_period: Any = NOT_SUPPLIED,
    first_treat: Any = NOT_SUPPLIED,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    survey_design: Any = None,
    trends_lin: bool = False,
    outcome_col: Any = NOT_SUPPLIED,
    dose_col: Any = NOT_SUPPLIED,
    time_col: Any = NOT_SUPPLIED,
    unit_col: Any = NOT_SUPPLIED,
    first_treat_col: Any = NOT_SUPPLIED,
) -> StuteJointResult:
    """Joint Stute pre-trends test (paper Section 4.2 step 2).

    Data-in wrapper around :func:`stute_joint_pretest` for the
    mean-independence null
    ``E[Y_{g,t} - Y_{g,base} | D_{g,treat}] = mu_t``
    across multiple pre-period placebos. For each ``t in pre_periods``,
    residuals are the deviations of ``Y_{g,t} - Y_{g,base}`` from their
    cross-unit mean (an intercept-only OLS fit); the joint CvM tests
    that the conditional mean depends on ``D``.

    Use this wrapper to close the paper's step-2 pre-trends gap that
    :func:`did_had_pretest_workflow` otherwise flags. On a panel with
    at least one earlier pre-period, the
    ``aggregate="event_study"`` dispatch calls this wrapper internally.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, dose, time, unit : str
    pre_periods : list
        Non-empty list of pre-period labels (all ``< base_period``, all
        with ``D = 0`` across every unit). Empty list raises; the
        workflow dispatch handles the "no earlier pre-period" case by
        setting ``pretrends_joint=None`` rather than calling this
        wrapper.
    base_period : period label
        The reference period. Must not be in ``pre_periods``. Must also
        satisfy ``D = 0`` across every unit (reciprocal of the pre-period
        HAD invariant - base is itself a pre-period in the four-step
        workflow).
    first_treat : str or None
        Forwarded to the underlying panel validator; matched cohort
        handling follows the HAD contract (staggered auto-filter warns
        and proceeds on last cohort; solo cohort proceeds).
    alpha, n_bootstrap, seed : as in :func:`stute_test`.
    survey_design : SurveyDesign or None, keyword-only, default None
        Survey design (Phase 4.5 C). Resolved on the filtered panel;
        replicate-weight designs raise ``NotImplementedError``;
        ``weight_type`` must be ``"pweight"``. Forwarded to
        :func:`stute_joint_pretest` as a per-unit
        ``ResolvedSurveyDesign``.
    trends_lin : bool, default False, keyword-only
        When ``True``, applies paper Eq 17 / Eq 18 linear-trend
        detrending: per-group slope estimated as ``Y[g, base] -
        Y[g, base - 1]`` and subtracted from each pre-period horizon's
        outcome evolution as ``(t - base) × slope``. Mirrors R
        ``DIDHAD::did_had(..., trends_lin=TRUE)`` on its joint Stute
        pre-trends surface (paper Section 5.2 Pierce-Schott
        application). **Requires** ``base_period`` to equal the last
        validated pre-period (``t_pre_list[-1]``, i.e. the canonical
        ``F-1`` anchor). Direct callers passing a non-terminal base
        get a ``ValueError`` — Eq 17 / R both anchor at ``F-1`` and
        any other anchor would compute a different slope and
        detrending. The previous validated pre-period
        (``t_pre_list[-2]``, ``F-2``) must also be present so the
        slope is identified. The "consumed" placebo at ``F-2`` is
        dropped from ``pre_periods`` explicitly (its detrended
        residual is mechanically zero by construction); a
        ``UserWarning`` fires when the filter triggers. If
        ``pre_periods`` becomes empty after the drop, raises
        ``ValueError`` (no testable placebo horizons remain).
        Mutually exclusive with survey weighting (``survey_design``);
        raises ``NotImplementedError`` if combined. Default ``False`` preserves bit-exact backcompat.

    Returns
    -------
    StuteJointResult with ``null_form = "mean_independence"``.
    """
    _q = "joint_pretrends_test"
    outcome = resolve_renamed_kwarg(
        _q, "outcome_col", outcome_col, "outcome", outcome, default=NOT_SUPPLIED
    )
    require_arg(_q, "outcome", outcome)
    dose = resolve_renamed_kwarg(_q, "dose_col", dose_col, "dose", dose, default=NOT_SUPPLIED)
    require_arg(_q, "dose", dose)
    time = resolve_renamed_kwarg(_q, "time_col", time_col, "time", time, default=NOT_SUPPLIED)
    require_arg(_q, "time", time)
    unit = resolve_renamed_kwarg(_q, "unit_col", unit_col, "unit", unit, default=NOT_SUPPLIED)
    require_arg(_q, "unit", unit)
    require_arg(_q, "pre_periods", pre_periods)
    require_arg(_q, "base_period", base_period)
    first_treat = resolve_renamed_kwarg(
        _q, "first_treat_col", first_treat_col, "first_treat", first_treat, default=None
    )
    # Body-local names; the public parameters are the bare names.
    outcome_col = outcome
    dose_col = dose
    time_col = time
    unit_col = unit
    first_treat_col = first_treat
    # Internal variable naming: downstream code reads `survey` (Phase 4.5 C
    # convention).
    survey = survey_design

    # ---- trends_lin × survey_design gate (PR #389 / Phase 4 R-parity). ----
    # Detrending under survey weighting (weighted slope? per-PSU slope?)
    # is not derived from the paper. Use trends_lin without survey weights,
    # OR survey weights without trends_lin. Tracked as DEFERRED.md follow-up.
    if trends_lin and survey is not None:
        raise NotImplementedError(
            "joint_pretrends_test(trends_lin=True) is not yet supported "
            "with survey weighting (`survey_design=`). The per-group "
            "slope estimator's weighted "
            "variant is not derived from the paper. Use trends_lin=True "
            "WITHOUT survey weights, or use survey weights WITHOUT "
            "trends_lin. Tracked in DEFERRED.md as a follow-up if user "
            "demand emerges."
        )

    if len(pre_periods) == 0:
        raise ValueError(
            "pre_periods must be non-empty. Workflow dispatch handles "
            "the empty case by setting pretrends_joint=None; direct "
            "callers should not pass an empty list."
        )
    if base_period in pre_periods:
        raise ValueError(
            f"base_period={base_period!r} must not appear in " f"pre_periods {list(pre_periods)!r}."
        )

    # Ordering check: all pre_periods strictly < base_period in
    # chronological order. Uses `_build_period_rank` to handle ordered-
    # categorical time columns correctly (raw Python `<` would fail on
    # categories whose lexical order disagrees with chronology, e.g.
    # ["q1", "q2", "q10"]). Numeric / datetime dtypes get natural order.
    period_rank = _build_period_rank(data, time_col)
    if base_period not in period_rank:
        raise ValueError(
            f"base_period={base_period!r} not found in the time column "
            f"{time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    missing_pre_in_data = [t for t in pre_periods if t not in period_rank]
    if missing_pre_in_data:
        raise ValueError(
            f"pre_periods entries {missing_pre_in_data!r} not found in "
            f"the time column {time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    base_rank = period_rank[base_period]
    out_of_order = [t for t in pre_periods if period_rank[t] >= base_rank]
    if out_of_order:
        raise ValueError(
            f"All pre_periods must be strictly < base_period in "
            f"chronological order. Violators: {out_of_order!r} "
            f"(base_period={base_period!r})."
        )

    # ---- trends_lin: defer the consumed-placebo drop and base-1
    # identification until AFTER the validator block runs (so we can
    # use t_pre_list to enforce the non-terminal-base guard and the
    # observed-period predecessor consistently). On 2-period panels
    # the validator does not run and trends_lin needs F-2, which is
    # impossible — front-door-reject here.
    base_minus_1_period: Any = None
    pre_periods_effective = list(pre_periods)

    # Event-study validation contract (paper Appendix B.2):
    # When the panel has >= 3 distinct periods, always route through
    # `_validate_had_panel_event_study`. This enforces (a) balanced
    # panel, (b) ordered time dtype, (c) D = 0 across every pre-period,
    # (d) last-cohort auto-filter under staggered timing with
    # UserWarning, (e) constant post-treatment dose within unit. When
    # first_treat_col is None and the panel is staggered, the validator
    # RAISES - matching the workflow dispatch contract. For 2-period
    # panels the validator does not apply; skip and fall through to the
    # simpler balance/invariant guards in `_aggregate_for_joint_test`.
    n_periods = int(data[time_col].nunique())
    if trends_lin and n_periods < 3:
        raise ValueError(
            f"joint_pretrends_test(trends_lin=True) requires a panel "
            f"with at least 3 distinct time periods so the per-group "
            f"slope Y[g, base] - Y[g, base - 1] is identified. Got "
            f"n_periods={n_periods}."
        )
    data_filtered: pd.DataFrame = data
    if n_periods >= 3:
        F_val, t_pre_list, _t_post_list, data_filtered, _filter_info = (
            _validate_had_panel_event_study(
                data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
            )
        )
        # `_validate_had_panel_event_study` already emits its own
        # `UserWarning` on the staggered-filter path; the wrapper
        # consumes `_filter_info` silently to avoid duplicated console
        # noise (R4 code-quality fix).
        # Subset invariants: the caller's base_period and pre_periods
        # must be pre-treatment periods under the validator's partition.
        if base_period not in t_pre_list:
            raise ValueError(
                f"base_period={base_period!r} is not in the validated "
                f"pre-period set {list(t_pre_list)!r} (periods before "
                f"first-treatment period F={F_val!r}). For the HAD "
                f"pre-trends workflow, base_period must be a pre-period "
                f"anchor (typically the last pre-period, F-1)."
            )
        not_pre = [t for t in pre_periods if t not in t_pre_list]
        if not_pre:
            raise ValueError(
                f"pre_periods must all be validated pre-treatment "
                f"periods. Not-pre entries: {not_pre!r}. Validator's "
                f"pre-period set: {list(t_pre_list)!r}."
            )
        # PR #392 R3 P1 (non-terminal base guard): paper Eq 17 / Eq 18
        # and R `DIDHAD::did_had(..., trends_lin=TRUE)` anchor the
        # detrending at F-1 (the last validated pre-period) and use
        # Y[F-1] - Y[F-2] as the slope. A direct caller passing
        # base_period < F-1 (e.g. F-2) would compute a different slope
        # at a different anchor, silently changing the methodology
        # away from the documented Eq 17/18 construction. Reject
        # explicitly. Workflow + HAD.fit always pass F-1; this check
        # only fires on direct user calls with non-terminal bases.
        if trends_lin and base_period != t_pre_list[-1]:
            raise ValueError(
                f"joint_pretrends_test(trends_lin=True) requires "
                f"base_period to equal the last validated pre-period "
                f"({t_pre_list[-1]!r}, the canonical Eq 17 anchor "
                f"F-1). Got base_period={base_period!r}. Anchoring at "
                f"any other pre-period would compute a different "
                f"slope and detrending that does not match paper "
                f"Eq 17 / Eq 18 or R DIDHAD::did_had(trends_lin=TRUE)."
            )
        # PR #392 R3 P1 (observed-period base-1 lookup) + R1 P0
        # (consumed-placebo drop) consolidated:
        # base_minus_1_period = t_pre_list[-2] (= F-2, the validated
        # observed pre-period immediately before F-1). Using
        # t_pre_list ensures correctness on ordered-categorical panels
        # with unused intermediate levels (the validator's t_pre_list
        # is built from observed contiguous pre-periods, not from the
        # full dtype's category list). Then drop t_pre_list[-2] from
        # pre_periods if present (the consumed placebo whose detrended
        # residual is mechanically zero).
        if trends_lin:
            if len(t_pre_list) < 2:
                raise ValueError(
                    f"joint_pretrends_test(trends_lin=True) requires "
                    f"at least 2 validated pre-periods so the per-"
                    f"group slope Y[g, F-1] - Y[g, F-2] is identified. "
                    f"Got t_pre_list={list(t_pre_list)!r}."
                )
            base_minus_1_period = t_pre_list[-2]
            if base_minus_1_period in pre_periods_effective:
                warnings.warn(
                    f"joint_pretrends_test(trends_lin=True): dropping "
                    f"period {base_minus_1_period!r} from pre_periods "
                    f"— it is the 'consumed' placebo (the F-2 → F-1 "
                    f"evolution used by the per-group slope "
                    f"estimator), so under trends_lin its detrended "
                    f"residual is mechanically zero. R's "
                    f"`did_had(trends_lin=TRUE)` reduces max placebo "
                    f"lag by 1 with the same effect.",
                    UserWarning,
                    stacklevel=2,
                )
                pre_periods_effective = [
                    t for t in pre_periods_effective if t != base_minus_1_period
                ]
            if len(pre_periods_effective) == 0:
                raise ValueError(
                    f"joint_pretrends_test(trends_lin=True): no testable "
                    f"placebo horizons remain after dropping the consumed "
                    f"placebo at base_period - 1 = {base_minus_1_period!r}. "
                    f"Pass at least one earlier observed pre-period when "
                    f"using trends_lin=True."
                )

    d_arr, dy_by_horizon, _ = _aggregate_for_joint_test(
        data_filtered,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        horizons=list(pre_periods_effective),
        base_period=base_period,
    )
    G = d_arr.shape[0]

    # HAD invariant: D_{g,t} = 0 for every g and every pre_period (and
    # for base_period - it is itself a pre-period relative to the
    # treatment onset). We check this on the passed-in panel subset.
    # Use pre_periods_effective so the consumed placebo (dropped above
    # under trends_lin) is not in the dose-zero check window.
    needed_all_zero = list(pre_periods_effective) + [base_period]
    subset_zero_check = data_filtered[data_filtered[time_col].isin(needed_all_zero)]
    if (subset_zero_check[dose_col] != 0).any():
        n_nonzero = int((subset_zero_check[dose_col] != 0).sum())
        raise ValueError(
            f"Pre-trends test requires D = 0 in every pre-period "
            f"(including base_period). Found {n_nonzero} non-zero "
            f"dose observation(s) across periods "
            f"{needed_all_zero!r}. HAD contract (paper Section 2) and "
            f"pre-trends test design both require the zero-dose "
            f"invariant to hold in ALL periods used as placebo or "
            f"anchor."
        )

    # ---- Apply trends_lin detrending (paper Eq 17 / Eq 18).
    # base_minus_1_period was computed and validated above (before the
    # consumed-placebo drop). Compute the per-group slope and apply the
    # `(t - base) × slope` adjustment to each remaining horizon.
    if trends_lin:
        # Extract Y[g, base] and Y[g, base-1] in unit_col-sorted order
        # (matching d_arr / dy_by_horizon ordering produced by
        # _aggregate_for_joint_test's wide-pivot sort by index).
        slope_subset = data_filtered[
            data_filtered[time_col].isin([base_period, base_minus_1_period])
        ]
        wide_y = slope_subset.pivot(index=unit_col, columns=time_col, values=outcome_col)
        wide_y = wide_y.sort_index()
        if wide_y[base_period].isna().any() or wide_y[base_minus_1_period].isna().any():
            raise ValueError(
                f"joint_pretrends_test(trends_lin=True): NaN value(s) "
                f"in outcome at base_period={base_period!r} or "
                f"base_period-1={base_minus_1_period!r}. The slope "
                f"estimator requires complete observations at both "
                f"periods for every unit."
            )
        slope = wide_y[base_period].to_numpy(dtype=np.float64) - wide_y[
            base_minus_1_period
        ].to_numpy(dtype=np.float64)
        # PR #392 R4 P0: build the detrending rank from OBSERVED
        # periods (on data_filtered), not from the full categorical
        # dtype. Otherwise unused intermediate categorical levels
        # silently change the (t - base) multiplier and corrupt the
        # joint statistic. Mirrors HAD.fit's
        # `_aggregate_multi_period_first_differences` convention which
        # uses `sorted(t_pre_list + t_post_list, ...)` for the
        # event-time rank.
        observed_rank = {
            p: i
            for i, p in enumerate(
                sorted(
                    set(data_filtered[time_col].unique()),
                    key=lambda p: period_rank[p],
                )
            )
        }
        base_rank_observed = observed_rank[base_period]
        # Apply detrending in place to remaining dy_by_horizon entries.
        for t in pre_periods_effective:
            label = str(t)
            delta = observed_rank[t] - base_rank_observed  # < 0 for pre-periods
            dy_by_horizon[label] = dy_by_horizon[label] - delta * slope

    weights_unit, resolved_unit = _resolve_pretest_unit_weights(
        data_filtered, unit_col, None, survey, "joint_pretrends_test"
    )
    # Reorder per-unit weights to match d_arr/dy_by_horizon ordering.
    # _aggregate_for_joint_test sorts the wide pivot by index (unit_col),
    # so per-unit order is the SAME as _aggregate_unit_weights' output.
    w_eff = resolved_unit.weights if resolved_unit is not None else weights_unit

    residuals_by_horizon: Dict[str, np.ndarray] = {}
    fitted_by_horizon: Dict[str, np.ndarray] = {}
    for label, dy_t in dy_by_horizon.items():
        if w_eff is None:
            mean_t = float(dy_t.mean())
        else:
            mean_t = float(np.sum(w_eff * dy_t) / np.sum(w_eff))
        fitted_t = np.full(G, mean_t, dtype=np.float64)
        residuals_t = dy_t - fitted_t
        residuals_by_horizon[label] = residuals_t
        fitted_by_horizon[label] = fitted_t

    design_matrix = np.ones((G, 1), dtype=np.float64)

    # Internal forwarding: pass survey_design= directly to stute_joint_pretest
    # (the canonical kwarg is the same on both ends).
    if resolved_unit is not None:
        joint_survey_design = resolved_unit
    elif weights_unit is not None:
        joint_survey_design = make_pweight_design(weights_unit)
    else:
        joint_survey_design = None
    return stute_joint_pretest(
        residuals_by_horizon=residuals_by_horizon,
        fitted_by_horizon=fitted_by_horizon,
        doses=d_arr,
        design_matrix=design_matrix,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        null_form="mean_independence",
        survey_design=joint_survey_design,
    )


def joint_homogeneity_test(
    data: pd.DataFrame,
    outcome: Any = NOT_SUPPLIED,
    dose: Any = NOT_SUPPLIED,
    time: Any = NOT_SUPPLIED,
    unit: Any = NOT_SUPPLIED,
    post_periods: Any = NOT_SUPPLIED,
    base_period: Any = NOT_SUPPLIED,
    first_treat: Any = NOT_SUPPLIED,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    survey_design: Any = None,
    trends_lin: bool = False,
    outcome_col: Any = NOT_SUPPLIED,
    dose_col: Any = NOT_SUPPLIED,
    time_col: Any = NOT_SUPPLIED,
    unit_col: Any = NOT_SUPPLIED,
    first_treat_col: Any = NOT_SUPPLIED,
) -> StuteJointResult:
    """Joint Stute homogeneity-linearity test (paper Section 4.3 joint).

    Data-in wrapper around :func:`stute_joint_pretest` for the
    linearity null
    ``E[Y_{g,t} - Y_{g,base} | D_{g,t}] = beta_{0,t} + beta_{fe,t} * D_{g,t}``
    across multiple post-period horizons. For each ``t in post_periods``,
    residuals are from an OLS regression of ``Y_{g,t} - Y_{g,base}`` on
    ``[1, D_g]``; the joint CvM tests whether the conditional mean is
    nonlinear in ``D`` in any horizon.

    Used by :func:`did_had_pretest_workflow` with
    ``aggregate="event_study"`` as the step-3 test (no joint Yatchew
    variant exists - the paper does not derive one; users who need
    Yatchew-style adjacent-difference robustness can call
    :func:`yatchew_hr_test` on each (base, post) pair manually).

    Parameters
    ----------
    data : pd.DataFrame
    outcome, dose, time, unit : str
    post_periods : list
        Non-empty list of post-period labels (all strictly ``>
        base_period`` by chronological order; each with ``D > 0`` for
        some unit, i.e. at least one treated unit per horizon).
    base_period : period label
        The reference period (last pre-period in the event-study
        convention). Must not be in ``post_periods``.
    first_treat : str or None
        Forwarded to the underlying panel validator.
    alpha, n_bootstrap, seed : as in :func:`stute_test`.
    survey_design : SurveyDesign or None, keyword-only, default None
        Survey design (Phase 4.5 C). Same contract as
        :func:`joint_pretrends_test`.
    trends_lin : bool, default False, keyword-only
        When ``True``, applies paper page-32 linear-trend detrending:
        per-group slope estimated as ``Y[g, base] - Y[g, base - 1]``
        and applied to each post-period horizon's outcome evolution as
        ``(t - base) × slope`` (forward extrapolation into post). Same
        slope estimator as :func:`joint_pretrends_test`. Mirrors R
        ``DIDHAD::did_had(..., trends_lin=TRUE)`` on its joint
        homogeneity surface (paper Section 4.3, Pierce-Schott p=0.40
        anchor). **Requires** ``base_period`` to equal the last
        validated pre-period (``t_pre_list[-1]``, the canonical
        ``F-1`` anchor) AND ``F-2`` to be present in the panel so
        the slope is identified. Direct callers passing a non-
        terminal base get a ``ValueError`` — Eq 17 / R both anchor
        at ``F-1``. Mutually exclusive with survey weighting; raises
        ``NotImplementedError`` if combined. Default ``False``
        preserves bit-exact backcompat.

    Returns
    -------
    StuteJointResult with ``null_form = "linearity"``.
    """
    _q = "joint_homogeneity_test"
    outcome = resolve_renamed_kwarg(
        _q, "outcome_col", outcome_col, "outcome", outcome, default=NOT_SUPPLIED
    )
    require_arg(_q, "outcome", outcome)
    dose = resolve_renamed_kwarg(_q, "dose_col", dose_col, "dose", dose, default=NOT_SUPPLIED)
    require_arg(_q, "dose", dose)
    time = resolve_renamed_kwarg(_q, "time_col", time_col, "time", time, default=NOT_SUPPLIED)
    require_arg(_q, "time", time)
    unit = resolve_renamed_kwarg(_q, "unit_col", unit_col, "unit", unit, default=NOT_SUPPLIED)
    require_arg(_q, "unit", unit)
    require_arg(_q, "post_periods", post_periods)
    require_arg(_q, "base_period", base_period)
    first_treat = resolve_renamed_kwarg(
        _q, "first_treat_col", first_treat_col, "first_treat", first_treat, default=None
    )
    # Body-local names; the public parameters are the bare names.
    outcome_col = outcome
    dose_col = dose
    time_col = time
    unit_col = unit
    first_treat_col = first_treat
    # Internal variable naming: downstream code reads `survey` (Phase 4.5 C
    # convention).
    survey = survey_design

    # ---- trends_lin × survey_design gate (PR #389 / Phase 4 R-parity).
    # Twin of joint_pretrends_test guard. ----
    if trends_lin and survey is not None:
        raise NotImplementedError(
            "joint_homogeneity_test(trends_lin=True) is not yet "
            "supported with survey weighting (`survey_design=`). "
            "The per-group slope estimator's "
            "weighted variant is not derived from the paper. Use "
            "trends_lin=True WITHOUT survey weights, or use survey "
            "weights WITHOUT trends_lin. Tracked in DEFERRED.md as a "
            "follow-up if user demand emerges."
        )

    if len(post_periods) == 0:
        raise ValueError(
            "post_periods must be non-empty. Workflow dispatch handles "
            "the empty case upstream; direct callers should not pass "
            "an empty list."
        )
    if base_period in post_periods:
        raise ValueError(
            f"base_period={base_period!r} must not appear in "
            f"post_periods {list(post_periods)!r}."
        )

    # Ordering: all post_periods strictly > base_period in
    # chronological order. Uses `_build_period_rank` for ordered-
    # categorical correctness (raw Python `>` would misorder e.g.
    # "q10" > "q2").
    period_rank = _build_period_rank(data, time_col)
    if base_period not in period_rank:
        raise ValueError(
            f"base_period={base_period!r} not found in the time column "
            f"{time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    missing_post_in_data = [t for t in post_periods if t not in period_rank]
    if missing_post_in_data:
        raise ValueError(
            f"post_periods entries {missing_post_in_data!r} not found in "
            f"the time column {time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    base_rank = period_rank[base_period]
    out_of_order = [t for t in post_periods if period_rank[t] <= base_rank]
    if out_of_order:
        raise ValueError(
            f"All post_periods must be strictly > base_period in "
            f"chronological order. Violators: {out_of_order!r} "
            f"(base_period={base_period!r})."
        )

    # Event-study validation contract (paper Appendix B.2) - twin of
    # `joint_pretrends_test`. Same gating by `n_periods >= 3`; same
    # subset-invariant checks; emits the staggered-filter UserWarning.
    # The validator also enforces constant post-treatment dose within
    # unit, which is critical for the homogeneity path because a
    # time-varying post-dose would make the per-horizon refit on
    # `[1, D_g]` misspecify the regressor.
    n_periods = int(data[time_col].nunique())
    if trends_lin and n_periods < 3:
        raise ValueError(
            f"joint_homogeneity_test(trends_lin=True) requires a "
            f"panel with at least 3 distinct time periods so the "
            f"per-group slope Y[g, base] - Y[g, base - 1] is "
            f"identified. Got n_periods={n_periods}."
        )
    base_minus_1_period_validated: Any = None  # set inside validator block under trends_lin
    data_filtered: pd.DataFrame = data
    if n_periods >= 3:
        F_val, t_pre_list, t_post_list, data_filtered, _filter_info = (
            _validate_had_panel_event_study(
                data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
            )
        )
        # `_validate_had_panel_event_study` already emits its own
        # `UserWarning` on the staggered-filter path; the wrapper
        # consumes `_filter_info` silently to avoid duplicated console
        # noise (R4 code-quality fix).
        if base_period not in t_pre_list:
            raise ValueError(
                f"base_period={base_period!r} is not in the validated "
                f"pre-period set {list(t_pre_list)!r} (periods before "
                f"first-treatment period F={F_val!r}). For the HAD "
                f"homogeneity workflow, base_period must be a pre-period "
                f"anchor (typically the last pre-period, F-1)."
            )
        not_post = [t for t in post_periods if t not in t_post_list]
        if not_post:
            raise ValueError(
                f"post_periods must all be validated post-treatment "
                f"periods. Not-post entries: {not_post!r}. Validator's "
                f"post-period set: {list(t_post_list)!r}."
            )
        # PR #392 R3 P1 (non-terminal base guard + observed-period
        # base-1 lookup, twin of joint_pretrends_test). Eq 17 anchors
        # at F-1 and uses Y[F-1] - Y[F-2] as slope; require base ==
        # t_pre_list[-1] AND derive base-1 from t_pre_list[-2].
        if trends_lin and base_period != t_pre_list[-1]:
            raise ValueError(
                f"joint_homogeneity_test(trends_lin=True) requires "
                f"base_period to equal the last validated pre-period "
                f"({t_pre_list[-1]!r}, the canonical Eq 17 anchor "
                f"F-1). Got base_period={base_period!r}. Anchoring at "
                f"any other pre-period would compute a different "
                f"slope and detrending that does not match paper "
                f"Eq 17 / page 32 or R DIDHAD::did_had(trends_lin=TRUE)."
            )
        if trends_lin and len(t_pre_list) < 2:
            raise ValueError(
                f"joint_homogeneity_test(trends_lin=True) requires "
                f"at least 2 validated pre-periods so the per-group "
                f"slope Y[g, F-1] - Y[g, F-2] is identified. Got "
                f"t_pre_list={list(t_pre_list)!r}."
            )
        # Capture the validator's predecessor for downstream use.
        if trends_lin:
            base_minus_1_period_validated = t_pre_list[-2]

    d_arr, dy_by_horizon, _ = _aggregate_for_joint_test(
        data_filtered,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        horizons=list(post_periods),
        base_period=base_period,
    )
    G = d_arr.shape[0]

    # HAD invariant for the homogeneity path: base_period has D = 0
    # (last pre-period contract); each post_period has D > 0 for SOME
    # unit (existence) and is NOT identically zero across all units
    # (reciprocal twin of the pretrends guard - an all-zero post-period
    # contradicts the HAD treatment-onset contract).
    base_doses = data_filtered.loc[data_filtered[time_col] == base_period, dose_col]
    if (base_doses != 0).any():
        n_nonzero = int((base_doses != 0).sum())
        raise ValueError(
            f"base_period={base_period!r} must have D = 0 across every "
            f"unit (HAD last-pre-period invariant). Found {n_nonzero} "
            f"non-zero dose observation(s) in base_period."
        )
    for t in post_periods:
        post_doses = data_filtered.loc[data_filtered[time_col] == t, dose_col]
        if not (post_doses > 0).any():
            raise ValueError(
                f"post_period={t!r} has D = 0 for every unit. HAD "
                f"contract requires at least some unit to have D > 0 "
                f"in each post-period (reciprocal of the pre-period "
                f"zero-dose invariant)."
            )

    # ---- Apply trends_lin detrending (paper Eq 17 / page 32 joint-Stute
    # post-period homogeneity null with industry-specific linear trends).
    # Twin of joint_pretrends_test detrending: per-group slope from
    # Y[g, base] - Y[g, base-1], applied to each post-period horizon's
    # dy_t. The post-period delta = t_rank - base_rank > 0, so the
    # subtraction extrapolates the linear trend FORWARD into post-periods.
    if trends_lin:
        # PR #392 R3 P1: use the validator's t_pre_list[-2] as the
        # predecessor (captured above as base_minus_1_period_validated).
        # This is robust to ordered-categorical panels with unused
        # intermediate levels because the validator builds t_pre_list
        # from observed contiguous pre-periods, not the full dtype
        # category list.
        base_minus_1_period_h = base_minus_1_period_validated
        slope_subset_h = data_filtered[
            data_filtered[time_col].isin([base_period, base_minus_1_period_h])
        ]
        wide_y_h = slope_subset_h.pivot(index=unit_col, columns=time_col, values=outcome_col)
        wide_y_h = wide_y_h.sort_index()
        if wide_y_h[base_period].isna().any() or wide_y_h[base_minus_1_period_h].isna().any():
            raise ValueError(
                f"joint_homogeneity_test(trends_lin=True): NaN value(s) "
                f"in outcome at base_period={base_period!r} or "
                f"base_period-1={base_minus_1_period_h!r}. The slope "
                f"estimator requires complete observations at both "
                f"periods for every unit."
            )
        slope_h = wide_y_h[base_period].to_numpy(dtype=np.float64) - wide_y_h[
            base_minus_1_period_h
        ].to_numpy(dtype=np.float64)
        # PR #392 R4 P0: build the detrending rank from OBSERVED
        # periods on data_filtered (matching HAD.fit). Twin of
        # joint_pretrends_test fix.
        observed_rank_h = {
            p: i
            for i, p in enumerate(
                sorted(
                    set(data_filtered[time_col].unique()),
                    key=lambda p: period_rank[p],
                )
            )
        }
        base_rank_observed_h = observed_rank_h[base_period]
        for t in post_periods:
            label = str(t)
            delta = observed_rank_h[t] - base_rank_observed_h  # > 0 for post-periods
            dy_by_horizon[label] = dy_by_horizon[label] - delta * slope_h

    # Phase 4.5 C: aggregate survey design to per-unit; thread through.
    weights_unit, resolved_unit = _resolve_pretest_unit_weights(
        data_filtered, unit_col, None, survey, "joint_homogeneity_test"
    )
    w_eff = resolved_unit.weights if resolved_unit is not None else weights_unit

    residuals_by_horizon: Dict[str, np.ndarray] = {}
    fitted_by_horizon: Dict[str, np.ndarray] = {}
    for label, dy_t in dy_by_horizon.items():
        if w_eff is None:
            a_hat, b_hat, residuals_t = _fit_ols_intercept_slope(d_arr, dy_t)
        else:
            a_hat, b_hat, residuals_t = _fit_weighted_ols_intercept_slope(d_arr, dy_t, w_eff)
        fitted_t = a_hat + b_hat * d_arr
        residuals_by_horizon[label] = residuals_t
        fitted_by_horizon[label] = fitted_t

    design_matrix = np.column_stack([np.ones(G, dtype=np.float64), d_arr.astype(np.float64)])

    # Internal forwarding via canonical kwarg (avoids deprecation warning).
    if resolved_unit is not None:
        joint_survey_design = resolved_unit
    elif weights_unit is not None:
        joint_survey_design = make_pweight_design(weights_unit)
    else:
        joint_survey_design = None
    return stute_joint_pretest(
        residuals_by_horizon=residuals_by_horizon,
        fitted_by_horizon=fitted_by_horizon,
        doses=d_arr,
        design_matrix=design_matrix,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        null_form="linearity",
        survey_design=joint_survey_design,
    )


_VALID_AGGREGATES = ("overall", "event_study")

_QUG_DEFERRED_SUFFIX = (
    " (linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0)"
)


def _compose_verdict_overall_survey(
    stute: Optional[StuteTestResults],
    yatchew: Optional[YatchewTestResults],
) -> str:
    """Build the overall-path :class:`HADPretestReport` verdict on the
    survey_design= branch (Phase 4.5 C).

    Drops the QUG step from consideration (skipped per Phase 4.5 C0)
    and composes the verdict from Stute + Yatchew alone, with the
    linearity-conditional suffix appended in every branch. R7 P1 fix:
    explicit survey-aware composer replaces the prior approach of
    composing the unweighted verdict with a NaN QUG and string-replacing
    the resulting "QUG NaN" suffix, which could leave pass cases starting
    with "inconclusive".

    Priority (mirrors :func:`_compose_verdict` minus QUG):
      1. Conclusive rejections of Stute or Yatchew lead.
      2. No conclusive rejection but linearity inconclusive (both NaN)
         -> "inconclusive - both linearity tests NaN".
      3. Linearity conclusive (at least one of Stute/Yatchew finite) AND
         no rejection -> fail-to-reject string.
    All branches end with `_QUG_DEFERRED_SUFFIX`.
    """
    if stute is not None and bool(np.isfinite(stute.p_value)):
        stute_ok, stute_rej = True, bool(stute.reject)
    else:
        stute_ok, stute_rej = False, False
    if yatchew is not None and bool(np.isfinite(yatchew.p_value)):
        yatchew_ok, yatchew_rej = True, bool(yatchew.reject)
    else:
        yatchew_ok, yatchew_rej = False, False

    reasons = []
    if stute_rej:
        reasons.append("linearity rejected - heterogeneity bias (Stute)")
    if yatchew_rej:
        reasons.append("linearity rejected - heterogeneity bias (Yatchew)")

    unresolved = []
    if not stute_ok:
        unresolved.append("Stute NaN")
    if not yatchew_ok:
        unresolved.append("Yatchew NaN")

    if reasons:
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict + _QUG_DEFERRED_SUFFIX

    # No rejections.
    if not (stute_ok or yatchew_ok):
        return "inconclusive - both Stute and Yatchew linearity tests NaN" + _QUG_DEFERRED_SUFFIX

    # At least one linearity test conclusive AND no rejection.
    skipped_note = ""
    if not stute_ok:
        skipped_note = " (Stute NaN - skipped)"
    elif not yatchew_ok:
        skipped_note = " (Yatchew NaN - skipped)"
    return (
        "Stute and Yatchew linearity diagnostics fail-to-reject"
        + skipped_note
        + _QUG_DEFERRED_SUFFIX
    )


def _compose_verdict_event_study_survey(
    pretrends_joint: Optional[StuteJointResult],
    homogeneity_joint: Optional[StuteJointResult],
) -> str:
    """Event-study survey-path verdict (R7 P1 fix; mirrors
    :func:`_compose_verdict_event_study` minus QUG)."""
    pretrends_ok = pretrends_joint is not None and bool(np.isfinite(pretrends_joint.p_value))
    homogeneity_ok = homogeneity_joint is not None and bool(np.isfinite(homogeneity_joint.p_value))
    pretrends_rej = pretrends_joint is not None and pretrends_ok and bool(pretrends_joint.reject)
    homogeneity_rej = (
        homogeneity_joint is not None and homogeneity_ok and bool(homogeneity_joint.reject)
    )

    reasons = []
    if pretrends_rej:
        reasons.append("joint pre-trends rejected - assumption 7 violated (joint Stute)")
    if homogeneity_rej:
        reasons.append("joint linearity rejected - heterogeneity bias (joint Stute)")

    unresolved = []
    if pretrends_joint is None:
        unresolved.append("joint pre-trends skipped (no earlier pre-period)")
    elif not pretrends_ok:
        unresolved.append("joint pre-trends NaN")
    if homogeneity_joint is None:
        unresolved.append("joint linearity skipped")
    elif not homogeneity_ok:
        unresolved.append("joint linearity NaN")

    if reasons:
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict + _QUG_DEFERRED_SUFFIX

    if unresolved:
        return "inconclusive - " + "; ".join(unresolved) + _QUG_DEFERRED_SUFFIX

    return "joint pre-trends and joint linearity diagnostics fail-to-reject" + _QUG_DEFERRED_SUFFIX


def did_had_pretest_workflow(
    data: pd.DataFrame,
    outcome: Any = NOT_SUPPLIED,
    dose: Any = NOT_SUPPLIED,
    time: Any = NOT_SUPPLIED,
    unit: Any = NOT_SUPPLIED,
    first_treat: Any = NOT_SUPPLIED,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    *,
    aggregate: str = "overall",
    survey_design: Any = None,
    trends_lin: bool = False,
    outcome_col: Any = NOT_SUPPLIED,
    dose_col: Any = NOT_SUPPLIED,
    time_col: Any = NOT_SUPPLIED,
    unit_col: Any = NOT_SUPPLIED,
    first_treat_col: Any = NOT_SUPPLIED,
) -> HADPretestReport:
    """Run the HAD pre-test workflow (paper Section 4.2-4.3).

    Two dispatch modes via ``aggregate``:

    ``aggregate="overall"`` (default, two-period panel): runs paper
    steps 1 (:func:`qug_test`) and 3 (:func:`stute_test` +
    :func:`yatchew_hr_test`). Step 2 (Assumption 7 pre-trends) is NOT
    implemented on this path because a single-pre-period panel cannot
    support the joint Stute variant; the returned verdict flags the
    Assumption 7 gap explicitly so callers do not receive a misleading
    "TWFE safe" signal. For multi-period panels, pass
    ``aggregate="event_study"`` to close the step-2 gap.

    ``aggregate="event_study"`` (multi-period panel, >= 3 periods): runs
    QUG + joint pre-trends Stute + joint homogeneity-linearity Stute,
    covering paper Section 4 steps 1-3 together. The step-3 Yatchew-HR
    alternative (a single-horizon swap-in for Stute) is subsumed by joint
    Stute on this path - the paper does not derive a joint Yatchew
    variant, so users who need Yatchew robustness under multi-period
    data should call :func:`yatchew_hr_test` on each ``(base, post)``
    pair manually. (Paper step 4 is the decision itself - "use TWFE if
    none of the tests rejects" - not a separate test, so it has no code
    path here. Mirrors the framing in the module-level docstring at
    line 54 and ``_compose_verdict_event_study`` at line 2735.)

    Eq 17 / Eq 18 linear-trend detrending (paper Section 5.2 Pierce-
    Schott application) is now SHIPPED on the event-study path via
    the ``trends_lin`` keyword-only parameter (PR #392 / Phase 4
    R-parity). When ``trends_lin=True``, this workflow forwards the
    flag to both :func:`joint_pretrends_test` and
    :func:`joint_homogeneity_test`; the consumed placebo at
    ``base_period - 1`` is auto-dropped from step 2 and the workflow
    skips step 2 (``pretrends_joint=None``) if no earlier placebo
    survives. Mirrors R ``DIDHAD::did_had(..., trends_lin=TRUE)``.
    Mutually exclusive with ``aggregate="overall"`` (raises
    ``NotImplementedError``).

    Parameters
    ----------
    data : pd.DataFrame
        HAD panel. For ``aggregate="overall"``: balanced two-period
        panel with pre-period dose = 0 for every unit. For
        ``aggregate="event_study"``: balanced multi-period panel with
        >= 3 periods, an ordered time dtype (numeric, datetime, or
        ordered categorical), and the pre-period D=0 invariant across
        all pre-periods.
    outcome, dose, time, unit : str
    first_treat : str or None, default None
        Optional first-treatment-period column. Required on the
        ``aggregate="event_study"`` path when the panel is staggered
        (multi-cohort); the panel validator auto-filters to the last
        cohort and emits ``UserWarning``. The overall path uses this for
        cross-validation only.
    alpha : float, default 0.05
    n_bootstrap : int, default 999
        Replication count for the single-horizon Stute (overall) or
        joint Stute (event_study).
    seed : int or None, default None
        Seed forwarded to the Stute bootstrap. QUG / Yatchew are
        deterministic.
    aggregate : str, keyword-only, default ``"overall"``
        Dispatch mode. Invalid values raise ``ValueError``.
    survey_design : SurveyDesign or None, keyword-only, default None
        Survey design for design-based pretest inference. Linearity-family
        pretests use PSU-level Mammen multiplier bootstrap (Stute family)
        and weighted OLS + weighted variance components (Yatchew). The QUG
        step is skipped under survey with a ``UserWarning`` (permanent
        deferral per Phase 4.5 C0). Replicate-weight designs raise
        ``NotImplementedError``.
    trends_lin : bool, default False, keyword-only
        Forwards into :func:`joint_pretrends_test` and
        :func:`joint_homogeneity_test` on the event-study dispatch
        path. Mirrors R ``DIDHAD::did_had(..., trends_lin=TRUE)``.
        Requires ``aggregate="event_study"``; raises
        ``NotImplementedError`` on ``aggregate="overall"`` (the
        overall path's qug + stute + yatchew block has no
        joint-pretest surface). Mutually exclusive with survey
        weighting at the joint-pretest layer; the joint wrappers
        raise ``NotImplementedError`` if combined. **Effective step-2
        rule under trends_lin**: the consumed placebo at
        ``base_period - 1`` is dropped before step 2 is dispatched;
        if no earlier placebo survives the drop (e.g., a minimal
        4-period panel with ``F=3`` where ``base_period=2`` and the
        only earlier placebo at ``t=1`` is the consumed one), step 2
        is skipped (``pretrends_joint=None``) and the workflow
        proceeds to step 3 (homogeneity). Default ``False`` preserves
        bit-exact backcompat.

    Returns
    -------
    HADPretestReport
        On the overall path: ``stute`` and ``yatchew`` populated,
        ``pretrends_joint`` / ``homogeneity_joint`` are ``None``. On the
        event-study path: ``pretrends_joint`` (``None`` if no earlier
        pre-period) and ``homogeneity_joint`` populated, ``stute`` /
        ``yatchew`` are ``None``. ``aggregate`` is recorded on the
        report for serialization dispatch. On the survey_design= path,
        ``qug`` is ``None`` (Phase 4.5 C0 deferral); other components
        populated as on the unweighted path.

    Raises
    ------
    ValueError
        On invalid ``aggregate``; or any downstream front-door failure
        (panel balance, dtype, dose invariant).
    NotImplementedError
        If ``survey.replicate_weights is not None`` (replicate-weight
        pretests deferred to a parallel follow-up after Phase 4.5 C).

    Notes
    -----
    **Scope (what this composite workflow does NOT cover).** The
    component pretests target the Theorem 4 / Design 1' support-infimum
    null (QUG: ``d_lower = 0``, adjacent evidence on the ``d_lower = 0``
    clause of Assumption 4 only — does not validate boundary density,
    conditional-mean smoothness, or variance regularity), Assumption 7
    (joint Stute pre-trends: mean-independence of placebo first-
    differences from dose), and Assumption 8 (Yatchew / joint
    homogeneity: linearity of treatment effects in dose). The workflow
    does NOT and CANNOT test Assumptions 5 and 6
    from de Chaisemartin et al. (2026) Section 3.1.2, which are required
    for sign / point identification of ``WAS_{d_lower}`` on the Design 1
    family (``d_lower > 0``). Assumptions 5/6 are non-testable via
    pre-trends. The composite verdict string does NOT mention
    Assumptions 5 or 6 — it only flags the Assumption 7 step-2 gap on
    the two-period ``aggregate="overall"`` path. The Assumption 5/6
    caveat is surfaced separately by (a) the
    ``HeterogeneousAdoptionDiD.fit()`` fit-time ``UserWarning`` (which
    fires whenever the resolved design is Design 1 family —
    ``continuous_near_d_lower`` or ``mass_point``) and (b) T21 (HAD
    pretest workflow tutorial) tutorial prose.

    Survey/weighted data (Phase 4.5 C): under ``survey_design=``,
    the workflow:

    1. **Skips QUG** with a ``UserWarning`` and sets ``qug=None`` on the
       report. QUG-under-survey is permanently deferred per Phase 4.5 C0;
       extreme-order-statistic tests are not smooth functionals of the
       empirical CDF and have no off-the-shelf survey-aware analog. See
       :func:`qug_test` Notes for the full methodology rationale.
    2. **Runs the linearity family** with the survey-aware mechanism
       (PSU-level Mammen multiplier bootstrap for Stute / joint variants;
       weighted OLS + weighted variance components for Yatchew) routed
       via the existing kernels.
    3. **Verdict** carries a ``"linearity-conditional verdict; QUG-under-
       survey deferred per Phase 4.5 C0"`` suffix to remind callers that
       admissibility is conditional on the linearity family alone.
    4. **`all_pass`** drops the QUG-conclusiveness gate (one less
       precondition). The linearity-conditional rule splits by aggregate:

       - ``aggregate="overall"`` survey: ``True`` iff at least one of
         Stute/Yatchew is conclusive AND no conclusive test rejects
         (paper Section 4 step-3 "Stute OR Yatchew" wording).
       - ``aggregate="event_study"`` survey: ``True`` iff
         ``pretrends_joint`` is non-None and conclusive,
         ``homogeneity_joint`` is conclusive, AND neither rejects.
         Both joint variants must be conclusive on the event-study
         path (same step-2 + step-3 closure as the unweighted
         aggregate, just without the QUG step).

    Sister pretests are unchanged on the workflow path; direct callers
    can also pass ``survey_design=`` to :func:`stute_test`,
    :func:`yatchew_hr_test`, etc. (Phase 4.5 C extends each helper's
    signature). Per-unit constant-within-unit invariant on weights /
    strata / psu / fpc is enforced by the workflow via
    :func:`diff_diff.had._aggregate_unit_weights` /
    :func:`diff_diff.had._aggregate_unit_resolved_survey`.

    References
    ----------
    de Chaisemartin et al. (2026), Section 4.2-4.3, Theorem 4, Appendix
    D, Theorem 7.
    """
    _q = "did_had_pretest_workflow"
    outcome = resolve_renamed_kwarg(
        _q, "outcome_col", outcome_col, "outcome", outcome, default=NOT_SUPPLIED
    )
    require_arg(_q, "outcome", outcome)
    dose = resolve_renamed_kwarg(_q, "dose_col", dose_col, "dose", dose, default=NOT_SUPPLIED)
    require_arg(_q, "dose", dose)
    time = resolve_renamed_kwarg(_q, "time_col", time_col, "time", time, default=NOT_SUPPLIED)
    require_arg(_q, "time", time)
    unit = resolve_renamed_kwarg(_q, "unit_col", unit_col, "unit", unit, default=NOT_SUPPLIED)
    require_arg(_q, "unit", unit)
    first_treat = resolve_renamed_kwarg(
        _q, "first_treat_col", first_treat_col, "first_treat", first_treat, default=None
    )
    # Body-local names; the public parameters are the bare names.
    outcome_col = outcome
    dose_col = dose
    time_col = time
    unit_col = unit
    first_treat_col = first_treat
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(
            f"aggregate must be one of {list(_VALID_AGGREGATES)!r}; " f"got {aggregate!r}."
        )

    # ---- trends_lin scope gate (PR #392 R1 P1).
    # `trends_lin=True` is only meaningful on the event-study path because
    # it forwards into joint_pretrends_test / joint_homogeneity_test. The
    # overall path runs qug + stute + yatchew on a 2-period panel and has
    # no joint-pretest surface to receive the kwarg. Front-door reject
    # rather than silently ignore.
    if trends_lin and aggregate != "event_study":
        raise NotImplementedError(
            "did_had_pretest_workflow(trends_lin=True) requires "
            "aggregate='event_study' (the trends_lin kwarg forwards "
            "into the joint pretests, which only run on the event-"
            "study path). The overall path's qug + stute + yatchew "
            "block has no per-group slope surface; pass a multi-"
            "period panel and aggregate='event_study'."
        )

    # R6 P1 fix: do NOT call _resolve_pretest_unit_weights on the FULL panel
    # here -- under aggregate='event_study' the panel may be staggered and the
    # cohort filter at _validate_multi_period_panel can drop units. If those
    # dropped units have zero/invalid weights, eager full-panel resolution
    # would abort an otherwise-valid event-study run. Defer resolution to the
    # per-aggregate branches: overall path resolves on the original data (no
    # filtering); event-study path lets the joint wrappers handle resolution
    # on data_filtered.
    # Internal variable naming: downstream code reads `survey` (Phase 4.5 C
    # convention). The bit-exact regression invariant is preserved because
    # we only bind names, not values.
    survey = survey_design

    use_survey_path = survey is not None

    if use_survey_path:
        # Phase 4.5 C0 deferral surface: skip QUG with educational warning.
        warnings.warn(
            "did_had_pretest_workflow: QUG step skipped under survey_design= "
            "(permanently deferred per Phase 4.5 C0; extreme-value theory "
            "under complex sampling is not a settled toolkit). Verdict "
            "reflects the linearity family only ('linearity-conditional').",
            UserWarning,
            stacklevel=2,
        )

    if aggregate == "event_study":
        F, t_pre_list, t_post_list, data_filtered, _filter_info = _validate_multi_period_panel(
            data,
            outcome_col=outcome_col,
            dose_col=dose_col,
            time_col=time_col,
            unit_col=unit_col,
            first_treat_col=first_treat_col,
        )
        # `_validate_multi_period_panel` delegates to
        # `_validate_had_panel_event_study`, which already emits its own
        # `UserWarning` on the staggered-filter path; we do NOT warn a
        # second time here (R4 code-quality fix - single emission point).

        # Base period for both joint tests is the last pre-period
        # (paper convention: anchor at F-1 under natural time order).
        # This is t_pre_list[-1] - NOT an arithmetic F-1, since the
        # time column may be non-integer (datetime, ordered categorical).
        base_period = t_pre_list[-1]

        # Step 1: QUG on dose distribution at F. Doses are
        # time-invariant in HAD, so D_g at F equals max_t D_{g,t}.
        # Phase 4.5 C: skipped under survey_design= (qug_res = None).
        doses_at_F = (
            data_filtered.loc[data_filtered[time_col] == F, [unit_col, dose_col]]
            .set_index(unit_col)
            .sort_index()[dose_col]
            .to_numpy(dtype=np.float64)
        )
        qug_res = None if use_survey_path else qug_test(doses_at_F, alpha=alpha)

        # `survey` carries column references resolved internally on
        # data_filtered by the joint wrappers, so no row-level subsetting
        # is needed when forwarding.
        joint_survey = survey if use_survey_path else None

        # Step 2: joint pre-trends on earlier pre-periods (those
        # strictly before base_period). If only the base pre-period is
        # available (len(t_pre_list) == 1), there are no earlier
        # placebos; set pretrends_joint=None and flag in verdict.
        # ``t_pre_list`` is returned chronologically sorted by
        # ``_validate_had_panel_event_study`` (using the column's
        # ordered-categorical category order or the natural numeric /
        # datetime order), so taking everything but the last element
        # gives the earlier pre-periods regardless of dtype. Raw
        # ``t < base_period`` would misorder ordered-categorical labels
        # whose lexical and chronological order disagree (e.g. "q10" <
        # "q2" lexically but > chronologically).
        earlier_pre = list(t_pre_list[:-1])
        # PR #392 R2 P1: under trends_lin=True, the consumed placebo at
        # base_period - 1 (= t_pre_list[-2] in the contiguous validated
        # pre-period list) is dropped by joint_pretrends_test downstream
        # because its detrended residual is mechanically zero. Pre-filter
        # it here so we can preserve the EXISTING "no earlier placebo →
        # pretrends_joint=None, skip step 2" verdict path (rather than
        # propagating joint_pretrends_test's `ValueError("no testable
        # placebo horizons remain")` and aborting the whole workflow).
        # The minimal valid trends_lin event-study panel (4 periods,
        # F=3, base=2, only earlier placebo at 1 = the consumed one)
        # hits this path; the workflow should still run step 3
        # (homogeneity) and emit the standard "step 2 skipped" verdict.
        if trends_lin and len(t_pre_list) >= 2:
            consumed_placebo_period = t_pre_list[-2]
            earlier_pre = [t for t in earlier_pre if t != consumed_placebo_period]
        if len(earlier_pre) >= 1:
            pretrends_joint = joint_pretrends_test(
                data_filtered,
                outcome=outcome_col,
                dose=dose_col,
                time=time_col,
                unit=unit_col,
                pre_periods=earlier_pre,
                base_period=base_period,
                first_treat=first_treat_col,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                seed=seed,
                survey_design=joint_survey,
                trends_lin=trends_lin,
            )
        else:
            pretrends_joint = None

        # Step 3: joint homogeneity-linearity on post-periods.
        homogeneity_joint = joint_homogeneity_test(
            data_filtered,
            outcome=outcome_col,
            dose=dose_col,
            time=time_col,
            unit=unit_col,
            post_periods=list(t_post_list),
            base_period=base_period,
            first_treat=first_treat_col,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
            survey_design=joint_survey,
            trends_lin=trends_lin,
        )

        # Event-study `all_pass`. On the unweighted path, every implemented
        # step must be conclusive AND none reject (Phase 3 convention). On
        # the survey_design= path, drop the QUG-conclusiveness condition
        # (qug=None per Phase 4.5 C0 deferral); admissibility becomes
        # linearity-conditional.
        pretrends_ok = pretrends_joint is not None and bool(np.isfinite(pretrends_joint.p_value))
        homogeneity_ok = bool(np.isfinite(homogeneity_joint.p_value))
        if use_survey_path:
            all_pass = bool(
                pretrends_ok
                and pretrends_joint is not None
                and not pretrends_joint.reject
                and homogeneity_ok
                and not homogeneity_joint.reject
            )
            # R7 P1 fix: explicit survey-aware verdict composer instead
            # of post-processing the unweighted-verdict output (the
            # previous string-replace approach could leave pass cases
            # starting with "inconclusive" even when all_pass=True).
            verdict = _compose_verdict_event_study_survey(pretrends_joint, homogeneity_joint)
        else:
            # use_survey_path=False branch: qug_res was computed above.
            assert qug_res is not None
            qug_ok = bool(np.isfinite(qug_res.p_value))
            all_pass = bool(
                qug_ok
                and pretrends_ok
                and pretrends_joint is not None
                and not pretrends_joint.reject
                and homogeneity_ok
                and not homogeneity_joint.reject
                and not qug_res.reject
            )
            verdict = _compose_verdict_event_study(qug_res, pretrends_joint, homogeneity_joint)

        return HADPretestReport(
            qug=qug_res,
            stute=None,
            yatchew=None,
            all_pass=all_pass,
            verdict=verdict,
            alpha=alpha,
            n_obs=int(doses_at_F.shape[0]),
            pretrends_joint=pretrends_joint,
            homogeneity_joint=homogeneity_joint,
            aggregate="event_study",
        )

    # aggregate == "overall" - Phase 3 behavior on the unweighted path
    # (bit-exact regression preserved); Phase 4.5 C survey path skips QUG
    # and dispatches stute / yatchew with the resolved survey design.
    t_pre, t_post = _validate_had_panel(
        data, outcome_col, dose_col, time_col, unit_col, first_treat_col
    )
    d_arr, dy_arr, _, _ = _aggregate_first_difference(
        data,
        outcome_col,
        dose_col,
        time_col,
        unit_col,
        t_pre,
        t_post,
        cluster_col=None,  # pretests do not use cluster-robust SE
    )

    # R6 P1 fix: resolve weights/survey HERE (overall path operates on
    # the original data; no cohort filter to interact with).
    weights_unit, resolved_unit = _resolve_pretest_unit_weights(
        data, unit_col, None, survey, "did_had_pretest_workflow"
    )

    qug_res = None if use_survey_path else qug_test(d_arr, alpha=alpha)
    # Forward weights/survey to per-test calls. The data-in workflow has
    # already aggregated to per-unit (weights_unit / resolved_unit); the
    # _aggregate_first_difference call above also collapses to per-unit
    # (one row per unit), so weights_unit and resolved_unit are aligned.
    # Internal forwarding uses the canonical survey_design= kwarg (the
    # sole weighting entry on every pretest surface).
    if resolved_unit is not None:
        per_test_survey_design = resolved_unit
    elif weights_unit is not None:
        per_test_survey_design = make_pweight_design(weights_unit)
    else:
        per_test_survey_design = None
    stute_res = stute_test(
        d_arr,
        dy_arr,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        survey_design=per_test_survey_design,
    )
    # Linearity null is correct for the workflow's post-treatment Yatchew step
    # (paper Theorem 7); placebo mean-independence routing lives in the
    # R-parity test, not the workflow.
    yatchew_res = yatchew_hr_test(
        d_arr,
        dy_arr,
        alpha=alpha,
        survey_design=per_test_survey_design,
    )

    # `all_pass` must be conclusive under the paper's four-step workflow
    # (step 1 QUG + step 3 linearity via Stute OR Yatchew):
    #   - QUG must produce a finite p-value (step 1 is required).
    #   - At least ONE of Stute / Yatchew must produce a finite p-value
    #     (step 3 accepts either; the paper's wording is "Stute OR
    #     Yatchew"). This accommodates common QUG-style panels with
    #     repeated d=0 units, where Yatchew's duplicate-dose guard trips
    #     but Stute's tie-safe CvM still produces a conclusive result.
    #   - No conclusive test may reject. NaN-p tests have reject=False by
    #     convention, so the OR across `.reject` naturally counts only
    #     the conclusive rejections.
    # On the survey path, drop QUG conclusiveness (qug=None per C0 deferral).
    linearity_conclusive = bool(np.isfinite(stute_res.p_value) or np.isfinite(yatchew_res.p_value))
    if use_survey_path:
        any_reject = stute_res.reject or yatchew_res.reject
        all_pass = bool(linearity_conclusive and not any_reject)
        # R7 P1 fix: explicit survey-aware verdict composer.
        verdict = _compose_verdict_overall_survey(stute_res, yatchew_res)
    else:
        # use_survey_path=False branch: qug_res was computed above.
        assert qug_res is not None
        qug_conclusive = bool(np.isfinite(qug_res.p_value))
        any_reject = qug_res.reject or stute_res.reject or yatchew_res.reject
        all_pass = bool(qug_conclusive and linearity_conclusive and not any_reject)
        verdict = _compose_verdict(qug_res, stute_res, yatchew_res)

    return HADPretestReport(
        qug=qug_res,
        stute=stute_res,
        yatchew=yatchew_res,
        all_pass=all_pass,
        verdict=verdict,
        alpha=alpha,
        n_obs=int(d_arr.shape[0]),
        aggregate="overall",
    )
