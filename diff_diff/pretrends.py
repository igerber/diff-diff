"""
Pre-trends power analysis for difference-in-differences designs.

This module implements the power analysis framework from Roth (2022) for assessing
the informativeness of pre-trends tests. It answers the question: "If my pre-trends
test passed, what violations would I have been able to detect?"

Key concepts:
- **Minimum Detectable Violation (MDV)**: The smallest pre-trends violation that
  would be detected with given power (e.g., 80%).
- **Power of Pre-Trends Test**: Probability of rejecting parallel trends given
  a specific violation pattern.
- **Relationship to HonestDiD**: If MDV is large relative to your estimated effect,
  a passing pre-trends test provides limited reassurance.

References
----------
Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for
    Parallel Trends. American Economic Review: Insights, 4(3), 305-322.
    https://doi.org/10.1257/aeri.20210236

See Also
--------
https://github.com/jonathandroth/pretrends - R package implementation
diff_diff.honest_did - Sensitivity analysis for parallel trends violations
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from diff_diff.results import MultiPeriodDiDResults
from diff_diff.results_base import Diagnostic


def _compute_nis_acceptance_prob(
    M: float,
    weights: np.ndarray,
    vcov: np.ndarray,
    z_alpha: float,
) -> float:
    """
    Compute the NIS box acceptance probability ``P(β̂_pre ∈ B_NIS(Σ))``.

    Used by both ``PreTrendsPower._compute_power_nis`` and
    ``PreTrendsPowerResults.power_at()`` to avoid code duplication and
    centralize the analytical-or-MC fallback path.

    Returns
    -------
    accept_prob : float
        Acceptance probability in [0, 1]. Always finite — falls back to
        Monte Carlo (N=20000) if the analytical scipy MVN CDF raises OR
        returns a non-finite value (e.g., on numerically degenerate Σ).
    """
    sigma = np.sqrt(np.maximum(np.diag(vcov), 0))
    delta = M * weights
    upper = z_alpha * sigma - delta
    lower = -z_alpha * sigma - delta

    accept_prob: float
    try:
        accept_prob = float(
            stats.multivariate_normal.cdf(
                upper,
                lower_limit=lower,
                mean=np.zeros(len(weights)),
                cov=vcov,
                allow_singular=True,
            )
        )
    except (ValueError, np.linalg.LinAlgError):
        accept_prob = float("nan")

    # MC fallback on non-finite analytical output. The scipy CDF can return
    # nan on numerically degenerate Σ even when no exception is raised
    # (Genz algorithm internal cancellation); detecting nan and falling
    # back to simulation keeps the downstream MDV solver from silently
    # propagating nan and returning a wrong-but-finite MDV.
    if not np.isfinite(accept_prob):
        rng = np.random.default_rng(0)
        samples = rng.multivariate_normal(mean=np.zeros(len(weights)), cov=vcov, size=20000)
        in_box = np.all((samples >= lower[None, :]) & (samples <= upper[None, :]), axis=1)
        accept_prob = float(in_box.mean())

    return float(np.clip(accept_prob, 0.0, 1.0))


def _coerce_relative_times_from_reference(
    estimated_pre_periods: List[Any],
    reference_period: Any,
) -> Optional[np.ndarray]:
    """
    Convert ``estimated_pre_periods`` to Roth-style relative-time offsets
    from a numeric / Period / datetime ``reference_period``.

    Returns ``np.ndarray`` of float relative times when conversion succeeds,
    or ``None`` when the labels are genuinely non-numeric / unordered
    (string period IDs, categoricals, etc.). In the ``None`` case, the
    caller's downstream linear-violation weight construction falls back to
    the legacy count-based normalized direction — the reported MDV is then
    NOT in Roth's γ units. We emit a ``UserWarning`` so the user knows
    the γ-unit contract did not hold and can re-fit with numeric labels.

    Supported regimes:

    - Numeric (``int`` / ``float`` / ``np.int64``): direct ``float()``
      coercion gives the correct relative offset.
    - ``pandas.Period`` / ``pandas.Timestamp`` / ``np.datetime64``: period
      arithmetic returns an offset / ``Timedelta`` that we coerce to a
      float via ``.n`` (for Period frequencies) or ``.days`` (for
      Timedelta-like). The result is in units of the reference's
      frequency for Period, days for Timestamp / datetime64 — the linear
      γ-units scale is per-unit-of-frequency.
    - Anything else (string period IDs, categoricals with no ordering,
      mixed types): returns ``None`` with a warning.
    """
    # Path 1: direct float coercion (numeric scalars).
    try:
        ref_float = float(reference_period)
        return np.asarray(
            [float(p) - ref_float for p in estimated_pre_periods],
            dtype=float,
        )
    except (TypeError, ValueError):
        pass

    # Path 2: pandas.Period / pandas.Timestamp / datetime64 — try
    # subtraction-based offset arithmetic.
    try:
        diffs = [p - reference_period for p in estimated_pre_periods]
        floats: List[float] = []
        for d in diffs:
            # pandas.tseries.offsets.* or pandas.Period offset — has `.n`.
            n_attr = getattr(d, "n", None)
            if n_attr is not None:
                floats.append(float(n_attr))
                continue
            # pandas.Timedelta / numpy.timedelta64 — convert to days.
            days_attr = getattr(d, "days", None)
            if days_attr is not None:
                floats.append(float(days_attr))
                continue
            # Bare numpy.timedelta64 fallback.
            try:
                floats.append(float(d / np.timedelta64(1, "D")))
                continue
            except (TypeError, ValueError):
                raise TypeError(
                    f"cannot coerce difference {d!r} of type {type(d).__name__} "
                    "to float days/periods"
                )
        return np.asarray(floats, dtype=float)
    except (TypeError, ValueError):
        pass

    # Path 3: genuinely non-numeric labels — warn and fall back to legacy.
    warnings.warn(
        f"PreTrendsPower: reference_period {reference_period!r} (type "
        f"{type(reference_period).__name__}) is not numeric or datetime-like, "
        "so per-period relative times cannot be derived. Linear-violation "
        "weights will use the legacy count-based [n_pre-1, ..., 0]/||·||_2 "
        "direction; the reported MDV is NOT in Roth (2022) γ units. Re-fit "
        "with numeric period labels (int year, pandas.Period, datetime) to "
        "obtain γ-unit MDV.",
        UserWarning,
        stacklevel=3,
    )
    return None


def _extract_event_study_vcov_subblock(
    results: Any,
    pre_periods: List[int],
    ses: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    Extract the pre-period sub-block of ``results.event_study_vcov`` when
    available; otherwise fall back to ``diag(ses**2)``.

    This is the canonical Σ_22 routing path for ``compute_pretrends_power``
    when the event-study result type exposes a full event-study covariance
    matrix (CallawaySantAnnaResults non-bootstrap fits at
    ``staggered_results.py:126-128`` and SunAbrahamResults non-bootstrap
    fits via the W-matrix construction added in PR-B Step 3). Bootstrap
    fits and replicate-weight survey fits clear ``event_study_vcov`` so
    the analytical VCV is not mixed with bootstrap / replicate SE
    overrides — those cases naturally fall through to the diag fallback.

    Parameters
    ----------
    results : event-study results object
        Must have ``event_study_vcov`` and ``event_study_vcov_index``
        attributes (CallawaySantAnnaResults and SunAbrahamResults both
        expose them; either may be None for the bootstrap / replicate
        paths).
    pre_periods : list of int
        Sorted relative-time labels of the pre-period coefficients to
        extract.
    ses : np.ndarray
        Per-period standard errors (used for the ``diag(ses**2)`` fallback
        path; must be in the same order as ``pre_periods``).

    Returns
    -------
    vcov : np.ndarray
        The (n_pre, n_pre) covariance sub-block. Full event_study_vcov
        sub-block when available; diag(ses**2) otherwise.
    source : str
        Provenance label for downstream report-layer tier classification:
        ``"full_pre_period_vcov"`` when the full event-study sub-block
        was used (no off-diagonal information was discarded), or
        ``"diag_fallback"`` when ``event_study_vcov`` was missing /
        cleared (bootstrap / replicate-weight CS or SA paths).
    """
    es_vcov = getattr(results, "event_study_vcov", None)
    es_vcov_index = getattr(results, "event_study_vcov_index", None)
    if es_vcov is None or es_vcov_index is None:
        return np.diag(ses**2), "diag_fallback"

    try:
        indices = [list(es_vcov_index).index(t) for t in pre_periods]
    except ValueError as e:
        # event_study_vcov_index out of sync with the filtered pre_periods.
        # This is a defensive guard — should not happen on the canonical
        # construction paths, but if it does we fail loud rather than
        # silently substituting diag.
        raise ValueError(
            f"event_study_vcov_index is missing one of the pre-period labels "
            f"{pre_periods}; cannot extract sub-block. Available index: "
            f"{list(es_vcov_index)}. Original error: {e}"
        ) from e

    return np.asarray(es_vcov)[np.ix_(indices, indices)], "full_pre_period_vcov"


# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class PreTrendsPowerResults(Diagnostic):
    """
    Results from pre-trends power analysis.

    Attributes
    ----------
    power : float
        Power to detect the specified violation pattern at given alpha.
    mdv : float
        Minimum detectable violation (smallest M detectable at target power).
    violation_magnitude : float
        The magnitude of violation tested (M parameter).
    violation_type : str
        Type of violation pattern ('linear', 'constant', 'last_period', 'custom').
    alpha : float
        Significance level for the pre-trends test.
    target_power : float
        Target power level used for MDV calculation.
    n_pre_periods : int
        Number of pre-treatment periods in the event study.
    test_statistic : float
        Expected test statistic under the specified violation (Wald only;
        NaN for NIS fits).
    critical_value : float
        Critical value for the pre-trends test.
    noncentrality : float
        Non-centrality parameter under the alternative hypothesis (Wald only;
        NaN for NIS fits).
    pre_period_effects : np.ndarray
        Estimated pre-period effects from the event study.
    pre_period_ses : np.ndarray
        Standard errors of pre-period effects.
    vcov : np.ndarray
        Variance-covariance matrix of pre-period effects.
    pretest_form : str
        Pretest acceptance-region form used: ``'nis'`` (no-individually-
        significant box probability — Roth 2022 Section II.A-B, default for new
        fits) or ``'wald'`` (noncentral-chi-squared on the quadratic form
        ``delta' Sigma_22^{-1} delta`` — paper-supported alternative, retained
        for backwards compatibility with shipped numerical baselines).
    nis_box_probability : float
        Acceptance probability ``P(beta_hat_pre in B_NIS(Sigma))`` under the
        alternative ``M * weights``. NIS-only; NaN for Wald fits.
    violation_weights : np.ndarray, optional
        The violation-direction vector used at fit time. Populated for all
        violation types on fresh fits. Normalization depends on the type
        so that ``M`` always matches the documented per-pattern contract:

        - ``linear`` threaded with ``relative_times`` (post PR-B Step 4):
          ``|t|`` directly, NOT L2-normalized, so ``δ_t = M·|t|`` and the
          reported MDV equals Roth's γ exactly.
        - ``linear`` without ``relative_times`` (legacy):
          ``[n_pre-1, ..., 0]`` L2-normalized.
        - ``constant`` (post PR-B R13): ``[1, ..., 1]`` directly, NOT
          L2-normalized, so ``δ_t = M`` is a true per-period level shift.
        - ``last_period``: ``[0, ..., 0, 1]`` (already unit-norm).
        - ``custom``: user vector L2-normalized to unit norm.

        Old serialized results may have ``None`` here; ``power_at()``
        falls back to reconstruction in that case (with the PR-A
        ``NotImplementedError`` guard retained only for
        ``violation_type='custom'`` with ``violation_weights=None``).
    """

    power: float
    mdv: float
    violation_magnitude: float
    violation_type: str
    alpha: float
    target_power: float
    n_pre_periods: int
    test_statistic: float
    critical_value: float
    noncentrality: float
    pre_period_effects: np.ndarray = field(repr=False)
    pre_period_ses: np.ndarray = field(repr=False)
    vcov: np.ndarray = field(repr=False)
    original_results: Optional[Any] = field(default=None, repr=False)
    pretest_form: Literal["nis", "wald"] = "wald"
    nis_box_probability: float = np.nan
    violation_weights: Optional[np.ndarray] = field(default=None, repr=False)
    # Provenance for downstream tier classification. Populated at fit time
    # from `_extract_pre_period_params`. ``"full_pre_period_vcov"`` when
    # off-diagonal pre-period covariances were used; ``"diag_fallback"``
    # when only per-period SEs were available; ``"unknown"`` for legacy
    # serialized results pre-PR-B (backwards-compat default). See
    # ``diagnostic_report._infer_cov_source`` for consumer-side use.
    covariance_source: str = "unknown"

    def __repr__(self) -> str:
        return (
            f"PreTrendsPowerResults(power={self.power:.3f}, "
            f"mdv={self.mdv:.4f}, M={self.violation_magnitude:.4f})"
        )

    @property
    def is_informative(self) -> bool:
        """
        Check if the pre-trends test is informative.

        A pre-trends test is considered informative if the MAX level-scale
        pre-period violation under the MDV is reasonably small relative to
        the per-period standard errors. Post PR-B Step 4 the `linear`
        MDV is in Roth's γ units (a slope), so comparing the raw ``mdv``
        scalar to the level-scale ``max(pre_period_ses)`` would mix units
        on irregular pre-period grids. The comparable level-scale scalar
        is ``mdv * max(|violation_weights|)`` (the largest pre-period
        deviation under the MDV — see ``max_abs_pre_violation``).
        """
        max_se = np.max(self.pre_period_ses) if len(self.pre_period_ses) > 0 else 1.0
        return bool(self.max_abs_pre_violation < 2 * max_se)

    @property
    def max_abs_pre_violation(self) -> float:
        """
        Largest level-scale pre-period deviation under the MDV.

        Returns ``mdv * max(|violation_weights|)`` — the maximum
        absolute pre-period violation ``δ_t`` when the violation
        magnitude equals the MDV. This is the right level-scale
        scalar for comparing pre-trends sensitivity against
        coefficient-scale quantities (post-treatment ATT, per-period
        SEs, HonestDiD's M bound).

        Why this matters: PR-B Step 4 made the linear ``mdv`` report
        Roth's γ units (a slope on relative time). On a regular grid
        ``[-3, -2, -1]`` the max deviation is ``γ * 3``; on an
        irregular grid ``[-5, -3, -1]`` it is ``γ * 5``. Raw ``mdv``
        alone cannot be compared to level effects without applying
        the weight scale.

        For non-linear violation types under the PR-B R13 level-shift
        convention: constant weights ``[1, ..., 1]`` (unnormalized)
        yield ``max_abs_pre_violation = mdv * 1 = mdv`` — raw ``mdv``
        IS the per-period level shift, so level- and γ-scales coincide.
        Last_period ``[0, ..., 0, 1]`` yields ``max_abs_pre_violation
        = mdv`` for the same reason. Custom uses the L2-normalized
        user-supplied weight vector, so ``max_abs_pre_violation``
        depends on the user's direction.

        Backwards-compat: legacy serialized results without
        ``violation_weights`` (pre-PR-B) fall back to the raw ``mdv``
        (which under the pre-PR-B count-based L2-normalized linear
        convention already had a roughly level-scale magnitude).
        """
        if self.violation_weights is None or len(self.violation_weights) == 0:
            return float(self.mdv)
        if not np.isfinite(self.mdv):
            return float(self.mdv)
        max_w = float(np.max(np.abs(self.violation_weights)))
        return float(self.mdv * max_w)

    @property
    def power_adequate(self) -> bool:
        """Check if power meets the target threshold."""
        return bool(self.power >= self.target_power)

    def summary(self) -> str:
        """
        Generate formatted summary of pre-trends power analysis.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = [
            "=" * 70,
            "Pre-Trends Power Analysis Results".center(70),
            "(Roth 2022)".center(70),
            "=" * 70,
            "",
            f"{'Number of pre-periods:':<35} {self.n_pre_periods}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            f"{'Target power:':<35} {self.target_power:.1%}",
            f"{'Violation type:':<35} {self.violation_type}",
            f"{'Pretest form:':<35} {self.pretest_form}",
            "",
            "-" * 70,
            "Power Analysis".center(70),
            "-" * 70,
            f"{'Violation magnitude (M):':<35} {self.violation_magnitude:.4f}",
            f"{'Power to detect this violation:':<35} {self.power:.1%}",
            f"{'Minimum detectable violation:':<35} {self.mdv:.4f}",
            "",
            f"{'Critical value:':<35} {self.critical_value:.4f}",
        ]
        # Dispatch on pretest_form: NIS reports the MVN box acceptance
        # probability, Wald reports the noncentral-chi-squared noncentrality.
        if self.pretest_form == "nis":
            lines.append(f"{'NIS box probability (accept):':<35} {self.nis_box_probability:.4f}")
        else:
            lines.append(f"{'Test statistic (expected):':<35} {self.test_statistic:.4f}")
            lines.append(f"{'Non-centrality parameter:':<35} {self.noncentrality:.4f}")
        lines.extend(
            [
                "",
                "-" * 70,
                "Interpretation".center(70),
                "-" * 70,
            ]
        )

        if self.power_adequate:
            lines.append(f"✓ Power ({self.power:.0%}) meets target ({self.target_power:.0%}).")
            lines.append(
                f"  The pre-trends test would detect violations of magnitude {self.violation_magnitude:.3f}."
            )
        else:
            lines.append(f"✗ Power ({self.power:.0%}) below target ({self.target_power:.0%}).")
            lines.append(
                f"  Would need violations of {self.mdv:.3f} to achieve {self.target_power:.0%} power."
            )

        lines.append("")
        lines.append(f"Minimum detectable violation (MDV): {self.mdv:.4f}")
        lines.append("  → Passing pre-trends test does NOT rule out violations up to this size.")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to JSON-serializable dictionary.

        Includes the post-PR-B provenance fields (``violation_weights``,
        ``covariance_source``) so callers that round-trip the result
        through ``to_dict``/``to_dataframe`` (e.g., for serialization
        or downstream transport) preserve the same information the
        reporting layer reads off the dataclass directly.

        ``violation_weights`` is emitted as ``list[float]`` (or ``None``)
        so ``json.dumps(result.to_dict())`` works out of the box. Use
        ``self.violation_weights`` directly on the dataclass when an
        ndarray is needed.
        """
        weights = self.violation_weights
        weights_list: Optional[List[float]]
        if weights is None:
            weights_list = None
        else:
            weights_list = [float(w) for w in np.asarray(weights).ravel()]
        return {
            "power": self.power,
            "mdv": self.mdv,
            "violation_magnitude": self.violation_magnitude,
            "violation_type": self.violation_type,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "n_pre_periods": self.n_pre_periods,
            "test_statistic": self.test_statistic,
            "critical_value": self.critical_value,
            "noncentrality": self.noncentrality,
            "pretest_form": self.pretest_form,
            "nis_box_probability": self.nis_box_probability,
            "violation_weights": weights_list,
            "covariance_source": self.covariance_source,
            "is_informative": self.is_informative,
            "power_adequate": self.power_adequate,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame.

        ``violation_weights`` is stored as a Python list in the single
        row (pandas-friendly); ``covariance_source`` is a plain string.
        Mirrors ``to_dict``.
        """
        return pd.DataFrame([self.to_dict()])

    def power_at(self, M: float) -> float:
        """
        Compute power to detect a specific violation magnitude.

        Uses the stored fitted ``violation_weights`` and the stored
        ``pretest_form`` to dispatch to the NIS or Wald power computation
        without re-fitting.

        Parameters
        ----------
        M : float
            Violation magnitude to evaluate.

        Returns
        -------
        float
            Power to detect violation of magnitude M.

        Raises
        ------
        NotImplementedError
            If the result was produced by an older library version (before
            the ``violation_weights`` field was added to ``PreTrendsPowerResults``)
            AND ``violation_type='custom'``. The reconstruction fallback can
            handle ``linear``/``constant``/``last_period`` from stored
            metadata, but custom weights cannot be reconstructed; refit
            ``PreTrendsPower(violation_type='custom', violation_weights=...)``
            with the new ``M`` instead.
        """
        from scipy import stats

        n_pre = self.n_pre_periods

        # Prefer the persisted fitted weights (populated for all violation
        # types on fresh fits after PR-B). Fall back to reconstruction only
        # for old serialized results lacking the field.
        if self.violation_weights is not None:
            weights = np.asarray(self.violation_weights, dtype=float)
        else:
            if self.violation_type == "custom":
                raise NotImplementedError(
                    "PreTrendsPowerResults.power_at() cannot reconstruct "
                    "custom violation weights from an older serialized result "
                    "(violation_weights field is None). Refit "
                    "PreTrendsPower(violation_type='custom', "
                    "violation_weights=...) with the new M instead. "
                    "Fresh fits from the current library version persist "
                    "violation_weights and do not hit this guard."
                )
            # Reconstruction fallback for legacy serialized results.
            # Matches the pre-PR-B count-based linear behavior (no
            # relative_times available on an old result). Only used when
            # violation_weights is None.
            if self.violation_type == "linear":
                weights = np.arange(-n_pre + 1, 1, dtype=float)
                weights = -weights  # [n-1, n-2, ..., 1, 0]
            elif self.violation_type == "constant":
                weights = np.ones(n_pre)
            elif self.violation_type == "last_period":
                weights = np.zeros(n_pre)
                weights[-1] = 1.0
            else:
                raise ValueError(
                    f"Unknown violation_type: {self.violation_type!r}. "
                    f"Expected one of: 'linear', 'constant', 'last_period', 'custom'."
                )
            # Normalize to unit L2 norm — matches the legacy normalize-at-end
            # path in _get_violation_weights for non-relative_times callers.
            norm = np.linalg.norm(weights)
            if norm > 0:
                weights = weights / norm

        # Dispatch on the stored pretest_form. Old serialized results default
        # to pretest_form='wald' (the dataclass default) which preserves the
        # previous power_at numerical output for backwards compat.
        if self.pretest_form == "nis":
            z_alpha = float(
                self.critical_value
                if np.isfinite(self.critical_value)
                else stats.norm.ppf(1 - self.alpha / 2)
            )
            # Centralized analytical-or-MC fallback (module-level helper).
            accept_prob = _compute_nis_acceptance_prob(M, weights, self.vcov, z_alpha)
            return float(1.0 - accept_prob)

        # Wald path (legacy default, also opt-in for new fits with
        # pretest_form='wald'). Matches the pre-PR-B numerical output.
        try:
            vcov_inv = np.linalg.inv(self.vcov)
        except np.linalg.LinAlgError:
            vcov_inv = np.linalg.pinv(self.vcov)
        noncentrality = M**2 * (weights @ vcov_inv @ weights)
        power = 1 - stats.ncx2.cdf(self.critical_value, df=n_pre, nc=noncentrality)
        return float(power)


@dataclass
class PreTrendsPowerCurve(Diagnostic):
    """
    Power curve across violation magnitudes.

    Attributes
    ----------
    M_values : np.ndarray
        Grid of violation magnitudes tested.
    powers : np.ndarray
        Power at each violation magnitude.
    mdv : float
        Minimum detectable violation.
    alpha : float
        Significance level.
    target_power : float
        Target power level.
    violation_type : str
        Type of violation pattern.
    pretest_form : str
        Pretest acceptance-region form (``'nis'`` or ``'wald'``) used to
        compute the curve. NIS and Wald curves can differ materially under
        correlated Σ_22; persisting the form prevents callers from
        misinterpreting a serialized/plotted curve.
    """

    M_values: np.ndarray
    powers: np.ndarray
    mdv: float
    alpha: float
    target_power: float
    violation_type: str
    pretest_form: Literal["nis", "wald"] = "wald"

    def __repr__(self) -> str:
        return f"PreTrendsPowerCurve(n_points={len(self.M_values)}, " f"mdv={self.mdv:.4f})"

    def summary(self) -> str:
        """Return a formatted summary of the power curve."""
        lines = [
            "Pre-Trends Power Curve",
            "=" * 60,
            f"{'Grid points:':<28} {len(self.M_values)}",
            f"{'Violation type:':<28} {self.violation_type}",
            f"{'Pretest form:':<28} {self.pretest_form}",
            f"{'Significance level (alpha):':<28} {self.alpha}",
            f"{'Target power:':<28} {self.target_power}",
            f"{'Minimum detectable violation:':<28} {self.mdv:.4f}",
            "-" * 60,
            f"{'M':>12} {'Power':>10}",
        ]
        for m, p in zip(self.M_values, self.powers):
            lines.append(f"{m:>12.4f} {p:>10.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with M, power, and pretest_form columns."""
        return pd.DataFrame(
            {
                "M": self.M_values,
                "power": self.powers,
                "pretest_form": self.pretest_form,
            }
        )

    def plot(
        self,
        ax=None,
        show_mdv: bool = True,
        show_target: bool = True,
        color: str = "#2563eb",
        mdv_color: str = "#dc2626",
        target_color: str = "#22c55e",
        **kwargs,
    ):
        """
        Plot the power curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_mdv : bool, default=True
            Whether to show vertical line at MDV.
        show_target : bool, default=True
            Whether to show horizontal line at target power.
        color : str
            Color for power curve line.
        mdv_color : str
            Color for MDV vertical line.
        target_color : str
            Color for target power horizontal line.
        **kwargs
            Additional arguments passed to plt.plot().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot power curve
        ax.plot(self.M_values, self.powers, color=color, linewidth=2, label="Power", **kwargs)

        # Target power line
        if show_target:
            ax.axhline(
                y=self.target_power,
                color=target_color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Target power ({self.target_power:.0%})",
            )

        # MDV line
        if show_mdv and self.mdv is not None and np.isfinite(self.mdv):
            ax.axvline(
                x=self.mdv,
                color=mdv_color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=f"MDV = {self.mdv:.3f}",
            )

        ax.set_xlabel("Violation Magnitude (M)")
        ax.set_ylabel("Power")
        ax.set_title("Pre-Trends Test Power Curve")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return ax


# =============================================================================
# Main Class
# =============================================================================


class PreTrendsPower:
    """
    Pre-trends power analysis (Roth 2022).

    Computes the power of pre-trends tests to detect violations of parallel
    trends, and the minimum detectable violation (MDV).

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for the pre-trends test.
    power : float, default=0.80
        Target power level for MDV calculation.
    violation_type : str, default='linear'
        Type of violation pattern to consider:
        - 'linear': Violations follow a linear trend (most common)
        - 'constant': Same violation in all pre-periods
        - 'last_period': Violation only in the last pre-period
        - 'custom': User-specified violation pattern (via violation_weights)
    violation_weights : array-like, optional
        Custom weights for violation pattern. Length must equal number of
        pre-periods. Only used when violation_type='custom'.
    pretest_form : {'nis', 'wald'}, default='nis'
        Pre-trends test acceptance-region form:

        - ``'nis'``: Roth (2022) no-individually-significant pretest (Section
          II.A-B). Acceptance region is ``B_NIS(Σ) = { b : |b_t| <= z_{1-α/2}
          σ_t for all t }``. Power computed via multivariate normal box
          probability. This is the new default (PR-B 2026-05-17), matching
          both the paper's primary analysis and the R ``pretrends`` package.
        - ``'wald'``: Noncentral chi-squared on the quadratic form
          ``δ' Σ_22^{-1} δ`` (the shipped behavior prior to PR-B 2026-05-17).
          Retained as a paper-supported alternative under Propositions 1+3+4
          (Wald acceptance region is a convex ellipsoid, so all four
          propositions apply). Use this for backwards-compat with shipped
          numerical baselines.

    Examples
    --------
    Basic usage with MultiPeriodDiD results:

    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.pretrends import PreTrendsPower
    >>>
    >>> # Fit event study
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                      time='period', post_periods=[4, 5, 6, 7])
    >>>
    >>> # Analyze pre-trends power
    >>> pt = PreTrendsPower(alpha=0.05, power=0.80)
    >>> power_results = pt.fit(results)
    >>> print(power_results.summary())
    >>>
    >>> # Get power curve
    >>> curve = pt.power_curve(results)
    >>> curve.plot()

    Notes
    -----
    The pre-trends test is typically a joint test that all pre-period
    coefficients are zero. This test has limited power to detect small
    violations, especially when:

    1. There are few pre-periods
    2. Standard errors are large
    3. The violation pattern is smooth (e.g., linear trend)

    Passing a pre-trends test does NOT mean parallel trends holds. It means
    violations smaller than the MDV cannot be ruled out. For robust inference,
    combine with HonestDiD sensitivity analysis.

    References
    ----------
    Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing
        for Parallel Trends. American Economic Review: Insights, 4(3), 305-322.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        violation_type: Literal["linear", "constant", "last_period", "custom"] = "linear",
        violation_weights: Optional[np.ndarray] = None,
        pretest_form: Literal["nis", "wald"] = "nis",
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}")
        if violation_type not in ["linear", "constant", "last_period", "custom"]:
            raise ValueError(
                f"violation_type must be 'linear', 'constant', 'last_period', or 'custom', "
                f"got '{violation_type}'"
            )
        if violation_type == "custom" and violation_weights is None:
            raise ValueError("violation_weights must be provided when violation_type='custom'")
        if pretest_form not in ("nis", "wald"):
            raise ValueError(f"pretest_form must be 'nis' or 'wald', got '{pretest_form}'")

        self.alpha = alpha
        self.target_power = power
        self.violation_type = violation_type
        self.violation_weights = (
            np.asarray(violation_weights) if violation_weights is not None else None
        )
        self.pretest_form = pretest_form

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "alpha": self.alpha,
            "power": self.target_power,
            "violation_type": self.violation_type,
            "violation_weights": self.violation_weights,
            "pretest_form": self.pretest_form,
        }

    def set_params(self, **params) -> "PreTrendsPower":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key == "power":
                self.target_power = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _get_violation_weights(
        self,
        n_pre: int,
        relative_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get violation weights based on violation type.

        Parameters
        ----------
        n_pre : int
            Number of pre-treatment periods.
        relative_times : np.ndarray, optional
            Sorted relative-time labels for the pre-period coefficients
            (e.g., ``[-3, -2, -1]`` for a regular grid, ``[-5, -3, -1]``
            for an irregular grid, ``[-3, -2]`` for an anticipation-shifted
            grid with ``anticipation=1``). When provided AND
            ``violation_type='linear'``, weights are set to ``|t|`` directly
            with NO L2 normalization, so ``δ_t = M * |t|`` and the reported
            MDV is in Roth's γ units (δ_t = γ·t convention). When None,
            falls back to the legacy count-based ``[n_pre-1, ..., 1, 0] /
            ||·||_2`` direction (preserves the pre-PR-B shipped behavior
            for callers that bypass ``fit()`` and call this helper
            directly without relative-time labels).

        Returns
        -------
        np.ndarray
            Violation weights, with per-violation-type normalization
            conventions chosen so the magnitude `M` matches what
            ``REGISTRY.md`` documents for the pattern:

            - ``'linear'`` with ``relative_times``: ``|t|`` directly,
              NOT L2-normalized (so ``δ_t = M * |t|`` and the reported
              MDV is in Roth's γ units). PR-B Step 4.
            - ``'linear'`` without ``relative_times`` (legacy): the
              count-based ``[n_pre-1, ..., 0]`` direction, L2-normalized
              to unit norm (preserves pre-PR-B shipped behavior).
            - ``'constant'``: ``[1, 1, ..., 1]`` directly, NOT
              normalized — ``δ_t = M`` per period (a true level shift,
              matching the documented ``δ_t = c`` convention). PR-B R13
              fix: pre-R13 normalization gave ``δ_t = M/√K``, a silent
              rescaling that the REGISTRY/API did not document.
            - ``'last_period'``: ``[0, ..., 0, 1]`` directly. Already
              unit-norm so the post-normalization output was identical;
              the unconditional early return locks the level-shift
              contract.
            - ``'custom'``: user-supplied ``violation_weights``,
              L2-normalized to unit norm (M is the magnitude along the
              user's direction; downstream
              ``max_abs_pre_violation = M * max(|weights|)`` exposes
              the level-scale max under the MDV).
        """
        if self.violation_type == "custom":
            assert self.violation_weights is not None
            if len(self.violation_weights) != n_pre:
                raise ValueError(
                    f"violation_weights has length {len(self.violation_weights)}, "
                    f"but there are {n_pre} pre-periods"
                )
            weights = self.violation_weights.copy()
        elif self.violation_type == "linear":
            if relative_times is not None:
                # Roth (2022) δ_t = γ · t convention. Use |t| because
                # pre-period labels are negative; the resulting violation
                # vector δ_pre = M * |t| satisfies M = γ exactly.
                # NO L2 normalization — keep the γ-unit scale so the
                # reported MDV is in Roth's γ units on irregular and
                # anticipation-shifted grids. Early return; skip the
                # normalize-at-end block below. See PR-A REGISTRY ##
                # PreTrendsPower "Note (deviation — linear violation
                # pattern)" — PR-B Step 4 resolves the deviation when
                # relative_times is threaded through.
                if len(relative_times) != n_pre:
                    raise ValueError(
                        f"relative_times has length {len(relative_times)}, "
                        f"but there are {n_pre} pre-periods"
                    )
                return np.abs(np.asarray(relative_times)).astype(float)
            # Backwards-compatible fallback (no relative_times threaded):
            # legacy count-based [n_pre-1, ..., 1, 0] / ||·||_2 direction.
            # Used by callers that bypass fit() (e.g., direct
            # _get_violation_weights() unit tests) or by code paths that
            # don't have access to the actual pre-period labels.
            weights = np.arange(-n_pre + 1, 1, dtype=float)
            weights = -weights  # Now [n-1, n-2, ..., 1, 0]
        elif self.violation_type == "constant":
            # δ_t = M for all pre-periods (level shift). Skip L2
            # normalization so M is exactly the per-period level shift
            # the REGISTRY documents (`δ_t = c`). Pre-PR-B (and the
            # pre-R13 PR-B state) divided by sqrt(K), making `δ_t =
            # M/sqrt(K)` and silently re-scaling reported MDV/power on
            # constant fits by sqrt(K). PR-B R13 fix: skip the norm
            # so the public contract matches the docs.
            return np.ones(n_pre, dtype=float)
        elif self.violation_type == "last_period":
            # Violation only in last pre-period (period -1). Unnormalized
            # `[0, ..., 0, 1]` already has L2 norm 1, so this path was
            # always equivalent to the post-normalization output; keep
            # the early return for symmetry with constant + linear-with-
            # relative_times so the level-shift contract is uniform
            # across all level-pattern violation types.
            weights = np.zeros(n_pre, dtype=float)
            weights[-1] = 1.0
            return weights
        else:
            raise ValueError(f"Unknown violation_type: {self.violation_type}")

        # Normalize to unit norm (if not all zeros). The early-return
        # branches above for linear-with-relative_times, constant, and
        # last_period intentionally skip this normalization to preserve
        # the level-shift contract documented in REGISTRY.md
        # `## PreTrendsPower`. This block only fires for the linear-
        # legacy-fallback path and `violation_type='custom'`.
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights = weights / norm

        return weights

    def _extract_pre_period_params(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        pre_periods: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray], str]:
        """
        Extract pre-period parameters from results.

        Parameters
        ----------
        results : MultiPeriodDiDResults or similar
            Results object from event study estimation.
        pre_periods : list of int, optional
            Explicit list of pre-treatment periods. If None, uses results.pre_periods.

        Returns
        -------
        effects : np.ndarray
            Pre-period effect estimates.
        ses : np.ndarray
            Pre-period standard errors.
        vcov : np.ndarray
            Variance-covariance matrix for pre-period effects.
        n_pre : int
            Number of pre-periods.
        relative_times : np.ndarray or None
            Pre-period relative-time labels (Roth's δ_t = γ·t convention),
            or None for callers that bypass the labeled-grid path.
        covariance_source : str
            Provenance label describing which covariance path the
            extraction actually took:

            - ``"full_pre_period_vcov"`` when a full pre-period
              covariance sub-block was used (MPD with
              ``interaction_indices``, or CS/SA with populated
              ``event_study_vcov``).
            - ``"diag_fallback"`` when only the per-period standard
              errors were available (bootstrap / replicate-weight CS or
              SA fits, MPD without ``interaction_indices``).

            ``DiagnosticReport`` consumes this label downstream to
            decide whether the power-tier should be conservatively
            downgraded (REPORTING.md "conservative deviation" rule),
            rather than re-inferring covariance provenance from the
            result type (which would diverge from the actual extraction
            path the moment the routing changes — see PR-B Step 3).
        """
        if isinstance(results, MultiPeriodDiDResults):
            # Get pre-period information - use explicit pre_periods if provided
            if pre_periods is not None:
                all_pre_periods = list(pre_periods)
            else:
                all_pre_periods = results.pre_periods

            if len(all_pre_periods) == 0:
                raise ValueError(
                    "No pre-treatment periods found in results. "
                    "Pre-trends power analysis requires pre-period coefficients. "
                    "If you estimated all periods as post_periods, use the pre_periods "
                    "parameter to specify which are actually pre-treatment."
                )

            # Pre-period effects are in period_effects (excluding reference period)
            estimated_pre_periods = [
                p
                for p in all_pre_periods
                if p in results.period_effects and results.period_effects[p].se > 0
            ]

            if len(estimated_pre_periods) == 0:
                raise ValueError(
                    "No estimated pre-period coefficients found. "
                    "The pre-trends test requires at least one estimated "
                    "pre-period coefficient (excluding the reference period)."
                )

            n_pre = len(estimated_pre_periods)
            effects = np.array([results.period_effects[p].effect for p in estimated_pre_periods])
            ses = np.array([results.period_effects[p].se for p in estimated_pre_periods])

            # Extract vcov using stored interaction indices for robust extraction
            if (
                results.vcov is not None
                and hasattr(results, "interaction_indices")
                and results.interaction_indices is not None
            ):
                indices = [results.interaction_indices[p] for p in estimated_pre_periods]
                vcov = results.vcov[np.ix_(indices, indices)]
                covariance_source = "full_pre_period_vcov"
            else:
                vcov = np.diag(ses**2)
                covariance_source = "diag_fallback"

            # For MultiPeriodDiDResults, period identifiers are generic
            # (often calendar years, sometimes pre-shifted relative times).
            # Roth's δ_t = γ·t convention needs RELATIVE offsets from the
            # treatment / reference period. Three label-type regimes:
            #
            #   1. Numeric (int / float / np.int64) — direct float() coercion
            #      gives the correct relative offset.
            #   2. pandas.Period — period arithmetic works on the Period
            #      object directly (``p - ref`` returns ordinal-difference);
            #      we cast via the `n` attribute on the resulting offset for
            #      sub-period frequencies. Datetime-like labels (Timestamp,
            #      np.datetime64) are caught the same way and converted to
            #      days via numpy timedelta semantics.
            #   3. Genuinely non-numeric / unordered labels (string period
            #      IDs, categoricals without a ranking) — emit an explicit
            #      UserWarning and fall back to the legacy count-based
            #      [n_pre-1, ..., 0] / ||·||_2 normalized direction. The
            #      reported MDV under this fallback is NOT in Roth's γ
            #      units; users on non-numeric labels who need γ-unit MDV
            #      should re-fit with numeric period labels.
            ref = getattr(results, "reference_period", None)
            relative_times: Optional[np.ndarray] = None
            if ref is not None:
                relative_times = _coerce_relative_times_from_reference(estimated_pre_periods, ref)
            return effects, ses, vcov, n_pre, relative_times, covariance_source

        # Try CallawaySantAnnaResults
        try:
            from diff_diff.staggered import CallawaySantAnnaResults

            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects is None:
                    raise ValueError(
                        "CallawaySantAnnaResults must have event_study_effects. "
                        "Fit with aggregate='event_study'. PreTrendsPower reads the "
                        "fit-time surface and does not yet accept the post-fit container "
                        "returned by results.aggregate('event_study')."
                    )

                # Get pre-period effects. Anticipation-aware cutoff per
                # REGISTRY.md §CallawaySantAnna lines 355-395: with
                # ``anticipation=k``, true pre-periods are ``t < -k``;
                # ``t ∈ [-k, -1]`` is the anticipation window and must
                # not be used for pre-trends power. Filter out
                # normalization constraints (n_groups=0) and non-finite
                # SEs as well.
                _ant = getattr(results, "anticipation", 0) or 0
                try:
                    _ant = int(_ant)
                except (TypeError, ValueError):
                    _ant = 0
                _pre_cutoff = -_ant
                # ``safe_inference`` treats ``se <= 0`` as undefined
                # inference; filter the same way here so pre-trends
                # power never silently includes rows whose per-period
                # SE collapsed (round-33 P0 CI review on PR #318).
                pre_effects = {
                    t: data
                    for t, data in results.event_study_effects.items()
                    if t < _pre_cutoff
                    and data.get("n_groups", 1) > 0
                    and np.isfinite(data.get("se", np.nan))
                    and float(data.get("se", 0.0)) > 0
                }

                if not pre_effects:
                    raise ValueError("No pre-treatment periods found in event study.")

                pre_periods = sorted(pre_effects.keys())
                n_pre = len(pre_periods)

                effects = np.array([pre_effects[t]["effect"] for t in pre_periods])
                ses = np.array([pre_effects[t]["se"] for t in pre_periods])

                # Route through full event_study_vcov when available
                # (non-bootstrap CS fits at staggered_results.py:126-128).
                # Bootstrap CS fits clear event_study_vcov at
                # staggered.py:2032-2036, falling through to diag.
                vcov, covariance_source = _extract_event_study_vcov_subblock(
                    results, pre_periods, ses
                )

                relative_times = np.asarray(pre_periods, dtype=float)
                return effects, ses, vcov, n_pre, relative_times, covariance_source
        except ImportError:
            pass

        # Try SunAbrahamResults
        try:
            from diff_diff.sun_abraham import SunAbrahamResults

            if isinstance(results, SunAbrahamResults):
                # Same anticipation-aware pre-period cutoff as
                # CallawaySantAnna above.
                _ant = getattr(results, "anticipation", 0) or 0
                try:
                    _ant = int(_ant)
                except (TypeError, ValueError):
                    _ant = 0
                _pre_cutoff = -_ant
                # Mirror the ``se > 0`` filter applied on the CS branch.
                pre_effects = {
                    t: data
                    for t, data in results.event_study_effects.items()
                    if t < _pre_cutoff
                    and data.get("n_groups", 1) > 0
                    and np.isfinite(data.get("se", np.nan))
                    and float(data.get("se", 0.0)) > 0
                }

                if not pre_effects:
                    raise ValueError("No pre-treatment periods found in event study.")

                pre_periods = sorted(pre_effects.keys())
                n_pre = len(pre_periods)

                effects = np.array([pre_effects[t]["effect"] for t in pre_periods])
                ses = np.array([pre_effects[t]["se"] for t in pre_periods])

                # Route through full event_study_vcov when available
                # (non-bootstrap SA fits — sun_abraham.py builds the matrix
                # via W @ vcov_cohort @ W.T after _compute_iw_effects).
                # Bootstrap SA fits and replicate-weight survey fits clear
                # event_study_vcov, falling through to diag.
                vcov, covariance_source = _extract_event_study_vcov_subblock(
                    results, pre_periods, ses
                )

                relative_times = np.asarray(pre_periods, dtype=float)
                return effects, ses, vcov, n_pre, relative_times, covariance_source
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported results type: {type(results)}. "
            "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, or SunAbrahamResults."
        )

    def _compute_power(
        self,
        M: float,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Dispatch to the configured pretest form (NIS by default)."""
        if self.pretest_form == "nis":
            return self._compute_power_nis(M, weights, vcov)
        return self._compute_power_wald(M, weights, vcov)

    def _compute_power_wald(
        self,
        M: float,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute power to detect violation of magnitude M under the Wald form.

        Wald pre-trends test: H0: delta = 0 vs H1: delta != 0. Under H1 with
        violation delta = M * weights, the test statistic ``delta' V^{-1} delta``
        follows a non-central chi-squared distribution with df=K and
        noncentrality lambda = M^2 * (w' V^{-1} w). Convex (ellipsoid)
        acceptance region, so Propositions 1+3+4 of Roth (2022) all apply.

        Parameters
        ----------
        M : float
            Violation magnitude.
        weights : np.ndarray
            Normalized violation pattern.
        vcov : np.ndarray
            Variance-covariance matrix.

        Returns
        -------
        power : float
            Power to detect this violation.
        noncentrality : float
            Non-centrality parameter.
        test_stat : float
            Expected test statistic under H1.
        critical_value : float
            Critical value for the test.
        """
        n_pre = len(weights)

        # Violation vector: delta = M * weights
        delta = M * weights

        # Non-centrality parameter for chi-squared test
        # lambda = delta' * V^{-1} * delta
        try:
            vcov_inv = np.linalg.inv(vcov)
            noncentrality = delta @ vcov_inv @ delta
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            vcov_inv = np.linalg.pinv(vcov)
            noncentrality = delta @ vcov_inv @ delta

        # Critical value from chi-squared distribution
        critical_value = stats.chi2.ppf(1 - self.alpha, df=n_pre)

        # Power = P(chi2_nc > critical_value) where chi2_nc is non-central chi2
        if noncentrality > 0:
            power = 1 - stats.ncx2.cdf(critical_value, df=n_pre, nc=noncentrality)
        else:
            power = self.alpha  # Size under null

        # Expected test statistic under H1
        test_stat = n_pre + noncentrality  # Mean of non-central chi2

        return power, noncentrality, test_stat, critical_value

    def _compute_power_nis(
        self,
        M: float,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute power to detect violation of magnitude M under the NIS form.

        NIS (no-individually-significant) pre-trends test: passes iff every
        pre-period coefficient lies within its own ``+/- z_{1-alpha/2} * sigma_t``
        confidence interval. Roth (2022) Section II.A-B; matches the empirical
        convention used in 12 of 12 surveyed papers (Section I.B).

        Under H1 with violation ``delta_pre = M * weights``, the rejection
        probability is computed via the centered change-of-variable
        ``Y = beta_hat_pre - delta_pre ~ N(0, Sigma_22)``:

        .. math::
            \\text{Power} = 1 - P\\bigl(Y_t \\in [-z\\sigma_t - \\delta_t,
                                                 z\\sigma_t - \\delta_t]
                                       \\text{ for all } t\\bigr)

        Implemented via ``scipy.stats.multivariate_normal.cdf`` with
        rectangular bounds (Genz method; supports K up to ~20 cleanly).

        Parameters
        ----------
        M : float
            Violation magnitude.
        weights : np.ndarray
            Violation pattern (Linear: ``|t|`` directly when fit() threads
            ``relative_times``; constant / last_period / custom: unit-normalized).
        vcov : np.ndarray
            Variance-covariance matrix Sigma_22 of the pre-period coefficients.

        Returns
        -------
        power : float
            Probability the NIS test rejects under the alternative.
        noncentrality : float
            ``np.nan``. NIS does not have a noncentrality scalar; the
            equivalent NIS-specific output is ``nis_box_probability`` (the
            acceptance probability ``1 - power``) stored on
            ``PreTrendsPowerResults``.
        test_stat : float
            ``np.nan``. NIS rejects via a rectangular acceptance event,
            not a scalar test statistic.
        critical_value : float
            ``z_{1-alpha/2}``, the per-period normal critical value used
            to define ``B_NIS(Sigma)``.
        """
        z_alpha = float(stats.norm.ppf(1 - self.alpha / 2))
        # Centralized analytical-or-MC fallback (module-level helper);
        # handles both exception and non-finite-CDF cases.
        accept_prob = _compute_nis_acceptance_prob(M, weights, vcov, z_alpha)
        power = float(1.0 - accept_prob)
        return power, float("nan"), float("nan"), z_alpha

    def _compute_mdv(
        self,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> float:
        """Dispatch to the configured pretest form (NIS by default)."""
        if self.pretest_form == "nis":
            return self._compute_mdv_nis(weights, vcov)
        return self._compute_mdv_wald(weights, vcov)

    def _compute_mdv_wald(
        self,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> float:
        """
        Compute minimum detectable violation under the Wald form.

        Find the smallest M such that ``_compute_power_wald(M, weights, vcov)
        >= target_power``. Uses binary search on the noncentrality parameter,
        then converts back to M via ``nc = M^2 * (w' V^{-1} w)``.

        Parameters
        ----------
        weights : np.ndarray
            Normalized violation pattern.
        vcov : np.ndarray
            Variance-covariance matrix.

        Returns
        -------
        mdv : float
            Minimum detectable violation in units of M (interpreted relative
            to the ``weights`` direction; for linear weights threaded with
            ``relative_times``, this is Roth's gamma in MDV units — see
            ``_get_violation_weights``).
        """
        n_pre = len(weights)

        # Critical value
        critical_value = stats.chi2.ppf(1 - self.alpha, df=n_pre)

        # Find non-centrality parameter for target power
        # We need: P(ncx2 > critical_value) = target_power
        # Use inverse: find lambda such that ncx2.cdf(cv, df, lambda) = 1 - target_power

        def power_minus_target(nc):
            if nc <= 0:
                return self.alpha - self.target_power
            return stats.ncx2.sf(critical_value, df=n_pre, nc=nc) - self.target_power

        # Binary search for non-centrality parameter
        # Start with bounds
        nc_low, nc_high = 0, 1

        # Expand upper bound until power exceeds target
        while power_minus_target(nc_high) < 0 and nc_high < 1000:
            nc_high *= 2

        if nc_high >= 1000:
            # Target power not achievable - return inf
            return np.inf

        # Binary search
        try:
            result = optimize.brentq(power_minus_target, nc_low, nc_high)
            target_nc = result
        except ValueError:
            # Fallback: use approximate formula
            # For chi2, power ≈ Phi(sqrt(2*nc) - sqrt(2*cv))
            # Solving: sqrt(2*nc) = z_power + sqrt(2*cv)
            z_power = stats.norm.ppf(self.target_power)
            target_nc = 0.5 * (z_power + np.sqrt(2 * critical_value)) ** 2

        # Convert non-centrality to M
        # nc = delta' * V^{-1} * delta = M^2 * w' * V^{-1} * w
        try:
            vcov_inv = np.linalg.inv(vcov)
            w_Vinv_w = weights @ vcov_inv @ weights
        except np.linalg.LinAlgError:
            vcov_inv = np.linalg.pinv(vcov)
            w_Vinv_w = weights @ vcov_inv @ weights

        if w_Vinv_w > 0:
            mdv = np.sqrt(target_nc / w_Vinv_w)
        else:
            mdv = np.inf

        return mdv

    def _compute_mdv_nis(
        self,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> float:
        """
        Compute minimum detectable violation under the NIS form.

        Solves ``_compute_power_nis(M, weights, vcov) = target_power`` for M
        via a doubling expansion to bracket the root, then ``brentq`` bisect.
        Non-convergence cap at ``M_high = 1000`` returns ``np.inf`` (matches
        the Wald path's existing 1000-cap fallback).

        Parameters
        ----------
        weights : np.ndarray
            Violation pattern.
        vcov : np.ndarray
            Variance-covariance matrix Sigma_22.

        Returns
        -------
        mdv : float
            Minimum detectable violation. For linear weights threaded with
            ``relative_times``, this is Roth's gamma at the target power.
        """

        def power_minus_target(M: float) -> float:
            return self._compute_power_nis(M, weights, vcov)[0] - self.target_power

        # Boundary short-circuit: if the NIS size under the null
        # (≈ 1 - (1-α)^K under independence) already meets target_power,
        # the MDV is zero — no violation needed to reject at target rate.
        # NIS size is generally LARGER than α (chi² size), so this case
        # is reachable for small target_power (e.g., target=0.10, α=0.05,
        # K=3 → null size ≈ 0.143 > 0.10).
        if power_minus_target(0.0) >= 0:
            return 0.0

        # Doubling expansion to find an upper bound where power >= target.
        # Cap M_high at 1000 to avoid pathological infinite doubling on
        # numerically extreme Σ_22, but the cap itself does NOT mean
        # "unreachable" — explicitly check power at the capped endpoint
        # before returning inf (codex R2 P0 fix: previously the cap
        # short-circuited to inf even when power(M_high) >= target,
        # producing silently wrong MDV=inf for finite-root cases like
        # vcov=[[50000]] where MDV lies between 512 and 1024).
        M_high = 1.0
        while power_minus_target(M_high) < 0 and M_high < 1000:
            M_high *= 2

        # Defensive: if the doubling exited because M_high*2 would exceed 1000,
        # the LAST value M_high actually reached might be either above or below
        # target. Evaluate explicitly at the final M_high to decide.
        if power_minus_target(M_high) < 0:
            # Power at the cap still fails to reach target_power.
            # Genuinely unreachable in the practical range.
            return np.inf

        # Bisect on [0, M_high]. Both sign-change endpoints verified above.
        try:
            mdv = float(optimize.brentq(power_minus_target, 0.0, M_high))
        except ValueError:
            # Defensive fallback. Should be unreachable.
            mdv = float(M_high)

        return mdv

    def fit(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: Optional[float] = None,
        pre_periods: Optional[List[int]] = None,
    ) -> PreTrendsPowerResults:
        """
        Compute pre-trends power analysis.

        Parameters
        ----------
        results : MultiPeriodDiDResults, CallawaySantAnnaResults, or SunAbrahamResults
            Results from an event study estimation.
        M : float, optional
            Specific violation magnitude to evaluate. If None, evaluates at
            a default magnitude based on the data.
        pre_periods : list of int, optional
            Explicit list of pre-treatment periods to use for power analysis.
            If None, attempts to infer from results.pre_periods. Use this when
            you've estimated an event study with all periods in post_periods
            and need to specify which are actually pre-treatment.

        Returns
        -------
        PreTrendsPowerResults
            Power analysis results including power and MDV.
        """
        # Extract pre-period parameters (now includes relative_times for
        # γ-unit MDV under linear violation_type, plus the covariance-source
        # provenance label for downstream DiagnosticReport / BusinessReport
        # tier classification).
        (
            effects,
            ses,
            vcov,
            n_pre,
            relative_times,
            covariance_source,
        ) = self._extract_pre_period_params(results, pre_periods)

        # Get violation weights. relative_times threaded through so the
        # linear-violation path produces γ-unit MDV per Roth's δ_t = γ·t
        # convention (skip L2 normalization for linear-with-relative_times).
        weights = self._get_violation_weights(n_pre, relative_times=relative_times)

        # Compute MDV (dispatches on self.pretest_form)
        mdv = self._compute_mdv(weights, vcov)

        # Default M: use MDV if not specified
        if M is None:
            M = mdv if np.isfinite(mdv) else np.max(ses)

        # Compute power at specified M (dispatches on self.pretest_form)
        power, noncentrality, test_stat, critical_value = self._compute_power(M, weights, vcov)

        # NIS-specific output: the box acceptance probability. Wald fits leave
        # this as NaN; the meaningful Wald-specific scalar is `noncentrality`.
        nis_box_probability = 1.0 - power if self.pretest_form == "nis" else float("nan")

        return PreTrendsPowerResults(
            power=power,
            mdv=mdv,
            violation_magnitude=M,
            violation_type=self.violation_type,
            alpha=self.alpha,
            target_power=self.target_power,
            n_pre_periods=n_pre,
            test_statistic=test_stat,
            critical_value=critical_value,
            noncentrality=noncentrality,
            pre_period_effects=effects,
            pre_period_ses=ses,
            vcov=vcov,
            original_results=results,
            pretest_form=self.pretest_form,
            nis_box_probability=nis_box_probability,
            violation_weights=weights,
            covariance_source=covariance_source,
        )

    def power_at(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: float,
        pre_periods: Optional[List[int]] = None,
    ) -> float:
        """
        Compute power to detect a specific violation magnitude.

        Parameters
        ----------
        results : results object
            Event study results.
        M : float
            Violation magnitude.
        pre_periods : list of int, optional
            Explicit list of pre-treatment periods. See fit() for details.

        Returns
        -------
        float
            Power to detect violation of magnitude M.
        """
        result = self.fit(results, M=M, pre_periods=pre_periods)
        return result.power

    def power_curve(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M_grid: Optional[List[float]] = None,
        n_points: int = 50,
        pre_periods: Optional[List[int]] = None,
    ) -> PreTrendsPowerCurve:
        """
        Compute power across a range of violation magnitudes.

        Parameters
        ----------
        results : results object
            Event study results.
        M_grid : list of float, optional
            Specific violation magnitudes to evaluate. If None, creates
            automatic grid from 0 to 2.5 * MDV.
        n_points : int, default=50
            Number of points in automatic grid.
        pre_periods : list of int, optional
            Explicit list of pre-treatment periods. See fit() for details.

        Returns
        -------
        PreTrendsPowerCurve
            Power curve data with plot method.
        """
        # Extract parameters (6-tuple includes relative_times + covariance
        # source; the source label is currently unused on the curve path but
        # the unpack must match the helper's signature).
        _, ses, vcov, n_pre, relative_times, _ = self._extract_pre_period_params(
            results, pre_periods
        )
        weights = self._get_violation_weights(n_pre, relative_times=relative_times)

        # Compute MDV
        mdv = self._compute_mdv(weights, vcov)

        # Create M grid if not provided
        if M_grid is None:
            max_M = min(2.5 * mdv if np.isfinite(mdv) else 10 * np.max(ses), 100)
            M_grid = np.linspace(0, max_M, n_points)
        else:
            M_grid = np.asarray(M_grid)

        # Compute power at each M
        assert M_grid is not None
        powers = np.array([self._compute_power(M, weights, vcov)[0] for M in M_grid])

        return PreTrendsPowerCurve(
            M_values=M_grid,
            powers=powers,
            mdv=mdv,
            alpha=self.alpha,
            target_power=self.target_power,
            violation_type=self.violation_type,
            pretest_form=self.pretest_form,
        )

    def sensitivity_to_honest_did(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        pre_periods: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compare pre-trends power analysis with HonestDiD sensitivity.

        This method helps interpret how informative a passing pre-trends
        test is in the context of HonestDiD's relative magnitudes restriction.

        Parameters
        ----------
        results : results object
            Event study results.
        pre_periods : list of int, optional
            Explicit list of pre-treatment periods. See fit() for details.

        Returns
        -------
        dict
            Dictionary with:
            - mdv: Minimum detectable violation from pre-trends test
            - honest_M_at_mdv: Corresponding M value for HonestDiD
            - interpretation: Text explaining the relationship
        """
        pt_results = self.fit(results, pre_periods=pre_periods)
        mdv = pt_results.mdv
        # Level-scale scalar for comparison against the level-scale
        # per-period SEs. PR-B Step 4: raw `mdv` for `linear` violations
        # is now Roth's γ units (a slope); the level-scale quantity is
        # `mdv * max(|violation_weights|)`. See PreTrendsPowerResults.
        max_abs_pre_violation = pt_results.max_abs_pre_violation

        # The MDV represents the size of violation the test could detect.
        # In HonestDiD's relative magnitudes framework, M=1 means
        # post-treatment violations can be as large as the max pre-period
        # violation. ``max_abs_pre_violation`` gives us that level-scale
        # number directly.

        max_pre_se = np.max(pt_results.pre_period_ses)

        interpretation = []
        interpretation.append(f"Minimum Detectable Violation (MDV): {mdv:.4f}")
        interpretation.append(f"Max pre-period level deviation at MDV: {max_abs_pre_violation:.4f}")
        interpretation.append(f"Max pre-period SE: {max_pre_se:.4f}")

        if np.isfinite(max_abs_pre_violation):
            # Ratio of max-level-deviation to max SE — how many SEs the
            # largest pre-period violation under the MDV would be.
            mdv_in_ses = max_abs_pre_violation / max_pre_se if max_pre_se > 0 else np.inf
            interpretation.append(f"Max level deviation / max(SE): {mdv_in_ses:.2f}")

            if mdv_in_ses < 1:
                interpretation.append("→ Pre-trends test is fairly sensitive to violations.")
            elif mdv_in_ses < 2:
                interpretation.append("→ Pre-trends test has moderate sensitivity.")
            else:
                interpretation.append("→ Pre-trends test has low power to detect violations.")
                interpretation.append(
                    "  Consider using HonestDiD with larger M values for robustness."
                )
        else:
            interpretation.append(
                "→ Pre-trends test cannot achieve target power for any violation size."
            )
            interpretation.append("  Use HonestDiD sensitivity analysis for inference.")

        return {
            "mdv": mdv,
            "max_abs_pre_violation": float(max_abs_pre_violation),
            "max_pre_se": max_pre_se,
            "mdv_in_ses": (
                max_abs_pre_violation / max_pre_se
                if max_pre_se > 0 and np.isfinite(max_abs_pre_violation)
                else np.inf
            ),
            "interpretation": "\n".join(interpretation),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_pretrends_power(
    results: Union[MultiPeriodDiDResults, Any],
    M: Optional[float] = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    violation_type: str = "linear",
    pre_periods: Optional[List[int]] = None,
    violation_weights: Optional[np.ndarray] = None,
    pretest_form: Literal["nis", "wald"] = "nis",
) -> PreTrendsPowerResults:
    """
    Convenience function for pre-trends power analysis.

    Parameters
    ----------
    results : results object
        Event study results.
    M : float, optional
        Violation magnitude to evaluate.
    alpha : float, default=0.05
        Significance level.
    target_power : float, default=0.80
        Target power for MDV calculation.
    violation_type : str, default='linear'
        Type of violation pattern: ``linear`` / ``constant`` / ``last_period``
        / ``custom``. For ``custom``, also pass ``violation_weights``.
    pre_periods : list of int, optional
        Explicit list of pre-treatment periods. If None, attempts to infer
        from results. Use when you've estimated all periods as post_periods.
    violation_weights : np.ndarray, optional
        Custom violation pattern weights. Required when
        ``violation_type='custom'``; ignored for other violation types.
    pretest_form : {'nis', 'wald'}, default='nis'
        Pretest acceptance-region form. ``'nis'`` (default) implements Roth
        (2022) Section II.A-B no-individually-significant box probability via
        ``scipy.stats.multivariate_normal.cdf``; ``'wald'`` is the
        noncentral-chi-squared form retained for backwards compatibility with
        the pre-PR-B shipped numerical output (also a paper-supported
        alternative under Propositions 1+3+4).

    Returns
    -------
    PreTrendsPowerResults
        Power analysis results.

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.pretrends import compute_pretrends_power
    >>>
    >>> results = MultiPeriodDiD().fit(data, ...)
    >>> power_results = compute_pretrends_power(results, pre_periods=[0, 1, 2, 3])
    >>> print(f"MDV: {power_results.mdv:.3f}")
    >>> print(f"Power: {power_results.power:.1%}")
    """
    pt = PreTrendsPower(
        alpha=alpha,
        power=target_power,
        violation_type=violation_type,
        violation_weights=violation_weights,
        pretest_form=pretest_form,
    )
    return pt.fit(results, M=M, pre_periods=pre_periods)


def compute_mdv(
    results: Union[MultiPeriodDiDResults, Any],
    alpha: float = 0.05,
    target_power: float = 0.80,
    violation_type: str = "linear",
    pre_periods: Optional[List[int]] = None,
    violation_weights: Optional[np.ndarray] = None,
    pretest_form: Literal["nis", "wald"] = "nis",
) -> float:
    """
    Compute minimum detectable violation.

    Parameters
    ----------
    results : results object
        Event study results.
    alpha : float, default=0.05
        Significance level.
    target_power : float, default=0.80
        Target power for MDV calculation.
    violation_type : str, default='linear'
        Type of violation pattern: ``linear`` / ``constant`` / ``last_period``
        / ``custom``. For ``custom``, also pass ``violation_weights``.
    pre_periods : list of int, optional
        Explicit list of pre-treatment periods. If None, attempts to infer
        from results. Use when you've estimated all periods as post_periods.
    violation_weights : np.ndarray, optional
        Custom violation pattern weights. Required when
        ``violation_type='custom'``; ignored for other violation types.
    pretest_form : {'nis', 'wald'}, default='nis'
        Pretest acceptance-region form. See ``compute_pretrends_power`` and
        ``PreTrendsPower`` for the NIS-vs-Wald discussion.

    Returns
    -------
    float
        Minimum detectable violation.
    """
    pt = PreTrendsPower(
        alpha=alpha,
        power=target_power,
        violation_type=violation_type,
        violation_weights=violation_weights,
        pretest_form=pretest_form,
    )
    result = pt.fit(results, pre_periods=pre_periods)
    return result.mdv
