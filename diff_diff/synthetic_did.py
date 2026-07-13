"""
Synthetic Difference-in-Differences estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from diff_diff.bootstrap_utils import generate_rao_wu_weights
from diff_diff.estimators import DifferenceInDifferences
from diff_diff.linalg import solve_ols
from diff_diff.results import SyntheticDiDResults, _SyntheticDiDFitSnapshot
from diff_diff.utils import (
    _compute_regularization,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_sdid_unit_weights_survey,
    compute_time_weights,
    compute_time_weights_survey,
    safe_inference,
    validate_binary,
)


class SyntheticDiD(DifferenceInDifferences):
    """
    Synthetic Difference-in-Differences (SDID) estimator.

    Combines the strengths of Difference-in-Differences and Synthetic Control
    methods by re-weighting control units to better match treated units'
    pre-treatment trends.

    This method is particularly useful when:

    - You have few treated units (possibly just one)
    - Parallel trends assumption may be questionable
    - Control units are heterogeneous and need reweighting
    - You want robustness to pre-treatment differences

    Parameters
    ----------
    zeta_omega : float, optional
        Regularization for unit weights. If None (default), auto-computed
        from data as ``(N1 * T1)^(1/4) * noise_level`` matching R's synthdid.
    zeta_lambda : float, optional
        Regularization for time weights. If None (default), auto-computed
        from data as ``1e-6 * noise_level`` matching R's synthdid.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    variance_method : str, default="placebo"
        Method for variance estimation:

        - "placebo": Placebo-based variance matching R's synthdid::vcov(method="placebo").
          Implements Algorithm 4 from Arkhangelsky et al. (2021). Library default
          (R's default is ``"bootstrap"``; we default to placebo because it is
          unconditionally available on pweight-only survey designs and avoids the
          ~5–30× slowdown of the refit bootstrap). See REGISTRY.md §SyntheticDiD
          ``Note (default variance_method deviation from R)`` for rationale.
        - "bootstrap": Paper-faithful pairs bootstrap — Arkhangelsky et al. (2021)
          Algorithm 2 step 2, also the behavior of R's default
          synthdid::vcov(method="bootstrap") (which rebinds ``attr(estimate, "opts")``
          with ``update.omega=TRUE``, so the renormalized ω is only Frank-Wolfe
          initialization). Re-estimates ω̂_b and λ̂_b via two-pass sparsified
          Frank-Wolfe on each bootstrap draw. **Survey support (PR #352):**
          pweight-only fits use the constant per-control survey weight as ``rw``;
          full-design fits (strata/PSU/FPC) use Rao-Wu rescaled weights per draw.
          Both compose with the **weighted Frank-Wolfe** kernel
          (``min ||A·diag(rw)·ω - b||² + ζ²·Σ rw_i ω_i²``); the FW returns ω on the
          standard simplex, then ``ω_eff = rw·ω/Σ(rw·ω)`` is composed for the SDID
          estimator. See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap
          composition)`` for the argmin-set caveat.
        - "jackknife": Jackknife variance matching R's synthdid::vcov(method="jackknife").
          Implements Algorithm 3 from Arkhangelsky et al. (2021). Deterministic
          (N_control + N_treated iterations), uses fixed weights (no re-estimation).
          The ``n_bootstrap`` parameter is ignored for this method.
    n_bootstrap : int, default=200
        Number of replications for variance estimation. Used for:
        - Bootstrap: Number of bootstrap samples
        - Placebo: Number of random permutations (matches R's `replications` argument)
        Ignored when ``variance_method="jackknife"``.
    seed : int, optional
        Random seed for reproducibility. If None (default), results
        will vary between runs.

    Attributes
    ----------
    results_ : SyntheticDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with panel data:

    >>> import pandas as pd
    >>> from diff_diff import SyntheticDiD
    >>>
    >>> # Panel data with units observed over multiple time periods
    >>> # Treatment occurs at period 5 for treated units
    >>> data = pd.DataFrame({
    ...     'unit': [...],      # Unit identifier
    ...     'period': [...],    # Time period
    ...     'outcome': [...],   # Outcome variable
    ...     'treated': [...]    # 1 if unit is ever treated, 0 otherwise
    ... })
    >>>
    >>> # Fit SDID model
    >>> sdid = SyntheticDiD()
    >>> results = sdid.fit(
    ...     data,
    ...     outcome='outcome',
    ...     treatment='treated',
    ...     unit='unit',
    ...     time='period',
    ...     post_periods=[5, 6, 7, 8]
    ... )
    >>>
    >>> # View results
    >>> results.print_summary()
    >>> print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
    >>>
    >>> # Examine unit weights
    >>> weights_df = results.get_unit_weights_df()
    >>> print(weights_df.head(10))

    Notes
    -----
    The SDID estimator (Arkhangelsky et al., 2021) computes:

        τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
            - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)

    Where:
    - ω_j are unit weights (sum to 1, non-negative)
    - λ_t are time weights (sum to 1, non-negative)

    Unit weights ω are chosen to match pre-treatment outcomes:
        min ||Σ_j ω_j * Y_j,pre - Y_treated,pre||²

    This interpolates between:
    - Standard DiD (uniform weights): ω_j = 1/N_control
    - Synthetic Control (exact matching): concentrated weights

    **Conley spatial-HAC rejection.** SyntheticDiD does not support the
    Conley (1999) spatial-HAC analytical sandwich. Passing
    ``vcov_type="conley"`` or any non-``None`` Conley keyword
    (``conley_coords``, ``conley_cutoff_km``, ``conley_metric``,
    ``conley_kernel``) to ``__init__`` or ``set_params`` raises
    ``TypeError``. Rationale: SyntheticDiD's variance is derived from
    bootstrap / jackknife / placebo resampling (Arkhangelsky et al. 2021
    Algorithms 2–4), not the sandwich identity Conley plugs into. Adding
    Conley support would require either an analytical SDID sandwich path
    or a spatial-block bootstrap (Politis-Romano 1994 territory). Tracked
    as a follow-up in ``TODO.md``.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
    (2021). Synthetic Difference-in-Differences. American Economic Review,
    111(12), 4088-4118.
    """

    def __init__(
        self,
        zeta_omega: Optional[float] = None,
        zeta_lambda: Optional[float] = None,
        alpha: float = 0.05,
        variance_method: str = "placebo",
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
        # Deprecated — accepted for backward compat, ignored with warning
        lambda_reg: Optional[float] = None,
        zeta: Optional[float] = None,
        # Defensive guard against silently-ignored Conley kwargs. SyntheticDiD
        # inherits __init__ from DifferenceInDifferences but overrides with
        # literal `super().__init__(robust=True, cluster=None, alpha=alpha)`,
        # so any user-passed `vcov_type=` or `conley_*=` would be silently
        # dropped. Per `feedback_no_silent_failures`, raise loudly. Tracked
        # in TODO.md for a follow-up that wires Conley to a non-bootstrap
        # variance path on SyntheticDiD.
        vcov_type: Optional[str] = None,
        conley_coords: Optional[Tuple[str, str]] = None,
        conley_cutoff_km: Optional[float] = None,
        conley_metric: Optional[str] = None,
        conley_kernel: Optional[str] = None,
        conley_lag_cutoff: Optional[int] = None,
    ):
        if vcov_type == "conley" or any(
            v is not None
            for v in (
                conley_coords,
                conley_cutoff_km,
                conley_metric,
                conley_kernel,
                conley_lag_cutoff,
            )
        ):
            raise TypeError(
                "SyntheticDiD does not yet support vcov_type='conley' or any "
                "conley_* kwargs. SyntheticDiD uses bootstrap/jackknife/placebo "
                "variance (variance_method=...), not the analytical sandwich "
                "routed through compute_robust_vcov. Tracked in TODO.md as "
                "a follow-up."
            )
        if vcov_type is not None and vcov_type != "conley":
            raise TypeError(
                f"SyntheticDiD does not accept vcov_type={vcov_type!r}. "
                f"SyntheticDiD's variance is bootstrap/jackknife/placebo "
                f"based; configure via variance_method=..."
            )
        if lambda_reg is not None:
            warnings.warn(
                "lambda_reg is deprecated and ignored. Regularization is now "
                "auto-computed from data. Use zeta_omega to override unit weight "
                "regularization. Will be removed in v4.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if zeta is not None:
            warnings.warn(
                "zeta is deprecated and ignored. Use zeta_lambda to override "
                "time weight regularization. Will be removed in v4.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(robust=True, cluster=None, alpha=alpha)
        self.zeta_omega = zeta_omega
        self.zeta_lambda = zeta_lambda
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self._validate_config()

        self._unit_weights = None
        self._time_weights = None

    _VALID_VARIANCE_METHODS = ("bootstrap", "jackknife", "placebo")

    def _validate_config(self) -> None:
        """Validate ``variance_method`` and ``n_bootstrap`` on the current state.

        Called from both ``__init__`` and ``set_params`` so updates via the
        sklearn-style setter path enforce the same contract as construction.
        """
        if self.variance_method not in self._VALID_VARIANCE_METHODS:
            raise ValueError(
                f"variance_method must be one of {self._VALID_VARIANCE_METHODS}, "
                f"got '{self.variance_method}'"
            )
        if self.n_bootstrap < 2 and self.variance_method != "jackknife":
            raise ValueError(
                f"n_bootstrap must be >= 2 (got {self.n_bootstrap}). At least 2 "
                f"iterations are needed to estimate standard errors."
            )

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
        covariates: Optional[List[str]] = None,
        survey_design=None,
    ) -> SyntheticDiDResults:
        """
        Fit the Synthetic Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with observations for multiple units over multiple
            time periods.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
            Should be 1 for all observations of treated units
            (both pre and post treatment).
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        post_periods : list, optional
            List of time period values that are post-treatment.
            If None, uses the last half of periods.
        covariates : list, optional
            List of covariate column names. Covariates are residualized
            out before computing the SDID estimator.
        survey_design : SurveyDesign, optional
            Survey design specification. Only pweight weight_type is
            supported. Replicate-weight designs are rejected. All three
            variance methods support both pweight-only and full
            strata/PSU/FPC designs:

                method     pweight-only     strata/PSU/FPC
                bootstrap  ✓ weighted FW    ✓ weighted FW + Rao-Wu (PR #355)
                placebo    ✓                ✓ stratified permutation + weighted FW
                jackknife  ✓                ✓ PSU-level LOO + stratum aggregation

            - **Bootstrap** composes Rao-Wu rescaled weights per draw with
              the weighted-Frank-Wolfe kernel; see REGISTRY.md §SyntheticDiD
              ``Note (survey + bootstrap composition)``.
            - **Placebo** under full design uses within-stratum permutation
              (pseudo-treated sampled from controls in each treated-containing
              stratum) with weighted-FW refit per draw; fit-time feasibility
              guards raise ``ValueError`` when a treated stratum has fewer
              controls than treated units (see ``Note (survey + placebo
              composition)``).
            - **Jackknife** under full design uses PSU-level LOO with
              stratum aggregation (Rust & Rao 1996); anti-conservative with
              few PSUs per stratum — prefer ``bootstrap`` when tight SE
              calibration matters in that regime (see ``Note (survey +
              jackknife composition)``).

        Returns
        -------
        SyntheticDiDResults
            Object containing the ATT estimate, standard error,
            unit weights, and time weights.

        Raises
        ------
        ValueError
            If required parameters are missing, data validation fails,
            or a non-pweight survey design is provided. Under survey
            designs, also raises when:

            - The total survey mass on either arm is zero
              (``w_control.sum() == 0`` or ``w_treated.sum() == 0``).
              Every unit on that arm would have weight 0, encoding an
              unidentified target population (PR #355 R7 P1).
            - The composed effective-control mass
              ``(unit_weights * w_control).sum()`` is zero. Frank-Wolfe
              sparsifies ``unit_weights`` to exact zeros by design, so
              even when at least one control has positive survey weight,
              the FW solution may concentrate all mass on controls whose
              survey weights are 0. Raising up front avoids a silent
              ``0/0`` in the ``omega_eff`` normalization (PR #355 R12 P1).
            - ``survey_design`` declares ``fpc`` with no explicit
              ``psu=``. SDID Rao-Wu then treats each unit as its own
              PSU, so ``fpc`` must be ``>=`` the number of units
              (unstratified) or ``>=`` the per-stratum unit count
              (stratified). Front-door checked after
              ``collapse_survey_to_unit_level`` so the user sees a
              targeted error instead of a bootstrap-exhaustion
              failure (PR #355 R8 P1).
        NotImplementedError
            If ``survey_design`` carries replicate weights (BRR/Fay/JK1/
            JKn/SDR) — SyntheticDiD has no replicate-weight variance
            path. All three variance methods (placebo, bootstrap,
            jackknife) accept pweight-only and full strata/PSU/FPC
            analytical designs; only replicate-weight designs are
            rejected.
        """
        # Validate inputs
        if outcome is None or treatment is None or unit is None or time is None:
            raise ValueError("Must provide 'outcome', 'treatment', 'unit', and 'time'")

        # Check columns exist
        required_cols = [outcome, treatment, unit, time]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Resolve survey design
        from diff_diff.survey import (
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        # R11 P1 fix: FPC is a documented no-op on placebo (Pesarin 2001
        # §1.5 — permutation tests condition on the observed sample), but
        # ``SurveyDesign.resolve()`` itself enforces ``FPC >= n_PSU``
        # design-validity constraints (survey.py:349-368). On placebo,
        # those constraints would block legitimate fits for a design
        # element that doesn't enter the placebo math. Drop FPC from a
        # copy of the survey design before resolution so placebo
        # bypasses the validator entirely; emit the FPC no-op warning
        # at the same time. The original survey_design object is
        # preserved (caller's reference unchanged).
        survey_design_for_resolve = survey_design
        if (
            self.variance_method == "placebo"
            and survey_design is not None
            and getattr(survey_design, "fpc", None) is not None
        ):
            # R13 P3 fix: validate the FPC column name exists in `data`
            # before dropping. Otherwise a typoed ``fpc="fpc_typo"`` is
            # silently ignored on the placebo path (the missing-column
            # check inside ``SurveyDesign.resolve()`` never runs because
            # we strip FPC pre-resolve). Raise the same targeted error
            # ``resolve()`` would have raised so input-spec mistakes
            # surface even when the value is mathematically a no-op.
            fpc_col = survey_design.fpc
            if fpc_col not in data.columns:
                raise ValueError(f"FPC column '{fpc_col}' not found in data")
            import dataclasses as _dc

            warnings.warn(
                "SurveyDesign(fpc=...) is a no-op on "
                "variance_method='placebo': permutation tests are "
                "conditional on the observed sample (Pesarin 2001 §1.5), "
                "so the sampling fraction does not enter Algorithm 4 or "
                "its stratified-permutation survey extension. The FPC "
                "column is dropped from the resolved survey design for "
                "the placebo fit (this also bypasses the FPC >= n_PSU "
                "design-validity check in SurveyDesign.resolve()). Use "
                "variance_method='bootstrap' or 'jackknife' if you need "
                "FPC to participate in the variance computation.",
                UserWarning,
                stacklevel=2,
            )
            survey_design_for_resolve = _dc.replace(survey_design, fpc=None)

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design_for_resolve, data, "analytical")
        )
        # Reject replicate-weight designs — SyntheticDiD has no replicate-
        # weight variance path. Analytical (pweight / strata / PSU / FPC)
        # designs are supported across all three variance methods:
        # bootstrap via weighted-FW + Rao-Wu (PR #355); placebo via
        # stratified permutation + weighted FW; jackknife via PSU-level
        # LOO with stratum aggregation (Rust & Rao 1996).
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "SyntheticDiD does not support replicate-weight survey "
                "designs. Analytical designs are supported across all "
                "three variance methods (placebo, bootstrap, jackknife), "
                "for both pweight-only and full strata/PSU/FPC. See "
                "docs/methodology/REGISTRY.md §SyntheticDiD for the "
                "full survey support matrix."
            )
        # Validate pweight only
        if resolved_survey is not None and resolved_survey.weight_type != "pweight":
            raise ValueError(
                "SyntheticDiD survey support requires weight_type='pweight'. "
                f"Got '{resolved_survey.weight_type}'."
            )

        # Strata/PSU/FPC support matrix:
        #   bootstrap  → supported via weighted Frank-Wolfe + hybrid
        #                pairs-bootstrap + Rao-Wu rescaling (PR #355;
        #                see _bootstrap_se Rao-Wu branch).
        #   placebo    → supported via stratified permutation + weighted
        #                Frank-Wolfe (this PR; _placebo_variance_se_survey).
        #   jackknife  → supported via PSU-level LOO with stratum
        #                aggregation (this PR; _jackknife_se_survey).

        # Validate treatment is binary
        validate_binary(data[treatment].values, "treatment")

        # Get all unique time periods
        all_periods = sorted(data[time].unique())

        if len(all_periods) < 2:
            raise ValueError("Need at least 2 time periods")

        # Determine pre and post periods
        if post_periods is None:
            mid = len(all_periods) // 2
            post_periods = list(all_periods[mid:])
            pre_periods = list(all_periods[:mid])
        else:
            post_periods = list(post_periods)
            pre_periods = [p for p in all_periods if p not in post_periods]

        if len(post_periods) == 0:
            raise ValueError("Must have at least one post-treatment period")
        if len(pre_periods) == 0:
            raise ValueError("Must have at least one pre-treatment period")

        # Validate post_periods are in data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Identify treated and control units
        # Treatment indicator should be constant within unit
        unit_treatment = data.groupby(unit)[treatment].first()

        # Validate treatment is constant within unit (SDID requires block treatment)
        treatment_nunique = data.groupby(unit)[treatment].nunique()
        varying_units = treatment_nunique[treatment_nunique > 1]
        if len(varying_units) > 0:
            example_unit = varying_units.index[0]
            example_vals = sorted(data.loc[data[unit] == example_unit, treatment].unique())
            raise ValueError(
                f"Treatment indicator varies within {len(varying_units)} unit(s) "
                f"(e.g., unit '{example_unit}' has values {example_vals}). "
                f"SyntheticDiD requires 'block' treatment where treatment is "
                f"constant within each unit across all time periods. "
                f"For staggered adoption designs, use CallawaySantAnna or "
                f"ImputationDiD instead."
            )

        treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        control_units = unit_treatment[unit_treatment == 0].index.tolist()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")
        if len(control_units) == 0:
            raise ValueError("No control units found")

        # Validate balanced panel (SDID requires all units observed in all periods)
        periods_per_unit = data.groupby(unit)[time].nunique()
        expected_n_periods = len(all_periods)
        unbalanced_units = periods_per_unit[periods_per_unit != expected_n_periods]
        if len(unbalanced_units) > 0:
            example_unit = unbalanced_units.index[0]
            actual_count = unbalanced_units.iloc[0]
            raise ValueError(
                f"Panel is not balanced: {len(unbalanced_units)} unit(s) do not "
                f"have observations in all {expected_n_periods} periods "
                f"(e.g., unit '{example_unit}' has {actual_count} periods). "
                f"SyntheticDiD requires a balanced panel. Use "
                f"diff_diff.prep.balance_panel() to balance the panel first."
            )

        # Validate and extract survey weights. Pweight-only fits feed
        # placebo / jackknife / bootstrap via ``w_control`` directly.
        # Strata/PSU/FPC fits feed bootstrap via the unit-collapsed
        # ``resolved_survey_unit`` (PR #352) which Rao-Wu rescaling
        # consumes per draw.
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            # Collapse to unit level for the bootstrap survey path. The
            # row order is [control_units..., treated_units...] so
            # boot_rw[:n_control] / boot_rw[n_control:] line up with the
            # bootstrap loop's column ordering. See
            # `collapse_survey_to_unit_level` in diff_diff/survey.py.
            # Use `data` (not `working_data`) for the groupby — survey
            # design columns are unit-constant (validated above) and
            # covariate residualization doesn't shuffle row order, so the
            # collapse is invariant to which view we group on.
            from diff_diff.survey import collapse_survey_to_unit_level

            all_units_for_bootstrap = list(control_units) + list(treated_units)
            resolved_survey_unit = collapse_survey_to_unit_level(
                resolved_survey,
                data,
                unit,
                all_units_for_bootstrap,
            )
            # Front-door FPC validation for implicit-PSU Rao-Wu (PR #355
            # R8 P1). When psu is None but fpc is set,
            # ``generate_rao_wu_weights`` (bootstrap_utils.py L654-L655)
            # treats each unit as its own PSU and rejects
            # ``FPC < n_units`` per stratum mid-draw. ``_bootstrap_se``
            # catches that ``ValueError`` and keeps retrying, so the user
            # sees a generic bootstrap-exhaustion message instead of a
            # targeted FPC/design error. Validate upstream so the user
            # gets a clean error before the bootstrap loop even starts.
            #
            # R10 P1 fix: gate this validator on variance methods that
            # actually use FPC. Bootstrap (Rao-Wu) and jackknife (Rust
            # & Rao stratum aggregation) both consume FPC; placebo is
            # documented as FPC-no-op (Pesarin 2001 §1.5 — permutation
            # tests condition on the observed sample). Running the
            # validator on placebo would block legitimate placebo fits
            # for a constraint that doesn't apply to permutation math.
            if (
                self.variance_method in ("bootstrap", "jackknife")
                and resolved_survey_unit.psu is None
                and resolved_survey_unit.fpc is not None
            ):
                if resolved_survey_unit.strata is None:
                    n_units_total = len(resolved_survey_unit.weights)
                    fpc_val = float(resolved_survey_unit.fpc[0])
                    if fpc_val < n_units_total:
                        raise ValueError(
                            f"FPC ({fpc_val}) is less than the number of "
                            f"units ({n_units_total}). With no explicit "
                            "psu= column, SDID Rao-Wu treats each unit as "
                            "its own PSU; FPC must be >= the number of "
                            "units. Declare an explicit psu= column or "
                            "increase FPC."
                        )
                else:
                    unique_strata = np.unique(resolved_survey_unit.strata)
                    for h in unique_strata:
                        mask_h = resolved_survey_unit.strata == h
                        n_h_units = int(mask_h.sum())
                        fpc_h = float(resolved_survey_unit.fpc[mask_h][0])
                        if fpc_h < n_h_units:
                            raise ValueError(
                                f"FPC ({fpc_h}) in stratum {h} is less than "
                                f"the number of units in that stratum "
                                f"({n_h_units}). With no explicit psu= "
                                "column, SDID Rao-Wu treats each unit as "
                                "its own PSU within strata; FPC must be "
                                ">= the per-stratum unit count. Declare an "
                                "explicit psu= column or increase FPC."
                            )
            # Source w_control / w_treated from resolved_survey_unit.weights
            # rather than re-extracting raw panel columns. resolved_survey.weights
            # is normalized to mean=1 by SurveyDesign.resolve() (survey.py L189-
            # L203), so the weighted-FW bootstrap objective — which is NOT
            # invariant to a global rescaling of rw — produces identical SE /
            # p-value / CI under SurveyDesign(weights="w") vs "c*w" (PR #355
            # R4 P0). Placebo / jackknife paths also consume w_control /
            # w_treated but are scale-invariant (np.average divides by sum;
            # ω_eff normalization likewise), so switching to resolved weights
            # doesn't change their numerics.
            n_control_for_split = len(control_units)
            w_control = resolved_survey_unit.weights[:n_control_for_split].astype(np.float64)
            w_treated = resolved_survey_unit.weights[n_control_for_split:].astype(np.float64)
            # Front-door positive-mass guard (PR #355 R7 P1). Survey weights
            # are non-negative post-resolve() (survey.py L171-L176 rejects
            # negatives), but all-zero mass on either arm is reachable — the
            # user can assign unit survey weights of 0 to every treated or
            # every control unit, which encodes an unidentified target
            # population. The fit-time ATT formulas downstream
            # (``np.average(..., weights=w_treated)`` around L551-L582 and
            # ``omega_eff = unit_weights * w_control`` in the bootstrap /
            # placebo / jackknife dispatchers) would otherwise hit 0/0
            # normalization or propagate NaNs silently. The bootstrap loop
            # already has per-draw zero-mass retries for degenerate resamples
            # (PR #355 R2 P0); this guard is the fit-time analogue.
            if w_control.sum() <= 0:
                raise ValueError(
                    "Survey-weighted control arm has zero total mass "
                    f"(sum of w_control = {w_control.sum():.3g}). "
                    "Every control unit has survey weight 0, so the target "
                    "population is unidentified. Drop units with zero weight, "
                    "or omit survey_design if unweighted estimation is intended."
                )
            if w_treated.sum() <= 0:
                raise ValueError(
                    "Survey-weighted treated arm has zero total mass "
                    f"(sum of w_treated = {w_treated.sum():.3g}). "
                    "Every treated unit has survey weight 0, so the target "
                    "population is unidentified. Drop units with zero weight, "
                    "or omit survey_design if unweighted estimation is intended."
                )
        else:
            w_treated = None
            w_control = None
            resolved_survey_unit = None

        # Residualize covariates if provided
        working_data = data.copy()
        if covariates:
            working_data = self._residualize_covariates(
                working_data,
                outcome,
                covariates,
                unit,
                time,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
            )

        # Create outcome matrices
        # Shape: (n_periods, n_units)
        Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated = (
            self._create_outcome_matrices(
                working_data,
                outcome,
                unit,
                time,
                pre_periods,
                post_periods,
                treated_units,
                control_units,
            )
        )

        # --- Y normalization ---------------------------------------------
        # τ is location-invariant and scale-equivariant in Y. Normalizing Y
        # once before weight optimization, the estimator, and the variance
        # procedures (and rescaling τ/SE/CI/effects by Y_scale at the end)
        # is mathematically a no-op but prevents ~6-digit precision loss in
        # the SDID double-difference when outcomes span millions-to-billions.
        # Normalization constants come from controls' pre-period only so the
        # reference is unaffected by treatment. See REGISTRY.md §SyntheticDiD
        # edge cases and synth-inference/synthdid#71 for R's version.
        Y_shift = float(np.mean(Y_pre_control))
        Y_scale_raw = float(np.std(Y_pre_control))
        # Relative floor: avoid amplifying roundoff when std is tiny but
        # nonzero (near-constant Y_pre_control). Fall back to 1.0 in that
        # case and on non-finite std.
        _scale_floor = 1e-12 * max(abs(Y_shift), 1.0)
        Y_scale = Y_scale_raw if np.isfinite(Y_scale_raw) and Y_scale_raw > _scale_floor else 1.0
        Y_pre_control_n = (Y_pre_control - Y_shift) / Y_scale
        Y_post_control_n = (Y_post_control - Y_shift) / Y_scale
        Y_pre_treated_n = (Y_pre_treated - Y_shift) / Y_scale
        Y_post_treated_n = (Y_post_treated - Y_shift) / Y_scale

        # Auto-regularization on normalized Y. FW's argmin is invariant under
        # (Y, ζ) -> (Y/s, ζ/s); auto-zetas computed on Y_n are already on the
        # normalized scale. User-supplied zetas are in original-Y units and
        # are divided by Y_scale for internal FW use. Original-scale values
        # are stored on results_ / self so diagnostic methods (in_time_placebo,
        # sensitivity_to_zeta_omega) — which operate on the stored original-
        # scale fit snapshot — see the same zeta the user specified.
        auto_zeta_omega_n, auto_zeta_lambda_n = _compute_regularization(
            Y_pre_control_n, len(treated_units), len(post_periods)
        )
        zeta_omega_n = (
            self.zeta_omega / Y_scale if self.zeta_omega is not None else auto_zeta_omega_n
        )
        zeta_lambda_n = (
            self.zeta_lambda / Y_scale if self.zeta_lambda is not None else auto_zeta_lambda_n
        )
        # Report the user-supplied value exactly (no roundoff from /*Y_scale
        # roundtrip); report auto-zeta rescaled to original Y units.
        zeta_omega = self.zeta_omega if self.zeta_omega is not None else auto_zeta_omega_n * Y_scale
        zeta_lambda = (
            self.zeta_lambda if self.zeta_lambda is not None else auto_zeta_lambda_n * Y_scale
        )

        # Store noise level for diagnostics (reported on original Y scale).
        from diff_diff.utils import _compute_noise_level

        noise_level_n = _compute_noise_level(Y_pre_control_n)
        noise_level = noise_level_n * Y_scale

        # Data-dependent convergence threshold (matches R's 1e-5 * noise.level),
        # evaluated on normalized Y since FW operates on normalized Y. Floor of
        # 1e-5 when noise is zero: R would use 0.0, causing FW to run all
        # max_iter iterations; the floor enables early stop on zero-variation
        # inputs without changing the optimum.
        min_decrease = 1e-5 * noise_level_n if noise_level_n > 0 else 1e-5

        # Compute unit weights (Frank-Wolfe with sparsification) on normalized Y.
        # Survey weights enter via the treated mean target.
        if w_treated is not None:
            Y_pre_treated_mean_n = np.average(Y_pre_treated_n, axis=1, weights=w_treated)
        else:
            Y_pre_treated_mean_n = np.mean(Y_pre_treated_n, axis=1)

        unit_weights = compute_sdid_unit_weights(
            Y_pre_control_n,
            Y_pre_treated_mean_n,
            zeta_omega=zeta_omega_n,
            min_decrease=min_decrease,
        )

        # Compute time weights (Frank-Wolfe on collapsed form) on normalized Y.
        time_weights = compute_time_weights(
            Y_pre_control_n,
            Y_post_control_n,
            zeta_lambda=zeta_lambda_n,
            min_decrease=min_decrease,
        )

        # Compose ω with control survey weights (WLS regression interpretation).
        # Frank-Wolfe finds best trajectory match; survey weights reweight by
        # population importance post-optimization.
        if w_control is not None:
            omega_eff = unit_weights * w_control
            # Front-door effective-control guard (PR #355 R12 P1). The R7 P1
            # check (``w_control.sum() > 0`` at the raw level, synthetic_did.py
            # L470-L499) is insufficient here: Frank-Wolfe sparsifies
            # ``unit_weights`` to exact zeros by design, so even when at least
            # one control has positive survey weight, FW may concentrate all
            # mass on a subset whose survey weights are all 0. The composed
            # vector ``unit_weights * w_control`` would then sum to 0, the
            # downstream ``omega_eff / omega_eff.sum()`` would emit NaN, and
            # the fit would return NaN ATT / SE silently. The analogous
            # guards already exist for the bootstrap loop
            # (``omega_scaled.sum() <= 0`` retry) and jackknife
            # (``effective_control > 0`` support gate); this restores the
            # contract at fit time.
            omega_eff_sum = float(omega_eff.sum())
            if omega_eff_sum <= 0:
                raise ValueError(
                    "SDID point estimate is unidentified: the Frank-Wolfe "
                    "solution concentrates all synthetic-control mass on "
                    "units with zero survey weight, so the composed "
                    "omega_eff = unit_weights * w_control sums to "
                    f"{omega_eff_sum:.3g}. Every control unit with positive "
                    "fit-time weight has survey weight 0. Drop zero-weight "
                    "controls, or omit survey_design if unweighted "
                    "estimation is intended."
                )
            omega_eff = omega_eff / omega_eff_sum
        else:
            omega_eff = unit_weights

        # Compute SDID estimate on normalized Y, then rescale to original units.
        if w_treated is not None:
            Y_post_treated_mean_n = np.average(Y_post_treated_n, axis=1, weights=w_treated)
        else:
            Y_post_treated_mean_n = np.mean(Y_post_treated_n, axis=1)

        att_n = compute_sdid_estimator(
            Y_pre_control_n,
            Y_post_control_n,
            Y_pre_treated_mean_n,
            Y_post_treated_mean_n,
            omega_eff,
            time_weights,
        )
        att = att_n * Y_scale

        # Recover original-scale treated means for diagnostics / trajectories.
        Y_pre_treated_mean = Y_pre_treated_mean_n * Y_scale + Y_shift
        Y_post_treated_mean = Y_post_treated_mean_n * Y_scale + Y_shift

        # Compute pre-treatment fit (RMSE) using composed weights on the
        # original Y (user-visible scale). omega_eff is a simplex — applies
        # cleanly to any linear rescale of Y — so trajectories live on the
        # original outcome scale for plotting and the poor-fit warning.
        synthetic_pre_trajectory = Y_pre_control @ omega_eff
        synthetic_post_trajectory = Y_post_control @ omega_eff
        pre_fit_rmse = np.sqrt(np.mean((Y_pre_treated_mean - synthetic_pre_trajectory) ** 2))

        # Warn if pre-treatment fit is poor (Registry requirement).
        # Threshold: 1× SD of treated pre-treatment outcomes — a natural baseline
        # since RMSE exceeding natural variation indicates the synthetic control
        # fails to reproduce the treated series' level or trend.
        pre_treatment_sd = (
            np.std(Y_pre_treated_mean, ddof=1) if len(Y_pre_treated_mean) > 1 else 0.0
        )
        if pre_treatment_sd > 0 and pre_fit_rmse > pre_treatment_sd:
            warnings.warn(
                f"Pre-treatment fit is poor: RMSE ({pre_fit_rmse:.4f}) exceeds "
                f"the standard deviation of treated pre-treatment outcomes "
                f"({pre_treatment_sd:.4f}). The synthetic control may not "
                f"adequately reproduce treated unit trends. Consider adding "
                f"more control units or adjusting regularization.",
                UserWarning,
                stacklevel=2,
            )

        # Treated-unit trajectories (the pre/post means already computed above).
        treated_pre_trajectory = Y_pre_treated_mean
        treated_post_trajectory = Y_post_treated_mean

        # Detect full-design survey (strata/PSU/FPC). The unit-collapsed
        # ``resolved_survey_unit`` carries the per-unit strata/psu/fpc
        # arrays ordered as [control..., treated...] to match the
        # downstream variance-method column layout.
        _full_design_survey = resolved_survey_unit is not None and (
            resolved_survey_unit.strata is not None
            or resolved_survey_unit.psu is not None
            or resolved_survey_unit.fpc is not None
        )
        if _full_design_survey:
            _n_c = len(control_units)
            _strata_control = (
                resolved_survey_unit.strata[:_n_c]
                if resolved_survey_unit.strata is not None
                else None
            )
            _strata_treated = (
                resolved_survey_unit.strata[_n_c:]
                if resolved_survey_unit.strata is not None
                else None
            )
            _psu_control = (
                resolved_survey_unit.psu[:_n_c] if resolved_survey_unit.psu is not None else None
            )
            _psu_treated = (
                resolved_survey_unit.psu[_n_c:] if resolved_survey_unit.psu is not None else None
            )
            _fpc_control = (
                resolved_survey_unit.fpc[:_n_c] if resolved_survey_unit.fpc is not None else None
            )
            _fpc_treated = (
                resolved_survey_unit.fpc[_n_c:] if resolved_survey_unit.fpc is not None else None
            )
        else:
            _strata_control = None
            _strata_treated = None
            _psu_control = None
            _psu_treated = None
            _fpc_control = None
            _fpc_treated = None

        # Placebo routes to the survey allocator whenever **strata or
        # PSU** is declared (FPC alone does NOT flip dispatch). For
        # PSU-without-strata designs, the whole panel is synthesized
        # as a single stratum (stratified permutation degenerates to
        # global within-stratum permutation, still dispatched through
        # the weighted-FW path).
        #
        # FPC handling on placebo (R8 P1 fix): permutation tests are
        # conditional on the observed sample (Pesarin 2001 §1.5), so
        # the sampling fraction does not enter Algorithm 4 or its
        # stratified-permutation extension. Including FPC in the
        # dispatch trigger would silently switch numerics (weighted-FW
        # vs unweighted-FW + post-hoc composition) on a survey design
        # element that has no place in the placebo math. Drop FPC from
        # the dispatch condition; emit a ``UserWarning`` below if FPC
        # is set with placebo to surface the no-op contract.
        _placebo_use_survey_path = (
            self.variance_method == "placebo"
            and resolved_survey_unit is not None
            and (resolved_survey_unit.strata is not None or resolved_survey_unit.psu is not None)
        )
        # NOTE: the FPC no-op warning for placebo is emitted earlier
        # (before ``_resolve_survey_for_fit``); ``resolved_survey_unit.fpc``
        # is already None on the placebo path because the FPC column is
        # dropped from a copy of the survey design pre-resolve. No
        # duplicate warning here.

        # Jackknife routes to the survey allocator whenever PSU or FPC or
        # strata is declared. PSU-without-strata is treated as a single
        # stratum (Rust & Rao 1996 JK1 form) inside
        # ``_jackknife_se_survey``.
        _jackknife_use_survey_path = _full_design_survey and self.variance_method == "jackknife"

        # Synthesize a single stratum for PSU/FPC-without-strata designs
        # so the placebo / jackknife survey paths can treat them as the
        # JK1 / global-permutation degenerate case of the stratified
        # allocator. The `_strata_*_eff` arrays are passed to the survey
        # methods; the original `_strata_*` arrays stay None so other
        # code paths (REGISTRY, metadata) see the true design.
        if _full_design_survey and _strata_control is None:
            _strata_control_eff: np.ndarray = np.zeros(len(control_units), dtype=np.int64)
            _strata_treated_eff: np.ndarray = np.zeros(len(treated_units), dtype=np.int64)
        else:
            _strata_control_eff = _strata_control
            _strata_treated_eff = _strata_treated

        # Fit-time feasibility guard for stratified-permutation placebo
        # (per `feedback_front_door_over_retry_swallow.md`). Case B / Case C
        # are hard failures — partial-permutation fallback would silently
        # change the null-distribution semantics and produce an incoherent
        # test. Must run *before* the retry loop below swallows ValueErrors
        # via `except (ValueError, LinAlgError, ZeroDivisionError): continue`.
        if _placebo_use_survey_path:
            unique_treated_strata, treated_counts = np.unique(
                _strata_treated_eff, return_counts=True
            )
            has_nondegenerate_stratum = False
            assert w_control is not None  # always set on full-design survey
            for h, n_t_h in zip(unique_treated_strata, treated_counts):
                n_c_h = int(np.sum(_strata_control_eff == h))
                if n_c_h == 0:
                    raise ValueError(
                        "Stratified-permutation placebo requires at least "
                        f"one control per stratum containing treated units; "
                        f"stratum {h} has 0 controls and {int(n_t_h)} "
                        "treated units. Either rebalance the panel, drop "
                        f"stratum {h} from the design, or use "
                        "variance_method='bootstrap' (which supports the "
                        "same full survey design via weighted-FW + Rao-Wu "
                        "without a permutation-feasibility constraint)."
                    )
                if n_c_h < int(n_t_h):
                    raise ValueError(
                        "Stratified-permutation placebo requires at least "
                        "n_treated controls per stratum containing treated "
                        "units (for exact-count within-stratum "
                        f"permutation); stratum {h} has {n_c_h} controls "
                        f"but {int(n_t_h)} treated units. Either rebalance "
                        "the panel, drop the undersupplied stratum, or use "
                        "variance_method='bootstrap' (which supports the "
                        "same full survey design via weighted-FW + Rao-Wu "
                        "without a permutation-feasibility constraint)."
                    )
                # Case E (R9 P1) — row-count guards passed (n_c_h ≥ n_t_h)
                # but the stratum has fewer positive-weight controls
                # than treated. The placebo allocator computes pseudo-
                # treated means as ``np.average(Y, weights=w_control[idx])``;
                # if too few controls have positive weight, draws can
                # pick all-zero-weight subsets (ZeroDivisionError on
                # np.average) and the retry loop swallows them as a
                # generic ``n_successful=0`` warning + ``SE=0.0``.
                # Front-door the targeted error.
                w_in_h = w_control[_strata_control_eff == h]
                n_c_h_positive = int(np.sum(w_in_h > 0))
                if n_c_h_positive < int(n_t_h):
                    raise ValueError(
                        "Stratified-permutation placebo requires at least "
                        "n_treated controls with positive survey weight "
                        "per stratum containing treated units (the "
                        "pseudo-treated mean uses survey-weighted "
                        f"averaging); stratum {h} has {n_c_h_positive} "
                        f"positive-weight controls (out of {n_c_h} total) "
                        f"but {int(n_t_h)} treated units. Either rebalance "
                        "the panel, drop the undersupplied stratum, or use "
                        "variance_method='bootstrap' (which supports the "
                        "same full survey design via weighted-FW + Rao-Wu "
                        "without a per-draw positive-mass constraint)."
                    )
                # Non-degenerate iff this stratum yields ≥2 distinct
                # positive-mass pseudo-treated draws. Two necessary
                # conditions, both required:
                #   * ``n_c_h > n_t_h`` — raw without-replacement count
                #     allows multiple subsets (otherwise only the
                #     "all-controls-as-pseudo-treated" subset exists,
                #     regardless of weights — Case D classical shape).
                #   * ``n_c_h_positive >= 2`` — at least 2 distinct
                #     positive-mass means are reachable. With only 1
                #     positive-weight control, every successful pick
                #     reduces to that single control's mean (zero-
                #     weight cohabitants contribute 0 to numerator and
                #     denominator), regardless of how many subsets the
                #     raw allocator can construct (Case D effective
                #     single-support shape, R11 P1).
                if n_c_h > int(n_t_h) and n_c_h_positive >= 2:
                    has_nondegenerate_stratum = True
            # Case D: every treated stratum is effectively single-
            # support, so the placebo null collapses to a single
            # positive-mass allocation. Two paths into this:
            #   * Raw exact-count (``n_c_h == n_t_h`` for every treated
            #     stratum, R4 P1): the without-replacement permutation
            #     yields a single subset, every draw is identical.
            #   * Effective single-support (``n_c_h_positive < 2`` for
            #     every treated stratum, R11 P1): positive-mass picks
            #     reduce to a single distinct mean even when raw count
            #     counts are larger, because zero-weight controls
            #     contribute 0 to numerator and denominator. Successful
            #     draws all collapse to the unique positive-weight
            #     subset.
            # Both shapes produce SE = FP noise (~1e-16) — reject up
            # front rather than silently reporting a near-zero SE.
            if not has_nondegenerate_stratum:
                detail = ", ".join(
                    f"stratum {h}: n_c={int(np.sum(_strata_control_eff == h))}, "
                    f"n_c_positive={int(np.sum(w_control[_strata_control_eff == h] > 0))}, "
                    f"n_t={int(n_t_h)}"
                    for h, n_t_h in zip(unique_treated_strata, treated_counts)
                )
                raise ValueError(
                    "Stratified-permutation placebo support is degenerate: "
                    "every treated-containing stratum has fewer than 2 "
                    "positive-weight controls, so within-stratum "
                    "permutation yields a single distinct positive-mass "
                    f"pseudo-treated mean across all draws ({detail}). "
                    "The resulting placebo distribution collapses to one "
                    "point and SE is not a meaningful null estimate. At "
                    "least one treated stratum must have ≥2 positive-"
                    "weight controls (and n_c_positive > n_t for the "
                    "test to have ≥2 distinct allocations). Either "
                    "rebalance the panel, or use "
                    "variance_method='bootstrap' (which supports the "
                    "same full survey design via weighted-FW + Rao-Wu "
                    "without a permutation-feasibility constraint)."
                )

        # Compute standard errors on normalized Y, rescale to original units.
        # Variance procedures resample / permute indices (independent of Y
        # values) so RNG streams stay aligned across scales.
        if self.variance_method == "bootstrap":
            # Paper-faithful pairs bootstrap (Algorithm 2 step 2): re-estimate
            # ω̂_b and λ̂_b via Frank-Wolfe on each draw. With survey designs
            # the FW switches to the weighted-FW variant and Rao-Wu rescaling
            # supplies per-draw weights (PR #355). Pweight-only designs use
            # constant w_control across draws; full designs use Rao-Wu draws.
            # Determine which survey path the bootstrap should use:
            #   - resolved_survey_unit + strata/PSU/FPC → Rao-Wu rescaling
            #   - pweight-only (resolved_survey_unit but no strata/PSU) →
            #     pass w_control/w_treated as constant rw per draw (the
            #     bootstrap branch sets `_pweight_only` from `w_control`
            #     when resolved_survey is None).
            #   - non-survey → pass nothing (legacy path).
            _boot_resolved_survey = resolved_survey_unit if _full_design_survey else None
            _boot_w_control = w_control if not _full_design_survey else None
            _boot_w_treated = w_treated if not _full_design_survey else None

            se_n, bootstrap_estimates_n = self._bootstrap_se(
                Y_pre_control_n,
                Y_post_control_n,
                Y_pre_treated_n,
                Y_post_treated_n,
                unit_weights,
                time_weights,
                w_control=_boot_w_control,
                w_treated=_boot_w_treated,
                resolved_survey=_boot_resolved_survey,
                zeta_omega_n=zeta_omega_n,
                zeta_lambda_n=zeta_lambda_n,
                min_decrease=min_decrease,
            )
            se = se_n * Y_scale
            variance_effects = np.asarray(bootstrap_estimates_n) * Y_scale
            inference_method = "bootstrap"
        elif self.variance_method == "jackknife":
            if _jackknife_use_survey_path:
                # PSU-level LOO + stratum aggregation (Rust & Rao 1996).
                assert w_control is not None and w_treated is not None
                # R5 P1 fix: validate ``lonely_psu`` mode. The survey
                # jackknife currently skips singleton strata (n_h < 2)
                # unconditionally — equivalent to R ``survey::svyjkn``'s
                # ``"remove"`` and ``"certainty"`` modes (both zero-
                # contribution for singleton strata). ``"adjust"`` (use
                # overall mean for singleton strata) is not implemented
                # for SDID jackknife; reject upfront rather than silently
                # treating it as ``"remove"``.
                _lonely_psu_mode = getattr(resolved_survey_unit, "lonely_psu", "remove")
                if _lonely_psu_mode not in ("remove", "certainty"):
                    raise NotImplementedError(
                        f"SurveyDesign(lonely_psu={_lonely_psu_mode!r}) is "
                        "not supported on the SDID jackknife survey path. "
                        "'remove' and 'certainty' are equivalent here "
                        "(both contribute 0 variance for singleton strata, "
                        "which is the canonical Rust & Rao 1996 behavior). "
                        "'adjust' requires an overall-mean fallback per "
                        "stratum that is not yet implemented for SDID "
                        "jackknife; use variance_method='bootstrap' (which "
                        "supports all three ``lonely_psu`` modes via the "
                        "weighted-FW + Rao-Wu path) or switch the design "
                        "to lonely_psu='remove'."
                    )
                # Unstratified designs use the synthesized single stratum
                # (``_strata_*_eff``) so the loop reduces to classical
                # JK1 (single-stratum PSU-LOO).
                se_n, jackknife_estimates_n = self._jackknife_se_survey(
                    Y_pre_control_n,
                    Y_post_control_n,
                    Y_pre_treated_n,
                    Y_post_treated_n,
                    unit_weights,
                    time_weights,
                    w_control=w_control,
                    w_treated=w_treated,
                    strata_control=_strata_control_eff,
                    strata_treated=_strata_treated_eff,
                    psu_control=_psu_control,
                    psu_treated=_psu_treated,
                    fpc_control=_fpc_control,
                    fpc_treated=_fpc_treated,
                    lonely_psu=_lonely_psu_mode,
                )
            else:
                # Fixed-weight jackknife (R's synthdid Algorithm 3)
                se_n, jackknife_estimates_n = self._jackknife_se(
                    Y_pre_control_n,
                    Y_post_control_n,
                    Y_pre_treated_n,
                    Y_post_treated_n,
                    unit_weights,
                    time_weights,
                    w_treated=w_treated,
                    w_control=w_control,
                )
            se = se_n * Y_scale
            variance_effects = np.asarray(jackknife_estimates_n) * Y_scale
            inference_method = "jackknife"
        else:
            # Use placebo-based variance (R's synthdid Algorithm 4).
            # Placebo re-estimates ω, λ inside the loop; it must receive the
            # normalized zetas and operate on normalized Y.
            if _placebo_use_survey_path:
                # Stratified permutation + weighted-FW (Pesarin 2001).
                # PSU/FPC-without-strata designs use a synthesized single
                # stratum (``_strata_*_eff``), which makes the stratified
                # permutation degenerate to a global within-stratum
                # permutation dispatched through the weighted-FW path.
                assert w_control is not None
                se_n, variance_effects_n = self._placebo_variance_se_survey(
                    Y_pre_control_n,
                    Y_post_control_n,
                    Y_pre_treated_mean_n,
                    Y_post_treated_mean_n,
                    strata_control=_strata_control_eff,
                    treated_strata=_strata_treated_eff,
                    zeta_omega=zeta_omega_n,
                    zeta_lambda=zeta_lambda_n,
                    min_decrease=min_decrease,
                    replications=self.n_bootstrap,
                    w_control=w_control,
                )
            else:
                se_n, variance_effects_n = self._placebo_variance_se(
                    Y_pre_control_n,
                    Y_post_control_n,
                    Y_pre_treated_mean_n,
                    Y_post_treated_mean_n,
                    n_treated=len(treated_units),
                    zeta_omega=zeta_omega_n,
                    zeta_lambda=zeta_lambda_n,
                    min_decrease=min_decrease,
                    replications=self.n_bootstrap,
                    w_control=w_control,
                    init_omega=unit_weights,
                    init_lambda=time_weights,
                )
            se = se_n * Y_scale
            variance_effects = np.asarray(variance_effects_n) * Y_scale
            inference_method = "placebo"

        # Compute test statistics
        t_stat, p_value_analytical, conf_int = safe_inference(att, se, alpha=self.alpha)
        # Empirical p-value is valid only for placebo (Algorithm 4): control
        # permutations produce draws from the null distribution, centered on 0.
        # Bootstrap draws (Algorithm 2) are resampled units centered on τ̂
        # (sampling distribution, not null), and jackknife pseudo-values are not
        # null-distribution draws either. Both use the analytical p-value from
        # the bootstrap/jackknife SE.
        if inference_method == "placebo" and len(variance_effects) > 0 and np.isfinite(t_stat):
            p_value = max(
                np.mean(np.abs(variance_effects) >= np.abs(att)),
                1.0 / (len(variance_effects) + 1),
            )
        else:
            p_value = p_value_analytical

        # Create weight dictionaries.  When survey weights are active, store
        # the effective (composed) weights that were actually used for the ATT
        # so that results.unit_weights matches the estimator.
        unit_weights_dict = {unit_id: w for unit_id, w in zip(control_units, omega_eff)}
        time_weights_dict = {period: w for period, w in zip(pre_periods, time_weights)}

        # Jackknife LOO ID/role arrays parallel to variance_effects positions
        # (first n_control entries are control-LOO, next n_treated are treated-LOO;
        # see _jackknife_se docstring).
        loo_unit_ids: Optional[List[Any]]
        loo_roles: Optional[List[str]]
        if inference_method == "jackknife" and len(variance_effects) > 0:
            loo_unit_ids = list(control_units) + list(treated_units)
            loo_roles = ["control"] * len(control_units) + ["treated"] * len(treated_units)
        else:
            loo_unit_ids = None
            loo_roles = None

        fit_snapshot = _SyntheticDiDFitSnapshot(
            Y_pre_control=Y_pre_control,
            Y_post_control=Y_post_control,
            Y_pre_treated=Y_pre_treated,
            Y_post_treated=Y_post_treated,
            control_unit_ids=list(control_units),
            treated_unit_ids=list(treated_units),
            pre_periods=list(pre_periods),
            post_periods=list(post_periods),
            w_control=w_control,
            w_treated=w_treated,
            Y_shift=Y_shift,
            Y_scale=Y_scale,
        )

        # Freeze the public diagnostic arrays so mutation via the results
        # object cannot silently invalidate later diagnostic calls.
        for _arr in (
            synthetic_pre_trajectory,
            synthetic_post_trajectory,
            treated_pre_trajectory,
            treated_post_trajectory,
            time_weights,
        ):
            _arr.setflags(write=False)

        # Store results. Internal diagnostic state (_loo_*, _fit_snapshot)
        # is attached as plain attributes after construction so that
        # dataclass-recursive serializers (dataclasses.asdict,
        # dataclasses.fields, dataclasses.replace) cannot reach retained
        # panel state or unit IDs.
        self.results_ = SyntheticDiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(data),
            n_treated=len(treated_units),
            n_control=len(control_units),
            unit_weights=unit_weights_dict,
            time_weights=time_weights_dict,
            pre_periods=pre_periods,
            post_periods=post_periods,
            alpha=self.alpha,
            variance_method=inference_method,
            noise_level=noise_level,
            zeta_omega=zeta_omega,
            zeta_lambda=zeta_lambda,
            pre_treatment_fit=pre_fit_rmse,
            variance_effects=variance_effects if len(variance_effects) > 0 else None,
            n_bootstrap=self.n_bootstrap if inference_method == "bootstrap" else None,
            survey_metadata=survey_metadata,
            synthetic_pre_trajectory=synthetic_pre_trajectory,
            synthetic_post_trajectory=synthetic_post_trajectory,
            treated_pre_trajectory=treated_pre_trajectory,
            treated_post_trajectory=treated_post_trajectory,
            time_weights_array=time_weights,
        )
        # results_ was just constructed above.
        assert self.results_ is not None
        # Explicit LOO granularity flag for ``get_loo_effects_df``. The
        # non-survey and pweight-only jackknife paths run unit-level LOO
        # (one estimate per unit, matching ``control_unit_ids +
        # treated_unit_ids``); the full-design survey jackknife runs
        # PSU-level LOO and returns a flat PSU-indexed replicate array.
        # Unit-level positional join onto ``_loo_unit_ids`` is well-
        # defined only for the unit-level path.
        if inference_method == "jackknife":
            self.results_._loo_granularity = "psu" if _jackknife_use_survey_path else "unit"
        else:
            self.results_._loo_granularity = None
        # Only populate unit-level LOO bookkeeping when the granularity
        # is actually unit-level (R7 P3). Leaving ``_loo_unit_ids`` /
        # ``_loo_roles`` populated on the PSU path would cause
        # ``_loo_unit_ids is not None`` availability checks (e.g.,
        # ``practitioner.py`` / canned guidance) to call
        # ``get_loo_effects_df()`` and hit ``NotImplementedError``.
        if self.results_._loo_granularity == "unit":
            self.results_._loo_unit_ids = loo_unit_ids
            self.results_._loo_roles = loo_roles
        else:
            self.results_._loo_unit_ids = None
            self.results_._loo_roles = None
        self.results_._fit_snapshot = fit_snapshot

        self._unit_weights = unit_weights
        self._time_weights = time_weights
        self.is_fitted_ = True

        return self.results_

    def _create_outcome_matrices(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create outcome matrices for SDID estimation.

        Returns
        -------
        tuple
            (Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated)
            Each is a 2D array with shape (n_periods, n_units)
        """
        # Pivot data to wide format
        pivot = data.pivot(index=time, columns=unit, values=outcome)

        # Extract submatrices
        Y_pre_control = pivot.loc[pre_periods, control_units].values
        Y_post_control = pivot.loc[post_periods, control_units].values
        Y_pre_treated = pivot.loc[pre_periods, treated_units].values
        Y_post_treated = pivot.loc[post_periods, treated_units].values

        return (
            Y_pre_control.astype(float),
            Y_post_control.astype(float),
            Y_pre_treated.astype(float),
            Y_post_treated.astype(float),
        )

    def _residualize_covariates(
        self,
        data: pd.DataFrame,
        outcome: str,
        covariates: List[str],
        unit: str,
        time: str,
        survey_weights=None,
        survey_weight_type=None,
    ) -> pd.DataFrame:
        """
        Residualize outcome by regressing out covariates.

        Uses two-way fixed effects to partial out covariates. When survey
        weights are provided, uses WLS for population-representative
        covariate removal.
        """
        data = data.copy()

        # Create design matrix with covariates
        X = data[covariates].values.astype(float)

        # Add unit and time dummies
        unit_dummies = pd.get_dummies(data[unit], prefix="u", drop_first=True)
        time_dummies = pd.get_dummies(data[time], prefix="t", drop_first=True)

        X_full = np.column_stack([np.ones(len(data)), X, unit_dummies.values, time_dummies.values])

        y = data[outcome].values.astype(float)

        # Fit and get residuals using unified backend
        coeffs, residuals, _ = solve_ols(
            X_full,
            y,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # Add back the mean for interpretability
        if survey_weights is not None:
            y_center = np.average(y, weights=survey_weights)
        else:
            y_center = np.mean(y)
        data[outcome] = residuals + y_center

        return data

    def _bootstrap_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        zeta_omega_n: float = 0.0,
        zeta_lambda_n: float = 0.0,
        min_decrease: float = 1e-5,
        *,
        w_control: Optional[np.ndarray] = None,
        w_treated: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[float, np.ndarray]:
        """Compute pairs-bootstrap standard error for SDID (Algorithm 2 step 2).

        Paper-faithful refit bootstrap: resamples all units (control + treated)
        with replacement, then re-estimates ω̂_b and λ̂_b via two-pass sparsified
        Frank-Wolfe on each resampled panel (Arkhangelsky et al. 2021
        Algorithm 2 step 2). Also matches R's default
        ``synthdid::vcov(method="bootstrap")`` behavior: R rebinds
        ``attr(estimate, "opts")`` (including ``update.omega=TRUE``) back into
        ``synthdid_estimate`` on each draw, so the renormalized ω serves only
        as Frank-Wolfe initialization.

        Survey-bootstrap (PR #352): if ``w_control``/``w_treated`` (pweight-
        only) or ``resolved_survey`` (full design with strata/PSU/FPC) is
        passed, switches to weighted-FW per draw. Per-draw rescaled weights
        come from either the constant ``w_control`` (pweight-only) or
        ``generate_rao_wu_weights(resolved_survey, rng)`` sliced over the
        resampled units (full design). The weighted-FW solves
        ``min Σ_t (Σ_i rw_i·ω_i·Y_i,pre[t] - b_t)² + ζ²·Σ rw_i·ω_i²``;
        downstream ``ω_eff = rw·ω / Σ(rw·ω)`` is composed before the SDID
        estimator (see REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap
        composition)`` for the argmin-set caveat).

        ``zeta_omega_n`` / ``zeta_lambda_n`` are the fit-time normalized-scale
        regularization parameters, used unchanged on each draw.

        ``unit_weights`` (fit-time ω) and ``time_weights`` (fit-time λ) are
        not reused as fixed estimator weights — every draw re-estimates ω̂_b
        and λ̂_b from scratch — but they ARE used as Frank-Wolfe warm-start
        initializations: ``_sum_normalize(unit_weights[boot_ctrl])`` seeds
        ω̂_b's first pass, ``time_weights`` seeds λ̂_b's. Matches R's
        ``vcov.R::bootstrap_sample`` shape.

        Retry-to-B: matches R's ``synthdid::bootstrap_sample`` while-loop;
        bounded attempt guard of ``20 * n_bootstrap``. Per-draw Frank-Wolfe
        non-convergence is tallied via the ``return_convergence`` flag from
        each helper and aggregated into one summary ``UserWarning`` if the
        rate exceeds 5% of valid draws.
        """
        rng = np.random.default_rng(self.seed)
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]
        n_total = n_control + n_treated

        # Survey-bootstrap dispatch (PR #352). _use_rao_wu fires for any
        # survey design (the helper itself handles strata=None / psu=None
        # by treating each obs as its own PSU); _pweight_only fires when
        # we have constant per-control survey weights but no
        # resolved_survey object (e.g., a future caller path).
        _use_rao_wu = resolved_survey is not None
        _pweight_only = (w_control is not None) and (resolved_survey is None)

        # Single-PSU short-circuit: unstratified design with <2 PSUs has no
        # identified bootstrap distribution (resampling one PSU yields the
        # same subset every draw). Returns NaN SE — same shape as PR #351's
        # n_successful=0 raise but caught upstream as NaN. Recovered from
        # 91082e5:diff_diff/synthetic_did.py.
        if _use_rao_wu and resolved_survey.psu is not None and resolved_survey.strata is None:
            from numpy import unique as _unique

            n_psu = len(_unique(resolved_survey.psu))
            if n_psu < 2:
                return np.nan, np.array([])

        # Build full panel matrix: (n_pre+n_post, n_control+n_treated)
        Y_full = np.block([[Y_pre_control, Y_pre_treated], [Y_post_control, Y_post_treated]])
        n_pre = Y_pre_control.shape[0]

        bootstrap_estimates: List[float] = []

        # Retry-until-B-valid semantic: matches R's synthdid::bootstrap_sample.
        max_attempts = 20 * self.n_bootstrap
        attempts = 0
        # Tally draws with any Frank-Wolfe non-convergence; aggregated into
        # one summary warning after the loop.
        fw_nonconvergence_count = 0

        while len(bootstrap_estimates) < self.n_bootstrap and attempts < max_attempts:
            attempts += 1
            boot_idx = rng.choice(n_total, size=n_total, replace=True)

            # Split resampled units into control vs treated
            boot_is_control = boot_idx < n_control
            n_co_b = int(boot_is_control.sum())

            # Retry if no control or no treated units in bootstrap sample
            if n_co_b == 0 or n_co_b == n_total:
                continue

            try:
                # Per-draw rescaled weights (PR #352 survey path). For
                # Rao-Wu, generate_rao_wu_weights returns per-unit rescaled
                # weights (resolved_survey is unit-collapsed upstream in
                # fit(); see synthetic_did.py callers). For pweight-only,
                # rw is the constant per-control survey weight.
                if _use_rao_wu:
                    boot_rw = generate_rao_wu_weights(resolved_survey, rng)
                    assert boot_rw is not None
                    rw_control_full = boot_rw[:n_control]
                    rw_treated_full = boot_rw[n_control:]
                    rw_control_draw = rw_control_full[boot_idx[boot_is_control]]
                    rw_treated_draw = rw_treated_full[boot_idx[~boot_is_control] - n_control]
                elif _pweight_only:
                    # pweight-only flag implies control weights exist.
                    assert w_control is not None
                    rw_control_draw = w_control[boot_idx[boot_is_control]]
                    rw_treated_draw = (
                        w_treated[boot_idx[~boot_is_control] - n_control]
                        if w_treated is not None
                        else None
                    )
                else:
                    rw_control_draw = None
                    rw_treated_draw = None

                # Degenerate-retry under ANY survey path: if a draw zeros
                # out the control or treated mass, retry. For Rao-Wu this
                # is expected behavior (PSUs not drawn get weight 0). For
                # pweight-only, zero-mass treated is reachable when at
                # least one unit has zero survey weight AND the
                # bootstrap resample happens to pick only zero-weight
                # treated units — silently falling back to an unweighted
                # mean would corrupt the bootstrap distribution because
                # fit-time ATT uses the survey-weighted mean (PR #355
                # R2 P0).
                if (
                    (_use_rao_wu or _pweight_only)
                    and rw_treated_draw is not None
                    and (rw_control_draw.sum() == 0 or rw_treated_draw.sum() == 0)
                ):
                    continue

                # Extract resampled outcome matrices
                Y_boot = Y_full[:, boot_idx]
                Y_boot_pre_c = Y_boot[:n_pre, boot_is_control]
                Y_boot_post_c = Y_boot[n_pre:, boot_is_control]
                Y_boot_pre_t = Y_boot[:n_pre, ~boot_is_control]
                Y_boot_post_t = Y_boot[n_pre:, ~boot_is_control]

                # Treated-unit mean: survey-weighted if rw_treated_draw is
                # set (PR #352), else unweighted.
                if rw_treated_draw is not None and rw_treated_draw.sum() > 0:
                    Y_boot_pre_t_mean = np.average(
                        Y_boot_pre_t,
                        axis=1,
                        weights=rw_treated_draw,
                    )
                    Y_boot_post_t_mean = np.average(
                        Y_boot_post_t,
                        axis=1,
                        weights=rw_treated_draw,
                    )
                else:
                    Y_boot_pre_t_mean = np.mean(Y_boot_pre_t, axis=1)
                    Y_boot_post_t_mean = np.mean(Y_boot_post_t, axis=1)

                # Warm-start init matching R's vcov.R::bootstrap_sample
                # shape. Under survey-bootstrap, scale the renormalized
                # init by rw before sum_normalize (matches the per-draw
                # weighted-FW geometry).
                boot_control_idx = boot_idx[boot_is_control]
                if rw_control_draw is not None:
                    boot_omega_init = _sum_normalize(
                        unit_weights[boot_control_idx] * rw_control_draw
                    )
                else:
                    boot_omega_init = _sum_normalize(unit_weights[boot_control_idx])
                boot_lambda_init = time_weights

                # Algorithm 2 step 2: re-estimate ω̂_b and λ̂_b via two-pass
                # sparsified Frank-Wolfe. Survey paths use the weighted-FW
                # helpers (PR #352 §1b); non-survey path unchanged.
                if rw_control_draw is not None:
                    boot_omega, omega_converged = compute_sdid_unit_weights_survey(
                        Y_boot_pre_c,
                        Y_boot_pre_t_mean,
                        rw_control_draw,
                        zeta_omega=zeta_omega_n,
                        min_decrease=min_decrease,
                        init_weights=boot_omega_init,
                        return_convergence=True,
                    )
                    boot_lambda, lambda_converged = compute_time_weights_survey(
                        Y_boot_pre_c,
                        Y_boot_post_c,
                        rw_control_draw,
                        zeta_lambda=zeta_lambda_n,
                        min_decrease=min_decrease,
                        init_weights=boot_lambda_init,
                        return_convergence=True,
                    )
                    # Compose ω_eff = rw·ω / Σ(rw·ω) for the SDID
                    # estimator (expects simplex weights). REGISTRY.md
                    # §SyntheticDiD covers the argmin-set caveat: the FW
                    # minimizes the weighted objective on ω, NOT the
                    # standard objective on ω_eff — intentional design
                    # choice validated by the stratified coverage MC.
                    omega_scaled = rw_control_draw * boot_omega
                    total = omega_scaled.sum()
                    if total <= 0:
                        # Degenerate: all mass on rw=0 controls
                        continue
                    boot_omega = omega_scaled / total
                else:
                    boot_omega, omega_converged = compute_sdid_unit_weights(
                        Y_boot_pre_c,
                        Y_boot_pre_t_mean,
                        zeta_omega=zeta_omega_n,
                        min_decrease=min_decrease,
                        init_weights=boot_omega_init,
                        return_convergence=True,
                    )
                    boot_lambda, lambda_converged = compute_time_weights(
                        Y_boot_pre_c,
                        Y_boot_post_c,
                        zeta_lambda=zeta_lambda_n,
                        min_decrease=min_decrease,
                        init_weights=boot_lambda_init,
                        return_convergence=True,
                    )

                tau = compute_sdid_estimator(
                    Y_boot_pre_c,
                    Y_boot_post_c,
                    Y_boot_pre_t_mean,
                    Y_boot_post_t_mean,
                    boot_omega,
                    boot_lambda,
                )
                if np.isfinite(tau):
                    bootstrap_estimates.append(float(tau))
                    # Count draws with ANY non-convergence (boolean per
                    # draw). Increment after the finite-τ gate so
                    # numerator and denominator (n_successful) match.
                    if not (omega_converged and lambda_converged):
                        fw_nonconvergence_count += 1

            except (ValueError, LinAlgError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Check bootstrap success rate and handle failures.
        # n_successful should equal self.n_bootstrap under the retry contract;
        # it only falls short if max_attempts was exhausted on pathological data.
        n_successful = len(bootstrap_estimates)
        failure_rate = 1 - (n_successful / self.n_bootstrap)

        if n_successful == 0:
            raise ValueError(
                f"Could not produce any valid bootstrap draw in {attempts} "
                f"attempts (target {self.n_bootstrap}). This typically occurs when:\n"
                f"  - Sample size is too small for reliable resampling\n"
                f"  - Weight matrices are singular or near-singular\n"
                f"  - Insufficient pre-treatment periods for weight estimation\n"
                f"  - Too few control units relative to treated units\n"
                f"Consider using variance_method='placebo' or increasing "
                f"the regularization parameters (zeta_omega, zeta_lambda)."
            )
        if n_successful == 1:
            warnings.warn(
                f"Only 1/{self.n_bootstrap} valid bootstrap draw accumulated in "
                f"{attempts} attempts. Standard error cannot be computed reliably "
                f"(requires at least 2). Returning SE=0.0. Consider using "
                f"variance_method='placebo' or increasing the regularization "
                f"(zeta_omega, zeta_lambda).",
                UserWarning,
                stacklevel=2,
            )
            return 0.0, bootstrap_estimates

        if failure_rate > 0.05:
            warnings.warn(
                f"Bootstrap attempt budget exhausted: only {n_successful}/"
                f"{self.n_bootstrap} valid draws accumulated in {attempts} "
                f"attempts. Standard errors may be unreliable; pathological "
                f"inputs can produce this signal (e.g., extreme treated/control "
                f"imbalance or singular weight matrices).",
                UserWarning,
                stacklevel=2,
            )

        # Aggregate Frank-Wolfe non-convergence across bootstrap draws. Per-draw
        # convergence warnings from compute_sdid_unit_weights / compute_time_weights
        # are suppressed inside the loop; emit one summary here if the rate
        # exceeds the same 5% threshold used for retry exhaustion.
        if fw_nonconvergence_count > 0.05 * max(n_successful, 1):
            warnings.warn(
                f"Frank-Wolfe did not converge on {fw_nonconvergence_count} of "
                f"{n_successful} valid bootstrap draws "
                f"(variance_method='bootstrap'). SE is still reported from "
                f"the final iterate of each draw, but non-convergent draws may be "
                f"noisier; consider relaxing min_decrease or increasing pre-period "
                f"length if regularization is already moderate.",
                UserWarning,
                stacklevel=2,
            )

        # SE formula matches R's synthdid::vcov(method="bootstrap"):
        # sqrt((r-1)/r) * sd(ddof=1), equivalent to Arkhangelsky et al. (2021)
        # Algorithm 2's σ̂² = (1/r) Σ (τ_b - τ̄)². Uses n_successful for
        # consistency with _placebo_variance_se.
        se = float(np.sqrt((n_successful - 1) / n_successful) * np.std(bootstrap_estimates, ddof=1))

        return se, bootstrap_estimates

    def _placebo_variance_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated_mean: np.ndarray,
        Y_post_treated_mean: np.ndarray,
        n_treated: int,
        zeta_omega: float = 0.0,
        zeta_lambda: float = 0.0,
        min_decrease: float = 1e-5,
        replications: int = 200,
        w_control=None,
        init_omega: Optional[np.ndarray] = None,
        init_lambda: Optional[np.ndarray] = None,
        _placebo_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute placebo-based variance matching R's synthdid methodology.

        This implements Algorithm 4 from Arkhangelsky et al. (2021),
        matching R's synthdid::vcov(method = "placebo"):

        1. Randomly sample N₀ control indices (permutation)
        2. Designate last N₁ as pseudo-treated, first (N₀-N₁) as pseudo-controls
        3. Re-estimate both omega and lambda on the permuted data with
           ``update.omega=TRUE, update.lambda=TRUE`` semantics. Per-draw FW
           is warm-started from the fit-time weights — ω is initialized with
           ``sum_normalize(init_omega[pseudo_control_idx])`` and λ is
           initialized with ``init_lambda`` — matching R's
           ``vcov.R::placebo_se`` ``weights.boot$omega = sum_normalize(
           weights$omega[ind[1:N0_placebo]])`` warm-start. The global FW
           optimum is init-independent (strict convexity), but the 100-iter
           pre-sparsify pass converges to different sparsification patterns
           under uniform vs warm init on a handful of draws — warm-start
           closes the resulting sub-percent SE drift against R.
        4. Compute SDID estimate with re-estimated weights
        5. Repeat `replications` times
        6. SE = sqrt((r-1)/r) * sd(estimates)

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated_mean : np.ndarray
            Mean treated outcomes in pre-treatment periods, shape (n_pre,).
        Y_post_treated_mean : np.ndarray
            Mean treated outcomes in post-treatment periods, shape (n_post,).
        n_treated : int
            Number of treated units in the original estimation.
        zeta_omega : float
            Regularization parameter for unit weights (for re-estimation).
        zeta_lambda : float
            Regularization parameter for time weights (for re-estimation).
        min_decrease : float
            Convergence threshold for Frank-Wolfe (for re-estimation).
        replications : int, default=200
            Number of placebo replications.
        init_omega : np.ndarray, optional
            Fit-time unit weights used to warm-start per-draw ω FW.
            Subset to pseudo-controls and renormalized inside the loop;
            mirrors R's ``weights.boot$omega = sum_normalize(weights$omega[
            ind[1:N0_placebo]])``. Cold-start (uniform init) when ``None``.
        init_lambda : np.ndarray, optional
            Fit-time time weights used to warm-start per-draw λ FW;
            mirrors R passing ``weights.boot$lambda = weights$lambda``
            through. Cold-start when ``None``.
        _placebo_indices : np.ndarray, optional
            Private R-parity test seam. When provided, each row of shape
            ``(replications, n_control)`` replaces the per-draw
            ``rng.permutation(n_control)`` so a Python fit can consume
            R's exact permutation sequence and produce a bit-identical SE
            (see ``test_placebo_se_matches_r``).

        Returns
        -------
        tuple
            (se, placebo_effects) where se is the standard error and
            placebo_effects is the array of placebo treatment effects.

        References
        ----------
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2021). Synthetic Difference-in-Differences. American Economic Review,
        111(12), 4088-4118. Algorithm 4.
        """
        rng = np.random.default_rng(self.seed)
        n_pre, n_control = Y_pre_control.shape

        # R-parity test seam (PR follow-up to #349). When
        # ``_placebo_indices`` is provided, each row replaces the
        # per-draw ``rng.permutation(n_control)`` so a Python fit can
        # consume R's exact permutation sequence and produce a
        # bit-identical SE. Shape: ``(replications, n_control)``,
        # 0-indexed. Underscore-prefixed → test-only / private API.
        if _placebo_indices is not None:
            _placebo_indices = np.asarray(_placebo_indices, dtype=np.int64)
            if _placebo_indices.ndim != 2 or _placebo_indices.shape[1] != n_control:
                raise ValueError(
                    f"_placebo_indices shape {_placebo_indices.shape} does not "
                    f"match expected (replications, {n_control})"
                )
            replications = _placebo_indices.shape[0]

        # Ensure we have enough controls for the split
        n_pseudo_control = n_control - n_treated
        if n_pseudo_control < 1:
            # Fallback guidance. All three variance methods support
            # pweight-only and full-design surveys (PR #355 and this PR).
            fallback = (
                "variance_method='bootstrap' or 'jackknife' (both support "
                "pweight-only and strata/PSU/FPC survey designs), or adding "
                "more control units"
                if w_control is not None
                else "variance_method='bootstrap', variance_method='jackknife', "
                "or adding more control units"
            )
            warnings.warn(
                f"Not enough control units ({n_control}) for placebo variance "
                f"estimation with {n_treated} treated units. "
                f"Consider using {fallback}.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, np.array([])

        placebo_estimates = []

        for _rep in range(replications):
            try:
                # Random permutation of control indices (Algorithm 4, step 1).
                # Test seam: when ``_placebo_indices`` is supplied, consume
                # the externally-provided permutation instead — used by
                # ``test_placebo_se_matches_r`` to feed R's exact
                # permutation sequence through Python for bit-identical SE.
                if _placebo_indices is not None:
                    perm = _placebo_indices[_rep]
                else:
                    perm = rng.permutation(n_control)

                # Split into pseudo-controls and pseudo-treated (step 2)
                pseudo_control_idx = perm[:n_pseudo_control]
                pseudo_treated_idx = perm[n_pseudo_control:]

                # Get pseudo-control and pseudo-treated outcomes
                Y_pre_pseudo_control = Y_pre_control[:, pseudo_control_idx]
                Y_post_pseudo_control = Y_post_control[:, pseudo_control_idx]

                # Pseudo-treated means: survey-weighted when available
                if w_control is not None:
                    pseudo_w_tr = w_control[pseudo_treated_idx]
                    Y_pre_pseudo_treated_mean = np.average(
                        Y_pre_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                    Y_post_pseudo_treated_mean = np.average(
                        Y_post_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                else:
                    Y_pre_pseudo_treated_mean = np.mean(
                        Y_pre_control[:, pseudo_treated_idx], axis=1
                    )
                    Y_post_pseudo_treated_mean = np.mean(
                        Y_post_control[:, pseudo_treated_idx], axis=1
                    )

                # Re-estimate weights on permuted data (matching R's behavior).
                # R's `placebo_se` (synthdid:::placebo_se in R/vcov.R) passes
                # `weights.boot$omega = sum_normalize(weights$omega[ind[1:N0]])`
                # as a warm-start to `synthdid_estimate` with `update.omega=TRUE`,
                # then re-estimates ω. Python mirrors that: when fit-time ω
                # (`init_omega`) is supplied, seed the FW first pass with the
                # subsetted-renormalized fit-time omega; otherwise use cold-
                # start (uniform). At the global FW optimum the two are
                # equivalent, but the warm-start matches R's exact iterates
                # for bit-identical SE under the R-parity test.
                if init_omega is not None:
                    pseudo_omega_init = _sum_normalize(init_omega[pseudo_control_idx])
                else:
                    pseudo_omega_init = None
                pseudo_omega = compute_sdid_unit_weights(
                    Y_pre_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    zeta_omega=zeta_omega,
                    min_decrease=min_decrease,
                    init_weights=pseudo_omega_init,
                )

                # Compose pseudo_omega with control survey weights
                if w_control is not None:
                    pseudo_w_co = w_control[pseudo_control_idx]
                    pseudo_omega_eff = pseudo_omega * pseudo_w_co
                    pseudo_omega_eff = pseudo_omega_eff / pseudo_omega_eff.sum()
                else:
                    pseudo_omega_eff = pseudo_omega

                # Time weights: re-estimate on pseudo-control data.
                # R's `placebo_se` reuses the same `weights.boot` (with
                # the original ``weights$lambda`` untouched) as warm-
                # start to ``synthdid_estimate``. Mirror by passing
                # fit-time λ as ``init_weights`` when supplied.
                pseudo_lambda = compute_time_weights(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    zeta_lambda=zeta_lambda,
                    min_decrease=min_decrease,
                    init_weights=init_lambda,
                )

                # Compute placebo SDID estimate (step 4)
                tau = compute_sdid_estimator(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    Y_post_pseudo_treated_mean,
                    pseudo_omega_eff,
                    pseudo_lambda,
                )
                if np.isfinite(tau):
                    placebo_estimates.append(tau)

            except (ValueError, LinAlgError, ZeroDivisionError):
                # Skip failed iterations
                continue

        placebo_estimates = np.array(placebo_estimates)
        n_successful = len(placebo_estimates)

        if n_successful < 2:
            # Same fallback guidance as the pre-replication guard above.
            # Bootstrap and jackknife both support pweight-only + full
            # strata/PSU/FPC survey designs, so either is a valid
            # fallback for survey users (though jackknife is anti-
            # conservative with few PSUs per stratum — see REGISTRY).
            fallback = (
                "variance_method='bootstrap' or 'jackknife' (both support "
                "pweight-only and strata/PSU/FPC survey designs), or "
                "increasing the number of control units"
                if w_control is not None
                else "variance_method='bootstrap' or variance_method='jackknife' "
                "or increasing the number of control units"
            )
            warnings.warn(
                f"Only {n_successful} placebo replications completed successfully. "
                f"Standard error cannot be estimated reliably. "
                f"Consider using {fallback}.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, placebo_estimates

        # Warn if many replications failed
        failure_rate = 1 - (n_successful / replications)
        if failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{replications} placebo replications succeeded "
                f"({failure_rate:.1%} failure rate). Standard errors may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        # Compute SE using R's formula: sqrt((r-1)/r) * sd(estimates)
        # This matches synthdid::vcov.R exactly
        se = np.sqrt((n_successful - 1) / n_successful) * np.std(placebo_estimates, ddof=1)

        return se, placebo_estimates

    def _placebo_variance_se_survey(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated_mean: np.ndarray,
        Y_post_treated_mean: np.ndarray,
        strata_control: np.ndarray,
        treated_strata: np.ndarray,
        zeta_omega: float = 0.0,
        zeta_lambda: float = 0.0,
        min_decrease: float = 1e-5,
        replications: int = 200,
        w_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Stratified-permutation placebo variance for survey designs.

        Extends Algorithm 4 of Arkhangelsky et al. (2021) to strata/PSU/FPC
        designs by restricting pseudo-treated sampling to controls in the
        same stratum as actual treated units (Pesarin 2001 stratified
        permutation test). Weighted Frank-Wolfe re-estimates ω and λ per
        draw on the pseudo-panel with per-control survey weights flowing
        into both the loss and the regularizer.

        The PSU axis is intentionally not randomized — within-stratum unit-
        level permutation is the classical stratified permutation test
        (Pesarin 2001 Ch. 3-4). PSU-level permutation on few PSUs (2-8
        typical) produces near-degenerate permutation support and poor
        power. Asymmetry with the jackknife allocator (which respects PSU)
        is by design; see REGISTRY.md §SyntheticDiD "Allocator asymmetry".

        Parameters
        ----------
        Y_pre_control, Y_post_control : np.ndarray
            Control outcomes, shapes (n_pre, n_control) / (n_post, n_control).
        Y_pre_treated_mean, Y_post_treated_mean : np.ndarray
            Survey-weighted treated means, shapes (n_pre,) / (n_post,).
        strata_control : np.ndarray
            Per-control stratum labels (already resolved via
            ``collapse_survey_to_unit_level``), shape (n_control,).
        treated_strata : np.ndarray
            Per-treated-unit stratum labels, shape (n_treated,).
        zeta_omega, zeta_lambda, min_decrease : float
            Weighted-FW hyperparameters (already normalized by Y_scale).
        replications : int, default 200
            Number of placebo draws.
        w_control : np.ndarray
            Per-control survey weights, shape (n_control,). Required for
            survey path (passed through from fit-time resolved weights).

        Returns
        -------
        tuple
            ``(se, placebo_effects)`` where ``se = sqrt((r-1)/r) * std(...)``
            (Algorithm 4 SE formula) and ``placebo_effects`` is the array
            of successful pseudo-τ̂ values.

        References
        ----------
        Arkhangelsky et al. (2021), *American Economic Review*, Algorithm 4.
        Pesarin (2001), *Multivariate Permutation Tests*, Ch. 3-4.
        Pesarin & Salmaso (2010), *Permutation Tests for Complex Data*.
        """
        rng = np.random.default_rng(self.seed)

        # Build per-stratum control index map (strata containing treated units)
        unique_treated_strata, treated_counts_per_stratum = np.unique(
            treated_strata, return_counts=True
        )
        control_idx_per_stratum: Dict[Any, np.ndarray] = {}
        for h in unique_treated_strata:
            control_idx_per_stratum[h] = np.where(strata_control == h)[0]

        placebo_estimates = []

        for _ in range(replications):
            try:
                pseudo_treated_parts = []
                for h, n_treated_h in zip(unique_treated_strata, treated_counts_per_stratum):
                    controls_in_h = control_idx_per_stratum[h]
                    pseudo_treated_h = rng.choice(
                        controls_in_h, size=int(n_treated_h), replace=False
                    )
                    pseudo_treated_parts.append(pseudo_treated_h)
                pseudo_treated_idx = np.concatenate(pseudo_treated_parts)

                # Pseudo-control = all controls \ pseudo-treated. Keep the
                # non-treated-stratum controls AND the unsampled controls
                # within each treated stratum.
                sampled_set = set(pseudo_treated_idx.tolist())
                pseudo_control_mask = np.array(
                    [i not in sampled_set for i in range(len(strata_control))]
                )
                pseudo_control_idx = np.where(pseudo_control_mask)[0]

                # Pseudo-panel
                # This survey pseudo-panel branch requires control weights.
                assert w_control is not None
                Y_pre_pseudo_control = Y_pre_control[:, pseudo_control_idx]
                Y_post_pseudo_control = Y_post_control[:, pseudo_control_idx]
                pseudo_w_tr = w_control[pseudo_treated_idx]
                pseudo_w_co = w_control[pseudo_control_idx]

                # Pseudo-treated means (survey-weighted)
                Y_pre_pseudo_treated_mean = np.average(
                    Y_pre_control[:, pseudo_treated_idx],
                    axis=1,
                    weights=pseudo_w_tr,
                )
                Y_post_pseudo_treated_mean = np.average(
                    Y_post_control[:, pseudo_treated_idx],
                    axis=1,
                    weights=pseudo_w_tr,
                )

                # Weighted FW for unit weights
                pseudo_omega = compute_sdid_unit_weights_survey(
                    Y_pre_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    rw_control=pseudo_w_co,
                    zeta_omega=zeta_omega,
                    min_decrease=min_decrease,
                )

                # Compose ω_eff = rw · ω / Σ(rw · ω). Zero-mass guard:
                # degenerate draw where FW sparsified onto zero-survey-
                # weight controls; retry (same convention as bootstrap
                # PR #355 R12 P1).
                omega_scaled = pseudo_w_co * pseudo_omega
                total = omega_scaled.sum()
                if total <= 0:
                    continue
                omega_eff = omega_scaled / total

                # Weighted FW for time weights
                pseudo_lambda = compute_time_weights_survey(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    rw_control=pseudo_w_co,
                    zeta_lambda=zeta_lambda,
                    min_decrease=min_decrease,
                )

                tau = compute_sdid_estimator(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    Y_post_pseudo_treated_mean,
                    omega_eff,
                    pseudo_lambda,
                )
                if np.isfinite(tau):
                    placebo_estimates.append(float(tau))

            except (ValueError, LinAlgError, ZeroDivisionError):
                continue

        placebo_estimates_arr = np.array(placebo_estimates)
        n_successful = len(placebo_estimates_arr)

        if n_successful < 2:
            warnings.warn(
                f"Only {n_successful} placebo replications completed successfully "
                f"on the survey path. Standard error cannot be estimated reliably. "
                "Consider variance_method='bootstrap' (supports the same full "
                "design via weighted-FW + Rao-Wu) or rebalance the panel.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, placebo_estimates_arr

        failure_rate = 1 - (n_successful / replications)
        if failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{replications} stratified-permutation "
                f"placebo replications succeeded ({failure_rate:.1%} failure "
                "rate). Standard errors may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        se = np.sqrt((n_successful - 1) / n_successful) * np.std(placebo_estimates_arr, ddof=1)
        return se, placebo_estimates_arr

    def _jackknife_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        w_treated=None,
        w_control=None,
    ) -> Tuple[float, np.ndarray]:
        """Compute jackknife standard error matching R's synthdid Algorithm 3.

        Delete-1 jackknife over all units (control + treated) with **fixed**
        weights.  For each leave-one-out sample the original omega is subsetted
        and renormalized; lambda stays unchanged.  No Frank-Wolfe
        re-estimation, making this the fastest variance method.

        This matches R's ``synthdid::vcov(method="jackknife")`` which sets
        ``update.omega=FALSE, update.lambda=FALSE``.

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated : np.ndarray
            Treated outcomes in pre-treatment periods, shape (n_pre, n_treated).
        Y_post_treated : np.ndarray
            Treated outcomes in post-treatment periods, shape (n_post, n_treated).
        unit_weights : np.ndarray
            Unit weights from Frank-Wolfe optimization, shape (n_control,).
        time_weights : np.ndarray
            Time weights from Frank-Wolfe optimization, shape (n_pre,).
        w_treated : np.ndarray, optional
            Survey probability weights for treated units.
        w_control : np.ndarray, optional
            Survey probability weights for control units.

        Returns
        -------
        tuple
            (se, jackknife_estimates) where se is the standard error and
            jackknife_estimates is a length-N array of leave-one-out estimates
            (first n_control entries are control-LOO, last n_treated are
            treated-LOO).

        References
        ----------
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2021). Synthetic Difference-in-Differences. American Economic Review,
        111(12), 4088-4118. Algorithm 3.
        """
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]
        n = n_control + n_treated

        # --- Early-return NaN: matches R's NA conditions ---
        if n_treated <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 treated unit. "
                "Use variance_method='placebo' for single treated unit.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        if np.sum(unit_weights > 0) <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 control unit with "
                "nonzero weight. Consider variance_method='placebo'.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        # --- Effective-support guards for survey-weighted path ---
        if w_control is not None:
            effective_control = unit_weights * w_control
            if np.sum(effective_control > 0) <= 1:
                warnings.warn(
                    "Jackknife variance requires more than 1 control unit with "
                    "positive effective weight (omega * survey_weight). "
                    "Consider variance_method='placebo'.",
                    UserWarning,
                    stacklevel=3,
                )
                return np.nan, np.array([])

        if w_treated is not None and np.sum(w_treated > 0) <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 treated unit with "
                "positive survey weight. "
                "Consider variance_method='placebo'.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        jackknife_estimates = np.empty(n)

        # --- Precompute treated means (constant across control-LOO) ---
        if w_treated is not None:
            treated_pre_mean = np.average(Y_pre_treated, axis=1, weights=w_treated)
            treated_post_mean = np.average(Y_post_treated, axis=1, weights=w_treated)
        else:
            treated_pre_mean = np.mean(Y_pre_treated, axis=1)
            treated_post_mean = np.mean(Y_post_treated, axis=1)

        # --- Precompute omega composed with survey weights (for treated-LOO) ---
        if w_control is not None:
            omega_eff_full = unit_weights * w_control
            omega_eff_full = omega_eff_full / omega_eff_full.sum()
        else:
            omega_eff_full = unit_weights

        # --- Leave-one-out over control units ---
        mask = np.ones(n_control, dtype=bool)
        for j in range(n_control):
            mask[j] = False

            # Subset and renormalize omega
            omega_jk = _sum_normalize(unit_weights[mask])

            # Compose with survey weights if present
            if w_control is not None:
                omega_jk = omega_jk * w_control[mask]
                if omega_jk.sum() == 0:
                    jackknife_estimates[j] = np.nan
                    mask[j] = True
                    continue
                omega_jk = omega_jk / omega_jk.sum()

            jackknife_estimates[j] = compute_sdid_estimator(
                Y_pre_control[:, mask],
                Y_post_control[:, mask],
                treated_pre_mean,
                treated_post_mean,
                omega_jk,
                time_weights,
            )

            mask[j] = True  # restore for next iteration

        # --- Leave-one-out over treated units ---
        mask = np.ones(n_treated, dtype=bool)
        for k in range(n_treated):
            mask[k] = False

            # Recompute treated means from remaining units
            if w_treated is not None:
                w_t_jk = w_treated[mask]
                if w_t_jk.sum() == 0:
                    jackknife_estimates[n_control + k] = np.nan
                    mask[k] = True
                    continue
                t_pre_mean = np.average(Y_pre_treated[:, mask], axis=1, weights=w_t_jk)
                t_post_mean = np.average(Y_post_treated[:, mask], axis=1, weights=w_t_jk)
            else:
                t_pre_mean = np.mean(Y_pre_treated[:, mask], axis=1)
                t_post_mean = np.mean(Y_post_treated[:, mask], axis=1)

            jackknife_estimates[n_control + k] = compute_sdid_estimator(
                Y_pre_control,
                Y_post_control,
                t_pre_mean,
                t_post_mean,
                omega_eff_full,
                time_weights,
            )

            mask[k] = True  # restore for next iteration

        # --- Check for non-finite estimates (propagate NaN like R's var()) ---
        if not np.all(np.isfinite(jackknife_estimates)):
            warnings.warn(
                "Some jackknife leave-one-out estimates are non-finite. "
                "Standard error cannot be computed.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, jackknife_estimates

        # --- Jackknife SE formula: sqrt((n-1)/n * sum((u - ubar)^2)) ---
        # Matches R's: sqrt(((n-1)/n) * (n-1) * var(u))
        u_bar = np.mean(jackknife_estimates)
        ss = np.sum((jackknife_estimates - u_bar) ** 2)
        se = np.sqrt((n - 1) / n * ss)

        return se, jackknife_estimates

    def _jackknife_se_survey(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        w_control: np.ndarray,
        w_treated: np.ndarray,
        strata_control: np.ndarray,
        strata_treated: np.ndarray,
        psu_control: Optional[np.ndarray],
        psu_treated: Optional[np.ndarray],
        fpc_control: Optional[np.ndarray],
        fpc_treated: Optional[np.ndarray],
        lonely_psu: str = "remove",
    ) -> Tuple[float, np.ndarray]:
        """PSU-level leave-one-out jackknife with stratum aggregation.

        Extends Algorithm 3 of Arkhangelsky et al. (2021) to survey designs
        via the stratified PSU-level jackknife variance estimator (Rust &
        Rao 1996)::

            V_J = Σ_h (1 - f_h) · (n_h - 1)/n_h · Σ_{j∈h} (τ̂_{(h,j)} - τ̄_h)²

        where ``n_h`` is the number of PSUs in stratum h, ``f_h = n_h/N_h``
        is the sampling fraction (0 if no FPC), and ``τ̄_h`` is the
        stratum-level mean of LOO estimates.

        Semantics:
        * **λ fixed** (not re-estimated per LOO) — matches non-survey
          Algorithm 3. Jackknife is variance-approximation; re-estimating
          λ per LOO conflates weight-estimation uncertainty (bootstrap's
          domain) with sampling uncertainty.
        * **ω subset + rw-composed-renormalize** (not re-estimated) — same
          rationale. Control units inside the dropped PSU are removed;
          remaining ω is composed with remaining survey weights and
          renormalized.
        * **Strata with n_h < 2 are silently skipped** (lonely-PSU case,
          matches R ``survey::svyjkn``). They contribute 0 to the total
          variance. If every stratum is skipped, ``SE=NaN`` with a
          ``UserWarning``.
        * **Undefined LOOs within a contributing stratum → SE=NaN.** The
          Rust & Rao formula requires every PSU-LOO in a contributing
          stratum (``n_h ≥ 2``) to produce a defined ``τ̂_{(h,j)}``. If
          any single LOO is undefined — (a) deletion removes all treated
          units, (b) kept ``ω_eff`` mass is zero, (c) kept treated
          survey mass is zero, (d) the SDID estimator raises or returns
          non-finite τ̂ — the overall SE is undefined and the method
          returns ``NaN`` with a targeted ``UserWarning`` naming the
          stratum / PSU / reason. Silently skipping the missing LOO
          while still applying the full ``(n_h-1)/n_h`` factor would
          systematically under-scale variance (silently wrong SE).

        PSU-None fallback: if ``psu_control is None``, each unit is treated
        as its own PSU within its stratum (matches PR #355 R8 P1
        implicit-PSU Rao-Wu semantics).

        Parameters
        ----------
        Y_pre_control, Y_post_control : np.ndarray
            Control outcomes.
        Y_pre_treated, Y_post_treated : np.ndarray
            Treated outcomes (raw per-unit, not averaged).
        unit_weights, time_weights : np.ndarray
            Fit-time ω, λ (kept fixed across LOOs).
        w_control, w_treated : np.ndarray
            Per-unit survey weights.
        strata_control, strata_treated : np.ndarray
            Per-unit stratum labels.
        psu_control, psu_treated : np.ndarray, optional
            Per-unit PSU labels. ``None`` → each unit is its own PSU.
        fpc_control, fpc_treated : np.ndarray, optional
            Per-unit FPC values (stratum-constant population counts from
            ``survey.py::SurveyDesign.resolve``, validated constant within
            each stratum).

        Returns
        -------
        tuple
            ``(se, tau_loo_all)`` where ``se`` is the stratum-aggregated
            jackknife SE (NaN if every stratum was skipped) and
            ``tau_loo_all`` is the flat array of successful LOO estimates
            (not grouped per stratum).

        References
        ----------
        Arkhangelsky et al. (2021), *American Economic Review*, Algorithm 3.
        Rust & Rao (1996), *Statistical Methods in Medical Research*, 5(3),
        283-310, "Variance Estimation for Complex Surveys Using Replication
        Techniques".
        """
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]

        # Build unit-level (stratum, psu, fpc, is_control, local_idx) index.
        # ``local_idx`` is the position in its arm (control_idx in [0,n_c)
        # or treated_idx in [0,n_t)). We loop over (stratum, psu) groups.
        if psu_control is None:
            # Each control unit is its own PSU — use the control's own index.
            psu_control_eff = np.arange(n_control, dtype=np.int64)
        else:
            psu_control_eff = np.asarray(psu_control)
        if psu_treated is None:
            psu_treated_eff = np.arange(n_control, n_control + n_treated, dtype=np.int64)
        else:
            psu_treated_eff = np.asarray(psu_treated)

        # Per-stratum PSU enumeration. PSU labels are globally unique
        # within strata by ``SurveyDesign.resolve`` (see survey.py
        # L308-L320 ``nest=False`` validation), so a (stratum, psu) pair
        # uniquely identifies a PSU.
        unique_strata_all = np.unique(np.concatenate([strata_control, strata_treated]))

        # Short-circuit: unstratified single-PSU design. ``strata_*`` arrays
        # are always populated after ``_resolve_survey_for_fit``, so a
        # single-stratum + single-PSU design is detectable as one unique
        # PSU across both arms.
        all_psus = np.concatenate([psu_control_eff, psu_treated_eff])
        if len(unique_strata_all) == 1 and len(np.unique(all_psus)) < 2:
            return np.nan, np.array([])

        # Precompute fixed-ω composition for the FULL sample (for LOOs that
        # drop only treated PSUs — control ω/w_control unchanged).
        omega_eff_full = unit_weights * w_control
        if omega_eff_full.sum() <= 0:
            # Fit-time guard should have caught this, but double-check for
            # defense-in-depth.
            warnings.warn(
                "Jackknife survey SE cannot be computed: the effective "
                "control omega mass (ω · w_control) sums to zero.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])
        omega_eff_full = omega_eff_full / omega_eff_full.sum()

        total_variance = 0.0
        tau_loo_all: List[float] = []
        any_stratum_contributed = False
        # Undefined-replicate tracking (PR #365 R1 P0 fix). The Rust & Rao
        # (1996) formula assumes every sampled PSU within a contributing
        # stratum has a defined delete-one replicate `τ̂_{(h,j)}`. If any
        # LOO within a contributing stratum (n_h ≥ 2) is undefined — e.g.,
        # all treated units are in that PSU, or the kept ω_eff mass is
        # zero, or the SDID estimator raises — the stratified SE formula
        # does not apply and the overall SE is undefined. Return NaN
        # rather than silently skipping the missing replicate while still
        # applying the full (n_h-1)/n_h factor (which would underscale).
        undefined_replicate_stratum: Optional[Any] = None
        undefined_replicate_psu: Optional[Any] = None
        undefined_replicate_reason: str = ""

        for h in unique_strata_all:
            # PSUs in stratum h (across both arms)
            control_in_h_mask = strata_control == h
            treated_in_h_mask = strata_treated == h
            psus_in_h_control = psu_control_eff[control_in_h_mask]
            psus_in_h_treated = psu_treated_eff[treated_in_h_mask]
            psus_in_h = np.unique(np.concatenate([psus_in_h_control, psus_in_h_treated]))
            n_h = len(psus_in_h)
            if n_h < 2:
                # Singleton-stratum handling. R12 P1 fix: distinguish
                # ``"certainty"`` from ``"remove"`` semantics. Both end
                # up adding zero variance for this stratum, but
                # ``"certainty"`` is an *explicit* zero-variance
                # contributor (the stratum is sampled with certainty,
                # so no sampling variance — this is a documented
                # legitimate zero, not a "skipped/undefined" case).
                # Mark as contributing so the all-singleton design
                # under ``"certainty"`` returns ``SE = 0.0`` instead
                # of falling through to the "every stratum was
                # skipped → NaN" branch (matches `survey.py`'s
                # ``test_all_certainty_psu_zero_vcov`` contract).
                # ``"remove"`` continues to silently skip — matches R
                # ``survey::svyjkn`` lonely-PSU="remove".
                if lonely_psu == "certainty":
                    any_stratum_contributed = True
                continue

            # Per-stratum FPC. ``fpc_*`` arrays are stratum-constant by
            # SurveyDesign.resolve (survey.py L343-L347). Read from either
            # arm; prefer control if any controls in the stratum.
            if fpc_control is not None and control_in_h_mask.any():
                fpc_h = float(fpc_control[control_in_h_mask][0])
                f_h = n_h / fpc_h if fpc_h > 0 else 0.0
            elif fpc_treated is not None and treated_in_h_mask.any():
                fpc_h = float(fpc_treated[treated_in_h_mask][0])
                f_h = n_h / fpc_h if fpc_h > 0 else 0.0
            else:
                f_h = 0.0

            # R6 P1 fix: full-census short-circuit. When f_h >= 1 the
            # Rust & Rao factor ``(1 - f_h) <= 0`` zeros this stratum's
            # contribution to total variance regardless of within-
            # stratum dispersion. Skip the delete-one feasibility loop
            # entirely — otherwise an undefined LOO inside a full-
            # census stratum (e.g., all treated in the dropped PSU)
            # would mistakenly short-circuit the whole design to
            # ``SE=NaN``, even though the stratum contributes zero by
            # legitimate design. Mark as contributing (so the overall
            # result returns ``SE=0`` or a finite non-zero from other
            # strata, not ``NaN`` from the "no stratum contributed"
            # branch).
            if f_h >= 1.0:
                any_stratum_contributed = True
                continue

            tau_loo_h: List[float] = []
            stratum_has_undefined_replicate = False
            for j in psus_in_h:
                # Mask: kept units across both arms
                control_kept_mask = psu_control_eff != j
                treated_kept_mask = psu_treated_eff != j

                # If this PSU contains no units in either arm, it cannot
                # produce a meaningful LOO (and should not have been
                # enumerated); treat as undefined for defensive consistency.
                if control_kept_mask.all() and treated_kept_mask.all():
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = "PSU contains no units in either arm"
                    break

                # All treated removed → LOO yields an undefined SDID
                # estimator (no treated mean to compare). The Rust & Rao
                # formula expects τ̂_{(h,j)} defined for every j; skipping
                # this PSU while keeping the (n_h-1)/n_h factor would
                # underscale variance (R1 P0).
                if not treated_kept_mask.any():
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = (
                        "deletion removes all treated units (no treated "
                        "mean for the LOO SDID estimator)"
                    )
                    break

                # Control ω composition on kept controls
                omega_kept = unit_weights[control_kept_mask]
                w_control_kept = w_control[control_kept_mask]
                omega_eff_kept = omega_kept * w_control_kept
                if omega_eff_kept.sum() <= 0:
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = (
                        "kept omega_eff mass is zero (all remaining "
                        "controls have zero fit-time or survey weight)"
                    )
                    break
                omega_eff_kept = omega_eff_kept / omega_eff_kept.sum()

                # Treated mean on kept treated units (survey-weighted)
                w_treated_kept = w_treated[treated_kept_mask]
                if w_treated_kept.sum() <= 0:
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = "kept treated survey mass is zero"
                    break
                Y_pre_t_mean = np.average(
                    Y_pre_treated[:, treated_kept_mask],
                    axis=1,
                    weights=w_treated_kept,
                )
                Y_post_t_mean = np.average(
                    Y_post_treated[:, treated_kept_mask],
                    axis=1,
                    weights=w_treated_kept,
                )

                try:
                    tau_j = compute_sdid_estimator(
                        Y_pre_control[:, control_kept_mask],
                        Y_post_control[:, control_kept_mask],
                        Y_pre_t_mean,
                        Y_post_t_mean,
                        omega_eff_kept,
                        time_weights,
                    )
                except (ValueError, LinAlgError, ZeroDivisionError):
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = "SDID estimator raised on the LOO panel"
                    break

                if not np.isfinite(tau_j):
                    stratum_has_undefined_replicate = True
                    undefined_replicate_stratum = h
                    undefined_replicate_psu = j
                    undefined_replicate_reason = "SDID estimator returned non-finite τ̂"
                    break
                tau_loo_h.append(float(tau_j))

            if stratum_has_undefined_replicate:
                # Record the partial LOOs for the returned array (useful
                # for debugging) but stop accumulating variance — the
                # stratified Rust & Rao formula requires all n_h
                # replicates.
                tau_loo_all.extend(tau_loo_h)
                break

            if len(tau_loo_h) == n_h:
                tau_bar_h = np.mean(tau_loo_h)
                ss_h = float(np.sum((np.asarray(tau_loo_h) - tau_bar_h) ** 2))
                total_variance += (1.0 - f_h) * (n_h - 1) / n_h * ss_h
                any_stratum_contributed = True
            tau_loo_all.extend(tau_loo_h)

        tau_loo_arr = np.asarray(tau_loo_all)
        if undefined_replicate_stratum is not None:
            # R1 P0 fix: Rust & Rao's stratified jackknife formula requires
            # every LOO within a contributing stratum to be defined. When
            # one is missing, the design is not covered by the formula and
            # the SE is undefined; returning a finite value on the
            # remaining replicates (still multiplied by (n_h-1)/n_h) would
            # systematically under-scale variance.
            warnings.warn(
                "Jackknife survey SE is undefined: delete-one replicate "
                f"for stratum {undefined_replicate_stratum} PSU "
                f"{undefined_replicate_psu} is not computable "
                f"({undefined_replicate_reason}). The stratified Rust & "
                "Rao (1996) jackknife formula requires τ̂_{(h,j)} defined "
                "for every j in every contributing stratum. Returning "
                "SE=NaN. Consider variance_method='bootstrap' (supports "
                "the same full design without a per-LOO feasibility "
                "constraint) or rebalance the panel so every PSU has at "
                "least one treated and one non-zero-mass control unit "
                "after deletion.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, tau_loo_arr
        if not any_stratum_contributed:
            warnings.warn(
                "Jackknife survey SE is undefined because every stratum "
                "was skipped (insufficient PSUs per stratum for variance "
                "contribution). Returning SE=NaN. Consider "
                "variance_method='bootstrap' (supports the same full "
                "design) or rebalance the panel.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, tau_loo_arr

        # R5 P1 fix: legitimate zero variance (e.g., full-census FPC with
        # f_h = 1 for every contributing stratum → (1 - f_h) = 0 factor
        # zeros the contribution even when within-stratum dispersion is
        # non-zero; or exact-zero within-stratum dispersion when all
        # LOOs produce identical τ̂). Rust & Rao gives V_J = 0, not
        # undefined. Reserve NaN for the "all strata skipped" /
        # undefined-replicate cases above; compute SE = 0 otherwise.
        variance_nonneg = max(total_variance, 0.0)
        return float(np.sqrt(variance_nonneg)), tau_loo_arr

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "zeta_omega": self.zeta_omega,
            "zeta_lambda": self.zeta_lambda,
            "alpha": self.alpha,
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            # Conley kwargs are inherited from DifferenceInDifferences.__init__
            # but rejected by SyntheticDiD's __init__ / set_params (Conley uses
            # the analytical sandwich, SyntheticDiD uses bootstrap variance).
            # Surface them here as None for sklearn-style API consistency; any
            # non-None value is rejected by set_params/__init__.
            "vcov_type": None,
            "conley_coords": None,
            "conley_cutoff_km": None,
            "conley_metric": None,
            "conley_kernel": None,
            "conley_lag_cutoff": None,
        }

    def set_params(self, **params) -> "SyntheticDiD":
        """Set estimator parameters.

        Applies updates transactionally: if ``_validate_config()`` rejects the
        post-update state, the instance is rolled back to the pre-call values
        so a raised ``ValueError`` leaves the object consistent with its
        pre-call configuration.

        Mirrors ``__init__``'s defensive rejection of ``vcov_type`` /
        ``conley_*`` non-None values: SyntheticDiD uses bootstrap/jackknife/
        placebo variance, not the analytical sandwich, so any Conley kwarg
        would be silently ignored otherwise (forbidden by
        ``feedback_no_silent_failures``). Tracked in TODO.md for a follow-up
        that wires Conley to a non-bootstrap variance path.
        """
        # Reject Conley kwargs / non-None vcov_type before any mutation —
        # mirrors __init__'s contract. Empty/None values are permitted so
        # round-tripping get_params() back through set_params() is a no-op.
        _conley_keys = (
            "conley_coords",
            "conley_cutoff_km",
            "conley_metric",
            "conley_kernel",
            "conley_lag_cutoff",
        )
        if params.get("vcov_type") is not None and params["vcov_type"] != "conley":
            raise TypeError(
                f"SyntheticDiD does not accept vcov_type={params['vcov_type']!r}. "
                "SyntheticDiD's variance is bootstrap/jackknife/placebo based; "
                "configure via variance_method=..."
            )
        if params.get("vcov_type") == "conley" or any(
            k in params and params[k] is not None for k in _conley_keys
        ):
            raise TypeError(
                "SyntheticDiD does not yet support vcov_type='conley' or any "
                "conley_* kwargs. SyntheticDiD uses bootstrap/jackknife/placebo "
                "variance (variance_method=...), not the analytical sandwich "
                "routed through compute_robust_vcov. Tracked in TODO.md as "
                "a follow-up."
            )
        # Deprecated parameter names — emit warning and ignore
        _deprecated = {"lambda_reg", "zeta"}
        # Conley kwargs are not stored as instance attributes; surfacing them
        # in get_params() returns None unconditionally. set_params() with None
        # values for these keys is a no-op (the rejection above only fires on
        # non-None values).
        _silent_conley_passthrough = {"vcov_type", *_conley_keys}
        # Snapshot original values for transactional rollback on validation failure.
        _rollback: Dict[str, Any] = {}
        for key in params:
            if key in _silent_conley_passthrough:
                continue
            if key not in _deprecated and hasattr(self, key):
                _rollback[key] = getattr(self, key)
        try:
            for key, value in params.items():
                if key in _silent_conley_passthrough:
                    # No-op: explicitly None passthrough for round-trip
                    # get_params() -> set_params() consistency.
                    continue
                if key in _deprecated:
                    warnings.warn(
                        f"{key} is deprecated and ignored. Use zeta_omega/zeta_lambda "
                        f"instead. Will be removed in v4.0.0.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                elif hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown parameter: {key}")
            self._validate_config()
        except (ValueError, TypeError):
            for key, prev in _rollback.items():
                setattr(self, key, prev)
            raise
        return self
