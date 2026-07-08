"""
Two-Way Fixed Effects estimator for panel Difference-in-Differences.
"""

import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.bacon import BaconDecompositionResults

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.linalg import LinearRegression
from diff_diff.results import DiDResults
from diff_diff.utils import (
    fe_dummy_names,
    pre_demean_norms,
    snap_absorbed_regressors,
    validate_covariate_names,
    validate_design_term_names,
)
from diff_diff.utils import (
    within_transform as _within_transform_util,
)


class TwoWayFixedEffects(DifferenceInDifferences):
    """
    Two-Way Fixed Effects (TWFE) estimator for panel DiD.

    Extends DifferenceInDifferences to handle panel data with unit
    and time fixed effects.

    Parameters
    ----------
    robust : bool, default=True
        Whether to use heteroskedasticity-robust standard errors.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, automatically clusters at the unit level (the `unit`
        parameter passed to `fit()`). This differs from
        DifferenceInDifferences where cluster=None means no clustering.

        **Exception (one-way analytical):** when
        ``vcov_type in {"classical", "hc2"}`` is explicit AND
        ``inference="analytical"``, the unit auto-cluster is dropped
        because these families are by construction one-way only and the
        validator rejects ``cluster_ids + classical`` / ``cluster_ids +
        hc2``. The user's explicit one-way choice wins over the TWFE
        default. Under ``inference="wild_bootstrap"`` the auto-cluster
        is preserved regardless of ``vcov_type`` (the bootstrap uses the
        cluster structure to resample residuals). On ``hc2_bm`` the
        auto-cluster is also preserved (routes to CR2-BM at unit).
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Notes
    -----
    This estimator uses the regression:

        Y_it = α_i + γ_t + β*(D_i × Post_t) + X_it'δ + ε_it

    where α_i are unit fixed effects and γ_t are time fixed effects.

    **HC2 / Bell-McCaffrey are supported via an internal full-dummy build.**
    Because TWFE's within-transformation preserves coefficients but not the
    hat matrix, HC2 leverage and CR2 Bell-McCaffrey corrections on the
    demeaned design would produce wrong small-sample SEs. When
    ``vcov_type in {"hc2","hc2_bm"}``, TWFE bypasses the within-transform
    and builds the full-dummy design ``[intercept, treated×post,
    covariates, unit_dummies, time_dummies]`` directly, so the leverage
    correction and BM DOF compute on the full FE projection. Under this
    path, ``result.coefficients``, ``result.vcov``, ``result.residuals``,
    ``result.fitted_values``, and ``result.r_squared`` reflect the
    full-dummy fit rather than the within-transformed reduced fit; the
    ATT coefficient, its SE, and analytical inference are unchanged.
    Auto-cluster-at-unit is preserved on ``hc2_bm`` (routes to CR2-BM at
    unit) and on ``hc2`` + ``wild_bootstrap``; dropped on explicit ``hc2``
    + ``analytical`` to match the one-way contract. **This wording applies
    to the non-survey analytical path**: under ``survey_design=`` with no
    explicit ``cluster=``, TWFE intentionally keeps the documented
    implicit-PSU path (auto-cluster is NOT injected into the survey PSU
    structure) — users who want unit-level PSU injection under a survey
    design must pass explicit ``cluster="unit"`` or set
    ``survey_design.psu``. Documented in
    ``docs/methodology/REGISTRY.md`` under the scope-limitation note.

    **Conley spatial-HAC (``vcov_type="conley"``) is supported via the
    block-decomposed panel sandwich (matches R ``conleyreg`` with
    ``lag_cutoff > 0``).** Pass ``conley_coords=(lat_col, lon_col)``,
    ``conley_cutoff_km=<float>``, and ``conley_lag_cutoff=<int>`` on the
    constructor; the ``time`` / ``unit`` arrays are auto-derived from the
    estimator's ``time`` and ``unit`` column-name arguments at fit-time.
    The sandwich sums within-period spatial pairs plus within-unit
    Bartlett serial pairs (lag=0 excluded to avoid double-counting); this
    is NOT a multiplicative product kernel. FWL composability: the
    within-transformed scores ``S = X_demeaned * residuals_demeaned`` form
    the same meat as the full-dummy-expansion design. The temporal kernel
    is hardcoded Bartlett regardless of ``conley_kernel`` (matches
    ``conleyreg::time_dist``). Explicit ``cluster=<col>`` + Conley
    enables the combined spatial + cluster product kernel
    ``K_total[i, j] = K_space(d_ij/h) · 1{cluster_i = cluster_j}``;
    cluster membership must be constant within each unit across periods
    (validator-enforced on the panel block-decomposed path). When
    ``cluster=`` is unset, TWFE's default auto-cluster at the unit level
    is silently dropped on the Conley path — Conley spatial HAC alone is
    applied, not the combined kernel. Restrictions:
    ``inference="wild_bootstrap"`` + Conley raises (incompatible inference
    modes); ``survey_design=`` + Conley raises (Phase 5 follow-up).

    Warning: TWFE can be biased with staggered treatment timing
    and heterogeneous treatment effects. Consider using
    more robust estimators (e.g., Callaway-Sant'Anna) for
    staggered designs.
    """

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        unit: str,
        covariates: Optional[List[str]] = None,
        survey_design: object = None,
    ) -> DiDResults:
        """
        Fit Two-Way Fixed Effects model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Name of outcome variable column.
        treatment : str
            Name of treatment indicator column.
        time : str
            Name of time period column.
        unit : str
            Name of unit identifier column.
        covariates : list, optional
            List of covariate column names. Names must not collide with reserved
            structural terms (``const``, ``ATT``, unit/time fixed-effect dummy
            names, or the internal ``_treatment_post`` column) and must be
            unique; a collision or duplicate raises ``ValueError`` (it would
            otherwise silently overwrite a structural coefficient on the
            full-dummy HC2/HC2-BM path).
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. When provided,
            uses Taylor Series Linearization for variance estimation and
            applies sampling weights to the regression.

        Returns
        -------
        DiDResults
            Estimation results.

        Raises
        ------
        ValueError
            If a covariate name collides with a reserved structural term name
            or duplicates another covariate.
        """
        # Validate unit column exists
        if unit not in data.columns:
            raise ValueError(f"Unit column '{unit}' not found in data")

        # HC2 / HC2 Bell-McCaffrey are now SUPPORTED via the inline
        # full-dummy build below (see "use_full_dummy" branch around the
        # design-construction block). FWL preserves coefficients and
        # residuals but NOT the hat matrix, so HC2 leverage and CR2-BM
        # DOF must compute on the full FE projection; building the design
        # with explicit unit + time dummies routes through ``solve_ols``'s
        # full-design hat matrix. HC1/CR1 paths remain on the demeaned
        # design (no leverage term).
        use_full_dummy = self.vcov_type in ("hc2", "hc2_bm")

        # Phase 2 panel block-decomposed Conley (matches R conleyreg).
        # FWL composability: the within-transformed scores S = X_demeaned *
        # residuals_demeaned form the same meat as the full-dummy expansion,
        # so passing the demeaned X / residuals to compute_conley_vcov along
        # with the original (un-demeaned) time / unit vectors and coords
        # yields the correct block-decomposed sandwich.
        if self.vcov_type == "conley":
            from diff_diff.conley import _validate_conley_estimator_inputs

            _validate_conley_estimator_inputs(
                estimator_name="TwoWayFixedEffects",
                data=data,
                unit=unit,
                conley_coords=self.conley_coords,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_lag_cutoff=self.conley_lag_cutoff,
                survey_design=survey_design,
                inference=self.inference,
                cluster=self.cluster,
            )

        # Check for staggered treatment timing and warn if detected
        self._check_staggered_treatment(data, treatment, time, unit)

        # Warn if time has more than 2 unique values (not a binary post indicator)
        n_unique_time = data[time].nunique()
        if n_unique_time > 2:
            warnings.warn(
                f"The '{time}' column has {n_unique_time} unique values. "
                f"TwoWayFixedEffects expects a binary (0/1) post indicator. "
                f"Multi-period time values produce 'treated * period_number' instead of "
                f"'treated * post_indicator', which may not estimate the standard DiD ATT. "
                f"Consider creating a binary post column: "
                f"df['post'] = (df['{time}'] >= cutoff).astype(int)",
                UserWarning,
                stacklevel=2,
            )
        elif n_unique_time == 2:
            unique_vals = set(data[time].unique())
            if unique_vals != {0, 1} and unique_vals != {False, True}:
                warnings.warn(
                    f"The '{time}' column has values {sorted(unique_vals)} instead of {{0, 1}}. "
                    f"The ATT estimate is mathematically correct (within-transformation "
                    f"absorbs the scaling), but 0/1 encoding is recommended for clarity. "
                    f"Consider: df['{time}'] = (df['{time}'] == {max(unique_vals)}).astype(int)",
                    UserWarning,
                    stacklevel=2,
                )

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_effective_cluster, _resolve_survey_for_fit

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, self.inference)
        )
        _uses_replicate_twfe = (
            resolved_survey is not None and resolved_survey.uses_replicate_variance
        )
        if _uses_replicate_twfe and self.inference == "wild_bootstrap":
            raise ValueError(
                "Cannot use inference='wild_bootstrap' with replicate-weight "
                "survey designs. Replicate weights provide their own variance "
                "estimation."
            )
        # Replicate designs replace the analytical vcov wholesale: the
        # per-replicate refits return point estimates only, so vcov_type
        # cannot influence any reported number (FWL: the full-dummy and
        # within-transformed fits give identical coefficients, hence an
        # identical replicate variance). An explicit vcov_type therefore
        # warns and the (discarded) base fit remaps to hc1 on the
        # within-transformed design — this also disables the full-dummy
        # HC2/HC2-BM auto-route, which does not compose with the
        # per-replicate re-demeaning. Previously hc2/hc2_bm raised
        # NotImplementedError here; the warn-and-proceed contract matches
        # DifferenceInDifferences/MultiPeriodDiD (shared helper).
        _replicate_vcov_remap_twfe = _uses_replicate_twfe and self._warn_replicate_vcov_ignored()
        if _replicate_vcov_remap_twfe:
            use_full_dummy = False

        # Unit-level clustering is the TWFE default when `cluster` is not
        # explicitly provided. But the one-way ``classical`` and ``hc2``
        # families are by construction not cluster-robust and the validator
        # in ``compute_robust_vcov`` rejects
        # ``cluster_ids + vcov_type in ("classical","hc2")``. When the user
        # EXPLICITLY asks for one of these analytical-one-way families AND
        # does NOT set ``cluster=``, honor that choice by disabling the
        # auto-cluster.
        #
        # When ``"classical"`` is IMPLICIT (from the legacy alias
        # ``robust=False``), keep the unit auto-cluster so
        # ``_resolve_effective_vcov_type`` below can remap it to ``"hc1"``
        # and preserve the historical CR1-at-unit behavior. Wild-bootstrap
        # inference also keeps the unit auto-cluster regardless of
        # ``vcov_type`` (bootstrap consumes cluster structure for
        # resampling, independent of the analytical sandwich). ``hc2_bm``
        # also keeps the auto-cluster (routes to CR2-BM at unit).
        if self.cluster is not None:
            cluster_var: Optional[str] = self.cluster
        elif (
            self.vcov_type in ("classical", "hc2")
            and self._vcov_type_explicit
            and self.inference == "analytical"
        ):
            # Explicit one-way analytical vcov: drop the auto-cluster so
            # the validator doesn't reject ``cluster_ids`` with these
            # families. Wild-bootstrap is exempt because the bootstrap
            # uses the cluster structure for resampling regardless of
            # the analytical sandwich choice.
            cluster_var = None
        else:
            cluster_var = unit

        # Create treatment × post interaction from raw data.
        data = data.copy()
        data["_treatment_post"] = data[treatment] * data[time]

        n_units = data[unit].nunique()
        n_times = data[time].nunique()

        # Reject covariate names that collide with reserved structural terms.
        # On the full-dummy (HC2/HC2-BM) path covariates are zipped into the
        # coefficient dict alongside "const"/"ATT" and the unit/time dummies, so
        # a colliding covariate would silently overwrite that coefficient (dict
        # last-write-wins). The within-transform path does not expose covariates
        # in the dict, but the covariate is still in X and a covariate named
        # "_treatment_post" would clobber the internal interaction column; we
        # therefore guard ALL paths. Unit/time dummy names are derived via
        # fe_dummy_names WITHOUT materializing the dummy matrix — critical here
        # because the within-transform path (the default hc1/classical/conley
        # branch) deliberately never expands the full FE dummies (its scaling
        # contract for high-cardinality panels); a get_dummies build would
        # defeat that. validate_design_term_names re-checks the full-dummy list.
        _reserved = {"const", "ATT", "_treatment_post"}
        _reserved.update(fe_dummy_names(data[unit], f"_fe_{unit}"))
        _reserved.update(fe_dummy_names(data[time], f"_fe_{time}"))
        validate_covariate_names(covariates, _reserved, estimator="TwoWayFixedEffects")

        if use_full_dummy:
            # HC2 / HC2-BM full-dummy build: bypass the within-transform
            # and stack [intercept, treated×post, covariates, unit_dummies,
            # time_dummies] explicitly. FWL preserves the ATT coefficient
            # and residuals, but NOT the hat matrix — so the leverage
            # correction `h_ii = x_i' (X'X)^{-1} x_i` and the CR2 Bell-
            # McCaffrey adjustment matrix `A_g = (I - H_gg)^{-1/2}` must
            # be computed on the full FE projection. Pivoted-QR rank
            # detection in `solve_ols` cleanly drops any collinear FE
            # dummies (e.g. an always-treated unit × treatment_post
            # collinearity) without poisoning the ATT.
            # Memory guard: the full-dummy design materializes a dense
            # n × (1 + 1 + n_covs + (n_units-1) + (n_times-1)) matrix.
            # On large TWFE panels (n_units > 5000 typical) this can blow
            # up working memory. Warn when the design exceeds ~50M float64
            # entries (~400 MB) so users can switch to HC1 (within-
            # transform path) for those panels.
            _design_cols = 2 + len(covariates or []) + max(0, n_units - 1) + max(0, n_times - 1)
            _design_entries = len(data) * _design_cols
            if _design_entries > 50_000_000:
                warnings.warn(
                    f"TwoWayFixedEffects(vcov_type={self.vcov_type!r}) builds a "
                    f"dense {len(data)} × {_design_cols} full-dummy design "
                    f"(~{_design_entries / 1e6:.1f}M float64 entries, "
                    f"~{_design_entries * 8 / 1e9:.2f} GB). For panels with "
                    f"many units/periods, consider vcov_type='hc1' (within-"
                    "transform path; no leverage term, lower memory) unless "
                    "small-sample HC2/HC2-BM inference is required.",
                    UserWarning,
                    stacklevel=2,
                )
            y = data[outcome].values.astype(np.float64)
            cov_arrs = [data[c].values.astype(np.float64) for c in (covariates or [])]
            unit_dummies_df = pd.get_dummies(data[unit], prefix=f"_fe_{unit}", drop_first=True)
            time_dummies_df = pd.get_dummies(data[time], prefix=f"_fe_{time}", drop_first=True)
            unit_dummies = unit_dummies_df.values.astype(np.float64)
            time_dummies = time_dummies_df.values.astype(np.float64)
            X = np.column_stack(
                [np.ones(len(data)), data["_treatment_post"].values]
                + cov_arrs
                + [unit_dummies, time_dummies]
            )
            # FEs are now in X explicitly; solve_ols's n - k accounting
            # already subtracts them, so the extra unit + time DOF
            # adjustment used on the within-transform path would
            # double-count.
            df_adjustment = 0
            # var_names parallels the X columns so the downstream
            # `coefficients` dict can mirror the full-dummy vcov shape
            # (matching the MPD invariant
            # ``len(result.coefficients) == result.vcov.shape[0]``).
            _twfe_var_names: Optional[List[str]] = (
                ["const", "ATT"]
                + list(covariates or [])
                + list(unit_dummies_df.columns)
                + list(time_dummies_df.columns)
            )
            # Backstop: reject any duplicate in the FINAL term list (e.g. a
            # unit/time dummy colliding with a structural term or another dummy)
            # before it silently overwrites a coefficient in the dict below.
            validate_design_term_names(_twfe_var_names, estimator="TwoWayFixedEffects")
        else:
            # Default within-transform path (HC1 / classical / Conley):
            # demean outcome, covariates, AND interaction in a single pass
            # so the regression uses demeaned regressors (FWL theorem).
            all_vars = [outcome] + (covariates or []) + ["_treatment_post"]
            _twfe_regressors = all_vars[1:]  # everything except outcome
            _pre_norms = pre_demean_norms(data, _twfe_regressors, weights=survey_weights)
            data_demeaned = _within_transform_util(
                data,
                all_vars,
                unit,
                time,
                suffix="_demeaned",
                weights=survey_weights,
            )
            # Snap FE-spanned regressors (e.g. a unit-constant covariate) to
            # exact zero so rank handling drops them deterministically.
            snap_absorbed_regressors(
                data_demeaned,
                _twfe_regressors,
                _pre_norms,
                absorbed_desc=f"unit '{unit}' and time '{time}' fixed effects",
                group_vars=[unit, time],
                rank_deficient_action=self.rank_deficient_action,
                suffix="_demeaned",
                display_names={"_treatment_post": f"{treatment}:{time}"},
                weights=survey_weights,
            )
            y = data_demeaned[f"{outcome}_demeaned"].values
            X_list = [data_demeaned["_treatment_post_demeaned"].values]
            if covariates:
                for cov in covariates:
                    X_list.append(data_demeaned[f"{cov}_demeaned"].values)
            X = np.column_stack([np.ones(len(y))] + X_list)
            df_adjustment = n_units + n_times - 2
            # Within-transform path: preserve the historical
            # `{"ATT": att}` user-facing `result.coefficients` contract.
            # Broadening this dict here would silently change the
            # API surface on HC1 / classical / Conley fits — the
            # full-dummy `_twfe_var_names` exposure is scoped to the
            # HC2 / HC2-BM paths only (the documented surface change).
            _twfe_var_names = None

        # ATT is the coefficient on treatment_post (index 1) on both branches.
        att_idx = 1

        # Always use LinearRegression for initial fit (unified code path)
        # For wild bootstrap, we don't need cluster SEs from the initial fit.
        # cluster_var may be None when the user selected a one-way vcov_type
        # (``classical``/``hc2``) without setting ``cluster=``; honor it.
        cluster_ids = data[cluster_var].values if cluster_var is not None else None

        # When survey PSU is present, it overrides cluster for variance estimation
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids, self.cluster
        )

        # For survey variance: only inject user-explicit cluster as PSU.
        # TWFE's default unit clustering should not override the documented
        # no-PSU survey path (implicit per-observation PSUs).
        if resolved_survey is not None and self.cluster is None:
            survey_cluster_ids = None
        else:
            survey_cluster_ids = effective_cluster_ids

        # Inject cluster as effective PSU for survey variance estimation
        if resolved_survey is not None and survey_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, survey_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Pass rank_deficient_action to LinearRegression
        # If "error", let LinearRegression raise immediately
        # If "warn" or "silent", suppress generic warning and use TWFE's context-specific
        # error/warning messages (more informative for panel data)
        # For replicate designs: pass survey_design=None to prevent LinearRegression
        # from computing replicate vcov on already-demeaned data (demeaning depends
        # on weights, so replicate refits must re-demean at the estimator level).
        _lr_survey_twfe = None if _uses_replicate_twfe else resolved_survey
        # Remap implicit "classical" + cluster to CR1 for legacy-alias
        # backward compatibility. TWFE auto-clusters at unit when the user
        # doesn't set cluster, so `robust=False` without an explicit
        # vcov_type historically produced CR1 at unit; we preserve that.
        # Don't forward `robust=self.robust` to LinearRegression when the
        # remapped vcov_type disagrees; the remapped `vcov_type` is the
        # single source of truth.
        _fit_vcov_type = (
            "hc1"
            if _replicate_vcov_remap_twfe
            else self._resolve_effective_vcov_type(survey_cluster_ids)
        )

        # Phase 2 panel-Conley: build coord array + row-aligned time/unit
        # vectors from the original (un-demeaned) data. FWL composability:
        # within-transformed X already encodes the FE; the meat is computed
        # on demeaned scores but the kernel grid uses the original space
        # (coords) and time/unit indexing.
        if _fit_vcov_type == "conley":
            _conley_coords_arr: Optional[np.ndarray] = np.column_stack(
                [
                    data[self.conley_coords[0]].values.astype(np.float64),
                    data[self.conley_coords[1]].values.astype(np.float64),
                ]
            )
            # Preserve the original time-label dtype (int, datetime64, pd.Period,
            # string). `_compute_conley_vcov` normalizes to dense 0..T-1 codes
            # internally; float coercion here would break datetime64 / Period /
            # string encodings before the normalizer runs.
            _conley_time_arr: Optional[np.ndarray] = np.asarray(data[time].values)
            _conley_unit_arr: Optional[np.ndarray] = data[unit].values
            # Combined spatial + cluster product kernel: thread the user's
            # EXPLICIT cluster=<col> through when set. TWFE's default auto-
            # cluster at the unit level is silently dropped on the Conley
            # path — combining Conley with the unit-cluster mask zeros out
            # all between-unit pairs (defeating Conley's spatial pooling).
            # Users who want the combined kernel must pass an above-unit
            # cluster (e.g. region) explicitly.
            _conley_cluster_override = (
                data[self.cluster].values if self.cluster is not None else None
            )
        else:
            _conley_coords_arr = None
            _conley_time_arr = None
            _conley_unit_arr = None
            _conley_cluster_override = (
                survey_cluster_ids if self.inference != "wild_bootstrap" else None
            )

        if self.rank_deficient_action == "error":
            reg = LinearRegression(
                include_intercept=False,
                cluster_ids=_conley_cluster_override,
                alpha=self.alpha,
                rank_deficient_action="error",
                weights=survey_weights,
                weight_type=survey_weight_type,
                survey_design=_lr_survey_twfe,
                vcov_type=_fit_vcov_type,
                conley_coords=_conley_coords_arr,
                conley_cutoff_km=self.conley_cutoff_km,
                conley_metric=self.conley_metric,
                conley_kernel=self.conley_kernel,
                conley_time=_conley_time_arr,
                conley_unit=_conley_unit_arr,
                conley_lag_cutoff=self.conley_lag_cutoff,
            ).fit(X, y, df_adjustment=df_adjustment)
        else:
            # Suppress generic warning, TWFE provides context-specific messages below
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Rank-deficient design matrix")
                reg = LinearRegression(
                    include_intercept=False,
                    cluster_ids=_conley_cluster_override,
                    alpha=self.alpha,
                    rank_deficient_action="silent",
                    weights=survey_weights,
                    weight_type=survey_weight_type,
                    survey_design=_lr_survey_twfe,
                    vcov_type=_fit_vcov_type,
                    conley_coords=_conley_coords_arr,
                    conley_cutoff_km=self.conley_cutoff_km,
                    conley_metric=self.conley_metric,
                    conley_kernel=self.conley_kernel,
                    conley_time=_conley_time_arr,
                    conley_unit=_conley_unit_arr,
                    conley_lag_cutoff=self.conley_lag_cutoff,
                ).fit(X, y, df_adjustment=df_adjustment)

        coefficients = reg.coefficients_
        residuals = reg.residuals_
        fitted = reg.fitted_values_
        r_squared = reg.r_squared()
        assert coefficients is not None
        att = coefficients[att_idx]

        # Check for unidentified coefficients (collinearity)
        # Build column names for informative error messages
        column_names = ["intercept", "treatment×post"]
        if covariates:
            column_names.extend(covariates)

        nan_mask = np.isnan(coefficients)
        if np.any(nan_mask):
            dropped_indices = np.where(nan_mask)[0]
            dropped_names = [
                column_names[i] if i < len(column_names) else f"column {i}" for i in dropped_indices
            ]

            # Determine the source of collinearity for better error message
            if att_idx in dropped_indices:
                # Treatment coefficient is unidentified
                raise ValueError(
                    f"Treatment effect cannot be identified due to collinearity. "
                    f"Dropped columns: {', '.join(dropped_names)}. "
                    "This can happen when: (1) treatment is perfectly collinear with "
                    "unit/time fixed effects, (2) all treated units are treated in all "
                    "periods, or (3) a covariate is collinear with the treatment indicator. "
                    "Check your data structure and model specification."
                )
            else:
                # Only covariates are dropped - this is a warning, not an error
                # The ATT can still be estimated
                # Respect rank_deficient_action setting for warning
                if self.rank_deficient_action == "warn":
                    warnings.warn(
                        f"Some covariates are collinear and were dropped: "
                        f"{', '.join(dropped_names)}. The treatment effect is still identified.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Get inference - replicate, bootstrap, or analytical
        if _uses_replicate_twfe:
            # Estimator-level replicate variance: re-do within-transform per replicate
            from diff_diff.linalg import solve_ols
            from diff_diff.survey import compute_replicate_refit_variance
            from diff_diff.utils import safe_inference as _safe_inf

            _all_vars_twfe = list(all_vars)
            _covariates_twfe = list(covariates) if covariates else []
            # Handle rank-deficient nuisance: refit only identified columns
            _id_mask_twfe = ~np.isnan(coefficients)
            _id_cols_twfe = np.where(_id_mask_twfe)[0]

            def _refit_twfe(w_r):
                # Drop zero-weight obs to prevent zero-sum demeaning
                # (JK1/BRR half-samples zero entire clusters)
                nz = w_r > 0
                data_nz = data[nz].copy()
                w_nz = w_r[nz]
                _rep_norms_twfe = pre_demean_norms(data_nz, _all_vars_twfe[1:], weights=w_nz)
                data_dem_r = _within_transform_util(
                    data_nz,
                    _all_vars_twfe,
                    unit,
                    time,
                    suffix="_demeaned",
                    weights=w_nz,
                )
                # Replicate-local FE spanning: snap silently (see DiD closure).
                snap_absorbed_regressors(
                    data_dem_r,
                    _all_vars_twfe[1:],
                    _rep_norms_twfe,
                    absorbed_desc=f"unit '{unit}' and time '{time}' fixed effects",
                    group_vars=[unit, time],
                    rank_deficient_action="silent",
                    suffix="_demeaned",
                    weights=w_nz,
                )
                y_r = data_dem_r[f"{outcome}_demeaned"].values
                X_list_r = [data_dem_r["_treatment_post_demeaned"].values]
                for cov_ in _covariates_twfe:
                    X_list_r.append(data_dem_r[f"{cov_}_demeaned"].values)
                X_r = np.column_stack([np.ones(len(y_r))] + X_list_r)
                coef_r, _, _ = solve_ols(
                    X_r[:, _id_cols_twfe],
                    y_r,
                    weights=w_nz,
                    weight_type=survey_weight_type,
                    rank_deficient_action="silent",
                    return_vcov=False,
                )
                return coef_r

            from diff_diff.linalg import _expand_vcov_with_nan as _expand_twfe

            vcov_reduced, _n_valid_rep_twfe = compute_replicate_refit_variance(
                _refit_twfe, coefficients[_id_mask_twfe], resolved_survey
            )
            vcov = _expand_twfe(vcov_reduced, len(coefficients), _id_cols_twfe)
            se = float(np.sqrt(max(vcov[att_idx, att_idx], 0.0)))
            _df_rep = (
                survey_metadata.df_survey
                if survey_metadata and survey_metadata.df_survey
                else 0  # rank-deficient replicate → NaN inference
            )
            if _n_valid_rep_twfe < resolved_survey.n_replicates:
                _df_rep = _n_valid_rep_twfe - 1 if _n_valid_rep_twfe > 1 else 0
            if survey_metadata is not None:
                survey_metadata.df_survey = _df_rep if _df_rep > 0 else None
            t_stat, p_value, conf_int = _safe_inf(att, se, alpha=self.alpha, df=_df_rep)
        elif self.inference == "wild_bootstrap":
            # Override with wild cluster bootstrap inference
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx
            )
        else:
            # Use analytical inference from LinearRegression
            vcov = reg.vcov_
            inference = reg.get_inference(att_idx)
            se = inference.se
            t_stat = inference.t_stat
            p_value = inference.p_value
            conf_int = inference.conf_int

        # Count observations
        treated_units = data[data[treatment] == 1][unit].unique()
        n_treated = len(treated_units)
        n_control = n_units - n_treated

        # Determine inference method and bootstrap info
        inference_method = "analytical"
        n_bootstrap_used = None
        n_clusters_used = None
        p_val_type_used = None
        if self._bootstrap_results is not None:
            inference_method = "wild_bootstrap"
            n_bootstrap_used = self._bootstrap_results.n_bootstrap
            n_clusters_used = self._bootstrap_results.n_clusters
            p_val_type_used = self._bootstrap_results.p_val_type

        # Cluster label for summary: TWFE auto-clusters at unit level when
        # `self.cluster is None` AND the vcov family is cluster-compatible.
        # One-way families (`classical`, `hc2`) disable the auto-cluster (see
        # the `cluster_var` block above); report None so summary() labels the
        # one-way family instead of "CR1 cluster-robust at unit". On the
        # Conley path, the auto-cluster is silently dropped (combining with
        # unit-level clusters would zero out all between-unit pairs and
        # defeat the spatial pooling); when the user passes an explicit
        # `cluster=<col>` ALONGSIDE Conley, the combined spatial + cluster
        # product kernel applies and the label tracks the user's column.
        # Either way, the cluster_name reflects the cluster IDs actually
        # passed to LinearRegression.
        if _fit_vcov_type == "conley":
            _twfe_cluster_label: Optional[str] = self.cluster if self.cluster is not None else None
        elif self.cluster is not None:
            _twfe_cluster_label = self.cluster
        elif cluster_var is None:
            # One-way family path: auto-cluster was intentionally dropped.
            _twfe_cluster_label = None
        else:
            _twfe_cluster_label = unit

        # Build the coefficients dict mirroring the actual X columns. On the
        # full-dummy path this surfaces the FE-dummy entries alongside the ATT;
        # on the within-transform path it only carries the visible
        # [const, ATT, covariates] columns. The "ATT" name at index 1 is
        # preserved as the ATT key on both paths, so existing
        # `result.coefficients["ATT"]` consumers continue to work. The
        # invariant ``len(coefficients) == vcov.shape[0]`` is now upheld on the
        # full-dummy path (matches the MPD absorb auto-route invariant
        # checked at tests/test_estimators_vcov_type.py:1611).
        coef_array = np.asarray(reg.coefficients_)
        _coefficients_dict: dict = (
            {name: float(c) for name, c in zip(_twfe_var_names, coef_array)}
            if _twfe_var_names is not None
            else {"ATT": float(att)}
        )
        self.results_ = DiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            alpha=self.alpha,
            coefficients=_coefficients_dict,
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
            inference_method=inference_method,
            n_bootstrap=n_bootstrap_used,
            n_clusters=n_clusters_used,
            p_val_type=p_val_type_used,
            survey_metadata=survey_metadata,
            # Report the family that actually produced the SE; may be the
            # remapped hc1 under the legacy alias path, not self.vcov_type.
            vcov_type=_fit_vcov_type,
            cluster_name=_twfe_cluster_label,
            conley_lag_cutoff=(self.conley_lag_cutoff if _fit_vcov_type == "conley" else None),
        )

        self.is_fitted_ = True
        return self.results_

    def _check_staggered_treatment(
        self,
        data: pd.DataFrame,
        treatment: str,
        time: str,
        unit: str,
    ) -> None:
        """
        Check for staggered treatment timing and warn if detected.

        Identifies if different units start treatment at different times,
        which can bias TWFE estimates when treatment effects are heterogeneous.

        Note: This check requires ``time`` to have actual period values (not
        binary 0/1). With binary time, all treated units appear to start at
        time=1, so staggering is undetectable.
        """
        # Find first treatment time for each unit
        treated_obs = data[data[treatment] == 1]
        if len(treated_obs) == 0:
            return  # No treated observations

        # Get first treatment time per unit
        first_treat_times = treated_obs.groupby(unit)[time].min()
        unique_treat_times = first_treat_times.unique()

        if len(unique_treat_times) > 1:
            n_groups = len(unique_treat_times)
            warnings.warn(
                f"Staggered treatment timing detected: {n_groups} treatment cohorts "
                f"start treatment at different times. TWFE can be biased when treatment "
                f"effects are heterogeneous across time. Consider using:\n"
                f"  - CallawaySantAnna estimator for robust estimates\n"
                f"  - TwoWayFixedEffects.decompose() to diagnose the decomposition\n"
                f"  - bacon_decompose() to see weight on 'forbidden' comparisons",
                UserWarning,
                stacklevel=3,
            )

    def decompose(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        weights: str = "exact",
    ) -> "BaconDecompositionResults":
        """
        Perform Goodman-Bacon decomposition of TWFE estimate.

        Decomposes the TWFE estimate into a weighted average of all possible
        2x2 DiD comparisons, revealing which comparisons drive the estimate
        and whether problematic "forbidden comparisons" are involved.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when each unit was first treated.
            The values ``0`` and ``np.inf`` are **reserved as
            never-treated sentinels**; a real treatment cohort with
            ``first_treat == 0`` would be folded into ``U`` and should
            be re-labeled to a non-sentinel value before fitting. Units
            whose ``first_treat`` is at or before the first observable
            period (``first_treat <= min(time)``, excluding the
            sentinels) are automatically remapped to the ``U``
            (untreated) bucket per Goodman-Bacon (2021) footnote 11
            with a ``UserWarning``. See ``BaconDecomposition.fit()`` for
            the full contract and
            ``BaconDecompositionResults.n_always_treated_remapped`` for
            the count. The user's original ``first_treat`` column is
            preserved unchanged.
        weights : str, default="exact"
            Weight calculation method:

            - "exact" (default): Variance-based weights from Goodman-Bacon
              (2021) Theorem 1, Eqs. 7-9 and 10e-g. Paper-faithful and
              the standard methodology contract.
            - "approximate": Fast simplified formula. Opt in for
              speed-sensitive diagnostic loops; numerical output may
              differ from R ``bacondecomp::bacon()``.

        Returns
        -------
        BaconDecompositionResults
            Decomposition results showing:
            - TWFE estimate and its weighted-average breakdown
            - List of all 2x2 comparisons with estimates and weights
            - Total weight by comparison type (clean vs forbidden)

        Examples
        --------
        >>> twfe = TwoWayFixedEffects()
        >>> decomp = twfe.decompose(
        ...     data, outcome='y', unit='id', time='t', first_treat='treat_year'
        ... )
        >>> decomp.print_summary()
        >>> # Check weight on forbidden comparisons
        >>> if decomp.total_weight_later_vs_earlier > 0.2:
        ...     print("Warning: significant forbidden comparison weight")

        Notes
        -----
        This decomposition is essential for understanding potential TWFE bias
        in staggered adoption designs. The three comparison types are:

        1. **Treated vs Never-treated**: Clean comparisons using never-treated
           units as controls. These are always valid.

        2. **Earlier vs Later treated**: Uses later-treated units as controls
           before they receive treatment. These are valid.

        3. **Later vs Earlier treated**: Uses already-treated units as controls.
           These "forbidden comparisons" can introduce bias when treatment
           effects are dynamic (changing over time since treatment).

        See Also
        --------
        bacon_decompose : Standalone decomposition function
        BaconDecomposition : Class-based decomposition interface
        CallawaySantAnna : Robust estimator that avoids forbidden comparisons
        """
        from diff_diff.bacon import BaconDecomposition

        decomp = BaconDecomposition(weights=weights)
        return decomp.fit(data, outcome, unit, time, first_treat)
