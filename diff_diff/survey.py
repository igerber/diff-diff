"""
Survey data support for diff-diff.

Provides SurveyDesign for specifying complex survey structures (stratification,
clustering, weights, FPC) and Taylor Series Linearization (TSL) variance
estimation for design-based inference.

References
----------
- Lumley (2004) "Analysis of Complex Survey Samples", JSS 9(8).
- Binder (1983) "On the Variances of Asymptotically Normal Estimators
  from Complex Surveys", International Statistical Review 51(3).
- Solon, Haider, & Wooldridge (2015) "What Are We Weighting For?",
  Journal of Human Resources 50(2).
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import _factorize_cluster_ids


def validate_raw_weights(raw_weights: np.ndarray, weight_type: Optional[str] = None) -> None:
    """Reject a raw (pre-normalization) weight vector that cannot be used.

    Shared by ``SurveyDesign.resolve`` and by callers that must read the raw
    weight column BEFORE resolution, so both paths raise the same errors in the
    same order rather than drifting apart.

    ``WooldridgeDiD`` is such a caller: it decides cohort-time cell SUPPORT from
    raw weights (a cell whose rows all carry zero weight is absent from the
    effective regression), and an unsupported reference period EXCLUDES that
    cohort's rows. An invalid weight sitting only in the excluded rows would
    therefore never reach ``resolve()`` -- the fit would drop a cohort, report a
    confident finite estimate, and warn that the cohort "has no pre-treatment
    period" when in fact its periods were merely poisoned. Validating the raw
    vector up front closes that path.

    Raises
    ------
    ValueError
        If any weight is NaN, infinite, or negative, or if every weight is zero.
    """
    if np.any(np.isnan(raw_weights)):
        raise ValueError("Weights contain NaN values")
    if np.any(~np.isfinite(raw_weights)):
        raise ValueError("Weights contain Inf values")
    if np.any(raw_weights < 0):
        raise ValueError("Weights must be non-negative")
    if np.any(raw_weights == 0) and np.all(raw_weights == 0):
        raise ValueError(
            "All weights are zero. At least one observation must " "have a positive weight."
        )
    # fweight validation: must be non-negative integers. Lives here, not only in
    # resolve(), because callers that read raw weights early must enforce EVERY
    # documented weight-type rule -- a fractional fweight in a cohort that
    # support-based exclusion later removes would otherwise never be seen.
    if weight_type == "fweight":
        pos_mask = raw_weights > 0
        if np.any(pos_mask):
            fractional = raw_weights[pos_mask] - np.round(raw_weights[pos_mask])
            if np.any(np.abs(fractional) > 1e-10):
                raise ValueError(
                    "Frequency weights (fweight) must be non-negative integers. "
                    "Fractional values detected. Use pweight for non-integer weights."
                )


@dataclass
class SurveyDesign:
    """
    User-facing class specifying complex survey design structure.

    Column names are resolved against the DataFrame at fit-time via
    :meth:`resolve`.

    Parameters
    ----------
    weights : str, optional
        Column name for observation weights (sampling weights).
    strata : str, optional
        Column name for stratification variable.
    psu : str, optional
        Column name for primary sampling unit (cluster).
    fpc : str, optional
        Column name for finite population size (N_h per stratum).
    weight_type : str, default "pweight"
        Weight type: "pweight" (inverse selection probability),
        "fweight" (frequency/expansion), or "aweight" (inverse variance).
    nest : bool, default False
        Whether PSU IDs are nested within strata (i.e., PSU IDs may repeat
        across strata). If True, PSU IDs are made unique by combining with
        strata.
    lonely_psu : str, default "remove"
        How to handle singleton strata (strata with only one PSU):
        "remove" (skip, emit warning), "certainty" (set f_h=1, zero
        variance contribution), or "adjust" (center around grand mean).
    """

    weights: Optional[str] = None
    strata: Optional[str] = None
    psu: Optional[str] = None
    fpc: Optional[str] = None
    weight_type: str = "pweight"
    nest: bool = False
    lonely_psu: str = "remove"
    replicate_weights: Optional[List[str]] = None
    replicate_method: Optional[str] = None
    fay_rho: float = 0.0
    replicate_strata: Optional[List[int]] = None
    combined_weights: bool = True
    replicate_scale: Optional[float] = None
    replicate_rscales: Optional[List[float]] = None
    mse: bool = False

    def __post_init__(self):
        valid_weight_types = {"pweight", "fweight", "aweight"}
        if self.weight_type not in valid_weight_types:
            raise ValueError(
                f"weight_type must be one of {valid_weight_types}, " f"got '{self.weight_type}'"
            )
        valid_lonely = {"remove", "certainty", "adjust"}
        if self.lonely_psu not in valid_lonely:
            raise ValueError(
                f"lonely_psu must be one of {valid_lonely}, " f"got '{self.lonely_psu}'"
            )
        # Replicate weight validation
        valid_rep_methods = {"BRR", "Fay", "JK1", "JKn", "SDR"}
        if self.replicate_method is not None:
            if self.replicate_method not in valid_rep_methods:
                raise ValueError(
                    f"replicate_method must be one of {valid_rep_methods}, "
                    f"got '{self.replicate_method}'"
                )
            if self.replicate_weights is None:
                raise ValueError("replicate_weights must be provided when replicate_method is set")
        if self.replicate_weights is not None and self.replicate_method is None:
            raise ValueError("replicate_method must be provided when replicate_weights is set")
        if self.replicate_method == "Fay":
            if not (0 < self.fay_rho < 1):
                raise ValueError(f"fay_rho must be in (0, 1) for Fay's method, got {self.fay_rho}")
        elif self.replicate_method is not None and self.fay_rho != 0.0:
            raise ValueError(
                f"fay_rho must be 0 for method '{self.replicate_method}', " f"got {self.fay_rho}"
            )
        # Replicate weights are mutually exclusive with strata/psu/fpc
        if self.replicate_weights is not None:
            if self.strata is not None or self.psu is not None or self.fpc is not None:
                raise ValueError(
                    "replicate_weights cannot be combined with strata/psu/fpc. "
                    "Replicate weights encode the design structure implicitly."
                )
            if self.weights is None:
                raise ValueError("Full-sample weights must be provided alongside replicate_weights")
        # JKn requires replicate_strata
        if self.replicate_method == "JKn":
            if self.replicate_strata is None:
                raise ValueError(
                    "replicate_strata is required for JKn method. Provide a list "
                    "of stratum assignments (one per replicate weight column)."
                )
            if len(self.replicate_strata) != len(self.replicate_weights):
                raise ValueError(
                    f"replicate_strata length ({len(self.replicate_strata)}) must "
                    f"match replicate_weights length ({len(self.replicate_weights)})"
                )
        # Validate scale/rscales values and length
        if self.replicate_scale is not None:
            if not (np.isfinite(self.replicate_scale) and self.replicate_scale > 0):
                raise ValueError(
                    f"replicate_scale must be a positive finite number, "
                    f"got {self.replicate_scale}"
                )
        if self.replicate_rscales is not None and self.replicate_weights is not None:
            if len(self.replicate_rscales) != len(self.replicate_weights):
                raise ValueError(
                    f"replicate_rscales length ({len(self.replicate_rscales)}) must "
                    f"match replicate_weights length ({len(self.replicate_weights)})"
                )
            rscales_arr = np.asarray(self.replicate_rscales, dtype=float)
            if not np.all(np.isfinite(rscales_arr)):
                raise ValueError("replicate_rscales must be finite")
            if np.any(rscales_arr < 0):
                raise ValueError("replicate_rscales must be non-negative")

    def resolve(self, data: pd.DataFrame) -> "ResolvedSurveyDesign":
        """
        Validate column names and extract numpy arrays from DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing survey design columns.

        Returns
        -------
        ResolvedSurveyDesign
            Internal representation with extracted numpy arrays.
        """
        n = len(data)

        # --- Weights ---
        if self.weights is not None:
            if self.weights not in data.columns:
                raise ValueError(f"Weight column '{self.weights}' not found in data")
            raw_weights = data[self.weights].values.astype(np.float64)

            validate_raw_weights(raw_weights, self.weight_type)

            # Normalize: pweights/aweights to sum=n (mean=1); fweights unchanged
            # Skip normalization for replicate designs — the IF path uses
            # w_r / w_full ratios that must be on the same raw scale
            if self.replicate_weights is not None:
                weights = raw_weights.copy()
            elif self.weight_type in ("pweight", "aweight"):
                raw_sum = float(np.sum(raw_weights))
                weights = raw_weights * (n / raw_sum)
                if not np.isclose(raw_sum, n):
                    warnings.warn(
                        f"{self.weight_type} weights normalized to mean=1 "
                        f"(sum={n}). Original sum was {raw_sum:.4g}.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                weights = raw_weights.copy()
        else:
            weights = np.ones(n, dtype=np.float64)

        # --- Replicate weights (short-circuit strata/psu/fpc) ---
        if self.replicate_weights is not None:
            rep_cols = self.replicate_weights
            for col in rep_cols:
                if col not in data.columns:
                    raise ValueError(f"Replicate weight column '{col}' not found in data")
            rep_arr = np.column_stack([data[col].values.astype(np.float64) for col in rep_cols])
            if np.any(np.isnan(rep_arr)):
                raise ValueError("Replicate weights contain NaN values")
            if np.any(~np.isfinite(rep_arr)):
                raise ValueError("Replicate weights contain Inf values")
            if np.any(rep_arr < 0):
                raise ValueError("Replicate weights must be non-negative")
            # Validate combined_weights contract: when True, replicate columns
            # include the full-sample weight, so w_r > 0 with w_full == 0 is
            # malformed (observation excluded from full sample but included in
            # a replicate).
            combined = self.combined_weights if self.combined_weights is not None else True
            if combined:
                zero_full = weights == 0
                if np.any(zero_full):
                    rep_positive_on_zero = np.any(rep_arr[zero_full] > 0, axis=1)
                    if np.any(rep_positive_on_zero):
                        raise ValueError(
                            "Malformed combined_weights=True design: some "
                            "replicate columns have positive weight where "
                            "full-sample weight is zero. Either fix the "
                            "replicate columns or use combined_weights=False."
                        )
            # Do NOT normalize replicate columns — the IF path uses w_r/w_full
            # ratios that must reflect the true replicate design, not rescaled sums
            n_rep = rep_arr.shape[1]
            if n_rep < 2:
                raise ValueError("At least 2 replicate weight columns are required")
            return ResolvedSurveyDesign(
                weights=weights,
                weight_type=self.weight_type,
                strata=None,
                psu=None,
                fpc=None,
                n_strata=0,
                n_psu=0,
                lonely_psu=self.lonely_psu,
                replicate_weights=rep_arr,
                replicate_method=self.replicate_method,
                fay_rho=self.fay_rho,
                n_replicates=n_rep,
                replicate_strata=(
                    np.asarray(self.replicate_strata, dtype=int)
                    if self.replicate_strata is not None
                    else None
                ),
                combined_weights=self.combined_weights,
                replicate_scale=self.replicate_scale,
                replicate_rscales=(
                    np.asarray(self.replicate_rscales, dtype=float)
                    if self.replicate_rscales is not None
                    else None
                ),
                mse=self.mse,
                nest=self.nest,
            )

        # --- Strata ---
        strata_arr = None
        n_strata = 0
        if self.strata is not None:
            if self.strata not in data.columns:
                raise ValueError(f"Strata column '{self.strata}' not found in data")
            strata_vals = data[self.strata].values
            if pd.isna(strata_vals).any():
                raise ValueError(
                    f"Strata column '{self.strata}' contains missing values. "
                    "All observations must have valid strata identifiers."
                )
            strata_arr = _factorize_cluster_ids(strata_vals)
            n_strata = len(np.unique(strata_arr))

        # --- PSU ---
        psu_arr = None
        n_psu = 0
        if self.psu is not None:
            if self.psu not in data.columns:
                raise ValueError(f"PSU column '{self.psu}' not found in data")
            psu_raw = data[self.psu].values
            if pd.isna(psu_raw).any():
                raise ValueError(
                    f"PSU column '{self.psu}' contains missing values. "
                    "All observations must have valid PSU identifiers."
                )

            if self.nest and strata_arr is not None:
                # Make PSU IDs unique within strata by combining
                combined = np.array([f"{s}_{p}" for s, p in zip(strata_arr, psu_raw)])
                psu_arr = _factorize_cluster_ids(combined)
            else:
                psu_arr = _factorize_cluster_ids(psu_raw)
                # Validate PSU labels are globally unique when nest=False
                # and strata are present. Repeated labels cause wrong n_psu,
                # df_survey, and lonely_psu="adjust" global mean.
                if strata_arr is not None:
                    seen_psus: set = set()
                    for h in np.unique(strata_arr):
                        psu_in_h = set(psu_raw[strata_arr == h])
                        overlap = seen_psus & psu_in_h
                        if overlap:
                            raise ValueError(
                                f"PSU labels {overlap} appear in multiple strata. "
                                "Set nest=True in SurveyDesign to make PSU IDs "
                                "unique within strata, or use globally unique "
                                "PSU labels."
                            )
                        seen_psus |= psu_in_h

            n_psu = len(np.unique(psu_arr))

        # --- FPC ---
        fpc_arr = None
        if self.fpc is not None:
            if self.fpc not in data.columns:
                raise ValueError(f"FPC column '{self.fpc}' not found in data")
            fpc_arr = data[self.fpc].values.astype(np.float64)

            if np.any(np.isnan(fpc_arr)) or np.any(~np.isfinite(fpc_arr)):
                raise ValueError("FPC values must be finite and non-NaN")

            # Validate FPC structure (constant within strata, positive).
            # FPC >= n_PSU validation is deferred to compute_survey_vcov()
            # where the final effective PSU structure is known (after
            # cluster-as-PSU injection and implicit per-obs PSU fallback).
            if strata_arr is not None:
                for h in np.unique(strata_arr):
                    mask_h = strata_arr == h
                    fpc_vals = fpc_arr[mask_h]
                    # Enforce FPC is constant within stratum
                    if len(np.unique(fpc_vals)) > 1:
                        raise ValueError(
                            f"FPC values must be constant within each stratum. "
                            f"Stratum {h} has values: {np.unique(fpc_vals)}"
                        )
                    fpc_h = fpc_vals[0]
                    # Validate FPC >= n_PSU when explicit PSU is declared
                    if psu_arr is not None:
                        n_psu_h = len(np.unique(psu_arr[mask_h]))
                        if fpc_h < n_psu_h:
                            raise ValueError(
                                f"FPC ({fpc_h}) is less than the number of PSUs "
                                f"({n_psu_h}) in stratum {h}. FPC must be >= n_PSU."
                            )
            else:
                # No strata: require FPC is a single constant value
                if len(np.unique(fpc_arr)) > 1:
                    raise ValueError(
                        "FPC values must be constant when no strata are specified. "
                        f"Found {len(np.unique(fpc_arr))} distinct values."
                    )
                # Validate FPC >= n_PSU when explicit PSU is declared
                if psu_arr is not None and fpc_arr[0] < n_psu:
                    raise ValueError(
                        f"FPC ({fpc_arr[0]}) is less than the number of PSUs "
                        f"({n_psu}). FPC must be >= number of PSUs."
                    )

        # --- Validate PSU counts per stratum ---
        if psu_arr is not None and strata_arr is not None:
            for h in np.unique(strata_arr):
                mask_h = strata_arr == h
                n_psu_h = len(np.unique(psu_arr[mask_h]))
                if n_psu_h < 2:
                    if self.lonely_psu == "remove":
                        warnings.warn(
                            f"Stratum {h} has only {n_psu_h} PSU(s). "
                            "It will be excluded from variance estimation "
                            "(lonely_psu='remove').",
                            UserWarning,
                            stacklevel=3,
                        )
                    elif self.lonely_psu == "certainty":
                        pass  # Handled in compute_survey_vcov
                    elif self.lonely_psu == "adjust":
                        pass  # Handled in compute_survey_vcov

        # Validate PSU count for unstratified designs
        if psu_arr is not None and strata_arr is None:
            if n_psu < 2:
                if self.lonely_psu == "remove":
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Variance cannot be estimated (lonely_psu='remove')."
                    )
                elif self.lonely_psu == "certainty":
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Treated as certainty PSU; zero variance contribution."
                    )
                else:
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Cannot adjust with a single cluster and no strata; "
                        "variance will be NaN."
                    )
                warnings.warn(msg, UserWarning, stacklevel=3)

        return ResolvedSurveyDesign(
            weights=weights,
            weight_type=self.weight_type,
            strata=strata_arr,
            psu=psu_arr,
            fpc=fpc_arr,
            n_strata=n_strata,
            n_psu=n_psu,
            lonely_psu=self.lonely_psu,
            nest=self.nest,
        )

    def subpopulation(
        self,
        data: pd.DataFrame,
        mask,
    ) -> Tuple["SurveyDesign", pd.DataFrame]:
        """Create a subpopulation design by zeroing out excluded observations.

        Preserves the full survey design structure (strata, PSU) while setting
        weights to zero for observations outside the subpopulation.  This is
        the correct approach for subpopulation analysis — unlike naive
        subsetting, it retains design information for variance estimation.

        Parameters
        ----------
        mask : array-like of bool, str, or callable
            Defines the subpopulation:
            - bool array/Series of length ``len(data)`` — True = included
            - str — column name in ``data`` containing boolean values
            - callable — applied to ``data``, must return bool array

        Returns
        -------
        (SurveyDesign, pd.DataFrame)
            A new SurveyDesign pointing to a ``_subpop_weight`` column in the
            returned DataFrame copy.  The pair should be used together: pass
            the returned DataFrame to ``fit()`` with the returned SurveyDesign.
        """
        # Resolve mask to boolean array
        if callable(mask):
            raw_mask = np.asarray(mask(data))
        elif isinstance(mask, str):
            if mask not in data.columns:
                raise ValueError(f"Mask column '{mask}' not found in data")
            raw_mask = np.asarray(data[mask].values)
        else:
            raw_mask = np.asarray(mask)

        # Validate: reject pd.NA/pd.NaT/None before bool coercion
        try:
            if pd.isna(raw_mask).any():
                raise ValueError(
                    "Subpopulation mask contains NA/missing values. "
                    "Provide a boolean mask with no missing values."
                )
        except (TypeError, ValueError) as e:
            if "NA/missing" in str(e):
                raise
            # pd.isna can't handle some dtypes — fall through to specific checks
        if raw_mask.dtype.kind == "f" and np.any(np.isnan(raw_mask)):
            raise ValueError(
                "Subpopulation mask contains NaN values. "
                "Provide a boolean mask with no missing values."
            )
        if hasattr(raw_mask, "dtype") and raw_mask.dtype == object:
            # Check for None values (pd.NA, None, etc.)
            if any(v is None for v in raw_mask):
                raise ValueError(
                    "Subpopulation mask contains None/NA values. "
                    "Provide a boolean mask with no missing values."
                )
            # Reject string/object masks — non-empty strings coerce to True
            # which silently defines the wrong domain
            if any(isinstance(v, str) for v in raw_mask):
                raise ValueError(
                    "Subpopulation mask has object dtype with string values. "
                    "Provide a boolean or numeric (0/1) mask, not strings."
                )
        if hasattr(raw_mask, "dtype") and raw_mask.dtype.kind in ("U", "S"):
            raise ValueError(
                "Subpopulation mask contains string values. "
                "Provide a boolean or numeric (0/1) mask."
            )
        # Validate numeric masks: only {0, 1} allowed (not {1, 2}, etc.)
        if hasattr(raw_mask, "dtype") and raw_mask.dtype.kind in ("i", "u", "f"):
            unique_vals = set(np.unique(raw_mask[np.isfinite(raw_mask)]).tolist())
            if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                raise ValueError(
                    f"Subpopulation mask contains non-binary numeric values "
                    f"{unique_vals - {0, 1, 0.0, 1.0}}. "
                    f"Provide a boolean or numeric (0/1) mask."
                )
        mask_arr = raw_mask.astype(bool)

        if len(mask_arr) != len(data):
            raise ValueError(
                f"Mask length ({len(mask_arr)}) does not match data " f"length ({len(data)})"
            )

        if not np.any(mask_arr):
            raise ValueError(
                "Subpopulation mask excludes all observations. "
                "At least one observation must be included."
            )

        # Build subpopulation weights
        if self.weights is not None:
            if self.weights not in data.columns:
                raise ValueError(f"Weight column '{self.weights}' not found in data")
            base_weights = data[self.weights].values.astype(np.float64)
        else:
            base_weights = np.ones(len(data), dtype=np.float64)

        subpop_weights = np.where(mask_arr, base_weights, 0.0)

        # Create data copy with synthetic weight column
        data_out = data.copy()
        data_out["_subpop_weight"] = subpop_weights

        # Zero out replicate weight columns for excluded observations
        if self.replicate_weights is not None:
            for col in self.replicate_weights:
                if col in data.columns:
                    data_out[col] = np.where(mask_arr, data[col].values, 0.0)

        # Return new SurveyDesign using the synthetic column
        new_design = SurveyDesign(
            weights="_subpop_weight",
            strata=self.strata,
            psu=self.psu,
            fpc=self.fpc,
            weight_type=self.weight_type,
            nest=self.nest,
            lonely_psu=self.lonely_psu,
            replicate_weights=self.replicate_weights,
            replicate_method=self.replicate_method,
            fay_rho=self.fay_rho,
            replicate_strata=self.replicate_strata,
            combined_weights=self.combined_weights,
            replicate_scale=self.replicate_scale,
            replicate_rscales=self.replicate_rscales,
            mse=self.mse,
        )

        return new_design, data_out


@dataclass
class ResolvedSurveyDesign:
    """
    Internal class with extracted numpy arrays from SurveyDesign.resolve().

    Not intended for direct construction by users.
    """

    weights: np.ndarray
    weight_type: str
    strata: Optional[np.ndarray]
    psu: Optional[np.ndarray]
    fpc: Optional[np.ndarray]
    n_strata: int
    n_psu: int
    lonely_psu: str
    replicate_weights: Optional[np.ndarray] = None  # (n, R) array
    replicate_method: Optional[str] = None
    fay_rho: float = 0.0
    n_replicates: int = 0
    replicate_strata: Optional[np.ndarray] = None  # (R,) for JKn
    combined_weights: bool = True
    replicate_scale: Optional[float] = None
    replicate_rscales: Optional[np.ndarray] = None  # (R,) per-replicate scales
    mse: bool = False
    # Preserved from SurveyDesign for use by downstream helpers (e.g.
    # `_inject_cluster_as_psu` honors `nest` when substituting cluster=<col>
    # for an absent PSU column).
    nest: bool = False

    @property
    def uses_replicate_variance(self) -> bool:
        """Whether replicate-based variance should be used instead of TSL."""
        return self.replicate_method is not None

    @property
    def df_survey(self) -> Optional[int]:
        """Survey degrees of freedom.

        For replicate designs: QR-rank of the analysis-weight matrix minus 1,
        matching R's ``survey::degf()`` which uses ``qr(..., tol=1e-5)$rank``.
        Returns ``None`` when rank <= 1 (insufficient for t-based inference).
        For TSL: n_PSU - n_strata.
        """
        if self.uses_replicate_variance:
            if self.replicate_weights is None or self.n_replicates < 2:
                return None
            # QR-rank of analysis-weight matrix, matching R's survey::degf()
            # which uses qr(weights(design, "analysis"), tol=1e-5)$rank.
            # For combined_weights=True, replicate cols ARE analysis weights.
            # For combined_weights=False, analysis weights = rep * full-sample.
            if self.combined_weights:
                analysis_weights = self.replicate_weights
            else:
                analysis_weights = self.replicate_weights * self.weights[:, np.newaxis]
            # Pivoted QR with R-compatible tolerance, matching R's
            # qr(..., tol=1e-5) which uses column pivoting (LAPACK dgeqp3)
            from scipy.linalg import qr as scipy_qr

            _, R_mat, _ = scipy_qr(analysis_weights, pivoting=True, mode="economic")
            diag_abs = np.abs(np.diag(R_mat))
            tol = 1e-5
            rank = int(np.sum(diag_abs > tol * diag_abs.max())) if diag_abs.max() > 0 else 0
            df = rank - 1
            return df if df > 0 else None
        if self.psu is not None and self.n_psu > 0:
            if self.strata is not None and self.n_strata > 0:
                return self.n_psu - self.n_strata
            return self.n_psu - 1
        # Implicit PSU: each observation is its own PSU
        n_obs = len(self.weights)
        if self.strata is not None and self.n_strata > 0:
            return n_obs - self.n_strata
        return n_obs - 1

    def subset_to_units(
        self,
        row_idx: np.ndarray,
        weights: np.ndarray,
        strata: Optional[np.ndarray],
        psu: Optional[np.ndarray],
        fpc: Optional[np.ndarray],
        n_strata: int,
        n_psu: int,
    ) -> "ResolvedSurveyDesign":
        """Create a unit-level copy preserving replicate metadata.

        Used by panel estimators (ContinuousDiD, EfficientDiD) that collapse
        panel-level survey info to one row per unit. Callers that only have a
        ``row_idx`` (first panel row per unit) should prefer the
        :meth:`subset_to_units_by_row_idx` convenience wrapper, which folds the
        index-and-recount preamble.

        Parameters
        ----------
        row_idx : np.ndarray
            Indices into the panel-level arrays to select one row per unit.
        weights, strata, psu, fpc, n_strata, n_psu
            Already-subsetted TSL fields (computed by the caller).
        """
        rep_weights_sub = None
        if self.replicate_weights is not None:
            rep_weights_sub = self.replicate_weights[row_idx, :]

        return ResolvedSurveyDesign(
            weights=weights,
            weight_type=self.weight_type,
            strata=strata,
            psu=psu,
            fpc=fpc,
            n_strata=n_strata,
            n_psu=n_psu,
            lonely_psu=self.lonely_psu,
            replicate_weights=rep_weights_sub,
            replicate_method=self.replicate_method,
            fay_rho=self.fay_rho,
            n_replicates=self.n_replicates,
            replicate_strata=self.replicate_strata,
            combined_weights=self.combined_weights,
            replicate_scale=self.replicate_scale,
            replicate_rscales=self.replicate_rscales,
            mse=self.mse,
            nest=self.nest,
        )

    def subset_to_units_by_row_idx(
        self,
        row_idx: np.ndarray,
        unit_weights: Optional[np.ndarray] = None,
    ) -> "ResolvedSurveyDesign":
        """Collapse this panel-level design to unit level given a ``row_idx``.

        Convenience wrapper over :meth:`subset_to_units` that folds the
        index-and-recount preamble panel estimators (ContinuousDiD,
        EfficientDiD) would otherwise hand-roll: index ``strata`` / ``psu`` /
        ``fpc`` at ``row_idx`` (one panel row per unit) and recount
        ``n_strata`` / ``n_psu`` from the collapsed arrays.

        Parameters
        ----------
        row_idx : np.ndarray
            First panel-row position for each unit (see
            :func:`build_unit_first_row_index`).
        unit_weights : np.ndarray, optional
            Pre-collapsed unit-level weights. When ``None`` (default), uses
            ``self.weights[row_idx]`` — identical to the caller-side
            ``survey_weights[row_idx]`` collapse.
        """
        if unit_weights is None:
            unit_weights = self.weights[row_idx]
        unit_strata = self.strata[row_idx] if self.strata is not None else None
        unit_psu = self.psu[row_idx] if self.psu is not None else None
        unit_fpc = self.fpc[row_idx] if self.fpc is not None else None
        n_strata_u = len(np.unique(unit_strata)) if unit_strata is not None else 0
        n_psu_u = len(np.unique(unit_psu)) if unit_psu is not None else 0
        return self.subset_to_units(
            row_idx,
            unit_weights,
            unit_strata,
            unit_psu,
            unit_fpc,
            n_strata_u,
            n_psu_u,
        )

    @property
    def needs_survey_vcov(self) -> bool:
        """Whether survey vcov (not generic sandwich) should be used."""
        return True  # Any resolved survey design uses the survey vcov path


def make_pweight_design(weights: np.ndarray) -> "ResolvedSurveyDesign":
    """Construct a pweight-only ResolvedSurveyDesign from a raw weight array.

    Use this on the array-in HAD pretest helpers (``stute_test``,
    ``yatchew_hr_test``, ``stute_joint_pretest``) when the caller has only
    a per-observation weight array and no PSU/strata/FPC structure::

        from diff_diff import stute_test, make_pweight_design
        result = stute_test(d, dy, survey_design=make_pweight_design(w))

    For the data-in HAD surfaces (``HeterogeneousAdoptionDiD.fit``,
    ``did_had_pretest_workflow``, ``joint_pretrends_test``,
    ``joint_homogeneity_test``), prefer adding the weights as a column on
    your dataframe and passing ``SurveyDesign(weights="col_name")`` instead;
    those surfaces resolve column references against ``data`` at fit time
    (the standard library convention used by ContinuousDiD, EfficientDiD,
    and ChaisemartinDHaultfoeuille).

    Internal note: this constructs a synthetic ``ResolvedSurveyDesign`` with
    each observation as its own PSU and no strata/FPC, so PSU-level
    multiplier-bootstrap kernels reduce bit-exactly to per-observation
    Mammen draws while sharing the survey-aware code path with full PSU /
    strata / FPC designs (mirrors the PR #363 synthetic-trivial-resolved
    pattern).

    Parameters
    ----------
    weights : np.ndarray, shape (n_obs,)
        Per-observation positive weights. Must be 1-D (shape ``(n_obs,)``);
        scalars, 0-D arrays, and column-vector inputs (shape ``(n, 1)``)
        raise ``ValueError`` at the front door. Caller is responsible for
        any non-negativity / per-unit-constancy validation. Typical usage
        is positional (``make_pweight_design(arr)``); the parameter name
        ``weights`` collides linguistically with the (removed, 3.7.x)
        ``weights=`` kwarg the HAD surfaces once carried, so prefer the
        positional form.

    Returns
    -------
    ResolvedSurveyDesign
        With ``weight_type="pweight"``, ``strata=psu=fpc=None``,
        ``n_strata=0``, ``n_psu=n_obs`` (each observation is its own PSU
        under the trivial design), ``lonely_psu="remove"``,
        ``replicate_weights=None``.

    Raises
    ------
    ValueError
        If ``weights`` is not 1-D (PR #376 R3 P1: catches scalar / 0-D /
        column-vector inputs with a clear front-door message instead of
        bubbling a low-level numpy or dataclass exception).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(
            f"make_pweight_design: weights must be 1-dimensional (1-D, shape "
            f"(n_obs,)), got shape {w.shape}. Common mistakes: scalar / 0-D "
            f"input (`make_pweight_design(1.0)`); column-vector "
            f"(`make_pweight_design(df[['w']].to_numpy())` produces (n, 1) "
            f"-- use `df['w'].to_numpy()` for (n,)); 2-D matrix input."
        )
    n_obs = int(w.shape[0])
    return ResolvedSurveyDesign(
        weights=w,
        weight_type="pweight",
        strata=None,
        psu=None,
        fpc=None,
        n_strata=0,
        n_psu=n_obs,
        lonely_psu="remove",
    )


_make_trivial_resolved = make_pweight_design


@dataclass
class SurveyMetadata:
    """
    Survey design metadata stored in results objects.

    Attributes
    ----------
    weight_type : str
        Type of weights used.
    effective_n : float
        Kish effective sample size: (sum(w))^2 / sum(w^2).
    design_effect : float
        DEFF: n * sum(w^2) / (sum(w))^2.
    sum_weights : float
        Sum of original (pre-normalization) weights.
    n_strata : int or None
        Number of strata (None if unstratified).
    n_psu : int or None
        Number of PSUs (None if no PSU specified).
    weight_range : tuple of (float, float)
        (min, max) of original weights.
    df_survey : int or None
        Survey degrees of freedom (n_psu - n_strata).
    """

    weight_type: str
    effective_n: float
    design_effect: float
    sum_weights: float
    n_strata: Optional[int] = None
    n_psu: Optional[int] = None
    weight_range: Tuple[float, float] = field(default=(0.0, 0.0))
    df_survey: Optional[int] = None
    replicate_method: Optional[str] = None
    n_replicates: Optional[int] = None
    deff_diagnostics: Optional["DEFFDiagnostics"] = None


def compute_survey_metadata(
    resolved: "ResolvedSurveyDesign",
    raw_weights: np.ndarray,
) -> SurveyMetadata:
    """
    Compute survey metadata from resolved design.

    Parameters
    ----------
    resolved : ResolvedSurveyDesign
        Resolved survey design.
    raw_weights : np.ndarray
        Original (pre-normalization) weights.

    Returns
    -------
    SurveyMetadata
    """
    sum_w = float(np.sum(raw_weights))
    sum_w2 = float(np.sum(raw_weights**2))
    n = len(raw_weights)

    effective_n = sum_w**2 / sum_w2 if sum_w2 > 0 else float(n)
    design_effect = n * sum_w2 / (sum_w**2) if sum_w > 0 else 1.0

    n_strata = resolved.n_strata if resolved.strata is not None else None
    if resolved.uses_replicate_variance:
        # Replicate designs don't have meaningful PSU/strata counts
        n_psu = None
        n_strata = None
    elif resolved.psu is not None:
        n_psu = resolved.n_psu
    else:
        # Implicit PSU: each observation is its own PSU
        n_psu = len(resolved.weights)
    df_survey = resolved.df_survey

    # Replicate info
    rep_method = resolved.replicate_method if resolved.uses_replicate_variance else None
    n_rep = resolved.n_replicates if resolved.uses_replicate_variance else None

    return SurveyMetadata(
        weight_type=resolved.weight_type,
        effective_n=effective_n,
        design_effect=design_effect,
        sum_weights=sum_w,
        n_strata=n_strata,
        n_psu=n_psu,
        weight_range=(float(np.min(raw_weights)), float(np.max(raw_weights))),
        df_survey=df_survey,
        replicate_method=rep_method,
        n_replicates=n_rep,
    )


@dataclass
class DEFFDiagnostics:
    """Per-coefficient design effect diagnostics.

    Compares survey-design variance to simple random sampling (SRS)
    variance for each coefficient, giving the variance inflation factor
    due to the survey design (clustering, stratification, weighting).

    Attributes
    ----------
    deff : np.ndarray
        Per-coefficient DEFF: survey_var / srs_var. Shape (k,).
    effective_n : np.ndarray
        Effective sample size per coefficient: n / DEFF. Shape (k,).
    srs_se : np.ndarray
        SRS (HC1) standard errors. Shape (k,).
    survey_se : np.ndarray
        Survey standard errors. Shape (k,).
    coefficient_names : list of str or None
        Names for display.
    """

    deff: np.ndarray
    effective_n: np.ndarray
    srs_se: np.ndarray
    survey_se: np.ndarray
    coefficient_names: Optional[List[str]] = None


def compute_deff_diagnostics(
    X: np.ndarray,
    residuals: np.ndarray,
    survey_vcov: np.ndarray,
    weights: np.ndarray,
    weight_type: str = "pweight",
    coefficient_names: Optional[List[str]] = None,
) -> DEFFDiagnostics:
    """Compute per-coefficient design effects.

    Compares the survey variance-covariance matrix to a simple random
    sampling (SRS) baseline (HC1 sandwich, ignoring strata/PSU/FPC).

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from the WLS fit, shape (n,).
    survey_vcov : np.ndarray
        Survey variance-covariance matrix, shape (k, k).
    weights : np.ndarray
        Observation weights (normalized), shape (n,).
    weight_type : str, default "pweight"
        Weight type for SRS computation.
    coefficient_names : list of str, optional
        Names for display.

    Returns
    -------
    DEFFDiagnostics
    """
    from diff_diff.linalg import compute_robust_vcov

    n = X.shape[0]
    # Use positive-weight count for effective n (zero-weight rows from
    # subpopulation don't contribute to the effective sample)
    n_eff = int(np.count_nonzero(weights > 0)) if np.any(weights == 0) else n

    # SRS baseline: HC1 weighted sandwich ignoring design structure
    srs_vcov = compute_robust_vcov(
        X,
        residuals,
        cluster_ids=None,
        weights=weights,
        weight_type=weight_type,
    )

    survey_var = np.diag(survey_vcov)
    srs_var = np.diag(srs_vcov)

    # DEFF = survey_var / srs_var
    with np.errstate(divide="ignore", invalid="ignore"):
        deff = np.where(srs_var > 0, survey_var / srs_var, np.nan)
        eff_n = np.where(deff > 0, n_eff / deff, np.nan)

    survey_se = np.sqrt(np.maximum(survey_var, 0.0))
    srs_se = np.sqrt(np.maximum(srs_var, 0.0))

    return DEFFDiagnostics(
        deff=deff,
        effective_n=eff_n,
        srs_se=srs_se,
        survey_se=survey_se,
        coefficient_names=coefficient_names,
    )


def _validate_unit_constant_survey(data, unit_col, survey_design):
    """Validate that survey design columns are constant within units.

    Panel estimators (ContinuousDiD, EfficientDiD) collapse panel-level
    survey info to one row per unit. This requires that survey columns
    do not vary across time periods within a unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    unit_col : str
        Unit identifier column name.
    survey_design : SurveyDesign
        Survey design specification (uses attribute names, not resolved arrays).

    Raises
    ------
    ValueError
        If any survey column varies within units.
    """
    cols_to_check = [
        survey_design.weights,
        survey_design.strata,
        survey_design.psu,
        survey_design.fpc,
    ]
    # Also validate replicate weight columns for within-unit constancy
    if survey_design.replicate_weights is not None:
        cols_to_check.extend(survey_design.replicate_weights)
    for col in cols_to_check:
        if col is not None and col in data.columns:
            n_unique = data.groupby(unit_col)[col].nunique()
            varying_units = n_unique[n_unique > 1]
            if len(varying_units) > 0:
                raise ValueError(
                    f"Survey column '{col}' varies within units "
                    f"(found {len(varying_units)} units with multiple values). "
                    f"Panel estimators require survey design columns to be "
                    f"constant within units."
                )


def _resolve_pweight_only(resolved_survey, estimator_name):
    """Guard: reject non-pweight and strata/PSU/FPC for pweight-only estimators.

    Parameters
    ----------
    resolved_survey : ResolvedSurveyDesign or None
        Resolved survey design. If None, returns immediately.
    estimator_name : str
        Estimator name for error messages.

    Raises
    ------
    ValueError
        If weight_type is not 'pweight'.
    NotImplementedError
        If strata, PSU, or FPC are present.
    """
    if resolved_survey is None:
        return
    if resolved_survey.weight_type != "pweight":
        raise ValueError(
            f"{estimator_name} survey support requires weight_type='pweight'. "
            f"Got '{resolved_survey.weight_type}'."
        )
    if (
        resolved_survey.strata is not None
        or resolved_survey.psu is not None
        or resolved_survey.fpc is not None
    ):
        raise NotImplementedError(
            f"{estimator_name} does not yet support strata/PSU/FPC in "
            "SurveyDesign. Use SurveyDesign(weights=...) only. Full "
            "design-based bootstrap is planned for the Bootstrap + "
            "Survey Interaction phase."
        )


def build_unit_first_row_index(unit_values: np.ndarray, unit_order: Sequence[Any]) -> np.ndarray:
    """Positional index of each unit's first row, aligned to ``unit_order``.

    Panel estimators that collapse a panel-level survey design to unit level
    (ContinuousDiD, EfficientDiD) need, for each unit, the position of its first
    row in the fit DataFrame so panel-length survey arrays can be indexed down
    to one row per unit. ``unit_values`` is the fit frame's unit column
    (``df[unit].values``, in row order); ``unit_order`` is the estimator's
    canonical unit ordering (typically ``all_units = sorted(df[unit].unique())``).

    Returns an ``int`` array ``idx`` where ``idx[i]`` is the first row position
    of ``unit_order[i]``.
    """
    first_pos: dict = {}
    for i, u in enumerate(unit_values):
        if u not in first_pos:
            first_pos[u] = i
    return np.array([first_pos[u] for u in unit_order], dtype=int)


def collapse_survey_to_unit_level(resolved_survey, df, unit_col, all_units):
    """Collapse observation-level ResolvedSurveyDesign to unit level.

    Panel estimators with influence-function-based variance need a
    unit-level survey design (one row per unit) rather than the
    observation-level design (one row per unit-time).  Survey design
    columns must be constant within units (validated upstream via
    ``_validate_unit_constant_survey``).

    Parameters
    ----------
    resolved_survey : ResolvedSurveyDesign
        Observation-level resolved survey design.
    df : pd.DataFrame
        Panel data used for groupby operations.
    unit_col : str
        Unit identifier column name.
    all_units : array-like
        Ordered sequence of unique unit identifiers.

    Returns
    -------
    ResolvedSurveyDesign
        Unit-level design with arrays indexed by ``all_units``.
    """
    n_units = len(all_units)

    weights_unit = (
        pd.Series(resolved_survey.weights, index=df.index)
        .groupby(df[unit_col])
        .first()
        .reindex(all_units)
        .values
    )

    strata_unit = None
    if resolved_survey.strata is not None:
        strata_unit = (
            pd.Series(resolved_survey.strata, index=df.index)
            .groupby(df[unit_col])
            .first()
            .reindex(all_units)
            .values
        )

    psu_unit = None
    if resolved_survey.psu is not None:
        psu_unit = (
            pd.Series(resolved_survey.psu, index=df.index)
            .groupby(df[unit_col])
            .first()
            .reindex(all_units)
            .values
        )

    fpc_unit = None
    if resolved_survey.fpc is not None:
        fpc_unit = (
            pd.Series(resolved_survey.fpc, index=df.index)
            .groupby(df[unit_col])
            .first()
            .reindex(all_units)
            .values
        )

    # Collapse replicate weights to unit level (same groupby pattern)
    rep_weights_unit = None
    if resolved_survey.replicate_weights is not None:
        R = resolved_survey.replicate_weights.shape[1]
        rep_weights_unit = np.zeros((n_units, R))
        for r in range(R):
            rep_weights_unit[:, r] = (
                pd.Series(resolved_survey.replicate_weights[:, r], index=df.index)
                .groupby(df[unit_col])
                .first()
                .reindex(all_units)
                .values
            )

    return ResolvedSurveyDesign(
        weights=weights_unit.astype(np.float64),
        weight_type=resolved_survey.weight_type,
        strata=strata_unit,
        psu=psu_unit,
        fpc=fpc_unit,
        n_strata=resolved_survey.n_strata,
        n_psu=resolved_survey.n_psu,
        lonely_psu=resolved_survey.lonely_psu,
        replicate_weights=rep_weights_unit,
        replicate_method=resolved_survey.replicate_method,
        fay_rho=resolved_survey.fay_rho,
        n_replicates=resolved_survey.n_replicates,
        replicate_strata=resolved_survey.replicate_strata,
        combined_weights=resolved_survey.combined_weights,
        replicate_scale=resolved_survey.replicate_scale,
        replicate_rscales=resolved_survey.replicate_rscales,
        mse=resolved_survey.mse,
        nest=resolved_survey.nest,
    )


def _extract_unit_survey_weights(data, unit_col, survey_design, unit_order):
    """Extract unit-level survey weights aligned to a given unit ordering.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with survey weight column.
    unit_col : str
        Unit identifier column name.
    survey_design : SurveyDesign
        Survey design. When ``survey_design.weights`` is a column name,
        the weights are pulled from ``data``. When ``survey_design.weights
        is None`` (a valid configuration — ``SurveyDesign.resolve()`` then
        synthesizes ones), returns a vector of ones of length
        ``len(unit_order)`` so downstream estimators can treat all units
        as having unit survey weight 1.
    unit_order : array-like
        Ordered sequence of unit identifiers to align weights to.

    Returns
    -------
    np.ndarray
        Float64 array of unit-level weights, one per unit in ``unit_order``.
    """
    if survey_design.weights is None:
        # SurveyDesign(weights=None, strata=..., psu=...) is a valid
        # configuration — the design element supplies clustering /
        # stratification without explicit per-unit weights. Synthesize
        # uniform unit weights of 1 to match SurveyDesign.resolve()'s
        # behavior (which emits ones when weights is None). Without this
        # branch the groupby below would raise a KeyError on ``None``.
        return np.ones(len(unit_order), dtype=np.float64)
    unit_w = data.groupby(unit_col)[survey_design.weights].first()
    return np.array([unit_w[u] for u in unit_order], dtype=np.float64)


def _resolve_survey_for_fit(
    survey_design: Optional["SurveyDesign"],
    data: pd.DataFrame,
    inference_mode: str = "analytical",
) -> Tuple[
    Optional["ResolvedSurveyDesign"],
    Optional[np.ndarray],
    str,
    Optional["SurveyMetadata"],
]:
    """
    Shared helper: validate and resolve a SurveyDesign for an estimator fit() call.

    Returns (resolved, weights, weight_type, metadata) or all-None tuple if
    survey_design is None.
    """
    if survey_design is None:
        return None, None, "pweight", None

    if not isinstance(survey_design, SurveyDesign):
        raise TypeError("survey_design must be a SurveyDesign instance")

    if inference_mode == "wild_bootstrap":
        raise NotImplementedError(
            "Wild bootstrap with survey weights is not yet supported. "
            "Use analytical survey inference (the default) instead."
        )

    resolved = survey_design.resolve(data)
    raw_w = (
        data[survey_design.weights].values.astype(np.float64)
        if survey_design.weights is not None
        else np.ones(len(data), dtype=np.float64)
    )
    metadata = compute_survey_metadata(resolved, raw_w)
    return resolved, resolved.weights, resolved.weight_type, metadata


def _resolve_effective_cluster(resolved_survey, cluster_ids, cluster_name=None):
    """
    Shared helper: determine effective cluster IDs for variance estimation.

    When survey PSU is present, it overrides the user-specified cluster.
    Warns if both are specified with different groupings.
    """
    if resolved_survey is None or resolved_survey.psu is None:
        return cluster_ids

    if cluster_ids is not None and cluster_name is not None:
        # Compare partition equivalence (not label equality)
        psu_codes, _ = pd.factorize(resolved_survey.psu)
        cluster_codes, _ = pd.factorize(cluster_ids)
        if not np.array_equal(psu_codes, cluster_codes):
            warnings.warn(
                f"Both survey_design.psu and cluster='{cluster_name}' specified "
                "with different groupings. PSU will be used for variance "
                "estimation (survey design-based inference).",
                UserWarning,
                stacklevel=3,
            )
    return resolved_survey.psu


def _inject_cluster_as_psu(resolved, cluster_ids):
    """
    When survey design has no PSU but cluster_ids are provided,
    inject cluster_ids as the effective PSU for TSL variance estimation.

    Honors ``resolved.nest`` matching the explicit-PSU resolver at
    ``SurveyDesign.resolve()`` (L299-L318):
      - ``nest=True`` (or no strata): nest cluster IDs within strata via
        ``f"{s}_{c}"`` so repeated cluster labels across strata become
        distinct PSUs.
      - ``nest=False`` AND strata present: cluster labels must be
        globally unique (no overlap across strata); raise if they
        repeat, mirroring the explicit-PSU `nest=False` contract.

    Returns a new ResolvedSurveyDesign (no mutation) or the original unchanged.
    """
    if resolved is None or cluster_ids is None:
        return resolved
    if resolved.psu is not None:
        return resolved  # PSU already present; _resolve_effective_cluster handles this

    # Validate no missing cluster IDs before factorization
    if pd.isna(cluster_ids).any():
        raise ValueError(
            "Cluster IDs contain missing values. "
            "All observations must have valid cluster identifiers "
            "when used as effective PSUs for survey variance estimation."
        )

    if resolved.strata is not None:
        if resolved.nest:
            # Nest cluster IDs within strata: combined `(stratum, cluster)`
            # labels are globally unique by construction.
            combined = np.array([f"{s}_{c}" for s, c in zip(resolved.strata, cluster_ids)])
            codes, uniques = pd.factorize(combined)
        else:
            # nest=False contract: cluster labels must be globally unique
            # across strata (same gate as `SurveyDesign.resolve()` L305-L316).
            # Validate by checking that each cluster label appears in
            # exactly one stratum.
            df_check = pd.DataFrame({"stratum": resolved.strata, "cluster": cluster_ids})
            overlap = df_check.groupby("cluster")["stratum"].nunique()
            ambiguous = overlap[overlap > 1]
            if len(ambiguous) > 0:
                bad_examples = list(ambiguous.index[:5])
                raise ValueError(
                    f"Cluster IDs repeat across strata under nest=False: "
                    f"{len(ambiguous)} cluster label(s) appear in multiple "
                    f"strata (examples: {bad_examples}). Either set "
                    f"nest=True in SurveyDesign so cluster IDs are made "
                    f"unique within strata, or pass an explicit unique "
                    f"`psu=<col>` to SurveyDesign."
                )
            codes, uniques = pd.factorize(cluster_ids)
    else:
        codes, uniques = pd.factorize(cluster_ids)
    n_clusters = len(uniques)

    return replace(resolved, psu=codes, n_psu=n_clusters)


def _compute_stratified_psu_meat(
    scores: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> tuple:
    """Compute the stratified PSU-level meat matrix for TSL variance.

    This is the core computation shared by :func:`compute_survey_vcov`
    (which wraps it in a sandwich with the bread matrix) and
    :func:`compute_survey_if_variance` (which uses it directly for
    influence-function-based estimators).

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape (n, k). For OLS-based estimators these are
        the weighted score contributions X_i * w_i * u_i.  For IF-based
        estimators these are the per-unit influence function values
        (reshaped to (n, 1) for scalar estimators).
    resolved : ResolvedSurveyDesign
        Resolved survey design with weights, strata, PSU arrays.

    Returns
    -------
    meat : np.ndarray
        Meat matrix of shape (k, k).
    variance_computed : bool
        Whether any actual variance computation happened.
    legitimate_zero_count : int
        Number of strata/sources that legitimately contribute zero variance.
    """
    n = scores.shape[0]
    k = scores.shape[1] if scores.ndim > 1 else 1
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    strata = resolved.strata
    psu = resolved.psu

    legitimate_zero_count = 0
    _variance_computed = False

    if strata is None and psu is None:
        # No survey structure beyond weights — implicit per-observation PSUs
        psu_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - psu_mean
        f_h = 0.0
        if resolved.fpc is not None:
            N_h = resolved.fpc[0]
            if N_h < n:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of observations "
                    f"({n}). FPC must be >= n_obs for implicit per-observation PSUs."
                )
            f_h = n / N_h
            if f_h >= 1.0:
                legitimate_zero_count += 1
        adjustment = (1.0 - f_h) * (n / (n - 1))
        meat = adjustment * (centered.T @ centered)
        _variance_computed = True
    elif strata is None and psu is not None:
        # No strata, but PSU present — single-stratum cluster-robust
        psu_scores = pd.DataFrame(scores).groupby(psu).sum().values
        n_psu = psu_scores.shape[0]

        if n_psu < 2:
            meat = np.zeros((k, k))
        else:
            psu_mean = psu_scores.mean(axis=0, keepdims=True)
            centered = psu_scores - psu_mean
            f_h = 0.0
            if resolved.fpc is not None:
                N_h = resolved.fpc[0]
                if N_h < n_psu:
                    raise ValueError(
                        f"FPC ({N_h}) is less than the number of effective PSUs "
                        f"({n_psu}). FPC must be >= n_PSU."
                    )
                f_h = n_psu / N_h
                if f_h >= 1.0:
                    legitimate_zero_count += 1
            adjustment = (1.0 - f_h) * (n_psu / (n_psu - 1))
            meat = adjustment * (centered.T @ centered)
            _variance_computed = True
    else:
        # Stratified with or without PSU
        unique_strata = np.unique(strata)
        meat = np.zeros((k, k))

        _global_psu_mean = None
        if resolved.lonely_psu == "adjust":
            if psu is not None:
                _global_psu_mean = (
                    pd.DataFrame(scores).groupby(psu).sum().values.mean(axis=0, keepdims=True)
                )
            else:
                _global_psu_mean = scores.mean(axis=0, keepdims=True)

        for h in unique_strata:
            mask_h = strata == h

            if psu is not None:
                psu_h = psu[mask_h]
                scores_h = scores[mask_h]
                psu_scores_h = pd.DataFrame(scores_h).groupby(psu_h).sum().values
                n_psu_h = psu_scores_h.shape[0]
            else:
                psu_scores_h = scores[mask_h]
                n_psu_h = psu_scores_h.shape[0]

            # Handle singleton strata
            if n_psu_h < 2:
                if resolved.lonely_psu == "remove":
                    continue
                elif resolved.lonely_psu == "certainty":
                    legitimate_zero_count += 1
                    continue
                elif resolved.lonely_psu == "adjust":
                    centered = psu_scores_h - _global_psu_mean
                    V_h = centered.T @ centered
                    meat += V_h
                    _variance_computed = True
                    continue

            # FPC
            f_h = 0.0
            if resolved.fpc is not None:
                N_h = resolved.fpc[mask_h][0]
                if N_h < n_psu_h:
                    raise ValueError(
                        f"FPC ({N_h}) is less than the number of effective PSUs "
                        f"({n_psu_h}) in stratum. FPC must be >= n_PSU."
                    )
                f_h = n_psu_h / N_h
                if f_h >= 1.0:
                    legitimate_zero_count += 1

            psu_mean_h = psu_scores_h.mean(axis=0, keepdims=True)
            centered = psu_scores_h - psu_mean_h

            adjustment = (1.0 - f_h) * (n_psu_h / (n_psu_h - 1))
            V_h = adjustment * (centered.T @ centered)
            meat += V_h
            _variance_computed = True

    return meat, _variance_computed, legitimate_zero_count


@dataclass(frozen=True)
class _PsuScaffolding:
    """Precomputed stratum/PSU layout for amortized TSL variance.

    Internal helper used by :func:`diff_diff.prep.aggregate_survey` to reuse
    design-dependent scaffolding across hundreds of per-cell variance calls.
    Holds integer codes, per-stratum counts, FPC ratios, and static
    variance-computability flags that depend only on the
    :class:`ResolvedSurveyDesign` (not on the psi / outcome being collapsed).

    See :func:`_compute_if_variance_fast` for the fast variance path that
    consumes this scaffolding.  Numerically equivalent to
    :func:`compute_survey_if_variance` up to sub-ULP reduction-order drift.
    """

    mode: str  # "no_strata_no_psu" | "psu_only" | "stratified"
    n: int
    lonely_psu: str
    variance_computable: bool
    legitimate_zero_count: int
    # stratified-mode fields (None in other modes):
    psu_codes: Optional[np.ndarray] = None  # (n,) int, global PSU id 0..P-1
    psu_stratum: Optional[np.ndarray] = None  # (P,) int, stratum of each PSU
    n_psu_per_stratum: Optional[np.ndarray] = None  # (S,) int
    singleton_strata: Optional[np.ndarray] = None  # (S,) bool
    adjustment_h: Optional[np.ndarray] = None  # (S,) float, (1-f_h)*n_h/(n_h-1); 0 for singletons
    # psu_only-mode fields (None in other modes):
    psu_codes_only: Optional[np.ndarray] = None  # (n,) int, PSU id 0..P-1
    n_psu_only: Optional[int] = None
    adjustment_only: Optional[float] = None  # (1-f)*n_psu/(n_psu-1) or 0
    # no_strata_no_psu-mode fields (None in other modes):
    adjustment_direct: Optional[float] = None  # (1-f)*n/(n-1) or 0


def _precompute_psu_scaffolding(resolved: "ResolvedSurveyDesign") -> _PsuScaffolding:
    """Precompute per-design PSU/stratum scaffolding for fast per-cell variance.

    Equivalent in effect to the per-call scaffolding work inside
    :func:`_compute_stratified_psu_meat`, but done once per design instead of
    once per output cell.  For the typical BRFSS-scale
    :func:`~diff_diff.prep.aggregate_survey` workload (~500 cells, ~20 strata),
    this amortizes the pandas-groupby + ``np.unique`` setup that otherwise
    dominates the chain runtime.

    Parameters
    ----------
    resolved : ResolvedSurveyDesign
        Resolved survey design.  Must NOT use replicate variance
        (``resolved.uses_replicate_variance`` False).

    Returns
    -------
    _PsuScaffolding
        Frozen dataclass with mode-appropriate precomputed fields.

    Raises
    ------
    ValueError
        Same FPC-vs-n guards as :func:`_compute_stratified_psu_meat`
        (FPC must be >= effective PSU count in each stratum).
    """
    weights = resolved.weights
    n = int(len(weights))
    strata = resolved.strata
    psu = resolved.psu
    fpc = resolved.fpc
    lonely_psu = resolved.lonely_psu

    if strata is None and psu is None:
        # Implicit per-observation PSUs
        f = 0.0
        lz_count = 0
        if fpc is not None:
            N = fpc[0]
            if N < n:
                raise ValueError(
                    f"FPC ({N}) is less than the number of observations "
                    f"({n}). FPC must be >= n_obs for implicit per-observation PSUs."
                )
            f = n / N
            if f >= 1.0:
                lz_count = 1
        var_computable = n >= 2
        adjustment = (1.0 - f) * (n / (n - 1)) if n >= 2 else 0.0
        return _PsuScaffolding(
            mode="no_strata_no_psu",
            n=n,
            lonely_psu=lonely_psu,
            variance_computable=var_computable,
            legitimate_zero_count=lz_count,
            adjustment_direct=float(adjustment),
        )

    if strata is None and psu is not None:
        # Single-stratum cluster-robust
        psu_arr = np.asarray(psu)
        codes, uniques = pd.factorize(psu_arr)
        n_psu = int(len(uniques))
        f = 0.0
        lz_count = 0
        if n_psu >= 2:
            if fpc is not None:
                N = fpc[0]
                if N < n_psu:
                    raise ValueError(
                        f"FPC ({N}) is less than the number of effective PSUs "
                        f"({n_psu}). FPC must be >= n_PSU."
                    )
                f = n_psu / N
                if f >= 1.0:
                    lz_count = 1
            adjustment = (1.0 - f) * (n_psu / (n_psu - 1))
            var_computable = True
        else:
            adjustment = 0.0
            var_computable = False
        return _PsuScaffolding(
            mode="psu_only",
            n=n,
            lonely_psu=lonely_psu,
            variance_computable=var_computable,
            legitimate_zero_count=lz_count,
            psu_codes_only=codes.astype(np.int64),
            n_psu_only=n_psu,
            adjustment_only=float(adjustment),
        )

    # Stratified branch (with or without PSU)
    strata_arr = np.asarray(strata)
    strata_codes, strata_uniques = pd.factorize(strata_arr, sort=True)
    strata_codes = strata_codes.astype(np.int64)
    S = int(len(strata_uniques))

    if psu is not None:
        # Global PSU codes unique across (stratum, psu) pairs — matches the
        # legacy per-stratum pandas groupby which never aggregated PSU labels
        # across strata.
        psu_arr = np.asarray(psu)
        psu_local_codes, _ = pd.factorize(psu_arr)
        psu_local_codes = psu_local_codes.astype(np.int64)
        psu_local_max = int(psu_local_codes.max()) if len(psu_local_codes) > 0 else 0
        compound = strata_codes * (psu_local_max + 1) + psu_local_codes
        psu_codes, _ = pd.factorize(compound)
        psu_codes = psu_codes.astype(np.int64)
        P = int(psu_codes.max() + 1) if len(psu_codes) > 0 else 0
        psu_stratum = np.zeros(P, dtype=np.int64)
        # Safe scatter: by construction, all observations sharing a global
        # PSU code share a stratum, so repeated writes to the same position
        # store the same value.
        if P > 0:
            psu_stratum[psu_codes] = strata_codes
    else:
        # Each observation is its own PSU within its stratum (legacy
        # behavior when strata is not None and psu is None).
        psu_codes = np.arange(n, dtype=np.int64)
        P = n
        psu_stratum = strata_codes.copy()

    n_psu_per_stratum = np.bincount(psu_stratum, minlength=S).astype(np.int64)
    singleton_strata = n_psu_per_stratum == 1

    # Per-stratum FPC ratio (stratum-level attribute; read from the first
    # observation of each stratum, matching legacy ``resolved.fpc[mask_h][0]``).
    f_h = np.zeros(S, dtype=np.float64)
    if fpc is not None:
        fpc_arr = np.asarray(fpc)
        # Vectorized "first-in-stratum" FPC lookup:
        # pd.factorize with sort=True iterates the array in input order, so
        # the first observation encountered for each stratum_code is the
        # reference row.
        first_idx = np.full(S, -1, dtype=np.int64)
        seen = np.zeros(S, dtype=bool)
        for i in range(n):
            h = strata_codes[i]
            if not seen[h]:
                seen[h] = True
                first_idx[h] = i
                if seen.all():
                    break
        for h in range(S):
            if first_idx[h] < 0:
                continue
            N_h = fpc_arr[first_idx[h]]
            n_h = n_psu_per_stratum[h]
            if n_h > 0 and N_h < n_h:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of effective PSUs "
                    f"({n_h}) in stratum. FPC must be >= n_PSU."
                )
            if n_h > 0:
                f_h[h] = n_h / N_h

    with np.errstate(divide="ignore", invalid="ignore"):
        adjustment_h = np.where(
            n_psu_per_stratum >= 2,
            (1.0 - f_h) * n_psu_per_stratum / np.maximum(n_psu_per_stratum - 1, 1),
            0.0,
        )

    # Static legitimate_zero_count (design-dependent only):
    #   - Non-singleton strata with f_h >= 1.0 contribute (legacy counter).
    #   - Singleton strata under lonely_psu == "certainty" contribute.
    fpc_saturated = (n_psu_per_stratum >= 2) & (f_h >= 1.0)
    legitimate_zero_count = int(fpc_saturated.sum())
    if lonely_psu == "certainty":
        legitimate_zero_count += int(singleton_strata.sum())

    # Static variance_computable flag:
    #   - Any non-singleton stratum (regardless of FPC) → variance_computed=True
    #     path is exercised.
    #   - Under "adjust", any singleton stratum also counts (adds V_h even if 0).
    has_non_singleton = bool(np.any(~singleton_strata))
    has_singleton = bool(np.any(singleton_strata))
    variance_computable = has_non_singleton or (lonely_psu == "adjust" and has_singleton)

    return _PsuScaffolding(
        mode="stratified",
        n=n,
        lonely_psu=lonely_psu,
        variance_computable=variance_computable,
        legitimate_zero_count=legitimate_zero_count,
        psu_codes=psu_codes,
        psu_stratum=psu_stratum,
        n_psu_per_stratum=n_psu_per_stratum,
        singleton_strata=singleton_strata,
        adjustment_h=adjustment_h,
    )


def _compute_if_variance_fast(
    psi: np.ndarray,
    scaffolding: _PsuScaffolding,
) -> float:
    """Fast TSL variance for aggregate_survey using precomputed scaffolding.

    Numerically equivalent to :func:`compute_survey_if_variance` for any
    TSL (non-replicate) design, up to sub-ULP reduction-order drift.  The
    speedup comes from replacing per-cell pandas groupbys and per-stratum
    Python loops with two ``np.bincount`` passes plus a fully vectorized
    per-stratum reduction.

    Parameters
    ----------
    psi : np.ndarray
        Per-unit influence function values, shape (n,).
    scaffolding : _PsuScaffolding
        Precomputed via :func:`_precompute_psu_scaffolding` for the same
        resolved design.

    Returns
    -------
    float
        Design-based variance.  Returns ``np.nan`` when variance is
        unidentified (matches legacy behavior).
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()

    def _finalize(meat_scalar: float) -> float:
        if meat_scalar == 0.0:
            if scaffolding.variance_computable or scaffolding.legitimate_zero_count > 0:
                return 0.0
            return float("nan")
        return meat_scalar

    if scaffolding.mode == "no_strata_no_psu":
        # Mode invariant: the scaffolding builder fills this field for this mode.
        assert scaffolding.adjustment_direct is not None
        if scaffolding.n < 2:
            return float("nan")
        psi_mean = psi.mean()
        centered = psi - psi_mean
        meat = scaffolding.adjustment_direct * float(centered @ centered)
        return _finalize(meat)

    if scaffolding.mode == "psu_only":
        # Mode invariant: the scaffolding builder fills these fields for this mode.
        assert scaffolding.n_psu_only is not None and scaffolding.adjustment_only is not None
        if scaffolding.n_psu_only < 2:
            if scaffolding.legitimate_zero_count > 0:
                return 0.0
            return float("nan")
        psu_sums = np.bincount(
            scaffolding.psu_codes_only, weights=psi, minlength=scaffolding.n_psu_only
        )
        psu_mean = psu_sums.mean()
        centered = psu_sums - psu_mean
        meat = scaffolding.adjustment_only * float(centered @ centered)
        return _finalize(meat)

    # Stratified
    S = len(scaffolding.n_psu_per_stratum)
    P = len(scaffolding.psu_stratum)

    # Mode invariant: the stratified scaffolding builder fills these fields.
    assert scaffolding.n_psu_per_stratum is not None and scaffolding.adjustment_h is not None
    psu_sums = np.bincount(scaffolding.psu_codes, weights=psi, minlength=P)
    sum_by_h = np.bincount(scaffolding.psu_stratum, weights=psu_sums, minlength=S)
    sum2_by_h = np.bincount(scaffolding.psu_stratum, weights=psu_sums * psu_sums, minlength=S)

    with np.errstate(divide="ignore", invalid="ignore"):
        centered_ss = np.where(
            scaffolding.n_psu_per_stratum >= 2,
            sum2_by_h - (sum_by_h * sum_by_h) / np.maximum(scaffolding.n_psu_per_stratum, 1),
            0.0,
        )
    meat_per_stratum = scaffolding.adjustment_h * centered_ss

    if np.any(scaffolding.singleton_strata) and scaffolding.lonely_psu == "adjust":
        # Singleton strata under "adjust": V_h = (psu_sum - global_mean)^2.
        # For a singleton stratum, the one PSU's sum equals sum_by_h[h].
        # No FPC, no (n-1) adjustment — matches legacy (survey.py:1276-1281).
        if P > 0:
            global_mean = psu_sums.mean()
            singleton_meat = (sum_by_h - global_mean) ** 2
            meat_per_stratum = np.where(
                scaffolding.singleton_strata, singleton_meat, meat_per_stratum
            )

    meat = float(meat_per_stratum.sum())
    return _finalize(meat)


def _compute_stratified_meat_from_psu_scores(
    psu_scores: np.ndarray,
    psu_strata: np.ndarray,
    fpc_per_psu: "Optional[np.ndarray]" = None,
    lonely_psu: str = "remove",
) -> np.ndarray:
    """Compute stratified meat matrix from pre-aggregated PSU-level scores.

    Like :func:`_compute_stratified_psu_meat`, but accepts scores that are
    already aggregated to the PSU level (one row per PSU). Used by
    TwoStageDiD's GMM sandwich where the score matrix ``S`` is built at
    the cluster/PSU level.

    Parameters
    ----------
    psu_scores : np.ndarray
        Score matrix of shape (G, k) — one row per PSU.
    psu_strata : np.ndarray
        Stratum assignment per PSU, shape (G,).
    fpc_per_psu : np.ndarray, optional
        FPC population size per PSU, shape (G,). All PSUs in the same
        stratum should share the same FPC value (first occurrence used).
    lonely_psu : str
        How to handle singleton strata: "remove", "certainty", or "adjust".

    Returns
    -------
    meat : np.ndarray
        Meat matrix of shape (k, k).
    variance_computed : bool
        Whether any actual variance computation happened.
    legitimate_zero_count : int
        Number of strata that legitimately contribute zero variance.
    """
    if psu_scores.ndim == 1:
        psu_scores = psu_scores[:, np.newaxis]
    k = psu_scores.shape[1]
    meat = np.zeros((k, k))

    unique_strata = np.unique(psu_strata)
    _variance_computed = False
    legitimate_zero_count = 0

    # Pre-compute global mean for lonely_psu="adjust"
    _global_psu_mean = None
    if lonely_psu == "adjust":
        _global_psu_mean = psu_scores.mean(axis=0, keepdims=True)

    for h in unique_strata:
        mask_h = psu_strata == h
        scores_h = psu_scores[mask_h]
        n_psu_h = scores_h.shape[0]

        # Handle singleton strata
        if n_psu_h < 2:
            if lonely_psu == "remove":
                continue
            elif lonely_psu == "certainty":
                legitimate_zero_count += 1
                continue
            elif lonely_psu == "adjust":
                centered = scores_h - _global_psu_mean
                with np.errstate(invalid="ignore", over="ignore"):
                    meat += centered.T @ centered
                _variance_computed = True
                continue

        # FPC
        f_h = 0.0
        if fpc_per_psu is not None:
            N_h = fpc_per_psu[mask_h][0]
            if N_h < n_psu_h:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of PSUs "
                    f"({n_psu_h}) in stratum. FPC must be >= n_PSU."
                )
            f_h = n_psu_h / N_h
            if f_h >= 1.0:
                legitimate_zero_count += 1

        psu_mean_h = scores_h.mean(axis=0, keepdims=True)
        centered = scores_h - psu_mean_h

        adjustment = (1.0 - f_h) * (n_psu_h / (n_psu_h - 1))
        with np.errstate(invalid="ignore", over="ignore"):
            meat += adjustment * (centered.T @ centered)
        _variance_computed = True

    return meat, _variance_computed, legitimate_zero_count


def _compute_stratified_conley_meat_from_psu_scores(
    psu_scores: np.ndarray,
    psu_strata: np.ndarray,
    psu_coords: np.ndarray,
    *,
    cutoff: float,
    metric,
    kernel: str,
    fpc_per_psu: "Optional[np.ndarray]" = None,
    lonely_psu: str = "remove",
) -> Tuple[np.ndarray, bool, int]:
    """Wave E.2 stratified-Conley meat on PSU-aggregated scores.

    Composes Conley (1999) spatial-HAC with Gerber (2026, arXiv:2605.04124)
    Proposition 1 Binder TSL (the Wave E.1 foundation) and the Wave D
    Gardner GMM first-stage uncertainty correction (Butts 2021 ss3.1 +
    Gardner 2022 ss4). Used by SpilloverDiD's Wave E.2 GMM sandwich when
    ``vcov_type="conley"`` is combined with ``survey_design=``.

    Per-stratum loop: demean PSU scores within the stratum, apply the
    cross-sectional Conley kernel between PSU centroids in that stratum,
    scale by the Binder finite-population correction
    ``(1 - f_h) * n_h/(n_h-1)``. Cross-stratum kernel weights are zero by
    sampling design (strata are exact independence partitions); total meat
    is the sum across strata.

    Parameters
    ----------
    psu_scores : np.ndarray
        Score matrix of shape (G, k) — one row per PSU.
    psu_strata : np.ndarray
        Stratum assignment per PSU, shape (G,).
    psu_coords : np.ndarray
        Per-PSU spatial centroid coordinates, shape (G, 2). Typically the
        mean of per-observation ``conley_coords`` within each PSU.
    cutoff : float
        Conley spatial-HAC bandwidth in the same units as ``psu_coords``
        (km when ``metric="haversine"``).
    metric : str or callable
        Distance metric; ``"haversine"`` / ``"euclidean"`` / callable per
        :mod:`diff_diff.conley` (``ConleyMetric``).
    kernel : str
        Spatial kernel: ``"bartlett"`` or ``"uniform"``.
    fpc_per_psu : np.ndarray, optional
        FPC population size per PSU, shape (G,). All PSUs in the same
        stratum should share the same FPC value (first occurrence used).
    lonely_psu : str
        How to handle singleton strata: ``"remove"``, ``"certainty"``, or
        ``"adjust"``. Matches the existing
        :func:`_compute_stratified_meat_from_psu_scores` behaviour exactly,
        including the ``"adjust"`` branch's ``continue`` that skips FPC
        scaling (with ``n_h=1`` the scale ``n_h/(n_h-1)`` would divide by
        zero).

    Returns
    -------
    meat : np.ndarray
        Meat matrix of shape (k, k).
    variance_computed : bool
        Whether any actual variance computation happened.
    legitimate_zero_count : int
        Number of strata that legitimately contribute zero variance.

    Notes
    -----
    Reduction semantics (load-bearing for tests):

    - bandwidth -> 0 (Bartlett: ``K(d/tiny) = 0`` for ``d > 0`` and
      ``K(0) = 1`` on the diagonal so K is the identity matrix): the
      within-stratum sandwich ``sum_{j,k} K_jk c_j c_k' = sum_j c_j c_j'
      = centered.T @ centered``, which is precisely Binder's formula at
      :func:`_compute_stratified_meat_from_psu_scores`.
    - Single stratum (H = 1, FPC = inf): reduces to ordinary Conley
      sandwich on PSU totals via :func:`diff_diff.conley._compute_conley_meat`.

    No reference software combines all three ingredients (Conley
    spatial-HAC + Binder TSL + Gardner GMM correction) on a two-stage
    influence function.
    """
    from diff_diff.conley import _compute_conley_meat

    if psu_scores.ndim == 1:
        psu_scores = psu_scores[:, np.newaxis]
    k = psu_scores.shape[1]
    meat = np.zeros((k, k))

    unique_strata = np.unique(psu_strata)
    _variance_computed = False
    legitimate_zero_count = 0

    _global_psu_mean = None
    if lonely_psu == "adjust":
        _global_psu_mean = psu_scores.mean(axis=0, keepdims=True)

    for h in unique_strata:
        mask_h = psu_strata == h
        scores_h = psu_scores[mask_h]
        coords_h = psu_coords[mask_h]
        n_psu_h = scores_h.shape[0]

        if n_psu_h < 2:
            if lonely_psu == "remove":
                continue
            elif lonely_psu == "certainty":
                legitimate_zero_count += 1
                continue
            elif lonely_psu == "adjust":
                # Degenerate one-PSU kernel K = [[K(0)]] = [[1.0]] for both
                # Bartlett and uniform; equivalent to centered.T @ centered.
                # MUST `continue` to skip the FPC block below — with n_h = 1
                # the scale n_h/(n_h-1) divides by zero. Mirrors the Binder
                # helper's singleton-adjust branch exactly.
                centered = scores_h - _global_psu_mean
                with np.errstate(invalid="ignore", over="ignore"):
                    meat += centered.T @ centered
                _variance_computed = True
                continue

        f_h = 0.0
        if fpc_per_psu is not None:
            N_h = fpc_per_psu[mask_h][0]
            if N_h < n_psu_h:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of PSUs "
                    f"({n_psu_h}) in stratum. FPC must be >= n_PSU."
                )
            f_h = n_psu_h / N_h
            if f_h >= 1.0:
                legitimate_zero_count += 1

        psu_mean_h = scores_h.mean(axis=0, keepdims=True)
        centered = scores_h - psu_mean_h

        # Within-stratum Conley sandwich on PSU-centered scores. Pass
        # ``cluster_ids=None`` explicitly: after PSU aggregation every PSU
        # is its own cluster, so a cluster product kernel would zero all
        # cross-PSU pairs. See Wave E.2 plan Chunk 3 step 4.
        conley_meat_h = _compute_conley_meat(
            centered,
            coords_h,
            cutoff,
            metric,
            kernel,
            cluster_ids=None,
        )

        adjustment = (1.0 - f_h) * (n_psu_h / (n_psu_h - 1))
        with np.errstate(invalid="ignore", over="ignore"):
            meat += adjustment * conley_meat_h
        _variance_computed = True

    return meat, _variance_computed, legitimate_zero_count


def compute_survey_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> np.ndarray:
    """
    Compute Taylor Series Linearization (TSL) variance-covariance matrix.

    Implements the stratified cluster sandwich estimator with optional
    finite population correction (FPC).

    V_TSL = (X'WX)^{-1} [sum_h V_h] (X'WX)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from WLS fit (y - X @ beta, on ORIGINAL scale).
    resolved : ResolvedSurveyDesign
        Resolved survey design with weights, strata, PSU arrays.

    Returns
    -------
    vcov : np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    n, k = X.shape
    weights = resolved.weights

    # Bread: (X'WX)^{-1}
    XtWX = X.T @ (X * weights[:, np.newaxis])

    # Compute weighted scores per observation: w_i * X_i * u_i
    if resolved.weight_type == "aweight":
        scores = X * residuals[:, np.newaxis]
        # Zero-weight observations should not contribute to aweight meat
        if np.any(weights == 0):
            scores[weights == 0] = 0.0
    else:
        scores = X * (weights * residuals)[:, np.newaxis]

    meat, _variance_computed, legitimate_zero_count = _compute_stratified_psu_meat(scores, resolved)

    # Guard: if meat is zero, distinguish legitimate zero from unidentified variance
    if not np.any(meat != 0):
        if _variance_computed or legitimate_zero_count > 0:
            return np.zeros((k, k))
        return np.full((k, k), np.nan)

    # Precondition check: near-singular X'WX lets np.linalg.solve return
    # unstable values without raising (finding #19, axis A). Threshold of
    # 1/sqrt(eps) ≈ 6.7e7 is the standard rule of thumb — above it, the
    # sandwich bread becomes numerically unreliable and the caller should
    # be told so.
    with np.errstate(invalid="ignore", over="ignore"):
        XtWX_cond = float(np.linalg.cond(XtWX))
    cond_threshold = 1.0 / np.sqrt(np.finfo(float).eps)
    if np.isfinite(XtWX_cond) and XtWX_cond > cond_threshold:
        warnings.warn(
            f"X'WX is ill-conditioned (cond={XtWX_cond:.2e}) in the "
            f"survey sandwich variance; variance estimates may be "
            f"numerically unstable. This typically indicates near "
            f"multicollinearity or zero-weight strata dominating the "
            f"bread matrix.",
            UserWarning,
            stacklevel=2,
        )

    # Sandwich: (X'WX)^{-1} meat (X'WX)^{-1}
    try:
        temp = np.linalg.solve(XtWX, meat)
        vcov = np.linalg.solve(XtWX, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'WX matrix). "
                "This indicates perfect multicollinearity."
            ) from e
        raise

    return vcov


def compute_survey_if_variance(
    psi: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> float:
    """Compute design-based variance of a scalar estimator from IF values.

    For influence-function-based estimators (e.g., CallawaySantAnna),
    the per-unit influence function values ``psi_i`` capture each unit's
    contribution to the estimating equation.  Under simple random sampling
    the variance is ``sum(psi_i^2)``.  This function computes the
    design-based analogue accounting for PSU clustering, stratification,
    and finite population correction.

    V_design = sum_h (1-f_h) * (n_h/(n_h-1)) * sum_j (psi_hj - psi_h_bar)^2

    where psi_hj = sum_{i in PSU j, stratum h} psi_i.

    Parameters
    ----------
    psi : np.ndarray
        Per-unit influence function values, shape (n,).
    resolved : ResolvedSurveyDesign
        Resolved survey design.

    Returns
    -------
    float
        Design-based variance.  Returns ``np.nan`` when variance is
        unidentified (e.g., all strata removed by lonely_psu='remove').
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()

    meat, _variance_computed, legitimate_zero_count = _compute_stratified_psu_meat(
        psi[:, np.newaxis], resolved
    )

    # meat is (1, 1) — extract scalar
    meat_scalar = float(meat[0, 0])

    if meat_scalar == 0.0:
        if _variance_computed or legitimate_zero_count > 0:
            return 0.0
        return np.nan

    return meat_scalar


def _replicate_variance_factor(
    method: str,
    n_replicates: int,
    fay_rho: float,
) -> float:
    """Compute the scalar variance factor for replicate methods."""
    if method == "BRR":
        return 1.0 / n_replicates
    elif method == "Fay":
        return 1.0 / (n_replicates * (1.0 - fay_rho) ** 2)
    elif method == "SDR":
        return 4.0 / n_replicates
    elif method == "JK1":
        return (n_replicates - 1.0) / n_replicates
    # JKn handled separately (per-stratum factors)
    raise ValueError(f"Unknown replicate method: {method}")


def compute_replicate_vcov(
    X: np.ndarray,
    y: np.ndarray,
    full_sample_coef: np.ndarray,
    resolved: "ResolvedSurveyDesign",
    weight_type: str = "pweight",
) -> np.ndarray:
    """Compute replicate-weight variance-covariance matrix.

    Re-runs WLS for each replicate weight column and computes variance
    from the distribution of replicate coefficient vectors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Response vector of shape (n,).
    full_sample_coef : np.ndarray
        Coefficients from the full-sample fit, shape (k,).
    resolved : ResolvedSurveyDesign
        Must have ``uses_replicate_variance == True``.
    weight_type : str, default "pweight"
        Weight type for per-replicate WLS.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    from diff_diff.linalg import solve_ols

    rep_weights = resolved.replicate_weights
    # Replicate-variance entry points are only reached on replicate designs.
    assert rep_weights is not None
    method = resolved.replicate_method
    R = resolved.n_replicates
    k = X.shape[1]

    # Collect replicate coefficient vectors
    coef_reps = np.full((R, k), np.nan)
    for r in range(R):
        w_r = rep_weights[:, r]
        # For non-combined weights, multiply by full-sample weights
        if not resolved.combined_weights:
            w_r = w_r * resolved.weights
        # Skip replicates where all weights are zero
        if np.sum(w_r) == 0:
            continue
        try:
            coef_r, _, _ = solve_ols(
                X,
                y,
                weights=w_r,
                weight_type=weight_type,
                rank_deficient_action="silent",
                return_vcov=False,
                check_finite=False,
            )
            coef_reps[r] = coef_r
        except (np.linalg.LinAlgError, ValueError):
            pass  # NaN row for singular/degenerate replicate solve

    # Remove replicates with NaN coefficients
    valid = np.all(np.isfinite(coef_reps), axis=1)
    n_invalid = int(R - np.sum(valid))
    if n_invalid > 0:
        warnings.warn(
            f"{n_invalid} of {R} replicate solves failed (singular or degenerate). "
            f"Variance computed from {int(np.sum(valid))} valid replicates.",
            UserWarning,
            stacklevel=2,
        )
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        if n_valid == 0:
            warnings.warn(
                "All replicate solves failed. Returning NaN variance.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Only {n_valid} valid replicate(s) — variance is not estimable "
                f"with fewer than 2. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
        return np.full((k, k), np.nan), n_valid
    coef_valid = coef_reps[valid]
    c = full_sample_coef

    # Compute variance by method
    # Support mse=False: center on replicate mean instead of full-sample estimate
    # When rscales present and mse=False, center only over rscales > 0
    # (R's svrVar convention — zero-scaled replicates should not shift center)
    if resolved.mse:
        center = c
    else:
        if resolved.replicate_rscales is not None:
            pos_scale = resolved.replicate_rscales[valid] > 0
            if np.any(pos_scale):
                center = np.mean(coef_valid[pos_scale], axis=0)
            else:
                center = np.mean(coef_valid, axis=0)
        else:
            center = np.mean(coef_valid, axis=0)
    diffs = coef_valid - center[np.newaxis, :]

    outer_sum = diffs.T @ diffs  # (k, k)

    # BRR/Fay: use fixed scaling, ignore user-supplied scale/rscales (R convention)
    if method in ("BRR", "Fay", "SDR"):
        if resolved.replicate_scale is not None or resolved.replicate_rscales is not None:
            warnings.warn(
                f"Custom replicate_scale/replicate_rscales ignored for {method} "
                f"(BRR/Fay/SDR use fixed scaling).",
                UserWarning,
                stacklevel=2,
            )
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return factor * outer_sum, n_valid

    # JK1/JKn: apply scale * rscales multiplicatively (R's svrVar contract)
    scale = resolved.replicate_scale if resolved.replicate_scale is not None else 1.0

    if resolved.replicate_rscales is not None:
        valid_rscales = resolved.replicate_rscales[valid]
        V = np.zeros((k, k))
        for i in range(len(diffs)):
            V += valid_rscales[i] * np.outer(diffs[i], diffs[i])
        return scale * V, n_valid

    if method == "JK1":
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return scale * factor * outer_sum, n_valid
    elif method == "JKn":
        # JKn: V = sum_h ((n_h-1)/n_h) * sum_{r in h} (c_r - c)(c_r - c)^T
        rep_strata = resolved.replicate_strata
        if rep_strata is None:
            raise ValueError("JKn requires replicate_strata")
        valid_strata = rep_strata[valid]
        V = np.zeros((k, k))
        for h in np.unique(rep_strata):
            n_h_original = int(np.sum(rep_strata == h))
            mask_h = valid_strata == h
            if not np.any(mask_h):
                continue
            diffs_h = diffs[mask_h]
            V += ((n_h_original - 1.0) / n_h_original) * (diffs_h.T @ diffs_h)
        return scale * V, n_valid
    else:
        raise ValueError(f"Unknown replicate method: {method}")


def compute_replicate_if_variance(
    psi: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> Tuple[float, int]:
    """Compute replicate-based variance for influence-function estimators.

    Instead of re-running the full estimator, reweights the influence
    function under each replicate weight set.

    Parameters
    ----------
    psi : np.ndarray
        Per-unit influence function values, shape (n,).
    resolved : ResolvedSurveyDesign
        Must have ``uses_replicate_variance == True``.

    Returns
    -------
    float
        Replicate-based variance estimate.
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()
    rep_weights = resolved.replicate_weights
    # Replicate-variance entry points are only reached on replicate designs.
    assert rep_weights is not None
    method = resolved.replicate_method
    R = resolved.n_replicates

    # Match the contract of compute_survey_if_variance(): psi is accepted
    # as-is (the combined IF/WIF object), with NO extra weight multiplication.
    # Replicate contrasts are formed by rescaling each unit's contribution
    # by the ratio w_r/w_full (Rao-Wu reweighting).
    full_weights = resolved.weights
    theta_full = float(np.sum(psi))

    # Validate: combined_weights=True requires w_full > 0 wherever w_r > 0
    if resolved.combined_weights:
        for r in range(R):
            bad = (rep_weights[:, r] > 0) & (full_weights <= 0)
            if np.any(bad):
                raise ValueError(
                    f"Replicate column {r} has positive weight where full-sample "
                    f"weight is zero. With combined_weights=True, every "
                    f"replicate-positive observation must have a positive "
                    f"full-sample weight."
                )

    # Compute replicate estimates via weight-ratio rescaling
    theta_reps = np.full(R, np.nan)
    for r in range(R):
        w_r = rep_weights[:, r]
        if np.any(w_r > 0):
            if resolved.combined_weights:
                # Combined: w_r already includes full-sample weight
                ratio = np.divide(
                    w_r,
                    full_weights,
                    out=np.zeros_like(w_r, dtype=np.float64),
                    where=full_weights > 0,
                )
            else:
                # Non-combined: w_r is perturbation factor directly
                ratio = w_r
            theta_reps[r] = np.sum(ratio * psi)

    valid = np.isfinite(theta_reps)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        return np.nan, n_valid

    # Support mse=False: center on replicate mean
    # When rscales present and mse=False, center only over rscales > 0
    # (R's svrVar convention — zero-scaled replicates should not shift center)
    if resolved.mse:
        center = theta_full
    else:
        if resolved.replicate_rscales is not None:
            pos_scale = resolved.replicate_rscales[valid] > 0
            if np.any(pos_scale):
                center = float(np.mean(theta_reps[valid][pos_scale]))
            else:
                center = float(np.mean(theta_reps[valid]))
        else:
            center = float(np.mean(theta_reps[valid]))
    diffs = theta_reps[valid] - center

    ss = float(np.sum(diffs**2))

    # BRR/Fay: use fixed scaling, ignore user-supplied scale/rscales (R convention)
    if method in ("BRR", "Fay", "SDR"):
        if resolved.replicate_scale is not None or resolved.replicate_rscales is not None:
            warnings.warn(
                f"Custom replicate_scale/replicate_rscales ignored for {method} "
                f"(BRR/Fay/SDR use fixed scaling).",
                UserWarning,
                stacklevel=2,
            )
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return factor * ss, n_valid

    # JK1/JKn: apply scale * rscales multiplicatively (R's svrVar contract)
    scale = resolved.replicate_scale if resolved.replicate_scale is not None else 1.0

    if resolved.replicate_rscales is not None:
        valid_rscales = resolved.replicate_rscales[valid]
        return scale * float(np.sum(valid_rscales * diffs**2)), n_valid

    if method == "JK1":
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return scale * factor * ss, n_valid
    elif method == "JKn":
        rep_strata = resolved.replicate_strata
        if rep_strata is None:
            raise ValueError("JKn requires replicate_strata")
        valid_strata = rep_strata[valid]
        result = 0.0
        for h in np.unique(rep_strata):
            n_h_original = int(np.sum(rep_strata == h))
            mask_h = valid_strata == h
            if not np.any(mask_h):
                continue
            result += ((n_h_original - 1.0) / n_h_original) * float(np.sum(diffs[mask_h] ** 2))
        return scale * result, n_valid
    else:
        raise ValueError(f"Unknown replicate method: {method}")


def compute_replicate_refit_variance(
    refit_fn: Callable[[np.ndarray], np.ndarray],
    full_sample_estimate: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> Tuple[np.ndarray, int]:
    """Compute replicate variance by re-running an arbitrary estimation function.

    For each replicate weight column, calls ``refit_fn(w_r)`` and collects
    the resulting estimate vector.  Variance is computed from the distribution
    of replicate estimates using method-specific scaling.

    This generalises :func:`compute_replicate_vcov` (which hard-codes
    ``solve_ols`` as the refit) for estimators whose estimation procedure
    is more complex than a single OLS call (e.g. within-transformation,
    two-stage imputation, stacked regression).

    Parameters
    ----------
    refit_fn : callable
        ``(n,) weight array -> (k,) estimate array``.  Must return the same
        length *k* on every call.  Should return all-NaN when the estimation
        fails for that replicate.
    full_sample_estimate : np.ndarray
        Estimate vector from the full-sample weights, shape ``(k,)``.
    resolved : ResolvedSurveyDesign
        Must have ``uses_replicate_variance == True``.

    Returns
    -------
    tuple of (np.ndarray, int)
        ``(vcov, n_valid)`` where *vcov* has shape ``(k, k)`` and *n_valid*
        is the number of replicates that produced finite estimates.
    """
    full_sample_estimate = np.asarray(full_sample_estimate, dtype=np.float64).ravel()
    k = len(full_sample_estimate)
    rep_weights = resolved.replicate_weights
    # Replicate-variance entry points are only reached on replicate designs.
    assert rep_weights is not None
    method = resolved.replicate_method
    R = resolved.n_replicates

    # Collect replicate estimate vectors
    est_reps = np.full((R, k), np.nan)
    for r in range(R):
        w_r = rep_weights[:, r].copy()
        if not resolved.combined_weights:
            w_r = w_r * resolved.weights
        if np.sum(w_r) == 0:
            continue
        try:
            est_r = refit_fn(w_r)
            est_r = np.asarray(est_r, dtype=np.float64).ravel()
            if len(est_r) == k:
                est_reps[r] = est_r
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            pass  # NaN row for failed replicate

    # Remove replicates with NaN estimates
    valid = np.all(np.isfinite(est_reps), axis=1)
    n_invalid = int(R - np.sum(valid))
    if n_invalid > 0:
        warnings.warn(
            f"{n_invalid} of {R} replicate refits failed. "
            f"Variance computed from {int(np.sum(valid))} valid replicates.",
            UserWarning,
            stacklevel=2,
        )
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        if n_valid == 0:
            warnings.warn(
                "All replicate refits failed. Returning NaN variance.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Only {n_valid} valid replicate(s) — variance is not estimable "
                f"with fewer than 2. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
        return np.full((k, k), np.nan), n_valid

    est_valid = est_reps[valid]
    c = full_sample_estimate

    # --- Centering (mse flag) ---
    if resolved.mse:
        center = c
    else:
        if resolved.replicate_rscales is not None:
            pos_scale = resolved.replicate_rscales[valid] > 0
            if np.any(pos_scale):
                center = np.mean(est_valid[pos_scale], axis=0)
            else:
                center = np.mean(est_valid, axis=0)
        else:
            center = np.mean(est_valid, axis=0)
    diffs = est_valid - center[np.newaxis, :]

    outer_sum = diffs.T @ diffs  # (k, k)

    # --- Method-specific scaling ---
    # BRR/Fay: fixed scaling, ignore user-supplied scale/rscales
    if method in ("BRR", "Fay", "SDR"):
        if resolved.replicate_scale is not None or resolved.replicate_rscales is not None:
            warnings.warn(
                f"Custom replicate_scale/replicate_rscales ignored for {method} "
                f"(BRR/Fay/SDR use fixed scaling).",
                UserWarning,
                stacklevel=2,
            )
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return factor * outer_sum, n_valid

    # JK1/JKn: apply scale * rscales multiplicatively
    scale = resolved.replicate_scale if resolved.replicate_scale is not None else 1.0

    if resolved.replicate_rscales is not None:
        valid_rscales = resolved.replicate_rscales[valid]
        V = np.zeros((k, k))
        for i in range(len(diffs)):
            V += valid_rscales[i] * np.outer(diffs[i], diffs[i])
        return scale * V, n_valid

    if method == "JK1":
        factor = _replicate_variance_factor(method, R, resolved.fay_rho)
        return scale * factor * outer_sum, n_valid
    elif method == "JKn":
        rep_strata = resolved.replicate_strata
        if rep_strata is None:
            raise ValueError("JKn requires replicate_strata")
        valid_strata = rep_strata[valid]
        V = np.zeros((k, k))
        for h in np.unique(rep_strata):
            n_h_original = int(np.sum(rep_strata == h))
            mask_h = valid_strata == h
            if not np.any(mask_h):
                continue
            diffs_h = diffs[mask_h]
            V += ((n_h_original - 1.0) / n_h_original) * (diffs_h.T @ diffs_h)
        return scale * V, n_valid
    else:
        raise ValueError(f"Unknown replicate method: {method}")


def aggregate_to_psu(
    values: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> tuple:
    """Sum values within PSUs for PSU-level bootstrap perturbation.

    Parameters
    ----------
    values : np.ndarray
        Per-observation values, shape (n,) or (n, k).
    resolved : ResolvedSurveyDesign
        Resolved survey design.

    Returns
    -------
    psu_sums : np.ndarray
        Aggregated values, shape (n_psu,) or (n_psu, k).
    psu_ids : np.ndarray
        Unique PSU identifiers in the same order as ``psu_sums``.
    """
    if resolved.psu is None:
        # Each observation is its own PSU — return as-is
        return values.copy(), np.arange(len(values))

    psu = resolved.psu
    unique_psu = np.unique(psu)
    if values.ndim == 1:
        psu_sums = np.array([values[psu == p].sum() for p in unique_psu])
    else:
        psu_sums = np.array([values[psu == p].sum(axis=0) for p in unique_psu])
    return psu_sums, unique_psu
