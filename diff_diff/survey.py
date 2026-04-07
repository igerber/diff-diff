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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import _factorize_cluster_ids


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
        valid_rep_methods = {"BRR", "Fay", "JK1", "JKn"}
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

            # Validate weights
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

            # fweight validation: must be non-negative integers
            if self.weight_type == "fweight":
                pos_mask = raw_weights > 0
                if np.any(pos_mask):
                    fractional = raw_weights[pos_mask] - np.round(raw_weights[pos_mask])
                    if np.any(np.abs(fractional) > 1e-10):
                        raise ValueError(
                            "Frequency weights (fweight) must be non-negative integers. "
                            "Fractional values detected. Use pweight for non-integer weights."
                        )

            # Normalize: pweights/aweights to sum=n (mean=1); fweights unchanged
            # Skip normalization for replicate designs — the IF path uses
            # w_r / w_full ratios that must be on the same raw scale
            if self.replicate_weights is not None:
                weights = raw_weights.copy()
            elif self.weight_type in ("pweight", "aweight"):
                weights = raw_weights * (n / np.sum(raw_weights))
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
        panel-level survey info to one row per unit.

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
        )

    @property
    def needs_survey_vcov(self) -> bool:
        """Whether survey vcov (not generic sandwich) should be used."""
        return True  # Any resolved survey design uses the survey vcov path


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
        Survey design (uses ``weights`` column name).
    unit_order : array-like
        Ordered sequence of unit identifiers to align weights to.

    Returns
    -------
    np.ndarray
        Float64 array of unit-level weights, one per unit in ``unit_order``.
    """
    unit_w = data.groupby(unit_col)[survey_design.weights].first()
    return np.array([unit_w[u] for u in unit_order], dtype=np.float64)


def _resolve_survey_for_fit(survey_design, data, inference_mode="analytical"):
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
            "Use inference='analytical' with survey_design, or see "
            "docs/survey-roadmap.md for planned Phase 5 support."
        )

    resolved = survey_design.resolve(data)
    raw_w = (
        data[survey_design.weights].values.astype(np.float64)
        if survey_design.weights
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

    # When strata are present, make cluster IDs unique within strata
    # (same nesting logic as SurveyDesign.resolve() with nest=True)
    if resolved.strata is not None:
        combined = np.array([f"{s}_{c}" for s, c in zip(resolved.strata, cluster_ids)])
        codes, uniques = pd.factorize(combined)
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
    if method in ("BRR", "Fay"):
        if resolved.replicate_scale is not None or resolved.replicate_rscales is not None:
            warnings.warn(
                f"Custom replicate_scale/replicate_rscales ignored for {method} "
                f"(BRR/Fay use fixed scaling).",
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
    if method in ("BRR", "Fay"):
        if resolved.replicate_scale is not None or resolved.replicate_rscales is not None:
            warnings.warn(
                f"Custom replicate_scale/replicate_rscales ignored for {method} "
                f"(BRR/Fay use fixed scaling).",
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
