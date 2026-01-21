"""
Bootstrap inference for Callaway-Sant'Anna estimator.

This module provides bootstrap weight generation functions, the bootstrap
results container, and the mixin class with bootstrap inference methods.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import HAS_RUST_BACKEND, _rust_bootstrap_weights

if TYPE_CHECKING:
    pass


# =============================================================================
# Bootstrap Weight Generators
# =============================================================================


def _generate_bootstrap_weights(
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap weights for multiplier bootstrap.

    Parameters
    ----------
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_units,).
    """
    if weight_type == "rademacher":
        # Rademacher: +1 or -1 with equal probability
        return rng.choice([-1.0, 1.0], size=n_units)

    elif weight_type == "mammen":
        # Mammen's two-point distribution
        # E[v] = 0, E[v^2] = 1, E[v^3] = 1
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2  # ≈ -0.618
        val2 = (sqrt5 + 1) / 2   # ≈ 1.618 (golden ratio)
        p1 = (sqrt5 + 1) / (2 * sqrt5)  # ≈ 0.724
        return rng.choice([val1, val2], size=n_units, p=[p1, 1 - p1])

    elif weight_type == "webb":
        # Webb's 6-point distribution (recommended for few clusters)
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
        ])
        probs = np.array([1, 2, 3, 3, 2, 1]) / 12
        return rng.choice(values, size=n_units, p=probs)

    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', "
            f"got '{weight_type}'"
        )


def _generate_bootstrap_weights_batch(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate all bootstrap weights at once (vectorized).

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    # Use Rust backend if available (parallel + fast RNG)
    if HAS_RUST_BACKEND and _rust_bootstrap_weights is not None:
        # Get seed from the NumPy RNG for reproducibility
        seed = rng.integers(0, 2**63 - 1)
        return _rust_bootstrap_weights(n_bootstrap, n_units, weight_type, seed)

    # Fallback to NumPy implementation
    return _generate_bootstrap_weights_batch_numpy(n_bootstrap, n_units, weight_type, rng)


def _generate_bootstrap_weights_batch_numpy(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    NumPy fallback implementation of _generate_bootstrap_weights_batch.

    Generates multiplier bootstrap weights for wild cluster bootstrap.
    All weight distributions satisfy E[w] = 0, E[w^2] = 1.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher" (+-1), "mammen" (2-point),
        or "webb" (6-point).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    if weight_type == "rademacher":
        # Rademacher: +1 or -1 with equal probability
        return rng.choice([-1.0, 1.0], size=(n_bootstrap, n_units))

    elif weight_type == "mammen":
        # Mammen's two-point distribution
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2
        p1 = (sqrt5 + 1) / (2 * sqrt5)
        return rng.choice([val1, val2], size=(n_bootstrap, n_units), p=[p1, 1 - p1])

    elif weight_type == "webb":
        # Webb's 6-point distribution
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
        ])
        probs = np.array([1, 2, 3, 3, 2, 1]) / 12
        return rng.choice(values, size=(n_bootstrap, n_units), p=probs)

    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', "
            f"got '{weight_type}'"
        )


# =============================================================================
# Bootstrap Results Container
# =============================================================================


@dataclass
class CSBootstrapResults:
    """
    Results from Callaway-Sant'Anna multiplier bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights used.
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : Tuple[float, float]
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    group_time_ses : Dict[Tuple[Any, Any], float]
        Bootstrap SEs for each ATT(g,t).
    group_time_cis : Dict[Tuple[Any, Any], Tuple[float, float]]
        Bootstrap CIs for each ATT(g,t).
    group_time_p_values : Dict[Tuple[Any, Any], float]
        Bootstrap p-values for each ATT(g,t).
    event_study_ses : Optional[Dict[int, float]]
        Bootstrap SEs for event study effects.
    event_study_cis : Optional[Dict[int, Tuple[float, float]]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Optional[Dict[int, float]]
        Bootstrap p-values for event study effects.
    group_effect_ses : Optional[Dict[Any, float]]
        Bootstrap SEs for group effects.
    group_effect_cis : Optional[Dict[Any, Tuple[float, float]]]
        Bootstrap CIs for group effects.
    group_effect_p_values : Optional[Dict[Any, float]]
        Bootstrap p-values for group effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT (if requested).
    """
    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    group_time_ses: Dict[Tuple[Any, Any], float]
    group_time_cis: Dict[Tuple[Any, Any], Tuple[float, float]]
    group_time_p_values: Dict[Tuple[Any, Any], float]
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_effect_ses: Optional[Dict[Any, float]] = None
    group_effect_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_effect_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


# =============================================================================
# Bootstrap Mixin Class
# =============================================================================


class CallawaySantAnnaBootstrapMixin:
    """
    Mixin class providing bootstrap inference methods for CallawaySantAnna.

    This class is not intended to be used standalone. It provides methods
    that are used by the main CallawaySantAnna class for multiplier bootstrap
    inference.
    """

    # Type hints for attributes accessed from the main class
    n_bootstrap: int
    bootstrap_weight_type: str
    alpha: float
    seed: Optional[int]
    anticipation: int

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        time_periods: List[Any],
    ) -> CSBootstrapResults:
        """
        Run multiplier bootstrap for inference on all parameters.

        This implements the multiplier bootstrap procedure from Callaway & Sant'Anna (2021).
        The key idea is to perturb the influence function contributions with random
        weights at the cluster (unit) level, then recompute aggregations.

        Parameters
        ----------
        group_time_effects : dict
            Dictionary of ATT(g,t) effects with analytical SEs.
        influence_func_info : dict
            Dictionary mapping (g,t) to influence function information.
        aggregate : str, optional
            Type of aggregation requested.
        balance_e : int, optional
            Balance parameter for event study.
        treatment_groups : list
            List of treatment cohorts.
        time_periods : list
            List of time periods.

        Returns
        -------
        CSBootstrapResults
            Bootstrap inference results.
        """
        # Warn about low bootstrap iterations
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference. Percentile confidence intervals and p-values "
                "may be unreliable with few iterations.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        # Collect all unique units across all (g,t) combinations
        all_units = set()
        for (g, t), info in influence_func_info.items():
            all_units.update(info['treated_units'])
            all_units.update(info['control_units'])
        all_units = sorted(all_units)
        n_units = len(all_units)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get list of (g,t) pairs
        gt_pairs = list(group_time_effects.keys())
        n_gt = len(gt_pairs)

        # Identify post-treatment (g,t) pairs for overall ATT
        # Pre-treatment effects are for parallel trends assessment, not aggregated
        post_treatment_mask = np.array([
            t >= g - self.anticipation for (g, t) in gt_pairs
        ])
        post_treatment_indices = np.where(post_treatment_mask)[0]

        # Compute aggregation weights for overall ATT (post-treatment only)
        all_n_treated = np.array([
            group_time_effects[gt]['n_treated'] for gt in gt_pairs
        ], dtype=float)
        post_n_treated = all_n_treated[post_treatment_mask]

        # Guard against empty post-treatment set - return early with NaN results
        if len(post_treatment_indices) == 0:
            warnings.warn(
                "No post-treatment effects for bootstrap aggregation. "
                "Returning NaN for overall ATT statistics.",
                UserWarning,
                stacklevel=2
            )
            # Return bootstrap results with NaN for overall ATT inference
            # Individual group-time effects may still have valid pre-treatment SEs
            gt_ses = {gt: group_time_effects[gt].get('se', np.nan) for gt in gt_pairs}
            gt_cis = {
                gt: group_time_effects[gt].get('conf_int', (np.nan, np.nan))
                for gt in gt_pairs
            }
            gt_p_values = {
                gt: group_time_effects[gt].get('p_value', np.nan) for gt in gt_pairs
            }
            return CSBootstrapResults(
                n_bootstrap=self.n_bootstrap,
                weight_type=self.bootstrap_weight_type,
                alpha=self.alpha,
                overall_att_se=np.nan,
                overall_att_ci=(np.nan, np.nan),
                overall_att_p_value=np.nan,
                group_time_ses=gt_ses,
                group_time_cis=gt_cis,
                group_time_p_values=gt_p_values,
                event_study_ses=None,
                event_study_cis=None,
                event_study_p_values=None,
                group_effect_ses=None,
                group_effect_cis=None,
                group_effect_p_values=None,
                bootstrap_distribution=np.array([]),
            )

        overall_weights_post = post_n_treated / np.sum(post_n_treated)

        # Original point estimates
        original_atts = np.array([group_time_effects[gt]['effect'] for gt in gt_pairs])
        original_overall = np.sum(overall_weights_post * original_atts[post_treatment_mask])

        # Prepare event study and group aggregation info if needed
        event_study_info = None
        group_agg_info = None

        if aggregate in ["event_study", "all"]:
            event_study_info = self._prepare_event_study_aggregation(
                gt_pairs, group_time_effects, balance_e
            )

        if aggregate in ["group", "all"]:
            group_agg_info = self._prepare_group_aggregation(
                gt_pairs, group_time_effects, treatment_groups
            )

        # Pre-compute unit index arrays for each (g,t) pair (done once, not per iteration)
        gt_treated_indices = []
        gt_control_indices = []
        gt_treated_inf = []
        gt_control_inf = []

        for j, gt in enumerate(gt_pairs):
            info = influence_func_info[gt]
            treated_idx = np.array([unit_to_idx[u] for u in info['treated_units']])
            control_idx = np.array([unit_to_idx[u] for u in info['control_units']])
            gt_treated_indices.append(treated_idx)
            gt_control_indices.append(control_idx)
            gt_treated_inf.append(np.asarray(info['treated_inf']))
            gt_control_inf.append(np.asarray(info['control_inf']))

        # Generate ALL bootstrap weights upfront: shape (n_bootstrap, n_units)
        # This is much faster than generating one at a time
        all_bootstrap_weights = _generate_bootstrap_weights_batch(
            self.n_bootstrap, n_units, self.bootstrap_weight_type, rng
        )

        # Vectorized bootstrap ATT(g,t) computation
        # Compute all bootstrap ATTs for all (g,t) pairs using matrix operations
        bootstrap_atts_gt = np.zeros((self.n_bootstrap, n_gt))

        for j in range(n_gt):
            treated_idx = gt_treated_indices[j]
            control_idx = gt_control_indices[j]
            treated_inf = gt_treated_inf[j]
            control_inf = gt_control_inf[j]

            # Extract weights for this (g,t)'s units across all bootstrap iterations
            # Shape: (n_bootstrap, n_treated) and (n_bootstrap, n_control)
            treated_weights = all_bootstrap_weights[:, treated_idx]
            control_weights = all_bootstrap_weights[:, control_idx]

            # Vectorized perturbation: matrix-vector multiply
            # Shape: (n_bootstrap,)
            # Suppress RuntimeWarnings for edge cases (small samples, extreme weights)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                perturbations = (
                    treated_weights @ treated_inf +
                    control_weights @ control_inf
                )

            # Let non-finite values propagate - they will be handled at statistics computation
            bootstrap_atts_gt[:, j] = original_atts[j] + perturbations

        # Vectorized overall ATT: matrix-vector multiply (post-treatment only)
        # Shape: (n_bootstrap,)
        # Suppress RuntimeWarnings for edge cases - non-finite values handled at statistics computation
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            bootstrap_overall = bootstrap_atts_gt[:, post_treatment_indices] @ overall_weights_post

        # Vectorized event study aggregation
        # Non-finite values handled at statistics computation stage
        rel_periods: List[int] = []
        bootstrap_event_study: Optional[Dict[int, np.ndarray]] = None
        if event_study_info is not None:
            rel_periods = sorted(event_study_info.keys())
            bootstrap_event_study = {}
            for e in rel_periods:
                agg_info = event_study_info[e]
                gt_indices = agg_info['gt_indices']
                weights = agg_info['weights']
                # Vectorized: select columns and multiply by weights
                # Suppress RuntimeWarnings for edge cases
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    bootstrap_event_study[e] = bootstrap_atts_gt[:, gt_indices] @ weights

        # Vectorized group aggregation
        # Non-finite values handled at statistics computation stage
        group_list: List[Any] = []
        bootstrap_group: Optional[Dict[Any, np.ndarray]] = None
        if group_agg_info is not None:
            group_list = sorted(group_agg_info.keys())
            bootstrap_group = {}
            for g in group_list:
                agg_info = group_agg_info[g]
                gt_indices = agg_info['gt_indices']
                weights = agg_info['weights']
                # Suppress RuntimeWarnings for edge cases
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    bootstrap_group[g] = bootstrap_atts_gt[:, gt_indices] @ weights

        # Compute bootstrap statistics for ATT(g,t)
        gt_ses = {}
        gt_cis = {}
        gt_p_values = {}

        for j, gt in enumerate(gt_pairs):
            se, ci, p_value = self._compute_effect_bootstrap_stats(
                original_atts[j], bootstrap_atts_gt[:, j],
                context=f"ATT(g={gt[0]}, t={gt[1]})"
            )
            gt_ses[gt] = se
            gt_cis[gt] = ci
            gt_p_values[gt] = p_value

        # Compute bootstrap statistics for overall ATT
        overall_se, overall_ci, overall_p_value = self._compute_effect_bootstrap_stats(
            original_overall, bootstrap_overall,
            context="overall ATT"
        )

        # Compute bootstrap statistics for event study effects
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if bootstrap_event_study is not None and event_study_info is not None:
            event_study_ses = {}
            event_study_cis = {}
            event_study_p_values = {}

            for e in rel_periods:
                se, ci, p_value = self._compute_effect_bootstrap_stats(
                    event_study_info[e]['effect'], bootstrap_event_study[e],
                    context=f"event study (e={e})"
                )
                event_study_ses[e] = se
                event_study_cis[e] = ci
                event_study_p_values[e] = p_value

        # Compute bootstrap statistics for group effects
        group_effect_ses = None
        group_effect_cis = None
        group_effect_p_values = None

        if bootstrap_group is not None and group_agg_info is not None:
            group_effect_ses = {}
            group_effect_cis = {}
            group_effect_p_values = {}

            for g in group_list:
                se, ci, p_value = self._compute_effect_bootstrap_stats(
                    group_agg_info[g]['effect'], bootstrap_group[g],
                    context=f"group effect (g={g})"
                )
                group_effect_ses[g] = se
                group_effect_cis[g] = ci
                group_effect_p_values[g] = p_value

        return CSBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weight_type,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p_value,
            group_time_ses=gt_ses,
            group_time_cis=gt_cis,
            group_time_p_values=gt_p_values,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_effect_ses=group_effect_ses,
            group_effect_cis=group_effect_cis,
            group_effect_p_values=group_effect_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def _prepare_event_study_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        balance_e: Optional[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare aggregation info for event study bootstrap."""
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}

        for j, (g, t) in enumerate(gt_pairs):
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                j,  # index in gt_pairs
                group_time_effects[(g, t)]['effect'],
                group_time_effects[(g, t)]['n_treated']
            ))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = set()
            for j, (g, t) in enumerate(gt_pairs):
                if t - g == balance_e:
                    groups_at_e.add(g)

            balanced_effects: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        j,
                        group_time_effects[(g, t)]['effect'],
                        group_time_effects[(g, t)]['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregation weights
        result = {}
        for e, effect_list in effects_by_e.items():
            indices = np.array([x[0] for x in effect_list])
            effects = np.array([x[1] for x in effect_list])
            n_treated = np.array([x[2] for x in effect_list], dtype=float)

            weights = n_treated / np.sum(n_treated)
            agg_effect = np.sum(weights * effects)

            result[e] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
            }

        return result

    def _prepare_group_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Prepare aggregation info for group-level bootstrap."""
        result = {}

        for g in treatment_groups:
            # Get all effects for this group (post-treatment only: t >= g)
            group_data = []
            for j, (gg, t) in enumerate(gt_pairs):
                if gg == g and t >= g:
                    group_data.append((
                        j,
                        group_time_effects[(gg, t)]['effect'],
                    ))

            if not group_data:
                continue

            indices = np.array([x[0] for x in group_data])
            effects = np.array([x[1] for x in group_data])

            # Equal weights across time periods
            weights = np.ones(len(effects)) / len(effects)
            agg_effect = np.sum(weights * effects)

            result[g] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
            }

        return result

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        lower = float(np.percentile(boot_dist, alpha / 2 * 100))
        upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return (lower, upper)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        n_valid: Optional[int] = None,
    ) -> float:
        """
        Compute two-sided bootstrap p-value.

        Uses the percentile method: p-value is the proportion of bootstrap
        estimates on the opposite side of zero from the original estimate,
        doubled for two-sided test.

        Parameters
        ----------
        original_effect : float
            Original point estimate.
        boot_dist : np.ndarray
            Bootstrap distribution of the effect.
        n_valid : int, optional
            Number of valid bootstrap samples. If None, uses self.n_bootstrap.
            Use this when boot_dist has already been filtered for non-finite values
            to ensure the p-value floor is based on the actual valid sample count.

        Returns
        -------
        float
            Two-sided bootstrap p-value.
        """
        if original_effect >= 0:
            # Proportion of bootstrap estimates <= 0
            p_one_sided = np.mean(boot_dist <= 0)
        else:
            # Proportion of bootstrap estimates >= 0
            p_one_sided = np.mean(boot_dist >= 0)

        # Two-sided p-value
        p_value = min(2 * p_one_sided, 1.0)

        # Ensure minimum p-value using n_valid if provided, otherwise n_bootstrap
        n_for_floor = n_valid if n_valid is not None else self.n_bootstrap
        p_value = max(p_value, 1 / (n_for_floor + 1))

        return float(p_value)

    def _compute_effect_bootstrap_stats(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        context: str = "bootstrap distribution",
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Compute bootstrap statistics for a single effect.

        Non-finite bootstrap samples are dropped and a warning is issued if any
        are present. If too few valid samples remain (<50%), returns NaN for all
        statistics to signal invalid inference.

        Parameters
        ----------
        original_effect : float
            Original point estimate.
        boot_dist : np.ndarray
            Bootstrap distribution of the effect.
        context : str, optional
            Description for warning messages, by default "bootstrap distribution".

        Returns
        -------
        se : float
            Bootstrap standard error.
        ci : Tuple[float, float]
            Percentile confidence interval.
        p_value : float
            Bootstrap p-value.
        """
        # Filter out non-finite values
        finite_mask = np.isfinite(boot_dist)
        n_valid = np.sum(finite_mask)
        n_total = len(boot_dist)

        if n_valid < n_total:
            import warnings
            n_nonfinite = n_total - n_valid
            warnings.warn(
                f"Dropping {n_nonfinite}/{n_total} non-finite bootstrap samples in {context}. "
                "This may occur with very small samples or extreme weights. "
                "Bootstrap estimates based on remaining valid samples.",
                RuntimeWarning,
                stacklevel=3
            )

        # Check if we have enough valid samples
        if n_valid < n_total * 0.5:
            import warnings
            warnings.warn(
                f"Too few valid bootstrap samples ({n_valid}/{n_total}) in {context}. "
                "Returning NaN for SE/CI/p-value to signal invalid inference.",
                RuntimeWarning,
                stacklevel=3
            )
            return np.nan, (np.nan, np.nan), np.nan

        # Use only valid samples
        valid_dist = boot_dist[finite_mask]
        n_valid_bootstrap = len(valid_dist)

        se = float(np.std(valid_dist, ddof=1))
        ci = self._compute_percentile_ci(valid_dist, self.alpha)

        # Compute p-value using shared method with correct floor based on valid sample count
        p_value = self._compute_bootstrap_pvalue(original_effect, valid_dist, n_valid=n_valid_bootstrap)

        return se, ci, p_value
